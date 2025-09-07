import os
import time

from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist  # single import for DDP
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import DictConfig, OmegaConf
from rich import pretty
import wandb

from load_data import load_data


from utils import Metric, wandb_init, build_model, build_criterion, build_optimizer, build_scheduler, gather_list, _evaluate, Sign_loss


class Trainer:
    def __init__(self, cfg, local_rank) -> None:
        self.cfg = cfg
        self.local_rank = local_rank
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_gpus = torch.cuda.device_count()
        self.device = torch.device("cuda", local_rank)
        self.global_master = self.global_rank == 0


        train_data, val_data, test_data = load_data(cfg)
        self.train_loader = train_data  # Store for loop
        self.val_loader = val_data
        self.test_loader = test_data
        self.train_sampler = self.train_loader.sampler

        model = build_model(self.cfg).to(self.device)
        self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)


        self.criterion = build_criterion(cfg, self.train_loader)
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)

        self.id_str = '[G %d/%d, L %d/%d]' % (self.global_rank, self.world_size, self.local_rank, self.local_gpus)
        self.save_every = cfg.save_every
        self.best_acc = 0.0

    def log(self, message, master_only=True):
        if (master_only and self.global_master) or (not master_only):
            print(self.id_str + ': ' + str(message))

    def train(self):
        if self.global_master:
            wandb_init(self.cfg)

        ema = torch.optim.swa_utils.AveragedModel(self.model) if self.cfg.get("use_ema", False) else None

        for epoch in range(self.cfg.epochs):
            self.train_sampler.set_epoch(epoch)
            metric = Metric()
            self.model.train()

            if self.global_master:
                wandb.log({'lr': self.optimizer.param_groups[0]['lr']})

            for it, (source, targets) in enumerate(self.train_loader):
                start_time = time.time()
                
                images = source.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()

                logits  = self.model(pixel_values=images)

                targets_onehot = F.one_hot(
                    targets, num_classes=15).float().to(self.device)
                
                loss_values = self.criterion(logits, targets_onehot.to(logits.device))
                loss = loss_values.mean()
                loss.backward()            
                self.optimizer.step()
                
                # EMA only
                if ema:
                    ema.update_parameters(self.model)

                if self.cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
                    self.scheduler.step(epoch + it / len(self.train_loader))
                else:
                    self.scheduler.step()  
              
                metric.add_val(loss.item())
                batch_time = time.time() - start_time

                if self.global_master and it % self.cfg.train.print_freq == 0:
                    self.log(
                        f"Epoch: {epoch}/{self.cfg.epochs}, "
                        f"Iter: {it}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f} ({metric.mean:.4f}), "
                        f"Time: {batch_time:.2f}s"
                    )

                    wandb.log({
                        "train/loss": loss.item(),
                        "train/loss_avg": metric.mean,
                        "train/batch_time": batch_time,
                        "train/epoch": epoch,
                    }, step=epoch * len(self.train_loader) + it)
            
            metrics = _evaluate(self.cfg, self.model, self.val_loader, self.device)
            if self.global_master and metrics:
                print(
                    f"Val Accuracy: {metrics['acc']:.2f}% | "
                    f"Precision: {metrics['prec']:.2f}% | "
                    f"Recall: {metrics['rec']:.2f}% | "
                    f"F1: {metrics['f1']:.2f}% | "
                    f"mAP: {metrics['mAP']:.2f}%"
                )
                wandb.log({
                    "val/acc": metrics["acc"],
                    "val/prec": metrics["prec"],
                    "val/rec": metrics["rec"],
                    "val/f1": metrics["f1"],
                    "val/mAP": metrics["mAP"],
                })
                
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
                if metrics['mAP'] > self.best_acc:
                    self.best_acc = metrics['mAP']
                    self._save_checkpoint(epoch, best=True)

        metrics = _evaluate(self.cfg, self.model, self.test_loader, self.device)
        if self.global_master and metrics:
            print(
                f"Val Accuracy: {metrics['acc']:.2f}% | "
                f"Precision: {metrics['prec']:.2f}% | "
                f"Recall: {metrics['rec']:.2f}% | "
                f"F1: {metrics['f1']:.2f}% | "
                f"mAP: {metrics['mAP']:.2f}%"
            )
            
            wandb.log({
                "test/acc": metrics["acc"],
                "test/f1": metrics["f1"],
                "test/mAP": metrics["mAP"],
            })

    def _save_checkpoint(self, epoch, best: bool = False):
        os.makedirs("checkpoint", exist_ok=True)
        ckp = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": self.best_acc,
        }
        filename = "best.pt" if best else f"checkpoint_epoch{epoch}.pt"
        path = os.path.join("checkpoint", filename)

        torch.save(ckp, path)


@hydra.main(version_base=None, config_path="configs", config_name="defaults")
def main(cfg: DictConfig):
    pretty.install()
    # OmegaConf.resolve(cfg)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=10),
    )
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    trainer = Trainer(cfg, local_rank)
    try:
        trainer.train()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()