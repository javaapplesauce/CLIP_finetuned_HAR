import os
import sys
import time
import random
import math
import psutil
from datetime import timedelta
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist  # single import for DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForImageClassification
from datasets import load_dataset, DownloadConfig

import hydra
from omegaconf import DictConfig, OmegaConf
from rich import pretty
import wandb

from load_data import load_data
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score
)
import numpy as np





### initialize wandb
def wandb_init(cfg: DictConfig):
    wandb.init(
        project='HAR',
        group=cfg.exp_group,
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))


### build the model (default ViT-32)
def build_model(cfg: DictConfig):
    return AutoModelForImageClassification.from_pretrained(
        "openai/clip-vit-base-patch32",
        num_labels=15)


### build the criterion
class SignLoss(nn.Module):
    def forward(self, class_logits, targets):
        assert class_logits.size() == targets.size(), "dimension mismatch"

        batch_size, n_classes = class_logits.size()
        zeros = torch.zeros(batch_size).view(batch_size, -1).to(targets.device)
        
        class_logits[targets == 1] *= -1
        class_logits = torch.cat((zeros, class_logits), dim=1)

        loss = torch.logsumexp(class_logits, 1)

        return loss

def build_criterion(cfg, train_loader):
    if cfg.loss.name == 'WBCE':
        pos_weights = train_loader.dataset.get_pos_weights()
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((pos_weights), device=train_loader.dataset.device))
    elif cfg.loss.name == 'Sign' or cfg.loss.name == 'SignLoss':
        return SignLoss()
    else:
        raise ValueError(f"Unknown loss: {cfg.loss.name}")

### build the optimizer
def build_optimizer(cfg, model):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.optim.base_lr, 
        weight_decay=cfg.optim.weight_decay)
    return optimizer

### build the scheduler
def build_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
            return lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.scheduler.T_0,
                T_mult=cfg.scheduler.T_mult
            )
    else:
        raise NotImplemented(f"Scheduler {cfg.scheduler.name} not implemented")

### data loading --> load_data.py

### validation
def gather_list(data_list):
    """
    Helper to gather a Python list from all processes.
    Returns a flat list on all ranks.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, data_list)
        # flatten
        flat = []
        for part in gathered:
            flat.extend(part)
        return flat
    else:
        return data_list

def _evaluate(model, loader, device):
    """
    Evaluates the model on validation set. Returns dict of metrics.
    Metrics: accuracy, precision, recall, f1, mAP
    """
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []


    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    if dist.is_initalized():
        all_preds = gather_list(all_preds)
        all_labels = gather_list(all_labels)
        all_probs = gather_list(all_probs)
        if dist.get_rank() != 0:
            return None

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    total = len(all_labels)
    correct = ( all_preds == all_labels).sum()
    acc = 100.0 * correct / total
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    num_classes = len(np.unique(all_labels))
    APs = []
    labels_oneshot = np.zeros((total, num_classes), dtype=int)
    labels_oneshot[np.arange(total), all_labels] = 1
    for c in range(num_classes):
        ap_c = average_precision_score(labels_oneshot[:, c], np.array(all_probs)[:, c])
        if np.isnan(ap_c):
            ap_c = 0.0
        APs.append(ap_c)
    mAP = np.mean(APs) * 100
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mAP": mAP}


### training loop
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
        self.cfg.data.n_iters = len(train_loader)

        model = build_model(self.cfg).to(self.local_rank)
        self.model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
        self.criterion = build_criterion(cfg, self.train_loader)
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)

        self.best_acc = 0.0

    def train(self):
        if self.global_master:
            wandb_init(self.cfg)

        
        ema = torch.optim.swa_utils.AveragedModel(model) if self.cfg.get("use_ema", False) else None

        for epoch in range(self.cfg.epochs):
            self.train_sampler.set_epoch(epoch)
            metric = Metric()
            self.model.train()

            if self.global_master:
                wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

            for it, (source, targets) in enumerate(self.train_loader):
                start_time = time.time()
                
                source = source.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()

                output = self.model(source)
                logits  = output.logits
                targets_onehot = torch.nn.F.one_hot(
                    targets, num_classes=15).float().to(self.device)
                
                loss_values = self.criterion(logits, targets_onehot.to(logits.device))
                loss = loss_values.mean()
                loss.backward()            
                self.optimizer.step()

                if ema:
                    ema.update_parameters(model)

                if self.cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
                    self.scheduler.step(epoch + it / self.cfg.data.n_iters)
                else:
                    self.scheduler.step()  
              
                metric.add_val(loss.item())

                batch_time = time.time() - start_time

                if self.global_master and it % self.cfg.train.print_freq == 0:
                    self.log(
                        f"Epoch: {epoch}/{self.cfg.epochs}, "
                        f"Iter: {it}/{len(train_data)}, "
                        f"Loss: {loss.item():.4f} ({metric.mean:.4f}), "
                        f"Time: {batch_time:.2f}s"
                    )

                    wandb.log({
                        "train/loss": loss.item(),
                        "train/loss_avg": metric.mean,
                        "train/batch_time": batch_time,
                        "train/epoch": epoch,
                    }, step=epoch * self.cfg.data.n_iters + it)
            
            
            metrics = self._evaluate(self.model, self.val_loader, self.device)
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
                    
                if metrics['acc'] > self.best_acc:
                    self.best_acc = metrics['acc']
                    self._save_checkpoint(epoch, best=True)






@hydra.main(config_path='configs', config_name='defaults')
def main(cfg: DictConfig):
    pretty.install()
    OmegaConf.resolve(cfg)

    os.environ["MASTER_ADDR"] = cfg.master.addr
    os.environ["MASTER_PORT"] = str(cfg.master.port)

    # 1) initialize DDP from torchrun's env vars
    dist.init_process_group(
        backend="nccl",
        init_method="env://",            # reads MASTER_ADDR & MASTER_PORT
        rank=cfg.local_rank,
        world_size=cfg.world_size,
        timeout=timedelta(minutes=10),
    )

    # 2) discover ranks
    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    local_rank = cfg.local_rank

    # 3) bind to the correct GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", cfg.local_rank)
    
    trainer = Trainer(cfg, cfg.local_rank)
    trainer.train()


if __name__ == "__main__":
    main()