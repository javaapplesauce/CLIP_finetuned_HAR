import os

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModelForImageClassification
import torch.distributed as dist
from torchvision.transforms import InterpolationMode
from PIL import Image

# Multiprocess imports
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Dataset imports
from datasets import load_dataset, DownloadConfig
from collections import Counter
import hydra
from omegaconf import DictConfig, OmegaConf
from rich import pretty
from load_data import load_data

# Metrics imports
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import random
import wandb
import time
import psutil
from datetime import datetime, timedelta
import sys
import math
import pandas





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
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights).cuda())
    elif cfg.loss.name == 'Sign' or cfg.loss.name == 'SignLoss':
        criterion = SignLoss()
    else:
        raise ValueError
    
    return criterion

### build the optimizer
def build_optimizer(cfg, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)

### build the scheduler
def build_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.scheduler.T_0,
                T_mult=cfg.scheduler.T_mult
            )
    else:
        raise NotImplemented()

    return scheduler

### data loading --> load_data.py


### training loop
class Trainer:

    def __init__(self, cfg, local_rank) -> None:
        self.cfg = cfg
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = local_rank
        self.local_gpus = torch.cuda.device_count()

        self.global_master = self.global_rank == 0
        self.local_master = self.local_rank == 0

    def train(self):
        if self.global_master:
            wandb_init(self.cfg)
        
        model = build_model(self.cfg).to(self.local_rank)
        model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)



        train_data, val_data, test_data = load_data.load_data(self.cfg)
        self.train_data = train_data  # Store for loop
        self.val_data = val_data
        self.test_data = test_data
        self.cfg.data.n_iters = len(train_data)

        criterion = build_criterion(self.cfg, train_loader)
        optimizer = build_optimizer(self.cfg, model)
        scheduler = build_scheduler(self.cfg, optimizer)
        ema = torch.optim.swa_utils.AveragedModel(model) if self.cfg.get("use_ema", False) else None

        for epoch in range(self.cfg.epochs):
            val_data.set_epoch(epoch)
            metric = Metric()
            model.train()

            if self.global_master:
                wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

            for it, (source, targets) in enumerate(self.train_data):
                start_time = time.time()
                
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
                
                optimizer.zero_grad()

                output = model(source)
                logits  = output.logits
                targets_onehot = torch.nn.F.one_hot(
                    targets, num_classes=15).float()
                
                loss_values = criterion(logits, targets_onehot.to(logits.device))
                loss = loss_values.mean()
                loss.backward()            
                optimizer.step()

                if ema:
                    ema.update_parameters(model)

                if self.cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
                    scheduler.step(epoch + it / self.cfg.data.n_iters)
                else:
                    scheduler.step()  
              
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
        




### validation


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
