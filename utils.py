import torch
import torch.nn as nn
import os
from typing import Union, List
from transformers import CLIPModel, CLIPTokenizer
import wandb
from omegaconf import OmegaConf, DictConfig
import sys
import torch.distributed as dist  # single import for DDP
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score
)
import numpy as np

from CLIP_HAR import build_model

class SignLoss(nn.Module):
    def forward(self, class_logits, targets):
        assert class_logits.size() == targets.size(), "dimension mismatch"

        batch_size, n_classes = class_logits.size()
        zeros = torch.zeros(batch_size).view(batch_size, -1).to(targets.device)
        
        class_logits[targets == 1] *= -1
        class_logits = torch.cat((zeros, class_logits), dim=1)

        loss = torch.logsumexp(class_logits, 1)

        return loss

### Wandb init function
def wandb_init(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        group=cfg.exp_group,
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=cfg.save_code,
        config=OmegaConf.to_container(cfg, resolve=False),
        reinit=True
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))
        
### Metrics class
class Metric:
    def __init__(self):
        self.cnt = 0
        self.val = 0
        self.total_val = 0
        self.mean = 0

    def add_val(self, val):
        self.cnt += 1
        self.total_val += val
        self.val = val
        self.mean = self.total_val / self.cnt        

### build model function
def build_model(cfg: DictConfig):
    if cfg.init_classifier == True:
        print("Building CLIP-ViT model with text-prompt classifier...")
        model = build_model(cfg)
    elif cfg.init_classifier == False:
        print("Building standard CLIP-ViT model...")
        model = build_model(cfg)

    return model
        
def build_criterion(cfg, train_loader):
    if cfg.loss.name == 'WBCE':
        pos_weights = train_loader.dataset.get_pos_weights()
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((pos_weights), device=train_loader.dataset.device))
    elif cfg.loss.name == 'Sign' or cfg.loss.name == 'SignLoss':
        return SignLoss()
    else:
        raise ValueError(f"Unknown loss: {cfg.loss.name}")

def build_optimizer(cfg, model):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.optim.base_lr, 
        weight_decay=cfg.optim.weight_decay)
    return optimizer

def build_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'CosineAnnealingWarmRestarts':
            return lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.scheduler.T_0,
                T_mult=cfg.scheduler.T_mult,
                eta_min=cfg.scheduler.eta_min
            )
    else:
        raise NotImplemented(f"Scheduler {cfg.scheduler.name} not implemented")

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
    
    
def _evaluate(cfg, model, loader, device):
    """
    Evaluates the model on validation set. Returns dict of metrics.
    Metrics: accuracy, precision, recall, f1, mAP
    """
    cfg = cfg
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            real_model = model.module if isinstance(model, DDP) else model

            logits = real_model(pixel_values=images)
            probs   = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            
    if dist.is_initialized():
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
