"""
scheduler.py
--------------
Optimizer and LR scheduler builders.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler



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
