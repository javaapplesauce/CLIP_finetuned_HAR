"""
criterion.py
-------------
Implements BCE and Log-Sum-Exp Sign loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, class_logits, targets, cfg):
        assert class_logits.size() == targets.size(), "dimension mismatch"

        batch_size, num_classes = class_logits.size()
        zeros = torch.zeros(batch_size).view(batch_size, -1).to(targets.device)
        
        class_logits[targets == 1] *= -1
        class_logits = torch.cat((zeros, class_logits), dim=1)

        loss = torch.logsumexp(class_logits, 1)

        return loss

def build_criterion(cfg):
    if cfg.loss.name == 'BCE':
        return nn.BCEWithLogitsLoss()
    elif cfg.loss.name == 'Sign' or cfg.loss.name == 'SignLoss':
        return SignLoss()
    else:
        raise ValueError(f"Unknown loss: {cfg.loss.name}")
