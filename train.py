import os

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    ReduceLROnPlateau,
    SequentialLR,
    LinearLR
)
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
    model = {
        'CLIP-ViT-B': build_model_clip,
        'CLIP-ViT-L': build_model_clip,
    }[cfg.model.backbone](cfg)
    return model


### build the criterion


### build the optimizer


### build the scheduler


### data loading --> load_data.py


### training loop


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
    
    train_data, val_data, test_data = load_data.load_data()

    trainer = Trainer(cfg, cfg.local_rank)
    trainer.train()
