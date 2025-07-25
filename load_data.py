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

class HFDataset(Dataset):
    """
    Wraps a Hugging Face Dataset for PyTorch DataLoader.
    Applies image transforms and returns (tensor, label).
    """
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"].convert("RGB")
        img_tensor = self.transform(img)
        label = item["labels"]  # integer label 0-14
        return img_tensor, label
    
    @staticmethod
    def transform():
        
        return transforms.Compose([

            transforms.RandomResizedCrop(
                224,
                scale=(0.9, 1.0),               # default in PyTorch
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        
        # return transforms.Compose([
        #     transforms.RandomResizedCrop(
        #         224,
        #         scale=(0.08, 1.0),               # default in PyTorch
        #         ratio=(0.75, 1.3333333),
        #         interpolation=InterpolationMode.BICUBIC
        #     ),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ToTensor(),
        #     transforms.Normalize(CLIP_MEAN, CLIP_STD),
        # ])

        # base = [
        #     transforms.Resize(256),
        #     transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,)*3, (0.5,)*3),
        # ]
        # return transforms.Compose(base)

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )

def load_data():

    # loads dataset
    ds = load_dataset("Bingsu/Human_Action_Recognition")  # load your dataset

    ds_train = dataset["train"]
    # Create train/val split since official test labels are all zero
    split = ds_train.train_test_split(test_size=0.2, seed=42, shuffle=True)
    
    training = split["train"].train_test_split(test_size=0.2, seed=42, shuffle=True)
    train_ds = training["train"]
    val_ds = training["test"]
    
    ## split train_ds --> train and val
    test_ds   = split["test"]

    # Wrap in our PyTorch Dataset
    train_dataset = HFDataset(train_ds, HFDataset.transform())
    val_dataset   = HFDataset(val_ds,   HFDataset.transform())
    test_dataset   = HFDataset(test_ds,   HFDataset.transform())

    # loads test, validation, and training dataset
    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, batch_size)

    return train_data, val_data, test_data