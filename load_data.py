import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import load_dataset
from omegaconf import DictConfig

class HFDataset(Dataset):
    """
    Wraps a Hugging Face Dataset for PyTorch DataLoader.
    Applies image transforms and returns (tensor, label).
    """
    def __init__(self, hf_dataset, transform: transforms.Compose):
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

def get_transform(cfg: DictConfig) -> transforms.Compose:
    t = cfg.data.transform
    ops = [
        transforms.RandomResizedCrop(
            t.size,
            scale=(t.scale[0], t.scale[1]),
            interpolation=getattr(InterpolationMode, t.interpolation)
        )
    ]
    if t.to_rgb:
        ops.append(transforms.Lambda(lambda img: img.convert("RGB")))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=tuple(t.normalize.mean),
            std=tuple(t.normalize.std)
        )
    ])
    return transforms.Compose(ops)

def get_transform2(cfg: DictConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.5, 1.0), ratio=(0.75, 1.33),
            interpolation=InterpolationMode.BICUBIC, antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    
def prepare_dataloader(dataset: Dataset, 
                        batch_size: int,
                        num_workers: int, 
                        shuffle: bool = False) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
        
def load_data(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # loads dataset
    dataset = load_dataset("Bingsu/Human_Action_Recognition")  # load your dataset

    ds_train = dataset["train"]
    # Create train/val split since official test labels are all zero
    split = ds_train.train_test_split(test_size=cfg.data.test_size, seed=cfg.data.seed, shuffle=cfg.data.shuffle)
    
    training = split["train"].train_test_split(test_size=cfg.data.test_size, seed=cfg.data.seed, shuffle=cfg.data.shuffle)
    train_ds = training["train"]
    val_ds = training["test"]
    
    ## split train_ds --> train and val
    test_ds   = split["test"]

    transform = get_transform(cfg)

    # Wrap in our PyTorch Dataset
    train_dataset = HFDataset(train_ds, transform)
    val_dataset   = HFDataset(val_ds,  transform)
    test_dataset   = HFDataset(test_ds,   transform)

    # loads test, validation, and training dataset
    train_data = prepare_dataloader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    val_data = prepare_dataloader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    test_data = prepare_dataloader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    return train_data, val_data, test_data