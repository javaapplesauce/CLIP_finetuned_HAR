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
    ds = load_dataset("Bingsu/Human_Action_Recognition")  # load your dataset

    ds_train = dataset["train"]
    # Create train/val split since official test labels are all zero
    split = ds_train.train_test_split(test_size=cfg.data.test_size, seed=cfg.data.seed, shuffle=cfg.data.shuffle)
    
    training = split["train"].train_test_split(test_size=cfg.data.test_size, seed=cfg.data.seed, shuffle=cfg.data.shuffle)
    train_ds = training["train"]
    val_ds = training["test"]
    
    ## split train_ds --> train and val
    test_ds   = split["test"]

    # Wrap in our PyTorch Dataset
    train_dataset = HFDataset(train_ds, HFDataset.transform())
    val_dataset   = HFDataset(val_ds,   HFDataset.transform())
    test_dataset   = HFDataset(test_ds,   HFDataset.transform())

    # loads test, validation, and training dataset
    train_data = prepare_dataloader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    val_data = prepare_dataloader(val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    test_data = prepare_dataloader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    return train_data, val_data, test_data