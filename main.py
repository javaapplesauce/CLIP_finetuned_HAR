import zeroshot
import finetune

import os
import clip
import torch
from datasets import load_dataset, DownloadConfig
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import torch.nn as nn


class HFDataset(Dataset):
    def __init__(self, hf_dataset, preprocess):
        self.dataset   = hf_dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item  = self.dataset[idx]
        # HF field is called "labels" in this split :contentReference[oaicite:0]{index=0}
        label = item["labels"]
        img   = item["image"].convert("RGB")
        img_t = self.preprocess(img)
        return img_t, label

if __name__ == '__main__':
    
    ### Load the dataset --> installed datasets and vision dependency
    dl_config = DownloadConfig(max_retries=10, use_etag=False)
    ds = load_dataset("Bingsu/Human_Action_Recognition", download_config=dl_config)
    ds_train, ds_test = finetune.split_dataset(ds)

    
    ### model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-B/32', device)
    model.to(device)
    
    train_ds = HFDataset(ds_train, preprocess)
    test_ds  = HFDataset(ds_test,  preprocess)

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(finetune.HARdataset(train_ds), batch_size=32, shuffle=True) # why is the batch size 32
    test_loader = DataLoader(finetune.HARdataset(test_ds), batch_size=32, shuffle=False)
    
    num_classes = 15
    model_ft = finetune.CLIPFineTuner(model, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

    trained_model = finetune.train_and_validate(
    model_ft, train_loader, test_loader,
    optimizer, criterion,
    device, num_epochs=5, save_path="clip_finetuned_HAR.pth")
    
    



