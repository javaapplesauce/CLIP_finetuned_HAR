import zeroshot
import finetune

import os
import clip
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import torch.nn as nn


tfms = Compose([Resize(224), ToTensor(),
                Normalize(mean=(0.481,0.457,0.408),
                          std=(0.268,0.261,0.275))])      # CLIP defaults
def collate(batch):
    sample0 = batch[0]
    if isinstance(sample0, dict):
        # old path: each x is a dict with keys "image" and "label"
        images = torch.stack([tfms(x["image"].convert("RGB")) for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
    else:
        # new path: each x is a tuple, e.g. (PIL.Image, label) or (tensor, label)
        images = torch.stack([
            tfms(x[0].convert("RGB")) if not torch.is_tensor(x[0])
            else x[0]
            for x in batch
        ])
        labels = torch.tensor([x[1] for x in batch])
    return images, labels

if __name__ == '__main__':
    
    ### Load the dataset --> installed datasets and vision dependency
    ds = load_dataset("Bingsu/Human_Action_Recognition")
    dataset = ds['train']
    dataset_test = ds["test"]
    
    ### model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-B/32', device)
    model.to(device)
    
    ds_train, ds_test = finetune.split_dataset(dataset)

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(finetune.HARdataset(ds_train), batch_size=32, shuffle=True, collate_fn=collate) # why is the batch size 32
    test_loader = DataLoader(finetune.HARdataset(ds_test), batch_size=32, shuffle=False, collate_fn=collate)
    
    num_classes = 15
    model_ft = finetune.CLIPFineTuner(model, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

    trained_model = finetune.train_and_validate(
    model_ft, train_loader, test_loader,
    optimizer, criterion,
    device, num_epochs=5, save_path="clip_finetuned_HAR.pth")
    
    



