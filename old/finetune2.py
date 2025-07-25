import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, DownloadConfig
from collections import Counter


# Metrics imports
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

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





def get_data_loaders(batch_size=64, num_workers=4, val_split=0.2, seed=42):
    
    # Image transforms matching ViT-B/32 expectations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
  
    # Load dataset
    ds = load_dataset("Bingsu/Human_Action_Recognition")  # no special download config needed
    ds_train = ds["train"]
    
    ds_test = ds["test"]
    ds_test_wrapped = HFDataset(ds_test,   transform)
    ds_test_loader = DataLoader(
        ds_test_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    labels = [example["labels"] for example in ds["test"]]
    print("Test split label counts:", Counter(labels))

    # Create train/val split since official test labels are all zero
    split = ds_train.train_test_split(test_size=val_split, seed=seed, shuffle=True)
    
    training = split["train"].train_test_split(test_size=val_split, seed=seed, shuffle=True)
    train_ds = training["train"]
    val_ds = training["test"]
    
    ## split train_ds --> train and val
    test_ds   = split["test"]

    # Wrap in our PyTorch Dataset
    train_dataset = HFDataset(train_ds, transform)
    val_dataset   = HFDataset(val_ds,   transform)
    test_dataset   = HFDataset(test_ds,   transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,            # **SINGLE-GPU: shuffle only for train**
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, ds_test_loader


def build_model(num_labels=15, lr=1e-4, weight_decay=1e-4, device=None):
    
    from transformers import AutoModelForImageClassification

    # Load pre-trained ViT and adapt for our 15 classes
    model = AutoModelForImageClassification.from_pretrained(
        "openai/clip-vit-base-patch32",
        num_labels=num_labels
    )
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (AdamW for decoupled weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Scheduler: step LR decay every 2 epochs by factor of 0.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    #scheduler = CosineAnnealingLR(
    #    optimizer,
    #    T_max = 5,   # one cosine cycle over all epochs
    #    eta_min = 1e-6        # floor LR
    #)
    
    return model, optimizer, criterion, scheduler


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs training for one epoch. Returns average loss.
    """
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on validation set. Returns dict of metrics.
    Metrics: accuracy, precision, recall, f1, mAP
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    total = len(all_labels)
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    acc = 100.0 * correct / total
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    # mAP: average precision per class then mean
    num_classes = len(set(all_labels))
    labels_onehot = np.zeros((total, num_classes), dtype=int)
    labels_onehot[np.arange(total), all_labels] = 1
    APs = []
    for c in range(num_classes):
        ap_c = average_precision_score(labels_onehot[:, c], np.array(all_probs)[:, c])
        APs.append(ap_c)
    mAP = np.mean(APs) * 100

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mAP": mAP}


def save_checkpoint(state, filename):
    """
    Saves training state dict to given filename.
    """
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=None):
    """
    Loads training state from checkpoint. Returns epoch.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint.get('epoch', 0)
