import os

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModelForImageClassification
import torch.distributed as dist

# Multiprocess imports
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Dataset imports
from datasets import load_dataset, DownloadConfig
from collections import Counter

# Metrics imports
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

from finetune_multiGPU import load_train_objs, HFDataset, prepare_dataloader, FineTune
from collections import OrderedDict


def evaluate(model, loader, device):
        """
        Evaluates the model on validation set. Returns dict of metrics.
        Metrics: accuracy, precision, recall, f1, mAP
        """
        model.eval()
        all_preds, all_labels, all_probs = [], [], []


        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images).logits
                probs = torch.softmax(logits, dim=1)
                
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        # Compute metrics
        total = len(all_labels)
        correct = np.sum(np.array(all_preds) == np.array(all_labels))
        acc = 100.0 * correct / total
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

        #   mAP: average precision per class then mean
        num_classes = len(set(all_labels))
        labels_onehot = np.zeros((total, num_classes), dtype=int)
        labels_onehot[np.arange(total), all_labels] = 1
        APs = []
        for c in range(num_classes):
            ap_c = average_precision_score(labels_onehot[:, c], np.array(all_probs)[:, c])
            APs.append(ap_c)
        mAP = np.mean(APs) * 100

        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mAP": mAP}
    

def main(batch_size: int):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### Hyperparameters
    learning_rate = 5e-5
    weight_decay = 1e-2
    
    ### loads raw dataset, CLIP model, and AdamW optimizer 
    dataset, model, optimizer, scheduler = load_train_objs(
        num_labels=15,
        lr=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    
    ds_train = dataset["train"]
    # Create train/val split since official test labels are all zero
    split = ds_train.train_test_split(test_size=0.2, seed=42, shuffle=True)
    
    ## split train_ds --> train and val
    test_ds   = split["test"]
    test_dataset   = HFDataset(test_ds,   HFDataset.transform())
    test_data = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,            # **SINGLE-GPU: shuffle only for train**
        pin_memory=True
    )

    # Wrap in our PyTorch Dataset

    state_dict = torch.load('best_model.pth', map_location=device)
    raw_sd = state_dict["model_state"]
    
    stripped_sd = OrderedDict()
    for k, v in raw_sd.items():
        new_key = k.replace("module.", "", 1)  # only strip the first occurrence
        stripped_sd[new_key] = v
    
    model.load_state_dict(stripped_sd)
    model.to(device)
    
    metrics = FineTune._evaluate(model, test_data, device)
    
    print(
        f"Test Accuracy: {metrics['acc']:.2f}% | "
        f"Precision: {metrics['prec']:.2f}% | "
        f"Recall: {metrics['rec']:.2f}% | "
        f"F1: {metrics['f1']:.2f}% | "
        f"mAP: {metrics['mAP']:.2f}%"
    )
    
if __name__ == "__main__":
    main(48)