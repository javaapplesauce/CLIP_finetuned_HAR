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

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def ddp_setup(rank, world_size):
    """
    peen
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356" 
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=10))

def gather_list(data_list):
    """
    Helper to gather a Python list from all processes.
    Returns a flat list on all ranks.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, data_list)
        # flatten
        flat = []
        for part in gathered:
            flat.extend(part)
        return flat
    else:
        return data_list

class SignLoss(nn.Module):
    def forward(self, class_logits, targets):
        assert class_logits.size() == targets.size(), "dimension mismatch"

        batch_size, n_classes = class_logits.size()
        zeros = torch.zeros(batch_size).view(batch_size, -1).to(targets.device)
        
        class_logits[targets == 1] *= -1
        class_logits = torch.cat((zeros, class_logits), dim=1)

        loss = torch.logsumexp(class_logits, 1)

        return loss

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

class FineTune:
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        lr: float,
        weight_decay: float,
        scheduler: optim.lr_scheduler,
        save_path: str = "best_model.pth"
    ) -> None:
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        self.scheduler = scheduler
        self.save_path = save_path
        self.best_acc = 0.0

    
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        logits  = output.logits
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        self.optimizer.step()
        

    
    def _run_epoch(self, epoch):
        criterion = SignLoss()
        b_sz = len(next(iter(self.train_data))[0])
        steps = len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {steps}")
        
        if self.gpu_id == 0:
            # Log LR at epoch start
            wandb.log({"train/lr": self.optimizer.param_groups[0]['lr']})
            print(f"Starting Epoch {epoch} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
        self.train_data.sampler.set_epoch(epoch)


        for it, (source, targets) in enumerate(self.train_data):
            
            start_time = time.time()
            
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            
            self.optimizer.zero_grad()
            output = self.model(source)
            logits  = output.logits
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes=15).float()
            
            loss_values = criterion(logits, targets_onehot.to(logits.device))
            loss = loss_values.mean()

            # loss = F.cross_entropy(logits, targets)
            loss.backward()            
            ema = torch.optim.swa_utils.AveragedModel(self.model)
            self.optimizer.step()
            ema.update_parameters(self.model)

            self.scheduler.step()
            
            batch_time = time.time() - start_time
            # — W&B: log training loss & step
            if self.gpu_id == 0 and it % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/batch_time": batch_time,
                    "train/epoch": epoch,
                })
            
            
                    
    def _save_checkpoint(self, epoch, best: bool = False):
        
        ckp = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": self.best_acc,
        }
        
        path = self.save_path if best else f"checkpoint_epoch{epoch}.pt"
        torch.save(ckp, path)

    
    @staticmethod
    def _evaluate(model, loader, device):
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
                
                outputs = model(images)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                
                
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            all_preds = gather_list(all_preds)
            all_labels = gather_list(all_labels)
            all_probs = gather_list(all_probs)
            rank_id = dist.get_rank()
        else:
            rank_id = 0
        if rank_id != 0:
            return None

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        total = len(all_labels)
        correct = (all_preds == all_labels).sum()
        acc = 100.0 * correct / total
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

        num_classes = len(np.unique(all_labels))
        APs = []
        labels_oneshot = np.zeros((total, num_classes), dtype=int)
        labels_oneshot[np.arange(total), all_labels] = 1
        for c in range(num_classes):
            ap_c = average_precision_score(labels_oneshot[:, c], np.array(all_probs)[:, c])
            if np.isnan(ap_c):
                ap_c = 0.0
            APs.append(ap_c)
        mAP = np.mean(APs) * 100
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mAP": mAP}
    
    
    
    def train(self, max_epochs: int):    
            
        for epoch in range(max_epochs):
            
            self._run_epoch(epoch)     
            
            metrics = self._evaluate(self.model, self.val_data, self.gpu_id)
            if self.gpu_id == 0 and metrics:
                print(
                    f"Val Accuracy: {metrics['acc']:.2f}% | "
                    f"Precision: {metrics['prec']:.2f}% | "
                    f"Recall: {metrics['rec']:.2f}% | "
                    f"F1: {metrics['f1']:.2f}% | "
                    f"mAP: {metrics['mAP']:.2f}%"
                )
                
                wandb.log({
                    "val/acc": metrics["acc"],
                    "val/prec": metrics["prec"],
                    "val/rec": metrics["rec"],
                    "val/f1": metrics["f1"],
                    "val/mAP": metrics["mAP"],
                })
                
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
                    
                if metrics['acc'] > self.best_acc:
                    self.best_acc = metrics['acc']
                    self._save_checkpoint(epoch, best=True)
                    
        
        
    
    def resume_train(self, checkpoint_path: str, max_epochs: int):
        
        ckp = torch.load(checkpoint_path, map_location=f"cuda:{self.gpu_id}")
        self.model.load_state_dict(ckp["model_state"])
        self.optimizer.load_state_dict(ckp["optimizer_state"])
        self.scheduler.load_state_dict(ckp["scheduler_state"])
        self.best_acc = ckp.get("best_acc", 0.0)
        start_epoch = ckp["epoch"] + 1
        if self.gpu_id == 0:
            print(f"Resuming from epoch {start_epoch} (best_acc={self.best_acc:.2f}%)")
        # Continue training
        self.train(max_epochs=max_epochs, start_epoch=start_epoch)



            
      
            

def load_train_objs(num_labels=15, lr=1e-5, weight_decay=4e-3, device=None, batch_size=48):
    
    # loads dataset
    ds = load_dataset("Bingsu/Human_Action_Recognition")  # load your dataset
    # loads model
    model = AutoModelForImageClassification.from_pretrained(
        "openai/clip-vit-base-patch32",
        num_labels=15)    
    
    
    base_lr       = lr           
    weight_decay  = weight_decay
    eta_min       = base_lr * 0.001 # floor LR = 1 % of base
    total_epochs  = 28


    cycle_epochs    = 4            
    # (20 epochs total → we’ll complete 3 cycles: epochs 2-7, 8-13, 14-19)

    steps_per_epoch = math.ceil(len(ds['train']) * 0.8 / batch_size)  # ≈ training batches
    warmup_iters   = int(0.2 * steps_per_epoch)
    cycle_iters    = 4 * steps_per_epoch

    head, base = [], []
    for n, p in model.named_parameters():
        (head if "classifier" in n else base).append(p)
    optimizer = optim.AdamW(
        [
            {"params": base, "lr": 0.5 * base_lr},
            {"params": head, "lr": 1.5 * base_lr},
        ],
        weight_decay  = weight_decay
    )

    # 1) linear warm-up
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor = 1e-2,       # start at 1 % of base_lr
        end_factor   = 1.0,        # reach base_lr
        total_iters  = warmup_iters
    )

    # 2) cosine-annealing warm restarts
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0     = cycle_iters,    # first cycle = 6 epochs
        T_mult  = 1,               # keep every cycle length the same
        eta_min = base_lr * 1e-4
    )

    # 3) chain them together
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup_scheduler, cosine_scheduler],
        milestones = [warmup_iters]
    )

    
    return ds, model, optimizer, scheduler



def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )



def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, 
         lr: float, weight_decay: float, resume_path: str | None = None, ):
        
    ddp_setup(rank, world_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Hyperparameters
    learning_rate = lr
    weight_decay = weight_decay
    
    ### loads raw dataset, CLIP model, and AdamW optimizer 
    dataset, model, optimizer, scheduler = load_train_objs(
        num_labels=15,
        lr=learning_rate,
        weight_decay=weight_decay,
        device=device,
        batch_size=batch_size
    )
    
    for n, p in model.named_parameters():
        if "text_model" in n:      # freeze text tower
            p.requires_grad_(False)
    
    ## delete text encoder from the model (?)
    
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
    
    if rank == 0:
        run_name = f"HAR-{datetime.now():%Y%m%d-%H%M%S}"
        wandb.init(
            project="HAR",      # or whatever slug you see in your URL
            group="plateau_fix", 
            tags=["lr=1.2e-5", "wd=4e-3"],
            config={                            # snap in your hyperparams
                "lr":       lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "save_every": save_every,
                "total_epochs": total_epochs
            }
        )
        wandb.watch(model, log="all", log_freq=100)
        wandb.watch_parameters = False
    
    trainer = FineTune(model, train_data, val_data, test_data, 
                       optimizer, rank, save_every, 
                       lr, weight_decay, scheduler, 
                       save_path="best_model.pth")
    
    try:
        if resume_path is not None:
            trainer.resume_train(resume_path, total_epochs)
        else:
            trainer.train(total_epochs)
        
    finally:
        destroy_process_group()
        
        # —— now that all DDP processes are torn down, run final test on rank 0
        if rank == 0:
            best_ckp = torch.load("best_model.pth", map_location="cuda:0", weights_only=False)
            # load into your non-DDP model
            trainer.model.load_state_dict(best_ckp['model_state'])
            
            test_metrics = trainer._evaluate(trainer.model, test_data, device="cuda:0")
            print(
                "running test|"
                f"{test_metrics['acc']:.2f}|"
                f"{test_metrics['prec']:.2f}|"
                f"{test_metrics['rec']:.2f}|"
                f"{test_metrics['f1']:.2f}|"
                f"{test_metrics['mAP']:.2f}|"
            )
            wandb.log({
                "test/acc": test_metrics['acc'],
                "test/mAP": test_metrics['mAP']
            })
        
    
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=48, type=int, help='Input batch size on each device (default: 48)')
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay")
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to a checkpoint (.pt) to resume training from",
    )

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,
                         args.save_every,
                         args.total_epochs,
                         args.batch_size,
                         args.lr,
                         args.weight_decay,
                         args.resume), nprocs=world_size)