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


def ddp_setup(rank, world_size):
    """
    peen
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" 
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def gather_list(data_list):
    """
    Helper to gather a Python list from all processes.
    Returns a flat list on all ranks.
    """
    world_size = dist.get_world_size()
    gathered = [None] * world_size
    # PyTorch ≥1.8: gather arbitrary Python objects
    dist.all_gather_object(gathered, data_list)
    # flatten
    flat = []
    for part in gathered:
        flat.extend(part)
    return flat

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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

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
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
        
    def _save_checkpoint(self, epoch, best: bool = False):
        
        ckp = {
            'model_state': self.model.state_dict(),
        }
        
        path = self.save_path if best else f"checkpoint_epoch{epoch}.pt"
        torch.save(ckp, path)

    
    @staticmethod
    def _evaluate(model, loader, device, rank):
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
                
        all_preds = gather_list(all_preds)
        all_labels = gather_list(all_labels)
        all_probs = gather_list(all_probs)
        
        if rank != 0:
            return None

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
    
    
    
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            
            self._run_epoch(epoch)
            self.scheduler.step()
            
            metrics = self._evaluate(self.model, self.val_data, self.gpu_id, self.gpu_id)
            if self.gpu_id == 0:
                print(
                    f"Val Accuracy: {metrics['acc']:.2f}% | "
                    f"Precision: {metrics['prec']:.2f}% | "
                    f"Recall: {metrics['rec']:.2f}% | "
                    f"F1: {metrics['f1']:.2f}% | "
                    f"mAP: {metrics['mAP']:.2f}%"
                )
                
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)
                    
                if metrics['acc'] > self.best_acc:
                    self.best_acc = metrics['acc']
                    self._save_checkpoint(epoch, best=True)
                    
        if self.gpu_id == 0:
            map_location = {'cuda:0': f'cuda:{self.gpu_id}'} if torch.cuda.is_available() else 'cpu'
            
            checkpoint = torch.load('best_model.pth', map_location=map_location)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(map_location)
            self.model.eval()
            
            # 2. Prepare containers for predictions & labels
            all_preds = []
            all_labels = []
            
            # 3. Inference loop
            with torch.no_grad():
                for images, labels in self.test_data:
                    # move to device
                    images = images.to(map_location)
                    labels = labels.to(map_location)

                    # forward pass
                    outputs = self.model(images)
                    # if using a Hugging Face-style model, grab .logits
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                    # get predicted class
                    preds = torch.argmax(logits, dim=1)

                    # collect
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            # 4. Concatenate and compute metrics
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            metrics = self._evaluate(self.model, self.test_data, map_location, self.gpu_id)
            print(
                f"test Accuracy: {metrics['acc']:.2f}% | "
                f"Precision: {metrics['prec']:.2f}% | "
                f"Recall: {metrics['rec']:.2f}% | "
                f"F1: {metrics['f1']:.2f}% | "
                f"mAP: {metrics['mAP']:.2f}%"
            )

            
            
            
            
            


            

def load_train_objs(num_labels=15, lr=1e-4, weight_decay=1e-4, device=None):
    
    # loads dataset
    ds = load_dataset("Bingsu/Human_Action_Recognition")  # load your dataset

    # loads model
    model = AutoModelForImageClassification.from_pretrained(
        "openai/clip-vit-base-patch32",
        num_labels=15)    
    
    
    
    # loads optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # loads scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    
    return ds, model, optimizer, scheduler



def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )



def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, 
         lr: float, weight_decay: float):
    
    ddp_setup(rank, world_size)
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
    
    trainer = FineTune(model, train_data, val_data, test_data, 
                       optimizer, rank, save_every, 
                       lr, weight_decay, scheduler, 
                       save_path="best_model.pth")
    
    try:
        trainer.train(total_epochs)
    finally:
        destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=48, type=int, help='Input batch size on each device (default: 48)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, 
                         args.save_every, 
                         args.total_epochs, 
                         args.batch_size, 
                         5e-5,
                         1e-2), nprocs=world_size)