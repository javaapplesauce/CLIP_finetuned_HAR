import os
import clip
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm




def split_dataset(dataset, *, test_size=0.2, seed=None, shuffle=True):
    # Split dataset into training and validation sets
    split = dataset.train_test_split(
        test_size=test_size,
        shuffle=shuffle,
        seed=seed
    )

    return split["train"], split["test"]



# Define a custom dataset class
class HARdataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) # this needs to get changed?
        ])

    def __len__(self): # make sure to look up why we do this
        return len(self.data)

    def __getitem__(self, idx):
        # def a way to make this more efficient
        class_to_label = {
        'calling': 0,
        'clapping': 1,
        'cycling': 2,
        'dancing': 3,
        'drinking': 4,
        'eating': 5,
        'fighting': 6,
        'hugging': 7,
        'laughing': 8,
        'listening_to_music': 9,
        'running': 10,
        'sitting': 11,
        'sleeping': 12,
        'texting': 13,
        'using_laptop': 14
        }
        label_to_class = {v: k for k, v in class_to_label.items()}
        
        item = self.data[idx]
        image = item['image']
        label_id  = item["labels"]
        actual_class = label_to_class[label_id]  # look up the string name
        return self.transform(image), actual_class



# Modify the model to include a classifier for subcategories --> figure this one out
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)



def train_and_validate(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    num_epochs=5,
    save_path="clip_finetuned_HAR.pth"):
    
    """
    Train and validate a PyTorch model, showing a tqdm progress bar and printing stats.

    Args:
        model (torch.nn.Module):        The model to train/evaluate.
        train_loader (DataLoader):      DataLoader for training data.
        test_loader (DataLoader):        DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (callable):           Loss function.
        device (torch.device or str):   Device to run on ('cpu' or 'cuda').
        num_epochs (int, default=5):    Number of epochs to train.
        save_path (str, default="clip_finetuned.pth"):
                                         Where to save the final model weights.

    Returns:
        torch.nn.Module: The fine-tuned model (in eval mode).
    """
    model.to(device)

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()            # step into this would be cool

            running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_loss:.4f}")

        # --- Validation phase ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {acc:.2f}%\n")

    # Save final model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to: {save_path}")

    # Return the model in evaluation mode
    model.eval()
    return model



if __name__ == '__main__':
    
    ### Load the dataset --> installed datasets and vision dependency
    ds = load_dataset("Bingsu/Human_Action_Recognition")
    dataset = ds['train']
    dataset_test = ds["test"]
    
    ### model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-B/32', device)
    model.to(device)
    
    
