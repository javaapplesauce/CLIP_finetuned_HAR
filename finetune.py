import os
import clip
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split



def split_dataset(dataset, *, test_size=0.2, seed=None, shuffle=True):
    # Split dataset into training and validation sets
    split = dataset["train"].train_test_split(
        test_size=test_size,
        shuffle=shuffle,
        seed=seed
    )

    return split["train"], split["test"]






if __name__ == '__main__':
    
    ### Load the dataset --> installed datasets and vision dependency
    ds = load_dataset("Bingsu/Human_Action_Recognition")
    dataset = ds['train']
    dataset_test = ds["test"]
    
    ### model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-B/32', device)
    model.to(device)
    
    
