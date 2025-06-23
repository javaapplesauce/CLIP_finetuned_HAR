import os
import clip
import torch
from datasets import load_dataset

### Load the dataset --> installed datasets and vision dependency
ds = load_dataset("Bingsu/Human_Action_Recognition")

dataset = ds['train']


### Load the models


model, preprocess = clip.load("ViT-B/32", jit=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


## test zero-shot prediction (baseline)


### dataset preprocessing


### loss


### Fine-Tuning CLIP Model


## retest performance 