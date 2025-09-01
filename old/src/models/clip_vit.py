

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

"""
clip_vit.py
------------
Wrapper around Hugging Face CLIP ViT for image embeddings,
with support for custom classifiers and text prototypes.
"""
class CLIPViT(nn.Module):
    def __init__(self, cfg, class_names):
        super().__init__()
        self.cfg = cfg
        self.class_names = class_names

        self.clip = CLIPModel.from_pretrained(cfg.model.backbone)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.backbone)

        self.return_hidden = cfg.model.return_hidden
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Custom classifier (trainable prototypes)
        self.classifier = nn.Parameter(torch.randn(len(class_names), self.clip.config.projection_dim))

    def encode_image_hidden(self, pixel_values):
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output  # (B, hidden_dim)

    def forward(self, pixel_values):
        if self.return_hidden:
            img = self.encode_image_hidden(pixel_values)  # (B, hidden_dim)
        else:
            img = self.clip.get_image_features(pixel_values)

        img = img / img.norm(dim=-1, keepdim=True)
        txt = self.classifier / self.classifier.norm(dim=-1, keepdim=True)

        logits = torch.exp(self.logit_scale) * img @ txt.t()
        return logits
