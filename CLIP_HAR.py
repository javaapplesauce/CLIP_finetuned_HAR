import torch.nn as nn
from transformers import CLIPModel
import torch





class CLIP_base(nn.Module):
    def __init__(self, cfg, num_labels: int):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(cfg.model.backbone)
        dim = self.clip.config.projection_dim  # usually 512
        self.head = nn.Linear(dim, num_labels)

    def forward(self, pixel_values):
        with torch.set_grad_enabled(self.training):
            feats = self.clip.get_image_features(pixel_values=pixel_values)  # (B, dim)
        # optional: L2-normalize like CLIPViT if you want cosine-style logits
        # feats = feats / feats.norm(dim=-1, keepdim=True)
        # w = self.head.weight / self.head.weight.norm(dim=-1, keepdim=True)
        # return feats @ w.t() + self.head.bias
        return self.head(feats)  # (B, num_labels), raw logits







def build_model(cfg):
    
    if cfg.init_classifier == True:
        return CLIP_init(cfg)
    elif cfg.init_classifier == False:
        return CLIP_base(cfg)