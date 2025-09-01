import torch
import torch.distributed as dist
from torch import nn
from omegaconf import DictConfig
from transformers import AutoModelForImageClassification, CLIPModel, CLIPTokenizer
import math
import torch.nn.functional as F

class CLIPViT(nn.Module):
    """
        Inputs:
            class_names: list with multiple class names
            backbone: from configs
            HvProjection: toggle to either project into Vision embedding space or to do standard
                          CLIP implmentation, from configs
            s: equivalent to temperature, from configs    
    """
    
    def __init__(self, cfg, class_names=None, ):
        super().__init__()
        assert class_names is not None
        self.class_names = class_names
        self.clip = CLIPModel.from_pretrained(cfg.model.backbone)
        self.tok = CLIPTokenizer.from_pretrained(cfg.model.backbone)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0/cfg.model.s)))
        self.HvProjection = bool(cfg.model.HvProjection)

        # build text weights
        TEMPLATES = [
                # "someone {}.",
                # "someone is {}.",
                # "a person {}.",
                # "a person is {}.",
                # "human action: {}.",
                "an image of a person {}."
                
        ]
        texts = [t.format(c.replace("_"," ")) for c in class_names for t in TEMPLATES]
        per   = len(TEMPLATES)
        N = len(class_names)
        eps   = 1e-12
        
        with torch.no_grad():
            assert len(texts) == N * per
            inputs = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
            text_features  = self.clip.get_text_features(**inputs)                         # (N*per, D=projection_dim)
            # average per-class
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)     # re-normalize
            text_features = text_features.view(N, per, -1)
            # text_features --> T_emb @ T_proj

        self.text_features_per = nn.Parameter(text_features.clone())
        self.template_logits_w  = nn.Parameter(torch.zeros(N, per)) # (N, per)
        
        self.per = per
        self.return_hidden = self.HvProjection

        
        
    ### Freezing practice here
    @torch.no_grad()
    def encode_image_hidden(self, pixel_values):
        """
        Return ViT hidden [CLS] before projection (B, 768)
        """
        # --> need to isolate V_cls
        
        vis_model = self.clip.vision_model
        outputs = vis_model(pixel_values=pixel_values, output_hidden_states=False)
        pooled = outputs.pooler_output
        pooled = F.normalize(pooled, dim=-1)
        return pooled

    def forward(self, pixel_values):
        
        eps = 1e-12
        per = self.per
        N   = len(self.class_names)
        s   = torch.exp(self.logit_scale)
        
        tf_per = F.normalize(self.text_features_per, dim=-1)
        
        if self.return_hidden:
            img = self.encode_image_hidden(pixel_values)            # (B, 768)
            
            VprojT = self.clip.visual_projection.weight.t() 
            Wv_per = torch.einsum("md,npd->npm", VprojT, tf_per)
            Wv_per = F.normalize(Wv_per, dim=-1)
            logits_per = torch.einsum("bd,npd->bpn", img, Wv_per)

        else:
            img = self.clip.get_image_features(pixel_values)        # (B, 512)
            img = img / (img.norm(dim=-1, keepdim=True) + eps)
            
            logits_per = torch.einsum("bd,npd->bpn", img, tf_per)
            
        w = torch.softmax(self.template_logits_w, dim=1)           # (N, per)
        w = w.transpose(0, 1).unsqueeze(0)    
        
        logits = s * (logits_per * w).sum(dim=1)
        return logits