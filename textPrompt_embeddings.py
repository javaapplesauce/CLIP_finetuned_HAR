import torch
import torch.distributed as dist
from torch import nn
from omegaconf import DictConfig
from transformers import AutoModelForImageClassification, CLIPModel, CLIPTokenizer
import math
import torch.nn.functional as F



# @torch.no_grad()
# def _build_text_prototypes(backbone: str, class_names: list[str], device: torch.device) -> torch.Tensor:
    
#     tok  = CLIPTokenizer.from_pretrained(backbone)
#     clip = CLIPModel.from_pretrained(backbone)
    
#     clip.to(device)
#     clip.eval()

#     TEMPLATES = [
#         "a photo of a person {}.",
#         "a person is {}.",
#         "human action: {}.",
#         "someone is {}.",
#     ]
#     texts = [t.format(c.replace("_", " ")) for c in class_names for t in TEMPLATES]
#     per   = len(TEMPLATES)

#     inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
#     feats  = clip.get_text_features(**inputs)                         # (C*per, D=projection_dim)
#     feats  = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

#     C = len(class_names)
#     W_text = torch.stack([feats[i*per:(i+1)*per].mean(0) for i in range(C)], dim=0)  # (C, D)
#     W_text = W_text / (W_text.norm(dim=-1, keepdim=True) + 1e-12)                    # L2 norm
#     return W_text, clip  # keep clip to access visual_projection if needed






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
        self.HvProjection = cfg.model.HvProjection
        
        # build text weights
        TEMPLATES = [
                "a person {}.",
                "someone is {}.",
                "a photo of a person {}.",
                # "human action: {}.",
                # "a person is {}.",
                # "a photo of someone {}.",
                # "a person engaged in {}.",
                # "an image of a person {}.",
                
        ]
        texts = [t.format(c.replace("_"," ")) for c in class_names for t in TEMPLATES]
        per   = len(TEMPLATES)
        
        with torch.no_grad():
            inputs = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
            text_features  = self.clip.get_text_features(**inputs)                         # (N*per, D=projection_dim)
            # average per-class
            text_features = text_features.reshape(len(class_names), per, -1).mean(dim=1)  # (N, D)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)     # re-normalize
            # text_features --> T_emb @ T_proj
            
        if self.HvProjection:
            
            ### goal: [V_proj @ (T_emb @ T_proj)^T)]^T --> [(768, 512) @ ((N,768)@(768,512))^T]^T
            
            # V_proj^T
            VprojT = self.clip.visual_projection.weight.t() 
            Wv = (VprojT @ text_features.t()).t()
            
            # Wv = F.normalize(Wv, dim=-1) # do i need to normalize here? probably not 

            self.classifier = nn.Parameter(Wv, requires_grad=True)
            self.return_hidden = True
        else:
            
            ### goal: (V_cls @ V_proj) @ (T_emb @ T_proj)
            
            self.classifier = nn.Parameter(text_features)
            self.return_hidden = False
            
            
        ### Freezing practice here
        
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
        if self.return_hidden:
            img = self.encode_image_hidden(pixel_values)            # (B, 768)
            txt = self.classifier / self.classifier.norm(dim=-1, keepdim=True)
            logits = torch.exp(self.logit_scale) * img @ txt.t()        # (B, ncls)

        elif (self.return_hidden == False):
            img = self.clip.get_image_features(pixel_values)        # (B, 512)
            txt = self.classifier / self.classifier.norm(dim=-1, keepdim=True)
            logits = torch.exp(self.logit_scale) * img @ txt.t()        # (B, ncls)

        
        return logits    