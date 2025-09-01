import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel, CLIPTokenizer

class CLIPViT(nn.Module):
    """
    Prompt-ensemble over TEXT embedding space (CLIP-style).
    - Build T template embeddings per class each forward (grads flow into text tower).
    - Average (optionally learn weights) in embedding space -> per-class prototype.
    - Re-normalize prototypes.
    - Score against image features in either:
        * projection space (default), or
        * vision hidden space via HvProjection (maps text proto proj->hidden).
    """
    def __init__(self, cfg, class_names):
        super().__init__()
        assert class_names is not None
        self.class_names = class_names
        self.clip = CLIPModel.from_pretrained(cfg.model.backbone)
        self.tok  = CLIPTokenizer.from_pretrained(cfg.model.backbone)

        self.logit_scale   = nn.Parameter(torch.log(torch.tensor(1.0 / cfg.model.s)))
        self.HvProjection  = bool(cfg.model.HvProjection)  # True => compare in hidden space
        self.return_hidden = self.HvProjection

        TEMPLATES = [
            "someone {}.",
            "someone is {}.",
            "a person {}.",
            "a person is {}.",
            "human action: {}.",
            "an image of a person {}.",
        ]
        N   = len(class_names)
        per = len(TEMPLATES)
        self.per = per

        # tokenize once
        texts = [t.format(c.replace("_"," ")) for c in class_names for t in TEMPLATES]  # len = N*per
        toks  = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
        L = toks["input_ids"].shape[-1]
        self.register_buffer("input_ids",      toks["input_ids"].view(N, per, L),      persistent=False)
        self.register_buffer("attention_mask", toks["attention_mask"].view(N, per, L), persistent=False)

        # learnable template weights PER CLASS (logits). Set requires_grad_(False) for uniform averaging.
        self.template_logits_w = nn.Parameter(torch.zeros(N, per))  # (N, per)

    # ---- vision hidden path (trainable)
    def encode_image_hidden(self, pixel_values):
        vis_model = self.clip.vision_model
        with torch.set_grad_enabled(self.training):
            outputs = vis_model(pixel_values=pixel_values, output_hidden_states=False)
            pooled  = outputs.pooler_output                 # (B, hidden_dim)
            pooled  = F.normalize(pooled, dim=-1)
        return pooled

    def _class_prototypes_text_space(self):
        """
        Recompute text features each forward so grads flow into text tower.
        Returns L2-normalized per-class prototypes in projection space: (N, D_proj)
        """
        N, per, L = self.input_ids.shape
        ids  = self.input_ids.view(-1, L)           # (N*per, L)  [device follows module]
        mask = self.attention_mask.view(-1, L)      # (N*per, L)

        with torch.set_grad_enabled(self.training):
            tf = self.clip.get_text_features(input_ids=ids, attention_mask=mask)  # (N*per, D_proj)
        tf = F.normalize(tf, dim=-1).view(N, per, -1)                             # (N, per, D_proj)

        w  = torch.softmax(self.template_logits_w, dim=1).unsqueeze(-1)           # (N, per, 1)
        proto = F.normalize((w * tf).sum(dim=1), dim=-1)                          # (N, D_proj)
        return proto

    def forward(self, pixel_values):
        eps = 1e-12
        s   = torch.exp(self.logit_scale)

        # (N, D_proj)
        text_proto = self._class_prototypes_text_space()

        if self.return_hidden:
            # Map text prototypes proj->hidden with visual_projection.weight (proj_dim, hidden_dim)
            W  = self.clip.visual_projection.weight              # (D_proj, D_hidden)
            Wv = F.normalize(text_proto @ W, dim=-1)             # (N, D_hidden)

            img = self.encode_image_hidden(pixel_values)         # (B, D_hidden), normalized
            logits = s * (img @ Wv.t())                          # (B, N)
            return logits

        else:
            img = self.clip.get_image_features(pixel_values=pixel_values)  # (B, D_proj)
            img = img / (img.norm(dim=-1, keepdim=True) + eps)
            logits = s * (img @ text_proto.t())                            # (B, N)
            return logits
