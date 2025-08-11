import torch
import torch.distributed as dist
from torch import nn
from omegaconf import DictConfig
from transformers import AutoModelForImageClassification, CLIPModel, CLIPTokenizer




@torch.no_grad()
def _build_text_prototypes(backbone: str, class_names: list[str], device: torch.device) -> torch.Tensor:
    
    tok  = CLIPTokenizer.from_pretrained(backbone)
    clip = CLIPModel.from_pretrained(backbone)
    
    clip.to(device)
    clip.eval()

    TEMPLATES = [
        "a photo of a person {}.",
        "a person is {}.",
        "human action: {}.",
        "someone is {}.",
    ]
    texts = [t.format(c.replace("_", " ")) for c in class_names for t in TEMPLATES]
    per   = len(TEMPLATES)

    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats  = clip.get_text_features(**inputs)                         # (C*per, D=projection_dim)
    feats  = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)

    C = len(class_names)
    W_text = torch.stack([feats[i*per:(i+1)*per].mean(0) for i in range(C)], dim=0)  # (C, D)
    W_text = W_text / (W_text.norm(dim=-1, keepdim=True) + 1e-12)                    # L2 norm
    return W_text, clip  # keep clip to access visual_projection if needed


def init_model(cfg: DictConfig, class_names: list[str]) -> nn.Module:

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone    = cfg.model.backbone
    num_cls     = int(cfg.num_classes)
    do_text_init= bool(cfg.joint_embed)
    alpha       = float(getattr(cfg.model, "alpha", 1.0))

    # 1) Build the image classification model
    model = AutoModelForImageClassification.from_pretrained(
        backbone,
        num_labels=num_cls,
    ).to(device)

    if not do_text_init:
        return model

    # 2) Determine target feature size of the classifier head
    head: nn.Linear = model.classifier
    in_f = head.in_features  # what the classifier expects

    # 3) Rank-0 computes the appropriate init matrix; others receive via broadcast
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0:
        W_text, clip_full = _build_text_prototypes(backbone, class_names, device)  # (C, D)

        # D = projection_dim; H = pre-projection hidden size
        D = W_text.shape[1]
        H = clip_full.visual_projection.weight.shape[1]  # (D, H)

        if in_f == D:
            W_init = W_text
        elif in_f == H:
            # Map prototypes from projection space back to hidden space
            V = clip_full.visual_projection.weight  # (D, H)
            W_init = W_text @ V                     # (C, H)
        else:
            raise RuntimeError(
                f"Classifier in_features={in_f} does not match CLIP projection_dim={D} "
                f"or hidden_size={H}."
            )

        if alpha is not None and alpha > 0:
            W_init = alpha * W_init

        W_buf = W_init.to(torch.float32).contiguous()
    else:
        W_buf = torch.empty((num_cls, in_f), device=device, dtype=torch.float32)

    if is_dist:
        dist.broadcast(W_buf, src=0)

    # 4) Load weights into the classifier
    if (W_buf.shape[0], W_buf.shape[1]) != (head.out_features, head.in_features):
        raise RuntimeError(
            f"Init shape {tuple(W_buf.shape)} doesn't match classifier "
            f"({head.out_features}, {head.in_features})."
        )

    with torch.no_grad():
        head.weight.copy_(W_buf.to(head.weight.dtype))
        if head.bias is not None:
            head.bias.zero_()

    return model
