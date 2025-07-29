import torch
from transformers import CLIPTextModel, CLIPTokenizer

def build_prompts(class_names):
    return [f"a person {c}" for c in class_names]


if __name__ == "__main__":
    # 1. your classes
    class_names = ["walking", "running", "jumping", …]

    # 2. load CLIP text encoder
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model = model.cuda()

    # 3. tokenize & encode
    texts = build_prompts(class_names)
    inputs = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model(**inputs).last_hidden_state  # (B, T, D)
        # use the [EOS] token
        eos_ix = (inputs.attention_mask.sum(dim=1) - 1).unsqueeze(-1)
        embeds = out[torch.arange(len(texts)), eos_ix.squeeze()]  # (B, D)

    # 4. normalize & save
    embeds = embeds / embeds.norm(dim=1, keepdim=True)
    torch.save(embeds.cpu(), "lan_emb.pt")
    print("Saved lan_emb.pt with shape", embeds.shape)