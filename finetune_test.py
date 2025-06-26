

import torch
import clip
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader

# Path to your fine-tuned weights
WEIGHTS_PATH = "clip_finetuned_HAR.pth"


def get_test_loader(preprocess, batch_size=32):
    # Load the test split
    har_test = load_dataset("Bingsu/Human_Action_Recognition", split="test")

    def collate_fn(batch):
        processed_images = []
        labels = []
        for ex in batch:
            img = ex["image"]
            # If already a PIL image, use directly; otherwise convert from array
            if isinstance(img, Image.Image):
                pil_img = img
            else:
                pil_img = Image.fromarray(img)
            processed_images.append(preprocess(pil_img))
            labels.append(ex["label"])

        images = torch.stack(processed_images)
        labels = torch.tensor(labels)
        return images, labels

    return DataLoader(
        har_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    ), har_test.features["labels"].names


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load base CLIP and its preprocessing function
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 2) Load fine-tuned weights
    state = torch.load(WEIGHTS_PATH, map_location=device)
    # Strip 'model.' prefix if present in keys
    if not any(k.startswith('positional_embedding') for k in state):
        filtered = {}
        for k, v in state.items():
            if k.startswith('model.'):
                new_key = k[len('model.') :]
                filtered[new_key] = v
        state = filtered
    # Load into CLIP model (ignore mismatched classifier keys)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 3) Prepare test loader and class names
    test_loader, class_names = get_test_loader(preprocess)

    # 4) Build zero-shot text features
    prompts = [f"a photo of someone {name.replace('_', ' ')}" for name in class_names]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 5) Run inference and accumulate predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            img_feats = model.encode_image(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            logits = img_feats @ text_features.T
            preds = logits.argmax(dim=-1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # 6) Compute and print accuracy
    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Zero-shot HAR test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
