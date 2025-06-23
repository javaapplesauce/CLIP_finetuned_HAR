import os
import clip
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt


def zeroshot():

    ### Load the dataset --> installed datasets and vision dependency
    ds = load_dataset("Bingsu/Human_Action_Recognition")
    dataset = ds['train']
    dataset_test = ds["test"]


    ### model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-B/32', device)
    model.to(device)

    # Select indices for three example images
    indices = [0, 2, 10]

    # Get the list of possible subcategories from the dataset
    class_to_label = {
        'calling': 0,
        'clapping': 1,
        'cycling': 2,
        'dancing': 3,
        'drinking': 4,
        'eating': 5,
        'fighting': 6,
        'hugging': 7,
        'laughing': 8,
        'listening_to_music': 9,
        'running': 10,
        'sitting': 11,
        'sleeping': 12,
        'texting': 13,
        'using_laptop': 14
    }
    label_to_class = {v: k for k, v in class_to_label.items()}
    text_inputs = torch.cat([clip.tokenize(f"a photo of someone {cls.replace('_', ' ')}") for cls in class_to_label]).to(device)


    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    

    # Loop through the indices and process each image
    for i, sample_idx in enumerate(indices):
        # Select an example image from the dataset
        example = dataset[sample_idx]
        image = example['image']
        label_id  = example["labels"]
        actual_class = label_to_class[label_id]  # look up the string name


        # Preprocess the image
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate image and text features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)


        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity between image and text features
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_value, top_idx_tensor = similarity[0].topk(1)
        
        pred_id = top_idx_tensor[0].item()                # Python int
        pred_class = label_to_class[pred_id]              # safe dict lookup

        # Display the image in the subplot
        axes[i].imshow(image)
        axes[i].set_title(f"Predicted: {pred_class}, Actual: {actual_class}")
        axes[i].axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    zeroshot()