import os
import torch
from finetune2 import (
    get_data_loaders,
    build_model,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint
)
from datasets import load_dataset, DownloadConfig


###
# change from google -> CLIP base model, ~80% -> 73% best loss 0.0520
# decrease learning rate from 1e-4 -> 5e-5 and weight_decay 1e-4 -> 1e-2, 81.71% best loss 0.0184
# change to cosine annealingLR, 83.10% best loss 0.0058
# change s.t. after each epoch, metrics are gathered from the test_loader not val_loader
# revoke that change COMPLETELY WRONG, also completely forgot to include my test_loader so I put that back oops


def main():
    # Device configuration (single GPU)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters (adjust as needed)
    batch_size = 64
    num_workers = 4
    learning_rate = 5e-5
    weight_decay = 1e-2
    num_epochs = 5

    # Prepare data loaders
    train_loader, val_loader, test_loader, ds_test_loader = get_data_loaders(
        batch_size=batch_size,
        num_workers=num_workers
    )
    


    # Build model, optimizer, loss, scheduler
    model, optimizer, criterion, scheduler = build_model(
        num_labels=15,
        lr=learning_rate,
        weight_decay=weight_decay,
        device=device
    )

    best_acc = 0.0

    # Optionally resume from checkpoint
    # start_epoch = load_checkpoint('checkpoint.pth', model, optimizer, scheduler, device)
    # print(f"Resuming training from epoch {start_epoch+1}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"Training Loss: {train_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        # Evaluate on validation set
        metrics = evaluate(model, val_loader, device)
        print(
            f"Val Accuracy: {metrics['acc']:.2f}% | "
            f"Precision: {metrics['prec']:.2f}% | "
            f"Recall: {metrics['rec']:.2f}% | "
            f"F1: {metrics['f1']:.2f}% | "
            f"mAP: {metrics['mAP']:.2f}%"
        )

        # Save best model and checkpoint
        checkpoint = {
            'model_state': model.state_dict(),
        }
        save_checkpoint(checkpoint, f"checkpoint_epoch{epoch+1}.pth")

        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            torch.save(model.state_dict(), 'best_model.pth')
            # save_checkpoint(checkpoint, 'best_model.pth')
            print(f"*** New best model saved (Accuracy: {best_acc:.2f}%) ***")

    # After training
    print("\nTraining complete.")

    state_dict = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Prepare containers for predictions & labels
    all_preds = []
    all_labels = []
    
    # 3. Inference loop
    with torch.no_grad():
        for images, labels in test_loader:
            # move to device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            # if using a Hugging Face-style model, grab .logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # get predicted class
            preds = torch.argmax(logits, dim=1)

            # collect
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # 4. Concatenate and compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    metrics = evaluate(model, test_loader, device)
    print(
        f"test Accuracy: {metrics['acc']:.2f}% | "
        f"Precision: {metrics['prec']:.2f}% | "
        f"Recall: {metrics['rec']:.2f}% | "
        f"F1: {metrics['f1']:.2f}% | "
        f"mAP: {metrics['mAP']:.2f}%"
    )
    # You can now perform inference on the test set or new images.



    ### 
    
    # 2. Prepare containers for predictions & labels
    all_preds = []
    all_labels = []
    
    # 3. Inference loop
    with torch.no_grad():
        for images, labels in ds_test_loader:
            # move to device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            # if using a Hugging Face-style model, grab .logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # get predicted class
            preds = torch.argmax(logits, dim=1)

            # collect
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # 4. Concatenate and compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    print("Evaluating on original test split")
    metrics = evaluate(model, ds_test_loader, device)
    print(
        f"test Accuracy: {metrics['acc']:.2f}% | "
        f"Precision: {metrics['prec']:.2f}% | "
        f"Recall: {metrics['rec']:.2f}% | "
        f"F1: {metrics['f1']:.2f}% | "
        f"mAP: {metrics['mAP']:.2f}%"
    )

if __name__ == '__main__':
    main()