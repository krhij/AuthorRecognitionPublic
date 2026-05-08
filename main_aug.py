import torch
from get_dataloader_augmentation import get_dataloaders # Z augmentacją
from model import CNNModel # STARY model
from training_loop import train_model 
from utils import finalize_and_plot

WEIGHTS_NAME = "author_weights_aug.pth"
IMAGE_NAME = "training_results_aug.png"

if __name__ == "__main__":
    print("Initializing Data and Model (CNN + Augmentation)...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    model = CNNModel()
    
    print("\nStarting Training Engine...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, 
        num_epochs=60, learning_rate=0.0001, weighted_decay=1e-4
    )

    print("\nTraining Complete! Saving weights...")
    torch.save(model.state_dict(), WEIGHTS_NAME)
    print(f"Weights successfully saved to {WEIGHTS_NAME}")

    print("Generating performance graphs...")
    finalize_and_plot(train_losses, val_losses, train_accs, val_accs, IMAGE_NAME)