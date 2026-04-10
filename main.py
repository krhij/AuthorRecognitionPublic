import torch
import matplotlib.pyplot as plt

from get_dataloader import get_dataloaders
from model import CNNModel
from training_loop import train_model 

def plot_results(train_losses, val_losses, train_accs, val_accs):
    """Generates a side-by-side graph of Loss and Accuracy."""
    epochs = range(1, len(train_losses) + 1)

    # Create a wide figure with 2 subplots (1 row, 2 columns)
    plt.figure(figsize=(12, 5))

    # --- Plot 1: The Loss Curve ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Learning Curve: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- Plot 2: The Accuracy Curve ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Learning Curve: Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the graph as a picture before showing it!
    plt.savefig('training_results.png')
    plt.show()


if __name__ == "__main__":
    print("Initializing Data and Model...")
    
    # 1. Get the data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    
    # 2. Create the blank brain
    model = CNNModel()
    
    print("\nStarting Training Engine...")
    
    # 3. Train the model and catch the lists it returns
    train_losses, val_losses, train_accs, val_accs = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=15, # Start with 15-20 epochs to see how it performs
        learning_rate=0.0001,
        weighted_decay=1e-4
    )

    print("\nTraining Complete! Saving weights...")
    
    # 4. Save the weights (The model's "memory")
    # .pth is the standard PyTorch extension for saving weights
    torch.save(model.state_dict(), "author_weights.pth")
    print("Weights successfully saved to 'author_weights.pth'")

    # 5. Plot the final results
    print("Generating performance graphs...")
    plot_results(train_losses, val_losses, train_accs, val_accs)

    final_val_acc = val_accs[-1]
    max_val_acc = max(val_accs)

    print(f"\nFinal Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Best Validation Accuracy Achieved: {max_val_acc:.2f}%")