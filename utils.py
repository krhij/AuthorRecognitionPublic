import matplotlib.pyplot as plt

def finalize_and_plot(train_losses, val_losses, train_accs, val_accs, image_name):
    """Generates graphs and prints the final summary."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # --- Wykres 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Learning Curve: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- Wykres 2: Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Learning Curve: Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.savefig(image_name)
    print(f"Graph saved as '{image_name}'")
    
    # Zamknięcie okna, żeby wykres nie blokował skryptu (opcjonalne, ale wygodne)
    plt.close() 

    # --- Podsumowanie w konsoli ---
    final_val_acc = val_accs[-1] * 100 # Mnożymy przez 100, żeby było 94% zamiast 0.94%
    max_val_acc = max(val_accs) * 100

    print(f"\nFinal Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Best Validation Accuracy Achieved: {max_val_acc:.2f}%")