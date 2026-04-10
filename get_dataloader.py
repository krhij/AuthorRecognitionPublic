import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_dataloaders(data_dir="processed_dataset", batch_size=32):

    transformations = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = ImageFolder(root=data_dir, transform=transformations)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the loader if this script is run directly
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Grab one batch to verify the shape
    images, labels = next(iter(train_loader))
    print(f"\nShape of one batch of images: {images.shape}")
    print(f"Shape of one batch of labels: {labels.shape}")