import torch
import os
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

def get_dataloaders(data_dir="processed_dataset", batch_size=32):
    
    #  BRAK AUGMENTACJI
    transformations = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. Ładujemy cały dataset jednym obiektem ImageFolder
    full_dataset = ImageFolder(root=data_dir, transform=transformations)

    train_idx, val_idx, test_idx = [], [], []
    all_samples = full_dataset.imgs 
    
    # 3. Tworzymy unikalny słownik na podstawie nazw plików
    all_unique_words = sorted(list(set([os.path.basename(path).split('_')[0] for path, _ in all_samples])))
    
    torch.manual_seed(42)
    shuffled_words = torch.randperm(len(all_unique_words))
    
    num_words = len(all_unique_words)
    tr_limit = int(0.7 * num_words)
    val_limit = int(0.85 * num_words)

    train_words = set([all_unique_words[i] for i in shuffled_words[:tr_limit]])
    val_words = set([all_unique_words[i] for i in shuffled_words[tr_limit:val_limit]])

    # 5. Przypisujemy konkretne pliki do odpowiednich indeksów
    for idx, (path, _) in enumerate(all_samples):
        word_name = os.path.basename(path).split('_')[0]
        if word_name in train_words:
            train_idx.append(idx)
        elif word_name in val_words:
            val_idx.append(idx)
        else:
            test_idx.append(idx)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=batch_size, shuffle=False)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
 
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"\nShape of one batch of images: {images.shape}")
    print(f"Shape of one batch of labels: {labels.shape}")