import torch
import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

def get_dataloaders(data_dir="processed_dataset", batch_size=32):
    #2 przypadki - augmentacja i bez
    train_transform = transforms.Compose([
        transforms.Grayscale(1),
        #augmentacja
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #ten sam folder 2x z różnymi transformacjami
    train_data = ImageFolder(root=data_dir, transform=train_transform)
    test_data = ImageFolder(root=data_dir, transform=test_transform)

    train_idx, val_idx, test_idx = [], [], []
    
    #pobieramy nazwy słów z plików
    all_samples = train_data.imgs 
    
    #unikalne slowa z datasetu
    all_unique_words = sorted(list(set([os.path.basename(path).split('_')[0] for path, _ in all_samples])))
    
    #podzial listy slow
    torch.manual_seed(42)
    shuffled_words = torch.randperm(len(all_unique_words))
    
    #podzial 70/15/15
    num_words = len(all_unique_words)
    tr_limit = int(0.7 * num_words)
    val_limit = int(0.85 * num_words)

    train_words = set([all_unique_words[i] for i in shuffled_words[:tr_limit]])
    val_words = set([all_unique_words[i] for i in shuffled_words[tr_limit:val_limit]])
    #reszta to test_words

    #indeksy plikow do odpowiedniej kategorii na podstawie nazwy
    for idx, (path, _) in enumerate(all_samples):
        word_name = os.path.basename(path).split('_')[0]
        if word_name in train_words:
            train_idx.append(idx)
        elif word_name in val_words:
            val_idx.append(idx)
        else:
            test_idx.append(idx)

    #laczymy obrazy z loaderem
    train_loader = DataLoader(Subset(train_data, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(test_data, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test_data, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader