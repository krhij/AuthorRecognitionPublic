import torch
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

    #robimy liste numerow obrazow, dzielimy na 3 i przypisujemy do train_idx, val_idx i test_idx
    total = len(train_data)
    train_idx, val_idx, test_idx = random_split(
        range(total), 
        [int(0.7*total), int(0.15*total), total - int(0.7*total) - int(0.15*total)],
        generator=torch.Generator().manual_seed(42)
    )

    #laczymy obrazy z loaderem
    train_loader = DataLoader(Subset(train_data, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(test_data, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(test_data, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader