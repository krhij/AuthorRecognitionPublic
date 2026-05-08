import torch
import torchmetrics
from model import CNNModel, ResNet18_Custom
#nowy dataloader zeby kazdy model byl sprawdzany na tych samych danych
from get_dataloader_augmentation import get_dataloaders 

def run_final_test(model, weights_path, test_loader, device):
    #wagi
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    
    model.eval() 
    test_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)
    
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_acc_metric.update(outputs, labels)

    final_score = test_acc_metric.compute().item() * 100
    print(f"Wynik dla {weights_path}: {final_score:.2f}%")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}\n")

    #ladowanie danych
    _, _, test_loader = get_dataloaders(batch_size=32)

    #tablica 4 modeli
    modele_do_sprawdzenia = [
        (CNNModel().to(device), "author_weights_baseline.pth"),
        (CNNModel().to(device), "author_weights_aug.pth"),
        (ResNet18_Custom(num_classes=8).to(device), "author_weights_resnet_baseline.pth"),
        (ResNet18_Custom(num_classes=8).to(device), "author_weights_resnet_aug.pth")
    ]

    
    for model, weights_path in modele_do_sprawdzenia:
        run_final_test(model, weights_path, test_loader, device)