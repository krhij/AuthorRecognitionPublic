import torch
import torchmetrics
from model import CNNModel
from get_dataloader import get_dataloaders 

def run_final_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}\n")

    _, _, test_loader = get_dataloaders(batch_size=32)
    model = CNNModel().to(device)
    
    model.load_state_dict(torch.load("author_weights.pth", map_location=device, weights_only=True))
    
    model.eval() 
    test_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)
    
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            test_acc_metric(outputs, labels)

    # 5. Calculate and print the final unbiased score
    final_score = test_acc_metric.compute().item() * 100

    print(f"  FINAL ACCURACY (TEST SET): {final_score:.2f}%  ")

if __name__ == "__main__":
    run_final_test()