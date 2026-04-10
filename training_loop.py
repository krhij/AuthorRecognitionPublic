import torch
import torch.nn as nn
import torch.optim as optim
from get_dataloader import get_dataloaders  # Importing your data pipeline
from model import CNNModel  
import torchmetrics



def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, weighted_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay= weighted_decay)
    train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)
    val_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_acc_metric.update(outputs, labels)
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        train_acc = train_acc_metric.compute().item()
        train_acc_metric.reset()
        train_accuracies.append(train_acc)


        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_acc_metric.update(outputs, labels)

            avg_epoch_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_epoch_val_loss)

            val_acc = val_acc_metric.compute().item()
            val_acc_metric.reset()
            val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_epoch_val_loss:.4f}")
    return train_losses, val_losses, train_accuracies, val_accuracies



