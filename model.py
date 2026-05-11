import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2,),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            )
        self.classifier = nn.Sequential(
            nn.Linear(128*16*8, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 8),
        )
        self._initialize_weights()
    def forward(self, x):
        x= self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for layer in self.modules():
            #kaiming jesli konwolucyjna albo liniowa zeby z lepszymi wagami zaczynac (nie calkowicie losowe)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                #bias na zero zeby stabilizowac uczenie
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

class ResNet18_Custom(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet18_Custom, self).__init__()
        
        #weights=None wiec uczenie od zera a nie transfer learning z ImageNet (bo tam 3 kanały a my mamy 1)
        self.resnet = models.resnet18(weights=None)
        
        #Modyfikacja pierwszej warstwy konw tak aby przyjmowała 1 kanał zamiast 3
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)