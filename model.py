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
            # If it's a Conv or Linear layer, apply Kaiming Normal
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # Set starting biases to exactly zero
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

class ResNet18_Custom(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet18_Custom, self).__init__()
        
        # 1. Load the empty ResNet18 blueprint (weights=None means training from scratch)
        self.resnet = models.resnet18(weights=None)
        
        # 2. ADAPT THE INPUT: Change the first layer to accept Grayscale (1 channel) instead of RGB (3 channels)
        # We copy the exact settings of the original layer, but change in_channels to 1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # 3. ADAPT THE OUTPUT: Change the final classification head
        # Find out how many neurons are feeding into the final layer (usually 512 for ResNet18)
        num_features = self.resnet.fc.in_features
        # Replace the 1000-class layer with your 8-class layer
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Simply pass the image through the modified ResNet
        return self.resnet(x)