import torch
import torchvision.models as models
import torch.nn as nn


class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        # Load pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)

        # Replace the fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through ResNet-50 model
        x = self.resnet(x)
        return x
