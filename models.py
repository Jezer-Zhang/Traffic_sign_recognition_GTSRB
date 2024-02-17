import torch
import torchvision.models as models
import torch.nn as nn


class CustomResNet50(nn.Module):
    def __init__(self, num_classes, weights_path=None):
        super(CustomResNet50, self).__init__()
        # Load ResNet-50 model
        self.resnet = models.resnet50(
            weights=None
        )  # Initialize without pre-trained weights

        if weights_path is not None:
            # Load the provided weights into the model
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            self.resnet.load_state_dict(state_dict)

        # Replace the fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through the ResNet-50 model
        x = self.resnet(x)

        return x
