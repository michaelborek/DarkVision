import torch
import torch.nn as nn
import torchvision.models as models

class Resnet18(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(Resnet18, self).__init__()
        if use_pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18()
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
