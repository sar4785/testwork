import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetModel:
    def __init__(self) -> None:
        pass
    @staticmethod
    def create_model(num_classes):
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model