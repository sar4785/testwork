# models/Vit.py
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTModel:
    @staticmethod
    def create_model(num_classes=8):
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
