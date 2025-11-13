# models/Vit.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTModel:
    @staticmethod
    def create_model(num_classes,pretrained=True):
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None

        model = vit_b_16(weights=weights)
        if hasattr(model.heads, "head"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        else:
            # fallback สำหรับบางเวอร์ชัน torchvision ที่ใช้ 'classifier'
            in_features = model.heads[0].in_features
            model.heads[0] = nn.Linear(in_features, num_classes)

        return model