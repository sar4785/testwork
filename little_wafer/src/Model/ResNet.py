import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock, ResNet

# 1️⃣ Define SE Block (Channel Attention)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 
       
class SEBasicBlock(BasicBlock):
    def __init__(self, *args, reduction=16, **kwargs):
        """
        Accept arbitrary args/kwargs like BasicBlock so ResNet(...) can instantiate it.
        We call super().__init__(*args, **kwargs) to keep BasicBlock behavior, then
        attach an SEBlock based on the conv2 output channels.
        """
        super().__init__(*args, **kwargs)
        # conv2.out_channels equals 'planes' for BasicBlock (after expansion)
        out_channels = self.conv2.out_channels
        self.se = SEBlock(out_channels, reduction=reduction)

    def forward(self, x):
        # Re-implement BasicBlock forward to insert SE before addition (like SE paper)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # apply SE here
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out        
        

class SEResNet18(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, reduction=16):
        super().__init__()
        # Use torchvision's ResNet class but replace block with SEBasicBlock
        self.model = ResNet(block=SEBasicBlock, layers=[2, 2, 2, 2])

        if pretrained:
            # load weights from canonical resnet18 but not strictly (SE layers won't match)
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            base_state = base.state_dict()
            # load base weights into our model (allow missing keys like fc and any se-specific params)
            self.model.load_state_dict(base_state, strict=False)

        # replace final fc
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)   
            
    def forward(self, x):
        return self.model(x)        
            
class ResNetModel:
    def __init__(self):
        pass

    @staticmethod
    def create_model(num_classes):
        # ใช้ SE-ResNet18 แทน ResNet ธรรมดา
        model = SEResNet18(num_classes=num_classes, pretrained=True, reduction=16)
        return model            
            
            