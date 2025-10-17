import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from torchvision import transforms, models
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import yaml

#SimCLR Trainer
def get_simclr_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = list(Path(root).rglob("*.png"))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            x1 = self.transform(img)
            x2 = self.transform(img)
        return x1, x2

    def __len__(self):
        return len(self.image_paths)
   
class SimCLRModel(nn.Module):
    def __init__(self, base_model='resnet18', projection_dim=128):
        super().__init__()
        # ✅ สร้าง backbone จาก timm (รองรับ EfficientNet, ConvNeXt, ResNet ฯลฯ)
        self.base_model = config["default"].get("base_model", "resnet18")
        self.backbone = timm.create_model(base_model, pretrained=True, num_classes=0)  
        
        # 🔹 หาขนาด output feature ของ encoder อัตโนมัติ
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        elif hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
        else:
            in_features = 512  # fallback สำหรับบาง backbone
        
        # ✅ projection head (SimCLR head)
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
        
    def forward(self, x):
        h = self.backbone(x)          # feature vector
        z = self.projection_head(h)   # projection space
        return F.normalize(z, dim=1) 
    
#Contrastive Loss
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature
    n = z.size(0)
    labels = torch.arange(n, device=z.device)
    labels = (labels + n // 2) % n
    loss = F.cross_entropy(sim, labels)
    return loss
    
#Trainer สำหรับ SimCLR
class SimCLRTrainer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = Path(config["data"].get("augment_data", "./Data/Augmented_Real/"))
        self.batch_size = config["default"].get("batch_size", 32)
        self.epochs = config["default"].get("epochs", 10)
        self.lr = config["default"].get("learning_rate", 0.003)
        self.checkpoint_dir = Path(config["output"]["checkpoints"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True) 
        
    def get_dataloader(self):
        dataset = SimCLRDataset(self.data_dir, transform=get_simclr_transform())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        print(f"📁 Found {len(dataset)} unlabeled wafer images for SimCLR pretraining")
        return loader    
        
    def train(self, save_name="simclr_pretrained.pth"):
        model = SimCLRModel(base_model=self.base_model).to(self.device)
        loader = self.get_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        print(f"\n🚀 Start SimCLR pretraining on {self.device}")
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for x1, x2 in tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"✅ Epoch [{epoch+1}/{self.epochs}] | Loss: {total_loss/len(loader):.4f}")

        save_path = self.checkpoint_dir / f"simclr_{self.base_model}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 SimCLR model ({self.base_model}) saved → {save_path}")
        return model
        
def train_simclr_from_config(config_path="configs/config.yaml"):
    trainer = SimCLRTrainer(config_path)
    return trainer.train()                 