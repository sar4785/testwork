# ==================== Trainer.py (Modified) ====================
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


class Trainer:
    def __init__(self, config, model_name="resnet"):
        self.config = config
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dir = Path(config["data"]["train"])
        
        # 🔹 นับจำนวน class (ไม่รวม 'none' แล้ว เพราะรวมกับ 'random')
        self.num_classes = len([f for f in train_dir.iterdir() if f.is_dir()])
        print(f"📊 Detected {self.num_classes} classes from {train_dir}")
        
        data_cfg = config["data"]
        self.train_dir = Path(data_cfg["train"])
        self.val_dir = Path(data_cfg["val"])
        self.test_dir = Path(data_cfg["test"])
        
        self.output_root = Path(config["output"]["checkpoints"]).resolve().parents[0]
        self.checkpoint_dir = Path(config["output"]["checkpoints"]).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_cfg = config.get("model", {}).get(model_name, config.get("default", {}))
        self.epochs = model_cfg.get("epochs", 5)
        self.batch_size = model_cfg.get("batch_size", 32)
        self.lr = model_cfg.get("learning_rate", 1e-4)

        print(f"💻 Using device: {self.device}")

    # -----------------------------------------------------
    # Load datasets with Data Augmentation (ตาม Paper)
    # -----------------------------------------------------
    def get_dataloaders(self):
        """
        🔹 ตาม Paper: ใช้ Horizontal Flip, Vertical Flip, Rotation
        เพื่อทำ data augmentation แบบ random
        
        🔧 แก้ไข: ลด augmentation ที่รุนแรง และใช้ normalization ที่เหมาะกับ grayscale
        """
        # Training augmentation (ปรับให้เบาลง)
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),      # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.5),        # Random vertical flip
            transforms.RandomRotation(degrees=15),       # 🔧 ลดจาก 360° → 15° เพื่อไม่ให้ผิดเพี้ยนมาก
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],   # 🔧 ใช้ค่าที่เหมาะกับ grayscale
                                std=[0.5, 0.5, 0.5])
        ])

        # Validation/Test (no augmentation, same normalization)
        transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],   # 🔧 ใช้ค่าเดียวกับ training
                                 std=[0.5, 0.5, 0.5])
        ])

        train_dataset = datasets.ImageFolder(self.train_dir, transform=transform_train)
        val_dataset = datasets.ImageFolder(self.val_dir, transform=transform_eval)
        self.class_names = train_dataset.classes

        # Test set (no labels) - สำหรับ real wafer maps
        test_image_paths = list(self.test_dir.rglob("*.png")) + list(self.test_dir.rglob("*.jpg"))
        test_images = [Image.open(p).convert("RGB") for p in test_image_paths]
        test_tensors = [transform_eval(img) for img in test_images]
        test_names = [p.name for p in test_image_paths]
        test_dataset = list(zip(test_tensors, test_names))
        
        # 🔹 Weighted sampler for imbalanced data (ตาม paper)
        class_counts = np.bincount(train_dataset.targets)
        weights = 1. / (class_counts + 1e-6)
        samples_weights = weights[train_dataset.targets]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_loader, val_loader, test_loader

    # -----------------------------------------------------
    # Supervised training with Pre-train / Fine-tune
    # -----------------------------------------------------
    def train_supervised(self, model, save_name="model.pth"):
        """
        🔹 ตาม Paper: 
        - Pre-train บนข้อมูล clean
        - Fine-tune บนข้อมูล noisy
        """
        train_loader, val_loader, test_loader = self.get_dataloaders()
        model = model.to(self.device)
        
        # 🔹 โหลด pretrained weights (ถ้ามีใน config)
        is_finetuning = "pretrained_weights" in self.config
        if is_finetuning:
            pretrained_path = self.config["pretrained_weights"]
            print(f"🔁 Loading pretrained weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=self.device), strict=False)
        
        # 🔹 ตั้งค่า optimizer ด้วย lr ต่ำสำหรับ fine-tune
        lr = self.lr if not is_finetuning else self.lr * 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        start_time = time.time()
        phase = "Fine-Tuning" if is_finetuning else "Pre-Training"
        
        print(f"\n🚀 Start {phase} {self.model_name.upper()} ...")

        for epoch in range(self.epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            val_acc = self.evaluate_acc(model, val_loader)
            print(f"📈 Epoch {epoch+1}: TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f}")

            # 🔹 Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # ตั้งชื่อไฟล์ให้ตรงกับ phase
                phase_tag = "pretrain" if not is_finetuning else "finetune"
                save_path = self.checkpoint_dir / f"{self.model_name}_{phase_tag}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"💾 Best model saved: {save_path}")
        
        # 🔹 ทำนาย test set และบันทึกผล
        elapsed_minutes = (time.time() - start_time) / 60.0
        print(f"✅ {phase} completed in {elapsed_minutes:.2f} minutes")
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
        
        self.predict_test_and_plot(model, test_loader)
        return elapsed_minutes
                      
    def evaluate_acc(self, model, loader):
        """Calculate accuracy on validation set"""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0
    
    def predict_test_and_plot(self, model, test_loader):
        """
        🔹 ทำนายบน test set (real wafer maps ที่ไม่มี label)
        และสร้าง bar chart แสดงผลการทำนาย
        """
        model.eval()
        preds, names = [], []
        
        with torch.no_grad():
            for imgs, fnames in test_loader:  # test_dataset คือ list ของ (tensor, filename)
                imgs = imgs.to(self.device)  # เพิ่ม batch dimension
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                preds.extend(pred.cpu().numpy())
                names.extend(fnames)
        
        # บันทึกผลการทำนาย
        df = pd.DataFrame({
            "filename": names, 
            "pred_label": [self.class_names[p] for p in preds]
        })
        out_path = self.output_root / f"test_predictions_{self.model_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Test predictions saved → {out_path}")
        
        # Plot bar graph
        plt.figure(figsize=(10, 6))
        counts = df["pred_label"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette="tab10")
        plt.title(f"Predicted Classes on Real Test Data ({self.model_name.upper()})")
        plt.xlabel("Predicted Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plot_path = self.output_root / f"bar_{self.model_name}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"🎨 Bar graph saved → {plot_path}")