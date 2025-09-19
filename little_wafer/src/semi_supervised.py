# src/semi_supervised.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml

class SemiSupervisedTrainer:
    def __init__(self, model, device, config, label_encoder):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.le = label_encoder

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def pseudo_label_unlabeled(self, unlabeled_loader, threshold=0.9):
        """
        สร้าง pseudo-label จาก unlabeled data
        """
        self.model.eval()
        pseudo_data = []
        pseudo_labels = []

        with torch.no_grad():
            for inputs, _ in tqdm(unlabeled_loader, desc="🔍 Generating Pseudo Labels"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

                for i in range(len(confs)):
                    if confs[i].item() >= threshold:  # ใช้เฉพาะที่มั่นใจสูง
                        pseudo_data.append(inputs[i].cpu())
                        pseudo_labels.append(preds[i].cpu())

        if len(pseudo_data) == 0:
            print("⚠️ No pseudo-labels above threshold, skipping.")
            return None

        pseudo_dataset = torch.utils.data.TensorDataset(
            torch.stack(pseudo_data), torch.tensor(pseudo_labels)
        )
        print(f"✅ Generated {len(pseudo_dataset)} pseudo-labeled samples")
        return pseudo_dataset

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        return running_loss / len(loader.dataset)

    def run(self, labeled_loader, unlabeled_loader, val_loader, epochs=5, threshold=0.9):
        for epoch in range(epochs):
            print(f"\n🚀 Epoch {epoch+1}/{epochs}")

            # Train with labeled data
            loss = self.train_epoch(labeled_loader)
            print(f"   Supervised loss: {loss:.4f}")

            # Generate pseudo-labels
            pseudo_dataset = self.pseudo_label_unlabeled(unlabeled_loader, threshold)
            if pseudo_dataset is not None:
                pseudo_loader = DataLoader(pseudo_dataset, batch_size=self.config['model']['batch_size'], shuffle=True)
                pseudo_loss = self.train_epoch(pseudo_loader)
                print(f"   Pseudo-label loss: {pseudo_loss:.4f}")

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)
            print(f"   Validation loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return running_loss / len(loader.dataset), correct / total


def run_from_config(config_path="configs/config.yaml", epochs=5, threshold=0.9):
    """
    Convenience function to run semi-supervised training from config.
    Loads model, datasets, and starts training.
    """
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load pre-trained model from Kaggle data
    from src.train import TrainModel
    trainer = TrainModel(config_path=config_path)
    model, le, history, test_loader = trainer.run()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    kaggle_dir = Path(config['data']['kaggle_png'])
    wafermap_dir = Path(config['data']['wafer_map_png'])

    labeled_dataset = datasets.ImageFolder(kaggle_dir, transform=transform)
    unlabeled_dataset = datasets.ImageFolder(wafermap_dir, transform=transform)

    # Split labeled data into train/val
    val_size = int(0.2 * len(labeled_dataset))
    train_size = len(labeled_dataset) - val_size
    train_dataset, val_dataset = random_split(
        labeled_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = config['model']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)

    # Start semi-supervised training
    semi_trainer = SemiSupervisedTrainer(model, trainer.device, config, le)
    semi_trainer.run(train_loader, unlabeled_loader, val_loader, epochs=epochs, threshold=threshold)

    return semi_trainer, model, le
