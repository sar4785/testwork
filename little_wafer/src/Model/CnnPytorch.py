import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,WeightedRandomSampler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix 

def train_cnn_pytorch(config_path="configs/config.yaml", val_label=False):
    # ---------------- Load Config ----------------
    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../", config_path))
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # ✅ แปลง path ใน config ให้เป็น absolute
    base_dir = os.path.abspath(os.path.join(os.path.dirname(CONFIG_PATH), ".."))
    train_dir = os.path.abspath(os.path.join(base_dir, cfg["data"]["train"]))
    val_dir = os.path.abspath(os.path.join(base_dir, cfg["data"]["val"]))
    test_dir = os.path.abspath(os.path.join(base_dir, cfg["data"]["test"]))
    
    print(f"🔍 Train dir: {train_dir}")
    print(f"🔍 Val dir:   {val_dir}")
    print(f"🔍 Test dir:  {test_dir}")
    
    allowed_classes = cfg["data"]["allowed_classes"]

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    PATIENCE = 10

    # ---------------- Device Setup ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # ---------------- Data Loaders ----------------
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    print("📂 Loading training dataset...")
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    class_counts = np.bincount(train_dataset.targets)
    weights = 1. / class_counts
    sample_weights = [weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    print(f"✅ Loaded {len(train_dataset)} training samples")

    # ---------------- Load Unlabeled Images ----------------
    def load_unlabeled_images(folder):
        files = list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpg"))
        imgs, names = [], []
        for f in files:
            img = Image.open(f).convert("RGB")
            img = transform(img)
            imgs.append(img)
            names.append(f.name)
        return imgs, names

    if val_label:
        print("📂 Loading labeled validation dataset...")
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
        val_targets = np.array(val_dataset.targets)
        val_class_counts = np.bincount(val_targets)
        print("📊 Validation class distribution:", val_class_counts)
    
        # ✅ ป้องกัน bias ด้วย class weight (ใช้ใน loss)
        val_class_weights = torch.tensor(1.0 / np.maximum(val_class_counts, 1), dtype=torch.float32)
        val_class_weights = val_class_weights / val_class_weights.sum() * len(val_class_counts)
        val_class_weights = val_class_weights.to(device)

        # ✅ เพิ่ม WeightedRandomSampler เพื่อ oversample class ที่มีน้อย
        val_weights = 1. / np.maximum(val_class_counts, 1)
        val_sample_weights = [val_weights[t] for t in val_targets]
        val_sampler = WeightedRandomSampler(val_sample_weights, len(val_sample_weights), replacement=True)
    
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

        print(f"✅ Loaded {len(val_dataset)} labeled validation samples with Weighted Sampling")
      
    else:
        print("📂 Loading unlabeled validation dataset...")
        val_imgs, val_names = load_unlabeled_images(val_dir)
        val_loader = DataLoader(val_imgs, batch_size=BATCH_SIZE, shuffle=False)
        print(f"✅ Loaded {len(val_imgs)} unlabeled validation samples")

    print("📂 Loading test dataset...")
    test_imgs, test_names = load_unlabeled_images(test_dir)
    test_loader = DataLoader(test_imgs, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Loaded {len(test_imgs)} test samples")

    # ---------------- Define CNN Model ----------------
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.fc(self.conv(x))

    num_classes = len(allowed_classes)
    model = SimpleCNN(num_classes).to(device)
    print(model)

    # ---------------- Loss & Optimizer ----------------
    criterion = nn.CrossEntropyLoss(weight=val_class_weights if val_label else None)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------- Early Stopping ----------------
    class EarlyStopping:
        def __init__(self, patience=10, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = np.inf
            self.counter = 0
            self.should_stop = False

        def step(self, current_loss):
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True

    torch.cuda.empty_cache()

    # ---------------- Training ----------------
    os.makedirs("./Output/checkpoints", exist_ok=True)
    print("\n🚀 Start training CNN model...")
    start_time = time.time()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0.0
    early_stopper = EarlyStopping(patience=PATIENCE)

    for epoch in range(EPOCHS):
        # ---------------- Train ----------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---------------- Validation ----------------
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                if val_label:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    val_total += labels.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                else:
                    images = batch.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    confidences, preds = torch.max(probs, 1)
                    val_running_loss += torch.mean(1 - confidences).item()
                    val_correct += torch.sum(confidences > 0.5).item()
                    val_total += preds.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ✅ ถ้ามี label — แสดง metrics เพิ่มเติม
        if val_label and epoch == EPOCHS - 1:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=allowed_classes, yticklabels=allowed_classes)
            plt.title("Validation Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig("./Output/confusion_matrix.png", dpi=300)
            print("✅ Confusion matrix saved to ./Output/confusion_matrix.png")


        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), "./Output/checkpoints/cnn_best.pth")
            print(f"💾 Saved new best model at epoch {epoch+1} with Val Acc: {best_acc:.4f}")

        # Early stopping
        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️ Training finished in {elapsed_time/60:.2f} minutes")

    torch.cuda.empty_cache()

    # ---------------- Plot Learning Curves ----------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy', linestyle='--')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='--')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("./Output/training_curves.png", dpi=300)
    # plt.show()

    # ---------------- Save Training Metrics ----------------
    metrics_df = pd.DataFrame({
        "epoch": range(1, len(train_losses)+1),
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "elapsed_time_minute": elapsed_time/60
    })
    metrics_df.to_csv("./Output/training_metrics.csv", index=False)
    print("✅ Metrics saved to ./Output/training_metrics.csv")

    # ---------------- Predict on Test Images ----------------
    if len(test_imgs) > 0:
        model.eval()
        all_preds, all_conf, all_names = [], [], []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, images in enumerate(test_loader):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_conf.extend(conf.cpu().numpy())
                start_idx = i * BATCH_SIZE
                end_idx = start_idx + images.size(0)
                all_names.extend(test_names[start_idx:end_idx])

        df_test = pd.DataFrame({
            "filename": all_names,
            "pred_label": [allowed_classes[i] for i in all_preds],
            "test_confidence": all_conf
        })
        os.makedirs("./Output/predictions", exist_ok=True)
        df_test.to_csv("./Output/predictions/test_predictions.csv", index=False)
        print("✅ Test predictions saved to ./Output/predictions/test_predictions.csv")

if __name__ == "__main__":
    train_cnn_pytorch(config_path="configs/config.yaml", val_label=True)
