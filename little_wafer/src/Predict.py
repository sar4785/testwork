# src/Predict.py (แก้ไขแล้ว)
import os
import time
import torch
import yaml
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# ============================ MODEL IMPORTS ============================
from src.Model.ResNet import ResNetModel
from src.Model.EfficientNet import EfficientNetModel
from src.Model.ConvNeXt import ConvNeXtModel
from src.Model.Vit import ViTModel

# ============================ UTILITY FUNCTIONS ============================
def load_config(config_path: str):
    """Load YAML configuration file"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_device():
    """Select GPU if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    return device

def get_num_classes_from_train(config):
    """
    🔹 นับจำนวน class จาก train_split folder
    (ใช้วิธีนี้เพื่อให้ตรงกับที่ใช้ตอนเทรน)
    """
    train_dir = Path(config["data"]["train"]).resolve()
    
    # ถ้า train ชี้ไปที่ train_split แล้ว
    if train_dir.name == "train_split" or (train_dir.parent / "train_split").exists():
        train_dir = train_dir.parent / "train_split"
    
    # นับ class folders
    class_folders = [f for f in train_dir.iterdir() if f.is_dir()]
    num_classes = len(class_folders)
    class_names = sorted([f.name for f in class_folders])
    
    print(f"📊 Detected {num_classes} classes from {train_dir}")
    print(f"   Classes: {class_names}")
    
    return num_classes, class_names

def get_test_dataset(config, transform=None):
    """
    🔹 โหลด test set (ไม่มี label หรือมี label ก็ได้)
    
    Returns:
    --------
    - ถ้า test_dir มี subfolder (labeled): ImageFolder dataset
    - ถ้า test_dir มีแค่ภาพ (unlabeled): list of (tensor, filename)
    """
    data_cfg = config["data"]
    test_dir = Path(data_cfg["test"]).resolve()

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],  # ใช้ค่าเดียวกับ training
                                 std=[0.5, 0.5, 0.5])
        ])

    # 🔹 ตรวจสอบว่า test_dir มี subfolder (labeled) หรือไม่
    subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
    
    if len(subdirs) > 0:
        # Test set มี label (ImageFolder structure)
        print(f"📂 Test set has labels (found {len(subdirs)} classes)")
        dataset = datasets.ImageFolder(test_dir, transform=transform)
        return dataset, True  # has_labels=True
    
    else:
        # Test set ไม่มี label (แค่ภาพเดี่ยวๆ)
        print(f"📂 Test set has NO labels (flat structure)")
        image_paths = sorted(list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")))
        
        if len(image_paths) == 0:
            raise FileNotFoundError(f"❌ No images found in {test_dir}")
        
        images = [transform(Image.open(p).convert("RGB")) for p in image_paths]
        filenames = [p.name for p in image_paths]
        
        print(f"📂 Loaded {len(images)} test images from {test_dir}")
        return list(zip(images, filenames)), False  # has_labels=False

# ============================ MODEL LOADING ============================
def load_model(model_name, checkpoint_dir, num_classes, phase="finetune"):
    """
    Load trained model with checkpoint
    
    Parameters:
    -----------
    model_name : str
        Model name (resnet, efficientnet, vit, convnext)
    checkpoint_dir : Path
        Directory containing checkpoints
    num_classes : int
        Number of output classes
    phase : str
        "pretrain" or "finetune" (default: finetune)
    """
    model_classes = {
        "resnet": ResNetModel,
        "efficientnet": EfficientNetModel,
        "vit": ViTModel,
        "convnext": ConvNeXtModel,
    }

    if model_name not in model_classes:
        raise ValueError(f"❌ Unknown model name: {model_name}")

    model_class = model_classes[model_name]
    
    # 🔹 สร้างโมเดล
    model = model_class.create_model(num_classes=num_classes)
    
    # 🔹 หา checkpoint file
    ckpt_path = Path(checkpoint_dir) / f"{model_name}_{phase}.pth"
    
    if not ckpt_path.exists():
        print(f"⚠️ {phase} checkpoint not found, trying alternative...")
        # ลองหา pretrain ถ้าไม่เจอ finetune
        alt_phase = "pretrain" if phase == "finetune" else "finetune"
        ckpt_path = Path(checkpoint_dir) / f"{model_name}_{alt_phase}.pth"
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"❌ No checkpoint found for {model_name} (tried both pretrain and finetune)")
    
    # 🔹 โหลด weights
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Loaded {model_name.upper()} weights from {ckpt_path}")
    return model

# ============================ PREDICTION (UNLABELED TEST SET) ============================
def predict_unlabeled_test(model, dataset, device, class_names, save_prefix, output_dir):
    """
    🔹 ทำนายบน test set ที่ไม่มี label
    แล้วบันทึกเป็น CSV และ bar chart
    """
    model.eval()
    preds, filenames = [], []
    
    batch_size = 32
    
    # 🔹 สร้าง custom collate function
    def collate_fn(batch):
        """Custom collate to handle (tensor, filename) pairs"""
        imgs = torch.stack([item[0] for item in batch])
        fnames = [item[1] for item in batch]
        return imgs, fnames
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    with torch.no_grad():
        for imgs, fnames in tqdm(dataloader, desc=f"Predicting with {save_prefix.upper()}"):
            imgs = imgs.to(device)
            
            outputs = model(imgs)
            batch_preds = torch.argmax(outputs, dim=1).cpu().tolist()
            
            preds.extend(batch_preds)
            filenames.extend(fnames)
    
    # 🔹 บันทึกผลเป็น CSV
    df = pd.DataFrame({
        "filename": filenames,
        "pred_label": [class_names[p] for p in preds],
        "pred_index": preds
    })
    
    csv_path = output_dir / f"test_predictions_{save_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 Predictions saved → {csv_path}")
    
    # 🔹 สรุปการกระจายของ predictions
    label_counts = df["pred_label"].value_counts()
    print(f"\n📊 Prediction Summary:")
    for label, count in label_counts.items():
        print(f"   {label:12s}: {count:4d} images ({100*count/len(df):.1f}%)")
    
    # 🔹 Plot bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="tab10")
    plt.title(f"Predicted Class Distribution ({save_prefix.upper()})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    
    # เพิ่ม count บน bar
    for i, (label, count) in enumerate(label_counts.items()):
        plt.text(i, count + 10, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    plot_path = output_dir / f"bar_{save_prefix}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"🎨 Bar chart saved → {plot_path}\n")
    
    return df

# ============================ EVALUATION (LABELED TEST SET) ============================
def evaluate_labeled_test(model, dataloader, device, class_names, save_prefix, output_dir):
    """
    🔹 ประเมินผลบน test set ที่มี label
    คำนวณ accuracy, precision, recall, F1-score และ confusion matrix
    """
    model.eval()
    y_true, y_pred = [], []

    start = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {save_prefix.upper()}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    elapsed = time.time() - start
    print(f"⏱️ Evaluation completed in {elapsed:.2f}s")

    # 🔹 Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    
    report_path = output_dir / f"{save_prefix}_metrics.csv"
    df_report.to_csv(report_path)
    print(f"📊 Classification report → {report_path}")
    
    # แสดงผลบางส่วน
    print("\n📈 Classification Report:")
    print(df_report[["precision", "recall", "f1-score", "support"]].round(3))

    # 🔹 Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {save_prefix.upper()}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    cm_path = output_dir / f"{save_prefix}_confusion.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix → {cm_path}\n")

# ============================ COMPARE MODELS ============================
def compare_models(config_path="configs/config.yaml", model_list=None):
    """
    🔹 เปรียบเทียบหลายโมเดลบน test set
    - ถ้า test มี label → evaluate (accuracy, confusion matrix)
    - ถ้า test ไม่มี label → predict (save CSV + bar chart)
    """
    config = load_config(config_path)
    device = get_device()
    
    # 🔹 นับจำนวน class และดึง class names
    num_classes, class_names = get_num_classes_from_train(config)
    
    # 🔹 โหลด test dataset
    dataset, has_labels = get_test_dataset(config)
    
    # 🔹 สร้าง DataLoader
    if has_labels:
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        print(f"✅ Test set loaded with labels (total: {len(dataset)} images)")
    else:
        print(f"✅ Test set loaded without labels (total: {len(dataset)} images)")
    
    # 🔹 Checkpoint และ output directories
    checkpoint_dir = Path(config["output"]["checkpoints"]).resolve()
    output_dir = Path("Output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔹 Model list
    available_models = ["resnet", "efficientnet", "vit", "convnext"]
    model_list = model_list or available_models
    model_list = [m for m in model_list if m in available_models]
    
    print(f"\n🔍 Evaluating models: {model_list}")
    print("="*70)
    
    # 🔹 วนประเมินแต่ละโมเดล
    for name in model_list:
        print(f"\n📊 Processing {name.upper()}...")
        
        try:
            # โหลดโมเดล (ลอง finetune ก่อน ถ้าไม่มีค่อยลอง pretrain)
            model = load_model(name, checkpoint_dir, num_classes=num_classes, phase="finetune")
            model.to(device)
            
            if has_labels:
                # Test set มี label → evaluate
                evaluate_labeled_test(model, dataloader, device, class_names, name, output_dir)
            else:
                # Test set ไม่มี label → predict
                predict_unlabeled_test(model, dataset, device, class_names, name, output_dir)
        
        except Exception as e:
            print(f"❌ Error with {name}: {e}")
            continue
    
    print("\n" + "="*70)
    print("🎉 All models processed successfully!")
    print(f"📁 Results saved to: {output_dir}")

# ============================ MAIN ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate/Predict with trained models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to config file")
    parser.add_argument("--models", type=str, nargs="*", 
                       help="Models to evaluate (resnet, efficientnet, convnext, vit)")
    args = parser.parse_args()

    compare_models(args.config, args.models)