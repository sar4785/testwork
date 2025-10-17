import os
import cv2
import time
import torch
import yaml
import argparse
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from src.kaggledata import KaggleDataProcessor

# Import Models
from src.Model.ResNet import ResNetModel
from src.Model.EfficientNet import EfficientNetModel
from src.Model.ConvNeXt import ConvNeXtModel
from src.Model.Vit import ViTModel

# DATASET HANDLER
class UnsupervisedDataset(Dataset):
    """Dataset for images without labels"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(self.root_dir.glob(f"**/{ext}"))
        print(f"📁 Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Failed to open {img_path}: {e}")
            img = Image.new("RGB", (224, 224), color=0)
        if self.transform:
            img = self.transform(img)
        return img, -1
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
# Utility Functions
def load_config(config_path: str):
    """Load configuration YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_device():
    """Select GPU if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    return device

def get_test_dataset(config, unsupervised=False):
    """Return test dataset and class list"""
    test_dir = Path(config["data"]["test"])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if unsupervised:
        dataset = UnsupervisedDataset(test_dir, transform=transform)
        allowed_classes = config["data"].get("allowed_classes")
        print(f"🔍 Using classes: {allowed_classes}")
        
        if not allowed_classes:
            allowed_classes = list(KaggleDataProcessor.CLASS_NAMES.values())
        return dataset, allowed_classes
    else:
        dataset = datasets.ImageFolder(test_dir, transform=transform)
        allowed_classes = config["data"].get("allowed_classes") or \
                          sorted({Path(p).parent.name for p, _ in dataset.samples})
        print(f"🔍 Classes detected: {allowed_classes}")
        return dataset, allowed_classes

# 🧠 MODEL LOADING
def load_model(model_name, checkpoint_dir):
    """Load model with given name and checkpoint"""
    model_classes = {
        "resnet": ResNetModel,
        "efficientnet": EfficientNetModel,
        "vit": ViTModel,
        "convnext": ConvNeXtModel,
    }
    
    ckpt_files = {
        "resnet": "resnet18.pth",
        "efficientnet": "efficientnet_b0.pth",
        "vit": "vit_b16.pth",
        "convnext": "convnext_tiny.pth",
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model name: {model_name}")

    ckpt_path = Path(checkpoint_dir) / ckpt_files[model_name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    last_layer_keys = [k for k in state_dict.keys() if "weight" in k and len(state_dict[k].shape) == 2]
    num_classes = state_dict[last_layer_keys[-1]].shape[0] if last_layer_keys else 8
    model_class = model_classes[model_name]   
    model = model_class.create_model(num_classes=num_classes)
    return model
    
# EVALUATION
def evaluate_model(model, dataloader, device, class_names, save_prefix, output_dir):
    """Evaluate model and save report + confusion matrix"""
    model.eval()
    y_true, y_pred = [], []

    start = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print(f"⏱️ Evaluation done in {time.time() - start:.2f}s")

    # Save classification report
    df_report = pd.DataFrame(
        classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    ).transpose()
    report_path = output_dir / f"{save_prefix}_metrics.csv"
    df_report.to_csv(report_path)
    print(f"📊 Saved report → {report_path}")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {save_prefix}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / f"{save_prefix}_confusion.png")
    plt.close()
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(output_dir / f"{save_prefix}_confusion_{timestamp}.png")

# UNSUPERVISED PREDICTION
def predict_unsupervised(model, dataloader, class_names, output_dir, save_csv="unsupervised_predictions.csv"):
    """Predict on unlabeled data"""
    model.eval()
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = dataloader.batch_size

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Predicting")):
            inputs = inputs.to(next(model.parameters()).device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(inputs)):
                img_path = dataloader.dataset.image_paths[batch_idx * batch_size + i]
                pred_class = class_names[preds[i].item()]
                pred_prob = probs[i][preds[i]].item()
                results.append({
                    'image_path': str(img_path),
                    'predicted_class': pred_class,
                    'confidence': pred_prob
                })

    df = pd.DataFrame(results)
    csv_path = output_dir / save_csv
    df.to_csv(csv_path, index=False)
    print(f"✅ Predictions saved to {csv_path}")
    visualize_predictions(df, output_dir)
    return df

def visualize_predictions(df_pred, output_dir, n_samples=5):
    """Display images with their predictions"""
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    if n_samples == 1:
        axes = [axes]

    for i, row in df_pred.head(n_samples).iterrows():
        img = cv2.imread(row['image_path'])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"{row['predicted_class']} ({row['confidence']:.2f})")
        axes[i].axis('off')

    plt.tight_layout()
    plot_path = output_dir / "prediction_samples.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"🖼️ Sample predictions saved to {plot_path}")

# ENSEMBLE VOTING
def ensemble_voting(config_path="configs/config.yaml", model_list=None, unsupervised=False):
    """Perform majority voting across multiple models"""
    print("\n🗳️ Starting Ensemble Voting...")
    config = load_config(config_path)
    dataset, class_names = get_test_dataset(config, unsupervised=unsupervised)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = get_device()

    checkpoint_dir = Path(config["output"]["checkpoints"])
    output_dir = Path(config["output"]["predictions"])
    output_dir.mkdir(parents=True, exist_ok=True)

    available = ["resnet", "efficientnet", "vit", "convnext"]
    model_list = [m for m in (model_list or available) if m in available]

    models = []
    for name in model_list:
        model = load_model(name, checkpoint_dir)
        if model:
            model.to(device).eval()
            models.append((name, model))

    if not models:
        print("❌ No models loaded. Abort ensemble.")
        return

    print(f"🧠 Loaded models: {[m[0] for m in models]}")

    if unsupervised:
        print("🔍 Running UNSUPERVISED ensemble prediction...")
        all_preds = []
        image_paths = []
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Predicting")):
                inputs = inputs.to(device)
                batch_votes = [torch.argmax(m(inputs), dim=1).cpu().numpy() for _, m in models]
                votes = np.array(batch_votes)
                final_preds = [Counter(votes[:, i]).most_common(1)[0][0] for i in range(votes.shape[1])]
                all_preds.extend(final_preds)
                # Store image paths
                start_idx = len(image_paths)
                for j in range(len(inputs)):
                    img_path = dataloader.dataset.image_paths[start_idx + j]
                    image_paths.append(img_path)

        df = pd.DataFrame({
            'image_path': image_paths,
            'predicted_class': [class_names[p] for p in all_preds],
            'confidence': [1.0] * len(all_preds)  # Placeholder as no real probabilities
        })
        csv_path = output_dir / "ensemble_unsupervised_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Ensemble predictions saved to {csv_path}")
        visualize_predictions(df, output_dir)
        return

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            batch_votes = [torch.argmax(m(inputs), dim=1).cpu().numpy() for _, m in models]
            votes = np.array(batch_votes)
            final_preds = [Counter(votes[:, i]).most_common(1)[0][0] for i in range(votes.shape[1])]
            all_preds.extend(final_preds)
            all_labels.extend(labels.numpy())

    print("\n📊 Evaluating Ensemble Results...")
    df_report = pd.DataFrame(
        classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    ).transpose()
    df_report.to_csv(output_dir / "ensemble_voting_metrics.csv")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Ensemble Confusion Matrix (Majority Vote)")
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_voting_confusion.png")
    plt.close()
    print(f"✅ Ensemble results saved to {output_dir}")

# COMPARE MODELS
def compare_models(config_path="configs/config.yaml", model_list=None, unsupervised=False):
    """Evaluate each model individually"""
    config = load_config(config_path)
    dataset, class_names = get_test_dataset(config, unsupervised=unsupervised)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = get_device()
    checkpoint_dir = Path(config["output"]["checkpoints"])
    output_dir = Path(config["output"]["predictions"])
    output_dir.mkdir(parents=True, exist_ok=True)

    available = ["resnet", "efficientnet", "vit", "convnext"]
    model_list = [m for m in (model_list or available) if m in available]

    if unsupervised:
        print("🔍 Running UNSUPERVISED prediction...")
        for name in model_list:
            print(f"\n🚀 Predicting with {name.upper()}...")
            model = load_model(name,checkpoint_dir)
            if model:
                model.to(device)
                predict_unsupervised(model, dataloader, class_names, output_dir, 
                                    save_csv=f"{name}_unsupervised_predictions.csv")
        print("✅ Unsupervised prediction completed!")
        return

    for name in model_list:
        print(f"\n🔍 Evaluating {name.upper()}...")
        model = load_model(name,checkpoint_dir)
        if model:
            model.to(device)
            evaluate_model(model, dataloader, device, class_names, name, output_dir)

    print("\n🎉 All models evaluated successfully.")

# MAIN ENTRY
if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--models", type=str, nargs="*", help="Models to evaluate")
    parser.add_argument("--ensemble", action="store_true", help="Run ensemble voting")
    parser.add_argument("--unsupervised", action="store_true", help="Run unsupervised prediction")
    args = parser.parse_args()

    if args.ensemble:
        ensemble_voting(args.config, args.models, args.unsupervised)
    else:
        compare_models(args.config, args.models, args.unsupervised)