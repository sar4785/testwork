import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from src.utils.metric import extract_features, cosine_knn_predict, compute_clustering_quality
import csv

class Trainer:
    def __init__(self, config, model_name=None):
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("Invalid config format")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_cfg = self.config.get("model", {}).get(model_name, self.config.get("default", {}))
        self.epochs = self.model_cfg.get('epochs', 10)
        self.batch_size = self.model_cfg.get('batch_size', 32)
        self.lr = self.model_cfg.get('learning_rate', 1e-4)
        self.patience = self.config['training'].get('patience', 3)
        self.save_best_only = self.config['training'].get('save_best_only', True)
        self.early_stopping = self.config['training'].get('early_stopping', True)

        data_cfg = self.config['data']
        self.train_dir = Path(data_cfg['train'])
        self.val_dir = Path(data_cfg['val'])
        self.test_dir = Path(data_cfg['test'])
        self.checkpoint_dir = Path(self.config['output']['checkpoints'])
        self.output_csv = Path(self.config['output'].get('predictions', "predictions.csv"))
        self.metric_csv = self.checkpoint_dir / "evaluation_metrics.csv"  # 🆕
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 🔹 Load datasets
    # -----------------------------------------------------
    def get_dataloaders(self):
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(self.train_dir, transform=transform_train)
        self.num_classes = len(train_dataset.classes)
        allowed_classes = train_dataset.classes

        if any((self.val_dir / cls).exists() for cls in allowed_classes):
            val_dataset = datasets.ImageFolder(self.val_dir, transform=transform_eval)
        else:
            val_dataset = UnlabeledDataset(self.val_dir, transform=transform_eval)

        test_dataset = UnlabeledDataset(self.test_dir, transform=transform_eval)

        # Balanced sampler
        train_targets = np.array(train_dataset.targets)
        class_counts = np.bincount(train_targets, minlength=self.num_classes)
        weights = 1. / (class_counts + 1e-6)
        samples_weights = weights[train_targets].tolist()
        sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    # -----------------------------------------------------
    # 🔹 Supervised training
    # -----------------------------------------------------
    def train_supervised(self, model, save_name="model.pth", fine_tune=False):
        train_loader, val_loader, test_loader = self.get_dataloaders()
        model = model.to(self.device)

        # 🧠 Load SimCLR pretrained weights
        simclr_path = Path("Output/checkpoints/simclr_pretrained.pth")
        if simclr_path.exists():
            print("🔁 Loading pretrained SimCLR encoder weights...")
            simclr_weights = torch.load(simclr_path, map_location=self.device)
            model.load_state_dict(simclr_weights, strict=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_val_score = -1
        save_path = self.checkpoint_dir / save_name

        print(f"\n🚀 Start Supervised Training ({self.model_name})")

        for epoch in range(self.epochs):
            model.train()
            train_loss, correct, total = 0, 0, 0

            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            val_sil, val_intra = self.evaluate(model, val_loader, phase="val")
            print(f"✅ Epoch [{epoch+1}/{self.epochs}] | Train Acc: {train_acc:.4f} | "
                  f"Val_Silhouette: {val_sil:.4f} | Val_IntraDist: {val_intra:.4f}")

            if val_sil > best_val_score:
                best_val_score = val_sil
                torch.save(model.state_dict(), save_path)
                print(f"💾 Saved best model → {save_path}")

        # 🧮 Evaluate on all sets after training
        print("\n📊 Final Evaluation on all phases...")
        results = []
        for phase, loader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
            sil, intra = self.evaluate(model, loader, phase=phase)
            results.append({"phase": phase, "model": self.model_name,
                            "silhouette_score": sil, "intra_cluster_distance": intra})

        model_metric_path = self.checkpoint_dir / f"metrics_{self.model_name}.csv"
        df = pd.DataFrame(results)
        df.to_csv(model_metric_path, index=False)
        print(f"📊 Saved metrics for {self.model_name} → {model_metric_path}")

        merged_path = self.checkpoint_dir / "all_models_metrics.csv"
        file_exists = merged_path.exists()
        with open(merged_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["phase", "model", "silhouette_score", "intra_cluster_distance"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
        print(f"📄 Updated combined metrics → {merged_path}")    

    # -----------------------------------------------------
    # 🔹 Evaluation (supports train/val/test)
    # -----------------------------------------------------
    def evaluate(self, model, loader, criterion=None, phase="val"):
        model.eval()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # 🧠 Extract feature-based metrics
        print(f"\n⚙️ Evaluating {phase.upper()} phase using feature similarity...")
        train_loader, _, _ = self.get_dataloaders()
        train_features, train_labels = extract_features(model, train_loader, self.device)
        features, _ = extract_features(model, loader, self.device)
        preds, confidences = cosine_knn_predict(train_features, train_labels, features)
        silhouette, intra_dist = compute_clustering_quality(features, n_clusters=self.num_classes)
        print(f"📈 {phase.upper()} → Silhouette: {silhouette:.4f} | Intra-cluster Dist: {intra_dist:.4f}")

        # 🧾 Pseudo-label CSV (only for val/test)
        if phase in ["val", "test"]:
            inv_map = {v: k for k, v in train_loader.dataset.class_to_idx.items()}
            pred_names = [inv_map[int(p.item())] for p in preds]
            val_paths = getattr(loader.dataset, "image_paths", [])
            if len(val_paths) > 0:
                df = pd.DataFrame({
                    "filename": [p.name for p in val_paths],
                    "pseudo_label_idx": preds.numpy(),
                    "pseudo_label_name": pred_names,
                    "confidence": confidences.numpy()
                })
                out_path = self.checkpoint_dir / f"pseudo_{phase}_predictions.csv"
                df.to_csv(out_path, index=False)
                print(f"📄 Saved pseudo-{phase} predictions → {out_path}")

        return silhouette, intra_dist


# -----------------------------------------------------
# 🔹 Dataset for unlabeled images
# -----------------------------------------------------
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            self.image_paths.extend(self.root_dir.glob(f"**/{ext}"))
        print(f"📁 Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Failed to open {path}: {e}")
            img = Image.new("RGB", (224, 224), color=0)
        if self.transform:
            img = self.transform(img)
        return img, path.name
