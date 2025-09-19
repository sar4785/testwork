# src/train.py
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class TrainModel:
    def __init__(self, config_path='configs/config.yaml'):
        """
        Initialize the training pipeline.
        Loads configuration and sets up the computation device (GPU/CPU).
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    def load_dataset_from_directory(self):
        """
        Load image dataset from directory using PyTorch's ImageFolder.
        Applies data augmentation and splits data into train/val/test sets.
        Returns DataLoaders for train and val, LabelEncoder, number of classes, and test DataLoader.
        """
        data_dirs = [
            Path(self.config['data']['wafer_map_png']),
            Path(self.config['data']['kaggle_png'])
        ]
        for d in data_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Dataset directory not found: {d}")

        # Define data transformations with augmentations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.1, 0.1))
        ])

        # โหลด dataset จากแต่ละโฟลเดอร์
        datasets = [ImageFolder(root=d, transform=transform) for d in data_dirs]
        
        # รวมเป็น dataset เดียว
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset(datasets)
        
        # class mapping ใช้อันแรก (ImageFolder จะใช้ชื่อโฟลเดอร์)
        class_names = datasets[0].classes
        num_classes = len(class_names)
        

        # Split dataset: 70% train, 15% val, 15% test
        total_size = len(full_dataset)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        batch_size = self.config['model']['batch_size']

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Create and fit LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(class_names)

        return train_loader, val_loader, test_loader, le, num_classes

    def create_custom_cnn_model(self, num_classes):
        """
        Create a custom CNN model architecture.
        """
        class CustomCNN(nn.Module):
            def __init__(self, num_classes):
                super(CustomCNN, self).__init__()
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 224 → 112
                    # Block 2
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 112 → 56
                    # Block 3
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 56 → 28
                    # Block 4
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # 28 → 14
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(128 * 14 * 14, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        model = CustomCNN(num_classes).to(self.device)
        return model

    def run(self):
        """
        Main training loop.
        Loads data, creates model, trains for specified epochs, and saves artifacts.
        Returns the trained model, label encoder, training history, and test loader for evaluation.
        """
        print("🧠 Starting Training Pipeline...")

        # Load datasets
        train_loader, val_loader, test_loader, le, num_classes = self.load_dataset_from_directory()
        print(f"📚 Found {num_classes} classes: {list(le.classes_)}")

        # Create model
        model = self.create_custom_cnn_model(num_classes)
        print("✅ Model created")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )

        # Training history
        epochs = self.config['model']['epochs']
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        # Training loop
        for epoch in range(epochs):
            # Training Phase
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix({'loss': loss.item()})

            epoch_train_loss = running_loss / len(train_loader.dataset)
            history['train_loss'].append(epoch_train_loss)

            # Validation Phase
            model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_running_loss / len(val_loader.dataset)
            val_acc = correct / total
            scheduler.step(val_loss) # Update learning rate based on validation loss

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {epoch_train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f} - "
                  f"val_acc: {val_acc:.4f}")

        print("✅ Training Done")

        # Save model and encoder
        checkpoint_dir = Path(self.config['output']['checkpoints'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / 'wafer_classifier.pth'
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to: {model_path}")

        le_path = checkpoint_dir / 'label_encoder.pkl'
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)
        print(f"💾 LabelEncoder saved to: {le_path}")

        # Return necessary components for evaluation
        return model, le, history, test_loader


def evaluate_model(model, dataloader, label_encoder, device, output_config):
    """
    Evaluate the trained model on a given dataloader (typically test set).
    Generates and saves a classification report and confusion matrix.
    
    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the test set.
        label_encoder: Fitted LabelEncoder for converting numeric labels back to class names.
        device: The device (CPU/GPU) the model is on.
        output_config: Output configuration dictionary from YAML for saving paths.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    class_names = label_encoder.classes_

    # Generate and save Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    predictions_dir = Path(output_config['predictions'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(predictions_dir / "test_metrics.csv")
    print("\n📊 Classification Report:")
    print(df_report)

    # Generate and save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.savefig(predictions_dir / "confusion_matrix.png")
    plt.close()
    print(f"📈 Confusion matrix saved to: {predictions_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    # Example usage: Run the entire pipeline including evaluation
    trainer = TrainModel(config_path='configs/config.yaml')
    model, le, history, test_loader = trainer.run()

    # Evaluate the model on the test set
    evaluate_model(
        model=model,
        dataloader=test_loader,
        label_encoder=le,
        device=trainer.device,
        output_config=trainer.config['output']
    )