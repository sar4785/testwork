# src/auto_label.py
import os
import torch
import shutil
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from src.Unlabeled import UnlabeledImageDataset  # สมมติว่าคุณมีไฟล์นี้แล้ว
from src.train import TrainModel


def auto_label(config_path="configs/config.yaml", threshold=0.9):
    """
    ใช้โมเดลที่ train แล้วมาสร้าง label ให้กับภาพในโฟลเดอร์ WaferMap_PNG
    แล้วจัดเรียงไฟล์ใหม่เป็นโครงสร้างโฟลเดอร์ตาม class ที่ทำนายได้
    """
    print("🤖 Starting Auto-Labeling Pipeline...")

    # โหลด config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ตั้งค่าอุปกรณ์
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # โหลด LabelEncoder
    checkpoint_dir = Path(config['output']['checkpoints'])
    le_path = checkpoint_dir / "label_encoder.pkl"
    if not le_path.exists():
        raise FileNotFoundError("❌ ไม่พบ LabelEncoder กรุณา train โมเดลก่อน")

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    # โหลดโมเดล
    trainer = TrainModel(config_path=config_path)
    # ดึง num_classes จาก LabelEncoder แทนการโหลด dataset
    num_classes = len(le.classes_)

    model = trainer.create_custom_cnn_model(num_classes)
    model_path = checkpoint_dir / "wafer_classifier.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("🧠 Model loaded successfully")

    # กำหนด path
    unlabeled_dir = Path(config['data']['wafer_map_png'])
    output_dir = Path(config['data']['auto_labeled'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ตั้งค่า transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # สร้าง dataset และ dataloader
    dataset = UnlabeledImageDataset(root=unlabeled_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"📂 Found {len(dataset)} images to auto-label")

    # เริ่มการทำนาย
    high_conf_count = 0
    low_conf_count = 0

    with torch.no_grad():
        for inputs, filenames in tqdm(dataloader, desc="🔍 Predicting labels"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(filenames)):
                confidence = confs[i].item()
                predicted_class_idx = preds[i].item()
                predicted_label = le.classes_[predicted_class_idx]
                src_path = Path(filenames[i])  # ชื่อไฟล์เต็มจาก dataset

                if confidence >= threshold:
                    # มั่นใจพอ → จัดเก็บตาม label
                    target_dir = output_dir / predicted_label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = target_dir / src_path.name

                    try:
                        shutil.copy(src_path, dst_path)
                        high_conf_count += 1
                    except Exception as e:
                        print(f"⚠️ Error copying {src_path.name}: {e}")
                else:
                    # ไม่มั่นใจ → เก็บไว้ในโฟลเดอร์ 'Uncertain'
                    uncertain_dir = output_dir / "Uncertain"
                    uncertain_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = uncertain_dir / f"{src_path.stem}_conf{confidence:.2f}{src_path.suffix}"
                    try:
                        shutil.copy(src_path, dst_path)
                        low_conf_count += 1
                    except Exception as e:
                        print(f"⚠️ Error copying {src_path.name}: {e}")

    print(f"✅ Auto-labeling complete!")
    print(f"   - จัดเก็บตาม label แล้ว: {high_conf_count} ไฟล์")
    print(f"   - ความมั่นใจต่ำ (< {threshold}): {low_conf_count} ไฟล์ (อยู่ในโฟลเดอร์ 'Uncertain')")
    print(f"📁 Results saved to: {output_dir}")