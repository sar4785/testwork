import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import os

def load_config(config_path="configs/config.yaml"):
    """โหลด config"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_circular_mask(img):
    """ล้างขอบภาพนอก wafer ให้เป็นสีดำ"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center) - 2
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(img, mask)
    return masked

def augment_image(img):
    """สร้างภาพ augmented (flip, rotate, mask)"""
    augmented = []
    augmented.append(apply_circular_mask(img.copy()))  # Original (masked)
    flip_lr = cv2.flip(img, 1)
    augmented.append(apply_circular_mask(flip_lr))
    flip_ud = cv2.flip(img, 0)
    augmented.append(apply_circular_mask(flip_ud))
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated = apply_circular_mask(rotated)
        augmented.append(rotated)
    return augmented

def process_real_wafer_data_single_folder():
    """ทำ Augment ภาพทั้งหมดในโฟลเดอร์เดียว"""
    config = load_config()
    input_dir = Path(config['data']['real'])
    output_dir = Path(config['data']['augment_data'])

    if not input_dir.exists():
        raise FileNotFoundError(f"❌ Input folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Reading wafer maps from: {input_dir}")
    print(f"💾 Saving augmented images to: {output_dir}")

    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    if not image_files:
        print(f"⚠️ No images found in {input_dir}")
        return

    total_saved = 0
    for count, img_file in enumerate(tqdm(image_files, desc="Augmenting images")):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipping invalid image: {img_file.name}")
            continue

        augmented_images = augment_image(img)
        for i, aug_img in enumerate(augmented_images):
            filename = f"{img_file.stem}_aug{i}_{count:05d}.png"
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), aug_img)
            total_saved += 1

    print(f"\n🎉 Augmentation completed!")
    print(f"📸 Total augmented images saved: {total_saved}")
    print(f"📁 Output path: {output_dir}")
