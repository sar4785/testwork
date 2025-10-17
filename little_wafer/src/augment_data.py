import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

def load_config(config_path="configs/config.yaml"):
    """โหลด config ภายในฟังก์ชัน"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
                
def apply_circular_mask(img):
    """ล้างขอบภาพนอก wafer ให้เป็นสีดำ"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center) - 2  # ลดเล็กน้อยกันขอบขาว
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(img, mask)
    return masked

def augment_image(img):
    augmented = []
    augmented.append(apply_circular_mask(img.copy()))  # Masked Original
    flip_lr = cv2.flip(img, 1)         # Flip Left-Right
    augmented.append(apply_circular_mask(flip_lr))
    flip_ud = cv2.flip(img, 0)         # Flip Up-Down
    augmented.append(apply_circular_mask(flip_ud))         # Flip Up-Down

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0) # Black border
        rotated = apply_circular_mask(rotated)
        augmented.append(rotated)
    return augmented

def process_real_wafer_data(   
    input_dir=None,
    output_dir="Data/Augmented_Real/",
    target_labels=["Edge-Loc", "Edge-Ring", "Random", "Scratch"]
):
    if input_dir is None:
        config = load_config()
        input_dir = Path(config['data']['real']) 
        
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    print(f"🔍 Reading real wafer maps from: {input_path}")
    print(f"🎯 Target labels: {target_labels}")

    total_saved = 0
    for label_folder in input_path.iterdir():
        if not label_folder.is_dir():
            continue

        label_name = label_folder.name
        if label_name not in target_labels:
            print(f"⏭️ Skipping {label_name}")
            continue

        label_output = output_path / label_name
        label_output.mkdir(exist_ok=True)

        print(f"\n📁 Processing label: {label_name}...")
        image_files = list(label_folder.glob("*.png")) + list(label_folder.glob("*.jpg"))
        if len(image_files) == 0:
            print(f"⚠️ No images found in {label_folder}")
            continue

        count = 0
        for img_file in tqdm(image_files, desc=f"Augmenting {label_name}"):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            augmented_images = augment_image(img)
            for aug_img in augmented_images:
                filename = f"{label_name}_{count:05d}.png"
                cv2.imwrite(str(label_output / filename), aug_img)
                count += 1
                total_saved += 1

        print(f"✅ Generated {count} images for '{label_name}'")

    print(f"\n🎉 Augmentation completed! Total saved: {total_saved}")
    print(f"📁 Output path: {output_path}")
