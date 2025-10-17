# src/auto_split.py
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.simclr_trainer import train_simclr_from_config  

def auto_split_dataset(config_path="configs/config.yaml", val_ratio=0.7, test_ratio=0.3, pretrain_simclr=True):
    """
    แบ่ง dataset ออกจาก train folder เดิม เป็น val / test เท่านั้น
    และสามารถ pretrain SimCLR ได้โดยอัตโนมัติ
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    source_dir = Path(config["data"]["augment_data"])
    base_dir = source_dir.parent  # ./Data/
    new_val_dir = base_dir / "val_split"
    new_test_dir = base_dir / "test_split"
    
    new_val_dir.mkdir(parents=True, exist_ok=True)
    new_test_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = list(source_dir.rglob("*.png"))
    if len(all_images) == 0:
        raise RuntimeError(f"❌ ไม่พบไฟล์ .png ใน {source_dir}")

    print(f"📦 พบภาพทั้งหมด {len(all_images)} จาก {source_dir}")
   
    val_imgs, test_imgs = train_test_split(all_images, test_size=test_ratio, random_state=42)
    print(f"Split data to → Validation: {len(val_imgs)} | Test: {len(test_imgs)}")
 
    def copy_images(image_list, target_root):
        for img_path in image_list:
            target_path = target_root / img_path.name
            if not target_path.exists():
                shutil.copy(img_path, target_path)

    print("Copy data to folder val/test ...")
    copy_images(val_imgs, new_val_dir)
    copy_images(test_imgs, new_test_dir)

    print(f"Split complete!\n - Validation: {new_val_dir}\n - Test: {new_test_dir}")
 
    if pretrain_simclr:
        print("\n🧠 เริ่ม pretraining SimCLR บนข้อมูล unlabeled ...")
        train_simclr_from_config(config_path)

    # ✅ คืนค่า path ทั้งหมด (string)
    return str(source_dir), str(new_val_dir), str(new_test_dir)
 
 