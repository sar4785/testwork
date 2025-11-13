# ===============================================
# auto_split.py (แยก pre-train และ fine-tune split)
# ===============================================
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset_generic(
    source_dir,
    train_output_dir,
    val_output_dir,
    val_ratio=0.2,
    move_files=False,
    dataset_name="Dataset"
):
    """
    🔹 ฟังก์ชันทั่วไปสำหรับแบ่ง dataset
    
    Parameters:
    -----------
    source_dir : Path
        โฟลเดอร์ต้นทางที่มีข้อมูล
    train_output_dir : Path
        โฟลเดอร์ปลายทางสำหรับ train
    val_output_dir : Path
        โฟลเดอร์ปลายทางสำหรับ validation
    val_ratio : float
        สัดส่วนของ validation set
    move_files : bool
        True = ย้ายไฟล์, False = copy ไฟล์
    dataset_name : str
        ชื่อสำหรับแสดงผล
    
    Returns:
    --------
    dict : สรุปผลการแบ่งข้อมูล
    """
    
    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)
    
    class_folders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
    print(f"\n📂 [{dataset_name}] Found {len(class_folders)} classes: {[f.name for f in class_folders]}")

    total_val = 0
    total_train = 0
    class_summary = []

    for class_dir in class_folders:
        class_name = class_dir.name
        
        # ดึงไฟล์ภาพทั้งหมด
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        
        if len(images) == 0:
            print(f"⚠️ No images found in {class_dir}, skipping...")
            continue

        # สร้างโฟลเดอร์ class
        (train_output_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_output_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # แบ่ง train/val
        train_imgs, val_imgs = train_test_split(
            images, 
            test_size=val_ratio, 
            random_state=42,
            shuffle=True
        )

        # ย้ายหรือ copy ไฟล์
        for img_path in train_imgs:
            target_path = train_output_dir / class_name / img_path.name
            if move_files:
                shutil.move(str(img_path), str(target_path))
            else:
                shutil.copy(str(img_path), str(target_path))
        
        for img_path in val_imgs:
            target_path = val_output_dir / class_name / img_path.name
            if move_files:
                shutil.move(str(img_path), str(target_path))
            else:
                shutil.copy(str(img_path), str(target_path))
        
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        
        class_summary.append({
            "class": class_name,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "total": len(images)
        })
        
        print(f"   [{class_name:12s}] Train: {len(train_imgs):4d} | Val: {len(val_imgs):4d}")
    
    return {
        "train_total": total_train,
        "val_total": total_val,
        "class_summary": class_summary,
        "train_dir": str(train_output_dir),
        "val_dir": str(val_output_dir)
    }


def split_pretrain_and_finetune(
    val_ratio=0.2,
    move_files=True,
    base_dir="./Data"
):
    """
    🔹 แบ่งข้อมูลเป็น 2 ชุดแยกกัน:
    
    1. Pre-train split:
       - Source: Data/Synthetic/png/clean
       - Output: Data/pre_train และ Data/pre_val
    
    2. Fine-tune split:
       - Source: Data/Synthetic/png/noisy (หรือ Data/train)
       - Output: Data/train_split และ Data/val_split
    
    Parameters:
    -----------
    val_ratio : float
        สัดส่วนของ validation set (default: 0.2 = 20%)
    move_files : bool
        True = ย้ายไฟล์, False = copy ไฟล์
    base_dir : str
        โฟลเดอร์หลักของข้อมูล
    """
    
    base_path = Path(base_dir).resolve()
    synth_path = Path("./Data/Synthetic/png").resolve()
    
    print("\n" + "="*70)
    print("🚀 STARTING DATASET SPLITTING")
    print("="*70)
    
    results = {}
    
    # ============ STEP 1: Pre-train Split (Clean Data) ============
    clean_dir = synth_path / "clean"
    pre_train_dir = base_path / "pre_train"
    pre_val_dir = base_path / "pre_val"
    
    if clean_dir.exists():
        print(f"\n📚 STEP 1: Splitting PRE-TRAIN data (clean)")
        print(f"   Source: {clean_dir}")
        
        results["pretrain"] = split_dataset_generic(
            source_dir=clean_dir,
            train_output_dir=pre_train_dir,
            val_output_dir=pre_val_dir,
            val_ratio=val_ratio,
            move_files=move_files,
            dataset_name="Pre-train (Clean)"
        )
        
        print(f"\n   ✅ Pre-train: {results['pretrain']['train_total']} images → {pre_train_dir}")
        print(f"   ✅ Pre-val:   {results['pretrain']['val_total']} images → {pre_val_dir}")
    else:
        print(f"\n⚠️ Clean data folder not found: {clean_dir}")
        print("   Skipping pre-train split...")
    
    # ============ STEP 2: Fine-tune Split (Noisy Data) ============
    noisy_dir = synth_path / "noisy"
    train_split_dir = base_path / "train_split"  # ✅ แก้ไขตรงนี้!
    val_split_dir = base_path / "val_split"
    
    # ถ้าไม่มี noisy folder ให้ใช้ Data/train
    if not noisy_dir.exists():
        noisy_dir = base_path / "train"
        print(f"\n⚠️ Noisy folder not found, using: {noisy_dir}")
    
    if noisy_dir.exists():
        print(f"\n🔧 STEP 2: Splitting FINE-TUNE data (noisy)")
        print(f"   Source: {noisy_dir}")
        
        results["finetune"] = split_dataset_generic(
            source_dir=noisy_dir,
            train_output_dir=train_split_dir,
            val_output_dir=val_split_dir,
            val_ratio=val_ratio,
            move_files=move_files,
            dataset_name="Fine-tune (Noisy)"
        )
        
        print(f"\n   ✅ Train: {results['finetune']['train_total']} images → {train_split_dir}")
        print(f"   ✅ Val:   {results['finetune']['val_total']} images → {val_split_dir}")
    else:
        print(f"\n⚠️ Noisy data folder not found: {noisy_dir}")
        print("   Skipping fine-tune split...")
    
    # ============ SUMMARY ============
    print("\n" + "="*70)
    print("📊 SPLIT SUMMARY")
    print("="*70)
    
    if "pretrain" in results:
        print(f"\n🔹 PRE-TRAIN (Clean Data):")
        print(f"   Train:      {results['pretrain']['train_total']:5d} images")
        print(f"   Validation: {results['pretrain']['val_total']:5d} images")
        print(f"   Ratio:      {100*(1-val_ratio):.0f}% / {100*val_ratio:.0f}%")
    
    if "finetune" in results:
        print(f"\n🔹 FINE-TUNE (Noisy Data):")
        print(f"   Train:      {results['finetune']['train_total']:5d} images")
        print(f"   Validation: {results['finetune']['val_total']:5d} images")
        print(f"   Ratio:      {100*(1-val_ratio):.0f}% / {100*val_ratio:.0f}%")
    
    print("="*70)
    
    # ============ SAVE SUMMARY ============
    summary_data = []
    
    if "pretrain" in results:
        for item in results["pretrain"]["class_summary"]:
            summary_data.append({
                "dataset": "pre-train",
                "class": item["class"],
                "train": item["train"],
                "val": item["val"],
                "total": item["total"]
            })
    
    if "finetune" in results:
        for item in results["finetune"]["class_summary"]:
            summary_data.append({
                "dataset": "fine-tune",
                "class": item["class"],
                "train": item["train"],
                "val": item["val"],
                "total": item["total"]
            })
    
    print("\n✅ All splits completed!\n")
    return results


def auto_split_dataset(
    config_path="configs/config.yaml",
    val_ratio=0.2,
    move_files=True,
    use_clean_only=True
):
    """
    🔹 ฟังก์ชันเดิมเพื่อ backward compatibility
    แบ่ง dataset จากโฟลเดอร์ train ออกเป็น train / val
    
    ⚠️ แนะนำให้ใช้ split_pretrain_and_finetune() แทน
    """
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_synth_dir = Path("./Data/Synthetic/png").resolve()
    clean_dir = base_synth_dir / "clean"
    
    if clean_dir.exists() and use_clean_only:
        train_dir = clean_dir
        print(f"✅ Using CLEAN data from: {train_dir}")
    else:
        train_dir = Path(config["data"]["train"]).resolve()
        print(f"✅ Using data from config: {train_dir}")
    
    base_dir = Path("./Data").resolve()
    train_split_dir = base_dir / "train_split"
    val_split_dir = base_dir / "val_split"
    
    result = split_dataset_generic(
        source_dir=train_dir,
        train_output_dir=train_split_dir,
        val_output_dir=val_split_dir,
        val_ratio=val_ratio,
        move_files=move_files,
        dataset_name="Default"
    )
    
    print("\n" + "="*70)
    print("📊 SPLIT SUMMARY")
    print("="*70)
    print(f"Total Train:      {result['train_total']:5d} images → {train_split_dir}")
    print(f"Total Validation: {result['val_total']:5d} images → {val_split_dir}")
    print(f"Split Ratio:      {100*(1-val_ratio):.0f}% train / {100*val_ratio:.0f}% val")
    print("="*70)
    
    summary_df = pd.DataFrame(result["class_summary"])
    summary_path = base_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Summary saved to: {summary_path}\n")

    return str(val_split_dir)


def merge_random_and_none(base_dir="./Data/Synthetic/png"):
    """
    🔹 รวมโฟลเดอร์ 'none' เข้ากับ 'random' (ทั้ง clean และ noisy)
    """
    base_path = Path(base_dir).resolve()
    
    print("\n" + "="*70)
    print("🔄 MERGING 'none' INTO 'random'")
    print("="*70 + "\n")
    
    for data_type in ["clean", "noisy"]:
        random_dir = base_path / data_type / "random"
        none_dir = base_path / data_type / "none"
        
        if not none_dir.exists():
            print(f"⚠️ {data_type}/none not found, skipping...")
            continue
        
        random_dir.mkdir(parents=True, exist_ok=True)
        
        none_images = list(none_dir.glob("*.png")) + list(none_dir.glob("*.jpg"))
        
        for img_path in none_images:
            target_path = random_dir / img_path.name
            shutil.move(str(img_path), str(target_path))
        
        shutil.rmtree(none_dir)
        
        print(f"✅ [{data_type:6s}] Merged {len(none_images):4d} images → random/")
    
    print("\n✅ Merge complete! 'none' folders removed.\n")


# ============ COMMAND LINE INTERFACE ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split dataset for pre-training and fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # แบ่งทั้ง pre-train และ fine-tune
  python auto_split.py --action split_all
  
  # Merge random+none ก่อน แล้วค่อย split
  python auto_split.py --action both
  
  # แบ่งเฉพาะ pre-train
  python auto_split.py --action split_pretrain
  
  # แบ่งเฉพาะ fine-tune
  python auto_split.py --action split_finetune
        """
    )
    
    parser.add_argument(
        "--action", 
        type=str, 
        choices=["merge", "split_all", "split_pretrain", "split_finetune", "both"],
        default="split_all",
        help="Action to perform"
    )
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.2, 
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--copy", 
        action="store_true", 
        help="Copy files instead of moving"
    )
    
    args = parser.parse_args()
    
    # Merge random+none
    if args.action in ["merge", "both"]:
        merge_random_and_none()
    
    # Split datasets
    if args.action in ["split_all", "both"]:
        split_pretrain_and_finetune(
            val_ratio=args.val_ratio,
            move_files=not args.copy
        )
    
    elif args.action == "split_pretrain":
        base_path = Path("./Data").resolve()
        synth_path = Path("./Data/Synthetic/png").resolve()
        clean_dir = synth_path / "clean"
        
        if clean_dir.exists():
            result = split_dataset_generic(
                source_dir=clean_dir,
                train_output_dir=base_path / "pre_train",
                val_output_dir=base_path / "pre_val",
                val_ratio=args.val_ratio,
                move_files=not args.copy,
                dataset_name="Pre-train"
            )
            print(f"\n✅ Pre-train split completed!")
        else:
            print(f"❌ Clean data folder not found: {clean_dir}")
    
    elif args.action == "split_finetune":
        base_path = Path("./Data").resolve()
        synth_path = Path("./Data/Synthetic/png").resolve()
        noisy_dir = synth_path / "noisy"
        
        if not noisy_dir.exists():
            noisy_dir = base_path / "train"
        
        if noisy_dir.exists():
            result = split_dataset_generic(
                source_dir=noisy_dir,
                train_output_dir=base_path / "train_split",  # ✅ แก้ไขตรงนี้!
                val_output_dir=base_path / "val_split",
                val_ratio=args.val_ratio,
                move_files=not args.copy,
                dataset_name="Fine-tune"
            )
            print(f"\n✅ Fine-tune split completed!")
        else:
            print(f"❌ Noisy data folder not found: {noisy_dir}")