# ========================= run.py (Fixed file renaming) =========================
import argparse
import sys
import time
from pathlib import Path
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# เพิ่ม root โปรเจกต์ลงใน PYTHONPATH
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import modules
from src.Model.ResNet import ResNetModel
from src.Model.EfficientNet import EfficientNetModel
from src.Model.Vit import ViTModel
from src.Model.ConvNeXt import ConvNeXtModel
from src.Model.Trainer import Trainer
from src.Model.CnnPytorch import train_cnn_pytorch
from src.auto_split import auto_split_dataset


# ================== Utility Function ==================
def load_config(config_path="configs/config.yaml"):
    """โหลด config ไฟล์"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # แปลง path ให้เป็น absolute
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for k, v in config["data"].items():
        if isinstance(v, str) and v.startswith("./"):
            config["data"][k] = os.path.join(base_dir, v[2:])
    return config


def plot_prediction_bar(csv_path, save_path):
    """Plot bar chart for predicted labels"""
    df = pd.read_csv(csv_path)
    if "pred_label" not in df.columns:
        print(f"⚠️ Warning: No column 'pred_label' found in {csv_path}")
        return

    counts = df["pred_label"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette="tab10")
    plt.title("Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Bar plot saved to {save_path}")


def train_and_measure(model_name, config, val_label=True, pretrained_path=None):
    """Train one model and measure time"""
    start_time = time.time()
    trainer = Trainer(config, model_name=model_name)

    if model_name.lower() == "cnn":
        print("\n🔧 Training Simple CNN ...")
        train_cnn_pytorch(config_path="configs/config.yaml", val_label=val_label)
        elapsed_min = (time.time() - start_time) / 60
    else:
        print(f"\n🚀 Training {model_name.upper()} ...")
        train_loader, val_loader, _ = trainer.get_dataloaders()

        # สร้างโมเดล
        model_map = {
            "resnet": (ResNetModel, "resnet18.pth"),
            "efficientnet": (EfficientNetModel, "efficientnet_b0.pth"),
            "convnext": (ConvNeXtModel, "convnext_tiny.pth"),
            "vit": (ViTModel, "vit_b_16.pth")
        }

        if model_name not in model_map:
            raise ValueError(f"❌ Unsupported model: {model_name}")

        model_class, save_name = model_map[model_name]
        model = model_class.create_model(num_classes=trainer.num_classes)

        # โหลด pretrained weights (ถ้ามี)
        if pretrained_path and pretrained_path.exists():
            print(f"🔁 Loading pretrained weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=trainer.device), strict=False)

        # Train and get elapsed time
        elapsed_min = trainer.train_supervised(model, save_name=save_name)

    print(f"✅ {model_name.upper()} finished in {elapsed_min:.2f} minutes")
    return elapsed_min


def clear_gpu_memory():
    """ล้าง GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU memory cleared")


def safe_rename(old_path, new_path):
    """
    ✅ Rename file safely (ลบไฟล์เป้าหมายก่อนถ้ามีอยู่แล้ว)
    """
    old_path = Path(old_path)
    new_path = Path(new_path)
    
    if not old_path.exists():
        print(f"⚠️ Source file not found: {old_path}")
        return False
    
    # ลบไฟล์เป้าหมายถ้ามีอยู่แล้ว
    if new_path.exists():
        print(f"🗑️ Removing existing file: {new_path}")
        new_path.unlink()
    
    # Rename
    old_path.rename(new_path)
    print(f"✅ Renamed {old_path.name} → {new_path.name}")
    return True


def run_full_pipeline_per_model(model_name, config, val_label, output_dir):
    """
    รัน pipeline ทั้งหมด (pre-train → fine-tune → predict) สำหรับ 1 model
    """
    print(f"\n{'='*70}")
    print(f"🚀 Starting FULL PIPELINE for {model_name.upper()}")
    print(f"{'='*70}\n")
    
    results = {
        "model": model_name.upper(),
        "pretrain_time": 0,
        "finetune_time": 0,
        "total_time": 0
    }
    
    pipeline_start = time.time()
    
    # ============ STEP 1: PRE-TRAIN ============
    print(f"\n📚 STEP 1/3: Pre-Training {model_name.upper()} on CLEAN data...")
    pre_config = config.copy()
    pre_config["data"]["train"] = "./Data/pre_train"
    pre_config["data"]["val"] = "./Data/pre_val"
    
    try:
        pretrain_time = train_and_measure(model_name, pre_config, val_label=val_label)
        results["pretrain_time"] = pretrain_time
        
        # ✅ ไม่ต้อง rename เพราะ Trainer.py บันทึกชื่อถูกต้องแล้ว
        print(f"✅ Pre-train checkpoint saved as {model_name}_pretrain.pth")
        
        clear_gpu_memory()
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Pre-training failed for {model_name}: {e}")
        return results
    
    # ============ STEP 2: FINE-TUNE ============
    print(f"\n🔧 STEP 2/3: Fine-Tuning {model_name.upper()} on NOISY data...")
    ft_config = config.copy()
    ft_config["data"]["train"] = "./Data/train_split"
    ft_config["data"]["val"] = "./Data/val_split"
    
    pretrained_path = output_dir / f"{model_name}_pretrain.pth"
    
    try:
        finetune_time = train_and_measure(
            model_name, ft_config,
            val_label=val_label,
            pretrained_path=pretrained_path if pretrained_path.exists() else None
        )
        results["finetune_time"] = finetune_time
        
        # ✅ เพิ่ม: Rename ไฟล์หลัง fine-tune
        model_map = {
            "resnet": "resnet_pretrain.pth",
            "efficientnet": "efficientnet_pretrain.pth",
            "convnext": "convnext_pretrain.pth",
            "vit": "vit_pretrain.pth",
            "cnn": "cnn_pretrain.pth"
        }
        old_name = output_dir / model_map.get(model_name, f"{model_name}.pth")
        new_name = output_dir / f"{model_name}_finetune.pth"
        safe_rename(old_name, new_name)
        
        clear_gpu_memory()
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Fine-tuning failed for {model_name}: {e}")
        return results
    
    # ============ STEP 3: PREDICT ============
    print(f"\n🔮 STEP 3/3: Running prediction for {model_name.upper()}...")
    try:
        from src.Predict import compare_models
        compare_models(config_path="configs/config.yaml", model_list=[model_name])
        
        # Plot bar chart
        pred_dir = Path("Output/predictions")
        csv_file = pred_dir / f"test_predictions_{model_name}.csv"
        if csv_file.exists():
            plot_path = csv_file.with_name(csv_file.stem + "_bar.png")
            plot_prediction_bar(csv_file, plot_path)
        
        clear_gpu_memory()
        
    except Exception as e:
        print(f"❌ Prediction failed for {model_name}: {e}")
    
    # คำนวณเวลารวม
    total_time = (time.time() - pipeline_start) / 60
    results["total_time"] = total_time
    
    print(f"\n{'='*70}")
    print(f"✅ {model_name.upper()} PIPELINE COMPLETED")
    print(f"   ⏱️  Pre-train: {results['pretrain_time']:.2f} min")
    print(f"   ⏱️  Fine-tune: {results['finetune_time']:.2f} min")
    print(f"   ⏱️  Total: {results['total_time']:.2f} min")
    print(f"{'='*70}\n")
    
    return results


# ================== Main ==================
def main():
    parser = argparse.ArgumentParser(description="Wafer Classification Trainer (Modified)")
    parser.add_argument('--step', type=str, required=True,
                        choices=['augment', 'train', 'predict',
                                 'auto_split', 'pre_train', 'evaluate', 'all'],
                        help='Choose pipeline step')
    parser.add_argument('--model', type=str, nargs='*',
                        default=["cnn", "resnet", "efficientnet", "convnext", "vit"],
                        choices=['resnet', 'efficientnet', 'convnext', 'cnn', 'vit'],
                        help='Model(s) to train or evaluate')
    parser.add_argument('--val_label', action='store_true',
                        help='Use labeled validation set when training CNN')
    args = parser.parse_args()

    # ตั้ง GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # โหลด config
    config = load_config()
    output_dir = Path("Output/checkpoints")

    # ============ NEW: ALL PIPELINE ============
    if args.step == "all":
        print("\n" + "="*70)
        print("🎯 Running COMPLETE PIPELINE for all models")
        print("   Models:", [m.upper() for m in args.model])
        print("="*70 + "\n")
        
        all_results = []
        overall_start = time.time()
        
        # รันทีละโมเดลเพื่อประหยัด memory
        for i, model_name in enumerate(args.model, 1):
            print(f"\n🔄 Processing model {i}/{len(args.model)}: {model_name.upper()}")
            
            result = run_full_pipeline_per_model(
                model_name, config, args.val_label, output_dir
            )
            all_results.append(result)
            
            # พักระหว่างโมเดล
            if i < len(args.model):
                print(f"\n⏸️  Waiting 5 seconds before next model...\n")
                time.sleep(5)
        
        # สรุปผลรวม
        overall_time = (time.time() - overall_start) / 60
        
        print("\n" + "="*70)
        print("📊 FINAL SUMMARY - ALL MODELS")
        print("="*70)
        
        df_summary = pd.DataFrame(all_results)
        print(df_summary.to_string(index=False))
        
        print(f"\n⏱️  Total Pipeline Time: {overall_time:.2f} minutes")
        print("="*70 + "\n")
        
        # บันทึกผลลัพธ์
        summary_path = Path("Output/all_pipeline_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(summary_path, index=False)
        print(f"💾 Complete summary saved to {summary_path}")
        
        return

    # ---------------- Split & Merge ---------------- #
    elif args.step == "auto_split":
        print("📂 Merging random+none and splitting dataset...")
        from src.auto_split import merge_random_and_none, auto_split_dataset
        
        # 🔹 ขั้นตอนที่ 1: รวม random + none เข้าด้วยกัน
        print("\n🔄 Step 1: Merging 'none' into 'random' folder...")
        merge_random_and_none()
        
        # 🔹 ขั้นตอนที่ 2: แบ่ง train/val จาก clean data
        print("\n📊 Step 2: Splitting clean data into train/val...")
        auto_split_dataset(val_ratio=0.2, move_files=True, use_clean_only=True)

    # ---------------- Pre-Train (Clean Data) ---------------- #
    elif args.step == "pre_train":
        print(f"🎯 Running Pre-Training (clean data) for models: {args.model}")
        total_times = {}
        pre_config = config.copy()
        
        # 🔹 ใช้ชุดข้อมูล clean ที่แบ่งแล้ว
        pre_config["data"]["train"] = "./Data/pre_train"
        pre_config["data"]["val"] = "./Data/pre_val"

        for model_name in args.model:
            elapsed_min = train_and_measure(model_name, pre_config, val_label=args.val_label)
            total_times[model_name] = elapsed_min

        # บันทึกสรุปผล
        df_summary = pd.DataFrame([
            {"Model": m.upper(), "PreTrain_Time_Minutes": t}
            for m, t in total_times.items()
        ])
        out_csv = Path("Output/pretrain_summary.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(out_csv, index=False)
        print(f"📝 Pre-training summary saved to {out_csv}")

    # ---------------- Fine-Tune (Noisy Data) ---------------- #
    elif args.step == "train":
        print(f"🔧 Running Fine-Tuning on NOISY data for models: {args.model}")
        total_times = {}

        for model_name in args.model:
            # โหลด pretrained weights
            pretrained_path = output_dir / f"{model_name}_pretrain.pth"
            if not pretrained_path.exists():
                print(f"⚠️ Warning: No pretrained weights for {model_name}, training from scratch.")
                pretrained_path = None

            # 🔹 ใช้ข้อมูล noisy ที่แบ่งแล้ว
            ft_config = config.copy()
            ft_config["data"]["train"] = "./Data/train_split"
            ft_config["data"]["val"] = "./Data/val_split"
            
            if pretrained_path:
                ft_config["pretrained_weights"] = str(pretrained_path)

            elapsed_min = train_and_measure(
                model_name, ft_config,
                val_label=args.val_label,
                pretrained_path=pretrained_path
            )
            total_times[model_name] = elapsed_min

        # Save training summary
        df_summary = pd.DataFrame([
            {"Model": m.upper(), "FineTune_Time_Minutes": t}
            for m, t in total_times.items()
        ])
        out_csv = Path("Output/finetune_summary.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(out_csv, index=False)
        print(f"📝 Fine-tuning summary saved to {out_csv}")

        # Plot test prediction bars
        pred_dir = Path("Output/predictions")
        for csv_file in pred_dir.glob("test_predictions_*.csv"):
            plot_path = csv_file.with_name(csv_file.stem + "_bar.png")
            plot_prediction_bar(csv_file, plot_path)

    # ---------------- Evaluate ---------------- #
    elif args.step == "evaluate":
        print("📊 Evaluating on TEST set...")
        from src.Predict import compare_models
        compare_models(config_path="configs/config.yaml", model_list=args.model)
        print("✅ Evaluation complete.")

    # ---------------- Predict ---------------- #
    elif args.step == "predict":
        print("🔮 Running prediction and comparison...")
        from src.Predict import compare_models
        compare_models(config_path="configs/config.yaml", model_list=args.model)

    # ---------------- Augment Real Data ---------------- #
    elif args.step == "augment":
        print("🎨 Running data augmentation for real wafer maps...")
        from src.augment_data import process_real_wafer_data_single_folder
        process_real_wafer_data_single_folder()

    else:
        print(f"❌ Unknown step: {args.step}")


if __name__ == "__main__":
    main()