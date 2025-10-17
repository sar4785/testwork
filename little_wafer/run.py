import argparse
import sys
from pathlib import Path
import torch
import yaml
import torchvision.transforms as transforms
from torchvision import datasets 

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
from src.Predict import ensemble_voting
from src.augment_data import augment_image
from src.auto_split import auto_split_dataset
from src.utils.simclr_trainer import train_simclr_from_config, SimCLRTrainer

# ================== Utility Function ==================
def load_config(config_path="configs/config.yaml"):
    """โหลด config ไฟล์"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_model(model_name, num_classes):
    models = {
        "resnet": (ResNetModel.create_model, "resnet18.pth"),
        "efficientnet": (EfficientNetModel.create_model, "efficientnet_b0.pth"),
        "vit": (ViTModel.create_model, "vit_b16.pth"),
        "convnext": (ConvNeXtModel.create_model, "convnext_tiny.pth")
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
    model_fn, save_name = models[model_name]
    return model_fn(num_classes=num_classes), save_name

    
# ================== Training Pipeline ==================
def train_pipeline(config, model_name, fine_tune=False,unsupervised=False):
    """Train and evaluate model"""
    trainer = Trainer(config, model_name=model_name)    
    train_path = Path(config['data']['train'])
    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_path}")
    
    # สร้าง transform พื้นฐานเพื่ออ่าน test dataset
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    train_loader, val_loader, class_counts = trainer.get_dataloaders()
    num_classes = trainer.num_classes   
    
    if not unsupervised:
        train_dataset_tmp = datasets.ImageFolder(train_path, transform=eval_transform)
        detected_classes = sorted(list(set(Path(p).parent.name for p, _ in train_dataset_tmp.samples)))
        config['data']['allowed_classes'] = detected_classes
        num_classes = len(detected_classes)        
        print(f"🔍 Classes detected: {detected_classes} ({num_classes} classes)")  
             
        if model_name == "resnet":
            model = ResNetModel.create_model(num_classes=num_classes)
            save_name = "resnet18.pth"
        elif model_name == "efficientnet":
            model = EfficientNetModel.create_model(num_classes=num_classes)
            save_name = "efficientnet_b0.pth"
        elif model_name == "vit":
            model = ViTModel.create_model( num_classes=num_classes)
            save_name = "vit_b16.pth"
        elif model_name == "convnext":
            model = ConvNeXtModel.create_model(num_classes=num_classes)
            save_name = "convnext_tiny.pth"
        else:
            raise ValueError(f"❌ Unsupported model: {model_name}")
    
        print(f"🚀 Running supervised training for {model_name.upper()} ...")
        trainer.train_supervised(model, save_name=save_name, fine_tune=fine_tune)
    
        # Evaluate after training
        model_path = trainer.checkpoint_dir / save_name
        
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=trainer.device))
            accuracy, auc_score = trainer.evaluate(model, val_loader,criterion=None)
            print(f"🔹 Accuracy: {accuracy:.4f}")
            if auc_score is not None:
                print(f"🔹 ROC-AUC: {auc_score:.4f}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
                      
# ================== Main ==================     
def main():
    parser = argparse.ArgumentParser(description="Wafer Classification Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        choices=['augment', 'train', 'predict', 'ensemble','kaggle','unsupervised','simclr','auto_split','pretrain','plot'],
                        help='Choose pipeline step: augment / train / predict / ensemble,simclr')
    parser.add_argument('--model', type=str, nargs='*', 
                        default=['resnet', 'efficientnet', 'convnext'],
                        choices=['resnet', 'efficientnet', 'vit', 'convnext'],
                        help='Model(s) to train or evaluate')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune all layers')
    parser.add_argument('--pretrain', action='store_true', help='Pretrain SimCLR before splitting dataset')
    
    args = parser.parse_args()

    # ตั้ง GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # โหลด config
    config = load_config()

    if args.step == 'kaggle':
        print("📦 Converting Kaggle wafer map dataset...")
        from src.kaggledata import KaggleDataProcessor
        KaggleDataProcessor.run_merge()

    # ---------------- Augment Data ---------------- #
    if args.step == 'augment':
        print("🖼️ Running data augmentation for real wafer maps...")
        augment_image(config)
        return

    elif args.step == 'simclr':
        print("🧩 Running SimCLR Unsupervised Pretraining ...")
        train_simclr_from_config("configs/config.yaml")
        return

    elif args.step == 'plot':
        from src.plot import PRRToPNGConverter
        print("📊 Converting PRR.csv to WaferMap PNG ...")
        PRRToPNGConverter.main()
        return

    elif args.step == 'train':
        print(f"🧠 Running Supervised Training for models: {args.model}")

        for model_name in args.model:
            train_pipeline(config, model_name, fine_tune=args.fine_tune, unsupervised=False)
        return
        
    elif args.step == 'unsupervised':
        print(f"🧩 Running Unsupervised Training for models: {args.model}")
        from src.Predict import compare_models
        compare_models(config_path="configs/config.yaml", model_list=args.model, unsupervised=True)
        return  
        
    elif args.step == 'predict':
        print("📊 Running model evaluation and comparison...")
        from src.Predict import compare_models
        compare_models(config_path="configs/config.yaml", model_list=args.model)
        return    

    elif args.step == 'auto_split':
        print("🔄 Splitting dataset into val/test ...")
        from src.auto_split import auto_split_dataset
        train_dir, val_dir, test_dir = auto_split_dataset("configs/config.yaml", pretrain_simclr=args.pretrain)
        print(f"✅ New split complete:\n  Train: {train_dir}\n  Val: {val_dir}\n  Test: {test_dir}")
        return

    # STEP: Ensemble (voting among models)
    elif args.step == 'ensemble':
        print("🗳️ Running ensemble voting among selected models...")
        ensemble_voting(config_path="configs/config.yaml", model_list=args.model)

    else:
        print(f"⚠️ Unknown step: {args.step}")
    
if __name__ == "__main__":
    main()