# run.py
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Wafer Classification Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        choices=['extract', 'convert', 'plot', 'resize', 'train', 'all','prepare_kaggle','_npz'],
                        help='Pipeline step to run')
    parser.add_argument('--session', type=str, default=None,
                        help='Session filter for .std files (e.g., "S11P")')
    args = parser.parse_args()

    if args.step == 'extract':
        from src.extract import ExtractZip
        print("📦 Running: Extract .std from ZIP...")
        ExtractZip.run(session_filter=args.session)

    elif args.step == 'convert':
        from src.convert import PRRConverter
        print("📊 Running: Convert .std to PRR.csv...")
        PRRConverter.convert_stdf_to_prr()

    elif args.step == 'plot':
        from src.plot import PRRToPNGConverter
        print("📈 Running: Plot PRR.csv to PNG...")
        PRRToPNGConverter.main()

    elif args.step == 'resize':
        from src.resize import resize_img #,convert_colors_to_grayscale
        print("🖼️ Running: Resize PNG images...")
        resize_img()
        #convert_colors_to_grayscale()
        
    elif args.step == 'prepare_kaggle':
        from src.kaggledata import KaggleDataProcessor
        print("🧩 Running: Prepare Kaggle dataset...")
        KaggleDataProcessor.run()
    elif args.step == '_npz':    
        from src.kaggledata import KaggleDataProcessor
        print("🧩 Running: Prepare Kaggle dataset from NPZ...")
        KaggleDataProcessor.runnpz()

    elif args.step == 'train':
        from src.train import TrainModel
        print("🧠 Running: Train model...")
        trainer = TrainModel()
        model, le, history, test_loader = trainer.run()
        
        # Evaluate the model on the test set
        from src.train import evaluate_model  # สมมติว่า evaluate_model อยู่ในไฟล์ train.py
        evaluate_model(
            model=model,
            dataloader=test_loader,
            label_encoder=le,
            device=trainer.device,
            output_config=trainer.config['output']
        )

    elif args.step == 'all':
        # รันทั้งหมดตามลำดับ
        from src.extract import ExtractZip
        from src.convert import PRRConverter
        from src.plot import PRRToPNGConverter
        from src.resize import resize_img
        from src.train import TrainModel

        print("🚀 Running ALL steps...")
        ExtractZip.run(session_filter=args.session)
        PRRConverter.convert_stdf_to_prr()
        PRRToPNGConverter.main()
        resize_img()
        trainer = TrainModel()
        trainer.run()

if __name__ == "__main__":
    main()