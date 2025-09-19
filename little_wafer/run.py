# run.py
import argparse

def main():
    parser = argparse.ArgumentParser(description="Wafer Classification Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        choices=[
                            'extract', 'convert', 'plot', 'resize',
                            'train', 'all', 'prepare_kaggle', '_npz',
                            'semi_supervised','auto_label',],
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

    elif args.step == 'prepare_kaggle':
        from src.kaggledata import KaggleDataProcessor
        print("🧩 Running: Prepare Kaggle dataset...")
        KaggleDataProcessor.run()

    elif args.step == '_npz':
        from src.kaggledata import KaggleDataProcessor
        print("🧩 Running: Prepare Kaggle dataset from NPZ...")
        KaggleDataProcessor.runnpz()

    elif args.step == 'train':
        from src.train import TrainModel, evaluate_model
        print("🧠 Running: Train model...")
        trainer = TrainModel()
        model, le, history, test_loader = trainer.run()
        evaluate_model(
            model=model,
            dataloader=test_loader,
            label_encoder=le,
            device=trainer.device,
            output_config=trainer.config['output']
        )

    elif args.step == 'semi_supervised':
        print("🧠 Running: Semi-supervised training...")
        from src.semi_supervised import run_from_config
        run_from_config("configs/config.yaml", epochs=5, threshold=0.9)
    
    elif args.step == 'auto_label':
        from src.auto_label import auto_label
        print("🤖 Running: Auto-labeling unlabeled images...")
        auto_label(config_path="configs/config.yaml", threshold=0.9)


    elif args.step == 'all':
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
