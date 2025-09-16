# run.py
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Wafer Classification Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        choices=['extract', 'convert', 'plot', 'resize', 'train', 'all'],
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
        from src.resize import resize_img
        print("🖼️ Running: Resize PNG images...")
        resize_img()

    elif args.step == 'train':
        from src.train import main as train_main
        print("🧠 Running: Train model...")
        train_main()

    elif args.step == 'all':
        # รันทั้งหมดตามลำดับ
        from src.extract import ExtractZip
        from src.convert import PRRConverter
        from src.plot import PRRToPNGConverter
        from src.resize import resize_img
        from src.train import main as train_main

        print("🚀 Running ALL steps...")
        ExtractZip.run(session_filter=args.session)
        PRRConverter.convert_stdf_to_prr()
        PRRToPNGConverter.main()
        resize_img()
        train_main()

if __name__ == "__main__":
    main()