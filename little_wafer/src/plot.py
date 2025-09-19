import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

# Load config
CONFIG_PATH = 'configs/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class PRRToPNGConverter:
    def main(input_dir=None, output_dir=None):
        # โหลด config
        CONFIG_PATH = 'configs/config.yaml'
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if input_dir is None:
            input_dir = config['data']['prr']
        if output_dir is None:
            output_dir = config['data']['wafer_map_png']

        # หาไฟล์ CSV ทั้งหมด
        csv_files = list(Path(input_dir).rglob("*.csv"))
        if not csv_files:
            print("❌ ไม่พบไฟล์ .csv")
            return

        for csv_file in csv_files:
            try:
                # โหลดข้อมูล CSV
                df = pd.read_csv(csv_file)
                row_count = len(df)
                print(f"📊 {csv_file.name}: {row_count} rows", "More than 5k" if row_count > 5000 else "Less than 5k")

                if df["X_COORD"].isna().all() or df["Y_COORD"].isna().all():
                    print(f"⚠️ ไม่มีข้อมูล X,Y ใน: {csv_file.name}")
                    continue

                # หาค่า min/max
                max_x, max_y = df["X_COORD"].max(), df["Y_COORD"].max()
                min_x, min_y = df["X_COORD"].min(), df["Y_COORD"].min()

                # สร้าง plot
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_axis_off()
                ax.set_xlim(min_x - 5, max_x + 5)
                ax.set_ylim(min_y - 5, max_y + 5)
                ax.set_aspect('auto', adjustable='box')

                # Plot pass dies (Hard_bin == 1)
                pass_dies = df[df["HARD_BIN"] == 1]
                ax.scatter(pass_dies["X_COORD"], pass_dies["Y_COORD"], c="green", marker="s", s=10)

                # Plot fail dies (Hard_bin != 1)
                fail_dies = df[df["HARD_BIN"] != 1]
                ax.scatter(fail_dies["X_COORD"], fail_dies["Y_COORD"], c="red", marker="s", s=10)

                # บันทึกไฟล์เป็น PNG
                out_path = Path(output_dir) / f"{csv_file.stem}.png"
                fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                print(f"✅ Saved wafer map: {out_path}")

            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")