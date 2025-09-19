import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

# Load config
CONFIG_PATH = 'configs/config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class PRRToPNGConverter:
    @staticmethod
    def main(input_dir=None, output_dir=None, target_size=(224, 224)):
        """แปลง PRR.csv → WaferMap PNG (Grayscale + Resize)"""
        if input_dir is None:
            input_dir = config['data']['prr']
        if output_dir is None:
            output_dir = config['data']['wafer_map_png']
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)    

        # หาไฟล์ CSV ทั้งหมด
        csv_files = list(Path(input_dir).rglob("*.csv"))
        if not csv_files:
            print("Not found.csv")
            return

        for csv_file in csv_files:
            try:
                # โหลดข้อมูล CSV
                df = pd.read_csv(csv_file)
                row_count = len(df)
                print(f"📊 {csv_file.name}: {row_count} rows", "More than 5k" if row_count > 5000 else "Less than 5k")

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_axis_off()
        
                if df["X_COORD"].isna().all() or df["Y_COORD"].isna().all():
                    print(f"⚠️ ไม่มีข้อมูล X,Y ใน: {csv_file.name}")
                    continue

                # หาค่า min/max
                max_x, max_y = df["X_COORD"].max(), df["Y_COORD"].max()
                min_x, min_y = df["X_COORD"].min(), df["Y_COORD"].min()

                width = max_x - min_x + 1
                height = max_y - min_y + 1

                # mask เริ่มต้นเป็น 0 (ดำ)
                mask = np.zeros((height, width), dtype=np.uint8)
                # Mapping ค่า bin → grayscale
                for _, row in df.iterrows():
                    x, y, hb = int(row["X_COORD"] - min_x), int(row["Y_COORD"] - min_y), row["HARD_BIN"]
                    if hb == 1:   # pass
                        mask[y, x] = 1   # เทา
                    else:         # fail
                        mask[y, x] = 2   # ขาว
                
                # แปลงเป็น grayscale (0=ดำ, 127=เทา, 254=ขาว)
                mask_img = Image.fromarray((mask * 127).astype(np.uint8), mode='L') 
                # Resize → 224x224
                mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
                
                
                """ # Plot pass dies (Hard_bin == 1)
                pass_dies = df[df["HARD_BIN"] == 1]
                ax.scatter(pass_dies["X_COORD"], pass_dies["Y_COORD"], c="green", marker="s", s=10)

                # Plot fail dies (Hard_bin != 1)
                fail_dies = df[df["HARD_BIN"] != 1]
                ax.scatter(fail_dies["X_COORD"], fail_dies["Y_COORD"], c="red", marker="s", s=10) """
                
                plt.close(fig)
                        
                # Save ไฟล์
                out_path = output_dir / f"{csv_file.stem}.png"
                mask_img.save(out_path)
                print(f"✅ Saved wafer map: {out_path}")

            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")