import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_DIR = r"C:\Users\User\Desktop\Wafer-classification-DCNN\convert_stdf_to_csv\Output-PRR"
OUTPUT_DIR = r"C:\Users\User\Desktop\Wafer-classification-DCNN\Wafer-Map"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class PRRToPNGConverter:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    @staticmethod
    def plot_pass_dies(ax, df):
        """
        1.1 Plot pass dies (Hard_bin == 1) เป็นสีเขียว
        """
        pass_dies = df[df["HARD_BIN"] == 1]
        ax.scatter(pass_dies["X_COORD"], pass_dies["Y_COORD"],
                   c="green", marker="s", s=10)

    @staticmethod
    def plot_fail_dies(ax, df):
        """
        1.2 Plot fail dies (Hard_bin != 1) เป็นสีแดง
        """
        fail_dies = df[df["HARD_BIN"] != 1]
        ax.scatter(fail_dies["X_COORD"], fail_dies["Y_COORD"],
                   c="red", marker="s", s=10)

    @classmethod
    def plot_wafer_map(cls, csv_file, output_dir=None):
        """
        Plot wafer map จาก PRR.csv ของแต่ละ lot
        """
        if output_dir is None:
            output_dir = cls.output_dir

        df = pd.read_csv(csv_file)

        # --- print count rows ---
        row_count = len(df)
        print(f"📊 {csv_file.name}: {row_count} rows", end=" ")
        print("More than 5k" if row_count > 5000 else "Less than 5k")

        # --- ตรวจสอบว่ามีข้อมูล X,Y หรือไม่ ---
        if df["X_COORD"].isna().all() or df["Y_COORD"].isna().all():
            print(f"⚠️ ไม่มีข้อมูล X,Y ใน: {csv_file.name}")
            return

        # --- กำหนดขนาดภาพจาก wafer size ---
        max_x, max_y = df["X_COORD"].max(), df["Y_COORD"].max()
        min_x, min_y = df["X_COORD"].min(), df["Y_COORD"].min()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_axis_off()

        # scaling: เผื่อ 10%
        ax.set_xlim(min_x - 5, max_x + 5)
        ax.set_ylim(min_y - 5, max_y + 5)
        ax.set_aspect('auto', adjustable='box')

        # --- plot step 1: pass dies ---
        cls.plot_pass_dies(ax, df)

        # --- plot step 2: fail dies ---
        cls.plot_fail_dies(ax, df)

        # --- save plot ---
        out_path = Path(output_dir) / f"{csv_file.stem}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"✅ Saved wafer map: {out_path}")

    @classmethod
    def main(cls, input_dir=None, output_dir=None):
        if input_dir is None:
            input_dir = cls.input_dir
        if output_dir is None:
            output_dir = cls.output_dir

        csv_files = list(Path(input_dir).rglob("*.csv"))

        if not csv_files:
            print("❌ ไม่พบไฟล์ .csv")
            return

        for csv_file in csv_files:
            try:
                cls.plot_wafer_map(csv_file, output_dir)
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")


if __name__ == "__main__":
    PRRToPNGConverter.main()
