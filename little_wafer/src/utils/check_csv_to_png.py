import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def main():
    # 🔹 Hard-coded path (แก้ไขตรงนี้ตามไฟล์ที่คุณต้องการทดสอบ)
    csv_path = "C:/Users/User/Documents/GitHub/testwork/little_wafer/Data/Output-PRR/ZA530361/ZA530361_001_S11P_20250814022454_M5116B16P03_WFTB33-01.csv"
    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"❌ ไม่พบไฟล์: {csv_file}")
        return

    print(f"📂 กำลังประมวลผล: {csv_file.name}")

    # โหลดข้อมูล
    df = pd.read_csv(csv_file)

    # สร้าง mask
    min_x, max_x = df["x_coord"].min(), df["x_coord"].max()
    min_y, max_y = df["y_coord"].min(), df["y_coord"].max()

    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)

    mask = np.zeros((height, width), dtype=np.uint8)

    for _, row in df.iterrows():
        x = int(row["x_coord"] - min_x)
        y = int(row["y_coord"] - min_y)
        hb = row["Hard_bin"]
        if hb == 1:
            mask[y, x] = 127   # เทา (pass)
        else:
            mask[y, x] = 255   # ขาว (fail)

    # แปลงเป็นภาพ
    mask_img = Image.fromarray(mask, mode='L')
    mask_img_resized = mask_img.resize((224, 224), Image.Resampling.NEAREST)

    # 🔹 บันทึกในโฟลเดอร์เดียวกับ .csv
    png_path = csv_file.with_suffix('.png')
    mask_img_resized.save(png_path)
    print(f"✅ บันทึกภาพแล้วที่: {png_path}")

    # 🔹 แสดงภาพทันที (เพื่อตรวจสอบ)
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_img_resized, cmap='gray')
    plt.title(f"Wafer Map: {csv_file.stem}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()