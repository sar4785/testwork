# src/kaggledata.py
import os
import pandas as pd
import numpy as np
import cv2
import yaml
from pathlib import Path

class KaggleDataProcessor:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def sanitize_label(self, label):
        """ลบอักขระพิเศษ [ ] ' และแทนที่ space ด้วย underscore"""
        return label.replace("[", "").replace("]", "").replace("'", "").replace(" ", "_").strip("_")

    def convert_failure_type(self, x):
        if isinstance(x, (list, np.ndarray)):
            return str(x[0]) if len(x) > 0 else "Unknown"
        elif pd.isna(x):
            return "Unknown"
        else:
            return str(x)

    def preprocess_and_save_wafer_map(self, wafer_map, output_path):
        """แปลง wafer_map เป็นภาพ PNG ขนาด 224x224 โดย:
        0 = ขาว, 1 = เขียว, 2 = แดง"""
        wafer_map = np.array(wafer_map)
        if wafer_map.size == 0:
            return False

        # สร้างภาพ RGB
        img = np.zeros((*wafer_map.shape, 3), dtype=np.uint8)
        img[wafer_map == 0] = 0    # ดำ
        img[wafer_map == 1] = 127  # เทา
        img[wafer_map == 2] = 255  # ขาว

        # Resize
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

        # บันทึกเป็น PNG
        success = cv2.imwrite(output_path, img)
        return success

    def export_kaggle_dataset(self):
        """โหลด LSWMD.pkl และ export ทุกภาพเป็น PNG ตาม label"""
        # Paths
        input_file = Path(self.config['data']['raw']) / 'LSWMD.pkl'
        output_root = Path(self.config['data']['kaggle_png'])
        output_root.mkdir(parents=True, exist_ok=True)
        # Load data
        print(f"📂 Loading dataset from: {input_file}")
        df = pd.read_pickle(input_file)
        print(f"📊 Total wafers: {len(df)}")
        df['failureType'] = df['failureType'].apply(self.convert_failure_type)

        # ✅ Drop labels ที่ไม่ต้องการ
        invalid_labels = {"none", "unknown", "", "nan"}
        filtered_df = df.dropna(subset=['failureType'])
        filtered_df = filtered_df[~filtered_df['failureType'].str.lower().isin(invalid_labels)]
        
        print(f"📊 Remaining wafers after filtering: {len(filtered_df)}")
        print(f"🏷️ Unique labels: {filtered_df['failureType'].unique()}")
        
        # Export images
        export_count = {}
        for idx, row in filtered_df.iterrows():
            label = row['failureType']
            safe_label = self.sanitize_label(label)
            label_folder = output_root / safe_label
            label_folder.mkdir(exist_ok=True)

            # ตั้งชื่อไฟล์
            count = export_count.get(label, 0)
            filename = f"{safe_label}_{count:05d}.png"
            output_path = label_folder / filename
            # แปลงและบันทึก
            if self.preprocess_and_save_wafer_map(row['waferMap'], str(output_path)):
                export_count[label] = count + 1
                if count % 1000 == 0:
                    print(f"✅ Exported {count} images for label '{label}'")
        # สรุป
        print("\n✅ Kaggle Dataset Export Complete!")
        for label, count in export_count.items():
            safe_label = self.sanitize_label(label)
            print(f"{label} → {safe_label}: {count} images")

        return export_count     
     
    def load_npz_dataset(self, npz_path=None):
        """
        อ่านไฟล์ .npz (เช่น Wafer_Map_Datasets.npz) และ export เป็น PNG
        """
        if npz_path is None:
            npz_path = Path(self.config['data']['npz_dataset']) 
        print(f"🔍 Loading .npz file: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        lst = data.files  # keys ทั้งหมด
        
        for item in lst:
            print(f"\n📁 Array Name: {item}")
            arr = data[item]
            print(f"Shape: {arr.shape}, Dtype: {arr.dtype}")
            if arr.size > 0:
                print(f"Sample data: {arr.flatten()[:5]}")
                
        # รองรับ keys มาตรฐาน
        output_root = Path(self.config['data']['kaggle_png'])

        if 'X' in lst and 'y' in lst:
            wafers = data['X']
            labels = data['y'].argmax(axis=1) if data['y'].ndim == 2 else data['y']
        elif 'wafers' in lst and 'labels' in lst:
            wafers = data['wafers']
            labels = data['labels'].argmax(axis=1) if data['labels'].ndim == 2 else data['labels']
        elif 'arr_0' in lst and 'arr_1' in lst:
            wafers = data['arr_0']  # shape: (38015, 52, 52)
            # แปลง one-hot เป็น class index
            labels = data['arr_1'].argmax(axis=1)  # shape: (38015,) จาก (38015, 8)
        else:
            print("⚠️ Unknown key format in .npz, please check manually.")
            return

        # ส่งไปบันทึก
        self._save_npz_images(wafers, labels, output_root)   
    
    def _save_npz_images(self, wafers, labels, output_root: Path):
        # 🔁 กำหนด mapping จาก class index → failure type name
        class_names = [
            "Center1",
            "Donut1",
            "Edge-Loc1",
            "Edge-Ring1",
            "Loc1",
            "Random1",
            "Scratch1",
            "Near-full1"
            ]
    
        output_root.mkdir(parents=True, exist_ok=True)
        export_count = {}

        for i, (wafer_map, label_idx) in enumerate(zip(wafers, labels)):
            label_idx = int(label_idx)

            # ✅ ใช้ class_names แทนการตั้งชื่อแบบ Class_0
            if 0 <= label_idx < len(class_names):
                label_name = class_names[label_idx]  # เช่น 0 → "Center"
            else:
                label_name = "Unknown"  # กรณี index ผิดพลาด

            safe_label = self.sanitize_label(label_name)
            label_folder = output_root / safe_label
            label_folder.mkdir(exist_ok=True)

            count = export_count.get(label_name, 0)
            filename = f"{safe_label}_{count:05d}.png"
            output_path = label_folder / filename

            if self.preprocess_and_save_wafer_map(wafer_map, str(output_path)):
                export_count[label_name] = count + 1
                if count % 100 == 0:
                    print(f"✅ Saved {count} images for '{label_name}'")

        print("\n✅ Images extracted from .npz!")
        for label, count in export_count.items():
            print(f"{label}: {count} images")
       
    @staticmethod
    def run():
        processor = KaggleDataProcessor()
        processor.export_kaggle_dataset()
    
    @staticmethod    
    def runnpz():
        # Export from NPZ
        print("\n🚀 Extracting from Wafer_Map_Datasets.npz...")
        processor = KaggleDataProcessor()
        processor.load_npz_dataset()