# src/kaggledata.py
import os
import pandas as pd
import numpy as np
import cv2
import yaml
from pathlib import Path


class KaggleDataProcessor:
    CLASS_NAMES = {
        0: "Center", 1: "Donut", 2: "Edge-Loc", 3: "Edge-Ring",
        4: "Loc", 5: "Random", 6: "Scratch", 7: "Near-full"
    }
    LABEL_MAP = {**{k: k for k in CLASS_NAMES.values()},
                 **{f"{k}1": k for k in CLASS_NAMES.values()}}

    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    @staticmethod
    def sanitize(label: str) -> str:
        return label.replace("[", "").replace("]", "").replace("'", "").replace(" ", "_").strip("_")

    @staticmethod
    def wafer_to_png(wafer_map, output_path, target_scale=8):
        """Convert wafer map to grayscale PNG (0=ดำ, 1=เทา, 2=ขาว)"""
        wafer_map = np.array(wafer_map, dtype=np.uint8)
        if wafer_map.size == 0: 
            return False
        
        #Map values to grayscale
        img = np.zeros_like(wafer_map, dtype=np.uint8)
        img[wafer_map == 1] = 127 #gray
        img[wafer_map == 2] = 255 #white
        
        target_size = 224
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        mask = np.zeros((target_size, target_size), dtype=np.uint8)
        center = (target_size // 2, target_size // 2)
        radius = target_size // 2 - 2
        cv2.circle(mask, center, radius, 255, -1)
        img = cv2.bitwise_and(img, mask)
        
        # 🔹 sharpen ภาพเล็กน้อย (เพิ่มความชัดของ die)
        kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)   
            
        return cv2.imwrite(str(output_path), img)

    def load_pkl(self, path):
        df = pd.read_pickle(path).dropna(subset=['failureType'])
        df['failureType'] = df['failureType'].astype(str).str.strip("[]' ")
        df = df[df['failureType'].isin(self.LABEL_MAP)]
        return [{'waferMap': r.waferMap, 'label': self.LABEL_MAP[r.failureType]}
                for r in df.itertuples()]

    def load_datasets(self):
        raw_dir, out_root = Path(self.config['data']['raw']), Path(self.config['data']['kaggle_png'])
        out_root.mkdir(parents=True, exist_ok=True)

        # โหลด + รวม
        pkl_records = self.load_pkl(raw_dir / "LSWMD.pkl")
        records = pkl_records 
        print(f"✅ Loaded {len(pkl_records)} from PKL")

        # Export
        export_count = {}
        for i, rec in enumerate(records):
            label = self.sanitize(rec['label'])
            label_dir = out_root / label
            label_dir.mkdir(exist_ok=True)

            count = export_count.get(label, 0)
            filename = f"{label}_{count:05d}.png"
            if self.wafer_to_png(rec['waferMap'], label_dir / filename):
                export_count[label] = count + 1
  
        print("\n✅ Export Complete!")
        for lbl, cnt in export_count.items():
            print(f"{lbl}: {cnt} images")
        print(f"📁 Saved to {out_root}")
        return export_count

    @staticmethod
    def run_merge():
        KaggleDataProcessor().load_datasets()
