import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Load dataset
file_path = r"C:\Users\User\Documents\GitHub\testwork\little_wafer\Kaggle_Dataset\LSWMD.pkl"
with open(file_path, 'rb') as f:
    df = pd.read_pickle(f)
    
# 2. Clean failureType → string
def convert_failure_type(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return "Unknown"
        else:
            return str(x[0])
    elif pd.isna(x):
        return "Unknown"
    else:
        return str(x)
df['failureType'] = df['failureType'].apply(convert_failure_type)

print("Unique labels:", df['failureType'].unique())

# 3. ตั้งค่าที่จะ save รูป
output_root = r"C:\Users\User\Documents\GitHub\testwork\little_wafer\WaferMap_PNG"
os.makedirs(output_root, exist_ok=True)

# 4. Export 100 รูป/label
max_images_per_label = 100
export_count = {}

for idx, row in df.iterrows():
    label = row['failureType']
    wafer_map = row['waferMap']

    # จำกัดจำนวนรูปต่อ label
    if export_count.get(label, 0) >= max_images_per_label:
        continue
     # preprocess wafer map
    wafer_map = np.array(wafer_map, dtype=np.uint8)
    if wafer_map.size == 0:
        continue  # ข้ามถ้าว่าง    
    # to 3 channel (RGB)
    wafer_map = np.stack([wafer_map] * 3, axis=-1)
    # resize → 224x224
    wafer_map = cv2.resize(wafer_map, (224, 224), interpolation=cv2.INTER_NEAREST)
    # สร้าง folder ของ label
    label_folder = os.path.join(output_root, label)
    os.makedirs(label_folder, exist_ok=True)
    # ตั้งชื่อไฟล์
    img_filename = os.path.join(label_folder, f"{label}_{export_count.get(label,0):03d}.png")
    # save image
    cv2.imwrite(img_filename, wafer_map)
    # update counter
    export_count[label] = export_count.get(label, 0) + 1

print("Export done ✅")
for label, count in export_count.items():
    print(f"{label}: {count} images saved")

sample = df.iloc[0]['waferMap']
plt.imshow(sample, cmap="gray")
plt.title(df.iloc[0]['failureType'])
plt.show()