import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import yaml
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# --- โหลด config (ถ้ามี) เพื่อรู้ class names ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

allowed_classes = cfg["data"]["allowed_classes"]
val_dir = cfg["data"]["val"]
model_path = "./Output/checkpoints/cnn_best.keras"

# --- โหลดโมเดล ---
model = load_model(model_path)
print("✅ Model loaded successfully.")
model.summary()

# ---------------- Load Validation Images ----------------
print(f"📂 Loading validation images from: {val_dir}")
image_paths = list(Path(val_dir).rglob("*.png")) + list(Path(val_dir).rglob("*.jpg"))
if not image_paths:
    raise FileNotFoundError(f"No images found in {val_dir}")

IMG_SIZE = (224, 224)

def load_and_preprocess_image(path):
    img = tf.keras.utils.load_img(str(path), target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ---------------- Predict Unlabeled Validation ----------------
results = []
for img_path in Path(val_dir).glob("*.png"):
    img = load_and_preprocess_image(img_path)
    pred = model.predict(img, verbose=0)
    pred_idx = np.argmax(pred)
    confidence = np.max(pred)
    pred_label = allowed_classes[pred_idx]
    results.append({
        "filename": img_path.name,
        "pred_label": pred_label,
        "confidence": confidence
    })

pd.DataFrame(results).to_csv("./Output/evaluation/cnn_predictions_unlabeled.csv", index=False)
print("✅ Saved predictions for unlabeled validation set.")
