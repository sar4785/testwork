import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import pickle

# -----------------------------------
# 1. Load Kaggle dataset
# -----------------------------------
file_path = r"C:\Users\Phimprasert\Documents\GitHub\testwork\little_wafer\Kaggle_Dataset\LSWMD.pkl"
df = pd.read_pickle(file_path)
print("Total wafers:", len(df))
print("Columns:", df.columns.tolist())

# -----------------------------------
# 2. Clean failureType → string
# -----------------------------------
def convert_failure_type(x):
    if isinstance(x, (list, np.ndarray)):
        return str(x[0]) if len(x) > 0 else "Unknown"
    elif pd.isna(x):
        return "Unknown"
    else:
        return str(x)

df['failureType'] = df['failureType'].apply(convert_failure_type)
print("Unique labels:", df['failureType'].unique())

# -----------------------------------
# 3. Preprocess wafer map → RGB image
# -----------------------------------
def preprocess_wafer_map(wafer_map, is_png=False):
    if is_png:
        img = wafer_map  # Already loaded via cv2
        if img.ndim == 2:  # Grayscale → RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:  # BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        wafer_map = np.array(wafer_map)
        if wafer_map.size == 0:
            return None
        img = np.zeros((*wafer_map.shape, 3), dtype=np.uint8)
        img[wafer_map == 1] = [0, 255, 0]  # green
        img[wafer_map == 2] = [0, 0, 255]  # red
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    return img.astype(np.float32) / 255.0

# -----------------------------------
# 4. Export test set (100 images/label)
# -----------------------------------
output_root = r"C:\Users\Phimprasert\Documents\GitHub\testwork\little_wafer\TestSet"
os.makedirs(output_root, exist_ok=True)
max_images_per_label = 100
export_count = {}
test_indices = []

for idx, row in df.iterrows():
    label = row['failureType']
    if export_count.get(label, 0) >= max_images_per_label:
        continue
    processed_map = preprocess_wafer_map(row['waferMap'])
    if processed_map is None:
        continue
    label_folder = os.path.join(output_root, label)
    os.makedirs(label_folder, exist_ok=True)
    filename = os.path.join(label_folder, f"{label}_{export_count.get(label, 0):03d}.png")
    cv2.imwrite(filename, (processed_map * 255).astype(np.uint8))
    export_count[label] = export_count.get(label, 0) + 1
    test_indices.append(idx)

print("✅ TestSet Export Done")
for label, count in export_count.items():
    print(f"{label}: {count} images saved")

# -----------------------------------
# 5. Load real-world images
# -----------------------------------
def load_real_images(real_path, label):
    imgs, labels = [], []
    for fname in os.listdir(real_path):
        if fname.lower().endswith(".png"):
            img = cv2.imread(os.path.join(real_path, fname))
            if img is None:
                continue
            img = preprocess_wafer_map(img, is_png=True)
            imgs.append(img)
            labels.append(label)
    return imgs, labels

real_path = r"C:\Users\Phimprasert\Documents\GitHub\testwork\little_wafer\PRR_to_PNG_and_Resize\Wafer-Map_resized"
real_imgs, real_labels = load_real_images(real_path, "Center")
df_real = pd.DataFrame({"waferMap": real_imgs, "failureType": real_labels})

# -----------------------------------
# 6. Combine Kaggle (train/val) + real data
# -----------------------------------
train_val_df = df.drop(index=test_indices).reset_index(drop=True)
df_all = pd.concat([train_val_df, df_real], ignore_index=True)
le = LabelEncoder()
df_all['label_id'] = le.fit_transform(df_all['failureType'])
num_classes = len(le.classes_)
y_onehot = tf.keras.utils.to_categorical(df_all['label_id'], num_classes)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    df_all[['waferMap', 'failureType']],
    y_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_onehot
)
print("Train size:", len(X_train))
print("Val size:", len(X_val))

# -----------------------------------
# 7. Use subset for FAST training
# -----------------------------------
train_subset = 2000
val_subset = 500
X_train = X_train.sample(n=min(train_subset, len(X_train)), random_state=42).reset_index(drop=True)
y_train = y_train[:len(X_train)]
X_val = X_val.sample(n=min(val_subset, len(X_val)), random_state=42).reset_index(drop=True)
y_val = y_val[:len(X_val)]
print(f"Train subset: {len(X_train)}, Val subset: {len(X_val)}")

# -----------------------------------
# 8. Convert DataFrame → NumPy
# -----------------------------------
def df_to_numpy(df, y):
    imgs = []
    for i, row in df.iterrows():
        img = preprocess_wafer_map(row['waferMap'])
        if img is not None:
            imgs.append(img)
    return np.array(imgs), y[:len(imgs)]

X_train_np, y_train_np = df_to_numpy(X_train, y_train)
X_val_np, y_val_np = df_to_numpy(X_val, y_val)
print(f"X_train_np shape: {X_train_np.shape}, y_train_np shape: {y_train_np.shape}")
print(f"X_val_np shape: {X_val_np.shape}, y_val_np shape: {y_val_np.shape}")

# -----------------------------------
# 9. Model (EfficientNet pretrained)
# -----------------------------------
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze backbone
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs=base_model.input, outputs=out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# -----------------------------------
# 10. Train (FAST)
# -----------------------------------
history = model.fit(
    X_train_np, y_train_np,
    validation_data=(X_val_np, y_val_np),
    epochs=5,
    batch_size=32,
    verbose=1
)
print("✅ Training Done")

# Save model + LabelEncoder
model.save("wafer_classifier_fast.h5")
pickle.dump(le, open("label_encoder.pkl", "wb"))

# -----------------------------------
# 11. Test prediction
# -----------------------------------
def load_test_folder(folder_path):
    imgs, names = [], []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".png"):
                img = cv2.imread(os.path.join(root, fname))
                if img is None:
                    continue
                img = preprocess_wafer_map(img, is_png=True)
                imgs.append(img)
                names.append(fname)
    return np.array(imgs), names

test_folder = output_root
X_test, test_names = load_test_folder(test_folder)
print("X_test shape:", X_test.shape)
preds = model.predict(X_test, verbose=0)
pred_ids = np.argmax(preds, axis=1)
pred_labels = le.inverse_transform(pred_ids)

df_results = pd.DataFrame({"filename": test_names, "predicted": pred_labels})
df_results.to_csv("test_predictions.csv", index=False)
print("✅ Test Predictions saved to test_predictions.csv")
print(df_results.head())