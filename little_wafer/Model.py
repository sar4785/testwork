import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import cv2 
import os

# ------------------------------
# 1. Load wm-811k dataset
# ------------------------------

file_path = r"C:\Users\User\Documents\GitHub\testwork\little_wafer\Kaggle_Dataset\LSWMD.pkl"
with open(file_path, 'rb') as f:
    df = pd.read_pickle(f)
print("Total wafers:", len(df))
print("Columns:", df.columns.tolist())

# ------------------------------
# 2. Clean failureType → string
# ------------------------------

def convert_failure_type(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return "Unknown"
        else:
            return str(x[0])  # ใช้ค่าแรกของ array
    elif pd.isna(x):
        return "Unknown"
    else:
        return str(x)
df['failureType'] = df['failureType'].apply(convert_failure_type)
print("\nUnique failure types:", df['failureType'].unique())
print("\nFailure type counts:\n", df['failureType'].value_counts())

# ------------------------------
# 3. Encode labels
# ------------------------------
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['failureType'])
# One-hot encoding สำหรับ CNN
y_onehot = tf.keras.utils.to_categorical(df['label_id'])
print("\nLabel mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print("y_onehot shape:", y_onehot.shape)

# ------------------------------
# 4. Split train/test
# ------------------------------
# เราจะใช้แค่ label และ waferMap index ในขั้นตอนนี้
# ภาพจะ preprocess ตอนใช้ generator เพื่อไม่กิน RAM
train_df, test_df, y_train, y_test = train_test_split(
    df[['waferMap', 'failureType', 'label_id']],
    y_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_onehot
)
print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))

# ------------------------------
# 5. สร้าง Model
# ------------------------------
num_classes = y_onehot.shape[1]
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ------------------------------
# 6. Train
# ------------------------------
history = model.fit(
    train_df,
    validation_data=test_df,
    epochs=5  # ลองน้อยๆ ก่อน
)

# ------------------------------
# 7. Predict รูป .png ของคุณเอง
# ------------------------------
def preprocess_png(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # โหลดเป็น grayscale
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    img = np.stack([img]*3, axis=-1)  # ทำเป็น RGB (3 channel)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)  # เพิ่ม batch dim

def predict_image(img_path):
    img = preprocess_png(img_path)
    preds = model.predict(img, verbose=0)  # ปิด progress bar
    class_id = np.argmax(preds, axis=1)[0]
    class_name = le.inverse_transform([class_id])[0]
    return class_name

def predict_folder(folder_path):
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".png"):
            file_path = os.path.join(folder_path, file_name)
            label = predict_image(file_path)
            results.append((file_name, label))
            print(f"{file_name} → Predicted class: {label}")
    return results

# ------------------------------
# ใช้งาน
# ------------------------w------
folder_path = r"C:\Users\User\Documents\GitHub\testwork\little_wafer\PRR_to_PNG_and_Resize\Wafer-Map_resized"
predict_results = predict_folder(folder_path)

df_results = pd.DataFrame(predict_results, columns=["filename", "predicted_label"])
print(df_results.head())