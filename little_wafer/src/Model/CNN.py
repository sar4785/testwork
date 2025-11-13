import os
import time
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ---------------- Load Config ----------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

train_dir = cfg["data"]["train"]
val_dir = cfg["data"]["val"]
allowed_classes = cfg["data"]["allowed_classes"]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3  # ตั้งไว้เยอะเพื่อให้ early stopping ตัดเอง

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')  # แก้เป็น [0]
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ Using GPU: {gpus[1].name}")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("⚠️ No GPU detected, running on CPU.")

# ---------------- Load Training Dataset ----------------
print("📂 Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    class_names=allowed_classes,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_ds = train_ds.map(lambda x, y: (x / 255.0, y))

# ---------------- Load Unlabeled Validation Images ----------------
print("📂 Loading validation images (unlabeled)...")
val_files = list(Path(val_dir).glob("*.png")) + list(Path(val_dir).glob("*.jpg"))
if not val_files:
    print("⚠️ No validation images found.")
    val_imgs = np.empty((0, *IMG_SIZE, 3))
else:
    val_imgs = np.stack([
        tf.keras.utils.img_to_array(
            tf.keras.utils.load_img(str(f), target_size=IMG_SIZE)
        )
        for f in val_files
    ])
    val_imgs = val_imgs / 255.0
print(f"✅ Loaded {len(val_imgs)} validation images")

# ---------------- Enable GPU ----------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # ใช้ GPU ตัวแรกเท่านั้น (GPU:0)
        tf.config.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ GPU enabled: {gpus[0].name} ({len(logical_gpus)} logical GPUs)")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("⚠️ No GPU detected, running on CPU.")

# ---------------- Define Model ----------------
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = IMG_SIZE + (3,)
num_classes = len(allowed_classes)

model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------- Callbacks ----------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./Output/checkpoints/cnn_best.keras",
        monitor='accuracy',
        save_best_only=True
    )
]

# ---------------- Training ----------------
os.makedirs("./Output/checkpoints", exist_ok=True)
print("\n🚀 Start training CNN model...")
start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏱️ Training time: {elapsed_time/60:.2f} minutes")

# ---------------- Save Model ----------------
model.save("./Output/checkpoints/cnn_final.keras")

# ---------------- Plot Learning Curves ----------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("./Output/training_curves.png", dpi=300)
plt.show()

# ---------------- Save Training Metrics ----------------
metrics_df = pd.DataFrame(history.history)
metrics_df["epoch"] = range(1, len(metrics_df) + 1)
metrics_df["train_time_sec"] = elapsed_time
metrics_df.to_csv("./Output/CNN_tensorflow_training_metrics.csv", index=False)
print("✅ Training metrics saved to ./Output/training_metrics.csv")

# ---------------- Optional: Predict on Validation Images ----------------
if len(val_imgs) > 0:
    preds = model.predict(val_imgs, verbose=1)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    
    df = pd.DataFrame({
        "filename": [p.name for p in val_files],
        "pred_label": [allowed_classes[i] for i in pred_classes],
        "confidence": confidences
    })
    
    df.to_csv("./Output/predictions/CNN_tensorflow_val_predictions.csv", index=False)
    print("✅ Predictions saved to ./Output/predictions/val_predictions.csv")
