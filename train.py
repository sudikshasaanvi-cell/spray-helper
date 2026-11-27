"""Train a MobileNetV2 classifier on the tomato leaf dataset."""
import os, json, time, glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config

SEED = config.SEED
IMG_SIZE = config.IMG_SIZE
BATCH = config.BATCH
MAX_PER_CLASS = config.MAX_PER_CLASS

DATA_ROOT = "data/tomato/train"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---- collect image paths + labels ----
class_names = sorted(os.listdir(DATA_ROOT))
class_names = [c for c in class_names if os.path.isdir(os.path.join(DATA_ROOT, c))]
print("Classes:", class_names)

paths, labels = [], []
for ci, cname in enumerate(class_names):
    folder = os.path.join(DATA_ROOT, cname)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = files[:MAX_PER_CLASS]
    for f in files:
        paths.append(os.path.join(folder, f))
        labels.append(ci)

paths = np.array(paths)
labels = np.array(labels)
print("Total images used:", len(paths))

# stratified 80/20 split
tr_p, va_p, tr_y, va_y = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
print("Train:", len(tr_p), "Val:", len(va_p))

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32")

def make_ds(paths_arr, ys, shuffle=False, augment=False):
    def gen():
        for p, y in zip(paths_arr, ys):
            yield load_img(p), y
    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)))
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    ds = ds.batch(BATCH)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(tr_p, tr_y, shuffle=True, augment=True)
val_ds = make_ds(va_p, va_y, shuffle=False, augment=False)

# ---- model ----
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ---- phase 1: frozen base, 5 epochs ----
print("\n=== Phase 1: frozen base, 5 epochs ===")
for ep in range(8):
    t0 = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    print(f"epoch {ep+1} time: {time.time()-t0:.1f}s  val_acc={h.history['val_accuracy'][-1]:.4f}")

# ---- phase 2: unfreeze last 30 layers, lr 1e-5, 3 epochs ----
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("\n=== Phase 2: unfreeze last 30, lr=1e-5, 3 epochs ===")
for ep in range(5):
    t0 = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    print(f"epoch {ep+1} time: {time.time()-t0:.1f}s  val_acc={h.history['val_accuracy'][-1]:.4f}")

# ---- evaluate ----
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\nValidation accuracy: {val_acc:.4f}")

# predictions for confusion matrix
y_true, y_pred = [], []
for x, y in val_ds:
    p = model.predict(x, verbose=0)
    y_true.extend(y.numpy().tolist())
    y_pred.extend(np.argmax(p, axis=1).tolist())
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

# save
model.save("model/tomato.keras")
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)
with open("results/accuracy.txt", "w") as f:
    f.write(f"Validation accuracy: {val_acc:.4f}\n")
with open("results/classification_report.txt", "w") as f:
    f.write(report)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
print("Saved model, class_names, accuracy, confusion matrix, report.")
