"""Fine-tune the tomato model on PlantDoc field photos (70/30 split)."""
import os, json, random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import config
import predict

SEED = config.SEED
IMG_SIZE = config.IMG_SIZE
BATCH = config.BATCH

# ---- gather lab training images ----
LAB_ROOT = "data/tomato/train"
model, class_names = predict.load()
name_to_idx = {n: i for i, n in enumerate(class_names)}

lab_paths, lab_labels = [], []
for cname in os.listdir(LAB_ROOT):
    folder = os.path.join(LAB_ROOT, cname)
    if not os.path.isdir(folder):
        continue
    if cname not in name_to_idx:
        continue
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = files[:config.MAX_PER_CLASS]
    for f in files:
        lab_paths.append(os.path.join(folder, f))
        lab_labels.append(name_to_idx[cname])

# ---- split field photos 70/30 by image ----
FIELD_DIR = "field_test"
field_train_paths, field_train_labels = [], []
field_test = []  # list of (path, true_idx)
rng = random.Random(SEED)
for cname in os.listdir(FIELD_DIR):
    folder = os.path.join(FIELD_DIR, cname)
    if not os.path.isdir(folder) or cname not in name_to_idx:
        continue
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    rng.shuffle(files)
    cut = int(0.7 * len(files))
    train_files = files[:cut]
    test_files = files[cut:]
    for f in train_files:
        field_train_paths.append(os.path.join(folder, f))
        field_train_labels.append(name_to_idx[cname])
    for f in test_files:
        field_test.append((os.path.join(folder, f), name_to_idx[cname]))

print(f"Lab train: {len(lab_paths)}, field train (70%): {len(field_train_paths)}, field test (30%): {len(field_test)}")

# ---- combined training set ----
all_paths = np.array(lab_paths + field_train_paths)
all_labels = np.array(lab_labels + field_train_labels)

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype("float32")

def make_ds(paths_arr, ys, shuffle=False):
    def gen():
        for p, y in zip(paths_arr, ys):
            yield load_img(p), y
    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)))
    if shuffle:
        ds = ds.shuffle(2000, seed=SEED)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(all_paths, all_labels, shuffle=True)

# ---- fine-tune: unfreeze last 50 layers, lr 1e-5, 3 epochs ----
base = model.get_layer("mobilenetv2_1.00_160")
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

for ep in range(3):
    h = model.fit(train_ds, epochs=1, verbose=1)
    print(f"finetune epoch {ep+1}: acc={h.history['accuracy'][-1]:.4f}")

model.save("model/tomato.keras")
# refresh cached predict model
predict._cached_model = None
predict._cached_names = None

# ---- evaluate on held-out 30% ----
records = []  # (true_idx, pred_idx, conf)
correct = 0
for path, true_idx in field_test:
    img = cv2.imread(path)
    label, conf = predict.predict(img)
    if label == "retake":
        pred_idx = -1
    else:
        pred_idx = class_names.index(label) if label in class_names else -1
    records.append((true_idx, pred_idx, conf, label))
    if pred_idx == true_idx:
        correct += 1

acc = correct / len(field_test) if field_test else 0.0
print(f"Held-out field accuracy after fine-tune: {acc:.4f} ({correct}/{len(field_test)})")
with open("results/field_accuracy_after.txt", "w") as f:
    f.write(f"Held-out field photos (30%): {len(field_test)}\n")
    f.write(f"Accuracy after fine-tune: {acc:.4f} ({correct}/{len(field_test)})\n")

# ---- threshold table ----
thresholds = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
answered_pcts = []
accs = []
for thr in thresholds:
    answered = [(t, p, c) for (t, p, c, _) in records if c >= thr and p != -1]
    n_ans = len(answered)
    pct_ans = 100.0 * n_ans / len(records) if records else 0.0
    corr = sum(1 for t, p, c in answered if t == p)
    a = corr / n_ans if n_ans else 0.0
    answered_pcts.append(pct_ans)
    accs.append(a)
    print(f"thr {thr:.2f}: answered {pct_ans:.1f}% acc {a:.3f}")

# plot
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(thresholds, answered_pcts, "o-", color="#1976D2", label="% answered")
ax1.set_xlabel("Confidence threshold")
ax1.set_ylabel("% photos answered", color="#1976D2")
ax2 = ax1.twinx()
ax2.plot(thresholds, [a * 100 for a in accs], "s-", color="#2E7D32", label="accuracy on answered")
ax2.set_ylabel("Accuracy on answered (%)", color="#2E7D32")
ax1.set_title("Confidence threshold vs coverage/accuracy (held-out 30%)")
fig.tight_layout()
fig.savefig("results/threshold_table.png", dpi=120)
plt.close(fig)
print("Saved results/threshold_table.png")

# recommend threshold: highest threshold where answered >= 50% and accuracy max
best = None
for thr, pa, a in zip(thresholds, answered_pcts, accs):
    if pa >= 50.0:
        if best is None or a > best[2]:
            best = (thr, pa, a)
print("Recommended threshold:", best)
with open("results/field_accuracy_after.txt", "a") as f:
    f.write(f"\nRecommended CONF_MIN from threshold table: {best[0]} "
            f"(answered {best[1]:.1f}%, accuracy {best[2]:.3f}).\n")
    f.write("Chosen as the highest accuracy among thresholds keeping >=50% coverage.\n")
