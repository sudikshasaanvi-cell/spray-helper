"""Evaluate the trained model on real-world PlantDoc field photos."""
import os, json
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import predict

LAB_ACC = 0.8125  # from results/accuracy.txt (validation accuracy)

FIELD_DIR = "field_test"
model, class_names = predict.load()

# only classes present in field_test
field_classes = sorted(os.listdir(FIELD_DIR))
field_classes = [c for c in field_classes if os.path.isdir(os.path.join(FIELD_DIR, c))]

y_true, y_pred = [], []
per_class = {c: [0, 0] for c in field_classes}  # correct, total
for ci, cname in enumerate(field_classes):
    folder = os.path.join(FIELD_DIR, cname)
    for f in os.listdir(folder):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        pred, conf = predict.predict(img)
        true_idx = class_names.index(cname) if cname in class_names else None
        # map prediction to index
        if pred == "retake":
            pred_idx = -1
        else:
            pred_idx = class_names.index(pred)
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        per_class[cname][1] += 1
        if pred == cname:
            per_class[cname][0] += 1

# accuracy treating retake as wrong
correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
total = len(y_true)
acc = correct / total if total else 0.0

lines = []
lines.append(f"Field photos used: {total}")
lines.append(f"Field accuracy: {acc:.4f} ({correct}/{total})")
lines.append(f"Lab validation accuracy: {LAB_ACC:.4f}")
lines.append(f"Drop: {(LAB_ACC - acc)*100:.1f} points")
lines.append("")
lines.append("Per-class (correct/total):")
for c in field_classes:
    cr, tot = per_class[c]
    lines.append(f"  {c}: {cr}/{tot} = {cr/tot:.3f}" if tot else f"  {c}: 0/0")

text = "\n".join(lines)
print(text)
with open("results/field_accuracy.txt", "w") as f:
    f.write(text + "\n")

# confusion matrix (map -1 retake to a separate index? just skip retake in cm by treating as last)
# Build cm over field_classes indices
idx_map = {class_names.index(c): i for i, c in enumerate(field_classes)}
yt = [idx_map[t] for t in y_true if t in idx_map]
yp = []
for t, p in zip(y_true, y_pred):
    if t not in idx_map:
        continue
    if p in idx_map and class_names[p] in field_classes:
        yp.append(idx_map[class_names[p]] if False else field_classes.index(class_names[p]))
    else:
        # retake or other -> put in a dummy? use -1 marker; instead count as wrong in first? 
        yp.append(-1)
# For cm, replace -1 with a separate column? Simpler: build cm with an extra 'retake/other' column.
labels = field_classes + ["retake/other"]
cm = np.zeros((len(field_classes), len(labels)), dtype=int)
for t, p in zip(y_true, y_pred):
    if t not in idx_map:
        continue
    ri = idx_map[t]
    if p == -1 or (p >= 0 and class_names[p] not in field_classes):
        ci = len(field_classes)
    else:
        ci = field_classes.index(class_names[p])
    cm[ri, ci] += 1

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=field_classes, cmap="Blues")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/field_confusion.png")
print("Saved results/field_confusion.png")
