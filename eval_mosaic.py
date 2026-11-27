"""Evaluate whole-field mode on synthetic mosaics."""
import os, json, random
import cv2

import make_mosaic
import field_map

N = 5
accs = []
for i in range(N):
    seed = 100 + i
    out_path = f"samples/mosaic_{i}.png"
    truth_path = f"samples/mosaic_{i}_truth.json"
    make_mosaic.create_mosaic(seed=seed, out_path=out_path, truth_path=truth_path)
    img = cv2.imread(out_path)
    grid, cells, result = field_map.analyze_field(img, 3, 3)
    with open(truth_path) as f:
        truth = json.load(f)["classes"]
    # flatten truth
    truth_flat = [truth[r][c] for r in range(3) for c in range(3)]
    correct = 0
    for cell, t in zip(cells, truth_flat):
        pred = cell["class"]
        if pred == t:
            correct += 1
    acc = correct / 9.0
    accs.append(acc)
    print(f"Mosaic {i+1} (seed {seed}): tile accuracy = {acc:.3f} ({correct}/9)")

avg = sum(accs) / len(accs)
print(f"Average tile accuracy over {N} mosaics: {avg:.3f}")

os.makedirs("results", exist_ok=True)
with open("results/mosaic_accuracy.txt", "w") as f:
    f.write(f"Average tile accuracy over {N} synthetic mosaics: {avg:.3f}\n")
    for i, a in enumerate(accs):
        f.write(f"  mosaic {i+1}: {a:.3f}\n")
    f.write("\nNOTE: These mosaics are SYNTHETIC (stitched validation photos). "
            "They do not represent real field images.\n")
