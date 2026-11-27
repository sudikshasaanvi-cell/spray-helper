"""Stitch 9 validation photos into a 3x3 synthetic field mosaic."""
import os, json, random
import cv2
import numpy as np

VAL = "data/tomato/val"
TILE = 256
HEALTHY = "Tomato___healthy"
DISEASED = [c for c in os.listdir(VAL) if c != HEALTHY and os.path.isdir(os.path.join(VAL, c))]


def create_mosaic(seed=42, out_path="samples/field_mosaic.png", truth_path="samples/field_mosaic_truth.json"):
    rng = random.Random(seed)
    # 4 + 2 diseased from (possibly repeated) classes, 3 healthy
    diseased_classes = rng.sample(DISEASED, 6) if len(DISEASED) >= 6 else DISEASED * 2
    # ensure 6
    while len(diseased_classes) < 6:
        diseased_classes.append(rng.choice(DISEASED))
    classes = diseased_classes[:4] + [HEALTHY] * 3 + diseased_classes[4:6]
    rng.shuffle(classes)

    truth = []
    tiles = []
    for cls in classes:
        d = os.path.join(VAL, cls)
        files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        f = rng.choice(files)
        img = cv2.imread(os.path.join(d, f))
        img = cv2.resize(img, (TILE, TILE))
        tiles.append(img)
        truth.append(cls)

    # stitch 3x3
    rows = []
    for r in range(3):
        row = np.hstack(tiles[r * 3:(r + 1) * 3])
        rows.append(row)
    mosaic = np.vstack(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, mosaic)
    truth_grid = [truth[r * 3:(r + 1) * 3] for r in range(3)]
    with open(truth_path, "w") as fh:
        json.dump({"seed": seed, "classes": truth_grid}, fh, indent=2)
    return out_path, truth_grid


if __name__ == "__main__":
    out, truth = create_mosaic()
    print("Saved", out)
    print(truth)
