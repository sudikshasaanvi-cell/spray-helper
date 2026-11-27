"""Build a 3x3 field from 9 sample photos and compare spray modes."""
import os, cv2, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import predict
from severity import estimate_severity
import field_sim

SAMPLES = "samples"
GRID_ROWS, GRID_COLS = 3, 3

model, class_names = predict.load()

grid = []
report_lines = []
for i in range(1, 10):
    path = os.path.join(SAMPLES, f"{i:02d}.jpg")
    img = cv2.imread(path)
    label, conf = predict.predict(img)
    if label == "retake":
        cell = ("retake", None)
    elif "healthy" in label.lower():
        cell = ("healthy", None)
    else:
        sev = estimate_severity(img)
        cell = ("infected", sev["level"])
    grid.append(cell)
    # fill row by row later; here collect flat
    report_lines.append(f"{i:02d}.jpg -> {label} (conf {conf:.2f}) -> {cell}")

# reshape flat list into 3x3 row-major
grid3 = [grid[r*GRID_COLS:(r+1)*GRID_COLS] for r in range(GRID_ROWS)]

out = field_sim.compare(grid3)
# save images
figA = field_sim.draw(grid3, out["A"], f"Mode A: spray all - {out['A']['litres']} L, {out['A']['seconds']} s")
figA.savefig("results/field_modeA.png", dpi=120)
figB = field_sim.draw(grid3, out["B"], f"Mode B: AI-targeted - {out['B']['litres']} L, {out['B']['seconds']} s")
figB.savefig("results/field_modeB.png", dpi=120)
plt.close("all")

summary = []
summary.append("Field grid (row-major):")
for r in grid3:
    summary.append(str(r))
summary.append("")
summary.append(f"Mode A: {out['A']['litres']} L, {out['A']['seconds']} s")
summary.append(f"Mode B: {out['B']['litres']} L, {out['B']['seconds']} s")
summary.append(f"Pesticide saved: {out['litres_saved_pct']}%")
summary.append(f"Time saved: {out['time_saved_pct']}%")
summary.append(f"Retake cells: {out['B']['retake_cells']}")
summary.append("")
summary.append("Assumption constants used:")
summary.append(f"  DOSE_FACTOR = {field_sim.DOSE_FACTOR}")
summary.append(f"  sec_move default 10s, sec_spray 15s, litres_per_cell 0.5 (from field_sim / app sliders)")
summary.append("")
summary.append("Predictions:")
summary.extend(report_lines)
text = "\n".join(summary)
print(text)
with open("results/field_summary.txt", "w") as f:
    f.write(text + "\n")
