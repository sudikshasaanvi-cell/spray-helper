"""Whole-field mode: split one photo into a 3x3 grid and decide spray per cell."""
import os, sys
import cv2
import numpy as np

import predict
from severity import estimate_severity, leaf_mask
import field_sim

EXG_MIN = 20     # assumption: ExG threshold for vegetation
VEG_MIN = 0.20   # assumption: minimum vegetation fraction in a tile
INFECT_CONF = 0.90  # assumption: min confidence to call a cell infected
SEV_MIN = 10.0   # assumption: min severity % to call a cell infected


def _clean(name):
    if not name:
        return ""
    return name.split("___")[-1].replace("_", " ").strip()


def _veg_mask(tile):
    b, g, r = cv2.split(tile)
    exg = 2 * g.astype("int16") - r.astype("int16") - b.astype("int16")
    veg = (exg > EXG_MIN).astype("uint8") * 255
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return veg


def analyze_field(img_bgr, rows=3, cols=3):
    h, w = img_bgr.shape[:2]
    th = h // rows
    tw = w // cols

    grid = []
    cells = []
    for r in range(rows):
        g_row = []
        for c in range(cols):
            y0 = r * th
            x0 = c * tw
            y1 = (r + 1) * th if r < rows - 1 else h
            x1 = (c + 1) * tw if c < cols - 1 else w
            tile = img_bgr[y0:y1, x0:x1]

            name = chr(ord("A") + r) + str(c + 1)
            tile_px = tile.shape[0] * tile.shape[1]
            veg = _veg_mask(tile)
            veg_px = int(np.count_nonzero(veg))
            veg_fraction = veg_px / tile_px if tile_px else 0.0

            # dilate vegetation mask by 5 px
            kernel = np.ones((11, 11), np.uint8)  # 5 px each side approx
            dilated = cv2.dilate(veg, kernel, iterations=1)
            # clean tile: paint outside dilated mask grey
            clean = tile.copy()
            clean[dilated == 0] = (175, 175, 175)

            if veg_fraction < VEG_MIN:
                status = "healthy"
                reason = "No crop here - skip"
                cls = None
                conf = None
                sev_pct = None
                level = None
                display_status = "healthy"
            else:
                cls, conf = predict.predict(clean)
                if "healthy" in str(cls).lower():
                    status = "healthy"
                    reason = "no clear disease"
                    sev_pct = 0.0
                    level = None
                    display_status = "healthy"
                elif conf is not None and conf < INFECT_CONF:
                    status = "retake"
                    reason = "not sure - take a close-up"
                    sev_pct = None
                    level = None
                    display_status = "retake"
                else:
                    sev = estimate_severity(clean, healthy_hue=(25, 95), sat_min=30, val_min=30, mask=dilated)
                    sev_pct = sev["pct"]
                    level = sev["level"]
                    if sev_pct is not None and sev_pct >= SEV_MIN:
                        status = "infected"
                        reason = ""
                        display_status = "suspected"
                    else:
                        status = "healthy"
                        reason = "no clear disease"
                        display_status = "healthy"

            if status == "retake":
                cell_tuple = ("retake", None)
            elif status == "healthy":
                cell_tuple = ("healthy", None)
            else:
                cell_tuple = ("infected", level)

            g_row.append(cell_tuple)
            cells.append({
                "name": name,
                "row": r,
                "col": c,
                "bbox": (x0, y0, x1, y1),
                "status": status,
                "display_status": display_status,
                "reason": reason,
                "class": cls,
                "confidence": (round(float(conf), 3) if conf is not None else None),
                "severity": sev_pct,
                "level": level,
                "veg_fraction": round(float(veg_fraction), 3),
                "next_step": "Take a close-up of a leaf in this square" if display_status in ("suspected", "retake") else "",
            })
        grid.append(g_row)

    result = field_sim.compare(grid)
    return grid, cells, result


def draw_field_map(img_bgr, cells, rows, cols):
    out = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    th = h // rows
    tw = w // cols

    overlay = out.copy()
    color_map = {"infected": (0, 0, 255), "retake": (0, 255, 255), "healthy": (0, 255, 0),
                 "suspected": (0, 0, 255)}
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        color = color_map.get(cell.get("display_status", cell["status"]), (200, 200, 200))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    for r in range(1, rows):
        y = r * th
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 2)
    for c in range(1, cols):
        x = c * tw
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 2)

    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        cx = x0 + 8
        cy = y0 + 22
        disease = _clean(cell["class"]) if cell["class"] else cell.get("display_status", cell["status"])
        if cell.get("reason") == "No crop here - skip":
            disease = "No crop"
        label1 = f"{cell['name']} {disease}"
        cv2.putText(out, label1, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label1, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if cell["severity"] is not None:
            label2 = f"{cell['severity']}%"
            cv2.putText(out, label2, (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label2, (cx, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def recommendation_text(cells, result):
    spray = []
    skip = []
    retake = []
    for cell in cells:
        name = cell["name"]
        disease = _clean(cell["class"]) if cell["class"] else ""
        level = cell["level"]
        disp = cell.get("display_status", cell["status"])
        if disp == "suspected":
            spray.append(f"{name} ({disease}, {level}) (suspected)")
        elif cell["status"] == "healthy":
            skip.append(f"{name} (healthy)")
        else:
            reason = cell["reason"] or "not sure"
            retake.append(f"{name} ({reason})")
    lines = []
    lines.append("Spray: " + (", ".join(spray) if spray else "none"))
    lines.append("Skip: " + (", ".join(skip) if skip else "none"))
    lines.append("Retake photo: " + (", ".join(retake) if retake else "none"))
    lines.append(f"Pesticide: Mode A {result['A']['litres']} L, Mode B {result['B']['litres']} L, saved {result['litres_saved_pct']}% (assumptions shown)")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python field_map.py image.jpg")
        sys.exit(0)
    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print("could not read", path)
        sys.exit(1)
    grid, cells, result = analyze_field(img)
    out = draw_field_map(img, cells, 3, 3)
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/field_map.png", out)
    for c in cells:
        print(f"{c['name']}: veg={c['veg_fraction']} class={c['class']} conf={c['confidence']} sev={c['severity']} decision={c.get('display_status',c['status'])} reason={c['reason']}")
    print()
    print(recommendation_text(cells, result))
    print("Saved results/field_map.png")
