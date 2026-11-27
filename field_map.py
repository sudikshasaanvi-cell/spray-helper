"""Whole-field mode: split one photo into a 3x3 grid and decide spray per cell."""
import os, sys
import cv2
import numpy as np

from severity import leaf_mask
import field_sim
import predict

EXG_MIN = 20       # assumption: ExG threshold for vegetation
VEG_MIN = 0.20     # assumption: minimum vegetation fraction in a tile/sub-crop
BLUR_MIN = 100.0   # assumption: Laplacian variance below this => blurry
F_SPRAY_DEFAULT = 0.5
F_SKIP_DEFAULT = 0.15

# Strictness presets: (T_AFFECTED, F_SPRAY, F_SKIP)
STRICTNESS = {
    "cautious": (0.85, 0.60, 0.10),
    "balanced": (0.80, 0.50, 0.15),
    "fewer": (0.50, 0.30, 0.30),
}


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


def _affected_score(img_bgr):
    model, names = predict.load()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160))
    arr = np.expand_dims(resized.astype("float32"), 0)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    arrp = preprocess_input(arr.copy())
    probs = model.predict(arrp, verbose=0)[0]
    healthy_idx = None
    for i, nm in enumerate(names):
        if "healthy" in nm.lower():
            healthy_idx = i
            break
    if healthy_idx is None:
        return 0.5
    return 1.0 - float(probs[healthy_idx])


def _subcrop_decisions(tile, T, F_SPRAY, F_SKIP):
    h, w = tile.shape[:2]
    th, tw = h // 3, w // 3
    used = 0
    affected = 0
    for r in range(3):
        for c in range(3):
            sub = tile[r*th:(r+1)*th, c*tw:(c+1)*tw]
            vm = _veg_mask(sub)
            frac = np.count_nonzero(vm) / (sub.shape[0] * sub.shape[1])
            if frac < VEG_MIN:
                continue
            kernel = np.ones((11, 11), np.uint8)
            dil = cv2.dilate(vm, kernel, iterations=1)
            clean = sub.copy()
            clean[dil == 0] = (175, 175, 175)
            sc = _affected_score(clean)
            used += 1
            if sc >= T:
                affected += 1
    if used == 0:
        return None, 0, 0
    f = affected / used
    return f, used, affected


def analyze_field(img_bgr, rows=3, cols=3, strictness="balanced"):
    T, F_SPRAY, F_SKIP = STRICTNESS.get(strictness, STRICTNESS["balanced"])
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

            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if blur_var < BLUR_MIN:
                status = "retake"
                reason = "blurry photo"
                decision = "Retake"
                f = None
                used = 0
            elif veg_fraction < VEG_MIN:
                status = "healthy"
                reason = "No crop here - skip"
                decision = "No crop - skip"
                f = None
                used = 0
            else:
                f, used, aff = _subcrop_decisions(tile, T, F_SPRAY, F_SKIP)
                if f is None:
                    status = "healthy"
                    reason = "No crop here - skip"
                    decision = "No crop - skip"
                elif f >= F_SPRAY:
                    status = "infected"
                    reason = ""
                    decision = "Spray"
                elif f <= F_SKIP:
                    status = "healthy"
                    reason = "no clear disease"
                    decision = "Skip"
                else:
                    status = "healthy"
                    reason = "watch"
                    decision = "Watch"

            if status == "retake":
                cell_tuple = ("retake", None)
            elif status == "healthy":
                cell_tuple = ("healthy", None)
            else:
                cell_tuple = ("infected", "Moderate")

            g_row.append(cell_tuple)
            next_step = ""
            if decision in ("Spray", "Retake", "Watch"):
                next_step = "Take a close-up of a leaf in this square"
            cells.append({
                "name": name,
                "row": r,
                "col": c,
                "bbox": (x0, y0, x1, y1),
                "status": status,
                "display_status": decision,
                "reason": reason,
                "class": None,
                "confidence": None,
                "severity": None,
                "level": "Moderate" if status == "infected" else None,
                "veg_fraction": round(float(veg_fraction), 3),
                "f": (round(float(f), 3) if f is not None else None),
                "subcrops_used": used,
                "blur_var": round(blur_var, 1),
                "next_step": next_step,
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
    color_map = {
        "Spray": (0, 0, 255),
        "Retake": (0, 255, 255),
        "Skip": (0, 255, 0),
        "Watch": (0, 165, 255),
        "No crop - skip": (180, 180, 180),
    }
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        disp = cell.get("display_status", cell["status"])
        color = color_map.get(disp, (200, 200, 200))
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
        disp = cell.get("display_status", cell["status"])
        short = disp
        if disp == "No crop - skip":
            short = "No crop"
        label1 = f"{cell['name']} {short}"
        cv2.putText(out, label1, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label1, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def recommendation_text(cells, result):
    spray = []
    skip = []
    watch = []
    retake = []
    nocrop = []
    for cell in cells:
        name = cell["name"]
        disp = cell.get("display_status", cell["status"])
        if disp == "Spray":
            spray.append(name)
        elif disp == "Skip":
            skip.append(name)
        elif disp == "Watch":
            watch.append(name)
        elif disp == "Retake":
            retake.append(name)
        else:
            nocrop.append(name)
    lines = []
    lines.append("Spray: " + (", ".join(spray) if spray else "none"))
    lines.append("Skip: " + (", ".join(skip) if skip else "none"))
    if watch:
        lines.append("Watch: " + ", ".join(watch))
    if retake:
        lines.append("Retake photo: " + ", ".join(retake))
    if nocrop:
        lines.append("No crop: " + ", ".join(nocrop))
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
        print(f"{c['name']}: veg={c['veg_fraction']} f={c['f']} subcrops={c['subcrops_used']} decision={c['display_status']} reason={c['reason']}")
    spray = sum(1 for c in cells if c["display_status"] == "Spray")
    skip = sum(1 for c in cells if c["display_status"] == "Skip")
    watch = sum(1 for c in cells if c["display_status"] == "Watch")
    retake = sum(1 for c in cells if c["display_status"] == "Retake")
    print(f"\nSpray={spray} Skip={skip} Watch={watch} Retake={retake}")
    print()
    print(recommendation_text(cells, result))
    print("Saved results/field_map.png")
