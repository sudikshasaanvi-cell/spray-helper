"""Leaf severity estimation: % of leaf area that looks diseased.

Baseline method (no training needed): find the leaf, then count the pixels
inside it that are NOT healthy green. Works best on single-leaf photos with a
plain background (PlantVillage style). Field photos will be noisier.

Cut-offs follow Horsfall-Barratt class boundaries (12% and 50% of leaf area).
Grouping those classes into 3 levels is OUR simplification - say so in review.
"""
import cv2
import numpy as np

MILD_MAX = 12.0      # % leaf area (Horsfall-Barratt class boundary)
MODERATE_MAX = 50.0  # above this = Severe


def leaf_mask(img_bgr):
    """Binary mask of the leaf: Otsu threshold on saturation, biggest blob, holes filled."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, m = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def lesion_mask(img_bgr, leaf):
    """Pixels inside the leaf that are not healthy green (OpenCV hue 35-85)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    healthy = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    les = cv2.bitwise_and(leaf, cv2.bitwise_not(healthy))
    return cv2.morphologyEx(les, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def level_from_pct(pct):
    if pct <= MILD_MAX:
        return "Mild"
    return "Moderate" if pct <= MODERATE_MAX else "Severe"


def estimate_severity(img_bgr):
    leaf = leaf_mask(img_bgr)
    leaf_px = int(np.count_nonzero(leaf))
    if leaf_px < 500:  # nothing leaf-like found -> do not guess
        return {"pct": None, "level": "Retake photo", "overlay": img_bgr}
    les = lesion_mask(img_bgr, leaf)
    pct = 100.0 * np.count_nonzero(les) / leaf_px
    overlay = img_bgr.copy()
    overlay[les > 0] = (0, 0, 255)  # lesions in red
    overlay = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)
    cv2.drawContours(overlay, cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 0, 0), 2)
    return {"pct": round(pct, 1), "level": level_from_pct(pct), "overlay": overlay}


if __name__ == "__main__":  # python severity.py photo.jpg
    import sys
    r = estimate_severity(cv2.imread(sys.argv[1]))
    print(r["pct"], r["level"])
    cv2.imwrite("severity_overlay.png", r["overlay"])
