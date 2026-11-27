"""Recommendation helper: where to spray, how much, and whether to wait."""
import cv2
import numpy as np

import predict
from severity import estimate_severity
import field_sim
from weather_rules import check_weather

LITRES_PER_CELL = 0.5  # assumption


def recommend(img_bgr, temp_c=28.0, humidity_pct=55.0, wind_kmh=8.0, rain_pct=10.0):
    label, conf = predict.predict(img_bgr)
    if label == "retake":
        return {"status": "retake", "disease": "retake", "confidence": conf,
                "severity": None, "grid": None, "sections_to_spray": [],
                "litres": None, "seconds": None, "weather": None, "advice": None}
    sev = estimate_severity(img_bgr)
    model, class_names = predict.load()
    ci = class_names.index(label)
    hm, _ = predict.gradcam(model, img_bgr, ci)
    grid = predict.grid_decision(hm, conf)

    # sections to spray
    sections = [(r + 1, c + 1) for r in range(3) for c in range(3)
                if grid[r][c] == "spray"]

    # build a 3x3 field grid for field_sim (spray cells -> infected Moderate)
    field_grid = []
    for r in range(3):
        row = []
        for c in range(3):
            if grid[r][c] == "spray":
                row.append(("infected", "Moderate"))
            elif grid[r][c] == "retake":
                row.append(("retake", None))
            else:
                row.append(("healthy", None))
        field_grid.append(row)
    out = field_sim.compare(field_grid, litres_per_cell=LITRES_PER_CELL)
    weather = check_weather(temp_c, humidity_pct, wind_kmh, rain_pct)

    return {
        "status": "ok",
        "disease": label,
        "confidence": conf,
        "severity": sev,
        "grid": grid,
        "sections_to_spray": sections,
        "litres_mode_a": out["A"]["litres"],
        "litres_mode_b": out["B"]["litres"],
        "seconds_mode_a": out["A"]["seconds"],
        "seconds_mode_b": out["B"]["seconds"],
        "litres_saved_pct": out["litres_saved_pct"],
        "weather": weather,
    }
