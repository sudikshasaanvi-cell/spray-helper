# Project: Smart Spray Helper (tomato)

Goal: a tomato-leaf photo tool that shows the disease, how bad it is (% leaf area),
which parts of a field to spray, pesticide saved vs spraying everything, and
official advice with its source.

## Stack
Python, TensorFlow/Keras (MobileNetV2 transfer learning), OpenCV, Qdrant (local mode,
hybrid search: dense + BM25, merged with RRF), Streamlit. Training runs in Google Colab.

## Files that ALREADY exist (reuse them, do NOT rewrite them)
- severity.py: estimate_severity(img_bgr) -> {"pct", "level", "overlay"}
- field_sim.py: compare(grid), draw(grid, result, title). A grid cell is
  ("infected", "Mild"|"Moderate"|"Severe"), ("healthy", None) or ("retake", None)
- weather_rules.py: check_weather(temp_c, humidity_pct, wind_kmh, rain_pct)
- app.py: Streamlit app with 3 tabs (severity, virtual field, weather)

## Rules
1. Work on ONE module at a time. After writing it, run it and fix errors before moving on.
2. Keep code simple, with short plain-English comments a student can explain.
3. NEVER invent pesticide names, doses or advice. Advice text must come only from
   the files in docs/, shown with its source link. No match -> "Ask your agriculture officer."
4. Low model confidence means "retake photo", never a guess.
5. Any number that is an assumption (thresholds, litres, seconds) must be a named
   constant with a comment saying it is an assumption.
6. Do not add features that were not asked for.
7. Everything lives in this one folder (next to severity.py).

## Folders
- model/ (saved model + class_names.json)
- docs/  (advisory .txt files; line 1 of each file is the source URL)
- samples/ (test photos)
