"""Model loading, prediction, Grad-CAM and spray-grid decision."""
import json, os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import config

_cached_model = None
_cached_names = None

CONF_MIN = config.CONF_MIN
HEAT_THRESH = config.HEAT_THRESH
IMG_SIZE = config.IMG_SIZE


def load():
    global _cached_model, _cached_names
    if _cached_model is None:
        _cached_model = tf.keras.models.load_model("model/tomato.keras")
        with open("model/class_names.json") as f:
            _cached_names = json.load(f)
    return _cached_model, _cached_names


def _prep(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(resized.astype("float32"), 0)
    return arr, rgb, resized


def predict(img_bgr):
    """Return (class_name, confidence). Low confidence -> 'retake'."""
    model, class_names = load()
    arr, _, _ = _prep(img_bgr)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    if conf < CONF_MIN:
        return "retake", conf
    return class_names[idx], conf


def gradcam(model, img_bgr, class_index):
    """Return (heatmap 0..1 at image size, overlay BGR)."""
    arr, rgb, resized = _prep(img_bgr)
    base = model.get_layer("mobilenetv2_1.00_160")
    conv_layer_name = "out_relu"
    try:
        base.get_layer(conv_layer_name)
    except ValueError:
        print("Layer names:", [l.name for l in base.layers])
        conv_layer_name = base.layers[-1].name
    last_conv = base.get_layer(conv_layer_name)

    # base sub-model that returns conv activations + base output
    base_conv = tf.keras.Model(inputs=base.input,
                               outputs=[last_conv.output, base.output])
    gap = model.get_layer("global_average_pooling2d")
    drop = model.get_layer("dropout")
    dense = model.get_layer("dense")

    arr_pre = preprocess_input(arr.copy())
    with tf.GradientTape() as tape:
        conv_out, base_out = base_conv(arr_pre)
        x = gap(base_out)
        x = drop(x, training=False)
        preds = dense(x)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    mx = tf.reduce_max(heatmap)
    if mx > 0:
        heatmap = heatmap / mx
    heatmap = heatmap.numpy()
    # resize to original image size
    h, w = img_bgr.shape[:2]
    heatmap_r = cv2.resize(heatmap, (w, h))
    heatmap_r = np.clip(heatmap_r, 0, 1)

    # overlay
    hm_color = cv2.applyColorMap((heatmap_r * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
    return heatmap_r, overlay


def grid_decision(heatmap, confidence, rows=3, cols=3):
    """Return rows x cols list of 'spray'/'skip' (or 'retake' if low confidence)."""
    if confidence < CONF_MIN:
        return [["retake"] * cols for _ in range(rows)]
    h, w = heatmap.shape[:2]
    rh, cw = h // rows, w // cols
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            region = heatmap[r*rh:(r+1)*rh, c*cw:(c+1)*cw]
            mean_heat = float(region.mean()) if region.size else 0.0
            row.append("spray" if mean_heat >= HEAT_THRESH else "skip")
        grid.append(row)
    return grid


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python predict.py image.jpg")
        sys.exit(0)
    img = cv2.imread(sys.argv[1])
    label, conf = predict(img)
    print(label, round(conf, 3))
    if label not in ("retake",):
        # find class index
        model, class_names = load()
        ci = class_names.index(label)
        hm, ov = gradcam(model, img, ci)
        cv2.imwrite("gradcam_test.png", ov)
        print("grid:", grid_decision(hm, conf))
