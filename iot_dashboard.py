"""AgriShield AI - Live IoT Dashboard (integrates with existing AI model)."""
import cv2
import numpy as np
import pandas as pd
import random
import time
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="AgriShield AI", page_icon="🛡️", layout="wide")

# ---- Theme CSS ----
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.agri-header { background: linear-gradient(90deg,#0B3D2E,#1B5E20); color:white; padding:20px; border-radius:12px; margin-bottom:16px; }
.card { background:#FFFFFF; border:1px solid #E0E0E0; border-radius:14px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin:8px 0; }
.metric-big { font-size:28px; font-weight:700; color:#1B5E20; }
.status-active { color:#2E7D32; font-weight:700; }
.status-inactive { color:#B71C1C; font-weight:700; }
.sidebar .sidebar-content { background:#0B3D2E; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='agri-header'><h1 style='margin:0;'>🛡️ AgriShield AI</h1>"
            "<p style='margin:0;opacity:0.9;'>Smarter Detection. Precise Spraying. Sustainable Farming.</p></div>",
            unsafe_allow_html=True)

import predict
import rag
import config

@st.cache_resource
def load_model():
    m, names = predict.load()
    return m, names

model, class_names = load_model()

# ---- session state ----
if "history" not in st.session_state:
    st.session_state["history"] = []
if "spray_count" not in st.session_state:
    st.session_state["spray_count"] = 0
if "last_spray" not in st.session_state:
    st.session_state["last_spray"] = None
if "pump_active" not in st.session_state:
    st.session_state["pump_active"] = False
if "auto_spray" not in st.session_state:
    st.session_state["auto_spray"] = True
if "threshold" not in st.session_state:
    st.session_state["threshold"] = config.CONF_MIN
if "notifications" not in st.session_state:
    st.session_state["notifications"] = []
if "sensor_history" not in st.session_state:
    st.session_state["sensor_history"] = []

# ---- sidebar nav ----
page = st.sidebar.radio("Navigation", ["Dashboard", "Live Monitoring", "Detections", "Analytics", "Notifications"])

# ---- simulated sensors ----
def new_reading():
    rng = random.Random(int(time.time() * 1000))
    return {
        "time": time.strftime("%H:%M:%S"),
        "temp": round(rng.uniform(22, 34), 1),
        "humidity": round(rng.uniform(30, 80), 1),
        "soil": round(rng.uniform(20, 70), 1),
        "tank": round(rng.uniform(40, 95), 1),
        "battery": round(rng.uniform(50, 100), 1),
        "connected": rng.random() > 0.05,
    }

reading = new_reading()
st.session_state["sensor_history"].append(reading)
st.session_state["sensor_history"] = st.session_state["sensor_history"][-30:]

# notifications for low resources
if reading["tank"] < 20:
    st.session_state["notifications"].append(f"{reading['time']}: Low water tank level ({reading['tank']}%)")
if reading["battery"] < 25:
    st.session_state["notifications"].append(f"{reading['time']}: Low battery ({reading['battery']}%)")
if not reading["connected"]:
    st.session_state["notifications"].append(f"{reading['time']}: Hardware connection lost")

# ---- helper to process an image ----
def process_image(img, ref="upload"):
    try:
        label, conf = predict.predict(img)
    except Exception:
        label, conf = "retake", 0.0
    healthy = "healthy" in str(label).lower()
    status = "Healthy" if healthy else ("Infected" if label != "retake" else "Retake")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    rec = {
        "timestamp": ts,
        "plant_id": f"P{len(st.session_state['history'])+1:03d}",
        "image_ref": ref,
        "disease": label,
        "confidence": round(float(conf), 3),
        "status": status,
        "sprayed": False,
    }
    # spraying logic
    if (not healthy) and label != "retake" and conf >= st.session_state["threshold"]:
        if st.session_state["auto_spray"] and st.session_state["pump_active"] is False:
            # simulate pump activation
            st.session_state["pump_active"] = True
            st.session_state["spray_count"] += 1
            st.session_state["last_spray"] = ts
            rec["sprayed"] = True
            st.session_state["notifications"].append(f"{ts}: Infected plant detected ({label}) - spray activated")
        else:
            rec["sprayed"] = False
    # turn pump off after (simulated)
    st.session_state["pump_active"] = False
    st.session_state["history"].append(rec)
    return rec

# ---- Pages ----
if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    total = len(st.session_state["history"])
    healthy_n = sum(1 for r in st.session_state["history"] if r["status"] == "Healthy")
    infected_n = sum(1 for r in st.session_state["history"] if r["status"] == "Infected")
    col1.metric("Total plants analysed", total)
    col2.metric("Healthy", healthy_n)
    col3.metric("Infected", infected_n)
    col4.metric("Sprays activated", st.session_state["spray_count"])
    st.markdown("---")
    # upload
    st.subheader("Upload / capture plant image")
    up = st.file_uploader("Upload plant photo", type=["jpg", "jpeg", "png"], key="iot_up")
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Latest captured image", width=300)
            rec = process_image(img, ref=up.name)
            st.success(f"Predicted: {rec['disease']} ({rec['confidence']*100:.1f}%) - {rec['status']}")
            if rec["sprayed"]:
                st.warning("Selective spray activated.")
    st.markdown("---")
    st.subheader("Live sensors")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Temp", f"{reading['temp']} C")
    c2.metric("Humidity", f"{reading['humidity']}%")
    c3.metric("Soil moisture", f"{reading['soil']}%")
    c4.metric("Tank level", f"{reading['tank']}%")
    c5.metric("Battery", f"{reading['battery']}%")
    c6.metric("Connectivity", "Online" if reading["connected"] else "Offline")
    st.caption("Simulated readings (hardware not connected).")

elif page == "Live Monitoring":
    st.subheader("Live IoT monitoring")
    hist = pd.DataFrame(st.session_state["sensor_history"])
    if not hist.empty:
        st.line_chart(hist[["temp", "humidity", "soil"]])
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Water tank level", f"{reading['tank']}%")
        st.metric("Battery level", f"{reading['battery']}%")
    with c2:
        st.metric("System connectivity", "Online" if reading["connected"] else "Offline")
        st.metric("Pump status", "ACTIVE" if st.session_state["pump_active"] else "INACTIVE")
    st.caption("Simulated readings when hardware is not connected.")

elif page == "Detections":
    st.subheader("Detection history")
    if st.session_state["history"]:
        df = pd.DataFrame(st.session_state["history"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No detections yet.")

elif page == "Analytics":
    st.subheader("Analytics")
    hist = st.session_state["history"]
    total = len(hist)
    healthy_n = sum(1 for r in hist if r["status"] == "Healthy")
    infected_n = sum(1 for r in hist if r["status"] == "Infected")
    # pie chart
    fig, ax = plt.subplots()
    ax.pie([healthy_n, infected_n, total - healthy_n - infected_n],
           labels=["Healthy", "Infected", "Retake"], autopct="%1.0f%%",
           colors=["#2E7D32", "#B71C1C", "#F57F17"])
    st.pyplot(fig)
    st.write(f"Total plants: {total}")
    st.write(f"Selective sprays: {st.session_state['spray_count']}")
    # traditional vs selective comparison
    trad = infected_n * 1.0  # assume 1 L per plant traditional
    sel = st.session_state["spray_count"] * 1.0
    st.write(f"Traditional pesticide usage (est.): {trad:.1f} L")
    st.write(f"Selective pesticide usage (est.): {sel:.1f} L")
    saved = max(trad - sel, 0)
    st.write(f"Estimated pesticide saved: {saved:.1f} L")
    st.caption("Estimates are illustrative (assumptions).")

elif page == "Notifications":
    st.subheader("Live notifications")
    for n in reversed(st.session_state["notifications"][-20:]):
        st.warning(n)
    if not st.session_state["notifications"]:
        st.info("No notifications.")

# ---- Smart spraying control (sidebar) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Smart spraying control")
st.session_state["auto_spray"] = st.sidebar.checkbox("Automatic spraying", value=st.session_state["auto_spray"])
st.session_state["threshold"] = st.sidebar.slider("Confidence threshold", 0.5, 0.95, st.session_state["threshold"], 0.05)
if st.sidebar.button("Manual spray (test)"):
    st.session_state["pump_active"] = True
    st.session_state["spray_count"] += 1
    st.session_state["last_spray"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["notifications"].append(f"{st.session_state['last_spray']}: Manual spray activated")
    st.session_state["pump_active"] = False
st.sidebar.write(f"Pump status: {'ACTIVE' if st.session_state['pump_active'] else 'INACTIVE'}")
st.sidebar.write(f"Spray count: {st.session_state['spray_count']}")
st.sidebar.write(f"Last spray: {st.session_state['last_spray'] or 'none'}")

# auto refresh
try:
    st.autorefresh(interval=5000, key="iot_refresh")
except Exception:
    pass
