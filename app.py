import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Spray Helper", page_icon="🌿", layout="wide")

# ---- CSS ----
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
html, body, [class*="css"] { font-size: 18px; }
.stButton>button { font-size: 20px; padding: 12px 24px; border-radius: 12px; }
.card { background: #F1F8E9; border-radius: 16px; padding: 20px; margin: 10px 0; }
.banner { border-radius: 16px; padding: 24px; font-size: 26px; font-weight: 700; text-align: center; margin: 10px 0; }
.banner-green { background: #C8E6C9; color: #1B5E20; }
.banner-red { background: #FFCDD2; color: #B71C1C; }
.banner-yellow { background: #FFF9C4; color: #F57F17; }
.metric-card { background: #FFFFFF; border: 2px solid #2E7D32; border-radius: 14px; padding: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

from severity import estimate_severity
import field_sim
from field_sim import compare, draw
from weather_rules import check_weather
import predict
import rag
import recommend
import field_map
import config

st.markdown("<h1 style='font-size:42px;'>🌿 Smart Spray Helper</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:22px; color:#2E7D32;'><b>Take a photo. Know where to spray.</b></p>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    m, names = predict.load()
    rag.build_index()
    return m, names


model, class_names = load_model()

tabW, tabS, tabWeather = st.tabs(["🌾 Whole field", "🍃 Single leaf", "⛅ Weather"])


def _clean(name):
    if not name:
        return ""
    return name.split("___")[-1].replace("_", " ").strip()


# ---------- Whole field ----------
with tabW:
    left, right = st.columns([0.4, 0.6])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        up = st.file_uploader("Upload a field photo", type=["jpg", "jpeg", "png"], key="wf")
        st.caption("Take a close-up photo of your plants (about arm's length)")
        if st.button("Try an example"):
            try:
                with open("samples/field_mosaic.png", "rb") as fh:
                    st.session_state["wf_bytes"] = fh.read()
            except Exception:
                st.warning("Could not load example photo.")
        if "wf_bytes" in st.session_state:
            up = None
            img_bytes = st.session_state["wf_bytes"]
        else:
            img_bytes = up.read() if up is not None else None
        with st.expander("Grid size (advanced)"):
            rows = st.slider("Rows", 1, 6, 3, key="wrows")
            cols = st.slider("Cols", 1, 6, 3, key="wcols")
        with st.expander("Simulation settings (advanced)"):
            sm = st.slider("Seconds to move across one section", 5, 30, 10, key="wsm")
            ss = st.slider("Extra seconds to spray one section", 5, 30, 15, key="wss")
            lp = st.slider("Litres per section (full dose)", 0.1, 2.0, 0.5, key="wlp")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if img_bytes is not None:
            with st.spinner("Checking your plants..."):
                try:
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("bad image")
                    grid, cells, _ = field_map.analyze_field(img, rows, cols)
                    result = field_sim.compare(grid, sec_move=sm, sec_spray=ss, litres_per_cell=lp)
                except Exception:
                    st.error("Could not read this photo. Please try another.")
                    st.stop()
            # 1. banner
            n_cells = rows * cols
            n_spray = sum(1 for c in cells if c["status"] == "infected")
            n_retake = sum(1 for c in cells if c["status"] == "retake")
            if n_spray == 0 and n_retake == 0:
                st.markdown("<div class='banner banner-green'>No spraying needed</div>", unsafe_allow_html=True)
            elif n_retake > 0:
                st.markdown(f"<div class='banner banner-yellow'>Retake {n_retake} photo(s)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='banner banner-red'>Spray {n_spray} of {n_cells} sections</div>", unsafe_allow_html=True)
            # 2. overlay
            overlay = field_map.draw_field_map(img, cells, rows, cols)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
            # 3. metric cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-card'><b>Pesticide saved</b><br><span style='font-size:32px;color:#2E7D32;'>{result['litres_saved_pct']}%</span></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-card'><b>Litres (A vs B)</b><br><span style='font-size:22px;'>{result['A']['litres']} L → {result['B']['litres']} L</span></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-card'><b>Time saved</b><br><span style='font-size:32px;color:#2E7D32;'>{result['time_saved_pct']}%</span></div>", unsafe_allow_html=True)
            # 4. plain list
            st.markdown(field_map.recommendation_text(cells, result).replace("\n", "<br>"), unsafe_allow_html=True)
            # 5. route figure radio
            route_choice = st.radio("Show route", ["Spray everything", "AI-targeted"], horizontal=True, key="route_choice")
            mode_key = "A" if route_choice == "Spray everything" else "B"
            fig = field_sim.draw(grid, result[mode_key],
                                 f"{'Mode A' if mode_key=='A' else 'Mode B'}: {result[mode_key]['litres']} L, {result[mode_key]['seconds']} s")
            st.pyplot(fig)
            # 6. advice card
            # find dominant disease
            diseases = [c["class"] for c in cells if c["status"] == "infected" and c["class"]]
            dom = max(set(diseases), key=diseases.count) if diseases else None
            if dom:
                advice = rag.advice_for(dom)
                if advice is None:
                    st.info("Ask your agriculture officer.")
                else:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"**{dom}**")
                    for ch in advice:
                        st.markdown(ch["text"][:400])
                        st.markdown(f"Source: [{ch['source']}]({ch['source']})")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Ask your agriculture officer.")
            # 7. details expander
            with st.expander("Details"):
                df = pd.DataFrame([{
                    "cell": c["name"],
                    "disease": _clean(c["class"]) if c["class"] else c["status"],
                    "confidence": c["confidence"],
                    "severity %": c["severity"],
                    "action": c["status"],
                } for c in cells])
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Upload a photo or try the example to see results.")


# ---------- Single leaf ----------
with tabS:
    up = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"], key="sl")
    if up:
        with st.spinner("Checking your plant..."):
            try:
                img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("bad")
                label, conf = predict.predict(img)
            except Exception:
                st.error("Could not read this photo. Please try another.")
                st.stop()
        st.write(f"**Disease:** {label}  |  **Confidence:** {conf:.2f}")
        if label == "retake":
            st.warning("Low confidence. Please retake the photo.")
        else:
            sev = estimate_severity(img)
            if sev["pct"] is None:
                st.warning("Could not find a leaf. Retake photo.")
            else:
                st.metric("Leaf area affected", f"{sev['pct']}%", sev["level"])
            st.image(cv2.cvtColor(sev["overlay"], cv2.COLOR_BGR2RGB), caption="Severity overlay")
            ci = class_names.index(label)
            hm, ov = predict.gradcam(model, img, ci)
            st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Grad-CAM")
            grid = predict.grid_decision(hm, conf)
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
            resB = field_sim.simulate(field_grid, "B")
            fig = field_sim.draw(field_grid, resB, "AI-targeted spray grid")
            st.pyplot(fig)
            advice = rag.advice_for(label)
            if advice is None:
                st.info("Ask your agriculture officer.")
            else:
                for ch in advice:
                    st.markdown(ch["text"])
                    st.markdown(f"Source: [{ch['source']}]({ch['source']})")


# ---------- Weather ----------
with tabWeather:
    t = st.number_input("Temperature (C)", value=28.0)
    h = st.number_input("Humidity (%)", value=55.0)
    w = st.number_input("Wind speed (km/h)", value=8.0)
    rn = st.number_input("Rain chance (%)", value=10.0)
    res = check_weather(t, h, w, rn)
    st.header(res["decision"])
    for rule in res["why"]:
        st.write(f"- {rule['text']}  Source: [{rule['source']}]({rule['url']})")
    for n in res["notes"]:
        st.info(n)
