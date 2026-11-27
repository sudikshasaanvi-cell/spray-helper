import cv2
import numpy as np
import pandas as pd
import streamlit as st
import io

st.set_page_config(page_title="Smart Spray Planner", page_icon="🌿", layout="wide")

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
from weather_rules import check_weather
import predict
import rag
import field_map
import spray_plan
import config

st.markdown("<h1 style='font-size:42px;'>🌿 Smart Spray Planner</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:22px; color:#2E7D32;'><b>Spray only where it is needed.</b></p>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    m, names = predict.load()
    rag.build_index()
    return m, names


model, class_names = load_model()

tabPlan, tabSim, tabWeather, tabDetails = st.tabs(
    ["Spray plan", "Simulation", "Weather", "Plant details"])


def _clean(name):
    if not name:
        return ""
    return name.split("___")[-1].replace("_", " ").strip()


# ---------- Spray plan ----------
with tabPlan:
    left, right = st.columns([0.4, 0.6])
    with left:
        up = st.file_uploader("Upload a field photo", type=["jpg", "jpeg", "png"], key="sp")
        st.caption("Take a close-up photo of your plants (about arm's length)")
        if st.button("Try an example", key="sp_example"):
            try:
                with open("samples/field_mosaic.png", "rb") as fh:
                    st.session_state["sp_bytes"] = fh.read()
            except Exception:
                st.warning("Could not load example photo.")
        if "sp_bytes" in st.session_state:
            img_bytes = st.session_state["sp_bytes"]
        else:
            img_bytes = up.read() if up is not None else None

        with st.expander("Inputs (advanced)"):
            rows = st.slider("Grid rows", 1, 6, 3, key="sp_rows")
            cols = st.slider("Grid cols", 1, 6, 3, key="sp_cols")
            strictness = st.selectbox("Strictness", ["cautious", "balanced", "fewer"], index=1, key="sp_strict")
            buffer = st.checkbox("Spread buffer (spray neighbours of infected)", key="sp_buffer")
            price = st.number_input("Price per litre (optional)", value=0.0, min_value=0.0, key="sp_price")
            st.subheader("Weather")
            t = st.number_input("Temp (C)", value=28.0, key="sp_t")
            h = st.number_input("Humidity (%)", value=55.0, key="sp_h")
            w = st.number_input("Wind (km/h)", value=8.0, key="sp_w")
            rn = st.number_input("Rain (%)", value=10.0, key="sp_rn")

    with right:
        if img_bytes is not None:
            with st.spinner("Checking your plants..."):
                try:
                    arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("bad")
                    grid, cells, _ = field_map.analyze_field(img, rows, cols, strictness=strictness)
                    result = field_sim.compare(grid, buffer=buffer,
                                              sec_move=10, sec_spray=15, litres_per_cell=0.5)
                    weather = {"temp_c": t, "humidity_pct": h, "wind_kmh": w, "rain_pct": rn}
                    plan = spray_plan.build_plan(cells, result, price_per_litre=(price if price > 0 else None), weather=weather)
                except Exception:
                    st.error("Could not read this photo. Please try another.")
                    st.stop()
            # 1. banner + gate
            gate = plan["gate"]
            headline = plan["headline"]
            if "No spraying" in headline:
                banner_class = "banner-green"
            elif "Retake" in headline:
                banner_class = "banner-yellow"
            else:
                banner_class = "banner-red"
            st.markdown(f"<div class='banner {banner_class}'>{headline}</div>", unsafe_allow_html=True)
            st.markdown(f"**Weather gate:** {gate}")
            if plan["gate_reasons"]:
                for r in plan["gate_reasons"]:
                    st.write(f"- {r}")
            # 2. spray map
            overlay = field_map.draw_field_map(img, cells, rows, cols)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
            # 3. metric cards
            tot = plan["totals"]
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-card'><b>Pesticide saved</b><br><span style='font-size:32px;color:#2E7D32;'>{tot['litres_saved'] and round(100*(tot['litres_mode_a']-tot['litres_mode_b'])/tot['litres_mode_a'],1) if tot['litres_mode_a'] else 0}%</span></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-card'><b>Litres (A vs B)</b><br><span style='font-size:22px;'>{tot['litres_mode_a']} → {tot['litres_mode_b']}</span></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-card'><b>Time saved</b><br><span style='font-size:32px;color:#2E7D32;'>{tot['time_saved_pct']}%</span></div>", unsafe_allow_html=True)
            if tot["money_saved"] is not None:
                st.write(f"Money saved: {tot['money_saved']}")
            # 4. route steps
            st.subheader("Route steps")
            for step in plan["steps"]:
                st.write(step)
            st.caption("Use the dose on your label.")
            # 5. Simulation fills automatically from the detected grid
            st.subheader("Simulation (auto-filled from this photo)")
            figA = field_sim.draw(grid, result["A"], f"Mode A: spray all - {result['A']['litres']} L, {result['A']['seconds']} s")
            figB = field_sim.draw(grid, result["B"], f"Mode B: AI-targeted - {result['B']['litres']} L, {result['B']['seconds']} s")
            ca, cb = st.columns(2)
            ca.pyplot(figA)
            cb.pyplot(figB)
            st.success(f"Pesticide saved: {result['litres_saved_pct']}%  |  Time saved: {result['time_saved_pct']}%")
            # 6. download buttons
            df = pd.DataFrame(plan["per_square"])
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button("Download spray_plan.csv", csv_bytes, "spray_plan.csv", "text/csv")
            # map png
            ok, buf = cv2.imencode(".png", overlay)
            if ok:
                st.download_button("Download map PNG", buf.tobytes(), "spray_map.png", "image/png")
            # small caption disease
            diseases = [c["class"] for c in cells if c["status"] == "infected" and c["class"]]
            if diseases:
                dom = max(set(diseases), key=diseases.count)
                sev = [c["severity"] for c in cells if c["status"]=="infected" and c["class"]==dom and c["severity"] is not None]
                pct = sev[0] if sev else ""
                st.caption(f"why: {_clean(dom)}, {pct}%")
        else:
            st.info("Upload a photo or try the example to see results.")


# ---------- Simulation ----------
with tabSim:
    st.write("Set each of the 9 field sections:")
    opts = ["healthy", "infected-Mild", "infected-Moderate", "infected-Severe", "retake"]
    grid = []
    for r_ in range(3):
        cols_, row = st.columns(3), []
        for c_ in range(3):
            v = cols_[c_].selectbox(f"Row {r_+1}, Col {c_+1}", opts, key=f"sim{r_}{c_}")
            row.append(("infected", v.split("-")[1]) if v.startswith("infected") else (v, None))
        grid.append(row)
    st.subheader("Assumptions (simulation only)")
    sm = st.slider("Seconds to move across one section", 5, 30, 10, key="sim_sm")
    ss = st.slider("Extra seconds to spray one section", 5, 30, 15, key="sim_ss")
    lp = st.slider("Litres per section (full dose)", 0.1, 2.0, 0.5, key="sim_lp")
    buf = st.checkbox("Spread buffer", key="sim_buf")
    out = field_sim.compare(grid, sec_move=sm, sec_spray=ss, litres_per_cell=lp, buffer=buf)
    choice = st.radio("Show route", ["Spray everything", "AI-targeted"], horizontal=True, key="sim_route")
    mk = "A" if choice == "Spray everything" else "B"
    fig = field_sim.draw(grid, out[mk], f"{'Mode A' if mk=='A' else 'Mode B'}: {out[mk]['litres']} L, {out[mk]['seconds']} s")
    st.pyplot(fig)
    st.success(f"Pesticide saved: {out['litres_saved_pct']}%  |  Time saved: {out['time_saved_pct']}%")
    if out["B"]["retake_cells"]:
        st.warning(f"Retake photos for sections: {[(r+1, c+1) for r, c in out['B']['retake_cells']]}")
    if out.get("buffer_cells"):
        st.info(f"Buffer added: {out['buffer_cells']}")


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


# ---------- Plant details ----------
with tabDetails:
    up = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"], key="pd")
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
            # per-square table for a single leaf (one cell)
            df = pd.DataFrame([{
                "cell": "A1",
                "disease": _clean(label),
                "confidence": round(float(conf), 3),
                "severity %": sev["pct"],
                "action": "Spray" if label != "retake" and "healthy" not in label.lower() else ("Skip" if "healthy" in label.lower() else "Retake"),
            }])
            st.dataframe(df, use_container_width=True)
