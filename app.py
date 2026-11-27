"""Streamlit demo: streamlit run app.py"""
import cv2
import numpy as np
import streamlit as st

from severity import estimate_severity
from field_sim import compare, draw
from weather_rules import check_weather
import predict
import rag
import recommend
import config

st.title("Smart Spray Helper (tomato)")


@st.cache_resource
def load_model():
    m, names = predict.load()
    rag.build_index()
    return m, names


model, class_names = load_model()

tab0, tabR, tab1, tab2, tab3 = st.tabs(
    ["Diagnose", "Recommendation", "1. How bad is the leaf?", "2. Virtual field", "3. Weather check"])

# ---------- Diagnose ----------
with tab0:
    up = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        label, conf = predict.predict(img)
        st.write(f"**Disease:** {label}  |  **Confidence:** {conf:.2f}")
        if label == "retake":
            st.warning("Low confidence. Please retake the photo.")
            st.stop()
        # severity
        sev = estimate_severity(img)
        if sev["pct"] is None:
            st.warning("Could not find a leaf for severity. Retake photo.")
        else:
            st.metric("Leaf area affected", f"{sev['pct']}%", sev["level"])
        st.image(cv2.cvtColor(sev["overlay"], cv2.COLOR_BGR2RGB),
                 caption="Red = affected area, blue line = leaf outline")
        # gradcam
        ci = class_names.index(label)
        hm, ov = predict.gradcam(model, img, ci)
        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Grad-CAM overlay")
        # spray grid (rendered as image)
        grid = predict.grid_decision(hm, conf)
        st.write("3x3 spray grid:")
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
        # advice
        advice = rag.advice_for(label)
        if advice is None:
            st.info("Ask your agriculture officer.")
        else:
            for ch in advice:
                st.markdown(ch["text"])
                src = ch["source"]
                st.markdown(f"Source: [{src}]({src})")

# ---------- Recommendation ----------
with tabR:
    up = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"], key="rec")
    c1, c2, c3, c4 = st.columns(4)
    t = c1.number_input("Temp (C)", value=28.0, key="rt")
    h = c2.number_input("Humidity (%)", value=55.0, key="rh")
    w = c3.number_input("Wind (km/h)", value=8.0, key="rw")
    rn = c4.number_input("Rain (%)", value=10.0, key="rr")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        r = recommend.recommend(img, t, h, w, rn)
        if r["status"] == "retake":
            st.warning("Low confidence. Retake the photo.")
        else:
            st.write(f"**Disease:** {r['disease']}  (conf {r['confidence']:.2f})")
            st.write(f"**Severity:** {r['severity']['level']} ({r['severity']['pct']}%)")
            st.write("**Spray these sections (row, col):**", r["sections_to_spray"])
            field_grid = []
            for row in r["grid"]:
                rr = []
                for cell in row:
                    if cell == "spray":
                        rr.append(("infected", "Moderate"))
                    elif cell == "retake":
                        rr.append(("retake", None))
                    else:
                        rr.append(("healthy", None))
                field_grid.append(rr)
            resB = field_sim.simulate(field_grid, "B")
            fig = field_sim.draw(field_grid, resB, "AI-targeted spray grid")
            st.pyplot(fig)
            st.write(f"Mode A (spray all): {r['litres_mode_a']} L, {r['seconds_mode_a']} s")
            st.write(f"Mode B (targeted): {r['litres_mode_b']} L, {r['seconds_mode_b']} s  (saved {r['litres_saved_pct']}%)")
            st.subheader(r["weather"]["decision"])
            for rule in r["weather"]["why"]:
                st.write(f"- {rule['text']}")
            advice = rag.advice_for(r["disease"])
            if advice is None:
                st.info("Ask your agriculture officer.")
            else:
                for ch in advice:
                    st.markdown(ch["text"])
                    st.markdown(f"Source: [{ch['source']}]({ch['source']})")

# ---------- Severity tab (existing) ----------
with tab1:
    up = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"], key="sev")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        r = estimate_severity(img)
        if r["pct"] is None:
            st.warning("Could not find a leaf. Please retake the photo.")
        else:
            st.metric("Leaf area affected", f"{r['pct']}%", r["level"])
        st.image(cv2.cvtColor(r["overlay"], cv2.COLOR_BGR2RGB), caption="Red = affected area, blue line = leaf outline")
        st.caption("Estimate from colour only. Severity levels are a simplified grouping of Horsfall-Barratt classes.")

# ---------- Virtual field (existing) ----------
with tab2:
    st.write("Set each of the 9 field sections:")
    opts = ["healthy", "infected-Mild", "infected-Moderate", "infected-Severe", "retake"]
    grid = []
    for r_ in range(3):
        cols, row = st.columns(3), []
        for c_ in range(3):
            v = cols[c_].selectbox(f"Row {r_+1}, Col {c_+1}", opts, key=f"c{r_}{c_}")
            row.append(("infected", v.split("-")[1]) if v.startswith("infected") else (v, None))
        grid.append(row)
    st.subheader("Assumptions (simulation only)")
    sm = st.slider("Seconds to move across one section", 5, 30, 10)
    ss = st.slider("Extra seconds to spray one section", 5, 30, 15)
    lp = st.slider("Litres per section (full dose)", 0.1, 2.0, 0.5)
    out = compare(grid, sec_move=sm, sec_spray=ss, litres_per_cell=lp)
    a, b = st.columns(2)
    a.pyplot(draw(grid, out["A"], f"Mode A: spray all - {out['A']['litres']} L, {out['A']['seconds']} s"))
    b.pyplot(draw(grid, out["B"], f"Mode B: AI-targeted - {out['B']['litres']} L, {out['B']['seconds']} s"))
    st.success(f"Pesticide saved: {out['litres_saved_pct']}%  |  Time saved: {out['time_saved_pct']}%")
    if out["B"]["retake_cells"]:
        st.warning(f"Retake photos for sections: {[(r+1, c+1) for r, c in out['B']['retake_cells']]}")

# ---------- Weather (existing) ----------
with tab3:
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
