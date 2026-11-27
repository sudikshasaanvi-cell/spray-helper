"""Streamlit demo:  streamlit run app.py
NOTE: written but not run in the build sandbox (Streamlit was not installed there).
Run it once yourself and fix anything it reports before the review.
"""
import cv2
import numpy as np
import streamlit as st
from severity import estimate_severity
from field_sim import compare, draw
from weather_rules import check_weather

st.title("Smart Spray Helper (tomato)")
tab1, tab2, tab3 = st.tabs(["1. How bad is the leaf?", "2. Virtual field", "3. Weather check"])

with tab1:
    up = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        r = estimate_severity(img)
        if r["pct"] is None:
            st.warning("Could not find a leaf. Please retake the photo.")
        else:
            st.metric("Leaf area affected", f"{r['pct']}%", r["level"])
        st.image(cv2.cvtColor(r["overlay"], cv2.COLOR_BGR2RGB), caption="Red = affected area, blue line = leaf outline")
        st.caption("Estimate from colour only. Severity levels are a simplified grouping of Horsfall-Barratt classes.")

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
