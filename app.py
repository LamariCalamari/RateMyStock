# app.py — Home page (styled title + centered boxes that actually contain the labels)

import streamlit as st
from app_utils import inject_css_and_script, brand_header

st.set_page_config(page_title="Rate My", layout="wide")

# global css + sidebar “Home” rename
inject_css_and_script()

# Brand header (logo sits next to title and is centered)
brand_header("Rate My")

# Tagline
st.markdown(
    "<div style='text-align:center;color:#9aa0a6;font-size:1.15rem;margin:.1rem 0 1.2rem;'>"
    "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    "</div>",
    unsafe_allow_html=True,
)

# ----- Centered CTA row: each label is inside the gradient box -----
c1, c2, c3 = st.columns([1,1,1], gap="large")

with c1:
    st.markdown('<div class="cta">', unsafe_allow_html=True)
    if st.button("📈  Rate My Stock", use_container_width=True, key="cta_stock"):
        st.switch_page("pages/1_Rate_My_Stock.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="cta dark">', unsafe_allow_html=True)
    if st.button("💼  Rate My Portfolio", use_container_width=True, key="cta_portfolio"):
        st.switch_page("pages/2_Rate_My_Portfolio.py")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="cta dark">', unsafe_allow_html=True)
    if st.button("📊  Portfolio Tracker", use_container_width=True, key="cta_tracker"):
        st.switch_page("pages/3_Portfolio_Tracker.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Helpful note
st.markdown(
    "<hr style='opacity:.08'>"
    "<div style='text-align:center;color:#9aa0a6;'>Your portfolios are saved to a local SQLite DB when signed in.</div>",
    unsafe_allow_html=True,
)
