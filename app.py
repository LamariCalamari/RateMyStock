# Home.py — Home / landing
import streamlit as st
from app_utils import inject_css, brand_header

st.set_page_config(page_title="Home — Rate My", layout="wide")
inject_css()

# Sidebar — force a proper "Home" label (the native 'app' pill is visually overridden via CSS)
with st.sidebar:
    st.header("Home")
    st.info("Sign up / Log in is in the **Portfolio Tracker** page sidebar.")

# Brand header (logo + gradient wordmark)
brand_header("Rate My")

# Tagline
st.markdown(
    "<p style='text-align:center;color:#a8b0b8;font-size:1.05rem;margin:.25rem 0 1.2rem'>"
    "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    "</p>",
    unsafe_allow_html=True,
)

# Three pretty, boxed CTAs — centered and on one row
col_l, col1, col2, col3, col_r = st.columns([1, 1, 1, 1, 1], gap="large")
with col1:
    st.markdown('<div class="cta-box primary">', unsafe_allow_html=True)
    st.page_link("pages/1_Rate_My_Stock.py", label="📈 Rate My Stock")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="cta-box">', unsafe_allow_html=True)
    st.page_link("pages/2_Rate_My_Portfolio.py", label="💼 Rate My Portfolio")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown('<div class="cta-box">', unsafe_allow_html=True)
    st.page_link("pages/3_Portfolio_Tracker.py", label="📊 Portfolio Tracker")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption("Your portfolios are saved to a local SQLite DB when signed in.")
