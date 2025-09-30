# Home.py — Home / landing
import streamlit as st
from app_utils import inject_css, brand_header

st.set_page_config(page_title="Home — Rate My", layout="wide")
inject_css()

# Sidebar
with st.sidebar:
    st.header("Home")  # explicit header; the tiny "app" pill is hidden via CSS
    st.info("Sign up / Log in is in the **Portfolio Tracker** page sidebar.")

# Brand + tagline
brand_header("Rate My")

st.markdown(
    "<p style='text-align:center;color:#a8b0b8;font-size:1.05rem;margin:.4rem 0 1.2rem'>"
    "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    "</p>",
    unsafe_allow_html=True,
)

# Centered, pretty CTAs on one row
st.markdown('<div id="ctas" class="cta-row">', unsafe_allow_html=True)
st.page_link("pages/1_Rate_My_Stock.py", label="📈 Rate My Stock")
st.page_link("pages/2_Rate_My_Portfolio.py", label="💼 Rate My Portfolio")
st.page_link("pages/3_Portfolio_Tracker.py", label="📊 Portfolio Tracker")
st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption("Your portfolios are saved to a local SQLite DB when signed in.")
