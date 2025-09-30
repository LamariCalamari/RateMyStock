# app.py — Home / landing
import streamlit as st
from app_utils import inject_css_and_script, brand_header

st.set_page_config(page_title="Home — Rate My", layout="wide")
inject_css_and_script()  # CSS + MutationObserver ("app" -> "Home")

# ----- Sidebar -----
with st.sidebar:
    st.header("Home")
    st.info("Sign up / Log in is in the **Portfolio Tracker** page sidebar.")

# ----- Hero -----
brand_header("Rate My")
st.markdown(
    "<p style='text-align:center;color:#a8b0b8;font-size:1.06rem;margin:.35rem 0 1.2rem'>"
    "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    "</p>",
    unsafe_allow_html=True,
)

# ----- Three boxed CTAs in one centered row -----
left, c1, c2, c3, right = st.columns([1, 1.15, 1.15, 1.15, 1], gap="large")
with c1:
    st.page_link("pages/1_Rate_My_Stock.py", label="📈 Rate My Stock")
with c2:
    st.page_link("pages/2_Rate_My_Portfolio.py", label="💼 Rate My Portfolio")
with c3:
    st.page_link("pages/3_Portfolio_Tracker.py", label="📊 Portfolio Tracker")

st.divider()
st.caption("Your portfolios are saved to a local SQLite DB when signed in.")
