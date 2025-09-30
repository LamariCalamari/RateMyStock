# app.py
import streamlit as st
from app_utils import inject_css, brand_header, inline_logo_svg
from db import ensure_db, get_current_user_from_state, logout

st.set_page_config(page_title="Rate My", layout="wide")
inject_css()
ensure_db()

# Sidebar auth status
with st.sidebar:
    user = get_current_user_from_state()
    if user:
        st.success(f"Signed in as {user['email']}")
        if st.button("Log out"):
            logout()
            st.rerun()
    else:
        st.info("Sign up / Log in appears in the **Portfolio Tracker** sidebar.")

brand_header("Rate My")

st.markdown(
    "<div style='text-align:center;color:#9aa0a6;font-size:1.05rem;margin-top:.2rem;'>"
    "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    "</div>",
    unsafe_allow_html=True,
)

colL, c1, c2, c3, colR = st.columns([1,2,2,2,1])
with c1: st.page_link("pages/1_Rate_My_Stock.py", label="📈 Rate My Stock", use_container_width=True)
with c2: st.page_link("pages/2_Rate_My_Portfolio.py", label="💼 Rate My Portfolio", use_container_width=True)
with c3: st.page_link("pages/3_Portfolio_Tracker.py", label="📊 Portfolio Tracker", use_container_width=True)

st.divider()
st.caption("Your portfolios are saved to a local SQLite DB when signed in.")
