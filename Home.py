# Home.py — Home / landing
import streamlit as st
from app_utils import inject_css, brand_header
from db import ensure_db, get_current_user_from_state, logout

st.set_page_config(page_title="Home — Rate My", layout="wide")
inject_css()
ensure_db()

# Sidebar — now the sidebar label shows "Home" because the file is Home.py
with st.sidebar:
    st.header("Home")
    user = get_current_user_from_state()
    if user:
        st.success(f"Signed in as {user['email']}")
        if st.button("Log out"):
            logout()
            st.rerun()
    else:
        st.info("Sign up / Log in is in the **Portfolio Tracker** page sidebar.")

# Brand + tagline
brand_header("Rate My")
st.markdown(
    "<div style='text-align:center;color:#9aa0a6;font-size:1.05rem;margin-top:.2rem;'>"
    "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    "</div>",
    unsafe_allow_html=True,
)

# Centered CTAs on one row
col_spacer_left, c1, c2, c3, col_spacer_right = st.columns([1, 1, 1, 1, 1])
with c1:
    st.page_link("pages/1_Rate_My_Stock.py", label="📈 Rate My Stock")
with c2:
    st.page_link("pages/2_Rate_My_Portfolio.py", label="💼 Rate My Portfolio")
with c3:
    st.page_link("pages/3_Portfolio_Tracker.py", label="📊 Portfolio Tracker")

st.divider()
st.caption("Your portfolios are saved to a local SQLite DB when signed in.")
