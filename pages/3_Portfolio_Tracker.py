# pages/3_Portfolio_Tracker.py
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from db import ensure_db, get_current_user_from_state, signup, login, list_portfolios, load_holdings, upsert_portfolio, delete_portfolio
from app_utils import (
    inject_css, brand_header, yf_symbol, fetch_prices_chunked_with_fallback,
    technical_scores, fetch_fundamentals_simple, zscore_series,
    fetch_vix_series, fetch_gold_series, fetch_dxy_series, fetch_tnx_series,
    fetch_credit_ratio_series, macro_from_signals, fetch_sector
)

st.set_page_config(page_title="Portfolio Tracker", layout="wide")
inject_css()
ensure_db()
brand_header("Portfolio Tracker")
st.caption("Track positions, get a live portfolio score, and review allocation risk.")

# Sidebar auth
with st.sidebar:
    st.subheader("Account")
    user = get_current_user_from_state()
    if not user:
        tabs = st.tabs(["Log in", "Sign up"])
        with tabs[0]:
            e = st.text_input("Email", key="login_email")
            p = st.text_input("Password", type="password", key="login_pw")
            if st.button("Log in", type="primary"):
                ok, msg = login(e, p); st.success(msg) if ok else st.error(msg); st.rerun()
        with tabs[1]:
            e = st.text_input("Email ", key="signup_email")
            p = st.text_input("Password ", type="password", key="signup_pw")
            if st.button("Create account", type="primary"):
                ok, msg = signup(e, p); st.success(msg) if ok else st.error(msg); st.rerun()
    else:
        st.success(f"Signed in as {user['email']}")
        if st.button("Log out"):
            from db import logout
            logout()
            st.rerun()

user = get_current_user_from_state()
user_id = user["id"] if user else None

with st.expander("Data controls", expanded=False):
    if st.button("Refresh market data"):
        st.cache_data.clear()
        st.rerun()

# Editor
portfolio_name = st.text_input("Portfolio name", value="My Portfolio")

if user_id:
    saved = list_portfolios(user_id)
    names = ["(Choose)"] + [nm for _, nm in saved]
    sel = st.selectbox("Load a saved portfolio", names)
    if sel != "(Choose)":
        pid = [pid for pid, nm in saved if nm == sel][0]
        df_loaded = load_holdings(pid)
        if not df_loaded.empty:
            st.session_state["tracker_ed"] = df_loaded.rename(columns={"ticker":"Ticker","shares":"Shares"})

ed_default = pd.DataFrame({"Ticker":["AAPL","MSFT","NVDA"], "Shares":[10,5,3], "Value":[np.nan,np.nan,np.nan]})
editor_df = st.session_state.get("tracker_ed", ed_default).copy()

input_mode = st.segmented_control(
    "Holdings input",
    options=["Shares", "Value ($)"],
    default="Shares",
    help="Use Value to enter position sizes; shares are auto-computed from latest prices."
)
st.caption("Edit holdings and click **Apply holdings** to refresh calculations.")

with st.form("tracker_holdings_form"):
    ed = st.data_editor(
        editor_df, num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(width="small"),
            "Shares": st.column_config.NumberColumn(format="%.6f", help="Number of shares (can be fractional)"),
            "Value": st.column_config.NumberColumn(format="%.2f", help="Position value in USD"),
        },
        key="tracker_editor_live"
    )
    apply_holdings = st.form_submit_button("Apply holdings", type="primary", use_container_width=True)

if apply_holdings:
    clean = ed.copy()
    clean["Ticker"] = clean["Ticker"].astype(str).str.strip().map(yf_symbol)
    clean["Shares"] = pd.to_numeric(clean["Shares"], errors="coerce")
    clean["Value"] = pd.to_numeric(clean["Value"], errors="coerce")
    clean = clean[clean["Ticker"].astype(bool)].reset_index(drop=True)

    if input_mode == "Value ($)" and not clean.empty:
        prices, ok = fetch_prices_chunked_with_fallback(
            clean["Ticker"].tolist(), period="5d", interval="1d", chunk=10, retries=2
        )
        latest = prices.ffill().iloc[-1] if not prices.empty else pd.Series(dtype=float)
        for i, row in clean.iterrows():
            value = row.get("Value")
            if pd.notna(value) and value > 0:
                px = latest.get(row["Ticker"])
                if pd.notna(px) and px > 0:
                    clean.at[i, "Shares"] = value / px
        if not latest.empty:
            clean["Value"] = clean["Ticker"].map(latest) * clean["Shares"]

    st.session_state["tracker_ed"] = clean[["Ticker","Shares","Value"]]

col_a, col_b = st.columns([1,1])
if user_id and st.button("💾 Save / Update", type="primary", use_container_width=True):
    committed = st.session_state.get("tracker_ed", ed_default).copy()
    clean = committed.copy()
    clean["ticker"] = clean["Ticker"].astype(str).str.strip().map(yf_symbol)
    clean["shares"] = pd.to_numeric(clean["Shares"], errors="coerce").fillna(0.0)
    clean = clean[clean["ticker"].astype(bool)][["ticker","shares"]]
    try:
        upsert_portfolio(user_id, portfolio_name, clean)
        st.success(f"Saved '{portfolio_name}'.")
        st.session_state["tracker_ed"] = ed
    except Exception as ex:
        st.error(str(ex))

if user_id and st.button("🗑️ Delete", use_container_width=True, disabled=not any(nm==portfolio_name for _, nm in (saved if user_id else []))):
    try:
        delete_portfolio(user_id, portfolio_name)
        st.warning(f"Deleted '{portfolio_name}'.")
        st.session_state.pop("tracker_ed", None)
        st.rerun()
    except Exception as ex:
        st.error(str(ex))

# Compute value + live score
port = st.session_state.get("tracker_ed", ed_default).copy()
port["Ticker"] = port["Ticker"].astype(str).str.strip().map(yf_symbol)
port = port[port["Ticker"].astype(bool)]
if port.empty:
    st.info("Add at least one holding to track.")
    st.stop()

def _cap_z(s: pd.Series, cap: float = 3.0) -> pd.Series:
    return s.clip(-cap, cap)

def _composite_row(row: pd.Series, weights: dict) -> float:
    total = sum(w for k, w in weights.items() if pd.notna(row.get(k)))
    if total == 0:
        return np.nan
    return float(sum(row.get(k) * w for k, w in weights.items() if pd.notna(row.get(k))) / total)

tickers = port["Ticker"].tolist()
shares = port.set_index("Ticker")["Shares"].fillna(0.0)

with st.status("Fetching prices & computing…", expanded=False):
    prices, ok = fetch_prices_chunked_with_fallback(tickers, period="1y", interval="1d")
    if prices.empty:
        st.error("Could not fetch prices for these tickers."); st.stop()

latest = prices.ffill().iloc[-1].reindex(tickers).fillna(0.0)
missing_prices = [t for t in tickers if t not in prices.columns]
if missing_prices:
    st.warning("Missing prices for: " + ", ".join(missing_prices))
values = latest * shares.reindex(tickers).fillna(0.0)
total_val = float(values.sum())
day_change = prices.ffill().iloc[-1] / prices.ffill().iloc[-2] - 1.0 if prices.shape[0] >= 2 else pd.Series(0, index=prices.columns)
day_pnl = float((day_change.reindex(tickers).fillna(0.0) * values / (1+day_change.reindex(tickers).fillna(0.0))).sum())

a,b,c = st.columns(3)
a.metric("Portfolio Value", f"${total_val:,.2f}")
b.metric("Day P&L", f"${day_pnl:,.2f}")
c.metric("Positions", f"{len(tickers)}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

aligned = prices.reindex(columns=tickers).ffill()
port_value = (aligned * shares.reindex(aligned.columns).fillna(0.0)).sum(axis=1)
st.line_chart(pd.DataFrame({"Portfolio value": port_value}), use_container_width=True)
st.caption("Sum of constituent values using fixed positions (buy & hold).")

st.markdown("### Allocation & Holdings")
weights_pct = (values / (values.sum() or 1.0)) * 100.0
day_pct = day_change.reindex(tickers).fillna(0.0) * 100.0
holdings_tbl = pd.DataFrame({
    "Price": latest.reindex(tickers),
    "Shares": shares.reindex(tickers),
    "Value": values.reindex(tickers),
    "Allocation %": weights_pct.reindex(tickers),
    "Day %": day_pct.reindex(tickers),
})
st.dataframe(holdings_tbl.round(4), use_container_width=True)

st.bar_chart(pd.DataFrame({"Allocation %": weights_pct.reindex(tickers)}), use_container_width=True)

sector_series = pd.Series({t: fetch_sector(t) for t in tickers})
sector_mix = weights_pct.groupby(sector_series).sum().sort_values(ascending=False)
if not sector_mix.empty:
    st.markdown("### Sector Mix")
    st.bar_chart(pd.DataFrame({"Allocation %": sector_mix}), use_container_width=True)

# Live rating (same weights as main portfolio page, no diversification here)
weights_by_value = (latest * shares).fillna(0.0)
weights = (weights_by_value / (weights_by_value.sum() or 1)).rename("weight")

panel_all = {t: prices[t].dropna() for t in tickers if t in prices}
tech_all = technical_scores(panel_all)
for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
    if col in tech_all.columns: tech_all[f"{col}_z"] = _cap_z(zscore_series(tech_all[col]))
TECH_score_all = tech_all[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech_all.columns]].mean(axis=1)

fund_raw_all = fetch_fundamentals_simple(list(panel_all.keys()))
core_fund = [
    "revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
    "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield",
    "trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"
]
core_present = [c for c in core_fund if c in fund_raw_all.columns]
if core_present:
    min_fund_cols = 4
    fund_raw_all = fund_raw_all[fund_raw_all[core_present].notna().sum(axis=1) >= min_fund_cols]
fdf_all = pd.DataFrame(index=fund_raw_all.index)
for col in ["revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
            "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield"]:
    if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=_cap_z(zscore_series(fund_raw_all[col]))
for col in ["trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"]:
    if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=_cap_z(zscore_series(-fund_raw_all[col]))
FUND_score_all = fdf_all.mean(axis=1) if len(fdf_all.columns) else pd.Series(dtype=float)

vix_series = fetch_vix_series(period="6mo", interval="1d")
gold_series = fetch_gold_series(period="6mo", interval="1d")
dxy_series = fetch_dxy_series(period="6mo", interval="1d")
tnx_series = fetch_tnx_series(period="6mo", interval="1d")
credit_series = fetch_credit_ratio_series(period="6mo", interval="1d")
MACRO = macro_from_signals(vix_series, gold_series, dxy_series, tnx_series, credit_series)["macro"]

idx_all = pd.Index(list(panel_all.keys()))
tmp = pd.DataFrame(index=idx_all)
tmp["FUND"] = FUND_score_all.reindex(idx_all)
tmp["TECH"] = TECH_score_all.reindex(idx_all)
tmp["MACRO"] = MACRO
wf,wt,wm = 0.50,0.45,0.05; wsum=(wf+wt+wm) or 1.0; wf,wt,wm = wf/wsum,wt/wsum,wm/wsum
comp_weights = {"FUND": wf, "TECH": wt, "MACRO": wm}
tmp["COMP"] = tmp.apply(_composite_row, axis=1, weights=comp_weights)
tmp = tmp.join(weights, how="left").fillna({"weight":0.0})
live_signal = float((tmp["COMP"]*tmp["weight"]).sum())
live_score = float(np.clip((live_signal+1)/2, 0, 1)*100)

st.markdown("### 🔮 Live ‘Rate My Portfolio’ Score")
st.metric("Portfolio Score (0–100)", f"{live_score:.1f}")
def _score_label(score: float) -> str:
    if score >= 80: return "Strong Buy"
    if score >= 60: return "Buy"
    if score >= 40: return "Hold"
    if score >= 20: return "Sell"
    return "Strong Sell"

st.caption(
    f"Interpretation: **{_score_label(live_score)}**. "
    "80+ strong, 60–79 buy, 40–59 hold, 20–39 sell, <20 strong sell."
)
