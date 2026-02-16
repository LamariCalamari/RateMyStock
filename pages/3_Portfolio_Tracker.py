# pages/3_Portfolio_Tracker.py (deprecated)
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from db import ensure_db, get_current_user_from_state, signup, login, list_portfolios, load_holdings, upsert_portfolio, delete_portfolio
from app_utils import (
    inject_css, brand_header, yf_symbol, fetch_prices_chunked_with_fallback,
    fetch_macro_pack, score_universe_panel, fetch_sector,
    SCORING_CONFIG, score_label, build_universe, FACTOR_LABELS, portfolio_signal_score
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
st.markdown(
    "- This is the historical value of your holdings with shares held constant.  \n"
    "- Rising line = portfolio grew; dips show drawdowns."
)

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
st.caption("Allocation shows current weight of each holding by market value.")

sector_series = pd.Series({t: fetch_sector(t) for t in tickers}).fillna("Unknown")
sector_mix = weights_pct.groupby(sector_series).sum().sort_values(ascending=False)
if not sector_mix.empty:
    st.markdown("### Sector Mix")
    st.bar_chart(pd.DataFrame({"Allocation %": sector_mix}), use_container_width=True)
    st.caption("Sector mix groups holdings by sector to show concentration risk.")

# Live rating (same weights as main portfolio page, no diversification here)
weights_by_value = (latest * shares).fillna(0.0)
weights = (weights_by_value / (weights_by_value.sum() or 1)).rename("weight")

panel_all = {t: prices[t].dropna() for t in tickers if t in prices}
score_universe, score_label_txt = build_universe(tickers, "Auto by index membership", 140, "")
score_prices, score_ok = fetch_prices_chunked_with_fallback(
    score_universe, period="1y", interval="1d", chunk=20, retries=3, sleep_between=0.35, singles_pause=0.25
)
score_panel = {
    t: score_prices[t].dropna()
    for t in score_ok
    if t in score_prices.columns and score_prices[t].dropna().size > 0
}
if not score_panel:
    score_panel = panel_all
    score_label_txt = "Holdings only (fallback)"

macro_pack = fetch_macro_pack(period="6mo", interval="1d")
score_pack = score_universe_panel(
    score_panel,
    target_count=max(len(score_universe), len(score_panel), 1),
    macro_pack=macro_pack,
    min_fund_cols=SCORING_CONFIG["min_fund_cols"],
)
out_all = score_pack["out"]
fdf_all = score_pack["fund"]
tech_all = score_pack["tech"]
MACRO = float(macro_pack["macro"])
if out_all.empty:
    st.error("Unable to compute live score from the current market data.")
    st.stop()

tmp = out_all.reindex(tickers)[["FUND_score", "TECH_score", "MACRO_score", "COMPOSITE", "CONFIDENCE"]].copy()
tmp = tmp.join(weights, how="left").fillna({"weight": 0.0})
missing_for_score = [t for t in tickers if t not in out_all.index]
if missing_for_score:
    st.caption("Scoring fallback excluded: " + ", ".join(missing_for_score))

live_pack = portfolio_signal_score(
    out_all["COMPOSITE"],
    tmp["COMPOSITE"],
    tmp["weight"],
    diversification=None,
)
live_signal = float(live_pack["signal"]) if pd.notna(live_pack["signal"]) else np.nan
live_pct = float(live_pack["signal_percentile"]) if pd.notna(live_pack["signal_percentile"]) else np.nan
live_score = float(live_pack["final_score"]) if pd.notna(live_pack["final_score"]) else np.nan
if pd.isna(live_score):
    st.error("Portfolio score unavailable due to missing holdings factor coverage.")
    st.stop()

st.markdown("### 🔮 Live ‘Rate My Portfolio’ Score")
st.metric("Portfolio Score (0–100)", f"{live_score:.1f}")
st.caption(f"Live score peer set: {score_label_txt}.")
st.caption(f"Signal percentile vs peers: {live_pct:.1f} (weighted composite {live_signal:.3f}).")
st.caption(
    f"Interpretation: **{score_label(live_score)}**. "
    "80+ strong, 60–79 buy, 40–59 hold, 20–39 sell, <20 strong sell."
)

st.markdown("### Score Bands")
def _band(score: float) -> int:
    if np.isnan(score): return -1
    if score >= 80: return 4
    if score >= 60: return 3
    if score >= 40: return 2
    if score >= 20: return 1
    return 0

active = _band(live_score)
bands = [
    ("0–19", "#e74c3c"),
    ("20–39", "#f39c12"),
    ("40–59", "#f1c40f"),
    ("60–79", "#8bc34a"),
    ("80–100", "#2ecc71"),
]
cells = []
for i, (label, color) in enumerate(bands):
    ring = "box-shadow:0 0 0 2px #fff inset;" if i == active else ""
    text = "color:#111;" if i >= 2 else ""
    cells.append(
        f"<div style='flex:1;background:{color};padding:8px;border-radius:6px;text-align:center;{text}{ring}'>"
        f"{label}</div>"
    )
st.markdown(f"<div style='display:flex;gap:6px;'>{''.join(cells)}</div>", unsafe_allow_html=True)

with st.expander("How to read this tracker score"):
    st.markdown(
        "- **Fundamentals** and **Technicals** are peer‑relative z‑scores. Positive = stronger vs peers.  \n"
        "- **Macro** is a regime overlay (VIX, USD, rates, credit, gold) and is the same for all holdings.  \n"
        "- Score is based on your weighted signal percentile within the selected peer universe.  \n"
        "- This tracker score is a live snapshot; use **Rate My Portfolio** for deeper analysis."
    )

def _weighted_mean_z(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    out = {}
    for col in [c for c in df.columns if c.endswith("_z")]:
        series = df[col].reindex(weights.index)
        mask = series.notna() & weights.notna()
        if mask.any():
            out[col] = float(np.average(series[mask], weights=weights[mask]))
    return pd.Series(out).sort_values()

weak_factors = pd.Series(dtype=float)
if not weights.empty:
    weights_norm = weights / (weights.sum() or 1.0)
    fund_weak = _weighted_mean_z(fdf_all.reindex(tickers), weights_norm)
    tech_weak = _weighted_mean_z(tech_all.reindex(tickers), weights_norm)
    weak_factors = pd.concat([fund_weak, tech_weak]).sort_values().head(3)

st.markdown("### Suggested Additions (factor‑based)")
if weak_factors.empty:
    st.caption("Not enough data to generate factor‑based ideas. Add more holdings or try again later.")
else:
    weak_labels = [FACTOR_LABELS.get(k, k.replace("_z", "")) for k in weak_factors.index]
    st.caption("Weakest factors: " + ", ".join(weak_labels) + ".")

    rec_sig = (tuple(sorted(tickers)), tuple(weak_factors.index))
    if st.session_state.get("rec_sig") != rec_sig:
        st.session_state.pop("rec_df", None)
        st.session_state["rec_sig"] = rec_sig

    if st.button("Generate recommendations", type="secondary", use_container_width=True):
        with st.status("Scanning peers for factor‑based ideas…", expanded=True) as status:
            prog = st.progress(0)
            msg = st.empty()
            msg.info("Step 1/3: building candidate universe.")
            universe, label = build_universe(tickers, "Auto (industry → sector → index)", 120, "")
            candidates = [t for t in universe if t not in tickers][:60]
            prog.progress(20)

            msg.info("Step 2/3: loading candidate price history.")
            prices_c, ok = fetch_prices_chunked_with_fallback(candidates, period="1y", interval="1d", chunk=20, retries=2, sleep_between=0.4, singles_pause=0.4)
            panel_c = {t: prices_c[t].dropna() for t in ok if t in prices_c.columns and prices_c[t].dropna().size > 0}
            if not panel_c:
                st.warning("No candidate prices available.")
            prog.progress(60)

            msg.info("Step 3/3: scoring candidates on weak factors.")
            cand_pack = score_universe_panel(
                panel_c,
                target_count=max(len(candidates), len(panel_c), 1),
                macro_pack=macro_pack,
                min_fund_cols=SCORING_CONFIG["min_fund_cols"],
            )
            fdf_c = cand_pack["fund"]
            tech_c = cand_pack["tech"]
            cand = pd.concat([fdf_c, tech_c[[c for c in tech_c.columns if c.endswith("_z")]]], axis=1)
            use_cols = [c for c in weak_factors.index if c in cand.columns]
            if use_cols:
                cand["improve_score"] = cand[use_cols].mean(axis=1)
                cand = cand.sort_values("improve_score", ascending=False).head(8)
                cand["Sector"] = [fetch_sector(t) for t in cand.index]
                why = []
                for t in cand.index:
                    top = cand.loc[t, use_cols].sort_values(ascending=False).head(2)
                    why.append(", ".join([FACTOR_LABELS.get(k, k) for k in top.index]))
                cand["Why"] = why
                st.session_state["rec_df"] = cand
            else:
                st.warning("Not enough candidate data to score weak factors.")
            prog.progress(100)
            status.update(label="Done!", state="complete")

    rec_df = st.session_state.get("rec_df")
    if isinstance(rec_df, pd.DataFrame) and not rec_df.empty:
        st.dataframe(rec_df[["Sector","improve_score","Why"]].round(3), use_container_width=True)
        st.caption(
            "Ideas are ranked by strength on your weakest factors. "
            "Use as a starting point, not investment advice."
        )
