# pages/2_Rate_My_Portfolio.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from app_utils import (
    inject_css, brand_header, yf_symbol, fetch_prices_chunked_with_fallback,
    fetch_vix_series, macro_from_vix, technical_scores, fetch_fundamentals_simple,
    zscore_series, percentile_rank, build_universe
)

st.set_page_config(page_title="Rate My Portfolio", layout="wide")
inject_css()
brand_header("Rate My Portfolio")

CURRENCY_MAP = {"$":"USD","€":"EUR","£":"GBP","CHF":"CHF","C$":"CAD","A$":"AUD","¥":"JPY"}
def _safe_num(x): return pd.to_numeric(x, errors="coerce")

def normalize_percents_to_100(p: pd.Series) -> pd.Series:
    p = _safe_num(p).fillna(0.0)
    s = p.sum()
    if s <= 0: return p
    return (p / s) * 100.0

def sync_percent_amount(df: pd.DataFrame, total: float, mode: str) -> pd.DataFrame:
    df=df.copy()
    df["Ticker"]=df["Ticker"].astype(str).str.strip()
    df=df[df["Ticker"].astype(bool)].reset_index(drop=True)
    n=len(df)
    if n==0:
        df["weight"]=[]
        return df

    df["Percent (%)"]=_safe_num(df.get("Percent (%)"))
    df["Amount"]=_safe_num(df.get("Amount"))
    has_total = (total is not None and total>0)

    if has_total:
        if mode=="percent":
            if df["Percent (%)"].fillna(0).sum()==0:
                df["Percent (%)"]=100.0/n
            df["Percent (%)"] = normalize_percents_to_100(df["Percent (%)"]).round(2)
            df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
        else:
            s=df["Amount"].fillna(0).sum()
            if s>0:
                df["Percent (%)"]= (df["Amount"]/total*100.0).round(2)
                df["Percent (%)"]= normalize_percents_to_100(df["Percent (%)"]).round(2)
                df["Amount"]= (df["Percent (%)"]/100.0*total).round(2)
            else:
                df["Percent (%)"]=100.0/n
                df["Amount"]= (df["Percent (%)"]/100.0*total).round(2)
    else:
        if df["Percent (%)"].fillna(0).sum()==0:
            df["Percent (%)"]=100.0/n
        df["Percent (%)"]= normalize_percents_to_100(df["Percent (%)"]).round(2)

    if has_total and df["Amount"].fillna(0).sum()>0:
        w=df["Amount"].fillna(0)/df["Amount"].fillna(0).sum()
    elif df["Percent (%)"].fillna(0).sum()>0:
        w=df["Percent (%)"].fillna(0)/df["Percent (%)"].fillna(0).sum()
    else:
        w=pd.Series([1.0/n]*n, index=df.index)
    df["weight"]=w
    return df

# Default grid
if st.session_state.get("grid_df") is None:
    st.session_state["grid_df"] = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA", "AMZN"],
        "Percent (%)": [25.0, 25.0, 25.0, 25.0],
        "Amount": [np.nan, np.nan, np.nan, np.nan],
    })

t1,t2,t3=st.columns([1,1,1])
with t1: cur = st.selectbox("Currency", list(CURRENCY_MAP.keys()), index=0)
with t2: total = st.number_input(f"Total portfolio value ({cur})", min_value=0.0, value=10000.0, step=500.0)
with t3: st.caption("Holdings update only when you click **Apply changes**.")

st.markdown(
    f"**Holdings**  \n"
    f"<span class='small-muted'>Enter <b>Ticker</b> and either <b>Percent (%)</b> or "
    f"<b>Amount ({cur})</b>. Values update only when you click "
    f"<b>Apply changes</b>. Use <b>Normalize</b> to force exactly 100% in percent mode.</span>",
    unsafe_allow_html=True,
)

committed = st.session_state["grid_df"].copy()
with st.form("holdings_form", clear_on_submit=False):
    sync_mode = st.segmented_control(
        "Sync mode",
        options=["Percent → Amount", "Amount → Percent"],
        default="Percent → Amount",
        help="Choose which side drives on Apply."
    )
    mode_key = {"Percent → Amount": "percent", "Amount → Percent": "amount"}[sync_mode]

    edited = st.data_editor(
        committed,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(width="small"),
            "Percent (%)": st.column_config.NumberColumn(format="%.2f"),
            "Amount": st.column_config.NumberColumn(format="%.2f", help=f"Amount in {cur}"),
        },
        key="grid_form",
    )

    col_a, col_b = st.columns([1, 1])
    apply_btn = col_a.form_submit_button("Apply changes", type="primary", use_container_width=True)
    normalize_btn = col_b.form_submit_button("Normalize to 100% (percent mode)", use_container_width=True)

if normalize_btn:
    syncd = edited.copy()
    syncd["Percent (%)"] = normalize_percents_to_100(_safe_num(syncd.get("Percent (%)")))
    if total and total>0:
        syncd["Amount"] = (syncd["Percent (%)"]/100.0*total).round(2)
    st.session_state["grid_df"] = syncd[["Ticker","Percent (%)","Amount"]]

elif apply_btn:
    syncd = sync_percent_amount(edited.copy(), total, mode_key)
    st.session_state["grid_df"] = syncd[["Ticker","Percent (%)","Amount"]]

current = st.session_state["grid_df"].copy()
out = current.copy()
out["ticker"] = out["Ticker"].map(yf_symbol)
out = out[out["ticker"].astype(bool)]

if out.empty:
    st.info("Add at least one holding to run the rating.")
    st.stop()

with st.status("Crunching the numbers…", expanded=True) as status:
    prog = st.progress(0)
    tickers = out["ticker"].tolist()
    universe, label = build_universe(tickers, "Auto by index membership", 180, "")
    target_count = len(universe)
    prog.progress(8)

    prices, ok = fetch_prices_chunked_with_fallback(
        universe, period="1y", interval="1d",
        chunk=25, retries=3, sleep_between=0.35, singles_pause=0.20
    )
    if prices.empty:
        st.error("No prices fetched."); st.stop()
    prog.progress(40)

    panel_all = {t: prices[t].dropna() for t in prices.columns if t in prices.columns and prices[t].dropna().size>0}
    tech_all = technical_scores(panel_all)
    for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        if col in tech_all.columns: tech_all[f"{col}_z"] = zscore_series(tech_all[col])
    TECH_score_all = tech_all[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech_all.columns]].mean(axis=1)
    prog.progress(65)

    fund_raw_all = fetch_fundamentals_simple(list(panel_all.keys()))
    fdf_all = pd.DataFrame(index=fund_raw_all.index)
    for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
                "grossMargins","operatingMargins","ebitdaMargins"]:
        if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=zscore_series(fund_raw_all[col])
    for col in ["trailingPE","forwardPE","debtToEquity"]:
        if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=zscore_series(-fund_raw_all[col])
    FUND_score_all = fdf_all.mean(axis=1) if len(fdf_all.columns) else pd.Series(0.0, index=fund_raw_all.index)
    prog.progress(85)

    vix_series = fetch_vix_series(period="6mo", interval="1d")
    MACRO, _, _, _ = macro_from_vix(vix_series)
    prog.progress(100)
    status.update(label="Done!", state="complete")

st.markdown(
    f'<div class="banner">Peers loaded: <b>{len(panel_all)}</b> / <b>{target_count}</b> '
    f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True
)

idx_all = pd.Index(list(panel_all.keys()))
out_all = pd.DataFrame(index=idx_all)
out_all["FUND_score"]  = FUND_score_all.reindex(idx_all).fillna(0.0)
out_all["TECH_score"]  = TECH_score_all.reindex(idx_all).fillna(0.0)
out_all["MACRO_score"] = MACRO
wsum=(0.45+0.40+0.10) or 1.0
wf,wt,wm = 0.45/wsum, 0.40/wsum, 0.10/wsum
out_all["COMPOSITE"] = wf*out_all["FUND_score"] + wt*out_all["TECH_score"] + wm*out_all["MACRO_score"]
out_all["RATING_0_100"] = percentile_rank(out_all["COMPOSITE"])

# Diversification metrics (same as original)
def fetch_sector(t):
    try: return yf.Ticker(t).info.get("sector", None)
    except Exception: return None

weights = None
if total and total>0 and _safe_num(out["Amount"]).sum()>0:
    weights = _safe_num(out["Amount"]) / _safe_num(out["Amount"]).sum()
elif _safe_num(out["Percent (%)"]).sum()>0:
    weights = _safe_num(out["Percent (%)"]) / _safe_num(out["Percent (%)"]).sum()
else:
    n = max(len(out),1)
    weights = pd.Series([1.0/n]*n, index=out.index)

tickers_shown = out["ticker"].tolist()
weights_named = pd.Series(list(weights.values), index=tickers_shown)

meta_sec = pd.Series({t: fetch_sector(t) for t in tickers_shown})
sec_mix = weights_named.groupby(meta_sec).sum()
if sec_mix.empty: sec_mix = pd.Series({"Unknown":1.0})
hhi = float((sec_mix**2).sum())
effN = 1.0/hhi if hhi>0 else 1.0
targetN = min(10, max(1,len(sec_mix)))
sector_div = float(np.clip((effN-1)/(targetN-1 if targetN>1 else 1), 0, 1))
max_w = float(weights_named.max())
if   max_w <= 0.10: name_div = 1.0
elif max_w >= 0.40: name_div = 0.0
else:               name_div = float((0.40-max_w)/0.30)

ret = prices[tickers_shown].pct_change().dropna(how="all")
if ret.shape[1]>=2:
    corr = ret.corr().values; n=corr.shape[0]
    avg_corr = (corr.sum()-np.trace(corr))/max(1,(n*n-n))
    corr_div = float(np.clip(1.0-max(0.0, avg_corr), 0.0, 1.0))
else:
    avg_corr=np.nan; corr_div=0.5

DIV = 0.5*sector_div + 0.3*corr_div + 0.2*name_div

per_name = out_all.reindex(tickers_shown).copy()
per_name = per_name.assign(weight = list(weights_named.values))
per_name["weighted_composite"] = per_name["COMPOSITE"]*per_name["weight"]
port_signal = float(per_name["weighted_composite"].sum())
total_for_final = 1.0 + 0.05
port_final = (port_signal)*(1/total_for_final) + DIV*(0.05/total_for_final)
port_score = float(np.clip((port_final+1)/2, 0, 1)*100)

st.markdown("## 🧺 Portfolio — Scores")
a,b,c,d = st.columns(4)
a.metric("Portfolio Score (0–100)", f"{port_score:.1f}")
b.metric("Signal (weighted composite)", f"{port_signal:.3f}")
c.metric("Macro (VIX)", f"{MACRO:.3f}")
d.metric("Diversification", f"{DIV:.3f}")

with st.expander("Why this portfolio rating?"):
    show = per_name.rename(columns={"weight":"Weight","FUND_score":"Fundamentals",
                                    "TECH_score":"Technicals","MACRO_score":"Macro (VIX)",
                                    "COMPOSITE":"Composite","weighted_composite":"Weight × Comp"})[
        ["Weight","Fundamentals","Technicals","Macro (VIX)","Composite","Weight × Comp"]
    ]
    st.dataframe(show.round(4), use_container_width=True)
    st.markdown("**Diversification explained**")
    st.markdown(
        f"- **Sector mix** → effective number of sectors ≈ **{effN:.1f}** → sector diversity score **{sector_div:.2f}**.  \n"
        f"- **Name concentration** → max single position ≈ **{max_w*100:.1f}%** → score **{name_div:.2f}**.  \n"
        f"- **Correlation** → average pairwise correlation ≈ "
        f"{('%.2f' % avg_corr) if not np.isnan(avg_corr) else 'N/A'} → score **{corr_div:.2f}**.  \n"
        f"- **Diversification score** = 50% sector + 30% correlation + 20% name concentration."
    )

# Risk/return charts — identical to original
px_held = prices[tickers_shown].dropna(how="all")
r = px_held.pct_change().fillna(0)
w_vec = weights_named.reindex(px_held.columns).fillna(0).values
port_r = (r * w_vec).sum(axis=1)
eq = (1+port_r).cumprod()
tabs = st.tabs(["Cumulative", "Volatility (60d) & Sharpe", "Drawdown"])
with tabs[0]:
    st.subheader("Cumulative growth (set = 1.0)")
    st.line_chart(pd.DataFrame({"Portfolio cumulative": eq}), use_container_width=True)
    st.caption("Growth of 1.0 invested, using your current weights over the chosen history.")
with tabs[1]:
    st.subheader("Volatility & rolling Sharpe (60-day)")
    vol60 = port_r.rolling(60).std()*np.sqrt(252)
    sharpe60 = (port_r.rolling(60).mean()/port_r.rolling(60).std())*np.sqrt(252)
    st.line_chart(pd.DataFrame({"Volatility 60d (ann.)": vol60, "Sharpe 60d": sharpe60}), use_container_width=True)
    st.caption("Lower volatility & higher Sharpe are preferred.")
with tabs[2]:
    st.subheader("Drawdown")
    roll_max = eq.cummax(); dd = eq/roll_max - 1
    st.line_chart(pd.DataFrame({"Drawdown": dd}), use_container_width=True)
    st.caption("Depth of falls from prior peaks (risk perspective).")
