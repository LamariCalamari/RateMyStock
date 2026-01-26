# pages/2_Rate_My_Portfolio.py
import numpy as np
import pandas as pd
import streamlit as st
from app_utils import (
    inject_css, brand_header, yf_symbol, fetch_prices_chunked_with_fallback,
    fetch_vix_series, fetch_gold_series, fetch_dxy_series, fetch_tnx_series, fetch_credit_ratio_series,
    macro_from_signals, technical_scores, fetch_fundamentals_simple,
    zscore_series, percentile_rank, build_universe, fetch_sector,
    SCORING_CONFIG, score_label
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

def sync_percent_amount(df: pd.DataFrame, total: float | None, mode: str) -> pd.DataFrame:
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
        if mode == "amount" and df["Amount"].fillna(0).sum() > 0:
            df["Percent (%)"] = (df["Amount"] / df["Amount"].sum() * 100.0).round(2)
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

def _cap_z(s: pd.Series, cap: float | None = None) -> pd.Series:
    cap = SCORING_CONFIG["z_cap"] if cap is None else cap
    return s.clip(-cap, cap)

def _composite_row(row: pd.Series, weights: dict) -> float:
    total = sum(w for k, w in weights.items() if pd.notna(row.get(k)))
    if total == 0:
        return np.nan
    return float(sum(row.get(k) * w for k, w in weights.items() if pd.notna(row.get(k))) / total)

def _coverage(df: pd.DataFrame, tickers: list[str], cols: list[str]) -> float:
    if df.empty or not cols:
        return 0.0
    valid = [t for t in tickers if t in df.index]
    if not valid:
        return 0.0
    return float(df.loc[valid, cols].notna().mean().mean())

# Default grid
if st.session_state.get("grid_df") is None:
    st.session_state["grid_df"] = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA", "AMZN"],
        "Percent (%)": [25.0, 25.0, 25.0, 25.0],
        "Amount": [np.nan, np.nan, np.nan, np.nan],
    })

t1,t2,t3=st.columns([1,1,1])
with t1: cur = st.selectbox("Currency", list(CURRENCY_MAP.keys()), index=0)
with t3: st.caption("Holdings update only when you click **Apply changes**.")

st.markdown(
    f"**Holdings**  \n"
    f"<span class='small-muted'>Enter <b>Ticker</b> and either <b>Percent (%)</b> or "
    f"<b>Amount ({cur})</b>. Use <b>Normalize</b> to force exactly 100% in percent mode.</span>",
    unsafe_allow_html=True,
)

sync_mode = st.segmented_control(
    "Sync mode",
    options=["Percent → Amount", "Amount → Percent"],
    default="Percent → Amount",
    help="Choose which side drives."
)
mode_key = {"Percent → Amount": "percent", "Amount → Percent": "amount"}[sync_mode]

auto_sync = st.checkbox("Auto-sync amounts", value=True)

with t2:
    if mode_key == "percent":
        total = st.number_input(
            f"Total portfolio value ({cur})",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )
    else:
        total = None
        current = st.session_state.get("grid_df", pd.DataFrame())
        derived_total = float(_safe_num(current.get("Amount")).sum()) if not current.empty else 0.0
        st.metric(f"Derived total ({cur})", f"{derived_total:,.2f}")

committed = st.session_state["grid_df"].copy()
percent_disabled = mode_key == "amount"
amount_disabled = mode_key == "percent"
edited = st.data_editor(
    committed,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Percent (%)": st.column_config.NumberColumn(format="%.2f", disabled=percent_disabled),
        "Amount": st.column_config.NumberColumn(
            format="%.2f", help=f"Amount in {cur}", disabled=amount_disabled
        ),
    },
    key="grid_form",
)

col_a, col_b = st.columns([1, 1])
apply_btn = col_a.button("Apply changes", type="primary", use_container_width=True, disabled=auto_sync)
normalize_btn = col_b.button("Normalize to 100% (percent mode)", use_container_width=True)

if normalize_btn:
    syncd = edited.copy()
    syncd["Percent (%)"] = normalize_percents_to_100(_safe_num(syncd.get("Percent (%)")))
    if total and total>0:
        syncd["Amount"] = (syncd["Percent (%)"]/100.0*total).round(2)
    st.session_state["grid_df"] = syncd[["Ticker","Percent (%)","Amount"]]

elif auto_sync:
    syncd = sync_percent_amount(edited.copy(), total, mode_key)
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

sig = (
    tuple(out["ticker"].tolist()),
    total if "total" in locals() else None,
    sync_mode,
)
if st.session_state.get("portfolio_sig") != sig:
    st.session_state["portfolio_ready"] = False

start_col = st.columns([1,2,1])[1]
with start_col:
    start = st.button("Start portfolio analysis", type="primary", use_container_width=True)
if start:
    st.session_state["portfolio_ready"] = True
    st.session_state["portfolio_sig"] = sig

if not st.session_state.get("portfolio_ready"):
    st.info("Review holdings, then click **Start portfolio analysis**.")
    st.stop()

with st.status("Crunching the numbers…", expanded=True) as status:
    prog = st.progress(0)
    pct = st.empty()
    msg = st.empty()
    tickers = out["ticker"].tolist()
    msg.info("Step 1/4: building peer universe.")
    universe, label = build_universe(tickers, "Auto by index membership", 180, "")
    target_count = len(universe)
    prog.progress(8)
    pct.markdown("**Progress: 8%**")

    msg.info("Step 2/4: downloading price history.")
    prices, ok = fetch_prices_chunked_with_fallback(
        universe, period="1y", interval="1d",
        chunk=25, retries=3, sleep_between=0.35, singles_pause=0.20
    )
    if prices.empty:
        st.error("No prices fetched."); st.stop()
    prog.progress(40)
    pct.markdown("**Progress: 40%**")

    panel_all = {t: prices[t].dropna() for t in prices.columns if t in prices.columns and prices[t].dropna().size>0}
    msg.info("Step 3/4: computing technical signals.")
    tech_all = technical_scores(panel_all)
    for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        if col in tech_all.columns: tech_all[f"{col}_z"] = _cap_z(zscore_series(tech_all[col]))
    TECH_score_all = tech_all[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech_all.columns]].mean(axis=1)
    prog.progress(65)
    pct.markdown("**Progress: 65%**")

    msg.info("Step 4/4: computing fundamentals + macro.")
    fund_raw_all = fetch_fundamentals_simple(list(panel_all.keys()))
    core_fund = [
        "revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
        "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield",
        "trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"
    ]
    core_present = [c for c in core_fund if c in fund_raw_all.columns]
    if core_present:
        min_fund_cols = SCORING_CONFIG["min_fund_cols"]
        fund_raw_all = fund_raw_all[fund_raw_all[core_present].notna().sum(axis=1) >= min_fund_cols]
    fdf_all = pd.DataFrame(index=fund_raw_all.index)
    for col in ["revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
                "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield"]:
        if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=_cap_z(zscore_series(fund_raw_all[col]))
    for col in ["trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"]:
        if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=_cap_z(zscore_series(-fund_raw_all[col]))
    FUND_score_all = fdf_all.mean(axis=1) if len(fdf_all.columns) else pd.Series(dtype=float)
    prog.progress(85)

    vix_series = fetch_vix_series(period="6mo", interval="1d")
    gold_series = fetch_gold_series(period="6mo", interval="1d")
    dxy_series = fetch_dxy_series(period="6mo", interval="1d")
    tnx_series = fetch_tnx_series(period="6mo", interval="1d")
    credit_series = fetch_credit_ratio_series(period="6mo", interval="1d")
    MACRO = macro_from_signals(vix_series, gold_series, dxy_series, tnx_series, credit_series)["macro"]
    prog.progress(100)
    pct.markdown("**Progress: 100%**")
    status.update(label="Done!", state="complete")

st.markdown(
    f'<div class="banner">Peers loaded: <b>{len(panel_all)}</b> / <b>{target_count}</b> '
    f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True
)

idx_all = pd.Index(list(panel_all.keys()))
out_all = pd.DataFrame(index=idx_all)
out_all["FUND_score"]  = FUND_score_all.reindex(idx_all)
out_all["TECH_score"]  = TECH_score_all.reindex(idx_all)
out_all["MACRO_score"] = MACRO
wsum = sum(SCORING_CONFIG["weights"].values()) or 1.0
wf = SCORING_CONFIG["weights"]["fund"] / wsum
wt = SCORING_CONFIG["weights"]["tech"] / wsum
wm = SCORING_CONFIG["weights"]["macro"] / wsum
weights = {"FUND_score": wf, "TECH_score": wt, "MACRO_score": wm}
out_all["COMPOSITE"] = out_all.apply(_composite_row, axis=1, weights=weights)
ratings = percentile_rank(out_all["COMPOSITE"].dropna())
out_all["RATING_0_100"] = ratings.reindex(out_all.index)

weights = None
if total and total>0 and _safe_num(out["Amount"]).sum()>0:
    weights = _safe_num(out["Amount"]) / _safe_num(out["Amount"]).sum()
elif _safe_num(out["Percent (%)"]).sum()>0:
    weights = _safe_num(out["Percent (%)"]) / _safe_num(out["Percent (%)"]).sum()
else:
    n = max(len(out),1)
    weights = pd.Series([1.0/n]*n, index=out.index)

tickers_shown = out["ticker"].tolist()
available_holdings = [t for t in tickers_shown if t in prices.columns]
missing_holdings = [t for t in tickers_shown if t not in prices.columns]
if missing_holdings:
    st.warning(
        "Some holdings had no price data and were excluded from charts/correlation: "
        + ", ".join(missing_holdings)
    )
if not available_holdings:
    st.error("No price data available for current holdings.")
    st.stop()

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

ret = prices[available_holdings].pct_change().dropna(how="all")
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
c.metric("Macro (Multi-signal)", f"{MACRO:.3f}")
d.metric("Diversification", f"{DIV:.3f}")
st.caption(
    f"Interpretation: **{score_label(port_score)}**. "
    "Score blends fundamentals, technicals, and macro; diversification rewards lower concentration and correlations."
)

st.markdown("### Score Bands")
def _band(score: float) -> int:
    if np.isnan(score): return -1
    if score >= 80: return 4
    if score >= 60: return 3
    if score >= 40: return 2
    if score >= 20: return 1
    return 0

active = _band(port_score)
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
st.caption(f"Current score band highlighted (score {port_score:.1f}).")

tech_cols = [c for c in tech_all.columns if c.endswith("_z")]
fund_cols = [c for c in fdf_all.columns if c.endswith("_z")]
peer_factor = min(len(out_all["COMPOSITE"].dropna()) / max(target_count, 1), 1.0)
cw = SCORING_CONFIG["confidence_weights"]
port_conf = 100.0 * (cw["peer"]*peer_factor + cw["fund"]*_coverage(fdf_all, tickers_shown, fund_cols) + cw["tech"]*_coverage(tech_all, tickers_shown, tech_cols))
st.caption(f"Confidence: {port_conf:.0f}/100 based on data coverage and peer sample size.")


st.markdown("### Allocation Snapshot")
weights_named = pd.Series(list(weights.values), index=tickers_shown)
alloc = pd.DataFrame({
    "Ticker": tickers_shown,
    "Weight %": (weights_named.values * 100.0),
}).sort_values("Weight %", ascending=False)
st.dataframe(alloc.round(2), use_container_width=True)

st.markdown("### Sector Mix")
sector_series = pd.Series({t: fetch_sector(t) for t in tickers_shown}).fillna("Unknown")
sector_mix = weights_named.groupby(sector_series).sum().sort_values(ascending=False) * 100.0
st.bar_chart(pd.DataFrame({"Allocation %": sector_mix}), use_container_width=True)
st.caption("Sector mix highlights concentration risk. High single‑sector exposure increases risk.")

st.markdown("### Top Contributors (Weight × Score)")
per_name_sorted = per_name.copy()
per_name_sorted["weight_pct"] = per_name_sorted["weight"] * 100.0
per_name_sorted = per_name_sorted.sort_values("weighted_composite", ascending=False)
top = per_name_sorted.head(5)[["weight_pct","COMPOSITE","weighted_composite"]].rename(
    columns={"weight_pct":"Weight %","COMPOSITE":"Composite","weighted_composite":"Contribution"}
)
st.dataframe(top.round(4), use_container_width=True)
st.caption("Contribution shows which holdings drive the portfolio signal the most.")

with st.expander("Why this portfolio rating?"):
    show = per_name.rename(columns={"weight":"Weight","FUND_score":"Fundamentals",
                                    "TECH_score":"Technicals","MACRO_score":"Macro (Multi-signal)",
                                    "COMPOSITE":"Composite","weighted_composite":"Weight × Comp"})[
        ["Weight","Fundamentals","Technicals","Macro (Multi-signal)","Composite","Weight × Comp"]
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
px_held = prices[available_holdings].dropna(how="all")
r = px_held.pct_change().fillna(0)
w_vec = weights_named.reindex(px_held.columns).fillna(0).values
port_r = (r * w_vec).sum(axis=1)
eq = (1+port_r).cumprod()
mdd = float((eq / eq.cummax() - 1).min())
vol_ann = float(port_r.std() * np.sqrt(252))
sharpe = float((port_r.mean() / (port_r.std() or 1e-9)) * np.sqrt(252))
st.markdown("### Risk Snapshot")
r1, r2, r3 = st.columns(3)
r1.metric("Max Drawdown", f"{mdd*100:.1f}%")
r2.metric("Volatility (ann.)", f"{vol_ann*100:.1f}%")
r3.metric("Sharpe (ann.)", f"{sharpe:.2f}")
st.caption("Max drawdown = worst peak‑to‑trough loss; volatility = typical fluctuation; Sharpe = risk‑adjusted return.")

st.markdown("### Executive Summary")
summary_bits = [
    f"Score **{port_score:.1f}** → **{score_label(port_score)}**",
    f"Diversification **{DIV:.2f}**",
    f"Max Drawdown **{mdd*100:.1f}%**",
    f"Volatility **{vol_ann*100:.1f}%**",
]
st.markdown("- " + "\n- ".join(summary_bits))

st.markdown("### Portfolio Diagnosis")
diag = []
if max_w >= 0.35:
    diag.append(f"High single‑name concentration (**{max_w*100:.1f}%** max weight).")
if sector_mix.iloc[0] >= 40:
    diag.append(f"Sector concentration risk: **{sector_mix.index[0]} {sector_mix.iloc[0]:.1f}%**.")
if not np.isnan(avg_corr) and avg_corr >= 0.6:
    diag.append(f"High internal correlation (**{avg_corr:.2f}**), diversification benefit is limited.")
if vol_ann >= 0.25:
    diag.append(f"Volatility is elevated (**{vol_ann*100:.1f}%** annualized).")
if mdd <= -0.30:
    diag.append(f"Large historical drawdown (**{mdd*100:.1f}%**).")
if sharpe >= 1.0:
    diag.append("Risk‑adjusted performance is strong (Sharpe ≥ 1).")
elif sharpe <= 0.2:
    diag.append("Risk‑adjusted performance is weak (Sharpe ≤ 0.2).")
if not diag:
    diag.append("Overall risk profile looks balanced with no major red flags.")
st.markdown("- " + "\n- ".join(diag))

st.markdown("### Actionable Improvements")
actions = []
if max_w >= 0.35:
    actions.append("Reduce the largest position to lower single‑name risk.")
if sector_mix.iloc[0] >= 40:
    actions.append(f"Add holdings outside **{sector_mix.index[0]}** to balance sector exposure.")
if not np.isnan(avg_corr) and avg_corr >= 0.6:
    actions.append("Add low‑correlation assets to improve diversification.")
if vol_ann >= 0.25:
    actions.append("Add defensive or lower‑volatility names to smooth returns.")
if not actions:
    actions.append("Current structure looks balanced; focus on quality/valuation upgrades.")
st.markdown("- " + "\n- ".join(actions))

st.markdown("### What Would Improve the Score Most?")
driver_weights = pd.Series({"Fundamentals": wf, "Technicals": wt, "Macro": wm})
drivers = pd.DataFrame({
    "Driver": ["Fundamentals", "Technicals", "Macro", "Diversification"],
    "Current": [
        float(np.nanmean(per_name["FUND_score"])),
        float(np.nanmean(per_name["TECH_score"])),
        float(MACRO),
        float(DIV),
    ],
    "Weight": [wf, wt, wm, 0.05],
})
drivers["Potential_Impact"] = drivers["Weight"] * (1.0 - drivers["Current"].clip(0, 1))
drivers = drivers.sort_values("Potential_Impact", ascending=False)
st.dataframe(drivers.round(4), use_container_width=True)
st.caption(
    "Potential impact estimates where improving a weaker, heavier‑weighted driver would lift the score most."
)

st.markdown("### Specific Improvement Actions")
factor_actions = []
if not fdf_all.empty:
    f_means = fdf_all.mean().sort_values()
    for col in f_means.index[:3]:
        if col.endswith("_z") and f_means[col] < 0:
            label = col.replace("_z","").replace("returnOnEquity","ROE").replace("returnOnAssets","ROA")
            label = label.replace("profitMargins","Profit Margin").replace("grossMargins","Gross Margin")
            label = label.replace("operatingMargins","Operating Margin").replace("ebitdaMargins","EBITDA Margin")
            label = label.replace("trailingPE","P/E").replace("forwardPE","Forward P/E")
            label = label.replace("enterpriseToEbitda","EV/EBITDA").replace("debtToEquity","Debt/Equity")
            label = label.replace("revenueGrowth","Revenue Growth").replace("earningsGrowth","Earnings Growth")
            label = label.replace("fcfYield","FCF Yield")
            direction = "increase exposure to names with stronger" if "PE" not in label and "Debt" not in label else "reduce exposure to expensive/leverage‑heavy names"
            factor_actions.append(f"- **{label}:** {direction}.")
if fdf_all.empty:
    factor_actions.append("- Fundamentals data is sparse; add more established, well‑covered stocks.")

tech_means = tech_all[[c for c in tech_all.columns if c.endswith("_z")]].mean()
if not tech_means.empty:
    if tech_means.min() < 0:
        weakest = tech_means.idxmin().replace("_z","")
        factor_actions.append(f"- **Technicals:** weakest factor is `{weakest}`; favor stronger momentum/trend names.")

if factor_actions:
    st.markdown("\n".join(factor_actions))
tabs = st.tabs(["Cumulative", "Volatility (60d) & Sharpe", "Drawdown"])
with tabs[0]:
    st.subheader("Cumulative growth (set = 1.0)")
    st.line_chart(pd.DataFrame({"Portfolio cumulative": eq}), use_container_width=True)
    st.caption("Growth of 1.0 invested, using your current weights over the chosen history.")
    st.markdown(
        "- Shows the compounded performance of the portfolio.  \n"
        "- Steeper slope = faster growth; flat/declining = weak period."
    )
with tabs[1]:
    st.subheader("Volatility & rolling Sharpe (60-day)")
    vol60 = port_r.rolling(60).std()*np.sqrt(252)
    sharpe60 = (port_r.rolling(60).mean()/port_r.rolling(60).std())*np.sqrt(252)
    st.line_chart(pd.DataFrame({"Volatility 60d (ann.)": vol60, "Sharpe 60d": sharpe60}), use_container_width=True)
    st.caption("Lower volatility & higher Sharpe are preferred.")
    st.markdown(
        "- **Volatility** = typical fluctuation; lower is smoother.  \n"
        "- **Sharpe** = return per unit of risk; higher is better."
    )
with tabs[2]:
    st.subheader("Drawdown")
    roll_max = eq.cummax(); dd = eq/roll_max - 1
    st.line_chart(pd.DataFrame({"Drawdown": dd}), use_container_width=True)
    st.caption("Depth of falls from prior peaks (risk perspective).")
    st.markdown(
        "- Drawdown shows the percent drop from the last peak.  \n"
        "- Deeper, longer drawdowns indicate higher risk."
    )
