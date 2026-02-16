# pages/2_Rate_My_Portfolio.py
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional
from app_utils import (
    inject_css, brand_header, yf_symbol, fetch_prices_chunked_with_fallback,
    fetch_macro_pack, score_universe_panel, build_universe, fetch_sector,
    SCORING_CONFIG, score_label, FACTOR_LABELS, weighted_series_mean, portfolio_signal_score
)

st.set_page_config(page_title="Rate My Portfolio", layout="wide")
inject_css()
brand_header("Rate My Portfolio")
st.page_link("pages/5_Stock_Battle.py", label="Compare holdings in Stock Battle")
st.page_link("pages/4_Build_My_Portfolio.py", label="Need a starter plan? Build My Portfolio")

CURRENCY_MAP = {"$":"USD","€":"EUR","£":"GBP","CHF":"CHF","C$":"CAD","A$":"AUD","¥":"JPY"}
def _safe_num(x): return pd.to_numeric(x, errors="coerce")

def normalize_percents_to_100(p: pd.Series) -> pd.Series:
    p = _safe_num(p).fillna(0.0)
    s = p.sum()
    if s <= 0: return p
    return (p / s) * 100.0

def sync_percent_amount(df: pd.DataFrame, total: Optional[float], mode: str) -> pd.DataFrame:
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

def _pretty_factor(col: str) -> str:
    return FACTOR_LABELS.get(col, col.replace("_z", "").replace("_", " ").title())

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
    msg.info("Step 3/4: loading macro regime.")
    macro_pack = fetch_macro_pack(period="6mo", interval="1d")
    prog.progress(65)
    pct.markdown("**Progress: 65%**")

    msg.info("Step 4/4: scoring universe with the shared engine.")
    score_pack = score_universe_panel(
        panel_all,
        target_count=target_count,
        macro_pack=macro_pack,
        min_fund_cols=SCORING_CONFIG["min_fund_cols"],
    )
    out_all = score_pack["out"]
    fdf_all = score_pack["fund"]
    tech_all = score_pack["tech"]
    score_weights = score_pack["weights"]
    wf = score_weights["FUND_score"]
    wt = score_weights["TECH_score"]
    wm = score_weights["MACRO_score"]
    MACRO = float(macro_pack["macro"])
    prog.progress(85)
    prog.progress(100)
    pct.markdown("**Progress: 100%**")
    status.update(label="Done!", state="complete")

st.markdown(
    f'<div class="banner">Peers loaded: <b>{len(panel_all)}</b> / <b>{target_count}</b> '
    f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True
)
if out_all.empty:
    st.error("Unable to compute scores for this universe.")
    st.stop()

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
eff_w = per_name["weight"].where(per_name["COMPOSITE"].notna(), 0.0).fillna(0.0)
eff_w = eff_w / (eff_w.sum() or 1.0)
per_name["weight_eff"] = eff_w
per_name["weighted_composite"] = per_name["COMPOSITE"] * per_name["weight_eff"]
port_pack = portfolio_signal_score(
    out_all["COMPOSITE"],
    per_name["COMPOSITE"],
    weights_named,
    diversification=DIV,
    diversification_bonus_points=5.0,
)
port_signal = float(port_pack["signal"]) if pd.notna(port_pack["signal"]) else np.nan
signal_pct = float(port_pack["signal_percentile"]) if pd.notna(port_pack["signal_percentile"]) else np.nan
port_score = float(port_pack["final_score"]) if pd.notna(port_pack["final_score"]) else np.nan
div_bonus_pts = float(port_pack["diversification_bonus"])
if pd.isna(port_score):
    st.error("Portfolio score unavailable due to missing holdings factor coverage.")
    st.stop()

st.markdown("## 🧺 Portfolio — Scores")
a,b,c,d = st.columns(4)
a.metric("Portfolio Score (0–100)", f"{port_score:.1f}")
b.metric("Signal Percentile vs Peers", f"{signal_pct:.1f}")
c.metric("Macro (Multi-signal)", f"{MACRO:.3f}")
d.metric("Diversification", f"{DIV:.3f}")
st.caption(
    f"Interpretation: **{score_label(port_score)}**. "
    "Score is peer-relative percentile of your weighted signal, then adjusted by diversification."
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

conf_row = out_all.reindex(tickers_shown)["CONFIDENCE"].fillna(0.0)
if conf_row.size and weights_named.sum() > 0:
    port_conf = float(np.average(conf_row, weights=weights_named.reindex(conf_row.index).fillna(0.0)))
else:
    port_conf = float(conf_row.mean()) if conf_row.size else 0.0
st.caption(f"Confidence: {port_conf:.0f}/100 based on data coverage and peer sample size.")

with st.expander("How to read this portfolio analysis"):
    st.markdown(
        "- **Portfolio Score** blends Fundamentals, Technicals, Macro, plus a small Diversification bonus.  \n"
        "- **Signal percentile** shows where your weighted portfolio signal sits vs the peer universe.  \n"
        "- **Macro** reflects the broader risk regime (VIX, USD, rates, credit, gold).  \n"
        "- **Diversification** rewards balanced sector mix, low concentration, and lower correlations.  \n"
        "- **Confidence** depends on peer size + data coverage; lower confidence means interpret cautiously."
    )

st.markdown("### Signal Decomposition")
fund_level = weighted_series_mean(per_name["FUND_score"], weights_named)
tech_level = weighted_series_mean(per_name["TECH_score"], weights_named)
macro_level = weighted_series_mean(per_name["MACRO_score"], weights_named)
fund_contrib = float(0.0 if pd.isna(fund_level) else fund_level * wf)
tech_contrib = float(0.0 if pd.isna(tech_level) else tech_level * wt)
macro_contrib = float(0.0 if pd.isna(macro_level) else macro_level * wm)
st.dataframe(
    pd.DataFrame(
        {
            "Component": [
                "Fundamentals contribution",
                "Technicals contribution",
                "Macro contribution",
                "Weighted signal",
                "Signal percentile",
                "Diversification bonus (points)",
            ],
            "Contribution": [fund_contrib, tech_contrib, macro_contrib, port_signal, signal_pct, div_bonus_pts],
        }
    ).set_index("Component").round(4),
    use_container_width=True,
)
st.caption("Contributions are coverage-adjusted; missing factor data no longer gets treated as zero weight.")


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

st.markdown("### Factor Tilt (Portfolio‑Weighted)")
tilt_notes = []
if not fdf_all.empty:
    fcols = [c for c in fdf_all.columns if c.endswith("_z")]
    f_hold = fdf_all.reindex(tickers_shown)[fcols].dropna(how="all")
    if not f_hold.empty:
        w = weights_named.reindex(f_hold.index).fillna(0.0)
        f_tilt = (f_hold.mul(w, axis=0)).sum() / (w.sum() or 1.0)
        pos = f_tilt.sort_values(ascending=False).head(3)
        neg = f_tilt.sort_values(ascending=True).head(3)
        tilt_notes.append("**Top positive tilts:**")
        for k, v in pos.items():
            tilt_notes.append(f"- {_pretty_factor(k)}: **{v:+.2f}z**")
        tilt_notes.append("**Top negative tilts:**")
        for k, v in neg.items():
            tilt_notes.append(f"- {_pretty_factor(k)}: **{v:+.2f}z**")
if tilt_notes:
    st.markdown("\n".join(tilt_notes))
else:
    st.caption("Not enough fundamentals data to compute factor tilts.")

st.markdown("### Risk Contribution")
if ret.shape[1] >= 2:
    cov = ret.cov() * 252.0
    w = weights_named.reindex(cov.index).fillna(0.0).values
    w = w / (w.sum() or 1.0)
    port_var = float(w.T @ cov.values @ w)
    if port_var > 0:
        mrc = cov.values @ w
        rc = w * mrc / port_var
        rc_df = pd.DataFrame({
            "Ticker": cov.index,
            "Risk Contribution %": rc * 100.0,
        }).sort_values("Risk Contribution %", ascending=False)
        st.dataframe(rc_df.round(2), use_container_width=True)
        st.caption("Risk contribution shows which holdings drive portfolio volatility the most.")
    else:
        st.caption("Risk contribution unavailable (insufficient variance).")
else:
    st.caption("Risk contribution requires at least two holdings with data.")

st.markdown("### Holdings Scorecard")
hold_scores = out_all.reindex(tickers_shown)[["COMPOSITE","RATING_0_100"]].copy()
hold_scores["Recommendation"] = hold_scores["RATING_0_100"].apply(score_label)
hold_scores["Weight %"] = (weights_named.reindex(tickers_shown).values * 100.0)
hold_scores = hold_scores.reset_index().rename(columns={"index": "Ticker"})
hold_scores = hold_scores.sort_values("Weight %", ascending=False)
st.dataframe(hold_scores.round(3), use_container_width=True)
st.caption("Holdings are ranked by weight; ratings are relative to the peer set.")

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
vol_ann = float(port_r.std() * np.sqrt(252)) if len(port_r) else np.nan
sharpe = float((port_r.mean() / (port_r.std() or 1e-9)) * np.sqrt(252)) if len(port_r) else np.nan
downside = port_r[port_r < 0]
down_std = float(downside.std()) if len(downside) else np.nan
sortino = float((port_r.mean() / (down_std or 1e-9)) * np.sqrt(252)) if len(port_r) else np.nan
var_95 = float(np.percentile(port_r, 5)) if len(port_r) else np.nan
cvar_95 = float(port_r[port_r <= var_95].mean()) if len(port_r) else np.nan
st.markdown("### Risk Snapshot")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Max Drawdown", f"{mdd*100:.1f}%")
r2.metric("Volatility (ann.)", f"{vol_ann*100:.1f}%" if not np.isnan(vol_ann) else "N/A")
r3.metric("Sharpe (ann.)", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
r4.metric("Sortino (ann.)", f"{sortino:.2f}" if not np.isnan(sortino) else "N/A")
st.caption(
    "Max drawdown = worst peak‑to‑trough loss; volatility = typical fluctuation; "
    "Sharpe = risk‑adjusted return; Sortino penalizes only downside volatility."
)

st.markdown("### Tail Risk")
t1, t2 = st.columns(2)
t1.metric("VaR 95% (daily)", f"{var_95*100:.2f}%" if not np.isnan(var_95) else "N/A")
t2.metric("CVaR 95% (daily)", f"{cvar_95*100:.2f}%" if not np.isnan(cvar_95) else "N/A")
st.caption("VaR/CVaR estimate potential daily loss in the worst 5% of days.")

with st.expander("How to interpret risk metrics"):
    st.markdown(
        "- **Volatility** = typical fluctuation of returns; higher = bumpier ride.  \n"
        "- **Max Drawdown** = worst peak‑to‑trough loss; gauges pain in bad regimes.  \n"
        "- **Sharpe** = return per unit of total risk; higher is better.  \n"
        "- **Sortino** = return per unit of downside risk; useful if you care about losses more than volatility.  \n"
        "- **VaR/CVaR** = tail loss estimates; CVaR is the average loss in the worst 5% of days."
    )

st.markdown("### Benchmark Comparison (SPY)")
bench_prices, _ = fetch_prices_chunked_with_fallback(["SPY"], period="1y", interval="1d", chunk=1, retries=2, sleep_between=0.2, singles_pause=0.2)
if not bench_prices.empty and "SPY" in bench_prices.columns:
    bench_r = bench_prices["SPY"].pct_change().dropna()
    aligned = port_r.align(bench_r, join="inner")
    if not aligned[0].empty and not aligned[1].empty:
        pr, br = aligned
        beta = float(np.cov(pr, br)[0,1] / (np.var(br) or 1e-9))
        alpha = float((pr.mean() - beta*br.mean()) * 252)
        tracking_err = float((pr - br).std() * np.sqrt(252))
        info_ratio = float(((pr.mean() - br.mean()) / (tracking_err or 1e-9)) * np.sqrt(252))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta vs SPY", f"{beta:.2f}")
        c2.metric("Alpha (ann.)", f"{alpha*100:.2f}%")
        c3.metric("Tracking Error", f"{tracking_err*100:.2f}%")
        c4.metric("Info Ratio", f"{info_ratio:.2f}")

        bench_eq = (1 + br).cumprod()
        st.line_chart(pd.DataFrame({"Portfolio": eq.reindex(bench_eq.index), "SPY": bench_eq}), use_container_width=True)
        st.caption("Benchmark chart compares cumulative performance vs SPY over the same period.")
        st.markdown(
            "- **Beta** = sensitivity to the benchmark.  \n"
            "- **Alpha** = return unexplained by benchmark exposure.  \n"
            "- **Tracking Error** = variability of active returns.  \n"
            "- **Information Ratio** = active return per unit of tracking error."
        )
    else:
        st.caption("Benchmark data unavailable for comparison.")
else:
    st.caption("Benchmark data unavailable for comparison.")

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
def _z_to_unit(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return float(0.5 + 0.5 * np.tanh(float(x)))

fund_level_z = weighted_series_mean(per_name["FUND_score"], weights_named)
tech_level_z = weighted_series_mean(per_name["TECH_score"], weights_named)
drivers = pd.DataFrame({
    "Driver": ["Fundamentals", "Technicals", "Macro", "Diversification"],
    "Current": [
        _z_to_unit(fund_level_z),
        _z_to_unit(tech_level_z),
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
            label = _pretty_factor(col)
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

st.markdown("### Suggested Additions (factor‑based)")
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
weights_norm = weights_named / (weights_named.sum() or 1.0)
fund_weak = _weighted_mean_z(fdf_all.reindex(tickers_shown), weights_norm)
tech_weak = _weighted_mean_z(tech_all.reindex(tickers_shown), weights_norm)
weak_factors = pd.concat([fund_weak, tech_weak]).sort_values().head(3)

if weak_factors.empty:
    st.caption("Not enough factor data to generate recommendations.")
else:
    weak_labels = [ _pretty_factor(k) for k in weak_factors.index ]
    st.caption("Weakest factors: " + ", ".join(weak_labels) + ".")
    if st.button("Generate recommendations", type="secondary", use_container_width=True):
        factor_z = pd.concat([fdf_all, tech_all[[c for c in tech_all.columns if c.endswith("_z")]]], axis=1)
        cand = factor_z.reindex(out_all.index).drop(index=tickers_shown, errors="ignore")
        use_cols = [c for c in weak_factors.index if c in cand.columns]
        if not use_cols:
            st.caption("Candidate data is sparse for weak factors.")
        else:
            cand["improve_score"] = cand[use_cols].mean(axis=1)
            cand = cand.dropna(subset=["improve_score"]).sort_values("improve_score", ascending=False).head(10)
            cand = cand.join(out_all[["RATING_0_100","COMPOSITE"]], how="left")
            cand["Sector"] = [fetch_sector(t) for t in cand.index]
            why = []
            for t in cand.index:
                top = cand.loc[t, use_cols].sort_values(ascending=False).head(2)
                why.append(", ".join([_pretty_factor(k) for k in top.index]))
            cand["Why"] = why
            show_cols = ["Sector","improve_score","RATING_0_100","Why"]
            st.dataframe(cand[show_cols].rename(columns={"RATING_0_100":"Score (0–100)"}).round(3), use_container_width=True)
            st.caption(
                "Ideas are ranked by strength on your weakest factors using the current peer universe. "
                "Use as a starting point, not investment advice."
            )
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
