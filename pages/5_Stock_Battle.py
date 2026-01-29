# pages/5_Stock_Battle.py
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import (
    inject_css, brand_header, yf_symbol, fetch_prices_chunked_with_fallback,
    fetch_fundamentals_simple, technical_scores, zscore_series, percentile_rank,
    fetch_vix_series, fetch_gold_series, fetch_dxy_series, fetch_tnx_series,
    fetch_credit_ratio_series, macro_from_signals, build_universe,
    SCORING_CONFIG, score_label
)

st.set_page_config(page_title="Stock Battle", layout="wide")
inject_css()
brand_header("Stock Battle")
st.caption("Compare two stocks or a stock vs an index with the same scoring system.")

def _cap_z(s: pd.Series, cap: float | None = None) -> pd.Series:
    cap = SCORING_CONFIG["z_cap"] if cap is None else cap
    return s.clip(-cap, cap)

def _composite_row(row: pd.Series, weights: dict) -> float:
    total = sum(w for k, w in weights.items() if pd.notna(row.get(k)))
    if total == 0:
        return np.nan
    return float(sum(row.get(k) * w for k, w in weights.items() if pd.notna(row.get(k))) / total)

def _coverage(df: pd.DataFrame, ticker: str, cols: list[str]) -> float:
    if not cols or ticker not in df.index:
        return 0.0
    return float(df.loc[ticker, cols].notna().mean())

def _risk_stats(px: pd.Series) -> dict:
    if px is None or px.dropna().empty:
        return {
            "ret": np.nan, "vol": np.nan, "mdd": np.nan,
            "sharpe": np.nan, "sortino": np.nan,
        }
    px = px.dropna()
    r = px.pct_change().dropna()
    if r.empty:
        return {
            "ret": np.nan, "vol": np.nan, "mdd": np.nan,
            "sharpe": np.nan, "sortino": np.nan,
        }
    total_ret = float(px.iloc[-1] / px.iloc[0] - 1.0)
    vol = float(r.std() * np.sqrt(252))
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    sharpe = float((r.mean() / (r.std() or 1e-9)) * np.sqrt(252))
    downside = r[r < 0]
    down_std = float(downside.std()) if len(downside) else np.nan
    sortino = float((r.mean() / (down_std or 1e-9)) * np.sqrt(252)) if not np.isnan(down_std) else np.nan
    return {"ret": total_ret, "vol": vol, "mdd": mdd, "sharpe": sharpe, "sortino": sortino}

def _normalize(px: pd.Series) -> pd.Series:
    px = px.dropna()
    if px.empty:
        return px
    return px / px.iloc[0]

battle_mode = st.segmented_control(
    "Battle mode",
    options=["Stock vs Stock", "Stock vs Index"],
    default="Stock vs Stock",
)

c1, c2 = st.columns(2)
with c1:
    ticker_a = st.text_input("Stock A", value="AAPL")
with c2:
    if battle_mode == "Stock vs Stock":
        ticker_b = st.text_input("Stock B", value="MSFT")
        index_choice = None
    else:
        index_choice = st.selectbox("Index benchmark", ["S&P 500", "NASDAQ 100", "Dow 30"], index=0)
        ticker_b = None

basis = "Same peer set (direct)"
if battle_mode == "Stock vs Stock":
    basis = st.segmented_control(
        "Comparison basis",
        options=["Same peer set (direct)", "Each vs own peers (fair cross‑industry)"],
        default="Each vs own peers (fair cross‑industry)",
        help="Fair mode scores each stock against its own industry/sector peers.",
    )
    if basis.startswith("Each vs own"):
        st.info("Fair mode compares each stock to its own peer group to avoid cross‑industry bias.")
else:
    st.caption("Index battles use the index peer set for the benchmark side.")

with st.expander("Advanced settings", expanded=False):
    universe_mode = st.selectbox(
        "Peer universe",
        ["Industry (auto fallback)", "S&P 500", "NASDAQ 100", "Dow 30"],
        index=0,
    )
    history = st.selectbox("History for signals", ["1y", "2y"], index=0)
    peer_n = st.slider("Peer sample size", 30, 300, 180, 10)
    fast_mode = st.checkbox("Fast mode (fewer peers, faster load)", value=False)
    w_f = st.slider("Weight: Fundamentals", 0.0, 1.0, SCORING_CONFIG["weights"]["fund"], 0.05)
    w_t = st.slider("Weight: Technicals", 0.0, 1.0, SCORING_CONFIG["weights"]["tech"], 0.05)
    w_m = st.slider("Weight: Macro", 0.0, 1.0, SCORING_CONFIG["weights"]["macro"], 0.05)

ticker_a = yf_symbol(ticker_a)
ticker_b = yf_symbol(ticker_b) if ticker_b else None

user_tickers = [t for t in [ticker_a, ticker_b] if t]
if not user_tickers:
    st.info("Enter a ticker to start the comparison.")
    st.stop()
if ticker_b and ticker_b == ticker_a:
    st.warning("Choose two different tickers for a head-to-head battle.")
    st.stop()

sig = (
    battle_mode,
    tuple(user_tickers),
    index_choice or "",
    universe_mode,
    basis,
    history, peer_n, fast_mode,
    w_f, w_t, w_m,
)
if st.session_state.get("battle_sig") != sig:
    st.session_state["battle_ready"] = False

start_col = st.columns([1, 2, 1])[1]
with start_col:
    start = st.button("Start battle", type="primary", use_container_width=True)
if start:
    st.session_state["battle_ready"] = True
    st.session_state["battle_sig"] = sig

if not st.session_state.get("battle_ready"):
    st.info("Adjust settings, then click **Start battle**.")
    st.stop()

peer_n_eff = min(peer_n, 120) if fast_mode else peer_n
mode = index_choice if battle_mode == "Stock vs Index" else universe_mode

def _compute_universe(tickers: list[str], mode: str, macro_value: float) -> dict:
    universe, label = build_universe(tickers, mode, peer_n_eff, "")
    target_count = peer_n_eff if mode != "Custom (paste list)" else len(universe)
    prices, ok = fetch_prices_chunked_with_fallback(
        universe, period=history, interval="1d",
        chunk=20, retries=4, sleep_between=0.6, singles_pause=0.6
    )
    if not ok:
        return {"ok": False}
    panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size > 0}
    tech = technical_scores(panel)
    for col in ["dma_gap", "macd_hist", "rsi_strength", "mom12m"]:
        if col in tech.columns:
            tech[f"{col}_z"] = _cap_z(zscore_series(tech[col]))
    TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech.columns]].mean(axis=1)

    fund_raw = fetch_fundamentals_simple(list(panel.keys()))
    core_fund = [
        "revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
        "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield",
        "trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"
    ]
    core_present = [c for c in core_fund if c in fund_raw.columns]
    if core_present:
        min_fund_cols = SCORING_CONFIG["min_fund_cols"]
        fund_raw = fund_raw[fund_raw[core_present].notna().sum(axis=1) >= min_fund_cols]
    fdf = pd.DataFrame(index=fund_raw.index)
    for col in ["revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
                "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield"]:
        if col in fund_raw.columns:
            fdf[f"{col}_z"] = _cap_z(zscore_series(fund_raw[col]))
    for col in ["trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"]:
        if col in fund_raw.columns:
            fdf[f"{col}_z"] = _cap_z(zscore_series(-fund_raw[col]))
    FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(dtype=float)

    idx = pd.Index(list(panel.keys()))
    out = pd.DataFrame(index=idx)
    out["FUND_score"] = FUND_score.reindex(idx)
    out["TECH_score"] = TECH_score.reindex(idx)
    out["MACRO_score"] = macro_value

    wsum = (w_f + w_t + w_m) or 1.0
    wf, wt, wm = w_f / wsum, w_t / wsum, w_m / wsum
    weights = {"FUND_score": wf, "TECH_score": wt, "MACRO_score": wm}
    out["COMPOSITE"] = out.apply(_composite_row, axis=1, weights=weights)
    ratings = percentile_rank(out["COMPOSITE"].dropna())
    out["RATING_0_100"] = ratings.reindex(out.index)
    out["RECO"] = out["RATING_0_100"].apply(score_label)

    tech_cols = [c for c in tech.columns if c.endswith("_z")]
    fund_cols = [c for c in fdf.columns if c.endswith("_z")]
    peer_factor = min(len(out["COMPOSITE"].dropna()) / max(target_count, 1), 1.0)
    cw = SCORING_CONFIG["confidence_weights"]
    out["CONFIDENCE"] = [
        100.0 * (cw["peer"] * peer_factor + cw["fund"] * _coverage(fdf, t, fund_cols) + cw["tech"] * _coverage(tech, t, tech_cols))
        for t in out.index
    ]

    return {
        "ok": True,
        "out": out,
        "panel": panel,
        "label": label,
        "target_count": target_count,
        "fdf": fdf,
        "tech": tech,
    }

with st.status("Building comparison...", expanded=True) as status:
    prog = st.progress(0)
    msg = st.empty()

    msg.info("Step 1/3: loading macro regime.")
    vix_series = fetch_vix_series(period="6mo", interval="1d")
    gold_series = fetch_gold_series(period="6mo", interval="1d")
    dxy_series = fetch_dxy_series(period="6mo", interval="1d")
    tnx_series = fetch_tnx_series(period="6mo", interval="1d")
    credit_series = fetch_credit_ratio_series(period="6mo", interval="1d")
    macro_value = macro_from_signals(vix_series, gold_series, dxy_series, tnx_series, credit_series)["macro"]
    prog.progress(20)

    if basis == "Each vs own peers (fair cross‑industry)" and battle_mode == "Stock vs Stock":
        msg.info("Step 2/3: building peers + signals for Stock A.")
        if universe_mode != "Industry (auto fallback)":
            st.caption("Using industry/sector fallback for fair comparison.")
        res_a = _compute_universe([ticker_a], "Industry (auto fallback)", macro_value)
        prog.progress(60)
        msg.info("Step 3/3: building peers + signals for Stock B.")
        res_b = _compute_universe([ticker_b], "Industry (auto fallback)", macro_value)
        prog.progress(100)
        status.update(label="Done!", state="complete")
    else:
        msg.info("Step 2/3: building peers + signals.")
        res_all = _compute_universe(user_tickers, mode, macro_value)
        prog.progress(100)
        status.update(label="Done!", state="complete")

if basis == "Each vs own peers (fair cross‑industry)" and battle_mode == "Stock vs Stock":
    if not res_a.get("ok") or not res_b.get("ok"):
        st.error("Failed to load peer data for one or both stocks.")
        st.stop()
    out_a, panel_a = res_a["out"], res_a["panel"]
    out_b, panel_b = res_b["out"], res_b["panel"]
    label_a, label_b = res_a["label"], res_b["label"]
    target_count_a, target_count_b = res_a["target_count"], res_b["target_count"]
    fdf_a, tech_a = res_a["fdf"], res_a["tech"]
    fdf_b, tech_b = res_b["fdf"], res_b["tech"]
    if ticker_a not in out_a.index or ticker_b not in out_b.index:
        st.error("Missing data for one or both tickers in their peer sets.")
        st.stop()
else:
    if not res_all.get("ok"):
        st.error("Failed to load peer data.")
        st.stop()
    out, panel = res_all["out"], res_all["panel"]
    label, target_count = res_all["label"], res_all["target_count"]
    fdf, tech = res_all["fdf"], res_all["tech"]

if basis == "Each vs own peers (fair cross‑industry)" and battle_mode == "Stock vs Stock":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="banner">{ticker_a}: peers loaded <b>{len(panel_a)}</b> / <b>{target_count_a}</b> '
            f'&nbsp;|&nbsp; Peer set: <b>{label_a}</b></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="banner">{ticker_b}: peers loaded <b>{len(panel_b)}</b> / <b>{target_count_b}</b> '
            f'&nbsp;|&nbsp; Peer set: <b>{label_b}</b></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f'<div class="banner">Peers loaded: <b>{len(panel)}</b> / <b>{target_count}</b> '
        f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>',
        unsafe_allow_html=True,
    )

    missing = [t for t in user_tickers if t not in out.index]
    if missing:
        st.error("Missing data for: " + ", ".join(missing))
        st.stop()

    peer_only = out.drop(index=user_tickers, errors="ignore")

def _summary_row(ticker: str, out_df: pd.DataFrame, panel_dict: dict) -> dict:
    row = out_df.loc[ticker]
    px = panel_dict.get(ticker)
    risk = _risk_stats(px)
    return {
        "Score": float(row["RATING_0_100"]),
        "Recommendation": row["RECO"],
        "Fundamentals": float(row["FUND_score"]),
        "Technicals": float(row["TECH_score"]),
        "Macro": float(row["MACRO_score"]),
        "Confidence": float(row["CONFIDENCE"]),
        "1y Return": risk["ret"],
        "Volatility": risk["vol"],
        "Max Drawdown": risk["mdd"],
        "Sharpe": risk["sharpe"],
        "Sortino": risk["sortino"],
    }

if basis == "Each vs own peers (fair cross‑industry)" and battle_mode == "Stock vs Stock":
    summary_a = _summary_row(ticker_a, out_a, panel_a)
    summary_b = _summary_row(ticker_b, out_b, panel_b)
else:
    summary_a = _summary_row(ticker_a, out, panel)
    summary_b = _summary_row(ticker_b, out, panel) if ticker_b else None

st.markdown("## Head-to-head")
left, right = st.columns(2)
with left:
    st.subheader(ticker_a)
    score_label_txt = "Score (0-100 vs own peers)" if basis.startswith("Each vs own") else "Score (0-100)"
    st.metric(score_label_txt, f"{summary_a['Score']:.1f}", help=summary_a["Recommendation"])
    st.caption(f"Recommendation: {summary_a['Recommendation']}")
    st.metric("Fundamentals", f"{summary_a['Fundamentals']:.3f}")
    st.metric("Technicals", f"{summary_a['Technicals']:.3f}")
    st.metric("Macro", f"{summary_a['Macro']:.3f}")
    st.metric("Confidence", f"{summary_a['Confidence']:.0f}/100")
    st.metric("1y Return", f"{summary_a['1y Return']*100:.1f}%" if not np.isnan(summary_a["1y Return"]) else "N/A")
    st.metric("Volatility", f"{summary_a['Volatility']*100:.1f}%" if not np.isnan(summary_a["Volatility"]) else "N/A")
    st.metric("Max Drawdown", f"{summary_a['Max Drawdown']*100:.1f}%" if not np.isnan(summary_a["Max Drawdown"]) else "N/A")
    st.metric("Sharpe", f"{summary_a['Sharpe']:.2f}" if not np.isnan(summary_a["Sharpe"]) else "N/A")
    st.metric("Sortino", f"{summary_a['Sortino']:.2f}" if not np.isnan(summary_a["Sortino"]) else "N/A")

with right:
    if battle_mode == "Stock vs Stock":
        st.subheader(ticker_b)
        st.metric(score_label_txt, f"{summary_b['Score']:.1f}", help=summary_b["Recommendation"])
        st.caption(f"Recommendation: {summary_b['Recommendation']}")
        st.metric("Fundamentals", f"{summary_b['Fundamentals']:.3f}")
        st.metric("Technicals", f"{summary_b['Technicals']:.3f}")
        st.metric("Macro", f"{summary_b['Macro']:.3f}")
        st.metric("Confidence", f"{summary_b['Confidence']:.0f}/100")
        st.metric("1y Return", f"{summary_b['1y Return']*100:.1f}%" if not np.isnan(summary_b["1y Return"]) else "N/A")
        st.metric("Volatility", f"{summary_b['Volatility']*100:.1f}%" if not np.isnan(summary_b["Volatility"]) else "N/A")
        st.metric("Max Drawdown", f"{summary_b['Max Drawdown']*100:.1f}%" if not np.isnan(summary_b["Max Drawdown"]) else "N/A")
        st.metric("Sharpe", f"{summary_b['Sharpe']:.2f}" if not np.isnan(summary_b["Sharpe"]) else "N/A")
        st.metric("Sortino", f"{summary_b['Sortino']:.2f}" if not np.isnan(summary_b["Sortino"]) else "N/A")
    else:
        st.subheader(f"{index_choice} (peer average)")
        if peer_only.empty:
            st.caption("Not enough peer data to compute index averages.")
        else:
            st.metric("Score (0-100)", f"{peer_only['RATING_0_100'].median():.1f}")
            st.metric("Fundamentals", f"{peer_only['FUND_score'].mean():.3f}")
            st.metric("Technicals", f"{peer_only['TECH_score'].mean():.3f}")
            st.metric("Macro", f"{peer_only['MACRO_score'].mean():.3f}")
            st.metric("Confidence", f"{out['CONFIDENCE'].mean():.0f}/100")

if battle_mode == "Stock vs Stock":
    diff = summary_a["Score"] - summary_b["Score"]
    if abs(diff) < 2:
        st.info("Verdict: close call. Scores are within 2 points.")
    elif diff > 0:
        st.success(f"Verdict: {ticker_a} leads by {diff:.1f} points.")
    else:
        st.success(f"Verdict: {ticker_b} leads by {abs(diff):.1f} points.")
else:
    if not peer_only.empty:
        delta = summary_a["Score"] - float(peer_only["RATING_0_100"].median())
        msg = "above" if delta > 0 else "below"
        st.success(f"{ticker_a} is {msg} the index median by {abs(delta):.1f} points.")

st.markdown("### Factor comparison")
factor_map = {
    "revenueGrowth_z": "Revenue growth",
    "earningsGrowth_z": "Earnings growth",
    "returnOnEquity_z": "ROE",
    "returnOnAssets_z": "ROA",
    "profitMargins_z": "Profit margin",
    "grossMargins_z": "Gross margin",
    "operatingMargins_z": "Operating margin",
    "ebitdaMargins_z": "EBITDA margin",
    "fcfYield_z": "FCF yield",
    "trailingPE_z": "P/E (lower is better)",
    "forwardPE_z": "Forward P/E (lower is better)",
    "enterpriseToEbitda_z": "EV/EBITDA (lower is better)",
    "debtToEquity_z": "Debt/Equity (lower is better)",
    "dma_gap_z": "Price vs EMA50",
    "macd_hist_z": "MACD momentum",
    "rsi_strength_z": "RSI strength",
    "mom12m_z": "12m momentum",
}

def _z_val(df: pd.DataFrame, t: str, col: str) -> float:
    if t in df.index and col in df.columns and pd.notna(df.loc[t, col]):
        return float(df.loc[t, col])
    return np.nan

rows = []
for col, label_txt in factor_map.items():
    rows.append({
        "Factor": label_txt,
        ticker_a: _z_val(fdf_a if col in fdf_a.columns else tech_a, ticker_a, col) if basis.startswith("Each vs own") and battle_mode == "Stock vs Stock"
                 else _z_val(fdf if col in fdf.columns else tech, ticker_a, col),
        (ticker_b or index_choice): _z_val(fdf_b if col in fdf_b.columns else tech_b, ticker_b, col) if (ticker_b and basis.startswith("Each vs own") and battle_mode == "Stock vs Stock")
                 else (_z_val(fdf if col in fdf.columns else tech, ticker_b, col) if ticker_b else np.nan),
    })

factor_df = pd.DataFrame(rows).set_index("Factor")
if battle_mode == "Stock vs Stock":
    if not basis.startswith("Each vs own"):
        factor_df["Delta"] = factor_df[ticker_a] - factor_df[ticker_b]
else:
    factor_df = factor_df[[ticker_a]]
st.dataframe(factor_df.round(3), use_container_width=True)
if basis.startswith("Each vs own") and battle_mode == "Stock vs Stock":
    st.caption("Z-scores are relative to each stock’s own peer group, so compare directionally.")
else:
    st.caption("Positive z-scores are stronger vs peers; valuation/leverage are inverted so lower is better.")

st.markdown("### Peer score distribution")
bins = [0, 20, 40, 60, 80, 100]
if basis.startswith("Each vs own") and battle_mode == "Stock vs Stock":
    c1, c2 = st.columns(2)
    with c1:
        counts, edges = np.histogram(out_a.drop(index=ticker_a)["RATING_0_100"].dropna(), bins=bins)
        hist = pd.DataFrame({"Score band": [f"{edges[i]}-{edges[i+1]}" for i in range(len(counts))], "Count": counts})
        st.caption(f"{ticker_a} peer distribution")
        st.bar_chart(hist.set_index("Score band"), use_container_width=True)
    with c2:
        counts, edges = np.histogram(out_b.drop(index=ticker_b)["RATING_0_100"].dropna(), bins=bins)
        hist = pd.DataFrame({"Score band": [f"{edges[i]}-{edges[i+1]}" for i in range(len(counts))], "Count": counts})
        st.caption(f"{ticker_b} peer distribution")
        st.bar_chart(hist.set_index("Score band"), use_container_width=True)
else:
    if not peer_only.empty:
        counts, edges = np.histogram(peer_only["RATING_0_100"].dropna(), bins=bins)
        hist = pd.DataFrame({"Score band": [f"{edges[i]}-{edges[i+1]}" for i in range(len(counts))], "Count": counts})
        st.bar_chart(hist.set_index("Score band"), use_container_width=True)
    else:
        st.caption("Not enough peers to show score distribution.")

st.markdown("### Performance comparison")
price_data = {}
if basis.startswith("Each vs own") and battle_mode == "Stock vs Stock":
    if ticker_a in panel_a:
        price_data[ticker_a] = _normalize(panel_a[ticker_a])
    if ticker_b in panel_b:
        price_data[ticker_b] = _normalize(panel_b[ticker_b])
else:
    for t in user_tickers:
        if t in panel:
            price_data[t] = _normalize(panel[t])

index_proxy = {"S&P 500": "SPY", "NASDAQ 100": "QQQ", "Dow 30": "DIA"}
if battle_mode == "Stock vs Index" and index_choice in index_proxy:
    proxy = index_proxy[index_choice]
    proxy_px, _ = fetch_prices_chunked_with_fallback([proxy], period=history, interval="1d", chunk=1, retries=2)
    if not proxy_px.empty and proxy in proxy_px.columns:
        price_data[proxy] = _normalize(proxy_px[proxy])

if price_data:
    perf_df = pd.DataFrame(price_data).dropna(how="all")
    st.line_chart(perf_df, use_container_width=True)
    st.caption("Normalized performance (start = 1.0) for an apples-to-apples comparison.")
else:
    st.caption("Not enough data for performance chart.")

st.markdown("### Export")
export = pd.DataFrame({
    "Ticker": [ticker_a] + ([ticker_b] if ticker_b else []),
    "Score": [summary_a["Score"]] + ([summary_b["Score"]] if summary_b else []),
    "Fundamentals": [summary_a["Fundamentals"]] + ([summary_b["Fundamentals"]] if summary_b else []),
    "Technicals": [summary_a["Technicals"]] + ([summary_b["Technicals"]] if summary_b else []),
    "Macro": [summary_a["Macro"]] + ([summary_b["Macro"]] if summary_b else []),
    "Confidence": [summary_a["Confidence"]] + ([summary_b["Confidence"]] if summary_b else []),
})
st.download_button(
    "Download battle summary (CSV)",
    data=export.to_csv(index=False).encode(),
    file_name="stock_battle_summary.csv",
    mime="text/csv",
    use_container_width=True,
)

with st.expander("View peer list", expanded=False):
    if basis.startswith("Each vs own") and battle_mode == "Stock vs Stock":
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"{ticker_a} peers: {len(panel_a)}")
            st.dataframe(pd.DataFrame({"Ticker": list(panel_a.keys())}), use_container_width=True)
        with c2:
            st.caption(f"{ticker_b} peers: {len(panel_b)}")
            st.dataframe(pd.DataFrame({"Ticker": list(panel_b.keys())}), use_container_width=True)
    else:
        st.caption(f"Peers shown: {len(panel)}")
        st.dataframe(pd.DataFrame({"Ticker": list(panel.keys())}), use_container_width=True)
