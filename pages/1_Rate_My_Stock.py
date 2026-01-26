# pages/Rate_My_Stock.py
import io
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from typing import Dict, List

from app_utils import (
    inject_css, brand_header, yf_symbol, ema, rsi, macd,
    zscore_series, percentile_rank, fetch_prices_chunked_with_fallback,
    fetch_fundamentals_simple, technical_scores, fetch_vix_series, fetch_gold_series,
    fetch_dxy_series, fetch_tnx_series, fetch_credit_ratio_series, macro_from_signals,
    fundamentals_interpretation, build_universe,
    # NEW:
    fetch_company_statements, statement_metrics, interpret_statement_metrics,
    build_compact_statements, _CURRENCY_SYMBOL
)

st.set_page_config(page_title="Rate My — Stock", layout="wide")
inject_css()
brand_header("Rate My Stock")

# -------------------- Inputs --------------------
c_in_left, c_in_mid, c_in_right = st.columns([1,2,1])
with c_in_mid:
    ticker = st.text_input(" ", "AAPL", label_visibility="collapsed",
                           placeholder="Type a ticker (e.g., AAPL)")

def _looks_like_ticker(val: str) -> bool:
    if not val:
        return False
    if " " in val:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9\.\-]{1,10}", val))

resolved = None
query = ticker.strip()
if st.session_state.get("last_query") != query:
    st.session_state["resolved_ticker"] = None
    st.session_state["last_query"] = query
if query and ("," not in query) and (not _looks_like_ticker(query)):
    st.caption("Looks like a company name. Search and confirm the ticker below.")
    with st.expander("Search company name", expanded=True):
        results = []
        try:
            search = yf.Search(query)
            results = search.quotes or []
        except Exception:
            results = []
        if results:
            best = results[0]
            best_symbol = best.get("symbol")
            best_name = best.get("shortname") or best.get("longname") or ""
            best_ex = best.get("exchange") or "N/A"
            if best_symbol:
                st.info(f"Best match: {best_symbol} — {best_name} ({best_ex})")
                if st.button("Use best match"):
                    resolved = best_symbol
                    st.session_state["resolved_ticker"] = resolved
        if results:
            options = [
                (r.get("symbol"), f"{r.get('symbol')} — {r.get('shortname') or r.get('longname') or ''} ({r.get('exchange') or 'N/A'})")
                for r in results
                if r.get("symbol")
            ]
            labels = [o[1] for o in options]
            picked = st.selectbox("Select company", labels, index=0)
            picked_symbol = dict(options).get(picked)
            if st.button("Use selected ticker"):
                resolved = picked_symbol
                st.session_state["resolved_ticker"] = resolved
        else:
            st.info("No matches found. Try a different company name.")

resolved = st.session_state.get("resolved_ticker") if resolved is None else resolved
if resolved:
    st.success(f"Using ticker: {resolved}")
    ticker = resolved

peer_label_ph = None

with st.expander("Advanced settings", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        universe_mode = st.selectbox(
            "Peer universe",
            ["Auto (industry → sector → index)","Custom (paste list)"], index=0
        )
    with c2:
        peer_n = st.slider("Peer sample size", 30, 300, 180, 10)
    with c3:
        history = st.selectbox("History for signals", ["1y","2y"], index=0)
    c4,c5,c6 = st.columns(3)
    with c4: w_f = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.50, 0.05)
    with c5: w_t = st.slider("Weight: Technicals",   0.0, 1.0, 0.45, 0.05)
    with c6: w_m = st.slider("Weight: Macro (Multi-signal)",  0.0, 1.0, 0.05, 0.05)
    custom_raw = st.text_area("Custom peers (comma-separated)", "") \
                  if universe_mode=="Custom (paste list)" else ""
    fast_mode = st.checkbox("Fast mode (fewer peers, faster load)", value=False)
    st.caption(
        "Industry mode picks same-industry peers; if too few, it falls back to sector, then index."
    )
    peer_label_ph = st.empty()

user_tickers = [yf_symbol(x) for x in ticker.split(",") if x.strip()]
if not user_tickers:
    st.info("Enter a ticker above to run the rating.")
    st.stop()

sig = (
    tuple(user_tickers), universe_mode, peer_n, history, w_f, w_t, w_m,
    custom_raw, fast_mode
)
if st.session_state.get("analysis_sig") != sig:
    st.session_state["analysis_ready"] = False

start_col = st.columns([1,2,1])[1]
with start_col:
    start = st.button("Start analysis", type="primary", use_container_width=True)
if start:
    st.session_state["analysis_ready"] = True
    st.session_state["analysis_sig"] = sig

if not st.session_state.get("analysis_ready"):
    st.info("Adjust settings, then click **Start analysis**.")
    st.stop()

def _cap_z(s: pd.Series, cap: float = 3.0) -> pd.Series:
    return s.clip(-cap, cap)

def _composite_row(row: pd.Series, weights: Dict[str, float]) -> float:
    total = sum(w for k, w in weights.items() if pd.notna(row.get(k)))
    if total == 0:
        return np.nan
    return float(sum(row.get(k) * w for k, w in weights.items() if pd.notna(row.get(k))) / total)

def _coverage(df: pd.DataFrame, ticker: str, cols: List[str]) -> float:
    if not cols or ticker not in df.index:
        return 0.0
    return float(df.loc[ticker, cols].notna().mean())

# -------------------- Pipeline --------------------
with st.status("Crunching the numbers…", expanded=True) as status:
    prog = st.progress(0)
    pct = st.empty()
    msg = st.empty()

    status.update(label="Building peer universe…")
    msg.info("Step 1/5: selecting the best peer set for comparison.")
    peer_n_eff = min(peer_n, 120) if fast_mode else peer_n
    universe, label = build_universe(user_tickers, universe_mode, peer_n_eff, custom_raw)
    if peer_label_ph:
        peer_label_ph.caption(f"Current peer set: {label}")
    target_count = peer_n_eff if universe_mode!="Custom (paste list)" else len(universe)
    prog.progress(10)
    pct.markdown("**Progress: 10%**")

    status.update(label="Downloading prices (chunked + retries)…")
    msg.info("Step 2/5: downloading price history (this is the slowest step).")
    prices, ok = fetch_prices_chunked_with_fallback(
        universe, period=history, interval="1d",
        chunk=20, retries=4, sleep_between=1.0, singles_pause=1.1
    )
    if not ok:
        st.error("No peer prices loaded.")
        st.stop()
    panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size>0}
    prog.progress(45)
    pct.markdown("**Progress: 45%**")

    status.update(label="Computing technicals…")
    msg.info("Step 3/5: calculating technical signals.")
    tech = technical_scores(panel)
    for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        if col in tech.columns:
            tech[f"{col}_z"] = _cap_z(zscore_series(tech[col]))
    TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech.columns]].mean(axis=1)
    prog.progress(65)
    pct.markdown("**Progress: 65%**")

    status.update(label="Fetching fundamentals…")
    msg.info("Step 4/5: loading fundamentals and quality metrics.")
    fund_raw = fetch_fundamentals_simple(list(panel.keys()))
    core_fund = [
        "revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
        "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield",
        "trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"
    ]
    core_present = [c for c in core_fund if c in fund_raw.columns]
    if core_present:
        min_fund_cols = 4
        fund_raw = fund_raw[fund_raw[core_present].notna().sum(axis=1) >= min_fund_cols]
    fdf = pd.DataFrame(index=fund_raw.index)
    for col in ["revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
                "profitMargins","grossMargins","operatingMargins","ebitdaMargins","fcfYield"]:
        if col in fund_raw.columns:
            fdf[f"{col}_z"] = _cap_z(zscore_series(fund_raw[col]))
    for col in ["trailingPE","forwardPE","enterpriseToEbitda","debtToEquity"]:
        if col in fund_raw.columns:
            fdf[f"{col}_z"] = _cap_z(zscore_series(-fund_raw[col]))
    FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fund_raw.index)
    prog.progress(85)
    pct.markdown("**Progress: 85%**")

    status.update(label="Assessing macro regime…")
    msg.info("Step 5/5: combining macro signals (VIX, USD, rates, credit, gold).")
    vix_series = fetch_vix_series(period="6mo", interval="1d")
    gold_series = fetch_gold_series(period="6mo", interval="1d")
    dxy_series = fetch_dxy_series(period="6mo", interval="1d")
    tnx_series = fetch_tnx_series(period="6mo", interval="1d")
    credit_series = fetch_credit_ratio_series(period="6mo", interval="1d")
    macro_pack = macro_from_signals(vix_series, gold_series, dxy_series, tnx_series, credit_series)
    MACRO = macro_pack["macro"]
    vix_last = macro_pack["vix_last"]
    vix_ema20 = macro_pack["vix_ema20"]
    vix_gap = macro_pack["vix_gap"]
    gold_ret = macro_pack["gold_ret"]
    dxy_ret = macro_pack["dxy_ret"]
    tnx_delta = macro_pack["tnx_delta"]
    credit_ret = macro_pack["credit_ret"]
    prog.progress(100)
    pct.markdown("**Progress: 100%**")
    status.update(label="Done!", state="complete")

st.markdown(
    f'<div class="banner">Peers loaded: <b>{len(panel)}</b> / <b>{target_count}</b> '
    f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True
)

with st.expander("View peer list", expanded=False):
    st.caption(f"Peers shown: {len(panel)}")
    st.dataframe(pd.DataFrame({"Ticker": list(panel.keys())}), use_container_width=True)

idx = pd.Index(list(panel.keys()))
out = pd.DataFrame(index=idx)
out["FUND_score"] = FUND_score.reindex(idx)
out["TECH_score"] = TECH_score.reindex(idx)
out["MACRO_score"]= MACRO
wsum = (w_f + w_t + w_m) or 1.0
wf, wt, wm = w_f/wsum, w_t/wsum, w_m/wsum
weights = {"FUND_score": wf, "TECH_score": wt, "MACRO_score": wm}
out["COMPOSITE"] = out.apply(_composite_row, axis=1, weights=weights)
ratings = percentile_rank(out["COMPOSITE"].dropna())
out["RATING_0_100"] = ratings.reindex(out.index)
def _reco(x):
    if pd.isna(x): return "Insufficient data"
    return "Strong Buy" if x>=80 else "Buy" if x>=60 else "Hold" if x>=40 else "Sell" if x>=20 else "Strong Sell"
out["RECO"] = out["RATING_0_100"].apply(_reco)

tech_cols = [c for c in tech.columns if c.endswith("_z")]
fund_cols = [c for c in fdf.columns if c.endswith("_z")]
peer_factor = min(len(out["COMPOSITE"].dropna()) / max(target_count, 1), 1.0)
out["CONFIDENCE"] = [
    100.0 * (0.4*peer_factor + 0.3*_coverage(fdf, t, fund_cols) + 0.3*_coverage(tech, t, tech_cols))
    for t in out.index
]

# 5Y momentum for SHOWN tickers only (quietly)
show_idx = [t for t in user_tickers if t in out.index]
tech = tech.copy()
for t in show_idx:
    try:
        px5 = yf.Ticker(t).history(period="5y", interval="1d")["Close"].dropna()
        if len(px5)>253:
            mom12 = px5.iloc[-1]/px5.iloc[-253]-1.0
            if t in tech.index:
                tech.loc[t,"mom12m"] = mom12
    except Exception:
        pass

table = out.reindex(show_idx).sort_values("RATING_0_100", ascending=False)

st.markdown("## 🏁 Ratings")
pretty = table.rename(columns={
    "FUND_score":"Fundamentals","TECH_score":"Technicals","MACRO_score":"Macro (Multi-signal)",
    "COMPOSITE":"Composite","RATING_0_100":"Score (0–100)","CONFIDENCE":"Confidence","RECO":"Recommendation"
})
st.dataframe(pretty.round(4), use_container_width=True)

# -------------------- Per-ticker deep dive --------------------
st.markdown("## 🔎 Why this rating?")
for t in show_idx:
    reco = table.loc[t,"RECO"]; sc = table.loc[t,"RATING_0_100"]
    with st.expander(f"{t} — {reco} (Score: {sc:.1f})", expanded=True):

        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="kpi-card"><div>Fundamentals</div><div class="kpi-num">{table.loc[t,"FUND_score"]:.3f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div>Technicals</div><div class="kpi-num">{table.loc[t,"TECH_score"]:.3f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div>Macro (Multi-signal)</div><div class="kpi-num">{table.loc[t,"MACRO_score"]:.3f}</div></div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {table.loc[t,'CONFIDENCE']:.0f}/100")

        # Fundamentals table
        st.markdown("#### Fundamentals — peer-relative z-scores")
        def _fget(col):
            return fdf.loc[t, col] if (t in fdf.index and col in fdf.columns) else np.nan
        fshow = pd.DataFrame({
            "Revenue growth (z)": _fget("revenueGrowth_z"),
            "Earnings growth (z)": _fget("earningsGrowth_z"),
            "ROE (z)": _fget("returnOnEquity_z"),
            "ROA (z)": _fget("returnOnAssets_z"),
            "Profit margin (z)": _fget("profitMargins_z"),
            "Gross margin (z)": _fget("grossMargins_z"),
            "Operating margin (z)": _fget("operatingMargins_z"),
            "EBITDA margin (z)": _fget("ebitdaMargins_z"),
            "PE (z, lower better)": _fget("trailingPE_z"),
            "Forward PE (z, lower better)": _fget("forwardPE_z"),
            "EV/EBITDA (z, lower better)": _fget("enterpriseToEbitda_z"),
            "FCF yield (z)": _fget("fcfYield_z"),
            "Debt/Equity (z, lower better)": _fget("debtToEquity_z"),
        }, index=[t]).T.rename(columns={t:"z-score"})
        st.dataframe(fshow.round(3), use_container_width=True)

        st.markdown("**Interpretation**")
        lines = fundamentals_interpretation(fdf.loc[t] if t in fdf.index else pd.Series(dtype=float))
        for L in lines: st.markdown(f"- {L}")

        # Technicals table
        st.markdown("#### Technicals")
        rsi_val = np.nan
        if (t in tech.index) and ("rsi_strength" in tech.columns) and pd.notna(tech.loc[t,"rsi_strength"]):
            rsi_val = 50 + 50*tech.loc[t,"rsi_strength"]
        tshow = pd.DataFrame({
            "Price vs EMA50 (gap)": tech.loc[t,"dma_gap"] if ("dma_gap" in tech.columns and t in tech.index) else np.nan,
            "MACD histogram": tech.loc[t,"macd_hist"] if ("macd_hist" in tech.columns and t in tech.index) else np.nan,
            "RSI (approx)": rsi_val,
            "12-mo momentum": tech.loc[t,"mom12m"] if ("mom12m" in tech.columns and t in tech.index) else np.nan,
        }, index=[t]).T.rename(columns={t:"value"})
        st.dataframe(tshow.round(3), use_container_width=True)

        notes=[]
        if "dma_gap" in tech.columns and t in tech.index and pd.notna(tech.loc[t,"dma_gap"]):
            if tech.loc[t,"dma_gap"] > 0.02: notes.append("Price above EMA50 → trend tailwind.")
            elif tech.loc[t,"dma_gap"] < -0.02: notes.append("Price below EMA50 → trend headwind.")
            else: notes.append("Price near EMA50.")
        if "macd_hist" in tech.columns and t in tech.index and pd.notna(tech.loc[t,"macd_hist"]):
            notes.append("MACD histogram positive → momentum building." if tech.loc[t,"macd_hist"]>0
                         else "MACD histogram negative → momentum fading.")
        if pd.notna(rsi_val):
            if rsi_val >= 65: notes.append(f"RSI ~{rsi_val:.0f} — strong/overbought.")
            elif rsi_val <= 35: notes.append(f"RSI ~{rsi_val:.0f} — weak/oversold.")
            else: notes.append(f"RSI ~{rsi_val:.0f} — neutral.")
        if "mom12m" in tech.columns and t in tech.index and pd.notna(tech.loc[t,"mom12m"]):
            notes.append("12m momentum positive." if tech.loc[t,"mom12m"]>0 else "12m momentum negative.")
        if notes:
            st.markdown("- " + "\n- ".join(notes))

        # Macro
        st.markdown("#### Macro (VIX + Gold + USD + Rates + Credit) — level & trend")
        if not np.isnan(vix_last):
            m1,m2,m3 = st.columns(3)
            m1.metric("VIX (last)", f"{vix_last:.2f}")
            m2.metric("VIX EMA20", f"{vix_ema20:.2f}")
            m3.metric("Gap vs EMA20", f"{(vix_gap*100):.1f}%")
            if vix_last <= 13: level_txt = "very calm (risk-friendly backdrop)"
            elif vix_last <= 18: level_txt = "calm (supportive for risk)"
            elif vix_last <= 24: level_txt = "elevated (more caution warranted)"
            else: level_txt = "stressed (risk-off backdrop)"
            if vix_gap > 0.03: trend_txt = "rising above its 20-day average (volatility building)"
            elif vix_gap < -0.03: trend_txt = "falling below its 20-day average (volatility easing)"
            else: trend_txt = "moving roughly in line with its 20-day average"
            st.markdown(f"- **VIX level:** {level_txt}.  \n- **VIX trend:** {trend_txt}.")
        else:
            st.info("VIX unavailable — Macro defaults to neutral.")

        if not np.isnan(gold_ret):
            g_txt = "risk-off bid" if gold_ret > 0.05 else "risk-on backdrop" if gold_ret < -0.05 else "neutral signal"
            st.markdown(f"- **Gold (3mo):** {gold_ret*100:.1f}% → {g_txt}.")
        if not np.isnan(dxy_ret):
            d_txt = "risk-off (USD bid)" if dxy_ret > 0.03 else "risk-on (USD weaker)" if dxy_ret < -0.03 else "neutral"
            st.markdown(f"- **USD (UUP, 3mo):** {dxy_ret*100:.1f}% → {d_txt}.")
        if not np.isnan(tnx_delta):
            r_txt = "rates rising" if tnx_delta > 0.25 else "rates falling" if tnx_delta < -0.25 else "rates steady"
            st.markdown(f"- **10Y yield (3mo):** {tnx_delta:+.2f} pp → {r_txt}.")
        if not np.isnan(credit_ret):
            c_txt = "credit improving" if credit_ret > 0.02 else "credit stress" if credit_ret < -0.02 else "neutral"
            st.markdown(f"- **Credit (HYG/LQD, 3mo):** {credit_ret*100:.1f}% → {c_txt}.")

        # -------------------- NEW: Financial Statements & Interpretation --------------------
        st.markdown("---")
        st.markdown("### 📑 Financial statements & interpretation")
        st.caption("Optional: click to load statements (slower).")

        show_statements = st.button("Load financial statements", key=f"stmts_{t}")
        if show_statements:
            stmts = fetch_company_statements(t)
            cur_code = stmts.get("currency") or "USD"
            cur_sym  = _CURRENCY_SYMBOL.get(cur_code, cur_code)

            freq = st.radio(f"{t}: Statement frequency", ["Annual","Quarterly"], horizontal=True, key=f"freq_{t}")
            if freq == "Annual":
                compact = build_compact_statements(stmts, freq="annual", take=4)
            else:
                compact = build_compact_statements(stmts, freq="quarterly", take=8)
            inc_d, bs_d, cf_d = compact["income"], compact["balance"], compact["cashflow"]

            # Scale to nice units for display
            inc_d, inc_unit = (inc_d/1, "")
            bs_d,  bs_unit  = (bs_d/1, "")
            cf_d,  cf_unit  = (cf_d/1, "")
            # Auto scale per table
            def _auto(df):
                if df is None or df.empty: return df, ""
                vals = pd.to_numeric(df.replace([np.inf,-np.inf], np.nan).stack(), errors="coerce").dropna()
                if vals.empty: return df, ""
                med = np.nanmedian(np.abs(vals))
                if med >= 1e9: return (df/1e9).round(2), " (billions)"
                if med >= 1e6: return (df/1e6).round(2), " (millions)"
                if med >= 1e3: return (df/1e3).round(2), " (thousands)"
                return df.round(2), ""
            inc_d, inc_unit = _auto(inc_d)
            bs_d,  bs_unit  = _auto(bs_d)
            cf_d,  cf_unit  = _auto(cf_d)

            tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Key ratios", "Narrative"])
            with tabs[0]:
                st.caption(f"Currency: **{cur_sym} {cur_code}** • Units: {inc_unit or 'raw'}")
                st.dataframe(inc_d, use_container_width=True)
            with tabs[1]:
                st.caption(f"Currency: **{cur_sym} {cur_code}** • Units: {bs_unit or 'raw'}")
                st.dataframe(bs_d, use_container_width=True)
            with tabs[2]:
                st.caption(f"Currency: **{cur_sym} {cur_code}** • Units: {cf_unit or 'raw'}")
                st.dataframe(cf_d, use_container_width=True)
            with tabs[3]:
                met = statement_metrics(stmts)
                ratios = pd.DataFrame({
                    "Revenue (last)": [met.get("revenue")],
                    "Revenue YoY": [met.get("revenue_yoy")],
                    "Revenue CAGR (3y)": [met.get("revenue_cagr3y")],
                    "Gross margin": [met.get("gross_margin")],
                    "Operating margin": [met.get("operating_margin")],
                    "Net margin": [met.get("net_margin")],
                    "FCF margin": [met.get("fcf_margin")],
                    "Current ratio": [met.get("current_ratio")],
                    "Quick ratio": [met.get("quick_ratio")],
                    "Debt/Equity": [met.get("debt_to_equity")],
                    "Interest coverage (EBIT/Int)": [met.get("interest_coverage")],
                    "ROE": [met.get("roe")],
                    "ROA": [met.get("roa")],
                }).T.rename(columns={0:"value"})
                # pretty formatting
                def fmt(x, pct=False):
                    if x is None or np.isnan(x): return "—"
                    return f"{x*100:.1f}%" if pct else f"{x:,.2f}"
                pretty_ratios = pd.DataFrame({
                    "value": [
                        fmt(met.get("revenue"), pct=False),
                        fmt(met.get("revenue_yoy"), pct=True),
                        fmt(met.get("revenue_cagr3y"), pct=True),
                        fmt(met.get("gross_margin"), pct=True),
                        fmt(met.get("operating_margin"), pct=True),
                        fmt(met.get("net_margin"), pct=True),
                        fmt(met.get("fcf_margin"), pct=True),
                        fmt(met.get("current_ratio"), pct=False),
                        fmt(met.get("quick_ratio"), pct=False),
                        fmt(met.get("debt_to_equity"), pct=False),
                        fmt(met.get("interest_coverage"), pct=False),
                        fmt(met.get("roe"), pct=True),
                        fmt(met.get("roa"), pct=True),
                    ]
                }, index=ratios.index)
                st.dataframe(pretty_ratios, use_container_width=True)

            with tabs[4]:
                met = statement_metrics(stmts)
                bullets = interpret_statement_metrics(met)
                st.markdown("> These points summarize what the **financial statements** say about growth, profitability, "
                            "cash generation, liquidity, and leverage — which underpin your **Fundamentals** score.")
                for b in bullets:
                    st.markdown(f"- {b}")

        # Per-ticker CSV export (expanded)
        def _tval(df: pd.DataFrame, col: str) -> float:
            if t in df.index and col in df.columns and pd.notna(df.loc[t, col]):
                return float(df.loc[t, col])
            return np.nan

        export = {
            "ticker": t,
            "peer_set": label,
            "peer_count": int(len(panel)),
            "fundamentals_score": float(table.loc[t, "FUND_score"]),
            "technicals_score":   float(table.loc[t, "TECH_score"]),
            "macro_score":        float(table.loc[t, "MACRO_score"]),
            "composite":          float(table.loc[t, "COMPOSITE"]),
            "score_0_100":        float(table.loc[t, "RATING_0_100"]),
            "confidence":         float(table.loc[t, "CONFIDENCE"]),
            "recommendation":     str(table.loc[t, "RECO"]),
            # Fundamentals z-scores
            "rev_growth_z":        _tval(fdf, "revenueGrowth_z"),
            "earnings_growth_z":   _tval(fdf, "earningsGrowth_z"),
            "roe_z":               _tval(fdf, "returnOnEquity_z"),
            "roa_z":               _tval(fdf, "returnOnAssets_z"),
            "profit_margin_z":     _tval(fdf, "profitMargins_z"),
            "gross_margin_z":      _tval(fdf, "grossMargins_z"),
            "operating_margin_z":  _tval(fdf, "operatingMargins_z"),
            "ebitda_margin_z":     _tval(fdf, "ebitdaMargins_z"),
            "pe_z":                _tval(fdf, "trailingPE_z"),
            "forward_pe_z":        _tval(fdf, "forwardPE_z"),
            "ev_ebitda_z":          _tval(fdf, "enterpriseToEbitda_z"),
            "fcf_yield_z":         _tval(fdf, "fcfYield_z"),
            "debt_equity_z":       _tval(fdf, "debtToEquity_z"),
            # Technicals (raw)
            "dma_gap":             _tval(tech, "dma_gap"),
            "macd_hist":           _tval(tech, "macd_hist"),
            "rsi_strength":        _tval(tech, "rsi_strength"),
            "mom12m":              _tval(tech, "mom12m"),
            # Macro context
            "vix_last":            float(vix_last) if not np.isnan(vix_last) else np.nan,
            "vix_gap":             float(vix_gap) if not np.isnan(vix_gap) else np.nan,
            "gold_3m_return":      float(gold_ret) if not np.isnan(gold_ret) else np.nan,
            "dxy_3m_return":       float(dxy_ret) if not np.isnan(dxy_ret) else np.nan,
            "tnx_3m_delta":        float(tnx_delta) if not np.isnan(tnx_delta) else np.nan,
            "credit_3m_return":    float(credit_ret) if not np.isnan(credit_ret) else np.nan,
        }

        export_df = pd.DataFrame([export])

        def _section(buf: io.StringIO, title: str, df: pd.DataFrame) -> None:
            buf.write(f"# {title}\n")
            df.to_csv(buf, index=True)
            buf.write("\n\n")

        buf = io.StringIO()
        _section(buf, "Summary", export_df.set_index("ticker"))

        fund_cols = [c for c in fdf.columns if c.endswith("_z")]
        if t in fdf.index and fund_cols:
            _section(buf, "Fundamentals (z-scores)", fdf.loc[[t], fund_cols].T)

        if t in tech.index:
            trow = tech.loc[[t]].T
            _section(buf, "Technicals", trow)

        macro_df = pd.DataFrame({
            "vix_last": [vix_last],
            "vix_gap": [vix_gap],
            "gold_3m_return": [gold_ret],
            "dxy_3m_return": [dxy_ret],
            "tnx_3m_delta": [tnx_delta],
            "credit_3m_return": [credit_ret],
        }).T.rename(columns={0: "value"})
        _section(buf, "Macro Context", macro_df)

        # Statements & ratios (annual only)
        stmts = fetch_company_statements(t)
        compact = build_compact_statements(stmts, freq="annual", take=4)
        if not compact["income"].empty:
            _section(buf, "Income Statement (annual)", compact["income"])
        if not compact["balance"].empty:
            _section(buf, "Balance Sheet (annual)", compact["balance"])
        if not compact["cashflow"].empty:
            _section(buf, "Cash Flow (annual)", compact["cashflow"])

        met = statement_metrics(stmts)
        ratios = pd.DataFrame({
            "Revenue (last)": [met.get("revenue")],
            "Revenue YoY": [met.get("revenue_yoy")],
            "Revenue CAGR (3y)": [met.get("revenue_cagr3y")],
            "Gross margin": [met.get("gross_margin")],
            "Operating margin": [met.get("operating_margin")],
            "Net margin": [met.get("net_margin")],
            "FCF margin": [met.get("fcf_margin")],
            "Current ratio": [met.get("current_ratio")],
            "Quick ratio": [met.get("quick_ratio")],
            "Debt/Equity": [met.get("debt_to_equity")],
            "Interest coverage (EBIT/Int)": [met.get("interest_coverage")],
            "ROE": [met.get("roe")],
            "ROA": [met.get("roa")],
        }).T.rename(columns={0: "value"})
        _section(buf, "Key Ratios", ratios)

        peers_df = pd.DataFrame({"peer_ticker": list(panel.keys())})
        _section(buf, "Peers Used", peers_df)

        st.download_button(
            "⬇️ Download full analysis (CSV)",
            data=buf.getvalue().encode(),
            file_name=f"{t}_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Charts (unchanged)
        def draw_stock_charts(series: pd.Series):
            if series is None or series.empty:
                st.info("Not enough history to show charts.")
                return
            st.subheader("📈 Price & EMAs")
            e20, e50 = ema(series,20), ema(series,50)
            price_df = pd.DataFrame({"Close": series, "EMA20": e20, "EMA50": e50})
            st.line_chart(price_df, use_container_width=True)
            st.caption(
                "EMA (exponential moving average) smooths price. "
                "EMA20 = short‑term trend, EMA50 = medium‑term trend."
            )
            st.markdown(
                "- If **price stays above EMA20/EMA50**, the trend is generally bullish.  \n"
                "- If **price crosses below**, it signals weakening momentum.  \n"
                "- When **EMA20 crosses above EMA50**, trend strength is improving (and vice‑versa)."
            )

            st.subheader("📉 MACD")
            line, sig, hist = macd(series)
            st.line_chart(pd.DataFrame({"MACD line": line, "Signal": sig}), use_container_width=True)
            st.bar_chart(pd.DataFrame({"Histogram": hist}), use_container_width=True)
            st.caption("MACD measures momentum by comparing a fast EMA to a slow EMA.")
            st.markdown(
                "- **MACD line** = momentum direction.  \n"
                "- **Signal line** = smoothed MACD for crossover signals.  \n"
                "- **Histogram** = MACD minus Signal (momentum acceleration).  \n"
                "- Cross above Signal + rising histogram → momentum building; below → momentum fading."
            )

            st.subheader("🔁 RSI (14)")
            st.line_chart(pd.DataFrame({"RSI(14)": rsi(series)}), use_container_width=True)
            st.caption("RSI is an oscillator that measures trend strength (0–100).")
            st.markdown(
                "- **>70** often means overbought (price extended).  \n"
                "- **<30** often means oversold (price stretched down).  \n"
                "- **~50** is neutral; rising RSI confirms uptrend strength."
            )

            st.subheader("🚀 12-month momentum")
            if len(series) > 252:
                mom12 = series/series.shift(253)-1.0
                st.line_chart(pd.DataFrame({"12m momentum": mom12}), use_container_width=True)
                st.caption("12m momentum compares today to ~1 year ago.")
                st.markdown(
                    "- **>0** = price above last year (relative strength).  \n"
                    "- **<0** = price below last year (weakness).  \n"
                    "- Rising momentum → improving trend strength."
                )
            else:
                st.info("Need > 1 year of data to show the 12-month momentum line.")

        try:
            px2 = yf.Ticker(t).history(period="2y", interval="1d")["Close"].dropna()
            if px2.size > 0:
                draw_stock_charts(px2)
            elif t in panel:
                draw_stock_charts(panel[t])
        except Exception:
            if t in panel:
                draw_stock_charts(panel[t])
