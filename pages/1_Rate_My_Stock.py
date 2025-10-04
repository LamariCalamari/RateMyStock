import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from app_utils import (
    inject_css, brand_header, yf_symbol, ema, rsi, macd,
    zscore_series, percentile_rank, fetch_prices_chunked_with_fallback,
    fetch_fundamentals_simple, technical_scores, fetch_vix_series, macro_from_vix,
    fundamentals_interpretation, build_universe,
    # NEW:
    fetch_company_statements, statement_metrics, interpret_statement_metrics,
    tidy_statement_for_display, _CURRENCY_SYMBOL
)

st.set_page_config(page_title="Rate My — Stock", layout="wide")
inject_css()
brand_header("Rate My Stock")

# -------------------- Inputs --------------------
c_in_left, c_in_mid, c_in_right = st.columns([1,2,1])
with c_in_mid:
    ticker = st.text_input(" ", "AAPL", label_visibility="collapsed",
                           placeholder="Type a ticker (e.g., AAPL)")

with st.expander("Advanced settings", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        universe_mode = st.selectbox(
            "Peer universe",
            ["Auto by index membership","S&P 500","Dow 30","NASDAQ 100","Custom (paste list)"], index=0
        )
    with c2:
        peer_n = st.slider("Peer sample size", 30, 300, 180, 10)
    with c3:
        history = st.selectbox("History for signals", ["1y","2y"], index=0)
    c4,c5,c6 = st.columns(3)
    with c4: w_f = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.5, 0.05)
    with c5: w_t = st.slider("Weight: Technicals",   0.0, 1.0, 0.40, 0.05)
    with c6: w_m = st.slider("Weight: Macro (VIX)",  0.0, 1.0, 0.10, 0.05)
    custom_raw = st.text_area("Custom peers (comma-separated)", "") \
                  if universe_mode=="Custom (paste list)" else ""

user_tickers = [yf_symbol(x) for x in ticker.split(",") if x.strip()]
if not user_tickers:
    st.info("Enter a ticker above to run the rating.")
    st.stop()

# -------------------- Pipeline --------------------
with st.status("Crunching the numbers…", expanded=True) as status:
    prog = st.progress(0)

    status.update(label="Building peer universe…")
    universe, label = build_universe(user_tickers, universe_mode, peer_n, custom_raw)
    target_count = peer_n if universe_mode!="Custom (paste list)" else len(universe)
    prog.progress(10)

    status.update(label="Downloading prices (chunked + retries)…")
    prices, ok = fetch_prices_chunked_with_fallback(
        universe, period=history, interval="1d",
        chunk=20, retries=4, sleep_between=1.0, singles_pause=1.1
    )
    if not ok:
        st.error("No peer prices loaded.")
        st.stop()
    panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size>0}
    prog.progress(45)

    status.update(label="Computing technicals…")
    tech = technical_scores(panel)
    for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        if col in tech.columns:
            tech[f"{col}_z"] = zscore_series(tech[col])
    TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech.columns]].mean(axis=1)
    prog.progress(65)

    status.update(label="Fetching fundamentals…")
    fund_raw = fetch_fundamentals_simple(list(panel.keys()))
    fdf = pd.DataFrame(index=fund_raw.index)
    for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
                "grossMargins","operatingMargins","ebitdaMargins"]:
        if col in fund_raw.columns:
            fdf[f"{col}_z"] = zscore_series(fund_raw[col])
    for col in ["trailingPE","forwardPE","debtToEquity"]:
        if col in fund_raw.columns:
            fdf[f"{col}_z"] = zscore_series(-fund_raw[col])
    FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fund_raw.index)
    prog.progress(85)

    status.update(label="Assessing macro regime…")
    vix_series = fetch_vix_series(period="6mo", interval="1d")
    MACRO, vix_last, vix_ema20, vix_gap = macro_from_vix(vix_series)
    prog.progress(100)
    status.update(label="Done!", state="complete")

st.markdown(
    f'<div class="banner">Peers loaded: <b>{len(panel)}</b> / <b>{target_count}</b> '
    f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True
)

idx = pd.Index(list(panel.keys()))
out = pd.DataFrame(index=idx)
out["FUND_score"] = FUND_score.reindex(idx).fillna(0.0)
out["TECH_score"] = TECH_score.reindex(idx).fillna(0.0)
out["MACRO_score"]= MACRO
wsum = (w_f + w_t + w_m) or 1.0
wf, wt, wm = w_f/wsum, w_t/wsum, w_m/wsum
out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wm*out["MACRO_score"]
out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])
out["RECO"] = out["RATING_0_100"].apply(
    lambda x: "Strong Buy" if x>=80 else "Buy" if x>=60 else "Hold" if x>=40 else "Sell" if x>=20 else "Strong Sell"
)

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
    "FUND_score":"Fundamentals","TECH_score":"Technicals","MACRO_score":"Macro (VIX)",
    "COMPOSITE":"Composite","RATING_0_100":"Score (0–100)","RECO":"Recommendation"
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
        c3.markdown(f'<div class="kpi-card"><div>Macro (VIX)</div><div class="kpi-num">{table.loc[t,"MACRO_score"]:.3f}</div></div>', unsafe_allow_html=True)

        # Fundamentals table
        st.markdown("#### Fundamentals — peer-relative z-scores")
        fshow = pd.DataFrame({
            "Revenue growth (z)": fdf.loc[t, "revenueGrowth_z"] if "revenueGrowth_z" in fdf.columns else np.nan,
            "Earnings growth (z)": fdf.loc[t, "earningsGrowth_z"] if "earningsGrowth_z" in fdf.columns else np.nan,
            "ROE (z)": fdf.loc[t, "returnOnEquity_z"] if "returnOnEquity_z" in fdf.columns else np.nan,
            "Profit margin (z)": fdf.loc[t, "profitMargins_z"] if "profitMargins_z" in fdf.columns else np.nan,
            "Gross margin (z)": fdf.loc[t, "grossMargins_z"] if "grossMargins_z" in fdf.columns else np.nan,
            "Operating margin (z)": fdf.loc[t, "operatingMargins_z"] if "operatingMargins_z" in fdf.columns else np.nan,
            "EBITDA margin (z)": fdf.loc[t, "ebitdaMargins_z"] if "ebitdaMargins_z" in fdf.columns else np.nan,
            "PE (z, lower better)": fdf.loc[t, "trailingPE_z"] if "trailingPE_z" in fdf.columns else np.nan,
            "Forward PE (z, lower better)": fdf.loc[t, "forwardPE_z"] if "forwardPE_z" in fdf.columns else np.nan,
            "Debt/Equity (z, lower better)": fdf.loc[t, "debtToEquity_z"] if "debtToEquity_z" in fdf.columns else np.nan,
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
        st.markdown("#### Macro (VIX) — level & trend")
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
            st.markdown(f"- **Level:** {level_txt}.  \n- **Trend:** {trend_txt}.")
        else:
            st.info("VIX unavailable — Macro defaults to neutral.")

        # -------------------- NEW: Financial Statements & Interpretation --------------------
        st.markdown("---")
        st.markdown("### 📑 Financial statements & interpretation")

        stmts = fetch_company_statements(t)
        cur_code = stmts.get("currency") or "USD"
        cur_sym  = _CURRENCY_SYMBOL.get(cur_code, cur_code)

        freq = st.radio(f"{t}: Statement frequency", ["Annual","Quarterly"], horizontal=True, key=f"freq_{t}")
        if freq == "Annual":
            inc, bs, cf = stmts["income"], stmts["balance"], stmts["cashflow"]
            take_annual = 4
            inc_d = tidy_statement_for_display(inc, take_annual)
            bs_d  = tidy_statement_for_display(bs,  take_annual)
            cf_d  = tidy_statement_for_display(cf,  take_annual)
        else:
            inc, bs, cf = stmts["income_q"], stmts["balance_q"], stmts["cashflow_q"]
            take_q = 8
            inc_d = tidy_statement_for_display(inc, take_q)
            bs_d  = tidy_statement_for_display(bs,  take_q)
            cf_d  = tidy_statement_for_display(cf,  take_q)

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

        # Per-ticker CSV export (unchanged)
        row = {
            "ticker": t,
            "fundamentals_score": float(table.loc[t, "FUND_score"]),
            "technicals_score":   float(table.loc[t, "TECH_score"]),
            "macro_score":        float(table.loc[t, "MACRO_score"]),
            "composite":          float(table.loc[t, "COMPOSITE"]),
            "score_0_100":        float(table.loc[t, "RATING_0_100"]),
            "recommendation":     str(table.loc[t, "RECO"]),
        }
        export_df = pd.DataFrame([row])
        st.download_button("⬇️ Download this breakdown (CSV)",
                           data=export_df.to_csv(index=False).encode(),
                           file_name=f"{t}_breakdown.csv", mime="text/csv", use_container_width=True)

        # Charts (unchanged)
        def draw_stock_charts(series: pd.Series):
            if series is None or series.empty:
                st.info("Not enough history to show charts.")
                return
            st.subheader("📈 Price & EMAs")
            e20, e50 = ema(series,20), ema(series,50)
            price_df = pd.DataFrame({"Close": series, "EMA20": e20, "EMA50": e50})
            st.line_chart(price_df, use_container_width=True)
            st.caption("If price is **above EMA50/EMA20**, trend bias is positive; **below** suggests a headwind.")

            st.subheader("📉 MACD")
            line, sig, hist = macd(series)
            st.line_chart(pd.DataFrame({"MACD line": line, "Signal": sig}), use_container_width=True)
            st.bar_chart(pd.DataFrame({"Histogram": hist}), use_container_width=True)
            st.caption("Rising histogram above zero → momentum building; falling below zero → fading.")

            st.subheader("🔁 RSI (14)")
            st.line_chart(pd.DataFrame({"RSI(14)": rsi(series)}), use_container_width=True)
            st.caption(">70 = overbought • <30 = oversold • around 50 = neutral trend strength.")

            st.subheader("🚀 12-month momentum")
            if len(series) > 252:
                mom12 = series/series.shift(253)-1.0
                st.line_chart(pd.DataFrame({"12m momentum": mom12}), use_container_width=True)
                st.caption("Positive vs one year ago → outperformance; negative → underperformance.")
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