# pages/1_Rate_My_Stock.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from app_utils import (
    inject_css, brand_header, yf_symbol, ema, rsi, macd,
    zscore_series, percentile_rank, fetch_prices_chunked_with_fallback,
    fetch_fundamentals_simple, technical_scores, fetch_vix_series, macro_from_vix,
    fundamentals_interpretation, build_universe
)

st.set_page_config(page_title="Rate My Stock", layout="wide")
inject_css()
brand_header("Rate My Stock")

c_in_left, c_in_mid, c_in_right = st.columns([1,2,1])
with c_in_mid:
    ticker = st.text_input(" ", "AAPL", label_visibility="collapsed",
                           placeholder="Type a ticker (e.g., AAPL)", key="ticker_in")

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

with st.status("Crunching the numbers…", expanded=True) as status:
    prog = st.progress(0)

    status.update(label="Building peer universe…")
    universe, label = build_universe(user_tickers, universe_mode, peer_n, custom_raw)
    target_count = peer_n if universe_mode!="Custom (paste list)" else len(universe)
    prog.progress(10)

    status.update(label="Downloading prices (chunked + retries)…")
    prices, ok = fetch_prices_chunked_with_fallback(
        universe, period=history, interval="1d",
        chunk=25, retries=3, sleep_between=0.35, singles_pause=0.20
    )
    if not ok:
        st.error("No peer prices loaded."); st.stop()
    panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size>0}
    prog.progress(50)

    status.update(label="Computing technicals…")
    tech = technical_scores(panel)
    for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        if col in tech.columns:
            tech[f"{col}_z"] = zscore_series(tech[col])
    TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech.columns]].mean(axis=1)
    prog.progress(75)

    status.update(label="Fetching fundamentals…")
    fund_raw = fetch_fundamentals_simple(list(panel.keys()))
    fdf = pd.DataFrame(index=fund_raw.index)
    for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
                "grossMargins","operatingMargins","ebitdaMargins"]:
        if col in fund_raw.columns: fdf[f"{col}_z"] = zscore_series(fund_raw[col])
    for col in ["trailingPE","forwardPE","debtToEquity"]:
        if col in fund_raw.columns: fdf[f"{col}_z"] = zscore_series(-fund_raw[col])
    FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fund_raw.index)
    prog.progress(92)

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

st.markdown("## 🔎 Why this rating?")
all_rows=[]
for t in show_idx:
    reco = table.loc[t,"RECO"]; sc = table.loc[t,"RATING_0_100"]
    with st.expander(f"{t} — {reco} (Score: {sc:.1f})", expanded=True):

        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="kpi-card"><div>Fundamentals</div><div class="kpi-num">{table.loc[t,"FUND_score"]:.3f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div>Technicals</div><div class="kpi-num">{table.loc[t,"TECH_score"]:.3f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div>Macro (VIX)</div><div class="kpi-num">{table.loc[t,"MACRO_score"]:.3f}</div></div>', unsafe_allow_html=True)

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

        # Per-ticker export
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

        # Charts
        try:
            px2 = yf.Ticker(t).history(period="2y", interval="1d")["Close"].dropna()
            if px2.size > 0:
                series = px2
            else:
                series = panel[t]
        except Exception:
            series = panel[t] if t in panel else None

        if series is not None and not series.empty:
            st.subheader("📈 Price & EMAs")
            e20, e50 = ema(series,20), ema(series,50)
            st.line_chart(pd.DataFrame({"Close": series, "EMA20": e20, "EMA50": e50}), use_container_width=True)
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

        all_rows.append(row)

if all_rows:
    df_all = pd.DataFrame(all_rows)
    st.markdown("### Export all shown tickers")
    st.download_button("⬇️ Download all (CSV)",
                       data=df_all.to_csv(index=False).encode(),
                       file_name="stock_breakdowns.csv", mime="text/csv", use_container_width=True)
    xlsx_all = io.BytesIO()
    with pd.ExcelWriter(xlsx_all, engine="openpyxl") as w:
        df_all.to_excel(w, index=False, sheet_name="Breakdowns")
    st.download_button("⬇️ Download all (Excel)",
                       data=xlsx_all.getvalue(),
                       file_name="stock_breakdowns.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
