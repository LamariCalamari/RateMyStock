# app.py
# Full Streamlit stock rating app:
# - Robust downloads (yfinance) with retries
# - Fundamentals (valuation, growth, profitability, leverage, margins)
# - Technicals (EMA gap, MACD hist, RSI strength, 12m momentum)
# - Relative-to-benchmark (ratio, 12m rel mom, 60d beta & alpha)
# - Macro overlay (VIX→risk-on scalar)
# - Weights & toggles in sidebar
# - Results table + recommendation buckets
# - Price & EMA chart + optional ratio chart
# - CSV/XLSX download AND local Excel save (timestamped)

import io, os, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Stock Rating Model", layout="wide")
st.title("📊 Stock Rating Model (Fundamentals • Technicals • Relative • Macro)")
st.caption("Type tickers, pick a benchmark, tweak weights, hit **Run**. Exports CSV/XLSX and saves a timestamped Excel locally.")

# ---------------------------
# Helpers / math
# ---------------------------
def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def macro_overlay_score(vix_level: float) -> float:
    """Map VIX to [0,1] risk-on score. 15 → 1.0, 35 → 0.0 linearly."""
    if pd.isna(vix_level): return 0.5
    if vix_level <= 15: return 1.0
    if vix_level >= 35: return 0.0
    return 1.0 - (vix_level - 15) / 20.0

# ---------------------------
# Robust data fetchers
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_prices_per_ticker(tickers, period="2y", interval="1d", max_retries=3, sleep_s=0.7):
    frames = []
    ok, fail = [], []
    for t in tickers:
        success = False
        for k in range(max_retries):
            try:
                df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
                if not df.empty and "Close" in df.columns:
                    frames.append(df["Close"].rename(t))
                    ok.append(t)
                    success = True
                    break
            except Exception:
                pass
            time.sleep(sleep_s * (k + 1))
        if not success:
            fail.append(t)
    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return prices, ok, fail

@st.cache_data(show_spinner=False)
def fetch_price_series(ticker: str, period="2y", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if not df.empty and "Close" in df.columns:
            return df["Close"].rename(ticker)
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_vix_level() -> float:
    try:
        vix = yf.Ticker("^VIX").history(period="6mo", interval="1d")
        if not vix.empty: return float(vix["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

@st.cache_data(show_spinner=False)
def fetch_fundamentals(tickers):
    """Best-effort pull from Yahoo. Snapshot-style metrics for cross-sectional z-scores."""
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            try:
                inf = tk.info or {}
            except Exception:
                inf = {}

            # Financial statements (annual)
            try:
                fin = tk.financials
            except Exception:
                fin = None
            try:
                bs = tk.balance_sheet
            except Exception:
                bs = None

            # Valuation
            pe = inf.get("trailingPE", np.nan)
            ev = inf.get("enterpriseValue", np.nan)

            # Income statement lines
            ebitda = np.nan
            total_revenue_recent = total_revenue_prev = np.nan
            net_income_recent = net_income_prev = np.nan
            if fin is not None and not fin.empty:
                # EBITDA
                try:
                    s = fin.loc[fin.index.str.contains("EBITDA", case=False)].T.squeeze().dropna()
                    ebitda = float(s.iloc[0]) if len(s) else np.nan
                except Exception:
                    pass

                # Revenue & Net Income (latest, previous)
                def latest_two(name):
                    try:
                        s = fin.loc[fin.index.str.contains(name, case=False)].T.squeeze().dropna()
                        return (float(s.iloc[0]) if len(s) else np.nan,
                                float(s.iloc[1]) if len(s) > 1 else np.nan)
                    except Exception:
                        return (np.nan, np.nan)
                total_revenue_recent, total_revenue_prev = latest_two("Total Revenue")
                net_income_recent, net_income_prev     = latest_two("Net Income")

            # EV/EBITDA
            ev_ebitda = np.nan
            if pd.notna(ev) and pd.notna(ebitda) and ebitda not in (0, None):
                ev_ebitda = ev / ebitda

            # Growth
            rev_growth = np.nan
            if pd.notna(total_revenue_recent) and pd.notna(total_revenue_prev) and total_revenue_prev != 0:
                rev_growth = total_revenue_recent / total_revenue_prev - 1.0

            eps_growth = np.nan
            shares_out = inf.get("sharesOutstanding", np.nan)
            if (pd.notna(shares_out) and shares_out not in (0, None)
                and pd.notna(net_income_recent) and pd.notna(net_income_prev) and net_income_prev != 0):
                eps_recent = net_income_recent / shares_out
                eps_prev   = net_income_prev / shares_out
                if pd.notna(eps_recent) and pd.notna(eps_prev) and eps_prev != 0:
                    eps_growth = eps_recent / eps_prev - 1.0

            # ROE / D/E
            roe = np.nan; de_ratio = np.nan
            total_equity = np.nan; total_liab = np.nan
            if bs is not None and not bs.empty:
                try:
                    eq = bs.loc[bs.index.str.contains("Total Stockholder", case=False)].T.squeeze().dropna()
                    total_equity = float(eq.iloc[0]) if len(eq) else np.nan
                except Exception:
                    pass
                try:
                    li = bs.loc[bs.index.str.contains("Total Liabilities", case=False)].T.squeeze().dropna()
                    total_liab = float(li.iloc[0]) if len(li) else np.nan
                except Exception:
                    pass
                if pd.notna(net_income_recent) and pd.notna(total_equity) and total_equity != 0:
                    roe = net_income_recent / total_equity
                if pd.notna(total_liab) and pd.notna(total_equity) and total_equity != 0:
                    de_ratio = total_liab / total_equity

            # Margins
            net_margin = np.nan; gross_margin = np.nan
            if pd.notna(net_income_recent) and pd.notna(total_revenue_recent) and total_revenue_recent != 0:
                net_margin = net_income_recent / total_revenue_recent
            try:
                gp = fin.loc[fin.index.str.contains("Gross Profit", case=False)].T.squeeze().dropna() if fin is not None and not fin.empty else pd.Series(dtype=float)
                if len(gp):
                    gp_recent = float(gp.iloc[0])
                    if pd.notna(gp_recent) and pd.notna(total_revenue_recent) and total_revenue_recent != 0:
                        gross_margin = gp_recent / total_revenue_recent
            except Exception:
                pass

            rows.append({
                "ticker": t,
                "pe": pe, "ev_ebitda": ev_ebitda,
                "rev_growth": rev_growth, "eps_growth": eps_growth,
                "roe": roe, "de_ratio": de_ratio,
                "net_margin": net_margin, "gross_margin": gross_margin
            })
        except Exception:
            rows.append({"ticker": t})
    df = pd.DataFrame(rows).set_index("ticker")
    return df

# ---------------------------
# Technicals & relative signals
# ---------------------------
def technical_scores(price_panel: dict) -> pd.DataFrame:
    rows = []
    for ticker, px in price_panel.items():
        px = px.dropna()
        if len(px) < 130:
            continue
        ema100 = ema(px, 100)
        dma_gap = (px.iloc[-1] - ema100.iloc[-1]) / ema100.iloc[-1]
        _, _, hist = macd(px)
        macd_hist = hist.iloc[-1]
        rsi_val = rsi(px).iloc[-1]
        rsi_strength = (rsi_val - 50.0) / 50.0
        mom12m = np.nan
        if len(px) > 252:
            mom12m = px.iloc[-1] / px.iloc[-253] - 1.0
        rows.append({
            "ticker": ticker,
            "dma_gap": dma_gap,
            "macd_hist": macd_hist,
            "rsi_strength": rsi_strength,
            "mom12m": mom12m
        })
    if not rows:
        return pd.DataFrame(columns=["dma_gap","macd_hist","rsi_strength","mom12m"])
    return pd.DataFrame(rows).set_index("ticker")

def rolling_beta_alpha(stock_px: pd.Series, bench_px: pd.Series, window: int = 60):
    df = pd.DataFrame({"r_s": stock_px.pct_change(), "r_b": bench_px.pct_change()}).dropna()
    out = []
    for i in range(window, len(df)):
        chunk = df.iloc[i-window:i]
        if chunk["r_b"].std(ddof=0) == 0:
            beta = np.nan
        else:
            cov = np.cov(chunk["r_s"], chunk["r_b"])[0,1]
            beta = cov / np.var(chunk["r_b"])
        alpha = df["r_s"].iloc[i] - beta * df["r_b"].iloc[i] if pd.notna(beta) else np.nan
        out.append((df.index[i], beta, alpha))
    res = pd.DataFrame(out, columns=["date", "beta", "alpha"]).set_index("date")
    return res

def relative_signals(stock_px: pd.Series, bench_px: pd.Series):
    ratio = (stock_px / bench_px).dropna()
    rel = {"rel_dma_gap": np.nan, "rel_mom12m": np.nan, "alpha_60d": np.nan}
    if len(ratio) >= 120:
        ema100 = ratio.ewm(span=100, adjust=False).mean()
        rel["rel_dma_gap"] = (ratio.iloc[-1] - ema100.iloc[-1]) / ema100.iloc[-1]
    if len(ratio) > 252:
        rel["rel_mom12m"] = ratio.iloc[-1] / ratio.iloc[-253] - 1.0
    ba = rolling_beta_alpha(stock_px, bench_px, window=60)
    if not ba.empty and "alpha" in ba:
        rel["alpha_60d"] = ba["alpha"].dropna().iloc[-1] if ba["alpha"].dropna().size else np.nan
    return rel

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Settings")

tickers_input = st.sidebar.text_area(
    "Tickers (comma-separated):",
    "AAPL, MSFT, NVDA, AMZN, META, GOOGL"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

benchmark = st.sidebar.selectbox(
    "Benchmark (for relative comparison):",
    ["SPY (S&P 500 ETF)", "^GSPC (S&P 500 Index)", "QQQ (Nasdaq ETF)", "^IXIC (Nasdaq Index)", "^DJI (Dow Jones)"],
    index=0
).split()[0]

history = st.sidebar.selectbox("History window", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("### Component Weights")
w_fund = st.sidebar.slider("Fundamentals", 0.0, 1.0, 0.5, 0.05)
w_tech = st.sidebar.slider("Technicals", 0.0, 1.0, 0.35, 0.05)
w_macro = st.sidebar.slider("Macro (VIX)", 0.0, 1.0, 0.10, 0.05)
w_rel  = st.sidebar.slider("Relative (vs Benchmark)", 0.0, 1.0, 0.05, 0.05)

st.sidebar.markdown("### Include / Exclude")
include_fund = st.sidebar.checkbox("Include Fundamentals", True)
include_tech = st.sidebar.checkbox("Include Technicals", True)
include_macro = st.sidebar.checkbox("Include Macro Overlay (VIX)", True)
include_rel = st.sidebar.checkbox("Include Relative-to-Benchmark", True)

st.sidebar.markdown("### Fundamental factors")
fund_hi = st.sidebar.multiselect(
    "Higher is better",
    ["rev_growth","eps_growth","roe","net_margin","gross_margin"],
    default=["rev_growth","eps_growth","roe","net_margin","gross_margin"]
)
fund_lo = st.sidebar.multiselect(
    "Lower is better",
    ["pe","ev_ebitda","de_ratio"],
    default=["pe","ev_ebitda","de_ratio"]
)

st.sidebar.markdown("### Technical components")
tech_parts = st.sidebar.multiselect(
    "Signals to include",
    ["dma_gap","macd_hist","rsi_strength","mom12m"],
    default=["dma_gap","macd_hist","rsi_strength","mom12m"]
)

run_btn = st.sidebar.button("🚀 Run Model")

# ---------------------------
# Main execution
# ---------------------------
if run_btn:
    # 1) Prices
    with st.spinner("Downloading prices..."):
        prices, ok, fail = fetch_prices_per_ticker(tickers, period=history, interval="1d")
    if fail:
        st.info(f"Skipped (no data): {', '.join(fail)}")

    price_panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size > 0}
    if not price_panel:
        st.error("No valid price data downloaded. Check tickers.")
        st.stop()

    # 2) Benchmark prices
    bench_px = fetch_price_series(benchmark, period=history, interval="1d")
    if bench_px.empty and include_rel:
        st.warning(f"Benchmark {benchmark} not available; relative signals will be skipped.")
        include_rel = False

    # 3) Technicals
    tech = technical_scores(price_panel)
    if tech.empty and include_tech:
        st.warning("Not enough history for technical indicators; disabling technicals.")
        include_tech = False

    for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        if include_tech and col in tech_parts and col in tech.columns:
            tech[f"{col}_z"] = zscore_series(tech[col])
        else:
            tech[f"{col}_z"] = np.nan

    TECH_score = (
        tech[[f"{c}_z" for c in tech_parts if f"{c}_z" in tech.columns]].mean(axis=1)
        if include_tech else pd.Series(0.0, index=tech.index if not tech.empty else list(price_panel.keys()))
    )

    # 4) Fundamentals
    with st.spinner("Fetching fundamentals..."):
        fundamentals = fetch_fundamentals(list(price_panel.keys()))
    fund_scores = pd.DataFrame(index=fundamentals.index)
    if include_fund and not fundamentals.empty:
        for col in fund_hi:
            if col in fundamentals.columns:
                fund_scores[f"{col}_z"] = zscore_series(fundamentals[col])
        for col in fund_lo:
            if col in fundamentals.columns:
                fund_scores[f"{col}_z"] = zscore_series(-fundamentals[col])
        FUND_score = fund_scores.mean(axis=1) if len(fund_scores.columns) else pd.Series(0.0, index=fundamentals.index)
    else:
        FUND_score = pd.Series(0.0, index=list(price_panel.keys()))

    # 5) Relative-to-benchmark
    rel_rows = []
    if include_rel and not bench_px.empty:
        for t, px in price_panel.items():
            s = px.reindex(bench_px.index).dropna()
            b = bench_px.reindex(s.index).dropna()
            if len(s) < 120 or len(b) < 120:  # need some history
                rel_rows.append({"ticker": t, "rel_dma_gap": np.nan, "rel_mom12m": np.nan, "alpha_60d": np.nan})
                continue
            feats = relative_signals(s, b)
            rel_rows.append({"ticker": t, **feats})
    rel = pd.DataFrame(rel_rows).set_index("ticker") if rel_rows else pd.DataFrame(index=list(price_panel.keys()))
    for col in ["rel_dma_gap","rel_mom12m","alpha_60d"]:
        if include_rel and col in rel.columns:
            rel[f"{col}_z"] = zscore_series(rel[col])
        else:
            rel[f"{col}_z"] = np.nan
    REL_score = (
        rel[[c for c in ["rel_dma_gap_z","rel_mom12m_z","alpha_60d_z"] if c in rel.columns]].mean(axis=1)
        if include_rel else pd.Series(0.0, index=list(price_panel.keys()))
    )

    # 6) Macro
    vix_level = fetch_vix_level() if include_macro else np.nan
    MACRO_score = macro_overlay_score(vix_level) if include_macro else 0.0

    # 7) Combine
    idx = pd.Index(list(price_panel.keys()))
    out = pd.DataFrame(index=idx)
    out["FUND_score"] = FUND_score.reindex(idx).fillna(0.0)
    out["TECH_score"] = TECH_score.reindex(idx).fillna(0.0)
    out["REL_score"]  = REL_score.reindex(idx).fillna(0.0)
    out["MACRO_score"] = MACRO_score

    # normalize weights if some toggles disabled
    wsum = (w_fund if include_fund else 0) + (w_tech if include_tech else 0) + (w_rel if include_rel else 0) + (w_macro if include_macro else 0)
    if wsum == 0:
        st.error("All components disabled (or weights are zero). Enable at least one component.")
        st.stop()
    wf = (w_fund / wsum) if include_fund else 0.0
    wt = (w_tech  / wsum) if include_tech else 0.0
    wr = (w_rel   / wsum) if include_rel  else 0.0
    wm = (w_macro / wsum) if include_macro else 0.0

    out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wr*out["REL_score"] + wm*out["MACRO_score"]
    out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])

    def bucket(x):
        if x >= 80: return "Strong Buy"
        if x >= 60: return "Buy"
        if x >= 40: return "Hold"
        if x >= 20: return "Sell"
        return "Strong Sell"
    out["RECO"] = out["RATING_0_100"].apply(bucket)
    out = out.sort_values("RATING_0_100", ascending=False)

    # 8) Display
    st.success(f"✅ Done! VIX level: {round(vix_level,2) if not np.isnan(vix_level) else 'N/A'} | Benchmark: {benchmark}")
    st.subheader("📈 Ratings")
    st.dataframe(out.round(4))

    # 9) Charts
    st.subheader("📊 Charts")
    c1, c2 = st.columns(2)
    with c1:
        sel = st.selectbox("Choose ticker for price chart", list(out.index))
        if sel in price_panel:
            px = price_panel[sel]
            ema20 = ema(px, 20)
            ema100 = ema(px, 100)
            st.line_chart(pd.DataFrame({"Price": px, "EMA20": ema20, "EMA100": ema100}))
    with c2:
        if include_rel and not bench_px.empty and sel in price_panel:
            ratio = (price_panel[sel] / bench_px.reindex(price_panel[sel].index)).dropna()
            st.line_chart(pd.DataFrame({"Relative Price (Stock/Benchmark)": ratio}))

    # 10) Downloads & local save
    csv_bytes = out.to_csv().encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "ratings.csv", "text/csv")

    xlsx_io = io.BytesIO()
    with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
        out.round(4).to_excel(writer, sheet_name="Ratings")
        if include_fund and not fund_scores.empty:
            fund_tab = fund_scores.copy()
            fund_tab.index.name = "ticker"
            fund_tab.round(4).to_excel(writer, sheet_name="Fundamentals_z")
        if include_tech and not tech.empty:
            tech_tab = tech[[c for c in tech.columns if c.endswith("_z")]].copy()
            tech_tab.index.name = "ticker"
            tech_tab.round(4).to_excel(writer, sheet_name="Technicals_z")
        if include_rel and not rel.empty:
            rel_tab = rel[[c for c in rel.columns if c.endswith("_z")]].copy()
            rel_tab.index.name = "ticker"
            rel_tab.round(4).to_excel(writer, sheet_name="Relative_z")
    xlsx_bytes = xlsx_io.getvalue()
    st.download_button("⬇️ Download Excel", xlsx_bytes, "ratings.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Local save (handy when running on your machine)
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ratings_{ts}.xlsx"
        with open(filename, "wb") as f:
            f.write(xlsx_bytes)
        st.caption(f"Saved Excel locally as **{filename}** in {os.getcwd()}")
    except Exception:
        st.caption("Local save skipped (read-only environment).")
