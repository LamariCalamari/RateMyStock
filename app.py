# app.py — Rate My (Stock + Portfolio)
# - Robust peer loader (small chunks, 3 retries, threads=False)
# - 12m momentum uses a dedicated 5Y fetch (only for shown tickers)
# - Clear fundamentals interpretation (growth/profitability/valuation/leverage)
# - Simplified, centered landing with clean triangle logo

import io
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -------------------- Page & CSS --------------------
st.set_page_config(page_title="Rate My", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1140px;}
.hero{display:flex;flex-direction:column;align-items:center;gap:.75rem;margin:2.4rem 0 2rem}
.logo{width:70px;height:64px}
.h1{font-size:56px;font-weight:800;letter-spacing:.3px}
.sub{color:#9aa0a6;font-size:1.05rem;text-align:center;max-width:720px}
.btns{display:flex;gap:18px;justify-content:center;margin:1.4rem 0 2.4rem}
.search-wrap{display:flex;justify-content:center;margin:.6rem 0 1.2rem}
.search-inner{width:min(760px,92%)}
.search-input input{border-radius:9999px !important;padding:1rem 1.25rem !important;font-size:1.1rem}
.topbar{display:flex;justify-content:flex-end;margin:.2rem 0 .6rem}
.banner{background:#0c2f22;color:#cdebdc;border-radius:10px;padding:.9rem 1.1rem;margin:.75rem 0 1.25rem}
.kpi-card{padding:1rem 1.1rem;border-radius:12px;background:#111418;border:1px solid #222}
.kpi-num{font-size:2.2rem;font-weight:800;margin-top:.25rem}
.small-muted{color:#9aa0a6;font-size:.9rem}
.chart-caption{color:#9aa0a6;margin:-.5rem 0 1rem}
</style>
""", unsafe_allow_html=True)

# -------------------- Session --------------------
for k, v in {"entered": False, "mode": None, "grid_df": None}.items():
    if k not in st.session_state: st.session_state[k] = v

def go_home():
    st.session_state.entered = False
    st.session_state.mode = None

def enter(mode):
    st.session_state.entered = True
    st.session_state.mode = mode

# ------------------- Helpers -------------------
def yf_symbol(t: str) -> str:
    if not isinstance(t, str): return t
    return t.strip().upper().replace(".", "-")

def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up, dn = d.clip(lower=0.0), -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    line = ema(series, fast) - ema(series, slow)
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0

# -------------- Robust 2-pass fetcher --------------
@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(
    tickers, period="1y", interval="1d",
    chunk=25, retries=3, sleep_between=0.35,
    singles_pause=0.20, hard_limit=350
):
    """
    Conservative settings for Streamlit Cloud:
      - small chunks (25), threads=False, longer sleeps, +3 single retries.
    """
    tickers = [yf_symbol(t) for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))[:hard_limit]
    frames, ok = [], []

    def _append_from_multi(df, names):
        got = set(df.columns.get_level_values(0))
        for t in names:
            if t in got:
                s = df[t]["Close"].dropna()
                if s.size:
                    frames.append(s.rename(t)); ok.append(t)

    # bulk pass
    for i in range(0, len(tickers), chunk):
        group = tickers[i:i+chunk]
        try:
            df = yf.download(group, period=period, interval=interval,
                             auto_adjust=True, group_by="ticker",
                             threads=False, progress=False)  # threads=False to reduce throttling
            if isinstance(df.columns, pd.MultiIndex):
                _append_from_multi(df, group)
            else:
                t = group[0]
                if "Close" in df:
                    s = df["Close"].dropna()
                    if s.size: frames.append(s.rename(t)); ok.append(t)
        except Exception:
            pass
        time.sleep(sleep_between + random.uniform(0, 0.15))

    ok_set = set(ok)
    missing = [t for t in tickers if t not in ok_set]

    # multiple singles retries
    if missing:
        for _ in range(retries):
            new_missing = []
            for t in missing:
                try:
                    df = yf.download(t, period=period, interval=interval,
                                     auto_adjust=True, group_by="ticker",
                                     threads=False, progress=False)
                    if "Close" in df:
                        s = df["Close"].dropna()
                        if s.size: frames.append(s.rename(t)); ok.append(t)
                        else: new_missing.append(t)
                    else:
                        new_missing.append(t)
                except Exception:
                    new_missing.append(t)
                time.sleep(singles_pause + random.uniform(0, 0.25))
            missing = new_missing
            if not missing: break

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return prices, ok

@st.cache_data(show_spinner=False)
def fetch_vix_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("^VIX").history(period=period, interval=interval)
        if not df.empty: return df["Close"].rename("^VIX")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_fundamentals_simple(tickers):
    keep = ["revenueGrowth","earningsGrowth","returnOnEquity",
            "profitMargins","grossMargins","operatingMargins","ebitdaMargins",
            "trailingPE","forwardPE","debtToEquity"]
    rows=[]
    for raw in tickers:
        t=yf_symbol(raw)
        try: info = yf.Ticker(t).info or {}
        except Exception: info={}
        row={"ticker":t}
        for k in keep:
            try: row[k]=float(info.get(k, np.nan))
            except Exception: row[k]=np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")

# -------------- Peer universes --------------
SP500_FALLBACK = ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","BRK-B","UNH","LLY","V","JPM"]
DOW30_FALLBACK = ["AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","DIS","HD","INTC","IBM","KO","MCD","NKE","TRV","VZ","CSCO"]
NASDAQ100_FALLBACK = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","NFLX","CSCO","AMD"]

def list_sp500():
    try:
        got = {yf_symbol(t) for t in yf.tickers_sp500()}
        if got: return got
    except Exception:
        pass
    return set(SP500_FALLBACK)

def list_dow30():
    try:
        got = {yf_symbol(t) for t in yf.tickers_dow()}
        if got: return got
    except Exception:
        pass
    return set(DOW30_FALLBACK)

def list_nasdaq100():
    try:
        if hasattr(yf,"tickers_nasdaq"):
            got = {yf_symbol(t) for t in yf.tickers_nasdaq()}
            if got: return got
    except Exception:
        pass
    return set(NASDAQ100_FALLBACK)

def build_universe(user_tickers, mode, sample_n=150, custom_raw=""):
    user = [yf_symbol(t) for t in user_tickers]
    if mode == "S&P 500":
        peers_all = list_sp500(); label="S&P 500"
    elif mode == "Dow 30":
        peers_all = list_dow30(); label="Dow 30"
    elif mode == "NASDAQ 100":
        peers_all = list_nasdaq100(); label="NASDAQ 100"
    elif mode == "Custom (paste list)":
        custom = {yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
        return sorted(set(user)|custom)[:350], "Custom"
    else:
        sp,dj,nd = list_sp500(), list_dow30(), list_nasdaq100()
        auto=set(); label="S&P 500"
        if len(user)==1:
            t=user[0]
            if   t in sp: auto=sp; label="S&P 500"
            elif t in dj: auto=dj; label="Dow 30"
            elif t in nd: auto=nd; label="NASDAQ 100"
        else:
            for t in user:
                if t in sp: auto|=sp; label="S&P 500"
                elif t in dj: auto|=dj; label="Dow 30"
                elif t in nd: auto|=nd; label="NASDAQ 100"
        peers_all = auto if auto else sp
    peers = sorted(peers_all.difference(set(user)))[:max(1, sample_n)]
    return sorted(set(user)|set(peers))[:350], label

# -------------- Feature builders --------------
def technical_scores(price_panel: dict) -> pd.DataFrame:
    rows=[]
    for ticker, px in price_panel.items():
        px=px.dropna()
        if len(px)<60:  # relaxed
            continue
        ema50  = ema(px,50)
        base50 = ema50.iloc[-1] if pd.notna(ema50.iloc[-1]) and ema50.iloc[-1]!=0 else np.nan
        dma_gap=(px.iloc[-1]-ema50.iloc[-1])/base50 if pd.notna(base50) else np.nan
        _,_,hist = macd(px)
        macd_hist = hist.iloc[-1]
        r = rsi(px).iloc[-1]
        rsi_strength = (r-50.0)/50.0
        # 12m mom placeholder; we overwrite only for presented tickers with 5Y fetch
        mom = np.nan
        rows.append({"ticker":ticker,"dma_gap":dma_gap,"macd_hist":macd_hist,
                     "rsi_strength":rsi_strength,"mom12m":mom})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

def macro_from_vix(vix_series: pd.Series):
    if vix_series is None or vix_series.empty:
        return 0.5, np.nan, np.nan, np.nan
    vix_last = float(vix_series.iloc[-1])
    ema20    = float(ema(vix_series,20).iloc[-1]) if len(vix_series)>=20 else vix_last
    rel_gap  = (vix_last-ema20)/max(ema20,1e-9)
    if   vix_last<=12: level=1.0
    elif vix_last>=28: level=0.0
    else: level = 1.0-(vix_last-12)/16.0
    if   rel_gap>=0.03: trend=0.0
    elif rel_gap<=-0.03: trend=1.0
    else: trend = float(np.clip(1.0-(rel_gap+0.03)/0.06,0,1))
    macro=float(np.clip(0.70*level+0.30*trend,0,1))
    return macro, vix_last, ema20, rel_gap

# -------------- UI blocks --------------
def inline_logo_svg():
    # clean triangle (no inner lines)
    return """
<svg class="logo" viewBox="0 0 100 90" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0" x2="0" y1="1" y2="0">
      <stop offset="0%" stop-color="#e74c3c"/>
      <stop offset="50%" stop-color="#f39c12"/>
      <stop offset="100%" stop-color="#2ecc71"/>
    </linearGradient>
  </defs>
  <polygon points="50,5 95,85 5,85" fill="url(#g)"/>
</svg>
"""

def landing():
    st.markdown(f'''
<div class="hero">
  {inline_logo_svg()}
  <div class="h1">Rate My</div>
  <div class="sub">Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts.</div>
</div>
''', unsafe_allow_html=True)
    st.markdown('<div class="btns">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.button("📈 Rate My Stock", type="primary", use_container_width=True, on_click=enter, args=("stock",))
    with c2:
        st.button("💼 Rate My Portfolio", use_container_width=True, on_click=enter, args=("portfolio",))
    st.markdown('</div>', unsafe_allow_html=True)

def topbar_back(key):
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    st.button("← Back", key=key, on_click=go_home)
    st.markdown('</div>', unsafe_allow_html=True)

# ======== STOCK CHARTS ========
def draw_stock_charts(t: str, series: pd.Series):
    if series is None or series.empty:
        st.info("Not enough history to show charts.")
        return
    st.subheader("📈 Price & EMAs")
    e20, e50 = ema(series,20), ema(series,50)
    price_df = pd.DataFrame({"Close": series, "EMA20": e20, "EMA50": e50})
    st.line_chart(price_df, use_container_width=True)
    st.caption("If price is above EMA50/EMA20 → trend is supportive; below → headwind.")

    st.subheader("📉 MACD")
    line, sig, hist = macd(series)
    st.line_chart(pd.DataFrame({"MACD line": line, "Signal": sig}), use_container_width=True)
    st.bar_chart(pd.DataFrame({"Histogram": hist}), use_container_width=True)
    st.caption("Rising histogram above zero → momentum building; falling below zero → fading.")

    st.subheader("🔁 RSI (14)")
    st.line_chart(pd.DataFrame({"RSI(14)": rsi(series)}), use_container_width=True)
    st.caption(">70 overbought • <30 oversold • around 50 neutral strength.")

    st.subheader("🚀 12-month momentum")
    if len(series) > 252:
        mom12 = series/series.shift(253)-1.0
        st.line_chart(pd.DataFrame({"12m momentum": mom12}), use_container_width=True)
        st.caption("Positive vs a year ago → outperformance; negative → underperformance.")
    else:
        st.info("Need > 1 year of data to show the 12-month momentum line.")

# =============== STOCK APP ===============
def fundamentals_interpretation(zrow: pd.Series):
    """Return a list of plain-English lines for fundamental tilts."""
    lines=[]
    def bucket(v, pos_good=True):
        if pd.isna(v): return "neutral"
        if pos_good:
            return "bullish" if v>=0.5 else "watch" if v<=-0.5 else "neutral"
        else:
            return "bullish (cheap)" if v>=0.5 else "watch (expensive)" if v<=-0.5 else "neutral"
    g = bucket(zrow.get("revenueGrowth_z"))  # growth
    e = bucket(zrow.get("earningsGrowth_z"))
    p = bucket(zrow.get("profitMargins_z"))
    roe = bucket(zrow.get("returnOnEquity_z"))
    val = bucket(zrow.get("forwardPE_z"), pos_good=False)
    lev = bucket(zrow.get("debtToEquity_z"), pos_good=False)
    # Build sentences
    if g=="bullish" or e=="bullish":
        lines.append("**Growth tilt:** above-peer revenue/earnings growth (supportive).")
    elif g=="watch" or e=="watch":
        lines.append("**Growth tilt:** below peers — watch trend stabilization.")
    else:
        lines.append("**Growth tilt:** broadly in line with peers.")
    if p=="bullish" or roe=="bullish":
        lines.append("**Profitability & margins:** strong vs peers (healthy quality).")
    elif p=="watch" or roe=="watch":
        lines.append("**Profitability:** below peer medians — monitor margin trajectory.")
    else:
        lines.append("**Profitability:** roughly peer-like.")
    if val.startswith("bullish"):
        lines.append("**Valuation tilt:** cheaper than peers (potential multiple support).")
    elif val.startswith("watch"):
        lines.append("**Valuation tilt:** richer than peers — execution must stay strong.")
    else:
        lines.append("**Valuation tilt:** roughly fair vs peers.")
    if lev.startswith("bullish"):
        lines.append("**Balance sheet:** lower leverage vs peers (lower financial risk).")
    elif lev.startswith("watch"):
        lines.append("**Balance sheet:** higher leverage vs peers — keep an eye on rates/cash flow.")
    else:
        lines.append("**Balance sheet:** typical for the peer set.")
    return lines

def app_stock():
    topbar_back("back_stock")
    st.markdown(f'''
<div class="hero" style="margin-top:.2rem;margin-bottom:.2rem">
  {inline_logo_svg()}
  <div class="h1">Rate My Stock</div>
</div>''', unsafe_allow_html=True)

    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    ticker = st.text_input(" ", "AAPL", label_visibility="collapsed",
                           placeholder="Type a ticker (e.g., AAPL)", key="ticker_in")
    st.markdown('</div></div>', unsafe_allow_html=True)

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
        st.info("Enter a ticker above to run the rating."); return

    with st.status("Crunching the numbers…", expanded=True) as status:
        prog = st.progress(0)
        status.update(label="Building peer universe…")
        universe, label = build_universe(user_tickers, universe_mode, peer_n, custom_raw)
        target_count = peer_n if universe_mode!="Custom (paste list)" else len(universe)
        prog.progress(10)

        status.update(label="Downloading prices (chunked + retries)…")
        prices, ok = fetch_prices_chunked_with_fallback(
            universe, period=history, interval="1d",
            chunk=25, retries=3, threads=False if True else False  # doc note
        )
        if not ok: st.error("No peer prices loaded."); return
        panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size>0}
        prog.progress(50)

        status.update(label="Computing technicals…")
        tech = technical_scores(panel)
        for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
            if col in tech.columns: tech[f"{col}_z"] = zscore_series(tech[col])
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
        prog.progress(100); status.update(label="Done!", state="complete")

    st.markdown(
        f'<div class="banner">Peers loaded: <b>{len(panel)}</b> / <b>{target_count}</b> '
        f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>',
        unsafe_allow_html=True
    )

    # Output
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

    # Compute 12-month momentum from 5Y only for shown tickers (silent upgrade of the table + chart)
    show_idx = [t for t in user_tickers if t in out.index]
    if show_idx:
        for t in show_idx:
            try:
                px5 = yf.Ticker(t).history(period="5y", interval="1d")["Close"].dropna()
                if len(px5)>253:
                    mom12 = px5.iloc[-1]/px5.iloc[-253]-1.0
                    if t in tech.index:
                        tech.loc[t,"mom12m"]=mom12
                        tech.loc[t,"mom12m_z"]=np.nan  # z unavailable across peers here; fine for explanation
                        out.loc[t,"TECH_score"] = np.nanmean([
                            tech.loc[t,"dma_gap_z"] if "dma_gap_z" in tech.columns else np.nan,
                            tech.loc[t,"macd_hist_z"] if "macd_hist_z" in tech.columns else np.nan,
                            tech.loc[t,"rsi_strength_z"] if "rsi_strength_z" in tech.columns else np.nan,
                            # keep z-only contributors; mom12m used in text/chart
                        ])
                        out.loc[t,"COMPOSITE"] = wf*out.loc[t,"FUND_score"] + wt*out.loc[t,"TECH_score"] + wm*out.loc[t,"MACRO_score"]
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
    for t in show_idx:
        reco = table.loc[t,"RECO"]; sc = table.loc[t,"RATING_0_100"]
        with st.expander(f"{t} — {reco} (Score: {sc:.1f})", expanded=True):
            k1,k2,k3 = st.columns(3)
            k1.markdown(f'<div class="kpi-card"><div>Fundamentals</div><div class="kpi-num">{table.loc[t,"FUND_score"]:.3f}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi-card"><div>Technicals</div><div class="kpi-num">{table.loc[t,"TECH_score"]:.3f}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi-card"><div>Macro (VIX)</div><div class="kpi-num">{table.loc[t,"MACRO_score"]:.3f}</div></div>', unsafe_allow_html=True)

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
            # Narrative
            st.markdown("**Interpretation**")
            for line in fundamentals_interpretation(fdf.loc[t] if t in fdf.index else pd.Series(dtype=float)):
                st.markdown("- " + line)

            # Technicals quick table
            st.markdown("#### Technicals")
            rsi_val = 50 + 50*tech.loc[t,"rsi_strength"] if ("rsi_strength" in tech.columns and t in tech.index and pd.notna(tech.loc[t,"rsi_strength"])) else np.nan
            tshow = pd.DataFrame({
                "Price vs EMA50 (gap)": tech.loc[t,"dma_gap"] if ("dma_gap" in tech.columns and t in tech.index) else np.nan,
                "MACD histogram": tech.loc[t,"macd_hist"] if ("macd_hist" in tech.columns and t in tech.index) else np.nan,
                "RSI (approx)": rsi_val,
                "12-mo momentum": tech.loc[t,"mom12m"] if ("mom12m" in tech.columns and t in tech.index) else np.nan,
            }, index=[t]).T.rename(columns={t:"value"})
            st.dataframe(tshow.round(3), use_container_width=True)
            notes=[]
            if "dma_gap" in tech.columns and t in tech.index and pd.notna(tech.loc[t,"dma_gap"]):
                notes.append("Price above EMA50 → trend tailwind." if tech.loc[t,"dma_gap"]>0.02 else "Price below EMA50 → trend headwind." if tech.loc[t,"dma_gap"]<-0.02 else "Price near EMA50.")
            if "macd_hist" in tech.columns and t in tech.index and pd.notna(tech.loc[t,"macd_hist"]):
                notes.append("MACD histogram positive → momentum building." if tech.loc[t,"macd_hist"]>0 else "MACD histogram negative → momentum fading.")
            if pd.notna(rsi_val):
                notes.append(f"RSI ~{rsi_val:.0f} — strong/overbought." if rsi_val>=65 else f"RSI ~{rsi_val:.0f} — weak/oversold." if rsi_val<=35 else f"RSI ~{rsi_val:.0f} — neutral.")
            if "mom12m" in tech.columns and t in tech.index and pd.notna(tech.loc[t,"mom12m"]):
                notes.append("12m momentum positive." if tech.loc[t,"mom12m"]>0 else "12m momentum negative.")
            if notes: st.markdown("- " + "\n- ".join(notes))

            # Macro read
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
                if vix_gap > 0.03: trend_txt = "rising and above its 20-day average (volatility building)"
                elif vix_gap < -0.03: trend_txt = "falling and below its 20-day average (volatility easing)"
                else: trend_txt = "moving roughly in line with its 20-day average"
                st.markdown(f"- **Level:** {level_txt}.  \n- **Trend:** {trend_txt}.")
            else:
                st.info("VIX unavailable — Macro defaults to neutral.")

            # Downloads
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
                px1y = yf.Ticker(t).history(period="2y", interval="1d")["Close"].dropna()
                draw_stock_charts(t, px1y)
            except Exception:
                if t in panel: draw_stock_charts(t, panel[t])

# =============== PORTFOLIO (same as before, submit-based editor) ===============
# ... (to keep the answer focused on your 4 asks, the portfolio block from the previous version can remain)
# You can paste your last working portfolio section below this comment unchanged.
# ------------------------------------------------------------------------------

def app_router():
    if not st.session_state.entered:
        landing(); return
    if st.session_state.mode=="portfolio":
        st.info("Portfolio section unchanged here — paste the portfolio block from your previous working version below the comment labeled 'PORTFOLIO'.")
    else:
        app_stock()

app_router()