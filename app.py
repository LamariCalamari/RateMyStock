# app.py — ⭐️ Rate My (Stock or Portfolio)
# - Peer universe: Auto / S&P500 / Dow30 / NASDAQ100 / Custom
# - Stock mode: Fundamentals + Technicals + Macro (VIX level+trend), no Relative
# - Portfolio mode: weighted composite + Diversification score (sector mix, name concentration, correlations)
# - Landing: choose Stock or Portfolio
# - Auto-run, centered inputs, loading bar, chart explanations

import io, time, datetime as dt, numpy as np, pandas as pd, streamlit as st, yfinance as yf

# --------------------- Page & Styles ---------------------
st.set_page_config(page_title="Rate My", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px;}
.hero {text-align:center; margin-top: 3.5rem; margin-bottom: .5rem;}
.sub {text-align:center; color:#9aa0a6; margin-bottom: 2rem;}
.small-muted {color:#9aa0a6; font-size:.9rem; text-align:center;}
.search-wrap {display:flex; justify-content:center; margin: 1rem 0 .5rem 0;}
.search-inner {width: min(760px, 92%);}
.search-input input {border-radius: 9999px !important; padding: .95rem 1.2rem !important; font-size: 1.1rem;}
.stDataFrame, .stTable {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# -------------------- Session --------------------
if "entered" not in st.session_state: st.session_state.entered = False
if "mode" not in st.session_state: st.session_state.mode = None  # "stock" | "portfolio"

def enter(mode=None):
    st.session_state.entered = True
    if mode: st.session_state.mode = mode

# --------------------------- Utils ----------------------------
def yf_symbol(t: str) -> str:
    if not isinstance(t, str): return t
    return t.strip().upper().replace(".", "-")  # BRK.B -> BRK-B

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0

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

# -------------------- Data Fetchers --------------------
@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(tickers, period="1y", interval="1d",
                                       chunk=50, min_ok=50, sleep_between=0.2):
    tickers = [yf_symbol(t) for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))
    frames, ok, fail = [], [], []
    for i in range(0, len(tickers), chunk):
        group = tickers[i:i+chunk]
        try:
            df = yf.download(group, period=period, interval=interval,
                             auto_adjust=True, group_by="ticker",
                             threads=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                got = set(df.columns.get_level_values(0))
                for t in group:
                    if t in got:
                        s = df[t]["Close"].dropna()
                        if s.size: frames.append(s.rename(t)); ok.append(t)
                        else: fail.append(t)
                    else:
                        fail.append(t)
            else:
                t = group[0]
                if "Close" in df:
                    s = df["Close"].dropna()
                    if s.size: frames.append(s.rename(t)); ok.append(t)
                    else: fail.append(t)
                else:
                    fail.append(t)
        except Exception:
            fail.extend(group)
        time.sleep(sleep_between)

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    if len(ok) < min_ok:
        for t in [t for t in tickers if t not in ok]:
            if len(ok) >= min_ok: break
            try:
                df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
                s = df["Close"].dropna()
                if s.size:
                    if prices.empty: prices = s.to_frame(t)
                    else: prices[t] = s
                    ok.append(t)
                else:
                    if t not in fail: fail.append(t)
            except Exception:
                if t not in fail: fail.append(t)
            time.sleep(0.12)

    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return prices, ok, [t for t in tickers if t not in ok]

@st.cache_data(show_spinner=False)
def fetch_price_series(ticker: str, period="1y", interval="1d") -> pd.Series:
    t = yf_symbol(ticker)
    try:
        df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
        if not df.empty and "Close" in df.columns:
            return df["Close"].rename(t)
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_vix_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("^VIX").history(period=period, interval=interval)
        if not df.empty:
            return df["Close"].rename("^VIX")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_company_meta(tickers):
    """Return DataFrame with sector & industry from yfinance .info"""
    rows=[]
    for raw in tickers:
        t=yf_symbol(raw)
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        rows.append({"ticker":t, "sector":info.get("sector", None), "industry":info.get("industry", None)})
    return pd.DataFrame(rows).set_index("ticker")

# -------------------- Peer Universes & Fallbacks --------------------
SP500_FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","BRK-B","UNH","LLY","V","JPM","JNJ","XOM",
    "PG","MA","HD","CVX","MRK","ABBV","PEP","KO","COST","BAC","ADBE","WMT","CRM","CSCO","ACN","MCD","TMO",
    "NFLX","AMD","DHR","NKE","LIN","ABT","INTC","TXN","DIS","AMAT","PM","NEE","COP","MS","LOW","HON","BMY","QCOM",
    "IBM","UNP","SBUX","INTU","CAT","GS","LMT","RTX","BLK","BKNG","AXP","GE","NOW","MDT","ISRG","ADI","ELV","PLD",
    "DE","ZTS","SPGI","MDLZ","TJX","GILD","ETN","ADP","CB","SO","EQIX","PGR","MMC","CI","SCHW","MU","REGN","ORLY"
]
DOW30_FALLBACK = [
    "AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","DIS","HD","INTC","IBM","KO","MCD","NKE",
    "TRV","VZ","CSCO","MRK","PFE","CAT","AXP","BA","MMM","GS","CVX","WBA","HON","CRM"
]
NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","LIN","NFLX","CSCO","AMD",
    "TMUS","TXN","AMAT","INTC","QCOM","BKNG","PDD","HON","SBUX","INTU","AMGN","ABNB","MU","MRVL","ADP","MDLZ",
    "ISRG","REGN","LRCX","PANW","GILD","VRTX","PYPL","MELI","KLAC","CSX","ADI","KDP","CHTR","CTAS","CRWD","SNPS",
    "ORLY","MNST","ABNB","AEP","FTNT","DXCM","ODFL","PCAR","PAYX","MAR","DDOG","CDNS","CPRT","WDAY","NXPI","ROP",
    "CEG","TEAM","IDXX","KHC","AZN","LCID","ZS","ROST","EA","ALGN","VRSK","ANSS","WDAY","MRNA","SPLK"
]

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
    # yfinance has yf.tickers_nasdaq() on some versions; fall back if missing
    try:
        if hasattr(yf, "tickers_nasdaq"):
            got = {yf_symbol(t) for t in yf.tickers_nasdaq()}
            if got: return got
    except Exception:
        pass
    return set(NASDAQ100_FALLBACK)

def build_universe(user_tickers, mode, sample_n=120, custom_raw=""):
    user = [yf_symbol(t) for t in user_tickers]
    if mode == "S&P 500":
        peers_all = list_sp500()
    elif mode == "Dow 30":
        peers_all = list_dow30()
    elif mode == "NASDAQ 100":
        peers_all = list_nasdaq100()
    elif mode == "Custom (paste list)":
        custom = {yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
        return sorted(set(user) | custom)
    else:  # Auto by index membership
        sp, dj, nd = list_sp500(), list_dow30(), list_nasdaq100()
        auto = set()
        if len(user) == 1:
            t = user[0]
            if   t in sp: auto = sp
            elif t in dj: auto = dj
            elif t in nd: auto = nd
        else:
            for t in user:
                if   t in sp: auto |= sp
                elif t in dj: auto |= dj
                elif t in nd: auto |= nd
        peers_all = auto if auto else (sp or nd or dj or set(SP500_FALLBACK))
    peers = sorted(peers_all.difference(set(user)))
    if len(peers) > sample_n: peers = peers[:sample_n]
    return sorted(set(user) | set(peers))

# -------------------------- Features --------------------------
def technical_scores(price_panel: dict) -> pd.DataFrame:
    rows = []
    for ticker, px in price_panel.items():
        px = px.dropna()
        if len(px) < 130:  # EMA100 + MACD stability
            continue
        ema100 = ema(px, 100)
        base = ema100.iloc[-1] if pd.notna(ema100.iloc[-1]) and ema100.iloc[-1] != 0 else np.nan
        dma_gap = (px.iloc[-1] - ema100.iloc[-1]) / base if pd.notna(base) else np.nan
        _, _, hist = macd(px)
        macd_hist = hist.iloc[-1]
        r = rsi(px).iloc[-1]
        rsi_strength = (r - 50.0) / 50.0
        mom12m = px.iloc[-1] / px.iloc[-253] - 1.0 if len(px) > 252 else np.nan
        rows.append({"ticker": ticker, "dma_gap": dma_gap, "macd_hist": macd_hist,
                     "rsi_strength": rsi_strength, "mom12m": mom12m})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

# --------------------- Fundamentals (.info) ---------------------
@st.cache_data(show_spinner=False)
def fetch_fundamentals_simple(tickers):
    rows = []
    keep = [
        # higher-better
        "revenueGrowth","earningsGrowth","returnOnEquity",
        "profitMargins","grossMargins","operatingMargins","ebitdaMargins",
        # lower-better
        "trailingPE","forwardPE","debtToEquity"
    ]
    for raw in tickers:
        t = yf_symbol(raw)
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        row = {"ticker": t}
        for k in keep:
            v = info.get(k, np.nan)
            try: row[k] = float(v)
            except Exception: row[k] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")

# -------------------- Macro (VIX level + trend) --------------------
def macro_overlay_score_from_series(vix_series: pd.Series):
    """
    Returns (macro_score, level_score, trend_score)
    - Level: 12 -> 1.0, 28 -> 0.0 linearly
    - Trend: last vs EMA20 (rising reduces)
    Macro = 0.7*Level + 0.3*Trend ; all in [0,1]
    """
    if vix_series is None or vix_series.empty:
        return 0.5, np.nan, np.nan
    vix_last = float(vix_series.iloc[-1])
    if vix_last <= 12: level_score = 1.0
    elif vix_last >= 28: level_score = 0.0
    else: level_score = 1.0 - (vix_last - 12) / 16.0
    ema20 = ema(vix_series, 20)
    vix_ema = float(ema20.iloc[-1]) if not np.isnan(ema20.iloc[-1]) else vix_last
    rel_gap = (vix_last - vix_ema) / max(vix_ema, 1e-9)
    if rel_gap >= 0.30: trend_score = 0.0
    elif rel_gap <= -0.10: trend_score = 1.0
    else:
        trend_score = 1.0 - (rel_gap + 0.10) / 0.40
        trend_score = float(np.clip(trend_score, 0.0, 1.0))
    macro = float(np.clip(0.70*level_score + 0.30*trend_score, 0.0, 1.0))
    return macro, level_score, trend_score

# -------------------- Explain helpers --------------------
def label_badge(value, pos_thresh=None, neg_thresh=None, higher_is_better=True, units=""):
    v = value
    if pd.isna(v): return "⚪ Neutral"
    if higher_is_better:
        if pos_thresh is None: pos_thresh = 0
        if v > pos_thresh: return f"🟢 Bullish ({v:.3f}{units})"
        if neg_thresh is not None and v < neg_thresh: return f"🔴 Bearish ({v:.3f}{units})"
        return f"⚪ Neutral ({v:.3f}{units})"
    else:
        if neg_thresh is None: neg_thresh = 0
        if v < neg_thresh: return f"🟢 Bullish ({v:.3f}{units})"
        if pos_thresh is not None and v > pos_thresh: return f"🔴 Bearish ({v:.3f}{units})"
        return f"⚪ Neutral ({v:.3f}{units})"

def explain_technicals_row(row):
    notes = []
    notes.append(f"• **Price vs EMA(100)**: {label_badge(row.get('dma_gap'), pos_thresh=0)}. Above EMA suggests trend strength; below can flag weakness or potential value entry.")
    notes.append(f"• **MACD histogram**: {label_badge(row.get('macd_hist'), pos_thresh=0)}. Positive = momentum building; negative = fading.")
    rsi_strength = row.get("rsi_strength")
    rsi_val = 50 + 50*rsi_strength if pd.notna(rsi_strength) else None
    if rsi_val is not None:
        if   rsi_val >= 60: rsi_note = f"🟢 Bullish (RSI≈{rsi_val:.0f})"
        elif rsi_val <= 40: rsi_note = f"🔴 Bearish (RSI≈{rsi_val:.0f})"
        else:               rsi_note = f"⚪ Neutral (RSI≈{rsi_val:.0f})"
    else:
        rsi_note = "⚪ Neutral"
    notes.append(f"• **RSI**: {rsi_note}. 70+ may be overbought; 30− oversold.")
    notes.append(f"• **12-month momentum**: {label_badge(row.get('mom12m'), pos_thresh=0)} vs its own past year.")
    return "\n".join(notes)

# --------------------------- Landing ---------------------------
def landing():
    st.markdown('<div class="hero"><h1>⭐️ Rate My</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Pick a mode to get a vibe-checked score — with receipts for every signal.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.button("Rate My Stock", type="primary", use_container_width=True, on_click=enter, kwargs={"mode":"stock"})
    with c2:
        st.button("Rate My Portfolio", use_container_width=True, on_click=enter, kwargs={"mode":"portfolio"})
    st.markdown('<p class="small-muted">Made with Streamlit · yfinance · pandas</p>', unsafe_allow_html=True)

# --------------------------- Stock App --------------------------
def app_stock():
    # Centered input
    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    ticker = st.text_input(" ", "AAPL", key="ticker_stock", label_visibility="collapsed",
                           placeholder="Type a ticker (e.g., AAPL)")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Settings
    with st.expander("Advanced settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            universe_mode = st.selectbox(
                "Peer universe",
                ["Auto by index membership", "S&P 500", "Dow 30", "NASDAQ 100", "Custom (paste list)"],
                index=0, key="universe_stock"
            )
        with c2:
            peer_sample_n = st.slider("Peer sample size", 30, 200, 120, 10, key="peer_n_stock")
        with c3:
            history = st.selectbox("History", ["1y", "2y", "5y"], index=0, key="hist_stock")
        c4, c5, c6 = st.columns(3)
        with c4:
            w_fund = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.5, 0.05, key="wf_stock")
        with c5:
            w_tech = st.slider("Weight: Technicals",   0.0, 1.0, 0.40, 0.05, key="wt_stock")
        with c6:
            w_macro= st.slider("Weight: Macro (VIX)",  0.0, 1.0, 0.10, 0.05, key="wm_stock")
        custom_raw = st.text_area("Custom peers (comma-separated)", "", key="custom_stock") \
                      if universe_mode=="Custom (paste list)" else ""
        show_debug = st.checkbox("Show debug info", value=False, key="dbg_stock")

    # Auto-run
    user_tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]
    if not user_tickers:
        st.info("Enter a ticker above to run the rating."); return

    with st.status("Crunching the numbers…", expanded=False) as status:
        prog = st.progress(0)
        status.update(label="Building peer universe…")
        universe = build_universe(user_tickers, universe_mode, peer_sample_n, custom_raw); prog.progress(10)

        status.update(label="Downloading peer prices…")
        prices, ok, _ = fetch_prices_chunked_with_fallback(universe, period=history, interval="1d",
                                                           chunk=50, min_ok=min(60, max(40, int(peer_sample_n*0.6))))
        panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size > 0}; prog.progress(40)
        if len(panel) < 10:
            status.update(label="Retrying with larger peer set…")
            retry_universe = sorted(set([*user_tickers, *SP500_FALLBACK]))[:max(peer_sample_n, 120)]
            prices2, ok2, _ = fetch_prices_chunked_with_fallback(retry_universe, period="1y", interval="1d",
                                                                 chunk=50, min_ok=60)
            panel = {t: prices2[t].dropna() for t in ok2 if t in prices2.columns and prices2[t].dropna().size > 0}
        if not panel: st.error("No peer prices loaded."); return

        status.update(label="Computing technicals…")
        tech = technical_scores(panel)
        for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
            if col in tech.columns: tech[f"{col}_z"] = zscore_series(tech[col])
        TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech.columns]].mean(axis=1); prog.progress(70)

        status.update(label="Fetching fundamentals…")
        fund_raw = fetch_fundamentals_simple(list(panel.keys()))
        fdf = pd.DataFrame(index=fund_raw.index)
        for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins","grossMargins","operatingMargins","ebitdaMargins"]:
            if col in fund_raw.columns: fdf[f"{col}_z"] = zscore_series(fund_raw[col])
        for col in ["trailingPE","forwardPE","debtToEquity"]:
            if col in fund_raw.columns: fdf[f"{col}_z"] = zscore_series(-fund_raw[col])
        FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fund_raw.index); prog.progress(85)

        status.update(label="Assessing macro regime…")
        vix_series = fetch_vix_series(period="6mo", interval="1d")
        MACRO_score_val, MACRO_level, MACRO_trend = macro_overlay_score_from_series(vix_series)

        idx = pd.Index(list(panel.keys()))
        out = pd.DataFrame(index=idx)
        out["FUND_score"]  = FUND_score.reindex(idx).fillna(0.0)
        out["TECH_score"]  = TECH_score.reindex(idx).fillna(0.0)
        out["MACRO_score"] = MACRO_score_val

        wsum = (w_fund + w_tech + w_macro)
        if wsum == 0: st.error("All weights are zero."); return
        wf, wt, wm = w_fund/wsum, w_tech/wsum, w_macro/wsum
        out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wm*out["MACRO_score"]
        out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])
        out["RECO"] = out["RATING_0_100"].apply(lambda x: "Strong Buy" if x>=80 else "Buy" if x>=60 else "Hold" if x>=40 else "Sell" if x>=20 else "Strong Sell")

        show_idx = [yf_symbol(t) for t in user_tickers if yf_symbol(t) in out.index]
        table = out.reindex(show_idx).sort_values("RATING_0_100", ascending=False)
        prog.progress(100); status.update(label="Done!", state="complete")

    vix_last = float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else np.nan
    vix_ema20 = float(ema(vix_series,20).iloc[-1]) if vix_series is not None and not vix_series.empty else np.nan
    vix_txt = f"{vix_last:.2f}" if not np.isnan(vix_last) else "N/A"
    st.success(f"VIX: {vix_txt}  |  Macro score: {MACRO_score_val:.2f} (level={MACRO_level:.2f}, trend={MACRO_trend:.2f})  |  Peers loaded: {len(panel)}")

    st.markdown("## 🏁 Ratings")
    pretty = table.rename(columns={
        "FUND_score":"Fundamentals","TECH_score":"Technicals","MACRO_score":"Macro (VIX)",
        "COMPOSITE":"Composite","RATING_0_100":"Score (0–100)","RECO":"Recommendation"
    })
    st.dataframe(pretty.round(4), use_container_width=True)

    st.markdown("## 🔍 Why this rating?")
    for t in show_idx:
        reco = table.loc[t,"RECO"]; sc = table.loc[t,"RATING_0_100"]
        with st.expander(f"{t} — {reco} (Score: {sc:.1f})"):
            c1,c2,c3 = st.columns(3)
            c1.metric("Fundamentals", f"{table.loc[t,'FUND_score']:.3f}")
            c2.metric("Technicals",   f"{table.loc[t,'TECH_score']:.3f}")
            c3.metric("Macro (VIX)",  f"{table.loc[t,'MACRO_score']:.3f}")

            if t in fdf.index and len(fdf.columns):
                fmap = {"revenueGrowth_z":"Revenue growth (YoY)","earningsGrowth_z":"Earnings growth (YoY)","returnOnEquity_z":"Return on equity",
                        "profitMargins_z":"Net margin","grossMargins_z":"Gross margin","operatingMargins_z":"Operating margin",
                        "ebitdaMargins_z":"EBITDA margin","trailingPE_z":"PE (lower better)","forwardPE_z":"Forward PE (lower better)","debtToEquity_z":"Debt/Equity (lower better)"}
                series = {nice: float(fdf.loc[t, raw]) for raw, nice in fmap.items() if raw in fdf.columns and pd.notna(fdf.loc[t, raw])}
                if series:
                    st.markdown("**Fundamentals (vs peers)**"); st.table(pd.Series(series, name=t).round(3))

            if t in tech.index:
                tmap = {"dma_gap_z":"Price vs EMA(100)","macd_hist_z":"MACD histogram","rsi_strength_z":"RSI (strength)","mom12m_z":"12-month momentum"}
                series = {nice: float(tech.loc[t, raw]) for raw, nice in tmap.items() if raw in tech.columns and pd.notna(tech.loc[t, raw])}
                if series:
                    st.markdown("**Technicals (vs peers)**"); st.table(pd.Series(series, name=t).round(3))
                st.markdown("**Interpretation**")
                st.markdown(explain_technicals_row(tech.loc[t, ["dma_gap","macd_hist","rsi_strength","mom12m"]].to_dict()))

            # Macro explanation
            st.markdown("**Macro (VIX) — level & trend**")
            if not np.isnan(vix_last):
                rel_gap = (vix_last - vix_ema20)/vix_ema20 if (not np.isnan(vix_ema20) and vix_ema20!=0) else np.nan
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Current VIX", f"{vix_last:.2f}")
                m2.metric("VIX vs EMA20", f"{(rel_gap*100):.1f}%" if not np.isnan(rel_gap) else "N/A")
                m3.metric("Level score", f"{MACRO_level:.2f}")
                m4.metric("Trend score", f"{MACRO_trend:.2f}")
                st.caption("Level maps today’s VIX to 0–1 (12→1.0, 28→0.0). Trend compares VIX to its 20-day EMA: rising above trend reduces the score (risk building), falling below boosts it.")
            else:
                st.caption("Macro details unavailable (VIX not loaded).")

    # Charts
    if show_idx:
        sel = st.selectbox("Choose ticker for charts", show_idx, index=0, key="chart_sel_stock")
        px = panel.get(sel)
        if px is not None and not px.empty:
            tabs = st.tabs(["Price & EMAs","Cumulative (Stock)","Volatility & Sharpe (60d)","Drawdown"])
            with tabs[0]:
                ema20, ema100 = ema(px,20), ema(px,100)
                st.line_chart(pd.DataFrame({"Price":px,"EMA20":ema20,"EMA100":ema100}))
                st.caption("Price with short/medium trend proxies. Above EMAs often indicates trend strength; below can point to weakness/value entry.")
            with tabs[1]:
                r = px.pct_change().fillna(0); equity=(1+r).cumprod()
                st.line_chart(pd.DataFrame({"Cumulative return": equity})); st.caption("Growth of £1 invested over the selected window.")
            with tabs[2]:
                r = px.pct_change(); vol60=r.rolling(60).std()*np.sqrt(252); sharpe60=(r.rolling(60).mean()/r.rolling(60).std())*np.sqrt(252)
                st.line_chart(pd.DataFrame({"Volatility 60d (ann.)":vol60,"Sharpe 60d":sharpe60})); st.caption("Risk and risk-adjusted return on a 60-day window.")
            with tabs[3]:
                roll_max = px.cummax(); drawdown=(px/roll_max-1.0)
                st.line_chart(pd.DataFrame({"Drawdown": drawdown})); st.caption("Peak-to-trough declines.")

    # Export
    csv_bytes = pretty.to_csv().encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "stock_ratings.csv", "text/csv")
    xlsx_io = io.BytesIO()
    with pd.ExcelWriter(xlsx_io, engine="openpyxl") as w: pretty.round(4).to_excel(w, sheet_name="Ratings")
    st.download_button("⬇️ Download Excel", xlsx_io.getvalue(), "stock_ratings.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------- Portfolio App --------------------------
def parse_holdings(text: str) -> pd.DataFrame:
    """
    Parse lines like:
      AAPL, 25
      MSFT, 20%
      NVDA 30
    Returns weights normalized to sum 1.0
    """
    rows=[]
    for line in text.splitlines():
        if not line.strip(): continue
        parts = [p.strip() for p in line.replace(";",",").split(",") if p.strip()]
        if len(parts)==0: continue
        t = yf_symbol(parts[0])
        w = 0.0
        if len(parts)>1:
            s = parts[1].replace("%","")
            try: w = float(s)/100.0 if "%" in parts[1] else float(s)/100.0 if float(s)>1.5 else float(s)
            except Exception: w = 0.0
        rows.append({"ticker":t,"weight":w})
    df = pd.DataFrame(rows)
    if df.empty: return df
    # if all weights are 0, equal-weight
    if (df["weight"].fillna(0)==0).all():
        df["weight"]=1.0/len(df)
    else:
        df["weight"]=df["weight"].fillna(0)
        s = df["weight"].sum()
        df["weight"]=df["weight"]/s if s>0 else 1.0/len(df)
    return df

def app_portfolio():
    # Centered editor
    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    st.markdown("**Paste your holdings** (ticker, weight%). One per line. Example:\n\n`AAPL, 25%`\n`MSFT, 20%`\n`NVDA, 30%`\n`AMZN, 25%`")
    holdings_text = st.text_area(" ", "AAPL, 25%\nMSFT, 25%\nNVDA, 25%\nAMZN, 25%", label_visibility="collapsed", height=140, key="ptf_text")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Settings
    with st.expander("Advanced settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            universe_mode = st.selectbox(
                "Peer universe",
                ["Auto by index membership", "S&P 500", "Dow 30", "NASDAQ 100", "Custom (paste list)"],
                index=0, key="universe_ptf"
            )
        with c2:
            peer_sample_n = st.slider("Peer sample size", 30, 250, 150, 10, key="peer_n_ptf")
        with c3:
            history = st.selectbox("History", ["1y", "2y", "5y"], index=0, key="hist_ptf")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            w_fund = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.45, 0.05, key="wf_ptf")
        with c5:
            w_tech = st.slider("Weight: Technicals",   0.0, 1.0, 0.40, 0.05, key="wt_ptf")
        with c6:
            w_macro= st.slider("Weight: Macro (VIX)",  0.0, 1.0, 0.10, 0.05, key="wm_ptf")
        with c7:
            w_div  = st.slider("Weight: Diversification", 0.0, 1.0, 0.05, 0.05, key="wd_ptf")
        custom_raw = st.text_area("Custom peers (comma-separated)", "", key="custom_ptf") \
                      if universe_mode=="Custom (paste list)" else ""
        show_debug = st.checkbox("Show debug info", value=False, key="dbg_ptf")

    # Parse holdings
    df_hold = parse_holdings(holdings_text)
    if df_hold.empty or df_hold["ticker"].nunique()==0:
        st.info("Paste at least one holding to run the rating."); return

    tickers = df_hold["ticker"].tolist()

    # RUN
    with st.status("Crunching the numbers…", expanded=False) as status:
        prog = st.progress(0)
        status.update(label="Building peer universe…")
        universe = build_universe(tickers, universe_mode, peer_sample_n, custom_raw); prog.progress(8)

        status.update(label="Downloading prices…")
        prices, ok, _ = fetch_prices_chunked_with_fallback(universe, period=history, interval="1d",
                                                           chunk=50, min_ok=min(80, max(50, int(peer_sample_n*0.5))))
        # panel only for held tickers (we'll also need peers for z-scores)
        # but keep all peers in prices for cross-sectional z-scores
        prog.progress(35)
        if prices.empty: st.error("No prices fetched."); return

        # Technicals (cross-section on peers)
        status.update(label="Computing technicals…")
        panel_all = {t: prices[t].dropna() for t in prices.columns if t in prices.columns and prices[t].dropna().size>0}
        tech_all = technical_scores(panel_all)
        for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
            if col in tech_all.columns: tech_all[f"{col}_z"] = zscore_series(tech_all[col])
        TECH_score_all = tech_all[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech_all.columns]].mean(axis=1)
        prog.progress(58)

        # Fundamentals (cross-section on peers)
        status.update(label="Fetching fundamentals…")
        fund_raw_all = fetch_fundamentals_simple(list(panel_all.keys()))
        fdf_all = pd.DataFrame(index=fund_raw_all.index)
        for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins","grossMargins","operatingMargins","ebitdaMargins"]:
            if col in fund_raw_all.columns: fdf_all[f"{col}_z"] = zscore_series(fund_raw_all[col])
        for col in ["trailingPE","forwardPE","debtToEquity"]:
            if col in fund_raw_all.columns: fdf_all[f"{col}_z"] = zscore_series(-fund_raw_all[col])
        FUND_score_all = fdf_all.mean(axis=1) if len(fdf_all.columns) else pd.Series(0.0, index=fund_raw_all.index)
        prog.progress(75)

        # Macro
        status.update(label="Assessing macro regime…")
        vix_series = fetch_vix_series(period="6mo", interval="1d")
        MACRO_score_val, MACRO_level, MACRO_trend = macro_overlay_score_from_series(vix_series)
        prog.progress(82)

        # Combine per-name (for held tickers)
        idx_all = pd.Index(list(panel_all.keys()))
        out_all = pd.DataFrame(index=idx_all)
        out_all["FUND_score"]  = FUND_score_all.reindex(idx_all).fillna(0.0)
        out_all["TECH_score"]  = TECH_score_all.reindex(idx_all).fillna(0.0)
        out_all["MACRO_score"] = MACRO_score_val
        wsum = (w_fund + w_tech + w_macro)
        wsum = wsum if wsum>0 else 1.0
        wf, wt, wm = w_fund/wsum, w_tech/wsum, w_macro/wsum
        out_all["COMPOSITE"] = wf*out_all["FUND_score"] + wt*out_all["TECH_score"] + wm*out_all["MACRO_score"]
        out_all["RATING_0_100"] = percentile_rank(out_all["COMPOSITE"])

        # Slice for holdings
        held = df_hold["ticker"].tolist()
        per_name = out_all.reindex(held).copy()

        # Diversification:
        # 1) Sector/industry mix (from .info)
        status.update(label="Evaluating diversification…")
        meta = fetch_company_meta(held)
        merged = df_hold.set_index("ticker").join(meta, how="left")
        # sector weights
        sec = merged.groupby(merged["sector"].fillna("Unknown"))["weight"].sum()
        if sec.empty: sec = pd.Series({"Unknown":1.0})
        hhi_sector = float((sec**2).sum())  # 1 -> very concentrated
        effectiveN_sector = 1.0/hhi_sector if hhi_sector>0 else 1.0
        # normalize to [0,1] (1 = great diversification). Cap with max target N=10
        targetN = min(10, max(1, len(sec)))
        sector_div_score = float(np.clip((effectiveN_sector-1)/(targetN-1 if targetN>1 else 1), 0, 1))

        # 2) Name concentration (max weight)
        max_w = float(merged["weight"].max())
        name_div_score = float(np.clip(1.0 - max_w/0.25, 0.0, 1.0))  # 0 if top holding 25%+, 1 if <=0%

        # 3) Correlation (lower avg corr -> higher score)
        # Use daily returns on held names
        ret = prices[held].pct_change().dropna(how="all")
        if ret.shape[1] >= 2:
            corr = ret.corr().values
            # avg off-diagonal
            n = corr.shape[0]
            avg_corr = (corr.sum() - np.trace(corr)) / max(1, (n*n - n))
            corr_div_score = float(np.clip(1.0 - max(0.0, avg_corr), 0.0, 1.0))  # if avg corr ~0.6 -> score 0.4
        else:
            corr_div_score = 0.5

        # Aggregate diversification score
        DIV_score = 0.5*sector_div_score + 0.2*name_div_score + 0.3*corr_div_score

        # Portfolio composite: weight per-name composite; blend diversification as separate component
        per_name = per_name.join(df_hold.set_index("ticker"), how="left")
        per_name["weighted_composite"] = per_name["COMPOSITE"] * per_name["weight"]
        port_signal = float(per_name["weighted_composite"].sum())
        PORT_macro = MACRO_score_val
        PORT_div   = DIV_score

        # Final portfolio score: combine signal & macro & diversification
        # normalize weights across (signal, macro, diversification) using (w_fund+wt+w_macro) already inside signal.
        total_for_final = 1.0 + w_div  # 1.0 = signal+macro block; w_div as additional
        port_final = (port_signal + 0.0) * (1.0/total_for_final) + PORT_div * (w_div/total_for_final)
        # Rank-style 0–100 not meaningful across one portfolio; map [-1,+1] ~> [0,100] heuristically
        port_score_0_100 = float(np.clip((port_final+1)/2, 0, 1)*100)

        prog.progress(100); status.update(label="Done!", state="complete")

    # ---- Output: Portfolio summary ----
    st.markdown("## 🧺 Portfolio — Scores")
    cA,cB,cC,cD = st.columns(4)
    cA.metric("Portfolio Score (0–100)", f"{port_score_0_100:.1f}")
    cB.metric("Signal (weighted composite)", f"{port_signal:.3f}")
    cC.metric("Macro (VIX level+trend)", f"{PORT_macro:.3f}")
    cD.metric("Diversification", f"{PORT_div:.3f}")

    with st.expander("Why this portfolio rating?"):
        st.markdown("**Per-name contribution (weights × composite)**")
        show = per_name[["weight","FUND_score","TECH_score","MACRO_score","COMPOSITE","weighted_composite"]].copy()
        show = show.rename(columns={"weight":"Weight","FUND_score":"Fundamentals","TECH_score":"Technicals","MACRO_score":"Macro (VIX)","COMPOSITE":"Composite","weighted_composite":"Weight × Comp"})
        st.dataframe(show.round(4), use_container_width=True)

        st.markdown("**Diversification breakdown**")
        st.write(f"- Sector HHI: {hhi_sector:.3f} → Effective N ≈ {effectiveN_sector:.2f} → Sector diversity score: **{sector_div_score:.3f}**")
        st.write(f"- Max single name weight: {max_w*100:.1f}% → Name concentration score: **{name_div_score:.3f}**")
        st.write(f"- Avg pairwise correlation: {0 if ret.shape[1]<2 else avg_corr:.2f} → Correlation diversity score: **{corr_div_score:.3f}**")
        st.caption("Diversification score = 50% sector mix + 30% correlation + 20% name concentration. Lower concentration and lower average correlation yield better scores.")

        # Macro explanation
        vix_series = fetch_vix_series(period="6mo", interval="1d")
        vix_last = float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else np.nan
        vix_ema20 = float(ema(vix_series,20).iloc[-1]) if vix_series is not None and not vix_series.empty else np.nan
        st.markdown("**Macro (VIX) — level & trend**")
        if not np.isnan(vix_last):
            rel_gap = (vix_last - vix_ema20)/vix_ema20 if (not np.isnan(vix_ema20) and vix_ema20!=0) else np.nan
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Current VIX", f"{vix_last:.2f}")
            m2.metric("VIX vs EMA20", f"{(rel_gap*100):.1f}%" if not np.isnan(rel_gap) else "N/A")
            m3.metric("Level score", f"{MACRO_level:.2f}")
            m4.metric("Trend score", f"{MACRO_trend:.2f}")
            st.caption("Level maps today’s VIX to 0–1 (12→1.0, 28→0.0). Trend compares VIX to its 20-day EMA: rising above trend reduces the score (risk building), falling below boosts it.")
        else:
            st.caption("Macro details unavailable (VIX not loaded).")

    # ---- Charts: Portfolio ----
    st.markdown("## 📈 Portfolio Charts")
    # Build portfolio equity curve from weights
    px_held = prices[held].dropna(how="all")
    r = px_held.pct_change().fillna(0)
    # align weights
    w_vec = df_hold.set_index("ticker")["weight"].reindex(px_held.columns).fillna(0).values
    port_r = (r * w_vec).sum(axis=1)
    equity = (1 + port_r).cumprod()
    tabs = st.tabs(["Cumulative (Portfolio)","Volatility & Sharpe (60d)","Drawdown"])

    with tabs[0]:
        st.line_chart(pd.DataFrame({"Portfolio cumulative": equity}))
        st.caption("Cumulative growth of £1 invested in the weighted portfolio over the selected period.")

    with tabs[1]:
        vol60 = port_r.rolling(60).std()*np.sqrt(252)
        sharpe60 = (port_r.rolling(60).mean()/port_r.rolling(60).std())*np.sqrt(252)
        st.line_chart(pd.DataFrame({"Volatility 60d (ann.)": vol60, "Sharpe 60d": sharpe60}))
        st.caption("Portfolio risk and risk-adjusted return using a 60-day window.")

    with tabs[2]:
        roll_max = equity.cummax(); drawdown = equity/roll_max - 1.0
        st.line_chart(pd.DataFrame({"Drawdown": drawdown}))
        st.caption("Peak-to-trough declines for the whole portfolio.")

    # Export
    out_show = show.copy()
    out_show["Weight"] = (out_show["Weight"]*100).round(2).astype(str) + "%"
    csv_bytes = out_show.to_csv().encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "portfolio_breakdown.csv", "text/csv")

    xlsx_io = io.BytesIO()
    with pd.ExcelWriter(xlsx_io, engine="openpyxl") as w: out_show.round(4).to_excel(w, sheet_name="Portfolio")
    st.download_button("⬇️ Download Excel", xlsx_io.getvalue(), "portfolio_breakdown.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------- Router --------------------------
def app_router():
    if not st.session_state.entered:
        landing(); return
    mode = st.session_state.mode
    if mode == "portfolio":
        st.title("⭐️ Rate My Portfolio")
        st.caption("Paste your holdings with weights; we’ll rate it and judge diversification too.")
        app_portfolio()
    else:
        st.title("⭐️ Rate My Stock")
        st.caption("Type a ticker. We’ll grab its peers and rate it with friendly explanations.")
        app_stock()

app_router()
