# app.py
# Rate My — Stocks & Portfolio
# ---------------------------------------------------------
# Streamlit app with: centered landing page + logo,
# stock rating with robust peer selection (built-in/upload/url/paste),
# richer explanations (fund/tech/macro), downloadable breakdown,
# portfolio rating with submit-based syncing, currency & total value,
# 12m momentum sourced from 5y history.
# ---------------------------------------------------------

import base64
import io
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# =========================
# ------- Utilities -------
# =========================

def nanmean_safe(vals):
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    return np.nan if not vals else float(np.mean(vals))

def _z_to_pct(z):
    # convert z-score to percentile (~normal)
    if pd.isna(z):
        return np.nan
    return float(100 * 0.5 * (1 + math.erf(float(z) / math.sqrt(2))))

@st.cache_data(show_spinner=False)
def _get(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r
    except Exception:
        return None
    return None

def try_float(x):
    try: return float(x)
    except: return np.nan

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================
# ------- Styling ---------
# =========================

# Minimal triangle logo as inline SVG
def logo_svg(size=64):
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle">
  <defs>
    <linearGradient id="grad" x1="0" y1="100" x2="0" y2="0">
      <stop offset="0%"  stop-color="#dc2626"/>
      <stop offset="45%" stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#16a34a"/>
    </linearGradient>
  </defs>
  <polygon points="50,5 95,95 5,95" fill="url(#grad)"/>
</svg>
"""

st.set_page_config(
    page_title="Rate My",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
<style>
/* Layout */
.block-container { max-width: 1200px; }

/* Header */
.header-shell     { display:flex; justify-content:center; margin:.8rem 0 1.2rem; }
.header-inner     { display:flex; align-items:center; gap:14px; }
.header-title     { font-size:3.6rem; font-weight:850; line-height:1.05; margin:0; letter-spacing:.25px; }
.header-sub       { text-align:center; color:#9aa0a6; margin-top:.35rem; font-size:1.06rem; }

/* Buttons */
.bigbtn .stButton>button { padding:.9rem 1.4rem; font-size:1.1rem; border-radius:12px; }

/* KPI cards */
.kpi        { padding:1rem 1.1rem; border-radius:12px; background:#111418; border:1px solid #222; }
.kpi-big    { font-size:2.2rem; font-weight:800; margin-top:.25rem; }
.small      { color:#9aa0a6; font-size:.92rem; }

/* Banner */
.banner     { background:#0c2f22; color:#cdebdc; border-radius:10px; padding:.9rem 1.1rem; margin:.75rem 0 1.25rem; }

/* Tables */
.table-mini td, .table-mini th { padding:.4rem .6rem !important; font-size:.92rem !important; }

/* Collapsible look */
.detail { background:#0e1217; border:1px solid #1c232d; border-radius:12px; padding:1rem 1.25rem; }

/* Center ticker input a bit wider */
.center-input .stTextInput>div>div>input { text-align:center; font-size:1.05rem; }

/* Chart caption */
.caption { color:#9aa0a6; font-size:.92rem; margin-top:.35rem; }
</style>
""",
    unsafe_allow_html=True,
)

def render_header_centered(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="header-shell">
          <div class="header-inner">
            <div class="header-title">{title}</div>
            <div>{logo_svg(64)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f"<div class='header-sub'>{subtitle}</div>", unsafe_allow_html=True)

# =========================
# --- Peer universe data ---
# =========================

@st.cache_data(show_spinner=False)
def peers_sp500():
    r = _get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if not r: return []
    try:
        tables = pd.read_html(r.text)
        df = next((t for t in tables if "Symbol" in t.columns), None)
        if df is None: return []
        syms = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        return sorted(syms.unique().tolist())
    except:
        return []

@st.cache_data(show_spinner=False)
def peers_nasdaq100():
    r = _get("https://en.wikipedia.org/wiki/Nasdaq-100")
    if not r: return []
    try:
        tables = pd.read_html(r.text)
        df = next((t for t in tables if ("Ticker" in t.columns) or ("Ticker symbol" in t.columns)), None)
        if df is None:
            # fallback: first table
            df = tables[0]
        col = "Ticker" if "Ticker" in df.columns else "Ticker symbol" if "Ticker symbol" in df.columns else df.columns[0]
        syms = df[col].astype(str).str.upper().str.replace(".", "-", regex=False)
        return sorted(syms.unique().tolist())
    except:
        return []

@st.cache_data(show_spinner=False)
def peers_dow30():
    r = _get("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
    if not r: return []
    try:
        tables = pd.read_html(r.text)
        df = next((t for t in tables if "Symbol" in t.columns), None)
        if df is None: return []
        syms = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        return sorted(syms.unique().tolist())
    except:
        return []

PEER_UNIVERSE_SOURCES = {
    "S&P 500": peers_sp500,
    "NASDAQ-100": peers_nasdaq100,
    "Dow 30": peers_dow30,
}

# Custom peers helpers
@st.cache_data(show_spinner=False)
def _clean_tickers(raw_list):
    out = []
    for t in raw_list:
        if not isinstance(t, str):
            t = str(t)
        t = t.upper().strip().replace(".", "-")
        if t and t.isascii():
            out.append(t)
    return sorted(list(set(out)))

@st.cache_data(show_spinner=False)
def parse_peer_file(uploaded_file) -> list:
    if uploaded_file is None: return []
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except:
        return []
    cols = {c.strip().lower(): c for c in df.columns}
    key = cols.get("ticker") or cols.get("symbol") or df.columns[0]
    return _clean_tickers(df[key].dropna().astype(str).tolist())

@st.cache_data(show_spinner=False)
def parse_peer_url(raw_url: str) -> list:
    if not raw_url: return []
    r = _get(raw_url)
    if not r: return []
    txt = r.text
    try:
        df = pd.read_csv(io.StringIO(txt))
    except:
        try:
            df = pd.read_csv(io.StringIO(txt), sep="\t")
        except:
            return []
    cols = {c.strip().lower(): c for c in df.columns}
    key = cols.get("ticker") or cols.get("symbol") or df.columns[0]
    return _clean_tickers(df[key].dropna().astype(str).tolist())

# =========================
# ------- Data fetch -------
# =========================

@st.cache_data(show_spinner=False)
def price_history(tickers, period="2y", interval="1d"):
    if isinstance(tickers, str): tickers = [tickers]
    try:
        df = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)["Close"]
    except Exception:
        return pd.DataFrame()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all")
    return df

@st.cache_data(show_spinner=False)
def price_history_5y(tickers):
    # For 12m momentum robustness
    return price_history(tickers, period="5y", interval="1d")

@st.cache_data(show_spinner=False)
def vix_history():
    try:
        v = yf.download("^VIX", period="2y", interval="1d", auto_adjust=False, progress=False)["Close"].dropna()
        return v
    except:
        return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def _yf_fast_info(tickers):
    # light fundamentals; fetching .info one-by-one often throttles — use fast_info + fallback
    out = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            fi = tk.fast_info
            # build small dict; we’ll compute z vs peers later
            out[t] = {
                "forwardPE": try_float(getattr(fi, "forward_pe", np.nan)),
                "trailingPE": try_float(getattr(fi, "trailing_pe", np.nan)),
                "profitMargins": np.nan,   # will try later
                "operatingMargins": np.nan,
                "grossMargins": np.nan,
                "ebitdaMargins": np.nan,
                "returnOnEquity": np.nan,
                "debtToEquity": np.nan,
                "revenueGrowth": np.nan,
                "earningsGrowth": np.nan,
            }
        except Exception:
            out[t] = {}
    return out

@st.cache_data(show_spinner=False)
def _yf_full_info_some(tickers, max_names=60):
    # Try to enrich for a subset to avoid heavy throttling
    out = {}
    for i, t in enumerate(tickers):
        if i >= max_names: break
        try:
            info = yf.Ticker(t).info
            out[t] = {
                "profitMargins": try_float(info.get("profitMargins")),
                "operatingMargins": try_float(info.get("operatingMargins")),
                "grossMargins": try_float(info.get("grossMargins")),
                "ebitdaMargins": try_float(info.get("ebitdaMargins")),
                "returnOnEquity": try_float(info.get("returnOnEquity")),
                "debtToEquity": try_float(info.get("debtToEquity")),
                "revenueGrowth": try_float(info.get("revenueGrowth")),
                "earningsGrowth": try_float(info.get("earningsGrowth")),
            }
        except Exception:
            out[t] = {}
    return out

def merge_fundamentals(primary: dict, enrich: dict):
    # merge dicts
    out = {}
    for t in primary.keys() | enrich.keys():
        a = primary.get(t) or {}
        b = enrich.get(t) or {}
        out[t] = {**a, **{k: (a.get(k) if not pd.isna(a.get(k)) else b.get(k)) for k in set(a) | set(b)}}
    return out

# =========================
# ------- Scoring ----------
# =========================

def _zscore(series: pd.Series, higher_is_better=True):
    s = series.astype(float).copy()
    m = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(m) or sd == 0 or pd.isna(sd):
        return pd.Series(np.nan, index=s.index)
    z = (s - m) / sd
    if not higher_is_better:
        z = -z
    return z

def fundamentals_panel(peers: list):
    base = _yf_fast_info(peers)
    enrich = _yf_full_info_some(peers, max_names=min(120, len(peers)))
    data = merge_fundamentals(base, enrich)
    df = pd.DataFrame.from_dict(data, orient="index")
    # Z-scores (sign so that “higher is better”)
    zcols = {}
    zcols["trailingPE_z"] = _zscore(df["trailingPE"], higher_is_better=False)
    zcols["forwardPE_z"]  = _zscore(df["forwardPE"],  higher_is_better=False)
    zcols["profitMargins_z"]   = _zscore(df["profitMargins"], higher_is_better=True)
    zcols["operatingMargins_z"]= _zscore(df["operatingMargins"], higher_is_better=True)
    zcols["grossMargins_z"]    = _zscore(df["grossMargins"], higher_is_better=True)
    zcols["ebitdaMargins_z"]   = _zscore(df["ebitdaMargins"], higher_is_better=True)
    zcols["returnOnEquity_z"]  = _zscore(df["returnOnEquity"], higher_is_better=True)
    zcols["debtToEquity_z"]    = _zscore(-df["debtToEquity"], higher_is_better=True)  # lower D/E better
    zcols["revenueGrowth_z"]   = _zscore(df["revenueGrowth"], higher_is_better=True)
    zcols["earningsGrowth_z"]  = _zscore(df["earningsGrowth"], higher_is_better=True)
    zdf = pd.DataFrame(zcols)
    # composite (equal buckets)
    growth   = zdf[["revenueGrowth_z", "earningsGrowth_z"]].mean(axis=1, skipna=True)
    quality  = zdf[["returnOnEquity_z","profitMargins_z","operatingMargins_z","grossMargins_z","ebitdaMargins_z"]].mean(axis=1, skipna=True)
    value    = zdf[["forwardPE_z","trailingPE_z"]].mean(axis=1, skipna=True)
    balance  = zdf[["debtToEquity_z"]].mean(axis=1, skipna=True)
    comp     = pd.concat([growth, quality, value, balance], axis=1).mean(axis=1, skipna=True)
    zdf["FUND_score"] = comp
    return zdf

def technicals_panel(peers: list, main: str):
    # Close (2y) for EMA/RSI/MACD; and 5y for 12m momentum
    prices_2y = price_history(peers, period="2y")
    prices_5y = price_history_5y([main])
    out = {}
    for t in peers:
        row = {"dma_gap": np.nan, "macd_hist": np.nan, "rsi_strength": np.nan, "mom12m": np.nan}
        s = prices_2y.get(t) if t in prices_2y.columns else None
        if s is not None:
            s = s.dropna()
            if len(s) > 60:
                ema50 = s.ewm(span=50, adjust=False).mean()
                row["dma_gap"] = (s.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]
                # MACD (12,26,9) histogram
                ema12 = s.ewm(span=12, adjust=False).mean()
                ema26 = s.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                sig  = macd.ewm(span=9, adjust=False).mean()
                row["macd_hist"] = (macd - sig).iloc[-1]
                # RSI (approx)
                delta = s.diff()
                up  = delta.clip(lower=0).rolling(14).mean()
                dn  = (-delta.clip(upper=0)).rolling(14).mean()
                rs  = up / (dn + 1e-9)
                rsi = 100 - (100/(1 + rs))
                row["rsi_strength"] = (rsi.iloc[-1] - 50)/50.0  # -1..+1
        if t == main and prices_5y.shape[1] == 1 and main in prices_5y.columns:
            s5 = prices_5y[main].dropna()
            if len(s5) > 252*2:
                # 12m momentum (close/close lagged 252)
                last = s5.iloc[-1]
                lag  = s5.shift(252).iloc[-1]
                if pd.notna(lag) and lag != 0:
                    row["mom12m"] = (last/lag - 1.0)
        out[t] = row

    df = pd.DataFrame(out).T
    # convert each feature to z across peers (higher=better)
    z = {}
    for c in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        z[f"{c}_z"] = _zscore(df[c], higher_is_better=True)
    zdf = pd.DataFrame(z)
    zdf["TECH_score"] = zdf.mean(axis=1, skipna=True)
    return zdf, prices_2y

def macro_kpi():
    v = vix_history()
    if v.empty:
        return {"macro_score": np.nan, "vix": np.nan, "vix_ema20": np.nan, "trend_gap": np.nan}
    v = v.dropna()
    last = v.iloc[-1]
    ema20 = v.ewm(span=20, adjust=False).mean().iloc[-1]
    # Level score: map low VIX to ~1, high VIX to ~0 (12→1, 28→0)
    lvl = clamp(1 - (last - 12)/(28 - 12), 0, 1)
    # Trend: if VIX below its EMA → safer (score↑); above → risk building (score↓)
    gap = (last - ema20)/ema20
    trn = clamp(1 - (gap+0.1)/0.3, 0, 1)  # gentle mapping
    macro = 0.6*lvl + 0.4*trn
    return {"macro_score": macro, "vix": last, "vix_ema20": ema20, "trend_gap": gap}

def composite_score(fund, tech, macro):
    # combine 3 pillars equally, then convert to 0-100
    z = np.nanmean([fund, tech, (macro-0.5)/0.25], axis=0)  # macro is in 0..1 → center to ~z-ish
    z = np.clip(z, -2.5, 2.5)
    pct = _z_to_pct(z)  # 0..100
    return z, pct

# =========================
# ------ Narratives --------
# =========================

def _pct_label(z): return f"{_z_to_pct(z):.0f}th pct" if pd.notna(z) else "n/a"

def _bucket_word(z):
    if pd.isna(z):      return "insufficient data"
    if z >= 1.3:        return "strongly above peers (bullish)"
    if z >= 0.6:        return "above peers (constructive)"
    if z <= -1.0:       return "well below peers (bearish)"
    if z <= -0.5:       return "below peers (caution)"
    return "around peer median (neutral)"

def fundamentals_story(zrow: pd.Series) -> str:
    growth   = nanmean_safe([zrow.get("revenueGrowth_z"), zrow.get("earningsGrowth_z")])
    quality  = nanmean_safe([
        zrow.get("returnOnEquity_z"), zrow.get("profitMargins_z"),
        zrow.get("operatingMargins_z"), zrow.get("grossMargins_z"),
        zrow.get("ebitdaMargins_z")
    ])
    value    = nanmean_safe([zrow.get("forwardPE_z"), zrow.get("trailingPE_z")])  # higher is cheaper
    leverage = zrow.get("debtToEquity_z")

    lines = []
    lines.append("**Growth**")
    if pd.isna(growth):
        lines.append("- Not enough growth data to compare with peers.")
    else:
        lines.append(f"- Revenue/EPS growth looks **{_bucket_word(growth)}** (≈ {_pct_label(growth)}).")

    lines.append("\n**Quality & margins**")
    if pd.isna(quality):
        lines.append("- Insufficient margin/ROE data for a clean comparison.")
    else:
        parts = []
        for k, label in [
            ("returnOnEquity_z","ROE"),
            ("profitMargins_z","profit margin"),
            ("operatingMargins_z","operating margin"),
            ("grossMargins_z","gross margin"),
        ]:
            v = zrow.get(k)
            if pd.notna(v): parts.append(f"{label} {_pct_label(v)}")
        tail = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"- Overall profitability is **{_bucket_word(quality)}**{tail}.")

    lines.append("\n**Valuation** (lower PE → better after sign-flip)")
    if pd.isna(value):
        lines.append("- Valuation unavailable.")
    else:
        subs = []
        if pd.notna(zrow.get("forwardPE_z")):  subs.append(f"forward PE {_pct_label(zrow['forwardPE_z'])}")
        if pd.notna(zrow.get("trailingPE_z")): subs.append(f"trailing PE {_pct_label(zrow['trailingPE_z'])}")
        lines.append(f"- Valuation screens **{_bucket_word(value)}** vs peers ({', '.join(subs) or 'n/a'}).")

    lines.append("\n**Leverage**")
    if pd.isna(leverage):
        lines.append("- Debt/Equity not reported.")
    else:
        lev_label = ("conservative" if leverage >= 0.6 else "elevated" if leverage <= -0.5 else "moderate")
        lines.append(f"- Balance sheet looks **{lev_label}** (≈ {_pct_label(leverage)}).")

    watch = []
    if pd.notna(growth) and growth < 0:  watch.append("watch for growth acceleration (rev/EPS)")
    if pd.notna(quality) and quality < 0: watch.append("scope for margin/ROE improvement")
    if pd.notna(value) and value < 0:     watch.append("valuation reset vs peers (lower PE)")
    if pd.notna(leverage) and leverage < 0: watch.append("deleveraging / interest-cover progress")

    if watch: lines.append("\n**What to watch:** " + ", ".join(watch) + ".")
    else:     lines.append("\n**What to watch:** sustain quality & growth; monitor valuation drift.")

    return "\n".join(lines)

def technicals_story(trow: pd.Series) -> str:
    gap = trow.get("dma_gap")
    macd = trow.get("macd_hist")
    rsi_strength = trow.get("rsi_strength")
    mom = trow.get("mom12m")

    lines = []
    if pd.notna(gap):
        gp = gap*100
        if gp > 6:    lines.append(f"- Price **{gp:.1f}% above** 50-day EMA → **bullish trend**.")
        elif gp > 2:  lines.append(f"- Price **{gp:.1f}% above** 50-day EMA → constructive.")
        elif gp < -6: lines.append(f"- Price **{abs(gp):.1f}% below** 50-day EMA → **trend risk**.")
        else:         lines.append("- Price near 50-day EMA → neutral trend.")

    if pd.notna(macd):
        if macd > 0:  lines.append("- MACD histogram **positive** → momentum building.")
        else:         lines.append("- MACD histogram **negative** → momentum fading.")
    if pd.notna(rsi_strength):
        rsi = 50 + 50*rsi_strength
        if rsi > 70:      lines.append(f"- RSI ≈ **{rsi:.0f}** → overbought risk; pullbacks common.")
        elif rsi < 30:    lines.append(f"- RSI ≈ **{rsi:.0f}** → oversold; mean-reversion possible.")
        else:             lines.append(f"- RSI ≈ **{rsi:.0f}** → mid-range.")
    if pd.notna(mom):
        lines.append(f"- 12-month momentum ≈ **{mom*100:.1f}%** (using 5-year history).")

    if not lines:
        lines = ["- Insufficient price history for technicals."]

    lines.append("\n**Trading lens:** use EMA50/MACD for trend confirmation; watch RSI for risk-management.")
    return "\n".join(lines)

def macro_story(k: dict) -> str:
    if not k or pd.isna(k.get("macro_score")):
        return "Macro data unavailable."
    vix = k["vix"]; ema = k["vix_ema20"]; gap = k["trend_gap"]
    level_note = "low & calm" if vix < 14 else "moderate" if vix < 22 else "elevated"
    if gap < -0.03:
        trend_note = "below trend (risk cooling)"
    elif gap > 0.03:
        trend_note = "above trend (risk building)"
    else:
        trend_note = "near trend"

    lines = [
        f"- VIX ≈ **{vix:.2f}** ({level_note}); vs EMA20 ≈ **{ema:.2f}** → {trend_note}.",
        "- Macro score blends **level** (lower is safer) and **trend** (falling volatility improves score).",
        "• When VIX rises above trend, tighten risk; when it falls below trend, conditions tend to support risk-on.",
    ]
    return "\n".join(lines)

# =========================
# ------- Stock page -------
# =========================

def page_stock():
    render_header_centered("Rate My Stock", "Pick a stock; we’ll grab its peers and rate it with explanations & charts.")

    # --- Ticker input ---
    with st.container():
        st.markdown('<div class="center-input">', unsafe_allow_html=True)
        ticker = st.text_input(" ", value="AAPL", placeholder="Type a ticker…").upper().strip().replace(".", "-")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Peer source ---
    st.markdown("### Peer universe")
    peer_source = st.radio("Source", ["Built-in", "Upload file", "Paste tickers", "From URL"], horizontal=True, index=0)

    peers = []; peer_label = ""
    if peer_source == "Built-in":
        choice = st.selectbox("Universes", list(PEER_UNIVERSE_SOURCES.keys()), index=0)
        peers = PEER_UNIVERSE_SOURCES[choice]() or []
        peer_label = choice
    elif peer_source == "Upload file":
        up = st.file_uploader("Upload CSV/XLSX with a `Ticker` or `Symbol` column", type=["csv","xlsx","xls"])
        tmpl = pd.DataFrame({"Ticker":["AAPL","MSFT","NVDA","AMZN"]}).to_csv(index=False).encode()
        st.download_button("Download peer template (CSV)", tmpl, "peer_template.csv")
        peers = parse_peer_file(up) if up else []
        peer_label = f"Custom upload ({len(peers)} tickers)"
    elif peer_source == "Paste tickers":
        txt = st.text_area("Paste: one ticker per line", value="AAPL\nMSFT\nNVDA\nAMZN", height=120)
        peers = _clean_tickers(txt.splitlines())
        peer_label = f"Pasted list ({len(peers)} tickers)"
    else:
        url = st.text_input("Raw CSV/TSV URL (GitHub raw etc.)")
        peers = parse_peer_url(url) if url else []
        peer_label = "URL source"

    if ticker and ticker not in peers:
        peers = [ticker] + peers

    # Banner (no macro here)
    st.markdown(
        f"<div class='banner'>Peers loaded: {max(0,len(peers)-1)} | Source: {peer_label}</div>",
        unsafe_allow_html=True,
    )

    # Run rating
    btn = st.button("Rate", type="primary")
    if not btn: return

    if not ticker or len(peers) < 2:
        st.warning("Please enter a ticker and provide a peer universe with at least a few names.")
        return

    with st.spinner("Fetching prices & fundamentals…"):
        fund = fundamentals_panel(peers)
        tech, px = technicals_panel(peers, ticker)
        k_macro = macro_kpi()

    # Merge row for main ticker
    row = pd.concat([fund.loc[[ticker]], tech.loc[[ticker]]], axis=1)
    fscore = row["FUND_score"].iloc[0]
    tscore = row["TECH_score"].iloc[0]
    mscore = k_macro["macro_score"]

    z, pct = composite_score(fscore, tscore, mscore)
    rating = "Strong Buy" if pct>=85 else "Buy" if pct>=60 else "Hold" if pct>=40 else "Sell" if pct>=20 else "Strong Sell"

    # KPIs
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1])
    with c1: st.markdown("**Fundamentals**"); st.markdown(f"<div class='kpi kpi-big'>{fscore:.3f}</div>", unsafe_allow_html=True)
    with c2: st.markdown("**Technicals**");   st.markdown(f"<div class='kpi kpi-big'>{tscore:.3f}</div>", unsafe_allow_html=True)
    with c3: st.markdown("**Macro (VIX)**");  st.markdown(f"<div class='kpi kpi-big'>{mscore:.3f}</div>", unsafe_allow_html=True)
    with c4: st.markdown("**Score (0–100)**");st.markdown(f"<div class='kpi kpi-big'>{pct:.1f}</div>", unsafe_allow_html=True)
    with c5: st.markdown("**Recommendation**"); st.markdown(f"<div class='kpi kpi-big'>{rating}</div>", unsafe_allow_html=True)

    st.markdown("### Why this rating?")
    with st.expander(f"{ticker} — {rating} (Score: {pct:.1f})", expanded=True):
        # Fundamentals details
        st.markdown("#### Fundamentals — details & interpretation")
        fcols = ["profitMargins_z","operatingMargins_z","grossMargins_z","ebitdaMargins_z",
                 "returnOnEquity_z","revenueGrowth_z","earningsGrowth_z","forwardPE_z","trailingPE_z","debtToEquity_z"]
        ftable = fund.loc[[ticker], fcols].T.rename(columns={ticker:"z"}).reset_index().rename(columns={"index":"metric"})
        st.dataframe(ftable, use_container_width=True)
        st.markdown(fundamentals_story(fund.loc[ticker]))

        # Technicals
        st.markdown("#### Technicals — indicators & interpretation")
        tcols = ["dma_gap","macd_hist","rsi_strength","mom12m","dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z","TECH_score"]
        ttable = tech.loc[[ticker], tcols].T.rename(columns={ticker:"value"}).reset_index().rename(columns={"index":"metric"})
        st.dataframe(ttable, use_container_width=True)
        st.markdown(technicals_story(tech.loc[ticker]))

        # Macro
        st.markdown("#### Macro (VIX) — level & trend")
        st.markdown(macro_story(k_macro))

        # Download breakdown
        out = pd.concat([fund.loc[[ticker]], tech.loc[[ticker]]], axis=1).T
        csv = out.to_csv().encode()
        st.download_button("⬇️ Download breakdown (CSV)", csv, file_name=f"{ticker}_breakdown.csv")

    # Charts
    st.markdown("### Charts")
    if ticker in px.columns and not px[ticker].dropna().empty:
        s = px[ticker].dropna()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(s.index, s, label=ticker)
        ax.plot(s.index, s.ewm(span=50, adjust=False).mean(), label="EMA50", alpha=0.7)
        ax.set_title(f"{ticker} — Price & EMA50")
        ax.legend()
        st.pyplot(fig)
        st.markdown("<div class='caption'>EMA50 gives a quick read on trend; sustained closes above it tend to support momentum.</div>", unsafe_allow_html=True)
    else:
        st.info("Not enough price data to draw charts.")

# =========================
# ----- Portfolio page -----
# =========================

CURRENCIES = ["$", "€", "£", "CHF", "CAD"]

def page_portfolio():
    render_header_centered("Rate My Portfolio", "Enter tickers and weights (or amounts). Submit to sync and rate.")

    # Currency & total
    c1,c2 = st.columns([1,3])
    with c1:
        cur = st.selectbox("Currency", CURRENCIES, index=0)
    with c2:
        total = st.number_input(f"Total portfolio value ({cur})", min_value=0.0, value=10000.0, step=100.0)

    st.markdown("### Holdings (submit to sync)")
    st.write("Enter **Ticker** and either **Percent (%)** or **Amount**. Use the toggle to control sync direction on submit.")

    mode = st.radio("Sync mode (applies on Submit)", ["Auto", "Percent → Amount", "Amount → Percent"], horizontal=True, index=0)

    # Editable grid (simple)
    df = st.session_state.get("holdings_df")
    if df is None:
        df = pd.DataFrame({"Ticker":["AAPL","MSFT","NVDA","AMZN"],
                           "Percent (%)":[30.0,20.0,20.0,30.0],
                           "Amount":[total*0.30, total*0.20, total*0.20, total*0.30]})
        st.session_state["holdings_df"] = df.copy()

    edited = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        key="holdings_editor",
        column_config={
            "Ticker": st.column_config.TextColumn(required=True, help="Yahoo symbol (BRK.B → BRK-B)"),
            "Percent (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
            "Amount": st.column_config.NumberColumn(min_value=0.0, step=50.0, format="%.2f"),
        },
    )

    # Submit button to apply sync
    if st.button("Submit holdings", type="primary"):
        df = edited.copy()
        # Clean tickers
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
        # Sync
        if mode == "Percent → Amount" or (mode=="Auto" and df["Percent (%)"].sum()>0):
            df["Amount"] = total * (df["Percent (%)"]/100.0)
        else:
            amt_sum = df["Amount"].sum()
            if amt_sum > 0:
                df["Percent (%)"] = 100.0 * df["Amount"]/amt_sum
            else:
                df["Percent (%)"] = 0.0
        # Constrain
        df["Percent (%)"] = df["Percent (%)"].clip(lower=0, upper=100)
        if df["Percent (%)"].sum() > 0:
            df["Percent (%)"] = df["Percent (%)"] * (100.0/df["Percent (%)"].sum())
            df["Amount"] = total * (df["Percent (%)"]/100.0)
        st.session_state["holdings_df"] = df.copy()
        st.success("Holdings updated.")

    df = st.session_state["holdings_df"].copy()
    if df["Percent (%)"].sum() < 99.5 or df["Percent (%)"].sum() > 100.5:
        st.warning("Portfolio weights do not sum to ~100%. Use Submit to normalize.")

    # Build peer set = all tickers in holdings + built-in choice (optional)
    st.markdown("### Peer universe for portfolio names")
    peer_choice = st.selectbox("Add a peer set for comparison (optional)", ["None"] + list(PEER_UNIVERSE_SOURCES.keys()), index=0)
    peers = []
    if peer_choice != "None":
        peers = PEER_UNIVERSE_SOURCES[peer_choice]() or []
    # union
    port_syms = df["Ticker"].dropna().unique().tolist()
    peers = sorted(list(set(peers) | set(port_syms)))

    st.markdown(
        f"<div class='banner'>Peers loaded: {max(0,len(peers)-len(port_syms))} | Source: {peer_choice or 'None'}</div>",
        unsafe_allow_html=True,
    )

    if st.button("Rate portfolio", type="primary"):
        with st.spinner("Fetching data…"):
            fund = fundamentals_panel(peers)
            tech, _ = technicals_panel(peers, port_syms[0] if port_syms else "")
            k = macro_kpi()
        # per name score
        rows=[]
        for t,w in zip(df["Ticker"], df["Percent (%)"]/100.0):
            f = fund.loc[t]["FUND_score"] if t in fund.index else np.nan
            te= tech.loc[t]["TECH_score"] if t in tech.index else np.nan
            z, pct = composite_score(f, te, k["macro_score"])
            rows.append({"Ticker":t,"Weight":w,"Fundamentals":f,"Technicals":te,"Macro (VIX)":k["macro_score"],"Composite":z,"Score (0-100)":pct})
        res = pd.DataFrame(rows).set_index("Ticker")
        res["Weight × Comp"] = res["Weight"] * res["Composite"]
        st.markdown("### Per-name contribution")
        st.dataframe(res[["Weight","Fundamentals","Technicals","Macro (VIX)","Composite","Weight × Comp","Score (0-100)"]], use_container_width=True)

        port_score = res["Weight × Comp"].sum()
        port_pct   = _z_to_pct(np.clip(port_score, -2.5, 2.5))
        rec = "Strong Buy" if port_pct>=85 else "Buy" if port_pct>=60 else "Hold" if port_pct>=40 else "Sell" if port_pct>=20 else "Strong Sell"

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown("**Portfolio composite**"); st.markdown(f"<div class='kpi kpi-big'>{port_score:.3f}</div>", unsafe_allow_html=True)
        with c2: st.markdown("**Score (0–100)**");      st.markdown(f"<div class='kpi kpi-big'>{port_pct:.1f}</div>", unsafe_allow_html=True)
        with c3: st.markdown("**Recommendation**");     st.markdown(f"<div class='kpi kpi-big'>{rec}</div>", unsafe_allow_html=True)

        # Simple diversification notes
        st.markdown("### Diversification — quick checks")
        weights = df["Percent (%)"]/100.0
        hhi = (weights**2).sum()
        effN = 1.0/hhi if hhi>0 else np.nan
        st.markdown(f"- **HHI** ≈ {hhi:.3f}; **Effective names** ≈ {effN:.2f}. Lower HHI / higher Effective N indicate better diversification.")
        st.markdown("- Correlations and sector mix can be layered in next (optional extension).")

# =========================
# ------- Router -----------
# =========================

def landing():
    render_header_centered("Rate My", "Pick a stock or your portfolio — we’ll rate it with clear, friendly explanations and charts.")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
            if st.button("📈 Rate My Stock", use_container_width=True, type="primary"):
                st.session_state["page"]="stock"; st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with cc2:
            st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
            if st.button("💼 Rate My Portfolio", use_container_width=True):
                st.session_state["page"]="portfolio"; st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def router():
    page = st.session_state.get("page","home")
    if page=="home":
        landing()
    elif page=="stock":
        if st.button("← Back", key="back1"): st.session_state["page"]="home"; st.experimental_rerun()
        page_stock()
    elif page=="portfolio":
        if st.button("← Back", key="back2"): st.session_state["page"]="home"; st.experimental_rerun()
        page_portfolio()
    else:
        landing()

if __name__ == "__main__":
    router()