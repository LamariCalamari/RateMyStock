# app_utils.py
import time
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==================== Global CSS & Brand ====================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container{max-width:1140px;}

        /* Sidebar: replace the small 'app' pill text with 'Home' */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div:first-child{
          position:relative;
          color:transparent !important;         /* hide original text */
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] > div:first-child::after{
          content:"Home";
          position:absolute; inset:auto auto auto 14px;
          top:8px; color:#e7ebf0; font-weight:700; letter-spacing:.2px;
        }

        /* Brand header: logo next to title, centered */
        .brand{ display:flex; align-items:center; justify-content:center; gap:16px; margin:1.1rem 0 .35rem; }
        .brand .wordmark{
          font-size:56px; line-height:1; font-weight:900; margin:0;
          background:linear-gradient(90deg,#e74c3c 0%, #f39c12 50%, #2ecc71 100%);
          -webkit-background-clip:text; background-clip:text; color:transparent;
          letter-spacing:.3px;
        }
        .logo{ width:56px; height:52px; flex:0 0 auto; }

        /* KPI + misc (from your original) */
        .kpi-card{padding:1rem 1.1rem;border-radius:12px;background:#111418;border:1px solid #222}
        .kpi-num{font-size:2.2rem;font-weight:800;margin-top:.25rem}
        .small-muted{color:#9aa0a6;font-size:.9rem}
        .banner{background:#0c2f22;color:#cdebdc;border-radius:10px;padding:.9rem 1.1rem;margin:.75rem 0 1.25rem}
        .chart-caption{color:#9aa0a6;margin:-.5rem 0 1rem}
        .topbar{display:flex;justify-content:flex-end;margin:.2rem 0 .6rem}

        /* CTA Boxes */
        .cta-box{
          display:block; text-align:center; padding:16px 18px; border-radius:14px;
          background:#15181d; border:1px solid #2e3238;
          box-shadow:0 1px 0 rgba(255,255,255,.06) inset, 0 8px 24px rgba(0,0,0,.35);
        }
        .cta-box:hover{ background:#181c22; border-color:#3a4048; }
        .cta-box a[data-testid="stPageLink"]{
          display:inline-flex; align-items:center; gap:.6rem; font-weight:800; color:#e9edf0; text-decoration:none;
        }
        /* Make first CTA stand out with the brand gradient frame */
        .cta-box.primary{
          background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(0,0,0,.06)) padding-box,
                     linear-gradient(90deg,#e85d58, #f39c12, #2ecc71) border-box;
          border:1px solid transparent;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def inline_logo_svg() -> str:
    return """<svg class="logo" viewBox="0 0 100 90" xmlns="http://www.w3.org/2000/svg" aria-label="Rate My">
  <defs>
    <linearGradient id="g" x1="0" x2="0" y1="1" y2="0">
      <stop offset="0%"  stop-color="#e74c3c"/>
      <stop offset="50%" stop-color="#f39c12"/>
      <stop offset="100%" stop-color="#2ecc71"/>
    </linearGradient>
  </defs>
  <polygon points="50,5 95,85 5,85" fill="url(#g)"/>
</svg>"""


def brand_header(title: str):
    """Centered logo + gradient wordmark — no Markdown escaping issues."""
    # No leading spaces before the <div> to avoid Markdown treating it as code.
    html = f"""<div class="brand">{inline_logo_svg()}<div class="wordmark">{title}</div></div>"""
    st.markdown(html, unsafe_allow_html=True)


def topbar_back(label: str = "← Back", url: str | None = None):
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if url:
        st.page_link(url, label=label)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== Finance Helpers (original logic retained) ====================

def yf_symbol(t: str) -> str:
    if not isinstance(t, str):
        return t
    return t.strip().upper().replace(".", "-")


def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up, dn = d.clip(lower=0.0), -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0

# ==================== Fetchers (chunked + retries) ====================

@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
    chunk: int = 25,
    retries: int = 3,
    sleep_between: float = 0.35,
    singles_pause: float = 0.20,
    hard_limit: int = 350,
) -> Tuple[pd.DataFrame, List[str]]:
    tickers = [yf_symbol(t) for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))[:hard_limit]
    if not tickers:
        return pd.DataFrame(), []

    frames: List[pd.Series] = []
    ok: List[str] = []

    def _append_from_multi(df, names):
        got = set(df.columns.get_level_values(0))
        for t in names:
            if t in got:
                s = df[t]["Close"].dropna()
                if s.size:
                    frames.append(s.rename(t)); ok.append(t)

    # Bulk
    for i in range(0, len(tickers), chunk):
        group = tickers[i:i+chunk]
        try:
            df = yf.download(group, period=period, interval=interval,
                             auto_adjust=True, group_by="ticker",
                             threads=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                _append_from_multi(df, group)
            else:
                t = group[0]
                if "Close" in df:
                    s = df["Close"].dropna()
                    if s.size:
                        frames.append(s.rename(t)); ok.append(t)
        except Exception:
            pass
        time.sleep(sleep_between + random.uniform(0, 0.15))

    ok_set = set(ok)
    missing = [t for t in tickers if t not in ok_set]

    # Singles
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
def fetch_vix_series(period: str = "6mo", interval: str = "1d") -> pd.Series:
    try:
        df = yf.Ticker("^VIX").history(period=period, interval=interval)
        if not df.empty:
            return df["Close"].rename("^VIX")
    except Exception:
        pass
    return pd.Series(dtype=float)


@st.cache_data(show_spinner=False)
def fetch_fundamentals_simple(tickers: List[str]) -> pd.DataFrame:
    keep = ["revenueGrowth","earningsGrowth","returnOnEquity",
            "profitMargins","grossMargins","operatingMargins","ebitdaMargins",
            "trailingPE","forwardPE","debtToEquity"]
    rows=[]
    for raw in tickers:
        t=yf_symbol(raw)
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info={}
        row={"ticker":t}
        for k in keep:
            try: row[k]=float(info.get(k, np.nan))
            except Exception: row[k]=np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")

# ==================== Peer Universes & Feature Builders (unchanged) ====================

SP500_FALLBACK = ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","BRK-B","UNH","LLY","V","JPM"]
DOW30_FALLBACK = ["AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","DIS","HD","INTC","IBM","KO","MCD","NKE","TRV","VZ","CSCO"]
NASDAQ100_FALLBACK = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","NFLX","CSCO","AMD"]

def list_sp500() -> set:
    try:
        got = {yf_symbol(t) for t in yf.tickers_sp500()}
        if got: return got
    except Exception: pass
    return set(SP500_FALLBACK)

def list_dow30() -> set:
    try:
        got = {yf_symbol(t) for t in yf.tickers_dow()}
        if got: return got
    except Exception: pass
    return set(DOW30_FALLBACK)

def list_nasdaq100() -> set:
    try:
        if hasattr(yf,"tickers_nasdaq"):
            got = {yf_symbol(t) for t in yf.tickers_nasdaq()}
            if got: return got
    except Exception: pass
    return set(NASDAQ100_FALLBACK)

def build_universe(user_tickers: List[str], mode: str, sample_n: int = 150, custom_raw: str = "") -> Tuple[List[str], str]:
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

def technical_scores(price_panel: Dict[str, pd.Series]) -> pd.DataFrame:
    rows=[]
    for ticker, px in price_panel.items():
        px=px.dropna()
        if len(px)<60: continue
        ema50  = ema(px,50)
        base50 = ema50.iloc[-1] if pd.notna(ema50.iloc[-1]) and ema50.iloc[-1]!=0 else np.nan
        dma_gap=(px.iloc[-1]-ema50.iloc[-1])/base50 if pd.notna(base50) else np.nan
        _,_,hist = macd(px)
        macd_hist = hist.iloc[-1] if len(hist)>0 else np.nan
        r = rsi(px).iloc[-1] if len(px)>14 else np.nan
        rsi_strength = (r-50.0)/50.0 if pd.notna(r) else np.nan
        mom = np.nan
        if len(px) > 252:
            try: mom = px.iloc[-1]/px.iloc[-253]-1.0
            except Exception: mom = np.nan
        rows.append({"ticker":ticker,"dma_gap":dma_gap,"macd_hist":macd_hist,
                     "rsi_strength":rsi_strength,"mom12m":mom})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

def macro_from_vix(vix_series: pd.Series) -> Tuple[float,float,float,float]:
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
    else:
        trend = 1.0-(rel_gap+0.03)/0.06
        trend = float(np.clip(trend,0,1))
    macro=float(np.clip(0.70*level+0.30*trend,0,1))
    return macro, vix_last, ema20, rel_gap

def fundamentals_interpretation(zrow: pd.Series) -> List[str]:
    lines=[]
    def bucket(v, pos_good=True):
        if pd.isna(v): return "neutral"
        if pos_good:
            return "bullish" if v>=0.5 else "watch" if v<=-0.5 else "neutral"
        else:
            return "bullish (cheap)" if v>=0.5 else "watch (expensive)" if v<=-0.5 else "neutral"

    g  = bucket(zrow.get("revenueGrowth_z"))
    e  = bucket(zrow.get("earningsGrowth_z"))
    pm = bucket(zrow.get("profitMargins_z"))
    gm = bucket(zrow.get("grossMargins_z"))
    om = bucket(zrow.get("operatingMargins_z"))
    roe= bucket(zrow.get("returnOnEquity_z"))
    val= bucket(zrow.get("forwardPE_z"), pos_good=False)
    lev= bucket(zrow.get("debtToEquity_z"), pos_good=False)

    if g=="bullish" or e=="bullish": lines.append("**Growth tilt:** above-peer revenue/earnings growth (supportive).")
    elif g=="watch" or e=="watch":   lines.append("**Growth tilt:** below peers — watch for stabilization or re-acceleration.")
    else:                            lines.append("**Growth tilt:** broadly in line with peers.")

    if (pm=="bullish" or gm=="bullish" or om=="bullish" or roe=="bullish"):
        lines.append("**Profitability & margins:** strong vs peers (healthy quality).")
    elif (pm=="watch" or gm=="watch" or om=="watch" or roe=="watch"):
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
