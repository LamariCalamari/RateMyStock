# app_utils.py — shared utilities for Rate My (Stock + Portfolio + Tracker)
# First run: fetches SP500, NASDAQ100, DOW30 from Wikipedia, normalizes,
# then writes peer_lists_snapshot.json so future runs use a STATIC snapshot.
# If you prefer fully hard-coded arrays later, just open that JSON and paste.

import io
import os
import json
import time
import random
from typing import Iterable, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==========================  UI: CSS + Brand  ==========================

def inject_css() -> None:
    """Global CSS + script to rename the first sidebar nav item to 'Home'."""
    st.markdown(
        """
        <style>
        .block-container{max-width:1140px;}

        /* KPI + misc */
        .kpi-card{padding:1rem 1.1rem;border-radius:12px;background:#111418;border:1px solid #222}
        .kpi-num{font-size:2.2rem;font-weight:800;margin-top:.25rem}
        .small-muted{color:#9aa0a6;font-size:.9rem}
        .banner{background:#0c2f22;color:#cdebdc;border-radius:10px;padding:.9rem 1.1rem;margin:.75rem 0 1.25rem}
        .chart-caption{color:#9aa0a6;margin:-.5rem 0 1rem}
        .topbar{display:flex;justify-content:flex-end;margin:.2rem 0 .6rem}

        /* Brand header: logo + gradient title centered */
        .brand{
          display:flex;align-items:center;justify-content:center;gap:16px;margin:1.0rem 0 .5rem;
        }
        .brand h1{
          font-size:56px;margin:0;line-height:1;font-weight:900;letter-spacing:.3px;
          background:linear-gradient(90deg,#e74c3c 0%,#f39c12 50%,#2ecc71 100%);
          -webkit-background-clip:text;background-clip:text;color:transparent;
        }
        .logo{width:56px;height:52px;flex:0 0 auto;}

        /* Home CTA boxes */
        .cta{ padding:.25rem; filter:drop-shadow(0 10px 18px rgba(0,0,0,.35)); }
        .cta .stButton>button{
          width:100%; padding:18px 22px; border-radius:14px; font-weight:800; font-size:1.05rem;
          border:1px solid rgba(255,255,255,.14);
          background:linear-gradient(90deg,#e85d58,#f39c12,#2ecc71);
          color:#0e1015; box-shadow:0 1px 0 rgba(255,255,255,.06) inset;
          transition:transform .08s ease, box-shadow .16s ease, filter .12s ease;
        }
        .cta.dark .stButton>button{
          background:#171a1f; color:#e6e8eb; border-color:#2e3339;
        }
        .cta .stButton>button:hover{ transform:translateY(-1px); filter:saturate(1.06) brightness(1.05); }
        .cta.dark .stButton>button:hover{ border-color:#3a3f46; }
        .hr-lite{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);border:0;margin:18px 0;}
        </style>

        <script>
        // Rename the first sidebar nav item ("app") to "Home"
        (function(){
          function renameFirst(){
            try{
              const nav = document.querySelector('[data-testid="stSidebarNav"]');
              if(!nav) return;
              const first = nav.querySelector('ul li:first-child a p');
              if(first && first.textContent.trim().toLowerCase()==='app'){ first.textContent = 'Home'; }
            }catch(e){}
          }
          const obs = new MutationObserver(renameFirst);
          obs.observe(document.body,{childList:true,subtree:true});
          renameFirst();
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

def inline_logo_svg() -> str:
    return """
<svg class="logo" viewBox="0 0 100 90" xmlns="http://www.w3.org/2000/svg" aria-label="Rate My">
  <defs>
    <linearGradient id="g" x1="0" x2="0" y1="1" y2="0">
      <stop offset="0%"  stop-color="#e74c3c"/>
      <stop offset="50%" stop-color="#f39c12"/>
      <stop offset="100%" stop-color="#2ecc71"/>
    </linearGradient>
  </defs>
  <polygon points="50,5 95,85 5,85" fill="url(#g)"/>
</svg>
"""

def brand_header(title: str) -> None:
    st.markdown(f'<div class="brand">{inline_logo_svg()}<h1>{title}</h1></div>', unsafe_allow_html=True)

def topbar_back(label: str = "← Back", url: str | None = None) -> None:
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if url: st.page_link(url, label=label)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================  Core helpers  ==========================

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

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
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

# ==========================  Peer lists (snapshot on first run) ==========================

_SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "peer_lists_snapshot.json")

def _normalize_list(tickers: Iterable[str]) -> List[str]:
    cleaned = []
    for s in tickers:
        if not isinstance(s, str): continue
        s = s.strip().upper().replace(".", "-")
        if s: cleaned.append(s)
    # dedupe keep order
    seen = set(); out=[]
    for t in cleaned:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _fetch_sp500_from_web() -> List[str]:
    # Wikipedia: List of S&P 500 companies
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, flavor="lxml")
    df = tables[0]
    return _normalize_list(df["Symbol"].tolist())

def _fetch_nasdaq100_from_web() -> List[str]:
    # Wikipedia: Nasdaq-100 (the constituents table)
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url, flavor="lxml")
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "symbol" in c for c in cols):
            df = t; break
    if df is None:
        raise RuntimeError("Could not find Nasdaq-100 table")
    for c in ["Ticker","Symbol","Ticker symbol","Ticker Symbol"]:
        if c in df.columns:
            return _normalize_list(df[c].tolist())
    return _normalize_list(df.iloc[:,0].tolist())

def _fetch_dow30_from_web() -> List[str]:
    # Wikipedia: Dow Jones Industrial Average components
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = pd.read_html(url, flavor="lxml")
    df = None
    for t in tables:
        if any("Symbol" in str(c) for c in t.columns):
            df = t; break
    if df is None:
        raise RuntimeError("Could not find Dow 30 table")
    col = None
    for c in df.columns:
        if "Symbol" in str(c):
            col = c; break
    if col is None:
        return _normalize_list(df.iloc[:,0].tolist())
    return _normalize_list(df[col].tolist())

@st.cache_resource(show_spinner=False)
def _load_or_snapshot_peer_lists() -> Dict[str, List[str]]:
    """Load static lists from JSON or fetch once and persist to JSON."""
    if os.path.exists(_SNAPSHOT_FILE):
        try:
            with open(_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # normalize again to be safe
            for k in obj:
                obj[k] = _normalize_list(obj[k])
            return obj
        except Exception:
            pass  # fall through to rebuild

    # First run: fetch from web, normalize, persist
    try:
        sp500 = _fetch_sp500_from_web()
        ndx   = _fetch_nasdaq100_from_web()
        dow   = _fetch_dow30_from_web()
        snap = {"S&P 500": sp500, "NASDAQ 100": ndx, "Dow 30": dow}
        try:
            with open(_SNAPSHOT_FILE, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
        except Exception:
            # if writing fails (read-only), still return in-memory lists
            pass
        return snap
    except Exception:
        # Emergency minimal fallbacks so the app still runs
        return {
            "S&P 500": ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","BRK-B","LLY","JPM","V","AVGO","TSLA"],
            "NASDAQ 100": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","NFLX","AMD"],
            "Dow 30": ["AAPL","MSFT","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM","MRK","MS","NKE","PG","TRV","UNH","V","VZ","WMT","DOW"],
        }

# Exposed catalog used by build_universe()
PEER_CATALOG: Dict[str, List[str]] = _load_or_snapshot_peer_lists()

def set_peer_catalog(sp500: List[str] | None = None,
                     ndx: List[str] | None = None,
                     dow: List[str] | None = None) -> None:
    """Optional hook to override lists at runtime (e.g., from a local CSV)."""
    if sp500 is not None: PEER_CATALOG["S&P 500"] = _normalize_list(sp500)
    if ndx is not None:   PEER_CATALOG["NASDAQ 100"] = _normalize_list(ndx)
    if dow is not None:   PEER_CATALOG["Dow 30"] = _normalize_list(dow)

# ==========================  Data fetchers  ==========================

@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(
    tickers: Iterable[str],
    period: str = "1y",
    interval: str = "1d",
    chunk: int = 25,
    retries: int = 3,
    sleep_between: float = 0.75,
    singles_pause: float = 0.60,
    hard_limit: int = 700,   # allow very large universes
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Robust fetcher for MANY symbols.
      - Pass 1: singles (reliable)
      - Pass 2: bulk for stragglers
    Tuned delays to be gentler on yfinance for 150–300 names.
    """
    names = [yf_symbol(t) for t in tickers if t]
    names = list(dict.fromkeys(names))[:hard_limit]
    if not names:
        return pd.DataFrame(), []

    frames: List[pd.Series] = []
    ok: List[str] = []

    # Pass 1 — singles with retries
    missing = names[:]
    for _ in range(retries):
        new_missing = []
        for t in missing:
            try:
                df = yf.download(
                    t, period=period, interval=interval,
                    auto_adjust=True, group_by="ticker",
                    threads=False, progress=False
                )
                if isinstance(df, pd.DataFrame) and "Close" in df:
                    s = df["Close"].dropna()
                    if s.size > 0:
                        frames.append(s.rename(t))
                        ok.append(t)
                    else:
                        new_missing.append(t)
                else:
                    new_missing.append(t)
            except Exception:
                new_missing.append(t)
            time.sleep(singles_pause + random.uniform(0, 0.25))
        missing = new_missing
        if not missing:
            break

    # Pass 2 — bulk for the rest
    def _append_from_multi(df: pd.DataFrame, group: List[str]):
        if not isinstance(df.columns, pd.MultiIndex):
            t = group[0]
            if "Close" in df:
                s = df["Close"].dropna()
                if s.size > 0:
                    frames.append(s.rename(t)); ok.append(t)
            return
        got = set(df.columns.get_level_values(0))
        for t in group:
            if t in got:
                s = df[t]["Close"].dropna()
                if s.size > 0:
                    frames.append(s.rename(t)); ok.append(t)

    for i in range(0, len(missing), chunk):
        group = missing[i:i+chunk]
        try:
            df = yf.download(
                group, period=period, interval=interval,
                auto_adjust=True, group_by="ticker",
                threads=False, progress=False
            )
            _append_from_multi(df, group)
        except Exception:
            pass
        time.sleep(sleep_between + random.uniform(0, 0.20))

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    ok = list(dict.fromkeys(ok))
    return prices, ok

@st.cache_data(show_spinner=False)
def fetch_vix_series(period: str = "6mo", interval: str = "1d") -> pd.Series:
    try:
        df = yf.Ticker("^VIX").history(period=period, interval=interval)
        if not df.empty: return df["Close"].rename("^VIX")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_fundamentals_simple(tickers: Iterable[str]) -> pd.DataFrame:
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

# ==========================  Universe builder  ==========================

def _get_catalog_list(label: str) -> List[str]:
    """Get frozen lists (from snapshot or runtime override)."""
    lst = PEER_CATALOG.get(label, [])
    return lst if lst else []

def build_universe(user_tickers: List[str], mode: str, sample_n: int = 150, custom_raw: str = "") -> Tuple[List[str], str]:
    """
    Build a peer universe from the STATIC snapshot (or custom paste).
    - Always excludes user tickers from the peers.
    - If mode == "Custom (paste list)", comma-separated string is used.
    - Otherwise uses the full index list and samples down to 'sample_n' deterministically.
    """
    user = [yf_symbol(t) for t in user_tickers if t]
    user_set = set(user)

    if mode == "Custom (paste list)":
        custom = {yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
        peers_all = sorted(list(custom - user_set))
        label = "Custom"
    elif mode in ("S&P 500", "NASDAQ 100", "Dow 30"):
        peers_all = [t for t in _get_catalog_list(mode) if t not in user_set]
        label = mode
    else:
        # Auto selection: detect membership; else default to S&P 500
        chosen = "S&P 500"
        for label_try in ("S&P 500","Dow 30","NASDAQ 100"):
            lst = set(_get_catalog_list(label_try))
            if user_set & lst:
                chosen = label_try; break
        label = chosen
        peers_all = [t for t in _get_catalog_list(chosen) if t not in user_set]

    # Respect sample_n while using the FULL index as source
    if sample_n and len(peers_all) > sample_n:
        # stable downsampling by stride (preserves broad coverage deterministically)
        step = max(1, len(peers_all)//sample_n)
        peers = peers_all[::step][:sample_n]
    else:
        peers = peers_all

    universe = sorted(list(user_set | set(peers)))[:700]  # final hard cap
    return universe, label

# ==========================  Features / Scores  ==========================

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
    else:
        trend = 1.0-(rel_gap+0.03)/0.06
        trend = float(np.clip(trend,0,1))
    macro=float(np.clip(0.70*level+0.30*trend,0,1))
    return macro, vix_last, ema20, rel_gap