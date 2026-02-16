from __future__ import annotations

import os, re, json, time, random, math
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =============================== UI: CSS + Brand ===============================

def inject_css() -> None:
    """Global CSS and a tiny script to rename the first sidebar item to 'Home'."""
    st.markdown(
        """
        <style>
        .block-container{max-width:1140px;}
        .kpi-card{padding:1rem 1.1rem;border-radius:12px;background:#111418;border:1px solid #222}
        .kpi-num{font-size:2.2rem;font-weight:800;margin-top:.25rem}
        .small-muted{color:#9aa0a6;font-size:.9rem}
        .banner{background:#0c2f22;color:#cdebdc;border-radius:10px;padding:.9rem 1.1rem;margin:.75rem 0 1.25rem}
        .topbar{display:flex;justify-content:flex-end;margin:.2rem 0 .6rem}

        /* Brand header */
        .brand{display:flex;align-items:center;justify-content:center;gap:16px;margin:1.0rem 0 .5rem;}
        .brand h1{
          font-size:56px;margin:0;line-height:1;font-weight:900;letter-spacing:.3px;
          background:linear-gradient(90deg,#e74c3c 0%,#f39c12 50%,#2ecc71 100%);
          -webkit-background-clip:text;background-clip:text;color:transparent;
        }
        .logo{width:56px;height:52px;flex:0 0 auto;}
        .pill{display:inline-block;padding:.15rem .5rem;border-radius:999px;border:1px solid #2c3239;background:#151920;color:#cfd4da;font-size:.85rem}
        </style>

        <script>
        // Rename "app" -> "Home" in sidebar nav
        (function(){
          function renameNav(){
            try{
              const nav=document.querySelector('[data-testid="stSidebarNav"]');
              if(!nav) return;
              const nodes = nav.querySelectorAll('a p, a span');
              nodes.forEach(n=>{
                if(n.textContent && n.textContent.trim().toLowerCase()==='app'){
                  n.textContent='Home';
                }
              });
            }catch(e){}
          }
          const obs=new MutationObserver(renameNav);
          obs.observe(document.body,{childList:true,subtree:true});
          renameNav();
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

def inline_logo_svg() -> str:
    return """
<svg class="logo" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" aria-label="Rate My">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="1" y2="0">
      <stop offset="0%"  stop-color="#e74c3c"/>
      <stop offset="50%" stop-color="#f39c12"/>
      <stop offset="100%" stop-color="#2ecc71"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="50%" r="60%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="60" cy="60" r="50" fill="none" stroke="url(#g)" stroke-width="7" stroke-linecap="round"/>
  <circle cx="60" cy="60" r="46" fill="url(#glow)"/>
  <rect x="34" y="66" width="11" height="20" rx="5" fill="#eef1f5"/>
  <rect x="52" y="52" width="11" height="34" rx="5" fill="#eef1f5"/>
  <rect x="70" y="38" width="11" height="48" rx="5" fill="#eef1f5"/>
  <path d="M32 70 Q44 61 56 50 T86 34" fill="none" stroke="url(#g)" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

def brand_header(title: str) -> None:
    st.markdown(f'<div class="brand">{inline_logo_svg()}<h1>{title}</h1></div>', unsafe_allow_html=True)

def topbar_back(label: str = "← Back", url: Optional[str] = None) -> None:
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if url: st.page_link(url, label=label)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================== Core helpers =================================

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


# ============================== Scoring defaults ==============================

SCORING_CONFIG = {
    "weights": {"fund": 0.50, "tech": 0.45, "macro": 0.05},
    "z_cap": 2.5,
    "min_fund_cols": 6,
    "factor_min_coverage": 0.25,
    "cross_section_shrink_target": 30,
    "peer_min_industry": 25,
    "peer_min_sector": 40,
    "confidence_weights": {"peer": 0.4, "fund": 0.3, "tech": 0.3},
    "rating_bins": [
        (80, "Strong Buy"),
        (60, "Buy"),
        (40, "Hold"),
        (20, "Sell"),
        (0, "Strong Sell"),
    ],
}

def score_label(score: float) -> str:
    if score >= 80: return "Strong Buy"
    if score >= 60: return "Buy"
    if score >= 40: return "Hold"
    if score >= 20: return "Sell"
    return "Strong Sell"


# ========================== Peer universes (indices.py) ========================

try:
    from indices import SP500_LIST as _SP500, NASDAQ100_LIST as _NDX, DOW30_LIST as _DOW
except Exception:
    _SP500, _NDX, _DOW = [], [], []

_DELISTED = {
    "FBHS", "NLSN", "PARA",
}

def _normalize_list(lst: Iterable[str]) -> List[str]:
    out, seen = [], set()
    for s in lst:
        if not isinstance(s, str): continue
        s = s.strip().upper().replace(".", "-")
        if s in _DELISTED: continue
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

_SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "peer_lists_snapshot.json")

@st.cache_resource(show_spinner=False)
def _load_or_snapshot_peer_lists() -> Dict[str, List[str]]:
    if _SP500 or _NDX or _DOW:
        return {"S&P 500": _normalize_list(_SP500),
                "NASDAQ 100": _normalize_list(_NDX),
                "Dow 30": _normalize_list(_DOW)}
    if os.path.exists(_SNAPSHOT_FILE):
        try:
            with open(_SNAPSHOT_FILE, "r", encoding="utf-8") as handle:
                obj = json.load(handle)
            return {k:_normalize_list(v) for k,v in obj.items()}
        except Exception:
            pass
    return {  # tiny fallback so app runs
        "S&P 500": ["AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","BRK-B","JPM","LLY","V"],
        "NASDAQ 100": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST"],
        "Dow 30": ["AAPL","MSFT","AMZN","AXP","BA","CAT","CSCO","CVX","HD","JNJ","JPM","KO","MCD","MRK","NKE","PG","TRV","UNH","V","WMT"],
    }

PEER_CATALOG = _load_or_snapshot_peer_lists()

def set_peer_catalog(sp500=None, ndx=None, dow=None) -> None:
    if sp500 is not None: PEER_CATALOG["S&P 500"]  = _normalize_list(sp500)
    if ndx   is not None: PEER_CATALOG["NASDAQ 100"]= _normalize_list(ndx)
    if dow   is not None: PEER_CATALOG["Dow 30"]   = _normalize_list(dow)


# ============================ Data fetchers (robust) ===========================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices_chunked_with_fallback(
    tickers: Iterable[str],
    period: str = "1y",
    interval: str = "1d",
    chunk: int = 20,
    retries: int = 4,
    sleep_between: float = 1.0,
    singles_pause: float = 1.1,
    hard_limit: int = 700,
) -> Tuple[pd.DataFrame, List[str]]:
    """3-pass yfinance loader to maximize peer coverage."""
    names = [yf_symbol(t) for t in tickers if t]
    names = list(dict.fromkeys(names))[:hard_limit]
    if not names: return pd.DataFrame(), []

    frames, ok = [], []

    def _sanitize_index(s: pd.Series) -> pd.Series:
        if not isinstance(s, pd.Series) or s.empty:
            return s
        out = s.copy()
        idx = out.index
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is not None:
                out.index = idx.tz_convert(None)
        else:
            try:
                out.index = pd.to_datetime(idx, errors="coerce")
                out = out[~out.index.isna()]
            except Exception:
                return out
        return out.sort_index()

    # Pass 1 — singles
    missing = names[:]
    for _ in range(retries):
        new_missing=[]
        for t in missing:
            try:
                df = yf.download(t, period=period, interval=interval,
                                 auto_adjust=True, group_by="ticker",
                                 threads=False, progress=False)
                if isinstance(df, pd.DataFrame) and "Close" in df:
                    s=df["Close"].dropna()
                    if s.size:
                        frames.append(_sanitize_index(s).rename(t)); ok.append(t)
                    else: new_missing.append(t)
                else: new_missing.append(t)
            except Exception:
                new_missing.append(t)
            time.sleep(singles_pause + random.uniform(0,0.25))
        missing = new_missing
        if not missing: break

    # Pass 2 — bulk
    def _append_from_multi(df: pd.DataFrame, group: List[str]):
        if not isinstance(df.columns, pd.MultiIndex):
            t0=group[0]
            if "Close" in df:
                s=df["Close"].dropna()
                if s.size: frames.append(_sanitize_index(s).rename(t0)); ok.append(t0)
            return
        got=set(df.columns.get_level_values(0))
        for t in group:
            if t in got:
                s=df[t]["Close"].dropna()
                if s.size: frames.append(_sanitize_index(s).rename(t)); ok.append(t)

    if missing:
        for i in range(0, len(missing), chunk):
            group = missing[i:i+chunk]
            try:
                df = yf.download(group, period=period, interval=interval,
                                 auto_adjust=True, group_by="ticker",
                                 threads=False, progress=False)
                _append_from_multi(df, group)
            except Exception:
                pass
            time.sleep(sleep_between + random.uniform(0,0.25))
        ok_set=set(ok)
        missing=[t for t in names if t not in ok_set]

    # Pass 3 — history()
    if missing:
        for t in missing:
            try:
                h=yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
                if isinstance(h,pd.DataFrame) and "Close" in h and not h["Close"].dropna().empty:
                    frames.append(_sanitize_index(h["Close"].dropna()).rename(t)); ok.append(t)
            except Exception:
                pass
            time.sleep(singles_pause + random.uniform(0.1,0.35))

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not prices.empty: prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    ok = list(dict.fromkeys(ok))
    return prices, ok

@st.cache_data(show_spinner=False, ttl=900)
def fetch_vix_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("^VIX").history(period=period, interval=interval)
        if not df.empty: return df["Close"].rename("^VIX")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_gold_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("GLD").history(period=period, interval=interval)
        if not df.empty: return df["Close"].rename("GLD")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_dxy_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("UUP").history(period=period, interval=interval)
        if not df.empty: return df["Close"].rename("UUP")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_tnx_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("^TNX").history(period=period, interval=interval)
        if not df.empty: return df["Close"].rename("^TNX")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False, ttl=900)
def fetch_credit_ratio_series(period="6mo", interval="1d") -> pd.Series:
    try:
        hyg = yf.Ticker("HYG").history(period=period, interval=interval)["Close"].rename("HYG")
        lqd = yf.Ticker("LQD").history(period=period, interval=interval)["Close"].rename("LQD")
        if not hyg.empty and not lqd.empty:
            ratio = (hyg / lqd).rename("HYG/LQD")
            return ratio.dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fundamentals_simple(tickers: Iterable[str]) -> pd.DataFrame:
    keep = ["revenueGrowth","earningsGrowth","returnOnEquity","returnOnAssets",
            "profitMargins","grossMargins","operatingMargins","ebitdaMargins",
            "trailingPE","forwardPE","enterpriseToEbitda","debtToEquity",
            "freeCashflow","marketCap"]
    rows=[]
    for raw in tickers:
        t=yf_symbol(raw)
        try:
            info=yf.Ticker(t).info or {}
        except Exception:
            info={}
        row={"ticker":t}
        for k in keep:
            try: row[k]=float(info.get(k, np.nan))
            except Exception: row[k]=np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ticker")
    if "freeCashflow" in df.columns and "marketCap" in df.columns:
        df["fcfYield"] = df["freeCashflow"] / df["marketCap"]
    return df

@st.cache_data(show_spinner=False, ttl=21600)
def fetch_profile(ticker: str) -> Dict[str, Optional[str]]:
    t = yf_symbol(ticker)
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}
    return {"sector": info.get("sector"), "industry": info.get("industry")}


# ================================ Universe ====================================

def build_universe(user_tickers: List[str], mode: str,
                   sample_n: int = 150, custom_raw: str = "",
                   min_industry: int = 25, min_sector: int = 40) -> Tuple[List[str], str]:
    user = [yf_symbol(t) for t in user_tickers if t]
    user_set=set(user)

    if mode=="Custom (paste list)":
        custom={yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
        peers_all=sorted(list(custom-user_set)); label="Custom"
    elif mode in ("S&P 500","NASDAQ 100","Dow 30"):
        peers_all=[t for t in PEER_CATALOG.get(mode,[]) if t not in user_set]; label=mode
    else:
        chosen="S&P 500"
        for lbl in ("S&P 500","Dow 30","NASDAQ 100"):
            if user_set & set(PEER_CATALOG.get(lbl,[])): chosen=lbl; break
        peers_all=[t for t in PEER_CATALOG.get(chosen,[]) if t not in user_set]; label=chosen

    auto_modes = ("Industry (auto fallback)", "Auto (industry → sector → index)", "Auto by index membership")
    if mode in auto_modes and user:
        target = user[0]
        target_profile = fetch_profile(target)
        target_industry = target_profile.get("industry")
        target_sector = target_profile.get("sector")
        profiles = {t: fetch_profile(t) for t in peers_all}

        industry_peers = [t for t in peers_all if target_industry and profiles[t].get("industry") == target_industry]
        sector_peers = [t for t in peers_all if target_sector and profiles[t].get("sector") == target_sector]

        if target_industry and len(industry_peers) >= min_industry:
            peers_all = industry_peers
            label = f"Industry: {target_industry} ({len(industry_peers)})"
        elif target_sector and len(sector_peers) >= min_sector:
            peers_all = sector_peers
            label = f"Sector: {target_sector} ({len(sector_peers)})"
        else:
            label = f"{label} (fallback)"

    if sample_n and len(peers_all)>sample_n:
        step = max(1, len(peers_all)//sample_n)
        peers=peers_all[::step][:sample_n]
    else:
        peers=peers_all

    universe=sorted(list(user_set | set(peers)))[:700]
    return universe, label


# ============================== Features / Scores ==============================

FUNDAMENTAL_POSITIVE_FACTORS = [
    "revenueGrowth", "earningsGrowth", "returnOnEquity", "returnOnAssets",
    "profitMargins", "grossMargins", "operatingMargins", "ebitdaMargins", "fcfYield",
]
FUNDAMENTAL_INVERTED_FACTORS = [
    "trailingPE", "forwardPE", "enterpriseToEbitda", "debtToEquity",
]
TECHNICAL_POSITIVE_FACTORS = [
    "dma_gap", "macd_hist", "rsi_strength", "mom12m", "mom6m", "mom3m", "trend_ema20_50", "drawdown6m",
]
TECHNICAL_INVERTED_FACTORS = ["vol63"]
FACTOR_LABELS = {
    "revenueGrowth_z": "Revenue growth",
    "earningsGrowth_z": "Earnings growth",
    "returnOnEquity_z": "ROE",
    "returnOnAssets_z": "ROA",
    "profitMargins_z": "Profit margin",
    "grossMargins_z": "Gross margin",
    "operatingMargins_z": "Operating margin",
    "ebitdaMargins_z": "EBITDA margin",
    "trailingPE_z": "P/E (lower is better)",
    "forwardPE_z": "Forward P/E (lower is better)",
    "enterpriseToEbitda_z": "EV/EBITDA (lower is better)",
    "fcfYield_z": "FCF yield",
    "debtToEquity_z": "Debt/Equity (lower is better)",
    "dma_gap_z": "Price vs EMA50",
    "macd_hist_z": "MACD momentum",
    "rsi_strength_z": "RSI strength",
    "mom12m_z": "12m momentum",
    "mom6m_z": "6m momentum",
    "mom3m_z": "3m momentum",
    "trend_ema20_50_z": "EMA20 vs EMA50",
    "vol63_z": "Volatility (lower is better)",
    "drawdown6m_z": "6m drawdown resilience",
}


def robust_zscore_series(s: pd.Series, cap: float | None = None) -> pd.Series:
    """
    Robust z-score using median/MAD with std-dev fallback.
    Falls back to classical z-score when MAD is degenerate.
    """
    x = pd.to_numeric(s, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if pd.isna(mad) or mad <= 1e-12:
        z = zscore_series(x)
    else:
        z = 0.67448975 * (x - med) / mad
    cap = SCORING_CONFIG["z_cap"] if cap is None else cap
    return z.clip(-cap, cap)


def _sample_shrinkage(count_non_null: int, target: Optional[int] = None) -> float:
    """
    Damp cross-sectional z-scores when sample size is very small.
    """
    target = SCORING_CONFIG.get("cross_section_shrink_target", 30) if target is None else target
    if count_non_null <= 1:
        return 0.0
    return float(np.clip(math.sqrt(count_non_null / max(target, 1)), 0.0, 1.0))


def _component_score(zdf: pd.DataFrame, min_coverage: Optional[float] = None) -> pd.Series:
    """
    Coverage-aware component score:
    - average of available z-factors
    - scaled by sqrt(coverage) so sparse rows are less extreme
    """
    min_coverage = (
        SCORING_CONFIG.get("factor_min_coverage", 0.25)
        if min_coverage is None else min_coverage
    )
    if zdf is None or zdf.empty:
        return pd.Series(dtype=float)
    coverage = zdf.notna().mean(axis=1)
    raw = zdf.mean(axis=1, skipna=True)
    score = raw * np.sqrt(np.clip(coverage, 0.0, 1.0))
    return score.where(coverage >= min_coverage, np.nan)


def _weighted_row_average(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Weighted average that renormalizes per row for missing components.
    """
    if df.empty:
        return pd.Series(dtype=float)
    w = pd.Series(weights, dtype=float)
    cols = [c for c in w.index if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    x = df[cols]
    valid = x.notna()
    num = x.mul(w[cols], axis=1).where(valid).sum(axis=1)
    den = valid.mul(w[cols], axis=1).sum(axis=1)
    return num / den.replace(0.0, np.nan)


def weighted_series_mean(values: pd.Series, weights: pd.Series) -> float:
    """
    Weighted mean with automatic renormalization over non-null values.
    """
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").reindex(v.index).fillna(0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    ww = w[mask]
    return float(np.average(v[mask], weights=ww))


def portfolio_signal_score(
    peer_composite: pd.Series,
    holding_composite: pd.Series,
    holding_weights: pd.Series,
    diversification: float | None = None,
    diversification_bonus_points: float = 5.0,
) -> Dict[str, float]:
    """
    Convert weighted holding composite into a peer-relative portfolio score.

    - signal: weighted mean composite across holdings (missing data renormalized)
    - signal_percentile: percentile of signal within peer composite distribution (0-100)
    - final_score: percentile plus optional diversification bonus (bounded 0-100)
    """
    peer = pd.to_numeric(peer_composite, errors="coerce").dropna()
    hold = pd.to_numeric(holding_composite, errors="coerce")
    w = pd.to_numeric(holding_weights, errors="coerce").reindex(hold.index).fillna(0.0)

    valid = hold.notna() & (w > 0)
    coverage_weight = float(w[valid].sum() / max(w.sum(), 1e-12)) if w.sum() > 0 else 0.0
    if not valid.any():
        return {
            "signal": np.nan,
            "signal_percentile": np.nan,
            "final_score": np.nan,
            "coverage_weight": coverage_weight,
            "diversification_bonus": 0.0,
        }

    signal = float(np.average(hold[valid], weights=w[valid]))

    if peer.empty:
        signal_percentile = np.nan
    else:
        signal_percentile = float(100.0 * (peer <= signal).mean())

    div_bonus = 0.0
    if diversification is not None and pd.notna(diversification):
        div = float(np.clip(diversification, 0.0, 1.0))
        div_bonus = (div - 0.5) * (2.0 * diversification_bonus_points)

    if pd.isna(signal_percentile):
        final_score = np.nan
    else:
        final_score = float(np.clip(signal_percentile + div_bonus, 0.0, 100.0))

    return {
        "signal": signal,
        "signal_percentile": signal_percentile,
        "final_score": final_score,
        "coverage_weight": coverage_weight,
        "diversification_bonus": div_bonus,
    }


def normalize_score_weights(
    fund_weight: Optional[float] = None,
    tech_weight: Optional[float] = None,
    macro_weight: Optional[float] = None,
) -> Dict[str, float]:
    wf = SCORING_CONFIG["weights"]["fund"] if fund_weight is None else float(max(0.0, fund_weight))
    wt = SCORING_CONFIG["weights"]["tech"] if tech_weight is None else float(max(0.0, tech_weight))
    wm = SCORING_CONFIG["weights"]["macro"] if macro_weight is None else float(max(0.0, macro_weight))
    total = wf + wt + wm
    if total <= 0:
        wf, wt, wm = 1.0, 0.0, 0.0
        total = 1.0
    return {"FUND_score": wf / total, "TECH_score": wt / total, "MACRO_score": wm / total}


def technical_scores(price_panel: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Build technical features.
    Backward-compatible fields are preserved:
      dma_gap, macd_hist, rsi_strength, mom12m
    and expanded with:
      mom6m, mom3m, trend_ema20_50, vol63, drawdown6m
    """
    rows = []
    for ticker, px_raw in price_panel.items():
        px = pd.to_numeric(px_raw, errors="coerce").dropna()
        if len(px) < 60:
            continue

        last = float(px.iloc[-1])
        ema20 = ema(px, 20)
        ema50 = ema(px, 50)
        e20 = float(ema20.iloc[-1]) if len(ema20) else np.nan
        e50 = float(ema50.iloc[-1]) if len(ema50) else np.nan

        dma_gap = (last - e50) / e50 if pd.notna(e50) and e50 != 0 else np.nan
        trend_ema20_50 = (e20 / e50 - 1.0) if (pd.notna(e20) and pd.notna(e50) and e50 != 0) else np.nan

        _, _, hist = macd(px)
        rets = np.log(px).diff().dropna()
        vol20 = float(rets.tail(20).std(ddof=0)) if rets.size >= 20 else np.nan
        hist_last = float(hist.iloc[-1]) if hist.size else np.nan
        if pd.notna(hist_last) and pd.notna(vol20) and vol20 > 1e-9 and last > 0:
            macd_hist = hist_last / (last * vol20)
        else:
            macd_hist = hist_last

        r = rsi(px).iloc[-1] if len(px) > 14 else np.nan
        rsi_strength = (r - 50.0) / 50.0 if pd.notna(r) else np.nan

        def _mom(days_back: int) -> float:
            if len(px) <= days_back:
                return np.nan
            base = float(px.iloc[-(days_back + 1)])
            return (last / base - 1.0) if base > 0 else np.nan

        mom12m = _mom(252)
        mom6m = _mom(126)
        mom3m = _mom(63)

        vol63 = float(rets.tail(63).std(ddof=1) * np.sqrt(252)) if rets.size >= 20 else np.nan
        px6 = px.tail(126)
        if px6.size >= 20:
            dd6 = px6 / px6.cummax() - 1.0
            drawdown6m = float(dd6.min())
        else:
            drawdown6m = np.nan

        rows.append({
            "ticker": ticker,
            "dma_gap": dma_gap,
            "macd_hist": macd_hist,
            "rsi_strength": rsi_strength,
            "mom12m": mom12m,
            "mom6m": mom6m,
            "mom3m": mom3m,
            "trend_ema20_50": trend_ema20_50,
            "vol63": vol63,
            "drawdown6m": drawdown6m,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def build_technical_zscores(
    tech_raw: pd.DataFrame,
    robust: bool = True,
    cap: float | None = None,
) -> pd.DataFrame:
    if tech_raw is None or tech_raw.empty:
        return pd.DataFrame()
    cap = SCORING_CONFIG["z_cap"] if cap is None else cap
    out = tech_raw.copy()

    for col in TECHNICAL_POSITIVE_FACTORS:
        if col in out.columns:
            vals = out[col]
            z = robust_zscore_series(vals, cap=cap) if robust else zscore_series(vals).clip(-cap, cap)
            z *= _sample_shrinkage(int(vals.notna().sum()))
            out[f"{col}_z"] = z

    for col in TECHNICAL_INVERTED_FACTORS:
        if col in out.columns:
            vals = -pd.to_numeric(out[col], errors="coerce")
            z = robust_zscore_series(vals, cap=cap) if robust else zscore_series(vals).clip(-cap, cap)
            z *= _sample_shrinkage(int(vals.notna().sum()))
            out[f"{col}_z"] = z
    return out


def build_fundamental_zscores(
    fund_raw: pd.DataFrame,
    min_fund_cols: Optional[int] = None,
    robust: bool = True,
    cap: float | None = None,
) -> pd.DataFrame:
    if fund_raw is None or fund_raw.empty:
        return pd.DataFrame()
    min_fund_cols = SCORING_CONFIG["min_fund_cols"] if min_fund_cols is None else int(min_fund_cols)
    cap = SCORING_CONFIG["z_cap"] if cap is None else cap

    core = [c for c in FUNDAMENTAL_POSITIVE_FACTORS + FUNDAMENTAL_INVERTED_FACTORS if c in fund_raw.columns]
    filt = fund_raw.copy()
    if core:
        required = min(max(1, min_fund_cols), len(core))
        filt = filt[filt[core].notna().sum(axis=1) >= required]
    if filt.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=filt.index)
    for col in FUNDAMENTAL_POSITIVE_FACTORS:
        if col in filt.columns:
            vals = filt[col]
            z = robust_zscore_series(vals, cap=cap) if robust else zscore_series(vals).clip(-cap, cap)
            z *= _sample_shrinkage(int(vals.notna().sum()))
            out[f"{col}_z"] = z

    for col in FUNDAMENTAL_INVERTED_FACTORS:
        if col in filt.columns:
            vals = -pd.to_numeric(filt[col], errors="coerce")
            z = robust_zscore_series(vals, cap=cap) if robust else zscore_series(vals).clip(-cap, cap)
            z *= _sample_shrinkage(int(vals.notna().sum()))
            out[f"{col}_z"] = z
    return out


def _coverage_for_ticker(df: pd.DataFrame, ticker: str, cols: List[str]) -> float:
    if df is None or df.empty or not cols or ticker not in df.index:
        return 0.0
    return float(df.loc[ticker, cols].notna().mean())


def _window_return(series: pd.Series, window: int) -> float:
    if series is None or series.empty or series.size < window + 5:
        return np.nan
    end = float(series.iloc[-1])
    start = float(series.iloc[-window])
    if start <= 0:
        return np.nan
    return end / start - 1.0


def macro_from_vix(vix: pd.Series):
    if vix is None or vix.empty:
        return 0.5, np.nan, np.nan, np.nan
    vix_last = float(vix.iloc[-1])
    ema20 = float(ema(vix, 20).iloc[-1]) if len(vix) >= 20 else vix_last
    rel = (vix_last - ema20) / max(ema20, 1e-9)
    if vix_last <= 12:
        level = 1.0
    elif vix_last >= 28:
        level = 0.0
    else:
        level = 1.0 - (vix_last - 12) / 16.0
    if rel >= 0.03:
        trend = 0.0
    elif rel <= -0.03:
        trend = 1.0
    else:
        trend = float(np.clip(1.0 - (rel + 0.03) / 0.06, 0.0, 1.0))
    macro = float(np.clip(0.70 * level + 0.30 * trend, 0.0, 1.0))
    return macro, vix_last, ema20, rel


def _scaled_score_from_return(ret: float, positive_is_risk_on: bool, scale: float = 0.08) -> float:
    """
    Smooth return-to-score mapping via tanh to reduce threshold artifacts.
    """
    if np.isnan(ret):
        return np.nan
    score = 0.5 + 0.5 * np.tanh(ret / max(scale, 1e-9))
    return float(score if positive_is_risk_on else 1.0 - score)


def macro_from_signals(
    vix: pd.Series,
    gold: pd.Series,
    dxy: pd.Series,
    tnx: pd.Series,
    credit_ratio: pd.Series,
    window: int = 63,
):
    vix_score, vix_last, ema20, rel = macro_from_vix(vix)

    gold_ret = _window_return(gold, window)
    dxy_ret = _window_return(dxy, window)
    credit_ret = _window_return(credit_ratio, window)

    if tnx is None or tnx.empty or tnx.size < window + 5:
        tnx_delta = np.nan
    else:
        tnx_delta = float((tnx.iloc[-1] - tnx.iloc[-window]) / 10.0)

    gold_score = _scaled_score_from_return(gold_ret, positive_is_risk_on=False, scale=0.08)
    dxy_score = _scaled_score_from_return(dxy_ret, positive_is_risk_on=False, scale=0.05)
    credit_score = _scaled_score_from_return(credit_ret, positive_is_risk_on=True, scale=0.06)
    tnx_score = np.nan if np.isnan(tnx_delta) else float(0.5 + 0.5 * np.tanh((-tnx_delta) / 0.30))

    parts = {
        "vix": (vix_score, 0.50),
        "gold": (gold_score, 0.15),
        "dxy": (dxy_score, 0.15),
        "tnx": (tnx_score, 0.10),
        "credit": (credit_score, 0.10),
    }
    valid_weight = sum(w for score, w in parts.values() if pd.notna(score))
    if valid_weight <= 0:
        macro = 0.5
        signal_coverage = 0.0
    else:
        macro = float(sum(score * w for score, w in parts.values() if pd.notna(score)) / valid_weight)
        signal_coverage = float(valid_weight / sum(w for _, w in parts.values()))

    return {
        "macro": float(np.clip(macro, 0.0, 1.0)),
        "vix_last": vix_last,
        "vix_ema20": ema20,
        "vix_gap": rel,
        "gold_ret": gold_ret,
        "dxy_ret": dxy_ret,
        "tnx_delta": tnx_delta,
        "credit_ret": credit_ret,
        "signal_coverage": signal_coverage,
    }


@st.cache_data(show_spinner=False, ttl=900)
def fetch_macro_pack(period: str = "6mo", interval: str = "1d", window: int = 63) -> Dict[str, float]:
    return macro_from_signals(
        fetch_vix_series(period=period, interval=interval),
        fetch_gold_series(period=period, interval=interval),
        fetch_dxy_series(period=period, interval=interval),
        fetch_tnx_series(period=period, interval=interval),
        fetch_credit_ratio_series(period=period, interval=interval),
        window=window,
    )


def score_universe_panel(
    panel: Dict[str, pd.Series],
    target_count: int,
    macro_pack: Optional[Dict[str, float]] = None,
    fund_weight: Optional[float] = None,
    tech_weight: Optional[float] = None,
    macro_weight: Optional[float] = None,
    min_fund_cols: Optional[int] = None,
) -> Dict[str, object]:
    """
    Shared scoring engine used across Stock, Portfolio, Tracker, and Battle pages.
    Returns:
      out, fund, tech, fund_raw, weights, macro_pack
    """
    idx = pd.Index(list(panel.keys()))
    out = pd.DataFrame(index=idx)
    if out.empty:
        return {
            "out": out,
            "fund": pd.DataFrame(),
            "tech": pd.DataFrame(),
            "fund_raw": pd.DataFrame(),
            "weights": normalize_score_weights(fund_weight, tech_weight, macro_weight),
            "macro_pack": macro_pack or {"macro": 0.5, "signal_coverage": 0.0},
        }

    tech_raw = technical_scores(panel)
    tech = build_technical_zscores(tech_raw, robust=True, cap=SCORING_CONFIG["z_cap"])
    tech_cols = [c for c in tech.columns if c.endswith("_z")]
    tech_score = _component_score(tech[tech_cols]) if tech_cols else pd.Series(dtype=float)

    fund_raw = fetch_fundamentals_simple(list(panel.keys()))
    fund = build_fundamental_zscores(
        fund_raw, min_fund_cols=min_fund_cols, robust=True, cap=SCORING_CONFIG["z_cap"]
    )
    fund_cols = [c for c in fund.columns if c.endswith("_z")]
    fund_score = _component_score(fund[fund_cols]) if fund_cols else pd.Series(dtype=float)

    macro_pack = fetch_macro_pack() if macro_pack is None else macro_pack
    macro_score = float(np.clip(macro_pack.get("macro", 0.5), 0.0, 1.0))
    macro_cov = float(np.clip(macro_pack.get("signal_coverage", 0.0), 0.0, 1.0))

    out["FUND_score"] = fund_score.reindex(idx)
    out["TECH_score"] = tech_score.reindex(idx)
    out["MACRO_score"] = macro_score

    weights = normalize_score_weights(fund_weight, tech_weight, macro_weight)
    out["COMPOSITE"] = _weighted_row_average(out[["FUND_score", "TECH_score", "MACRO_score"]], weights)
    ratings = percentile_rank(out["COMPOSITE"].dropna())
    out["RATING_0_100"] = ratings.reindex(out.index)
    out["RECO"] = out["RATING_0_100"].apply(score_label)

    peer_loaded = int(out["COMPOSITE"].notna().sum())
    peer_factor = float(np.clip(peer_loaded / max(target_count, 1), 0.0, 1.0))
    depth_factor = float(np.clip(math.sqrt(peer_loaded / max(min(target_count, 60), 1)), 0.0, 1.0))
    component_cov = out[["FUND_score", "TECH_score", "MACRO_score"]].notna().mean(axis=1).fillna(0.0)
    confidence = []
    for t in out.index:
        fund_cov = _coverage_for_ticker(fund, t, fund_cols)
        tech_cov = _coverage_for_ticker(tech, t, tech_cols)
        comp_cov = float(component_cov.loc[t]) if t in component_cov.index else 0.0
        conf = 100.0 * (
            0.30 * peer_factor
            + 0.15 * depth_factor
            + 0.25 * fund_cov
            + 0.20 * tech_cov
            + 0.10 * macro_cov * comp_cov
        )
        confidence.append(float(np.clip(conf, 0.0, 100.0)))
    out["CONFIDENCE"] = confidence

    return {
        "out": out,
        "fund": fund,
        "tech": tech,
        "fund_raw": fund_raw,
        "weights": weights,
        "macro_pack": macro_pack,
    }


# ========================= Fundamentals interpretation ========================

def fundamentals_interpretation(zrow: pd.Series) -> List[str]:
    lines=[]
    def bucket(v, pos_good=True):
        if pd.isna(v): return "neutral"
        if pos_good:
            return "bullish" if v>=0.5 else "watch" if v<=-0.5 else "neutral"
        else:
            return "bullish (cheap)" if v>=0.5 else "watch (expensive)" if v<=-0.5 else "neutral"

    g=bucket(zrow.get("revenueGrowth_z"))
    e=bucket(zrow.get("earningsGrowth_z"))
    pm=bucket(zrow.get("profitMargins_z"))
    gm=bucket(zrow.get("grossMargins_z"))
    om=bucket(zrow.get("operatingMargins_z"))
    roa=bucket(zrow.get("returnOnAssets_z"))
    roe=bucket(zrow.get("returnOnEquity_z"))
    val=bucket(zrow.get("forwardPE_z"), pos_good=False)
    ev=bucket(zrow.get("enterpriseToEbitda_z"), pos_good=False)
    lev=bucket(zrow.get("debtToEquity_z"), pos_good=False)
    fcf=bucket(zrow.get("fcfYield_z"))

    if g=="bullish" or e=="bullish": lines.append("**Growth tilt:** above-peer growth (supportive).")
    elif g=="watch" or e=="watch":   lines.append("**Growth tilt:** below peers — watch for re-acceleration.")
    else:                            lines.append("**Growth tilt:** peer-like.")

    if (pm=="bullish" or gm=="bullish" or om=="bullish" or roa=="bullish" or roe=="bullish"):
        lines.append("**Profitability:** strong vs peers.")
    elif (pm=="watch" or gm=="watch" or om=="watch" or roa=="watch" or roe=="watch"):
        lines.append("**Profitability:** below peer medians — monitor margins.")
    else:
        lines.append("**Profitability:** roughly peer-like.")

    if val.startswith("bullish") or ev.startswith("bullish"):
        lines.append("**Valuation:** cheaper than peers.")
    elif val.startswith("watch") or ev.startswith("watch"):
        lines.append("**Valuation:** richer than peers — execution must stay strong.")
    else:                         lines.append("**Valuation:** around peer medians.")

    if fcf=="bullish": lines.append("**Cash generation:** attractive FCF yield.")
    elif fcf=="watch": lines.append("**Cash generation:** weak FCF yield vs peers.")

    if lev.startswith("bullish"): lines.append("**Balance sheet:** lower leverage.")
    elif lev.startswith("watch"): lines.append("**Balance sheet:** higher leverage — keep an eye on cash flow.")
    else:                         lines.append("**Balance sheet:** typical for the peer set.")
    return lines


# ============================ Portfolio editor helpers =========================

CURRENCY_MAP = {"$":"USD","€":"EUR","£":"GBP","CHF":"CHF","C$":"CAD","A$":"AUD","¥":"JPY"}
def _safe_num(x): return pd.to_numeric(x, errors="coerce")

def normalize_percents_to_100(p: pd.Series) -> pd.Series:
    p=_safe_num(p).fillna(0.0); s=p.sum()
    return (p/s*100.0) if s>0 else p

def sync_percent_amount(df: pd.DataFrame, total: float, mode: str) -> pd.DataFrame:
    df=df.copy()
    df["Ticker"]=df["Ticker"].astype(str).str.strip()
    df=df[df["Ticker"].astype(bool)].reset_index(drop=True)
    n=len(df)
    if n==0: df["weight"]=[]; return df

    df["Percent (%)"]=_safe_num(df.get("Percent (%)"))
    df["Amount"]=_safe_num(df.get("Amount"))
    has_total = (total is not None and total>0)

    if has_total:
        if mode=="percent":
            if df["Percent (%)"].fillna(0).sum()==0: df["Percent (%)"]=100.0/n
            df["Percent (%)"]=normalize_percents_to_100(df["Percent (%)"]).round(2)
            df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
        else:
            if df["Amount"].fillna(0).sum()>0:
                df["Percent (%)"]=(df["Amount"]/total*100.0).round(2)
                df["Percent (%)"]=normalize_percents_to_100(df["Percent (%)"]).round(2)
                df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
            else:
                df["Percent (%)"]=100.0/n
                df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
    else:
        if df["Percent (%)"].fillna(0).sum()==0: df["Percent (%)"]=100.0/n
        df["Percent (%)"]=normalize_percents_to_100(df["Percent (%)"]).round(2)

    if has_total and df["Amount"].fillna(0).sum()>0:
        w=df["Amount"].fillna(0)/df["Amount"].fillna(0).sum()
    elif df["Percent (%)"].fillna(0).sum()>0:
        w=df["Percent (%)"].fillna(0)/df["Percent (%)"].fillna(0).sum()
    else:
        w=pd.Series([1.0/n]*n, index=df.index)
    df["weight"]=w
    return df

def holdings_editor_form(currency_symbol: str, total_value: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if st.session_state.get("grid_df") is None:
        st.session_state["grid_df"]=pd.DataFrame({
            "Ticker":["AAPL","MSFT","NVDA","AMZN"],
            "Percent (%)":[25.0,25.0,25.0,25.0],
            "Amount":[np.nan,np.nan,np.nan,np.nan],
        })

    st.markdown(
        f"**Holdings**  \n"
        f"<span class='small-muted'>Enter <b>Ticker</b> and either <b>Percent (%)</b> or "
        f"<b>Amount ({currency_symbol})</b>. Click <b>Apply changes</b> to update.</span>",
        unsafe_allow_html=True,
    )

    committed=st.session_state["grid_df"].copy()
    with st.form("holdings_form", clear_on_submit=False):
        sync_mode = st.segmented_control("Sync mode",
                    options=["Percent → Amount","Amount → Percent"],
                    default="Percent → Amount")
        mode_key={"Percent → Amount":"percent","Amount → Percent":"amount"}[sync_mode]

        edited = st.data_editor(
            committed, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn(width="small"),
                "Percent (%)": st.column_config.NumberColumn(format="%.2f"),
                "Amount": st.column_config.NumberColumn(format="%.2f", help=f"Amount in {currency_symbol}"),
            },
            key="grid_form",
        )
        c1,c2=st.columns(2)
        apply_btn=c1.form_submit_button("Apply changes", type="primary", use_container_width=True)
        normalize_btn=c2.form_submit_button("Normalize to 100%", use_container_width=True)

    if normalize_btn:
        syncd=edited.copy()
        syncd["Percent (%)"]=normalize_percents_to_100(_safe_num(syncd.get("Percent (%)")))
        if total_value and total_value>0:
            syncd["Amount"]=(syncd["Percent (%)"]/100.0*total_value).round(2)
        st.session_state["grid_df"]=syncd[["Ticker","Percent (%)","Amount"]]
    elif apply_btn:
        syncd=sync_percent_amount(edited.copy(), total_value, mode_key)
        st.session_state["grid_df"]=syncd[["Ticker","Percent (%)","Amount"]]

    current=st.session_state["grid_df"].copy()
    out=current.copy()
    out["ticker"]=out["Ticker"].map(yf_symbol)
    out=out[out["ticker"].astype(bool)]

    if total_value and total_value>0 and _safe_num(out["Amount"]).sum()>0:
        w=_safe_num(out["Amount"]) / _safe_num(out["Amount"]).sum()
    elif _safe_num(out["Percent (%)"]).sum()>0:
        w=_safe_num(out["Percent (%)"]) / _safe_num(out["Percent (%)"]).sum()
    else:
        n=max(len(out),1); w=pd.Series([1.0/n]*n, index=out.index)

    df_hold=pd.DataFrame({"ticker":out["ticker"],"weight":w})
    return df_hold, current


# =============== Financial statements: essential, clean, ordered ===============

# Public currency map export (avoid import errors in pages)
_CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CAD": "C$", "AUD": "A$", "CHF": "CHF",
    "SEK": "kr", "NOK": "kr", "DKK": "kr", "HKD": "HK$", "CNY": "¥", "INR": "₹", "SGD": "S$"
}
CURRENCY_SYMBOL = _CURRENCY_SYMBOL

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _clean_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or isinstance(df,float) or df.empty: return pd.DataFrame()
    out=df.copy()
    out=out[~out.index.duplicated(keep="first")]
    for c in out.columns: out[c]=pd.to_numeric(out[c], errors="coerce")
    try:
        out.columns=pd.to_datetime(out.columns); out=out.sort_index(axis=1)
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False, ttl=21600)
def fetch_company_statements(ticker: str) -> Dict[str, object]:
    t=yf.Ticker(yf_symbol(ticker))
    try: info=t.info or {}
    except Exception: info={}
    currency = info.get("financialCurrency") or info.get("currency") or "USD"

    def safe(attr):
        try: return _clean_statement(getattr(t, attr))
        except Exception: return pd.DataFrame()

    return {
        "currency": currency,
        "income": safe("financials"),
        "balance": safe("balance_sheet"),
        "cashflow": safe("cashflow"),
        "income_q": safe("quarterly_financials"),
        "balance_q": safe("quarterly_balance_sheet"),
        "cashflow_q": safe("quarterly_cashflow"),
    }

@st.cache_data(show_spinner=False, ttl=21600)
def fetch_sector(ticker: str) -> Optional[str]:
    t = yf_symbol(ticker)
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}
    return info.get("sector")

# Essential, accountant-ordered specs
INCOME_SPEC: List[Tuple[str, List[str]]] = [
    ("Revenue", ["Total Revenue","Revenue","Operating Revenue","Sales"]),
    ("Cost of Revenue", ["Cost Of Revenue","Cost of Goods Sold"]),
    ("Gross Profit", ["Gross Profit"]),  # if missing, we compute as Revenue - CoR
    ("Research & Development", ["Research Development","Research And Development"]),
    ("SG&A", ["Selling General Administrative","Selling General And Administrative","SG&A","General And Administrative Expense"]),
    ("Operating Income", ["Operating Income","Operating Profit","Ebit"]),
    ("Interest Expense", ["Interest Expense","Interest Expense Non Operating"]),
    ("Other Income/Expense", ["Other Income Expense","Other Non Operating Income Expense","Other Non Operating Income/Expense"]),
    ("Pretax Income", ["Pretax Income","Income Before Tax","Earnings Before Tax"]),
    ("Income Tax", ["Income Tax Expense","Provision For Income Taxes"]),
    ("Net Income", ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares",
                    "Net Income Including Noncontrolling Interests","Net Income From Continuing Operations"]),
]

BALANCE_SPEC: List[Tuple[str, List[str]]] = [
    ("Cash & ST Investments", ["Cash And Cash Equivalents","Cash","Cash Equivalents","Short Term Investments","Cash And Short Term Investments"]),
    ("Accounts Receivable", ["Net Receivables","Accounts Receivable"]),
    ("Inventory", ["Inventory"]),
    ("Other Current Assets", ["Other Current Assets"]),
    ("Total Current Assets", ["Total Current Assets"]),
    ("PP&E", ["Property Plant Equipment","Net Property Plant And Equipment","Property, Plant & Equipment"]),
    ("Goodwill & Intangibles", ["Good Will","Goodwill","Intangible Assets"]),
    ("Other Non-current Assets", ["Other Assets","Non Current Assets","Other Non-Current Assets"]),
    ("Total Assets", ["Total Assets"]),
    ("Accounts Payable", ["Accounts Payable"]),
    ("Short-term Debt", ["Short Long Term Debt","Short-Term Debt","Current Portion of Long Term Debt"]),
    ("Other Current Liabilities", ["Other Current Liab","Other Current Liabilities"]),
    ("Total Current Liabilities", ["Total Current Liabilities"]),
    ("Long-term Debt", ["Long Term Debt"]),
    ("Other Non-current Liabilities", ["Long Term Liabilities","Other Non-Current Liabilities"]),
    ("Total Liabilities", ["Total Liab","Total Liabilities"]),
    ("Total Equity", ["Total Stockholder Equity","Total Shareholder Equity","Total Equity","Common Stock Equity"]),
    ("Total Liabilities & Equity", ["Total Liabilities & Stockholders' Equity","Total Liabilities And Stockholders Equity"]),
]

CASHFLOW_SPEC: List[Tuple[str, List[str]]] = [
    ("Cash from Operations", ["Total Cash From Operating Activities","Operating Cash Flow"]),
    ("Capital Expenditures", ["Capital Expenditures","Investments In Property Plant And Equipment"]),
    ("Free Cash Flow", []),  # computed if possible
    ("Cash from Investing", ["Total Cashflows From Investing Activities","Investing Cash Flow"]),
    ("Cash from Financing", ["Total Cash From Financing Activities","Financing Cash Flow"]),
    ("Net Change in Cash", ["Change In Cash","Change In Cash And Cash Equivalents"]),
]

def _find_row(df: pd.DataFrame, synonyms: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    idx_map={_norm(i):i for i in df.index.astype(str)}
    # exact
    for name in synonyms:
        key=_norm(name)
        if key in idx_map:
            return pd.to_numeric(df.loc[idx_map[key]], errors="coerce")
    # contains
    for name in synonyms:
        key=_norm(name)
        for k,orig in idx_map.items():
            if key in k:
                return pd.to_numeric(df.loc[orig], errors="coerce")
    return None

def _ordered_table(df: pd.DataFrame, spec: List[Tuple[str, List[str]]], take: int = 4) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    rows = []
    for label, syns in spec:
        if label == "Free Cash Flow":
            # compute later if possible
            rows.append((label, None))
            continue
        s = _find_row(df, syns)
        if s is not None:
            rows.append((label, s))

    tbl = pd.DataFrame({name: ser for name,ser in rows if ser is not None}).T if rows else pd.DataFrame()

    # Derived lines:
    # Gross Profit = Revenue - CoR (if missing)
    if "Gross Profit" not in tbl.index:
        if "Revenue" in tbl.index and "Cost of Revenue" in tbl.index:
            try:
                gp = tbl.loc["Revenue"].sub(tbl.loc["Cost of Revenue"], fill_value=np.nan)
                tbl.loc["Gross Profit"] = gp
            except Exception:
                pass

    # Free Cash Flow = CFO - CapEx (if missing and CFO/CapEx exist)
    if any(lab == "Free Cash Flow" for lab,_ in spec):
        if "Free Cash Flow" not in tbl.index:
            try:
                cfo = _find_row(df, ["Total Cash From Operating Activities","Operating Cash Flow"])
                capex = _find_row(df, ["Capital Expenditures","Investments In Property Plant And Equipment"])
                if cfo is not None and capex is not None:
                    fcf = cfo.sub(capex, fill_value=np.nan)
                    tbl.loc["Free Cash Flow"] = fcf
            except Exception:
                pass

    # Balance: enforce total L&E if possible
    if spec is BALANCE_SPEC and "Total Liabilities & Equity" not in tbl.index:
        try:
            if "Total Liabilities" in tbl.index and "Total Equity" in tbl.index:
                tle = tbl.loc["Total Liabilities"].add(tbl.loc["Total Equity"], fill_value=np.nan)
                tbl.loc["Total Liabilities & Equity"] = tle
        except Exception:
            pass

    # column order & last N periods
    try:
        tbl.columns=pd.to_datetime(tbl.columns); tbl=tbl.sort_index(axis=1)
        tbl.columns=[c.date() for c in tbl.columns]
    except Exception:
        tbl.columns=[str(c) for c in tbl.columns]
    tbl = tbl.iloc[:, -take:]

    # row order exactly as spec (existing only)
    wanted = [lab for lab,_ in spec]
    tbl = tbl.loc[[r for r in wanted if r in tbl.index]]

    return tbl

def build_compact_statements(stmts: Dict[str, object],
                             freq: str = "annual",
                             take: int = 4) -> Dict[str, pd.DataFrame]:
    inc = stmts["income"]   if freq=="annual" else stmts["income_q"]
    bal = stmts["balance"]  if freq=="annual" else stmts["balance_q"]
    cfs = stmts["cashflow"] if freq=="annual" else stmts["cashflow_q"]
    return {
        "income":   _ordered_table(inc, INCOME_SPEC, take=take),
        "balance":  _ordered_table(bal, BALANCE_SPEC, take=take),
        "cashflow": _ordered_table(cfs, CASHFLOW_SPEC, take=take),
    }

def tidy_statement_for_display(df: pd.DataFrame, take: int = 4) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out=df.copy()
    try:
        out.columns=pd.to_datetime(out.columns); out=out.sort_index(axis=1); out=out.iloc[:, -take:]; out.columns=[c.date() for c in out.columns]
    except Exception:
        out=out.iloc[:, -take:]; out.columns=[str(c) for c in out.columns]
    return out


# ---------------------- Statement-derived metrics + text -----------------------

def statement_metrics(stmts: Dict[str, object]) -> Dict[str, float]:
    """
    Compute key ratios/trends from (prefer) annual statements.
    Returns a dict with revenue, margins, growth, liquidity, leverage, returns, coverage, FCF.
    """
    inc = stmts.get("income", pd.DataFrame())
    bs  = stmts.get("balance", pd.DataFrame())
    cf  = stmts.get("cashflow", pd.DataFrame())

    def pick(df, names): return _find_row(df, names)

    rev   = pick(inc, ["Total Revenue","Revenue","Sales","Operating Revenue"])
    gross = pick(inc, ["Gross Profit"])
    opinc = pick(inc, ["Operating Income","Operating Profit","Ebit"])
    net   = pick(inc, ["Net Income","Net Income Common Stockholders","Net Income Applicable To Common Shares",
                       "Net Income Including Noncontrolling Interests","Net Income From Continuing Operations"])
    intr  = pick(inc, ["Interest Expense","Interest Expense Non Operating"])

    ta    = pick(bs,  ["Total Assets"])
    teq   = pick(bs,  ["Total Stockholder Equity","Total Shareholder Equity","Total Equity","Common Stock Equity"])
    tliab = pick(bs,  ["Total Liab","Total Liabilities"])
    ca    = pick(bs,  ["Total Current Assets"])
    cl    = pick(bs,  ["Total Current Liabilities"])
    inv   = pick(bs,  ["Inventory"])
    ltd   = pick(bs,  ["Long Term Debt"])
    std   = pick(bs,  ["Short Long Term Debt","Short-Term Debt","Current Portion of Long Term Debt"])
    notes = pick(bs,  ["Notes Payable"])

    ocf   = pick(cf,  ["Total Cash From Operating Activities","Operating Cash Flow"])
    capex = pick(cf,  ["Capital Expenditures","Investments In Property Plant And Equipment"])

    def last(series):
        return (series.dropna().iloc[-1] if series is not None and series.dropna().size else np.nan)
    def prev(series):
        return (series.dropna().iloc[-2] if series is not None and series.dropna().size>1 else np.nan)

    revenue      = float(last(rev));      revenue_prev = float(prev(rev))
    gross_profit = float(last(gross))
    operating_inc= float(last(opinc))
    net_income   = float(last(net))
    interest_exp = abs(float(last(intr))) if not np.isnan(last(intr)) else np.nan

    total_assets = float(last(ta))
    equity       = float(last(teq)) if teq is not None else np.nan
    if np.isnan(equity) and (ta is not None and tliab is not None):
        equity = float(last(ta)) - float(last(tliab))

    current_assets      = float(last(ca))
    current_liabilities = float(last(cl))
    inventory           = float(last(inv))
    lt_debt             = float(last(ltd))
    st_debt             = float(last(std))
    notes_pay           = float(last(notes))
    total_debt          = np.nansum([lt_debt, st_debt, notes_pay])

    op_cash_flow = float(last(ocf))
    capex_val    = float(last(capex))
    fcf          = op_cash_flow - (capex_val if not np.isnan(capex_val) else 0.0)

    # margins & growth
    gross_margin      = (gross_profit/revenue) if revenue else np.nan
    operating_margin  = (operating_inc/revenue) if revenue else np.nan
    net_margin        = (net_income/revenue) if revenue else np.nan
    fcf_margin        = (fcf/revenue) if revenue else np.nan
    revenue_yoy       = (revenue/revenue_prev - 1.0) if (revenue_prev and not np.isnan(revenue_prev)) else np.nan

    # liquidity
    current_ratio = (current_assets/current_liabilities) if current_liabilities else np.nan
    quick_ratio   = ((current_assets - (inventory if not np.isnan(inventory) else 0.0)) / current_liabilities) if current_liabilities else np.nan

    # leverage & returns
    d_to_e   = (total_debt/equity) if equity else np.nan
    roe      = (net_income/equity) if equity else np.nan
    roa      = (net_income/total_assets) if total_assets else np.nan
    icov     = (operating_inc/interest_exp) if interest_exp and not np.isnan(operating_inc) else np.nan

    # margin trends (YoY)
    def _ratio_yoy(series_num, series_den):
        try:
            n0, n1 = float(last(series_num)), float(prev(series_num))
            d0, d1 = float(last(series_den)), float(prev(series_den))
            if d0 and d1:
                m0, m1 = (n0/d0), (n1/d1)
                return m0, (m0 - m1)
        except Exception:
            pass
        return np.nan, np.nan

    gm, gm_chg = _ratio_yoy(gross, rev)
    om, om_chg = _ratio_yoy(opinc, rev)
    nm, nm_chg = _ratio_yoy(net, rev)

    # 3Y revenue CAGR
    revenue_cagr = np.nan
    if rev is not None and rev.dropna().size >= 4:
        try:
            v0 = float(rev.dropna().iloc[-4]); v1 = float(rev.dropna().iloc[-1])
            if v0>0 and v1>0:
                revenue_cagr = (v1/v0)**(1/3) - 1
        except Exception:
            pass

    return {
        "revenue": revenue, "revenue_prev": revenue_prev, "revenue_yoy": revenue_yoy, "revenue_cagr3y": revenue_cagr,
        "gross_margin": gross_margin, "operating_margin": operating_margin, "net_margin": net_margin,
        "gross_margin_chg": gm_chg, "operating_margin_chg": om_chg, "net_margin_chg": nm_chg,
        "fcf": fcf, "fcf_margin": fcf_margin, "ocf": op_cash_flow, "capex": capex_val,
        "current_ratio": current_ratio, "quick_ratio": quick_ratio,
        "debt_total": total_debt, "debt_to_equity": d_to_e,
        "roe": roe, "roa": roa, "interest_coverage": icov,
        "assets": total_assets, "equity": equity
    }

def interpret_statement_metrics(m: Dict[str, float]) -> List[str]:
    """Plain-English narrative tying statements to fundamentals."""
    out = []

    # Growth
    y = m.get("revenue_yoy"); c = m.get("revenue_cagr3y")
    if not np.isnan(y):
        if y > 0.1: out.append(f"**Top-line growth:** Revenue grew ~{y*100:.1f}% YoY (strong).")
        elif y > 0.0: out.append(f"**Top-line growth:** Revenue grew ~{y*100:.1f}% YoY (modest).")
        else: out.append(f"**Top-line growth:** Revenue declined ~{abs(y)*100:.1f}% YoY (headwind).")
    if not np.isnan(c):
        if c > 0.1: out.append(f"**3Y CAGR:** ~{c*100:.1f}% (strong multi-year).")
        elif c > 0.0: out.append(f"**3Y CAGR:** ~{c*100:.1f}% (slow but positive).")
        else: out.append("**3Y CAGR:** roughly flat/negative.")

    # Margins & profitability
    gm, om, nm = m.get("gross_margin"), m.get("operating_margin"), m.get("net_margin")
    if not np.isnan(gm):
        qual = "high" if gm >= 0.5 else "mid" if gm >= 0.3 else "low"
        out.append(f"**Gross margin:** {gm*100:.1f}% ({qual}).")
    if not np.isnan(om):
        if om >= 0.2: lvl="excellent"
        elif om >= 0.1: lvl="healthy"
        elif om >= 0.0: lvl="thin"
        else: lvl="negative"
        out.append(f"**Operating margin:** {om*100:.1f}% ({lvl}).")
    if not np.isnan(nm):
        out.append(f"**Net margin:** {nm*100:.1f}%.")

    for nm_key, label in [("gross_margin_chg","gross"),("operating_margin_chg","operating"),("net_margin_chg","net")]:
        delta = m.get(nm_key)
        if not np.isnan(delta) and abs(delta) >= 0.02:
            out.append(f"**{label.title()} margin trend:** {('expanding' if delta>0 else 'contracting')} ~{abs(delta)*100:.1f} pp YoY.")

    # Cash flow quality
    fcfm = m.get("fcf_margin")
    if not np.isnan(fcfm):
        if fcfm >= 0.10: txt="strong free-cash-flow generation"
        elif fcfm >= 0.0: txt="modest free-cash-flow generation"
        else: txt="negative free cash flow (investment/pressure)"
        out.append(f"**FCF margin:** {fcfm*100:.1f}% — {txt}.")

    # Liquidity
    cr, qr = m.get("current_ratio"), m.get("quick_ratio")
    if not np.isnan(cr):
        if cr >= 2.0: s="strong"
        elif cr >= 1.0: s="adequate"
        else: s="tight"
        out.append(f"**Liquidity:** current ratio {cr:.2f} ({s}).")
    if not np.isnan(qr): out.append(f"**Quick ratio:** {qr:.2f}.")

    # Leverage
    de = m.get("debt_to_equity")
    if not np.isnan(de):
        if de <= 0.5: s="conservative"
        elif de <= 1.5: s="moderate"
        else: s="high"
        out.append(f"**Leverage:** Debt/Equity ≈ {de:.2f} ({s}).")

    ic = m.get("interest_coverage")
    if not np.isnan(ic):
        if ic >= 6: s="comfortable"
        elif ic >= 2: s="manageable"
        else: s="stressed"
        out.append(f"**Interest coverage:** ~{ic:.1f}× ({s}).")

    # Returns
    for k, nm in [("roe","ROE"),("roa","ROA")]:
        v = m.get(k)
        if not np.isnan(v): out.append(f"**{nm}:** {v*100:.1f}%.")

    return out


# ========================== Simple stock price predictor =======================

def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float,float]:
    """Return slope, intercept for y ~ a*x + b (least squares)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n == 0: return 0.0, float('nan')
    xbar = x.mean(); ybar = y.mean()
    xx = ((x - xbar) ** 2).sum()
    if xx == 0: return 0.0, ybar
    a = ((x - xbar) * (y - ybar)).sum() / xx
    b = ybar - a * xbar
    return a, b

def predict_price(series: pd.Series, horizon_days: int = 30, lookback_days: int = 252) -> Dict[str, object]:
    """
    Very simple, robust predictor:
      - fit a straight line to log-price over last `lookback_days`
      - extrapolate `horizon_days`
      - build +/- 1σ bands using realized daily volatility
    Returns dict: {'forecast': DataFrame, 'slope': ..., 'vol_ann': ..., 'last_price': ...}
    """
    s = series.dropna().astype(float)
    if s.size < 30:
        return {"forecast": pd.DataFrame(), "slope": np.nan, "vol_ann": np.nan, "last_price": s.iloc[-1] if s.size else np.nan}

    s = s.iloc[-min(lookback_days, s.size):]
    t = np.arange(s.size)
    logp = np.log(s.values)
    a, b = _ols_slope_intercept(t, logp)  # log-price = a*t + b

    # realized vol
    ret = np.diff(logp)
    vol_d = np.nanstd(ret, ddof=1)
    vol_ann = vol_d * np.sqrt(252)

    # forecast path
    future_t = np.arange(s.size, s.size + horizon_days)
    base = a * future_t + b
    mid = np.exp(base)

    # 1σ bands on price scale via log bands
    up = np.exp(base + vol_d)
    dn = np.exp(base - vol_d)

    idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="B")
    df = pd.DataFrame({"mid": mid, "lo_1s": dn, "hi_1s": up}, index=idx)

    return {"forecast": df, "slope": a, "vol_ann": vol_ann, "last_price": float(s.iloc[-1])}


# =============== 2D multivariate Gaussian centrality / anomaly score ==========

def _mah2(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> float:
    d = x - mu
    return float(d.T @ inv_cov @ d)

def gaussian2d_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit mean, cov, inv_cov from 2D samples (rows = samples, cols = 2).
    Returns (mu, cov, inv_cov). Adds tiny ridge if singular.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 5:
        raise ValueError("Need at least 5 samples of 2D points.")
    mu = X.mean(axis=0)
    cov = np.cov(X.T, ddof=1)
    # ridge if singular
    eps = 1e-8
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.inv(cov + eps*np.eye(2))
    return mu, cov, inv_cov

def gaussian2d_centrality(x: Tuple[float,float], mu: np.ndarray, inv_cov: np.ndarray) -> Dict[str, float]:
    """
    Centrality score in (0,1], where 1 is most central (Mahalanobis=0).
    For df=2, Chi-square CDF is 1 - exp(-MD2/2). We return exp(-MD2/2) as a
    "central density" style score (higher = more typical).
    """
    x = np.asarray(x, dtype=float)
    md2 = _mah2(x, mu, inv_cov)
    score = math.exp(-0.5*md2)      # likelihood-like centrality
    tail  = math.exp(-0.5*md2)      # same functional form for df=2
    return {"md2": md2, "centrality": score, "tail_prob": tail}

def stock_return_vol_features(px: pd.Series, window: int = 60) -> Tuple[float, float]:
    r = np.log(px).diff().dropna()
    if r.size < window: window = max(10, min(window, r.size))
    ret_w = float(np.exp(r.tail(window).sum()) - 1.0)  # window total return (geom)
    vol_w = float(r.tail(window).std(ddof=1) * np.sqrt(252))  # annualized
    return ret_w, vol_w

def peer_return_vol_matrix(price_panel: Dict[str, pd.Series], window: int = 60) -> np.ndarray:
    pts=[]
    for t, s in price_panel.items():
        s=s.dropna()
        if s.size<window+5: continue
        pts.append(stock_return_vol_features(s, window))
    return np.array(pts, dtype=float) if pts else np.empty((0,2), dtype=float)


# =========================== END OF PUBLIC INTERFACE ===========================

# Re-export names your pages import:
__all__ = [
    # UI
    "inject_css","brand_header","topbar_back","inline_logo_svg",
    # helpers
    "yf_symbol","ema","rsi","macd","zscore_series","robust_zscore_series","percentile_rank",
    # loaders & universes
    "fetch_prices_chunked_with_fallback","fetch_vix_series","fetch_gold_series","fetch_dxy_series",
    "fetch_tnx_series","fetch_credit_ratio_series","fetch_fundamentals_simple","fetch_sector",
    "build_universe","PEER_CATALOG","set_peer_catalog",
    # scoring
    "SCORING_CONFIG","score_label",
    "FACTOR_LABELS",
    "technical_scores","build_technical_zscores","build_fundamental_zscores",
    "normalize_score_weights","score_universe_panel",
    "weighted_series_mean","portfolio_signal_score",
    "macro_from_vix","macro_from_signals","fetch_macro_pack","fundamentals_interpretation",
    # portfolio editor
    "CURRENCY_MAP","holdings_editor_form",
    # statements & metrics
    "CURRENCY_SYMBOL","fetch_company_statements","build_compact_statements",
    "tidy_statement_for_display","statement_metrics","interpret_statement_metrics",
    # predictor & gaussian 2D
    "predict_price","gaussian2d_fit","gaussian2d_centrality",
    "stock_return_vol_features","peer_return_vol_matrix",
]
