# app_utils.py — shared utilities for Rate My (Stock + Portfolio + Tracker)
# UI (CSS/brand), peer universes (via indices.py or snapshot), robust price loader,
# features (technicals/macro), fundamentals fetch + interpretation, portfolio editor,
# and well-ordered company financial statements with narrative interpretation.

from __future__ import annotations

import os
import re
import json
import time
import random
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ==========================  UI: CSS + Brand  ==========================

def inject_css() -> None:
    """Global CSS + a small script to rename the first sidebar item to 'Home'."""
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
        .brand{display:flex;align-items:center;justify-content:center;gap:16px;margin:1.0rem 0 .5rem;}
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
        .cta.dark .stButton>button{ background:#171a1f; color:#e6e8eb; border-color:#2e3339; }
        .cta .stButton>button:hover{ transform:translateY(-1px); filter:saturate(1.06) brightness(1.05); }
        .cta.dark .stButton>button:hover{ border-color:#3a3f46; }
        .hr-lite{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);border:0;margin:18px 0;}

        /* Statement sections */
        .pill{display:inline-block;padding:.15rem .5rem;border-radius:999px;border:1px solid #2c3239;background:#151920;color:#cfd4da;font-size:.85rem}
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

def topbar_back(label: str = "← Back", url: Optional[str] = None) -> None:
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if url:
        st.page_link(url, label=label)
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


# ==========================  Peer universes (indices.py → snapshot → fallback) ==========================

try:
    from indices import SP500_LIST as _SP500, NASDAQ100_LIST as _NDX, DOW30_LIST as _DOW
except Exception:
    _SP500, _NDX, _DOW = [], [], []

def _normalize_list(tickers: Iterable[str]) -> List[str]:
    cleaned, seen = [], set()
    for s in tickers:
        if not isinstance(s, str): continue
        s = s.strip().upper().replace(".", "-")
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned

_SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "peer_lists_snapshot.json")

@st.cache_resource(show_spinner=False)
def _load_or_snapshot_peer_lists() -> Dict[str, List[str]]:
    if _SP500 or _NDX or _DOW:
        return {
            "S&P 500": _normalize_list(_SP500),
            "NASDAQ 100": _normalize_list(_NDX),
            "Dow 30": _normalize_list(_DOW),
        }
    if os.path.exists(_SNAPSHOT_FILE):
        try:
            with open(_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for k in obj:
                obj[k] = _normalize_list(obj[k])
            return obj
        except Exception:
            pass
    # minimal fallback if nothing is available
    return {
        "S&P 500": ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","BRK-B","LLY","JPM","V","AVGO","TSLA"],
        "NASDAQ 100": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","NFLX","AMD"],
        "Dow 30": ["AAPL","MSFT","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","GS","HD","HON","IBM","INTC",
                   "JNJ","JPM","KO","MCD","MMM","MRK","MS","NKE","PG","TRV","UNH","V","VZ","WMT","DOW"],
    }

PEER_CATALOG: Dict[str, List[str]] = _load_or_snapshot_peer_lists()

def set_peer_catalog(sp500: Optional[List[str]] = None,
                     ndx: Optional[List[str]] = None,
                     dow: Optional[List[str]] = None) -> None:
    """Update the in-memory peer lists (used by build_universe)."""
    if sp500 is not None: PEER_CATALOG["S&P 500"] = _normalize_list(sp500)
    if ndx is not None:   PEER_CATALOG["NASDAQ 100"] = _normalize_list(ndx)
    if dow is not None:   PEER_CATALOG["Dow 30"] = _normalize_list(dow)


# ==========================  Data fetchers (robust 3-pass)  ==========================

@st.cache_data(show_spinner=False)
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
    """
    Conservative, multi-pass loader:
      1) singles w/ retries,
      2) group download in chunks,
      3) final Ticker.history() singles.
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

    # Pass 2 — bulk groups
    def _append_from_multi(df: pd.DataFrame, group: List[str]):
        if not isinstance(df.columns, pd.MultiIndex):
            t0 = group[0]
            if "Close" in df:
                s = df["Close"].dropna()
                if s.size > 0:
                    frames.append(s.rename(t0)); ok.append(t0)
            return
        got = set(df.columns.get_level_values(0))
        for t in group:
            if t in got:
                s = df[t]["Close"].dropna()
                if s.size > 0:
                    frames.append(s.rename(t)); ok.append(t)

    if missing:
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
            time.sleep(sleep_between + random.uniform(0, 0.25))
        ok_set = set(ok)
        missing = [t for t in names if t not in ok_set]

    # Pass 3 — Ticker.history() fallback
    if missing:
        for t in missing:
            try:
                h = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
                if isinstance(h, pd.DataFrame) and "Close" in h and not h["Close"].dropna().empty:
                    frames.append(h["Close"].dropna().rename(t))
                    ok.append(t)
            except Exception:
                pass
            time.sleep(singles_pause + random.uniform(0.1, 0.35))

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
    lst = PEER_CATALOG.get(label, [])
    return lst if lst else []

def build_universe(user_tickers: List[str], mode: str, sample_n: int = 150, custom_raw: str = "") -> Tuple[List[str], str]:
    """
    Returns (universe, label). Universe always includes user's tickers and
    sampled peers from the chosen set.
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
        chosen = "S&P 500"
        for label_try in ("S&P 500","Dow 30","NASDAQ 100"):
            base = set(_get_catalog_list(label_try))
            if user_set & base:
                chosen = label_try
                break
        label = chosen
        peers_all = [t for t in _get_catalog_list(chosen) if t not in user_set]

    if sample_n and len(peers_all) > sample_n:
        step = max(1, len(peers_all)//sample_n)
        peers = peers_all[::step][:sample_n]
    else:
        peers = peers_all

    universe = sorted(list(user_set | set(peers)))[:700]
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


# ==========================  Fundamentals interpretation  ==========================

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


# ==========================  Portfolio editor helpers  ==========================

CURRENCY_MAP = {"$":"USD","€":"EUR","£":"GBP","CHF":"CHF","C$":"CAD","A$":"AUD","¥":"JPY"}
def _safe_num(x): return pd.to_numeric(x, errors="coerce")

def normalize_percents_to_100(p: pd.Series) -> pd.Series:
    p = _safe_num(p).fillna(0.0)
    s = p.sum()
    if s <= 0: return p
    return (p / s) * 100.0

def sync_percent_amount(df: pd.DataFrame, total: float, mode: str) -> pd.DataFrame:
    df=df.copy()
    df["Ticker"]=df["Ticker"].astype(str).str.strip()
    df=df[df["Ticker"].astype(bool)].reset_index(drop=True)
    n=len(df)
    if n==0:
        df["weight"]=[]
        return df

    df["Percent (%)"]=_safe_num(df.get("Percent (%)"))
    df["Amount"]=_safe_num(df.get("Amount"))
    has_total = (total is not None and total>0)

    if has_total:
        if mode=="percent":
            if df["Percent (%)"].fillna(0).sum()==0:
                df["Percent (%)"]=100.0/n
            df["Percent (%)"] = normalize_percents_to_100(df["Percent (%)"]).round(2)
            df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
        else:
            s=df["Amount"].fillna(0).sum()
            if s>0:
                df["Percent (%)"]= (df["Amount"]/total*100.0).round(2)
                df["Percent (%)"]= normalize_percents_to_100(df["Percent (%)"]).round(2)
                df["Amount"]= (df["Percent (%)"]/100.0*total).round(2)
            else:
                df["Percent (%)"]=100.0/n
                df["Amount"]= (df["Percent (%)"]/100.0*total).round(2)
    else:
        if df["Percent (%)"].fillna(0).sum()==0:
            df["Percent (%)"]=100.0/n
        df["Percent (%)"]= normalize_percents_to_100(df["Percent (%)"]).round(2)

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
        st.session_state["grid_df"] = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT", "NVDA", "AMZN"],
            "Percent (%)": [25.0, 25.0, 25.0, 25.0],
            "Amount": [np.nan, np.nan, np.nan, np.nan],
        })

    st.markdown(
        f"**Holdings**  \n"
        f"<span class='small-muted'>Enter <b>Ticker</b> and either <b>Percent (%)</b> or "
        f"<b>Amount ({currency_symbol})</b>. Values update only when you click "
        f"<b>Apply changes</b>. Use <b>Normalize</b> to force exactly 100% in percent mode.</span>",
        unsafe_allow_html=True,
    )

    committed = st.session_state["grid_df"].copy()

    with st.form("holdings_form", clear_on_submit=False):
        sync_mode = st.segmented_control(
            "Sync mode",
            options=["Percent → Amount", "Amount → Percent"],
            default="Percent → Amount",
            help="Choose which side drives on Apply."
        )
        mode_key = {"Percent → Amount": "percent", "Amount → Percent": "amount"}[sync_mode]

        edited = st.data_editor(
            committed,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn(width="small"),
                "Percent (%)": st.column_config.NumberColumn(format="%.2f"),
                "Amount": st.column_config.NumberColumn(format="%.2f", help=f"Amount in {currency_symbol}"),
            },
            key="grid_form",
        )

        col_a, col_b = st.columns([1, 1])
        apply_btn = col_a.form_submit_button("Apply changes", type="primary", use_container_width=True)
        normalize_btn = col_b.form_submit_button("Normalize to 100% (percent mode)", use_container_width=True)

    if normalize_btn:
        syncd = edited.copy()
        syncd["Percent (%)"] = normalize_percents_to_100(_safe_num(syncd.get("Percent (%)")))
        if total_value and total_value>0:
            syncd["Amount"] = (syncd["Percent (%)"]/100.0*total_value).round(2)
        st.session_state["grid_df"] = syncd[["Ticker","Percent (%)","Amount"]]

    elif apply_btn:
        syncd = sync_percent_amount(edited.copy(), total_value, mode_key)
        st.session_state["grid_df"] = syncd[["Ticker","Percent (%)","Amount"]]

    current = st.session_state["grid_df"].copy()
    out = current.copy()
    out["ticker"] = out["Ticker"].map(yf_symbol)
    out = out[out["ticker"].astype(bool)]

    if total_value and total_value>0 and _safe_num(out["Amount"]).sum()>0:
        w = _safe_num(out["Amount"]) / _safe_num(out["Amount"]).sum()
    elif _safe_num(out["Percent (%)"]).sum()>0:
        w = _safe_num(out["Percent (%)"]) / _safe_num(out["Percent (%)"]).sum()
    else:
        n = max(len(out),1)
        w = pd.Series([1.0/n]*n, index=out.index)

    df_hold = pd.DataFrame({"ticker": out["ticker"], "weight": w})
    return df_hold, current


# ==========================  Company financial statements  ==========================

CURRENCY_SYMBOL: Dict[str, str] = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CAD": "C$", "AUD": "A$", "CHF": "CHF",
    "SEK": "kr", "NOK": "kr", "DKK": "kr", "HKD": "HK$", "CNY": "¥", "INR": "₹", "SGD": "S$"
}
_CURRENCY_SYMBOL = CURRENCY_SYMBOL  # backward-compat alias for pages that import this name

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    norm_index = {_norm(idx): idx for idx in df.index.astype(str)}
    # exact
    for c in candidates:
        key = _norm(c)
        if key in norm_index:
            return pd.to_numeric(df.loc[norm_index[key]], errors="coerce")
    # contains
    for c in candidates:
        key = _norm(c)
        for k, orig in norm_index.items():
            if key in k:
                return pd.to_numeric(df.loc[orig], errors="coerce")
    return None

def _clean_statement(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or isinstance(df, float): return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    out = df.copy()
    out = out[~out.index.duplicated(keep="first")]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    try:
        out.columns = pd.to_datetime(out.columns)
        out = out.sort_index(axis=1)
    except Exception:
        pass
    return out

# Canonical row orders (normalized) for pretty display
INCOME_ORDER = [
    "totalrevenue","revenue","sales",
    "costofrevenue","costofgoodsold","cogs",
    "grossprofit",
    "sellinggeneraladministrative","sga","researchdevelopment","rd",
    "depreciationamortization","reconciledepreciation","amortization",
    "otheroperatingexpenses",
    "operatingincome","operatingprofit","ebit",
    "interestincome","interestexpense","netinterestincome","otherincomeexpensenet",
    "pretaxincome","incomebeforetax","ebt",
    "incometaxexpense","taxexpense","taxrateforcalcs",
    "netincome","netincomefromcontinuing","netincometocommon","netincomeapplicabletocommonshares"
]

BALANCE_ORDER = [
    # Assets
    "cashandcashequivalents","shortterminvestments","restrictedcash",
    "netreceivables","inventory","othercurrentassets","totalcurrentassets",
    "propertyplantequipment","goodwill","intangibles","otherassets","totalassets",
    # Liabilities
    "accountspayable","othercurrentliab","shortlongtermdebt","currentportionoflongtermdebt",
    "totallongtermdebt","otherliabilities","totalcurrentliabilities","totalliab","totalliabilities",
    # Equity
    "retainedearnings","treasurystock","additionalpaidincapital","commonstock",
    "totalstockholderequity","totalequity"
]

CASHFLOW_ORDER = [
    # Operating
    "netincome","depreciation","depreciationamortization","reconciledepreciation",
    "deferredtaxes","stockbasedcompensation","changestoworkingcapital",
    "totalcashfromoperatingactivities","netcashprovidedbyoperatingactivities",
    # Investing
    "capitalexpenditures","investments","purchaseofinvestments","saleofinvestments",
    "acquisitionsnet","otherinvestingcashflows","totalcashflowsfrominvestingactivities",
    # Financing
    "dividendspaid","repurchaseofstock","issuanceofstock","netissuanceofdebt",
    "otherfinancingcashflows","totalcashfromfinancingactivities","totalcashflowsfromfinancingactivities",
    # Net change
    "effectofexchangeratechanges","changeincash","netchangeincash"
]

def _priority_index(order: List[str]) -> Dict[str, int]:
    return {k:i for i,k in enumerate(order)}

def _reorder_by_priority(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    if df is None or df.empty: return df
    prio = _priority_index(order)
    keys = []
    for idx in df.index.astype(str):
        n = _norm(idx)
        score = prio.get(n, 10_000)  # unknowns at the end
        keys.append((score, idx))
    keys.sort(key=lambda x: (x[0], str(x[1]).lower()))
    ordered_index = [idx for _, idx in keys]
    return df.loc[ordered_index]

def _detect_statement_kind(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "unknown"
    names = {_norm(i) for i in df.index.astype(str)}
    if {"totalrevenue","revenue","sales"} & names or "grossprofit" in names:
        return "income"
    if {"totalassets","totalliab","totalliabilities","totalstockholderequity"} & names:
        return "balance"
    if {"totalcashfromoperatingactivities","netcashprovidedbyoperatingactivities",
        "capitalexpenditures","totalcashflowsfrominvestingactivities"} & names:
        return "cashflow"
    return "unknown"

def reorder_income_statement(df: pd.DataFrame) -> pd.DataFrame:
    return _reorder_by_priority(df, INCOME_ORDER)

def reorder_balance_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return _reorder_by_priority(df, BALANCE_ORDER)

def reorder_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    return _reorder_by_priority(df, CASHFLOW_ORDER)

@st.cache_data(show_spinner=False)
def fetch_company_statements(ticker: str) -> Dict[str, object]:
    t = yf.Ticker(yf_symbol(ticker))
    try:
        info = t.info or {}
    except Exception:
        info = {}
    currency = info.get("financialCurrency") or info.get("currency") or "USD"

    def safe(attr):
        try:
            df = getattr(t, attr)
            return _clean_statement(df)
        except Exception:
            return pd.DataFrame()

    data = {
        "currency": currency,
        "income": safe("financials"),
        "balance": safe("balance_sheet"),
        "cashflow": safe("cashflow"),
        "income_q": safe("quarterly_financials"),
        "balance_q": safe("quarterly_balance_sheet"),
        "cashflow_q": safe("quarterly_cashflow"),
    }
    return data

def _scale_unit(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if df is None or df.empty: return df, ""
    vals = pd.to_numeric(df.replace([np.inf, -np.inf], np.nan).stack(), errors="coerce").dropna()
    if vals.empty: return df, ""
    m = np.nanmedian(np.abs(vals))
    if m >= 1e9:
        return (df / 1e9).round(2), " (billions)"
    if m >= 1e6:
        return (df / 1e6).round(2), " (millions)"
    if m >= 1e3:
        return (df / 1e3).round(2), " (thousands)"
    return df.round(2), ""

def tidy_statement_for_display(
    df: pd.DataFrame,
    take: int = 4,
    newest_first: bool = True,
    kind: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a nicely ordered statement:
      - Auto-detects statement type (income/balance/cashflow) if not provided.
      - Reorders rows per canonical order (so income begins with revenue and ends with net income).
      - Keeps the last `take` periods.
      - Displays newest period on the LEFT by default.
      - Column headers as YYYY-MM-DD (no 'bound method' text).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    k = (kind or _detect_statement_kind(df)).lower()
    if k == "income":
        work = reorder_income_statement(df)
    elif k == "balance":
        work = reorder_balance_sheet(df)
    elif k == "cashflow":
        work = reorder_cashflow(df)
    else:
        work = df.copy()

    try:
        work.columns = pd.to_datetime(work.columns)
        work = work.sort_index(axis=1)  # old -> new
    except Exception:
        pass

    work = work.iloc[:, -take:]  # keep last N

    if newest_first:
        work = work[work.columns[::-1]]  # newest left

    # Pretty column names
    pretty_cols = []
    for c in work.columns:
        try:
            pretty_cols.append(pd.to_datetime(c).date().strftime("%Y-%m-%d"))
        except Exception:
            pretty_cols.append(str(c))
    work.columns = pretty_cols

    # Numeric cleanup
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce").round(2)

    return work

def statement_metrics(stmts: Dict[str, object]) -> Dict[str, float]:
    inc = stmts.get("income", pd.DataFrame())
    bs  = stmts.get("balance", pd.DataFrame())
    cf  = stmts.get("cashflow", pd.DataFrame())

    # helpers
    rev   = _pick(inc, ["Total Revenue","Revenue","Sales"])
    gross = _pick(inc, ["Gross Profit"])
    opinc = _pick(inc, ["Operating Income","Operating Profit","Ebit"])
    net   = _pick(inc, ["Net Income","Net Income Applicable To Common Shares","Net Income Common Stockholders"])
    intr  = _pick(inc, ["Interest Expense","Interest Expense Non Operating"])

    ta    = _pick(bs,  ["Total Assets"])
    teq   = _pick(bs,  ["Total Stockholder Equity","Total Equity","Total Shareholder Equity"])
    tliab = _pick(bs,  ["Total Liab","Total Liabilities"])
    ca    = _pick(bs,  ["Total Current Assets"])
    cl    = _pick(bs,  ["Total Current Liabilities"])
    inv   = _pick(bs,  ["Inventory"])
    ltd   = _pick(bs,  ["Long Term Debt"])
    std   = _pick(bs,  ["Short Long Term Debt","Short-Term Debt","Current Portion of Long Term Debt"])
    notes = _pick(bs,  ["Notes Payable"])

    ocf   = _pick(cf,  ["Total Cash From Operating Activities","Net Cash Provided By Operating Activities"])
    capex = _pick(cf,  ["Capital Expenditures","Investments In Property Plant And Equipment"])

    def last(series): 
        return (series.dropna().iloc[-1] if series is not None and series.dropna().size else np.nan)
    def prev(series):
        return (series.dropna().iloc[-2] if series is not None and series.dropna().size>1 else np.nan)

    revenue      = float(last(rev))
    revenue_prev = float(prev(rev))
    gross_profit = float(last(gross))
    operating_inc= float(last(opinc))
    net_income   = float(last(net))
    interest_exp = abs(float(last(intr))) if not np.isnan(last(intr)) else np.nan

    total_assets = float(last(ta))
    equity       = float(last(teq)) if teq is not None else np.nan
    if np.isnan(equity):
        equity = float(last(ta)) - float(last(tliab)) if not (np.isnan(last(ta)) or np.isnan(last(tliab))) else np.nan

    current_assets     = float(last(ca))
    current_liabilities= float(last(cl))
    inventory          = float(last(inv))
    lt_debt            = float(last(ltd))
    st_debt            = float(last(std))
    notes_pay          = float(last(notes))
    total_debt         = np.nansum([lt_debt, st_debt, notes_pay])

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
            v0 = float(rev.dropna().iloc[-4])
            v1 = float(rev.dropna().iloc[-1])
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
    out = []
    y = m.get("revenue_yoy"); c = m.get("revenue_cagr3y")
    if not np.isnan(y):
        if y > 0.1: out.append(f"**Top-line growth:** Revenue grew ~{y*100:.1f}% YoY (strong momentum).")
        elif y > 0.0: out.append(f"**Top-line growth:** Revenue grew ~{y*100:.1f}% YoY (modest).")
        else: out.append(f"**Top-line growth:** Revenue declined ~{abs(y)*100:.1f}% YoY (headwind).")
    if not np.isnan(c):
        if c > 0.1: out.append(f"**Multi-year growth:** ~{c*100:.1f}% CAGR over ~3 years.")
        elif c > 0.0: out.append(f"**Multi-year growth:** ~{c*100:.1f}% CAGR (slow but positive).")
        else: out.append("**Multi-year growth:** roughly flat/negative over ~3 years.")

    gm, om, nm = m.get("gross_margin"), m.get("operating_margin"), m.get("net_margin")
    if not np.isnan(gm):
        qual = "high" if gm >= 0.5 else "mid" if gm >= 0.3 else "low"
        out.append(f"**Gross margin:** {gm*100:.1f}% ({qual} structural margin).")
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
        if not np.isnan(delta):
            if abs(delta) >= 0.02:
                out.append(f"**{label.title()} margin trend:** {('expanding' if delta>0 else 'contracting')} ~{abs(delta)*100:.1f} pp YoY.")

    fcfm = m.get("fcf_margin")
    if not np.isnan(fcfm):
        if fcfm >= 0.10: txt="strong free-cash-flow generation"
        elif fcfm >= 0.0: txt="modest free-cash-flow generation"
        else: txt="negative free cash flow (investment/pressure)"
        out.append(f"**FCF margin:** {fcfm*100:.1f}% — {txt}.")

    cr, qr = m.get("current_ratio"), m.get("quick_ratio")
    if not np.isnan(cr):
        if cr >= 2.0: s="strong"
        elif cr >= 1.0: s="adequate"
        else: s="tight"
        out.append(f"**Liquidity:** current ratio {cr:.2f} ({s}).")
    if not np.isnan(qr):
        out.append(f"**Quick ratio:** {qr:.2f}.")

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

    for k, nm in [("roe","ROE"),("roa","ROA")]:
        v = m.get(k)
        if not np.isnan(v):
            out.append(f"**{nm}:** {v*100:.1f}%.")

    return out