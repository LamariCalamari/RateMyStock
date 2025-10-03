# app_utils.py — shared utilities for Rate My (Stock + Portfolio + Tracker)
# Uses static index lists from indices.py if available; otherwise falls back to local snapshot/web fallbacks.
# Peer loader is strengthened to improve coverage (singles → bulk → Ticker.history).

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

# ==========================  Index lists: from indices.py if present ==========================

# If you paste full arrays in indices.py, we’ll use them. Otherwise we fall back to snapshot/fallbacks.
try:
    from indices import SP500_LIST as _SP500, NASDAQ100_LIST as _NDX, DOW30_LIST as _DOW
except Exception:
    _SP500, _NDX, _DOW = [], [], []

def _normalize_list(tickers: Iterable[str]) -> List[str]:
    cleaned = []
    seen = set()
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
    # 1) indices.py wins if provided
    if _SP500 or _NDX or _DOW:
        return {
            "S&P 500": _normalize_list(_SP500),
            "NASDAQ 100": _normalize_list(_NDX),
            "Dow 30": _normalize_list(_DOW),
        }

    # 2) else try local snapshot
    if os.path.exists(_SNAPSHOT_FILE):
        try:
            with open(_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for k in obj:
                obj[k] = _normalize_list(obj[k])
            return obj
        except Exception:
            pass

    # 3) else try a one-time fetch (Wikipedia) -> snapshot.
    try:
        sp500 = _fetch_sp500_from_web()
        ndx   = _fetch_nasdaq100_from_web()
        dow   = _fetch_dow30_from_web()
        snap = {"S&P 500": sp500, "NASDAQ 100": ndx, "Dow 30": dow}
        try:
            with open(_SNAPSHOT_FILE, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return snap
    except Exception:
        # 4) emergency minimal fallback
        return {
            "S&P 500": ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","BRK-B","LLY","JPM","V","AVGO","TSLA"],
            "NASDAQ 100": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","NFLX","AMD"],
            "Dow 30": ["AAPL","MSFT","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","GS","HD","HON","IBM","INTC",
                       "JNJ","JPM","KO","MCD","MMM","MRK","MS","NKE","PG","TRV","UNH","V","VZ","WMT","DOW"],
        }

# Optional web fetchers (used only if no indices.py and no snapshot)
def _fetch_sp500_from_web() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, flavor="lxml")
    df = tables[0]
    return _normalize_list(df["Symbol"].tolist())

def _fetch_nasdaq100_from_web() -> List[str]:
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

PEER_CATALOG: Dict[str, List[str]] = _load_or_snapshot_peer_lists()

def set_peer_catalog(sp500: List[str] | None = None,
                     ndx: List[str] | None = None,
                     dow: List[str] | None = None) -> None:
    """Optional runtime override."""
    if sp500 is not None: PEER_CATALOG["S&P 500"] = _normalize_list(sp500)
    if ndx is not None:   PEER_CATALOG["NASDAQ 100"] = _normalize_list(ndx)
    if dow is not None:   PEER_CATALOG["Dow 30"] = _normalize_list(dow)

# ==========================  Data fetchers (stronger)  ==========================

@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(
    tickers: Iterable[str],
    period: str = "1y",
    interval: str = "1d",
    chunk: int = 20,          # slightly smaller bulks; more reliable
    retries: int = 4,         # extra tries
    sleep_between: float = 1.0,
    singles_pause: float = 1.1,
    hard_limit: int = 700,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    3-pass loader to maximize coverage:
      1) Singles (most reliable)
      2) Bulk for stragglers (faster)
      3) Ticker.history() fallback for any remaining
    """
    names = [yf_symbol(t) for t in tickers if t]
    names = list(dict.fromkeys(names))[:hard_limit]
    if not names:
        return pd.DataFrame(), []

    frames: List[pd.Series] = []
    ok: List[str] = []

    # -------- Pass 1: singles with retries --------
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

    # -------- Pass 2: bulk groups --------
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
        # recompute missing after pass 2
        ok_set = set(ok)
        missing = [t for t in names if t not in ok_set]

    # -------- Pass 3: Ticker.history() fallback --------
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
    Build a peer universe from indices.py lists (or snapshot/fallbacks).
    - Always excludes user tickers from peers.
    - Custom mode uses comma-separated list.
    - Index modes start from full list then sample down (deterministic stride) to sample_n.
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
        # Auto by membership → else default to S&P 500
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

# ==========================  Portfolio editor bits (unchanged APIs)  ==========================

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