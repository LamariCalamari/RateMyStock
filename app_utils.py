# app_utils.py — shared styles + ALL helpers (full rewrite, with resilient peer loader)

import io
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =====================================================================================
# Global CSS + Sidebar "app" → "Home" script + brand header + topbar
# =====================================================================================

def inject_css_and_script():
    """Sitewide CSS and a robust script that renames the first sidebar item to 'Home'."""
    st.markdown(
        """
        <style>
        .block-container{max-width:1140px;}

        /* Brand header (logo + gradient wordmark) */
        .brand{ display:flex; align-items:center; justify-content:center; gap:16px; margin:1.0rem 0 .25rem; }
        .logo{ width:56px; height:52px; flex:0 0 auto; }

        /* KPI + misc (original styles) */
        .kpi-card{padding:1rem 1.1rem;border-radius:12px;background:#111418;border:1px solid #222}
        .kpi-num{font-size:2.2rem;font-weight:800;margin-top:.25rem}
        .small-muted{color:#9aa0a6;font-size:.9rem}
        .banner{background:#0c2f22;color:#cdebdc;border-radius:10px;padding:.9rem 1.1rem;margin:.75rem 0 1.25rem}
        .chart-caption{color:#9aa0a6;margin:-.5rem 0 1rem}
        .topbar{display:flex;justify-content:flex-end;margin:.2rem 0 .6rem}

        /* ---- Home CTA Buttons (style st.page_link directly in main area) ---- */
        [data-testid="stMain"] a[data-testid="stPageLink"]{
          display:flex; align-items:center; justify-content:center; gap:.6rem;
          padding:16px 22px; border-radius:14px; text-decoration:none; font-weight:800;
          min-width:260px; white-space:nowrap;
          background:linear-gradient(90deg,#e85d58, #f39c12, #2ecc71) padding-box,
                     linear-gradient(90deg,rgba(255,255,255,.12),rgba(255,255,255,.04)) border-box;
          border:1px solid transparent; color:#0d0f12;
          box-shadow:0 1px 0 rgba(255,255,255,.06) inset, 0 12px 26px rgba(0,0,0,.42);
          transition:transform .08s ease, box-shadow .16s ease, filter .12s ease;
        }
        [data-testid="stMain"] a[data-testid="stPageLink"]:hover{
          transform:translateY(-1px);
          filter:saturate(1.06) brightness(1.05);
          box-shadow:0 1px 0 rgba(255,255,255,.08) inset, 0 16px 34px rgba(0,0,0,.50);
        }
        [data-testid="stMain"] a[data-testid="stPageLink"] p{
          margin:0 !important; color:inherit !important; font-weight:800 !important;
        }

        /* Do not affect topbar back link */
        .topbar a[data-testid="stPageLink"]{ all:unset !important; cursor:pointer; color:inherit; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Robustly rename the first sidebar entry to "Home" on every re-render
    st.markdown(
        """
        <script>
        (function(){
          function renameFirstSidebarItem(){
            try{
              const nav = document.querySelector('[data-testid="stSidebarNav"]');
              if(!nav) return;
              const firstLabel = nav.querySelector('ul li:first-child a p');
              if(firstLabel && firstLabel.textContent.trim().toLowerCase() === 'app'){
                firstLabel.textContent = 'Home';
              }
            }catch(e){}
          }
          const obs = new MutationObserver(renameFirstSidebarItem);
          obs.observe(document.body, {childList:true, subtree:true});
          renameFirstSidebarItem();
        })();
        </script>
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
    gradient = ("background:linear-gradient(90deg,#e74c3c 0%,#f39c12 50%,#2ecc71 100%);"
                "-webkit-background-clip:text;background-clip:text;color:transparent;")
    html = (
        f'<div class="brand">{inline_logo_svg()}'
        f'<h1 style="font-size:56px;margin:0;line-height:1;font-weight:900;letter-spacing:.3px;{gradient}">{title}</h1>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def topbar_back(label: str = "← Back", url: Optional[str] = None):
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if url:
        st.page_link(url, label=label)
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================================
# Core finance helpers
# =====================================================================================

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

# =====================================================================================
# Robust yfinance fetchers (chunked + retries, slower pacing + fallback period)
# =====================================================================================

@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
    chunk: int = 25,              # kept for compatibility (we use tuned sizes below)
    retries: int = 3,             # kept for compatibility (we do stronger retries)
    sleep_between: float = 0.75,  # slower group pacing (helps Yahoo reliability)
    singles_pause: float = 0.60,  # slower single pacing
    hard_limit: int = 350,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Resilient multi-stage loader with fallback period:

      Stage A: Medium group downloads (CHUNK1=18) using primary period (default 1y)
      Stage A-fallback: Same groups for misses using fallback period (6mo)
      Stage B: Small group downloads (CHUNK2=6) for remaining misses (primary then fallback)
      Stage C: Per-ticker retries with exponential backoff (primary, then fallback)

    Returns
    -------
    (prices_df, ok_tickers)
    """
    FALLBACK_PERIOD = "6mo"

    # ---------- normalize universe ----------
    tickers = [yf_symbol(t) for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))[:hard_limit]
    if not tickers:
        return pd.DataFrame(), []

    frames: List[pd.Series] = []
    ok: List[str] = []

    # ---------- helpers ----------
    def _append_from_multi(df: pd.DataFrame, names: List[str]):
        if df is None or df.empty:
            return
        if isinstance(df.columns, pd.MultiIndex):
            got = set(df.columns.get_level_values(0))
            for t in names:
                if t in got and ("Close" in df[t]):
                    s = df[t]["Close"].dropna()
                    if s.size:
                        frames.append(s.rename(t)); ok.append(t)
        else:
            t = names[0] if names else None
            if t and ("Close" in df):
                s = df["Close"].dropna()
                if s.size:
                    frames.append(s.rename(t)); ok.append(t)

    def _group_try(names: List[str], use_period: str, pause: float):
        try:
            df = yf.download(
                names,
                period=use_period,
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                threads=False,       # more reliable
                progress=False,
            )
            _append_from_multi(df, names)
        except Exception:
            pass
        time.sleep(pause + random.uniform(0.05, 0.15))

    def _single_try(t: str, use_period: str) -> bool:
        try:
            df = yf.download(
                t,
                period=use_period,
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                threads=False,
                progress=False,
            )
            if isinstance(df, pd.DataFrame) and ("Close" in df):
                s = df["Close"].dropna()
                if s.size:
                    frames.append(s.rename(t)); ok.append(t)
                    return True
        except Exception:
            pass
        return False

    # ---------- Stage A: medium chunks on primary period ----------
    CHUNK1 = 18
    for i in range(0, len(tickers), CHUNK1):
        group = tickers[i:i+CHUNK1]
        _group_try(group, period, sleep_between)

    # ---------- Stage A-fallback: same groups on FALLBACK_PERIOD ----------
    ok_set = set(ok)
    missing = [t for t in tickers if t not in ok_set]
    if missing:
        for i in range(0, len(missing), CHUNK1):
            group = missing[i:i+CHUNK1]
            _group_try(group, FALLBACK_PERIOD, sleep_between * 1.1)

    # ---------- Stage B: small chunks (primary, then fallback) ----------
    ok_set = set(ok)
    missing = [t for t in tickers if t not in ok_set]
    CHUNK2 = 6
    if missing:
        for i in range(0, len(missing), CHUNK2):
            group = missing[i:i+CHUNK2]
            _group_try(group, period, sleep_between * 1.15)

    ok_set = set(ok)
    missing = [t for t in tickers if t not in ok_set]
    if missing:
        for i in range(0, len(missing), CHUNK2):
            group = missing[i:i+CHUNK2]
            _group_try(group, FALLBACK_PERIOD, sleep_between * 1.25)

    # ---------- Stage C: per-ticker with exponential backoff ----------
    ok_set = set(ok)
    missing = [t for t in tickers if t not in ok_set]
    if missing:
        MAX_ROUNDS = 6
        base = max(singles_pause, 0.5)   # start slower
        for _ in range(MAX_ROUNDS):
            if not missing:
                break
            new_missing = []
            for t in missing:
                # try primary first, then fallback for this ticker
                got = _single_try(t, period) or _single_try(t, FALLBACK_PERIOD)
                time.sleep(base * (1.0 + 0.3*random.random()))  # jittered backoff
                if not got:
                    new_missing.append(t)
            base *= 1.7
            missing = new_missing

    # ---------- Build final frame ----------
    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()

    ok_unique = [t for t in tickers if (not prices.empty and t in prices.columns)]
    prices = prices.reindex(columns=ok_unique) if not prices.empty else prices
    return prices, ok_unique


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
    keep = [
        "revenueGrowth","earningsGrowth","returnOnEquity",
        "profitMargins","grossMargins","operatingMargins","ebitdaMargins",
        "trailingPE","forwardPE","debtToEquity",
    ]
    rows = []
    for raw in tickers:
        t = yf_symbol(raw)
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        row = {"ticker": t}
        for k in keep:
            try:
                row[k] = float(info.get(k, np.nan))
            except Exception:
                row[k] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")

# =====================================================================================
# Peer universes
# =====================================================================================

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

def build_universe(
    user_tickers: List[str], mode: str, sample_n: int = 150, custom_raw: str = ""
) -> Tuple[List[str], str]:
    user = [yf_symbol(t) for t in user_tickers]
    if mode == "S&P 500":
        peers_all = list_sp500(); label = "S&P 500"
    elif mode == "Dow 30":
        peers_all = list_dow30(); label = "Dow 30"
    elif mode == "NASDAQ 100":
        peers_all = list_nasdaq100(); label = "NASDAQ 100"
    elif mode == "Custom (paste list)":
        custom = {yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
        return sorted(set(user) | custom)[:350], "Custom"
    else:
        sp, dj, nd = list_sp500(), list_dow30(), list_nasdaq100()
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
    return sorted(set(user) | set(peers))[:350], label

# =====================================================================================
# Feature builders + interpretations
# =====================================================================================

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
        if pos_good:  return "bullish" if v>=0.5 else "watch" if v<=-0.5 else "neutral"
        else:         return "bullish (cheap)" if v>=0.5 else "watch (expensive)" if v<=-0.5 else "neutral"
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

# =====================================================================================
# Reusable stock charts
# =====================================================================================

def draw_stock_charts(ticker: str, series: pd.Series):
    if series is None or series.empty:
        st.info("Not enough history to show charts.")
        return
    st.subheader("📈 Price & EMAs")
    e20, e50 = ema(series, 20), ema(series, 50)
    price_df = pd.DataFrame({"Close": series, "EMA20": e20, "EMA50": e50})
    st.line_chart(price_df, use_container_width=True)
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
    else:
        st.info("Need > 1 year of data to show the 12-month momentum line.")

# =====================================================================================
# Portfolio helpers (normalize/sync) + submit-based grid editor
# =====================================================================================

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

def holdings_editor_form(currency_symbol: str, total_value: Optional[float]):
    """Submit-based editable grid (Apply / Normalize)."""
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
        syncd = sync_percent_amount(edited.copy(), total_value or 0.0, mode_key)
        st.session_state["grid_df"] = syncd[["Ticker","Percent (%)","Amount"]]

    current = st.session_state["grid_df"].copy()
    view = current.copy()
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
    return df_hold, view
