# app.py
# ======================================================================
# Rate My (Stock | Portfolio)
# Full Streamlit app with centered title + inline logo, rich narratives,
# robust peer-universe options (built-ins, upload, paste, URL),
# 12-month momentum from 5y data, Macro KPI (VIX level+trend),
# charts with captions, download breakdown, and portfolio "Apply changes".
# ======================================================================

import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# ----- App Configuration -----
# -----------------------------
st.set_page_config(
    page_title="Rate My",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# -------- Utilities ----------
# -----------------------------
@st.cache_data(show_spinner=False)
def _get(url, timeout=12):
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r
    except Exception:
        pass
    return None

def _z_to_pct(z):
    if pd.isna(z):
        return np.nan
    return float(100.0 * 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2))))

def clamp01(x):
    if pd.isna(x):
        return np.nan
    return max(0.0, min(1.0, float(x)))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def nanmean_safe(vals):
    vals = [v for v in vals if v is not None and not pd.isna(v)]
    return np.nan if not vals else float(np.mean(vals))

# -----------------------------
# ------- Inline Logo ---------
# -----------------------------
def triangle_logo_svg(size=64):
    # Smooth gradient triangle (green top -> orange mid -> red bottom)
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle">
  <defs>
    <linearGradient id="grad" x1="0" y1="100" x2="0" y2="0">
      <stop offset="0%"  stop-color="#d7191c"/>
      <stop offset="50%" stop-color="#fdae61"/>
      <stop offset="100%" stop-color="#1a9641"/>
    </linearGradient>
  </defs>
  <polygon points="50,6 96,96 4,96" fill="url(#grad)" stroke="#e5e7eb" stroke-width="1"/>
</svg>
"""

# -----------------------------
# ---------- Styling ----------
# -----------------------------
st.markdown(
    """
<style>
.block-container { max-width: 1200px; }

/* Header */
.header-shell { display:flex; justify-content:center; margin:.8rem 0 1.2rem; }
.header-inner { display:flex; align-items:center; gap:14px; }
.header-title { font-size:3.6rem; font-weight:850; line-height:1.06; margin:0; letter-spacing:.25px; }
.header-sub   { text-align:center; color:#9aa0a6; margin-top:.35rem; font-size:1.06rem; }

/* Landing CTA */
.landing-cta { display:flex; align-items:center; justify-content:center; gap:20px; margin-top:18px; }

/* KPI cards */
.kpi { padding:1rem 1.1rem; border-radius:12px; background:#111418; border:1px solid #222; }
.kpi-big { font-size:2.2rem; font-weight:800; margin-top:.25rem; }
.small { color:#9aa0a6; font-size:.92rem; }

/* Banner */
.banner { background:#0c2f22; color:#cdebdc; border-radius:10px; padding:.9rem 1.1rem; margin:.75rem 0 1.25rem; }

/* Inputs */
.center-input .stTextInput>div>div>input { text-align:center; font-size:1.05rem; }

/* Tables */
.table-mini td, .table-mini th { padding:.4rem .6rem !important; font-size:.92rem !important; }

/* Captions */
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
            <div>{triangle_logo_svg(64)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f"<div class='header-sub'>{subtitle}</div>", unsafe_allow_html=True)

# -----------------------------
# ------ Peer universes -------
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_sp500():
    # Wikipedia primary
    r = _get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if r:
        try:
            tables = pd.read_html(r.text, displayed_only=False)
            # try to find table with 'Symbol'
            df = None
            for t in tables:
                if "Symbol" in t.columns:
                    df = t; break
            if df is None:
                df = tables[0]
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            syms = df[col].astype(str).str.upper().str.replace(".", "-", regex=False)
            vals = [s for s in syms if s and s.isascii()]
            if len(vals) >= 200:
                return sorted(list(set(vals)))
        except Exception:
            pass
    # Fallback dataset
    r2 = _get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv")
    if r2:
        try:
            df = pd.read_csv(io.StringIO(r2.text))
            syms = df.iloc[:,0].astype(str).str.upper().str.replace(".", "-", regex=False)
            return sorted(list(set([s for s in syms if s and s.isascii()])))
        except Exception:
            pass
    return []

@st.cache_data(show_spinner=False)
def fetch_nasdaq100():
    r = _get("https://en.wikipedia.org/wiki/Nasdaq-100")
    if r:
        try:
            tables = pd.read_html(r.text, displayed_only=False)
            for df in tables:
                cols = [c.lower() for c in df.columns]
                if any(("ticker" in c) or ("symbol" in c) for c in cols):
                    col = next(c for c in df.columns if ("ticker" in c.lower() or "symbol" in c.lower()))
                    syms = df[col].astype(str).str.upper().str.replace(".", "-", regex=False)
                    vals = [s for s in syms if s and s.isascii()]
                    if len(vals) >= 80:
                        return sorted(list(set(vals)))
        except Exception:
            pass
    # Fallback community CSV
    r2 = _get("https://raw.githubusercontent.com/nikbearbrown/Financial-Data-Science/main/data/NASDAQ_100_Company_List.csv")
    if r2:
        try:
            df = pd.read_csv(io.StringIO(r2.text))
            col = next(c for c in df.columns if "symbol" in c.lower())
            syms = df[col].astype(str).str.upper().str.replace(".", "-", regex=False)
            return sorted(list(set([s for s in syms if s and s.isascii()])))
        except Exception:
            pass
    return []

@st.cache_data(show_spinner=False)
def fetch_dow30():
    r = _get("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
    if r:
        try:
            tables = pd.read_html(r.text, displayed_only=False)
            for df in tables:
                cols = [c.lower() for c in df.columns]
                if any(("symbol" in c) or ("ticker" in c) for c in cols):
                    col = next(c for c in df.columns if ("symbol" in c.lower() or "ticker" in c.lower()))
                    syms = df[col].astype(str).str.upper().str.replace(".", "-", regex=False)
                    vals = [s for s in syms if s and s.isascii()]
                    if len(vals) >= 25:
                        return sorted(list(set(vals)))
        except Exception:
            pass
    return []

BUILT_INS = {
    "S&P 500": fetch_sp500,
    "NASDAQ-100": fetch_nasdaq100,
    "Dow 30": fetch_dow30,
}

# Custom peers
@st.cache_data(show_spinner=False)
def _clean_tickers(raw):
    out = []
    for t in raw:
        s = str(t).upper().strip().replace(".", "-")
        if s and s.isascii():
            out.append(s)
    return sorted(list(dict.fromkeys(out)))

@st.cache_data(show_spinner=False)
def parse_peer_file(upload):
    if upload is None:
        return []
    try:
        if upload.name.lower().endswith(".csv"):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload, engine="openpyxl")
        cols = {c.lower(): c for c in df.columns}
        key = cols.get("ticker") or cols.get("symbol") or df.columns[0]
        return _clean_tickers(df[key].dropna().tolist())
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def parse_peer_url(url):
    if not url:
        return []
    r = _get(url)
    if not r:
        return []
    txt = r.text
    try:
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(txt), sep="\t")
        except Exception:
            return []
    cols = {c.lower(): c for c in df.columns}
    key = cols.get("ticker") or cols.get("symbol") or df.columns[0]
    return _clean_tickers(df[key].dropna().tolist())

# -----------------------------
# ----- Market data fetch -----
# -----------------------------
@st.cache_data(show_spinner=False)
def load_history(tickers, start, end):
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(
            tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = df["Close"] if isinstance(df, pd.DataFrame) and "Close" in df else df
        if isinstance(close, pd.Series):
            close = close.to_frame()
        return close.dropna(how="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def vix_series(days=500):
    if yf is None:
        return pd.Series(dtype=float)
    try:
        v = yf.download("^VIX", period=f"{days}d", interval="1d", progress=False)
        return v["Close"].dropna() if isinstance(v, pd.DataFrame) and "Close" in v else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def load_fundamentals(tickers, enrich_cap=140):
    """Fetch fundamentals with light footprint; enrich a subset with .info."""
    out = {}
    if yf is None:
        return pd.DataFrame()
    # fast_info pass first
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            out[t] = {
                "forwardPE": float(getattr(fi, "forward_pe", np.nan)),
                "trailingPE": float(getattr(fi, "trailing_pe", np.nan)),
            }
        except Exception:
            out[t] = {"forwardPE": np.nan, "trailingPE": np.nan}
    # enrich part of the universe with .info (margins, growth, ROE, D/E, sector, name)
    for i, t in enumerate(tickers):
        if i >= enrich_cap:
            break
        try:
            info = yf.Ticker(t).info
            out[t].update({
                "sector": info.get("sector"),
                "longName": info.get("longName") or info.get("shortName"),
                "profitMargins": info.get("profitMargins"),
                "operatingMargins": info.get("operatingMargins"),
                "grossMargins": info.get("grossMargins"),
                "ebitdaMargins": info.get("ebitdaMargins"),
                "debtToEquity": info.get("debtToEquity"),
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsGrowth"),
                "returnOnEquity": info.get("returnOnEquity"),
            })
        except Exception:
            pass
    return pd.DataFrame.from_dict(out, orient="index")

# -----------------------------
# ---------- Scoring ----------
# -----------------------------
def zscore(s, high_better=True):
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(skipna=True), s.std(skipna=True)
    if pd.isna(m) or pd.isna(sd) or sd == 0:
        z = pd.Series(np.nan, index=s.index)
    else:
        z = (s - m) / sd
    if not high_better:
        z = -z
    return z

def fundamentals_z(df):
    Z = pd.DataFrame(index=df.index)
    Z["profitMargins_z"]    = zscore(df.get("profitMargins"), True)
    Z["operatingMargins_z"] = zscore(df.get("operatingMargins"), True)
    Z["grossMargins_z"]     = zscore(df.get("grossMargins"), True)
    Z["ebitdaMargins_z"]    = zscore(df.get("ebitdaMargins"), True)
    Z["returnOnEquity_z"]   = zscore(df.get("returnOnEquity"), True)
    Z["revenueGrowth_z"]    = zscore(df.get("revenueGrowth"), True)
    Z["earningsGrowth_z"]   = zscore(df.get("earningsGrowth"), True)
    Z["forwardPE_z"]        = zscore(df.get("forwardPE"), False)   # lower better
    Z["trailingPE_z"]       = zscore(df.get("trailingPE"), False)  # lower better
    Z["debtToEquity_z"]     = zscore(-df.get("debtToEquity"), True)  # lower D/E → better
    # Composite (equal buckets: growth, quality, value, balance)
    growth   = Z[["revenueGrowth_z","earningsGrowth_z"]].mean(axis=1, skipna=True)
    quality  = Z[["returnOnEquity_z","profitMargins_z","operatingMargins_z","grossMargins_z","ebitdaMargins_z"]].mean(axis=1, skipna=True)
    value    = Z[["forwardPE_z","trailingPE_z"]].mean(axis=1, skipna=True)
    balance  = Z[["debtToEquity_z"]].mean(axis=1, skipna=True)
    Z["FUND_score"] = pd.concat([growth, quality, value, balance], axis=1).mean(axis=1, skipna=True)
    return Z

def compute_technicals(prices, main):
    """Return raw indicators + z-scores + TECH_score; also return the 2y price panel."""
    if prices.empty:
        return pd.DataFrame(), prices
    # 5y panel only for main to compute 12m mom robustly
    px5 = load_history([main], start=(datetime.utcnow().date()-timedelta(days=int(365*5))), end=datetime.utcnow().date())
    rows = {}
    for t in prices.columns:
        s = prices[t].dropna()
        gap = macd = rsi_s = mom12 = np.nan
        if len(s) >= 60:
            ema50 = ema(s, 50)
            if ema50.iloc[-1] != 0:
                gap = (s.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]
            ema12 = ema(s, 12); ema26 = ema(s, 26)
            macd_line = ema12 - ema26
            macd_sig  = ema(macd_line, 9)
            macd = (macd_line - macd_sig).iloc[-1]
            d = s.diff()
            up = d.clip(lower=0).rolling(14).mean()
            dn = (-d.clip(upper=0)).rolling(14).mean()
            rs = up / (dn + 1e-9)
            rsi = 100 - (100/(1+rs))
            rsi_s = (rsi.iloc[-1] - 50)/50.0
        # 12m momentum from 5y for main ticker only (robustness)
        if t == main and (px5 is not None) and (main in px5.columns):
            s5 = px5[main].dropna()
            if len(s5) >= 252 and s5.iloc[-252] != 0:
                mom12 = s5.iloc[-1] / s5.iloc[-252] - 1.0
        rows[t] = {"dma_gap": gap, "macd_hist": macd, "rsi_strength": rsi_s, "mom12m": mom12}
    DF = pd.DataFrame(rows).T
    Z = pd.DataFrame({
        "dma_gap_z": zscore(DF["dma_gap"], True),
        "macd_hist_z": zscore(DF["macd_hist"], True),
        "rsi_strength_z": zscore(DF["rsi_strength"], True),
        "mom12m_z": zscore(DF["mom12m"], True),
    })
    Z["TECH_score"] = Z.mean(axis=1, skipna=True)
    return DF.join(Z), prices

def macro_vix():
    s = vix_series(500)
    if s.empty:
        return {"macro": np.nan, "vix": np.nan, "ema20": np.nan, "gap": np.nan}
    last = s.iloc[-1]
    e20  = ema(s, 20).iloc[-1]
    gap  = (last - e20)/e20 if e20 else 0.0
    # level map: 12→1, 28→0
    level = clamp01(1 - (last-12)/(28-12))
    # trend map: below EMA → safer
    trend = clamp01(1 - (gap+0.1)/0.3)
    m = 0.6*level + 0.4*trend
    return {"macro": m, "vix": last, "ema20": e20, "gap": gap}

def composite_score(f, t, m):
    # Convert macro 0..1 to z-ish around 0, then average with FUND/TECH z’s and map to 0..100
    z = np.nanmean([f, t, (m-0.5)/0.25])
    z = float(np.clip(z, -2.5, 2.5))
    return z, _z_to_pct(z)

# -----------------------------
# ------- Narratives ----------
# -----------------------------
def label_pct(z):
    return f"{_z_to_pct(z):.0f}th pct" if pd.notna(z) else "n/a"

def bucket(z):
    if pd.isna(z):      return "insufficient data"
    if z >= 1.2:        return "strongly above peers (bullish)"
    if z >= 0.6:        return "above peers (constructive)"
    if z <= -1.0:       return "well below peers (bearish)"
    if z <= -0.5:       return "below peers (caution)"
    return "around peer median (neutral)"

def fundamentals_story(zrow: pd.Series):
    growth  = nanmean_safe([zrow.get("revenueGrowth_z"), zrow.get("earningsGrowth_z")])
    quality = nanmean_safe([zrow.get("returnOnEquity_z"), zrow.get("profitMargins_z"),
                            zrow.get("operatingMargins_z"), zrow.get("grossMargins_z"),
                            zrow.get("ebitdaMargins_z")])
    value   = nanmean_safe([zrow.get("forwardPE_z"), zrow.get("trailingPE_z")])   # higher z = cheaper (we inverted)
    lev     = zrow.get("debtToEquity_z")

    lines=[]
    lines.append("**Growth**")
    if pd.isna(growth):
        lines.append("- Not enough growth data to compare.")
    else:
        lines.append(f"- Revenue/EPS growth screens **{bucket(growth)}** (≈ {label_pct(growth)}).")

    lines.append("\n**Quality & margins**")
    if pd.isna(quality):
        lines.append("- Insufficient margin/ROE data.")
    else:
        parts=[]
        for k, nm in [("returnOnEquity_z","ROE"),("profitMargins_z","profit margin"),
                      ("operatingMargins_z","operating margin"),("grossMargins_z","gross margin")]:
            v = zrow.get(k); 
            if pd.notna(v): parts.append(f"{nm} {label_pct(v)}")
        tail = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"- Overall profitability is **{bucket(quality)}**{tail}.")

    lines.append("\n**Valuation (PE)**")
    if pd.isna(value):
        lines.append("- Valuation unavailable.")
    else:
        subs=[]
        if pd.notna(zrow.get("forwardPE_z")):  subs.append(f"forward PE {label_pct(zrow['forwardPE_z'])}")
        if pd.notna(zrow.get("trailingPE_z")): subs.append(f"trailing PE {label_pct(zrow['trailingPE_z'])}")
        lines.append(f"- Looks **{bucket(value)}** vs peers ({', '.join(subs) or 'n/a'}). (Higher z = cheaper.)")

    lines.append("\n**Balance sheet**")
    if pd.isna(lev):
        lines.append("- Debt/Equity not reported.")
    else:
        lev_tag = "conservative" if lev>=0.6 else "elevated" if lev<=-0.5 else "moderate"
        lines.append(f"- Leverage appears **{lev_tag}** (≈ {label_pct(lev)}).")

    watch=[]
    if pd.notna(growth) and growth<0:   watch.append("growth acceleration (rev/EPS)")
    if pd.notna(quality) and quality<0: watch.append("margin/ROE improvement")
    if pd.notna(value) and value<0:     watch.append("valuation reset vs peers (lower PE)")
    if pd.notna(lev) and lev<0:         watch.append("deleveraging / interest cover")
    lines.append("\n**What to watch:** " + (", ".join(watch) if watch else "sustain quality & growth; monitor valuation drift.") + ".")
    return "\n".join(lines)

def technicals_story(trow: pd.Series):
    lines=[]
    g = trow.get("dma_gap")
    if pd.notna(g):
        gp = 100*g
        if gp>6: lines.append(f"- Price **{gp:.1f}% above** EMA50 → **bullish trend**.")
        elif gp>2: lines.append(f"- Price **{gp:.1f}% above** EMA50 → constructive.")
        elif gp<-6: lines.append(f"- Price **{abs(gp):.1f}% below** EMA50 → **trend risk**.")
        else: lines.append("- Price near EMA50 → neutral trend.")
    mh = trow.get("macd_hist")
    if pd.notna(mh):
        lines.append("- MACD histogram **positive** → momentum building." if mh>0 else "- MACD histogram **negative** → momentum fading.")
    rs = trow.get("rsi_strength")
    if pd.notna(rs):
        r = 50 + 50*rs
        if r>70: lines.append(f"- RSI ≈ **{r:.0f}** → overbought; pullback risk.")
        elif r<30: lines.append(f"- RSI ≈ **{r:.0f}** → oversold; mean-reversion possible.")
        else: lines.append(f"- RSI ≈ **{r:.0f}** → mid-range.")
    m12 = trow.get("mom12m")
    if pd.notna(m12):
        lines.append(f"- 12-month momentum ≈ **{m12*100:.1f}%** (from 5y history).")
    if not lines:
        lines = ["- Insufficient price history."]
    lines.append("**Trading lens:** confirm with EMA50/MACD; manage entries with RSI extremes.")
    return "\n".join(lines)

def macro_story(m):
    if not m or pd.isna(m.get("macro")):
        return "Macro data unavailable; treating as neutral."
    v, e20, gap = m["vix"], m["ema20"], m["gap"]
    level = "low & calm → **supportive**" if v<14 else ("moderate → **OK**" if v<22 else "elevated → **caution**")
    trend = "below trend (risk cooling)" if gap<-0.03 else ("above trend (risk building)" if gap>0.03 else "near trend")
    return (f"- VIX ≈ **{v:.2f}**, EMA20 ≈ **{e20:.2f}** → {level}; trend: {trend}.\n"
            "- Macro score blends **level** (lower is safer) and **trend** (falling volatility improves conditions).")

# -----------------------------
# ---------- Pages ------------
# -----------------------------
def page_landing():
    render_header_centered("Rate My", "Pick a stock or your portfolio — we’ll rate it with clear, friendly explanations and charts.")
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div class="landing-cta">', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("📈 Rate My Stock", use_container_width=True, type="primary"):
                st.session_state.page="stock"; st.rerun()
        with cc2:
            if st.button("💼 Rate My Portfolio", use_container_width=True):
                st.session_state.page="portfolio"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def page_stock():
    render_header_centered("Rate My Stock", "Type a ticker; pick peers (or upload/paste your own); we’ll score it and show receipts.")
    # Ticker centered
    st.markdown('<div class="center-input">', unsafe_allow_html=True)
    ticker = st.text_input(" ", value="AAPL", placeholder="Type a ticker…").upper().strip().replace(".", "-")
    st.markdown('</div>', unsafe_allow_html=True)

    # Peer source
    st.markdown("### Peer universe")
    src = st.radio("Source", ["Built-in", "Upload file", "Paste tickers", "Raw URL"], horizontal=True, index=0)
    peers=[]; label=""
    if src=="Built-in":
        which = st.selectbox("Built-ins", list(BUILT_INS.keys()), index=0)
        peers = BUILT_INS[which]() or []
        label = which
    elif src=="Upload file":
        up = st.file_uploader("Upload CSV/XLSX with 'Ticker' / 'Symbol'", type=["csv","xlsx","xls"])
        if up: peers = parse_peer_file(up)
        label = f"Upload ({len(peers)})"
        # template
        tmpl = pd.DataFrame({"Ticker":["AAPL","MSFT","NVDA","AMZN"]}).to_csv(index=False).encode()
        st.download_button("Download template (CSV)", tmpl, file_name="peer_template.csv")
    elif src=="Paste tickers":
        txt = st.text_area("One ticker per line", value="AAPL\nMSFT\nNVDA\nAMZN", height=120)
        peers = _clean_tickers(txt.splitlines()); label=f"Pasted ({len(peers)})"
    else:
        url = st.text_input("Raw CSV/TSV URL (e.g., GitHub raw)")
        peers = parse_peer_url(url) if url else []
        label = "URL"

    if ticker and ticker not in peers:
        peers = [ticker] + peers

    # Banner (no macro duplication)
    st.markdown(f"<div class='banner'>Peers loaded: {max(0,len(peers)-1)} | Source: {label}</div>", unsafe_allow_html=True)

    # Button
    if not st.button("Rate", type="primary"):
        return

    # Progress
    prog = st.progress(0, text="Fetching data…")

    end = datetime.utcnow().date()
    start_2y = end - timedelta(days=int(365*2))
    prices_2y = load_history(peers, start_2y, end)
    prog.progress(30, text="Pulling fundamentals…")
    fdf = load_fundamentals(peers, enrich_cap=min(160, len(peers)))
    fz = fundamentals_z(fdf)
    prog.progress(55, text="Computing technicals…")
    tdf, px = compute_technicals(prices_2y, ticker)
    prog.progress(72, text="Evaluating macro regime…")
    mk = macro_vix()
    prog.progress(90, text="Scoring…")

    # Scores for main ticker
    if ticker not in fz.index and ticker not in tdf.index:
        prog.empty()
        st.error("We couldn't fetch enough data for this ticker. Try another or add a custom peer file.")
        return

    fscore = fz.loc[ticker]["FUND_score"] if ticker in fz.index else np.nan
    tscore = tdf.loc[ticker]["TECH_score"] if ticker in tdf.index else np.nan
    mscore = mk["macro"]
    z, pct = composite_score(fscore, tscore, mscore)
    rating = "Strong Buy" if pct>=85 else "Buy" if pct>=60 else "Hold" if pct>=40 else "Sell" if pct>=20 else "Strong Sell"

    prog.progress(100, text="Done!")

    # KPI row
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.markdown("**Fundamentals**"); st.markdown(f"<div class='kpi kpi-big'>{fscore if pd.notna(fscore) else 'n/a'}</div>", unsafe_allow_html=True)
    with k2: st.markdown("**Technicals**");   st.markdown(f"<div class='kpi kpi-big'>{tscore if pd.notna(tscore) else 'n/a'}</div>", unsafe_allow_html=True)
    with k3: st.markdown("**Macro (VIX)**");  st.markdown(f"<div class='kpi kpi-big'>{mscore:.3f}</div>", unsafe_allow_html=True)
    with k4: st.markdown("**Score (0–100)**");st.markdown(f"<div class='kpi kpi-big'>{pct:.1f}</div>", unsafe_allow_html=True)
    with k5: st.markdown("**Recommendation**"); st.markdown(f"<div class='kpi kpi-big'>{rating}</div>", unsafe_allow_html=True)

    # Why this rating
    st.markdown("### Why this rating?")
    with st.expander(f"{ticker} — {rating} (Score: {pct:.1f})", expanded=True):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Fundamentals — details**")
            if ticker in fz.index:
                st.markdown(fundamentals_story(fz.loc[ticker]))
                cols = [c for c in fz.columns if c.endswith("_z") and c!="FUND_score"]
                ftable = fz.loc[[ticker], cols].T.rename(columns={ticker:"z"}).reset_index().rename(columns={"index":"metric"})
                st.dataframe(ftable, use_container_width=True, hide_index=True)
            else:
                st.info("No fundamental snapshot.")
        with c2:
            st.markdown("**Technicals — indicators**")
            if ticker in tdf.index:
                st.markdown(technicals_story(tdf.loc[ticker]))
            else:
                st.info("No technical indicators available.")
        with c3:
            st.markdown("**Macro — VIX regime**")
            st.markdown(macro_story(mk))

        # Download breakdown
        to_join = []
        if ticker in fz.index: to_join.append(fz.loc[[ticker]])
        if ticker in tdf.index: to_join.append(tdf.loc[[ticker]])
        if to_join:
            out = pd.concat(to_join, axis=1).T
            st.download_button("⬇️ Download breakdown (CSV)", out.to_csv().encode(), file_name=f"{ticker}_breakdown.csv")

    # Charts
    st.markdown("### Charts")
    if ticker in px.columns:
        s = px[ticker].dropna()
        if not s.empty:
            # Price & EMA50
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(s.index, s, label=ticker)
            ax.plot(s.index, ema(s,50), label="EMA50", alpha=0.8)
            ax.set_title(f"{ticker} — Price & EMA50")
            ax.legend()
            st.pyplot(fig)
            st.markdown("<div class='caption'>Trend gauge: sustained closes above a rising EMA50 tend to support momentum.</div>", unsafe_allow_html=True)

            # MACD histogram
            if len(s)>=60:
                macd_line = ema(s,12)-ema(s,26)
                macd_sig  = ema(macd_line,9)
                hist = (macd_line - macd_sig)
                fig2, ax2 = plt.subplots(figsize=(10,2.5))
                ax2.fill_between(hist.index, 0, hist.values, alpha=0.4)
                ax2.set_title("MACD Histogram")
                st.pyplot(fig2)
                st.markdown("<div class='caption'>Above zero suggests momentum building; below zero suggests fading.</div>", unsafe_allow_html=True)

            # 12m momentum (from 5y series)
            s5 = load_history([ticker], start=(datetime.utcnow().date()-timedelta(days=int(365*5))), end=datetime.utcnow().date())
            if ticker in s5.columns and len(s5[ticker].dropna())>=252:
                mom = s5[ticker]/s5[ticker].shift(252)-1.0
                fig3, ax3 = plt.subplots(figsize=(10,3))
                ax3.plot(mom.index, mom.values)
                ax3.axhline(0, color="gray", linewidth=1)
                ax3.set_title("12-month Momentum")
                st.pyplot(fig3)
                st.markdown("<div class='caption'>Positive 12-month returns favor relative strength vs. peers.</div>", unsafe_allow_html=True)
    else:
        st.info("Not enough price data to draw charts.")

def page_portfolio():
    render_header_centered("Rate My Portfolio", "Enter tickers + % or $, click Apply changes to sync, then rate.")
    c1,c2 = st.columns([1,3])
    with c1: cur = st.selectbox("Currency", ["$", "€", "£", "CHF", "CAD"], index=0)
    with c2: total = st.number_input(f"Total portfolio value ({cur})", min_value=0.0, value=10000.0, step=100.0)

    st.markdown("### Holdings (Submit to apply sync)")
    mode = st.radio("Sync mode (applies on Submit)", ["Auto", "Percent → Amount", "Amount → Percent"], horizontal=True)

    # Initial table
    init = pd.DataFrame({"Ticker":["AAPL","MSFT","NVDA","AMZN"],
                         "Percent (%)":[25.0,25.0,25.0,25.0],
                         "Amount":[total*0.25, total*0.25, total*0.25, total*0.25]})
    if "pf_df" not in st.session_state:
        st.session_state["pf_df"] = init.copy()

    edited = st.data_editor(
        st.session_state["pf_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="pf_editor",
    )

    if st.button("Apply changes", type="primary"):
        df = edited.copy()
        df["Ticker"] = df["Ticker"].astype(str).upper().str.strip().str.replace(".", "-", regex=False)
        df = df[df["Ticker"].str.len()>0]

        if mode=="Percent → Amount" or (mode=="Auto" and df["Percent (%)"].fillna(0).sum()>0):
            df["Amount"] = total * df["Percent (%)"].fillna(0)/100.0
        else:
            amt_sum = df["Amount"].fillna(0).sum()
            if amt_sum>0:
                df["Percent (%)"] = 100.0 * df["Amount"].fillna(0)/amt_sum
            else:
                df["Percent (%)"] = 0.0

        # Normalize to 100%
        s = df["Percent (%)"].sum()
        if s>0:
            df["Percent (%)"] = df["Percent (%)"] * (100.0/s)
            df["Amount"] = total * df["Percent (%)"]/100.0

        st.session_state["pf_df"] = df.copy()
        st.success("Holdings updated.")
        st.rerun()

    df = st.session_state["pf_df"].copy()
    if abs(df["Percent (%)"].sum()-100.0) > 0.5:
        st.warning("Weights do not sum to ~100%. Click **Apply changes** to normalize.")

    # Optional peer set
    peer_src = st.selectbox("Add a peer set for comparison (optional)", ["None"]+list(BUILT_INS.keys()))
    peers = BUILT_INS[peer_src]() if peer_src!="None" else []
    # Ensure portfolio tickers included
    tickers = df["Ticker"].dropna().unique().tolist()
    peers = sorted(list(set(peers) | set(tickers)))

    st.markdown(f"<div class='banner'>Peers loaded: {max(0,len(peers)-len(tickers))} | Source: {peer_src}</div>", unsafe_allow_html=True)

    if not st.button("Rate portfolio", type="primary"):
        return

    with st.spinner("Fetching data and computing portfolio scores…"):
        end = datetime.utcnow().date()
        start = end - timedelta(days=int(365*2))
        px = load_history(peers, start, end)
        fdf = load_fundamentals(peers, enrich_cap=min(160, len(peers)))
        fz = fundamentals_z(fdf)
        tdf, _ = compute_technicals(px, tickers[0] if tickers else "")
        mk = macro_vix()

    rows=[]
    for _, r in df.iterrows():
        t = r["Ticker"]; w = float(r["Percent (%)"] or 0)/100.0
        f = fz.loc[t]["FUND_score"] if t in fz.index else np.nan
        te= tdf.loc[t]["TECH_score"] if t in tdf.index else np.nan
        z, pct = composite_score(f, te, mk["macro"])
        rows.append({"Ticker":t,"Weight":w,"Fundamentals":f,"Technicals":te,"Macro (VIX)":mk["macro"],"Composite_z":z,"Score (0-100)":pct})
    res = pd.DataFrame(rows).set_index("Ticker")
    res["Weight × CompZ"] = res["Weight"] * res["Composite_z"]

    # Portfolio KPIs
    port_z = float(res["Weight × CompZ"].sum())
    port_pct = _z_to_pct(np.clip(port_z, -2.5, 2.5))
    rec = "Strong Buy" if port_pct>=85 else "Buy" if port_pct>=60 else "Hold" if port_pct>=40 else "Sell" if port_pct>=20 else "Strong Sell"

    st.markdown("### Portfolio rating")
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown("**Portfolio composite (z)**"); st.markdown(f"<div class='kpi kpi-big'>{port_z:.3f}</div>", unsafe_allow_html=True)
    with c2: st.markdown("**Score (0–100)**"); st.markdown(f"<div class='kpi kpi-big'>{port_pct:.1f}</div>", unsafe_allow_html=True)
    with c3: st.markdown("**Recommendation**"); st.markdown(f"<div class='kpi kpi-big'>{rec}</div>", unsafe_allow_html=True)

    st.markdown("### Per-name contribution")
    st.dataframe(res[["Weight","Fundamentals","Technicals","Macro (VIX)","Composite_z","Weight × CompZ","Score (0-100)"]],
                 use_container_width=True)

    # Simple diversification notes
    st.markdown("### Diversification (quick checks)")
    # Sector mix (from fundamentals table if available)
    sectors = fdf.get("sector").reindex(df["Ticker"]).fillna("Unknown")
    sec_weights = (df.set_index("Ticker")["Percent (%)"]/100.0).groupby(sectors).sum().sort_values(ascending=False)
    if not sec_weights.empty:
        st.bar_chart(sec_weights, height=260, use_container_width=True)
        st.markdown("<div class='caption'>Sector weights by portfolio % — broader spread usually reduces idiosyncratic risk.</div>", unsafe_allow_html=True)

    # Correlation
    if not px.empty and len(tickers) >= 2 and all(t in px.columns for t in tickers):
        rets = px[tickers].pct_change().dropna(how="all")
        if not rets.empty:
            corr = rets.corr().values
            upper = corr[np.triu_indices_from(corr, k=1)]
            if upper.size:
                avg_corr = float(np.nanmean(upper))
                st.write(f"- Average pairwise correlation: **{avg_corr:.2f}** (lower = better diversification).")
    else:
        st.write("- Not enough overlapping price history to compute correlations.")

# -----------------------------
# --------- Router ------------
# -----------------------------
def router():
    page = st.session_state.get("page","home")
    if page=="home":
        page_landing()
    elif page=="stock":
        if st.button("← Back"): st.session_state["page"]="home"; st.rerun()
        page_stock()
    elif page=="portfolio":
        if st.button("← Back"): st.session_state["page"]="home"; st.rerun()
        page_portfolio()
    else:
        st.session_state["page"]="home"; page_landing()

if __name__ == "__main__":
    router()