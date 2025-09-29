# app.py
# ============================
# Rate My (Stock | Portfolio)
# Comprehensive Streamlit app
# ============================

import base64
import io
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup

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
# --------- Utilities ---------
# -----------------------------

@st.cache_data(show_spinner=False)
def _get(url, **kwargs):
    try:
        r = requests.get(url, timeout=15, **kwargs)
        r.raise_for_status()
        return r
    except Exception:
        return None


def _z_to_pct(z):
    if pd.isna(z):
        return np.nan
    # normal CDF
    return 100.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def clamp01(x):
    if pd.isna(x):
        return np.nan
    return max(0.0, min(1.0, float(x)))


def nanmean_safe(vals):
    vals = [v for v in vals if pd.notna(v)]
    return np.nan if not vals else float(np.mean(vals))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


# -----------------------------
# ------- Logo (inline) -------
# -----------------------------

@st.cache_data(show_spinner=False)
def make_triangle_logo(w=220, h=220):
    """Make a simple green→yellow→red triangle (no inner lines)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    # Triangle points
    A = (w // 2, int(h * 0.08))
    B = (int(w * 0.08), int(h * 0.92))
    C = (int(w * 0.92), int(h * 0.92))

    # Gradient bands from bottom to top
    steps = 80
    for i in range(steps):
        t = i / (steps - 1.0)
        # color ramp red -> orange -> yellow -> light green -> green
        if t < 0.25:
            # red to orange
            k = t / 0.25
            col = (int(220 + 20 * k), int(40 + 60 * k), 40, 255)
        elif t < 0.5:
            k = (t - 0.25) / 0.25
            col = (255, int(100 + 120 * k), 50, 255)  # orange->yellow
        elif t < 0.75:
            k = (t - 0.5) / 0.25
            col = (240 - int(60 * k), 220 - int(60 * k), 60, 255)  # yellow->lime
        else:
            k = (t - 0.75) / 0.25
            col = (180 - int(90 * k), 220 + int(20 * k), 80 + int(60 * k), 255)  # lime->green

        # Horizontal slice polygon inside triangle
        y1 = int(B[1] - (B[1] - A[1]) * t)
        y2 = int(B[1] - (B[1] - A[1]) * (t + 1 / steps))
        # Interpolate left/right edges
        def edge_point(p1, p2, y):
            if p2[1] == p1[1]:
                return p1
            u = (y - p1[1]) / (p2[1] - p1[1])
            u = max(0, min(1, u))
            x = int(p1[0] + u * (p2[0] - p1[0]))
            return (x, y)

        L1 = edge_point(B, A, y1)
        R1 = edge_point(C, A, y1)
        L2 = edge_point(B, A, y2)
        R2 = edge_point(C, A, y2)
        dr.polygon([L1, R1, R2, L2], fill=col)

    # thin border
    dr.polygon([A, B, C], outline=(230, 230, 230, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_LOGO_BYTES = make_triangle_logo()


def render_logo_img(height=64):
    b64 = base64.b64encode(_LOGO_BYTES).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="logo" style="height:{height}px;vertical-align:middle;">'


# -----------------------------
# --------- Styling -----------
# -----------------------------
st.markdown(
    """
<style>
.block-container { max-width: 1180px; }
.header-row { display:flex; align-items:center; justify-content:center; gap:14px; margin:.6rem 0 1.2rem; }
.header-title { font-size:3.4rem; font-weight:850; line-height:1.08; margin:0; letter-spacing:.25px; }
.header-sub { text-align:center; color:#9aa0a6; margin-top:.3rem; font-size:1.06rem; }
.landing-cta { display:flex; align-items:center; justify-content:center; gap:24px; margin-top:20px; }
.kpi { padding:1rem 1.1rem; border-radius:12px; background:#111418; border:1px solid #222; }
.kpi-big { font-size:2.2rem; font-weight:800; margin-top:.25rem; }
.small { color:#9aa0a6; font-size:.92rem; }
.banner { background:#0c2f22; color:#cdebdc; border-radius:10px; padding:.9rem 1.1rem; margin:.75rem 0 1.25rem; }
.topbar { display:flex; justify-content:flex-end; margin:.2rem 0 .8rem; }
.table-mini td, .table-mini th { padding:.4rem .6rem !important; font-size:.92rem !important; }
</style>
""",
    unsafe_allow_html=True,
)


def render_header_centered(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="header-row">
          <div class="header-title">{title}</div>
          {render_logo_img(64)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f"<div class='header-sub'>{subtitle}</div>", unsafe_allow_html=True)


# ----------------------------------------
# -------- Peer universe fetchers --------
# ----------------------------------------

SP500_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
]
NDX_URLS = [
    "https://en.wikipedia.org/wiki/Nasdaq-100",
    "https://raw.githubusercontent.com/nikbearbrown/Financial-Data-Science/main/data/NASDAQ_100_Company_List.csv",
]
DOW_URLS = [
    "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
]
R2000_FALLBACK = [
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf",
]  # we’ll provide a light fallback set if scrape fails


@st.cache_data(show_spinner=False)
def fetch_sp500_constituents():
    tickers = []
    # Wikipedia
    r = _get(SP500_URLS[0])
    if r:
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", {"id": "constituents"})
        if table:
            for tr in table.select("tbody tr"):
                t = tr.find("td")
                if t:
                    sym = t.text.strip().upper()
                    if "." in sym:  # BRK.B → BRK-B for Yahoo
                        sym = sym.replace(".", "-")
                    tickers.append(sym)
    # Fallback CSV
    if not tickers:
        r = _get(SP500_URLS[1])
        if r:
            df = pd.read_csv(io.StringIO(r.text))
            for sym in df.iloc[:, 0].astype(str).str.upper():
                sym = sym.replace(".", "-")
                tickers.append(sym)
    return sorted(list(set(tickers)))


@st.cache_data(show_spinner=False)
def fetch_nasdaq100():
    tickers = []
    r = _get(NDX_URLS[0])
    if r:
        soup = BeautifulSoup(r.text, "lxml")
        # new page layout can vary — collect all "Symbol" columns
        for table in soup.find_all("table"):
            headers = [th.text.strip().lower() for th in table.find_all("th")]
            if any("symbol" in h for h in headers):
                for tr in table.select("tbody tr"):
                    tds = [td.text.strip() for td in tr.find_all("td")]
                    if tds:
                        sym = tds[0].upper()
                        sym = sym.replace(".", "-")
                        if sym and sym.isascii():
                            tickers.append(sym)
    if not tickers:
        r = _get(NDX_URLS[1])
        if r:
            df = pd.read_csv(io.StringIO(r.text))
            col = [c for c in df.columns if "symbol" in c.lower()]
            if col:
                for sym in df[col[0]].astype(str).str.upper():
                    tickers.append(sym.replace(".", "-"))
    return sorted(list(set([t for t in tickers if t.isascii()])))


@st.cache_data(show_spinner=False)
def fetch_dow30():
    tickers = []
    r = _get(DOW_URLS[0])
    if r:
        soup = BeautifulSoup(r.text, "lxml")
        for table in soup.find_all("table"):
            headers = [th.text.strip().lower() for th in table.find_all("th")]
            if "symbol" in headers or "ticker symbol" in headers:
                for tr in table.select("tbody tr"):
                    tds = [td.text.strip() for td in tr.find_all("td")]
                    if tds:
                        # symbol column can be first or second, detect
                        cand = [t for t in tds if t.isupper() or "." in t]
                        if cand:
                            sym = cand[0].upper().replace(".", "-")
                            tickers.append(sym)
    # Dow is small; ensure unique
    return sorted(list(set(tickers)))


@st.cache_data(show_spinner=False)
def fetch_russell2000_fallback(limit=350):
    """We don't have a clean free source for the full R2000. Provide a broad
    fallback by sampling from iShares IWM holdings download if accessible,
    else return empty to avoid blocking."""
    try:
        # iShares CSV behind a dynamic link; keep this very soft
        # If it fails, we return empty and won’t block the app.
        return []
    except Exception:
        return []


PEER_UNIVERSE_SOURCES = {
    "S&P 500": fetch_sp500_constituents,
    "NASDAQ-100": fetch_nasdaq100,
    "Dow 30": fetch_dow30,
    "Russell 2000 (lite)": fetch_russell2000_fallback,  # partial
}


# ------------------------------------------------
# --------- Market data & fundamentals ----------
# ------------------------------------------------

@st.cache_data(show_spinner=False)
def load_history(tickers, start, end):
    if yf is None:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        if isinstance(data, pd.DataFrame) and "Close" in data:
            px = data["Close"]
        else:
            px = data
        return px.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_fundamental_snapshot(tickers):
    """Light snapshot from yfinance. We only read a few fields to avoid rate limits.
    Returns DataFrame with rows=tickers."""
    rows = []
    if yf is None:
        return pd.DataFrame()
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            # yfinance fast_info is thin; we add a few with .info (guarded)
            # We'll try to enrich with margins/growth if available
            hard = {}
            try:
                full = yf.Ticker(t).info
                # normalize keys we'll use
                hard["sector"] = full.get("sector")
                hard["longName"] = full.get("longName") or full.get("shortName")
                for k in [
                    "profitMargins", "grossMargins", "operatingMargins",
                    "ebitdaMargins", "forwardPE", "trailingPE",
                    "debtToEquity", "revenueGrowth", "earningsGrowth",
                    "returnOnEquity",
                ]:
                    hard[k] = full.get(k)
            except Exception:
                pass
            rows.append(
                {
                    "ticker": t,
                    "sector": hard.get("sector"),
                    "name": hard.get("longName"),
                    "profitMargins": hard.get("profitMargins"),
                    "grossMargins": hard.get("grossMargins"),
                    "operatingMargins": hard.get("operatingMargins"),
                    "ebitdaMargins": hard.get("ebitdaMargins"),
                    "forwardPE": hard.get("forwardPE"),
                    "trailingPE": hard.get("trailingPE"),
                    "debtToEquity": hard.get("debtToEquity"),
                    "revenueGrowth": hard.get("revenueGrowth"),
                    "earningsGrowth": hard.get("earningsGrowth"),
                    "returnOnEquity": hard.get("returnOnEquity"),
                }
            )
        except Exception:
            rows.append({"ticker": t})
    df = pd.DataFrame(rows).set_index("ticker")
    return df


def zscore_series(s, invert=False):
    s = s.astype(float)
    z = (s - s.mean(skipna=True)) / (s.std(skipna=True) + 1e-9)
    if invert:
        z = -z
    return z


def compute_fundamental_z(peers_df):
    """Compute z-scores vs peers; where 'lower is better' (PE, leverage) we invert."""
    df = peers_df.copy()
    for col, inv in [
        ("profitMargins", False),
        ("grossMargins", False),
        ("operatingMargins", False),
        ("ebitdaMargins", False),
        ("forwardPE", True),    # lower better
        ("trailingPE", True),   # lower better
        ("debtToEquity", True), # lower better
        ("revenueGrowth", False),
        ("earningsGrowth", False),
        ("returnOnEquity", False),
    ]:
        if col in df and df[col].notna().sum() >= 5:
            df[f"{col}_z"] = zscore_series(df[col].astype(float), invert=inv)
        else:
            df[f"{col}_z"] = np.nan
    return df


def compute_technicals(prices: pd.DataFrame):
    """Return dict of per-ticker technical metrics based on recent price history."""
    out = {}
    if prices.empty:
        return out
    # Work per column
    for t in prices.columns:
        s = prices[t].dropna()
        if s.size < 80:
            out[t] = {
                "dma_gap": np.nan,
                "macd_hist": np.nan,
                "rsi_strength": np.nan,
                "mom12m": np.nan,
            }
            continue

        # 50d EMA gap
        ema50 = ema(s, 50)
        gap = (s.iloc[-1] / ema50.iloc[-1]) - 1.0 if ema50.iloc[-1] != 0 else np.nan

        # MACD histogram (12/26 EMA and 9 signal)
        ema12 = ema(s, 12)
        ema26 = ema(s, 26)
        macd = ema12 - ema26
        signal = ema(macd, 9)
        hist = (macd - signal).iloc[-1]

        # RSI (approx)
        diff = s.diff()
        up = diff.clip(lower=0.0).rolling(14).mean()
        dn = (-diff.clip(upper=0.0)).rolling(14).mean()
        rs = (up / (dn + 1e-9)).iloc[-1]
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_strength = (rsi - 50.0) / 50.0  # -1..+1 roughly

        # 12-month momentum (need 252 trading days)
        mom = np.nan
        if s.size >= 252:
            mom = (s.iloc[-1] / s.iloc[-252]) - 1.0

        out[t] = {
            "dma_gap": float(gap),
            "macd_hist": float(hist),
            "rsi_strength": float(rsi_strength),
            "mom12m": float(mom) if pd.notna(mom) else np.nan,
        }
    return out


@st.cache_data(show_spinner=False)
def vix_series(days=300):
    if yf is None:
        return pd.Series(dtype=float)
    try:
        v = yf.download("^VIX", period=f"{days}d", auto_adjust=False, progress=False)
        if isinstance(v, pd.DataFrame) and "Close" in v:
            return v["Close"].dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def macro_vix_score():
    s = vix_series(400)
    if s.empty or s.size < 25:
        return 0.5, np.nan, np.nan, 0.0
    last = s.iloc[-1]
    ema20 = ema(s, 20).iloc[-1]
    gap = (last / ema20) - 1.0 if ema20 != 0 else 0.0

    # Level score: low VIX => high score; clamp to [0,1]
    # 12 -> ~1, 30 -> ~0
    lvl = 1.0 - clamp01((last - 12.0) / (30.0 - 12.0))

    # Trend score: VIX below its EMA => bullish
    if gap > 0.03:
        tr = 0.25
    elif gap < -0.03:
        tr = 0.85
    else:
        tr = 0.55

    return clamp01(0.6 * lvl + 0.4 * tr), last, ema20, gap


# -----------------------------------------
# --------- Scoring & Narratives ----------
# -----------------------------------------

def fundamentals_story(zrow: pd.Series) -> str:
    def pct(z): 
        return f"~{_z_to_pct(z):.0f}th pct" if pd.notna(z) else "n/a"

    g_rev = zrow.get("revenueGrowth_z")
    g_eps = zrow.get("earningsGrowth_z")
    roe   = zrow.get("returnOnEquity_z")
    pm    = zrow.get("profitMargins_z")
    gm    = zrow.get("grossMargins_z")
    om    = zrow.get("operatingMargins_z")
    ebit  = zrow.get("ebitdaMargins_z")
    pe_f  = zrow.get("forwardPE_z")
    pe_t  = zrow.get("trailingPE_z")
    dte   = zrow.get("debtToEquity_z")

    growth_g      = nanmean_safe([g_rev, g_eps])
    profitability = nanmean_safe([roe, pm, gm, om, ebit])
    valuation_g   = nanmean_safe([pe_f, pe_t])
    leverage_inv  = dte

    def bucket(z):
        if pd.isna(z): return "insufficient data"
        if z >= 1.2:   return "top decile vs peers"
        if z >= 0.7:   return "well above average"
        if z >= 0.2:   return "slightly above average"
        if z <= -1.0:  return "bottom decile vs peers"
        if z <= -0.5:  return "below average"
        return "roughly in line"

    lines = []
    lines.append("**Growth tilt**")
    if pd.isna(growth_g):
        lines.append("- Not enough data on growth.")
    else:
        lines.append(f"- Aggregate growth (revenue/earnings) is **{bucket(growth_g)}** ({pct(growth_g)}).")

    lines.append("\n**Profitability & quality**")
    if pd.isna(profitability):
        lines.append("- Not enough margin/ROE data.")
    else:
        pts = []
        if pd.notna(roe): pts.append(f"ROE {pct(roe)}")
        if pd.notna(pm):  pts.append(f"profit margin {pct(pm)}")
        if pd.notna(om):  pts.append(f"operating margin {pct(om)}")
        lines.append(f"- Overall profitability is **{bucket(profitability)}** ({', '.join(pts) or 'n/a'}).")

    lines.append("\n**Valuation**")
    if pd.isna(valuation_g):
        lines.append("- Not enough valuation data.")
    else:
        parts = []
        if pd.notna(pe_f): parts.append("forward " + pct(pe_f))
        if pd.notna(pe_t): parts.append("trailing " + pct(pe_t))
        lines.append(f"- Valuation (PE) screens **{bucket(valuation_g)}** vs peers ({', '.join(parts)}). "
                     "Higher z here means cheaper vs peers (lower PE).")

    lines.append("\n**Balance sheet**")
    if pd.isna(leverage_inv):
        lines.append("- Debt/Equity not available.")
    else:
        if leverage_inv >= 0.7:
            lines.append(f"- Leverage looks **conservative** ({pct(leverage_inv)}).")
        elif leverage_inv <= -0.5:
            lines.append(f"- Leverage is **on the higher side** ({pct(leverage_inv)}); watch interest coverage.")
        else:
            lines.append(f"- Leverage is **moderate** ({pct(leverage_inv)}).")

    pos = [z for z in [growth_g, profitability, valuation_g, leverage_inv] if pd.notna(z) and z >= 0.5]
    neg = [z for z in [growth_g, profitability, valuation_g, leverage_inv] if pd.notna(z) and z <= -0.5]
    if pos and not neg:
        lines.append("\n**Summary**: broad-based **fundamental strength** with supportive quality/valuation.")
    elif neg and not pos:
        lines.append("\n**Summary**: **fundamentals lag** peers; monitor margins, growth durability, and leverage.")
    else:
        lines.append("\n**Summary**: mixed profile — strengths offset by weak spots; neutral to slightly mixed overall.")
    return "\n".join(lines)


def technicals_story(trow: pd.Series) -> str:
    pts = []

    g = trow.get("dma_gap")
    if pd.notna(g):
        gp = g * 100
        if gp > 5:
            pts.append(f"- Price is **{gp:.1f}% above** the 50-day EMA → **strong trend tailwind**; dips often get bought.")
        elif gp > 2:
            pts.append(f"- Price is **{gp:.1f}% above** the 50-day EMA → **mild tailwind**.")
        elif gp < -5:
            pts.append(f"- Price is **{abs(gp):.1f}% below** the 50-day EMA → **strong headwind**; rallies may fade.")
        elif gp < -2:
            pts.append(f"- Price is **{abs(gp):.1f}% below** the 50-day EMA → **mild headwind**.")
        else:
            pts.append("- Price is **near** the 50-day EMA (±2%) → neutral reference.")
    else:
        pts.append("- 50-day EMA comparison unavailable.")

    mh = trow.get("macd_hist")
    if pd.notna(mh):
        if mh > 0:
            pts.append("- MACD histogram **positive** → **momentum building**; pullbacks often constructive.")
        elif mh < 0:
            pts.append("- MACD histogram **negative** → **momentum fading**; be selective on entries.")
        else:
            pts.append("- MACD histogram around zero → transition.")
    else:
        pts.append("- MACD histogram unavailable.")

    r = trow.get("rsi_strength")
    if pd.notna(r):
        rsi = 50 + 50*r
        if rsi >= 70:
            pts.append(f"- RSI ≈ **{rsi:.0f}** → **overbought**; short-term pullback risk ↑.")
        elif rsi <= 30:
            pts.append(f"- RSI ≈ **{rsi:.0f}** → **oversold**; mean-reversion potential ↑.")
        else:
            pts.append(f"- RSI ≈ **{rsi:.0f}** → neutral.")
    else:
        pts.append("- RSI unavailable.")

    m = trow.get("mom12m")
    if pd.notna(m):
        pts.append(f"- 12-month price change: **{m*100:+.1f}%**; positive readings bias to relative strength.")
    else:
        pts.append("- 12-month momentum not available (needs ≥1y history).")

    return "\n".join(pts)


def macro_story(vix_last, vix_ema20, vix_gap) -> str:
    if np.isnan(vix_last):
        return "VIX unavailable; using a neutral macro score."
    if vix_last <= 13:
        lvl = "very calm → **risk-friendly** backdrop"
    elif vix_last <= 18:
        lvl = "calm → **supportive** for risk"
    elif vix_last <= 24:
        lvl = "elevated → **caution** warranted"
    else:
        lvl = "stressed → **risk-off** tone"

    if vix_gap > 0.03:
        tr = "rising **above** its 20-day average → volatility **building** (headwind)."
    elif vix_gap < -0.03:
        tr = "falling **below** its 20-day average → volatility **easing** (tailwind)."
    else:
        tr = "hovering **near** its 20-day average → **neutral** short-term trend."

    return f"- **Level**: {lvl}.  \n- **Trend**: {tr}"


# -----------------------------------------
# --------------- Landing -----------------
# -----------------------------------------

def page_landing():
    render_header_centered(
        "Rate My",
        "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="landing-cta">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📈  Rate My Stock", use_container_width=True, type="primary"):
                st.session_state.page = "stock"
                st.rerun()
        with c2:
            if st.button("💼  Rate My Portfolio", use_container_width=True):
                st.session_state.page = "portfolio"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------
# -------- Stock Rating Page --------------
# -----------------------------------------

def page_stock():
    st.markdown('<div class="topbar"><a href="#" onclick="window.location.reload()" style="text-decoration:none;"></a></div>', unsafe_allow_html=True)
    render_header_centered("Rate My Stock")

    # Inputs
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    peer_choice = st.selectbox("Peer universe", list(PEER_UNIVERSE_SOURCES.keys()), index=0)
    with st.expander("Advanced settings", expanded=False):
        hist_years = st.slider("History window (years, for technicals)", 1, 5, 2)

    # Progress
    prog = st.progress(0, text="Fetching fundamentals…")

    # Peers
    get_peers = PEER_UNIVERSE_SOURCES[peer_choice]
    peers = get_peers() or []
    prog.progress(10, text=f"Loading peer set ({peer_choice})")
    if ticker not in peers:
        peers = [ticker] + peers
    peer_sample = peers  # full set

    # Fundamentals
    fdf = load_fundamental_snapshot(peer_sample)
    fdf_z = compute_fundamental_z(fdf)
    prog.progress(45, text="Computing technicals…")

    # Technicals (use longer window to fill 12m momentum if needed)
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(365 * max(hist_years, 2)))
    px = load_history(peer_sample, start, end)
    tech = compute_technicals(px)
    prog.progress(70, text="Building macro…")

    # Macro (VIX)
    macro_s, vix_last, vix_ema20, vix_gap = macro_vix_score()
    prog.progress(85, text="Scoring…")

    # Compose final table row for the chosen ticker
    def comp_fund_score(row):
        parts = []
        for k in [
            "profitMargins_z","grossMargins_z","operatingMargins_z","ebitdaMargins_z",
            "forwardPE_z","trailingPE_z","debtToEquity_z",
            "revenueGrowth_z","earningsGrowth_z","returnOnEquity_z",
        ]:
            parts.append(row.get(k))
        return clamp01((nanmean_safe(parts) + 3) / 6.0)  # normalize roughly into 0..1

    fund_score = comp_fund_score(fdf_z.loc[ticker]) if ticker in fdf_z.index else np.nan
    tech_row = pd.Series(tech.get(ticker, {}))
    # technical score blend: dma_gap (scaled), macd sign, rsi neutral ~0.5, momentum sign
    t_gap = 0.5 + 0.1 * np.tanh(tech_row.get("dma_gap", 0) / 0.05) if pd.notna(tech_row.get("dma_gap")) else np.nan
    t_macd = 0.7 if tech_row.get("macd_hist", 0) > 0 else (0.3 if tech_row.get("macd_hist", 0) < 0 else 0.5)
    t_rsi = clamp01(0.5 + 0.5 * tech_row.get("rsi_strength", 0)) if pd.notna(tech_row.get("rsi_strength")) else np.nan
    t_mom = 0.7 if pd.notna(tech_row.get("mom12m")) and tech_row["mom12m"] > 0 else (0.3 if pd.notna(tech_row.get("mom12m")) else np.nan)
    tech_score = clamp01(nanmean_safe([t_gap, t_macd, t_rsi, t_mom]))

    weights = dict(fund=0.38, tech=0.38, macro=0.24)
    comp = nanmean_safe([weights["fund"] * fund_score, weights["tech"] * tech_score, weights["macro"] * macro_s])
    score100 = int(round(clamp01(comp) * 100))
    reco = "Strong Buy" if score100 >= 85 else ("Buy" if score100 >= 65 else ("Hold" if score100 >= 45 else ("Sell" if score100 >= 30 else "Strong Sell")))

    prog.progress(100, text="Done!")
    st.markdown(f"<div class='banner'>Peers loaded: {max(0, len(peers)-1)} | Peer set: {peer_choice}</div>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns([1.0, 1.0, 1.0, 1.0, 1.2])
    with k1:
        st.markdown("**Fundamentals**")
        st.markdown(f"<div class='kpi kpi-big'>{fund_score if pd.notna(fund_score) else 'n/a'!s}</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("**Technicals**")
        st.markdown(f"<div class='kpi kpi-big'>{tech_score if pd.notna(tech_score) else 'n/a'!s}</div>", unsafe_allow_html=True)
    with k3:
        st.markdown("**Macro (VIX)**")
        st.markdown(f"<div class='kpi kpi-big'>{macro_s:.3f}</div>", unsafe_allow_html=True)
    with k4:
        st.markdown("**Composite**")
        st.markdown(f"<div class='kpi kpi-big'>{comp:.3f}</div>", unsafe_allow_html=True)
    with k5:
        st.markdown("**Score (0–100) / Recommendation**")
        st.markdown(f"<div class='kpi kpi-big'>{score100} — {reco}</div>", unsafe_allow_html=True)

    # Ratings table for the one ticker
    st.subheader("Ratings")
    tbl = pd.DataFrame(
        [{
            "Ticker": ticker,
            "Fundamentals": round(fund_score, 4) if pd.notna(fund_score) else np.nan,
            "Technicals": round(tech_score, 4) if pd.notna(tech_score) else np.nan,
            "Macro (VIX)": round(macro_s, 4),
            "Composite": round(comp, 4),
            "Score (0–100)": score100,
            "Recommendation": reco,
        }]
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Why this rating
    st.subheader("🔍 Why this rating?")
    with st.expander(f"{ticker} — {reco} (Score: {score100})", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Fundamentals**")
            if ticker in fdf_z.index:
                st.markdown(fundamentals_story(fdf_z.loc[ticker]))
                fundamentals_break = fdf_z.filter(like="_z").loc[ticker].rename("z").to_frame()
                st.markdown("**Fundamental z-scores vs peers (higher better):**")
                st.dataframe(fundamentals_break, use_container_width=True)
            else:
                st.info("No fundamentals available.")

        with c2:
            st.markdown("**Technicals**")
            st.markdown(technicals_story(tech_row))

        with c3:
            st.markdown("**Macro**")
            st.markdown(macro_story(vix_last, vix_ema20, vix_gap))
            if pd.notna(vix_last) and pd.notna(vix_ema20):
                st.write(f"Current VIX: **{vix_last:.2f}**, 20-day EMA: **{vix_ema20:.2f}**")

    # Charts
    st.subheader("Charts")
    if ticker in px.columns:
        st.markdown("**Price & 50-day EMA** — prices trending above a rising EMA tend to show persistent strength.")
        s = px[ticker].dropna()
        if not s.empty:
            dfp = pd.DataFrame({"Close": s, "EMA50": ema(s, 50)})
            st.line_chart(dfp, height=260, use_container_width=True)

    if ticker in px.columns:
        st.markdown("**MACD histogram** — positive histogram suggests momentum building; negative suggests fading.")
        s = px[ticker].dropna()
        if s.size >= 60:
            macd = ema(s, 12) - ema(s, 26)
            signal = ema(macd, 9)
            hist = macd - signal
            st.area_chart(hist.rename("MACD_hist"), height=180, use_container_width=True)

        st.markdown("**12-month momentum** — positive 12-m return favours relative strength vs peers.")
        if s.size >= 252:
            mom = s / s.shift(252) - 1.0
            st.line_chart(mom.rename("12m_momentum"), height=180, use_container_width=True)

    # Download breakdown
    csv = tbl.to_csv(index=False).encode()
    st.download_button("Download stock rating (CSV)", csv, file_name=f"{ticker}_rating.csv", use_container_width=False)


# -----------------------------------------
# -------- Portfolio Rating Page ----------
# -----------------------------------------

def page_portfolio():
    render_header_centered("Rate My Portfolio")

    # Controls
    cols = st.columns([1, 1, 2])
    with cols[0]:
        currency = st.selectbox("Currency", ["$", "€", "£", "CHF", "CAD"], index=0)
    with cols[1]:
        total_val = st.number_input("Total portfolio value", min_value=0.0, value=10000.0, step=100.0, format="%.2f")

    st.write("Enter (Ticker, Percent %) **or** (Ticker, Amount). Click **Apply changes** to sync & lock the table.")
    init_rows = pd.DataFrame(
        [
            {"Ticker": "AAPL", "Percent (%)": 25.0, "Amount": total_val * 0.25},
            {"Ticker": "MSFT", "Percent (%)": 25.0, "Amount": total_val * 0.25},
            {"Ticker": "NVDA", "Percent (%)": 25.0, "Amount": total_val * 0.25},
            {"Ticker": "AMZN", "Percent (%)": 25.0, "Amount": total_val * 0.25},
        ]
    )

    if "pf_table" not in st.session_state:
        st.session_state.pf_table = init_rows.copy()

    # Editable table
    edt = st.data_editor(
        st.session_state.pf_table,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="holdings_editor",
    )

    # Submit action (no auto sync while typing)
    if st.button("Apply changes", type="primary"):
        df = edt.copy()
        # Clean tickers
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        df = df[df["Ticker"].str.len() > 0]
        # If Amount missing, compute from Percent and total
        if df["Amount"].isna().any() and df["Percent (%)"].notna().any():
            df["Amount"] = df["Amount"].fillna(df["Percent (%)"].fillna(0) / 100.0 * total_val)
        # If Percent missing, compute from Amount
        if df["Percent (%)"].isna().any() and df["Amount"].notna().any() and total_val > 0:
            df["Percent (%)"] = df["Percent (%)"].fillna(df["Amount"].fillna(0) / total_val * 100.0)

        # Final normalization to 100% if off by small drift
        pct_sum = df["Percent (%)"].fillna(0).sum()
        if pct_sum > 0:
            df["Percent (%)"] = df["Percent (%)"] / pct_sum * 100.0
            df["Amount"] = df["Percent (%)"] / 100.0 * total_val

        st.session_state.pf_table = df
        st.success("Holdings updated.")
        st.rerun()

    dfh = st.session_state.pf_table.copy()
    if dfh.empty:
        st.info("Add some holdings above.")
        return

    # Peer detection: use S&P 500 by default for breadth
    peers = fetch_sp500_constituents()
    # ensure portfolio names included
    for t in dfh["Ticker"]:
        if t not in peers:
            peers.append(t)

    # Data loads
    tickers = list(dfh["Ticker"].unique())
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * 2)
    px = load_history(sorted(set(peers)), start, end)
    fdf = load_fundamental_snapshot(sorted(set(peers)))
    fdf_z = compute_fundamental_z(fdf)
    tech = compute_technicals(px)
    macro_s, vix_last, vix_ema20, vix_gap = macro_vix_score()

    # Compute per-name scores
    rows = []
    for _, r in dfh.iterrows():
        t = r["Ticker"]
        w = (r["Percent (%)"] or 0) / 100.0
        fund = np.nan
        if t in fdf_z.index:
            fund = clamp01((nanmean_safe([fdf_z.loc[t, c] for c in fdf_z.columns if c.endswith("_z")]) + 3) / 6.0)
        tro = pd.Series(tech.get(t, {}))
        t_gap = 0.5 + 0.1 * np.tanh(tro.get("dma_gap", 0) / 0.05) if pd.notna(tro.get("dma_gap")) else np.nan
        t_macd = 0.7 if tro.get("macd_hist", 0) > 0 else (0.3 if tro.get("macd_hist", 0) < 0 else 0.5)
        t_rsi = clamp01(0.5 + 0.5 * tro.get("rsi_strength", 0)) if pd.notna(tro.get("rsi_strength")) else np.nan
        t_mom = 0.7 if pd.notna(tro.get("mom12m")) and tro["mom12m"] > 0 else (0.3 if pd.notna(tro.get("mom12m")) else np.nan)
        tech_s = clamp01(nanmean_safe([t_gap, t_macd, t_rsi, t_mom]))
        comp = nanmean_safe([0.38 * fund, 0.38 * tech_s, 0.24 * macro_s])
        rows.append(
            {"Ticker": t, "Weight": w, "Fundamentals": fund, "Technicals": tech_s, "Macro": macro_s, "Composite": comp}
        )
    rated = pd.DataFrame(rows)

    # Portfolio score
    pf_score = clamp01((rated["Composite"] * rated["Weight"]).sum())
    pf100 = int(round(pf_score * 100))
    pf_reco = "Strong Buy" if pf100 >= 85 else ("Buy" if pf100 >= 65 else ("Hold" if pf100 >= 45 else ("Sell" if pf100 >= 30 else "Strong Sell")))

    st.subheader("Portfolio rating")
    k1, k2, k3 = st.columns(3)
    with k1: st.markdown("**Composite**"); st.markdown(f"<div class='kpi kpi-big'>{pf_score:.3f}</div>", unsafe_allow_html=True)
    with k2: st.markdown("**Score (0–100)**"); st.markdown(f"<div class='kpi kpi-big'>{pf100}</div>", unsafe_allow_html=True)
    with k3: st.markdown("**Recommendation**"); st.markdown(f"<div class='kpi kpi-big'>{pf_reco}</div>", unsafe_allow_html=True)

    # Weighted table
    table = rated.copy()
    table["Weight × Comp"] = table["Weight"] * table["Composite"]
    table = table.sort_values("Weight", ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Diversification breakdown (sectors + correlations)
    st.subheader("Diversification")
    secs = fdf.reindex(dfh["Ticker"]).get("sector")
    sec_pct = (dfh.set_index("Ticker")["Percent (%)"] / 100.0).groupby(secs).sum().sort_values(ascending=False)
    st.write("**Sector mix** (by portfolio weight)")
    st.bar_chart(sec_pct)

    # Correlation diversity (pairwise correlation of price returns)
    ret = px[dfh["Ticker"].tolist()].pct_change().dropna(how="all")
    if not ret.empty and ret.shape[1] >= 2:
        c = ret.corr().values
        upper = c[np.triu_indices_from(c, k=1)]
        avg_corr = float(np.nanmean(upper)) if upper.size else np.nan
    else:
        avg_corr = np.nan

    st.write(f"- Average pairwise correlation: **{avg_corr:.2f}**" if pd.notna(avg_corr) else "- Not enough data to compute correlations.")

    # Macro banner
    st.markdown(f"<div class='banner'>Peers loaded: {max(0, len(peers)-len(dfh))} | Benchmark: SPY | Peer set: S&P 500</div>", unsafe_allow_html=True)

    # Download portfolio details
    out = table.copy()
    out.insert(1, "Percent (%)", (out["Weight"] * 100).round(2))
    out = out.drop(columns=["Weight"])
    csv = out.to_csv(index=False).encode()
    st.download_button("Download portfolio breakdown (CSV)", csv, file_name="portfolio_rating.csv")


# -----------------------------------------
# --------------- Router ------------------
# -----------------------------------------

def app_router():
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    page = st.session_state.page
    if page == "landing":
        page_landing()
    elif page == "stock":
        if st.button("← Back", use_container_width=False): 
            st.session_state.page = "landing"; st.rerun()
        page_stock()
    elif page == "portfolio":
        if st.button("← Back", use_container_width=False): 
            st.session_state.page = "landing"; st.rerun()
        page_portfolio()
    else:
        st.session_state.page = "landing"
        page_landing()


if __name__ == "__main__":
    app_router()