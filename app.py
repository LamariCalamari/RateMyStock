# app.py — Rate My (centered header + richer explanations)
# -----------------------------------------------------------------------------
# What’s new vs prior build:
# • Centered header: big inline triangle logo + large title, perfectly centered
# • Landing page centered and cleaner
# • Explanations rewritten: fundamentals, technicals, macro now provide
#   concrete, percentile-based narratives with “what it means” summaries
# • All previous features retained (peers, charts, portfolio editor, exports)
# -----------------------------------------------------------------------------

import io
import math
import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from io import BytesIO
from PIL import Image, ImageDraw

# ----------------------------- Page & CSS ------------------------------------
st.set_page_config(page_title="Rate My", layout="wide")

st.markdown("""
<style>
/* Center the whole app a bit tighter */
.block-container { max-width: 1140px; }

/* Centered header with inline logo + title */
.header-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 18px;
  margin: .75rem 0 1.25rem;
}
.header-title {
  font-size: 3.1rem;
  font-weight: 850;
  line-height: 1.1;
  margin: 0;
}
.header-sub {
  text-align: center;
  color: #9aa0a6;
  margin-top: .35rem;
  font-size: 1.05rem;
}

/* Centered landing CTA row */
.landing-cta {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 22px;
  margin-top: 18px;
}

/* KPI chips */
.kpi { padding:1rem 1.1rem; border-radius:12px; background:#111418; border:1px solid #222; }
.kpi-big { font-size:2.2rem; font-weight:800; margin-top:.25rem; }
.small { color:#9aa0a6; font-size:.92rem; }

/* Banner */
.banner { background:#0c2f22; color:#cdebdc; border-radius:10px; padding:.9rem 1.1rem; margin:.75rem 0 1.25rem; }

/* Top-right back */
.topbar { display:flex; justify-content:flex-end; margin:.2rem 0 .8rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------- Session State ------------------------------------
for k, v in {"entered": False, "mode": None, "grid_df": None}.items():
    st.session_state.setdefault(k, v)

def go_home():
    st.session_state.update({"entered": False, "mode": None})

def enter(mode):
    st.session_state.update({"entered": True, "mode": mode})

# ----------------------------- Helpers ---------------------------------------
def yf_symbol(t: str) -> str:
    if not isinstance(t, str): return t
    return t.strip().upper().replace(".", "-")

def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up, dn = d.clip(lower=0.0), -d.clip(upper=0.0)
    rs = up.ewm(alpha=1/window, adjust=False).mean() / dn.ewm(alpha=1/window, adjust=False).mean().replace(0,np.nan)
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

def _z_to_pct(z): 
    if pd.isna(z): return np.nan
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * 100.0

# --------------------- Inline Triangle Logo (large, clean) -------------------
def _make_logo_png_bytes(size: int = 112) -> bytes:
    # draw a filled triangle with a smooth red→orange→green gradient bottom→top
    w = int(size * 1.1); h = size
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    top = (w//2, 4); left=(6,h-4); right=(w-6,h-4)

    def inside_tri(x, y):
        x1,y1=top; x2,y2=left; x3,y3=right
        det = (y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)
        if det==0: return False
        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
        c = 1 - a - b
        return (a>=0) and (b>=0) and (c>=0)

    for y in range(h):
        t = y/(h-1)
        # red at bottom → orange → green at top
        if t < 0.5:
            u = (0.5 - t)/0.5
            r = int(231*(1-u) + 46*u); g = int(76*(1-u) + 204*u); b = int(60*(1-u) + 113*u)
        else:
            u = (t-0.5)/0.5
            r = int(231*(1-u) + 243*u); g = int(76*(1-u) + 156*u); b = int(60*(1-u) + 18*u)
        for x in range(w):
            if inside_tri(x,y): draw.point((x,y), fill=(r,g,b,255))
    buf = BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()

_LOGO_BYTES = _make_logo_png_bytes(112)  # large

def render_header_centered(title: str, subtitle: str = ""):
    st.markdown(
        """
        <div class="header-wrap">
          <img src="data:image/png;base64,{}" width="112" height="112">
          <div>
            <div class="header-title">{}</div>
          </div>
        </div>
        """.format(
            _LOGO_BYTES.hex() if False else _LOGO_BYTES.encode("latin1").hex(),  # (placeholder kept; Streamlit ignores)
            title
        ),
        unsafe_allow_html=True,
    )
    # Streamlit doesn't allow base64 via hex trick; use st.image next to markdown:
    cols = st.columns([1,1,1])
    with cols[1]:
        # Real centered header (image + title)
        st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
        st.image(_LOGO_BYTES, width=112)
        st.markdown(f"<div class='header-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<div class='header-sub'>{subtitle}</div>", unsafe_allow_html=True)

def topbar_back(key):
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    st.button("← Back", key=key, on_click=go_home)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- Data Fetch (robust) -------------------------------
@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(
    tickers, period="1y", interval="1d",
    chunk=30, retries=4, sleep_between=0.45, singles_pause=0.30,
    hard_limit=350
):
    tickers = [yf_symbol(t) for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))[:hard_limit]
    frames, ok = [], []

    def _append(df, group):
        got = set(df.columns.get_level_values(0))
        for t in group:
            if t in got:
                s = df[t]["Close"].dropna()
                if s.size: frames.append(s.rename(t)); ok.append(t)

    # bulk
    for i in range(0, len(tickers), chunk):
        group = tickers[i:i+chunk]
        try:
            df = yf.download(group, period=period, interval=interval,
                             auto_adjust=True, group_by="ticker",
                             threads=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                _append(df, group)
            else:
                t = group[0]
                if "Close" in df:
                    s = df["Close"].dropna()
                    if s.size: frames.append(s.rename(t)); ok.append(t)
        except Exception:
            pass
        time.sleep(sleep_between + random.uniform(0,0.15))

    # singles fallback
    seen=set(ok)
    missing=[t for t in tickers if t not in seen]
    for _ in range(retries):
        new=[]
        for t in missing:
            try:
                df=yf.download(t, period=period, interval=interval,
                               auto_adjust=True, group_by="ticker",
                               threads=False, progress=False)
                if "Close" in df:
                    s=df["Close"].dropna()
                    if s.size: frames.append(s.rename(t)); ok.append(t)
                    else: new.append(t)
                else: new.append(t)
            except Exception:
                new.append(t)
            time.sleep(singles_pause + random.uniform(0,0.25))
        if not new: break
        missing=new

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return prices, ok

@st.cache_data(show_spinner=False)
def fetch_vix_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("^VIX").history(period=period, interval=interval)
        if not df.empty: return df["Close"].rename("^VIX")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_fundamentals_simple(tickers):
    keep=["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
          "grossMargins","operatingMargins","ebitdaMargins",
          "trailingPE","forwardPE","debtToEquity"]
    rows=[]
    for raw in tickers:
        t=yf_symbol(raw)
        try: info=yf.Ticker(t).info or {}
        except Exception: info={}
        row={"ticker":t}
        for k in keep:
            try: row[k]=float(info.get(k, np.nan))
            except Exception: row[k]=np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")

# -------------------------- Universes ----------------------------------------
SP500_FALLBACK = ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","BRK-B","UNH","LLY","V","JPM"]
DOW30_FALLBACK = ["AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","DIS","HD","INTC","IBM","KO","MCD","NKE","TRV","VZ","CSCO"]
NASDAQ100_FALLBACK = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","NFLX","CSCO","AMD"]

def list_sp500():
    try:
        got = {yf_symbol(t) for t in yf.tickers_sp500()}
        if got: return got
    except Exception: pass
    return set(SP500_FALLBACK)

def list_dow30():
    try:
        got = {yf_symbol(t) for t in yf.tickers_dow()}
        if got: return got
    except Exception: pass
    return set(DOW30_FALLBACK)

def list_nasdaq100():
    try:
        if hasattr(yf,"tickers_nasdaq"):
            got = {yf_symbol(t) for t in yf.tickers_nasdaq()}
            if got: return got
    except Exception: pass
    return set(NASDAQ100_FALLBACK)

def build_universe(user_tickers, mode, sample_n=180, custom_raw=""):
    user = [yf_symbol(t) for t in user_tickers]
    if mode == "S&P 500": peers_all=list_sp500(); label="S&P 500"
    elif mode == "Dow 30": peers_all=list_dow30(); label="Dow 30"
    elif mode == "NASDAQ 100": peers_all=list_nasdaq100(); label="NASDAQ 100"
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

# --------------------------- Signals -----------------------------------------
def technical_scores(price_panel: dict) -> pd.DataFrame:
    rows=[]
    for ticker, px in price_panel.items():
        px=px.dropna()
        if len(px)<60: continue
        ema50=ema(px,50)
        base50=ema50.iloc[-1] if pd.notna(ema50.iloc[-1]) and ema50.iloc[-1]!=0 else np.nan
        dma_gap=(px.iloc[-1]-ema50.iloc[-1])/base50 if pd.notna(base50) else np.nan
        _,_,hist=macd(px); macd_hist = hist.iloc[-1] if len(hist)>0 else np.nan
        r = rsi(px).iloc[-1] if len(px)>14 else np.nan
        rsi_strength = (r-50.0)/50.0 if pd.notna(r) else np.nan
        mom = np.nan
        if len(px)>252:
            try: mom = px.iloc[-1]/px.iloc[-253]-1.0
            except Exception: mom=np.nan
        rows.append({"ticker":ticker,"dma_gap":dma_gap,"macd_hist":macd_hist,
                     "rsi_strength":rsi_strength,"mom12m":mom})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

def macro_from_vix(vix_series: pd.Series):
    if vix_series is None or vix_series.empty: return 0.5, np.nan, np.nan, np.nan
    vix_last=float(vix_series.iloc[-1])
    ema20=float(ema(vix_series,20).iloc[-1]) if len(vix_series)>=20 else vix_last
    gap=(vix_last-ema20)/max(ema20,1e-9)
    if vix_last<=12: level=1.0
    elif vix_last>=28: level=0.0
    else: level=1.0-(vix_last-12)/16.0
    if   gap>=0.03: trend=0.0
    elif gap<=-0.03: trend=1.0
    else:
        trend = 1.0-(gap+0.03)/0.06
        trend = float(np.clip(trend,0,1))
    macro=float(np.clip(0.70*level+0.30*trend,0,1))
    return macro, vix_last, ema20, gap

# ---------------------- Narratives (richer text) -----------------------------
def _bucket_from_z(z):
    if pd.isna(z): return "N/A"
    if z >= 1.2: return "top-tier (≈90th+ pct)"
    if z >= 0.7: return "strong (≈75th+ pct)"
    if z >= 0.3: return "above average"
    if z <= -1.0: return "bottom-tier (≈15th pct or less)"
    if z <= -0.5: return "below average"
    return "in line with peers"

def fundamentals_story(zrow: pd.Series) -> str:
    def one(label, key, invert=False):
        z=zrow.get(key)
        if pd.isna(z): return f"- {label}: not available."
        pct=_z_to_pct(z)
        tone=_bucket_from_z(z)
        better="higher" if not invert else "lower"
        return f"- **{label}**: {tone} ({z:+.2f}, ~{pct:.0f}th pct). Typically, {better} is better for this metric."

    parts=[
        one("Revenue growth", "revenueGrowth_z"),
        one("Earnings growth", "earningsGrowth_z"),
        one("Return on equity", "returnOnEquity_z"),
        one("Profit margin", "profitMargins_z"),
        one("Gross margin", "grossMargins_z"),
        one("Operating margin", "operatingMargins_z"),
        one("EBITDA margin", "ebitdaMargins_z"),
        one("Valuation (forward PE; lower is better)", "forwardPE_z", invert=True),
        one("Valuation (trailing PE; lower is better)", "trailingPE_z", invert=True),
        one("Leverage (Debt/Equity; lower is better)", "debtToEquity_z", invert=True),
    ]

    # high-level conclusion
    strong = [k for k in zrow.index if k.endswith("_z") and not pd.isna(zrow[k]) and zrow[k]>=0.6]
    weak   = [k for k in zrow.index if k.endswith("_z") and not pd.isna(zrow[k]) and zrow[k]<=-0.6]
    if strong and not weak:
        parts.append("→ **Summary**: broad-based fundamental strength vs peers with supportive valuation/quality tilt.")
    elif weak and not strong:
        parts.append("→ **Summary**: fundamentals trail peers; watch margins, growth durability and valuation risk.")
    else:
        parts.append("→ **Summary**: a mix of positives and negatives; overall neutral to slightly mixed.")
    return "\n".join(parts)

def technicals_story(trow: pd.Series) -> str:
    pts=[]
    # EMA gap
    g=trow.get("dma_gap")
    if pd.notna(g):
        gp=g*100
        if gp>2: pts.append(f"- Price is **{gp:.1f}% above** its 50-day EMA → trend tailwind.")
        elif gp<-2: pts.append(f"- Price is **{abs(gp):.1f}% below** its 50-day EMA → trend headwind.")
        else: pts.append(f"- Price is **near** its 50-day EMA (±2%) → neutral trend reference.")
    # MACD
    mh=trow.get("macd_hist")
    if pd.notna(mh):
        pts.append("- MACD histogram **positive** → momentum building." if mh>0 else
                   "- MACD histogram **negative** → momentum fading.")
    # RSI
    r = trow.get("rsi_strength")
    if pd.notna(r):
        rsi = 50+50*r
        if rsi>=70: pts.append(f"- RSI ≈ **{rsi:.0f}** → overbought/strong (short-term risk of pullback).")
        elif rsi<=30: pts.append(f"- RSI ≈ **{rsi:.0f}** → oversold/weak (potential mean-reversion).")
        else: pts.append(f"- RSI ≈ **{rsi:.0f}** → neutral.")
    # 12m momentum
    m=trow.get("mom12m")
    if pd.notna(m): pts.append(f"- 12-month momentum: **{m*100:+.1f}%** vs one year ago.")
    if not pts: pts.append("- Not enough data for technical commentary.")
    return "\n".join(pts)

def macro_story(vix_last, vix_ema20, vix_gap) -> str:
    if np.isnan(vix_last): return "VIX unavailable; treating macro as neutral."
    if vix_last<=13: lvl="very calm (risk-friendly)"
    elif vix_last<=18: lvl="calm (supportive)"
    elif vix_last<=24: lvl="elevated (caution)"
    else: lvl="stressed (risk-off)"
    if vix_gap>0.03: tr="**rising above** its 20-day average (volatility building → risk-off headwind)."
    elif vix_gap<-0.03: tr="**falling below** its 20-day average (volatility easing → risk-on tailwind)."
    else: tr="**near** its 20-day average (neutral short-term trend)."
    return f"- **Level**: {lvl}.  \n- **Trend**: {tr}"

# ---------------------------- Charts -----------------------------------------
def draw_stock_charts(t: str, series: pd.Series):
    if series is None or series.empty:
        st.info("Not enough history to show charts."); return
    st.subheader("📈 Price & EMAs")
    e20,e50=ema(series,20), ema(series,50)
    st.line_chart(pd.DataFrame({"Close":series,"EMA20":e20,"EMA50":e50}), use_container_width=True)
    st.caption("Above EMA50/EMA20 → trend tailwind; below → headwind.")

    st.subheader("📉 MACD")
    line,sig,hist=macd(series)
    st.line_chart(pd.DataFrame({"MACD line":line,"Signal":sig}), use_container_width=True)
    st.bar_chart(pd.DataFrame({"Histogram":hist}), use_container_width=True)
    st.caption("Positive & rising histogram → momentum building; below zero & falling → momentum fading.")

    st.subheader("🔁 RSI (14)")
    st.line_chart(pd.DataFrame({"RSI(14)": rsi(series)}), use_container_width=True)
    st.caption(">70 overbought • <30 oversold • ~50 neutral.")

    st.subheader("🚀 12-month momentum")
    if len(series)>252:
        mom12=series/series.shift(253)-1.0
        st.line_chart(pd.DataFrame({"12m momentum": mom12}), use_container_width=True)
        st.caption("Positive vs one year ago → outperformance; negative → underperformance.")
    else:
        st.info("Need > 1y to show the 12-month momentum line.")

# --------------------------- STOCK PAGE --------------------------------------
def app_stock():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    st.button("← Back", key="back_stock", on_click=go_home)
    st.markdown('</div>', unsafe_allow_html=True)

    render_header_centered("Rate My Stock")

    center=st.columns([1,2,1])[1]
    with center:
        ticker = st.text_input(" ", "AAPL", label_visibility="collapsed",
                               placeholder="Type a ticker (e.g., AAPL)")

    with st.expander("Advanced settings", expanded=False):
        c1,c2,c3=st.columns(3)
        with c1:
            universe_mode = st.selectbox("Peer universe",
                ["Auto by index membership","S&P 500","Dow 30","NASDAQ 100","Custom (paste list)"], index=0)
        with c2: peer_n = st.slider("Peer sample size", 30, 300, 180, 10)
        with c3: history = st.selectbox("History for signals", ["1y","2y"], index=0)
        c4,c5,c6=st.columns(3)
        with c4: w_f = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.50, 0.05)
        with c5: w_t = st.slider("Weight: Technicals",   0.0, 1.0, 0.40, 0.05)
        with c6: w_m = st.slider("Weight: Macro (VIX)",  0.0, 1.0, 0.10, 0.05)
        custom_raw = st.text_area("Custom peers (comma-separated)", "") \
                      if universe_mode=="Custom (paste list)" else ""

    user_tickers = [yf_symbol(x) for x in ticker.split(",") if x.strip()]
    if not user_tickers:
        st.info("Enter a ticker above to run the rating."); return

    with st.status("Crunching the numbers…", expanded=True) as status:
        p=st.progress(0)
        status.update(label="Building peer universe…")
        universe,label = build_universe(user_tickers, universe_mode, peer_n, custom_raw)
        target_count = peer_n if universe_mode!="Custom (paste list)" else len(universe); p.progress(10)

        status.update(label="Downloading prices (chunked + retries)…")
        prices, ok = fetch_prices_chunked_with_fallback(universe, period=history, interval="1d")
        if not ok: st.error("No peer prices loaded."); return
        panel={t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size>0}; p.progress(50)

        status.update(label="Computing technicals…")
        tech=technical_scores(panel)
        for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
            if col in tech.columns: tech[f"{col}_z"]=zscore_series(tech[col])
        TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech.columns]].mean(axis=1)
        p.progress(75)

        status.update(label="Fetching fundamentals…")
        f_raw=fetch_fundamentals_simple(list(panel.keys()))
        fdf=pd.DataFrame(index=f_raw.index)
        for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
                    "grossMargins","operatingMargins","ebitdaMargins"]:
            if col in f_raw.columns: fdf[f"{col}_z"]=zscore_series(f_raw[col])
        for col in ["trailingPE","forwardPE","debtToEquity"]:
            if col in f_raw.columns: fdf[f"{col}_z"]=zscore_series(-f_raw[col])  # invert
        FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=f_raw.index); p.progress(92)

        status.update(label="Assessing macro regime…")
        vix_series=fetch_vix_series(period="6mo", interval="1d")
        MACRO, vix_last, vix_ema20, vix_gap = macro_from_vix(vix_series); p.progress(100)
        status.update(label="Done!", state="complete")

    st.markdown(
        f'<div class="banner">Peers loaded: <b>{len(panel)}</b> / <b>{target_count}</b> '
        f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True
    )

    idx=pd.Index(list(panel.keys()))
    out=pd.DataFrame(index=idx)
    out["FUND_score"]=FUND_score.reindex(idx).fillna(0.0)
    out["TECH_score"]=TECH_score.reindex(idx).fillna(0.0)
    out["MACRO_score"]=MACRO
    wsum=(w_f+w_t+w_m) or 1.0; wf,wt,wm=w_f/wsum,w_t/wsum,w_m/wsum
    out["COMPOSITE"]=wf*out["FUND_score"]+wt*out["TECH_score"]+wm*out["MACRO_score"]
    out["RATING_0_100"]=percentile_rank(out["COMPOSITE"])
    out["RECO"]=out["RATING_0_100"].apply(
        lambda x: "Strong Buy" if x>=80 else "Buy" if x>=60 else "Hold" if x>=40 else "Sell" if x>=20 else "Strong Sell"
    )

    # 5y momentum for just-shown names
    show_idx=[t for t in user_tickers if t in out.index]
    for t in show_idx:
        try:
            px5=yf.Ticker(t).history(period="5y", interval="1d")["Close"].dropna()
            if len(px5)>253:
                mom12=px5.iloc[-1]/px5.iloc[-253]-1.0
                if t in tech.index: tech.loc[t,"mom12m"]=mom12
        except Exception:
            pass

    st.markdown("## 🏁 Ratings")
    table=out.reindex(show_idx).sort_values("RATING_0_100", ascending=False)
    pretty=table.rename(columns={"FUND_score":"Fundamentals","TECH_score":"Technicals","MACRO_score":"Macro (VIX)",
                                 "COMPOSITE":"Composite","RATING_0_100":"Score (0–100)","RECO":"Recommendation"})
    st.dataframe(pretty.round(4), use_container_width=True)

    st.markdown("## 🔎 Why this rating?")
    rows=[]
    for t in show_idx:
        reco=table.loc[t,"RECO"]; sc=table.loc[t,"RATING_0_100"]
        with st.expander(f"{t} — {reco} (Score: {sc:.1f})", expanded=True):
            c1,c2,c3=st.columns(3)
            c1.markdown(f'<div class="kpi"><div>Fundamentals</div><div class="kpi-big">{table.loc[t,"FUND_score"]:.3f}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="kpi"><div>Technicals</div><div class="kpi-big">{table.loc[t,"TECH_score"]:.3f}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="kpi"><div>Macro (VIX)</div><div class="kpi-big">{table.loc[t,"MACRO_score"]:.3f}</div></div>', unsafe_allow_html=True)

            st.markdown("#### Fundamentals — peer-relative details")
            if t in fdf.index:
                st.markdown(fundamentals_story(fdf.loc[t]), unsafe_allow_html=True)
            else:
                st.info("Not enough peer fundamentals to show details.")

            st.markdown("#### Technicals")
            if t in tech.index:
                st.markdown(technicals_story(tech.loc[t]))
            else:
                st.info("Not enough price history for technical commentary.")

            st.markdown("#### Macro (VIX) — what it means")
            st.markdown(macro_story(vix_last, vix_ema20, vix_gap))

            row={"ticker":t,
                 "fundamentals_score":float(table.loc[t,"FUND_score"]),
                 "technicals_score":float(table.loc[t,"TECH_score"]),
                 "macro_score":float(table.loc[t,"MACRO_score"]),
                 "composite":float(table.loc[t,"COMPOSITE"]),
                 "score_0_100":float(table.loc[t,"RATING_0_100"]),
                 "recommendation":str(table.loc[t,"RECO"])}
            rows.append(row)

            # Charts (2y for speed; momentum pulls 5y quietly above)
            try:
                px2=yf.Ticker(t).history(period="2y", interval="1d")["Close"].dropna()
                draw_stock_charts(t, px2 if px2.size else panel.get(t))
            except Exception:
                if t in panel: draw_stock_charts(t, panel[t])

    if rows:
        df_all=pd.DataFrame(rows)
        st.markdown("### Export")
        st.download_button("⬇️ Download shown breakdowns (CSV)",
                           data=df_all.to_csv(index=False).encode(),
                           file_name="stock_breakdowns.csv", mime="text/csv", use_container_width=True)

# --------------------------- PORTFOLIO PAGE ----------------------------------
CURRENCY_MAP={"$":"USD","€":"EUR","£":"GBP","CHF":"CHF","C$":"CAD","A$":"AUD","¥":"JPY"}
def _num(x): return pd.to_numeric(x, errors="coerce")
def normalize_100(p: pd.Series) -> pd.Series:
    p=_num(p).fillna(0.0); s=p.sum(); return p if s<=0 else (p/s*100.0)

def sync_percent_amount(df: pd.DataFrame, total: float, mode: str) -> pd.DataFrame:
    df=df.copy(); df["Ticker"]=df["Ticker"].astype(str).str.strip()
    df=df[df["Ticker"].astype(bool)].reset_index(drop=True)
    if df.empty: df["weight"]=[]; return df
    df["Percent (%)"]=_num(df.get("Percent (%)"))
    df["Amount"]=_num(df.get("Amount"))
    has_total = (total is not None and total>0)

    if has_total:
        if mode=="percent":
            if df["Percent (%)"].fillna(0).sum()==0: df["Percent (%)"]=100.0/len(df)
            df["Percent (%)"]=normalize_100(df["Percent (%)"]).round(2)
            df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
        else:
            s=df["Amount"].fillna(0).sum()
            if s>0:
                df["Percent (%)"]=(df["Amount"]/total*100.0).round(2)
                df["Percent (%)"]=normalize_100(df["Percent (%)"]).round(2)
                df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
            else:
                df["Percent (%)"]=100.0/len(df); df["Amount"]=(df["Percent (%)"]/100.0*total).round(2)
    else:
        if df["Percent (%)"].fillna(0).sum()==0: df["Percent (%)"]=100.0/len(df)
        df["Percent (%)"]=normalize_100(df["Percent (%)"]).round(2)

    if has_total and df["Amount"].fillna(0).sum()>0:
        w=df["Amount"].fillna(0)/df["Amount"].fillna(0).sum()
    elif df["Percent (%)"].fillna(0).sum()>0:
        w=df["Percent (%)"].fillna(0)/df["Percent (%)"].fillna(0).sum()
    else:
        w=pd.Series([1.0/len(df)]*len(df), index=df.index)
    df["weight"]=w
    return df

def holdings_editor_form(currency_symbol, total_value):
    if st.session_state.get("grid_df") is None:
        st.session_state["grid_df"]=pd.DataFrame({
            "Ticker":["AAPL","MSFT","NVDA","AMZN"],
            "Percent (%)":[25.0,25.0,25.0,25.0],
            "Amount":[np.nan,np.nan,np.nan,np.nan],
        })
    st.markdown(
        f"**Holdings**  \n"
        f"<span class='small'>Enter <b>Ticker</b> and either <b>Percent (%)</b> or "
        f"<b>Amount ({currency_symbol})</b>. Values update only when you click <b>Apply changes</b>. "
        f"Use <b>Normalize</b> to force exactly 100% in percent mode.</span>",
        unsafe_allow_html=True,
    )
    committed=st.session_state["grid_df"].copy()
    with st.form("holdings_form", clear_on_submit=False):
        sync_mode = st.segmented_control("Sync mode",
                    options=["Percent → Amount","Amount → Percent"],
                    default="Percent → Amount",
                    help="Choose which side drives on Apply.")
        mode_key = {"Percent → Amount":"percent","Amount → Percent":"amount"}[sync_mode]
        edited = st.data_editor(
            committed, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn(width="small"),
                "Percent (%)": st.column_config.NumberColumn(format="%.2f"),
                "Amount": st.column_config.NumberColumn(format="%.2f", help=f"Amount in {currency_symbol}"),
            }, key="grid_form"
        )
        col_a,col_b=st.columns([1,1])
        apply_btn = col_a.form_submit_button("Apply changes", type="primary", use_container_width=True)
        normalize_btn = col_b.form_submit_button("Normalize to 100% (percent mode)", use_container_width=True)

    if normalize_btn:
        syncd=edited.copy()
        syncd["Percent (%)"]=normalize_100(_num(syncd.get("Percent (%)")))
        if total_value and total_value>0:
            syncd["Amount"]=(syncd["Percent (%)"]/100.0*total_value).round(2)
        st.session_state["grid_df"]=syncd[["Ticker","Percent (%)","Amount"]]
    elif apply_btn:
        syncd=sync_percent_amount(edited.copy(), total_value, mode_key)
        st.session_state["grid_df"]=syncd[["Ticker","Percent (%)","Amount"]]

    current=st.session_state["grid_df"].copy()
    out=current.copy(); out["ticker"]=out["Ticker"].map(yf_symbol)
    out=out[out["ticker"].astype(bool)]
    if total_value and total_value>0 and _num(out["Amount"]).sum()>0:
        w=_num(out["Amount"])/_num(out["Amount"]).sum()
    elif _num(out["Percent (%)"]).sum()>0:
        w=_num(out["Percent (%)"])/_num(out["Percent (%)"]).sum()
    else:
        n=max(len(out),1); w=pd.Series([1.0/n]*n, index=out.index)
    df_hold=pd.DataFrame({"ticker":out["ticker"],"weight":w})
    return df_hold, current

def app_portfolio():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    st.button("← Back", key="back_port", on_click=go_home)
    st.markdown('</div>', unsafe_allow_html=True)

    render_header_centered("Rate My Portfolio")

    t1,t2,t3=st.columns([1,1,1])
    with t1: cur = st.selectbox("Currency", list(CURRENCY_MAP.keys()), index=0)
    with t2: total = st.number_input(f"Total portfolio value ({cur})", min_value=0.0, value=10000.0, step=500.0)
    with t3: st.caption("Holdings update only when you click **Apply changes**.")

    df_hold, grid_view = holdings_editor_form(cur, total)
    if df_hold.empty: st.info("Add at least one holding to run the rating."); return
    tickers=df_hold["ticker"].tolist()

    with st.expander("Advanced settings", expanded=False):
        c1,c2,c3=st.columns(3)
        with c1:
            universe_mode=st.selectbox("Peer universe",
                ["Auto by index membership","S&P 500","Dow 30","NASDAQ 100","Custom (paste list)"], index=0)
        with c2: peer_n=st.slider("Peer sample size", 30, 300, 180, 10)
        with c3: history=st.selectbox("History for signals", ["1y","2y"], index=0)
        c4,c5,c6,c7=st.columns(4)
        with c4: w_f=st.slider("Weight: Fundamentals", 0.0, 1.0, 0.45, 0.05)
        with c5: w_t=st.slider("Weight: Technicals",   0.0, 1.0, 0.40, 0.05)
        with c6: w_m=st.slider("Weight: Macro (VIX)",  0.0, 1.0, 0.10, 0.05)
        with c7: w_d=st.slider("Weight: Diversification", 0.0, 1.0, 0.05, 0.05)
        custom_raw = st.text_area("Custom peers (comma-separated)", "") \
                      if universe_mode=="Custom (paste list)" else ""

    with st.status("Crunching the numbers…", expanded=True) as status:
        p=st.progress(0)
        status.update(label="Building peer universe…")
        universe,label = build_universe(tickers, universe_mode, peer_n, custom_raw)
        target_count = peer_n if universe_mode!="Custom (paste list)" else len(universe); p.progress(8)

        status.update(label="Downloading prices (chunked + retries)…")
        prices, ok = fetch_prices_chunked_with_fallback(universe, period=history, interval="1d")
        if prices.empty: st.error("No prices fetched."); return
        panel_all={t: prices[t].dropna() for t in prices.columns if t in prices.columns and prices[t].dropna().size>0}; p.progress(40)

        status.update(label="Computing technicals…")
        tech_all=technical_scores(panel_all)
        for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
            if col in tech_all.columns: tech_all[f"{col}_z"]=zscore_series(tech_all[col])
        TECH_score_all=tech_all[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech_all.columns]].mean(axis=1); p.progress(65)

        status.update(label="Fetching fundamentals…")
        f_raw_all=fetch_fundamentals_simple(list(panel_all.keys()))
        fdf_all=pd.DataFrame(index=f_raw_all.index)
        for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
                    "grossMargins","operatingMargins","ebitdaMargins"]:
            if col in f_raw_all.columns: fdf_all[f"{col}_z"]=zscore_series(f_raw_all[col])
        for col in ["trailingPE","forwardPE","debtToEquity"]:
            if col in f_raw_all.columns: fdf_all[f"{col}_z"]=zscore_series(-f_raw_all[col])
        FUND_score_all=fdf_all.mean(axis=1) if len(fdf_all.columns) else pd.Series(0.0, index=f_raw_all.index); p.progress(85)

        status.update(label="Assessing macro regime…")
        vix_series=fetch_vix_series(period="6mo", interval="1d")
        MACRO,_,_,_=macro_from_vix(vix_series); p.progress(100)
        status.update(label="Done!", state="complete")

    st.markdown(
        f'<div class="banner">Peers loaded: <b>{len(panel_all)}</b> / <b>{target_count}</b> '
        f'&nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True
    )

    idx_all=pd.Index(list(panel_all.keys()))
    out_all=pd.DataFrame(index=idx_all)
    out_all["FUND_score"]=FUND_score_all.reindex(idx_all).fillna(0.0)
    out_all["TECH_score"]=TECH_score_all.reindex(idx_all).fillna(0.0)
    out_all["MACRO_score"]=MACRO
    wsum=(w_f+w_t+w_m) or 1.0; wf,wt,wm=w_f/wsum,w_t/wsum,w_m/wsum
    out_all["COMPOSITE"]=wf*out_all["FUND_score"]+wt*out_all["TECH_score"]+wm*out_all["MACRO_score"]
    out_all["RATING_0_100"]=percentile_rank(out_all["COMPOSITE"])

    # diversification (sectors + corr + name concentration)
    def fetch_sector(t):
        try: return yf.Ticker(t).info.get("sector", None)
        except Exception: return None
    weights=df_hold.set_index("ticker")["weight"]
    meta_sec=pd.Series({t:fetch_sector(t) for t in tickers})
    sec_mix=weights.groupby(meta_sec).sum()
    if sec_mix.empty: sec_mix=pd.Series({"Unknown":1.0})
    hhi=float((sec_mix**2).sum()); effN=1.0/hhi if hhi>0 else 1.0
    targetN=min(10, max(1,len(sec_mix))); sector_div=float(np.clip((effN-1)/(targetN-1 if targetN>1 else 1),0,1))
    max_w=float(weights.max())
    if   max_w<=0.10: name_div=1.0
    elif max_w>=0.40: name_div=0.0
    else: name_div=float((0.40-max_w)/0.30)
    ret=prices[tickers].pct_change().dropna(how="all")
    if ret.shape[1]>=2:
        corr=ret.corr().values; n=corr.shape[0]
        avg_corr=(corr.sum()-np.trace(corr))/max(1,(n*n-n))
        corr_div = float(np.clip(1.0-max(0.0, avg_corr), 0.0, 1.0))
    else: avg_corr=np.nan; corr_div=0.5
    DIV=0.5*sector_div + 0.3*corr_div + 0.2{name_div}

    per_name=out_all.reindex(tickers).copy().join(weights, how="left")
    per_name["weighted_composite"]=per_name["COMPOSITE"]*per_name["weight"]
    port_signal=float(per_name["weighted_composite"].sum())
    total_for_final=1.0 + w_d
    port_final=(port_signal)*(1/total_for_final) + DIV*(w_d/total_for_final)
    port_score=float(np.clip((port_final+1)/2, 0, 1)*100)

    st.markdown("## 🧺 Portfolio — Scores")
    a,b,c,d=st.columns(4)
    a.metric("Portfolio Score (0–100)", f"{port_score:.1f}")
    b.metric("Signal (weighted composite)", f"{port_signal:.3f}")
    c.metric("Macro (VIX)", f"{MACRO:.3f}")
    d.metric("Diversification", f"{DIV:.3f}")

    with st.expander("Why this portfolio rating?"):
        show=per_name.rename(columns={"weight":"Weight","FUND_score":"Fundamentals",
                                      "TECH_score":"Technicals","MACRO_score":"Macro (VIX)",
                                      "COMPOSITE":"Composite","weighted_composite":"Weight × Comp"})[
            ["Weight","Fundamentals","Technicals","Macro (VIX)","Composite","Weight × Comp"]
        ]
        st.dataframe(show.round(4), use_container_width=True)
        st.markdown(
            f"- **Sector mix**: effective #sectors ≈ **{effN:.1f}** → score **{sector_div:.2f}**.  \n"
            f"- **Name concentration**: max weight ≈ **{max_w*100:.1f}%** → score **{name_div:.2f}**.  \n"
            f"- **Correlation**: average pairwise corr ≈ {('%.2f' % avg_corr) if not np.isnan(avg_corr) else 'N/A'} "
            f"→ score **{corr_div:.2f}**.  \n"
            f"- **Diversification score** = 50% sector + 30% correlation + 20% name concentration."
        )

    # Charts
    px_held=prices[tickers].dropna(how="all")
    r=px_held.pct_change().fillna(0)
    w_vec=weights.reindex(px_held.columns).fillna(0).values
    port_r=(r*w_vec).sum(axis=1)
    eq=(1+port_r).cumprod()
    tabs=st.tabs(["Cumulative","Volatility (60d) & Sharpe","Drawdown"])
    with tabs[0]:
        st.subheader("Cumulative growth (set = 1.0)")
        st.line_chart(pd.DataFrame({"Portfolio cumulative":eq}), use_container_width=True)
        st.caption("Growth of 1.0 invested, using your current weights over the chosen history.")
    with tabs[1]:
        st.subheader("Volatility & rolling Sharpe (60-day)")
        vol60=port_r.rolling(60).std()*np.sqrt(252)
        sharpe60=(port_r.rolling(60).mean()/port_r.rolling(60).std())*np.sqrt(252)
        st.line_chart(pd.DataFrame({"Volatility 60d (ann.)":vol60,"Sharpe 60d":sharpe60}), use_container_width=True)
        st.caption("Lower volatility & higher Sharpe are preferred.")
    with tabs[2]:
        st.subheader("Drawdown")
        roll_max=eq.cummax(); dd=eq/roll_max - 1
        st.line_chart(pd.DataFrame({"Drawdown":dd}), use_container_width=True)
        st.caption("Depth of falls from prior peaks (risk perspective).")

# --------------------------- Landing -----------------------------------------
def landing():
    render_header_centered(
        "Rate My",
        "Pick a stock or your entire portfolio — we’ll rate it with clear, friendly explanations and charts."
    )
    st.markdown('<div class="landing-cta">', unsafe_allow_html=True)
    st.button("📈 Rate My Stock", type="primary", use_container_width=True, on_click=enter, args=("stock",))
    st.button("💼 Rate My Portfolio", use_container_width=True, on_click=enter, args=("portfolio",))
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ Router ---------------------------------------
def app_router():
    if not st.session_state.entered: landing(); return
    if st.session_state.mode=="portfolio": app_portfolio()
    else: app_stock()

# ------------------------------- Run -----------------------------------------
app_router()