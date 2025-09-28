# app.py — ⭐ Rate My (Stock + Portfolio) — full build with charts, macro explanations, exports, and bug-fixed editor

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------- Page & Styles ---------------------
st.set_page_config(page_title="Rate My", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1140px;}
.hero {text-align:center; margin-top: 3.5rem; margin-bottom: .5rem;}
.sub {text-align:center; color:#9aa0a6; margin-bottom: 2rem;}
.small-muted {color:#9aa0a6; font-size:.9rem; text-align:center;}
.btn-wrap {display:flex; gap:18px; justify-content:center; margin: 1.25rem 0 2.4rem;}
.search-wrap {display:flex; justify-content:center; margin: .5rem 0 1rem;}
.search-inner {width: min(760px, 92%);}
.search-input input {border-radius: 9999px !important; padding: 1rem 1.25rem !important; font-size: 1.1rem;}
.banner {background:#0c2f22; color:#cdebdc; border-radius:10px; padding:.9rem 1.1rem; margin:.75rem 0 1.25rem;}
.topbar {display:flex; justify-content:flex-end; margin:.25rem 0 .5rem;}
.kpi-num {font-size:2.2rem; font-weight:700; margin-top:.25rem;}
.kpi-card {padding:.9rem 1rem; border-radius:12px; background:#111418; border:1px solid #222;}
</style>
""", unsafe_allow_html=True)

# -------------------- Session --------------------
for k, v in {"entered": False, "mode": None, "grid_df": None, "prev_grid_df": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go_home():
    st.session_state.entered = False
    st.session_state.mode = None

def enter(mode):
    st.session_state.entered = True
    st.session_state.mode = mode

# ------------------- Utilities -------------------
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

# -------------- Cached fetchers --------------
@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(tickers, period="1y", interval="1d",
                                       chunk=50, min_ok=50, sleep_between=0.2):
    tickers = [yf_symbol(t) for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))
    frames, ok = [], []
    for i in range(0, len(tickers), chunk):
        group = tickers[i:i+chunk]
        try:
            df = yf.download(group, period=period, interval=interval,
                             auto_adjust=True, group_by="ticker",
                             threads=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                got = set(df.columns.get_level_values(0))
                for t in group:
                    if t in got:
                        s = df[t]["Close"].dropna()
                        if s.size: frames.append(s.rename(t)); ok.append(t)
            else:
                t = group[0]
                if "Close" in df:
                    s = df["Close"].dropna()
                    if s.size: frames.append(s.rename(t)); ok.append(t)
        except Exception:
            pass
        time.sleep(sleep_between)
    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return prices, ok

@st.cache_data(show_spinner=False)
def fetch_vix_series(period="6mo", interval="1d") -> pd.Series:
    try:
        df = yf.Ticker("^VIX").history(period=period, interval=interval)
        if not df.empty:
            return df["Close"].rename("^VIX")
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_fundamentals_simple(tickers):
    keep = ["revenueGrowth","earningsGrowth","returnOnEquity",
            "profitMargins","grossMargins","operatingMargins","ebitdaMargins",
            "trailingPE","forwardPE","debtToEquity"]
    rows=[]
    for raw in tickers:
        t=yf_symbol(raw)
        try: info = yf.Ticker(t).info or {}
        except Exception: info={}
        row={"ticker":t}
        for k in keep:
            try: row[k]=float(info.get(k, np.nan))
            except Exception: row[k]=np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")

# -------------- Peer universes --------------
SP500_FALLBACK = ["AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","BRK-B","UNH","LLY","V","JPM"]
DOW30_FALLBACK = ["AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","DIS","HD","INTC","IBM","KO","MCD","NKE","TRV","VZ","CSCO"]
NASDAQ100_FALLBACK = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST","PEP","ADBE","NFLX","CSCO","AMD"]

def list_sp500():
    try:
        got = {yf_symbol(t) for t in yf.tickers_sp500()}
        if got: return got
    except Exception:
        pass
    return set(SP500_FALLBACK)

def list_dow30():
    try:
        got = {yf_symbol(t) for t in yf.tickers_dow()}
        if got: return got
    except Exception:
        pass
    return set(DOW30_FALLBACK)

def list_nasdaq100():
    try:
        if hasattr(yf,"tickers_nasdaq"):
            got = {yf_symbol(t) for t in yf.tickers_nasdaq()}
            if got: return got
    except Exception:
        pass
    return set(NASDAQ100_FALLBACK)

def build_universe(user_tickers, mode, sample_n=120, custom_raw=""):
    user = [yf_symbol(t) for t in user_tickers]
    if mode == "S&P 500":
        peers_all = list_sp500(); bench="SPY"; label="S&P 500"
    elif mode == "Dow 30":
        peers_all = list_dow30(); bench="DIA"; label="Dow 30"
    elif mode == "NASDAQ 100":
        peers_all = list_nasdaq100(); bench="QQQ"; label="NASDAQ 100"
    elif mode == "Custom (paste list)":
        custom = {yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
        return sorted(set(user)|custom), "Custom", "Custom"
    else:
        sp,dj,nd = list_sp500(), list_dow30(), list_nasdaq100()
        auto=set(); bench,label="SPY","S&P 500"
        if len(user)==1:
            t=user[0]
            if   t in sp: auto=sp; bench,label="SPY","S&P 500"
            elif t in dj: auto=dj; bench,label="DIA","Dow 30"
            elif t in nd: auto=nd; bench,label="QQQ","NASDAQ 100"
        else:
            for t in user:
                if t in sp: auto|=sp; bench,label="SPY","S&P 500"
                elif t in dj: auto|=dj; bench,label="DIA","Dow 30"
                elif t in nd: auto|=nd; bench,label="QQQ","NASDAQ 100"
        peers_all = auto if auto else sp
    peers = sorted(peers_all.difference(set(user)))[:sample_n]
    return sorted(set(user)|set(peers)), bench, label

# -------------- Feature builders --------------
def technical_scores(price_panel: dict) -> pd.DataFrame:
    rows=[]
    for ticker, px in price_panel.items():
        px=px.dropna()
        if len(px)<130: continue
        ema100=ema(px,100)
        base=ema100.iloc[-1] if pd.notna(ema100.iloc[-1]) and ema100.iloc[-1]!=0 else np.nan
        dma_gap=(px.iloc[-1]-ema100.iloc[-1])/base if pd.notna(base) else np.nan
        _,_,hist = macd(px)
        macd_hist = hist.iloc[-1]
        r = rsi(px).iloc[-1]
        rsi_strength = (r-50.0)/50.0
        mom12m = px.iloc[-1]/px.iloc[-253]-1.0 if len(px)>252 else np.nan
        rows.append({"ticker":ticker,"dma_gap":dma_gap,"macd_hist":macd_hist,
                     "rsi_strength":rsi_strength,"mom12m":mom12m})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

def macro_from_vix(vix_series: pd.Series):
    if vix_series is None or vix_series.empty:
        return 0.5, np.nan, np.nan, np.nan
    vix_last = float(vix_series.iloc[-1])
    ema20    = float(ema(vix_series,20).iloc[-1]) if len(vix_series)>=20 else vix_last
    rel_gap  = (vix_last-ema20)/max(ema20,1e-9)
    # Map level 12→1.0, 28→0.0 (clip)
    if   vix_last<=12: level=1.0
    elif vix_last>=28: level=0.0
    else: level = 1.0-(vix_last-12)/16.0
    # Map trend: above EMA hurts, below helps
    if   rel_gap>=0.30: trend=0.0
    elif rel_gap<=-0.10: trend=1.0
    else:
        trend = 1.0-(rel_gap+0.10)/0.40
        trend=float(np.clip(trend,0,1))
    macro=float(np.clip(0.70*level+0.30*trend,0,1))
    return macro, vix_last, ema20, rel_gap

# -------------- UI helpers --------------
def landing():
    st.markdown('<div class="hero"><h1>⭐️ Rate My</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Choose whether to rate a <b>stock</b> or your <b>portfolio</b>.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.button("📈 Rate My Stock", type="primary", use_container_width=True, on_click=enter, args=("stock",))
    with c2:
        st.button("💼 Rate My Portfolio", use_container_width=True, on_click=enter, args=("portfolio",))
    st.markdown('<p class="small-muted">Made with Streamlit · yfinance · pandas</p>', unsafe_allow_html=True)

def topbar_back(key):
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    st.button("← Back", key=key, on_click=go_home)
    st.markdown('</div>', unsafe_allow_html=True)

# ======== STOCK CHARTS ========
def draw_stock_charts(t: str, series: pd.Series):
    st.markdown("### 📈 Charts")
    if series is None or series.empty:
        st.info("Not enough history to show charts.")
        return

    # Price + EMAs
    e20, e50, e100 = ema(series,20), ema(series,50), ema(series,100)
    price_df = pd.DataFrame({"Close": series, "EMA20": e20, "EMA50": e50, "EMA100": e100})
    st.line_chart(price_df)

    # MACD
    line, sig, hist = macd(series)
    st.line_chart(pd.DataFrame({"MACD line": line, "Signal": sig}))
    st.bar_chart(pd.DataFrame({"Histogram": hist}))

    # RSI
    st.line_chart(pd.DataFrame({"RSI(14)": rsi(series)}))

    # 12m momentum
    mom12 = series/series.shift(253)-1.0
    st.line_chart(pd.DataFrame({"12m momentum": mom12}))

# =============== STOCK APP ===============
def app_stock():
    topbar_back("back_stock")
    st.title("⭐️ Rate My Stock")
    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    ticker = st.text_input(" ", "AAPL", label_visibility="collapsed", placeholder="Type a ticker (e.g., AAPL)", key="ticker_in")
    st.markdown('</div></div>', unsafe_allow_html=True)

    with st.expander("Advanced settings", expanded=False):
        c1,c2,c3 = st.columns(3)
        with c1:
            universe_mode = st.selectbox(
                "Peer universe",
                ["Auto by index membership","S&P 500","Dow 30","NASDAQ 100","Custom (paste list)"], index=0
            )
        with c2:
            peer_n = st.slider("Peer sample size", 30, 220, 120, 10)
        with c3:
            history = st.selectbox("History", ["1y","2y","5y"], index=0)
        c4,c5,c6 = st.columns(3)
        with c4: w_f = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.5, 0.05)
        with c5: w_t = st.slider("Weight: Technicals",   0.0, 1.0, 0.40, 0.05)
        with c6: w_m = st.slider("Weight: Macro (VIX)",  0.0, 1.0, 0.10, 0.05)
        custom_raw = st.text_area("Custom peers (comma-separated)", "") \
                      if universe_mode=="Custom (paste list)" else ""

    user_tickers = [yf_symbol(x) for x in ticker.split(",") if x.strip()]
    if not user_tickers:
        st.info("Enter a ticker above to run the rating.")
        return

    # Auto-loading status
    with st.status("Crunching the numbers…", expanded=True) as status:
        prog = st.progress(0)
        status.update(label="Building peer universe…")
        universe, bench, label = build_universe(user_tickers, universe_mode, peer_n, custom_raw)
        prog.progress(10)

        status.update(label="Downloading prices…")
        prices, ok = fetch_prices_chunked_with_fallback(universe, period=history, interval="1d",
                                                        chunk=50, min_ok=min(60, max(40, int(peer_n*0.6))))
        if not ok:
            st.error("No peer prices loaded.")
            return
        panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size>0}
        prog.progress(40)

        status.update(label="Computing technicals…")
        tech = technical_scores(panel)
        for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
            if col in tech.columns: tech[f"{col}_z"] = zscore_series(tech[col])
        TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"]
                           if c in tech.columns]].mean(axis=1)
        prog.progress(70)

        status.update(label="Fetching fundamentals…")
        fund_raw = fetch_fundamentals_simple(list(panel.keys()))
        fdf = pd.DataFrame(index=fund_raw.index)
        for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
                    "grossMargins","operatingMargins","ebitdaMargins"]:
            if col in fund_raw.columns: fdf[f"{col}_z"] = zscore_series(fund_raw[col])
        for col in ["trailingPE","forwardPE","debtToEquity"]:
            if col in fund_raw.columns: fdf[f"{col}_z"] = zscore_series(-fund_raw[col])
        FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fund_raw.index)
        prog.progress(86)

        status.update(label="Assessing macro regime…")
        vix_series = fetch_vix_series(period="6mo", interval="1d")
        MACRO, vix_last, vix_ema20, vix_gap = macro_from_vix(vix_series)
        prog.progress(100)
        status.update(label="Done!", state="complete")

    # Banner (no VIX number here)
    peers_loaded = len(panel)
    st.markdown(f'<div class="banner">Peers loaded: <b>{peers_loaded}</b> &nbsp;|&nbsp; '
                f'Benchmark: <b>{bench}</b> &nbsp;|&nbsp; Peer set: <b>{label}</b></div>',
                unsafe_allow_html=True)

    # Build output
    idx = pd.Index(list(panel.keys()))
    out = pd.DataFrame(index=idx)
    out["FUND_score"] = FUND_score.reindex(idx).fillna(0.0)
    out["TECH_score"] = TECH_score.reindex(idx).fillna(0.0)
    out["MACRO_score"]= MACRO
    wsum = (w_f + w_t + w_m) or 1.0
    wf, wt, wm = w_f/wsum, w_t/wsum, w_m/wsum
    out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wm*out["MACRO_score"]
    out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])
    out["RECO"] = out["RATING_0_100"].apply(
        lambda x: "Strong Buy" if x>=80 else "Buy" if x>=60 else "Hold" if x>=40 else "Sell" if x>=20 else "Strong Sell"
    )

    show_idx = [t for t in user_tickers if t in out.index]
    table = out.reindex(show_idx).sort_values("RATING_0_100", ascending=False)

    st.markdown("## 🏁 Ratings")
    pretty = table.rename(columns={
        "FUND_score":"Fundamentals","TECH_score":"Technicals","MACRO_score":"Macro (VIX)",
        "COMPOSITE":"Composite","RATING_0_100":"Score (0–100)","RECO":"Recommendation"
    })
    st.dataframe(pretty.round(4), use_container_width=True)

    st.markdown("## 🔎 Why this rating?")
    all_rows = []  # for "download all"
    for t in show_idx:
        reco = table.loc[t,"RECO"]; sc = table.loc[t,"RATING_0_100"]
        with st.expander(f"{t} — {reco} (Score: {sc:.1f})", expanded=True):
            # KPI row
            k1,k2,k3 = st.columns(3)
            k1.markdown('<div class="kpi-card"><div>Fundamentals</div>'
                        f'<div class="kpi-num">{table.loc[t,"FUND_score"]:.3f}</div></div>', unsafe_allow_html=True)
            k2.markdown('<div class="kpi-card"><div>Technicals</div>'
                        f'<div class="kpi-num">{table.loc[t,"TECH_score"]:.3f}</div></div>', unsafe_allow_html=True)
            k3.markdown('<div class="kpi-card"><div>Macro (VIX)</div>'
                        f'<div class="kpi-num">{table.loc[t,"MACRO_score"]:.3f}</div></div>', unsafe_allow_html=True)

            # Fundamentals detail
            st.markdown("#### Fundamentals (z-scores vs peers)")
            fshow = pd.DataFrame({
                "Revenue growth (z)": fdf.loc[t, "revenueGrowth_z"] if "revenueGrowth_z" in fdf.columns else np.nan,
                "Earnings growth (z)": fdf.loc[t, "earningsGrowth_z"] if "earningsGrowth_z" in fdf.columns else np.nan,
                "ROE (z)": fdf.loc[t, "returnOnEquity_z"] if "returnOnEquity_z" in fdf.columns else np.nan,
                "Profit margin (z)": fdf.loc[t, "profitMargins_z"] if "profitMargins_z" in fdf.columns else np.nan,
                "Gross margin (z)": fdf.loc[t, "grossMargins_z"] if "grossMargins_z" in fdf.columns else np.nan,
                "Operating margin (z)": fdf.loc[t, "operatingMargins_z"] if "operatingMargins_z" in fdf.columns else np.nan,
                "EBITDA margin (z)": fdf.loc[t, "ebitdaMargins_z"] if "ebitdaMargins_z" in fdf.columns else np.nan,
                "PE (z, lower better)": fdf.loc[t, "trailingPE_z"] if "trailingPE_z" in fdf.columns else np.nan,
                "Forward PE (z, lower better)": fdf.loc[t, "forwardPE_z"] if "forwardPE_z" in fdf.columns else np.nan,
                "Debt/Equity (z, lower better)": fdf.loc[t, "debtToEquity_z"] if "debtToEquity_z" in fdf.columns else np.nan,
            }, index=[t]).T.rename(columns={t:"z-score"})
            st.dataframe(fshow.round(3), use_container_width=True)
            st.caption("Positive z-scores: above-peer values. For valuation/leverage rows, negative values are better (cheaper / less debt).")

            # Technicals detail
            st.markdown("#### Technicals")
            if t in tech.index:
                rsi_val = 50 + 50*tech.loc[t,"rsi_strength"] if "rsi_strength" in tech.columns and pd.notna(tech.loc[t,"rsi_strength"]) else np.nan
                tshow = pd.DataFrame({
                    "Price vs EMA100 (gap)": tech.loc[t,"dma_gap"] if "dma_gap" in tech.columns else np.nan,
                    "MACD histogram": tech.loc[t,"macd_hist"] if "macd_hist" in tech.columns else np.nan,
                    "RSI (approx)": rsi_val,
                    "12-mo momentum": tech.loc[t,"mom12m"] if "mom12m" in tech.columns else np.nan,
                }, index=[t]).T.rename(columns={t:"value"})
                st.dataframe(tshow.round(3), use_container_width=True)
                st.caption("Above EMA100 + positive MACD = strength. RSI ~70 overbought / ~30 oversold. 12-mo momentum compares to 1 year ago.")
            else:
                st.info("Not enough price history for technicals.")

            # Macro detail (with interpretation)
            st.markdown("#### Macro (VIX) — level & trend")
            if not np.isnan(vix_last):
                m1,m2,m3 = st.columns(3)
                m1.metric("VIX (last)", f"{vix_last:.2f}")
                m2.metric("VIX EMA20", f"{vix_ema20:.2f}")
                m3.metric("Gap vs EMA20", f"{(vix_gap*100):.1f}%")
                # Interpretation
                if vix_last <= 13:
                    level_txt = "very calm"
                elif vix_last <= 18:
                    level_txt = "calm"
                elif vix_last <= 24:
                    level_txt = "elevated"
                else:
                    level_txt = "stressed"

                if vix_gap > 0.03:
                    trend_txt = "rising above trend (risk building)"
                elif vix_gap < -0.03:
                    trend_txt = "falling below trend (risk easing)"
                else:
                    trend_txt = "near trend"

                st.markdown(f"- **Level** suggests **{level_txt}** conditions. "
                            f"**Trend** is **{trend_txt}**. "
                            "A higher Macro score reflects calmer levels and/or easing trend.")
            else:
                st.info("VIX unavailable — Macro defaults to a neutral value.")

            # ---------- Export: this ticker ----------
            row = {
                "ticker": t,
                "fundamentals_score": float(table.loc[t, "FUND_score"]),
                "technicals_score":   float(table.loc[t, "TECH_score"]),
                "macro_score":        float(table.loc[t, "MACRO_score"]),
                "composite":          float(table.loc[t, "COMPOSITE"]),
                "score_0_100":        float(table.loc[t, "RATING_0_100"]),
                "recommendation":     str(table.loc[t, "RECO"]),
                "vix_last": float(vix_last) if not np.isnan(vix_last) else np.nan,
                "vix_ema20": float(vix_ema20) if not np.isnan(vix_ema20) else np.nan,
                "vix_gap_vs_ema": float(vix_gap) if not np.isnan(vix_gap) else np.nan,
            }
            for col in [
                "revenueGrowth_z","earningsGrowth_z","returnOnEquity_z",
                "profitMargins_z","grossMargins_z","operatingMargins_z","ebitdaMargins_z",
                "trailingPE_z","forwardPE_z","debtToEquity_z",
            ]:
                row[col] = float(fdf.loc[t, col]) if (t in fdf.index and col in fdf.columns and pd.notna(fdf.loc[t, col])) else np.nan
            for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
                row[col] = float(tech.loc[t, col]) if (t in tech.index and col in tech.columns and pd.notna(tech.loc[t, col])) else np.nan
                zc=f"{col}_z"
                row[zc] = float(tech.loc[t, zc]) if (t in tech.index and zc in tech.columns and pd.notna(tech.loc[t, zc])) else np.nan

            export_df = pd.DataFrame([row])
            csv_bytes = export_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download this breakdown (CSV)",
                data=csv_bytes,
                file_name=f"{t}_breakdown.csv",
                mime="text/csv",
                use_container_width=True,
            )
            xlsx_io = io.BytesIO()
            with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name=t)
            st.download_button(
                "⬇️ Download this breakdown (Excel)",
                data=xlsx_io.getvalue(),
                file_name=f"{t}_breakdown.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            # Stock charts (price+EMAs, MACD, RSI, momentum)
            if t in panel:
                draw_stock_charts(t, panel[t])

            all_rows.append(row)

    # ---------- Export: all shown tickers ----------
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        st.markdown("### Export all shown tickers")
        st.download_button(
            "⬇️ Download all (CSV)",
            data=df_all.to_csv(index=False).encode(),
            file_name="stock_breakdowns.csv",
            mime="text/csv",
            use_container_width=True,
        )
        xlsx_all = io.BytesIO()
        with pd.ExcelWriter(xlsx_all, engine="openpyxl") as w:
            df_all.to_excel(w, index=False, sheet_name="Breakdowns")
        st.download_button(
            "⬇️ Download all (Excel)",
            data=xlsx_all.getvalue(),
            file_name="stock_breakdowns.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# =============== PORTFOLIO APP ===============
CURRENCY_MAP = {"$":"USD","€":"EUR","£":"GBP","CHF":"CHF","C$":"CAD","A$":"AUD","¥":"JPY"}

def _safe_num(x): return pd.to_numeric(x, errors="coerce")

# >>> FIXED: robust change detection (no Series truth-value errors)
def detect_change_side(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> str:
    """
    Decide which side changed more between prev and curr:
    returns "amount", "percent" or "none".
    Robust to missing columns and different lengths.
    """
    def col(df, name):
        if df is None or not isinstance(df, pd.DataFrame) or name not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[name], errors="coerce")

    a_prev = col(prev_df, "Amount")
    a_curr = col(curr_df, "Amount")
    p_prev = col(prev_df, "Percent (%)")
    p_curr = col(curr_df, "Percent (%)")

    n = max(len(a_prev), len(a_curr), len(p_prev), len(p_curr), 1)
    idx = pd.RangeIndex(n)

    a_prev = a_prev.reindex(idx).fillna(0.0)
    a_curr = a_curr.reindex(idx).fillna(0.0)
    p_prev = p_prev.reindex(idx).fillna(0.0)
    p_curr = p_curr.reindex(idx).fillna(0.0)

    d_amt = (a_curr - a_prev).abs().sum()
    d_pct = (p_curr - p_prev).abs().sum()

    if d_amt > d_pct and d_amt > 0: return "amount"
    if d_pct > d_amt and d_pct > 0: return "percent"
    return "none"

def sync_percent_amount(df: pd.DataFrame, total: float, mode: str) -> pd.DataFrame:
    df=df.copy()
    df["Ticker"]=df["Ticker"].astype(str).str.strip()
    df=df[df["Ticker"].astype(bool)].reset_index(drop=True)
    n=len(df)
    if n==0: df["weight"]=[]; return df
    df["Percent (%)"]=_safe_num(df.get("Percent (%)"))
    df["Amount"]=_safe_num(df.get("Amount"))
    has_total = (total is not None and total>0)

    drive=mode
    if mode=="auto":
        prev=st.session_state.get("prev_grid_df")
        try:
            side=detect_change_side(prev, df)
        except Exception:
            side="none"
        if side=="none":
            if has_total and df["Amount"].fillna(0).sum()>0: drive="amount"
            elif df["Percent (%)"].fillna(0).sum()>0: drive="percent"
            else: drive="percent"
        else:
            drive=side

    if has_total:
        if drive=="percent":
            if df["Percent (%)"].fillna(0).sum()==0: df["Percent (%)"]=100.0/n
            df["Amount"]=(df["Percent (%)"].fillna(0)/100.0*total).round(2)
        else:
            s=df["Amount"].fillna(0).sum()
            if s>0:
                df["Percent (%)"]=(df["Amount"].fillna(0)/s*100.0).round(2)
            else:
                df["Percent (%)"]=100.0/n
    else:
        if drive=="percent" and df["Percent (%)"].fillna(0).sum()==0:
            df["Percent (%)"]=100.0/n

    if has_total and df["Amount"].fillna(0).sum()>0:
        w=df["Amount"].fillna(0)/df["Amount"].fillna(0).sum()
    elif df["Percent (%)"].fillna(0).sum()>0:
        s=df["Percent (%)"].fillna(0).sum()
        w=df["Percent (%)"].fillna(0)/s
    else:
        w=pd.Series([1.0/n]*n, index=df.index)
    df["weight"]=w
    return df

def holdings_editor(currency_symbol, total_value):
    if st.session_state["grid_df"] is None:
        st.session_state["grid_df"]=pd.DataFrame({
            "Ticker":["AAPL","MSFT","NVDA","AMZN"],
            "Percent (%)":[25.0,25.0,25.0,25.0],
            "Amount":[np.nan,np.nan,np.nan,np.nan]
        })

    st.markdown(
        f"**Holdings**  \n"
        f"<span class='small-muted'>Enter <b>Ticker</b> and either <b>Percent (%)</b> or <b>Amount ({currency_symbol})</b>. "
        f"Set a total value to auto-sync % and {currency_symbol}.</span>", unsafe_allow_html=True)

    sync_mode = st.segmented_control(
        "Sync mode", options=["Auto","Percent → Amount","Amount → Percent"], default="Auto",
        help="Choose which side drives the other."
    )
    mode_key = {"Auto":"auto","Percent → Amount":"percent","Amount → Percent":"amount"}[sync_mode]

    edited = st.data_editor(
        st.session_state["grid_df"], num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(width="small"),
            "Percent (%)": st.column_config.NumberColumn(format="%.2f"),
            "Amount": st.column_config.NumberColumn(format="%.2f", help=f"Amount in {currency_symbol}"),
        },
        key="grid"
    )

    synced = sync_percent_amount(edited.copy(), total_value, mode_key)
    st.session_state["prev_grid_df"]=st.session_state["grid_df"].copy()
    st.session_state["grid_df"]=synced[["Ticker","Percent (%)","Amount"]]

    out=synced.copy()
    out["ticker"]=out["Ticker"].map(yf_symbol)
    out=out[out["ticker"].astype(bool)]
    return out[["ticker","weight"]], synced

def app_portfolio():
    topbar_back("back_port")
    st.title("💼 Rate My Portfolio")

    top1,top2,top3=st.columns([1,1,1])
    with top1:
        cur = st.selectbox("Currency", list(CURRENCY_MAP.keys()), index=0)
    with top2:
        total = st.number_input(f"Total portfolio value ({cur})", min_value=0.0, value=10000.0, step=500.0)
    with top3:
        st.caption("Change currency/total — grid syncs instantly.")

    df_hold, synced_view = holdings_editor(cur, total)
    if df_hold.empty:
        st.info("Add at least one holding to run the rating.")
        return

    with st.expander("Advanced settings", expanded=False):
        c1,c2,c3=st.columns(3)
        with c1:
            universe_mode = st.selectbox(
                "Peer universe", ["Auto by index membership","S&P 500","Dow 30","NASDAQ 100","Custom (paste list)"], index=0
            )
        with c2: peer_n = st.slider("Peer sample size", 30, 250, 150, 10)
        with c3: history = st.selectbox("History", ["1y","2y","5y"], index=0)
        c4,c5,c6,c7 = st.columns(4)
        with c4: w_f = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.45, 0.05)
        with c5: w_t = st.slider("Weight: Technicals",   0.0, 1.0, 0.40, 0.05)
        with c6: w_m = st.slider("Weight: Macro (VIX)",  0.0, 1.0, 0.10, 0.05)
        with c7: w_d = st.slider("Weight: Diversification", 0.0, 1.0, 0.05, 0.05)
        custom_raw = st.text_area("Custom peers (comma-separated)", "") \
                      if universe_mode=="Custom (paste list)" else ""

    # Auto run
    tickers = df_hold["ticker"].tolist()
    with st.status("Crunching the numbers…", expanded=True) as status:
        prog = st.progress(0)
        status.update(label="Building peer universe…")
        universe, bench, label = build_universe(tickers, universe_mode, peer_n, custom_raw); prog.progress(8)

        status.update(label="Downloading prices…")
        prices, ok = fetch_prices_chunked_with_fallback(universe, period=history, interval="1d",
                                                        chunk=50, min_ok=min(80, max(50, int(peer_n*0.5))))
        if prices.empty: st.error("No prices fetched."); return
        prog.progress(35)

        status.update(label="Computing technicals…")
        panel_all = {t: prices[t].dropna() for t in prices.columns if t in prices.columns and prices[t].dropna().size>0}
        tech_all = technical_scores(panel_all)
        for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
            if col in tech_all.columns: tech_all[f"{col}_z"] = zscore_series(tech_all[col])
        TECH_score_all = tech_all[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"]
                                   if c in tech_all.columns]].mean(axis=1)
        prog.progress(58)

        status.update(label="Fetching fundamentals…")
        fund_raw_all = fetch_fundamentals_simple(list(panel_all.keys()))
        fdf_all = pd.DataFrame(index=fund_raw_all.index)
        for col in ["revenueGrowth","earningsGrowth","returnOnEquity","profitMargins",
                    "grossMargins","operatingMargins","ebitdaMargins"]:
            if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=zscore_series(fund_raw_all[col])
        for col in ["trailingPE","forwardPE","debtToEquity"]:
            if col in fund_raw_all.columns: fdf_all[f"{col}_z"]=zscore_series(-fund_raw_all[col])
        FUND_score_all = fdf_all.mean(axis=1) if len(fdf_all.columns) else pd.Series(0.0, index=fund_raw_all.index)
        prog.progress(75)

        status.update(label="Assessing macro regime…")
        vix_series = fetch_vix_series(period="6mo", interval="1d")
        MACRO, _, _, _ = macro_from_vix(vix_series)
        prog.progress(100); status.update(label="Done!", state="complete")

    # Banner
    st.markdown(f'<div class="banner">Peers loaded: <b>{len(panel_all)}</b> &nbsp;|&nbsp; '
                f'Benchmark: <b>{bench}</b> &nbsp;|&nbsp; Peer set: <b>{label}</b></div>', unsafe_allow_html=True)

    # Compose per-name scores
    idx_all = pd.Index(list(panel_all.keys()))
    out_all = pd.DataFrame(index=idx_all)
    out_all["FUND_score"]  = FUND_score_all.reindex(idx_all).fillna(0.0)
    out_all["TECH_score"]  = TECH_score_all.reindex(idx_all).fillna(0.0)
    out_all["MACRO_score"] = MACRO
    wsum=(w_f+w_t+w_m) or 1.0
    wf,wt,wm=w_f/wsum, w_t/wsum, w_m/wsum
    out_all["COMPOSITE"] = wf*out_all["FUND_score"] + wt*out_all["TECH_score"] + wm*out_all["MACRO_score"]
    out_all["RATING_0_100"] = percentile_rank(out_all["COMPOSITE"])

    # Diversification
    def fetch_sector(t):
        try: return yf.Ticker(t).info.get("sector", None)
        except Exception: return None
    meta_sec = pd.Series({t: fetch_sector(t) for t in tickers})
    weights = df_hold.set_index("ticker")["weight"]
    sec_mix = weights.groupby(meta_sec).sum()
    if sec_mix.empty: sec_mix = pd.Series({"Unknown":1.0})
    hhi = float((sec_mix**2).sum())
    effN = 1.0/hhi if hhi>0 else 1.0
    targetN = min(10, max(1,len(sec_mix)))
    sector_div = float(np.clip((effN-1)/(targetN-1 if targetN>1 else 1), 0, 1))
    max_w = float(weights.max())
    if   max_w <= 0.10: name_div = 1.0
    elif max_w >= 0.40: name_div = 0.0
    else:               name_div = float((0.40-max_w)/0.30)
    ret = prices[tickers].pct_change().dropna(how="all")
    if ret.shape[1]>=2:
        corr = ret.corr().values; n=corr.shape[0]
        avg_corr = (corr.sum()-np.trace(corr))/max(1,(n*n-n))
        corr_div = float(np.clip(1.0-max(0.0, avg_corr), 0.0, 1.0))
    else:
        avg_corr=np.nan; corr_div=0.5
    DIV = 0.5*sector_div + 0.3*corr_div + 0.2*name_div

    per_name = out_all.reindex(tickers).copy()
    per_name = per_name.join(weights, how="left")
    per_name["weighted_composite"] = per_name["COMPOSITE"]*per_name["weight"]
    port_signal = float(per_name["weighted_composite"].sum())
    total_for_final = 1.0 + w_d
    port_final = (port_signal)*(1/total_for_final) + DIV*(w_d/total_for_final)
    port_score = float(np.clip((port_final+1)/2, 0, 1)*100)

    st.markdown("## 🧺 Portfolio — Scores")
    a,b,c,d = st.columns(4)
    a.metric("Portfolio Score (0–100)", f"{port_score:.1f}")
    b.metric("Signal (weighted composite)", f"{port_signal:.3f}")
    c.metric("Macro (VIX)", f"{MACRO:.3f}")
    d.metric("Diversification", f"{DIV:.3f}")

    with st.expander("Why this portfolio rating?"):
        show = per_name.rename(columns={"weight":"Weight","FUND_score":"Fundamentals",
                                        "TECH_score":"Technicals","MACRO_score":"Macro (VIX)",
                                        "COMPOSITE":"Composite","weighted_composite":"Weight × Comp"})[
            ["Weight","Fundamentals","Technicals","Macro (VIX)","Composite","Weight × Comp"]
        ]
        st.dataframe(show.round(4), use_container_width=True)
        st.markdown("**Diversification explained**")
        st.markdown(
            f"- **Sector HHI** converts to an effective number of sectors ≈ {effN:.1f}. "
            f"More sectors → higher score (your sector diversity score = {sector_div:.2f}).\n"
            f"- **Name concentration**: your largest position ≈ {max_w*100:.1f}%. "
            f"Lower concentration is better (score = {name_div:.2f}).\n"
            f"- **Correlation**: average pairwise correlation ≈ "
            f"{('%.2f' % avg_corr) if not np.isnan(avg_corr) else 'N/A'}. "
            f"Lower correlation improves diversification (score = {corr_div:.2f}).\n"
            f"- **Diversification score** = 50% sector + 30% correlation + 20% name concentration."
        )

    # Charts
    px_held = prices[tickers].dropna(how="all")
    r = px_held.pct_change().fillna(0)
    w_vec = weights.reindex(px_held.columns).fillna(0).values
    port_r = (r * w_vec).sum(axis=1)
    eq = (1+port_r).cumprod()
    tabs = st.tabs(["Cumulative", "Volatility (60d) & Sharpe", "Drawdown"])
    with tabs[0]:
        st.line_chart(pd.DataFrame({"Portfolio cumulative": eq}))
    with tabs[1]:
        vol60 = port_r.rolling(60).std()*np.sqrt(252)
        sharpe60 = (port_r.rolling(60).mean()/port_r.rolling(60).std())*np.sqrt(252)
        st.line_chart(pd.DataFrame({"Volatility 60d (ann.)": vol60, "Sharpe 60d": sharpe60}))
    with tabs[2]:
        roll_max = eq.cummax(); dd = eq/roll_max - 1
        st.line_chart(pd.DataFrame({"Drawdown": dd}))

# -------------- Router --------------
def app_router():
    if not st.session_state.entered:
        landing(); return
    if st.session_state.mode=="portfolio":
        app_portfolio()
    else:
        app_stock()

app_router()