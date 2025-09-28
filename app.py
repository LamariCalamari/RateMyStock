# app.py — ⭐️ Rate My Stock (landing page + centered search + robust REL + explanations)

import io, os, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------- Page/Style ----------
st.set_page_config(page_title="Rate My Stock", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1100px;}
.hero {text-align:center; margin-top: 4rem; margin-bottom: .5rem;}
.sub {text-align:center; color:#9aa0a6; margin-bottom: 2rem;}
.search-wrap {display:flex; justify-content:center; margin-top: 1rem; margin-bottom: 1rem;}
.search-inner {width: min(680px, 90%);}
.search-input input {border-radius: 9999px !important; padding: 0.9rem 1.2rem !important; font-size: 1.1rem;}
.small-muted {color:#9aa0a6; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Session gate (landing) ----------
if "entered" not in st.session_state:
    st.session_state.entered = False

def enter_app():
    st.session_state.entered = True

# ---------- Utils / Math ----------
def yf_symbol(t: str) -> str:
    if not isinstance(t, str): return t
    return t.strip().upper().replace(".", "-")

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0

def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window=14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def macro_overlay_score(vix: float) -> float:
    if pd.isna(vix): return 0.5
    if vix <= 15: return 1.0
    if vix >= 35: return 0.0
    return 1.0 - (vix - 15) / 20.0

# ---------- Robust data fetch ----------
@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(tickers, period="1y", interval="1d",
                                       chunk=50, min_ok=40, sleep_between=0.2):
    tickers = [yf_symbol(t) for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))
    frames, ok, fail = [], [], []
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
                        else: fail.append(t)
                    else:
                        fail.append(t)
            else:
                t = group[0]
                if "Close" in df:
                    s = df["Close"].dropna()
                    if s.size: frames.append(s.rename(t)); ok.append(t)
                    else: fail.append(t)
                else:
                    fail.append(t)
        except Exception:
            fail.extend(group)
        time.sleep(sleep_between)
    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    # single-name fallback
    if len(ok) < min_ok:
        for t in [t for t in tickers if t not in ok]:
            if len(ok) >= min_ok: break
            try:
                df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
                s = df["Close"].dropna()
                if s.size:
                    if prices.empty: prices = s.to_frame(t)
                    else: prices[t] = s
                    ok.append(t)
                else:
                    if t not in fail: fail.append(t)
            except Exception:
                if t not in fail: fail.append(t)
            time.sleep(0.12)

    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return prices, ok, [t for t in tickers if t not in ok]

@st.cache_data(show_spinner=False)
def fetch_price_series(ticker: str, period="1y", interval="1d") -> pd.Series:
    t = yf_symbol(ticker)
    try:
        df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
        if not df.empty and "Close" in df.columns:
            return df["Close"].rename(t)
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_vix_level() -> float:
    try:
        vix = yf.Ticker("^VIX").history(period="6mo", interval="1d")
        if not vix.empty: return float(vix["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

# ---------- Fallback peer lists ----------
SP500_FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","BRK-B","UNH","LLY","V","JPM","JNJ","XOM",
    "PG","MA","HD","CVX","MRK","ABBV","PEP","KO","COST","BAC","ADBE","WMT","CRM","PFE","CSCO","ACN","MCD","TMO",
    "NFLX","AMD","DHR","NKE","LIN","ABT","INTC","TXN","DIS","AMAT","PM","NEE","COP","MS","LOW","HON","BMY","QCOM",
    "IBM","UNP","SBUX","INTU","CAT","GS","LMT","RTX","BLK","BKNG","AXP","GE","NOW","MDT","ISRG","ADI","ELV","PLD",
    "DE","ZTS","SPGI","MDLZ","TJX","GILD","ETN","ADP","CB","SO","EQIX","PGR","MMC","CI","SCHW","MU","REGN","ORLY"
]
DOW30_FALLBACK = [
    "AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","DIS","HD","INTC","IBM","KO","MCD","NKE","TRV","VZ","CSCO",
    "MRK","PFE","CAT","AXP","BA","MMM","GS","CVX","WBA","HON","CRM"
]

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

# ---------- Features ----------
def technical_scores(price_panel: dict) -> pd.DataFrame:
    rows = []
    for ticker, px in price_panel.items():
        px = px.dropna()
        if len(px) < 130:
            continue
        ema100 = ema(px, 100)
        base = ema100.iloc[-1] if pd.notna(ema100.iloc[-1]) and ema100.iloc[-1] != 0 else np.nan
        dma_gap = (px.iloc[-1] - ema100.iloc[-1]) / base if pd.notna(base) else np.nan
        _, _, hist = macd(px)
        macd_hist = hist.iloc[-1]
        r = rsi(px).iloc[-1]
        rsi_strength = (r - 50.0) / 50.0
        mom12m = px.iloc[-1] / px.iloc[-253] - 1.0 if len(px) > 252 else np.nan
        rows.append({"ticker":ticker,"dma_gap":dma_gap,"macd_hist":macd_hist,"rsi_strength":rsi_strength,"mom12m":mom12m})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

def rolling_beta_alpha(stock_px: pd.Series, bench_px: pd.Series, window: int = 60):
    df = pd.DataFrame({"r_s": stock_px.pct_change(), "r_b": bench_px.pct_change()}).dropna()
    out = []
    for i in range(window, len(df)):
        chunk = df.iloc[i-window:i]
        if chunk["r_b"].std(ddof=0) == 0: beta = np.nan
        else:
            cov = np.cov(chunk["r_s"], chunk["r_b"])[0,1]
            beta = cov / np.var(chunk["r_b"])
        alpha = df["r_s"].iloc[i] - (beta * df["r_b"].iloc[i] if pd.notna(beta) else 0.0)
        out.append((df.index[i], beta, alpha))
    return pd.DataFrame(out, columns=["date","beta","alpha"]).set_index("date")

def relative_signals(stock_px: pd.Series, bench_px: pd.Series):
    ratio = (stock_px / bench_px).dropna()
    d = {"rel_dma_gap": np.nan, "rel_mom12m": np.nan, "alpha_60d": np.nan}
    if len(ratio) >= 100:  # slightly more forgiving than 120
        ema100 = ratio.ewm(span=100, adjust=False).mean()
        base = ema100.iloc[-1] if pd.notna(ema100.iloc[-1]) and ema100.iloc[-1] != 0 else np.nan
        d["rel_dma_gap"] = (ratio.iloc[-1] - ema100.iloc[-1]) / base if pd.notna(base) else np.nan
    if len(ratio) > 252:
        d["rel_mom12m"] = ratio.iloc[-1] / ratio.iloc[-253] - 1.0
    ba = rolling_beta_alpha(stock_px, bench_px, window=60)
    if not ba.empty and "alpha" in ba:
        s = ba["alpha"].dropna()
        if s.size: d["alpha_60d"] = s.iloc[-1]
    return d

# ---------- Friendly explainers ----------
def label_badge(value, pos_thresh=None, neg_thresh=None, higher_is_better=True, units=""):
    v = value
    if pd.isna(v): return "⚪ Neutral"
    if higher_is_better:
        if pos_thresh is None: pos_thresh = 0
        if v > pos_thresh: return f"🟢 Bullish ({v:.3f}{units})"
        if neg_thresh is not None and v < neg_thresh: return f"🔴 Bearish ({v:.3f}{units})"
        return f"⚪ Neutral ({v:.3f}{units})"
    else:
        if neg_thresh is None: neg_thresh = 0
        if v < neg_thresh: return f"🟢 Bullish ({v:.3f}{units})"
        if pos_thresh is not None and v > pos_thresh: return f"🔴 Bearish ({v:.3f}{units})"
        return f"⚪ Neutral ({v:.3f}{units})"

def explain_technicals_row(row):
    notes = []
    notes.append(f"• **Price vs EMA(100)**: {label_badge(row.get('dma_gap'), pos_thresh=0)}. Above EMA suggests an up-trend; below EMA can flag weakness or value entry (context matters).")
    notes.append(f"• **MACD histogram**: {label_badge(row.get('macd_hist'), pos_thresh=0)}. Positive = momentum building; negative = fading.")
    rsi_strength = row.get("rsi_strength"); rsi_val = 50 + 50*rsi_strength if pd.notna(rsi_strength) else None
    if rsi_val is not None:
        if rsi_val >= 60: rsi_note = f"🟢 Bullish (RSI≈{rsi_val:.0f})"
        elif rsi_val <= 40: rsi_note = f"🔴 Bearish (RSI≈{rsi_val:.0f})"
        else: rsi_note = f"⚪ Neutral (RSI≈{rsi_val:.0f})"
    else:
        rsi_note = "⚪ Neutral"
    notes.append(f"• **RSI**: {rsi_note}. 70+ often overbought; 30− oversold.")
    notes.append(f"• **12-month momentum**: {label_badge(row.get('mom12m'), pos_thresh=0)} vs its own past year.")
    return "\n".join(notes)

def explain_relative_row(row):
    notes = []
    notes.append(f"• **Ratio EMA gap (stock/benchmark)**: {label_badge(row.get('rel_dma_gap'), pos_thresh=0)}.")
    notes.append(f"• **12-month relative momentum**: {label_badge(row.get('rel_mom12m'), pos_thresh=0)}.")
    notes.append(f"• **Rolling alpha (60d)**: {label_badge(row.get('alpha_60d'), pos_thresh=0)}. Positive = beating benchmark on risk-adjusted basis recently.")
    return "\n".join(notes)

FRIENDLY_NAMES = {
    "FUND_score":"Fundamentals", "TECH_score":"Technicals", "REL_score":"Relative vs Benchmark",
    "MACRO_score":"Macro (VIX)", "COMPOSITE":"Composite", "RATING_0_100":"Score (0–100)", "RECO":"Recommendation"
}

# ---------- Landing ----------
def landing():
    st.markdown('<div class="hero"><h1>⭐️ Rate My Stock</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Pick a stock, we’ll grab its peers from the index and give you a vibe-checked rating — with receipts.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.button("Enter", type="primary", use_container_width=True, on_click=enter_app)
    st.markdown('<p class="small-muted" style="text-align:center;margin-top:1rem;">Made with Streamlit · yfinance · pandas</p>', unsafe_allow_html=True)

# ---------- Main App ----------
def main_app():
    # Centered search
    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    ticker = st.text_input(" ", "AAPL", key="ticker", label_visibility="collapsed",
                           placeholder="Type a ticker (e.g., AAPL)")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Advanced settings (collapsed)
    with st.expander("Advanced settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            universe_mode = st.selectbox(
                "Peer universe",
                ["Auto by index membership", "S&P 500", "Dow 30", "Custom (paste list)"],
                index=0
            )
        with c2:
            peer_sample_n = st.slider("Peer sample size", 30, 200, 120, 10)
        with c3:
            history = st.selectbox("History", ["1y","2y","5y"], index=0)
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            benchmark = st.selectbox("Benchmark (for REL)", ["SPY","^GSPC","QQQ","^IXIC","^DJI"], index=0)
        with c5:
            w_fund = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.5, 0.05)
        with c6:
            w_tech = st.slider("Weight: Technicals", 0.0, 1.0, 0.35, 0.05)
        with c7:
            w_macro = st.slider("Weight: Macro (VIX)", 0.0, 1.0, 0.10, 0.05)
        w_rel = st.slider("Weight: Relative vs Benchmark", 0.0, 1.0, 0.05, 0.05)
        custom_raw = ""
        if universe_mode == "Custom (paste list)":
            custom_raw = st.text_area("Custom peers (comma-separated)", "AMD, AVGO, CRM, COST, NFLX")
        show_debug = st.checkbox("Show debug info", False)

    # Run (centered)
    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    run_btn = st.button("Rate it 🚀", type="primary", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if not run_btn: return

    user_tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]
    if not user_tickers:
        st.error("Please enter a ticker."); return

    # Universe
    def build_universe(user_tickers, mode, sample_n=120, custom_raw=""):
        user = [yf_symbol(t) for t in user_tickers]
        if mode == "S&P 500":
            peers_all = list_sp500()
        elif mode == "Dow 30":
            peers_all = list_dow30()
        elif mode == "Custom (paste list)":
            custom = {yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
            return sorted(set(user) | custom)
        else:
            sp, dj = list_sp500(), list_dow30()
            auto = set()
            if len(user) == 1:
                t = user[0]
                if t in sp: auto = sp
                elif t in dj: auto = dj
            else:
                for t in user:
                    if t in sp: auto |= sp
                    elif t in dj: auto |= dj
            peers_all = auto if auto else (sp or dj or set(SP500_FALLBACK))
        peers = sorted(peers_all.difference(set(user)))
        if len(peers) > sample_n: peers = peers[:sample_n]
        return sorted(set(user) | set(peers))

    universe = build_universe(user_tickers, universe_mode, peer_sample_n, custom_raw)
    with st.spinner("Downloading prices for peers…"):
        prices, ok, _ = fetch_prices_chunked_with_fallback(
            universe, period=history, interval="1d",
            chunk=50, min_ok=min(60, max(40, int(peer_sample_n*0.6)))
        )
    panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size > 0}

    # If too few peers, retry with S&P fallback
    if len(panel) < 10:
        retry_universe = sorted(set([*user_tickers, *SP500_FALLBACK]))[:max(peer_sample_n, 120)]
        prices2, ok2, _ = fetch_prices_chunked_with_fallback(retry_universe, period="1y", interval="1d",
                                                             chunk=50, min_ok=60)
        panel = {t: prices2[t].dropna() for t in ok2 if t in prices2.columns and prices2[t].dropna().size > 0}

    loaded = len(panel)
    if loaded < 5:
        st.warning(f"Only {loaded} peers loaded. Try 1y history and larger peer sample.")
    if show_debug:
        st.info(f"Peers loaded: {loaded}")

    if not panel:
        st.error("No peer prices loaded."); return

    # Benchmark (robust)
    bench_px = fetch_price_series(benchmark, period=history, interval="1d")
    used_bench = benchmark
    if bench_px.empty:
        bench_px = fetch_price_series("SPY", period=history, interval="1d")
        used_bench = "SPY"
    if bench_px.empty:
        st.warning("Benchmark not available; Relative score disabled.")
        enable_rel = False
    else:
        enable_rel = True

    # --- Technicals
    tech = technical_scores(panel)
    for col in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        tech[f"{col}_z"] = zscore_series(tech[col]) if col in tech.columns else np.nan
    TECH_score = tech[[c for c in ["dma_gap_z","macd_hist_z","rsi_strength_z","mom12m_z"] if c in tech.columns]].mean(axis=1)

    # --- Fundamentals
    @st.cache_data(show_spinner=False)
    def fetch_fundamentals(tickers):
        rows=[]
        for raw in tickers:
            t=yf_symbol(raw)
            try:
                tk=yf.Ticker(t)
                try: inf=tk.info or {}
                except Exception: inf={}
                try: fin=tk.financials
                except Exception: fin=None
                try: bs=tk.balance_sheet
                except Exception: bs=None
                pe=inf.get("trailingPE", np.nan); ev=inf.get("enterpriseValue", np.nan)
                ebitda=np.nan; rev1=rev0=inc1=inc0=np.nan
                if fin is not None and not fin.empty:
                    try:
                        s=fin.loc[fin.index.str.contains("EBITDA", case=False)].T.squeeze().dropna()
                        ebitda=float(s.iloc[0]) if len(s) else np.nan
                    except Exception: pass
                    def two(name):
                        try:
                            s=fin.loc[fin.index.str.contains(name, case=False)].T.squeeze().dropna()
                            return (float(s.iloc[0]) if len(s) else np.nan,
                                    float(s.iloc[1]) if len(s)>1 else np.nan)
                        except Exception: return (np.nan,np.nan)
                    rev1,rev0 = two("Total Revenue"); inc1,inc0 = two("Net Income")
                ev_ebitda=np.nan
                if pd.notna(ev) and pd.notna(ebitda) and ebitda not in (0,None): ev_ebitda=ev/ebitda
                rev_g=np.nan
                if pd.notna(rev1) and pd.notna(rev0) and rev0!=0: rev_g=rev1/rev0-1.0
                eps_g=np.nan
                shares=inf.get("sharesOutstanding", np.nan)
                if pd.notna(shares) and shares not in (0,None) and pd.notna(inc1) and pd.notna(inc0) and inc0!=0:
                    eps1=inc1/shares; eps0=inc0/shares
                    if pd.notna(eps1) and pd.notna(eps0) and eps0!=0: eps_g=eps1/eps0-1.0
                roe=np.nan; de_ratio=np.nan; net_margin=np.nan; gross_margin=np.nan
                if pd.notna(inc1) and pd.notna(rev1) and rev1!=0: net_margin=inc1/rev1
                if fin is not None and not fin.empty:
                    try:
                        gp=fin.loc[fin.index.str.contains("Gross Profit", case=False)].T.squeeze().dropna()
                        if len(gp) and pd.notna(rev1) and rev1!=0: gross_margin=float(gp.iloc[0])/rev1
                    except Exception: pass
                if bs is not None and not bs.empty:
                    try:
                        eq=bs.loc[bs.index.str.contains("Total Stockholder", case=False)].T.squeeze().dropna()
                    except Exception:
                        eq=pd.Series(dtype=float)
                    try:
                        li=bs.loc[bs.index.str.contains("Total Liabilities", case=False)].T.squeeze().dropna()
                    except Exception:
                        li=pd.Series(dtype=float)
                    total_equity=float(eq.iloc[0]) if len(eq) else np.nan
                    total_liab=float(li.iloc[0]) if len(li) else np.nan
                    if pd.notna(inc1) and pd.notna(total_equity) and total_equity!=0: roe=inc1/total_equity
                    if pd.notna(total_liab) and pd.notna(total_equity) and total_equity!=0: de_ratio=total_liab/total_equity
                rows.append({"ticker":t,"pe":pe,"ev_ebitda":ev_ebitda,"rev_growth":rev_g,"eps_growth":eps_g,
                             "roe":roe,"de_ratio":de_ratio,"net_margin":net_margin,"gross_margin":gross_margin})
            except Exception:
                rows.append({"ticker":t})
        return pd.DataFrame(rows).set_index("ticker")

    fundamentals = fetch_fundamentals(list(panel.keys()))
    fdf = pd.DataFrame(index=fundamentals.index)
    for col in ["rev_growth","eps_growth","roe","net_margin","gross_margin"]:
        if col in fundamentals.columns: fdf[f"{col}_z"] = zscore_series(fundamentals[col])
    for col in ["pe","ev_ebitda","de_ratio"]:
        if col in fundamentals.columns: fdf[f"{col}_z"] = zscore_series(-fundamentals[col])
    FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fundamentals.index)

    # --- Relative (robust)
    rel_rows=[]
    if enable_rel:
        for t, px in panel.items():
            s = px.reindex(bench_px.index).dropna()
            b = bench_px.reindex(s.index).dropna()
            if len(s) < 100 or len(b) < 100:
                rel_rows.append({"ticker":t,"rel_dma_gap":np.nan,"rel_mom12m":np.nan,"alpha_60d":np.nan})
                continue
            rel_rows.append({"ticker":t, **relative_signals(s, b)})
    rel = pd.DataFrame(rel_rows).set_index("ticker") if rel_rows else pd.DataFrame(index=list(panel.keys()))
    for col in ["rel_dma_gap","rel_mom12m","alpha_60d"]:
        rel[f"{col}_z"] = zscore_series(rel[col]) if col in rel.columns else np.nan
    REL_score = rel[[c for c in ["rel_dma_gap_z","rel_mom12m_z","alpha_60d_z"] if c in rel.columns]].mean(axis=1) if enable_rel else pd.Series(0.0, index=list(panel.keys()))

    # --- Macro
    vix_level = fetch_vix_level()
    MACRO_score = macro_overlay_score(vix_level)

    # --- Combine
    idx = pd.Index(list(panel.keys()))
    out = pd.DataFrame(index=idx)
    out["FUND_score"]  = FUND_score.reindex(idx).fillna(0.0)
    out["TECH_score"]  = TECH_score.reindex(idx).fillna(0.0)
    out["REL_score"]   = REL_score.reindex(idx).fillna(0.0)
    out["MACRO_score"] = MACRO_score

    # weights
    wsum = (w_fund + w_tech + w_rel + w_macro)
    if wsum == 0:
        st.error("All component weights are zero."); return
    wf, wt, wr, wm = w_fund/wsum, w_tech/wsum, w_rel/wsum, w_macro/wsum

    out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wr*out["REL_score"] + wm*out["MACRO_score"]
    out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])

    def bucket(x):
        if x >= 80: return "Strong Buy"
        if x >= 60: return "Buy"
        if x >= 40: return "Hold"
        if x >= 20: return "Sell"
        return "Strong Sell"
    out["RECO"] = out["RATING_0_100"].apply(bucket)

    # show only user tickers
    show_idx = [yf_symbol(t) for t in user_tickers if yf_symbol(t) in out.index]
    table = out.reindex(show_idx).sort_values("RATING_0_100", ascending=False)
    pretty = table.rename(columns=FRIENDLY_NAMES)

    # Header status
    vix_txt = f"{round(vix_level,2)}" if not np.isnan(vix_level) else "N/A"
    st.success(f"VIX: {vix_txt} | Benchmark used: {used_bench} | Peers loaded: {len(panel)}")

    # Ratings
    st.markdown("## 🏁 Ratings")
    st.dataframe(pretty.round(4), use_container_width=True)

    # Why this rating
    st.markdown("## 🔍 Why this rating?")
    for t in show_idx:
        reco = table.loc[t,"RECO"]; sc = table.loc[t,"RATING_0_100"]
        with st.expander(f"{t} — {reco} (Score: {sc:.1f})"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Fundamentals", f"{table.loc[t,'FUND_score']:.3f}")
            c2.metric("Technicals",   f"{table.loc[t,'TECH_score']:.3f}")
            c3.metric("Relative vs Benchmark", f"{table.loc[t,'REL_score']:.3f}")
            c4.metric("Macro (VIX)",  f"{table.loc[t,'MACRO_score']:.3f}")

            # Fundamentals detail (friendly labels)
            if t in fdf.index:
                fmap = {
                    "rev_growth_z":"Revenue growth (YoY)", "eps_growth_z":"EPS growth (YoY)",
                    "roe_z":"Return on equity", "net_margin_z":"Net margin",
                    "gross_margin_z":"Gross margin", "pe_z":"PE (lower better)",
                    "ev_ebitda_z":"EV/EBITDA (lower better)", "de_ratio_z":"Debt/Equity (lower better)"
                }
                series = {nice: float(fdf.loc[t, raw]) for raw, nice in fmap.items()
                          if raw in fdf.columns and pd.notna(fdf.loc[t, raw])}
                if series:
                    st.markdown("**Fundamentals (vs peers)**")
                    st.table(pd.Series(series, name=t).round(3))

            # Technical detail
            if t in tech.index:
                tmap = {
                    "dma_gap_z":"Price vs EMA(100)", "macd_hist_z":"MACD histogram",
                    "rsi_strength_z":"RSI (strength)", "mom12m_z":"12-month momentum"
                }
                series = {nice: float(tech.loc[t, raw]) for raw, nice in tmap.items()
                          if raw in tech.columns and pd.notna(tech.loc[t, raw])}
                if series:
                    st.markdown("**Technicals (vs peers)**")
                    st.table(pd.Series(series, name=t).round(3))
                st.markdown("**Interpretation**")
                st.markdown(explain_technicals_row(tech.loc[t, ["dma_gap","macd_hist","rsi_strength","mom12m"]].to_dict()))

            # Relative detail
            if enable_rel and t in rel.index:
                rmap = {"rel_dma_gap_z":"Ratio EMA gap", "rel_mom12m_z":"Relative momentum (12m)", "alpha_60d_z":"Rolling alpha (60d)"}
                series = {nice: float(rel.loc[t, raw]) for raw, nice in rmap.items()
                          if raw in rel.columns and pd.notna(rel.loc[t, raw])}
                if series:
                    st.markdown("**Relative to benchmark (vs peers)**")
                    st.table(pd.Series(series, name=t).round(3))
                    st.markdown("**Interpretation**")
                    st.markdown(explain_relative_row(rel.loc[t].to_dict()))
                else:
                    st.caption("Relative details not available (insufficient overlap).")

    # Charts
    if show_idx:
        sel = st.selectbox("Choose ticker for charts", show_idx, index=0)
        px = panel.get(sel)
        if px is not None and not px.empty:
            ema20, ema100 = ema(px, 20), ema(px, 100)
            st.subheader("📊 Price & EMAs")
            st.line_chart(pd.DataFrame({"Price": px, "EMA20": ema20, "EMA100": ema100}))
            if enable_rel and not bench_px.empty:
                ratio = (px / bench_px.reindex(px.index)).dropna()
                if not ratio.empty:
                    st.subheader("📊 Relative (Stock / Benchmark)")
                    st.line_chart(pd.DataFrame({"Relative price": ratio}))
        else:
            st.warning("No price data for chart.")

    # Export
    csv_bytes = table.to_csv().encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "ratings.csv", "text/csv")
    xlsx_io = io.BytesIO()
    with pd.ExcelWriter(xlsx_io, engine="openpyxl") as w:
        table.round(4).to_excel(w, sheet_name="Ratings")
    xlsx_bytes = xlsx_io.getvalue()
    st.download_button("⬇️ Download Excel", xlsx_bytes, "ratings.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"ratings_{ts}.xlsx","wb") as f: f.write(xlsx_bytes)
        st.caption(f"Saved Excel locally as ratings_{ts}.xlsx")
    except Exception:
        st.caption("Local save skipped.")

# ---------- Router ----------
if not st.session_state.entered:
    landing()
else:
    st.title("⭐️ Rate My Stock")
    st.caption("Type a ticker. We’ll grab its peers from the index and rate it with friendly explanations.")
    main_app()
