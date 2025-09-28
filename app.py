# app.py — ⭐️ Rate My Stock (landing page + centered search + friendly labels)

import io, os, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------- Page ----------------
st.set_page_config(page_title="Rate My Stock", layout="wide")

# ------------- Small CSS polish -------------
CENTER_CSS = """
<style>
/* narrower content & nicer fonts */
.block-container {max-width: 1100px;}
/* center the hero and search */
.hero {text-align:center; margin-top: 4rem; margin-bottom: .5rem;}
.sub {text-align:center; color:#9aa0a6; margin-bottom: 2rem;}
.search-wrap {display:flex; justify-content:center; margin-top: 1rem; margin-bottom: 1rem;}
.search-inner {width: min(680px, 90%);}
.search-input input {border-radius: 9999px !important; padding: 0.9rem 1.2rem !important; font-size: 1.1rem;}
.small-muted {color:#9aa0a6; font-size: .9rem;}
.table-center .stDataFrame {border-radius: 12px;}
.badge {display:inline-block; padding:.15rem .5rem; border-radius:9999px; font-size:.85rem}
.badge-green {background:#0f5132; color:#d1fae5;}
.badge-red {background:#5c1a1a; color:#fee2e2;}
.badge-grey {background:#2f2f2f; color:#e5e7eb;}
</style>
"""
st.markdown(CENTER_CSS, unsafe_allow_html=True)

# ---------------- Session: landing gate ----------------
if "entered" not in st.session_state:
    st.session_state.entered = False

def enter_app():
    st.session_state.entered = True

# ------------------- Helpers / Math --------------------
def yf_symbol(t: str) -> str:
    if not isinstance(t, str): return t
    t = t.strip().upper()
    return t.replace(".", "-")  # BRK.B -> BRK-B

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100.0

def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
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

# ----------- Data Fetchers (cached) ----
@st.cache_data(show_spinner=False)
def fetch_prices_chunked_with_fallback(tickers, period="1y", interval="1d",
                                       chunk=50, min_ok=30, sleep_between=0.2):
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

    # fallback: single-name pulls
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
            time.sleep(0.15)
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

@st.cache_data(show_spinner=False)
def fetch_fundamentals(tickers):
    rows = []
    for raw in tickers:
        t = yf_symbol(raw)
        try:
            tk = yf.Ticker(t)
            try: inf = tk.info or {}
            except Exception: inf = {}
            try: fin = tk.financials
            except Exception: fin = None
            try: bs  = tk.balance_sheet
            except Exception: bs  = None

            pe = inf.get("trailingPE", np.nan)
            ev = inf.get("enterpriseValue", np.nan)

            ebitda = np.nan
            tot_rev_recent = tot_rev_prev = np.nan
            net_inc_recent = net_inc_prev = np.nan
            if fin is not None and not fin.empty:
                try:
                    s = fin.loc[fin.index.str.contains("EBITDA", case=False)].T.squeeze().dropna()
                    ebitda = float(s.iloc[0]) if len(s) else np.nan
                except Exception:
                    pass
                def latest_two(name):
                    try:
                        s = fin.loc[fin.index.str.contains(name, case=False)].T.squeeze().dropna()
                        return (float(s.iloc[0]) if len(s) else np.nan,
                                float(s.iloc[1]) if len(s) > 1 else np.nan)
                    except Exception:
                        return (np.nan, np.nan)
                tot_rev_recent, tot_rev_prev = latest_two("Total Revenue")
                net_inc_recent, net_inc_prev = latest_two("Net Income")

            ev_ebitda = np.nan
            if pd.notna(ev) and pd.notna(ebitda) and ebitda not in (0, None):
                ev_ebitda = ev / ebitda

            rev_growth = np.nan
            if pd.notna(tot_rev_recent) and pd.notna(tot_rev_prev) and tot_rev_prev != 0:
                rev_growth = tot_rev_recent / tot_rev_prev - 1.0

            eps_growth = np.nan
            shares_out = inf.get("sharesOutstanding", np.nan)
            if (pd.notna(shares_out) and shares_out not in (0, None)
                and pd.notna(net_inc_recent) and pd.notna(net_inc_prev) and net_inc_prev != 0):
                eps_recent = net_inc_recent / shares_out
                eps_prev   = net_inc_prev / shares_out
                if pd.notna(eps_recent) and pd.notna(eps_prev) and eps_prev != 0:
                    eps_growth = eps_recent / eps_prev - 1.0

            roe = np.nan; de_ratio = np.nan
            total_equity = np.nan; total_liab = np.nan
            if bs is not None and not bs.empty:
                try:
                    eq = bs.loc[bs.index.str.contains("Total Stockholder", case=False)].T.squeeze().dropna()
                    total_equity = float(eq.iloc[0]) if len(eq) else np.nan
                except Exception: pass
                try:
                    li = bs.loc[bs.index.str.contains("Total Liabilities", case=False)].T.squeeze().dropna()
                    total_liab = float(li.iloc[0]) if len(li) else np.nan
                except Exception: pass
                if pd.notna(net_inc_recent) and pd.notna(total_equity) and total_equity != 0:
                    roe = net_inc_recent / total_equity
                if pd.notna(total_liab) and pd.notna(total_equity) and total_equity != 0:
                    de_ratio = total_liab / total_equity

            net_margin = np.nan; gross_margin = np.nan
            if pd.notna(net_inc_recent) and pd.notna(tot_rev_recent) and tot_rev_recent != 0:
                net_margin = net_inc_recent / tot_rev_recent
            try:
                gp = fin.loc[fin.index.str.contains("Gross Profit", case=False)].T.squeeze().dropna() if fin is not None and not fin.empty else pd.Series(dtype=float)
                if len(gp):
                    gp_recent = float(gp.iloc[0])
                    if pd.notna(gp_recent) and pd.notna(tot_rev_recent) and tot_rev_recent != 0:
                        gross_margin = gp_recent / tot_rev_recent
            except Exception:
                pass

            rows.append({
                "ticker": t,
                "pe": pe, "ev_ebitda": ev_ebitda,
                "rev_growth": rev_growth, "eps_growth": eps_growth,
                "roe": roe, "de_ratio": de_ratio,
                "net_margin": net_margin, "gross_margin": gross_margin
            })
        except Exception:
            rows.append({"ticker": yf_symbol(raw)})
    return pd.DataFrame(rows).set_index("ticker")

# ----------- Technical / Relative ----------
def technical_scores(price_panel: dict) -> pd.DataFrame:
    rows = []
    for ticker, px in price_panel.items():
        px = px.dropna()
        if len(px) < 130:  # need EMA100 & MACD stability
            continue
        ema100 = ema(px, 100)
        dma_gap = (px.iloc[-1] - ema100.iloc[-1]) / ema100.iloc[-1]
        _, _, hist = macd(px)
        macd_hist = hist.iloc[-1]
        r = rsi(px).iloc[-1]
        rsi_strength = (r - 50.0) / 50.0
        mom12m = np.nan
        if len(px) > 252:
            mom12m = px.iloc[-1] / px.iloc[-253] - 1.0
        rows.append({"ticker": ticker, "dma_gap": dma_gap, "macd_hist": macd_hist,
                     "rsi_strength": rsi_strength, "mom12m": mom12m})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

def rolling_beta_alpha(stock_px: pd.Series, bench_px: pd.Series, window: int = 60):
    df = pd.DataFrame({"r_s": stock_px.pct_change(), "r_b": bench_px.pct_change()}).dropna()
    out = []
    for i in range(window, len(df)):
        chunk = df.iloc[i-window:i]
        if chunk["r_b"].std(ddof=0) == 0:
            beta = np.nan
        else:
            cov = np.cov(chunk["r_s"], chunk["r_b"])[0,1]
            beta = cov / np.var(chunk["r_b"])
        alpha = df["r_s"].iloc[i] - beta * df["r_b"].iloc[i] if pd.notna(beta) else np.nan
        out.append((df.index[i], beta, alpha))
    return pd.DataFrame(out, columns=["date","beta","alpha"]).set_index("date")

def relative_signals(stock_px: pd.Series, bench_px: pd.Series):
    ratio = (stock_px / bench_px).dropna()
    rel = {"rel_dma_gap": np.nan, "rel_mom12m": np.nan, "alpha_60d": np.nan}
    if len(ratio) >= 120:
        ema100 = ratio.ewm(span=100, adjust=False).mean()
        rel["rel_dma_gap"] = (ratio.iloc[-1] - ema100.iloc[-1]) / ema100.iloc[-1]
    if len(ratio) > 252:
        rel["rel_mom12m"] = ratio.iloc[-1] / ratio.iloc[-253] - 1.0
    ba = rolling_beta_alpha(stock_px, bench_px, window=60)
    if not ba.empty and "alpha" in ba:
        s = ba["alpha"].dropna()
        if s.size: rel["alpha_60d"] = s.iloc[-1]
    return rel

# ---------- Universe builders -----------
def list_sp500():
    try:   return {yf_symbol(t) for t in yf.tickers_sp500()}
    except Exception: return set()

def list_dow30():
    try:   return {yf_symbol(t) for t in yf.tickers_dow()}
    except Exception: return set()

def build_universe(user_tickers, mode, custom_raw="", sample_n=80):
    """Always returns user + sampled peers so 1 ticker is compared to peers."""
    user = [yf_symbol(t) for t in user_tickers if t]
    if mode == "S&P 500":
        peers_all = list_sp500()
    elif mode == "Dow 30":
        peers_all = list_dow30()
    elif mode == "User list only":
        return user
    elif mode == "Custom (paste list)":
        custom = {yf_symbol(t) for t in custom_raw.split(",") if t.strip()}
        return sorted(set(user) | custom)
    else:  # Auto detect
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
        peers_all = auto if auto else (sp or dj)

    peers = sorted(peers_all.difference(set(user)))
    if len(peers) > sample_n: peers = peers[:sample_n]
    return sorted(set(user) | set(peers))

# ---------------- Interpretation helpers ----------------
def badge(v):
    if pd.isna(v): return '<span class="badge badge-grey">Neutral</span>'
    if v > 0:      return '<span class="badge badge-green">Bullish</span>'
    if v < 0:      return '<span class="badge badge-red">Bearish</span>'
    return '<span class="badge badge-grey">Neutral</span>'

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
    notes.append(f"• **Price vs EMA(100)**: {label_badge(row.get('dma_gap'), pos_thresh=0)}. Above EMA can mean up-trend; below can mean weakness or a value entry (context matters).")
    notes.append(f"• **MACD histogram**: {label_badge(row.get('macd_hist'), pos_thresh=0)}. Positive = bullish momentum building; negative = momentum fading.")
    rsi_strength = row.get("rsi_strength"); rsi_val = 50 + 50*rsi_strength if pd.notna(rsi_strength) else None
    if rsi_val is not None:
        if   rsi_val >= 60: rsi_note = f"🟢 Bullish (RSI≈{rsi_val:.0f})"
        elif rsi_val <= 40: rsi_note = f"🔴 Bearish (RSI≈{rsi_val:.0f})"
        else:               rsi_note = f"⚪ Neutral (RSI≈{rsi_val:.0f})"
    else:
        rsi_note = "⚪ Neutral"
    notes.append(f"• **RSI**: {rsi_note}. 70+ can be overbought; 30− can be oversold.")
    notes.append(f"• **12-month momentum**: {label_badge(row.get('mom12m'), pos_thresh=0)}. Positive = outperformed its own past year.")
    return "\n".join(notes)

def explain_relative_row(row):
    notes = []
    notes.append(f"• **Ratio EMA gap (stock/benchmark)**: {label_badge(row.get('rel_dma_gap'), pos_thresh=0)}.")
    notes.append(f"• **12-month relative momentum**: {label_badge(row.get('rel_mom12m'), pos_thresh=0)}.")
    notes.append(f"• **Rolling alpha (60d)**: {label_badge(row.get('alpha_60d'), pos_thresh=0)}. Positive = beating benchmark on a risk-adjusted basis recently.")
    return "\n".join(notes)

# ---------------- Fixed factors ----------------
FUND_HIGHER_BETTER = ["rev_growth", "eps_growth", "roe", "net_margin", "gross_margin"]
FUND_LOWER_BETTER  = ["pe", "ev_ebitda", "de_ratio"]
TECH_PARTS = ["dma_gap", "macd_hist", "rsi_strength", "mom12m"]

FRIENDLY_NAMES = {
    "FUND_score": "Fundamentals",
    "TECH_score": "Technicals",
    "REL_score":  "Relative vs Benchmark",
    "MACRO_score":"Macro (VIX)",
    "COMPOSITE":  "Composite",
    "RATING_0_100":"Score (0–100)",
    "RECO":       "Recommendation",
    # fundamentals
    "rev_growth_z":"Revenue growth (YoY)",
    "eps_growth_z":"EPS growth (YoY)",
    "roe_z":"Return on equity",
    "net_margin_z":"Net margin",
    "gross_margin_z":"Gross margin",
    "pe_z":"PE (lower better)",
    "ev_ebitda_z":"EV/EBITDA (lower better)",
    "de_ratio_z":"Debt/Equity (lower better)",
    # technicals
    "dma_gap_z":"Price vs EMA(100)",
    "macd_hist_z":"MACD histogram",
    "rsi_strength_z":"RSI (strength)",
    "mom12m_z":"12-month momentum",
    # relative
    "rel_dma_gap_z":"Ratio EMA gap",
    "rel_mom12m_z":"Relative momentum (12m)",
    "alpha_60d_z":"Rolling alpha (60d)"
}

# ---------------- Landing or App ----------------
def landing():
    st.markdown('<div class="hero"><h1>⭐️ Rate My Stock</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">A clean, explainable rating for any stock — benchmarked against its peers.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.button("Enter", type="primary", use_container_width=True, on_click=enter_app)
    st.markdown('<p class="small-muted" style="text-align:center;margin-top:1rem;">Made with Streamlit · yfinance · pandas</p>', unsafe_allow_html=True)

def main_app():
    # --- Centered search box ---
    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    ticker = st.text_input(" ", "AAPL", key="ticker", label_visibility="collapsed", placeholder="Type a ticker (e.g., AAPL)",
                           help="Enter one or more tickers separated by commas.")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # --- Advanced settings (collapsible) ---
    with st.expander("Advanced settings"):
        c1, c2, c3 = st.columns(3)
        with c1:
            universe_mode = st.selectbox(
                "Peer universe (what we compare against)",
                ["Auto by index membership", "S&P 500", "Dow 30", "User list only", "Custom (paste list)"],
                index=0
            )
        with c2:
            peer_sample_n = st.slider("Peer sample size", 20, 150, 80, 10,
                                      help="We sample peers from the chosen universe for speed/reliability.")
        with c3:
            history = st.selectbox("History", ["1y", "2y", "5y"], index=0)

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            benchmark = st.selectbox("Benchmark (for relative features)",
                                     ["SPY (S&P 500 ETF)", "^GSPC (S&P 500 Index)",
                                      "QQQ (Nasdaq ETF)", "^IXIC (Nasdaq Index)", "^DJI (Dow Jones)"],
                                     index=0).split()[0]
        with c5:
            w_fund = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.5, 0.05)
        with c6:
            w_tech = st.slider("Weight: Technicals", 0.0, 1.0, 0.35, 0.05)
        with c7:
            w_macro = st.slider("Weight: Macro (VIX)", 0.0, 1.0, 0.10, 0.05)
        c8, c9 = st.columns(2)
        with c8:
            w_rel  = st.slider("Weight: Relative vs Benchmark", 0.0, 1.0, 0.05, 0.05)
        with c9:
            include_macro = st.checkbox("Include Macro overlay (VIX)", True)

    # defaults if expander closed
    if "universe_mode" not in locals(): universe_mode = "Auto by index membership"
    if "peer_sample_n" not in locals(): peer_sample_n = 80
    if "history" not in locals(): history = "1y"
    if "benchmark" not in locals(): benchmark = "SPY"
    if "w_fund" not in locals(): w_fund = 0.5
    if "w_tech" not in locals(): w_tech = 0.35
    if "w_macro" not in locals(): w_macro = 0.10
    if "w_rel" not in locals():  w_rel  = 0.05
    if "include_macro" not in locals(): include_macro = True

    # Run button centered
    st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
    run_btn = st.button("Rate it 🚀", type="primary", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if not run_btn:
        return

    # -------- build universe from ticker(s) --------
    user_tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]
    if not user_tickers:
        st.error("Please enter at least one ticker."); return

    def list_sp500():
        try:   return {yf_symbol(t) for t in yf.tickers_sp500()}
        except Exception: return set()
    def list_dow30():
        try:   return {yf_symbol(t) for t in yf.tickers_dow()}
        except Exception: return set()

    def build_universe(user_tickers, mode, sample_n=80, custom_raw=""):
        user = [yf_symbol(t) for t in user_tickers if t]
        if mode == "S&P 500":
            peers_all = list_sp500()
        elif mode == "Dow 30":
            peers_all = list_dow30()
        elif mode == "User list only":
            return user
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
            peers_all = auto if auto else (sp or dj)
        peers = sorted(peers_all.difference(set(user)))
        if len(peers) > sample_n: peers = peers[:sample_n]
        return sorted(set(user) | set(peers))

    universe = build_universe(user_tickers, universe_mode, peer_sample_n)
    if not universe:
        st.error("Could not build a peer set."); return

    # -------- prices and benchmark --------
    with st.spinner("Downloading prices for peers…"):
        prices, ok, fail = fetch_prices_chunked_with_fallback(
            universe, period=history, interval="1d",
            chunk=50, min_ok=min(30, max(peer_sample_n, 30))
        )
    panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size > 0}
    loaded = len(panel)
    if loaded < 5:
        st.warning(f"Only {loaded} tickers loaded; try a larger sample or 1y history.")
    if not panel:
        st.error("No price data for peers."); return

    bench_px = fetch_price_series(benchmark, period=history, interval="1d")
    include_rel = True
    if bench_px.empty:
        st.warning(f"Benchmark {benchmark} not available; relative features disabled.")
        include_rel = False

    # -------- scores --------
    tech = technical_scores(panel)
    for col in TECH_PARTS:
        tech[f"{col}_z"] = zscore_series(tech[col]) if col in tech.columns else np.nan
    TECH_score = tech[[f"{c}_z" for c in TECH_PARTS if f"{c}_z" in tech.columns]].mean(axis=1)

    fundamentals = fetch_fundamentals(list(panel.keys()))
    fdf = pd.DataFrame(index=fundamentals.index)
    for col in FUND_HIGHER_BETTER:
        if col in fundamentals.columns: fdf[f"{col}_z"] = zscore_series(fundamentals[col])
    for col in FUND_LOWER_BETTER:
        if col in fundamentals.columns: fdf[f"{col}_z"] = zscore_series(-fundamentals[col])
    FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fundamentals.index)

    rel_rows = []
    if include_rel:
        for t, px in panel.items():
            s = px.reindex(bench_px.index).dropna()
            b = bench_px.reindex(s.index).dropna()
            if len(s) < 120 or len(b) < 120:
                rel_rows.append({"ticker": t, "rel_dma_gap": np.nan, "rel_mom12m": np.nan, "alpha_60d": np.nan})
                continue
            feats = relative_signals(s, b)
            rel_rows.append({"ticker": t, **feats})
    rel = pd.DataFrame(rel_rows).set_index("ticker") if rel_rows else pd.DataFrame(index=list(panel.keys()))
    for col in ["rel_dma_gap","rel_mom12m","alpha_60d"]:
        rel[f"{col}_z"] = zscore_series(rel[col]) if include_rel and col in rel.columns else np.nan
    REL_score = rel[[c for c in ["rel_dma_gap_z","rel_mom12m_z","alpha_60d_z"] if c in rel.columns]].mean(axis=1) if include_rel else pd.Series(0.0, index=list(panel.keys()))

    vix_level = fetch_vix_level()
    MACRO_score = macro_overlay_score(vix_level) if include_macro else 0.0

    idx = pd.Index(list(panel.keys()))
    out = pd.DataFrame(index=idx)
    out["FUND_score"]  = FUND_score.reindex(idx).fillna(0.0)
    out["TECH_score"]  = TECH_score.reindex(idx).fillna(0.0)
    out["REL_score"]   = REL_score.reindex(idx).fillna(0.0)
    out["MACRO_score"] = MACRO_score

    wsum = w_fund + w_tech + w_rel + (w_macro if include_macro else 0)
    wf, wt, wr, wm = w_fund/wsum, w_tech/wsum, w_rel/wsum, (w_macro/wsum if include_macro else 0.0)

    out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wr*out["REL_score"] + wm*out["MACRO_score"]
    out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])

    def bucket(x):
        if x >= 80: return "Strong Buy"
        if x >= 60: return "Buy"
        if x >= 40: return "Hold"
        if x >= 20: return "Sell"
        return "Strong Sell"
    out["RECO"] = out["RATING_0_100"].apply(bucket)

    # show ONLY the user tickers (scores vs peers)
    show_idx = [yf_symbol(t) for t in user_tickers if yf_symbol(t) in out.index]
    table = out.reindex(show_idx).sort_values("RATING_0_100", ascending=False)

    # friendly columns
    pretty = table.rename(columns={k: v for k, v in FRIENDLY_NAMES.items() if k in table.columns})

    # -------- header badges --------
    vix_txt = f"{round(vix_level,2)}" if not np.isnan(vix_level) else "N/A"
    st.success(f"VIX: {vix_txt} | Benchmark: {benchmark} | Peer set: {universe_mode} (loaded {loaded})")

    # -------- Ratings table --------
    st.markdown("## 🏁 Ratings")
    st.dataframe(pretty.round(4), use_container_width=True)

    # -------- Why this rating (friendly labels + badges) --------
    st.markdown("## 🔍 Why this rating?")
    for t in show_idx:
        reco = table.loc[t, "RECO"]; sc = table.loc[t, "RATING_0_100"]
        with st.expander(f"{t} — {reco} (Score: {sc:.1f})"):
            # summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fundamentals", f"{table.loc[t,'FUND_score']:.3f}")
            c2.metric("Technicals",   f"{table.loc[t,'TECH_score']:.3f}")
            c3.metric("Relative vs Benchmark", f"{table.loc[t,'REL_score']:.3f}")
            c4.metric("Macro (VIX)",  f"{table.loc[t,'MACRO_score']:.3f}")

            # fundamentals detail
            if t in fdf.index:
                frow = {}
                for raw, nice in [("rev_growth_z","Revenue growth (YoY)"),
                                  ("eps_growth_z","EPS growth (YoY)"),
                                  ("roe_z","Return on equity"),
                                  ("net_margin_z","Net margin"),
                                  ("gross_margin_z","Gross margin"),
                                  ("pe_z","PE (lower better)"),
                                  ("ev_ebitda_z","EV/EBITDA (lower better)"),
                                  ("de_ratio_z","Debt/Equity (lower better)")]:
                    if raw in fdf.columns and pd.notna(fdf.loc[t, raw]):
                        frow[nice] = float(fdf.loc[t, raw])
                if frow:
                    st.markdown("**Fundamentals (vs peers)**")
                    st.table(pd.Series(frow, name=t).round(3))

            # technicals detail + human text
            if t in tech.index:
                st.markdown("**Technicals (vs peers)**")
                tz = {}
                mapping = {"dma_gap_z":"Price vs EMA(100)", "macd_hist_z":"MACD histogram",
                           "rsi_strength_z":"RSI (strength)", "mom12m_z":"12-month momentum"}
                for raw, nice in mapping.items():
                    if raw in tech.columns and pd.notna(tech.loc[t, raw]):
                        tz[nice] = float(tech.loc[t, raw])
                if tz: st.table(pd.Series(tz, name=t).round(3))

                row = tech.loc[t, ["dma_gap","macd_hist","rsi_strength","mom12m"]].to_dict()
                st.markdown("**Interpretation**")
                st.markdown(explain_technicals_row(row))

            # relative detail + text
            if t in rel.index:
                rz = {}
                mapping = {"rel_dma_gap_z":"Ratio EMA gap", "rel_mom12m_z":"Relative momentum (12m)", "alpha_60d_z":"Rolling alpha (60d)"}
                for raw, nice in mapping.items():
                    if raw in rel.columns and pd.notna(rel.loc[t, raw]):
                        rz[nice] = float(rel.loc[t, raw])
                if rz:
                    st.markdown("**Relative to benchmark (vs peers)**")
                    st.table(pd.Series(rz, name=t).round(3))
                    st.markdown("**Interpretation**")
                    st.markdown(explain_relative_row(rel.loc[t].to_dict()))

    # -------- Charts + EMA callouts --------
    if show_idx:
        sel = st.selectbox("Choose ticker for charts", show_idx, index=0)
        px = panel.get(sel)
        if px is not None:
            ema20, ema100 = ema(px,20), ema(px,100)
            last = px.iloc[-1]; e20 = ema20.iloc[-1]; e100 = ema100.iloc[-1]
            gap20 = (last - e20)/e20 if e20 else np.nan
            gap100= (last - e100)/e100 if e100 else np.nan
            msg=[]
            if pd.notna(gap20):  msg.append(f"Price vs **EMA20**: {label_badge(gap20, pos_thresh=0)}")
            if pd.notna(gap100): msg.append(f"Price vs **EMA100**: {label_badge(gap100, pos_thresh=0)}")
            st.info(" | ".join(msg) + " — Above EMA suggests an up-trend; below EMA can signal weakness or potential value entry (context matters).")

            st.subheader("📊 Price & EMAs")
            st.line_chart(pd.DataFrame({"Price": px, "EMA20": ema20, "EMA100": ema100}))

            # relative chart
            bench = bench_px
            if not bench.empty:
                ratio = (px / bench.reindex(px.index)).dropna()
                st.subheader("📊 Relative (Stock / Benchmark)")
                st.line_chart(pd.DataFrame({"Relative price": ratio}))

    # -------- Export --------
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

# ---------------- Router ----------------
if not st.session_state.entered:
    landing()
else:
    main_app()
