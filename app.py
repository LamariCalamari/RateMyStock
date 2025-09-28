# app.py — ⭐️ Rate My Stock (peer sampling + explainable signals, fixed factors)

import io, os, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------- Page ----------------
st.set_page_config(page_title="Rate My Stock", layout="wide")
st.title("⭐️ Rate My Stock")
st.caption("Pick a stock, pick a squad (peer universe), and get a vibe-checked rating — with friendly explanations for every signal.")

# ------------- Helpers / Math ----------
def yf_symbol(t: str) -> str:
    if not isinstance(t, str): return t
    t = t.strip().upper()
    return t.replace(".", "-")  # BRK.B -> BRK-B for Yahoo

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
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
            df = yf.download(
                group, period=period, interval=interval,
                auto_adjust=True, group_by="ticker", threads=True, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                got = set(df.columns.get_level_values(0))
                for t in group:
                    if t in got:
                        s = df[t]["Close"].dropna()
                        if s.size:
                            frames.append(s.rename(t)); ok.append(t)
                        else:
                            fail.append(t)
                    else:
                        fail.append(t)
            else:
                t = group[0]
                if "Close" in df:
                    s = df["Close"].dropna()
                    if s.size: frames.append(s.rename(t)); ok.append(t)
                    else: fail.append(t)
                else: fail.append(t)
        except Exception:
            fail.extend(group)
        time.sleep(sleep_between)

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    if len(ok) < min_ok:
        to_try = [t for t in tickers if t not in ok]
        for t in to_try:
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
        prices = prices.sort_index()
        prices = prices.loc[:, ~prices.columns.duplicated()]
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
        rows.append({"ticker":ticker,"dma_gap":dma_gap,"macd_hist":macd_hist,"rsi_strength":rsi_strength,"mom12m":mom12m})
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
    try:
        return {yf_symbol(t) for t in yf.tickers_sp500()}
    except Exception:
        return set()

def list_dow30():
    try:
        return {yf_symbol(t) for t in yf.tickers_dow()}
    except Exception:
        return set()

def build_universe(user_tickers, mode, custom_raw="", sample_n=80):
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
    else:  # Auto by membership (fall back to S&P)
        sp, dj = list_sp500(), list_dow30()
        auto = set()
        for t in user:
            if t in sp: auto |= sp
            elif t in dj: auto |= dj
        peers_all = auto if auto else (sp or dj)

    peers = sorted(peers_all.difference(set(user)))
    if len(peers) > sample_n:
        peers = peers[:sample_n]
    return sorted(set(user) | set(peers))

# -------------- Sidebar -----------------
st.sidebar.header("⚙️ Settings")

tickers_input = st.sidebar.text_area("Tickers (comma-separated):", "AAPL, MSFT, NVDA")
user_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

universe_mode = st.sidebar.selectbox(
    "Peer universe",
    ["Auto by index membership", "S&P 500", "Dow 30", "User list only", "Custom (paste list)"],
    index=0
)
peer_sample_n = st.sidebar.slider("Peer sample size (for speed)", 20, 150, 80, 10)
custom_universe_raw = ""
if universe_mode == "Custom (paste list)":
    custom_universe_raw = st.sidebar.text_area("Paste custom peers (comma-separated):", "AMD, AVGO, CRM, COST, NFLX")

benchmark = st.sidebar.selectbox(
    "Benchmark (for relative comparison):",
    ["SPY (S&P 500 ETF)", "^GSPC (S&P 500 Index)", "QQQ (Nasdaq ETF)", "^IXIC (Nasdaq Index)", "^DJI (Dow Jones)"],
    index=0
).split()[0]

history = st.sidebar.selectbox("History window", ["1y", "2y", "5y"], index=0)

with st.expander("ℹ️ Peer universe vs Benchmark"):
    st.markdown(
        "- **Peer universe**: group we compare against to compute cross-sectional z-scores for **Fundamentals/Technicals/Relative**.\n"
        "- **Benchmark**: single index/ETF used for **relative features** (price vs benchmark, 12-month relative momentum, 60-day alpha).\n"
        "You can set both to S&P — they do different jobs."
    )

st.sidebar.markdown("### Component Weights")
w_fund = st.sidebar.slider("Fundamentals", 0.0, 1.0, 0.5, 0.05)
w_tech = st.sidebar.slider("Technicals", 0.0, 1.0, 0.35, 0.05)
w_macro = st.sidebar.slider("Macro (VIX)", 0.0, 1.0, 0.10, 0.05)
w_rel  = st.sidebar.slider("Relative (vs Benchmark)", 0.0, 1.0, 0.05, 0.05)

# Fixed factors (users cannot change)
FUND_HIGHER_BETTER = ["rev_growth", "eps_growth", "roe", "net_margin", "gross_margin"]
FUND_LOWER_BETTER  = ["pe", "ev_ebitda", "de_ratio"]
TECH_PARTS = ["dma_gap", "macd_hist", "rsi_strength", "mom12m"]

# Component toggles (keep these to let users weight/omit whole blocks)
include_fund = st.sidebar.checkbox("Include Fundamentals", True)
include_tech = st.sidebar.checkbox("Include Technicals", True)
include_macro = st.sidebar.checkbox("Include Macro Overlay (VIX)", True)
include_rel = st.sidebar.checkbox("Include Relative-to-Benchmark", True)

run_btn = st.sidebar.button("🚀 Rate it!")

# ---------------- Interpretation helpers ----------------
def label_badge(value, pos_thresh=None, neg_thresh=None, higher_is_better=True, units=""):
    """Return emoji label for a scalar signal."""
    v = value
    if pd.isna(v):
        return "⚪ Neutral"
    if higher_is_better:
        if pos_thresh is None: pos_thresh = 0
        if v > pos_thresh: return f"🟢 Bullish ({v:.3f}{units})"
        if neg_thresh is not None and v < neg_thresh: return f"🔴 Bearish ({v:.3f}{units})"
        return f"⚪ Neutral ({v:.3f}{units})"
    else:
        # lower-is-better case (rare here)
        if neg_thresh is None: neg_thresh = 0
        if v < neg_thresh: return f"🟢 Bullish ({v:.3f}{units})"
        if pos_thresh is not None and v > pos_thresh: return f"🔴 Bearish ({v:.3f}{units})"
        return f"⚪ Neutral ({v:.3f}{units})"

def explain_technicals_row(row):
    notes = []
    # dma_gap: price vs EMA100
    notes.append(f"• **Price vs EMA(100)**: {label_badge(row.get('dma_gap'), pos_thresh=0, units='')}. "
                 "Above EMA can mean up-trend; below can mean weakness or value entry depending on context.")
    # macd_hist
    notes.append(f"• **MACD histogram**: {label_badge(row.get('macd_hist'), pos_thresh=0)}. "
                 "Positive implies bullish momentum building; negative implies momentum fading.")
    # RSI
    rsi_strength = row.get("rsi_strength")
    rsi_val = None
    if pd.notna(rsi_strength):
        rsi_val = 50 + 50*rsi_strength
    if rsi_val is not None:
        if rsi_val >= 60: rsi_note = "🟢 Bullish (RSI≈{:.0f})".format(rsi_val)
        elif rsi_val <= 40: rsi_note = "🔴 Bearish (RSI≈{:.0f})".format(rsi_val)
        else: rsi_note = "⚪ Neutral (RSI≈{:.0f})".format(rsi_val)
    else:
        rsi_note = "⚪ Neutral"
    notes.append(f"• **RSI**: {rsi_note}. 70+ can be overbought; 30− can be oversold.")
    # 12-month momentum
    notes.append(f"• **12-month momentum**: {label_badge(row.get('mom12m'), pos_thresh=0)}. "
                 "Positive = outperformed own past year.")
    return "\n".join(notes)

def explain_relative_row(row):
    notes = []
    notes.append(f"• **Ratio EMA gap (stock/benchmark)**: {label_badge(row.get('rel_dma_gap'), pos_thresh=0)}.")
    notes.append(f"• **12-month relative momentum**: {label_badge(row.get('rel_mom12m'), pos_thresh=0)}.")
    notes.append(f"• **Rolling alpha (60d)**: {label_badge(row.get('alpha_60d'), pos_thresh=0)}. "
                 "Positive alpha = beating benchmark on a risk-adjusted basis recently.")
    return "\n".join(notes)

# -------------- Main flow ---------------
if run_btn:
    # Universe (user + sampled peers)
    universe_tickers = build_universe(user_tickers, universe_mode, custom_universe_raw, peer_sample_n)
    if not universe_tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    # Prices (chunked + fallback)
    with st.spinner("Downloading prices for peer universe..."):
        prices, ok, fail = fetch_prices_chunked_with_fallback(
            universe_tickers, period=history, interval="1d",
            chunk=50, min_ok=min(30, peer_sample_n)
        )
    price_panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size > 0}
    peer_loaded = len(price_panel)
    if peer_loaded < 5:
        st.warning(f"Only {peer_loaded} tickers loaded (rate limits or long history). "
                   f"Try a larger sample size or shorter history (1y).")
    skipped_user = [yf_symbol(t) for t in user_tickers if yf_symbol(t) not in price_panel]
    if skipped_user:
        st.info(f"Skipped (no price data): {', '.join(skipped_user)}")

    if not price_panel:
        st.error("No valid price data downloaded.")
        st.stop()

    # Benchmark
    bench_px = fetch_price_series(benchmark, period=history, interval="1d")
    if bench_px.empty and include_rel:
        st.warning(f"Benchmark {benchmark} not available; relative signals disabled.")
        include_rel = False

    # Technicals (universe z-scores)
    tech = technical_scores(price_panel)
    for col in TECH_PARTS:
        if col in tech.columns and include_tech:
            tech[f"{col}_z"] = zscore_series(tech[col])
        else:
            tech[f"{col}_z"] = np.nan
    TECH_score = (
        tech[[f"{c}_z" for c in TECH_PARTS if f"{c}_z" in tech.columns]].mean(axis=1)
        if include_tech else pd.Series(0.0, index=list(price_panel.keys()))
    )

    # Fundamentals (universe z-scores)
    with st.spinner("Fetching fundamentals for peer universe..."):
        fundamentals = fetch_fundamentals(list(price_panel.keys()))
    fund_scores = pd.DataFrame(index=fundamentals.index)
    if include_fund and not fundamentals.empty:
        for col in FUND_HIGHER_BETTER:
            if col in fundamentals.columns:
                fund_scores[f"{col}_z"] = zscore_series(fundamentals[col])
        for col in FUND_LOWER_BETTER:
            if col in fundamentals.columns:
                fund_scores[f"{col}_z"] = zscore_series(-fundamentals[col])
        FUND_score = fund_scores.mean(axis=1) if len(fund_scores.columns) else pd.Series(0.0, index=fundamentals.index)
    else:
        FUND_score = pd.Series(0.0, index=list(price_panel.keys()))

    # Relative-to-benchmark (universe z-scores)
    rel_rows = []
    if include_rel and not bench_px.empty:
        for t, px in price_panel.items():
            s = px.reindex(bench_px.index).dropna()
            b = bench_px.reindex(s.index).dropna()
            if len(s) < 120 or len(b) < 120:
                rel_rows.append({"ticker": t, "rel_dma_gap": np.nan, "rel_mom12m": np.nan, "alpha_60d": np.nan})
                continue
            feats = relative_signals(s, b)
            rel_rows.append({"ticker": t, **feats})
    rel = pd.DataFrame(rel_rows).set_index("ticker") if rel_rows else pd.DataFrame(index=list(price_panel.keys()))
    for col in ["rel_dma_gap","rel_mom12m","alpha_60d"]:
        rel[f"{col}_z"] = zscore_series(rel[col]) if include_rel and col in rel.columns else np.nan
    REL_score = (
        rel[[c for c in ["rel_dma_gap_z","rel_mom12m_z","alpha_60d_z"] if c in rel.columns]].mean(axis=1)
        if include_rel else pd.Series(0.0, index=list(price_panel.keys()))
    )

    # Macro
    vix_level = fetch_vix_level() if include_macro else np.nan
    MACRO_score = macro_overlay_score(vix_level) if include_macro else 0.0

    # Combine
    idx = pd.Index(list(price_panel.keys()))
    out = pd.DataFrame(index=idx)
    out["FUND_score"]  = FUND_score.reindex(idx).fillna(0.0)
    out["TECH_score"]  = TECH_score.reindex(idx).fillna(0.0)
    out["REL_score"]   = REL_score.reindex(idx).fillna(0.0)
    out["MACRO_score"] = MACRO_score

    wsum = (w_fund if include_fund else 0) + (w_tech if include_tech else 0) + (w_rel if include_rel else 0) + (w_macro if include_macro else 0)
    if wsum == 0:
        st.error("All components are disabled/zero. Enable at least one.")
        st.stop()
    wf = (w_fund / wsum) if include_fund else 0.0
    wt = (w_tech  / wsum) if include_tech else 0.0
    wr = (w_rel   / wsum) if include_rel  else 0.0
    wm = (w_macro / wsum) if include_macro else 0.0

    out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wr*out["REL_score"] + wm*out["MACRO_score"]
    out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])

    def bucket(x):
        if x >= 80: return "Strong Buy"
        if x >= 60: return "Buy"
        if x >= 40: return "Hold"
        if x >= 20: return "Sell"
        return "Strong Sell"
    out["RECO"] = out["RATING_0_100"].apply(bucket)

    # Show only user inputs (scores computed vs peers)
    display_idx = [yf_symbol(t) for t in user_tickers if yf_symbol(t) in out.index]
    out = out.reindex(display_idx).sort_values("RATING_0_100", ascending=False)

    st.success(f"VIX: {round(vix_level,2) if not np.isnan(vix_level) else 'N/A'} | "
               f"Benchmark: {benchmark} | Peer set: {universe_mode} (loaded {peer_loaded})")
    st.subheader("🏁 Ratings")
    st.dataframe(out.round(4))

    # ---------- Explainable breakdowns ----------
    st.subheader("🔍 Why this rating?")
    for t in display_idx:
        with st.expander(f"{t} — {out.loc[t,'RECO']} (Score: {out.loc[t,'RATING_0_100']:.1f})"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fundamentals", f"{out.loc[t,'FUND_score']:.3f}")
            c2.metric("Technicals",   f"{out.loc[t,'TECH_score']:.3f}")
            c3.metric("Relative",     f"{out.loc[t,'REL_score']:.3f}")
            c4.metric("Macro (VIX)",  f"{out.loc[t,'MACRO_score']:.3f}")

            # Fundamentals detail
            if include_fund and 'fund_scores' in locals() and t in fund_scores.index:
                st.markdown("**Fundamentals (z-scores vs peers)**")
                fz = {k: v for k, v in fund_scores.loc[t].dropna().to_dict().items() if k.endswith("_z")}
                if fz:
                    st.table(pd.Series(fz, name=t).round(3))
                    st.caption("Higher-is-better: rev_growth, eps_growth, roe, net_margin, gross_margin. "
                               "Lower-is-better: pe, ev_ebitda, de_ratio.")
                else:
                    st.caption("No fundamental details available.")

            # Technicals detail + plain-English interpretation
            if include_tech and t in tech.index:
                st.markdown("**Technicals (z-scores vs peers)**")
                tz = {f"{c}_z": tech.loc[t, f"{c}_z"] for c in TECH_PARTS
                      if f"{c}_z" in tech.columns and pd.notna(tech.loc[t, f"{c}_z"])}
                if tz: st.table(pd.Series(tz, name=t).round(3))
                row = tech.loc[t, ["dma_gap","macd_hist","rsi_strength","mom12m"]].to_dict()
                st.markdown("**Interpretation**")
                st.markdown(explain_technicals_row(row))

            # Relative detail + interpretation
            if include_rel and t in rel.index:
                st.markdown("**Relative to benchmark (z-scores vs peers)**")
                rz = {f"{c}_z": rel.loc[t, f"{c}_z"] for c in ["rel_dma_gap","rel_mom12m","alpha_60d"]
                      if f"{c}_z" in rel.columns and pd.notna(rel.loc[t, f"{c}_z"])}
                if rz: st.table(pd.Series(rz, name=t).round(3))
                st.markdown("**Interpretation**")
                st.markdown(explain_relative_row(rel.loc[t].to_dict()))

    # ---------- Charts + live EMA callout ----------
    if display_idx:
        sel = st.selectbox("Choose ticker for charts", display_idx)
        px = price_panel.get(sel)
        if px is not None:
            ema20 = ema(px, 20); ema100 = ema(px, 100)
            last = px.iloc[-1]; e20 = ema20.iloc[-1]; e100 = ema100.iloc[-1]
            gap20 = (last - e20) / e20 if e20 else np.nan
            gap100 = (last - e100) / e100 if e100 else np.nan
            # Friendly guidance
            msg = []
            if pd.notna(gap20):
                msg.append(f"Price vs **EMA20**: {label_badge(gap20, pos_thresh=0)}")
            if pd.notna(gap100):
                msg.append(f"Price vs **EMA100**: {label_badge(gap100, pos_thresh=0)}")
            st.info(" | ".join(msg) + " — Above EMA suggests an up-trend; below EMA can be a sign of weakness or potential value entry (context matters).")

            st.subheader("📊 Price & EMAs")
            st.line_chart(pd.DataFrame({"Price": px, "EMA20": ema20, "EMA100": ema100}))

            if include_rel and not bench_px.empty:
                ratio = (px / bench_px.reindex(px.index)).dropna()
                st.subheader("📊 Relative (Stock/Benchmark)")
                st.line_chart(pd.DataFrame({"Rel Price": ratio}))

    # ---------- Export ----------
    csv_bytes = out.to_csv().encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "ratings.csv", "text/csv")

    xlsx_io = io.BytesIO()
    with pd.ExcelWriter(xlsx_io, engine="openpyxl") as writer:
        out.round(4).to_excel(writer, sheet_name="Ratings")
    xlsx_bytes = xlsx_io.getvalue()
    st.download_button("⬇️ Download Excel", xlsx_bytes, "ratings.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"ratings_{ts}.xlsx","wb") as f: f.write(xlsx_bytes)
        st.caption(f"Saved Excel locally as ratings_{ts}.xlsx in {os.getcwd()}")
    except Exception:
        st.caption("Local save skipped.")
