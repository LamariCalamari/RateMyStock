# app.py — Rate My Stock (robust peer fallback + chart guard)

import io, os, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Rate My Stock", layout="wide")

# ---------- CSS ----------
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

if "entered" not in st.session_state:
    st.session_state.entered = True  # skip landing for now to focus on the fix; set False if you want splash again

# ---------- helpers ----------
def yf_symbol(t):
    if not isinstance(t, str): return t
    return t.strip().upper().replace(".", "-")

def zscore_series(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(); sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd): return pd.Series(0.0, index=s.index)
    return (s - mu)/sd

def percentile_rank(s): return s.rank(pct=True)*100.0
def ema(x, span): return x.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return (100 - (100/(1+rs))).fillna(50.0)

def macd(series, fast=12, slow=26, signal=9):
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def macro_overlay_score(vix):
    if pd.isna(vix): return 0.5
    if vix <= 15: return 1.0
    if vix >= 35: return 0.0
    return 1.0 - (vix-15)/20.0

# ---------- robust data fetch ----------
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

    # single-name fallback if too few ok
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
            time.sleep(0.12)

    if not prices.empty:
        prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return prices, ok, [t for t in tickers if t not in ok]

@st.cache_data(show_spinner=False)
def fetch_price_series(ticker, period="1y", interval="1d"):
    t = yf_symbol(ticker)
    try:
        df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
        if not df.empty and "Close" in df:
            return df["Close"].rename(t)
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fetch_vix_level():
    try:
        vix = yf.Ticker("^VIX").history(period="6mo", interval="1d")
        if not vix.empty: return float(vix["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

# ---------- FALLBACK PEER LISTS ----------
# A solid subset of S&P 500 mega/large caps (100+) so z-scores are meaningful
SP500_FALLBACK = [ "AAPL","MSFT","AMZN","NVDA","META","GOOGL","GOOG","TSLA","AVGO","BRK-B","UNH","LLY","V","JPM","JNJ","XOM",
"PG","MA","HD","CVX","MRK","ABBV","PEP","KO","COST","BAC","ADBE","WMT","CRM","PFE","CSCO","ACN","MCD","TMO","NFLX","AMD","DHR",
"NKE","LIN","ABT","INTC","TXN","DIS","AMAT","PM","NEE","COP","MS","LOW","HON","BMY","QCOM","IBM","UNP","SBUX","INTU","CAT",
"GS","LMT","RTX","BLK","BKNG","IBM","AXP","GE","NOW","AMT","MDT","ISRG","ADI","ELV","PLD","DE","ZTS","SPGI","MDLZ","T",
"USB","REGN","MU","MMC","ATVI","PGR","SYK","CI","SCHW","GILD","PNC","C","ETN","ADP","CB","SO","EQIX","TJX","BDX","DUK",
"SHW","CL","APH","MRNA","FISV","AON","CTAS","FDX","EOG","KLAC","MAR","CSX","CCI","ORLY","ILMN" ]

DOW30_FALLBACK = [ "AAPL","MSFT","JPM","V","JNJ","WMT","PG","UNH","DIS","HD","INTC","IBM","KO","MCD","NKE",
"TRV","VZ","CSCO","MRK","PFE","CAT","AXP","BA","MMM","GS","CVX","DD","WBA","HON","CRM" ]

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

# ---------- feature builders ----------
def technical_scores(price_panel):
    rows = []
    for ticker, px in price_panel.items():
        px = px.dropna()
        if len(px) < 130:  # need EMA100 & MACD
            continue
        ema100 = ema(px, 100)
        dma_gap = (px.iloc[-1] - ema100.iloc[-1]) / (ema100.iloc[-1] if ema100.iloc[-1] else np.nan)
        _, _, hist = macd(px)
        macd_hist = hist.iloc[-1]
        r = rsi(px).iloc[-1]
        rsi_strength = (r - 50.0)/50.0
        mom12m = np.nan
        if len(px) > 252:
            mom12m = px.iloc[-1] / px.iloc[-253] - 1.0
        rows.append({"ticker":ticker,"dma_gap":dma_gap,"macd_hist":macd_hist,"rsi_strength":rsi_strength,"mom12m":mom12m})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

def rolling_beta_alpha(stock_px, bench_px, window=60):
    df = pd.DataFrame({"r_s":stock_px.pct_change(), "r_b":bench_px.pct_change()}).dropna()
    out=[]
    for i in range(window, len(df)):
        ch = df.iloc[i-window:i]
        if ch["r_b"].std(ddof=0)==0: beta=np.nan
        else:
            cov = np.cov(ch["r_s"], ch["r_b"])[0,1]
            beta = cov/np.var(ch["r_b"])
        alpha = df["r_s"].iloc[i] - (beta*df["r_b"].iloc[i] if pd.notna(beta) else 0)
        out.append((df.index[i], beta, alpha))
    return pd.DataFrame(out, columns=["date","beta","alpha"]).set_index("date")

def relative_signals(stock_px, bench_px):
    ratio = (stock_px/bench_px).dropna()
    d = {"rel_dma_gap":np.nan,"rel_mom12m":np.nan,"alpha_60d":np.nan}
    if len(ratio)>=120:
        ema100 = ratio.ewm(span=100, adjust=False).mean()
        d["rel_dma_gap"] = (ratio.iloc[-1]-ema100.iloc[-1])/(ema100.iloc[-1] if ema100.iloc[-1] else np.nan)
    if len(ratio)>252:
        d["rel_mom12m"] = ratio.iloc[-1]/ratio.iloc[-253]-1.0
    ba = rolling_beta_alpha(stock_px, bench_px, window=60)
    if not ba.empty and "alpha" in ba:
        s = ba["alpha"].dropna()
        if s.size: d["alpha_60d"]=s.iloc[-1]
    return d

# ---------- UI ----------
st.title("⭐️ Rate My Stock")
st.caption("Type a ticker. We’ll grab its peers from the index and rate it with friendly explanations.")

with st.expander("Advanced settings", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        universe_mode = st.selectbox("Peer universe", ["Auto by index membership","S&P 500","Dow 30"], index=0)
    with c2:
        peer_sample_n = st.slider("Peer sample size", 30, 200, 120, 10)
    with c3:
        history = st.selectbox("History", ["1y","2y","5y"], index=0)
    c4,c5,c6,c7 = st.columns(4)
    with c4:
        benchmark = st.selectbox("Benchmark", ["SPY","^GSPC","QQQ","^IXIC","^DJI"], index=0)
    with c5:
        w_fund = st.slider("Weight: Fundamentals", 0.0, 1.0, 0.5, 0.05)
    with c6:
        w_tech = st.slider("Weight: Technicals", 0.0, 1.0, 0.35, 0.05)
    with c7:
        w_rel  = st.slider("Weight: Relative", 0.0, 1.0, 0.1, 0.05)
    c8,c9 = st.columns(2)
    with c8:
        w_macro = st.slider("Weight: Macro (VIX)", 0.0, 1.0, 0.05, 0.05)
    with c9:
        show_debug = st.checkbox("Show debug", False)

st.markdown('<div class="search-wrap"><div class="search-inner">', unsafe_allow_html=True)
tickers_input = st.text_input(" ", "AAPL", label_visibility="collapsed", help="One or more tickers (comma-separated)")
run_btn = st.button("Rate it 🚀", type="primary", use_container_width=True)
st.markdown('</div></div>', unsafe_allow_html=True)

if run_btn:
    user_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not user_tickers:
        st.error("Please enter at least one ticker."); st.stop()

    # build universe from index with FALLBACK lists
    def build_universe(user_tickers, mode, sample_n=120):
        user = [yf_symbol(t) for t in user_tickers]
        if mode=="S&P 500":
            peers_all = list_sp500()
        elif mode=="Dow 30":
            peers_all = list_dow30()
        else:
            sp = list_sp500(); dj = list_dow30()
            peers_all=set()
            if len(user)==1:
                t=user[0]
                if t in sp: peers_all=sp
                elif t in dj: peers_all=dj
            else:
                for t in user:
                    if t in sp: peers_all |= sp
                    elif t in dj: peers_all |= dj
            if not peers_all: peers_all = sp or dj or set(SP500_FALLBACK)
        peers = sorted(peers_all.difference(set(user)))
        if len(peers) > sample_n: peers = peers[:sample_n]
        return sorted(set(user)|set(peers))

    universe = build_universe(user_tickers, universe_mode, peer_sample_n)

    # pull prices (attempt 1)
    prices, ok, fail = fetch_prices_chunked_with_fallback(universe, period=history, interval="1d",
                                                          chunk=50, min_ok=min(50, max(30, int(peer_sample_n*0.6))))
    panel = {t: prices[t].dropna() for t in ok if t in prices.columns and prices[t].dropna().size>0}

    # if too few, RETRY with hard fallback S&P subset
    if len(panel) < 10:
        retry_universe = sorted(set([*user_tickers, *SP500_FALLBACK]))[:max(peer_sample_n, 120)]
        prices2, ok2, _ = fetch_prices_chunked_with_fallback(retry_universe, period="1y", interval="1d",
                                                             chunk=50, min_ok=60)
        panel = {t: prices2[t].dropna() for t in ok2 if t in prices2.columns and prices2[t].dropna().size>0}

    loaded = len(panel)
    if show_debug:
        st.info(f"Loaded peers: {loaded}")

    if loaded < 5:
        st.warning(f"Only {loaded} tickers loaded (rate limits or history). Try 1y and larger peer sample.")
    if not panel:
        st.error("No peer prices loaded."); st.stop()

    bench_px = fetch_price_series(benchmark, period=history, interval="1d")

    # ---- compute scores
    tech = technical_scores(panel)
    for c in ["dma_gap","macd_hist","rsi_strength","mom12m"]:
        tech[f"{c}_z"] = zscore_series(tech[c]) if c in tech.columns else np.nan
    TECH_score = tech[[f"{c}_z" for c in ["dma_gap","macd_hist","rsi_strength","mom12m"] if f"{c}_z" in tech.columns]].mean(axis=1)

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
                ebitda=np.nan
                if fin is not None and not fin.empty:
                    try:
                        s=fin.loc[fin.index.str.contains("EBITDA", case=False)].T.squeeze().dropna()
                        ebitda=float(s.iloc[0]) if len(s) else np.nan
                    except Exception: pass
                    def latest_two(name):
                        try:
                            s=fin.loc[fin.index.str.contains(name, case=False)].T.squeeze().dropna()
                            return (float(s.iloc[0]) if len(s) else np.nan, float(s.iloc[1]) if len(s)>1 else np.nan)
                        except Exception: return (np.nan,np.nan)
                    rev1, rev0 = latest_two("Total Revenue")
                    inc1, inc0 = latest_two("Net Income")
                else:
                    rev1=rev0=inc1=inc0=np.nan
                ev_ebitda=np.nan
                if pd.notna(ev) and pd.notna(ebitda) and ebitda not in (0,None): ev_ebitda=ev/ebitda
                rev_g=np.nan
                if pd.notna(rev1) and pd.notna(rev0) and rev0!=0: rev_g = rev1/rev0-1.0
                eps_g=np.nan
                shares=inf.get("sharesOutstanding", np.nan)
                if pd.notna(shares) and shares not in (0,None) and pd.notna(inc1) and pd.notna(inc0) and inc0!=0:
                    eps1=inc1/shares; eps0=inc0/shares
                    if pd.notna(eps1) and pd.notna(eps0) and eps0!=0: eps_g=eps1/eps0-1.0
                roe=np.nan; de_ratio=np.nan; net_margin=np.nan; gross_margin=np.nan
                if fin is not None and not fin.empty:
                    if pd.notna(inc1) and pd.notna(rev1) and rev1!=0: net_margin=inc1/rev1
                    try:
                        gp=fin.loc[fin.index.str.contains("Gross Profit", case=False)].T.squeeze().dropna()
                        if len(gp) and pd.notna(rev1) and rev1!=0: gross_margin=float(gp.iloc[0])/rev1
                    except Exception: pass
                if bs is not None and not bs.empty:
                    try:
                        eq=bs.loc[bs.index.str_contains("Total Stockholder", case=False)].T.squeeze().dropna()
                    except Exception:
                        eq=pd.Series(dtype=float)
                    try:
                        li=bs.loc[bs.index.str_contains("Total Liabilities", case=False)].T.squeeze().dropna()
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
        if col in fundamentals.columns: fdf[f"{col}_z"]=zscore_series(fundamentals[col])
    for col in ["pe","ev_ebitda","de_ratio"]:
        if col in fundamentals.columns: fdf[f"{col}_z"]=zscore_series(-fundamentals[col])
    FUND_score = fdf.mean(axis=1) if len(fdf.columns) else pd.Series(0.0, index=fundamentals.index)

    rel_rows=[]
    if not bench_px.empty:
        for t, px in panel.items():
            s = px.reindex(bench_px.index).dropna()
            b = bench_px.reindex(s.index).dropna()
            if len(s)<120 or len(b)<120:
                rel_rows.append({"ticker":t,"rel_dma_gap":np.nan,"rel_mom12m":np.nan,"alpha_60d":np.nan}); continue
            rel_rows.append({"ticker":t, **relative_signals(s,b)})
    rel = pd.DataFrame(rel_rows).set_index("ticker") if rel_rows else pd.DataFrame(index=list(panel.keys()))
    for c in ["rel_dma_gap","rel_mom12m","alpha_60d"]:
        rel[f"{c}_z"]=zscore_series(rel[c]) if c in rel.columns else np.nan
    REL_score = rel[[c for c in ["rel_dma_gap_z","rel_mom12m_z","alpha_60d_z"] if c in rel.columns]].mean(axis=1) if not rel.empty else pd.Series(0.0, index=list(panel.keys()))

    vix_level = fetch_vix_level()
    MACRO_score = macro_overlay_score(vix_level)

    idx = pd.Index(list(panel.keys()))
    out = pd.DataFrame(index=idx)
    out["FUND_score"]=FUND_score.reindex(idx).fillna(0.0)
    out["TECH_score"]=TECH_score.reindex(idx).fillna(0.0)
    out["REL_score"]=REL_score.reindex(idx).fillna(0.0)
    out["MACRO_score"]=MACRO_score

    wsum = w_fund + w_tech + w_rel + w_macro
    wf,wt,wr,wm = w_fund/wsum, w_tech/wsum, w_rel/wsum, w_macro/wsum
    out["COMPOSITE"] = wf*out["FUND_score"] + wt*out["TECH_score"] + wr*out["REL_score"] + wm*out["MACRO_score"]
    out["RATING_0_100"] = percentile_rank(out["COMPOSITE"])

    def bucket(x):
        if x>=80: return "Strong Buy"
        if x>=60: return "Buy"
        if x>=40: return "Hold"
        if x>=20: return "Sell"
        return "Strong Sell"
    out["RECO"] = out["RATING_0_100"].apply(bucket)

    show_idx = [yf_symbol(t) for t in user_tickers if yf_symbol(t) in out.index]
    table = out.reindex(show_idx).sort_values("RATING_0_100", ascending=False)

    vix_txt = f"{round(vix_level,2)}" if not np.isnan(vix_level) else "N/A"
    st.success(f"VIX: {vix_txt} | Benchmark: {benchmark} | Peers loaded: {loaded}")

    st.subheader("🏁 Ratings")
    st.dataframe(table.round(4), use_container_width=True)

    # ---------- charts (guarded) ----------
    if show_idx:
        sel = st.selectbox("Choose ticker for charts", show_idx, index=0)
        px = panel.get(sel)
        if px is None or px.empty:
            st.warning("No price data for chart.")
        else:
            ema20 = ema(px,20); ema100 = ema(px,100)
            st.subheader("📊 Price & EMAs")
            st.line_chart(pd.DataFrame({"Price":px, "EMA20":ema20, "EMA100":ema100}))

    # ---------- export ----------
    csv_bytes = table.to_csv().encode()
    st.download_button("⬇️ Download CSV", csv_bytes, "ratings.csv", "text/csv")
