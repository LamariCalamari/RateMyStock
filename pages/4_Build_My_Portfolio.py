from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from app_utils import (
    SCORING_CONFIG,
    brand_header,
    fetch_macro_pack,
    fetch_prices_chunked_with_fallback,
    inject_css,
    score_universe_panel,
    technical_scores,
    yf_symbol,
)


BUCKET_CANDIDATES: Dict[str, List[str]] = {
    "US Core Equity": ["VTI", "ITOT", "SCHB", "SPY"],
    "US Quality/Dividend": ["SCHD", "VIG", "DGRO", "QUAL"],
    "US Growth": ["QQQM", "VUG", "SCHG", "IWY"],
    "US Value": ["VTV", "SCHV", "IWD", "IVE"],
    "US Small Cap Value": ["AVUV", "IJR", "VB"],
    "US Min Vol Equity": ["USMV", "SPLV", "VFMV"],
    "Intl Developed": ["VEA", "IEFA", "SCHF"],
    "Emerging Markets": ["VWO", "IEMG", "SCHE"],
    "US Aggregate Bonds": ["BND", "AGG", "SCHZ"],
    "Treasury Intermediate": ["VGIT", "IEF", "SCHR"],
    "Short Treasuries / Cash": ["SGOV", "BIL", "SHV"],
    "TIPS": ["SCHP", "TIP", "VTIP"],
    "REITs": ["VNQ", "SCHH", "USRT"],
    "Gold": ["GLDM", "IAU"],
    "Crypto Sleeve": ["IBIT", "FBTC", "BITB"],
}

BUCKET_RETURN_ASSUMPTION = {
    "US Core Equity": 0.082,
    "US Quality/Dividend": 0.076,
    "US Growth": 0.096,
    "US Value": 0.079,
    "US Small Cap Value": 0.090,
    "US Min Vol Equity": 0.071,
    "Intl Developed": 0.073,
    "Emerging Markets": 0.087,
    "US Aggregate Bonds": 0.045,
    "Treasury Intermediate": 0.040,
    "Short Treasuries / Cash": 0.037,
    "TIPS": 0.041,
    "REITs": 0.070,
    "Gold": 0.042,
    "Crypto Sleeve": 0.135,
}

BUCKET_VOL_ASSUMPTION = {
    "US Core Equity": 0.155,
    "US Quality/Dividend": 0.140,
    "US Growth": 0.220,
    "US Value": 0.165,
    "US Small Cap Value": 0.205,
    "US Min Vol Equity": 0.120,
    "Intl Developed": 0.170,
    "Emerging Markets": 0.230,
    "US Aggregate Bonds": 0.070,
    "Treasury Intermediate": 0.090,
    "Short Treasuries / Cash": 0.015,
    "TIPS": 0.080,
    "REITs": 0.190,
    "Gold": 0.180,
    "Crypto Sleeve": 0.550,
}

STOCK_IDEA_UNIVERSE = [
    "MSFT", "AAPL", "AMZN", "NVDA", "GOOGL", "META", "AVGO", "TSM", "ASML",
    "JPM", "V", "MA", "BRK-B", "UNH", "LLY", "JNJ", "ABBV", "MRK",
    "XOM", "CVX", "COP", "CAT", "GE", "DE", "RTX", "LMT",
    "PG", "COST", "WMT", "KO", "PEP", "MCD", "NKE", "HD", "LOW",
    "CRM", "ORCL", "ADBE", "NOW", "INTU", "QCOM", "AMD", "TXN", "PANW",
]

POSITIVE_NEWS_WORDS = (
    "beat", "upgrade", "surge", "record", "strong", "growth", "approval",
    "partnership", "expands", "buyback", "outperform", "bullish",
)
NEGATIVE_NEWS_WORDS = (
    "miss", "downgrade", "probe", "lawsuit", "decline", "warning", "cut",
    "delay", "fraud", "recall", "bearish", "underperform",
)


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {}
    for key, value in weights.items():
        v = _coerce_float(value)
        if np.isnan(v) or v <= 0:
            continue
        clean[key] = float(v)
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def _risk_label(score: float) -> str:
    if score < 30:
        return "Conservative"
    if score < 55:
        return "Moderate"
    if score < 75:
        return "Growth"
    return "Aggressive"


def _recommendation_mean_label(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    if value <= 1.8:
        return "Strong Buy bias"
    if value <= 2.4:
        return "Buy bias"
    if value <= 3.0:
        return "Hold bias"
    if value <= 3.6:
        return "Underperform bias"
    return "Sell bias"


def _zscore(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(0.0, index=vals.index)
    sd = vals.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=vals.index)
    return (vals - vals.mean()) / sd


def _series_stats(px: pd.Series) -> Dict[str, float]:
    out = {"ret_1y": np.nan, "ret_6m": np.nan, "ret_3m": np.nan, "vol_ann": np.nan, "mdd_1y": np.nan}
    s = pd.to_numeric(px, errors="coerce").dropna()
    if s.size < 30:
        return out

    out["ret_1y"] = float(s.iloc[-1] / s.iloc[0] - 1.0)

    def _ret(window: int) -> float:
        if s.size <= window:
            return np.nan
        base = float(s.iloc[-(window + 1)])
        if base <= 0:
            return np.nan
        return float(s.iloc[-1] / base - 1.0)

    out["ret_6m"] = _ret(126)
    out["ret_3m"] = _ret(63)

    r = s.pct_change().dropna()
    if r.size >= 20:
        out["vol_ann"] = float(r.std(ddof=1) * np.sqrt(252))
    dd = s / s.cummax() - 1.0
    out["mdd_1y"] = float(dd.min()) if not dd.empty else np.nan
    return out


def _headline_sentiment(news_rows: List[Dict[str, object]]) -> str:
    score = 0
    for item in news_rows:
        title = str(item.get("title") or "").lower()
        score += sum(1 for w in POSITIVE_NEWS_WORDS if w in title)
        score -= sum(1 for w in NEGATIVE_NEWS_WORDS if w in title)
    if score >= 2:
        return "Positive"
    if score <= -2:
        return "Negative"
    return "Neutral"


def _risk_score_from_answers(
    risk_tolerance: str,
    expected_return: float,
    horizon: str,
    max_drawdown: int,
    goal: str,
) -> int:
    risk_map = {
        "Low": 20,
        "Medium": 50,
        "High": 72,
        "Very High": 88,
    }
    horizon_map = {
        "< 3 years": 18,
        "3 - 7 years": 44,
        "7 - 15 years": 67,
        "15+ years": 82,
    }
    goal_adj = {
        "Capital preservation": -12,
        "Balanced growth": 0,
        "Maximum growth": 8,
        "Income focus": -6,
    }

    risk_base = risk_map.get(risk_tolerance, 50)
    horizon_score = horizon_map.get(horizon, 50)
    expected_score = np.interp(float(expected_return), [4.0, 18.0], [8.0, 92.0])
    drawdown_score = np.interp(float(max_drawdown), [5.0, 60.0], [8.0, 92.0])
    adj = goal_adj.get(goal, 0)

    score = 0.38 * risk_base + 0.22 * horizon_score + 0.22 * expected_score + 0.18 * drawdown_score + adj
    return int(np.clip(round(score), 5, 95))


def _strategic_allocation(
    risk_score: int,
    horizon: str,
    goal: str,
    region: str,
    style: str,
    include_alts: bool,
    include_crypto: bool,
    inflation_concern: bool,
) -> Dict[str, float]:
    horizon_adj = {
        "< 3 years": -0.20,
        "3 - 7 years": -0.05,
        "7 - 15 years": 0.03,
        "15+ years": 0.07,
    }
    goal_adj = {
        "Capital preservation": -0.15,
        "Balanced growth": 0.0,
        "Maximum growth": 0.08,
        "Income focus": -0.08,
    }
    region_map = {
        "US-heavy": 0.78,
        "Global balanced": 0.60,
        "International tilt": 0.45,
    }

    equity = 0.24 + 0.66 * (risk_score / 100.0)
    equity += horizon_adj.get(horizon, 0.0)
    equity += goal_adj.get(goal, 0.0)
    equity = float(np.clip(equity, 0.18, 0.92))

    alternatives = 0.0
    if include_alts:
        alternatives = 0.04 + (0.02 if inflation_concern else 0.0)
        alternatives = float(np.clip(alternatives, 0.0, 0.10))

    cash = 0.03 + max(0.0, (35.0 - risk_score) / 100.0) * 0.14
    if horizon == "< 3 years":
        cash = max(cash, 0.12)
    if goal == "Capital preservation":
        cash = max(cash, 0.14)
    if goal == "Maximum growth":
        cash = min(cash, 0.03)
    cash = float(np.clip(cash, 0.02, 0.22))

    crypto = 0.0
    if include_crypto and risk_score >= 70 and horizon != "< 3 years":
        crypto = float(np.clip(0.01 + (risk_score - 70) * 0.0012, 0.0, 0.05))

    fixed_income = 1.0 - equity - alternatives - cash - crypto
    if fixed_income < 0.08:
        needed = 0.08 - fixed_income
        pull_cash = min(max(cash - 0.02, 0.0), needed)
        cash -= pull_cash
        fixed_income += pull_cash
        needed -= pull_cash
        if needed > 0:
            pull_alt = min(alternatives, needed)
            alternatives -= pull_alt
            fixed_income += pull_alt
        fixed_income = max(0.08, fixed_income)

    us_share = region_map.get(region, 0.60)
    us_equity = equity * us_share
    intl_equity = max(0.0, equity - us_equity)

    em_frac = 0.10 + 0.18 * (risk_score / 100.0)
    if horizon == "< 3 years":
        em_frac = min(em_frac, 0.08)
    em_frac = float(np.clip(em_frac, 0.06, 0.30))
    em_w = intl_equity * em_frac
    intl_dev_w = intl_equity - em_w

    style_mix = {
        "Blend": {"US Core Equity": 0.68, "US Quality/Dividend": 0.20, "US Small Cap Value": 0.12},
        "Growth tilt": {"US Core Equity": 0.43, "US Growth": 0.42, "US Quality/Dividend": 0.15},
        "Value tilt": {"US Core Equity": 0.44, "US Value": 0.41, "US Quality/Dividend": 0.15},
        "Dividend/Income": {"US Core Equity": 0.34, "US Quality/Dividend": 0.46, "US Value": 0.20},
        "Lower-volatility": {"US Core Equity": 0.35, "US Min Vol Equity": 0.45, "US Quality/Dividend": 0.20},
    }.get(style, {"US Core Equity": 0.65, "US Quality/Dividend": 0.20, "US Small Cap Value": 0.15})

    if risk_score < 35:
        growth_take = style_mix.get("US Growth", 0.0)
        small_take = style_mix.get("US Small Cap Value", 0.0)
        style_mix["USMinVolBuffer"] = growth_take * 0.7 + small_take * 0.6
        if "US Growth" in style_mix:
            style_mix["US Growth"] *= 0.3
        if "US Small Cap Value" in style_mix:
            style_mix["US Small Cap Value"] *= 0.4
        style_mix["US Min Vol Equity"] = style_mix.get("US Min Vol Equity", 0.0) + style_mix.pop("USMinVolBuffer")

    fixed_mix = {}
    if risk_score < 35:
        fixed_mix = {
            "US Aggregate Bonds": 0.34,
            "Treasury Intermediate": 0.24,
            "Short Treasuries / Cash": 0.32,
            "TIPS": 0.10,
        }
    elif risk_score < 60:
        fixed_mix = {
            "US Aggregate Bonds": 0.48,
            "Treasury Intermediate": 0.20,
            "Short Treasuries / Cash": 0.16,
            "TIPS": 0.16,
        }
    else:
        fixed_mix = {
            "US Aggregate Bonds": 0.58,
            "Treasury Intermediate": 0.14,
            "Short Treasuries / Cash": 0.10,
            "TIPS": 0.18,
        }
    if goal == "Income focus":
        fixed_mix["US Aggregate Bonds"] += 0.08
        fixed_mix["Short Treasuries / Cash"] -= 0.05
        fixed_mix["Treasury Intermediate"] -= 0.03
    if horizon == "< 3 years":
        fixed_mix["Short Treasuries / Cash"] += 0.12
        fixed_mix["US Aggregate Bonds"] -= 0.08
        fixed_mix["Treasury Intermediate"] -= 0.04

    fixed_mix = _normalize_weights(fixed_mix)

    weights: Dict[str, float] = {}
    style_mix = _normalize_weights(style_mix)
    for bucket, part in style_mix.items():
        weights[bucket] = weights.get(bucket, 0.0) + us_equity * part

    weights["Intl Developed"] = weights.get("Intl Developed", 0.0) + intl_dev_w
    weights["Emerging Markets"] = weights.get("Emerging Markets", 0.0) + em_w

    for bucket, part in fixed_mix.items():
        weights[bucket] = weights.get(bucket, 0.0) + fixed_income * part
    weights["Short Treasuries / Cash"] = weights.get("Short Treasuries / Cash", 0.0) + cash

    if alternatives > 0:
        gold_share = 0.38 if inflation_concern else 0.28
        weights["Gold"] = weights.get("Gold", 0.0) + alternatives * gold_share
        weights["REITs"] = weights.get("REITs", 0.0) + alternatives * (1.0 - gold_share)
    if crypto > 0:
        weights["Crypto Sleeve"] = weights.get("Crypto Sleeve", 0.0) + crypto

    return _normalize_weights(weights)


def _apply_macro_overlay(weights: Dict[str, float], macro_pack: Dict[str, float], risk_score: int) -> Tuple[Dict[str, float], str]:
    w = dict(weights)
    macro = float(np.clip(macro_pack.get("macro", 0.5), 0.0, 1.0))
    note = "No macro tilt applied."

    if macro < 0.45:
        intensity = (0.45 - macro) / 0.45
        cut = 0.0
        for bucket in ("US Growth", "Emerging Markets", "US Small Cap Value", "Crypto Sleeve"):
            before = w.get(bucket, 0.0)
            if before <= 0:
                continue
            drop = before * (0.16 + 0.16 * intensity)
            w[bucket] = max(0.0, before - drop)
            cut += drop
        if cut > 0:
            w["Short Treasuries / Cash"] = w.get("Short Treasuries / Cash", 0.0) + cut * 0.55
            w["US Aggregate Bonds"] = w.get("US Aggregate Bonds", 0.0) + cut * 0.30
            w["US Quality/Dividend"] = w.get("US Quality/Dividend", 0.0) + cut * 0.15
        note = "Macro is risk-off, so growth/emerging sleeves were trimmed toward cash/bonds."

    elif macro > 0.65 and risk_score >= 55:
        intensity = (macro - 0.65) / 0.35
        max_shift = 0.02 + 0.04 * intensity
        from_short = min(w.get("Short Treasuries / Cash", 0.0), max_shift * 0.65)
        from_bonds = min(w.get("US Aggregate Bonds", 0.0), max_shift - from_short)
        moved = from_short + from_bonds
        if moved > 0:
            w["Short Treasuries / Cash"] = w.get("Short Treasuries / Cash", 0.0) - from_short
            w["US Aggregate Bonds"] = w.get("US Aggregate Bonds", 0.0) - from_bonds
            w["US Growth"] = w.get("US Growth", 0.0) + moved * 0.45
            w["US Core Equity"] = w.get("US Core Equity", 0.0) + moved * 0.35
            w["Emerging Markets"] = w.get("Emerging Markets", 0.0) + moved * 0.20
            note = "Macro is risk-on, so part of defensive weight was shifted to growth sleeves."

    return _normalize_weights(w), note


def _extract_news_fields(item: Dict[str, object]) -> Dict[str, object]:
    content = item.get("content")
    content = content if isinstance(content, dict) else {}

    title = item.get("title") or content.get("title") or ""
    publisher = item.get("publisher")
    if not publisher:
        provider = content.get("provider")
        if isinstance(provider, dict):
            publisher = provider.get("displayName")
    publisher = publisher or "Yahoo Finance"

    link = item.get("link")
    if not link:
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            link = canonical.get("url")
    link = link or ""

    published = pd.NaT
    ts = item.get("providerPublishTime")
    if ts is not None:
        try:
            published = pd.to_datetime(int(ts), unit="s", utc=True)
        except Exception:
            published = pd.NaT
    if pd.isna(published):
        pub_date = content.get("pubDate")
        if pub_date:
            published = pd.to_datetime(pub_date, utc=True, errors="coerce")

    return {
        "title": str(title).strip(),
        "publisher": str(publisher).strip(),
        "link": str(link).strip(),
        "published": published,
    }


@st.cache_data(show_spinner=False, ttl=900)
def _fetch_news(ticker: str, limit: int = 5) -> List[Dict[str, object]]:
    tkr = yf_symbol(ticker)
    try:
        raw = yf.Ticker(tkr).news or []
    except Exception:
        raw = []

    rows: List[Dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clean = _extract_news_fields(item)
        if not clean["title"]:
            continue
        rows.append(clean)
        if len(rows) >= limit:
            break
    return rows


@st.cache_data(show_spinner=False, ttl=900)
def _fetch_quote_estimates(ticker: str) -> Dict[str, object]:
    tkr = yf_symbol(ticker)
    info = {}
    fast = {}
    tk = yf.Ticker(tkr)
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    try:
        fast_obj = tk.fast_info
        fast = dict(fast_obj) if fast_obj is not None else {}
    except Exception:
        fast = {}

    price = _coerce_float(info.get("regularMarketPrice"))
    if np.isnan(price):
        for key in ("lastPrice", "last_price", "regular_market_price"):
            if key in fast:
                price = _coerce_float(fast.get(key))
                if not np.isnan(price):
                    break

    return {
        "ticker": tkr,
        "name": info.get("longName") or info.get("shortName") or tkr,
        "currency": info.get("currency") or "USD",
        "price": price,
        "change_pct": _coerce_float(info.get("regularMarketChangePercent")),
        "expense_ratio": _coerce_float(
            info.get("annualReportExpenseRatio", info.get("expenseRatio", info.get("netExpenseRatio")))
        ),
        "dividend_yield": _coerce_float(info.get("yield", info.get("trailingAnnualDividendYield"))),
        "target_mean_price": _coerce_float(info.get("targetMeanPrice")),
        "target_high_price": _coerce_float(info.get("targetHighPrice")),
        "target_low_price": _coerce_float(info.get("targetLowPrice")),
        "recommendation_mean": _coerce_float(info.get("recommendationMean")),
        "analyst_count": _coerce_float(info.get("numberOfAnalystOpinions")),
        "forward_pe": _coerce_float(info.get("forwardPE")),
        "trailing_pe": _coerce_float(info.get("trailingPE")),
        "revenue_growth": _coerce_float(info.get("revenueGrowth")),
        "earnings_growth": _coerce_float(info.get("earningsGrowth")),
    }


def _score_bucket_candidates(
    candidates: List[str],
    panel: Dict[str, pd.Series],
    tech_raw: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for raw in candidates:
        tkr = yf_symbol(raw)
        px = panel.get(tkr, pd.Series(dtype=float))
        stats = _series_stats(px)
        rows.append({
            "Ticker": tkr,
            "ret_1y": stats["ret_1y"],
            "ret_6m": stats["ret_6m"],
            "ret_3m": stats["ret_3m"],
            "vol_ann": stats["vol_ann"],
            "mdd_1y": stats["mdd_1y"],
            "mom12m": _coerce_float(tech_raw.loc[tkr, "mom12m"]) if tkr in tech_raw.index and "mom12m" in tech_raw.columns else np.nan,
            "mom6m": _coerce_float(tech_raw.loc[tkr, "mom6m"]) if tkr in tech_raw.index and "mom6m" in tech_raw.columns else np.nan,
            "mom3m": _coerce_float(tech_raw.loc[tkr, "mom3m"]) if tkr in tech_raw.index and "mom3m" in tech_raw.columns else np.nan,
            "trend_ema20_50": _coerce_float(tech_raw.loc[tkr, "trend_ema20_50"]) if tkr in tech_raw.index and "trend_ema20_50" in tech_raw.columns else np.nan,
            "drawdown6m": _coerce_float(tech_raw.loc[tkr, "drawdown6m"]) if tkr in tech_raw.index and "drawdown6m" in tech_raw.columns else np.nan,
            "vol63": _coerce_float(tech_raw.loc[tkr, "vol63"]) if tkr in tech_raw.index and "vol63" in tech_raw.columns else np.nan,
        })
    df = pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()
    if df.empty:
        return df

    signal = (
        0.38 * _zscore(df["mom12m"])
        + 0.24 * _zscore(df["mom6m"])
        + 0.10 * _zscore(df["mom3m"])
        + 0.13 * _zscore(df["trend_ema20_50"])
        + 0.10 * _zscore(-df["vol63"])
        + 0.05 * _zscore(df["drawdown6m"])
    )
    df["signal_score"] = signal.fillna(0.0)
    return df.sort_values("signal_score", ascending=False)


def _build_etf_recommendations(
    weights: Dict[str, float],
    invest_amount: float,
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    active_buckets = [b for b, w in weights.items() if w >= 0.01 and b in BUCKET_CANDIDATES]
    if not active_buckets:
        return pd.DataFrame(), None

    candidates = []
    for b in active_buckets:
        candidates.extend(BUCKET_CANDIDATES[b])
    candidates = sorted(set(yf_symbol(x) for x in candidates))

    prices, ok = fetch_prices_chunked_with_fallback(
        candidates,
        period="1y",
        interval="1d",
        chunk=20,
        retries=3,
        sleep_between=0.35,
        singles_pause=0.25,
    )
    panel = {
        t: prices[t].dropna()
        for t in ok
        if t in prices.columns and prices[t].dropna().size > 0
    }
    tech_raw = technical_scores(panel) if panel else pd.DataFrame()

    market_timestamp = None
    if not prices.empty and isinstance(prices.index, pd.DatetimeIndex):
        market_timestamp = prices.index.max()

    rows = []
    for bucket in sorted(active_buckets, key=lambda x: weights.get(x, 0.0), reverse=True):
        bucket_weight = float(weights.get(bucket, 0.0))
        ranked = _score_bucket_candidates(BUCKET_CANDIDATES[bucket], panel, tech_raw)
        choice = ranked.index[0] if not ranked.empty else yf_symbol(BUCKET_CANDIDATES[bucket][0])
        backup = ranked.index[1] if ranked.shape[0] > 1 else (
            yf_symbol(BUCKET_CANDIDATES[bucket][1]) if len(BUCKET_CANDIDATES[bucket]) > 1 else ""
        )
        stats = ranked.loc[choice] if choice in ranked.index else pd.Series(dtype=float)

        quote = _fetch_quote_estimates(choice)
        news_rows = _fetch_news(choice, limit=3)
        news_tone = _headline_sentiment(news_rows)
        latest_news = news_rows[0]["title"] if news_rows else "No recent headline fetched"
        latest_source = news_rows[0]["publisher"] if news_rows else "N/A"
        latest_time = news_rows[0]["published"] if news_rows else pd.NaT
        latest_time_str = (
            pd.to_datetime(latest_time).strftime("%Y-%m-%d %H:%M UTC")
            if pd.notna(latest_time) else "N/A"
        )

        rows.append({
            "Bucket": bucket,
            "Weight %": bucket_weight * 100.0,
            "Amount ($)": invest_amount * bucket_weight if invest_amount > 0 else np.nan,
            "Suggested ETF": choice,
            "Backup ETF": backup,
            "Signal Score": _coerce_float(stats.get("signal_score", np.nan)),
            "1Y Return %": _coerce_float(stats.get("ret_1y", np.nan)) * 100.0,
            "Ann. Vol %": _coerce_float(stats.get("vol_ann", np.nan)) * 100.0,
            "Max Drawdown %": _coerce_float(stats.get("mdd_1y", np.nan)) * 100.0,
            "Price": quote.get("price", np.nan),
            "Expense Ratio %": _coerce_float(quote.get("expense_ratio", np.nan)) * 100.0,
            "Yield %": _coerce_float(quote.get("dividend_yield", np.nan)) * 100.0,
            "News Tone": news_tone,
            "Latest Headline": latest_news,
            "Headline Source": latest_source,
            "Headline Time": latest_time_str,
        })
    out = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
    return out, market_timestamp


def _build_stock_ideas(
    risk_score: int,
    style: str,
    macro_pack: Dict[str, float],
    n_ideas: int,
) -> pd.DataFrame:
    if n_ideas <= 0:
        return pd.DataFrame()

    universe = [yf_symbol(x) for x in STOCK_IDEA_UNIVERSE]

    if style == "Growth tilt":
        universe = [t for t in universe if t not in ("KO", "PG", "JNJ", "PEP")] + ["CRWD", "PLTR"]
    elif style == "Dividend/Income":
        universe.extend(["O", "NEE", "DUK"])
    elif style == "Value tilt":
        universe.extend(["BAC", "C", "PFE"])
    universe = list(dict.fromkeys(universe))

    prices, ok = fetch_prices_chunked_with_fallback(
        universe,
        period="1y",
        interval="1d",
        chunk=20,
        retries=3,
        sleep_between=0.35,
        singles_pause=0.25,
    )
    panel = {
        t: prices[t].dropna()
        for t in ok
        if t in prices.columns and prices[t].dropna().size > 0
    }
    if not panel:
        return pd.DataFrame()

    score_pack = score_universe_panel(
        panel,
        target_count=max(len(universe), 1),
        macro_pack=macro_pack,
        min_fund_cols=SCORING_CONFIG["min_fund_cols"],
    )
    out = score_pack.get("out", pd.DataFrame())
    if out.empty or "RATING_0_100" not in out.columns:
        return pd.DataFrame()

    if risk_score < 40:
        out = out.drop(index=["NVDA", "AMD", "PLTR", "CRWD"], errors="ignore")

    ranked = out.sort_values("RATING_0_100", ascending=False).head(max(n_ideas * 2, n_ideas))
    rows = []
    for ticker, row in ranked.iterrows():
        if len(rows) >= n_ideas:
            break

        quote = _fetch_quote_estimates(ticker)
        news_rows = _fetch_news(ticker, limit=2)
        latest_news = news_rows[0]["title"] if news_rows else "No recent headline fetched"
        news_tone = _headline_sentiment(news_rows)
        price = _coerce_float(quote.get("price"))
        target = _coerce_float(quote.get("target_mean_price"))
        upside = (target / price - 1.0) if pd.notna(price) and price > 0 and pd.notna(target) else np.nan
        rec_mean = _coerce_float(quote.get("recommendation_mean"))
        analysts = _coerce_float(quote.get("analyst_count"))

        # Filter very weak estimate coverage when enough alternatives are available.
        if len(rows) < n_ideas and pd.notna(analysts) and analysts > 0 and analysts < 4 and ranked.shape[0] > n_ideas:
            continue

        rows.append({
            "Ticker": ticker,
            "Score": _coerce_float(row.get("RATING_0_100", np.nan)),
            "Reco": str(row.get("RECO", "N/A")),
            "Confidence": _coerce_float(row.get("CONFIDENCE", np.nan)),
            "Price": price,
            "Target Mean": target,
            "Upside vs Target %": upside * 100.0 if pd.notna(upside) else np.nan,
            "Analyst Rec Mean": rec_mean,
            "Analyst Bias": _recommendation_mean_label(rec_mean),
            "Analyst Count": analysts,
            "Forward P/E": _coerce_float(quote.get("forward_pe")),
            "Revenue Growth %": _coerce_float(quote.get("revenue_growth")) * 100.0,
            "Earnings Growth %": _coerce_float(quote.get("earnings_growth")) * 100.0,
            "News Tone": news_tone,
            "Latest Headline": latest_news,
        })

    return pd.DataFrame(rows).sort_values("Score", ascending=False) if rows else pd.DataFrame()


st.set_page_config(page_title="Build My Portfolio", layout="wide")
inject_css()
brand_header("Build My Portfolio")
st.caption(
    "Answer a few questions and get a live starter portfolio with concrete ETFs, optional stock ideas, "
    "latest headlines, and current analyst estimates."
)
st.page_link("pages/2_Rate_My_Portfolio.py", label="Already have holdings? Rate My Portfolio")

with st.expander("Data controls", expanded=False):
    if st.button("Refresh live data now", use_container_width=False):
        st.cache_data.clear()
        st.rerun()
    st.caption("Refresh clears Streamlit cache so prices, news, and estimates are pulled again.")

st.markdown("### Investor Questionnaire")
c1, c2, c3 = st.columns(3)
with c1:
    goal = st.selectbox(
        "Primary goal",
        ["Balanced growth", "Maximum growth", "Income focus", "Capital preservation"],
        index=0,
    )
with c2:
    horizon = st.selectbox(
        "Investment horizon",
        ["< 3 years", "3 - 7 years", "7 - 15 years", "15+ years"],
        index=2,
    )
with c3:
    risk_tolerance = st.selectbox(
        "Risk tolerance",
        ["Low", "Medium", "High", "Very High"],
        index=1,
    )

c4, c5, c6 = st.columns(3)
with c4:
    expected_return = st.slider("Expected long-run annual return (%)", 4.0, 18.0, 8.0, 0.5)
with c5:
    max_drawdown = st.slider("Temporary drawdown you can tolerate (%)", 5, 60, 25, 5)
with c6:
    invest_amount = st.number_input("Amount to invest now ($)", min_value=0.0, value=10000.0, step=500.0)

c7, c8, c9 = st.columns(3)
with c7:
    region = st.selectbox("Regional preference", ["US-heavy", "Global balanced", "International tilt"], index=1)
with c8:
    style = st.selectbox(
        "Equity style preference",
        ["Blend", "Growth tilt", "Value tilt", "Dividend/Income", "Lower-volatility"],
        index=0,
    )
with c9:
    n_ideas = st.slider("Optional stock ideas", 0, 8, 4, 1)

c10, c11, c12 = st.columns(3)
with c10:
    include_alts = st.checkbox("Include REITs/Gold diversifiers", value=True)
with c11:
    inflation_concern = st.checkbox("Inflation protection focus", value=True)
with c12:
    include_crypto = st.checkbox("Allow small crypto sleeve", value=False)

run_sig = (
    goal, horizon, risk_tolerance, expected_return, max_drawdown,
    invest_amount, region, style, n_ideas, include_alts, inflation_concern, include_crypto,
)
if st.session_state.get("builder_sig") != run_sig:
    st.session_state["builder_ready"] = False

start_col = st.columns([1, 2, 1])[1]
with start_col:
    start = st.button("Build my portfolio plan", type="primary", use_container_width=True)
if start:
    st.session_state["builder_ready"] = True
    st.session_state["builder_sig"] = run_sig

if not st.session_state.get("builder_ready"):
    st.info("Tune your answers, then click **Build my portfolio plan**.")
    st.stop()

run_ts = datetime.now(timezone.utc)

with st.status("Designing your portfolio...", expanded=True) as status:
    prog = st.progress(0)
    msg = st.empty()

    msg.info("Step 1/5: mapping your answers to a risk profile.")
    risk_score = _risk_score_from_answers(
        risk_tolerance=risk_tolerance,
        expected_return=expected_return,
        horizon=horizon,
        max_drawdown=max_drawdown,
        goal=goal,
    )
    base_weights = _strategic_allocation(
        risk_score=risk_score,
        horizon=horizon,
        goal=goal,
        region=region,
        style=style,
        include_alts=include_alts,
        include_crypto=include_crypto,
        inflation_concern=inflation_concern,
    )
    prog.progress(20)

    msg.info("Step 2/5: pulling the latest macro regime signals.")
    macro_pack = fetch_macro_pack(period="6mo", interval="1d")
    tilted_weights, macro_note = _apply_macro_overlay(base_weights, macro_pack, risk_score)
    prog.progress(40)

    msg.info("Step 3/5: selecting concrete ETFs within each allocation bucket.")
    etf_df, market_ts = _build_etf_recommendations(tilted_weights, invest_amount)
    prog.progress(70)

    msg.info("Step 4/5: generating optional stock ideas with live estimates.")
    stock_df = _build_stock_ideas(
        risk_score=risk_score,
        style=style,
        macro_pack=macro_pack,
        n_ideas=n_ideas,
    )
    prog.progress(90)

    msg.info("Step 5/5: finalizing return/risk estimates and outputs.")
    est_return = sum(tilted_weights.get(k, 0.0) * BUCKET_RETURN_ASSUMPTION.get(k, 0.06) for k in tilted_weights)
    est_vol = sum(tilted_weights.get(k, 0.0) * BUCKET_VOL_ASSUMPTION.get(k, 0.12) for k in tilted_weights)
    downside_case = est_return - 1.65 * est_vol
    prog.progress(100)
    status.update(label="Portfolio plan ready.", state="complete")

macro_value = float(np.clip(macro_pack.get("macro", 0.5), 0.0, 1.0))
if macro_value >= 0.66:
    macro_label = "Risk-on"
elif macro_value <= 0.40:
    macro_label = "Risk-off"
else:
    macro_label = "Neutral"

st.markdown(
    f'<div class="banner">'
    f'Plan generated: <b>{run_ts.strftime("%Y-%m-%d %H:%M UTC")}</b>'
    f' &nbsp;|&nbsp; Market regime: <b>{macro_label}</b> ({macro_value * 100:.0f}/100)'
    f'</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Risk profile", f"{_risk_label(risk_score)} ({risk_score}/100)")
m2.metric("Estimated annual return", f"{est_return * 100:.1f}%")
m3.metric("Estimated annual volatility", f"{est_vol * 100:.1f}%")
m4.metric("Stress-case year (rough)", f"{downside_case * 100:.1f}%")

if market_ts is not None and pd.notna(market_ts):
    st.caption(f"Latest market close used in ETF ranking: {pd.to_datetime(market_ts).strftime('%Y-%m-%d')}")
st.caption(macro_note)

alloc_df = (
    pd.DataFrame({"Bucket": list(tilted_weights.keys()), "Weight %": [v * 100.0 for v in tilted_weights.values()]})
    .sort_values("Weight %", ascending=False)
    .reset_index(drop=True)
)
if invest_amount > 0:
    alloc_df["Amount ($)"] = alloc_df["Weight %"] / 100.0 * invest_amount

st.markdown("### Strategic Allocation")
st.dataframe(alloc_df.round(2), use_container_width=True, hide_index=True)
st.bar_chart(alloc_df.set_index("Bucket")["Weight %"], use_container_width=True)

st.markdown("### Core ETF Recommendations (Live)")
if etf_df.empty:
    st.warning("Could not fetch live ETF data right now. The strategic allocation above is still valid.")
else:
    show_cols = [
        "Bucket", "Weight %", "Amount ($)", "Suggested ETF", "Backup ETF",
        "Signal Score", "1Y Return %", "Ann. Vol %", "Max Drawdown %",
        "Price", "Expense Ratio %", "Yield %", "News Tone", "Latest Headline",
        "Headline Source", "Headline Time",
    ]
    show_cols = [c for c in show_cols if c in etf_df.columns]
    st.dataframe(etf_df[show_cols].round(2), use_container_width=True, hide_index=True)
    st.caption("Signal Score blends momentum + trend + volatility inside each ETF bucket.")

st.markdown("### Optional Stock Ideas (Live Estimates + News)")
if n_ideas <= 0:
    st.info("Stock ideas are turned off. Increase 'Optional stock ideas' if you want a satellite list.")
elif stock_df.empty:
    st.warning("No live stock ideas available from data providers right now.")
else:
    stock_show_cols = [
        "Ticker", "Score", "Reco", "Confidence", "Price", "Target Mean",
        "Upside vs Target %", "Analyst Rec Mean", "Analyst Bias", "Analyst Count",
        "Forward P/E", "Revenue Growth %", "Earnings Growth %", "News Tone", "Latest Headline",
    ]
    stock_show_cols = [c for c in stock_show_cols if c in stock_df.columns]
    st.dataframe(stock_df[stock_show_cols].round(2), use_container_width=True, hide_index=True)
    st.caption("Analyst fields come from current Yahoo Finance estimates and can be missing for some names.")

st.markdown("### What This Engine Is Doing")
st.markdown(
    "1. Converts your questionnaire answers into a risk score and strategic bucket mix.\n"
    "2. Applies a macro overlay using live multi-signal regime data.\n"
    "3. Picks ETF candidates within each bucket using recent momentum/volatility behavior.\n"
    "4. Enriches picks with latest headlines and estimate fields to keep outputs timely.\n"
    "5. Produces a beginner-friendly starter plan you can refine in `Rate My Portfolio`."
)

export_name = f"portfolio_builder_plan_{run_ts.strftime('%Y%m%d_%H%M%S')}.csv"
if not etf_df.empty:
    export_df = etf_df.copy()
else:
    export_df = alloc_df.copy()
st.download_button(
    "Download plan as CSV",
    data=export_df.to_csv(index=False),
    file_name=export_name,
    mime="text/csv",
)

st.warning("Educational tool only, not personalized financial advice.")
