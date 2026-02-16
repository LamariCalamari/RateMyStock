import streamlit as st
import pandas as pd
from app_utils import inject_css, brand_header, SCORING_CONFIG

st.set_page_config(page_title="Methodology", layout="wide")
inject_css()
brand_header("Methodology")

st.markdown(
    "<div style='text-align:center;color:#9aa0a6;font-size:1.05rem;margin:.1rem 0 1.2rem;'>"
    "How the ratings are built: inputs, formulas, and interpretation."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("## 🧠 Overview")
st.markdown(
    "Each stock is scored relative to a peer group (industry → sector → index fallback). "
    "We compute standardized factors, combine them into a composite score, then map that score "
    "to a 0–100 rating and a recommendation label."
)

st.markdown("## 🔎 Worked Example (Simplified)")
st.markdown(
    "Below is a simplified, illustrative example to show how a score is built and interpreted."
)
st.markdown("**Example ticker:** Acme Corp (fictional)")
st.markdown(
    "- Peer set: Industry peers (n=120)  \n"
    "- Weights: Fundamentals 50%, Technicals 45%, Macro 5%"
)
st.table(
    {
        "Component": ["Fundamentals", "Technicals", "Macro", "Composite"],
        "Score": [0.55, 0.30, 0.40, 0.43],
        "Interpretation": [
            "Strong growth + margins; valuation slightly expensive",
            "Momentum positive but cooling (MACD flattening)",
            "Neutral risk backdrop",
            "Above‑average vs peers",
        ],
    }
)
st.markdown(
    "**Interpretation:** The stock scores above peer average because fundamentals are strong. "
    "Technicals are positive but not exceptional, so the composite is good but not top‑tier. "
    "This would typically map to a **Buy** (e.g., 60–79 percentile) depending on peers."
)

st.markdown("## 🧮 Detailed Walkthrough (Numbers)")
st.markdown("**Step A — Raw inputs (sample values):**")
st.table(
    {
        "Metric": ["Revenue growth", "ROE", "FCF yield", "Forward PE", "EMA gap", "MACD hist", "RSI", "12m momentum", "VIX", "Gold 3m"],
        "Value": ["18%", "22%", "4.5%", "28×", "6%", "0.12", "62", "14%", "16", "-2%"],
    }
)

st.markdown("**Step B — Convert to peer‑relative z‑scores:**")
st.table(
    {
        "Metric": ["Revenue growth", "ROE", "FCF yield", "Forward PE (inverted)", "EMA gap", "MACD hist", "RSI strength", "12m momentum"],
        "Z‑score": [0.9, 0.7, 0.4, -0.6, 0.5, 0.2, 0.3, 0.6],
        "Notes": [
            "Above peer average",
            "Above peer average",
            "Slightly above",
            "Valuation richer than peers",
            "Price above EMA50",
            "Momentum modestly positive",
            "Trend strength moderate",
            "Solid 12‑month trend",
        ],
    }
)

st.markdown("**Step C — Aggregate factor groups:**")
st.table(
    {
        "Group": ["Fundamentals", "Technicals", "Macro"],
        "Formula": [
            "avg(z of growth, profitability, valuation, balance)",
            "avg(z of EMA gap, MACD hist, RSI strength, 12m mom)",
            "multi‑signal risk score",
        ],
        "Score": [0.55, 0.30, 0.40],
    }
)
st.markdown(
    "**Interpretation:**  \n"
    "- Fundamentals are strong because growth and profitability z‑scores are positive, even though valuation is a headwind.  \n"
    "- Technicals are mildly positive (uptrend, but momentum not extreme).  \n"
    "- Macro is neutral‑positive (risk backdrop not restrictive)."
)

st.markdown("**Step D — Composite score:**")
st.code("Composite = 0.50×0.55 + 0.45×0.30 + 0.05×0.40 = 0.43")

st.markdown("**Step E — Percentile ranking to 0–100:**")
st.markdown(
    "If the composite ranks at the 68th percentile among peers, the rating = **68/100**, "
    "which maps to **Buy**."
)

st.markdown("**Step F — Confidence (example):**")
st.markdown(
    "Confidence blends peer coverage, fundamentals coverage, and technicals coverage.  \n"
    "Example values:  \n"
    "- Peer coverage = 0.85 (102 of 120 peers loaded)  \n"
    "- Fundamentals coverage = 0.75  \n"
    "- Technicals coverage = 0.90  \n"
    "Confidence = 100 × (0.4×0.85 + 0.3×0.75 + 0.3×0.90) = **83/100**"
)

st.markdown("## 1) Peer Universe")
st.markdown(
    "- **Primary:** same industry peers.  \n"
    "- **Fallback 1:** same sector.  \n"
    "- **Fallback 2:** broad index peers.  \n"
    "Peers are validated using available price data; missing symbols are excluded."
)

st.markdown("## 2) Factor Construction")
st.markdown(
    "**Fundamentals (examples)**  \n"
    "- Growth: revenue growth, earnings growth  \n"
    "- Profitability: ROE, ROA, margins  \n"
    "- Valuation: PE, forward PE, EV/EBITDA  \n"
    "- Balance sheet & cash: Debt/Equity, FCF yield"
)
st.markdown(
    "**Technicals (examples)**  \n"
    "- EMA gap (price vs EMA50)  \n"
    "- MACD histogram  \n"
    "- RSI strength  \n"
    "- 3m / 6m / 12m momentum  \n"
    "- Trend structure (EMA20 vs EMA50)  \n"
    "- Volatility + drawdown resilience"
)
st.markdown(
    "**Macro (multi‑signal)**  \n"
    "- VIX level & trend  \n"
    "- Gold 3‑month return  \n"
    "- USD trend (UUP)  \n"
    "- 10Y yield change  \n"
    "- Credit ratio (HYG/LQD)"
)

st.markdown("## 2b) Technical Indicator Formulas (Simplified)")
st.markdown(
    "- **EMA:** `EMA_t = α·Price_t + (1−α)·EMA_(t−1)`  \n"
    "- **MACD:** `MACD = EMA(12) − EMA(26)`; **Signal** = EMA(9) of MACD; **Histogram** = MACD − Signal  \n"
    "- **RSI:** based on average gains/losses over 14 periods (0–100)  \n"
    "- **Momentum (12m):** `Price_t / Price_(t−12m) − 1`"
)

st.markdown("## 3) Standardization")
st.markdown(
    "Each numeric factor is standardized against the peer group and clipped to reduce outliers.  \n"
    "By default we use a **robust z‑score** (median/MAD) with classical z-score fallback when needed:  \n"
    "`z_robust ≈ 0.6745 × (x − median) / MAD`  \n"
    "For valuation and leverage where *lower is better*, we invert before z‑scoring."
)

st.markdown("## 4) Weighting (Defaults)")
st.markdown(
    f"- **Fundamentals:** {SCORING_CONFIG['weights']['fund']*100:.0f}%  \n"
    f"- **Technicals:** {SCORING_CONFIG['weights']['tech']*100:.0f}%  \n"
    f"- **Macro:** {SCORING_CONFIG['weights']['macro']*100:.0f}%  \n"
    "Weights can be adjusted in Advanced Settings."
)

st.markdown("## 5) Composite Score")
st.markdown(
    "We compute a weighted average of available factor groups:  \n"
    "`Composite = wF·Fund + wT·Tech + wM·Macro`  \n"
    "If a factor group is missing for a ticker, weights are re‑normalized over the available groups."
)

st.markdown("## 6) Portfolio Score (Weighted Holdings)")
st.markdown(
    "For a portfolio, each holding’s composite score is weighted by its portfolio allocation:  \n"
    "`PortfolioSignal = Σ(Weight_i × Composite_i)`  \n"
    "Missing-factor holdings are re‑normalized (not treated as zero).  \n"
    "The signal is converted into a **peer percentile score (0–100)**, then adjusted by a small diversification bonus."
)
st.markdown(
    "**Diversification (simplified):**  \n"
    "- **Sector diversity:** effective number of sectors based on weights  \n"
    "- **Name concentration:** penalizes very large single positions  \n"
    "- **Correlation:** penalizes highly correlated holdings  \n"
    "Final diversification score = `0.5·Sector + 0.3·Correlation + 0.2·Concentration`"
)
st.markdown(
    "**Risk contribution:** Each holding’s share of portfolio volatility, derived from the covariance matrix. "
    "This identifies which positions drive risk the most."
)
st.markdown(
    "**Risk metrics:**  \n"
    "- **Volatility (ann.)** = std(daily returns) × √252  \n"
    "- **Max drawdown** = worst peak‑to‑trough decline  \n"
    "- **Sharpe** = mean return / volatility  \n"
    "- **Sortino** = mean return / downside volatility  \n"
    "- **VaR/CVaR** = tail loss estimates (worst 5% of days)"
)
st.markdown(
    "**Benchmarking (SPY):**  \n"
    "- **Beta** = covariance(portfolio, benchmark) / variance(benchmark)  \n"
    "- **Alpha** = excess return after adjusting for beta  \n"
    "- **Tracking error** = std(portfolio − benchmark)  \n"
    "- **Information ratio** = active return / tracking error"
)

st.markdown("## 7) Rating & Recommendation")
st.markdown(
    "We rank the composite score among peers into a percentile score:  \n"
    "`Rating = percentile_rank(Composite) × 100`  \n"
    "**Labels:**  \n"
    + "\n".join([f"- {int(cutoff)}+ = {label}" for cutoff, label in SCORING_CONFIG["rating_bins"]])
)

st.markdown("## 8) Confidence")
st.markdown(
    "Confidence reflects data coverage and peer sample size:  \n"
    "- Peer coverage (how many peers actually loaded)  \n"
    "- Fundamentals coverage (% of factor z‑scores available)  \n"
    "- Technicals coverage (% of factor z‑scores available)  \n"
    "- Macro signal coverage (how many macro inputs were available)"
)
st.markdown(
    "**What coverage means:**  \n"
    "- **Fundamentals coverage** = how many fundamental factors were available for the ticker (e.g., ROE, margins, FCF).  \n"
    "- **Technicals coverage** = how many technical factors were computed from price data (EMA gap, MACD, RSI, momentum).  \n"
    "Missing data reduces coverage and lowers confidence."
)

st.markdown("## 📚 Factor Glossary (Short)")
st.markdown(
    "- **FCF yield:** Free cash flow / market cap (higher is better).  \n"
    "- **EV/EBITDA:** Valuation multiple (lower is better).  \n"
    "- **DMA gap:** Price relative to EMA50 (trend).  \n"
    "- **MACD histogram:** Momentum acceleration.  \n"
    "- **RSI strength:** Trend strength (0–100)."
)

st.markdown("## ⚠️ Notes & Limitations")
st.markdown(
    "- Scores are **relative** to the chosen peers, not absolute predictions.  \n"
    "- Data comes from public sources (yfinance) and can be delayed or incomplete.  \n"
    "- This tool is educational and not investment advice."
)

st.markdown("## 🎯 How to Read Scores (Quick Legend)")
legend_rows = [
    ("80–100", "Strong Buy", "#2ecc71", "Top of peers; strongest signals across factors"),
    ("60–79", "Buy", "#8bc34a", "Above‑average vs peers"),
    ("40–59", "Hold", "#f1c40f", "Around peer average"),
    ("20–39", "Sell", "#f39c12", "Below‑average vs peers"),
    ("0–19", "Strong Sell", "#e74c3c", "Bottom of peers; weakest signals"),
]
st.markdown("<div style='display:grid;gap:8px;'>", unsafe_allow_html=True)
for rng, label, color, meaning in legend_rows:
    st.markdown(
        f"<div style='display:grid;grid-template-columns:80px 140px 1fr;align-items:center;"
        f"background:#151920;border:1px solid #2c3239;border-radius:10px;padding:10px 12px;'>"
        f"<div style='font-weight:700;'>{rng}</div>"
        f"<div style='font-weight:700;color:{color};'>{label}</div>"
        f"<div style='color:#cfd4da;'>{meaning}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)
