import streamlit as st
from app_utils import inject_css, brand_header

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
    "Imagine a stock with strong revenue growth, above‑peer ROE, and positive momentum, but a high "
    "valuation. The growth and profitability z‑scores push the fundamentals score up, valuation pulls it down. "
    "Technicals add a positive lift if momentum is strong, and macro can tilt the score if risk conditions are favorable. "
    "The final rating is the stock’s percentile rank within its peer group."
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
    "- 12‑month momentum"
)
st.markdown(
    "**Macro (multi‑signal)**  \n"
    "- VIX level & trend  \n"
    "- Gold 3‑month return  \n"
    "- USD trend (UUP)  \n"
    "- 10Y yield change  \n"
    "- Credit ratio (HYG/LQD)"
)

st.markdown("## 3) Standardization")
st.markdown(
    "Each numeric factor is **z‑scored** against the peer group and clipped to reduce outliers:  \n"
    "`z = (x − μ) / σ`  \n"
    "For valuation and leverage where *lower is better*, we invert before z‑scoring."
)

st.markdown("## 4) Weighting (Defaults)")
st.markdown(
    "- **Fundamentals:** 50%  \n"
    "- **Technicals:** 45%  \n"
    "- **Macro:** 5%  \n"
    "Weights can be adjusted in Advanced Settings."
)

st.markdown("## 4) Composite Score")
st.markdown(
    "We compute a weighted average of available factor groups:  \n"
    "`Composite = wF·Fund + wT·Tech + wM·Macro`  \n"
    "If a factor group is missing for a ticker, weights are re‑normalized over the available groups."
)

st.markdown("## 5) Rating & Recommendation")
st.markdown(
    "We rank the composite score among peers into a percentile score:  \n"
    "`Rating = percentile_rank(Composite) × 100`  \n"
    "**Labels:**  \n"
    "- 80+ = Strong Buy  \n"
    "- 60–79 = Buy  \n"
    "- 40–59 = Hold  \n"
    "- 20–39 = Sell  \n"
    "- <20 = Strong Sell"
)

st.markdown("## 6) Confidence")
st.markdown(
    "Confidence reflects data coverage and peer sample size:  \n"
    "- Peer coverage (how many peers actually loaded)  \n"
    "- Fundamentals coverage (% of factor z‑scores available)  \n"
    "- Technicals coverage (% of factor z‑scores available)"
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
