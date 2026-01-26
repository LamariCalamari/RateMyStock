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

st.markdown("## 4b) Portfolio Score (Weighted Holdings)")
st.markdown(
    "For a portfolio, each holding’s composite score is weighted by its portfolio allocation:  \n"
    "`PortfolioSignal = Σ(Weight_i × Composite_i)`  \n"
    "We then blend a small diversification bonus to reward lower concentration and lower correlations."
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
legend = pd.DataFrame(
    {
        "Score Range": ["80–100", "60–79", "40–59", "20–39", "0–19"],
        "Label": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
        "Color": ["#2ecc71", "#8bc34a", "#f1c40f", "#f39c12", "#e74c3c"],
        "Meaning": [
            "Top of peers; strongest signals across factors",
            "Above‑average vs peers",
            "Around peer average",
            "Below‑average vs peers",
            "Bottom of peers; weakest signals",
        ],
    }
)
st.dataframe(legend, use_container_width=True)
