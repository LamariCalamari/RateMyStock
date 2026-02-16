from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app_utils import inject_css, brand_header


st.set_page_config(page_title="Build My Portfolio", layout="wide")
inject_css()
brand_header("Build My Portfolio")
st.caption(
    "Beginner-friendly portfolio builder. Answer a few questions and get a simple, diversified ETF plan."
)


ASSET_META: Dict[str, Dict[str, float | str]] = {
    "VTI": {"name": "US Total Stock Market", "bucket": "Equity", "exp_return": 0.082, "exp_vol": 0.17},
    "IJR": {"name": "US Small Cap", "bucket": "Equity", "exp_return": 0.088, "exp_vol": 0.22},
    "VEA": {"name": "International Developed", "bucket": "Equity", "exp_return": 0.074, "exp_vol": 0.18},
    "VWO": {"name": "Emerging Markets", "bucket": "Equity", "exp_return": 0.084, "exp_vol": 0.24},
    "VNQ": {"name": "US REITs", "bucket": "Equity", "exp_return": 0.075, "exp_vol": 0.21},
    "BND": {"name": "US Aggregate Bonds", "bucket": "Bonds", "exp_return": 0.045, "exp_vol": 0.06},
    "VGIT": {"name": "Intermediate Treasuries", "bucket": "Bonds", "exp_return": 0.040, "exp_vol": 0.05},
    "TIP": {"name": "TIPS", "bucket": "Bonds", "exp_return": 0.042, "exp_vol": 0.06},
    "SGOV": {"name": "Cash / T-Bills", "bucket": "Cash", "exp_return": 0.035, "exp_vol": 0.01},
    "IAU": {"name": "Gold", "bucket": "Diversifier", "exp_return": 0.045, "exp_vol": 0.15},
}

RISK_CHOICES = ["Low", "Medium", "High"]
HORIZON_CHOICES = ["<3 years", "3-5 years", "5-10 years", "10+ years"]
RETURN_CHOICES = ["4%-6%", "6%-8%", "8%-10%", "10%+"]
DRAWDOWN_CHOICES = ["Up to 10%", "Up to 20%", "Up to 30%", "30%+"]
INCOME_CHOICES = ["No, growth first", "Some income", "Yes, steady income"]
INVOLVEMENT_CHOICES = ["Set and forget", "Review quarterly", "Actively rebalance"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: max(0.0, v) * 100.0 / total for k, v in weights.items()}


def _equity_mix(risk: str, horizon: str, income_need: str) -> Dict[str, float]:
    mix = {
        "Low": {"VTI": 0.62, "IJR": 0.04, "VEA": 0.22, "VWO": 0.04, "VNQ": 0.08},
        "Medium": {"VTI": 0.53, "IJR": 0.08, "VEA": 0.20, "VWO": 0.09, "VNQ": 0.10},
        "High": {"VTI": 0.43, "IJR": 0.14, "VEA": 0.18, "VWO": 0.15, "VNQ": 0.10},
    }[risk].copy()

    if horizon in ("<3 years", "3-5 years"):
        reduce_factor = 0.50 if horizon == "<3 years" else 0.25
        cut_small = mix["IJR"] * reduce_factor
        cut_em = mix["VWO"] * reduce_factor
        mix["IJR"] -= cut_small
        mix["VWO"] -= cut_em
        mix["VTI"] += 0.60 * (cut_small + cut_em)
        mix["VEA"] += 0.40 * (cut_small + cut_em)

    inc_shift = {"No, growth first": 0.0, "Some income": 0.03, "Yes, steady income": 0.06}[income_need]
    if inc_shift > 0:
        reducible = mix["IJR"] + mix["VWO"]
        shift = min(inc_shift, reducible)
        if reducible > 0:
            mix["IJR"] -= shift * (mix["IJR"] / reducible)
            mix["VWO"] -= shift * (mix["VWO"] / reducible)
        mix["VNQ"] += shift

    return _normalize(mix)


def _bond_mix(risk: str, horizon: str, income_need: str) -> Dict[str, float]:
    mix = {"BND": 0.50, "VGIT": 0.30, "TIP": 0.20}
    if risk == "Low":
        mix["VGIT"] += 0.08
        mix["BND"] -= 0.08
    elif risk == "High":
        mix["BND"] += 0.08
        mix["VGIT"] -= 0.08

    if income_need == "Some income":
        mix["BND"] += 0.07
        mix["VGIT"] -= 0.05
        mix["TIP"] -= 0.02
    elif income_need == "Yes, steady income":
        mix["BND"] += 0.14
        mix["VGIT"] -= 0.08
        mix["TIP"] -= 0.06

    if horizon == "<3 years":
        mix["VGIT"] += 0.10
        mix["TIP"] -= 0.10

    return _normalize(mix)


def _build_weights(
    risk: str,
    horizon: str,
    target_return: str,
    drawdown: str,
    income_need: str,
    involvement: str,
) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    risk_base = {"Low": 35.0, "Medium": 60.0, "High": 80.0}
    horizon_adj = {"<3 years": -25.0, "3-5 years": -10.0, "5-10 years": 0.0, "10+ years": 5.0}
    drawdown_adj = {"Up to 10%": -20.0, "Up to 20%": -10.0, "Up to 30%": 0.0, "30%+": 5.0}
    income_adj = {"No, growth first": 0.0, "Some income": -5.0, "Yes, steady income": -10.0}
    target_map = {"4%-6%": 0.050, "6%-8%": 0.070, "8%-10%": 0.090, "10%+": 0.105}

    equity = (
        risk_base[risk] + horizon_adj[horizon] + drawdown_adj[drawdown] + income_adj[income_need]
    )
    implied_equity = (target_map[target_return] - 0.042) / (0.083 - 0.042) * 100.0
    equity = 0.65 * equity + 0.35 * implied_equity
    if involvement == "Set and forget":
        equity -= 2.0
    elif involvement == "Actively rebalance":
        equity += 2.0
    equity = _clamp(equity, 15.0, 92.0)

    gold = {"Low": 2.0, "Medium": 3.5, "High": 5.0}[risk]
    if horizon == "<3 years":
        gold = max(1.5, gold - 1.0)

    safe_total = max(5.0, 100.0 - equity - gold)
    bond_share = {"<3 years": 0.55, "3-5 years": 0.70, "5-10 years": 0.82, "10+ years": 0.90}[horizon]
    bond_total = safe_total * bond_share
    cash_total = safe_total - bond_total

    eq_mix = _equity_mix(risk, horizon, income_need)
    fi_mix = _bond_mix(risk, horizon, income_need)

    weights: Dict[str, float] = {}
    for t, w in eq_mix.items():
        weights[t] = equity * w / 100.0
    for t, w in fi_mix.items():
        weights[t] = bond_total * w / 100.0
    weights["SGOV"] = cash_total
    weights["IAU"] = gold

    if involvement == "Set and forget":
        weights["VTI"] += weights.pop("IJR", 0.0) + weights.pop("VWO", 0.0)
        if weights.get("TIP", 0.0) < 4.0:
            weights["BND"] += weights.pop("TIP", 0.0)
        if weights.get("IAU", 0.0) < 2.0:
            weights["BND"] += weights.pop("IAU", 0.0)

    weights = _normalize(weights)
    tiny = [k for k, v in weights.items() if v < 1.0]
    for t in tiny:
        value = weights.pop(t, 0.0)
        if weights:
            kmax = max(weights, key=weights.get)
            weights[kmax] += value
    weights = _normalize(weights)

    diagnostics = {
        "equity_target": sum(weights.get(t, 0.0) for t, m in ASSET_META.items() if m["bucket"] == "Equity"),
        "bond_target": sum(weights.get(t, 0.0) for t, m in ASSET_META.items() if m["bucket"] == "Bonds"),
        "cash_target": weights.get("SGOV", 0.0),
        "gold_target": weights.get("IAU", 0.0),
        "target_return_input": target_map[target_return] * 100.0,
    }

    notes = [
        f"Your stock allocation was set near **{diagnostics['equity_target']:.0f}%** based on risk, horizon, and drawdown comfort.",
        f"Defensive sleeve: **{diagnostics['bond_target']:.0f}% bonds** + **{diagnostics['cash_target']:.0f}% cash**.",
        f"Diversifier sleeve: **{diagnostics['gold_target']:.0f}% gold** to reduce concentration in stock/bond moves.",
    ]
    if involvement == "Set and forget":
        notes.append("Portfolio was simplified to fewer funds because you chose a low-maintenance style.")

    return weights, diagnostics, notes


def _expected_stats(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: v / 100.0 for k, v in weights.items()}
    exp_ret = sum(w.get(t, 0.0) * float(ASSET_META[t]["exp_return"]) for t in ASSET_META if t in w)
    raw_vol = np.sqrt(sum((w.get(t, 0.0) * float(ASSET_META[t]["exp_vol"])) ** 2 for t in ASSET_META if t in w))
    exp_vol = float(raw_vol * 0.80)  # rough diversification adjustment

    equity_w = sum(w.get(t, 0.0) for t, m in ASSET_META.items() if m["bucket"] == "Equity")
    bond_w = sum(w.get(t, 0.0) for t, m in ASSET_META.items() if m["bucket"] == "Bonds")
    cash_w = w.get("SGOV", 0.0)
    gold_w = w.get("IAU", 0.0)
    stress_dd = float(0.45 * equity_w + 0.12 * bond_w + 0.02 * cash_w + 0.18 * gold_w)

    return {"exp_return": exp_ret, "exp_vol": exp_vol, "stress_drawdown": stress_dd}


def _future_value(initial: float, monthly: float, years: int, annual_return: float) -> float:
    n = int(12 * years)
    rm = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
    fv_init = initial * ((1.0 + rm) ** n)
    if abs(rm) < 1e-12:
        fv_add = monthly * n
    else:
        fv_add = monthly * (((1.0 + rm) ** n - 1.0) / rm)
    return float(fv_init + fv_add)


with st.form("portfolio_builder_form"):
    st.markdown("### Your Inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        risk = st.select_slider("Risk tolerance", options=RISK_CHOICES, value="Medium")
    with c2:
        horizon = st.selectbox("Investment horizon", HORIZON_CHOICES, index=2)
    with c3:
        target_return = st.selectbox("Expected annual return", RETURN_CHOICES, index=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        drawdown = st.selectbox("Max temporary drop you can tolerate", DRAWDOWN_CHOICES, index=1)
    with c5:
        income_need = st.selectbox("Need investment income soon?", INCOME_CHOICES, index=0)
    with c6:
        involvement = st.selectbox("How hands-on do you want to be?", INVOLVEMENT_CHOICES, index=0)

    c7, c8 = st.columns(2)
    with c7:
        starting_amount = st.number_input("Starting amount ($)", min_value=0.0, value=10000.0, step=1000.0)
    with c8:
        monthly_contrib = st.number_input("Monthly contribution ($)", min_value=0.0, value=500.0, step=50.0)

    build = st.form_submit_button("Build My Beginner Portfolio", type="primary", use_container_width=True)

if not build:
    st.info("Fill the questionnaire and click **Build My Beginner Portfolio**.")
    st.stop()

weights, diag, notes = _build_weights(risk, horizon, target_return, drawdown, income_need, involvement)
stats = _expected_stats(weights)
horizon_years = {"<3 years": 3, "3-5 years": 5, "5-10 years": 8, "10+ years": 15}[horizon]

st.markdown("## Recommended Allocation")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Equity", f"{diag['equity_target']:.1f}%")
k2.metric("Bonds", f"{diag['bond_target']:.1f}%")
k3.metric("Cash", f"{diag['cash_target']:.1f}%")
k4.metric("Diversifiers", f"{diag['gold_target']:.1f}%")

r1, r2, r3 = st.columns(3)
r1.metric("Expected long-run return", f"{stats['exp_return']*100:.1f}%")
r2.metric("Expected volatility", f"{stats['exp_vol']*100:.1f}%")
r3.metric("Stress drawdown estimate", f"-{stats['stress_drawdown']*100:.1f}%")

rows = []
for ticker, weight in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
    meta = ASSET_META[ticker]
    rows.append(
        {
            "Asset": meta["name"],
            "Bucket": meta["bucket"],
            "ETF": ticker,
            "Weight (%)": weight,
            "Dollar Amount ($)": starting_amount * weight / 100.0,
        }
    )
alloc_df = pd.DataFrame(rows)

st.dataframe(alloc_df.round(2), use_container_width=True)
st.bar_chart(alloc_df.set_index("Asset")["Weight (%)"], use_container_width=True)

st.markdown("### Why this mix")
for note in notes:
    st.markdown(f"- {note}")
st.markdown(
    f"- You selected a target return near **{diag['target_return_input']:.1f}%**, and this allocation aims for roughly **{stats['exp_return']*100:.1f}%** long-run."
)

st.markdown("### Projection (educational estimate)")
low = _future_value(starting_amount, monthly_contrib, horizon_years, max(0.0, stats["exp_return"] - 0.02))
base = _future_value(starting_amount, monthly_contrib, horizon_years, stats["exp_return"])
high = _future_value(starting_amount, monthly_contrib, horizon_years, stats["exp_return"] + 0.02)

p1, p2, p3 = st.columns(3)
p1.metric(f"Lower case ({horizon_years}y)", f"${low:,.0f}")
p2.metric(f"Base case ({horizon_years}y)", f"${base:,.0f}")
p3.metric(f"Upper case ({horizon_years}y)", f"${high:,.0f}")
st.caption(
    "Projection assumes constant returns and contributions. Real market returns are volatile and path-dependent."
)

st.markdown("### Beginner Checklist")
if involvement == "Set and forget":
    rebalance_txt = "Rebalance once per year."
elif involvement == "Review quarterly":
    rebalance_txt = "Review quarterly and rebalance if any weight drifts by more than 5%."
else:
    rebalance_txt = "Review quarterly with tighter drift bands (3%-5%)."
st.markdown(f"- {rebalance_txt}")
st.markdown("- Keep emergency savings separate from investment portfolio.")
st.markdown("- Avoid changing strategy after short-term market moves.")
st.markdown("- Revisit risk level only if goals, income, or time horizon changes.")

csv_df = alloc_df[["Asset", "Bucket", "ETF", "Weight (%)", "Dollar Amount ($)"]].copy()
buf = io.StringIO()
csv_df.to_csv(buf, index=False)
st.download_button(
    "Download Allocation CSV",
    data=buf.getvalue().encode(),
    file_name="beginner_portfolio_plan.csv",
    mime="text/csv",
    use_container_width=True,
)

st.page_link("pages/2_Rate_My_Portfolio.py", label="Open Rate My Portfolio to evaluate this allocation")
st.caption("Educational tool only. Not financial advice.")
