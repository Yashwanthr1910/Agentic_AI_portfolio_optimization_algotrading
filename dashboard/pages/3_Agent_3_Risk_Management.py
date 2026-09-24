"""
Agent 3 - Risk Management Dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# PROJECT SETUP
# =============================================================================

PAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGE_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dashboard.utils.data_loader import (
    get_agent3_risk_counts,
    load_agent3_risk,
    load_agent3_summary,
    load_full_trace,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Agent 3 - Risk Management",
    page_icon="🛡️",
    layout="wide",
)


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .agent-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .agent-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .flow-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        min-height: 120px;
        background: rgba(128,128,128,0.04);
    }

    .flow-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 6px;
    }

    .flow-text {
        font-size: 0.85rem;
        opacity: 0.80;
        line-height: 1.45;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def safe_float(value, default: float = 0.0) -> float:

    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def numeric(
    dataframe: pd.DataFrame,
    column: Optional[str],
) -> pd.Series:

    if column is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_page_data():

    return {
        "risk": load_agent3_risk(),
        "summary": load_agent3_summary(),
        "trace": load_full_trace(),
    }


try:
    data = load_page_data()

except Exception as error:

    st.error("Unable to load Agent-3 data.")
    st.exception(error)
    st.stop()


risk = data["risk"]
summary = data["summary"]
trace = data["trace"]


# =============================================================================
# COLUMN DETECTION
# =============================================================================

rank_column = first_existing_column(
    risk,
    (
        "agent3_rank",
        "risk_rank",
        "rank",
    ),
)

score_column = first_existing_column(
    risk,
    (
        "agent3_score",
        "risk_score",
        "score",
    ),
)

risk_level_column = first_existing_column(
    risk,
    (
        "risk_level",
        "risk_class",
    ),
)

weight_column = first_existing_column(
    risk,
    (
        "risk_adjusted_weight",
        "weight",
    ),
)

volatility_column = first_existing_column(
    risk,
    (
        "historical_volatility_annual",
        "annualized_volatility",
        "volatility",
    ),
)

var_column = first_existing_column(
    risk,
    (
        "individual_var_percent",
        "var_percent",
        "historical_var_percent",
        "var_95",
        "value_at_risk",
    ),
)

sharpe_column = first_existing_column(
    risk,
    (
        "sharpe_ratio",
        "sharpe",
    ),
)

drawdown_column = first_existing_column(
    risk,
    (
        "max_drawdown",
        "drawdown",
    ),
)


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="agent-title">
        Agent 3 — Risk Management Agent
    </div>

    <div class="agent-subtitle">
        Financial-risk evaluation using volatility, Value at Risk,
        Sharpe ratio, drawdown and trend information.
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Agent 3 does not simply remove high-risk stocks. "
    "It evaluates the Agent-2 trend signal together with financial risk "
    "and produces a risk-adjusted ranking and provisional allocation."
)


# =============================================================================
# WORKFLOW
# =============================================================================

st.subheader("Agent 3 Workflow")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Agent 2 Signal</div>
            <div class="flow-text">
                P(TOP30)<br>
                Trend rank<br>
                Trend class
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Volatility</div>
            <div class="flow-text">
                Historical price<br>
                variability and<br>
                market risk
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Value at Risk</div>
            <div class="flow-text">
                Downside-loss<br>
                estimate at the<br>
                configured confidence level
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Risk-Adjusted Return</div>
            <div class="flow-text">
                Sharpe ratio<br>
                Drawdown<br>
                Risk measures
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Agent 3 Output</div>
            <div class="flow-text">
                Risk score<br>
                Risk rank<br>
                Provisional weight
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# KPI SUMMARY
# =============================================================================

st.divider()
st.subheader("Risk Summary")

risk_counts = get_agent3_risk_counts()

top_stock = "N/A"

if rank_column and not risk.empty:

    temp = risk.copy()

    temp[rank_column] = pd.to_numeric(
        temp[rank_column],
        errors="coerce",
    )

    temp = temp.sort_values(
        rank_column,
        ascending=True,
    )

    top_stock = str(
        temp.iloc[0]["symbol"]
    )


weight_sum = 0.0

if weight_column:

    weight_sum = numeric(
        risk,
        weight_column,
    ).sum()


k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.metric(
        "Stocks Assessed",
        len(risk),
    )

with k2:
    st.metric(
        "LOW Risk",
        risk_counts.get("LOW", 0),
    )

with k3:
    st.metric(
        "MEDIUM Risk",
        risk_counts.get("MEDIUM", 0),
    )

with k4:
    st.metric(
        "HIGH Risk",
        risk_counts.get("HIGH", 0),
    )

with k5:
    st.metric(
        "Top Risk Rank",
        top_stock,
    )

with k6:
    st.metric(
        "Weight Sum",
        f"{weight_sum:.6f}",
    )


# =============================================================================
# RISK SCORE RANKING
# =============================================================================

st.divider()
st.subheader("Risk-Adjusted Ranking")

if score_column:

    chart_data = risk.copy()

    chart_data[score_column] = pd.to_numeric(
        chart_data[score_column],
        errors="coerce",
    )

    chart_data = chart_data.sort_values(
        score_column,
        ascending=True,
    )

    left, right = st.columns([1.25, 1])

    with left:

        fig = px.bar(
            chart_data,
            x=score_column,
            y="symbol",
            orientation="h",
            color=(
                risk_level_column
                if risk_level_column
                else None
            ),
            text=(
                chart_data[score_column]
                .map(lambda x: f"{x:.4f}")
            ),
            title="Agent-3 Risk Score",
        )

        fig.update_layout(
            height=520,
            xaxis_title="Agent-3 Score",
            yaxis_title="Stock",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        columns = [
            column
            for column in (
                rank_column,
                "symbol",
                score_column,
                risk_level_column,
                volatility_column,
                var_column,
                sharpe_column,
                drawdown_column,
                weight_column,
            )
            if (
                column is not None
                and column in risk.columns
            )
        ]

        display = risk.copy()

        if rank_column:
            display = display.sort_values(
                rank_column,
                ascending=True,
            )

        st.dataframe(
            display[columns],
            use_container_width=True,
            hide_index=True,
            height=520,
        )


# =============================================================================
# RISK CLASSIFICATION
# =============================================================================

st.divider()
st.subheader("Risk Classification")

risk_count_df = pd.DataFrame(
    {
        "Risk Level": list(risk_counts.keys()),
        "Stocks": list(risk_counts.values()),
    }
)

p1, p2 = st.columns(2)

with p1:

    fig = px.pie(
        risk_count_df,
        names="Risk Level",
        values="Stocks",
        hole=0.48,
        title="LOW / MEDIUM / HIGH Risk Distribution",
    )

    fig.update_layout(height=400)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with p2:

    st.dataframe(
        risk_count_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        The risk level summarizes the financial-risk characteristics
        used by Agent 3.

        A HIGH-risk classification does not automatically mean that
        Agent 4 must issue a SELL action.

        Agent 4 independently performs the final DQN trade decision.
        """
    )


# =============================================================================
# PROVISIONAL RISK WEIGHTS
# =============================================================================

st.divider()
st.subheader("Risk-Adjusted Provisional Weights")

if weight_column:

    weight_data = risk.copy()

    weight_data[weight_column] = numeric(
        weight_data,
        weight_column,
    )

    weight_data["weight_percent"] = (
        weight_data[weight_column] * 100
    )

    weight_data = weight_data.sort_values(
        "weight_percent",
        ascending=False,
    )

    fig = px.bar(
        weight_data,
        x="symbol",
        y="weight_percent",
        color=(
            risk_level_column
            if risk_level_column
            else None
        ),
        text=(
            weight_data["weight_percent"]
            .map(lambda x: f"{x:.2f}%")
        ),
        title="Agent-3 Provisional Portfolio Weights",
    )

    fig.update_layout(
        height=450,
        xaxis_title="Stock",
        yaxis_title="Weight (%)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "These are Agent-3 provisional risk-adjusted weights. "
        "They are not the final portfolio allocation. Agent 5 produces "
        "the final weights after Agent-4 trade decisions and MPT optimization."
    )


# =============================================================================
# VOLATILITY VS VAR
# =============================================================================

if (
    volatility_column
    and var_column
):

    st.divider()
    st.subheader("Volatility vs Value at Risk")

    scatter = risk.copy()

    scatter[volatility_column] = numeric(
        scatter,
        volatility_column,
    )

    scatter[var_column] = numeric(
        scatter,
        var_column,
    )

    fig = px.scatter(
        scatter,
        x=volatility_column,
        y=var_column,
        text="symbol",
        color=(
            risk_level_column
            if risk_level_column
            else None
        ),
        size=(
            score_column
            if score_column
            else None
        ),
        title="Risk Profile of Agent-1 Selected Stocks",
    )

    fig.update_traces(
        textposition="top center"
    )

    fig.update_layout(
        height=500,
        xaxis_title="Volatility",
        yaxis_title="Value at Risk",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# SHARPE
# =============================================================================

if sharpe_column:

    st.divider()
    st.subheader("Sharpe Ratio")

    sharpe_data = risk.copy()

    sharpe_data[sharpe_column] = numeric(
        sharpe_data,
        sharpe_column,
    )

    sharpe_data = sharpe_data.sort_values(
        sharpe_column,
        ascending=False,
    )

    fig = px.bar(
        sharpe_data,
        x="symbol",
        y=sharpe_column,
        color=(
            risk_level_column
            if risk_level_column
            else None
        ),
        text=(
            sharpe_data[sharpe_column]
            .map(lambda x: f"{x:.3f}")
        ),
        title="Risk-Adjusted Return by Stock",
    )

    fig.update_layout(
        height=430,
        xaxis_title="Stock",
        yaxis_title="Sharpe Ratio",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# AGENT 2 -> AGENT 3
# =============================================================================

st.divider()
st.subheader("Agent 2 → Agent 3 Ranking Change")

columns = [
    column
    for column in (
        "symbol",
        "agent2_top30_probability",
        "agent2_rank",
        "agent2_trend_class",
        "agent3_score",
        "agent3_rank",
        "agent3_risk_level",
        "agent3_risk_adjusted_weight",
    )
    if column in trace.columns
]

transfer = trace[columns].copy()

if "agent3_rank" in transfer.columns:

    transfer = transfer.sort_values(
        "agent3_rank",
        ascending=True,
    )

st.dataframe(
    transfer,
    use_container_width=True,
    hide_index=True,
)

st.success(
    f"{len(transfer)} Agent-2 predictions were successfully evaluated "
    "by the Risk Management Agent."
)


# =============================================================================
# AGENT 3 -> AGENT 4
# =============================================================================

st.divider()
st.subheader("Agent 3 → Agent 4 Transfer")

columns = [
    column
    for column in (
        "symbol",
        "agent3_score",
        "agent3_rank",
        "agent3_risk_level",
        "agent3_risk_adjusted_weight",
        "agent4_trade_action",
        "agent4_q_hold",
        "agent4_q_buy",
        "agent4_q_sell",
    )
    if column in trace.columns
]

handoff = trace[columns].copy()

if "agent3_rank" in handoff.columns:

    handoff = handoff.sort_values(
        "agent3_rank",
        ascending=True,
    )

st.dataframe(
    handoff,
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# STOCK INSPECTOR
# =============================================================================

st.divider()
st.subheader("Inspect a Stock")

stock = st.selectbox(
    "Select stock",
    sorted(
        risk["symbol"]
        .astype(str)
        .tolist()
    ),
)

row = risk[
    risk["symbol"] == stock
].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Risk Rank",
        row.get(
            rank_column,
            "N/A",
        )
        if rank_column
        else "N/A",
    )

with c2:
    st.metric(
        "Risk Score",
        (
            f"{safe_float(row.get(score_column)):.4f}"
            if score_column
            else "N/A"
        ),
    )

with c3:
    st.metric(
        "Risk Level",
        row.get(
            risk_level_column,
            "N/A",
        )
        if risk_level_column
        else "N/A",
    )

with c4:
    st.metric(
        "VaR",
        (
            f"{safe_float(row.get(var_column)):.4f}"
            if var_column
            else "N/A"
        ),
    )

with c5:
    st.metric(
        "Risk Weight",
        (
            f"{safe_float(row.get(weight_column)) * 100:.2f}%"
            if weight_column
            else "N/A"
        ),
    )

with st.expander("View all Agent-3 fields"):

    table = (
        risk[
            risk["symbol"] == stock
        ]
        .T
        .reset_index()
    )

    table.columns = [
        "Field",
        "Value",
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# SUMMARY JSON
# =============================================================================

if summary:

    with st.expander(
        "View Agent-3 risk summary report"
    ):

        st.json(summary)


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()

with st.expander("Agent-3 Methodology"):

    st.markdown(
        """
        ### Historical volatility

        Measures the variability of stock returns over a historical
        observation window.

        ### Value at Risk

        VaR estimates downside-loss exposure at a selected confidence
        level.

        ### Sharpe ratio

        Sharpe compares return with volatility and helps distinguish
        high raw returns from stronger risk-adjusted returns.

        ### Drawdown

        Drawdown measures decline from a previous portfolio or price peak.

        ### Risk score

        Agent 3 combines risk information with the incoming Agent-2 trend
        information to produce a risk-adjusted stock score.

        ### Risk-adjusted weight

        Agent 3 creates a provisional normalized portfolio state.

        This becomes an input to later agents and is not the final
        portfolio allocation.
        """
    )

st.divider()

st.caption(
    "Agent 3 provides financial-risk context for Agent 4. "
    "The final BUY/HOLD/SELL action is produced by the DQN Trade Execution Agent."
)