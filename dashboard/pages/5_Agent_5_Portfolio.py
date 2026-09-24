"""
Agent 5 - Portfolio Rebalancing Dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

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
    get_active_portfolio,
    get_exited_portfolio,
    load_agent5_allocation_comparison,
    load_agent5_evaluation,
    load_agent5_integration,
    load_agent5_mpt_summary,
    load_agent5_portfolio,
    load_agent5_rebalancing_environment,
    load_agent5_summary,
    load_full_trace,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Agent 5 - Portfolio Rebalancing",
    page_icon="⚖️",
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
    }

    .agent-subtitle {
        color: #6b7280;
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

def safe_float(
    value,
    default: float = 0.0,
) -> float:

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def flatten_json(
    data: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:

    result = {}

    for key, value in data.items():

        current = (
            f"{prefix}.{key}"
            if prefix
            else str(key)
        )

        if isinstance(value, dict):

            result.update(
                flatten_json(
                    value,
                    current,
                )
            )

        else:

            result[current] = value

    return result


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_page_data():

    return {
        "portfolio":
            load_agent5_portfolio(),

        "active":
            get_active_portfolio(),

        "exited":
            get_exited_portfolio(),

        "trace":
            load_full_trace(),

        "evaluation":
            load_agent5_evaluation(),

        "summary":
            load_agent5_summary(),

        "environment":
            load_agent5_rebalancing_environment(),

        "integration":
            load_agent5_integration(),

        "mpt":
            load_agent5_mpt_summary(),

        "comparison":
            load_agent5_allocation_comparison(),
    }


try:
    data = load_page_data()

except Exception as error:

    st.error(
        "Unable to load Agent-5 data."
    )

    st.exception(error)
    st.stop()


portfolio = data["portfolio"]
active = data["active"]
exited = data["exited"]
trace = data["trace"]
evaluation = data["evaluation"]
summary = data["summary"]
environment = data["environment"]
integration = data["integration"]
mpt = data["mpt"]
comparison = data["comparison"]


# =============================================================================
# PREPARE WEIGHTS
# =============================================================================

portfolio = portfolio.copy()

if "final_weight" in portfolio.columns:

    portfolio["final_weight"] = pd.to_numeric(
        portfolio["final_weight"],
        errors="coerce",
    )

    portfolio["allocation_percent"] = (
        portfolio["final_weight"] * 100
    )


for column in (
    "current_weight",
    "target_weight",
    "weight_change",
):

    if column in portfolio.columns:

        portfolio[column] = pd.to_numeric(
            portfolio[column],
            errors="coerce",
        )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="agent-title">
        Agent 5 — Portfolio Rebalancing Agent
    </div>

    <div class="agent-subtitle">
        Final portfolio construction using Markowitz Mean-Variance
        Optimization and Agent-4 trade feedback.
    </div>
    """,
    unsafe_allow_html=True,
)

st.success(
    "Agent 5 is the final decision layer of the five-agent pipeline. "
    "It converts upstream predictions, risk information and DQN actions "
    "into normalized final portfolio weights."
)


# =============================================================================
# WORKFLOW
# =============================================================================

st.subheader("Agent 5 Workflow")

a1, a2, a3, a4, a5 = st.columns(5)

with a1:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Agent-4 Actions</div>
            <div class="flow-text">
                BUY<br>
                HOLD<br>
                SELL
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a2:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Eligible Assets</div>
            <div class="flow-text">
                SELL assets removed<br>
                BUY/HOLD stocks<br>
                remain eligible
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a3:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">MPT</div>
            <div class="flow-text">
                Mean-Variance<br>
                portfolio<br>
                optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a4:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">RL Feedback</div>
            <div class="flow-text">
                Trade-action<br>
                feedback and<br>
                constraints
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a5:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Final Portfolio</div>
            <div class="flow-text">
                Normalized weights<br>
                active positions<br>
                exits
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# KPIs
# =============================================================================

st.divider()
st.subheader("Final Portfolio Summary")

weight_sum = (
    portfolio["final_weight"].sum()
    if "final_weight" in portfolio.columns
    else 0.0
)

top_stock = "N/A"
top_weight = 0.0

if (
    "final_weight" in portfolio.columns
    and not portfolio.empty
):

    top_row = (
        portfolio.sort_values(
            "final_weight",
            ascending=False,
        )
        .iloc[0]
    )

    top_stock = str(
        top_row["symbol"]
    )

    top_weight = safe_float(
        top_row["final_weight"]
    )


k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Portfolio Stocks",
        len(portfolio),
    )

with k2:
    st.metric(
        "Active Assets",
        len(active),
    )

with k3:
    st.metric(
        "Exited Assets",
        len(exited),
    )

with k4:
    st.metric(
        "Largest Allocation",
        top_stock,
        f"{top_weight * 100:.2f}%",
    )

with k5:
    st.metric(
        "Weight Sum",
        f"{weight_sum:.8f}",
    )


# =============================================================================
# FINAL ALLOCATION
# =============================================================================

st.divider()
st.subheader("Final Portfolio Allocation")

if "allocation_percent" in portfolio.columns:

    allocation = portfolio.sort_values(
        "allocation_percent",
        ascending=False,
    )

    c1, c2 = st.columns([1.25, 1])

    with c1:

        fig = px.bar(
            allocation,
            x="symbol",
            y="allocation_percent",
            color=(
                "trade_action"
                if "trade_action" in allocation.columns
                else None
            ),
            text=(
                allocation["allocation_percent"]
                .map(lambda x: f"{x:.2f}%")
            ),
            title="Final Agent-5 Portfolio Weights",
        )

        fig.update_layout(
            height=500,
            xaxis_title="Stock",
            yaxis_title="Allocation (%)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c2:

        columns = [
            column
            for column in (
                "symbol",
                "trade_action",
                "final_weight",
                "final_allocation_percent",
                "portfolio_status",
                "final_decision",
            )
            if column in allocation.columns
        ]

        st.dataframe(
            allocation[columns],
            use_container_width=True,
            hide_index=True,
            height=500,
        )


# =============================================================================
# PORTFOLIO DONUT
# =============================================================================

st.divider()
st.subheader("Portfolio Composition")

if (
    "final_weight" in active.columns
    and not active.empty
):

    pie_data = active.copy()

    pie_data["allocation_percent"] = (
        pd.to_numeric(
            pie_data["final_weight"],
            errors="coerce",
        )
        * 100
    )

    fig = px.pie(
        pie_data,
        names="symbol",
        values="allocation_percent",
        hole=0.42,
        title="Active Portfolio Allocation",
    )

    fig.update_layout(height=520)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# CURRENT VS FINAL
# =============================================================================

if (
    "current_weight" in portfolio.columns
    and "final_weight" in portfolio.columns
):

    st.divider()
    st.subheader("Portfolio Rebalancing Transition")

    transition = portfolio[
        [
            "symbol",
            "current_weight",
            "final_weight",
        ]
    ].copy()

    transition["current_percent"] = (
        transition["current_weight"] * 100
    )

    transition["final_percent"] = (
        transition["final_weight"] * 100
    )

    melted = transition.melt(
        id_vars="symbol",
        value_vars=[
            "current_percent",
            "final_percent",
        ],
        var_name="Portfolio State",
        value_name="Weight (%)",
    )

    fig = px.bar(
        melted,
        x="symbol",
        y="Weight (%)",
        color="Portfolio State",
        barmode="group",
        title="Current Agent-3 Weight vs Final Agent-5 Weight",
    )

    fig.update_layout(
        height=500,
        xaxis_title="Stock",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# ACTIVE / EXITED
# =============================================================================

st.divider()

c1, c2 = st.columns(2)

with c1:

    st.subheader("Active Positions")

    columns = [
        column
        for column in (
            "symbol",
            "trade_action",
            "final_weight",
            "portfolio_status",
            "final_decision",
        )
        if column in active.columns
    ]

    st.dataframe(
        active[columns],
        use_container_width=True,
        hide_index=True,
    )

with c2:

    st.subheader("Exited Positions")

    columns = [
        column
        for column in (
            "symbol",
            "trade_action",
            "final_weight",
            "portfolio_status",
            "final_decision",
        )
        if column in exited.columns
    ]

    st.dataframe(
        exited[columns],
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# AGENT 4 -> 5
# =============================================================================

st.divider()
st.subheader("Agent 4 → Agent 5 Transfer")

columns = [
    column
    for column in (
        "symbol",
        "agent4_trade_action",
        "agent5_current_weight",
        "agent5_target_weight",
        "agent5_weight_change",
        "agent5_rebalance_direction",
        "agent5_final_weight",
        "agent5_final_allocation_percent",
        "agent5_portfolio_status",
        "agent5_final_decision",
    )
    if column in trace.columns
]

transfer = trace[columns].copy()

if "agent5_final_weight" in transfer.columns:

    transfer = transfer.sort_values(
        "agent5_final_weight",
        ascending=False,
    )

st.dataframe(
    transfer,
    use_container_width=True,
    hide_index=True,
)

st.success(
    "Agent-4 trade actions have been transferred into the final "
    "portfolio construction process."
)


# =============================================================================
# ALLOCATION COMPARISON FILE
# =============================================================================

if not comparison.empty:

    st.divider()
    st.subheader("Allocation Comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# EVALUATION REPORT
# =============================================================================

st.divider()
st.subheader("Portfolio Evaluation")

if evaluation:

    flat_evaluation = flatten_json(
        evaluation
    )

    evaluation_table = pd.DataFrame(
        {
            "Metric": list(
                flat_evaluation.keys()
            ),
            "Value": list(
                flat_evaluation.values()
            ),
        }
    )

    st.dataframe(
        evaluation_table,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Agent-5 evaluation summary is unavailable."
    )


# =============================================================================
# MPT
# =============================================================================

st.divider()
st.subheader("Markowitz Optimization")

if mpt:

    flat_mpt = flatten_json(
        mpt
    )

    table = pd.DataFrame(
        {
            "Parameter": list(
                flat_mpt.keys()
            ),
            "Value": list(
                flat_mpt.values()
            ),
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "MPT optimization report is unavailable."
    )


# =============================================================================
# STOCK INSPECTOR
# =============================================================================

st.divider()
st.subheader("Inspect a Final Portfolio Position")

stock = st.selectbox(
    "Select stock",
    sorted(
        portfolio["symbol"]
        .astype(str)
        .tolist()
    ),
)

row = portfolio[
    portfolio["symbol"] == stock
].iloc[0]

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.metric(
        "Trade Action",
        row.get(
            "trade_action",
            "N/A",
        ),
    )

with p2:
    st.metric(
        "Final Weight",
        (
            f"{safe_float(row.get('final_weight')) * 100:.2f}%"
        ),
    )

with p3:
    st.metric(
        "Portfolio Status",
        row.get(
            "portfolio_status",
            "N/A",
        ),
    )

with p4:
    st.metric(
        "Final Decision",
        row.get(
            "final_decision",
            "N/A",
        ),
    )


with st.expander(
    "View all Agent-5 fields"
):

    table = (
        portfolio[
            portfolio["symbol"] == stock
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
# REPORTS
# =============================================================================

with st.expander(
    "Agent-5 complete summary"
):

    if summary:
        st.json(summary)
    else:
        st.info("Summary not available.")


with st.expander(
    "Rebalancing environment report"
):

    if environment:
        st.json(environment)
    else:
        st.info("Environment report not available.")


with st.expander(
    "Agent 4 → Agent 5 integration report"
):

    if integration:
        st.json(integration)
    else:
        st.info("Integration report not available.")


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()

with st.expander("Agent-5 Methodology"):

    st.markdown(
        """
        ### 1. Agent-4 constraints

        Agent-4 BUY/HOLD/SELL decisions enter Agent 5.

        Stocks with a SELL decision receive zero target allocation.

        ### 2. Historical return matrix

        Agent 5 constructs historical return observations for the eligible
        stocks.

        ### 3. Covariance matrix

        The covariance matrix estimates how asset returns move relative to
        one another.

        ### 4. Markowitz Mean-Variance Optimization

        Portfolio weights are optimized using the risk/return relationship
        represented by expected returns and covariance.

        The implementation is long-only and fully invested.

        ### 5. RL feedback

        Agent-4 trading feedback is applied to the portfolio optimization
        process.

        ### 6. Rebalancing environment

        The system compares current provisional weights with target weights.

        ### 7. Final portfolio

        Final portfolio weights are normalized so that:

        `sum(weights) = 1`

        Agent-4 SELL positions must have:

        `final_weight = 0`

        ### Important

        Agent-5 historical evaluation metrics describe the historical
        estimation/evaluation window used by the implementation.

        They should not be interpreted as guaranteed future performance.
        """
    )


st.divider()

st.caption(
    "Agent 5 is the final portfolio-allocation stage of the current "
    "five-agent system."
)