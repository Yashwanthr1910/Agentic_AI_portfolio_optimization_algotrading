"""
Agent 4 - Trade Execution Dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
    get_agent4_action_counts,
    load_agent4_integration,
    load_agent4_trades,
    load_full_trace,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Agent 4 - Trade Execution",
    page_icon="🎯",
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
# LOAD DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_page_data():

    return {
        "trades": load_agent4_trades(),
        "trace": load_full_trace(),
        "integration": load_agent4_integration(),
    }


try:
    data = load_page_data()

except Exception as error:

    st.error("Unable to load Agent-4 data.")
    st.exception(error)
    st.stop()


trades = data["trades"]
trace = data["trace"]
integration = data["integration"]


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

for column in (
    "q_hold",
    "q_buy",
    "q_sell",
):

    if column in trades.columns:

        trades[column] = pd.to_numeric(
            trades[column],
            errors="coerce",
        )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="agent-title">
        Agent 4 — Trade Execution Agent
    </div>

    <div class="agent-subtitle">
        Deep Q-Network decision layer producing BUY, HOLD and SELL actions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Agent 4 receives trend and risk information from previous agents "
    "and uses a Deep Q-Network to choose one of three actions: "
    "HOLD, BUY or SELL."
)


# =============================================================================
# WORKFLOW
# =============================================================================

st.subheader("Agent 4 Workflow")

a1, a2, a3, a4, a5 = st.columns(5)

with a1:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Agent-3 State</div>
            <div class="flow-text">
                Trend signal<br>
                Risk score<br>
                Risk weight
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a2:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">DQN State</div>
            <div class="flow-text">
                Financial and<br>
                model features<br>
                form state vector
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a3:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Q-Network</div>
            <div class="flow-text">
                Neural network<br>
                estimates value<br>
                of each action
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a4:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Q Values</div>
            <div class="flow-text">
                Q(HOLD)<br>
                Q(BUY)<br>
                Q(SELL)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a5:
    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">Trade Action</div>
            <div class="flow-text">
                argmax(Q)<br>
                →<br>
                BUY / HOLD / SELL
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# ACTION COUNTS
# =============================================================================

st.divider()
st.subheader("Trade Decision Summary")

counts = get_agent4_action_counts()

prediction_date = "N/A"

if (
    "prediction_date" in trades.columns
    and not trades.empty
):

    prediction_date = str(
        trades["prediction_date"].iloc[0]
    )


buy_symbols = (
    trades.loc[
        trades["trade_action"]
        .astype(str)
        .str.upper()
        .eq("BUY"),
        "symbol",
    ]
    .astype(str)
    .tolist()
)

sell_symbols = (
    trades.loc[
        trades["trade_action"]
        .astype(str)
        .str.upper()
        .eq("SELL"),
        "symbol",
    ]
    .astype(str)
    .tolist()
)


k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Stocks",
        len(trades),
    )

with k2:
    st.metric(
        "BUY",
        counts.get("BUY", 0),
    )

with k3:
    st.metric(
        "HOLD",
        counts.get("HOLD", 0),
    )

with k4:
    st.metric(
        "SELL",
        counts.get("SELL", 0),
    )

with k5:
    st.metric(
        "Prediction Date",
        prediction_date,
    )


if buy_symbols:
    st.success(
        "BUY action: "
        + ", ".join(buy_symbols)
    )

if sell_symbols:
    st.warning(
        "SELL actions: "
        + ", ".join(sell_symbols)
    )


# =============================================================================
# ACTION DISTRIBUTION
# =============================================================================

st.divider()
st.subheader("Action Distribution")

action_df = pd.DataFrame(
    {
        "Action": list(counts.keys()),
        "Stocks": list(counts.values()),
    }
)

c1, c2 = st.columns(2)

with c1:

    fig = px.pie(
        action_df,
        names="Action",
        values="Stocks",
        hole=0.48,
        title="DQN BUY / HOLD / SELL Distribution",
    )

    fig.update_layout(height=400)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with c2:

    st.dataframe(
        action_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        Agent 4 selects the action corresponding to the largest Q-value.

        **Action mapping**

        `0 → HOLD`

        `1 → BUY`

        `2 → SELL`
        """
    )


# =============================================================================
# Q VALUES
# =============================================================================

st.divider()
st.subheader("DQN Q-Values")

required_q = {
    "q_hold",
    "q_buy",
    "q_sell",
}

if required_q.issubset(
    trades.columns
):

    q_table = trades[
        [
            "symbol",
            "q_hold",
            "q_buy",
            "q_sell",
            "trade_action",
        ]
    ].copy()

    melted = q_table.melt(
        id_vars=[
            "symbol",
            "trade_action",
        ],
        value_vars=[
            "q_hold",
            "q_buy",
            "q_sell",
        ],
        var_name="Q Action",
        value_name="Q Value",
    )

    fig = px.bar(
        melted,
        x="symbol",
        y="Q Value",
        color="Q Action",
        barmode="group",
        title="Q(HOLD), Q(BUY), Q(SELL) by Stock",
    )

    fig.update_layout(
        height=500,
        xaxis_title="Stock",
        yaxis_title="Q Value",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# Q MARGIN
# =============================================================================

st.divider()
st.subheader("Decision Confidence Margin")

if required_q.issubset(
    trades.columns
):

    confidence = trades.copy()

    q_array = confidence[
        [
            "q_hold",
            "q_buy",
            "q_sell",
        ]
    ].to_numpy(dtype=float)

    sorted_q = np.sort(
        q_array,
        axis=1,
    )

    confidence["q_margin"] = (
        sorted_q[:, -1]
        -
        sorted_q[:, -2]
    )

    confidence = confidence.sort_values(
        "q_margin",
        ascending=False,
    )

    fig = px.bar(
        confidence,
        x="symbol",
        y="q_margin",
        color="trade_action",
        text=(
            confidence["q_margin"]
            .map(lambda x: f"{x:.5f}")
        ),
        title="Difference Between Best and Second-Best Q Value",
    )

    fig.update_layout(
        height=450,
        xaxis_title="Stock",
        yaxis_title="Q Margin",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "A larger Q-margin indicates a larger separation between the "
        "selected DQN action and the second-highest estimated action value."
    )


# =============================================================================
# COMPLETE TRADE TABLE
# =============================================================================

st.divider()
st.subheader("Trade Decisions")

columns = [
    column
    for column in (
        "symbol",
        "prediction_date",
        "q_hold",
        "q_buy",
        "q_sell",
        "action_id",
        "trade_action",
    )
    if column in trades.columns
]

st.dataframe(
    trades[columns],
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# AGENT 3 -> 4 TRANSFER
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
        "agent4_q_hold",
        "agent4_q_buy",
        "agent4_q_sell",
        "agent4_trade_action",
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
# AGENT 4 -> 5
# =============================================================================

st.divider()
st.subheader("Agent 4 → Agent 5 Effect")

columns = [
    column
    for column in (
        "symbol",
        "agent4_trade_action",
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

st.caption(
    "Agent-4 SELL actions become zero-weight EXIT positions in Agent 5."
)


# =============================================================================
# STOCK INSPECTOR
# =============================================================================

st.divider()
st.subheader("Inspect a DQN Decision")

stock = st.selectbox(
    "Select stock",
    sorted(
        trades["symbol"]
        .astype(str)
        .tolist()
    ),
)

row = trades[
    trades["symbol"] == stock
].iloc[0]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Q(HOLD)",
        f"{float(row.get('q_hold', 0)):.6f}",
    )

with c2:
    st.metric(
        "Q(BUY)",
        f"{float(row.get('q_buy', 0)):.6f}",
    )

with c3:
    st.metric(
        "Q(SELL)",
        f"{float(row.get('q_sell', 0)):.6f}",
    )

with c4:
    st.metric(
        "Selected Action",
        row.get(
            "trade_action",
            "N/A",
        ),
    )


if required_q.issubset(
    trades.columns
):

    stock_q = pd.DataFrame(
        {
            "Action": [
                "HOLD",
                "BUY",
                "SELL",
            ],
            "Q Value": [
                row["q_hold"],
                row["q_buy"],
                row["q_sell"],
            ],
        }
    )

    fig = px.bar(
        stock_q,
        x="Action",
        y="Q Value",
        text=(
            stock_q["Q Value"]
            .map(lambda x: f"{x:.6f}")
        ),
        title=f"DQN Decision for {stock}",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# INTEGRATION REPORT
# =============================================================================

if integration:

    with st.expander(
        "Agent 3 → Agent 4 integration report"
    ):

        st.json(integration)


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()

with st.expander("Agent-4 DQN Methodology"):

    st.markdown(
        """
        ### State

        Agent 4 receives the processed financial state generated by
        upstream agents.

        ### Actions

        The DQN action space contains:

        - HOLD
        - BUY
        - SELL

        ### Q-function

        The neural network estimates the expected long-term value of
        taking each action.

        For a state `s`:

        `Q(s, HOLD)`

        `Q(s, BUY)`

        `Q(s, SELL)`

        ### Action selection

        During final inference:

        `action = argmax(Q-values)`

        ### Agent 5 handoff

        Agent 4 does not determine exact final portfolio percentages.

        Instead its BUY/HOLD/SELL decisions become constraints and
        feedback for Agent 5.

        In particular, SELL positions are removed from the final allocation.
        """
    )


st.divider()

st.caption(
    "DQN actions are model outputs. Agent 5 performs the final portfolio "
    "construction and allocation step."
)