"""
End-to-End Five-Agent Pipeline Dashboard

Shows the complete stock journey:

Agent 1
    ↓
Agent 2
    ↓
Agent 3
    ↓
Agent 4
    ↓
Agent 5

Also displays:
- System validation
- Agent handoffs
- Full trace
- Stage runtimes
- Historical system runs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# PROJECT SETUP
# =============================================================================

PAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAGE_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from dashboard.utils.data_loader import (
    list_saved_runs,
    load_full_trace,
    load_integration_summary,
    load_preflight_summary,
    load_stage_timings,
    load_system_summary,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="End-to-End Pipeline",
    page_icon="🔗",
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

    .page-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .page-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .pipeline-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        min-height: 150px;
        background: rgba(128,128,128,0.04);
    }

    .pipeline-agent {
        font-size: 0.78rem;
        font-weight: 700;
        opacity: 0.65;
    }

    .pipeline-name {
        font-size: 1.05rem;
        font-weight: 750;
        margin-top: 5px;
        margin-bottom: 8px;
    }

    .pipeline-method {
        font-size: 0.83rem;
        opacity: 0.82;
        line-height: 1.4;
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


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_page_data():

    return {
        "trace":
            load_full_trace(),

        "system":
            load_system_summary(),

        "integration":
            load_integration_summary(),

        "preflight":
            load_preflight_summary(),

        "timings":
            load_stage_timings(),

        "saved_runs":
            list_saved_runs(),
    }


try:

    data = load_page_data()

except Exception as error:

    st.error(
        "Unable to load end-to-end pipeline data."
    )

    st.exception(error)

    st.stop()


trace = data["trace"]
system = data["system"]
integration = data["integration"]
preflight = data["preflight"]
timings = data["timings"]
saved_runs = data["saved_runs"]


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="page-title">
        End-to-End Five-Agent Pipeline
    </div>

    <div class="page-subtitle">
        Complete stock journey from selection through prediction,
        risk management, trade execution and portfolio allocation.
    </div>
    """,
    unsafe_allow_html=True,
)


overall_pass = bool(
    integration.get(
        "overall_pass",
        False,
    )
)


if overall_pass:

    st.success(
        "Complete Agent 1 → Agent 2 → Agent 3 → "
        "Agent 4 → Agent 5 integration: PASS"
    )

else:

    st.error(
        "Complete five-agent integration: FAIL"
    )


# =============================================================================
# SYSTEM KPIs
# =============================================================================

run_id = system.get(
    "run_id",
    "N/A",
)

mode = system.get(
    "mode",
    "N/A",
)

inference_date = integration.get(
    "inference_date",
    "N/A",
)


k1, k2, k3, k4, k5 = st.columns(5)


with k1:
    st.metric(
        "Run ID",
        run_id,
    )

with k2:
    st.metric(
        "Mode",
        str(mode).upper(),
    )

with k3:
    st.metric(
        "Inference Date",
        inference_date,
    )

with k4:
    st.metric(
        "Stocks Through Pipeline",
        len(trace),
    )

with k5:
    st.metric(
        "Integration",
        (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    )


# =============================================================================
# PIPELINE
# =============================================================================

st.divider()

st.subheader(
    "System Architecture"
)


a1, a2, a3, a4, a5 = st.columns(5)


with a1:

    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-agent">AGENT 1</div>
            <div class="pipeline-name">Stock Selection</div>
            <div class="pipeline-method">
                Decision Tree<br>
                +<br>
                BGSTO
                <br><br>
                Select 10 stocks
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with a2:

    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-agent">AGENT 2</div>
            <div class="pipeline-name">Trend Prediction</div>
            <div class="pipeline-method">
                Dilated LSTM<br>
                Transformer<br>
                Progressive Attention
                <br><br>
                P(TOP30)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with a3:

    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-agent">AGENT 3</div>
            <div class="pipeline-name">Risk Management</div>
            <div class="pipeline-method">
                Volatility<br>
                Value at Risk<br>
                Sharpe / Drawdown
                <br><br>
                Risk ranking
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with a4:

    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-agent">AGENT 4</div>
            <div class="pipeline-name">Trade Execution</div>
            <div class="pipeline-method">
                Deep Q-Network
                <br><br>
                HOLD<br>
                BUY<br>
                SELL
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with a5:

    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-agent">AGENT 5</div>
            <div class="pipeline-name">Rebalancing</div>
            <div class="pipeline-method">
                Markowitz MPT<br>
                +<br>
                RL Feedback
                <br><br>
                Final weights
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STOCK JOURNEY
# =============================================================================

st.divider()

st.subheader(
    "Stock-by-Stock Journey"
)


stocks = sorted(
    trace[
        "symbol"
    ]
    .astype(str)
    .tolist()
)


stock = st.selectbox(
    "Choose stock",
    stocks,
)


row = (
    trace[
        trace[
            "symbol"
        ]
        ==
        stock
    ]
    .iloc[
        0
    ]
)


j1, j2, j3, j4, j5 = st.columns(5)


with j1:

    rank = row.get(
        "agent1_final_rank",
        row.get(
            "agent1_selection_rank",
            "N/A",
        ),
    )

    st.metric(
        "Agent 1 Rank",
        rank,
    )

    st.caption(
        "BGSTO: "
        f"{safe_float(row.get('agent1_bgsto_score')):.4f}"
    )


with j2:

    st.metric(
        "Agent 2 Rank",
        row.get(
            "agent2_rank",
            "N/A",
        ),
    )

    st.caption(
        "P(TOP30): "
        f"{safe_float(row.get('agent2_top30_probability')):.4f}"
    )

    st.caption(
        str(
            row.get(
                "agent2_trend_class",
                "",
            )
        )
    )


with j3:

    st.metric(
        "Agent 3 Rank",
        row.get(
            "agent3_rank",
            "N/A",
        ),
    )

    st.caption(
        "Risk: "
        + str(
            row.get(
                "agent3_risk_level",
                "N/A",
            )
        )
    )


with j4:

    st.metric(
        "Agent 4 Action",
        row.get(
            "agent4_trade_action",
            "N/A",
        ),
    )

    st.caption(
        "Q margin: "
        f"{safe_float(row.get('agent4_q_margin')):.5f}"
    )


with j5:

    final_weight = safe_float(
        row.get(
            "agent5_final_weight",
            0,
        )
    )

    st.metric(
        "Agent 5 Allocation",
        f"{final_weight * 100:.2f}%",
    )

    st.caption(
        str(
            row.get(
                "agent5_portfolio_status",
                "N/A",
            )
        )
    )


# =============================================================================
# RANK EVOLUTION
# =============================================================================

st.divider()

st.subheader(
    "Ranking Evolution"
)


rank_columns = [
    column
    for column in (
        "agent1_final_rank",
        "agent2_rank",
        "agent3_rank",
    )
    if column in trace.columns
]


if len(rank_columns) >= 2:

    ranking = trace[
        [
            "symbol",
            *rank_columns,
        ]
    ].copy()

    rename_map = {
        "agent1_final_rank":
            "Agent 1",

        "agent2_rank":
            "Agent 2",

        "agent3_rank":
            "Agent 3",
    }

    ranking = ranking.rename(
        columns=rename_map
    )


    value_columns = [
        column
        for column in (
            "Agent 1",
            "Agent 2",
            "Agent 3",
        )
        if column in ranking.columns
    ]


    melted = ranking.melt(
        id_vars="symbol",
        value_vars=value_columns,
        var_name="Agent",
        value_name="Rank",
    )


    fig = px.line(
        melted,
        x="Agent",
        y="Rank",
        color="symbol",
        markers=True,
        title="How Stock Ranking Changes Across Agents",
    )

    fig.update_layout(
        height=520,
        yaxis=dict(
            autorange="reversed"
        ),
        yaxis_title="Rank",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# COMPLETE TRACE
# =============================================================================

st.divider()

st.subheader(
    "Unified Five-Agent Trace"
)


display_columns = [
    column
    for column in (
        "symbol",
        "agent1_final_rank",
        "agent1_final_score",
        "agent1_buy_probability",
        "agent1_bgsto_score",
        "agent2_top30_probability",
        "agent2_rank",
        "agent2_trend_class",
        "agent3_score",
        "agent3_rank",
        "agent3_risk_level",
        "agent3_risk_adjusted_weight",
        "agent4_q_hold",
        "agent4_q_buy",
        "agent4_q_sell",
        "agent4_q_margin",
        "agent4_trade_action",
        "agent5_final_weight",
        "agent5_final_allocation_percent",
        "agent5_portfolio_status",
        "agent5_final_decision",
    )
    if column in trace.columns
]


st.dataframe(
    trace[
        display_columns
    ],
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# SANKEY FLOW
# =============================================================================

st.divider()

st.subheader(
    "Decision Flow"
)


if (
    "agent2_trend_class"
    in trace.columns
    and
    "agent3_risk_level"
    in trace.columns
    and
    "agent4_trade_action"
    in trace.columns
):

    trend_values = sorted(
        trace[
            "agent2_trend_class"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    risk_values = sorted(
        trace[
            "agent3_risk_level"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    action_values = sorted(
        trace[
            "agent4_trade_action"
        ]
        .astype(str)
        .unique()
        .tolist()
    )


    nodes = (
        [
            f"Trend: {value}"
            for value in trend_values
        ]
        +
        [
            f"Risk: {value}"
            for value in risk_values
        ]
        +
        [
            f"Action: {value}"
            for value in action_values
        ]
    )


    node_index = {
        node: index
        for index, node
        in enumerate(nodes)
    }


    sources = []
    targets = []
    values = []


    # Agent 2 -> Agent 3
    first_flow = (
        trace.groupby(
            [
                "agent2_trend_class",
                "agent3_risk_level",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


    for _, flow_row in first_flow.iterrows():

        sources.append(
            node_index[
                "Trend: "
                + str(
                    flow_row[
                        "agent2_trend_class"
                    ]
                )
            ]
        )

        targets.append(
            node_index[
                "Risk: "
                + str(
                    flow_row[
                        "agent3_risk_level"
                    ]
                )
            ]
        )

        values.append(
            int(
                flow_row[
                    "count"
                ]
            )
        )


    # Agent 3 -> Agent 4
    second_flow = (
        trace.groupby(
            [
                "agent3_risk_level",
                "agent4_trade_action",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


    for _, flow_row in second_flow.iterrows():

        sources.append(
            node_index[
                "Risk: "
                + str(
                    flow_row[
                        "agent3_risk_level"
                    ]
                )
            ]
        )

        targets.append(
            node_index[
                "Action: "
                + str(
                    flow_row[
                        "agent4_trade_action"
                    ]
                )
            ]
        )

        values.append(
            int(
                flow_row[
                    "count"
                ]
            )
        )


    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=nodes,
                    pad=20,
                    thickness=20,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                ),
            )
        ]
    )


    fig.update_layout(
        title_text=(
            "Trend Classification → Risk Level → DQN Action"
        ),
        height=540,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# INTEGRATION CHECKS
# =============================================================================

st.divider()

st.subheader(
    "Integration Validation"
)


group_status = integration.get(
    "group_status",
    {},
)


if group_status:

    integration_df = pd.DataFrame(
        [
            {
                "Validation":
                    name.replace(
                        "_",
                        " "
                    ).title(),

                "Status":
                    (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
            }
            for name, passed
            in group_status.items()
        ]
    )


    st.dataframe(
        integration_df,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# PREFLIGHT STATUS
# =============================================================================

with st.expander(
    "System preflight report"
):

    st.json(
        preflight
    )


# =============================================================================
# RUNTIME
# =============================================================================

st.divider()

st.subheader(
    "Pipeline Stage Runtime"
)


if (
    not timings.empty
    and
    "duration_seconds"
    in timings.columns
):

    runtime = timings.copy()

    runtime[
        "duration_seconds"
    ] = pd.to_numeric(
        runtime[
            "duration_seconds"
        ],
        errors="coerce",
    )

    runtime = runtime.dropna(
        subset=[
            "duration_seconds"
        ]
    )


    fig = px.bar(
        runtime,
        x="stage",
        y="duration_seconds",
        text=(
            runtime[
                "duration_seconds"
            ]
            .map(
                lambda x:
                    f"{x:.2f}s"
            )
        ),
        title="Execution Time by Pipeline Stage",
    )


    fig.update_layout(
        height=500,
        xaxis_title="Stage",
        yaxis_title="Seconds",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.dataframe(
        runtime,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# RUN HISTORY
# =============================================================================

st.divider()

st.subheader(
    "Saved System Runs"
)


if not saved_runs.empty:

    st.dataframe(
        saved_runs,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No saved full-system run snapshots are available."
    )


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()

with st.expander(
    "How the complete pipeline works"
):

    st.markdown(
        """
        ### Agent 1 — Selection

        Reduces the broad universe to a smaller set of candidate stocks.

        ### Agent 2 — Prediction

        Estimates relative future trend strength using the neural
        sequence model.

        ### Agent 3 — Risk

        Incorporates financial-risk measures and changes the ranking
        based on risk-adjusted characteristics.

        ### Agent 4 — Execution

        Uses a Deep Q-Network to select BUY, HOLD or SELL.

        ### Agent 5 — Portfolio construction

        Uses the Agent-4 decisions as constraints/feedback and performs
        portfolio optimization.

        ### Final integration validation

        The system validates:

        - same symbol universe across all agents
        - consistent inference dates
        - Agent-2 signal transfer
        - Agent-3 risk transfer
        - DQN action consistency
        - Agent-4 action preservation
        - SELL → zero-weight enforcement
        - normalized final portfolio
        - unified-trace consistency
        """
    )


st.divider()

st.caption(
    "The end-to-end page displays the latest validated complete-system run."
)