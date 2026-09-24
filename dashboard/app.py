"""
dashboard/app.py

Main Streamlit dashboard for the complete five-agent
Agentic AI Portfolio Optimization system.

Run using:

    .\\.venv\\Scripts\\python.exe -m streamlit run .\\dashboard\\app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# PROJECT IMPORT SETUP
# =============================================================================

DASHBOARD_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    DASHBOARD_DIR.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from dashboard.utils.data_loader import (
    get_agent2_trend_counts,
    get_agent3_risk_counts,
    get_agent4_action_counts,
    get_active_portfolio,
    get_exited_portfolio,
    load_agent5_evaluation,
    load_agent5_portfolio,
    load_dashboard_bundle,
    load_full_trace,
    load_integration_summary,
    load_stage_timings,
)


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=(
        "Agentic AI Portfolio Optimization"
    ),
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .agent-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        min-height: 165px;
        background: rgba(128,128,128,0.04);
    }

    .agent-number {
        font-size: 0.80rem;
        font-weight: 700;
        opacity: 0.70;
    }

    .agent-name {
        font-size: 1.08rem;
        font-weight: 750;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .agent-method {
        font-size: 0.87rem;
        opacity: 0.80;
        line-height: 1.45;
    }

    .status-pass {
        font-weight: 700;
    }

    .section-space {
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(
    show_spinner=False
)
def load_data():
    """
    Load all latest validated dashboard data.
    """

    return (
        load_dashboard_bundle()
    )


def refresh_data():
    """
    Clear dashboard cache.
    """

    st.cache_data.clear()


# =============================================================================
# SAFE HELPERS
# =============================================================================

def safe_number(
    value,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def format_percent(
    value,
    decimals: int = 2,
) -> str:

    number = safe_number(
        value
    )

    return (
        f"{number:.{decimals}f}%"
    )


def status_text(
    value,
) -> str:

    if bool(
        value
    ):

        return "PASS"

    return "FAIL"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title(
        "Portfolio AI"
    )

    st.caption(
        "Five-Agent Portfolio Optimization System"
    )

    st.divider()

    st.markdown(
        """
        **Pipeline**

        Agent 1 — Stock Selection

        Agent 2 — Trend Prediction

        Agent 3 — Risk Management

        Agent 4 — Trade Execution

        Agent 5 — Portfolio Rebalancing
        """
    )

    st.divider()

    if st.button(
        "Refresh latest data",
        use_container_width=True,
    ):

        refresh_data()

        st.rerun()

    st.caption(
        "Dashboard reads validated outputs from "
        "`reports/system/latest`."
    )


# =============================================================================
# LOAD DATA
# =============================================================================

try:

    bundle = (
        load_data()
    )

except Exception as error:

    st.error(
        "Unable to load dashboard data."
    )

    st.exception(
        error
    )

    st.stop()


status = (
    bundle[
        "system_status"
    ]
)

trace = (
    bundle[
        "full_trace"
    ]
)

agent1 = (
    bundle[
        "agent1_selected"
    ]
)

agent2 = (
    bundle[
        "agent2_predictions"
    ]
)

agent3 = (
    bundle[
        "agent3_risk"
    ]
)

agent4 = (
    bundle[
        "agent4_trades"
    ]
)

agent5 = (
    bundle[
        "agent5_portfolio"
    ]
)

stage_timings = (
    bundle[
        "stage_timings"
    ]
)

saved_runs = (
    bundle[
        "saved_runs"
    ]
)


# =============================================================================
# PAGE HEADER
# =============================================================================

st.markdown(
    """
    <div class="main-title">
        Agentic AI Portfolio Optimization
    </div>

    <div class="main-subtitle">
        End-to-End Multi-Agent Algorithmic Trading and Portfolio
        Rebalancing Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SYSTEM STATUS BANNER
# =============================================================================

if (
    status.get(
        "overall_pass",
        False,
    )
):

    st.success(
        "Full five-agent system status: PASS"
    )

else:

    st.error(
        "Full five-agent system status: FAIL"
    )


# =============================================================================
# TOP KPI CARDS
# =============================================================================

col1, col2, col3, col4, col5, col6 = (
    st.columns(
        6
    )
)

with col1:

    st.metric(
        "Run ID",
        status.get(
            "run_id",
            "N/A",
        ),
    )

with col2:

    st.metric(
        "Inference Date",
        status.get(
            "inference_date",
            "N/A",
        ),
    )

with col3:

    st.metric(
        "Selected Stocks",
        status.get(
            "total_stocks",
            0,
        ),
    )

with col4:

    st.metric(
        "Active Assets",
        status.get(
            "active_assets",
            0,
        ),
    )

with col5:

    st.metric(
        "Exited Assets",
        status.get(
            "exited_assets",
            0,
        ),
    )

with col6:

    weight_sum = safe_number(
        status.get(
            "final_weight_sum",
            0,
        )
    )

    st.metric(
        "Weight Sum",
        f"{weight_sum:.6f}",
    )


# =============================================================================
# PIPELINE ARCHITECTURE
# =============================================================================

st.divider()

st.subheader(
    "Five-Agent Decision Pipeline"
)

a1, a2, a3, a4, a5 = (
    st.columns(
        5
    )
)

with a1:

    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-number">
                AGENT 1
            </div>
            <div class="agent-name">
                Stock Selection
            </div>
            <div class="agent-method">
                Decision Tree<br>
                +<br>
                BGSTO Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a2:

    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-number">
                AGENT 2
            </div>
            <div class="agent-name">
                Trend Prediction
            </div>
            <div class="agent-method">
                Dilated LSTM<br>
                Transformer<br>
                Progressive Attention
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a3:

    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-number">
                AGENT 3
            </div>
            <div class="agent-name">
                Risk Management
            </div>
            <div class="agent-method">
                Volatility<br>
                VaR<br>
                Sharpe Ratio
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a4:

    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-number">
                AGENT 4
            </div>
            <div class="agent-name">
                Trade Execution
            </div>
            <div class="agent-method">
                Deep Q-Network<br>
                BUY<br>
                HOLD / SELL
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a5:

    st.markdown(
        """
        <div class="agent-card">
            <div class="agent-number">
                AGENT 5
            </div>
            <div class="agent-name">
                Portfolio Rebalancing
            </div>
            <div class="agent-method">
                Markowitz MPT<br>
                +<br>
                RL Feedback
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DECISION DISTRIBUTIONS
# =============================================================================

st.divider()

st.subheader(
    "Agent Decision Summary"
)

trend_counts = (
    get_agent2_trend_counts()
)

risk_counts = (
    get_agent3_risk_counts()
)

action_counts = (
    get_agent4_action_counts()
)


trend_df = pd.DataFrame(
    {
        "Trend Class":
            list(
                trend_counts.keys()
            ),

        "Stocks":
            list(
                trend_counts.values()
            ),
    }
)


risk_df = pd.DataFrame(
    {
        "Risk Level":
            list(
                risk_counts.keys()
            ),

        "Stocks":
            list(
                risk_counts.values()
            ),
    }
)


action_df = pd.DataFrame(
    {
        "Trade Action":
            list(
                action_counts.keys()
            ),

        "Stocks":
            list(
                action_counts.values()
            ),
    }
)


chart1, chart2, chart3 = (
    st.columns(
        3
    )
)


with chart1:

    st.markdown(
        "**Agent 2 — Trend Classification**"
    )

    fig = px.bar(
        trend_df,
        x="Trend Class",
        y="Stocks",
        text="Stocks",
    )

    fig.update_layout(
        height=330,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with chart2:

    st.markdown(
        "**Agent 3 — Risk Classification**"
    )

    fig = px.bar(
        risk_df,
        x="Risk Level",
        y="Stocks",
        text="Stocks",
    )

    fig.update_layout(
        height=330,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with chart3:

    st.markdown(
        "**Agent 4 — DQN Decisions**"
    )

    fig = px.bar(
        action_df,
        x="Trade Action",
        y="Stocks",
        text="Stocks",
    )

    fig.update_layout(
        height=330,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# FINAL PORTFOLIO
# =============================================================================

st.divider()

st.subheader(
    "Agent 5 — Final Portfolio Allocation"
)


portfolio = (
    agent5.copy()
)


if (
    "final_weight"
    in portfolio.columns
):

    portfolio[
        "final_weight"
    ] = pd.to_numeric(
        portfolio[
            "final_weight"
        ],
        errors="coerce",
    )

    portfolio[
        "allocation_percent"
    ] = (
        portfolio[
            "final_weight"
        ]
        *
        100
    )

    portfolio = (
        portfolio.sort_values(
            "allocation_percent",
            ascending=False,
        )
    )


portfolio_chart_col, portfolio_table_col = (
    st.columns(
        [
            1.35,
            1,
        ]
    )
)


with portfolio_chart_col:

    if (
        "allocation_percent"
        in portfolio.columns
    ):

        fig = px.bar(
            portfolio,
            x="symbol",
            y="allocation_percent",
            text=(
                portfolio[
                    "allocation_percent"
                ]
                .map(
                    lambda x:
                    f"{x:.2f}%"
                )
            ),
        )

        fig.update_layout(
            height=450,
            xaxis_title="Stock",
            yaxis_title="Final Allocation (%)",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


with portfolio_table_col:

    display_columns = [
        column
        for column in (
            "symbol",
            "trade_action",
            "final_weight",
            "portfolio_status",
            "final_decision",
        )
        if column in portfolio.columns
    ]

    st.dataframe(
        portfolio[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# ACTIVE VS EXITED
# =============================================================================

active = (
    get_active_portfolio()
)

exited = (
    get_exited_portfolio()
)

active_col, exit_col = (
    st.columns(
        2
    )
)


with active_col:

    st.markdown(
        "### Active Portfolio"
    )

    active_columns = [
        column
        for column in (
            "symbol",
            "trade_action",
            "final_weight",
            "portfolio_status",
        )
        if column in active.columns
    ]

    st.dataframe(
        active[
            active_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


with exit_col:

    st.markdown(
        "### Exited Positions"
    )

    exit_columns = [
        column
        for column in (
            "symbol",
            "trade_action",
            "final_weight",
            "portfolio_status",
        )
        if column in exited.columns
    ]

    st.dataframe(
        exited[
            exit_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# STOCK JOURNEY
# =============================================================================

st.divider()

st.subheader(
    "Stock Journey Through All Five Agents"
)


stock_options = (
    sorted(
        trace[
            "symbol"
        ]
        .astype(str)
        .tolist()
    )
)


selected_stock = st.selectbox(
    "Choose a stock",
    options=stock_options,
)


stock_row = (
    trace[
        trace[
            "symbol"
        ]
        ==
        selected_stock
    ]
    .iloc[
        0
    ]
)


j1, j2, j3, j4, j5 = (
    st.columns(
        5
    )
)


with j1:

    st.metric(
        "Agent 1 Rank",
        int(
            safe_number(
                stock_row.get(
                    "agent1_final_rank",
                    stock_row.get(
                        "agent1_selection_rank",
                        0,
                    ),
                )
            )
        ),
    )

    if (
        "agent1_bgsto_score"
        in trace.columns
    ):

        st.caption(
            "BGSTO Score: "
            f"{safe_number(stock_row.get('agent1_bgsto_score')):.4f}"
        )


with j2:

    probability = safe_number(
        stock_row.get(
            "agent2_top30_probability",
            0,
        )
    )

    st.metric(
        "Agent 2 P(TOP30)",
        f"{probability:.4f}",
    )

    st.caption(
        str(
            stock_row.get(
                "agent2_trend_class",
                "N/A",
            )
        )
    )


with j3:

    st.metric(
        "Agent 3 Risk",
        str(
            stock_row.get(
                "agent3_risk_level",
                "N/A",
            )
        ),
    )

    st.caption(
        "Risk Score: "
        f"{safe_number(stock_row.get('agent3_score')):.4f}"
    )


with j4:

    st.metric(
        "Agent 4 Action",
        str(
            stock_row.get(
                "agent4_trade_action",
                "N/A",
            )
        ),
    )

    if (
        "agent4_q_margin"
        in trace.columns
    ):

        st.caption(
            "Q Margin: "
            f"{safe_number(stock_row.get('agent4_q_margin')):.5f}"
        )


with j5:

    final_weight = safe_number(
        stock_row.get(
            "agent5_final_weight",
            0,
        )
    )

    st.metric(
        "Agent 5 Weight",
        f"{final_weight * 100:.2f}%",
    )

    st.caption(
        str(
            stock_row.get(
                "agent5_portfolio_status",
                "N/A",
            )
        )
    )


# =============================================================================
# FULL TRACE TABLE
# =============================================================================

with st.expander(
    "View complete five-agent trace",
    expanded=False,
):

    important_columns = [
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
            "agent4_trade_action",
            "agent4_q_hold",
            "agent4_q_buy",
            "agent4_q_sell",
            "agent4_q_margin",
            "agent5_final_weight",
            "agent5_final_allocation_percent",
            "agent5_portfolio_status",
        )
        if column in trace.columns
    ]

    st.dataframe(
        trace[
            important_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# PIPELINE EXECUTION TIME
# =============================================================================

st.divider()

st.subheader(
    "Pipeline Runtime"
)


if (
    not stage_timings.empty
    and
    "duration_seconds"
    in stage_timings.columns
):

    runtime = (
        stage_timings.copy()
    )

    runtime[
        "duration_seconds"
    ] = pd.to_numeric(
        runtime[
            "duration_seconds"
        ],
        errors="coerce",
    )

    runtime = (
        runtime.dropna(
            subset=[
                "duration_seconds"
            ]
        )
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
                lambda value:
                f"{value:.2f}s"
            )
        ),
    )

    fig.update_layout(
        height=430,
        xaxis_title="Pipeline Stage",
        yaxis_title="Execution Time (seconds)",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:

    st.info(
        "Stage timing data is currently unavailable."
    )


# =============================================================================
# INTEGRATION STATUS
# =============================================================================

st.divider()

st.subheader(
    "System Integration Validation"
)


integration_groups = (
    status.get(
        "integration_groups",
        {},
    )
)


if integration_groups:

    integration_df = pd.DataFrame(
        [
            {
                "Validation":
                    key.replace(
                        "_",
                        " "
                    ).title(),

                "Status":
                    (
                        "PASS"
                        if value
                        else "FAIL"
                    ),
            }
            for key, value
            in integration_groups.items()
        ]
    )

    st.dataframe(
        integration_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.warning(
        "Integration group results were not found."
    )


# =============================================================================
# SAVED RUNS
# =============================================================================

st.divider()

st.subheader(
    "Saved Full-System Runs"
)


if not saved_runs.empty:

    st.dataframe(
        saved_runs,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No historical pipeline snapshots found."
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Current dashboard displays the validated latest system run. "
    "Portfolio performance values generated by the current Agent-5 "
    "evaluation are diagnostic historical-window measurements and "
    "should not be interpreted as future investment performance."
)