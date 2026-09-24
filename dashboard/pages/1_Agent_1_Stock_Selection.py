"""
Agent 1 - Stock Selection Dashboard

Visualizes the complete Agent-1 workflow:

    Market Data
        ↓
    Feature Engineering
        ↓
    Decision Tree
        ↓
    BUY Candidates
        ↓
    BGSTO Optimization
        ↓
    Selected Stocks
        ↓
    Final Ranking
        ↓
    Agent 2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# PROJECT IMPORT SETUP
# =============================================================================

PAGE_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    PAGE_DIR.parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from dashboard.utils.data_loader import (
    load_agent1_bgsto_history,
    load_agent1_dt_metrics,
    load_agent1_feature_importance,
    load_agent1_rankings,
    load_agent1_selected,
    load_full_trace,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Agent 1 - Stock Selection",
    page_icon="🔎",
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
        margin-bottom: 5px;
    }

    .flow-text {
        font-size: 0.85rem;
        opacity: 0.80;
        line-height: 1.4;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA HELPERS
# =============================================================================

def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


def numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:

    return pd.to_numeric(
        dataframe[
            column
        ],
        errors="coerce",
    )


def safe_float(
    value,
    default: float = 0.0,
) -> float:

    try:

        if pd.isna(
            value
        ):

            return default

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(
    show_spinner=False
)
def load_page_data():

    return {
        "selected":
            load_agent1_selected(),

        "rankings":
            load_agent1_rankings(),

        "metrics":
            load_agent1_dt_metrics(),

        "feature_importance":
            load_agent1_feature_importance(),

        "bgsto_history":
            load_agent1_bgsto_history(),

        "trace":
            load_full_trace(),
    }


try:

    data = (
        load_page_data()
    )

except Exception as error:

    st.error(
        "Unable to load Agent-1 data."
    )

    st.exception(
        error
    )

    st.stop()


selected = (
    data[
        "selected"
    ]
)

rankings = (
    data[
        "rankings"
    ]
)

metrics = (
    data[
        "metrics"
    ]
)

feature_importance = (
    data[
        "feature_importance"
    ]
)

bgsto_history = (
    data[
        "bgsto_history"
    ]
)

trace = (
    data[
        "trace"
    ]
)


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="agent-title">
        Agent 1 — Stock Selection Agent
    </div>

    <div class="agent-subtitle">
        Decision Tree candidate filtering followed by BGSTO stock-set
        optimization and final ranking.
    </div>
    """,
    unsafe_allow_html=True,
)


st.info(
    "Agent 1 reduces the broad stock universe to a focused set of "
    "10 stocks that are transferred to the Trend Prediction Agent."
)


# =============================================================================
# WORKFLOW
# =============================================================================

st.subheader(
    "Agent 1 Workflow"
)


c1, c2, c3, c4, c5 = (
    st.columns(
        5
    )
)


with c1:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Historical Data
            </div>
            <div class="flow-text">
                NIFTY-500-derived<br>
                OHLCV market data
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Features
            </div>
            <div class="flow-text">
                Returns<br>
                Trend<br>
                Volatility<br>
                Momentum
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Decision Tree
            </div>
            <div class="flow-text">
                Predict BUY probability<br>
                and remove weak candidates
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                BGSTO
            </div>
            <div class="flow-text">
                Binary Genetic<br>
                Siberian Tiger<br>
                Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with c5:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Final Selection
            </div>
            <div class="flow-text">
                10 stocks<br>
                ranked for<br>
                Agent 2
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TOP KPIs
# =============================================================================

st.divider()

st.subheader(
    "Current Selection Summary"
)


selection_date_column = (
    first_existing_column(
        selected,
        (
            "selection_date",
            "date",
        ),
    )
)


buy_probability_column = (
    first_existing_column(
        selected,
        (
            "buy_probability",
            "prediction_probability",
            "probability",
        ),
    )
)


bgsto_score_column = (
    first_existing_column(
        selected,
        (
            "bgsto_stock_score",
            "bgsto_score",
            "score",
        ),
    )
)


selection_rank_column = (
    first_existing_column(
        selected,
        (
            "selection_rank",
            "bgsto_rank",
            "rank",
        ),
    )
)


selection_date = "N/A"

if (
    selection_date_column
    is not None
    and
    not selected.empty
):

    selection_date = str(
        selected[
            selection_date_column
        ]
        .iloc[
            0
        ]
    )


average_buy_probability = 0.0

if buy_probability_column:

    average_buy_probability = (
        numeric_column(
            selected,
            buy_probability_column,
        )
        .mean()
    )


average_bgsto_score = 0.0

if bgsto_score_column:

    average_bgsto_score = (
        numeric_column(
            selected,
            bgsto_score_column,
        )
        .mean()
    )


top_stock = "N/A"

if not selected.empty:

    if selection_rank_column:

        temporary = (
            selected.copy()
        )

        temporary[
            selection_rank_column
        ] = pd.to_numeric(
            temporary[
                selection_rank_column
            ],
            errors="coerce",
        )

        temporary = temporary.sort_values(
            selection_rank_column,
            ascending=True,
        )

        top_stock = str(
            temporary[
                "symbol"
            ]
            .iloc[
                0
            ]
        )

    else:

        top_stock = str(
            selected[
                "symbol"
            ]
            .iloc[
                0
            ]
        )


k1, k2, k3, k4, k5 = (
    st.columns(
        5
    )
)


with k1:

    st.metric(
        "Selected Stocks",
        len(
            selected
        ),
    )


with k2:

    st.metric(
        "Selection Date",
        selection_date,
    )


with k3:

    st.metric(
        "Top Ranked Stock",
        top_stock,
    )


with k4:

    st.metric(
        "Average BUY Probability",
        f"{average_buy_probability:.4f}",
    )


with k5:

    st.metric(
        "Average BGSTO Score",
        f"{average_bgsto_score:.4f}",
    )


# =============================================================================
# SELECTED STOCKS
# =============================================================================

st.divider()

st.subheader(
    "Selected Stocks"
)


chart_col, table_col = (
    st.columns(
        [
            1.15,
            1,
        ]
    )
)


with chart_col:

    if (
        bgsto_score_column
        is not None
    ):

        chart_data = (
            selected.copy()
        )

        chart_data[
            bgsto_score_column
        ] = pd.to_numeric(
            chart_data[
                bgsto_score_column
            ],
            errors="coerce",
        )

        chart_data = (
            chart_data.sort_values(
                bgsto_score_column,
                ascending=True,
            )
        )

        fig = px.bar(
            chart_data,
            x=bgsto_score_column,
            y="symbol",
            orientation="h",
            text=(
                chart_data[
                    bgsto_score_column
                ]
                .map(
                    lambda value:
                    f"{value:.4f}"
                )
            ),
            title="BGSTO Stock Score",
        )

        fig.update_layout(
            height=520,
            xaxis_title="BGSTO Score",
            yaxis_title="Stock",
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    elif (
        buy_probability_column
        is not None
    ):

        chart_data = (
            selected.copy()
        )

        fig = px.bar(
            chart_data,
            x="symbol",
            y=buy_probability_column,
            title="Decision Tree BUY Probability",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


with table_col:

    selected_display_columns = [
        column
        for column in (
            selection_rank_column,
            "symbol",
            buy_probability_column,
            bgsto_score_column,
            "return_21d",
            "return_63d",
            "volatility_60d",
            "sharpe_252d",
        )
        if (
            column is not None
            and
            column in selected.columns
        )
    ]

    st.dataframe(
        selected[
            selected_display_columns
        ],
        use_container_width=True,
        hide_index=True,
        height=520,
    )


# =============================================================================
# DECISION TREE PERFORMANCE
# =============================================================================

st.divider()

st.subheader(
    "Decision Tree Performance"
)


if metrics.empty:

    st.warning(
        "Decision Tree metrics file is not available."
    )

else:

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
    )

    split_column = first_existing_column(
        metrics,
        (
            "split",
            "dataset",
            "data_split",
            "period",
            "set",
        ),
    )

    metric_candidates = [
        column
        for column in (
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1_score",
            "f1",
            "roc_auc",
            "auc",
        )
        if column in metrics.columns
    ]

    if metric_candidates:

        metric_frame = (
            metrics.copy()
        )

        for column in metric_candidates:

            metric_frame[
                column
            ] = pd.to_numeric(
                metric_frame[
                    column
                ],
                errors="coerce",
            )

        if split_column:

            melted = metric_frame.melt(
                id_vars=[
                    split_column
                ],
                value_vars=metric_candidates,
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                melted,
                x=split_column,
                y="Score",
                color="Metric",
                barmode="group",
                title=(
                    "Decision Tree Performance "
                    "Across Data Splits"
                ),
            )

        else:

            metric_values = []

            for column in metric_candidates:

                value = (
                    metric_frame[
                        column
                    ]
                    .iloc[
                        0
                    ]
                )

                metric_values.append(
                    {
                        "Metric":
                            column,

                        "Score":
                            value,
                    }
                )

            fig = px.bar(
                pd.DataFrame(
                    metric_values
                ),
                x="Metric",
                y="Score",
                title="Decision Tree Metrics",
            )

        fig.update_layout(
            height=420,
            yaxis_title="Score",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

st.divider()

st.subheader(
    "Decision Tree Feature Importance"
)


if feature_importance.empty:

    st.warning(
        "Feature-importance output is not available."
    )

else:

    feature_column = first_existing_column(
        feature_importance,
        (
            "feature",
            "feature_name",
            "variable",
        ),
    )

    importance_column = (
        first_existing_column(
            feature_importance,
            (
                "importance",
                "feature_importance",
                "score",
            ),
        )
    )

    if (
        feature_column
        and
        importance_column
    ):

        importance = (
            feature_importance.copy()
        )

        importance[
            importance_column
        ] = pd.to_numeric(
            importance[
                importance_column
            ],
            errors="coerce",
        )

        importance = (
            importance.dropna(
                subset=[
                    importance_column
                ]
            )
            .sort_values(
                importance_column,
                ascending=False,
            )
        )

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=min(
                35,
                max(
                    5,
                    len(
                        importance
                    ),
                ),
            ),
            value=min(
                15,
                len(
                    importance
                ),
            ),
        )

        top_features = (
            importance.head(
                top_n
            )
            .sort_values(
                importance_column,
                ascending=True,
            )
        )

        fig = px.bar(
            top_features,
            x=importance_column,
            y=feature_column,
            orientation="h",
            text=(
                top_features[
                    importance_column
                ]
                .map(
                    lambda value:
                    f"{value:.4f}"
                )
            ),
            title=(
                f"Top {top_n} Decision Tree Features"
            ),
        )

        fig.update_layout(
            height=max(
                400,
                top_n * 30,
            ),
            xaxis_title="Importance",
            yaxis_title="Feature",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        with st.expander(
            "View complete feature-importance table"
        ):

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# BGSTO OPTIMIZATION HISTORY
# =============================================================================

st.divider()

st.subheader(
    "BGSTO Optimization"
)


if bgsto_history.empty:

    st.warning(
        "BGSTO optimization-history file is not available."
    )

else:

    iteration_column = (
        first_existing_column(
            bgsto_history,
            (
                "iteration",
                "generation",
                "iter",
            ),
        )
    )

    fitness_column = (
        first_existing_column(
            bgsto_history,
            (
                "best_fitness",
                "fitness",
                "global_best_fitness",
            ),
        )
    )

    selected_count_column = (
        first_existing_column(
            bgsto_history,
            (
                "selected_stocks",
                "selected_count",
                "n_selected",
            ),
        )
    )

    bg1, bg2 = (
        st.columns(
            2
        )
    )


    with bg1:

        if (
            iteration_column
            and
            fitness_column
        ):

            history = (
                bgsto_history.copy()
            )

            history[
                iteration_column
            ] = pd.to_numeric(
                history[
                    iteration_column
                ],
                errors="coerce",
            )

            history[
                fitness_column
            ] = pd.to_numeric(
                history[
                    fitness_column
                ],
                errors="coerce",
            )

            fig = px.line(
                history,
                x=iteration_column,
                y=fitness_column,
                markers=True,
                title="BGSTO Fitness Convergence",
            )

            fig.update_layout(
                height=420,
                xaxis_title="Iteration",
                yaxis_title="Best Fitness",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    with bg2:

        if (
            iteration_column
            and
            selected_count_column
        ):

            history = (
                bgsto_history.copy()
            )

            history[
                selected_count_column
            ] = pd.to_numeric(
                history[
                    selected_count_column
                ],
                errors="coerce",
            )

            fig = px.line(
                history,
                x=iteration_column,
                y=selected_count_column,
                markers=True,
                title=(
                    "Number of Stocks Selected "
                    "During Optimization"
                ),
            )

            fig.update_layout(
                height=420,
                xaxis_title="Iteration",
                yaxis_title="Selected Stocks",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    with st.expander(
        "View BGSTO optimization history"
    ):

        st.dataframe(
            bgsto_history,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# FINAL AGENT 1 RANKING
# =============================================================================

st.divider()

st.subheader(
    "Final Agent-1 Ranking"
)


if rankings.empty:

    st.warning(
        "Agent-1 ranking output is unavailable."
    )

else:

    final_rank_column = (
        first_existing_column(
            rankings,
            (
                "final_rank",
                "selection_rank",
                "rank",
            ),
        )
    )

    final_score_column = (
        first_existing_column(
            rankings,
            (
                "final_stock_score",
                "final_score",
                "stock_score",
                "score",
            ),
        )
    )

    ranking_data = (
        rankings.copy()
    )

    if final_rank_column:

        ranking_data[
            final_rank_column
        ] = pd.to_numeric(
            ranking_data[
                final_rank_column
            ],
            errors="coerce",
        )

        ranking_data = ranking_data.sort_values(
            final_rank_column,
            ascending=True,
        )


    rank_chart, rank_table = (
        st.columns(
            [
                1.15,
                1,
            ]
        )
    )


    with rank_chart:

        if final_score_column:

            ranking_data[
                final_score_column
            ] = pd.to_numeric(
                ranking_data[
                    final_score_column
                ],
                errors="coerce",
            )

            graph_data = (
                ranking_data.sort_values(
                    final_score_column,
                    ascending=True,
                )
            )

            fig = px.bar(
                graph_data,
                x=final_score_column,
                y="symbol",
                orientation="h",
                text=(
                    graph_data[
                        final_score_column
                    ]
                    .map(
                        lambda value:
                        f"{value:.2f}"
                    )
                ),
                title="Final Agent-1 Stock Score",
            )

            fig.update_layout(
                height=500,
                xaxis_title="Final Score",
                yaxis_title="Stock",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    with rank_table:

        st.dataframe(
            ranking_data,
            use_container_width=True,
            hide_index=True,
            height=500,
        )


# =============================================================================
# TRANSFER TO AGENT 2
# =============================================================================

st.divider()

st.subheader(
    "Agent 1 → Agent 2 Transfer"
)


agent1_trace_columns = [
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
    )
    if column in trace.columns
]


transfer = (
    trace[
        agent1_trace_columns
    ]
    .copy()
)


if (
    "agent1_final_rank"
    in transfer.columns
):

    transfer = (
        transfer.sort_values(
            "agent1_final_rank",
            ascending=True,
        )
    )


st.dataframe(
    transfer,
    use_container_width=True,
    hide_index=True,
)


st.success(
    f"{len(transfer)} Agent-1 selected stocks were successfully "
    "transferred to Agent 2."
)


# =============================================================================
# STOCK INSPECTOR
# =============================================================================

st.divider()

st.subheader(
    "Inspect an Agent-1 Stock"
)


stock = st.selectbox(
    "Select stock",
    sorted(
        selected[
            "symbol"
        ]
        .astype(str)
        .tolist()
    ),
)


stock_data = (
    selected[
        selected[
            "symbol"
        ]
        ==
        stock
    ]
)


if not stock_data.empty:

    row = (
        stock_data.iloc[
            0
        ]
    )

    p1, p2, p3, p4 = (
        st.columns(
            4
        )
    )


    with p1:

        rank_value = (
            row.get(
                selection_rank_column,
                "N/A",
            )
            if selection_rank_column
            else "N/A"
        )

        st.metric(
            "Selection Rank",
            rank_value,
        )


    with p2:

        value = (
            safe_float(
                row.get(
                    buy_probability_column
                )
            )
            if buy_probability_column
            else 0.0
        )

        st.metric(
            "BUY Probability",
            f"{value:.4f}",
        )


    with p3:

        value = (
            safe_float(
                row.get(
                    bgsto_score_column
                )
            )
            if bgsto_score_column
            else 0.0
        )

        st.metric(
            "BGSTO Score",
            f"{value:.4f}",
        )


    with p4:

        trace_stock = (
            trace[
                trace[
                    "symbol"
                ]
                ==
                stock
            ]
        )

        if (
            not trace_stock.empty
            and
            "agent1_final_rank"
            in trace_stock.columns
        ):

            final_rank = (
                trace_stock[
                    "agent1_final_rank"
                ]
                .iloc[
                    0
                ]
            )

        else:

            final_rank = "N/A"

        st.metric(
            "Final Agent-1 Rank",
            final_rank,
        )


    with st.expander(
        "View all available Agent-1 fields"
    ):

        stock_display = (
            stock_data.T
            .reset_index()
        )

        stock_display.columns = [
            "Field",
            "Value",
        ]

        st.dataframe(
            stock_display,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()

with st.expander(
    "Agent-1 Methodology"
):

    st.markdown(
        """
        ### Stage 1 — Decision Tree

        The Decision Tree uses engineered market features to estimate a
        stock's BUY probability. Features include price behaviour,
        momentum, volatility, moving averages, returns, drawdown,
        Sharpe-related measures and trading-volume information.

        ### Stage 2 — Candidate Filtering

        Stocks predicted as BUY candidates form the candidate universe
        for the optimization stage.

        ### Stage 3 — BGSTO

        Binary Genetic Siberian Tiger Optimization searches combinations
        of candidate stocks rather than ranking each stock independently.

        Its purpose is to identify a stronger subset while considering
        the stock-selection objective used by the project.

        ### Stage 4 — Final Selection

        The resulting selected stocks are ranked and written to the
        Agent-1 output files.

        ### Stage 5 — Transfer

        The selected stock universe is transferred unchanged to
        Agent 2 for neural trend prediction.
        """
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Agent 1 provides the stock universe for downstream analysis. "
    "Its score is a selection signal and is not itself a final "
    "BUY/SELL recommendation; trading decisions are produced later "
    "by Agent 4."
)