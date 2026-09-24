"""
Agent 2 - Trend Prediction Dashboard

Visualizes:

    Agent 1 selected stocks
        ↓
    Historical feature sequences
        ↓
    Dilated LSTM
        ↓
    Transformer Encoder
        ↓
    Progressive Attention
        ↓
    Softmax
        ↓
    3-seed ensemble
        ↓
    P(TOP30)
        ↓
    Ranking
        ↓
    Agent 3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    get_agent2_trend_counts,
    load_agent2_predictions,
    load_agent3_risk,
    load_full_trace,
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Agent 2 - Trend Prediction",
    page_icon="🧠",
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
        padding: 14px;
        text-align: center;
        min-height: 125px;
        background: rgba(128,128,128,0.04);
    }

    .flow-title {
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 6px;
    }

    .flow-text {
        font-size: 0.84rem;
        opacity: 0.80;
        line-height: 1.45;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


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


def safe_int(
    value,
    default: int = 0,
) -> int:

    try:

        if pd.isna(
            value
        ):

            return default

        return int(
            float(
                value
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def make_numeric(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:

    result = (
        dataframe.copy()
    )

    for column in columns:

        if column in result.columns:

            result[
                column
            ] = pd.to_numeric(
                result[
                    column
                ],
                errors="coerce",
            )

    return result


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(
    show_spinner=False
)
def load_page_data():

    return {
        "predictions":
            load_agent2_predictions(),

        "risk":
            load_agent3_risk(),

        "trace":
            load_full_trace(),
    }


try:

    data = (
        load_page_data()
    )

except Exception as error:

    st.error(
        "Unable to load Agent-2 data."
    )

    st.exception(
        error
    )

    st.stop()


predictions = (
    data[
        "predictions"
    ]
)

risk = (
    data[
        "risk"
    ]
)

trace = (
    data[
        "trace"
    ]
)


# =============================================================================
# IDENTIFY IMPORTANT COLUMNS
# =============================================================================

probability_column = (
    first_existing_column(
        predictions,
        (
            "top30_probability",
            "probability",
            "ensemble_probability",
        ),
    )
)


rank_column = (
    first_existing_column(
        predictions,
        (
            "agent2_rank",
            "rank",
        ),
    )
)


class_column = (
    first_existing_column(
        predictions,
        (
            "trend_class",
            "prediction_class",
            "class",
        ),
    )
)


uncertainty_column = (
    first_existing_column(
        predictions,
        (
            "ensemble_std",
            "prediction_std",
            "uncertainty",
        ),
    )
)


date_column = (
    first_existing_column(
        predictions,
        (
            "prediction_date",
            "date",
        ),
    )
)


seed_42_column = (
    first_existing_column(
        predictions,
        (
            "probability_seed_42",
            "seed_42_probability",
            "seed42_probability",
        ),
    )
)


seed_123_column = (
    first_existing_column(
        predictions,
        (
            "probability_seed_123",
            "seed_123_probability",
            "seed123_probability",
        ),
    )
)


seed_456_column = (
    first_existing_column(
        predictions,
        (
            "probability_seed_456",
            "seed_456_probability",
            "seed456_probability",
        ),
    )
)


numeric_columns = [
    column
    for column in (
        probability_column,
        rank_column,
        uncertainty_column,
        seed_42_column,
        seed_123_column,
        seed_456_column,
    )
    if column is not None
]


predictions = make_numeric(
    predictions,
    numeric_columns,
)


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="agent-title">
        Agent 2 — Trend Prediction Agent
    </div>

    <div class="agent-subtitle">
        Dilated LSTM + Transformer + Progressive Attention with
        three-seed probability ensemble.
    </div>
    """,
    unsafe_allow_html=True,
)


st.info(
    "Agent 2 estimates the probability that each Agent-1 stock belongs "
    "to the stronger relative-return group. It ranks the selected stocks "
    "before passing them to the Risk Management Agent."
)


# =============================================================================
# ARCHITECTURE
# =============================================================================

st.subheader(
    "Neural Architecture"
)


f1, f2, f3, f4, f5, f6 = (
    st.columns(
        6
    )
)


with f1:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Input Sequence
            </div>
            <div class="flow-text">
                60 trading observations<br>
                ×<br>
                engineered features
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with f2:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Dilated LSTM
            </div>
            <div class="flow-text">
                Learns temporal patterns<br>
                across multiple<br>
                time scales
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with f3:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Transformer
            </div>
            <div class="flow-text">
                Multi-head attention<br>
                captures relationships<br>
                across time
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with f4:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Progressive Attention
            </div>
            <div class="flow-text">
                Learns which temporal<br>
                representations are<br>
                most relevant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with f5:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Softmax
            </div>
            <div class="flow-text">
                P(NOT_TOP30)<br>
                and<br>
                P(TOP30)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with f6:

    st.markdown(
        """
        <div class="flow-card">
            <div class="flow-title">
                Ensemble
            </div>
            <div class="flow-text">
                Seed 42<br>
                Seed 123<br>
                Seed 456
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SUMMARY KPIs
# =============================================================================

st.divider()

st.subheader(
    "Current Prediction Summary"
)


prediction_date = "N/A"

if (
    date_column
    and
    not predictions.empty
):

    prediction_date = str(
        predictions[
            date_column
        ]
        .iloc[
            0
        ]
    )


top_stock = "N/A"
top_probability = 0.0


if (
    probability_column
    and
    not predictions.empty
):

    ranked_probability = (
        predictions.sort_values(
            probability_column,
            ascending=False,
        )
    )

    top_stock = str(
        ranked_probability[
            "symbol"
        ]
        .iloc[
            0
        ]
    )

    top_probability = safe_float(
        ranked_probability[
            probability_column
        ]
        .iloc[
            0
        ]
    )


average_probability = 0.0

if probability_column:

    average_probability = safe_float(
        predictions[
            probability_column
        ]
        .mean()
    )


average_uncertainty = 0.0

if uncertainty_column:

    average_uncertainty = safe_float(
        predictions[
            uncertainty_column
        ]
        .mean()
    )


trend_counts = (
    get_agent2_trend_counts()
)


k1, k2, k3, k4, k5, k6 = (
    st.columns(
        6
    )
)


with k1:

    st.metric(
        "Stocks Evaluated",
        len(
            predictions
        ),
    )


with k2:

    st.metric(
        "Prediction Date",
        prediction_date,
    )


with k3:

    st.metric(
        "TOP30 Stocks",
        trend_counts.get(
            "TOP30",
            0,
        ),
    )


with k4:

    st.metric(
        "Top Candidate",
        top_stock,
    )


with k5:

    st.metric(
        "Top P(TOP30)",
        f"{top_probability:.4f}",
    )


with k6:

    st.metric(
        "Average Probability",
        f"{average_probability:.4f}",
    )


if uncertainty_column:

    st.caption(
        f"Average ensemble disagreement: "
        f"{average_uncertainty:.5f}"
    )


# =============================================================================
# PROBABILITY RANKING
# =============================================================================

st.divider()

st.subheader(
    "P(TOP30) Ranking"
)


if probability_column:

    ranking_data = (
        predictions.copy()
    )

    ranking_data = (
        ranking_data.sort_values(
            probability_column,
            ascending=True,
        )
    )

    ranking_data[
        "probability_percent"
    ] = (
        ranking_data[
            probability_column
        ]
        *
        100
    )


    chart_col, table_col = (
        st.columns(
            [
                1.3,
                1,
            ]
        )
    )


    with chart_col:

        color_argument = (
            class_column
            if class_column
            else None
        )

        fig = px.bar(
            ranking_data,
            x="probability_percent",
            y="symbol",
            orientation="h",
            color=color_argument,
            text=(
                ranking_data[
                    "probability_percent"
                ]
                .map(
                    lambda value:
                    f"{value:.2f}%"
                )
            ),
            title="Estimated TOP30 Probability",
        )

        fig.add_vline(
            x=50,
            line_dash="dash",
            annotation_text="0.50 threshold",
        )

        fig.update_layout(
            height=520,
            xaxis_title="P(TOP30) (%)",
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


    with table_col:

        display_columns = [
            column
            for column in (
                rank_column,
                "symbol",
                probability_column,
                uncertainty_column,
                class_column,
            )
            if (
                column is not None
                and
                column in predictions.columns
            )
        ]

        display_data = (
            predictions.copy()
        )

        if rank_column:

            display_data = (
                display_data.sort_values(
                    rank_column,
                    ascending=True,
                )
            )

        st.dataframe(
            display_data[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
            height=520,
        )


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

st.divider()

st.subheader(
    "Trend Classification"
)


trend_dataframe = pd.DataFrame(
    {
        "Class":
            list(
                trend_counts.keys()
            ),

        "Stocks":
            list(
                trend_counts.values()
            ),
    }
)


class_chart, class_table = (
    st.columns(
        [
            1,
            1
        ]
    )
)


with class_chart:

    fig = px.pie(
        trend_dataframe,
        names="Class",
        values="Stocks",
        hole=0.48,
        title="TOP30 vs NOT_TOP30",
    )

    fig.update_layout(
        height=400,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with class_table:

    st.dataframe(
        trend_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Interpretation**

        `TOP30` means the model estimates the stock belongs to the
        stronger relative-return class for the prediction horizon.

        `NOT_TOP30` means it falls outside that model-defined stronger
        group.

        Agent 2 does not generate BUY / HOLD / SELL actions.
        """
    )


# =============================================================================
# THREE-SEED ENSEMBLE
# =============================================================================

st.divider()

st.subheader(
    "Three-Seed Ensemble"
)


seed_columns = [
    column
    for column in (
        seed_42_column,
        seed_123_column,
        seed_456_column,
    )
    if column is not None
]


if len(
    seed_columns
) >= 2:

    seed_data = (
        predictions[
            [
                "symbol",
                *seed_columns,
            ]
        ]
        .copy()
    )

    rename_map = {}

    if seed_42_column:

        rename_map[
            seed_42_column
        ] = "Seed 42"

    if seed_123_column:

        rename_map[
            seed_123_column
        ] = "Seed 123"

    if seed_456_column:

        rename_map[
            seed_456_column
        ] = "Seed 456"


    seed_data = seed_data.rename(
        columns=rename_map
    )


    melted = seed_data.melt(
        id_vars=[
            "symbol"
        ],
        var_name="Model",
        value_name="P(TOP30)",
    )


    fig = px.bar(
        melted,
        x="symbol",
        y="P(TOP30)",
        color="Model",
        barmode="group",
        title=(
            "Prediction Variation Across "
            "Random-Seed Models"
        ),
    )

    fig.update_layout(
        height=470,
        xaxis_title="Stock",
        yaxis_title="P(TOP30)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.caption(
        "The final probability is obtained by combining the outputs "
        "of the separately initialized neural models."
    )


else:

    st.info(
        "Individual seed probabilities are not present in the current "
        "prediction output. The final ensemble prediction is still "
        "available."
    )


# =============================================================================
# ENSEMBLE UNCERTAINTY
# =============================================================================

st.divider()

st.subheader(
    "Prediction Uncertainty"
)


if uncertainty_column:

    uncertainty = (
        predictions[
            [
                "symbol",
                uncertainty_column,
                probability_column,
            ]
        ]
        .copy()
    )

    uncertainty[
        uncertainty_column
    ] = pd.to_numeric(
        uncertainty[
            uncertainty_column
        ],
        errors="coerce",
    )

    uncertainty = (
        uncertainty.sort_values(
            uncertainty_column,
            ascending=False,
        )
    )


    u1, u2 = (
        st.columns(
            [
                1.25,
                1,
            ]
        )
    )


    with u1:

        fig = px.bar(
            uncertainty,
            x="symbol",
            y=uncertainty_column,
            text=(
                uncertainty[
                    uncertainty_column
                ]
                .map(
                    lambda value:
                    f"{value:.4f}"
                )
            ),
            title="Ensemble Disagreement",
        )

        fig.update_layout(
            height=420,
            xaxis_title="Stock",
            yaxis_title="Ensemble Standard Deviation",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    with u2:

        highest_uncertainty = (
            uncertainty.iloc[
                0
            ]
        )

        lowest_uncertainty = (
            uncertainty.iloc[
                -1
            ]
        )

        st.metric(
            "Highest Model Disagreement",
            str(
                highest_uncertainty[
                    "symbol"
                ]
            ),
            f"{safe_float(highest_uncertainty[uncertainty_column]):.5f}",
        )

        st.metric(
            "Lowest Model Disagreement",
            str(
                lowest_uncertainty[
                    "symbol"
                ]
            ),
            f"{safe_float(lowest_uncertainty[uncertainty_column]):.5f}",
        )

        st.markdown(
            """
            A larger ensemble standard deviation means the independently
            initialized models disagree more strongly.

            This represents **model disagreement**, not market volatility
            or portfolio financial risk.
            """
        )


else:

    st.info(
        "No ensemble uncertainty column is available."
    )


# =============================================================================
# PROBABILITY VS UNCERTAINTY
# =============================================================================

if (
    probability_column
    and
    uncertainty_column
):

    st.divider()

    st.subheader(
        "Signal Strength vs Model Disagreement"
    )


    scatter = (
        predictions.copy()
    )


    fig = px.scatter(
        scatter,
        x=probability_column,
        y=uncertainty_column,
        text="symbol",
        color=(
            class_column
            if class_column
            else None
        ),
        size=(
            uncertainty_column
        ),
        title=(
            "P(TOP30) Compared with "
            "Ensemble Uncertainty"
        ),
    )

    fig.update_traces(
        textposition="top center"
    )

    fig.update_layout(
        height=500,
        xaxis_title="P(TOP30)",
        yaxis_title="Ensemble Standard Deviation",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# STOCK INSPECTOR
# =============================================================================

st.divider()

st.subheader(
    "Inspect a Prediction"
)


stock_options = sorted(
    predictions[
        "symbol"
    ]
    .astype(str)
    .tolist()
)


stock = st.selectbox(
    "Select stock",
    stock_options,
)


stock_prediction = (
    predictions[
        predictions[
            "symbol"
        ]
        ==
        stock
    ]
)


if not stock_prediction.empty:

    row = (
        stock_prediction.iloc[
            0
        ]
    )


    s1, s2, s3, s4 = (
        st.columns(
            4
        )
    )


    with s1:

        value = (
            safe_int(
                row.get(
                    rank_column
                )
            )
            if rank_column
            else 0
        )

        st.metric(
            "Agent-2 Rank",
            value,
        )


    with s2:

        value = (
            safe_float(
                row.get(
                    probability_column
                )
            )
            if probability_column
            else 0.0
        )

        st.metric(
            "P(TOP30)",
            f"{value:.4f}",
        )


    with s3:

        value = (
            str(
                row.get(
                    class_column,
                    "N/A",
                )
            )
            if class_column
            else "N/A"
        )

        st.metric(
            "Trend Class",
            value,
        )


    with s4:

        value = (
            safe_float(
                row.get(
                    uncertainty_column
                )
            )
            if uncertainty_column
            else 0.0
        )

        st.metric(
            "Ensemble Std",
            f"{value:.5f}",
        )


    # =========================================================================
    # INDIVIDUAL MODEL OUTPUTS
    # =========================================================================

    seed_metric_columns = [
        column
        for column in (
            seed_42_column,
            seed_123_column,
            seed_456_column,
        )
        if column
    ]


    if seed_metric_columns:

        st.markdown(
            "#### Individual Ensemble Models"
        )

        seed_metric_ui = (
            st.columns(
                len(
                    seed_metric_columns
                )
            )
        )

        for index, column in enumerate(
            seed_metric_columns
        ):

            label_map = {
                seed_42_column:
                    "Seed 42",

                seed_123_column:
                    "Seed 123",

                seed_456_column:
                    "Seed 456",
            }

            with seed_metric_ui[
                index
            ]:

                st.metric(
                    label_map.get(
                        column,
                        column,
                    ),
                    (
                        f"{safe_float(row.get(column)):.4f}"
                    ),
                )


    with st.expander(
        "View all Agent-2 output fields"
    ):

        stock_display = (
            stock_prediction.T
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
# AGENT 2 -> AGENT 3 TRANSFER
# =============================================================================

st.divider()

st.subheader(
    "Agent 2 → Agent 3 Transfer"
)


transfer_columns = [
    column
    for column in (
        "symbol",
        "agent2_top30_probability",
        "agent2_ensemble_std",
        "agent2_rank",
        "agent2_trend_class",
        "agent3_score",
        "agent3_rank",
        "agent3_risk_level",
        "agent3_risk_adjusted_weight",
    )
    if column in trace.columns
]


transfer = (
    trace[
        transfer_columns
    ]
    .copy()
)


if (
    "agent2_rank"
    in transfer.columns
):

    transfer = (
        transfer.sort_values(
            "agent2_rank",
            ascending=True,
        )
    )


st.dataframe(
    transfer,
    use_container_width=True,
    hide_index=True,
)


st.success(
    f"{len(transfer)} Agent-2 predictions were successfully "
    "transferred to Agent 3."
)


# =============================================================================
# AGENT 2 RANK VS AGENT 3 RANK
# =============================================================================

if (
    "agent2_rank"
    in trace.columns
    and
    "agent3_rank"
    in trace.columns
):

    st.divider()

    st.subheader(
        "Trend Rank vs Risk-Adjusted Rank"
    )


    comparison = (
        trace[
            [
                "symbol",
                "agent2_rank",
                "agent3_rank",
            ]
        ]
        .copy()
    )


    comparison[
        "agent2_rank"
    ] = pd.to_numeric(
        comparison[
            "agent2_rank"
        ],
        errors="coerce",
    )


    comparison[
        "agent3_rank"
    ] = pd.to_numeric(
        comparison[
            "agent3_rank"
        ],
        errors="coerce",
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=comparison[
                "symbol"
            ],
            y=comparison[
                "agent2_rank"
            ],
            mode="lines+markers",
            name="Agent 2 Trend Rank",
        )
    )


    fig.add_trace(
        go.Scatter(
            x=comparison[
                "symbol"
            ],
            y=comparison[
                "agent3_rank"
            ],
            mode="lines+markers",
            name="Agent 3 Risk Rank",
        )
    )


    fig.update_layout(
        height=450,
        xaxis_title="Stock",
        yaxis_title="Rank",
        yaxis=dict(
            autorange="reversed"
        ),
        title=(
            "How Risk Management Changes "
            "the Trend-Based Ranking"
        ),
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# METHODOLOGY
# =============================================================================

st.divider()


with st.expander(
    "Agent-2 Methodology",
    expanded=False,
):

    st.markdown(
        """
        ### 1. Input sequences

        Agent 2 uses historical engineered stock features arranged into
        sequential windows.

        In the current implementation, the model uses a 60-observation
        historical sequence.

        ### 2. Dilated LSTM

        The LSTM component captures temporal patterns.

        Dilated temporal processing allows the model to learn information
        across different time scales instead of relying only on immediately
        adjacent observations.

        ### 3. Transformer Encoder

        Multi-head self-attention allows different points in the historical
        sequence to interact directly.

        This helps the model capture relationships that may occur across
        longer temporal distances.

        ### 4. Progressive Attention

        Progressive Attention learns how much importance to assign to
        different temporal representations before producing the final
        prediction.

        ### 5. Softmax Classification

        The neural network produces two class probabilities:

        - NOT_TOP30
        - TOP30

        The dashboard primarily displays:

        `P(TOP30)`

        ### 6. Relative prediction target

        Agent 2 ranks stocks by the estimated probability of belonging to
        the stronger relative-return class for the future prediction horizon.

        ### 7. Ensemble

        Three independently initialized models are combined:

        - Seed 42
        - Seed 123
        - Seed 456

        Their probability outputs are averaged to reduce dependence on one
        random neural-network initialization.

        ### 8. Ensemble standard deviation

        `ensemble_std` measures disagreement between the neural models.

        It is a model uncertainty indicator.

        It should not be interpreted as financial volatility or Value at Risk.

        ### 9. Downstream role

        Agent 2 does not directly issue a trading command.

        Its trend prediction becomes one input to Agent 3, which combines the
        trend signal with financial risk information.
        """
    )


# =============================================================================
# COMPLETE OUTPUT
# =============================================================================

with st.expander(
    "View complete Agent-2 prediction table"
):

    display = (
        predictions.copy()
    )

    if rank_column:

        display = (
            display.sort_values(
                rank_column,
                ascending=True,
            )
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Agent-2 probabilities are predictive model outputs, not guaranteed "
    "future returns. Final trading decisions and portfolio allocation are "
    "handled by downstream agents."
)