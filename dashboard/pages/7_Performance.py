"""
Portfolio Performance Dashboard

Displays the current Agent-5 historical diagnostic evaluation:

- Current portfolio
- Equal-weight benchmark
- Final Agent-5 portfolio

Metrics:
- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Historical VaR
- Effective assets
- Turnover

Important:
These are diagnostic historical-window values, not guaranteed
future investment performance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
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
    load_agent5_allocation_comparison,
    load_agent5_evaluation,
    load_agent5_portfolio,
    load_full_trace,
    load_stage_timings,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Portfolio Performance",
    page_icon="📊",
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

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(
    value,
    default: Optional[float] = None,
):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def normalize_key(
    key: str,
) -> str:

    return (
        str(key)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "percent")
    )


def find_recursive(
    data: Any,
    names: tuple[str, ...],
):
    """
    Recursively search nested JSON for one of several key names.
    """

    normalized_names = {
        normalize_key(name)
        for name in names
    }

    if isinstance(
        data,
        dict,
    ):

        for key, value in data.items():

            if normalize_key(
                key
            ) in normalized_names:

                return value


        for value in data.values():

            result = find_recursive(
                value,
                names,
            )

            if result is not None:
                return result


    if isinstance(
        data,
        list,
    ):

        for value in data:

            result = find_recursive(
                value,
                names,
            )

            if result is not None:
                return result


    return None


def find_section(
    data: Dict[str, Any],
    possible_names: tuple[str, ...],
) -> Dict[str, Any]:

    normalized_names = {
        normalize_key(name)
        for name in possible_names
    }


    for key, value in data.items():

        if (
            normalize_key(
                key
            )
            in normalized_names
            and
            isinstance(
                value,
                dict,
            )
        ):

            return value


    for value in data.values():

        if isinstance(
            value,
            dict,
        ):

            found = find_section(
                value,
                possible_names,
            )

            if found:
                return found


    return {}


def metric_value(
    section: Dict[str, Any],
    names: tuple[str, ...],
):
    return find_recursive(
        section,
        names,
    )


# =============================================================================
# DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_page_data():

    return {
        "evaluation":
            load_agent5_evaluation(),

        "allocation_comparison":
            load_agent5_allocation_comparison(),

        "portfolio":
            load_agent5_portfolio(),

        "trace":
            load_full_trace(),

        "timings":
            load_stage_timings(),
    }


try:

    data = load_page_data()

except Exception as error:

    st.error(
        "Unable to load performance data."
    )

    st.exception(
        error
    )

    st.stop()


evaluation = data["evaluation"]
comparison = data["allocation_comparison"]
portfolio = data["portfolio"]
trace = data["trace"]
timings = data["timings"]


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="page-title">
        Portfolio Performance
    </div>

    <div class="page-subtitle">
        Historical diagnostic evaluation of the final Agent-5 portfolio.
    </div>
    """,
    unsafe_allow_html=True,
)


st.warning(
    "Performance metrics on this page are computed from the historical "
    "evaluation/estimation window used by the current implementation. "
    "They are not future-return forecasts or guaranteed investment results."
)


# =============================================================================
# FIND EVALUATION SECTIONS
# =============================================================================

current_section = find_section(
    evaluation,
    (
        "current_portfolio",
        "current_metrics",
        "current",
    ),
)


benchmark_section = find_section(
    evaluation,
    (
        "equal_weight",
        "equal_weight_portfolio",
        "benchmark",
        "benchmark_metrics",
    ),
)


final_section = find_section(
    evaluation,
    (
        "final_portfolio",
        "final_agent5",
        "final_agent_5",
        "final_metrics",
        "target_portfolio",
    ),
)


# =============================================================================
# RAW METRIC EXTRACTION
# =============================================================================

metric_definitions = {
    "Cumulative Return": (
        "cumulative_return",
        "cumulative_return_percent",
    ),

    "Annualized Return": (
        "annualized_return",
        "annual_return",
        "annualized_return_percent",
    ),

    "Annualized Volatility": (
        "annualized_volatility",
        "annual_volatility",
        "volatility",
    ),

    "Sharpe Ratio": (
        "sharpe_ratio",
        "sharpe",
    ),

    "Maximum Drawdown": (
        "maximum_drawdown",
        "max_drawdown",
    ),

    "Historical VaR 95%": (
        "historical_var_95",
        "historical_var",
        "var_95",
    ),

    "Effective Assets": (
        "effective_assets",
    ),
}


rows = []


for metric_name, aliases in metric_definitions.items():

    current_value = metric_value(
        current_section,
        aliases,
    )

    benchmark_value = metric_value(
        benchmark_section,
        aliases,
    )

    final_value = metric_value(
        final_section,
        aliases,
    )


    if (
        current_value is not None
        or
        benchmark_value is not None
        or
        final_value is not None
    ):

        rows.append(
            {
                "Metric":
                    metric_name,

                "Current":
                    current_value,

                "Equal Weight":
                    benchmark_value,

                "Final Agent 5":
                    final_value,
            }
        )


metrics_df = pd.DataFrame(
    rows
)


# =============================================================================
# FALLBACK TO KNOWN REPORT STRUCTURE DISPLAY
# =============================================================================

st.subheader(
    "Evaluation Summary"
)


if not metrics_df.empty:

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Structured metric fields were not automatically detected. "
        "The complete evaluation report is available below."
    )


# =============================================================================
# FINAL PORTFOLIO KPI EXTRACTION
# =============================================================================

def final_metric(
    aliases: tuple[str, ...],
):

    value = metric_value(
        final_section,
        aliases,
    )

    return safe_float(
        value
    )


cum_return = final_metric(
    (
        "cumulative_return",
        "cumulative_return_percent",
    )
)

annual_return = final_metric(
    (
        "annualized_return",
        "annual_return",
        "annualized_return_percent",
    )
)

annual_vol = final_metric(
    (
        "annualized_volatility",
        "annual_volatility",
        "volatility",
    )
)

sharpe = final_metric(
    (
        "sharpe_ratio",
        "sharpe",
    )
)

drawdown = final_metric(
    (
        "maximum_drawdown",
        "max_drawdown",
    )
)

var_95 = final_metric(
    (
        "historical_var_95",
        "historical_var",
        "var_95",
    )
)


# =============================================================================
# METRIC CARDS
# =============================================================================

if any(
    value is not None
    for value in (
        cum_return,
        annual_return,
        annual_vol,
        sharpe,
        drawdown,
        var_95,
    )
):

    st.divider()

    st.subheader(
        "Final Agent-5 Diagnostic Metrics"
    )


    k1, k2, k3 = st.columns(3)
    k4, k5, k6 = st.columns(3)


    with k1:

        st.metric(
            "Cumulative Return",
            (
                f"{cum_return:.4f}"
                if cum_return is not None
                else "N/A"
            ),
        )


    with k2:

        st.metric(
            "Annualized Return",
            (
                f"{annual_return:.4f}"
                if annual_return is not None
                else "N/A"
            ),
        )


    with k3:

        st.metric(
            "Annualized Volatility",
            (
                f"{annual_vol:.4f}"
                if annual_vol is not None
                else "N/A"
            ),
        )


    with k4:

        st.metric(
            "Sharpe Ratio",
            (
                f"{sharpe:.4f}"
                if sharpe is not None
                else "N/A"
            ),
        )


    with k5:

        st.metric(
            "Maximum Drawdown",
            (
                f"{drawdown:.4f}"
                if drawdown is not None
                else "N/A"
            ),
        )


    with k6:

        st.metric(
            "Historical VaR 95%",
            (
                f"{var_95:.4f}"
                if var_95 is not None
                else "N/A"
            ),
        )


# =============================================================================
# METRIC COMPARISON CHART
# =============================================================================

if not metrics_df.empty:

    st.divider()

    st.subheader(
        "Portfolio Metric Comparison"
    )


    numerical = metrics_df.copy()


    for column in (
        "Current",
        "Equal Weight",
        "Final Agent 5",
    ):

        numerical[column] = pd.to_numeric(
            numerical[column],
            errors="coerce",
        )


    selected_metric = st.selectbox(
        "Choose metric to compare",
        numerical[
            "Metric"
        ].tolist(),
    )


    selected_row = numerical[
        numerical[
            "Metric"
        ]
        ==
        selected_metric
    ].iloc[0]


    chart = pd.DataFrame(
        {
            "Portfolio": [
                "Current",
                "Equal Weight",
                "Final Agent 5",
            ],

            "Value": [
                selected_row[
                    "Current"
                ],

                selected_row[
                    "Equal Weight"
                ],

                selected_row[
                    "Final Agent 5"
                ],
            ],
        }
    )


    fig = px.bar(
        chart,
        x="Portfolio",
        y="Value",
        text=(
            chart[
                "Value"
            ]
            .map(
                lambda x:
                    (
                        f"{x:.4f}"
                        if pd.notna(x)
                        else ""
                    )
            )
        ),
        title=selected_metric,
    )


    fig.update_layout(
        height=430,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# FINAL ALLOCATION
# =============================================================================

st.divider()

st.subheader(
    "Final Portfolio Allocation"
)


portfolio = portfolio.copy()


if "final_weight" in portfolio.columns:

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

    portfolio = portfolio.sort_values(
        "allocation_percent",
        ascending=False,
    )


    fig = px.bar(
        portfolio,
        x="symbol",
        y="allocation_percent",
        color=(
            "trade_action"
            if "trade_action"
            in portfolio.columns
            else None
        ),
        text=(
            portfolio[
                "allocation_percent"
            ]
            .map(
                lambda x:
                    f"{x:.2f}%"
            )
        ),
        title="Final Portfolio Weights",
    )


    fig.update_layout(
        height=470,
        xaxis_title="Stock",
        yaxis_title="Allocation (%)",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# ALLOCATION COMPARISON
# =============================================================================

if not comparison.empty:

    st.divider()

    st.subheader(
        "Current vs Equal Weight vs Final Allocation"
    )


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )


    symbol_column = (
        "symbol"
        if "symbol"
        in comparison.columns
        else None
    )


    if symbol_column:

        candidate_columns = [
            column
            for column in comparison.columns
            if (
                column
                !=
                symbol_column
                and
                pd.api.types.is_numeric_dtype(
                    comparison[
                        column
                    ]
                )
            )
        ]


        if candidate_columns:

            melted = comparison.melt(
                id_vars=[
                    symbol_column
                ],
                value_vars=candidate_columns,
                var_name="Allocation",
                value_name="Weight",
            )


            fig = px.bar(
                melted,
                x=symbol_column,
                y="Weight",
                color="Allocation",
                barmode="group",
                title="Portfolio Allocation Comparison",
            )


            fig.update_layout(
                height=500,
            )


            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# =============================================================================
# TURNOVER
# =============================================================================

st.divider()

st.subheader(
    "Portfolio Turnover"
)


turnover = find_recursive(
    evaluation,
    (
        "turnover",
        "current_to_final_turnover",
        "portfolio_turnover",
    ),
)


if turnover is not None:

    st.metric(
        "Current → Final Turnover",
        str(
            turnover
        ),
    )

else:

    st.info(
        "Turnover value was not automatically detected in the JSON report."
    )


# =============================================================================
# PIPELINE RUNTIME
# =============================================================================

st.divider()

st.subheader(
    "System Runtime"
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
        title="Runtime by Pipeline Stage",
    )


    fig.update_layout(
        height=470,
        xaxis_title="Stage",
        yaxis_title="Seconds",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# COMPLETE EVALUATION REPORT
# =============================================================================

st.divider()


with st.expander(
    "View complete Agent-5 evaluation JSON"
):

    if evaluation:

        st.json(
            evaluation
        )

    else:

        st.info(
            "Evaluation JSON is unavailable."
        )


# =============================================================================
# INTERPRETATION
# =============================================================================

with st.expander(
    "How to interpret these metrics"
):

    st.markdown(
        """
        ### Cumulative return

        Total portfolio return over the evaluation window.

        ### Annualized return

        Historical return scaled to an annualized measure.

        ### Annualized volatility

        Annualized variability of portfolio returns.

        ### Sharpe ratio

        Measures historical return relative to volatility.

        ### Maximum drawdown

        Largest peak-to-trough decline observed during the historical
        evaluation period.

        ### Historical VaR

        Historical downside-risk estimate based on the return distribution.

        ### Effective assets

        Measures portfolio diversification based on the concentration
        of portfolio weights.

        ### Turnover

        Measures the amount of portfolio-weight change required to move
        from the current allocation to the final allocation.

        ### Important limitation

        The current metrics are diagnostic values calculated over the
        historical window used by the Agent-5 implementation.

        They are not yet a leakage-free rolling end-to-end backtest of the
        complete five-agent system.
        """
    )


# =============================================================================
# RESEARCH STATUS
# =============================================================================

st.divider()

st.info(
    "Current status: component and end-to-end inference pipeline validated. "
    "A research-grade rolling historical backtest should later rerun Agent 1 "
    "and all downstream agents as-of each historical decision date."
)


st.caption(
    "Historical diagnostic performance should not be interpreted as "
    "guaranteed future investment performance."
)