"""
output_collector.py

Collect the outputs of all five portfolio agents into one unified
stock-level trace.

Purpose
-------
The complete multi-agent system produces separate files:

Agent 1
    selected_stocks.csv / parquet
    rankings.csv / parquet

Agent 2
    trend_predictions.csv / parquet

Agent 3
    risk_assessment.csv / parquet

Agent 4
    trade_decisions.csv / parquet

Agent 5
    rebalanced_portfolio.csv / parquet

This module combines them into:

    reports/system/latest/full_agent_trace.csv
    reports/system/latest/full_agent_trace.parquet

The resulting file will later be especially useful for:

    - Streamlit visualization
    - pipeline debugging
    - stock journey explanation
    - experiment comparison
    - base-paper comparison
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT IMPORT SETUP
# =============================================================================

SYSTEM_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SYSTEM_DIR.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from system import pipeline_config as cfg


# =============================================================================
# REPORT
# =============================================================================

COLLECTOR_REPORT = (
    cfg.SYSTEM_LATEST_DIR
    / "output_collector_summary.json"
)


# =============================================================================
# DISPLAY
# =============================================================================

def separator(
    char: str = "=",
    width: int = 110,
) -> str:

    return char * width


def header(
    title: str,
) -> None:

    print()
    print(
        separator()
    )

    print(
        title
    )

    print(
        separator()
    )


def section(
    title: str,
) -> None:

    print()
    print(
        title
    )

    print(
        "-" * 110
    )


# =============================================================================
# GENERIC DATA HELPERS
# =============================================================================

def path_ready(
    path: Path,
) -> bool:
    """
    File exists and is non-empty.
    """

    try:

        return (
            path.exists()
            and
            path.is_file()
            and
            path.stat().st_size > 0
        )

    except OSError:

        return False


def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize dataframe column names.
    """

    result = dataframe.copy()

    result.columns = [
        str(column)
        .strip()
        .lower()
        for column
        in result.columns
    ]

    return result


def normalize_symbols(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize stock symbols.
    """

    result = dataframe.copy()

    if "symbol" not in result.columns:

        raise ValueError(
            "Dataframe does not contain a 'symbol' column."
        )

    result[
        "symbol"
    ] = (
        result[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return result


def load_dataframe(
    csv_path: Optional[Path],
    parquet_path: Optional[Path],
    label: str,
) -> Tuple[pd.DataFrame, Path, str]:
    """
    Prefer parquet, otherwise CSV.
    """

    if (
        parquet_path is not None
        and
        path_ready(
            parquet_path
        )
    ):

        dataframe = pd.read_parquet(
            parquet_path
        )

        source = parquet_path

        source_format = "PARQUET"

    elif (
        csv_path is not None
        and
        path_ready(
            csv_path
        )
    ):

        dataframe = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        source = csv_path

        source_format = "CSV"

    else:

        raise FileNotFoundError(
            f"{label} output was not found.\n"
            f"CSV     : {csv_path}\n"
            f"Parquet : {parquet_path}"
        )

    dataframe = normalize_columns(
        dataframe
    )

    dataframe = normalize_symbols(
        dataframe
    )

    if dataframe.empty:

        raise ValueError(
            f"{label} dataframe is empty."
        )

    if dataframe[
        "symbol"
    ].duplicated().any():

        duplicates = (
            dataframe.loc[
                dataframe[
                    "symbol"
                ].duplicated(
                    keep=False
                ),
                "symbol",
            ]
            .tolist()
        )

        raise ValueError(
            f"{label} contains duplicate symbols: "
            f"{duplicates}"
        )

    return (
        dataframe,
        source,
        source_format,
    )


# =============================================================================
# OPTIONAL COLUMN HELPERS
# =============================================================================

def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    """
    Return first matching column.
    """

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


def add_column_if_present(
    destination: pd.DataFrame,
    source: pd.DataFrame,
    source_candidates: Iterable[str],
    output_name: str,
) -> pd.DataFrame:
    """
    Copy one optional column into the destination dataframe.

    The first available source column is used.
    """

    column = first_existing_column(
        source,
        source_candidates,
    )

    if column is not None:

        destination[
            output_name
        ] = source[
            column
        ].values

    return destination


def numeric_series(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# =============================================================================
# AGENT 1
# =============================================================================

def load_agent1() -> Tuple[pd.DataFrame, Dict]:
    """
    Build Agent-1 trace dataframe.

    Uses selected stocks as the primary source and supplements it using
    rankings when available.
    """

    selected, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT1_SELECTED_STOCKS_CSV,
            parquet_path=cfg.AGENT1_SELECTED_STOCKS_PARQUET,
            label="Agent 1 selected stocks",
        )
    )

    result = pd.DataFrame(
        {
            "symbol":
                selected[
                    "symbol"
                ]
        }
    )

    # -------------------------------------------------------------------------
    # Selection date
    # -------------------------------------------------------------------------

    result = add_column_if_present(
        destination=result,
        source=selected,
        source_candidates=(
            "selection_date",
            "date",
        ),
        output_name="agent1_selection_date",
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    result = add_column_if_present(
        result,
        selected,
        (
            "selection_rank",
            "bgsto_rank",
            "rank",
        ),
        "agent1_selection_rank",
    )

    # -------------------------------------------------------------------------
    # Decision Tree probability
    # -------------------------------------------------------------------------

    result = add_column_if_present(
        result,
        selected,
        (
            "buy_probability",
            "prediction_probability",
            "probability",
        ),
        "agent1_buy_probability",
    )

    # -------------------------------------------------------------------------
    # BGSTO
    # -------------------------------------------------------------------------

    result = add_column_if_present(
        result,
        selected,
        (
            "bgsto_stock_score",
            "bgsto_score",
            "score",
        ),
        "agent1_bgsto_score",
    )

    # -------------------------------------------------------------------------
    # Feature values useful for visualization
    # -------------------------------------------------------------------------

    result = add_column_if_present(
        result,
        selected,
        (
            "return_21d",
        ),
        "agent1_return_21d",
    )

    result = add_column_if_present(
        result,
        selected,
        (
            "return_63d",
        ),
        "agent1_return_63d",
    )

    result = add_column_if_present(
        result,
        selected,
        (
            "volatility_60d",
        ),
        "agent1_volatility_60d",
    )

    result = add_column_if_present(
        result,
        selected,
        (
            "sharpe_252d",
        ),
        "agent1_sharpe_252d",
    )

    # -------------------------------------------------------------------------
    # Final ranking output
    # -------------------------------------------------------------------------

    rankings_source = None

    rankings_format = None

    try:

        rankings, rankings_source, rankings_format = (
            load_dataframe(
                csv_path=cfg.AGENT1_RANKINGS_CSV,
                parquet_path=cfg.AGENT1_RANKINGS_PARQUET,
                label="Agent 1 rankings",
            )
        )

        ranking_columns = [
            "symbol",
        ]

        optional_ranking_columns = (
            "final_rank",
            "final_stock_score",
            "score_category",
        )

        for column in optional_ranking_columns:

            if column in rankings.columns:

                ranking_columns.append(
                    column
                )

        rankings = rankings[
            ranking_columns
        ].copy()

        rankings = rankings.rename(
            columns={
                "final_rank":
                    "agent1_final_rank",

                "final_stock_score":
                    "agent1_final_score",

                "score_category":
                    "agent1_score_category",
            }
        )

        result = result.merge(
            rankings,
            on="symbol",
            how="left",
            validate="one_to_one",
        )

    except FileNotFoundError:

        pass

    metadata = {
        "selected_source":
            str(
                source
            ),

        "selected_format":
            source_format,

        "rankings_source":
            (
                str(
                    rankings_source
                )
                if rankings_source
                else None
            ),

        "rankings_format":
            rankings_format,

        "rows":
            len(
                result
            ),
    }

    return (
        result,
        metadata,
    )


# =============================================================================
# AGENT 2
# =============================================================================

def load_agent2() -> Tuple[pd.DataFrame, Dict]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT2_TREND_PREDICTIONS_CSV,
            parquet_path=cfg.AGENT2_TREND_PREDICTIONS_PARQUET,
            label="Agent 2 trend predictions",
        )
    )

    result = pd.DataFrame(
        {
            "symbol":
                dataframe[
                    "symbol"
                ]
        }
    )

    mappings = {
        "agent2_prediction_date": (
            "prediction_date",
        ),

        "agent2_probability_seed_42": (
            "probability_seed_42",
        ),

        "agent2_probability_seed_123": (
            "probability_seed_123",
        ),

        "agent2_probability_seed_456": (
            "probability_seed_456",
        ),

        "agent2_top30_probability": (
            "top30_probability",
        ),

        "agent2_ensemble_std": (
            "ensemble_std",
        ),

        "agent2_trend_class": (
            "trend_class",
        ),

        "agent2_rank": (
            "agent2_rank",
        ),
    }

    for output_name, candidates in mappings.items():

        result = add_column_if_present(
            result,
            dataframe,
            candidates,
            output_name,
        )

    metadata = {
        "source":
            str(
                source
            ),

        "format":
            source_format,

        "rows":
            len(
                result
            ),
    }

    return (
        result,
        metadata,
    )


# =============================================================================
# AGENT 3
# =============================================================================

def load_agent3() -> Tuple[pd.DataFrame, Dict]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT3_RISK_CSV,
            parquet_path=cfg.AGENT3_RISK_PARQUET,
            label="Agent 3 risk assessment",
        )
    )

    result = pd.DataFrame(
        {
            "symbol":
                dataframe[
                    "symbol"
                ]
        }
    )

    mappings = {
        "agent3_rank": (
            "agent3_rank",
        ),

        "agent3_score": (
            "agent3_score",
        ),

        "agent3_risk_level": (
            "risk_level",
        ),

        "agent3_risk_decision": (
            "risk_decision",
        ),

        "agent3_volatility": (
            "historical_volatility_annual",
            "annualized_volatility",
            "volatility",
        ),

        "agent3_var_percent": (
            "individual_var_percent",
            "var_percent",
            "historical_var_percent",
        ),

        "agent3_sharpe_ratio": (
            "sharpe_ratio",
            "sharpe",
        ),

        "agent3_drawdown": (
            "max_drawdown",
            "drawdown",
        ),

        "agent3_risk_adjusted_weight": (
            "risk_adjusted_weight",
        ),

        "agent3_risk_adjusted_weight_percent": (
            "risk_adjusted_weight_percent",
        ),
    }

    for output_name, candidates in mappings.items():

        result = add_column_if_present(
            result,
            dataframe,
            candidates,
            output_name,
        )

    metadata = {
        "source":
            str(
                source
            ),

        "format":
            source_format,

        "rows":
            len(
                result
            ),
    }

    return (
        result,
        metadata,
    )


# =============================================================================
# AGENT 4
# =============================================================================

def load_agent4() -> Tuple[pd.DataFrame, Dict]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT4_TRADE_DECISIONS_CSV,
            parquet_path=cfg.AGENT4_TRADE_DECISIONS_PARQUET,
            label="Agent 4 trade decisions",
        )
    )

    result = pd.DataFrame(
        {
            "symbol":
                dataframe[
                    "symbol"
                ]
        }
    )

    mappings = {
        "agent4_prediction_date": (
            "prediction_date",
        ),

        "agent4_q_hold": (
            "q_hold",
        ),

        "agent4_q_buy": (
            "q_buy",
        ),

        "agent4_q_sell": (
            "q_sell",
        ),

        "agent4_action_id": (
            "action_id",
        ),

        "agent4_trade_action": (
            "trade_action",
        ),
    }

    for output_name, candidates in mappings.items():

        result = add_column_if_present(
            result,
            dataframe,
            candidates,
            output_name,
        )

    # -------------------------------------------------------------------------
    # Add Q margin
    # -------------------------------------------------------------------------

    q_columns = [
        "agent4_q_hold",
        "agent4_q_buy",
        "agent4_q_sell",
    ]

    if all(
        column
        in result.columns
        for column in q_columns
    ):

        q_matrix = (
            result[
                q_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
        )

        if np.isfinite(
            q_matrix
        ).all():

            sorted_q = np.sort(
                q_matrix,
                axis=1,
            )

            result[
                "agent4_q_margin"
            ] = (
                sorted_q[
                    :,
                    -1
                ]
                -
                sorted_q[
                    :,
                    -2
                ]
            )

    metadata = {
        "source":
            str(
                source
            ),

        "format":
            source_format,

        "rows":
            len(
                result
            ),
    }

    return (
        result,
        metadata,
    )


# =============================================================================
# AGENT 5
# =============================================================================

def load_agent5() -> Tuple[pd.DataFrame, Dict]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT5_REBALANCED_PORTFOLIO_CSV,
            parquet_path=cfg.AGENT5_REBALANCED_PORTFOLIO_PARQUET,
            label="Agent 5 final portfolio",
        )
    )

    result = pd.DataFrame(
        {
            "symbol":
                dataframe[
                    "symbol"
                ]
        }
    )

    mappings = {
        "agent5_current_weight": (
            "current_weight",
        ),

        "agent5_target_weight": (
            "target_weight",
        ),

        "agent5_weight_change": (
            "weight_change",
        ),

        "agent5_rebalance_direction": (
            "rebalance_direction",
        ),

        "agent5_final_weight": (
            "final_weight",
        ),

        "agent5_final_allocation_percent": (
            "final_allocation_percent",
        ),

        "agent5_portfolio_status": (
            "portfolio_status",
        ),

        "agent5_final_decision": (
            "final_decision",
        ),
    }

    for output_name, candidates in mappings.items():

        result = add_column_if_present(
            result,
            dataframe,
            candidates,
            output_name,
        )

    metadata = {
        "source":
            str(
                source
            ),

        "format":
            source_format,

        "rows":
            len(
                result
            ),
    }

    return (
        result,
        metadata,
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_symbol_sets(
    agent_frames: Dict[int, pd.DataFrame],
) -> Dict[str, bool]:

    symbol_sets = {
        number:
            set(
                frame[
                    "symbol"
                ]
            )
        for number, frame
        in agent_frames.items()
    }

    checks = {
        "agent1_agent2_symbols_match":
            symbol_sets[1]
            ==
            symbol_sets[2],

        "agent2_agent3_symbols_match":
            symbol_sets[2]
            ==
            symbol_sets[3],

        "agent3_agent4_symbols_match":
            symbol_sets[3]
            ==
            symbol_sets[4],

        "agent4_agent5_symbols_match":
            symbol_sets[4]
            ==
            symbol_sets[5],

        "all_agents_same_symbols":
            (
                symbol_sets[1]
                ==
                symbol_sets[2]
                ==
                symbol_sets[3]
                ==
                symbol_sets[4]
                ==
                symbol_sets[5]
            ),
    }

    return checks


def validate_final_trace(
    dataframe: pd.DataFrame,
) -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks[
        "trace_not_empty"
    ] = not dataframe.empty

    checks[
        "ten_rows"
    ] = (
        len(
            dataframe
        )
        ==
        cfg.EXPECTED_SELECTED_STOCKS
    )

    checks[
        "ten_unique_symbols"
    ] = (
        dataframe[
            "symbol"
        ].nunique()
        ==
        cfg.EXPECTED_SELECTED_STOCKS
    )

    required_stage_columns = {
        "symbol",
        "agent2_top30_probability",
        "agent3_score",
        "agent3_risk_level",
        "agent4_trade_action",
        "agent5_final_weight",
    }

    checks[
        "core_stage_columns_present"
    ] = required_stage_columns.issubset(
        dataframe.columns
    )

    if "agent2_top30_probability" in dataframe.columns:

        probabilities = numeric_series(
            dataframe[
                "agent2_top30_probability"
            ]
        )

        checks[
            "agent2_probability_finite"
        ] = bool(
            np.isfinite(
                probabilities
            ).all()
        )

        checks[
            "agent2_probability_range"
        ] = bool(
            probabilities
            .between(
                0.0,
                1.0,
                inclusive="both",
            )
            .all()
        )

    if "agent3_score" in dataframe.columns:

        scores = numeric_series(
            dataframe[
                "agent3_score"
            ]
        )

        checks[
            "agent3_scores_finite"
        ] = bool(
            np.isfinite(
                scores
            ).all()
        )

    if "agent4_trade_action" in dataframe.columns:

        actions = (
            dataframe[
                "agent4_trade_action"
            ]
            .astype(str)
            .str.upper()
        )

        checks[
            "agent4_actions_valid"
        ] = bool(
            actions
            .isin(
                cfg.VALID_TRADE_ACTIONS
            )
            .all()
        )

    if "agent5_final_weight" in dataframe.columns:

        weights = numeric_series(
            dataframe[
                "agent5_final_weight"
            ]
        )

        checks[
            "agent5_weights_finite"
        ] = bool(
            np.isfinite(
                weights
            ).all()
        )

        checks[
            "agent5_weights_nonnegative"
        ] = bool(
            (
                weights
                >=
                -1e-8
            )
            .all()
        )

        checks[
            "agent5_weights_sum_to_one"
        ] = bool(
            np.isclose(
                float(
                    weights.sum()
                ),
                1.0,
                atol=1e-5,
            )
        )

        if "agent4_trade_action" in dataframe.columns:

            sell_mask = (
                dataframe[
                    "agent4_trade_action"
                ]
                .astype(str)
                .str.upper()
                .eq(
                    cfg.ACTION_SELL
                )
            )

            checks[
                "agent4_sell_agent5_zero"
            ] = bool(
                np.allclose(
                    weights[
                        sell_mask
                    ].to_numpy(
                        dtype=float
                    ),
                    0.0,
                    atol=1e-8,
                )
            )

    return checks


# =============================================================================
# BUILD TRACE
# =============================================================================

def build_full_agent_trace() -> Tuple[pd.DataFrame, Dict]:
    """
    Load all five agent outputs and build the complete stock journey table.
    """

    header(
        "AGENTIC AI PORTFOLIO OPTIMIZATION - OUTPUT COLLECTOR"
    )

    cfg.ensure_system_directories()

    metadata: Dict[str, Dict] = {}

    agent_frames: Dict[int, pd.DataFrame] = {}

    # -------------------------------------------------------------------------
    # Agent 1
    # -------------------------------------------------------------------------

    section(
        "1. LOADING AGENT 1 - STOCK SELECTION"
    )

    agent1, metadata[
        "agent1"
    ] = load_agent1()

    agent_frames[
        1
    ] = agent1

    print(
        f"Rows    : {len(agent1)}"
    )

    print(
        f"Columns : {len(agent1.columns)}"
    )

    # -------------------------------------------------------------------------
    # Agent 2
    # -------------------------------------------------------------------------

    section(
        "2. LOADING AGENT 2 - TREND PREDICTION"
    )

    agent2, metadata[
        "agent2"
    ] = load_agent2()

    agent_frames[
        2
    ] = agent2

    print(
        f"Rows    : {len(agent2)}"
    )

    print(
        f"Columns : {len(agent2.columns)}"
    )

    # -------------------------------------------------------------------------
    # Agent 3
    # -------------------------------------------------------------------------

    section(
        "3. LOADING AGENT 3 - RISK MANAGEMENT"
    )

    agent3, metadata[
        "agent3"
    ] = load_agent3()

    agent_frames[
        3
    ] = agent3

    print(
        f"Rows    : {len(agent3)}"
    )

    print(
        f"Columns : {len(agent3.columns)}"
    )

    # -------------------------------------------------------------------------
    # Agent 4
    # -------------------------------------------------------------------------

    section(
        "4. LOADING AGENT 4 - TRADE EXECUTION"
    )

    agent4, metadata[
        "agent4"
    ] = load_agent4()

    agent_frames[
        4
    ] = agent4

    print(
        f"Rows    : {len(agent4)}"
    )

    print(
        f"Columns : {len(agent4.columns)}"
    )

    # -------------------------------------------------------------------------
    # Agent 5
    # -------------------------------------------------------------------------

    section(
        "5. LOADING AGENT 5 - PORTFOLIO REBALANCING"
    )

    agent5, metadata[
        "agent5"
    ] = load_agent5()

    agent_frames[
        5
    ] = agent5

    print(
        f"Rows    : {len(agent5)}"
    )

    print(
        f"Columns : {len(agent5.columns)}"
    )

    # =========================================================================
    # SYMBOL TRANSFER
    # =========================================================================

    section(
        "6. VALIDATING SYMBOL TRANSFER"
    )

    symbol_checks = (
        validate_symbol_sets(
            agent_frames
        )
    )

    for name, passed in (
        symbol_checks.items()
    ):

        print(
            f"{name:<45} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(
        symbol_checks.values()
    ):

        raise ValueError(
            "Stock universes are not consistent "
            "across all five agents."
        )

    # =========================================================================
    # MERGE
    # =========================================================================

    section(
        "7. BUILDING FULL FIVE-AGENT TRACE"
    )

    trace = (
        agent1
        .merge(
            agent2,
            on="symbol",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            agent3,
            on="symbol",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            agent4,
            on="symbol",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            agent5,
            on="symbol",
            how="inner",
            validate="one_to_one",
        )
    )

    # =========================================================================
    # FINAL SORTING
    # =========================================================================

    if "agent5_final_weight" in trace.columns:

        trace[
            "agent5_final_weight"
        ] = numeric_series(
            trace[
                "agent5_final_weight"
            ]
        )

        trace = trace.sort_values(
            by="agent5_final_weight",
            ascending=False,
            kind="stable",
        )

    elif "agent1_final_rank" in trace.columns:

        trace = trace.sort_values(
            by="agent1_final_rank",
            ascending=True,
            kind="stable",
        )

    trace = trace.reset_index(
        drop=True
    )

    trace.insert(
        0,
        "system_row",
        range(
            1,
            len(
                trace
            )
            +
            1
        ),
    )

    # =========================================================================
    # VALIDATE
    # =========================================================================

    checks = validate_final_trace(
        trace
    )

    section(
        "8. FULL TRACE VALIDATION"
    )

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name:<45} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

    validation_pass = (
        all(
            symbol_checks.values()
        )
        and
        all(
            checks.values()
        )
    )

    if not validation_pass:

        raise ValueError(
            "Full Agent 1 -> Agent 5 trace validation failed."
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    section(
        "9. SAVING SYSTEM TRACE"
    )

    trace.to_csv(
        cfg.SYSTEM_FULL_TRACE_CSV,
        index=False,
    )

    trace.to_parquet(
        cfg.SYSTEM_FULL_TRACE_PARQUET,
        index=False,
    )

    print(
        f"CSV     : {cfg.SYSTEM_FULL_TRACE_CSV}"
    )

    print(
        f"Parquet : {cfg.SYSTEM_FULL_TRACE_PARQUET}"
    )

    report = {
        "status":
            "COMPLETE",

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "inference_date":
            cfg.FINAL_INFERENCE_DATE,

        "agents":
            metadata,

        "symbol_checks":
            symbol_checks,

        "trace_checks":
            checks,

        "rows":
            len(
                trace
            ),

        "columns":
            len(
                trace.columns
            ),

        "symbols":
            trace[
                "symbol"
            ].tolist(),

        "output_csv":
            str(
                cfg.SYSTEM_FULL_TRACE_CSV
            ),

        "output_parquet":
            str(
                cfg.SYSTEM_FULL_TRACE_PARQUET
            ),
    }

    with open(
        COLLECTOR_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    # =========================================================================
    # DISPLAY IMPORTANT TRACE
    # =========================================================================

    section(
        "10. COMPLETE STOCK JOURNEY"
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
            "agent4_trade_action",
            "agent4_q_margin",
            "agent5_final_weight",
            "agent5_final_allocation_percent",
            "agent5_portfolio_status",
        )
        if column in trace.columns
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        30,
    ):

        print(
            trace[
                display_columns
            ].to_string(
                index=False
            )
        )

    header(
        "OUTPUT COLLECTOR STATUS"
    )

    print(
        f"Stocks traced        : {len(trace)}"
    )

    print(
        f"Trace columns        : {len(trace.columns)}"
    )

    if "agent5_final_weight" in trace.columns:

        print(
            "Final weight sum     : "
            f"{trace['agent5_final_weight'].sum():.6f}"
        )

    print(
        f"Collector report     : {COLLECTOR_REPORT}"
    )

    print()

    print(
        "OUTPUT COLLECTOR STATUS: COMPLETE"
    )

    print(
        separator()
    )

    return (
        trace,
        report,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    try:

        _, report = (
            build_full_agent_trace()
        )

        success = (
            str(
                report.get(
                    "status",
                    "",
                )
            )
            .upper()
            ==
            "COMPLETE"
        )

        return (
            0
            if success
            else 1
        )

    except Exception as error:

        header(
            "OUTPUT COLLECTOR FAILED"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )