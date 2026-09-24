"""
integration_validator.py

End-to-end integration validation for the complete five-agent
Agentic AI Portfolio Optimization system.

This module DOES NOT train or execute the agents.

It validates:

    Agent 1
    Stock Selection
        ↓
    Agent 2
    Trend Prediction
        ↓
    Agent 3
    Risk Management
        ↓
    Agent 4
    Trade Execution
        ↓
    Agent 5
    Portfolio Rebalancing
        ↓
    Final Portfolio

Validation areas
----------------
1. All required agent outputs exist.
2. Each output contains 10 unique stocks.
3. The same stock universe moves through all five agents.
4. Inference / decision dates are consistent.
5. Agent-2 probabilities and rankings are valid.
6. Agent-2 information is preserved by Agent 3.
7. Agent-3 risk information is valid.
8. Agent-3 information is preserved by Agent 4.
9. Agent-4 Q-values are finite.
10. Agent-4 action equals DQN argmax.
11. Agent-4 actions are preserved by Agent 5.
12. Agent-4 SELL stocks receive zero final portfolio weight.
13. Agent-5 weights are finite, non-negative, and sum to one.
14. Unified full_agent_trace agrees with the original agent outputs.
15. Complete integration report is saved.

Usage
-----
python .\\system\\integration_validator.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
# REPORT PATH
# =============================================================================

INTEGRATION_REPORT = (
    cfg.SYSTEM_INTEGRATION_JSON
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def separator(
    char: str = "=",
    width: int = 110,
) -> str:
    """
    Return formatted separator.
    """

    return char * width


def header(
    title: str,
) -> None:
    """
    Print large section header.
    """

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
    """
    Print subsection.
    """

    print()
    print(
        title
    )

    print(
        "-" * 110
    )


def print_check(
    name: str,
    passed: bool,
    detail: str = "",
) -> None:
    """
    Print PASS / FAIL validation result.
    """

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{name:<55} : {status}"
    )

    if detail:

        print(
            f"{'':<55}   {detail}"
        )


# =============================================================================
# GENERIC FILE HELPERS
# =============================================================================

def path_ready(
    path: Path,
) -> bool:
    """
    Return True when a file exists and is non-empty.
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


# =============================================================================
# DATAFRAME NORMALIZATION
# =============================================================================

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
    Standardize stock symbols.
    """

    result = dataframe.copy()

    if "symbol" not in result.columns:

        raise ValueError(
            "Dataframe is missing required column: symbol"
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataframe(
    csv_path: Optional[Path],
    parquet_path: Optional[Path],
    label: str,
) -> Tuple[pd.DataFrame, Path, str]:
    """
    Load one agent output.

    Preference:
        Parquet
        ↓
        CSV
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
            f"{label} not found.\n"
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
            f"{label} is empty."
        )

    if dataframe[
        "symbol"
    ].duplicated().any():

        duplicated_symbols = (
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
            f"{duplicated_symbols}"
        )

    return (
        dataframe,
        source,
        source_format,
    )


# =============================================================================
# NUMERIC HELPERS
# =============================================================================

def numeric_values(
    series: pd.Series,
) -> np.ndarray:
    """
    Convert a Series safely into float array.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


def finite_column(
    dataframe: pd.DataFrame,
    column: str,
) -> bool:
    """
    Check numeric column contains only finite values.
    """

    if column not in dataframe.columns:

        return False

    values = numeric_values(
        dataframe[
            column
        ]
    )

    return bool(
        len(
            values
        ) > 0
        and
        np.isfinite(
            values
        ).all()
    )


# =============================================================================
# SYMBOL HELPERS
# =============================================================================

def symbols_from(
    dataframe: pd.DataFrame,
) -> set[str]:
    """
    Get normalized stock-symbol set.
    """

    return set(
        dataframe[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    """
    Return first available column from candidate names.
    """

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


# =============================================================================
# NUMERIC COLUMN COMPARISON
# =============================================================================

def compare_numeric_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_column: str,
    right_column: str,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> bool:
    """
    Compare two stock-level numeric columns after aligning by symbol.

    IMPORTANT
    ---------
    This function works when the source columns have either:

        identical names

    OR

        different names

    Example:

        full trace:
            agent2_top30_probability

        raw Agent-2 output:
            top30_probability

    Earlier implementation relied on pandas merge suffixes and failed
    when the two names were different.

    The updated implementation explicitly renames the compared values
    before merging.
    """

    if (
        left_column not in left.columns
        or
        right_column not in right.columns
    ):

        return False

    left_frame = (
        left[
            [
                "symbol",
                left_column,
            ]
        ]
        .copy()
        .rename(
            columns={
                left_column:
                    "__left_value"
            }
        )
    )

    right_frame = (
        right[
            [
                "symbol",
                right_column,
            ]
        ]
        .copy()
        .rename(
            columns={
                right_column:
                    "__right_value"
            }
        )
    )

    try:

        merged = left_frame.merge(
            right_frame,
            on="symbol",
            how="inner",
            validate="one_to_one",
        )

    except Exception:

        return False

    if (
        len(
            merged
        )
        !=
        len(
            left
        )
    ):

        return False

    if (
        len(
            merged
        )
        !=
        len(
            right
        )
    ):

        return False

    left_values = numeric_values(
        merged[
            "__left_value"
        ]
    )

    right_values = numeric_values(
        merged[
            "__right_value"
        ]
    )

    if not (
        np.isfinite(
            left_values
        ).all()
        and
        np.isfinite(
            right_values
        ).all()
    ):

        return False

    return bool(
        np.allclose(
            left_values,
            right_values,
            atol=atol,
            rtol=rtol,
        )
    )


# =============================================================================
# STRING COLUMN COMPARISON
# =============================================================================

def compare_string_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_column: str,
    right_column: str,
) -> bool:
    """
    Compare stock-level string columns after aligning by symbol.

    Works correctly whether the original column names are identical
    or different.
    """

    if (
        left_column not in left.columns
        or
        right_column not in right.columns
    ):

        return False

    left_frame = (
        left[
            [
                "symbol",
                left_column,
            ]
        ]
        .copy()
        .rename(
            columns={
                left_column:
                    "__left_value"
            }
        )
    )

    right_frame = (
        right[
            [
                "symbol",
                right_column,
            ]
        ]
        .copy()
        .rename(
            columns={
                right_column:
                    "__right_value"
            }
        )
    )

    try:

        merged = left_frame.merge(
            right_frame,
            on="symbol",
            how="inner",
            validate="one_to_one",
        )

    except Exception:

        return False

    if (
        len(
            merged
        )
        !=
        len(
            left
        )
    ):

        return False

    if (
        len(
            merged
        )
        !=
        len(
            right
        )
    ):

        return False

    left_values = (
        merged[
            "__left_value"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    right_values = (
        merged[
            "__right_value"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return bool(
        (
            left_values
            ==
            right_values
        ).all()
    )


# =============================================================================
# DATE HELPERS
# =============================================================================

DATE_COLUMN_CANDIDATES = (
    "selection_date",
    "prediction_date",
    "assessment_date",
    "risk_date",
    "rebalance_date",
    "final_date",
    "date",
)


def extract_dates(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Extract unique valid dates from likely decision-date column.
    """

    column = first_existing_column(
        dataframe,
        DATE_COLUMN_CANDIDATES,
    )

    if column is None:

        return []

    parsed = pd.to_datetime(
        dataframe[
            column
        ],
        errors="coerce",
    )

    parsed = parsed.dropna()

    return sorted(
        {
            value.strftime(
                "%Y-%m-%d"
            )
            for value
            in parsed
        }
    )


# =============================================================================
# LOAD ALL FIVE AGENTS
# =============================================================================

def load_all_agent_outputs() -> Dict[int, Dict[str, Any]]:
    """
    Load final outputs from all five agents.
    """

    results: Dict[int, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # Agent 1
    # -------------------------------------------------------------------------

    agent1, source1, format1 = (
        load_dataframe(
            csv_path=cfg.AGENT1_SELECTED_STOCKS_CSV,
            parquet_path=cfg.AGENT1_SELECTED_STOCKS_PARQUET,
            label="Agent 1 selected stocks",
        )
    )

    results[
        1
    ] = {
        "data":
            agent1,

        "source":
            source1,

        "format":
            format1,
    }

    # -------------------------------------------------------------------------
    # Agent 2
    # -------------------------------------------------------------------------

    agent2, source2, format2 = (
        load_dataframe(
            csv_path=cfg.AGENT2_TREND_PREDICTIONS_CSV,
            parquet_path=cfg.AGENT2_TREND_PREDICTIONS_PARQUET,
            label="Agent 2 trend predictions",
        )
    )

    results[
        2
    ] = {
        "data":
            agent2,

        "source":
            source2,

        "format":
            format2,
    }

    # -------------------------------------------------------------------------
    # Agent 3
    # -------------------------------------------------------------------------

    agent3, source3, format3 = (
        load_dataframe(
            csv_path=cfg.AGENT3_RISK_CSV,
            parquet_path=cfg.AGENT3_RISK_PARQUET,
            label="Agent 3 risk assessment",
        )
    )

    results[
        3
    ] = {
        "data":
            agent3,

        "source":
            source3,

        "format":
            format3,
    }

    # -------------------------------------------------------------------------
    # Agent 4
    # -------------------------------------------------------------------------

    agent4, source4, format4 = (
        load_dataframe(
            csv_path=cfg.AGENT4_TRADE_DECISIONS_CSV,
            parquet_path=cfg.AGENT4_TRADE_DECISIONS_PARQUET,
            label="Agent 4 trade decisions",
        )
    )

    results[
        4
    ] = {
        "data":
            agent4,

        "source":
            source4,

        "format":
            format4,
    }

    # -------------------------------------------------------------------------
    # Agent 5
    # -------------------------------------------------------------------------

    agent5, source5, format5 = (
        load_dataframe(
            csv_path=cfg.AGENT5_REBALANCED_PORTFOLIO_CSV,
            parquet_path=cfg.AGENT5_REBALANCED_PORTFOLIO_PARQUET,
            label="Agent 5 final portfolio",
        )
    )

    results[
        5
    ] = {
        "data":
            agent5,

        "source":
            source5,

        "format":
            format5,
    }

    return results


# =============================================================================
# OUTPUT STRUCTURE VALIDATION
# =============================================================================

def validate_output_structure(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Verify every agent has exactly ten unique stock rows.
    """

    checks: Dict[str, bool] = {}

    for number in cfg.PIPELINE_ORDER:

        dataframe = (
            outputs[
                number
            ][
                "data"
            ]
        )

        checks[
            f"agent{number}_not_empty"
        ] = (
            not dataframe.empty
        )

        checks[
            f"agent{number}_unique_symbols"
        ] = (
            not dataframe[
                "symbol"
            ]
            .duplicated()
            .any()
        )

        checks[
            f"agent{number}_ten_stocks"
        ] = (
            len(
                dataframe
            )
            ==
            cfg.EXPECTED_SELECTED_STOCKS
        )

    return checks


# =============================================================================
# SYMBOL TRANSFER VALIDATION
# =============================================================================

def validate_symbol_transfer(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Validate same stock universe through all five agents.
    """

    symbols = {
        number:
            symbols_from(
                outputs[
                    number
                ][
                    "data"
                ]
            )
        for number
        in cfg.PIPELINE_ORDER
    }

    return {
        "agent1_agent2_same_symbols":
            (
                symbols[
                    1
                ]
                ==
                symbols[
                    2
                ]
            ),

        "agent2_agent3_same_symbols":
            (
                symbols[
                    2
                ]
                ==
                symbols[
                    3
                ]
            ),

        "agent3_agent4_same_symbols":
            (
                symbols[
                    3
                ]
                ==
                symbols[
                    4
                ]
            ),

        "agent4_agent5_same_symbols":
            (
                symbols[
                    4
                ]
                ==
                symbols[
                    5
                ]
            ),

        "all_five_agents_same_symbols":
            (
                symbols[
                    1
                ]
                ==
                symbols[
                    2
                ]
                ==
                symbols[
                    3
                ]
                ==
                symbols[
                    4
                ]
                ==
                symbols[
                    5
                ]
            ),
    }


# =============================================================================
# DATE VALIDATION
# =============================================================================

def validate_dates(
    outputs: Dict[int, Dict[str, Any]],
) -> Tuple[
    Dict[str, bool],
    Dict[str, list[str]],
]:
    """
    Validate decision / inference dates across agents.
    """

    observed: Dict[
        str,
        list[str],
    ] = {}

    for number in cfg.PIPELINE_ORDER:

        dataframe = (
            outputs[
                number
            ][
                "data"
            ]
        )

        dates = extract_dates(
            dataframe
        )

        if dates:

            observed[
                f"agent{number}"
            ] = dates

    checks: Dict[str, bool] = {}

    checks[
        "all_observed_agent_dates_single"
    ] = all(
        len(
            values
        )
        ==
        1
        for values
        in observed.values()
    )

    checks[
        "all_observed_dates_match_config"
    ] = all(
        values[
            0
        ]
        ==
        cfg.FINAL_INFERENCE_DATE
        for values
        in observed.values()
        if values
    )

    unique_dates = {
        values[
            0
        ]
        for values
        in observed.values()
        if len(
            values
        )
        ==
        1
    }

    checks[
        "all_observed_dates_consistent"
    ] = (
        len(
            unique_dates
        )
        <=
        1
    )

    return (
        checks,
        observed,
    )


# =============================================================================
# AGENT 1 -> AGENT 2 VALIDATION
# =============================================================================

def validate_agent1_agent2(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Validate stock-selection to trend-prediction transfer.
    """

    agent1 = (
        outputs[
            1
        ][
            "data"
        ]
    )

    agent2 = (
        outputs[
            2
        ][
            "data"
        ]
    )

    probabilities = numeric_values(
        agent2[
            "top30_probability"
        ]
    )

    checks = {
        "same_symbol_count":
            (
                len(
                    agent1
                )
                ==
                len(
                    agent2
                )
            ),

        "same_symbol_set":
            (
                symbols_from(
                    agent1
                )
                ==
                symbols_from(
                    agent2
                )
            ),

        "top30_probability_finite":
            np.isfinite(
                probabilities
            ).all(),

        "top30_probability_range":
            (
                (
                    probabilities
                    >=
                    0.0
                )
                &
                (
                    probabilities
                    <=
                    1.0
                )
            ).all(),

        "agent2_rank_finite":
            finite_column(
                agent2,
                "agent2_rank",
            ),

        "trend_class_present":
            (
                "trend_class"
                in agent2.columns
            ),
    }

    return {
        key:
            bool(
                value
            )
        for key, value
        in checks.items()
    }


# =============================================================================
# AGENT 2 -> AGENT 3 VALIDATION
# =============================================================================

def validate_agent2_agent3(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Validate trend information entering risk management.
    """

    agent2 = (
        outputs[
            2
        ][
            "data"
        ]
    )

    agent3 = (
        outputs[
            3
        ][
            "data"
        ]
    )

    checks: Dict[str, bool] = {}

    checks[
        "same_symbol_set"
    ] = (
        symbols_from(
            agent2
        )
        ==
        symbols_from(
            agent3
        )
    )

    checks[
        "agent3_score_finite"
    ] = finite_column(
        agent3,
        "agent3_score",
    )

    checks[
        "agent3_rank_finite"
    ] = finite_column(
        agent3,
        "agent3_rank",
    )

    checks[
        "risk_level_valid"
    ] = bool(
        agent3[
            "risk_level"
        ]
        .astype(str)
        .str.upper()
        .isin(
            {
                "LOW",
                "MEDIUM",
                "HIGH",
            }
        )
        .all()
    )

    checks[
        "risk_adjusted_weight_finite"
    ] = finite_column(
        agent3,
        "risk_adjusted_weight",
    )

    risk_weights = numeric_values(
        agent3[
            "risk_adjusted_weight"
        ]
    )

    checks[
        "risk_weights_nonnegative"
    ] = bool(
        (
            risk_weights
            >=
            -1e-8
        )
        .all()
    )

    checks[
        "risk_weights_sum_to_one"
    ] = bool(
        np.isclose(
            float(
                risk_weights.sum()
            ),
            1.0,
            atol=1e-5,
        )
    )

    # -------------------------------------------------------------------------
    # Agent-2 information preserved by Agent 3
    # -------------------------------------------------------------------------

    if (
        "top30_probability"
        in agent3.columns
    ):

        checks[
            "top30_probability_preserved"
        ] = compare_numeric_columns(
            left=agent2,
            right=agent3,
            left_column="top30_probability",
            right_column="top30_probability",
        )

    if (
        "agent2_rank"
        in agent3.columns
    ):

        checks[
            "agent2_rank_preserved"
        ] = compare_numeric_columns(
            left=agent2,
            right=agent3,
            left_column="agent2_rank",
            right_column="agent2_rank",
        )

    if (
        "trend_class"
        in agent3.columns
    ):

        checks[
            "trend_class_preserved"
        ] = compare_string_columns(
            left=agent2,
            right=agent3,
            left_column="trend_class",
            right_column="trend_class",
        )

    return {
        key:
            bool(
                value
            )
        for key, value
        in checks.items()
    }


# =============================================================================
# AGENT 3 -> AGENT 4 VALIDATION
# =============================================================================

def validate_agent3_agent4(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Validate risk-management to DQN trade-execution transfer.
    """

    agent3 = (
        outputs[
            3
        ][
            "data"
        ]
    )

    agent4 = (
        outputs[
            4
        ][
            "data"
        ]
    )

    checks: Dict[str, bool] = {}

    checks[
        "same_symbol_set"
    ] = (
        symbols_from(
            agent3
        )
        ==
        symbols_from(
            agent4
        )
    )

    # -------------------------------------------------------------------------
    # Q-values
    # -------------------------------------------------------------------------

    for column in (
        "q_hold",
        "q_buy",
        "q_sell",
    ):

        checks[
            f"{column}_finite"
        ] = finite_column(
            agent4,
            column,
        )

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    actions = (
        agent4[
            "trade_action"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    checks[
        "actions_valid"
    ] = bool(
        actions
        .isin(
            cfg.VALID_TRADE_ACTIONS
        )
        .all()
    )

    # -------------------------------------------------------------------------
    # DQN argmax validation
    #
    # Action order:
    #
    # 0 -> HOLD
    # 1 -> BUY
    # 2 -> SELL
    # -------------------------------------------------------------------------

    q_matrix = np.column_stack(
        [
            numeric_values(
                agent4[
                    "q_hold"
                ]
            ),

            numeric_values(
                agent4[
                    "q_buy"
                ]
            ),

            numeric_values(
                agent4[
                    "q_sell"
                ]
            ),
        ]
    )

    action_names = np.array(
        [
            cfg.ACTION_HOLD,
            cfg.ACTION_BUY,
            cfg.ACTION_SELL,
        ],
        dtype=object,
    )

    predicted_actions = (
        action_names[
            np.argmax(
                q_matrix,
                axis=1,
            )
        ]
    )

    checks[
        "dqn_argmax_matches_action"
    ] = bool(
        (
            predicted_actions
            ==
            actions.to_numpy()
        )
        .all()
    )

    # -------------------------------------------------------------------------
    # Agent-3 numeric information preserved
    # -------------------------------------------------------------------------

    optional_numeric_transfers = (
        "agent3_score",
        "agent3_rank",
        "risk_adjusted_weight",
    )

    for column in optional_numeric_transfers:

        if (
            column in agent3.columns
            and
            column in agent4.columns
        ):

            checks[
                f"{column}_preserved"
            ] = compare_numeric_columns(
                left=agent3,
                right=agent4,
                left_column=column,
                right_column=column,
            )

    # -------------------------------------------------------------------------
    # Agent-3 categorical information preserved
    # -------------------------------------------------------------------------

    optional_string_transfers = (
        "risk_level",
        "risk_decision",
    )

    for column in optional_string_transfers:

        if (
            column in agent3.columns
            and
            column in agent4.columns
        ):

            checks[
                f"{column}_preserved"
            ] = compare_string_columns(
                left=agent3,
                right=agent4,
                left_column=column,
                right_column=column,
            )

    return {
        key:
            bool(
                value
            )
        for key, value
        in checks.items()
    }


# =============================================================================
# AGENT 4 -> AGENT 5 VALIDATION
# =============================================================================

def validate_agent4_agent5(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Validate DQN decisions entering portfolio rebalancing.
    """

    agent4 = (
        outputs[
            4
        ][
            "data"
        ]
    )

    agent5 = (
        outputs[
            5
        ][
            "data"
        ]
    )

    checks: Dict[str, bool] = {}

    checks[
        "same_symbol_set"
    ] = (
        symbols_from(
            agent4
        )
        ==
        symbols_from(
            agent5
        )
    )

    checks[
        "trade_actions_preserved"
    ] = compare_string_columns(
        left=agent4,
        right=agent5,
        left_column="trade_action",
        right_column="trade_action",
    )

    # -------------------------------------------------------------------------
    # Final weights
    # -------------------------------------------------------------------------

    weights = numeric_values(
        agent5[
            "final_weight"
        ]
    )

    checks[
        "final_weights_finite"
    ] = bool(
        np.isfinite(
            weights
        ).all()
    )

    checks[
        "final_weights_nonnegative"
    ] = bool(
        (
            weights
            >=
            -1e-8
        )
        .all()
    )

    checks[
        "final_weights_sum_to_one"
    ] = bool(
        np.isclose(
            float(
                weights.sum()
            ),
            1.0,
            atol=1e-5,
        )
    )

    # -------------------------------------------------------------------------
    # Agent-4 SELL -> Agent-5 zero weight
    # -------------------------------------------------------------------------

    merged = (
        agent4[
            [
                "symbol",
                "trade_action",
            ]
        ]
        .merge(
            agent5[
                [
                    "symbol",
                    "final_weight",
                ]
            ],
            on="symbol",
            how="inner",
            validate="one_to_one",
        )
    )

    sell_rows = (
        merged[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .eq(
            cfg.ACTION_SELL
        )
    )

    sell_weights = numeric_values(
        merged.loc[
            sell_rows,
            "final_weight",
        ]
    )

    checks[
        "sell_assets_exist"
    ] = bool(
        int(
            sell_rows.sum()
        )
        >
        0
    )

    checks[
        "sell_assets_zero_weight"
    ] = bool(
        np.allclose(
            sell_weights,
            0.0,
            atol=1e-8,
        )
    )

    # -------------------------------------------------------------------------
    # No positive-weight stock may be Agent-4 SELL
    # -------------------------------------------------------------------------

    all_weights = numeric_values(
        merged[
            "final_weight"
        ]
    )

    active_rows = (
        all_weights
        >
        1e-8
    )

    checks[
        "active_assets_are_not_sell"
    ] = bool(
        merged.loc[
            active_rows,
            "trade_action",
        ]
        .astype(str)
        .str.upper()
        .ne(
            cfg.ACTION_SELL
        )
        .all()
    )

    return {
        key:
            bool(
                value
            )
        for key, value
        in checks.items()
    }


# =============================================================================
# LOAD UNIFIED TRACE
# =============================================================================

def load_full_trace() -> pd.DataFrame:
    """
    Load full Agent-1 -> Agent-5 trace generated by output_collector.py.
    """

    if path_ready(
        cfg.SYSTEM_FULL_TRACE_PARQUET
    ):

        dataframe = pd.read_parquet(
            cfg.SYSTEM_FULL_TRACE_PARQUET
        )

    elif path_ready(
        cfg.SYSTEM_FULL_TRACE_CSV
    ):

        dataframe = pd.read_csv(
            cfg.SYSTEM_FULL_TRACE_CSV,
            low_memory=False,
        )

    else:

        raise FileNotFoundError(
            "Full system trace does not exist.\n"
            "Run:\n"
            ".\\.venv\\Scripts\\python.exe "
            ".\\system\\output_collector.py"
        )

    dataframe = normalize_columns(
        dataframe
    )

    dataframe = normalize_symbols(
        dataframe
    )

    return dataframe


# =============================================================================
# FULL TRACE VALIDATION
# =============================================================================

def validate_full_trace(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Verify unified trace contains the same values as original agent outputs.
    """

    trace = load_full_trace()

    agent1 = (
        outputs[
            1
        ][
            "data"
        ]
    )

    agent2 = (
        outputs[
            2
        ][
            "data"
        ]
    )

    agent3 = (
        outputs[
            3
        ][
            "data"
        ]
    )

    agent4 = (
        outputs[
            4
        ][
            "data"
        ]
    )

    agent5 = (
        outputs[
            5
        ][
            "data"
        ]
    )

    checks: Dict[str, bool] = {}

    checks[
        "trace_not_empty"
    ] = (
        not trace.empty
    )

    checks[
        "trace_has_ten_stocks"
    ] = (
        len(
            trace
        )
        ==
        cfg.EXPECTED_SELECTED_STOCKS
    )

    checks[
        "trace_unique_symbols"
    ] = bool(
        not trace[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "trace_symbols_match_agent1"
    ] = (
        symbols_from(
            trace
        )
        ==
        symbols_from(
            agent1
        )
    )

    checks[
        "trace_symbols_match_agent5"
    ] = (
        symbols_from(
            trace
        )
        ==
        symbols_from(
            agent5
        )
    )

    # -------------------------------------------------------------------------
    # Agent 2
    #
    # DIFFERENT COLUMN NAMES:
    #
    # trace:
    #     agent2_top30_probability
    #
    # raw Agent 2:
    #     top30_probability
    # -------------------------------------------------------------------------

    if (
        "agent2_top30_probability"
        in trace.columns
    ):

        checks[
            "trace_agent2_probability_matches"
        ] = compare_numeric_columns(
            left=trace,
            right=agent2,
            left_column="agent2_top30_probability",
            right_column="top30_probability",
        )

    else:

        checks[
            "trace_agent2_probability_matches"
        ] = False

    # -------------------------------------------------------------------------
    # Agent 3
    # -------------------------------------------------------------------------

    if (
        "agent3_score"
        in trace.columns
    ):

        checks[
            "trace_agent3_score_matches"
        ] = compare_numeric_columns(
            left=trace,
            right=agent3,
            left_column="agent3_score",
            right_column="agent3_score",
        )

    else:

        checks[
            "trace_agent3_score_matches"
        ] = False

    # -------------------------------------------------------------------------
    # Agent 4
    #
    # DIFFERENT COLUMN NAMES:
    #
    # trace:
    #     agent4_trade_action
    #
    # raw Agent 4:
    #     trade_action
    # -------------------------------------------------------------------------

    if (
        "agent4_trade_action"
        in trace.columns
    ):

        checks[
            "trace_agent4_action_matches"
        ] = compare_string_columns(
            left=trace,
            right=agent4,
            left_column="agent4_trade_action",
            right_column="trade_action",
        )

    else:

        checks[
            "trace_agent4_action_matches"
        ] = False

    # -------------------------------------------------------------------------
    # Agent 5
    #
    # DIFFERENT COLUMN NAMES:
    #
    # trace:
    #     agent5_final_weight
    #
    # raw Agent 5:
    #     final_weight
    # -------------------------------------------------------------------------

    if (
        "agent5_final_weight"
        in trace.columns
    ):

        checks[
            "trace_agent5_weight_matches"
        ] = compare_numeric_columns(
            left=trace,
            right=agent5,
            left_column="agent5_final_weight",
            right_column="final_weight",
        )

    else:

        checks[
            "trace_agent5_weight_matches"
        ] = False

    return checks


# =============================================================================
# CURRENT AGENT-4 ACTION DISTRIBUTION
# =============================================================================

def current_action_distribution(
    outputs: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Read current BUY / HOLD / SELL distribution.
    """

    agent4 = (
        outputs[
            4
        ][
            "data"
        ]
    )

    actions = (
        agent4[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
    )

    counts = {
        cfg.ACTION_BUY:
            int(
                (
                    actions
                    ==
                    cfg.ACTION_BUY
                )
                .sum()
            ),

        cfg.ACTION_HOLD:
            int(
                (
                    actions
                    ==
                    cfg.ACTION_HOLD
                )
                .sum()
            ),

        cfg.ACTION_SELL:
            int(
                (
                    actions
                    ==
                    cfg.ACTION_SELL
                )
                .sum()
            ),
    }

    return {
        "counts":
            counts,

        "matches_current_locked_output":
            (
                counts
                ==
                cfg.CURRENT_EXPECTED_ACTION_COUNTS
            ),
    }


# =============================================================================
# SAVE REPORT
# =============================================================================

def save_report(
    report: Dict[str, Any],
) -> None:
    """
    Persist final system integration report.
    """

    cfg.ensure_system_directories()

    with open(
        INTEGRATION_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )


# =============================================================================
# COMPLETE VALIDATION
# =============================================================================

def run_integration_validation() -> Dict[str, Any]:
    """
    Run complete Agent 1 -> Agent 5 integration validation.
    """

    header(
        "AGENTIC AI PORTFOLIO OPTIMIZATION - "
        "FULL SYSTEM INTEGRATION VALIDATION"
    )

    print(
        f"Project root         : "
        f"{cfg.PROJECT_ROOT}"
    )

    print(
        f"Final inference date : "
        f"{cfg.FINAL_INFERENCE_DATE}"
    )

    # =========================================================================
    # 1. LOAD AGENT OUTPUTS
    # =========================================================================

    section(
        "1. LOADING AGENT OUTPUTS"
    )

    outputs = (
        load_all_agent_outputs()
    )

    for number in cfg.PIPELINE_ORDER:

        info = (
            outputs[
                number
            ]
        )

        dataframe = (
            info[
                "data"
            ]
        )

        print(
            f"Agent {number:<2} "
            f"{cfg.AGENT_NAMES[number]:<30} "
            f"| rows={len(dataframe):<3} "
            f"| format={info['format']}"
        )

        print(
            f"       "
            f"{info['source']}"
        )

    # =========================================================================
    # 2. OUTPUT STRUCTURE
    # =========================================================================

    section(
        "2. OUTPUT STRUCTURE"
    )

    structure_checks = (
        validate_output_structure(
            outputs
        )
    )

    for name, passed in (
        structure_checks.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 3. SYMBOL TRANSFER
    # =========================================================================

    section(
        "3. SYMBOL TRANSFER"
    )

    symbol_checks = (
        validate_symbol_transfer(
            outputs
        )
    )

    for name, passed in (
        symbol_checks.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 4. DATE CONSISTENCY
    # =========================================================================

    section(
        "4. DATE CONSISTENCY"
    )

    (
        date_checks,
        observed_dates,
    ) = validate_dates(
        outputs
    )

    for agent, dates in (
        observed_dates.items()
    ):

        print(
            f"{agent:<20} : "
            f"{', '.join(dates)}"
        )

    for name, passed in (
        date_checks.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 5. AGENT 1 -> AGENT 2
    # =========================================================================

    section(
        "5. AGENT 1 -> AGENT 2"
    )

    agent1_agent2 = (
        validate_agent1_agent2(
            outputs
        )
    )

    for name, passed in (
        agent1_agent2.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 6. AGENT 2 -> AGENT 3
    # =========================================================================

    section(
        "6. AGENT 2 -> AGENT 3"
    )

    agent2_agent3 = (
        validate_agent2_agent3(
            outputs
        )
    )

    for name, passed in (
        agent2_agent3.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 7. AGENT 3 -> AGENT 4
    # =========================================================================

    section(
        "7. AGENT 3 -> AGENT 4"
    )

    agent3_agent4 = (
        validate_agent3_agent4(
            outputs
        )
    )

    for name, passed in (
        agent3_agent4.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 8. AGENT 4 -> AGENT 5
    # =========================================================================

    section(
        "8. AGENT 4 -> AGENT 5"
    )

    agent4_agent5 = (
        validate_agent4_agent5(
            outputs
        )
    )

    for name, passed in (
        agent4_agent5.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 9. FULL TRACE
    # =========================================================================

    section(
        "9. UNIFIED FULL-AGENT TRACE"
    )

    trace_checks = (
        validate_full_trace(
            outputs
        )
    )

    for name, passed in (
        trace_checks.items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # 10. ACTION DISTRIBUTION
    # =========================================================================

    section(
        "10. CURRENT AGENT-4 ACTION DISTRIBUTION"
    )

    action_distribution = (
        current_action_distribution(
            outputs
        )
    )

    counts = (
        action_distribution[
            "counts"
        ]
    )

    print(
        f"BUY  : "
        f"{counts[cfg.ACTION_BUY]}"
    )

    print(
        f"HOLD : "
        f"{counts[cfg.ACTION_HOLD]}"
    )

    print(
        f"SELL : "
        f"{counts[cfg.ACTION_SELL]}"
    )

    print()

    print(
        "Matches current locked 2025 output : "
        f"{action_distribution['matches_current_locked_output']}"
    )

    # =========================================================================
    # 11. FINAL PORTFOLIO
    # =========================================================================

    section(
        "11. FINAL PORTFOLIO"
    )

    agent5 = (
        outputs[
            5
        ][
            "data"
        ]
        .copy()
    )

    agent5[
        "final_weight"
    ] = pd.to_numeric(
        agent5[
            "final_weight"
        ],
        errors="coerce",
    )

    agent5 = agent5.sort_values(
        "final_weight",
        ascending=False,
        kind="stable",
    )

    active_assets = int(
        (
            agent5[
                "final_weight"
            ]
            >
            1e-8
        )
        .sum()
    )

    exited_assets = int(
        (
            agent5[
                "final_weight"
            ]
            <=
            1e-8
        )
        .sum()
    )

    final_weight_sum = float(
        agent5[
            "final_weight"
        ]
        .sum()
    )

    print(
        f"Active assets         : "
        f"{active_assets}"
    )

    print(
        f"Exited assets         : "
        f"{exited_assets}"
    )

    print(
        f"Weight sum            : "
        f"{final_weight_sum:.8f}"
    )

    print()

    final_display_columns = [
        column
        for column in (
            "symbol",
            "trade_action",
            "final_weight",
            "final_allocation_percent",
            "portfolio_status",
        )
        if column
        in agent5.columns
    ]

    with pd.option_context(
        "display.width",
        180,
        "display.max_columns",
        None,
    ):

        print(
            agent5[
                final_display_columns
            ]
            .to_string(
                index=False
            )
        )

    # =========================================================================
    # GLOBAL VALIDATION GROUPS
    # =========================================================================

    validation_groups = {
        "output_structure":
            structure_checks,

        "symbol_transfer":
            symbol_checks,

        "date_consistency":
            date_checks,

        "agent1_agent2":
            agent1_agent2,

        "agent2_agent3":
            agent2_agent3,

        "agent3_agent4":
            agent3_agent4,

        "agent4_agent5":
            agent4_agent5,

        "full_trace":
            trace_checks,
    }

    group_status = {
        group:
            bool(
                checks
                and
                all(
                    bool(
                        value
                    )
                    for value
                    in checks.values()
                )
            )
        for group, checks
        in validation_groups.items()
    }

    overall_pass = bool(
        all(
            group_status.values()
        )
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    report = {
        "system_name":
            cfg.SYSTEM_NAME,

        "system_version":
            cfg.SYSTEM_VERSION,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "inference_date":
            cfg.FINAL_INFERENCE_DATE,

        "agent_sources": {
            f"agent{number}": {
                "name":
                    cfg.AGENT_NAMES[
                        number
                    ],

                "source":
                    str(
                        outputs[
                            number
                        ][
                            "source"
                        ]
                    ),

                "format":
                    outputs[
                        number
                    ][
                        "format"
                    ],

                "rows":
                    len(
                        outputs[
                            number
                        ][
                            "data"
                        ]
                    ),
            }
            for number
            in cfg.PIPELINE_ORDER
        },

        "validation_groups":
            validation_groups,

        "group_status":
            group_status,

        "observed_dates":
            observed_dates,

        "action_distribution":
            action_distribution,

        "final_portfolio": {
            "stocks":
                len(
                    agent5
                ),

            "active_assets":
                active_assets,

            "exited_assets":
                exited_assets,

            "weight_sum":
                final_weight_sum,
        },

        "overall_pass":
            overall_pass,

        "status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
    }

    save_report(
        report
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    header(
        "FULL SYSTEM INTEGRATION SUMMARY"
    )

    for group, passed in (
        group_status.items()
    ):

        print_check(
            group,
            passed,
        )

    print()

    print(
        f"Integration report : "
        f"{INTEGRATION_REPORT}"
    )

    print()

    if overall_pass:

        print(
            "AGENT 1 -> AGENT 2 -> AGENT 3 -> "
            "AGENT 4 -> AGENT 5 : PASS"
        )

        print()

        print(
            "FULL SYSTEM INTEGRATION STATUS: PASS"
        )

    else:

        print(
            "FULL SYSTEM INTEGRATION STATUS: FAIL"
        )

    print(
        separator()
    )

    return report


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """
    Command-line entry point.
    """

    try:

        report = (
            run_integration_validation()
        )

        return (
            0
            if report[
                "overall_pass"
            ]
            else 1
        )

    except Exception as error:

        header(
            "FULL SYSTEM INTEGRATION VALIDATION FAILED"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )