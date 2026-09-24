"""
preflight.py

System-level preflight validation for the complete five-agent
Agentic AI Portfolio Optimization project.

This module DOES NOT execute any agent.

It verifies that the environment, agent entry points, support files,
existing outputs, and cross-agent contracts are valid before the
complete pipeline is executed.

Supported modes
---------------
verify
    Validate the complete set of outputs that already exists.

run
    Validate that the system has the static resources required to
    execute Agent 1 -> Agent 5. Existing outputs are inspected when
    available but are not required.

Usage
-----
python .\\system\\preflight.py --mode verify

python .\\system\\preflight.py --mode run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


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
# THIRD-PARTY IMPORTS
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# REPORT PATH
# =============================================================================

PREFLIGHT_REPORT = (
    cfg.SYSTEM_LATEST_DIR
    / "preflight_summary.json"
)


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_MODES = {
    "verify",
    "run",
}

EXPECTED_PYTHON_MAJOR = 3
MINIMUM_PYTHON_MINOR = 11

REQUIRED_PACKAGES = {
    "numpy": "NumPy",
    "pandas": "Pandas",
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "pyarrow": "PyArrow",
    "tensorflow": "TensorFlow",
}

OPTIONAL_PACKAGES = {
    "pytest": "Pytest",
    "matplotlib": "Matplotlib",
}


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def separator(
    char: str = "=",
    width: int = 100,
) -> str:
    return char * width


def print_header(
    title: str,
) -> None:

    print()
    print(separator())
    print(title)
    print(separator())


def print_section(
    title: str,
) -> None:

    print()
    print(title)
    print("-" * 100)


def print_check(
    name: str,
    passed: bool,
    detail: str = "",
) -> None:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{name:<48} : {status}"
    )

    if detail:
        print(
            f"{'':<48}   {detail}"
        )


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def package_available(
    module_name: str,
) -> bool:
    """
    Test whether a Python package/module can be discovered.

    We intentionally use find_spec rather than importing heavy packages
    such as TensorFlow during preflight.
    """

    try:

        return (
            importlib.util.find_spec(
                module_name
            )
            is not None
        )

    except Exception:

        return False


def path_nonempty(
    path: Path,
) -> bool:
    """
    Return True when path exists and is not an empty file.
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


def first_existing_nonempty(
    paths: Iterable[Path],
) -> Optional[Path]:
    """
    Return first existing non-empty file.
    """

    for path in paths:

        if path_nonempty(
            path
        ):

            return path

    return None


def serialize_path(
    path: Optional[Path],
) -> Optional[str]:

    if path is None:

        return None

    return str(
        path
    )


def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return copy with standardized lower-case column names.
    """

    result = dataframe.copy()

    result.columns = [
        str(column)
        .strip()
        .lower()
        for column in result.columns
    ]

    return result


def load_dataframe(
    csv_path: Optional[Path] = None,
    parquet_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path, str]:
    """
    Load Parquet preferentially, otherwise CSV.

    Returns
    -------
    dataframe, source_path, source_format
    """

    if (
        parquet_path is not None
        and
        path_nonempty(
            parquet_path
        )
    ):

        dataframe = pd.read_parquet(
            parquet_path
        )

        return (
            normalize_columns(
                dataframe
            ),
            parquet_path,
            "PARQUET",
        )

    if (
        csv_path is not None
        and
        path_nonempty(
            csv_path
        )
    ):

        dataframe = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        return (
            normalize_columns(
                dataframe
            ),
            csv_path,
            "CSV",
        )

    paths = [
        path
        for path in (
            parquet_path,
            csv_path,
        )
        if path is not None
    ]

    raise FileNotFoundError(
        "No valid dataframe file found. Candidates:\n"
        +
        "\n".join(
            str(path)
            for path in paths
        )
    )


def numeric_is_finite(
    series: pd.Series,
) -> bool:
    """
    Validate that a numeric dataframe column contains only finite values.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return bool(
        len(values) > 0
        and
        np.isfinite(
            values
        ).all()
    )


def column_set_present(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
) -> bool:

    return set(
        required_columns
    ).issubset(
        set(
            dataframe.columns
        )
    )


def get_first_date_column(
    dataframe: pd.DataFrame,
) -> Optional[str]:
    """
    Detect a likely decision/inference/rebalance date column.
    """

    candidates = (
        "selection_date",
        "prediction_date",
        "assessment_date",
        "risk_date",
        "rebalance_date",
        "final_date",
        "date",
    )

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


def extract_unique_dates(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Read all valid unique dates from the first recognized date field.
    """

    column = get_first_date_column(
        dataframe
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

    if parsed.empty:

        return []

    values = sorted(
        {
            timestamp.strftime(
                "%Y-%m-%d"
            )
            for timestamp in parsed
        }
    )

    return values


def expected_date_matches(
    dataframe: pd.DataFrame,
) -> Optional[bool]:
    """
    Check current final inference date when a usable date column exists.

    Returns None if the dataframe contains no recognized date field.
    """

    dates = extract_unique_dates(
        dataframe
    )

    if not dates:

        return None

    return (
        len(
            dates
        ) == 1
        and
        dates[
            0
        ] == cfg.FINAL_INFERENCE_DATE
    )


# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

def validate_python_environment() -> Dict[str, Any]:

    checks: Dict[str, bool] = {}

    version_ok = (
        sys.version_info.major
        ==
        EXPECTED_PYTHON_MAJOR
        and
        sys.version_info.minor
        >=
        MINIMUM_PYTHON_MINOR
    )

    checks[
        "python_version_supported"
    ] = version_ok

    checks[
        "running_inside_project_venv"
    ] = (
        ".venv"
        in str(
            Path(
                sys.executable
            )
        )
        .lower()
    )

    package_status: Dict[str, bool] = {}

    for module_name in REQUIRED_PACKAGES:

        package_status[
            module_name
        ] = package_available(
            module_name
        )

    optional_status: Dict[str, bool] = {}

    for module_name in OPTIONAL_PACKAGES:

        optional_status[
            module_name
        ] = package_available(
            module_name
        )

    checks[
        "required_packages_available"
    ] = all(
        package_status.values()
    )

    return {
        "checks": checks,
        "python": {
            "version": platform.python_version(),
            "executable": str(
                Path(
                    sys.executable
                ).resolve()
            ),
            "platform": platform.platform(),
        },
        "required_packages": package_status,
        "optional_packages": optional_status,
    }


# =============================================================================
# AGENT DIRECTORY + ENTRYPOINT VALIDATION
# =============================================================================

def validate_agent_structure() -> Dict[str, Any]:

    checks: Dict[str, bool] = {}

    agents: Dict[str, Any] = {}

    for number in cfg.PIPELINE_ORDER:

        definition = (
            cfg.get_agent_description(
                number
            )
        )

        entrypoint = (
            cfg.resolve_agent_entrypoint(
                number
            )
        )

        root_exists = (
            definition.root.exists()
            and
            definition.root.is_dir()
        )

        entrypoint_exists = (
            entrypoint is not None
            and
            path_nonempty(
                entrypoint
            )
        )

        checks[
            f"agent{number}_root_exists"
        ] = root_exists

        checks[
            f"agent{number}_entrypoint_exists"
        ] = entrypoint_exists

        agents[
            f"agent{number}"
        ] = {
            "name": definition.name,
            "root": str(
                definition.root
            ),
            "entrypoint": (
                serialize_path(
                    entrypoint
                )
            ),
        }

    return {
        "checks": checks,
        "agents": agents,
    }


# =============================================================================
# STATIC SUPPORT FILE VALIDATION
# =============================================================================

def validate_support_files(
    mode: str,
) -> Dict[str, Any]:
    """
    Validate files needed by the production system.

    In --mode run we distinguish between:

    STATIC resources
        Files that must already exist before execution.

    UPSTREAM resources
        Files that may be regenerated by an earlier agent in the same run.

    For the current implementation all are reported, but missing upstream
    outputs do not automatically block run mode.
    """

    checks: Dict[str, bool] = {}

    details: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Agent 1 static resources
    # -------------------------------------------------------------------------

    agent1_static = {
        "market_data":
            cfg.AGENT1_MARKET_DATA_PARQUET,

        "features":
            cfg.AGENT1_FEATURES_PARQUET,

        "labeled_features":
            cfg.AGENT1_LABELED_FEATURES,
    }

    for name, path in agent1_static.items():

        passed = path_nonempty(
            path
        )

        checks[
            f"agent1_{name}"
        ] = passed

        details[
            f"agent1_{name}"
        ] = str(
            path
        )

    # -------------------------------------------------------------------------
    # Agent 2 locked inference resources
    # -------------------------------------------------------------------------

    agent2_static = {
        "scaler":
            cfg.AGENT2_SCALER,

        "sequence_metadata":
            cfg.AGENT2_SEQUENCE_METADATA,

        "seed42_weights":
            cfg.AGENT2_SEED_42_WEIGHTS,

        "seed123_weights":
            cfg.AGENT2_SEED_123_WEIGHTS,

        "seed456_weights":
            cfg.AGENT2_SEED_456_WEIGHTS,
    }

    for name, path in agent2_static.items():

        passed = path_nonempty(
            path
        )

        checks[
            f"agent2_{name}"
        ] = passed

        details[
            f"agent2_{name}"
        ] = str(
            path
        )

    # -------------------------------------------------------------------------
    # Upstream dependencies
    #
    # These may legitimately be regenerated in --mode run.
    # -------------------------------------------------------------------------

    upstream_groups = {

        "agent1_selected_stocks": (
            cfg.AGENT1_SELECTED_STOCKS_PARQUET,
            cfg.AGENT1_SELECTED_STOCKS_CSV,
        ),

        "agent2_predictions": (
            cfg.AGENT2_TREND_PREDICTIONS_PARQUET,
            cfg.AGENT2_TREND_PREDICTIONS_CSV,
        ),

        "agent3_risk_output": (
            cfg.AGENT3_RISK_PARQUET,
            cfg.AGENT3_RISK_CSV,
        ),

        "agent4_trade_output": (
            cfg.AGENT4_TRADE_DECISIONS_PARQUET,
            cfg.AGENT4_TRADE_DECISIONS_CSV,
        ),
    }

    upstream_status: Dict[str, bool] = {}

    for name, group in upstream_groups.items():

        source = first_existing_nonempty(
            group
        )

        upstream_status[
            name
        ] = (
            source
            is not None
        )

        details[
            name
        ] = (
            serialize_path(
                source
            )
        )

        if mode == "verify":

            checks[
                name
            ] = (
                source
                is not None
            )

    return {
        "checks": checks,
        "upstream_outputs": upstream_status,
        "details": details,
    }


# =============================================================================
# AGENT 1 OUTPUT VALIDATION
# =============================================================================

def validate_agent1_output() -> Dict[str, Any]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT1_SELECTED_STOCKS_CSV,
            parquet_path=cfg.AGENT1_SELECTED_STOCKS_PARQUET,
        )
    )

    checks: Dict[str, bool] = {}

    checks[
        "not_empty"
    ] = not dataframe.empty

    checks[
        "required_columns"
    ] = column_set_present(
        dataframe,
        cfg.AGENT1_REQUIRED_COLUMNS,
    )

    checks[
        "expected_rows"
    ] = (
        len(
            dataframe
        )
        ==
        cfg.EXPECTED_SELECTED_STOCKS
    )

    checks[
        "unique_symbols"
    ] = (
        "symbol"
        in dataframe.columns
        and
        dataframe[
            "symbol"
        ].nunique()
        ==
        cfg.EXPECTED_SELECTED_STOCKS
    )

    checks[
        "symbols_non_null"
    ] = (
        "symbol"
        in dataframe.columns
        and
        dataframe[
            "symbol"
        ].notna().all()
    )

    date_check = expected_date_matches(
        dataframe
    )

    if date_check is not None:

        checks[
            "final_date_matches"
        ] = date_check

    return {
        "checks": checks,
        "rows": len(
            dataframe
        ),
        "symbols": (
            sorted(
                dataframe[
                    "symbol"
                ]
                .astype(str)
                .str.strip()
                .tolist()
            )
            if "symbol"
            in dataframe.columns
            else []
        ),
        "dates": extract_unique_dates(
            dataframe
        ),
        "source": str(
            source
        ),
        "format": source_format,
    }


# =============================================================================
# AGENT 2 OUTPUT VALIDATION
# =============================================================================

def validate_agent2_output() -> Dict[str, Any]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT2_TREND_PREDICTIONS_CSV,
            parquet_path=cfg.AGENT2_TREND_PREDICTIONS_PARQUET,
        )
    )

    checks: Dict[str, bool] = {}

    checks[
        "not_empty"
    ] = not dataframe.empty

    checks[
        "required_columns"
    ] = column_set_present(
        dataframe,
        cfg.AGENT2_REQUIRED_COLUMNS,
    )

    checks[
        "expected_rows"
    ] = (
        len(
            dataframe
        )
        ==
        cfg.EXPECTED_AGENT2_STOCKS
    )

    checks[
        "unique_symbols"
    ] = (
        dataframe[
            "symbol"
        ].nunique()
        ==
        cfg.EXPECTED_AGENT2_STOCKS
    )

    checks[
        "top30_probability_finite"
    ] = numeric_is_finite(
        dataframe[
            "top30_probability"
        ]
    )

    probabilities = pd.to_numeric(
        dataframe[
            "top30_probability"
        ],
        errors="coerce",
    )

    checks[
        "probability_range_valid"
    ] = bool(
        (
            probabilities
            .between(
                0.0,
                1.0,
                inclusive="both",
            )
        ).all()
    )

    checks[
        "rank_finite"
    ] = numeric_is_finite(
        dataframe[
            "agent2_rank"
        ]
    )

    checks[
        "trend_class_valid"
    ] = (
        dataframe[
            "trend_class"
        ]
        .astype(str)
        .str.upper()
        .isin(
            {
                "TOP30",
                "NOT_TOP30",
            }
        )
        .all()
    )

    date_check = expected_date_matches(
        dataframe
    )

    if date_check is not None:

        checks[
            "final_date_matches"
        ] = date_check

    return {
        "checks": checks,
        "rows": len(
            dataframe
        ),
        "symbols": sorted(
            dataframe[
                "symbol"
            ]
            .astype(str)
            .str.strip()
            .tolist()
        ),
        "dates": extract_unique_dates(
            dataframe
        ),
        "source": str(
            source
        ),
        "format": source_format,
    }


# =============================================================================
# AGENT 3 OUTPUT VALIDATION
# =============================================================================

def validate_agent3_output() -> Dict[str, Any]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT3_RISK_CSV,
            parquet_path=cfg.AGENT3_RISK_PARQUET,
        )
    )

    checks: Dict[str, bool] = {}

    checks[
        "not_empty"
    ] = not dataframe.empty

    checks[
        "required_columns"
    ] = column_set_present(
        dataframe,
        cfg.AGENT3_REQUIRED_COLUMNS,
    )

    checks[
        "expected_rows"
    ] = (
        len(
            dataframe
        )
        ==
        cfg.EXPECTED_AGENT3_STOCKS
    )

    checks[
        "unique_symbols"
    ] = (
        dataframe[
            "symbol"
        ].nunique()
        ==
        cfg.EXPECTED_AGENT3_STOCKS
    )

    checks[
        "agent3_score_finite"
    ] = numeric_is_finite(
        dataframe[
            "agent3_score"
        ]
    )

    checks[
        "agent3_rank_finite"
    ] = numeric_is_finite(
        dataframe[
            "agent3_rank"
        ]
    )

    checks[
        "risk_weight_finite"
    ] = numeric_is_finite(
        dataframe[
            "risk_adjusted_weight"
        ]
    )

    weights = pd.to_numeric(
        dataframe[
            "risk_adjusted_weight"
        ],
        errors="coerce",
    )

    checks[
        "risk_weights_nonnegative"
    ] = bool(
        (
            weights
            >=
            -1e-8
        ).all()
    )

    checks[
        "risk_weights_sum_to_one"
    ] = bool(
        np.isclose(
            float(
                weights.sum()
            ),
            1.0,
            atol=1e-5,
        )
    )

    checks[
        "risk_levels_valid"
    ] = (
        dataframe[
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

    date_check = expected_date_matches(
        dataframe
    )

    if date_check is not None:

        checks[
            "final_date_matches"
        ] = date_check

    return {
        "checks": checks,
        "rows": len(
            dataframe
        ),
        "symbols": sorted(
            dataframe[
                "symbol"
            ]
            .astype(str)
            .str.strip()
            .tolist()
        ),
        "dates": extract_unique_dates(
            dataframe
        ),
        "source": str(
            source
        ),
        "format": source_format,
    }


# =============================================================================
# AGENT 4 OUTPUT VALIDATION
# =============================================================================

def validate_agent4_output() -> Dict[str, Any]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT4_TRADE_DECISIONS_CSV,
            parquet_path=cfg.AGENT4_TRADE_DECISIONS_PARQUET,
        )
    )

    checks: Dict[str, bool] = {}

    checks[
        "not_empty"
    ] = not dataframe.empty

    checks[
        "required_columns"
    ] = column_set_present(
        dataframe,
        cfg.AGENT4_REQUIRED_COLUMNS,
    )

    checks[
        "expected_rows"
    ] = (
        len(
            dataframe
        )
        ==
        cfg.EXPECTED_AGENT4_STOCKS
    )

    checks[
        "unique_symbols"
    ] = (
        dataframe[
            "symbol"
        ].nunique()
        ==
        cfg.EXPECTED_AGENT4_STOCKS
    )

    actions = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
    )

    checks[
        "actions_valid"
    ] = actions.isin(
        cfg.VALID_TRADE_ACTIONS
    ).all()

    for column in (
        "q_hold",
        "q_buy",
        "q_sell",
    ):

        checks[
            f"{column}_finite"
        ] = numeric_is_finite(
            dataframe[
                column
            ]
        )

    action_counts = {
        action:
            int(
                (
                    actions
                    ==
                    action
                ).sum()
            )
        for action
        in cfg.VALID_TRADE_ACTIONS
    }

    # This is current-run-specific.
    # We report it but do not make normal future runs depend on it.
    current_distribution_match = (
        action_counts
        ==
        cfg.CURRENT_EXPECTED_ACTION_COUNTS
    )

    date_check = expected_date_matches(
        dataframe
    )

    if date_check is not None:

        checks[
            "final_date_matches"
        ] = date_check

    return {
        "checks": checks,
        "rows": len(
            dataframe
        ),
        "symbols": sorted(
            dataframe[
                "symbol"
            ]
            .astype(str)
            .str.strip()
            .tolist()
        ),
        "dates": extract_unique_dates(
            dataframe
        ),
        "action_counts": action_counts,
        "current_distribution_match":
            current_distribution_match,
        "source": str(
            source
        ),
        "format": source_format,
    }


# =============================================================================
# AGENT 5 OUTPUT VALIDATION
# =============================================================================

def validate_agent5_output() -> Dict[str, Any]:

    dataframe, source, source_format = (
        load_dataframe(
            csv_path=cfg.AGENT5_REBALANCED_PORTFOLIO_CSV,
            parquet_path=cfg.AGENT5_REBALANCED_PORTFOLIO_PARQUET,
        )
    )

    checks: Dict[str, bool] = {}

    checks[
        "not_empty"
    ] = not dataframe.empty

    checks[
        "required_columns"
    ] = column_set_present(
        dataframe,
        cfg.AGENT5_REQUIRED_COLUMNS,
    )

    checks[
        "expected_rows"
    ] = (
        len(
            dataframe
        )
        ==
        cfg.EXPECTED_AGENT5_STOCKS
    )

    checks[
        "unique_symbols"
    ] = (
        dataframe[
            "symbol"
        ].nunique()
        ==
        cfg.EXPECTED_AGENT5_STOCKS
    )

    checks[
        "final_weight_finite"
    ] = numeric_is_finite(
        dataframe[
            "final_weight"
        ]
    )

    final_weights = pd.to_numeric(
        dataframe[
            "final_weight"
        ],
        errors="coerce",
    )

    checks[
        "final_weights_nonnegative"
    ] = bool(
        (
            final_weights
            >=
            -1e-8
        ).all()
    )

    checks[
        "final_weights_sum_to_one"
    ] = bool(
        np.isclose(
            float(
                final_weights.sum()
            ),
            1.0,
            atol=1e-5,
        )
    )

    actions = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
    )

    checks[
        "actions_valid"
    ] = actions.isin(
        cfg.VALID_TRADE_ACTIONS
    ).all()

    sell_mask = (
        actions
        ==
        cfg.ACTION_SELL
    )

    sell_weights = (
        final_weights[
            sell_mask
        ]
    )

    checks[
        "sell_weights_zero"
    ] = bool(
        np.allclose(
            sell_weights.to_numpy(
                dtype=float
            ),
            0.0,
            atol=1e-8,
        )
    )

    active_assets = int(
        (
            final_weights
            >
            1e-8
        ).sum()
    )

    exited_assets = int(
        (
            final_weights
            <=
            1e-8
        ).sum()
    )

    date_check = expected_date_matches(
        dataframe
    )

    if date_check is not None:

        checks[
            "final_date_matches"
        ] = date_check

    return {
        "checks": checks,
        "rows": len(
            dataframe
        ),
        "symbols": sorted(
            dataframe[
                "symbol"
            ]
            .astype(str)
            .str.strip()
            .tolist()
        ),
        "dates": extract_unique_dates(
            dataframe
        ),
        "active_assets": active_assets,
        "exited_assets": exited_assets,
        "weight_sum": float(
            final_weights.sum()
        ),
        "source": str(
            source
        ),
        "format": source_format,
    }


# =============================================================================
# CROSS-AGENT SYMBOL VALIDATION
# =============================================================================

def validate_symbol_chain(
    agent_results: Dict[str, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Confirm that exactly the same stock universe is transferred through
    all five current final outputs.
    """

    checks: Dict[str, bool] = {}

    symbols = {}

    for number in cfg.PIPELINE_ORDER:

        key = (
            f"agent{number}"
        )

        symbols[
            number
        ] = set(
            agent_results[
                key
            ].get(
                "symbols",
                [],
            )
        )

    checks[
        "agent1_agent2_same_symbols"
    ] = (
        symbols[1]
        ==
        symbols[2]
    )

    checks[
        "agent2_agent3_same_symbols"
    ] = (
        symbols[2]
        ==
        symbols[3]
    )

    checks[
        "agent3_agent4_same_symbols"
    ] = (
        symbols[3]
        ==
        symbols[4]
    )

    checks[
        "agent4_agent5_same_symbols"
    ] = (
        symbols[4]
        ==
        symbols[5]
    )

    checks[
        "all_agents_same_symbols"
    ] = (
        symbols[1]
        ==
        symbols[2]
        ==
        symbols[3]
        ==
        symbols[4]
        ==
        symbols[5]
    )

    return checks


# =============================================================================
# CROSS-AGENT DATE VALIDATION
# =============================================================================

def validate_date_chain(
    agent_results: Dict[str, Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Validate date consistency for all agents exposing a decision date.

    An agent without a date field is ignored here instead of being
    considered automatically invalid.
    """

    observed_dates: Dict[int, list[str]] = {}

    for number in cfg.PIPELINE_ORDER:

        key = (
            f"agent{number}"
        )

        dates = (
            agent_results[
                key
            ].get(
                "dates",
                [],
            )
        )

        if dates:

            observed_dates[
                number
            ] = dates

    checks: Dict[str, bool] = {}

    checks[
        "all_observed_dates_single"
    ] = all(
        len(
            values
        ) == 1
        for values in observed_dates.values()
    )

    checks[
        "all_observed_dates_match_final_date"
    ] = all(
        values[
            0
        ]
        ==
        cfg.FINAL_INFERENCE_DATE
        for values in observed_dates.values()
        if values
    )

    flat_dates = {
        values[
            0
        ]
        for values in observed_dates.values()
        if len(
            values
        ) == 1
    }

    checks[
        "observed_agent_dates_consistent"
    ] = (
        len(
            flat_dates
        )
        <=
        1
    )

    return checks


# =============================================================================
# OUTPUT EXISTENCE VALIDATION
# =============================================================================

def validate_required_outputs() -> Dict[str, bool]:
    """
    Validate all output groups defined in pipeline_config.py.
    """

    checks: Dict[str, bool] = {}

    for number in cfg.PIPELINE_ORDER:

        definition = (
            cfg.get_agent_description(
                number
            )
        )

        for index, group in enumerate(
            definition.required_output_groups,
            start=1,
        ):

            checks[
                (
                    f"agent{number}"
                    f"_output_group_{index}"
                )
            ] = (
                cfg.output_group_exists(
                    group
                )
            )

    return checks


# =============================================================================
# SAFE AGENT VALIDATION WRAPPER
# =============================================================================

def safely_validate_agent(
    agent_number: int,
) -> Dict[str, Any]:

    validators = {
        1: validate_agent1_output,
        2: validate_agent2_output,
        3: validate_agent3_output,
        4: validate_agent4_output,
        5: validate_agent5_output,
    }

    try:

        return validators[
            agent_number
        ]()

    except Exception as error:

        return {
            "checks": {
                "validation_completed":
                    False,
            },
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
            "symbols": [],
            "dates": [],
        }


# =============================================================================
# REPORT WRITER
# =============================================================================

def save_report(
    report: Dict[str, Any],
) -> None:

    cfg.ensure_system_directories()

    with open(
        PREFLIGHT_REPORT,
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
# SUMMARY HELPERS
# =============================================================================

def all_checks_pass(
    checks: Dict[str, bool],
) -> bool:

    return bool(
        checks
        and
        all(
            bool(
                value
            )
            for value in checks.values()
        )
    )


def flatten_agent_checks(
    agent_results: Dict[str, Dict[str, Any]],
) -> Dict[str, bool]:

    flattened: Dict[str, bool] = {}

    for key, result in agent_results.items():

        for name, passed in (
            result
            .get(
                "checks",
                {},
            )
            .items()
        ):

            flattened[
                f"{key}_{name}"
            ] = bool(
                passed
            )

    return flattened


# =============================================================================
# MAIN PREFLIGHT ROUTINE
# =============================================================================

def run_preflight(
    mode: str = "verify",
) -> Dict[str, Any]:

    mode = (
        str(
            mode
        )
        .strip()
        .lower()
    )

    if mode not in SUPPORTED_MODES:

        raise ValueError(
            f"Unsupported preflight mode: {mode}"
        )

    cfg.ensure_system_directories()

    print_header(
        "AGENTIC AI PORTFOLIO OPTIMIZATION - SYSTEM PREFLIGHT"
    )

    print(
        f"Mode                 : {mode.upper()}"
    )

    print(
        f"Project root         : {cfg.PROJECT_ROOT}"
    )

    print(
        f"Final inference date : {cfg.FINAL_INFERENCE_DATE}"
    )

    print(
        f"Python executable    : {sys.executable}"
    )

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    environment = (
        validate_python_environment()
    )

    print_section(
        "1. PYTHON ENVIRONMENT"
    )

    print(
        f"Python version       : "
        f"{environment['python']['version']}"
    )

    for name, passed in (
        environment[
            "checks"
        ].items()
    ):

        print_check(
            name,
            passed,
        )

    for module_name, passed in (
        environment[
            "required_packages"
        ].items()
    ):

        print_check(
            f"package_{module_name}",
            passed,
            REQUIRED_PACKAGES[
                module_name
            ],
        )

    # =========================================================================
    # AGENT STRUCTURE
    # =========================================================================

    structure = (
        validate_agent_structure()
    )

    print_section(
        "2. AGENT STRUCTURE"
    )

    for number in cfg.PIPELINE_ORDER:

        key = (
            f"agent{number}"
        )

        info = (
            structure[
                "agents"
            ][
                key
            ]
        )

        print()
        print(
            f"Agent {number} - "
            f"{info['name']}"
        )

        print(
            f"  Root       : "
            f"{info['root']}"
        )

        print(
            f"  Entrypoint : "
            f"{info['entrypoint']}"
        )

        print_check(
            f"agent{number}_root_exists",
            structure[
                "checks"
            ][
                f"agent{number}_root_exists"
            ],
        )

        print_check(
            f"agent{number}_entrypoint_exists",
            structure[
                "checks"
            ][
                f"agent{number}_entrypoint_exists"
            ],
        )

    # =========================================================================
    # SUPPORT FILES
    # =========================================================================

    support = (
        validate_support_files(
            mode
        )
    )

    print_section(
        "3. SYSTEM SUPPORT FILES"
    )

    for name, passed in (
        support[
            "checks"
        ].items()
    ):

        print_check(
            name,
            passed,
        )

    print()
    print(
        "Existing upstream outputs"
    )

    for name, passed in (
        support[
            "upstream_outputs"
        ].items()
    ):

        print_check(
            name,
            passed,
        )

    # =========================================================================
    # VERIFY EXISTING OUTPUTS
    # =========================================================================

    agent_results: Dict[str, Dict[str, Any]] = {}

    output_group_checks: Dict[str, bool] = {}

    symbol_checks: Dict[str, bool] = {}

    date_checks: Dict[str, bool] = {}

    if mode == "verify":

        print_section(
            "4. EXISTING AGENT OUTPUTS"
        )

        output_group_checks = (
            validate_required_outputs()
        )

        for name, passed in (
            output_group_checks.items()
        ):

            print_check(
                name,
                passed,
            )

        print_section(
            "5. AGENT OUTPUT CONTENT"
        )

        for number in cfg.PIPELINE_ORDER:

            key = (
                f"agent{number}"
            )

            result = (
                safely_validate_agent(
                    number
                )
            )

            agent_results[
                key
            ] = result

            print()
            print(
                f"Agent {number} - "
                f"{cfg.AGENT_NAMES[number]}"
            )

            if (
                "source"
                in result
            ):

                print(
                    f"  Source : "
                    f"{result['source']}"
                )

                print(
                    f"  Format : "
                    f"{result['format']}"
                )

                print(
                    f"  Rows   : "
                    f"{result['rows']}"
                )

            if (
                "error"
                in result
            ):

                print(
                    f"  Error  : "
                    f"{result['error']}"
                )

            for name, passed in (
                result
                .get(
                    "checks",
                    {},
                )
                .items()
            ):

                print_check(
                    name,
                    passed,
                )

        # =====================================================================
        # SYMBOL CHAIN
        # =====================================================================

        print_section(
            "6. AGENT 1 -> 2 -> 3 -> 4 -> 5 SYMBOL CHAIN"
        )

        symbol_checks = (
            validate_symbol_chain(
                agent_results
            )
        )

        for name, passed in (
            symbol_checks.items()
        ):

            print_check(
                name,
                passed,
            )

        # =====================================================================
        # DATE CHAIN
        # =====================================================================

        print_section(
            "7. DATE CONSISTENCY"
        )

        date_checks = (
            validate_date_chain(
                agent_results
            )
        )

        for name, passed in (
            date_checks.items()
        ):

            print_check(
                name,
                passed,
            )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    environment_pass = (
        all_checks_pass(
            environment[
                "checks"
            ]
        )
        and
        all(
            environment[
                "required_packages"
            ].values()
        )
    )

    structure_pass = (
        all_checks_pass(
            structure[
                "checks"
            ]
        )
    )

    support_pass = (
        all_checks_pass(
            support[
                "checks"
            ]
        )
    )

    if mode == "verify":

        agent_content_checks = (
            flatten_agent_checks(
                agent_results
            )
        )

        outputs_pass = (
            all_checks_pass(
                output_group_checks
            )
        )

        content_pass = (
            all_checks_pass(
                agent_content_checks
            )
        )

        symbols_pass = (
            all_checks_pass(
                symbol_checks
            )
        )

        dates_pass = (
            all_checks_pass(
                date_checks
            )
        )

        overall_pass = all(
            [
                environment_pass,
                structure_pass,
                support_pass,
                outputs_pass,
                content_pass,
                symbols_pass,
                dates_pass,
            ]
        )

    else:

        outputs_pass = None
        content_pass = None
        symbols_pass = None
        dates_pass = None

        overall_pass = all(
            [
                environment_pass,
                structure_pass,
                support_pass,
            ]
        )

    report: Dict[str, Any] = {
        "system_name":
            cfg.SYSTEM_NAME,

        "system_version":
            cfg.SYSTEM_VERSION,

        "mode":
            mode,

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "project_root":
            str(
                cfg.PROJECT_ROOT
            ),

        "final_inference_date":
            cfg.FINAL_INFERENCE_DATE,

        "environment":
            environment,

        "structure":
            structure,

        "support":
            support,

        "agent_results":
            agent_results,

        "symbol_chain":
            symbol_checks,

        "date_chain":
            date_checks,

        "stage_status": {
            "environment":
                environment_pass,

            "structure":
                structure_pass,

            "support":
                support_pass,

            "outputs":
                outputs_pass,

            "content":
                content_pass,

            "symbol_chain":
                symbols_pass,

            "date_chain":
                dates_pass,
        },

        "overall_pass":
            overall_pass,
    }

    save_report(
        report
    )

    print_header(
        "PREFLIGHT SUMMARY"
    )

    print_check(
        "Python environment",
        environment_pass,
    )

    print_check(
        "Agent structure",
        structure_pass,
    )

    print_check(
        "Support files",
        support_pass,
    )

    if mode == "verify":

        print_check(
            "Required outputs",
            bool(
                outputs_pass
            ),
        )

        print_check(
            "Output content",
            bool(
                content_pass
            ),
        )

        print_check(
            "Cross-agent symbols",
            bool(
                symbols_pass
            ),
        )

        print_check(
            "Cross-agent dates",
            bool(
                dates_pass
            ),
        )

    print()
    print(
        f"Preflight report : "
        f"{PREFLIGHT_REPORT}"
    )

    print()

    if overall_pass:

        print(
            "SYSTEM PREFLIGHT STATUS: PASS"
        )

    else:

        print(
            "SYSTEM PREFLIGHT STATUS: FAIL"
        )

    print(
        separator()
    )

    return report


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete five-agent "
            "portfolio optimization system."
        )
    )

    parser.add_argument(
        "--mode",
        choices=sorted(
            SUPPORTED_MODES
        ),
        default="verify",
        help=(
            "verify = inspect existing complete pipeline outputs; "
            "run = inspect resources needed before execution."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    arguments = (
        parse_arguments()
    )

    report = (
        run_preflight(
            mode=arguments.mode
        )
    )

    return (
        0
        if report[
            "overall_pass"
        ]
        else 1
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )