"""
Agent 5 - Portfolio Rebalancing Data Loader
===============================================================================

Purpose
-------
Prepare all inputs required by the Agent-5 Markowitz portfolio optimizer.

Inputs
------
1. Agent-4 final BUY / SELL / HOLD decisions
2. Agent-3 risk-management output
3. Historical NIFTY-500 market prices

Outputs
-------
1. returns_matrix.csv / parquet
2. covariance_matrix.csv
3. correlation_matrix.csv
4. asset_statistics.csv / parquet
5. rebalancing_input.csv / parquet
6. data_loader_summary.json

Reference-paper core
--------------------
Portfolio rebalancing uses Markowitz Modern Portfolio Theory.

Core inputs:
    - Mean / expected return
    - Standard deviation
    - Covariance
    - Correlation
    - Portfolio weights

Agent-4 DQN decisions provide RL feedback before rebalancing.

Implementation reconstruction
-----------------------------
The paper does not specify an exact mathematical conversion from
BUY / SELL / HOLD to MPT portfolio constraints.

Current transparent interpretation:

    SELL -> target allocation = 0
    BUY  -> eligible for MPT optimization
    HOLD -> eligible for MPT optimization

No arbitrary BUY or HOLD weighting multiplier is introduced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# IMPORT PATH
# =============================================================================

AGENT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(AGENT_ROOT),
    )


from config import config as cfg


# =============================================================================
# GENERIC DATAFRAME READER
# =============================================================================

def read_with_fallback(
    parquet_path: Path,
    csv_path: Path,
    name: str,
) -> tuple[pd.DataFrame, str]:
    """
    Prefer parquet.

    If parquet cannot be read, fall back to CSV.
    """

    parquet_path = Path(
        parquet_path
    )

    csv_path = Path(
        csv_path
    )

    if parquet_path.exists():

        try:

            dataframe = pd.read_parquet(
                parquet_path
            )

            print(
                f"{name:<32}: PARQUET"
            )

            return (
                dataframe,
                "PARQUET",
            )

        except Exception as error:

            print(
                f"{name:<32}: PARQUET FAILED"
            )

            print(
                f"Reason                        : "
                f"{type(error).__name__}: {error}"
            )

    if csv_path.exists():

        dataframe = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        print(
            f"{name:<32}: CSV"
        )

        return (
            dataframe,
            "CSV",
        )

    raise FileNotFoundError(
        f"{name} was not found.\n\n"
        f"Parquet:\n"
        f"{parquet_path}\n\n"
        f"CSV:\n"
        f"{csv_path}"
    )


# =============================================================================
# GENERIC DATAFRAME WRITER
# =============================================================================

def save_dataframe(
    dataframe: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path | None = None,
) -> dict[str, Any]:
    """
    Save dataframe to CSV and, where requested, parquet.
    """

    csv_path = Path(
        csv_path
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    result: dict[str, Any] = {
        "csv":
            str(csv_path),

        "csv_saved":
            True,
    }

    if parquet_path is not None:

        parquet_path = Path(
            parquet_path
        )

        parquet_saved = False
        parquet_error = None

        try:

            dataframe.to_parquet(
                parquet_path,
                index=False,
            )

            parquet_saved = True

        except Exception as error:

            parquet_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        result[
            "parquet"
        ] = str(
            parquet_path
        )

        result[
            "parquet_saved"
        ] = parquet_saved

        result[
            "parquet_error"
        ] = parquet_error

    return result


# =============================================================================
# AGENT 4 OUTPUT
# =============================================================================

def load_agent4_output() -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load final Agent-4 trade decisions.
    """

    (
        dataframe,
        source_format,
    ) = read_with_fallback(
        parquet_path=cfg.AGENT4_TRADE_DECISIONS_PARQUET,
        csv_path=cfg.AGENT4_TRADE_DECISIONS_CSV,
        name="Agent-4 trade decisions",
    )

    if dataframe.empty:

        raise ValueError(
            "Agent-4 trade decisions are empty."
        )

    required_columns = set(
        cfg.AGENT4_REQUIRED_COLUMNS
    )

    missing_columns = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Agent-4 output is missing required columns:\n"
            +
            "\n".join(
                sorted(
                    missing_columns
                )
            )
        )

    dataframe = dataframe.copy()

    dataframe[
        "symbol"
    ] = (
        dataframe[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe[
        "prediction_date"
    ] = pd.to_datetime(
        dataframe[
            "prediction_date"
        ],
        errors="coerce",
    )

    if "market_state_date" in dataframe.columns:

        dataframe[
            "market_state_date"
        ] = pd.to_datetime(
            dataframe[
                "market_state_date"
            ],
            errors="coerce",
        )

    dataframe[
        "trade_action"
    ] = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if dataframe[
        "prediction_date"
    ].isna().any():

        raise ValueError(
            "Agent-4 contains invalid prediction dates."
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
            "Agent-4 contains duplicate symbols:\n"
            +
            "\n".join(
                sorted(
                    set(
                        duplicates
                    )
                )
            )
        )

    if not dataframe[
        "trade_action"
    ].isin(
        cfg.VALID_ACTIONS
    ).all():

        invalid_actions = (
            dataframe.loc[
                ~dataframe[
                    "trade_action"
                ].isin(
                    cfg.VALID_ACTIONS
                ),
                "trade_action",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Invalid Agent-4 actions: "
            f"{invalid_actions}"
        )

    unique_dates = (
        dataframe[
            "prediction_date"
        ]
        .dropna()
        .unique()
    )

    if len(
        unique_dates
    ) != 1:

        raise ValueError(
            "Agent-4 output must contain exactly "
            "one prediction date."
        )

    prediction_date = pd.Timestamp(
        unique_dates[
            0
        ]
    )

    expected_date = pd.Timestamp(
        cfg.FINAL_REBALANCE_DATE
    )

    if prediction_date != expected_date:

        raise ValueError(
            "Agent-4 prediction date does not match "
            "Agent-5 rebalance date.\n"
            f"Agent-4: {prediction_date.date()}\n"
            f"Agent-5: {expected_date.date()}"
        )

    return (
        dataframe,
        source_format,
    )


# =============================================================================
# AGENT 3 OUTPUT
# =============================================================================

def load_agent3_output() -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load Agent-3 risk-management results.
    """

    (
        dataframe,
        source_format,
    ) = read_with_fallback(
        parquet_path=cfg.AGENT3_RISK_PARQUET,
        csv_path=cfg.AGENT3_RISK_CSV,
        name="Agent-3 risk assessment",
    )

    if dataframe.empty:

        raise ValueError(
            "Agent-3 risk assessment is empty."
        )

    if "symbol" not in dataframe.columns:

        raise ValueError(
            "Agent-3 output does not contain "
            "the required 'symbol' column."
        )

    dataframe = dataframe.copy()

    dataframe[
        "symbol"
    ] = (
        dataframe[
            "symbol"
        ]
        .astype(str)
        .str.strip()
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
            "Agent-3 contains duplicate symbols:\n"
            +
            "\n".join(
                sorted(
                    set(
                        duplicates
                    )
                )
            )
        )

    return (
        dataframe,
        source_format,
    )


# =============================================================================
# MARKET-DATA COLUMN NORMALIZATION
# =============================================================================

def normalize_market_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize required market-data columns to:

        symbol
        date
        close
    """

    column_lookup = {
        str(
            column
        ).lower().strip():
            column

        for column
        in dataframe.columns
    }

    candidates = {
        "symbol": [
            "symbol",
            "ticker",
        ],

        "date": [
            "date",
            "timestamp",
            "datetime",
        ],

        "close": [
            "close",
            "adj close",
            "adj_close",
            "adjusted_close",
        ],
    }

    rename_map = {}

    for target, options in candidates.items():

        source_column = None

        for option in options:

            option_key = (
                option
                .lower()
                .strip()
            )

            if option_key in column_lookup:

                source_column = (
                    column_lookup[
                        option_key
                    ]
                )

                break

        if source_column is None:

            raise ValueError(
                "Historical market data is missing "
                f"required column: {target}"
            )

        rename_map[
            source_column
        ] = target

    return dataframe.rename(
        columns=rename_map
    )


# =============================================================================
# HISTORICAL MARKET DATA
# =============================================================================

def load_market_data(
    symbols: list[str],
    rebalance_date: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load historical prices for Agent-4 stocks only.

    Future observations after the final rebalance date are removed.
    """

    (
        dataframe,
        source_format,
    ) = read_with_fallback(
        parquet_path=cfg.MARKET_DATA_PARQUET,
        csv_path=cfg.MARKET_DATA_CSV,
        name="Historical market data",
    )

    dataframe = normalize_market_columns(
        dataframe
    )

    dataframe = dataframe.copy()

    dataframe[
        "symbol"
    ] = (
        dataframe[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="coerce",
    )

    dataframe[
        "close"
    ] = pd.to_numeric(
        dataframe[
            "close"
        ],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "symbol",
            "date",
            "close",
        ]
    )

    dataframe = dataframe[
        dataframe[
            "symbol"
        ].isin(
            symbols
        )
    ].copy()

    # -------------------------------------------------------------------------
    # NO FUTURE MARKET DATA
    # -------------------------------------------------------------------------

    dataframe = dataframe[
        dataframe[
            "date"
        ]
        <=
        rebalance_date
    ].copy()

    dataframe = (
        dataframe
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .drop_duplicates(
            subset=[
                "symbol",
                "date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    required_symbols = set(
        symbols
    )

    available_symbols = set(
        dataframe[
            "symbol"
        ].unique()
    )

    missing_symbols = (
        required_symbols
        -
        available_symbols
    )

    if missing_symbols:

        raise ValueError(
            "Historical market data is missing "
            "Agent-4 symbols:\n"
            +
            "\n".join(
                sorted(
                    missing_symbols
                )
            )
        )

    return (
        dataframe,
        source_format,
    )


# =============================================================================
# RETURN MATRIX
# =============================================================================

def build_returns_matrix(
    market_dataframe: pd.DataFrame,
    symbols: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build aligned close-price and daily-return matrices.

    Covariance estimation requires common dates across all final assets.
    """

    price_matrix = (
        market_dataframe
        .pivot(
            index="date",
            columns="symbol",
            values="close",
        )
        .sort_index()
    )

    missing_symbols = [
        symbol

        for symbol
        in symbols

        if symbol
        not in price_matrix.columns
    ]

    if missing_symbols:

        raise ValueError(
            "Price matrix is missing symbols:\n"
            +
            "\n".join(
                missing_symbols
            )
        )

    # Preserve Agent-4 order.
    price_matrix = price_matrix[
        symbols
    ]

    returns = (
        price_matrix
        .pct_change(
            fill_method=None
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    # Keep only dates shared by all stocks.
    aligned_returns = (
        returns
        .dropna(
            how="any"
        )
        .copy()
    )

    if len(
        aligned_returns
    ) < cfg.MIN_REQUIRED_RETURN_OBSERVATIONS:

        raise ValueError(
            "Insufficient common return history for "
            "Agent-5 portfolio optimization.\n"
            f"Available aligned observations: "
            f"{len(aligned_returns)}\n"
            f"Minimum required: "
            f"{cfg.MIN_REQUIRED_RETURN_OBSERVATIONS}"
        )

    aligned_returns = (
        aligned_returns
        .tail(
            cfg.ESTIMATION_LOOKBACK_DAYS
        )
        .copy()
    )

    return (
        price_matrix,
        aligned_returns,
    )


# =============================================================================
# ASSET STATISTICS
# =============================================================================

def build_asset_statistics(
    returns: pd.DataFrame,
    agent4: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate asset-level return and volatility statistics.

    Agent-4 contextual information is preserved.
    """

    mean_daily_return = (
        returns.mean()
    )

    daily_volatility = (
        returns.std(
            ddof=1
        )
    )

    annualized_expected_return = (
        (
            1.0
            +
            mean_daily_return
        )
        **
        cfg.TRADING_DAYS_PER_YEAR
        -
        1.0
    )

    annualized_volatility = (
        daily_volatility
        *
        np.sqrt(
            cfg.TRADING_DAYS_PER_YEAR
        )
    )

    statistics = pd.DataFrame(
        {
            "symbol":
                mean_daily_return.index,

            "mean_daily_return":
                mean_daily_return.values,

            "annualized_expected_return":
                annualized_expected_return.values,

            "daily_volatility":
                daily_volatility.values,

            "annualized_volatility":
                annualized_volatility.values,

            "return_observations":
                int(
                    len(
                        returns
                    )
                ),
        }
    )

    context_columns = [
        column

        for column
        in [
            "symbol",
            "prediction_date",
            "market_state_date",
            "q_hold",
            "q_buy",
            "q_sell",
            "selected_q_value",
            "action_id",
            "trade_action",
            "top30_probability",
            "agent2_rank",
            "agent3_score",
            "agent3_rank",
            "risk_level",
            "risk_decision",
            "risk_adjusted_weight",
        ]

        if column
        in agent4.columns
    ]

    statistics = statistics.merge(
        agent4[
            context_columns
        ],
        on="symbol",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # DQN -> MPT ELIGIBILITY
    # -------------------------------------------------------------------------

    statistics[
        "eligible_for_mpt"
    ] = (
        statistics[
            "trade_action"
        ]
        .isin(
            cfg.MPT_ELIGIBLE_ACTIONS
        )
    )

    # -------------------------------------------------------------------------
    # SELL -> ZERO TARGET WEIGHT
    # -------------------------------------------------------------------------

    statistics[
        "forced_weight"
    ] = np.where(
        statistics[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL,

        cfg.SELL_TARGET_WEIGHT,

        np.nan,
    )

    return statistics


# =============================================================================
# COVARIANCE MATRIX
# =============================================================================

def build_covariance_matrix(
    returns: pd.DataFrame,
    eligible_symbols: list[str],
) -> pd.DataFrame:
    """
    Build annualized covariance matrix for MPT-eligible assets only.
    """

    if not eligible_symbols:

        raise ValueError(
            "No MPT-eligible assets are available."
        )

    eligible_returns = (
        returns[
            eligible_symbols
        ]
        .copy()
    )

    covariance_matrix = (
        eligible_returns
        .cov()
        *
        cfg.TRADING_DAYS_PER_YEAR
    )

    covariance_values = (
        covariance_matrix
        .to_numpy(
            dtype=float
        )
    )

    # Numerical stabilization.
    covariance_values = (
        covariance_values
        +
        (
            np.eye(
                len(
                    eligible_symbols
                )
            )
            *
            cfg.COVARIANCE_RIDGE
        )
    )

    covariance_matrix = pd.DataFrame(
        covariance_values,
        index=eligible_symbols,
        columns=eligible_symbols,
    )

    return covariance_matrix


# =============================================================================
# CORRELATION MATRIX
# =============================================================================

def build_correlation_matrix(
    returns: pd.DataFrame,
    eligible_symbols: list[str],
) -> pd.DataFrame:
    """
    Build return correlation matrix for MPT-eligible assets.
    """

    if not eligible_symbols:

        raise ValueError(
            "No MPT-eligible assets are available."
        )

    correlation_matrix = (
        returns[
            eligible_symbols
        ]
        .corr()
    )

    return correlation_matrix


# =============================================================================
# FINAL REBALANCING INPUT
# =============================================================================

def build_rebalancing_input(
    asset_statistics: pd.DataFrame,
    agent3: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the complete Agent-5 optimization input.

    IMPORTANT FIX
    -------------
    The merge key 'symbol' is always retained.

    Previous code built a list of Agent-3 columns by excluding columns
    already present in asset_statistics. Since 'symbol' was already present,
    it could accidentally be removed before:

        merge(..., on="symbol")

    causing:

        KeyError: 'symbol'

    This implementation explicitly adds 'symbol' to the merge dataframe.
    """

    result = asset_statistics.copy()

    # -------------------------------------------------------------------------
    # REQUIRED MERGE KEY
    # -------------------------------------------------------------------------

    if "symbol" not in result.columns:

        raise ValueError(
            "asset_statistics does not contain "
            "the required 'symbol' column."
        )

    if "symbol" not in agent3.columns:

        raise ValueError(
            "Agent-3 output does not contain "
            "the required 'symbol' column."
        )

    result[
        "symbol"
    ] = (
        result[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

    agent3_copy = agent3.copy()

    agent3_copy[
        "symbol"
    ] = (
        agent3_copy[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # DUPLICATE VALIDATION
    # -------------------------------------------------------------------------

    if result[
        "symbol"
    ].duplicated().any():

        duplicate_symbols = (
            result.loc[
                result[
                    "symbol"
                ].duplicated(
                    keep=False
                ),
                "symbol",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate symbols in asset_statistics:\n"
            +
            "\n".join(
                sorted(
                    set(
                        duplicate_symbols
                    )
                )
            )
        )

    if agent3_copy[
        "symbol"
    ].duplicated().any():

        duplicate_symbols = (
            agent3_copy.loc[
                agent3_copy[
                    "symbol"
                ].duplicated(
                    keep=False
                ),
                "symbol",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate symbols in Agent-3 output:\n"
            +
            "\n".join(
                sorted(
                    set(
                        duplicate_symbols
                    )
                )
            )
        )

    # -------------------------------------------------------------------------
    # SYMBOL-COVERAGE VALIDATION
    # -------------------------------------------------------------------------

    result_symbols = set(
        result[
            "symbol"
        ]
    )

    agent3_symbols = set(
        agent3_copy[
            "symbol"
        ]
    )

    missing_from_agent3 = (
        result_symbols
        -
        agent3_symbols
    )

    if missing_from_agent3:

        raise ValueError(
            "Agent-3 is missing Agent-5 symbols:\n"
            +
            "\n".join(
                sorted(
                    missing_from_agent3
                )
            )
        )

    # -------------------------------------------------------------------------
    # AGENT-3 FIELDS
    # -------------------------------------------------------------------------

    possible_agent3_columns = [
        "historical_volatility_annual",
        "individual_var_percent",
        "sharpe_ratio",
        "risk_contribution_percent",
        "max_drawdown_percent",
        "paper_core_risk_score",
        "agent3_score",
        "agent3_rank",
        "risk_level",
        "risk_decision",
        "risk_adjusted_weight",
    ]

    # Add only fields not already transferred through Agent 4.
    new_agent3_columns = [
        column

        for column
        in possible_agent3_columns

        if (
            column
            in agent3_copy.columns

            and

            column
            not in result.columns
        )
    ]

    # -------------------------------------------------------------------------
    # FIXED MERGE
    # -------------------------------------------------------------------------

    if new_agent3_columns:

        merge_columns = [
            "symbol",
            *new_agent3_columns,
        ]

        result = result.merge(
            agent3_copy[
                merge_columns
            ],
            on="symbol",
            how="left",
            validate="one_to_one",
        )

    # -------------------------------------------------------------------------
    # VERIFY MERGED FIELDS
    # -------------------------------------------------------------------------

    for column in new_agent3_columns:

        if column not in result.columns:

            raise ValueError(
                "Agent-3 field failed to merge: "
                f"{column}"
            )

    # -------------------------------------------------------------------------
    # SORT
    # -------------------------------------------------------------------------

    sort_columns = [
        "eligible_for_mpt",
    ]

    ascending = [
        False,
    ]

    if "agent3_rank" in result.columns:

        sort_columns.append(
            "agent3_rank"
        )

        ascending.append(
            True
        )

    else:

        sort_columns.append(
            "symbol"
        )

        ascending.append(
            True
        )

    result = (
        result
        .sort_values(
            by=sort_columns,
            ascending=ascending,
        )
        .reset_index(
            drop=True
        )
    )

    return result


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(
    agent4: pd.DataFrame,
    agent3: pd.DataFrame,
    market_data: pd.DataFrame,
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    correlation: pd.DataFrame,
    asset_statistics: pd.DataFrame,
) -> dict[str, bool]:
    """
    Validate Agent-5 input preparation.
    """

    agent4_symbols = set(
        agent4[
            "symbol"
        ]
    )

    agent3_symbols = set(
        agent3[
            "symbol"
        ]
    )

    market_symbols = set(
        market_data[
            "symbol"
        ].unique()
    )

    eligible_symbols = set(
        asset_statistics.loc[
            asset_statistics[
                "eligible_for_mpt"
            ],
            "symbol",
        ]
    )

    covariance_symbols = set(
        covariance.columns
    )

    covariance_index_symbols = set(
        covariance.index
    )

    correlation_symbols = set(
        correlation.columns
    )

    correlation_index_symbols = set(
        correlation.index
    )

    covariance_values = (
        covariance.to_numpy(
            dtype=float
        )
    )

    correlation_values = (
        correlation.to_numpy(
            dtype=float
        )
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance_values
    )

    checks: dict[str, bool] = {}

    # -------------------------------------------------------------------------
    # Agent-4 checks
    # -------------------------------------------------------------------------

    checks[
        "agent4_not_empty"
    ] = (
        not agent4.empty
    )

    checks[
        "agent4_unique_symbols"
    ] = (
        not agent4[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    # -------------------------------------------------------------------------
    # Agent-3 checks
    # -------------------------------------------------------------------------

    checks[
        "agent3_not_empty"
    ] = (
        not agent3.empty
    )

    checks[
        "agent3_unique_symbols"
    ] = (
        not agent3[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "agent3_agent4_same_symbols"
    ] = (
        agent3_symbols
        ==
        agent4_symbols
    )

    # -------------------------------------------------------------------------
    # Market-data checks
    # -------------------------------------------------------------------------

    checks[
        "market_symbols_complete"
    ] = (
        agent4_symbols
        <=
        market_symbols
    )

    checks[
        "no_future_market_data"
    ] = bool(
        (
            market_data[
                "date"
            ]
            <=
            pd.Timestamp(
                cfg.FINAL_REBALANCE_DATE
            )
        )
        .all()
    )

    # -------------------------------------------------------------------------
    # Return checks
    # -------------------------------------------------------------------------

    checks[
        "returns_not_empty"
    ] = (
        not returns.empty
    )

    checks[
        "minimum_return_observations"
    ] = (
        len(
            returns
        )
        >=
        cfg.MIN_REQUIRED_RETURN_OBSERVATIONS
    )

    checks[
        "lookback_limit_respected"
    ] = (
        len(
            returns
        )
        <=
        cfg.ESTIMATION_LOOKBACK_DAYS
    )

    checks[
        "returns_all_symbols"
    ] = (
        set(
            returns.columns
        )
        ==
        agent4_symbols
    )

    checks[
        "returns_finite"
    ] = bool(
        np.isfinite(
            returns.to_numpy(
                dtype=float
            )
        )
        .all()
    )

    # -------------------------------------------------------------------------
    # MPT eligibility
    # -------------------------------------------------------------------------

    checks[
        "eligible_assets_exist"
    ] = (
        len(
            eligible_symbols
        )
        >
        0
    )

    # -------------------------------------------------------------------------
    # Covariance checks
    # -------------------------------------------------------------------------

    checks[
        "covariance_symbols_valid"
    ] = (
        covariance_symbols
        ==
        eligible_symbols

        and

        covariance_index_symbols
        ==
        eligible_symbols
    )

    checks[
        "covariance_finite"
    ] = bool(
        np.isfinite(
            covariance_values
        )
        .all()
    )

    checks[
        "covariance_symmetric"
    ] = bool(
        np.allclose(
            covariance_values,
            covariance_values.T,
            atol=1e-10,
        )
    )

    checks[
        "covariance_positive_semidefinite"
    ] = bool(
        np.min(
            eigenvalues
        )
        >=
        -1e-8
    )

    # -------------------------------------------------------------------------
    # Correlation checks
    # -------------------------------------------------------------------------

    checks[
        "correlation_symbols_valid"
    ] = (
        correlation_symbols
        ==
        eligible_symbols

        and

        correlation_index_symbols
        ==
        eligible_symbols
    )

    checks[
        "correlation_finite"
    ] = bool(
        np.isfinite(
            correlation_values
        )
        .all()
    )

    checks[
        "correlation_symmetric"
    ] = bool(
        np.allclose(
            correlation_values,
            correlation_values.T,
            atol=1e-10,
        )
    )

    # -------------------------------------------------------------------------
    # Asset statistics
    # -------------------------------------------------------------------------

    checks[
        "asset_statistics_complete"
    ] = (
        set(
            asset_statistics[
                "symbol"
            ]
        )
        ==
        agent4_symbols
    )

    # -------------------------------------------------------------------------
    # SELL checks
    # -------------------------------------------------------------------------

    sell_rows = (
        asset_statistics[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL
    )

    if sell_rows.any():

        checks[
            "sell_assets_not_mpt_eligible"
        ] = bool(
            (
                ~asset_statistics.loc[
                    sell_rows,
                    "eligible_for_mpt",
                ]
            )
            .all()
        )

        checks[
            "sell_target_weight_zero"
        ] = bool(
            np.allclose(
                asset_statistics.loc[
                    sell_rows,
                    "forced_weight",
                ]
                .to_numpy(
                    dtype=float
                ),

                cfg.SELL_TARGET_WEIGHT,

                atol=cfg.FLOAT_TOLERANCE,
            )
        )

    else:

        checks[
            "sell_assets_not_mpt_eligible"
        ] = True

        checks[
            "sell_target_weight_zero"
        ] = True

    return checks


# =============================================================================
# COMPLETE AGENT-5 DATA PREPARATION
# =============================================================================

def prepare_rebalancing_data() -> dict[str, Any]:
    """
    Execute complete Agent-5 data preparation pipeline.
    """

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 5 - PORTFOLIO REBALANCING DATA LOADER"
    )

    print(
        "=" * 100
    )

    print()

    # =========================================================================
    # STEP 1 - AGENT 4
    # =========================================================================

    (
        agent4,
        agent4_format,
    ) = load_agent4_output()

    rebalance_date = pd.Timestamp(
        agent4[
            "prediction_date"
        ]
        .iloc[
            0
        ]
    )

    symbols = (
        agent4[
            "symbol"
        ]
        .tolist()
    )

    print()

    print(
        f"Agent-4 stocks         : "
        f"{len(symbols)}"
    )

    print(
        f"Rebalance date         : "
        f"{rebalance_date.date()}"
    )

    action_counts = (
        agent4[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"BUY actions            : "
        f"{action_counts.get('BUY', 0)}"
    )

    print(
        f"HOLD actions           : "
        f"{action_counts.get('HOLD', 0)}"
    )

    print(
        f"SELL actions           : "
        f"{action_counts.get('SELL', 0)}"
    )

    # =========================================================================
    # STEP 2 - AGENT 3
    # =========================================================================

    print()

    (
        agent3,
        agent3_format,
    ) = load_agent3_output()

    # =========================================================================
    # STEP 3 - MARKET DATA
    # =========================================================================

    print()

    (
        market_data,
        market_format,
    ) = load_market_data(
        symbols=symbols,
        rebalance_date=rebalance_date,
    )

    print()

    print(
        f"Historical market rows : "
        f"{len(market_data):,}"
    )

    print(
        f"Historical stocks      : "
        f"{market_data['symbol'].nunique()}"
    )

    print(
        f"Market date range      : "
        f"{market_data['date'].min().date()} "
        f"-> "
        f"{market_data['date'].max().date()}"
    )

    # =========================================================================
    # STEP 4 - RETURN MATRIX
    # =========================================================================

    (
        price_matrix,
        returns,
    ) = build_returns_matrix(
        market_dataframe=market_data,
        symbols=symbols,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "RETURN ESTIMATION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Aligned return rows    : "
        f"{len(returns)}"
    )

    print(
        f"Return start           : "
        f"{returns.index.min().date()}"
    )

    print(
        f"Return end             : "
        f"{returns.index.max().date()}"
    )

    # =========================================================================
    # STEP 5 - ASSET STATISTICS
    # =========================================================================

    asset_statistics = (
        build_asset_statistics(
            returns=returns,
            agent4=agent4,
        )
    )

    eligible_symbols = (
        asset_statistics.loc[
            asset_statistics[
                "eligible_for_mpt"
            ],
            "symbol",
        ]
        .tolist()
    )

    excluded_symbols = (
        asset_statistics.loc[
            ~asset_statistics[
                "eligible_for_mpt"
            ],
            "symbol",
        ]
        .tolist()
    )

    print()

    print(
        f"MPT eligible stocks    : "
        f"{len(eligible_symbols)}"
    )

    print(
        f"Excluded SELL stocks   : "
        f"{len(excluded_symbols)}"
    )

    print(
        f"Eligible symbols       : "
        f"{', '.join(eligible_symbols)}"
    )

    print(
        f"Excluded symbols       : "
        f"{', '.join(excluded_symbols)}"
    )

    # =========================================================================
    # STEP 6 - COVARIANCE
    # =========================================================================

    covariance = (
        build_covariance_matrix(
            returns=returns,
            eligible_symbols=eligible_symbols,
        )
    )

    # =========================================================================
    # STEP 7 - CORRELATION
    # =========================================================================

    correlation = (
        build_correlation_matrix(
            returns=returns,
            eligible_symbols=eligible_symbols,
        )
    )

    # =========================================================================
    # STEP 8 - REBALANCING INPUT
    # =========================================================================

    rebalancing_input = (
        build_rebalancing_input(
            asset_statistics=asset_statistics,
            agent3=agent3,
        )
    )

    # =========================================================================
    # STEP 9 - VALIDATE
    # =========================================================================

    checks = validate_data(
        agent4=agent4,
        agent3=agent3,
        market_data=market_data,
        returns=returns,
        covariance=covariance,
        correlation=correlation,
        asset_statistics=asset_statistics,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "AGENT 5 DATA VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<55}: "
            f"{passed}"
        )

    overall_pass = all(
        checks.values()
    )

    print()

    print(
        "VALIDATION RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    if not overall_pass:

        failed_checks = [
            name

            for name, passed
            in checks.items()

            if not passed
        ]

        raise ValueError(
            "Agent-5 data validation failed:\n"
            +
            "\n".join(
                f"  - {name}"

                for name
                in failed_checks
            )
        )

    # =========================================================================
    # STEP 10 - SAVE RETURN MATRIX
    # =========================================================================

    returns_to_save = (
        returns
        .reset_index()
    )

    returns_output = save_dataframe(
        dataframe=returns_to_save,
        csv_path=cfg.RETURNS_MATRIX_CSV,
        parquet_path=cfg.RETURNS_MATRIX_PARQUET,
    )

    # =========================================================================
    # STEP 11 - SAVE COVARIANCE
    # =========================================================================

    cfg.COVARIANCE_MATRIX_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    covariance.to_csv(
        cfg.COVARIANCE_MATRIX_CSV,
        index=True,
    )

    # =========================================================================
    # STEP 12 - SAVE CORRELATION
    # =========================================================================

    correlation.to_csv(
        cfg.CORRELATION_MATRIX_CSV,
        index=True,
    )

    # =========================================================================
    # STEP 13 - SAVE ASSET STATISTICS
    # =========================================================================

    asset_statistics_output = save_dataframe(
        dataframe=asset_statistics,
        csv_path=cfg.ASSET_STATISTICS_CSV,
        parquet_path=cfg.ASSET_STATISTICS_PARQUET,
    )

    # =========================================================================
    # STEP 14 - SAVE REBALANCING INPUT
    # =========================================================================

    rebalancing_input_output = save_dataframe(
        dataframe=rebalancing_input,
        csv_path=cfg.REBALANCING_INPUT_CSV,
        parquet_path=cfg.REBALANCING_INPUT_PARQUET,
    )

    # =========================================================================
    # STEP 15 - SUMMARY
    # =========================================================================

    summary = {
        "status":
            "COMPLETE",

        "rebalance_date":
            str(
                rebalance_date.date()
            ),

        "agent4_source_format":
            agent4_format,

        "agent3_source_format":
            agent3_format,

        "market_source_format":
            market_format,

        "selected_stocks":
            int(
                len(
                    symbols
                )
            ),

        "eligible_stocks":
            int(
                len(
                    eligible_symbols
                )
            ),

        "eligible_symbols":
            eligible_symbols,

        "excluded_sell_stocks":
            int(
                len(
                    excluded_symbols
                )
            ),

        "excluded_symbols":
            excluded_symbols,

        "return_observations":
            int(
                len(
                    returns
                )
            ),

        "return_start":
            str(
                returns.index.min().date()
            ),

        "return_end":
            str(
                returns.index.max().date()
            ),

        "estimation_lookback_days":
            int(
                cfg.ESTIMATION_LOOKBACK_DAYS
            ),

        "action_distribution": {
            "BUY":
                int(
                    action_counts.get(
                        "BUY",
                        0,
                    )
                ),

            "HOLD":
                int(
                    action_counts.get(
                        "HOLD",
                        0,
                    )
                ),

            "SELL":
                int(
                    action_counts.get(
                        "SELL",
                        0,
                    )
                ),
        },

        "validation": {
            name:
                bool(
                    passed
                )

            for name, passed
            in checks.items()
        },

        "outputs": {
            "returns":
                returns_output,

            "covariance":
                str(
                    cfg.COVARIANCE_MATRIX_CSV
                ),

            "correlation":
                str(
                    cfg.CORRELATION_MATRIX_CSV
                ),

            "asset_statistics":
                asset_statistics_output,

            "rebalancing_input":
                rebalancing_input_output,
        },

        "methodology_note":
            (
                "Agent 5 prepares Markowitz MPT inputs using Agent-4 "
                "DQN trade decisions and historical returns. SELL "
                "assets are excluded from positive target allocation, "
                "while BUY and HOLD assets remain eligible for MPT. "
                "The 105-trading-day lookback is an implementation "
                "mapping of the paper's reported five-month holding "
                "period."
            ),
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        cfg.DATA_LOADER_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # =========================================================================
    # DISPLAY GENERATED FILES
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "GENERATED FILES"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Returns matrix         : "
        f"{cfg.RETURNS_MATRIX_CSV}"
    )

    print(
        f"Covariance matrix      : "
        f"{cfg.COVARIANCE_MATRIX_CSV}"
    )

    print(
        f"Correlation matrix     : "
        f"{cfg.CORRELATION_MATRIX_CSV}"
    )

    print(
        f"Asset statistics       : "
        f"{cfg.ASSET_STATISTICS_CSV}"
    )

    print(
        f"Rebalancing input      : "
        f"{cfg.REBALANCING_INPUT_CSV}"
    )

    print(
        f"Summary report         : "
        f"{cfg.DATA_LOADER_SUMMARY}"
    )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "AGENT 5 DATA LOADER STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Agent-4 trade decisions : COMPLETE"
    )

    print(
        "Agent-3 risk context    : COMPLETE"
    )

    print(
        "Historical prices       : COMPLETE"
    )

    print(
        "Return matrix           : COMPLETE"
    )

    print(
        "Asset statistics        : COMPLETE"
    )

    print(
        "Covariance matrix       : COMPLETE"
    )

    print(
        "Correlation matrix      : COMPLETE"
    )

    print(
        "MPT input preparation   : COMPLETE"
    )

    print()

    print(
        "DATA LOADER STATUS: COMPLETE"
    )

    return {
        "agent4":
            agent4,

        "agent3":
            agent3,

        "market_data":
            market_data,

        "price_matrix":
            price_matrix,

        "returns":
            returns,

        "covariance":
            covariance,

        "correlation":
            correlation,

        "asset_statistics":
            asset_statistics,

        "rebalancing_input":
            rebalancing_input,

        "checks":
            checks,
    }


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    prepare_rebalancing_data()