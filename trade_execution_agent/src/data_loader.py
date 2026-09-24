"""
Agent 4 - Trade Execution Data Loader
===============================================================================

Paper-aligned temporal preparation.

Available historical source:
    2015-2025

This implementation uses:

    2015-2018 -> historical DQN training
    2019-2023 -> paper-aligned investing/backtest
    2024      -> optional robustness test
    2025      -> final inference

The paper itself uses 2000-2018 for historical calculations and
2019-2023 for investing. Because this project's source data starts in
2015, 2015-2018 is the closest available historical-training period.

Historical DQN states intentionally use only causally computable
market features. Final Agent-2 / Agent-3 snapshot information is
attached only to final inference data.
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

AGENT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    AGENT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            AGENT_ROOT
        ),
    )


from config import config as cfg


# =============================================================================
# FORMAT-AWARE READER
# =============================================================================

def read_with_fallback(
    parquet_path: Path,
    csv_path: Path,
    name: str,
) -> tuple[pd.DataFrame, str]:

    if parquet_path.exists():

        try:

            df = pd.read_parquet(
                parquet_path
            )

            print(
                f"{name:<28}: PARQUET"
            )

            return (
                df,
                "parquet",
            )

        except Exception as error:

            print(
                f"{name}: parquet unavailable"
            )

            print(
                f"Reason: "
                f"{type(error).__name__}: {error}"
            )

    if csv_path.exists():

        df = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        print(
            f"{name:<28}: CSV"
        )

        return (
            df,
            "csv",
        )

    raise FileNotFoundError(
        f"Could not load {name}.\n"
        f"Parquet: {parquet_path}\n"
        f"CSV: {csv_path}"
    )


# =============================================================================
# FORMAT-AWARE WRITER
# =============================================================================

def save_dataframe(
    df: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path,
) -> dict[str, Any]:

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    parquet_saved = False
    parquet_error = None

    try:

        df.to_parquet(
            parquet_path,
            index=False,
        )

        parquet_saved = True

    except Exception as error:

        parquet_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    return {
        "csv":
            str(
                csv_path
            ),

        "parquet":
            str(
                parquet_path
            ),

        "parquet_saved":
            parquet_saved,

        "parquet_error":
            parquet_error,
    }


# =============================================================================
# AGENT-3 INPUT
# =============================================================================

def load_agent3_data() -> tuple[
    pd.DataFrame,
    str,
]:

    (
        df,
        source_format,
    ) = read_with_fallback(
        parquet_path=cfg.AGENT3_RISK_PARQUET_PATH,
        csv_path=cfg.AGENT3_RISK_CSV_PATH,
        name="Agent-3 risk assessment",
    )

    if df.empty:

        raise ValueError(
            "Agent-3 risk assessment is empty."
        )

    missing = [
        column
        for column
        in cfg.AGENT3_REQUIRED_COLUMNS
        if column
        not in df.columns
    ]

    if missing:

        raise ValueError(
            "Agent-3 data missing columns:\n"
            +
            "\n".join(
                f"  - {column}"
                for column
                in missing
            )
        )

    df = df.copy()

    df[
        "symbol"
    ] = (
        df[
            "symbol"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    df[
        "prediction_date"
    ] = pd.to_datetime(
        df[
            "prediction_date"
        ],
        errors="coerce",
    )

    if df[
        "prediction_date"
    ].isna().any():

        raise ValueError(
            "Invalid Agent-3 prediction dates."
        )

    if df[
        "symbol"
    ].duplicated().any():

        raise ValueError(
            "Agent-3 contains duplicate symbols."
        )

    return (
        df,
        source_format,
    )


# =============================================================================
# MARKET DATA
# =============================================================================

def normalize_market_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    lookup = {
        str(column).lower():
            column
        for column
        in df.columns
    }

    required = {
        "symbol": [
            "symbol",
        ],

        "date": [
            "date",
            "timestamp",
        ],

        "open": [
            "open",
        ],

        "high": [
            "high",
        ],

        "low": [
            "low",
        ],

        "close": [
            "close",
        ],

        "volume": [
            "volume",
            "total_traded_quantity",
        ],
    }

    rename_map = {}

    for standard_name, candidates in required.items():

        found = None

        for candidate in candidates:

            if candidate.lower() in lookup:

                found = lookup[
                    candidate.lower()
                ]

                break

        if found is None:

            raise ValueError(
                f"Market-data column missing: "
                f"{standard_name}"
            )

        rename_map[
            found
        ] = standard_name

    return df.rename(
        columns=rename_map
    )


def load_market_data(
    selected_symbols: list[str],
) -> tuple[
    pd.DataFrame,
    str,
]:

    (
        df,
        source_format,
    ) = read_with_fallback(
        parquet_path=cfg.MARKET_DATA_PARQUET_PATH,
        csv_path=cfg.MARKET_DATA_CSV_PATH,
        name="Historical market data",
    )

    df = normalize_market_columns(
        df
    )

    df = df.copy()

    df[
        "symbol"
    ] = (
        df[
            "symbol"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    df = df[
        df[
            "symbol"
        ].isin(
            selected_symbols
        )
    ].copy()

    df = df.dropna(
        subset=[
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    df = (
        df
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

    actual_symbols = set(
        df[
            "symbol"
        ].unique()
    )

    missing_symbols = (
        set(
            selected_symbols
        )
        -
        actual_symbols
    )

    if missing_symbols:

        raise ValueError(
            "Market data missing symbols:\n"
            +
            "\n".join(
                sorted(
                    missing_symbols
                )
            )
        )

    return (
        df,
        source_format,
    )


# =============================================================================
# STATE FEATURE ENGINEERING
# =============================================================================

def engineer_state_features(
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    df = (
        market_df
        .copy()
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    grouped = df.groupby(
        "symbol",
        sort=False,
        group_keys=False,
    )

    # -------------------------------------------------------------------------
    # 1. Previous-open return
    # -------------------------------------------------------------------------

    df[
        "open_return_1d"
    ] = grouped[
        "open"
    ].pct_change(
        fill_method=None
    )

    # -------------------------------------------------------------------------
    # 2. Previous-close return
    # -------------------------------------------------------------------------

    df[
        "close_return_1d"
    ] = grouped[
        "close"
    ].pct_change(
        fill_method=None
    )

    # -------------------------------------------------------------------------
    # 3. Intraday high-low range
    # -------------------------------------------------------------------------

    safe_close = (
        df[
            "close"
        ]
        .replace(
            0.0,
            np.nan,
        )
    )

    df[
        "high_low_range"
    ] = (
        (
            df[
                "high"
            ]
            -
            df[
                "low"
            ]
        )
        /
        safe_close
    )

    # -------------------------------------------------------------------------
    # 4. Opening gap
    # -------------------------------------------------------------------------

    previous_close = grouped[
        "close"
    ].shift(
        1
    )

    df[
        "gap_return"
    ] = (
        df[
            "open"
        ]
        /
        previous_close.replace(
            0.0,
            np.nan,
        )
        -
        1.0
    )

    # -------------------------------------------------------------------------
    # 5. Volume ratio
    # -------------------------------------------------------------------------

    rolling_volume = (
        df
        .groupby(
            "symbol",
            sort=False,
        )[
            "volume"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=cfg.VOLUME_LOOKBACK,
                    min_periods=cfg.VOLUME_LOOKBACK,
                ).mean()
        )
    )

    df[
        "volume_ratio_20d"
    ] = (
        df[
            "volume"
        ]
        /
        rolling_volume.replace(
            0.0,
            np.nan,
        )
    )

    # -------------------------------------------------------------------------
    # 6. Five-day return
    # -------------------------------------------------------------------------

    df[
        "return_5d"
    ] = (
        df
        .groupby(
            "symbol",
            sort=False,
        )[
            "close"
        ]
        .pct_change(
            periods=cfg.RETURN_LOOKBACK,
            fill_method=None,
        )
    )

    # -------------------------------------------------------------------------
    # 7. 20-day volatility
    # -------------------------------------------------------------------------

    df[
        "volatility_20d"
    ] = (
        df
        .groupby(
            "symbol",
            sort=False,
        )[
            "close_return_1d"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=cfg.VOLATILITY_LOOKBACK,
                    min_periods=cfg.VOLATILITY_LOOKBACK,
                ).std()
        )
    )

    df[
        cfg.STATE_FEATURES
    ] = (
        df[
            cfg.STATE_FEATURES
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    return df


# =============================================================================
# TEMPORAL DATASETS
# =============================================================================

def filter_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    result = df[
        (
            df[
                "date"
            ]
            >=
            pd.Timestamp(
                start_date
            )
        )
        &
        (
            df[
                "date"
            ]
            <=
            pd.Timestamp(
                end_date
            )
        )
    ].copy()

    return (
        result
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_historical_datasets(
    state_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    complete = state_df.dropna(
        subset=cfg.STATE_FEATURES
    ).copy()

    training = filter_period(
        complete,
        cfg.TRAIN_START_DATE,
        cfg.TRAIN_END_DATE,
    )

    paper_backtest = filter_period(
        complete,
        cfg.EVALUATION_START_DATE,
        cfg.EVALUATION_END_DATE,
    )

    robustness = filter_period(
        complete,
        cfg.ROBUSTNESS_START_DATE,
        cfg.ROBUSTNESS_END_DATE,
    )

    if training.empty:

        raise ValueError(
            "Historical training dataset is empty."
        )

    if paper_backtest.empty:

        raise ValueError(
            "2019-2023 paper backtest dataset is empty."
        )

    if robustness.empty:

        raise ValueError(
            "2024 robustness dataset is empty."
        )

    return (
        training,
        paper_backtest,
        robustness,
    )


# =============================================================================
# FINAL INFERENCE DATA
# =============================================================================

def build_inference_data(
    state_df: pd.DataFrame,
    agent3_df: pd.DataFrame,
) -> pd.DataFrame:

    unique_dates = (
        agent3_df[
            "prediction_date"
        ]
        .dropna()
        .unique()
    )

    if len(
        unique_dates
    ) != 1:

        raise ValueError(
            "Agent-3 must contain one prediction date."
        )

    prediction_date = pd.Timestamp(
        unique_dates[
            0
        ]
    )

    configured_date = pd.Timestamp(
        cfg.FINAL_INFERENCE_DATE
    )

    if prediction_date != configured_date:

        print(
            "WARNING: Agent-3 prediction date "
            "differs from FINAL_INFERENCE_DATE."
        )

    valid_state = state_df.dropna(
        subset=cfg.STATE_FEATURES
    )

    valid_state = valid_state[
        valid_state[
            "date"
        ]
        <=
        prediction_date
    ].copy()

    latest = (
        valid_state
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .groupby(
            "symbol",
            as_index=False,
        )
        .tail(
            1
        )
        .rename(
            columns={
                "date":
                    "market_state_date"
            }
        )
    )

    inference = latest.merge(
        agent3_df,
        on="symbol",
        how="inner",
        validate="one_to_one",
    )

    return (
        inference
        .sort_values(
            [
                "agent3_rank",
                "symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_data(
    agent3_df: pd.DataFrame,
    training: pd.DataFrame,
    paper_backtest: pd.DataFrame,
    robustness: pd.DataFrame,
    inference: pd.DataFrame,
) -> dict[str, bool]:

    checks = {}

    checks[
        "training_not_empty"
    ] = (
        not training.empty
    )

    checks[
        "paper_backtest_not_empty"
    ] = (
        not paper_backtest.empty
    )

    checks[
        "robustness_not_empty"
    ] = (
        not robustness.empty
    )

    checks[
        "inference_not_empty"
    ] = (
        not inference.empty
    )

    checks[
        "training_features_complete"
    ] = bool(
        training[
            cfg.STATE_FEATURES
        ]
        .notna()
        .all()
        .all()
    )

    checks[
        "paper_backtest_features_complete"
    ] = bool(
        paper_backtest[
            cfg.STATE_FEATURES
        ]
        .notna()
        .all()
        .all()
    )

    checks[
        "robustness_features_complete"
    ] = bool(
        robustness[
            cfg.STATE_FEATURES
        ]
        .notna()
        .all()
        .all()
    )

    checks[
        "training_ends_pre_2019"
    ] = (
        training[
            "date"
        ].max()
        <=
        pd.Timestamp(
            cfg.TRAIN_END_DATE
        )
    )

    checks[
        "paper_backtest_2019_2023"
    ] = (
        paper_backtest[
            "date"
        ].min()
        >=
        pd.Timestamp(
            cfg.EVALUATION_START_DATE
        )
        and
        paper_backtest[
            "date"
        ].max()
        <=
        pd.Timestamp(
            cfg.EVALUATION_END_DATE
        )
    )

    checks[
        "robustness_2024"
    ] = (
        robustness[
            "date"
        ].min()
        >=
        pd.Timestamp(
            cfg.ROBUSTNESS_START_DATE
        )
        and
        robustness[
            "date"
        ].max()
        <=
        pd.Timestamp(
            cfg.ROBUSTNESS_END_DATE
        )
    )

    checks[
        "temporal_order_valid"
    ] = (
        training[
            "date"
        ].max()
        <
        paper_backtest[
            "date"
        ].min()
        <
        robustness[
            "date"
        ].min()
    )

    checks[
        "agent3_symbols_preserved"
    ] = (
        set(
            agent3_df[
                "symbol"
            ]
        )
        ==
        set(
            inference[
                "symbol"
            ]
        )
    )

    checks[
        "inference_no_future_data"
    ] = bool(
        (
            inference[
                "market_state_date"
            ]
            <=
            inference[
                "prediction_date"
            ]
        )
        .all()
    )

    return checks


# =============================================================================
# COMPLETE DATA PREPARATION
# =============================================================================

def prepare_trade_execution_data() -> dict[str, Any]:

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 4 - PAPER-ALIGNED DATA LOADER"
    )

    print(
        "=" * 100
    )

    print()

    (
        agent3_df,
        agent3_format,
    ) = load_agent3_data()

    selected_symbols = (
        agent3_df[
            "symbol"
        ]
        .tolist()
    )

    print(
        f"Agent-3 stocks         : "
        f"{len(selected_symbols)}"
    )

    print(
        f"Prediction date        : "
        f"{agent3_df['prediction_date'].iloc[0].date()}"
    )

    print()

    (
        market_df,
        market_format,
    ) = load_market_data(
        selected_symbols
    )

    print(
        f"Historical rows        : "
        f"{len(market_df):,}"
    )

    print(
        f"Historical stocks      : "
        f"{market_df['symbol'].nunique()}"
    )

    print(
        f"Historical dates       : "
        f"{market_df['date'].min().date()} "
        f"-> "
        f"{market_df['date'].max().date()}"
    )

    state_df = engineer_state_features(
        market_df
    )

    (
        training_df,
        paper_backtest_df,
        robustness_df,
    ) = build_historical_datasets(
        state_df
    )

    inference_df = build_inference_data(
        state_df,
        agent3_df,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "TEMPORAL SPLITS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Training              : "
        f"{training_df['date'].min().date()} "
        f"-> "
        f"{training_df['date'].max().date()} "
        f"({len(training_df):,} rows)"
    )

    print(
        f"Paper backtest        : "
        f"{paper_backtest_df['date'].min().date()} "
        f"-> "
        f"{paper_backtest_df['date'].max().date()} "
        f"({len(paper_backtest_df):,} rows)"
    )

    print(
        f"Robustness            : "
        f"{robustness_df['date'].min().date()} "
        f"-> "
        f"{robustness_df['date'].max().date()} "
        f"({len(robustness_df):,} rows)"
    )

    print(
        f"Final inference rows  : "
        f"{len(inference_df):,}"
    )

    checks = validate_data(
        agent3_df=agent3_df,
        training=training_df,
        paper_backtest=paper_backtest_df,
        robustness=robustness_df,
        inference=inference_df,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "DATA VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<50}: {passed}"
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

        failed = [
            name
            for name, value
            in checks.items()
            if not value
        ]

        raise ValueError(
            "Agent-4 data validation failed:\n"
            +
            "\n".join(
                failed
            )
        )

    # -------------------------------------------------------------------------
    # Save datasets
    # -------------------------------------------------------------------------

    state_output = save_dataframe(
        state_df,
        cfg.STATE_DATA_PATH,
        cfg.STATE_DATA_CSV_PATH,
    )

    training_output = save_dataframe(
        training_df,
        cfg.DQN_TRAINING_DATA_PATH,
        cfg.DQN_TRAINING_DATA_CSV,
    )

    backtest_output = save_dataframe(
        paper_backtest_df,
        cfg.DQN_INPUT_DATA_PATH,
        cfg.DQN_INPUT_DATA_CSV,
    )

    robustness_output = save_dataframe(
        robustness_df,
        cfg.DQN_ROBUSTNESS_DATA_PATH,
        cfg.DQN_ROBUSTNESS_DATA_CSV,
    )

    inference_output = save_dataframe(
        inference_df,
        cfg.DQN_INFERENCE_DATA_PATH,
        cfg.DQN_INFERENCE_DATA_CSV,
    )

    summary = {
        "status":
            "COMPLETE",

        "methodology":
            (
                "Paper-aligned temporal reconstruction: "
                "2015-2018 historical training because source data begins "
                "in 2015; 2019-2023 paper-aligned investing/backtesting; "
                "2024 optional robustness test; 2025 final inference."
            ),

        "agent3_source_format":
            agent3_format,

        "market_source_format":
            market_format,

        "training": {
            "start":
                str(
                    training_df[
                        "date"
                    ]
                    .min()
                    .date()
                ),

            "end":
                str(
                    training_df[
                        "date"
                    ]
                    .max()
                    .date()
                ),

            "rows":
                int(
                    len(
                        training_df
                    )
                ),
        },

        "paper_backtest": {
            "start":
                str(
                    paper_backtest_df[
                        "date"
                    ]
                    .min()
                    .date()
                ),

            "end":
                str(
                    paper_backtest_df[
                        "date"
                    ]
                    .max()
                    .date()
                ),

            "rows":
                int(
                    len(
                        paper_backtest_df
                    )
                ),
        },

        "robustness": {
            "start":
                str(
                    robustness_df[
                        "date"
                    ]
                    .min()
                    .date()
                ),

            "end":
                str(
                    robustness_df[
                        "date"
                    ]
                    .max()
                    .date()
                ),

            "rows":
                int(
                    len(
                        robustness_df
                    )
                ),
        },

        "inference_rows":
            int(
                len(
                    inference_df
                )
            ),

        "validation":
            {
                key:
                    bool(
                        value
                    )
                for key, value
                in checks.items()
            },

        "outputs": {
            "state":
                state_output,

            "training":
                training_output,

            "paper_backtest":
                backtest_output,

            "robustness":
                robustness_output,

            "inference":
                inference_output,
        },
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        cfg.DATA_LOADER_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print(
        "DATA LOADER STATUS: COMPLETE"
    )

    return {
        "training":
            training_df,

        "paper_backtest":
            paper_backtest_df,

        "robustness":
            robustness_df,

        "inference":
            inference_df,

        "checks":
            checks,
    }


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    prepare_trade_execution_data()