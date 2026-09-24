"""
Agent 2 - Trend Prediction Data Preprocessor
============================================

This version supports the broad NIFTY-500 training universe and
automatically repairs / derives Agent 2 features that are not already
present in Agent 1's features.parquet.

Main safeguards
---------------
1. Broad historical universe is used for model learning.
2. Feature calculations are performed separately per stock.
3. Only current/past observations are used for technical indicators.
4. Train / backtest / validation splits are chronological.
5. StandardScaler is fitted ONLY on training data.
6. Backtest and validation data never participate in scaler fitting.
7. Target generation is NOT performed here.
8. Future-return leakage columns are excluded.

Current implementation periods
------------------------------
TRAIN      : 2015-01-01 -> 2018-12-31
BACKTEST   : 2019-01-01 -> 2024-12-31
VALIDATION : 2025-01-01 -> 2025-12-31

These split dates are implementation choices for the current project.
"""

from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

TREND_AGENT_ROOT = CURRENT_FILE.parents[1]

PROJECT_ROOT = TREND_AGENT_ROOT.parent


if str(TREND_AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(TREND_AGENT_ROOT),
    )


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

PROCESSED_DIR = (
    TREND_AGENT_ROOT
    / "data"
    / "processed"
)


TRAINING_UNIVERSE_FILE = (
    PROCESSED_DIR
    / "training_universe.parquet"
)


PREPROCESSED_FILE = (
    PROCESSED_DIR
    / "preprocessed_trend_data.parquet"
)


SCALER_FILE = (
    PROCESSED_DIR
    / "scaler.pkl"
)


PREPROCESSING_SUMMARY_FILE = (
    PROCESSED_DIR
    / "preprocessing_summary.csv"
)


FEATURE_REPORT_FILE = (
    PROCESSED_DIR
    / "feature_preprocessing_report.csv"
)


# ============================================================
# DATE SPLITS
# ============================================================

TRAIN_START_DATE = pd.Timestamp(
    "2015-01-01"
)

TRAIN_END_DATE = pd.Timestamp(
    "2018-12-31"
)


BACKTEST_START_DATE = pd.Timestamp(
    "2019-01-01"
)

BACKTEST_END_DATE = pd.Timestamp(
    "2024-12-31"
)


VALIDATION_START_DATE = pd.Timestamp(
    "2025-01-01"
)

VALIDATION_END_DATE = pd.Timestamp(
    "2025-12-31"
)


# ============================================================
# MODEL FEATURE SET
# ============================================================

FEATURE_COLUMNS = [

    # Returns
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",

    # Moving averages
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",

    # Exponential moving averages
    "ema_12",
    "ema_26",

    # Price / MA ratios
    "close_to_sma_5",
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_sma_200",

    # MA relationships
    "sma_5_to_sma_20",
    "sma_20_to_sma_50",

    # Momentum
    "rsi_14",

    "macd",
    "macd_signal",
    "macd_hist",

    # Risk
    "volatility_20d",
    "volatility_60d",

    "sharpe_60d",
    "sharpe_252d",

    "drawdown",

    # Volume
    "volume_change_1d",
    "volume_ratio_20d",

    # Price structure
    "intraday_range",
    "gap_return",
]


# ============================================================
# EXACTLY 30 FEATURES CHECK
# ============================================================

if len(FEATURE_COLUMNS) != 30:

    raise RuntimeError(
        f"Agent 2 expects 30 features, "
        f"but FEATURE_COLUMNS contains "
        f"{len(FEATURE_COLUMNS)}."
    )


# ============================================================
# POSSIBLE COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {

    "return_1d": [
        "daily_return",
        "return",
        "ret_1d",
        "returns_1d",
        "pct_change_1d",
    ],

    "return_5d": [
        "ret_5d",
        "returns_5d",
    ],

    "return_21d": [
        "ret_21d",
        "returns_21d",
    ],

    "return_63d": [
        "ret_63d",
        "returns_63d",
    ],

    "sma_5": [
        "ma_5",
        "sma5",
    ],

    "sma_10": [
        "ma_10",
        "sma10",
    ],

    "sma_20": [
        "ma_20",
        "sma20",
    ],

    "sma_50": [
        "ma_50",
        "sma50",
    ],

    "sma_200": [
        "ma_200",
        "sma200",
    ],

    "ema_12": [
        "ema12",
    ],

    "ema_26": [
        "ema26",
    ],

    "rsi_14": [
        "rsi",
        "rsi14",
    ],

    "macd_signal": [
        "signal_line",
        "macd_signal_line",
        "signal",
    ],

    "macd_hist": [
        "macd_histogram",
        "macd_diff",
    ],

    "volatility_20d": [
        "volatility_20",
        "vol_20",
        "vol_20d",
    ],

    "volatility_60d": [
        "volatility_60",
        "vol_60",
        "vol_60d",
    ],

    "sharpe_60d": [
        "sharpe_60",
        "sharpe60",
    ],

    "sharpe_252d": [
        "sharpe_252",
        "sharpe252",
        "sharpe_ratio",
    ],

    "drawdown": [
        "current_drawdown",
        "drawdown_pct",
    ],

    "volume_change_1d": [
        "volume_change",
        "volume_return",
        "volume_pct_change",
    ],

    "volume_ratio_20d": [
        "volume_ratio",
        "volume_to_avg20",
        "volume_to_sma20",
    ],

    "intraday_range": [
        "daily_range",
        "range_pct",
        "high_low_range",
    ],

    "gap_return": [
        "gap",
        "overnight_return",
        "opening_gap",
    ],
}


# ============================================================
# LEAKAGE COLUMNS
# ============================================================

LEAKAGE_COLUMNS = {

    "future_return",
    "future_returns",

    "future_price",
    "future_close",

    "target",
    "label",

    "trend_target",
    "high_return_target",

    "predicted_target",

    "prediction",

    "predicted_class",

    "predicted_probability",

    "high_return_probability",

    "low_return_probability",
}


# ============================================================
# PRINT HELPER
# ============================================================

def print_section(title):

    print()

    print(
        "=" * 82
    )

    print(
        title
    )

    print(
        "=" * 82
    )

    print()


# ============================================================
# VALIDATE INPUT FILE
# ============================================================

def validate_input_file():

    print_section(
        "VALIDATING PREPROCESSOR INPUT"
    )


    print(
        "Input file:"
    )

    print(
        TRAINING_UNIVERSE_FILE
    )

    print()


    if not TRAINING_UNIVERSE_FILE.exists():

        raise FileNotFoundError(
            "training_universe.parquet "
            "not found.\n\n"
            "Run data_loader.py first."
        )


    print(
        "Input file exists."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_training_universe():

    print_section(
        "LOADING BROAD TRAINING UNIVERSE"
    )


    df = pd.read_parquet(
        TRAINING_UNIVERSE_FILE
    )


    if df.empty:

        raise ValueError(
            "training_universe.parquet "
            "is empty."
        )


    df.columns = [

        str(column)
        .strip()
        .lower()

        for column in df.columns

    ]


    print(
        f"Rows loaded   : "
        f"{len(df):,}"
    )

    print(
        f"Stocks        : "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"Columns       : "
        f"{len(df.columns):,}"
    )


    print()

    print(
        "Columns available:"
    )


    for column in df.columns:

        print(
            f"  - {column}"
        )


    return df


# ============================================================
# STANDARDIZE CORE DATA
# ============================================================

def standardize_core_columns(df):

    print_section(
        "STANDARDIZING CORE DATA"
    )


    required = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


    missing = [

        column

        for column in required

        if column not in df.columns

    ]


    if missing:

        raise KeyError(
            "Historical feature dataset "
            "is missing required base columns:\n"
            f"{missing}"
        )


    df = df.copy()


    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )


    invalid_dates = (
        df["date"]
        .isna()
        .sum()
    )


    print(
        f"Invalid date rows : "
        f"{invalid_dates:,}"
    )


    if invalid_dates > 0:

        df = df[
            df["date"].notna()
        ].copy()


    # --------------------------------------------------------
    # NUMERIC BASE COLUMNS
    # --------------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = (
        df
        .duplicated(
            subset=[
                "symbol",
                "date",
            ]
        )
        .sum()
    )


    print(
        f"Duplicate rows    : "
        f"{duplicate_count:,}"
    )


    if duplicate_count > 0:

        df = (
            df
            .drop_duplicates(
                subset=[
                    "symbol",
                    "date",
                ],
                keep="last",
            )
        )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = (
        df
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


    return df


# ============================================================
# REMOVE LEAKAGE COLUMNS
# ============================================================

def remove_leakage_columns(df):

    print_section(
        "CHECKING LEAKAGE COLUMNS"
    )


    columns_to_remove = [

        column

        for column in df.columns

        if column.lower()
        in LEAKAGE_COLUMNS

    ]


    if columns_to_remove:

        print(
            "Removing:"
        )


        for column in columns_to_remove:

            print(
                f"  - {column}"
            )


        df = df.drop(
            columns=columns_to_remove,
            errors="ignore",
        )


    else:

        print(
            "No target/future leakage "
            "columns found."
        )


    return df


# ============================================================
# APPLY COLUMN ALIASES
# ============================================================

def apply_feature_aliases(df):

    print_section(
        "CHECKING FEATURE ALIASES"
    )


    df = df.copy()


    mappings_used = []


    for required_name, aliases in (
        COLUMN_ALIASES.items()
    ):

        # Already present.

        if required_name in df.columns:

            continue


        for alias in aliases:

            if alias in df.columns:

                df[required_name] = df[
                    alias
                ]

                mappings_used.append(
                    (
                        alias,
                        required_name,
                    )
                )

                break


    if mappings_used:

        print(
            "Alias mappings used:"
        )

        print()


        for old, new in mappings_used:

            print(
                f"  {old:25s} -> {new}"
            )


    else:

        print(
            "No alias mappings required."
        )


    return df, mappings_used


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(
    numerator,
    denominator,
):

    denominator = denominator.replace(
        0,
        np.nan,
    )


    result = (
        numerator
        /
        denominator
    )


    return result


# ============================================================
# COMPUTE MISSING FEATURES
# ============================================================

def compute_missing_features(df):

    print_section(
        "DERIVING MISSING AGENT 2 FEATURES"
    )


    df = df.copy()


    original_columns = set(
        df.columns
    )


    # --------------------------------------------------------
    # GROUPED OBJECT
    # --------------------------------------------------------

    grouped = df.groupby(
        "symbol",
        sort=False,
        group_keys=False,
    )


    # ========================================================
    # RETURN FEATURES
    # ========================================================

    if "return_1d" not in df.columns:

        print(
            "Computing return_1d"
        )

        df["return_1d"] = (
            grouped["close"]
            .pct_change(
                periods=1,
                fill_method=None,
            )
        )


    if "return_5d" not in df.columns:

        print(
            "Computing return_5d"
        )

        df["return_5d"] = (
            grouped["close"]
            .pct_change(
                periods=5,
                fill_method=None,
            )
        )


    if "return_21d" not in df.columns:

        print(
            "Computing return_21d"
        )

        df["return_21d"] = (
            grouped["close"]
            .pct_change(
                periods=21,
                fill_method=None,
            )
        )


    if "return_63d" not in df.columns:

        print(
            "Computing return_63d"
        )

        df["return_63d"] = (
            grouped["close"]
            .pct_change(
                periods=63,
                fill_method=None,
            )
        )


    # ========================================================
    # SIMPLE MOVING AVERAGES
    # ========================================================

    sma_windows = [
        5,
        10,
        20,
        50,
        200,
    ]


    for window in sma_windows:

        column = (
            f"sma_{window}"
        )


        if column not in df.columns:

            print(
                f"Computing {column}"
            )


            df[column] = (
                df.groupby(
                    "symbol",
                    sort=False,
                )["close"]
                .transform(
                    lambda series:
                    series
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )


    # ========================================================
    # EXPONENTIAL MOVING AVERAGES
    # ========================================================

    for span in [
        12,
        26,
    ]:

        column = (
            f"ema_{span}"
        )


        if column not in df.columns:

            print(
                f"Computing {column}"
            )


            df[column] = (
                df.groupby(
                    "symbol",
                    sort=False,
                )["close"]
                .transform(
                    lambda series:
                    series
                    .ewm(
                        span=span,
                        adjust=False,
                        min_periods=span,
                    )
                    .mean()
                )
            )


    # ========================================================
    # PRICE / MA RATIOS
    # ========================================================

    ratio_map = {

        "close_to_sma_5": (
            "close",
            "sma_5",
        ),

        "close_to_sma_20": (
            "close",
            "sma_20",
        ),

        "close_to_sma_50": (
            "close",
            "sma_50",
        ),

        "close_to_sma_200": (
            "close",
            "sma_200",
        ),

        "sma_5_to_sma_20": (
            "sma_5",
            "sma_20",
        ),

        "sma_20_to_sma_50": (
            "sma_20",
            "sma_50",
        ),
    }


    for output_column, (
        numerator_column,
        denominator_column,
    ) in ratio_map.items():

        if output_column not in df.columns:

            print(
                f"Computing {output_column}"
            )


            df[output_column] = (
                safe_divide(
                    df[numerator_column],
                    df[denominator_column],
                )
                -
                1.0
            )


    # ========================================================
    # RSI 14
    # ========================================================

    if "rsi_14" not in df.columns:

        print(
            "Computing rsi_14"
        )


        delta = (
            df.groupby(
                "symbol",
                sort=False,
            )["close"]
            .diff()
        )


        gain = delta.clip(
            lower=0
        )


        loss = (
            -delta.clip(
                upper=0
            )
        )


        avg_gain = (
            gain.groupby(
                df["symbol"],
                sort=False,
            )
            .transform(
                lambda series:
                series
                .rolling(
                    14,
                    min_periods=14,
                )
                .mean()
            )
        )


        avg_loss = (
            loss.groupby(
                df["symbol"],
                sort=False,
            )
            .transform(
                lambda series:
                series
                .rolling(
                    14,
                    min_periods=14,
                )
                .mean()
            )
        )


        rs = safe_divide(
            avg_gain,
            avg_loss,
        )


        df["rsi_14"] = (
            100
            -
            (
                100
                /
                (
                    1
                    +
                    rs
                )
            )
        )


    # ========================================================
    # MACD
    # ========================================================

    if "macd" not in df.columns:

        print(
            "Computing macd"
        )


        if (
            "ema_12" not in df.columns
            or
            "ema_26" not in df.columns
        ):

            raise RuntimeError(
                "EMA columns unavailable "
                "for MACD calculation."
            )


        df["macd"] = (
            df["ema_12"]
            -
            df["ema_26"]
        )


    # ========================================================
    # MACD SIGNAL
    # ========================================================

    if "macd_signal" not in df.columns:

        print(
            "Computing macd_signal"
        )


        df["macd_signal"] = (
            df.groupby(
                "symbol",
                sort=False,
            )["macd"]
            .transform(
                lambda series:
                series
                .ewm(
                    span=9,
                    adjust=False,
                    min_periods=9,
                )
                .mean()
            )
        )


    # ========================================================
    # MACD HISTOGRAM
    # ========================================================

    if "macd_hist" not in df.columns:

        print(
            "Computing macd_hist"
        )


        df["macd_hist"] = (
            df["macd"]
            -
            df["macd_signal"]
        )


    # ========================================================
    # VOLATILITY
    # ========================================================

    if "volatility_20d" not in df.columns:

        print(
            "Computing volatility_20d"
        )


        df["volatility_20d"] = (
            df.groupby(
                "symbol",
                sort=False,
            )["return_1d"]
            .transform(
                lambda series:
                series
                .rolling(
                    20,
                    min_periods=20,
                )
                .std()
                *
                np.sqrt(
                    252
                )
            )
        )


    if "volatility_60d" not in df.columns:

        print(
            "Computing volatility_60d"
        )


        df["volatility_60d"] = (
            df.groupby(
                "symbol",
                sort=False,
            )["return_1d"]
            .transform(
                lambda series:
                series
                .rolling(
                    60,
                    min_periods=60,
                )
                .std()
                *
                np.sqrt(
                    252
                )
            )
        )


    # ========================================================
    # SHARPE 60
    # ========================================================

    if "sharpe_60d" not in df.columns:

        print(
            "Computing sharpe_60d"
        )


        rolling_mean_60 = (
            df.groupby(
                "symbol",
                sort=False,
            )["return_1d"]
            .transform(
                lambda series:
                series
                .rolling(
                    60,
                    min_periods=60,
                )
                .mean()
            )
        )


        rolling_std_60 = (
            df.groupby(
                "symbol",
                sort=False,
            )["return_1d"]
            .transform(
                lambda series:
                series
                .rolling(
                    60,
                    min_periods=60,
                )
                .std()
            )
        )


        df["sharpe_60d"] = (
            safe_divide(
                rolling_mean_60,
                rolling_std_60,
            )
            *
            np.sqrt(
                252
            )
        )


    # ========================================================
    # SHARPE 252
    # ========================================================

    if "sharpe_252d" not in df.columns:

        print(
            "Computing sharpe_252d"
        )


        rolling_mean_252 = (
            df.groupby(
                "symbol",
                sort=False,
            )["return_1d"]
            .transform(
                lambda series:
                series
                .rolling(
                    252,
                    min_periods=252,
                )
                .mean()
            )
        )


        rolling_std_252 = (
            df.groupby(
                "symbol",
                sort=False,
            )["return_1d"]
            .transform(
                lambda series:
                series
                .rolling(
                    252,
                    min_periods=252,
                )
                .std()
            )
        )


        df["sharpe_252d"] = (
            safe_divide(
                rolling_mean_252,
                rolling_std_252,
            )
            *
            np.sqrt(
                252
            )
        )


    # ========================================================
    # DRAWDOWN
    # ========================================================

    if "drawdown" not in df.columns:

        print(
            "Computing drawdown"
        )


        rolling_peak = (
            df.groupby(
                "symbol",
                sort=False,
            )["close"]
            .cummax()
        )


        df["drawdown"] = (
            safe_divide(
                df["close"],
                rolling_peak,
            )
            -
            1.0
        )


    # ========================================================
    # VOLUME CHANGE
    # ========================================================

    if "volume_change_1d" not in df.columns:

        print(
            "Computing volume_change_1d"
        )


        df["volume_change_1d"] = (
            df.groupby(
                "symbol",
                sort=False,
            )["volume"]
            .pct_change(
                periods=1,
                fill_method=None,
            )
        )


    # ========================================================
    # VOLUME RATIO
    # ========================================================

    if "volume_ratio_20d" not in df.columns:

        print(
            "Computing volume_ratio_20d"
        )


        volume_sma_20 = (
            df.groupby(
                "symbol",
                sort=False,
            )["volume"]
            .transform(
                lambda series:
                series
                .rolling(
                    20,
                    min_periods=20,
                )
                .mean()
            )
        )


        df["volume_ratio_20d"] = (
            safe_divide(
                df["volume"],
                volume_sma_20,
            )
        )


    # ========================================================
    # INTRADAY RANGE
    # ========================================================

    if "intraday_range" not in df.columns:

        print(
            "Computing intraday_range"
        )


        df["intraday_range"] = (
            safe_divide(
                (
                    df["high"]
                    -
                    df["low"]
                ),
                df["close"],
            )
        )


    # ========================================================
    # GAP RETURN
    # ========================================================

    if "gap_return" not in df.columns:

        print(
            "Computing gap_return"
        )


        previous_close = (
            df.groupby(
                "symbol",
                sort=False,
            )["close"]
            .shift(
                1
            )
        )


        df["gap_return"] = (
            safe_divide(
                df["open"],
                previous_close,
            )
            -
            1.0
        )


    # --------------------------------------------------------
    # REPORT DERIVED COLUMNS
    # --------------------------------------------------------

    derived_columns = [

        column

        for column in FEATURE_COLUMNS

        if (
            column not in original_columns
            and
            column in df.columns
        )

    ]


    print()

    print(
        f"Derived features: "
        f"{len(derived_columns)}"
    )


    for column in derived_columns:

        print(
            f"  + {column}"
        )


    return df, derived_columns


# ============================================================
# ENSURE NUMERIC FEATURES
# ============================================================

def convert_features_to_numeric(df):

    print_section(
        "CONVERTING MODEL FEATURES TO NUMERIC"
    )


    df = df.copy()


    missing = [

        feature

        for feature in FEATURE_COLUMNS

        if feature not in df.columns

    ]


    if missing:

        raise KeyError(
            "The following Agent 2 features "
            "still could not be created:\n"
            f"{missing}"
        )


    for feature in FEATURE_COLUMNS:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )


    print(
        f"All {len(FEATURE_COLUMNS)} "
        f"features available."
    )


    return df


# ============================================================
# REPLACE INFINITE VALUES
# ============================================================

def replace_infinite_values(df):

    print_section(
        "CHECKING INFINITE VALUES"
    )


    df = df.copy()


    matrix = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float64,
            copy=False,
        )
    )


    inf_count = int(
        np.isinf(
            matrix
        ).sum()
    )


    print(
        f"Infinite values found : "
        f"{inf_count:,}"
    )


    if inf_count > 0:

        df[
            FEATURE_COLUMNS
        ] = (
            df[
                FEATURE_COLUMNS
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


# ============================================================
# MISSING FEATURE REPORT
# ============================================================

def report_missing_features(df):

    print_section(
        "FEATURE MISSING-VALUE REPORT"
    )


    missing_counts = (
        df[
            FEATURE_COLUMNS
        ]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )


    for feature, count in (
        missing_counts.items()
    ):

        print(
            f"{feature:25s} : "
            f"{int(count):10,d}"
        )


    return missing_counts


# ============================================================
# REMOVE INCOMPLETE FEATURE ROWS
# ============================================================

def remove_incomplete_rows(df):

    print_section(
        "REMOVING INCOMPLETE FEATURE ROWS"
    )


    before_rows = len(
        df
    )


    before_stocks = (
        df[
            "symbol"
        ]
        .nunique()
    )


    df = (
        df
        .dropna(
            subset=FEATURE_COLUMNS
        )
        .copy()
    )


    after_rows = len(
        df
    )


    after_stocks = (
        df[
            "symbol"
        ]
        .nunique()
    )


    removed_rows = (
        before_rows
        -
        after_rows
    )


    print(
        f"Rows before  : "
        f"{before_rows:,}"
    )

    print(
        f"Rows after   : "
        f"{after_rows:,}"
    )

    print(
        f"Rows removed : "
        f"{removed_rows:,}"
    )

    print()

    print(
        f"Stocks before: "
        f"{before_stocks:,}"
    )

    print(
        f"Stocks after : "
        f"{after_stocks:,}"
    )


    if df.empty:

        raise ValueError(
            "All rows were removed during "
            "feature preprocessing."
        )


    return df


# ============================================================
# CREATE CHRONOLOGICAL SPLITS
# ============================================================

def assign_dataset_splits(df):

    print_section(
        "CREATING CHRONOLOGICAL SPLITS"
    )


    df = df.copy()


    df["dataset_split"] = (
        pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_mask = (

        (
            df["date"]
            >=
            TRAIN_START_DATE
        )

        &

        (
            df["date"]
            <=
            TRAIN_END_DATE
        )
    )


    df.loc[
        train_mask,
        "dataset_split",
    ] = "train"


    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    backtest_mask = (

        (
            df["date"]
            >=
            BACKTEST_START_DATE
        )

        &

        (
            df["date"]
            <=
            BACKTEST_END_DATE
        )
    )


    df.loc[
        backtest_mask,
        "dataset_split",
    ] = "backtest"


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_mask = (

        (
            df["date"]
            >=
            VALIDATION_START_DATE
        )

        &

        (
            df["date"]
            <=
            VALIDATION_END_DATE
        )
    )


    df.loc[
        validation_mask,
        "dataset_split",
    ] = "validation"


    # --------------------------------------------------------
    # OUTSIDE PERIOD
    # --------------------------------------------------------

    outside_rows = int(
        df[
            "dataset_split"
        ]
        .isna()
        .sum()
    )


    print(
        f"Rows outside configured periods: "
        f"{outside_rows:,}"
    )


    if outside_rows > 0:

        df = df[
            df[
                "dataset_split"
            ]
            .notna()
        ].copy()


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()


    for split in [
        "train",
        "backtest",
        "validation",
    ]:

        part = df[
            df[
                "dataset_split"
            ]
            ==
            split
        ]


        if part.empty:

            print(
                f"{split.upper():10s}: "
                f"0 rows"
            )

        else:

            print(
                f"{split.upper():10s}: "
                f"{len(part):10,d} rows | "
                f"{part['symbol'].nunique():4,d} stocks | "
                f"{part['date'].min().date()} "
                f"-> "
                f"{part['date'].max().date()}"
            )


    for required_split in [
        "train",
        "backtest",
        "validation",
    ]:

        if not (
            df["dataset_split"]
            ==
            required_split
        ).any():

            raise ValueError(
                f"No {required_split} "
                f"observations were created."
            )


    return df


# ============================================================
# FIT TRAIN-ONLY STANDARD SCALER
# ============================================================

def fit_training_scaler(df):

    print_section(
        "FITTING STANDARD SCALER - TRAIN DATA ONLY"
    )


    train_df = df[
        df[
            "dataset_split"
        ]
        ==
        "train"
    ]


    X_train = (
        train_df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    print(
        f"Training rows : "
        f"{X_train.shape[0]:,}"
    )

    print(
        f"Features      : "
        f"{X_train.shape[1]}"
    )


    if not np.isfinite(
        X_train
    ).all():

        raise ValueError(
            "Training feature matrix "
            "contains NaN or infinity."
        )


    scaler = StandardScaler()


    scaler.fit(
        X_train
    )


    print()

    print(
        "Scaler fitted using TRAIN "
        "data only."
    )


    return scaler


# ============================================================
# SCALE EACH SPLIT
# ============================================================

def scale_features(
    df,
    scaler,
):

    print_section(
        "SCALING TRAIN / BACKTEST / VALIDATION"
    )


    df = df.copy()


    for split in [
        "train",
        "backtest",
        "validation",
    ]:

        mask = (
            df[
                "dataset_split"
            ]
            ==
            split
        )


        X = (
            df.loc[
                mask,
                FEATURE_COLUMNS,
            ]
            .to_numpy(
                dtype=np.float64
            )
        )


        scaled = scaler.transform(
            X
        )


        if not np.isfinite(
            scaled
        ).all():

            raise ValueError(
                f"Scaled {split} "
                f"matrix contains "
                f"non-finite values."
            )


        # Store as float32 to reduce parquet size and
        # downstream sequence memory consumption.

        df.loc[
            mask,
            FEATURE_COLUMNS,
        ] = scaled.astype(
            np.float32
        )


        print(
            f"{split.upper():10s}: "
            f"{len(X):10,d} rows scaled"
        )


    # Make sure final feature dtypes are float32.

    for feature in FEATURE_COLUMNS:

        df[feature] = (
            df[feature]
            .astype(
                np.float32
            )
        )


    return df


# ============================================================
# VALIDATE TRAIN STANDARDIZATION
# ============================================================

def validate_scaled_data(df):

    print_section(
        "VALIDATING SCALED DATA"
    )


    train = df[
        df[
            "dataset_split"
        ]
        ==
        "train"
    ]


    train_matrix = (
        train[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    mean_values = np.mean(
        train_matrix,
        axis=0,
    )


    std_values = np.std(
        train_matrix,
        axis=0,
    )


    max_abs_mean = float(
        np.max(
            np.abs(
                mean_values
            )
        )
    )


    mean_std = float(
        np.mean(
            std_values
        )
    )


    print(
        f"Max absolute train mean : "
        f"{max_abs_mean:.8f}"
    )

    print(
        f"Mean train std          : "
        f"{mean_std:.8f}"
    )


    remaining_nan = int(
        df[
            FEATURE_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )


    duplicates = int(
        df
        .duplicated(
            [
                "symbol",
                "date",
            ]
        )
        .sum()
    )


    print(
        f"Remaining feature NaNs  : "
        f"{remaining_nan:,}"
    )

    print(
        f"Duplicate symbol/date   : "
        f"{duplicates:,}"
    )


    full_matrix = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32,
            copy=False,
        )
    )


    if not np.isfinite(
        full_matrix
    ).all():

        raise ValueError(
            "Final feature matrix "
            "contains NaN or infinity."
        )


    if remaining_nan > 0:

        raise ValueError(
            "Final preprocessed data "
            "contains feature NaNs."
        )


    if duplicates > 0:

        raise ValueError(
            "Duplicate symbol/date rows "
            "remain."
        )


    print()

    print(
        "Scaled dataset validation PASSED."
    )


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(df):

    rows = []


    for split in [
        "train",
        "backtest",
        "validation",
    ]:

        part = df[
            df[
                "dataset_split"
            ]
            ==
            split
        ]


        rows.append(
            {
                "dataset_split": split,

                "rows": int(
                    len(part)
                ),

                "stocks": int(
                    part[
                        "symbol"
                    ]
                    .nunique()
                ),

                "start_date": (
                    part["date"].min()
                    if not part.empty
                    else None
                ),

                "end_date": (
                    part["date"].max()
                    if not part.empty
                    else None
                ),

                "feature_count": (
                    len(
                        FEATURE_COLUMNS
                    )
                ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# CREATE FEATURE REPORT
# ============================================================

def create_feature_report(
    original_columns,
    alias_mappings,
    derived_columns,
):

    rows = []


    alias_target_map = {

        new: old

        for old, new in alias_mappings

    }


    for feature in FEATURE_COLUMNS:

        if feature in original_columns:

            source_type = (
                "existing"
            )

            source = (
                feature
            )


        elif feature in alias_target_map:

            source_type = (
                "alias"
            )

            source = (
                alias_target_map[
                    feature
                ]
            )


        elif feature in derived_columns:

            source_type = (
                "derived"
            )

            source = (
                "OHLCV / historical features"
            )


        else:

            source_type = (
                "unknown"
            )

            source = (
                ""
            )


        rows.append(
            {
                "feature": feature,

                "source_type": (
                    source_type
                ),

                "source": (
                    source
                ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    df,
    scaler,
    summary,
    feature_report,
):

    print_section(
        "SAVING PREPROCESSING OUTPUTS"
    )


    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    df.to_parquet(
        PREPROCESSED_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # SCALER PACKAGE
    # --------------------------------------------------------
    #
    # Store more than only StandardScaler so predictor.py can
    # later verify exactly which features and split were used.
    #

    scaler_package = {

        "scaler": scaler,

        "feature_columns": (
            FEATURE_COLUMNS
        ),

        "train_start_date": (
            str(
                TRAIN_START_DATE.date()
            )
        ),

        "train_end_date": (
            str(
                TRAIN_END_DATE.date()
            )
        ),

        "backtest_start_date": (
            str(
                BACKTEST_START_DATE.date()
            )
        ),

        "backtest_end_date": (
            str(
                BACKTEST_END_DATE.date()
            )
        ),

        "validation_start_date": (
            str(
                VALIDATION_START_DATE.date()
            )
        ),

        "validation_end_date": (
            str(
                VALIDATION_END_DATE.date()
            )
        ),
    }


    with open(
        SCALER_FILE,
        "wb",
    ) as file:

        pickle.dump(
            scaler_package,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


    summary.to_csv(
        PREPROCESSING_SUMMARY_FILE,
        index=False,
    )


    feature_report.to_csv(
        FEATURE_REPORT_FILE,
        index=False,
    )


    print(
        "Preprocessed dataset:"
    )

    print(
        PREPROCESSED_FILE
    )

    print()

    print(
        "Scaler package:"
    )

    print(
        SCALER_FILE
    )

    print()

    print(
        "Preprocessing summary:"
    )

    print(
        PREPROCESSING_SUMMARY_FILE
    )

    print()

    print(
        "Feature report:"
    )

    print(
        FEATURE_REPORT_FILE
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_preprocessor():

    print_section(
        "AGENT 2 - CORRECTED BROAD-UNIVERSE PREPROCESSOR"
    )


    print(
        "Input:"
    )

    print(
        "  Broad historical NIFTY-500 "
        "training universe"
    )

    print()

    print(
        "Feature policy:"
    )

    print(
        "  Existing feature -> use it"
    )

    print(
        "  Alternate name   -> map it"
    )

    print(
        "  Missing feature  -> derive it "
        "from historical OHLCV"
    )

    print()

    print(
        "Scaling policy:"
    )

    print(
        "  Fit StandardScaler on TRAIN "
        "only"
    )


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    validate_input_file()


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    df = (
        load_training_universe()
    )


    original_columns = set(
        df.columns
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    df = (
        standardize_core_columns(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    df = (
        remove_leakage_columns(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    (
        df,
        alias_mappings,
    ) = (
        apply_feature_aliases(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    (
        df,
        derived_columns,
    ) = (
        compute_missing_features(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    df = (
        convert_features_to_numeric(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    df = (
        replace_infinite_values(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    report_missing_features(
        df
    )


    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    df = (
        remove_incomplete_rows(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    df = (
        assign_dataset_splits(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 12
    # --------------------------------------------------------

    scaler = (
        fit_training_scaler(
            df
        )
    )


    # --------------------------------------------------------
    # STEP 13
    # --------------------------------------------------------

    df = (
        scale_features(
            df,
            scaler,
        )
    )


    # --------------------------------------------------------
    # STEP 14
    # --------------------------------------------------------

    validate_scaled_data(
        df
    )


    # --------------------------------------------------------
    # STEP 15
    # --------------------------------------------------------

    summary = (
        create_summary(
            df
        )
    )


    feature_report = (
        create_feature_report(
            original_columns,
            alias_mappings,
            derived_columns,
        )
    )


    # --------------------------------------------------------
    # STEP 16
    # --------------------------------------------------------

    save_outputs(
        df,
        scaler,
        summary,
        feature_report,
    )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "PREPROCESSING COMPLETED SUCCESSFULLY"
    )


    print(
        f"Final rows     : "
        f"{len(df):,}"
    )

    print(
        f"Final stocks   : "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"Model features : "
        f"{len(FEATURE_COLUMNS)}"
    )


    print()

    print(
        "FEATURE SOURCES"
    )

    print(
        "-" * 60
    )


    existing_count = int(
        (
            feature_report[
                "source_type"
            ]
            ==
            "existing"
        ).sum()
    )


    alias_count = int(
        (
            feature_report[
                "source_type"
            ]
            ==
            "alias"
        ).sum()
    )


    derived_count = int(
        (
            feature_report[
                "source_type"
            ]
            ==
            "derived"
        ).sum()
    )


    print(
        f"Existing Agent 1 features : "
        f"{existing_count}"
    )

    print(
        f"Alias-mapped features     : "
        f"{alias_count}"
    )

    print(
        f"Derived features          : "
        f"{derived_count}"
    )


    print()

    print(
        "SPLIT SUMMARY"
    )

    print(
        "-" * 60
    )


    for _, row in summary.iterrows():

        print(
            f"{row['dataset_split'].upper():10s} "
            f"{int(row['rows']):10,d} rows | "
            f"{int(row['stocks']):4,d} stocks"
        )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Target generation has NOT been "
        "performed here."
    )

    print(
        "The next stage is "
        "sequence_builder.py."
    )


    return {

        "data": df,

        "scaler": scaler,

        "summary": summary,

        "feature_report": (
            feature_report
        ),

        "feature_columns": (
            FEATURE_COLUMNS
        ),
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_preprocessor()