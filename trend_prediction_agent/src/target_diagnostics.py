"""
Agent 2 - Target Definition Diagnostics
========================================

Purpose
-------
Test whether the current trend-prediction target is the main reason
for weak temporal generalization.

Current implementation target:
    21 trading days forward
    HIGH_RETURN if future return >= +2%

This exact 21-day / +2% rule is an implementation choice rather than
a clearly specified numerical rule from the base paper.

This script tests:

ABSOLUTE TARGETS
----------------
5d  > 0%
5d  > 1%
10d > 0%
10d > 2%
21d > 0%
21d > 2%     <- current target
21d > 5%
63d > 0%
63d > 5%

RELATIVE / CROSS-SECTIONAL TARGETS
----------------------------------
Top 30% of stocks by future return at each prediction date:
5d, 10d, 21d, 63d

Research protocol
-----------------
Training fitting period:
    before 2018-07-04

Backtest:
    2019-2024

Final validation:
    2025

CRITICAL:
2025 is NOT evaluated in this script.

Candidate target selection must be based only on historical
TRAIN/BACKTEST results.

After choosing a target, 2025 may be evaluated exactly once in
a separate final-validation stage.

Models used
-----------
1. Logistic Regression
2. Random Forest

These models are diagnostic probes. They are not replacements for
the paper's Dilated-LSTM + Transformer architecture.

The goal is to determine whether each target contains temporally
stable predictive information.
"""

from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

TREND_AGENT_ROOT = CURRENT_FILE.parents[1]


if str(TREND_AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(TREND_AGENT_ROOT),
    )


PROCESSED_DIR = (
    TREND_AGENT_ROOT
    / "data"
    / "processed"
)


REPORTS_DIR = (
    TREND_AGENT_ROOT
    / "reports"
)


RESULTS_DIR = (
    REPORTS_DIR
    / "results"
)


# ============================================================
# INPUT FILES
# ============================================================

PREPROCESSED_FILE = (
    PROCESSED_DIR
    / "preprocessed_trend_data.parquet"
)


TRAINING_UNIVERSE_FILE = (
    PROCESSED_DIR
    / "training_universe.parquet"
)


EVALUATION_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "trend_evaluation_predictions.parquet"
)


# ============================================================
# OUTPUT FILES
# ============================================================

TARGET_METRICS_FILE = (
    REPORTS_DIR
    / "target_candidate_metrics.csv"
)


TARGET_YEAR_METRICS_FILE = (
    REPORTS_DIR
    / "target_candidate_year_metrics.csv"
)


TARGET_PREVALENCE_FILE = (
    REPORTS_DIR
    / "target_candidate_prevalence.csv"
)


TARGET_RANKING_FILE = (
    REPORTS_DIR
    / "target_candidate_ranking.csv"
)


TARGET_SUMMARY_FILE = (
    REPORTS_DIR
    / "target_diagnostics_summary.json"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [

    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",

    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",

    "ema_12",
    "ema_26",

    "close_to_sma_5",
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_sma_200",

    "sma_5_to_sma_20",
    "sma_20_to_sma_50",

    "rsi_14",

    "macd",
    "macd_signal",
    "macd_hist",

    "volatility_20d",
    "volatility_60d",

    "sharpe_60d",
    "sharpe_252d",

    "drawdown",

    "volume_change_1d",
    "volume_ratio_20d",

    "intraday_range",
    "gap_return",
]


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50


# EXACT CUT-OFF USED BY THE COMPLETED DEEP TRAINER

FIT_CUTOFF = pd.Timestamp(
    "2018-07-04"
)


BACKTEST_START = pd.Timestamp(
    "2019-01-01"
)


BACKTEST_END = pd.Timestamp(
    "2024-12-31"
)


FINAL_VALIDATION_START = pd.Timestamp(
    "2025-01-01"
)


# ------------------------------------------------------------
# Absolute target candidates
# ------------------------------------------------------------

ABSOLUTE_TARGETS = [

    (5, 0.00),
    (5, 0.01),

    (10, 0.00),
    (10, 0.02),

    (21, 0.00),
    (21, 0.02),
    (21, 0.05),

    (63, 0.00),
    (63, 0.05),
]


# ------------------------------------------------------------
# Relative targets
# ------------------------------------------------------------

RELATIVE_HORIZONS = [

    5,
    10,
    21,
    63,
]


RELATIVE_TOP_FRACTION = 0.30


# ============================================================
# DISPLAY
# ============================================================

def print_section(title):

    print()

    print(
        "=" * 95
    )

    print(
        title
    )

    print(
        "=" * 95
    )

    print()


def safe_float(value):

    if pd.isna(
        value
    ):

        return None

    return float(
        value
    )


# ============================================================
# FILE SCHEMA HELPERS
# ============================================================

def parquet_columns(path):

    return (
        pq.ParquetFile(
            path
        )
        .schema
        .names
    )


def find_column(
    columns,
    candidates,
):

    for candidate in candidates:

        if candidate in columns:

            return candidate

    return None


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs():

    print_section(
        "VALIDATING TARGET-DIAGNOSTIC INPUTS"
    )


    required = [

        PREPROCESSED_FILE,
        EVALUATION_PREDICTIONS_FILE,
    ]


    for path in required:

        print(
            path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Required file missing:\n{path}"
            )


    print()

    print(
        "Required files found."
    )

    print(
        f"Exact model fitting cutoff : "
        f"{FIT_CUTOFF.date()}"
    )

    print(
        "Final 2025 validation      : PROTECTED"
    )


# ============================================================
# LOAD SEQUENCE-ELIGIBLE ANCHORS
# ============================================================

def load_prediction_anchors():

    print_section(
        "LOADING SEQUENCE-ELIGIBLE PREDICTION DATES"
    )


    columns = [

        "sequence_id",
        "symbol",
        "dataset_split",
        "prediction_date",
    ]


    anchors = pd.read_parquet(

        EVALUATION_PREDICTIONS_FILE,

        columns=columns,
    )


    anchors[
        "symbol"
    ] = (
        anchors[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    anchors[
        "dataset_split"
    ] = (
        anchors[
            "dataset_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    anchors[
        "prediction_date"
    ] = pd.to_datetime(
        anchors[
            "prediction_date"
        ]
    )


    duplicate_count = anchors.duplicated(
        subset=[
            "symbol",
            "prediction_date",
        ]
    ).sum()


    print(
        f"Anchors          : "
        f"{len(anchors):,}"
    )

    print(
        f"Stocks           : "
        f"{anchors['symbol'].nunique():,}"
    )

    print(
        f"Duplicate keys   : "
        f"{duplicate_count:,}"
    )


    if duplicate_count > 0:

        raise ValueError(
            "Duplicate anchor symbol/date rows found."
        )


    # --------------------------------------------------------
    # 2025 stays present in the source only so we can prove
    # that it exists, but it is excluded before modelling.
    # --------------------------------------------------------

    validation_rows = (
        anchors[
            "prediction_date"
        ]
        >=
        FINAL_VALIDATION_START
    ).sum()


    print(
        f"Protected 2025 anchors : "
        f"{validation_rows:,}"
    )


    return anchors


# ============================================================
# LOAD FEATURES
# ============================================================

def load_feature_data():

    print_section(
        "LOADING 30 PREPROCESSED FEATURES"
    )


    columns = parquet_columns(
        PREPROCESSED_FILE
    )


    symbol_column = find_column(
        columns,
        [
            "symbol",
            "Symbol",
            "SYMBOL",
        ],
    )


    date_column = find_column(
        columns,
        [
            "date",
            "Date",
            "DATE",
        ],
    )


    close_column = find_column(
        columns,
        [
            "close",
            "Close",
            "CLOSE",
            "adj_close",
            "Adj Close",
        ],
    )


    if symbol_column is None:

        raise KeyError(
            "Could not locate symbol column."
        )


    if date_column is None:

        raise KeyError(
            "Could not locate date column."
        )


    missing_features = [

        feature

        for feature in FEATURE_COLUMNS

        if feature not in columns
    ]


    if missing_features:

        raise KeyError(
            "Missing features:\n"
            f"{missing_features}"
        )


    needed = [

        symbol_column,
        date_column,

        *FEATURE_COLUMNS,
    ]


    if close_column is not None:

        needed.append(
            close_column
        )


    needed = list(
        dict.fromkeys(
            needed
        )
    )


    df = pd.read_parquet(
        PREPROCESSED_FILE,
        columns=needed,
    )


    rename_map = {

        symbol_column: "symbol",

        date_column: "date",
    }


    if close_column is not None:

        rename_map[
            close_column
        ] = "close"


    df = df.rename(
        columns=rename_map
    )


    df[
        "symbol"
    ] = (
        df[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ]
    )


    print(
        f"Rows       : "
        f"{len(df):,}"
    )

    print(
        f"Stocks     : "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"Features   : "
        f"{len(FEATURE_COLUMNS)}"
    )


    # --------------------------------------------------------
    # If close was removed from the preprocessed file,
    # recover it from training_universe.parquet.
    # --------------------------------------------------------

    if "close" not in df.columns:

        print()

        print(
            "Close price not present in "
            "preprocessed file."
        )

        print(
            "Attempting recovery from "
            "training_universe.parquet..."
        )


        if not TRAINING_UNIVERSE_FILE.exists():

            raise FileNotFoundError(
                "Close price is required, but it was "
                "not found in preprocessed data and "
                "training_universe.parquet is missing."
            )


        raw_columns = parquet_columns(
            TRAINING_UNIVERSE_FILE
        )


        raw_symbol = find_column(
            raw_columns,
            [
                "symbol",
                "Symbol",
                "SYMBOL",
            ],
        )


        raw_date = find_column(
            raw_columns,
            [
                "date",
                "Date",
                "DATE",
            ],
        )


        raw_close = find_column(
            raw_columns,
            [
                "close",
                "Close",
                "CLOSE",
                "adj_close",
                "Adj Close",
            ],
        )


        if (
            raw_symbol is None
            or
            raw_date is None
            or
            raw_close is None
        ):

            raise KeyError(
                "Could not identify symbol/date/close "
                "inside training_universe.parquet."
            )


        prices = pd.read_parquet(

            TRAINING_UNIVERSE_FILE,

            columns=[
                raw_symbol,
                raw_date,
                raw_close,
            ],
        )


        prices = prices.rename(
            columns={
                raw_symbol: "symbol",
                raw_date: "date",
                raw_close: "close",
            }
        )


        prices[
            "symbol"
        ] = (
            prices[
                "symbol"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )


        prices[
            "date"
        ] = pd.to_datetime(
            prices[
                "date"
            ]
        )


        df = df.merge(

            prices,

            how="left",

            on=[
                "symbol",
                "date",
            ],

            validate="one_to_one",
        )


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    df[
        "close"
    ] = pd.to_numeric(
        df[
            "close"
        ],
        errors="coerce",
    )


    invalid_close = (
        ~np.isfinite(
            df[
                "close"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )
    ).sum()


    if invalid_close > 0:

        raise ValueError(
            f"Invalid close prices: "
            f"{invalid_close:,}"
        )


    feature_matrix = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    if not np.isfinite(
        feature_matrix
    ).all():

        raise ValueError(
            "Feature matrix contains "
            "NaN or infinity."
        )


    duplicates = df.duplicated(
        subset=[
            "symbol",
            "date",
        ]
    ).sum()


    if duplicates > 0:

        raise ValueError(
            f"Duplicate feature rows: "
            f"{duplicates:,}"
        )


    df = df.sort_values(
        [
            "symbol",
            "date",
        ]
    ).reset_index(
        drop=True
    )


    print(
        "Feature and close-price validation PASSED."
    )


    return df


# ============================================================
# CREATE FUTURE RETURN CACHE
# ============================================================

def create_future_return_cache(
    feature_data,
):

    print_section(
        "GENERATING FUTURE RETURN CACHE"
    )


    horizons = sorted(
        set(
            [
                horizon
                for horizon, threshold
                in ABSOLUTE_TARGETS
            ]
            +
            RELATIVE_HORIZONS
        )
    )


    working = feature_data[
        [
            "symbol",
            "date",
            "close",
        ]
    ].copy()


    grouped = working.groupby(
        "symbol",
        sort=False,
        observed=True,
    )


    cache = {}


    for horizon in horizons:

        print(
            f"Generating {horizon}-day future return..."
        )


        future_close = grouped[
            "close"
        ].shift(
            -horizon
        )


        future_date = grouped[
            "date"
        ].shift(
            -horizon
        )


        future_return = (

            future_close
            /
            working[
                "close"
            ]

            -

            1.0
        )


        cache[
            horizon
        ] = pd.DataFrame(
            {

                "symbol": (
                    working[
                        "symbol"
                    ]
                ),

                "date": (
                    working[
                        "date"
                    ]
                ),

                "target_date": (
                    future_date
                ),

                "future_return": (
                    future_return
                ),
            }
        )


    print()

    print(
        f"Cached horizons: {horizons}"
    )


    return cache


# ============================================================
# BUILD MASTER ANCHOR DATASET
# ============================================================

def build_master_dataset(
    anchors,
    feature_data,
):

    print_section(
        "BUILDING MASTER DIAGNOSTIC DATASET"
    )


    features = feature_data[
        [
            "symbol",
            "date",
            *FEATURE_COLUMNS,
        ]
    ].rename(
        columns={
            "date": "prediction_date",
        }
    )


    dataset = anchors.merge(

        features,

        how="left",

        on=[
            "symbol",
            "prediction_date",
        ],

        validate="one_to_one",
    )


    missing = (
        dataset[
            FEATURE_COLUMNS
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )


    print(
        f"Anchor rows          : "
        f"{len(anchors):,}"
    )

    print(
        f"Merged rows          : "
        f"{len(dataset):,}"
    )

    print(
        f"Missing feature rows : "
        f"{missing:,}"
    )


    if (
        len(dataset)
        !=
        len(anchors)
    ):

        raise ValueError(
            "Master merge changed row count."
        )


    if missing > 0:

        raise ValueError(
            "Some anchor dates have missing features."
        )


    # --------------------------------------------------------
    # REMOVE 2025 BEFORE ANY MODEL EXPERIMENT
    # --------------------------------------------------------

    research = dataset[
        dataset[
            "prediction_date"
        ]
        <
        FINAL_VALIDATION_START
    ].copy()


    protected = dataset[
        dataset[
            "prediction_date"
        ]
        >=
        FINAL_VALIDATION_START
    ].copy()


    print()

    print(
        f"Research rows (<2025) : "
        f"{len(research):,}"
    )

    print(
        f"Protected 2025 rows   : "
        f"{len(protected):,}"
    )


    return research


# ============================================================
# ATTACH FUTURE RETURNS
# ============================================================

def attach_horizon_data(
    research_data,
    future_cache,
    horizon,
):

    future = future_cache[
        horizon
    ].rename(
        columns={
            "date": "prediction_date",
        }
    )


    dataset = research_data.merge(

        future,

        how="left",

        on=[
            "symbol",
            "prediction_date",
        ],

        validate="one_to_one",
    )


    # --------------------------------------------------------
    # Future target must exist
    # --------------------------------------------------------

    dataset = dataset[
        dataset[
            "future_return"
        ]
        .notna()
    ].copy()


    dataset = dataset[
        dataset[
            "target_date"
        ]
        .notna()
    ].copy()


    # --------------------------------------------------------
    # Prevent split boundary leakage
    # --------------------------------------------------------

    train_mask = (
        dataset[
            "dataset_split"
        ]
        ==
        "train"
    )


    backtest_mask = (
        dataset[
            "dataset_split"
        ]
        ==
        "backtest"
    )


    valid_train_target = (

        train_mask

        &

        (
            dataset[
                "target_date"
            ]
            <
            BACKTEST_START
        )
    )


    valid_backtest_target = (

        backtest_mask

        &

        (
            dataset[
                "target_date"
            ]
            <=
            BACKTEST_END
        )
    )


    dataset = dataset[
        valid_train_target
        |
        valid_backtest_target
    ].copy()


    return dataset


# ============================================================
# TARGET BUILDERS
# ============================================================

def build_absolute_target(
    dataset,
    threshold,
):

    result = dataset.copy()


    result[
        "target"
    ] = (

        result[
            "future_return"
        ]

        >=

        threshold
    ).astype(
        np.int8
    )


    return result


def build_relative_target(
    dataset,
    top_fraction,
):

    result = dataset.copy()


    # --------------------------------------------------------
    # Rank future returns cross-sectionally on each
    # prediction date.
    #
    # Higher percentile = stronger future return.
    # --------------------------------------------------------

    result[
        "cross_section_percentile"
    ] = (

        result.groupby(
            "prediction_date",
            observed=True,
        )[
            "future_return"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )


    cutoff = (
        1.0
        -
        top_fraction
    )


    result[
        "target"
    ] = (

        result[
            "cross_section_percentile"
        ]

        >

        cutoff
    ).astype(
        np.int8
    )


    return result


# ============================================================
# EXACT TRAIN/BACKTEST PARTITION
# ============================================================

def create_model_partitions(
    dataset,
):

    # --------------------------------------------------------
    # Model fitting:
    #
    # Prediction date before exact DL cutoff AND target date
    # before exact cutoff.
    #
    # This provides the same embargo principle used by the
    # completed neural-network trainer.
    # --------------------------------------------------------

    fitting = dataset[

        (
            dataset[
                "dataset_split"
            ]
            ==
            "train"
        )

        &

        (
            dataset[
                "prediction_date"
            ]
            <
            FIT_CUTOFF
        )

        &

        (
            dataset[
                "target_date"
            ]
            <
            FIT_CUTOFF
        )

    ].copy()


    backtest = dataset[

        (
            dataset[
                "dataset_split"
            ]
            ==
            "backtest"
        )

        &

        (
            dataset[
                "prediction_date"
            ]
            >=
            BACKTEST_START
        )

        &

        (
            dataset[
                "prediction_date"
            ]
            <=
            BACKTEST_END
        )

    ].copy()


    return (
        fitting,
        backtest,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probability,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )


    probability = np.asarray(
        probability,
        dtype=np.float64,
    )


    prediction = (

        probability

        >=

        DEFAULT_THRESHOLD
    ).astype(
        np.int32
    )


    cm = confusion_matrix(
        y_true,
        prediction,
        labels=[
            0,
            1,
        ],
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    specificity = (

        tn
        /
        (
            tn
            +
            fp
        )

        if (
            tn
            +
            fp
        ) > 0

        else np.nan
    )


    try:

        auc = roc_auc_score(
            y_true,
            probability,
        )

    except ValueError:

        auc = np.nan


    try:

        ap = average_precision_score(
            y_true,
            probability,
        )

    except ValueError:

        ap = np.nan


    try:

        brier = brier_score_loss(
            y_true,
            probability,
        )

    except ValueError:

        brier = np.nan


    return {

        "samples": int(
            len(
                y_true
            )
        ),

        "actual_high_rate": float(
            np.mean(
                y_true
            )
        ),

        "predicted_high_rate": float(
            np.mean(
                prediction
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "roc_auc": safe_float(
            auc
        ),

        "average_precision": safe_float(
            ap
        ),

        "brier_score": safe_float(
            brier
        ),

        "tn": int(
            tn
        ),

        "fp": int(
            fp
        ),

        "fn": int(
            fn
        ),

        "tp": int(
            tp
        ),
    }


# ============================================================
# MODELS
# ============================================================

def create_models():

    return {

        "logistic_regression":

            LogisticRegression(

                solver="lbfgs",

                max_iter=1000,

                class_weight="balanced",

                random_state=RANDOM_STATE,
            ),


        "random_forest":

            RandomForestClassifier(

                n_estimators=100,

                max_depth=12,

                min_samples_split=100,

                min_samples_leaf=30,

                max_features="sqrt",

                class_weight="balanced_subsample",

                n_jobs=-1,

                random_state=RANDOM_STATE,
            ),
    }


# ============================================================
# YEARLY PREVALENCE
# ============================================================

def calculate_prevalence(
    dataset,
    candidate_name,
    target_type,
    horizon,
    threshold,
):

    records = []


    working = dataset.copy()


    working[
        "year"
    ] = (
        working[
            "prediction_date"
        ]
        .dt
        .year
    )


    for year, group in working.groupby(
        "year",
        sort=True,
    ):

        records.append(
            {

                "candidate": candidate_name,

                "target_type": target_type,

                "horizon": horizon,

                "threshold": threshold,

                "year": int(
                    year
                ),

                "samples": int(
                    len(
                        group
                    )
                ),

                "stocks": int(
                    group[
                        "symbol"
                    ]
                    .nunique()
                ),

                "positive_rate": float(
                    group[
                        "target"
                    ]
                    .mean()
                ),

                "mean_future_return": float(
                    group[
                        "future_return"
                    ]
                    .mean()
                ),

                "std_future_return": float(
                    group[
                        "future_return"
                    ]
                    .std()
                ),
            }
        )


    return records


# ============================================================
# EVALUATE ONE TARGET CANDIDATE
# ============================================================

def evaluate_candidate(
    dataset,
    candidate_name,
    target_type,
    horizon,
    threshold,
):

    print_section(
        f"TARGET CANDIDATE: {candidate_name}"
    )


    fitting, backtest = (
        create_model_partitions(
            dataset
        )
    )


    print(
        f"Fitting samples   : "
        f"{len(fitting):,}"
    )

    print(
        f"Fitting HIGH rate : "
        f"{fitting['target'].mean():.2%}"
    )

    print(
        f"Backtest samples  : "
        f"{len(backtest):,}"
    )

    print(
        f"Backtest HIGH rate: "
        f"{backtest['target'].mean():.2%}"
    )


    if len(
        np.unique(
            fitting[
                "target"
            ]
        )
    ) < 2:

        print(
            "[SKIP] Fitting target has only "
            "one class."
        )

        return [], []


    X_train = (
        fitting[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y_train = (
        fitting[
            "target"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )


    X_backtest = (
        backtest[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y_backtest = (
        backtest[
            "target"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )


    metric_records = []

    year_records = []


    models = create_models()


    for model_name, model in models.items():

        print()

        print(
            f"Training {model_name}..."
        )


        start = time.time()


        model.fit(
            X_train,
            y_train,
        )


        training_seconds = (
            time.time()
            -
            start
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_probability = (
            model.predict_proba(
                X_train
            )[
                :,
                1
            ]
        )


        train_metrics = calculate_metrics(
            y_train,
            train_probability,
        )


        # ----------------------------------------------------
        # BACKTEST
        # ----------------------------------------------------

        backtest_probability = (
            model.predict_proba(
                X_backtest
            )[
                :,
                1
            ]
        )


        backtest_metrics = calculate_metrics(
            y_backtest,
            backtest_probability,
        )


        print(
            f"  train AUC    : "
            f"{train_metrics['roc_auc']:.4f}"
        )

        print(
            f"  backtest AUC : "
            f"{backtest_metrics['roc_auc']:.4f}"
        )

        print(
            f"  training sec : "
            f"{training_seconds:.2f}"
        )


        metric_records.append(
            {

                "candidate": candidate_name,

                "target_type": target_type,

                "horizon": horizon,

                "threshold": threshold,

                "model": model_name,

                "fit_samples": int(
                    len(
                        fitting
                    )
                ),

                "fit_positive_rate": float(
                    y_train.mean()
                ),

                "backtest_samples": int(
                    len(
                        backtest
                    )
                ),

                "backtest_positive_rate": float(
                    y_backtest.mean()
                ),

                "training_seconds": float(
                    training_seconds
                ),

                "train_accuracy": (
                    train_metrics[
                        "accuracy"
                    ]
                ),

                "train_f1": (
                    train_metrics[
                        "f1"
                    ]
                ),

                "train_roc_auc": (
                    train_metrics[
                        "roc_auc"
                    ]
                ),

                "backtest_accuracy": (
                    backtest_metrics[
                        "accuracy"
                    ]
                ),

                "backtest_f1": (
                    backtest_metrics[
                        "f1"
                    ]
                ),

                "backtest_roc_auc": (
                    backtest_metrics[
                        "roc_auc"
                    ]
                ),

                "backtest_average_precision": (
                    backtest_metrics[
                        "average_precision"
                    ]
                ),

                "backtest_brier": (
                    backtest_metrics[
                        "brier_score"
                    ]
                ),
            }
        )


        # ----------------------------------------------------
        # YEAR-BY-YEAR BACKTEST AUC
        # ----------------------------------------------------

        temp = backtest.copy()


        temp[
            "probability"
        ] = (
            backtest_probability
        )


        temp[
            "year"
        ] = (
            temp[
                "prediction_date"
            ]
            .dt
            .year
        )


        for year, group in temp.groupby(
            "year",
            sort=True,
        ):

            metrics = calculate_metrics(

                group[
                    "target"
                ],

                group[
                    "probability"
                ],
            )


            year_records.append(
                {

                    "candidate": candidate_name,

                    "target_type": target_type,

                    "horizon": horizon,

                    "threshold": threshold,

                    "model": model_name,

                    "year": int(
                        year
                    ),

                    "samples": int(
                        len(
                            group
                        )
                    ),

                    "positive_rate": float(
                        group[
                            "target"
                        ]
                        .mean()
                    ),

                    "roc_auc": (
                        metrics[
                            "roc_auc"
                        ]
                    ),

                    "accuracy": (
                        metrics[
                            "accuracy"
                        ]
                    ),

                    "f1": (
                        metrics[
                            "f1"
                        ]
                    ),

                    "average_precision": (
                        metrics[
                            "average_precision"
                        ]
                    ),
                }
            )


    return (
        metric_records,
        year_records,
    )


# ============================================================
# BUILD RANKING
# ============================================================

def build_ranking(
    metrics_df,
    year_df,
):

    print_section(
        "BUILDING TARGET RANKING USING 2019-2024 ONLY"
    )


    ranking_records = []


    for (
        candidate,
        model,
    ), group in year_df.groupby(
        [
            "candidate",
            "model",
        ],
        sort=True,
    ):

        valid_auc = (
            group[
                "roc_auc"
            ]
            .dropna()
        )


        if valid_auc.empty:

            continue


        mean_year_auc = float(
            valid_auc.mean()
        )


        std_year_auc = float(
            valid_auc.std(
                ddof=0
            )
        )


        minimum_year_auc = float(
            valid_auc.min()
        )


        maximum_year_auc = float(
            valid_auc.max()
        )


        years_above_050 = int(
            (
                valid_auc
                >
                0.50
            )
            .sum()
        )


        years_above_052 = int(
            (
                valid_auc
                >
                0.52
            )
            .sum()
        )


        years_above_055 = int(
            (
                valid_auc
                >
                0.55
            )
            .sum()
        )


        # ----------------------------------------------------
        # Conservative stability score.
        #
        # Reward mean AUC, penalize unstable targets.
        # ----------------------------------------------------

        stability_score = (

            mean_year_auc

            -

            0.50
            *
            std_year_auc
        )


        aggregate_row = metrics_df[

            (
                metrics_df[
                    "candidate"
                ]
                ==
                candidate
            )

            &

            (
                metrics_df[
                    "model"
                ]
                ==
                model
            )

        ]


        aggregate_auc = (

            float(
                aggregate_row.iloc[
                    0
                ][
                    "backtest_roc_auc"
                ]
            )

            if not aggregate_row.empty

            else np.nan
        )


        ranking_records.append(
            {

                "candidate": candidate,

                "model": model,

                "aggregate_backtest_auc": (
                    aggregate_auc
                ),

                "mean_year_auc": (
                    mean_year_auc
                ),

                "std_year_auc": (
                    std_year_auc
                ),

                "minimum_year_auc": (
                    minimum_year_auc
                ),

                "maximum_year_auc": (
                    maximum_year_auc
                ),

                "years_above_050": (
                    years_above_050
                ),

                "years_above_052": (
                    years_above_052
                ),

                "years_above_055": (
                    years_above_055
                ),

                "stability_score": (
                    stability_score
                ),
            }
        )


    ranking = pd.DataFrame(
        ranking_records
    )


    ranking = ranking.sort_values(

        [
            "stability_score",
            "aggregate_backtest_auc",
        ],

        ascending=[
            False,
            False,
        ],

    ).reset_index(
        drop=True
    )


    ranking[
        "rank"
    ] = (
        np.arange(
            1,
            len(
                ranking
            )
            +
            1
        )
    )


    display = ranking[
        [
            "rank",
            "candidate",
            "model",
            "aggregate_backtest_auc",
            "mean_year_auc",
            "std_year_auc",
            "minimum_year_auc",
            "years_above_052",
            "stability_score",
        ]
    ]


    print(
        display.to_string(
            index=False
        )
    )


    return ranking


# ============================================================
# MAIN
# ============================================================

def run_target_diagnostics():

    print_section(
        "AGENT 2 - TARGET DEFINITION DIAGNOSTICS"
    )


    print(
        "Research rule:"
    )

    print(
        "2025 WILL NOT be used to compare "
        "or select target candidates."
    )

    print()

    print(
        "Candidate ranking will use "
        "2019-2024 only."
    )


    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # 1. INPUT CHECK
    # --------------------------------------------------------

    validate_inputs()


    # --------------------------------------------------------
    # 2. ANCHORS
    # --------------------------------------------------------

    anchors = (
        load_prediction_anchors()
    )


    # --------------------------------------------------------
    # 3. FEATURE DATA
    # --------------------------------------------------------

    feature_data = (
        load_feature_data()
    )


    # --------------------------------------------------------
    # 4. FUTURE RETURN CACHE
    # --------------------------------------------------------

    future_cache = (
        create_future_return_cache(
            feature_data
        )
    )


    # --------------------------------------------------------
    # 5. MASTER DATA
    # --------------------------------------------------------

    research_data = (
        build_master_dataset(
            anchors,
            feature_data,
        )
    )


    all_metric_records = []

    all_year_records = []

    all_prevalence_records = []


    # ========================================================
    # ABSOLUTE TARGETS
    # ========================================================

    for horizon, threshold in ABSOLUTE_TARGETS:

        candidate_name = (

            f"absolute_"
            f"{horizon}d_"
            f"{threshold * 100:.0f}pct"
        )


        horizon_data = (
            attach_horizon_data(

                research_data,

                future_cache,

                horizon,
            )
        )


        candidate_data = (
            build_absolute_target(

                horizon_data,

                threshold,
            )
        )


        prevalence_records = (
            calculate_prevalence(

                candidate_data,

                candidate_name,

                "absolute",

                horizon,

                threshold,
            )
        )


        all_prevalence_records.extend(
            prevalence_records
        )


        metrics, years = (
            evaluate_candidate(

                candidate_data,

                candidate_name,

                "absolute",

                horizon,

                threshold,
            )
        )


        all_metric_records.extend(
            metrics
        )


        all_year_records.extend(
            years
        )


    # ========================================================
    # RELATIVE TARGETS
    # ========================================================

    for horizon in RELATIVE_HORIZONS:

        candidate_name = (

            f"relative_"
            f"{horizon}d_"
            f"top{int(RELATIVE_TOP_FRACTION * 100)}"
        )


        horizon_data = (
            attach_horizon_data(

                research_data,

                future_cache,

                horizon,
            )
        )


        candidate_data = (
            build_relative_target(

                horizon_data,

                RELATIVE_TOP_FRACTION,
            )
        )


        prevalence_records = (
            calculate_prevalence(

                candidate_data,

                candidate_name,

                "relative",

                horizon,

                RELATIVE_TOP_FRACTION,
            )
        )


        all_prevalence_records.extend(
            prevalence_records
        )


        metrics, years = (
            evaluate_candidate(

                candidate_data,

                candidate_name,

                "relative",

                horizon,

                RELATIVE_TOP_FRACTION,
            )
        )


        all_metric_records.extend(
            metrics
        )


        all_year_records.extend(
            years
        )


    # ========================================================
    # DATAFRAMES
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metric_records
    )


    year_df = pd.DataFrame(
        all_year_records
    )


    prevalence_df = pd.DataFrame(
        all_prevalence_records
    )


    # ========================================================
    # RANK TARGETS
    # ========================================================

    ranking_df = build_ranking(
        metrics_df,
        year_df,
    )


    # ========================================================
    # SAVE
    # ========================================================

    print_section(
        "SAVING TARGET DIAGNOSTIC RESULTS"
    )


    metrics_df.to_csv(
        TARGET_METRICS_FILE,
        index=False,
    )


    year_df.to_csv(
        TARGET_YEAR_METRICS_FILE,
        index=False,
    )


    prevalence_df.to_csv(
        TARGET_PREVALENCE_FILE,
        index=False,
    )


    ranking_df.to_csv(
        TARGET_RANKING_FILE,
        index=False,
    )


    summary = {

        "fit_cutoff": str(
            FIT_CUTOFF.date()
        ),

        "backtest_period": (
            "2019-01-01 to 2024-12-31"
        ),

        "final_validation": (
            "2025 protected and not evaluated"
        ),

        "feature_count": len(
            FEATURE_COLUMNS
        ),

        "absolute_targets": [

            {
                "horizon": horizon,
                "threshold": threshold,
            }

            for horizon, threshold
            in ABSOLUTE_TARGETS
        ],

        "relative_horizons": (
            RELATIVE_HORIZONS
        ),

        "relative_top_fraction": (
            RELATIVE_TOP_FRACTION
        ),

        "ranking": (

            ranking_df
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
    }


    with open(
        TARGET_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            default=str,
        )


    print(
        TARGET_METRICS_FILE
    )

    print(
        TARGET_YEAR_METRICS_FILE
    )

    print(
        TARGET_PREVALENCE_FILE
    )

    print(
        TARGET_RANKING_FILE
    )

    print(
        TARGET_SUMMARY_FILE
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print_section(
        "TARGET DIAGNOSTICS COMPLETE"
    )


    print(
        "2025 validation was NOT evaluated."
    )

    print()

    print(
        "Use target_candidate_ranking.csv "
        "to identify targets with stable "
        "2019-2024 discrimination."
    )


    if not ranking_df.empty:

        print()

        print(
            "TOP 10 HISTORICAL TARGET/MODEL "
            "COMBINATIONS:"
        )

        print(
            "-" * 95
        )


        print(
            ranking_df[
                [
                    "rank",
                    "candidate",
                    "model",
                    "aggregate_backtest_auc",
                    "mean_year_auc",
                    "std_year_auc",
                    "minimum_year_auc",
                    "years_above_052",
                    "stability_score",
                ]
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )
        )


    return {

        "metrics": metrics_df,

        "year_metrics": year_df,

        "prevalence": prevalence_df,

        "ranking": ranking_df,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_target_diagnostics()