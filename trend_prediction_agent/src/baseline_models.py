"""
Agent 2 - Classical Baseline Models
====================================

Purpose
-------
Determine whether Agent 2's weak out-of-sample performance is caused
primarily by:

1. the deep architecture,
2. the feature/target relationship,
3. temporal regime shift.

Models
------
1. Majority / constant-probability baseline
2. Logistic Regression
3. Decision Tree
4. Random Forest

Important
---------
These models use the SAME prediction dates and targets as the
Dilated-LSTM + Transformer model.

For classical baselines, the 30 engineered features available on the
prediction date are used directly.

This is therefore a diagnostic feature-signal baseline rather than
a direct sequence-model architecture comparison.

No 2025 information is used for model fitting.
"""

from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

PROJECT_ROOT = TREND_AGENT_ROOT.parent


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

FEATURE_FILE = (
    PROCESSED_DIR
    / "preprocessed_trend_data.parquet"
)


EVALUATION_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "trend_evaluation_predictions.parquet"
)


# ============================================================
# OUTPUT FILES
# ============================================================

BASELINE_METRICS_FILE = (
    REPORTS_DIR
    / "baseline_metrics.csv"
)


BASELINE_YEAR_METRICS_FILE = (
    REPORTS_DIR
    / "baseline_year_metrics.csv"
)


BASELINE_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "baseline_predictions.parquet"
)


BASELINE_FEATURE_IMPORTANCE_FILE = (
    REPORTS_DIR
    / "baseline_feature_importance.csv"
)


BASELINE_SUMMARY_FILE = (
    REPORTS_DIR
    / "baseline_summary.json"
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
# SETTINGS
# ============================================================

DEFAULT_THRESHOLD = 0.50


SPLIT_ORDER = [
    "train",
    "backtest",
    "validation",
]


RANDOM_STATE = 42


# ============================================================
# DISPLAY
# ============================================================

def print_section(title):

    print()

    print(
        "=" * 90
    )

    print(
        title
    )

    print(
        "=" * 90
    )

    print()


def safe_float(
    value,
):

    if pd.isna(
        value
    ):

        return None


    return float(
        value
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_inputs():

    print_section(
        "VALIDATING BASELINE INPUTS"
    )


    required_files = [

        FEATURE_FILE,
        EVALUATION_PREDICTIONS_FILE,
    ]


    for file in required_files:

        print(
            file
        )


        if not file.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )


    print()

    print(
        "All required baseline files found."
    )


# ============================================================
# LOAD TARGET INDEX
# ============================================================

def load_targets():

    print_section(
        "LOADING SEQUENCE TARGET INDEX"
    )


    target_columns = [

        "sequence_id",
        "symbol",
        "dataset_split",

        "prediction_date",
        "target_date",

        "future_return",
        "target",
    ]


    df = pd.read_parquet(
        EVALUATION_PREDICTIONS_FILE,
        columns=target_columns,
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
        "dataset_split"
    ] = (
        df[
            "dataset_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    df[
        "prediction_date"
    ] = pd.to_datetime(
        df[
            "prediction_date"
        ]
    )


    df[
        "target_date"
    ] = pd.to_datetime(
        df[
            "target_date"
        ]
    )


    df[
        "target"
    ] = (
        df[
            "target"
        ]
        .astype(
            np.int8
        )
    )


    print(
        f"Sequences : "
        f"{len(df):,}"
    )

    print(
        f"Stocks    : "
        f"{df['symbol'].nunique():,}"
    )


    for split in SPLIT_ORDER:

        part = df[
            df[
                "dataset_split"
            ]
            ==
            split
        ]


        if part.empty:

            continue


        print(
            f"{split.upper():10s}: "
            f"{len(part):10,d} | "
            f"HIGH "
            f"{part['target'].mean():7.2%}"
        )


    return df


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features():

    print_section(
        "LOADING PREPROCESSED FEATURES"
    )


    available_columns = pd.read_parquet(
        FEATURE_FILE
    ).columns.tolist()


    print(
        f"Columns in feature file : "
        f"{len(available_columns)}"
    )


    missing_features = [

        feature

        for feature in FEATURE_COLUMNS

        if feature not in available_columns
    ]


    if missing_features:

        raise KeyError(
            "Missing baseline features:\n"
            f"{missing_features}"
        )


    symbol_candidates = [

        "symbol",
        "Symbol",
        "SYMBOL",
    ]


    date_candidates = [

        "date",
        "Date",
        "DATE",
    ]


    symbol_column = next(
        (
            column
            for column in symbol_candidates
            if column in available_columns
        ),
        None,
    )


    date_column = next(
        (
            column
            for column in date_candidates
            if column in available_columns
        ),
        None,
    )


    if symbol_column is None:

        raise KeyError(
            "Could not locate symbol column "
            "in preprocessed feature file."
        )


    if date_column is None:

        raise KeyError(
            "Could not locate date column "
            "in preprocessed feature file."
        )


    needed_columns = [

        symbol_column,
        date_column,

        *FEATURE_COLUMNS,
    ]


    features = pd.read_parquet(
        FEATURE_FILE,
        columns=needed_columns,
    )


    features = features.rename(
        columns={
            symbol_column: "symbol",
            date_column: "prediction_date",
        }
    )


    features[
        "symbol"
    ] = (
        features[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    features[
        "prediction_date"
    ] = pd.to_datetime(
        features[
            "prediction_date"
        ]
    )


    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    duplicate_count = features.duplicated(
        subset=[
            "symbol",
            "prediction_date",
        ]
    ).sum()


    print(
        f"Feature rows       : "
        f"{len(features):,}"
    )

    print(
        f"Feature stocks     : "
        f"{features['symbol'].nunique():,}"
    )

    print(
        f"Duplicate keys     : "
        f"{duplicate_count:,}"
    )


    if duplicate_count > 0:

        raise ValueError(
            "Duplicate symbol/date feature "
            "rows detected."
        )


    # --------------------------------------------------------
    # NUMERICAL VALIDATION
    # --------------------------------------------------------

    matrix = features[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )


    if not np.isfinite(
        matrix
    ).all():

        raise ValueError(
            "Preprocessed features contain "
            "NaN or infinity."
        )


    print(
        f"Feature count      : "
        f"{len(FEATURE_COLUMNS)}"
    )


    return features


# ============================================================
# JOIN FEATURES TO SEQUENCES
# ============================================================

def build_baseline_dataset(
    targets,
    features,
):

    print_section(
        "BUILDING BASELINE DATASET"
    )


    dataset = targets.merge(

        features,

        how="left",

        on=[
            "symbol",
            "prediction_date",
        ],

        validate="many_to_one",
    )


    missing_feature_rows = (
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
        f"Target rows          : "
        f"{len(targets):,}"
    )

    print(
        f"Merged rows          : "
        f"{len(dataset):,}"
    )

    print(
        f"Missing feature rows : "
        f"{missing_feature_rows:,}"
    )


    if len(
        dataset
    ) != len(
        targets
    ):

        raise ValueError(
            "Target/feature merge changed "
            "the sequence count."
        )


    if missing_feature_rows > 0:

        missing_example = (

            dataset[
                dataset[
                    FEATURE_COLUMNS
                ]
                .isna()
                .any(
                    axis=1
                )
            ][
                [
                    "symbol",
                    "prediction_date",
                ]
            ]
            .head(
                10
            )
        )


        print()

        print(
            missing_example.to_string(
                index=False
            )
        )


        raise ValueError(
            "Some prediction-date features "
            "could not be matched."
        )


    print()

    print(
        "Feature/target merge PASSED."
    )


    return dataset


# ============================================================
# CREATE SAME CHRONOLOGICAL FIT SUBSET
# ============================================================

def create_training_partition(
    dataset,
):

    print_section(
        "CREATING CHRONOLOGICAL BASELINE FIT PARTITION"
    )


    train = dataset[
        dataset[
            "dataset_split"
        ]
        ==
        "train"
    ].copy()


    if train.empty:

        raise ValueError(
            "No TRAIN sequences found."
        )


    # --------------------------------------------------------
    # Use chronological dates only.
    #
    # Approximately final 20% of prediction dates are reserved.
    # Target-date embargo prevents future targets from crossing
    # into the internal validation period.
    # --------------------------------------------------------

    unique_dates = np.array(
        sorted(
            train[
                "prediction_date"
            ]
            .unique()
        )
    )


    cutoff_index = int(
        len(
            unique_dates
        )
        *
        0.80
    )


    cutoff_index = min(
        max(
            cutoff_index,
            1,
        ),
        len(
            unique_dates
        )
        -
        1,
    )


    cutoff = pd.Timestamp(
        unique_dates[
            cutoff_index
        ]
    )


    fitting = train[

        (
            train[
                "prediction_date"
            ]
            <
            cutoff
        )

        &

        (
            train[
                "target_date"
            ]
            <
            cutoff
        )

    ].copy()


    internal_validation = train[

        train[
            "prediction_date"
        ]
        >=
        cutoff

    ].copy()


    embargoed = train[

        (
            train[
                "prediction_date"
            ]
            <
            cutoff
        )

        &

        (
            train[
                "target_date"
            ]
            >=
            cutoff
        )

    ].copy()


    print(
        f"Internal cutoff      : "
        f"{cutoff.date()}"
    )

    print()

    print(
        f"Original TRAIN       : "
        f"{len(train):,}"
    )

    print(
        f"Model-fitting rows   : "
        f"{len(fitting):,}"
    )

    print(
        f"Internal validation  : "
        f"{len(internal_validation):,}"
    )

    print(
        f"Embargoed rows       : "
        f"{len(embargoed):,}"
    )


    if fitting.empty:

        raise ValueError(
            "Baseline fitting dataset is empty."
        )


    if (
        fitting[
            "target_date"
        ]
        .max()
        >=
        cutoff
    ):

        raise ValueError(
            "Target-date leakage detected."
        )


    print()

    print(
        "Chronological baseline split PASSED."
    )


    return (
        fitting,
        internal_validation,
        cutoff,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probability,
    threshold=0.50,
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

        threshold
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

        average_precision = (
            average_precision_score(
                y_true,
                probability,
            )
        )

    except ValueError:

        average_precision = np.nan


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
            average_precision
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

    models = {}


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    models[
        "logistic_regression"
    ] = LogisticRegression(

        solver="lbfgs",

        max_iter=1000,

        class_weight="balanced",

        random_state=RANDOM_STATE,
    )


    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    models[
        "decision_tree"
    ] = DecisionTreeClassifier(

        max_depth=8,

        min_samples_split=100,

        min_samples_leaf=50,

        class_weight="balanced",

        random_state=RANDOM_STATE,
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    models[
        "random_forest"
    ] = RandomForestClassifier(

        n_estimators=200,

        max_depth=12,

        min_samples_split=100,

        min_samples_leaf=30,

        max_features="sqrt",

        class_weight="balanced_subsample",

        n_jobs=-1,

        random_state=RANDOM_STATE,

        verbose=0,
    )


    return models


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def run_baselines(
    dataset,
    fitting,
    internal_validation,
):

    print_section(
        "TRAINING CLASSICAL BASELINES"
    )


    X_fit = fitting[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )


    y_fit = fitting[
        "target"
    ].to_numpy(
        dtype=np.int32
    )


    print(
        f"Fit matrix : "
        f"{X_fit.shape}"
    )

    print(
        f"HIGH rate  : "
        f"{y_fit.mean():.2%}"
    )


    # --------------------------------------------------------
    # CONSTANT BASELINE
    # --------------------------------------------------------

    training_prevalence = float(
        y_fit.mean()
    )


    print()

    print(
        f"Constant HIGH probability : "
        f"{training_prevalence:.4f}"
    )


    all_predictions = []

    metric_records = []

    year_records = []

    importance_records = []


    # ========================================================
    # CONSTANT MODEL
    # ========================================================

    print_section(
        "BASELINE 1 - CONSTANT / MAJORITY"
    )


    for split in SPLIT_ORDER:

        subset = dataset[
            dataset[
                "dataset_split"
            ]
            ==
            split
        ]


        if subset.empty:

            continue


        probability = np.full(
            len(
                subset
            ),
            training_prevalence,
            dtype=np.float64,
        )


        metrics = calculate_metrics(

            subset[
                "target"
            ],

            probability,
        )


        metric_records.append(
            {

                "model": "constant_baseline",

                "dataset_split": split,

                "training_seconds": 0.0,

                **metrics,
            }
        )


        temp = subset[
            [
                "sequence_id",
                "symbol",
                "dataset_split",
                "prediction_date",
                "target_date",
                "future_return",
                "target",
            ]
        ].copy()


        temp[
            "model"
        ] = "constant_baseline"


        temp[
            "high_return_probability"
        ] = probability


        temp[
            "predicted_target"
        ] = (

            probability

            >=

            DEFAULT_THRESHOLD
        ).astype(
            np.int8
        )


        all_predictions.append(
            temp
        )


    # ========================================================
    # TRAIN REAL MODELS
    # ========================================================

    models = create_models()


    for model_name, model in models.items():

        print_section(
            f"TRAINING {model_name.upper()}"
        )


        start_time = time.time()


        model.fit(
            X_fit,
            y_fit,
        )


        training_seconds = (
            time.time()
            -
            start_time
        )


        print(
            f"Training time : "
            f"{training_seconds:.2f} sec"
        )


        # ----------------------------------------------------
        # FEATURE IMPORTANCE / COEFFICIENTS
        # ----------------------------------------------------

        if hasattr(
            model,
            "feature_importances_",
        ):

            values = model.feature_importances_


            for feature, value in zip(
                FEATURE_COLUMNS,
                values,
            ):

                importance_records.append(
                    {

                        "model": model_name,

                        "feature": feature,

                        "importance": float(
                            value
                        ),
                    }
                )


        elif hasattr(
            model,
            "coef_",
        ):

            values = model.coef_[
                0
            ]


            for feature, value in zip(
                FEATURE_COLUMNS,
                values,
            ):

                importance_records.append(
                    {

                        "model": model_name,

                        "feature": feature,

                        "importance": float(
                            value
                        ),

                        "absolute_importance": float(
                            abs(
                                value
                            )
                        ),
                    }
                )


        # ----------------------------------------------------
        # SPLIT EVALUATION
        # ----------------------------------------------------

        for split in SPLIT_ORDER:

            subset = dataset[
                dataset[
                    "dataset_split"
                ]
                ==
                split
            ]


            if subset.empty:

                continue


            X = subset[
                FEATURE_COLUMNS
            ].to_numpy(
                dtype=np.float32
            )


            probability = (
                model.predict_proba(
                    X
                )[
                    :,
                    1
                ]
            )


            metrics = calculate_metrics(

                subset[
                    "target"
                ],

                probability,
            )


            metric_records.append(
                {

                    "model": model_name,

                    "dataset_split": split,

                    "training_seconds": float(
                        training_seconds
                    ),

                    **metrics,
                }
            )


            print()

            print(
                f"{split.upper():10s} "
                f"AUC="
                f"{metrics['roc_auc']:.4f} | "
                f"ACC="
                f"{metrics['accuracy']:.4f} | "
                f"F1="
                f"{metrics['f1']:.4f}"
            )


            temp = subset[
                [
                    "sequence_id",
                    "symbol",
                    "dataset_split",
                    "prediction_date",
                    "target_date",
                    "future_return",
                    "target",
                ]
            ].copy()


            temp[
                "model"
            ] = model_name


            temp[
                "high_return_probability"
            ] = probability.astype(
                np.float32
            )


            temp[
                "predicted_target"
            ] = (

                probability

                >=

                DEFAULT_THRESHOLD
            ).astype(
                np.int8
            )


            all_predictions.append(
                temp
            )


        # ----------------------------------------------------
        # YEAR-BY-YEAR PERFORMANCE
        # ----------------------------------------------------

        for year in sorted(
            dataset[
                "prediction_date"
            ]
            .dt
            .year
            .unique()
        ):

            subset = dataset[
                dataset[
                    "prediction_date"
                ]
                .dt
                .year
                ==
                year
            ]


            if subset.empty:

                continue


            X = subset[
                FEATURE_COLUMNS
            ].to_numpy(
                dtype=np.float32
            )


            probability = (
                model.predict_proba(
                    X
                )[
                    :,
                    1
                ]
            )


            metrics = calculate_metrics(

                subset[
                    "target"
                ],

                probability,
            )


            split_names = "|".join(
                sorted(
                    subset[
                        "dataset_split"
                    ]
                    .unique()
                )
            )


            year_records.append(
                {

                    "model": model_name,

                    "year": int(
                        year
                    ),

                    "dataset_split": (
                        split_names
                    ),

                    "stocks": int(
                        subset[
                            "symbol"
                        ]
                        .nunique()
                    ),

                    **metrics,
                }
            )


    # ========================================================
    # BUILD OUTPUTS
    # ========================================================

    metrics_df = pd.DataFrame(
        metric_records
    )


    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )


    year_df = pd.DataFrame(
        year_records
    )


    importance_df = pd.DataFrame(
        importance_records
    )


    return (
        metrics_df,
        predictions_df,
        year_df,
        importance_df,
    )


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    metrics_df,
):

    print_section(
        "BASELINE MODEL COMPARISON"
    )


    display_columns = [

        "model",
        "dataset_split",

        "samples",

        "accuracy",
        "f1",

        "roc_auc",
        "average_precision",

        "actual_high_rate",
        "predicted_high_rate",
    ]


    print(
        metrics_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


    print_section(
        "OUT-OF-SAMPLE ROC-AUC COMPARISON"
    )


    pivot = metrics_df.pivot(

        index="model",

        columns="dataset_split",

        values="roc_auc",
    )


    ordered_columns = [

        column

        for column in SPLIT_ORDER

        if column in pivot.columns
    ]


    pivot = pivot[
        ordered_columns
    ]


    if "backtest" in pivot.columns:

        pivot = pivot.sort_values(
            "backtest",
            ascending=False,
        )


    print(
        pivot.to_string()
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    metrics_df,
    predictions_df,
    year_df,
    importance_df,
    cutoff,
):

    print_section(
        "SAVING BASELINE RESULTS"
    )


    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    metrics_df.to_csv(
        BASELINE_METRICS_FILE,
        index=False,
    )


    predictions_df.to_parquet(
        BASELINE_PREDICTIONS_FILE,
        index=False,
    )


    year_df.to_csv(
        BASELINE_YEAR_METRICS_FILE,
        index=False,
    )


    importance_df.to_csv(
        BASELINE_FEATURE_IMPORTANCE_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "feature_count": len(
            FEATURE_COLUMNS
        ),

        "features": (
            FEATURE_COLUMNS
        ),

        "internal_cutoff": str(
            cutoff.date()
        ),

        "models": sorted(
            metrics_df[
                "model"
            ]
            .unique()
            .tolist()
        ),

        "threshold": (
            DEFAULT_THRESHOLD
        ),

        "important_note": (
            "Classical baselines use the 30 features "
            "available at the prediction date. "
            "The deep model uses a 60-day sequence."
        ),

        "metrics": (

            metrics_df
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
        BASELINE_SUMMARY_FILE,
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
        BASELINE_METRICS_FILE
    )

    print(
        BASELINE_YEAR_METRICS_FILE
    )

    print(
        BASELINE_FEATURE_IMPORTANCE_FILE
    )

    print(
        BASELINE_PREDICTIONS_FILE
    )

    print(
        BASELINE_SUMMARY_FILE
    )


# ============================================================
# MAIN
# ============================================================

def run_baseline_analysis():

    print_section(
        "AGENT 2 - CLASSICAL BASELINE ANALYSIS"
    )


    print(
        "Purpose:"
    )

    print(
        "Determine whether useful predictive "
        "signal exists before modifying the "
        "deep-learning architecture."
    )


    # --------------------------------------------------------
    # 1. INPUT CHECK
    # --------------------------------------------------------

    validate_inputs()


    # --------------------------------------------------------
    # 2. TARGETS
    # --------------------------------------------------------

    targets = load_targets()


    # --------------------------------------------------------
    # 3. FEATURES
    # --------------------------------------------------------

    features = load_features()


    # --------------------------------------------------------
    # 4. MERGE
    # --------------------------------------------------------

    dataset = build_baseline_dataset(
        targets,
        features,
    )


    # --------------------------------------------------------
    # 5. TRAIN PARTITION
    # --------------------------------------------------------

    (
        fitting,
        internal_validation,
        cutoff,

    ) = create_training_partition(
        dataset
    )


    # --------------------------------------------------------
    # 6. BASELINES
    # --------------------------------------------------------

    (
        metrics_df,
        predictions_df,
        year_df,
        importance_df,

    ) = run_baselines(

        dataset,
        fitting,
        internal_validation,
    )


    # --------------------------------------------------------
    # 7. DISPLAY
    # --------------------------------------------------------

    print_comparison(
        metrics_df
    )


    # --------------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------------

    save_outputs(

        metrics_df,
        predictions_df,
        year_df,
        importance_df,
        cutoff,
    )


    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print_section(
        "BASELINE ANALYSIS COMPLETE"
    )


    print(
        "Do not change Agent 2 architecture yet."
    )

    print()

    print(
        "First compare baseline BACKTEST and "
        "VALIDATION ROC-AUC against:"
    )

    print()

    print(
        "Deep model BACKTEST   : 0.5093"
    )

    print(
        "Deep model VALIDATION : 0.4651"
    )


    return {

        "metrics": (
            metrics_df
        ),

        "predictions": (
            predictions_df
        ),

        "year_metrics": (
            year_df
        ),

        "feature_importance": (
            importance_df
        ),
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_baseline_analysis()