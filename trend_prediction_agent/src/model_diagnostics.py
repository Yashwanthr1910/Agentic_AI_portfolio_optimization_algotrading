"""
Agent 2 - Broad-Universe Model Diagnostics
===========================================

Purpose
-------
Evaluate the trained Agent 2 trend prediction model after
broad-universe training.

This module DOES NOT train or modify the model.

Primary input
-------------
reports/results/trend_evaluation_predictions.parquet

These predictions were produced by trainer.py from the
memory-efficient sequence pipeline.

Therefore diagnostics do NOT need to recreate all
822k 60x30 sequences or run neural-network inference again.

Diagnostics
-----------
1. Train / Backtest / Validation metrics
2. Probability-distribution analysis
3. Threshold diagnostics
4. Per-stock diagnostics
5. Per-year diagnostics
6. Temporal generalization analysis
7. Probability separation analysis
8. Calibration-style bins
9. Report generation
10. Final interpretation

Important
---------
Threshold analysis is diagnostic only.

The final 2025 validation set must NOT be used to choose an
operational classification threshold or tune model parameters.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# GENERAL
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
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


REPORTS_DIR = (
    TREND_AGENT_ROOT
    / "reports"
)


RESULTS_DIR = (
    REPORTS_DIR
    / "results"
)


PROCESSED_DIR = (
    TREND_AGENT_ROOT
    / "data"
    / "processed"
)


MODELS_DIR = (
    TREND_AGENT_ROOT
    / "models"
)


# ============================================================
# INPUT FILES
# ============================================================

EVALUATION_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "trend_evaluation_predictions.parquet"
)


TRAINING_HISTORY_FILE = (
    REPORTS_DIR
    / "training_history.csv"
)


TRAINING_SUMMARY_FILE = (
    REPORTS_DIR
    / "training_summary.json"
)


SEQUENCE_METADATA_FILE = (
    PROCESSED_DIR
    / "sequence_metadata.json"
)


SEQUENCE_INDEX_FILE = (
    PROCESSED_DIR
    / "sequence_index.parquet"
)


# ============================================================
# OUTPUT FILES
# ============================================================

DIAGNOSTIC_METRICS_FILE = (
    REPORTS_DIR
    / "diagnostic_metrics.csv"
)


PROBABILITY_SUMMARY_FILE = (
    REPORTS_DIR
    / "probability_summary.csv"
)


THRESHOLD_ANALYSIS_FILE = (
    REPORTS_DIR
    / "threshold_analysis.csv"
)


PER_STOCK_DIAGNOSTICS_FILE = (
    REPORTS_DIR
    / "per_stock_diagnostics.csv"
)


PER_YEAR_DIAGNOSTICS_FILE = (
    REPORTS_DIR
    / "per_year_diagnostics.csv"
)


CALIBRATION_FILE = (
    REPORTS_DIR
    / "probability_calibration_bins.csv"
)


TEMPORAL_GENERALIZATION_FILE = (
    REPORTS_DIR
    / "temporal_generalization.csv"
)


MODEL_DIAGNOSIS_SUMMARY_FILE = (
    REPORTS_DIR
    / "model_diagnosis_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_THRESHOLD = 0.50


THRESHOLDS = np.round(
    np.arange(
        0.10,
        0.91,
        0.05,
    ),
    2,
)


SPLIT_ORDER = [
    "train",
    "backtest",
    "validation",
]


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


def safe_float_text(
    value,
    decimals=4,
):

    if pd.isna(
        value
    ):

        return "N/A"


    return (
        f"{float(value):.{decimals}f}"
    )


# ============================================================
# VALIDATE FILES
# ============================================================

def validate_inputs():

    print_section(
        "VALIDATING DIAGNOSTIC INPUTS"
    )


    print(
        "Primary prediction file:"
    )

    print(
        EVALUATION_PREDICTIONS_FILE
    )

    print()


    if not EVALUATION_PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            "trend_evaluation_predictions.parquet "
            "was not found.\n\n"
            "Run trainer.py successfully first."
        )


    print(
        "Prediction file found."
    )


    optional_files = [

        TRAINING_HISTORY_FILE,
        TRAINING_SUMMARY_FILE,
        SEQUENCE_METADATA_FILE,
        SEQUENCE_INDEX_FILE,
    ]


    print()

    print(
        "Supporting files:"
    )


    for file in optional_files:

        status = (
            "FOUND"
            if file.exists()
            else "MISSING"
        )


        print(
            f"  [{status}] {file.name}"
        )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    print_section(
        "LOADING BROAD-UNIVERSE MODEL PREDICTIONS"
    )


    df = pd.read_parquet(
        EVALUATION_PREDICTIONS_FILE
    )


    required_columns = [

        "sequence_id",
        "symbol",
        "dataset_split",

        "prediction_date",
        "target_date",

        "future_return",
        "target",

        "low_return_probability",
        "high_return_probability",

        "predicted_target",
    ]


    missing = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing:

        raise KeyError(
            "Prediction file is missing "
            f"required columns:\n{missing}"
        )


    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

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
        ],
        errors="raise",
    )


    df[
        "target_date"
    ] = pd.to_datetime(
        df[
            "target_date"
        ],
        errors="raise",
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


    df[
        "predicted_target"
    ] = (
        df[
            "predicted_target"
        ]
        .astype(
            np.int8
        )
    )


    probability_columns = [

        "low_return_probability",
        "high_return_probability",
    ]


    for column in probability_columns:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            "Evaluation prediction file is empty."
        )


    if (
        df[
            required_columns
        ]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Prediction dataset contains "
            "missing required values."
        )


    if not np.isfinite(
        df[
            probability_columns
        ]
        .to_numpy(
            dtype=np.float64
        )
    ).all():

        raise ValueError(
            "Prediction probabilities contain "
            "NaN or infinity."
        )


    probability_sum = (

        df[
            "low_return_probability"
        ]

        +

        df[
            "high_return_probability"
        ]
    )


    if not np.allclose(
        probability_sum,
        1.0,
        atol=1e-4,
    ):

        raise ValueError(
            "LOW/HIGH probabilities do not "
            "sum approximately to 1."
        )


    invalid_targets = (
        ~df[
            "target"
        ]
        .isin(
            [
                0,
                1,
            ]
        )
    ).sum()


    if invalid_targets > 0:

        raise ValueError(
            "Invalid targets detected."
        )


    # Recreate prediction at 0.50 independently.
    #
    # This verifies trainer output.

    diagnostic_prediction = (

        df[
            "high_return_probability"
        ]

        >=

        DEFAULT_THRESHOLD
    ).astype(
        np.int8
    )


    mismatches = (

        diagnostic_prediction

        !=

        df[
            "predicted_target"
        ]
    ).sum()


    print(
        f"Rows             : "
        f"{len(df):,}"
    )

    print(
        f"Stocks           : "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"Prediction dates : "
        f"{df['prediction_date'].min().date()} "
        f"-> "
        f"{df['prediction_date'].max().date()}"
    )

    print(
        f"Threshold check  : "
        f"{mismatches:,} mismatches"
    )


    if mismatches > 0:

        raise ValueError(
            "Trainer predicted_target values "
            "do not match threshold 0.50."
        )


    df[
        "actual_class"
    ] = (
        df[
            "target"
        ]
    )


    df[
        "predicted_class"
    ] = (
        diagnostic_prediction
    )


    df[
        "year"
    ] = (
        df[
            "prediction_date"
        ]
        .dt
        .year
    )


    df[
        "prediction_correct"
    ] = (

        df[
            "actual_class"
        ]

        ==

        df[
            "predicted_class"
        ]
    )


    print()

    print(
        "Split distribution:"
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
            f"  {split.upper():10s}: "
            f"{len(part):10,d} | "
            f"Actual HIGH "
            f"{part['actual_class'].mean():7.2%} | "
            f"Predicted HIGH "
            f"{part['predicted_class'].mean():7.2%}"
        )


    return df


# ============================================================
# METRIC CALCULATION
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


    y_pred = (

        probability

        >=

        threshold
    ).astype(
        np.int32
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
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

        roc_auc = roc_auc_score(
            y_true,
            probability,
        )

    except ValueError:

        roc_auc = np.nan


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

        brier = (
            brier_score_loss(
                y_true,
                probability,
            )
        )

    except ValueError:

        brier = np.nan


    return {

        "samples": int(
            len(
                y_true
            )
        ),

        "actual_positive_rate": float(
            np.mean(
                y_true
            )
        ),

        "predicted_positive_rate": float(
            np.mean(
                y_pred
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "roc_auc": (
            float(
                roc_auc
            )
            if not pd.isna(
                roc_auc
            )
            else np.nan
        ),

        "average_precision": (
            float(
                average_precision
            )
            if not pd.isna(
                average_precision
            )
            else np.nan
        ),

        "brier_score": (
            float(
                brier
            )
            if not pd.isna(
                brier
            )
            else np.nan
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
# SPLIT METRICS
# ============================================================

def evaluate_splits(
    results,
):

    print_section(
        "TRAIN / BACKTEST / VALIDATION DIAGNOSTICS"
    )


    records = []


    for split in SPLIT_ORDER:

        subset = results[
            results[
                "dataset_split"
            ]
            ==
            split
        ]


        if subset.empty:

            continue


        metrics = calculate_metrics(

            subset[
                "actual_class"
            ],

            subset[
                "high_return_probability"
            ],

            threshold=(
                DEFAULT_THRESHOLD
            ),
        )


        records.append(
            {
                "dataset_split": split,
                **metrics,
            }
        )


        print(
            f"{split.upper()}"
        )

        print(
            "-" * 70
        )

        print(
            f"Samples                 : "
            f"{metrics['samples']:,}"
        )

        print(
            f"Actual HIGH rate        : "
            f"{metrics['actual_positive_rate']:.2%}"
        )

        print(
            f"Predicted HIGH rate     : "
            f"{metrics['predicted_positive_rate']:.2%}"
        )

        print(
            f"Accuracy                : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision               : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall                  : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Specificity             : "
            f"{metrics['specificity']:.4f}"
        )

        print(
            f"F1                      : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC                 : "
            f"{safe_float_text(metrics['roc_auc'])}"
        )

        print(
            f"Average Precision       : "
            f"{safe_float_text(metrics['average_precision'])}"
        )

        print(
            f"Brier score             : "
            f"{safe_float_text(metrics['brier_score'])}"
        )

        print(
            f"TN / FP / FN / TP       : "
            f"{metrics['tn']} / "
            f"{metrics['fp']} / "
            f"{metrics['fn']} / "
            f"{metrics['tp']}"
        )

        print()


    metrics_df = pd.DataFrame(
        records
    )


    metrics_df.to_csv(
        DIAGNOSTIC_METRICS_FILE,
        index=False,
    )


    return metrics_df


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def analyze_probabilities(
    results,
):

    print_section(
        "HIGH-RETURN PROBABILITY DISTRIBUTION"
    )


    records = []


    for split in SPLIT_ORDER:

        subset = results[
            results[
                "dataset_split"
            ]
            ==
            split
        ]


        if subset.empty:

            continue


        probabilities = subset[
            "high_return_probability"
        ]


        positive_probabilities = (
            subset.loc[
                subset[
                    "actual_class"
                ]
                ==
                1,
                "high_return_probability",
            ]
        )


        negative_probabilities = (
            subset.loc[
                subset[
                    "actual_class"
                ]
                ==
                0,
                "high_return_probability",
            ]
        )


        positive_mean = (
            positive_probabilities.mean()
            if len(
                positive_probabilities
            )
            else np.nan
        )


        negative_mean = (
            negative_probabilities.mean()
            if len(
                negative_probabilities
            )
            else np.nan
        )


        separation = (
            positive_mean
            -
            negative_mean
        )


        record = {

            "dataset_split": split,

            "samples": int(
                len(
                    subset
                )
            ),

            "mean_probability": float(
                probabilities.mean()
            ),

            "std_probability": float(
                probabilities.std()
            ),

            "min_probability": float(
                probabilities.min()
            ),

            "p05": float(
                probabilities.quantile(
                    0.05
                )
            ),

            "p25": float(
                probabilities.quantile(
                    0.25
                )
            ),

            "median_probability": float(
                probabilities.median()
            ),

            "p75": float(
                probabilities.quantile(
                    0.75
                )
            ),

            "p95": float(
                probabilities.quantile(
                    0.95
                )
            ),

            "max_probability": float(
                probabilities.max()
            ),

            "mean_probability_actual_positive": float(
                positive_mean
            ),

            "mean_probability_actual_negative": float(
                negative_mean
            ),

            "probability_separation": float(
                separation
            ),
        }


        records.append(
            record
        )


        print(
            split.upper()
        )

        print(
            "-" * 70
        )

        print(
            f"Mean P(HIGH)            : "
            f"{record['mean_probability']:.4f}"
        )

        print(
            f"Std                     : "
            f"{record['std_probability']:.4f}"
        )

        print(
            f"Minimum                 : "
            f"{record['min_probability']:.4f}"
        )

        print(
            f"P05                     : "
            f"{record['p05']:.4f}"
        )

        print(
            f"P25                     : "
            f"{record['p25']:.4f}"
        )

        print(
            f"Median                  : "
            f"{record['median_probability']:.4f}"
        )

        print(
            f"P75                     : "
            f"{record['p75']:.4f}"
        )

        print(
            f"P95                     : "
            f"{record['p95']:.4f}"
        )

        print(
            f"Maximum                 : "
            f"{record['max_probability']:.4f}"
        )

        print(
            f"Mean P(HIGH) actual HIGH: "
            f"{record['mean_probability_actual_positive']:.4f}"
        )

        print(
            f"Mean P(HIGH) actual LOW : "
            f"{record['mean_probability_actual_negative']:.4f}"
        )

        print(
            f"Probability separation  : "
            f"{record['probability_separation']:+.4f}"
        )

        print()


    probability_df = pd.DataFrame(
        records
    )


    probability_df.to_csv(
        PROBABILITY_SUMMARY_FILE,
        index=False,
    )


    return probability_df


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def run_threshold_analysis(
    results,
):

    print_section(
        "THRESHOLD ANALYSIS - DIAGNOSTIC ONLY"
    )


    records = []


    for split in SPLIT_ORDER:

        subset = results[
            results[
                "dataset_split"
            ]
            ==
            split
        ]


        if subset.empty:

            continue


        y_true = (
            subset[
                "actual_class"
            ]
            .to_numpy()
        )


        probability = (
            subset[
                "high_return_probability"
            ]
            .to_numpy()
        )


        for threshold in THRESHOLDS:

            metrics = calculate_metrics(

                y_true,

                probability,

                threshold=float(
                    threshold
                ),
            )


            records.append(
                {
                    "dataset_split": split,

                    "threshold": float(
                        threshold
                    ),

                    **metrics,
                }
            )


    threshold_df = pd.DataFrame(
        records
    )


    threshold_df.to_csv(
        THRESHOLD_ANALYSIS_FILE,
        index=False,
    )


    for split in SPLIT_ORDER:

        subset = threshold_df[
            threshold_df[
                "dataset_split"
            ]
            ==
            split
        ]


        if subset.empty:

            continue


        best_row = subset.loc[
            subset[
                "f1"
            ]
            .idxmax()
        ]


        print(
            f"{split.upper()} - "
            f"best diagnostic F1 threshold"
        )

        print(
            "-" * 70
        )

        print(
            f"Threshold          : "
            f"{best_row['threshold']:.2f}"
        )

        print(
            f"F1                 : "
            f"{best_row['f1']:.4f}"
        )

        print(
            f"Accuracy           : "
            f"{best_row['accuracy']:.4f}"
        )

        print(
            f"Precision          : "
            f"{best_row['precision']:.4f}"
        )

        print(
            f"Recall             : "
            f"{best_row['recall']:.4f}"
        )

        print(
            f"Specificity        : "
            f"{best_row['specificity']:.4f}"
        )

        print(
            f"Predicted HIGH     : "
            f"{best_row['predicted_positive_rate']:.2%}"
        )

        print()


    print(
        "IMPORTANT:"
    )

    print(
        "These thresholds are diagnostic only."
    )

    print(
        "Do NOT choose a production threshold "
        "from the 2025 validation results."
    )


    return threshold_df


# ============================================================
# PER-STOCK DIAGNOSTICS
# ============================================================

def analyze_by_stock(
    results,
):

    print_section(
        "PER-STOCK MODEL DIAGNOSTICS"
    )


    records = []


    grouped = results.groupby(
        [
            "dataset_split",
            "symbol",
        ],
        sort=True,
        observed=True,
    )


    for (
        split,
        symbol,
    ), group in grouped:

        if len(
            group
        ) < 2:

            continue


        metrics = calculate_metrics(

            group[
                "actual_class"
            ],

            group[
                "high_return_probability"
            ],

            threshold=(
                DEFAULT_THRESHOLD
            ),
        )


        records.append(
            {

                "dataset_split": split,

                "symbol": symbol,

                "first_date": (
                    group[
                        "prediction_date"
                    ]
                    .min()
                ),

                "last_date": (
                    group[
                        "prediction_date"
                    ]
                    .max()
                ),

                "mean_future_return": float(
                    group[
                        "future_return"
                    ]
                    .mean()
                ),

                "mean_high_probability": float(
                    group[
                        "high_return_probability"
                    ]
                    .mean()
                ),

                **metrics,
            }
        )


    stock_df = pd.DataFrame(
        records
    )


    stock_df.to_csv(
        PER_STOCK_DIAGNOSTICS_FILE,
        index=False,
    )


    print(
        f"Stock/split records: "
        f"{len(stock_df):,}"
    )


    # --------------------------------------------------------
    # VALIDATION BEST / WORST
    # --------------------------------------------------------

    validation_df = stock_df[
        stock_df[
            "dataset_split"
        ]
        ==
        "validation"
    ].copy()


    validation_df = validation_df[
        validation_df[
            "roc_auc"
        ]
        .notna()
    ]


    if not validation_df.empty:

        print()

        print(
            "TOP 20 VALIDATION STOCKS BY ROC-AUC"
        )

        print(
            "-" * 90
        )


        top = (
            validation_df
            .sort_values(
                "roc_auc",
                ascending=False,
            )
            .head(
                20
            )
        )


        print(
            top[
                [
                    "symbol",
                    "samples",
                    "actual_positive_rate",
                    "predicted_positive_rate",
                    "roc_auc",
                    "f1",
                ]
            ]
            .to_string(
                index=False
            )
        )


        print()

        print(
            "BOTTOM 20 VALIDATION STOCKS BY ROC-AUC"
        )

        print(
            "-" * 90
        )


        bottom = (
            validation_df
            .sort_values(
                "roc_auc",
                ascending=True,
            )
            .head(
                20
            )
        )


        print(
            bottom[
                [
                    "symbol",
                    "samples",
                    "actual_positive_rate",
                    "predicted_positive_rate",
                    "roc_auc",
                    "f1",
                ]
            ]
            .to_string(
                index=False
            )
        )


    return stock_df


# ============================================================
# PER-YEAR DIAGNOSTICS
# ============================================================

def analyze_by_year(
    results,
):

    print_section(
        "PER-YEAR TEMPORAL DIAGNOSTICS"
    )


    records = []


    grouped = results.groupby(
        "year",
        sort=True,
    )


    for year, group in grouped:

        if len(
            group
        ) < 2:

            continue


        metrics = calculate_metrics(

            group[
                "actual_class"
            ],

            group[
                "high_return_probability"
            ],

            threshold=(
                DEFAULT_THRESHOLD
            ),
        )


        splits = "|".join(
            sorted(
                group[
                    "dataset_split"
                ]
                .unique()
            )
        )


        positive_mean = (
            group.loc[
                group[
                    "actual_class"
                ]
                ==
                1,
                "high_return_probability",
            ]
            .mean()
        )


        negative_mean = (
            group.loc[
                group[
                    "actual_class"
                ]
                ==
                0,
                "high_return_probability",
            ]
            .mean()
        )


        records.append(
            {

                "year": int(
                    year
                ),

                "dataset_split": (
                    splits
                ),

                "stocks": int(
                    group[
                        "symbol"
                    ]
                    .nunique()
                ),

                "mean_future_return": float(
                    group[
                        "future_return"
                    ]
                    .mean()
                ),

                "mean_high_probability": float(
                    group[
                        "high_return_probability"
                    ]
                    .mean()
                ),

                "probability_separation": float(
                    positive_mean
                    -
                    negative_mean
                ),

                **metrics,
            }
        )


    year_df = pd.DataFrame(
        records
    )


    year_df.to_csv(
        PER_YEAR_DIAGNOSTICS_FILE,
        index=False,
    )


    if not year_df.empty:

        display_columns = [

            "year",
            "dataset_split",
            "samples",
            "stocks",

            "actual_positive_rate",
            "predicted_positive_rate",

            "roc_auc",
            "average_precision",
            "f1",

            "probability_separation",
        ]


        print(
            year_df[
                display_columns
            ]
            .to_string(
                index=False
            )
        )


    return year_df


# ============================================================
# CALIBRATION BINS
# ============================================================

def analyze_calibration(
    results,
):

    print_section(
        "PROBABILITY CALIBRATION DIAGNOSTICS"
    )


    records = []


    bin_edges = np.linspace(
        0.0,
        1.0,
        11,
    )


    for split in SPLIT_ORDER:

        subset = results[
            results[
                "dataset_split"
            ]
            ==
            split
        ].copy()


        if subset.empty:

            continue


        subset[
            "probability_bin"
        ] = pd.cut(

            subset[
                "high_return_probability"
            ],

            bins=bin_edges,

            include_lowest=True,

            duplicates="drop",
        )


        grouped = subset.groupby(
            "probability_bin",
            observed=True,
        )


        for probability_bin, group in grouped:

            if group.empty:

                continue


            records.append(
                {

                    "dataset_split": split,

                    "probability_bin": str(
                        probability_bin
                    ),

                    "samples": int(
                        len(
                            group
                        )
                    ),

                    "mean_predicted_probability": float(
                        group[
                            "high_return_probability"
                        ]
                        .mean()
                    ),

                    "actual_positive_rate": float(
                        group[
                            "actual_class"
                        ]
                        .mean()
                    ),

                    "calibration_error": float(

                        group[
                            "high_return_probability"
                        ]
                        .mean()

                        -

                        group[
                            "actual_class"
                        ]
                        .mean()
                    ),
                }
            )


    calibration_df = pd.DataFrame(
        records
    )


    calibration_df.to_csv(
        CALIBRATION_FILE,
        index=False,
    )


    print(
        f"Calibration rows saved: "
        f"{len(calibration_df):,}"
    )


    return calibration_df


# ============================================================
# TEMPORAL GENERALIZATION
# ============================================================

def analyze_temporal_generalization(
    metrics_df,
    year_df,
):

    print_section(
        "TEMPORAL GENERALIZATION ANALYSIS"
    )


    records = []


    train_auc = np.nan

    backtest_auc = np.nan

    validation_auc = np.nan


    for split in SPLIT_ORDER:

        part = metrics_df[
            metrics_df[
                "dataset_split"
            ]
            ==
            split
        ]


        if part.empty:

            continue


        auc = part.iloc[
            0
        ][
            "roc_auc"
        ]


        if split == "train":

            train_auc = auc

        elif split == "backtest":

            backtest_auc = auc

        elif split == "validation":

            validation_auc = auc


    records.append(
        {

            "comparison": (
                "train_to_backtest"
            ),

            "source_auc": train_auc,

            "destination_auc": backtest_auc,

            "auc_change": (
                backtest_auc
                -
                train_auc
            ),
        }
    )


    records.append(
        {

            "comparison": (
                "train_to_validation"
            ),

            "source_auc": train_auc,

            "destination_auc": validation_auc,

            "auc_change": (
                validation_auc
                -
                train_auc
            ),
        }
    )


    records.append(
        {

            "comparison": (
                "backtest_to_validation"
            ),

            "source_auc": backtest_auc,

            "destination_auc": validation_auc,

            "auc_change": (
                validation_auc
                -
                backtest_auc
            ),
        }
    )


    temporal_df = pd.DataFrame(
        records
    )


    temporal_df.to_csv(
        TEMPORAL_GENERALIZATION_FILE,
        index=False,
    )


    print(
        temporal_df.to_string(
            index=False
        )
    )


    if not year_df.empty:

        print()

        print(
            "Year-by-year ROC-AUC:"
        )

        print(
            "-" * 50
        )


        for row in year_df.itertuples(
            index=False
        ):

            print(
                f"{row.year}: "
                f"{safe_float_text(row.roc_auc)}"
            )


    return temporal_df


# ============================================================
# TRAINING HISTORY
# ============================================================

def analyze_training_history():

    print_section(
        "TRAINING HISTORY DIAGNOSTICS"
    )


    if not TRAINING_HISTORY_FILE.exists():

        print(
            "training_history.csv not found."
        )

        return None


    history = pd.read_csv(
        TRAINING_HISTORY_FILE
    )


    if history.empty:

        print(
            "Training history is empty."
        )

        return history


    print(
        f"Epochs completed : "
        f"{len(history)}"
    )


    if "val_loss" in history.columns:

        best_index = (
            history[
                "val_loss"
            ]
            .idxmin()
        )


        best_epoch = (
            history.loc[
                best_index
            ]
        )


        print(
            f"Best epoch       : "
            f"{int(best_epoch['epoch'])}"
        )

        print(
            f"Best val_loss    : "
            f"{best_epoch['val_loss']:.6f}"
        )


    if (
        "accuracy"
        in history.columns

        and

        "val_accuracy"
        in history.columns
    ):

        final = history.iloc[
            -1
        ]


        print(
            f"Final train acc  : "
            f"{final['accuracy']:.4f}"
        )

        print(
            f"Final val acc    : "
            f"{final['val_accuracy']:.4f}"
        )


        gap = (

            final[
                "accuracy"
            ]

            -

            final[
                "val_accuracy"
            ]
        )


        print(
            f"Final acc gap    : "
            f"{gap:+.4f}"
        )


    return history


# ============================================================
# INTERPRETATION
# ============================================================

def interpret_results(
    metrics_df,
    probability_df,
    year_df,
):

    print_section(
        "DIAGNOSTIC INTERPRETATION"
    )


    if metrics_df.empty:

        print(
            "No metrics available."
        )

        return {}


    split_metrics = {}


    for split in SPLIT_ORDER:

        subset = metrics_df[
            metrics_df[
                "dataset_split"
            ]
            ==
            split
        ]


        if subset.empty:

            continue


        row = subset.iloc[
            0
        ]


        split_metrics[
            split
        ] = row


        print(
            f"{split.upper():10s} "
            f"ROC-AUC="
            f"{safe_float_text(row['roc_auc'])} | "
            f"F1="
            f"{safe_float_text(row['f1'])} | "
            f"actual HIGH="
            f"{row['actual_positive_rate']:.2%} | "
            f"predicted HIGH="
            f"{row['predicted_positive_rate']:.2%}"
        )


    diagnosis = {

        "class_collapse_detected": False,

        "training_signal_detected": False,

        "backtest_generalization_good": False,

        "validation_generalization_good": False,

        "probability_separation_train": None,

        "probability_separation_backtest": None,

        "probability_separation_validation": None,

        "interpretation": [],
    }


    # --------------------------------------------------------
    # CLASS COLLAPSE
    # --------------------------------------------------------

    validation = split_metrics.get(
        "validation"
    )


    if validation is not None:

        predicted_rate = float(
            validation[
                "predicted_positive_rate"
            ]
        )


        if (
            predicted_rate
            <
            0.01

            or

            predicted_rate
            >
            0.99
        ):

            diagnosis[
                "class_collapse_detected"
            ] = True


            diagnosis[
                "interpretation"
            ].append(
                "Prediction class collapse detected."
            )


        else:

            diagnosis[
                "interpretation"
            ].append(
                "Previous all-one-class prediction "
                "collapse is no longer present."
            )


    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    train = split_metrics.get(
        "train"
    )


    backtest = split_metrics.get(
        "backtest"
    )


    if train is not None:

        train_auc = float(
            train[
                "roc_auc"
            ]
        )


        if train_auc >= 0.60:

            diagnosis[
                "training_signal_detected"
            ] = True


    if backtest is not None:

        backtest_auc = float(
            backtest[
                "roc_auc"
            ]
        )


        if backtest_auc >= 0.55:

            diagnosis[
                "backtest_generalization_good"
            ] = True


    if validation is not None:

        validation_auc = float(
            validation[
                "roc_auc"
            ]
        )


        if validation_auc >= 0.55:

            diagnosis[
                "validation_generalization_good"
            ] = True


    # --------------------------------------------------------
    # PROBABILITY SEPARATION
    # --------------------------------------------------------

    for split in SPLIT_ORDER:

        subset = probability_df[
            probability_df[
                "dataset_split"
            ]
            ==
            split
        ]


        if subset.empty:

            continue


        separation = float(
            subset.iloc[
                0
            ][
                "probability_separation"
            ]
        )


        diagnosis[
            f"probability_separation_{split}"
        ] = separation


    print()

    print(
        "Interpretation:"
    )

    print(
        "-" * 70
    )


    if not diagnosis[
        "class_collapse_detected"
    ]:

        print(
            "[PASS] Model is no longer "
            "collapsed to one output class."
        )


    if diagnosis[
        "training_signal_detected"
    ]:

        print(
            "[INFO] Model learns signal "
            "inside the training distribution."
        )


    if not diagnosis[
        "backtest_generalization_good"
    ]:

        print(
            "[WARNING] Backtest discrimination "
            "is weak."
        )


    if not diagnosis[
        "validation_generalization_good"
    ]:

        print(
            "[WARNING] Final validation "
            "discrimination is weak."
        )


    if (
        train is not None

        and

        backtest is not None
    ):

        gap = (

            float(
                train[
                    "roc_auc"
                ]
            )

            -

            float(
                backtest[
                    "roc_auc"
                ]
            )
        )


        print(
            f"Train -> Backtest AUC gap: "
            f"{gap:+.4f}"
        )


        if gap > 0.08:

            print(
                "[WARNING] Large temporal "
                "generalization gap detected."
            )


    if validation is not None:

        validation_auc = float(
            validation[
                "roc_auc"
            ]
        )


        if validation_auc < 0.50:

            print(
                "[IMPORTANT] Validation ROC-AUC "
                "is below 0.50."
            )

            print(
                "The ranking relationship is "
                "not merely suffering from an "
                "incorrect 0.50 threshold."
            )


    print()

    print(
        "Threshold diagnostics must not be "
        "used to tune against 2025."
    )


    print()

    print(
        "Recommended research order:"
    )

    print(
        "  1. inspect year-by-year performance"
    )

    print(
        "  2. inspect per-stock performance"
    )

    print(
        "  3. compare simple baselines"
    )

    print(
        "  4. investigate target definition"
    )

    print(
        "  5. only then modify architecture"
    )


    return diagnosis


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

def save_diagnosis_summary(
    diagnosis,
    metrics_df,
    year_df,
):

    summary = {

        "default_threshold": (
            DEFAULT_THRESHOLD
        ),

        "prediction_source": str(
            EVALUATION_PREDICTIONS_FILE
        ),

        "diagnosis": (
            diagnosis
        ),

        "split_metrics": (
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

        "year_metrics": (
            year_df
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
        MODEL_DIAGNOSIS_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            default=str,
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_model_diagnostics():

    print_section(
        "AGENT 2 - BROAD-UNIVERSE MODEL DIAGNOSTICS"
    )


    print(
        "This diagnostic run does NOT:"
    )

    print(
        "  - retrain the neural network"
    )

    print(
        "  - alter model weights"
    )

    print(
        "  - recreate 822k sequence tensors"
    )

    print(
        "  - run expensive model inference again"
    )


    print()

    print(
        "It uses predictions already produced "
        "by trainer.py."
    )


    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # 1. VALIDATE
    # --------------------------------------------------------

    validate_inputs()


    # --------------------------------------------------------
    # 2. LOAD
    # --------------------------------------------------------

    results = (
        load_predictions()
    )


    # --------------------------------------------------------
    # 3. TRAINING HISTORY
    # --------------------------------------------------------

    history = (
        analyze_training_history()
    )


    # --------------------------------------------------------
    # 4. CORE METRICS
    # --------------------------------------------------------

    metrics_df = (
        evaluate_splits(
            results
        )
    )


    # --------------------------------------------------------
    # 5. PROBABILITY ANALYSIS
    # --------------------------------------------------------

    probability_df = (
        analyze_probabilities(
            results
        )
    )


    # --------------------------------------------------------
    # 6. THRESHOLD ANALYSIS
    # --------------------------------------------------------

    threshold_df = (
        run_threshold_analysis(
            results
        )
    )


    # --------------------------------------------------------
    # 7. PER STOCK
    # --------------------------------------------------------

    stock_df = (
        analyze_by_stock(
            results
        )
    )


    # --------------------------------------------------------
    # 8. PER YEAR
    # --------------------------------------------------------

    year_df = (
        analyze_by_year(
            results
        )
    )


    # --------------------------------------------------------
    # 9. CALIBRATION
    # --------------------------------------------------------

    calibration_df = (
        analyze_calibration(
            results
        )
    )


    # --------------------------------------------------------
    # 10. TEMPORAL
    # --------------------------------------------------------

    temporal_df = (
        analyze_temporal_generalization(
            metrics_df,
            year_df,
        )
    )


    # --------------------------------------------------------
    # 11. INTERPRET
    # --------------------------------------------------------

    diagnosis = (
        interpret_results(

            metrics_df,
            probability_df,
            year_df,
        )
    )


    # --------------------------------------------------------
    # 12. SAVE SUMMARY
    # --------------------------------------------------------

    save_diagnosis_summary(

        diagnosis,
        metrics_df,
        year_df,
    )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_section(
        "MODEL DIAGNOSTICS COMPLETE"
    )


    print(
        "Reports generated:"
    )

    print()

    print(
        f"  {DIAGNOSTIC_METRICS_FILE.name}"
    )

    print(
        f"  {PROBABILITY_SUMMARY_FILE.name}"
    )

    print(
        f"  {THRESHOLD_ANALYSIS_FILE.name}"
    )

    print(
        f"  {PER_STOCK_DIAGNOSTICS_FILE.name}"
    )

    print(
        f"  {PER_YEAR_DIAGNOSTICS_FILE.name}"
    )

    print(
        f"  {CALIBRATION_FILE.name}"
    )

    print(
        f"  {TEMPORAL_GENERALIZATION_FILE.name}"
    )

    print(
        f"  {MODEL_DIAGNOSIS_SUMMARY_FILE.name}"
    )


    print()

    print(
        "No retraining was performed."
    )

    print(
        "No model weights were modified."
    )


    return {

        "metrics": (
            metrics_df
        ),

        "probability_summary": (
            probability_df
        ),

        "threshold_analysis": (
            threshold_df
        ),

        "per_stock": (
            stock_df
        ),

        "per_year": (
            year_df
        ),

        "calibration": (
            calibration_df
        ),

        "temporal_generalization": (
            temporal_df
        ),

        "diagnosis": (
            diagnosis
        ),

        "training_history": (
            history
        ),
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_model_diagnostics()