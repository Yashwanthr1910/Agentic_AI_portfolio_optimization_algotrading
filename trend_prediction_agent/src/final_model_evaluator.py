"""
Agent 2 - Final Locked Ensemble Evaluator
==========================================

Purpose
-------
Evaluate the FINAL locked Agent 2 ensemble.

Locked models:
    Seed 42
    Seed 123
    Seed 456

Architecture:
    Dilated LSTM
    -> Transformer
    -> Progressive Attention
    -> Softmax

Target:
    relative_21d_top30

Procedure:
    1. Load existing trained seed weights.
    2. DO NOT retrain.
    3. Generate 2019-2024 backtest probabilities.
    4. Average probabilities across seeds.
    5. Evaluate ensemble on 2019-2024.
    6. Evaluate the same locked ensemble on 2025.
    7. Save final metrics, predictions, and model manifest.

IMPORTANT:
    No architecture, target, feature, threshold, or seed changes
    should be made after examining the 2025 results.
"""

from pathlib import Path

import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

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
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
TREND_AGENT_ROOT = CURRENT_FILE.parents[1]

if str(TREND_AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(TREND_AGENT_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import config as cfg

from src.trainer import (
    DynamicSequence,
    load_sequence_metadata,
    load_feature_matrix,
    load_sequence_index,
    validate_row_mapping,
)

from src.trend_model import (
    build_trend_model,
)


# ============================================================
# LOCKED FINAL CONFIGURATION
# ============================================================

FINAL_SEEDS = [
    42,
    123,
    456,
]

SEQUENCE_LENGTH = int(
    cfg.SEQUENCE_LENGTH
)

FEATURE_COLUMNS = list(
    cfg.TREND_FEATURE_COLUMNS
)

FEATURE_COUNT = len(
    FEATURE_COLUMNS
)

DECISION_THRESHOLD = 0.50

PREDICTION_BATCH_SIZE = 256


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path(
    cfg.MODEL_DIR
)

REPORT_DIR = Path(
    cfg.REPORT_DIR
)

SEED_MODEL_DIR = (
    MODEL_DIR
    / "seed_stability"
)

FINAL_REPORT_DIR = (
    REPORT_DIR
    / "final_model"
)

FINAL_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_METRICS_FILE = (
    FINAL_REPORT_DIR
    / "final_ensemble_metrics.csv"
)

FINAL_YEAR_METRICS_FILE = (
    FINAL_REPORT_DIR
    / "final_ensemble_year_metrics.csv"
)

FINAL_PREDICTIONS_FILE = (
    FINAL_REPORT_DIR
    / "final_ensemble_predictions.parquet"
)

FINAL_MANIFEST_FILE = (
    FINAL_REPORT_DIR
    / "final_model_manifest.json"
)

FINAL_SEED_METRICS_FILE = (
    FINAL_REPORT_DIR
    / "final_seed_metrics.csv"
)


# ============================================================
# DISPLAY
# ============================================================

def section(title):

    print()

    print(
        "=" * 95
    )

    print(title)

    print(
        "=" * 95
    )

    print()


# ============================================================
# MODEL PATH
# ============================================================

def get_seed_weights_path(
    seed,
):

    return (
        SEED_MODEL_DIR
        /
        f"full_paper_model_seed_{seed}.weights.h5"
    )


# ============================================================
# VALIDATE FINAL WEIGHTS
# ============================================================

def validate_weight_files():

    section(
        "VALIDATING FINAL SEED MODELS"
    )

    missing = []

    for seed in FINAL_SEEDS:

        path = get_seed_weights_path(
            seed
        )

        exists = path.exists()

        print(
            f"Seed {seed:<4} : "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        print(
            f"            {path}"
        )

        if not exists:
            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(
            "One or more final seed weight files "
            "could not be found:\n\n"
            +
            "\n".join(
                missing
            )
        )

    print()

    print(
        "All final seed model files found."
    )


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model():

    model = build_trend_model(
        compile_model=False
    )

    dummy = np.zeros(
        (
            1,
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ),
        dtype=np.float32,
    )

    _ = model(
        dummy,
        training=False,
    )

    if model.count_params() != 210052:

        print(
            "[WARNING] Expected approximately "
            "210,052 parameters."
        )

        print(
            f"Actual parameters: "
            f"{model.count_params():,}"
        )

    return model


# ============================================================
# GENERATOR
# ============================================================

def create_prediction_generator(
    feature_matrix,
    rows,
):

    return DynamicSequence(

        feature_matrix=(
            feature_matrix
        ),

        sequence_rows=(
            rows
        ),

        batch_size=(
            PREDICTION_BATCH_SIZE
        ),

        shuffle=False,

        class_weights=None,

        return_targets=False,
    )


# ============================================================
# GENERATE MODEL PROBABILITIES
# ============================================================

def predict_seed(
    seed,
    feature_matrix,
    rows,
):

    tf.keras.backend.clear_session()

    gc.collect()

    section(
        f"LOADING SEED {seed}"
    )

    model = build_model()

    weights_file = (
        get_seed_weights_path(
            seed
        )
    )

    model.load_weights(
        weights_file
    )

    print(
        f"Architecture : {model.name}"
    )

    print(
        f"Parameters   : "
        f"{model.count_params():,}"
    )

    print(
        f"Weights      : "
        f"{weights_file}"
    )

    generator = (
        create_prediction_generator(
            feature_matrix,
            rows,
        )
    )

    start = time.time()

    prediction = model.predict(
        generator,
        verbose=1,
    )

    seconds = (
        time.time()
        -
        start
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if (
        prediction.ndim != 2
        or
        prediction.shape[1] != 2
    ):

        raise ValueError(
            f"Unexpected prediction shape "
            f"for seed {seed}: "
            f"{prediction.shape}"
        )

    if len(prediction) != len(
        rows
    ):

        raise ValueError(
            f"Prediction count mismatch "
            f"for seed {seed}."
        )

    if not np.isfinite(
        prediction
    ).all():

        raise ValueError(
            f"Seed {seed} generated "
            "non-finite probabilities."
        )

    probability = (
        prediction[
            :,
            1
        ]
    )

    print()

    print(
        f"Samples         : "
        f"{len(probability):,}"
    )

    print(
        f"Mean P(TOP30)   : "
        f"{probability.mean():.6f}"
    )

    print(
        f"Prediction time : "
        f"{seconds / 60:.2f} minutes"
    )

    del model
    del generator

    tf.keras.backend.clear_session()

    gc.collect()

    return probability


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >=
        DECISION_THRESHOLD
    ).astype(
        np.int32
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {

        "samples":
            int(
                len(
                    y_true
                )
            ),

        "actual_top30_rate":
            float(
                y_true.mean()
            ),

        "predicted_top30_rate":
            float(
                predictions.mean()
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),

        "specificity":
            float(
                specificity
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probabilities,
                )
            ),

        "mean_probability":
            float(
                probabilities.mean()
            ),

        "min_probability":
            float(
                probabilities.min()
            ),

        "max_probability":
            float(
                probabilities.max()
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics,
):

    section(
        name
    )

    print(
        f"Samples              : "
        f"{metrics['samples']:,}"
    )

    print(
        f"Actual TOP30 rate    : "
        f"{metrics['actual_top30_rate']:.2%}"
    )

    print(
        f"Predicted TOP30 rate : "
        f"{metrics['predicted_top30_rate']:.2%}"
    )

    print()

    print(
        f"ROC-AUC              : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Average Precision    : "
        f"{metrics['average_precision']:.6f}"
    )

    print(
        f"Accuracy             : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision            : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall               : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1                   : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"Specificity          : "
        f"{metrics['specificity']:.6f}"
    )

    print(
        f"Brier score          : "
        f"{metrics['brier_score']:.6f}"
    )

    print()

    print(
        "Confusion matrix:"
    )

    print(
        f"TN = {metrics['tn']:,}"
    )

    print(
        f"FP = {metrics['fp']:,}"
    )

    print(
        f"FN = {metrics['fn']:,}"
    )

    print(
        f"TP = {metrics['tp']:,}"
    )


# ============================================================
# CREATE ENSEMBLE
# ============================================================

def generate_ensemble_predictions(
    feature_matrix,
    rows,
    split_name,
):

    section(
        f"GENERATING {split_name.upper()} "
        "ENSEMBLE PREDICTIONS"
    )

    seed_probabilities = {}

    for seed in FINAL_SEEDS:

        probability = predict_seed(

            seed=seed,

            feature_matrix=(
                feature_matrix
            ),

            rows=rows,
        )

        seed_probabilities[
            seed
        ] = probability

    stacked = np.column_stack(
        [
            seed_probabilities[
                seed
            ]
            for seed in FINAL_SEEDS
        ]
    )

    ensemble_probability = (
        stacked.mean(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Model disagreement measure
    # --------------------------------------------------------

    ensemble_std = (
        stacked.std(
            axis=1
        )
    )

    return (
        seed_probabilities,
        ensemble_probability,
        ensemble_std,
    )


# ============================================================
# BUILD RESULT FRAME
# ============================================================

def build_result_frame(
    rows,
    seed_probabilities,
    ensemble_probability,
    ensemble_std,
):

    columns = [
        "sequence_id",
        "symbol",
        "dataset_split",
        "prediction_date",
        "target_date",
        "future_return",
        "cross_section_percentile",
        "target",
    ]

    available = [
        column
        for column in columns
        if column in rows.columns
    ]

    result = rows[
        available
    ].copy()

    for seed in FINAL_SEEDS:

        result[
            f"probability_seed_{seed}"
        ] = (
            seed_probabilities[
                seed
            ]
        )

    result[
        "ensemble_probability"
    ] = (
        ensemble_probability
    )

    result[
        "ensemble_std"
    ] = (
        ensemble_std
    )

    result[
        "predicted_class"
    ] = (
        ensemble_probability
        >=
        DECISION_THRESHOLD
    ).astype(
        np.int8
    )

    return result


# ============================================================
# YEAR METRICS
# ============================================================

def calculate_year_metrics(
    result,
):

    data = result.copy()

    data[
        "year"
    ] = (
        pd.to_datetime(
            data[
                "prediction_date"
            ]
        )
        .dt
        .year
    )

    records = []

    for year, group in data.groupby(
        "year",
        sort=True,
    ):

        metrics = calculate_metrics(

            group[
                "target"
            ].to_numpy(),

            group[
                "ensemble_probability"
            ].to_numpy(),
        )

        records.append(
            {
                "dataset_split":
                    str(
                        group[
                            "dataset_split"
                        ].iloc[0]
                    ),

                "year":
                    int(year),

                "stocks":
                    int(
                        group[
                            "symbol"
                        ].nunique()
                    ),

                **metrics,
            }
        )

    return records


# ============================================================
# INDIVIDUAL SEED METRICS
# ============================================================

def calculate_seed_metrics(
    split_name,
    rows,
    seed_probabilities,
):

    records = []

    y_true = (
        rows[
            "target"
        ]
        .to_numpy()
    )

    for seed in FINAL_SEEDS:

        metrics = calculate_metrics(

            y_true,

            seed_probabilities[
                seed
            ],
        )

        records.append(
            {
                "dataset_split":
                    split_name,

                "seed":
                    int(seed),

                **metrics,
            }
        )

    return records


# ============================================================
# MAIN
# ============================================================

def run_final_model_evaluation():

    overall_start = time.time()

    section(
        "AGENT 2 - FINAL LOCKED ENSEMBLE EVALUATION"
    )

    print(
        "Architecture:"
    )

    print(
        "Dilated LSTM -> Transformer -> "
        "Progressive Attention -> Softmax"
    )

    print()

    print(
        "Target:"
    )

    print(
        "relative_21d_top30"
    )

    print()

    print(
        f"Locked seeds       : "
        f"{FINAL_SEEDS}"
    )

    print(
        f"Ensemble method    : "
        "Arithmetic mean probability"
    )

    print(
        f"Decision threshold : "
        f"{DECISION_THRESHOLD:.2f}"
    )

    print(
        f"Sequence shape     : "
        f"({SEQUENCE_LENGTH}, "
        f"{FEATURE_COUNT})"
    )

    print()

    print(
        "NO TRAINING WILL OCCUR."
    )

    # ========================================================
    # VALIDATE WEIGHTS
    # ========================================================

    validate_weight_files()

    # ========================================================
    # LOAD DATA
    # ========================================================

    metadata = (
        load_sequence_metadata()
    )

    (
        feature_data,
        feature_matrix,

    ) = load_feature_matrix(
        metadata
    )

    sequence_index = (
        load_sequence_index(
            metadata
        )
    )

    validate_row_mapping(

        sequence_index,

        feature_data,
    )

    # ========================================================
    # SPLITS
    # ========================================================

    backtest = (

        sequence_index[
            sequence_index[
                "dataset_split"
            ]
            ==
            "backtest"
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )

    validation = (

        sequence_index[
            sequence_index[
                "dataset_split"
            ]
            ==
            "validation"
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )

    section(
        "FINAL EVALUATION DATA"
    )

    print(
        f"2019-2024 backtest : "
        f"{len(backtest):,}"
    )

    print(
        f"2025 validation    : "
        f"{len(validation):,}"
    )

    print()

    print(
        "Backtest period:"
    )

    print(
        f"{backtest['prediction_date'].min()} "
        f"-> "
        f"{backtest['prediction_date'].max()}"
    )

    print()

    print(
        "Final validation period:"
    )

    print(
        f"{validation['prediction_date'].min()} "
        f"-> "
        f"{validation['prediction_date'].max()}"
    )

    # ========================================================
    # BACKTEST ENSEMBLE
    # ========================================================

    (
        backtest_seed_probabilities,
        backtest_ensemble,
        backtest_disagreement,

    ) = generate_ensemble_predictions(

        feature_matrix=(
            feature_matrix
        ),

        rows=(
            backtest
        ),

        split_name=(
            "2019-2024 backtest"
        ),
    )

    backtest_metrics = (
        calculate_metrics(

            backtest[
                "target"
            ].to_numpy(),

            backtest_ensemble,
        )
    )

    print_metrics(

        "FINAL ENSEMBLE - "
        "2019-2024 BACKTEST",

        backtest_metrics,
    )

    backtest_result = (
        build_result_frame(

            rows=(
                backtest
            ),

            seed_probabilities=(
                backtest_seed_probabilities
            ),

            ensemble_probability=(
                backtest_ensemble
            ),

            ensemble_std=(
                backtest_disagreement
            ),
        )
    )

    # ========================================================
    # FINAL 2025 EVALUATION
    # ========================================================

    section(
        "STARTING FINAL 2025 EVALUATION"
    )

    print(
        "The ensemble configuration is now locked."
    )

    print(
        "No changes should be made based on "
        "2025 results and then re-evaluated "
        "on the same period."
    )

    (
        validation_seed_probabilities,
        validation_ensemble,
        validation_disagreement,

    ) = generate_ensemble_predictions(

        feature_matrix=(
            feature_matrix
        ),

        rows=(
            validation
        ),

        split_name=(
            "2025 validation"
        ),
    )

    validation_metrics = (
        calculate_metrics(

            validation[
                "target"
            ].to_numpy(),

            validation_ensemble,
        )
    )

    print_metrics(

        "FINAL LOCKED ENSEMBLE - "
        "2025 VALIDATION",

        validation_metrics,
    )

    validation_result = (
        build_result_frame(

            rows=(
                validation
            ),

            seed_probabilities=(
                validation_seed_probabilities
            ),

            ensemble_probability=(
                validation_ensemble
            ),

            ensemble_std=(
                validation_disagreement
            ),
        )
    )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    all_predictions = pd.concat(
        [
            backtest_result,
            validation_result,
        ],
        ignore_index=True,
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    metric_records = [

        {
            "dataset_split":
                "backtest_2019_2024",

            **backtest_metrics,
        },

        {
            "dataset_split":
                "validation_2025",

            **validation_metrics,
        },
    ]

    metrics_df = pd.DataFrame(
        metric_records
    )

    # ========================================================
    # YEAR METRICS
    # ========================================================

    year_records = []

    year_records.extend(
        calculate_year_metrics(
            backtest_result
        )
    )

    year_records.extend(
        calculate_year_metrics(
            validation_result
        )
    )

    year_df = pd.DataFrame(
        year_records
    )

    # ========================================================
    # INDIVIDUAL SEED METRICS
    # ========================================================

    seed_metric_records = []

    seed_metric_records.extend(
        calculate_seed_metrics(

            split_name=(
                "backtest_2019_2024"
            ),

            rows=(
                backtest
            ),

            seed_probabilities=(
                backtest_seed_probabilities
            ),
        )
    )

    seed_metric_records.extend(
        calculate_seed_metrics(

            split_name=(
                "validation_2025"
            ),

            rows=(
                validation
            ),

            seed_probabilities=(
                validation_seed_probabilities
            ),
        )
    )

    seed_metrics_df = pd.DataFrame(
        seed_metric_records
    )

    # ========================================================
    # SAVE
    # ========================================================

    metrics_df.to_csv(
        FINAL_METRICS_FILE,
        index=False,
    )

    year_df.to_csv(
        FINAL_YEAR_METRICS_FILE,
        index=False,
    )

    seed_metrics_df.to_csv(
        FINAL_SEED_METRICS_FILE,
        index=False,
    )

    all_predictions.to_parquet(
        FINAL_PREDICTIONS_FILE,
        index=False,
    )

    # ========================================================
    # MODEL MANIFEST
    # ========================================================

    manifest = {

        "agent":
            "trend_prediction_agent",

        "model_status":
            "LOCKED_FINAL_ENSEMBLE",

        "target":
            "relative_21d_top30",

        "prediction_horizon_days":
            21,

        "relative_top_fraction":
            0.30,

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_count":
            FEATURE_COUNT,

        "features":
            FEATURE_COLUMNS,

        "architecture":
            (
                "Dilated LSTM -> Transformer -> "
                "Progressive Attention -> Softmax"
            ),

        "ensemble_method":
            "mean_probability",

        "ensemble_seeds":
            FINAL_SEEDS,

        "decision_threshold":
            DECISION_THRESHOLD,

        "weights":
            {
                str(seed):
                    str(
                        get_seed_weights_path(
                            seed
                        )
                    )
                for seed in FINAL_SEEDS
            },

        "selection_backtest":
            "2019-2024",

        "final_evaluation":
            "2025",

        "backtest_metrics":
            backtest_metrics,

        "validation_2025_metrics":
            validation_metrics,

        "methodology_note":
            (
                "The ensemble configuration was locked "
                "before this final evaluation. "
                "The 2025 period had appeared in earlier "
                "development diagnostics, therefore it "
                "should not be described as a completely "
                "untouched holdout in the final report."
            ),
    }

    with open(
        FINAL_MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=4,
            default=str,
        )

    # ========================================================
    # YEAR DISPLAY
    # ========================================================

    section(
        "FINAL YEAR-BY-YEAR ENSEMBLE ROC-AUC"
    )

    display_year = year_df[
        [
            "dataset_split",
            "year",
            "samples",
            "roc_auc",
            "average_precision",
            "accuracy",
            "f1",
        ]
    ]

    print(
        display_year.to_string(
            index=False
        )
    )

    # ========================================================
    # SEED VS ENSEMBLE
    # ========================================================

    section(
        "SEED VS ENSEMBLE COMPARISON"
    )

    print(
        seed_metrics_df[
            [
                "dataset_split",
                "seed",
                "roc_auc",
                "average_precision",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "ENSEMBLE:"
    )

    print(
        metrics_df[
            [
                "dataset_split",
                "roc_auc",
                "average_precision",
                "accuracy",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    section(
        "FINAL AGENT 2 MODEL ASSESSMENT"
    )

    backtest_auc = (
        backtest_metrics[
            "roc_auc"
        ]
    )

    validation_auc = (
        validation_metrics[
            "roc_auc"
        ]
    )

    print(
        f"2019-2024 ensemble AUC : "
        f"{backtest_auc:.6f}"
    )

    print(
        f"2025 final AUC         : "
        f"{validation_auc:.6f}"
    )

    print()

    if (
        backtest_auc > 0.53
        and
        validation_auc > 0.52
    ):

        status = (
            "AGENT 2 MODEL ACCEPTED"
        )

        next_step = (
            "Proceed to final predictor.py "
            "and Agent 1 -> Agent 2 inference."
        )

    elif (
        backtest_auc > 0.52
        and
        validation_auc > 0.50
    ):

        status = (
            "AGENT 2 MODEL ACCEPTED WITH "
            "MODEST PREDICTIVE EDGE"
        )

        next_step = (
            "Proceed to predictor.py, "
            "but document the modest "
            "out-of-sample predictive strength."
        )

    else:

        status = (
            "AGENT 2 MODEL REQUIRES CAUTION"
        )

        next_step = (
            "Do not tune against 2025. "
            "Record the result as final and "
            "decide whether the project should "
            "continue using the current model "
            "as a research prototype."
        )

    print(
        f"Status:"
    )

    print(
        status
    )

    print()

    print(
        next_step
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    section(
        "FINAL MODEL EVALUATION COMPLETE"
    )

    elapsed = (
        time.time()
        -
        overall_start
    )

    print(
        f"Runtime:"
        f" {elapsed / 60:.2f} minutes"
    )

    print()

    print(
        "Saved:"
    )

    print(
        FINAL_METRICS_FILE
    )

    print(
        FINAL_YEAR_METRICS_FILE
    )

    print(
        FINAL_SEED_METRICS_FILE
    )

    print(
        FINAL_PREDICTIONS_FILE
    )

    print(
        FINAL_MANIFEST_FILE
    )

    print()

    print(
        "NEXT STAGE:"
    )

    print(
        "Finalize predictor.py "
        "using this exact 3-seed ensemble."
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_final_model_evaluation()