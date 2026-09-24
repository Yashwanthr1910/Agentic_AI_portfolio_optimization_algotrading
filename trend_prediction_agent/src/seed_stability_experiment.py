"""
Agent 2 - Seed Stability Experiment
===================================

Purpose
-------
Test whether the winning Agent 2 architecture is stable across
different neural-network random seeds.

Locked methodology
------------------
Target:
    relative_21d_top30

Architecture:
    Dilated LSTM
    -> Transformer
    -> Progressive Attention
    -> Softmax

Data:
    Same preprocessed feature matrix
    Same 60 x 30 sequences
    Same chronological model-fitting split
    Same target-date embargo
    Same 2019-2024 backtest

IMPORTANT:
    2025 validation is NEVER evaluated in this script.

Seeds:
    42
    123
    456
"""

from pathlib import Path

import gc
import json
import random
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

CURRENT_FILE = Path(
    __file__
).resolve()

TREND_AGENT_ROOT = (
    CURRENT_FILE.parents[1]
)

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
    create_internal_validation,
    calculate_class_weights,
)

from src.trend_model import (
    build_trend_model,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = [
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

LEARNING_RATE = float(
    cfg.LEARNING_RATE
)

BATCH_SIZE = int(
    getattr(
        cfg,
        "BATCH_SIZE",
        64,
    )
)

PREDICTION_BATCH_SIZE = 256

MAX_EPOCHS = 6

EARLY_STOPPING_PATIENCE = 2

DECISION_THRESHOLD = 0.50


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

REPORT_DIR = Path(
    cfg.REPORT_DIR
)

MODEL_DIR = Path(
    cfg.MODEL_DIR
)

SEED_REPORT_DIR = (
    REPORT_DIR
    /
    "seed_stability"
)

SEED_MODEL_DIR = (
    MODEL_DIR
    /
    "seed_stability"
)

SEED_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OUTPUT FILES
# ============================================================

SUMMARY_FILE = (
    SEED_REPORT_DIR
    /
    "seed_stability_summary.csv"
)

YEAR_METRICS_FILE = (
    SEED_REPORT_DIR
    /
    "seed_stability_year_metrics.csv"
)

PREDICTIONS_FILE = (
    SEED_REPORT_DIR
    /
    "seed_stability_predictions.parquet"
)

DETAIL_FILE = (
    SEED_REPORT_DIR
    /
    "seed_stability_summary.json"
)


# ============================================================
# DISPLAY
# ============================================================

def section(
    title,
):

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


# ============================================================
# RANDOM-SEED CONTROL
# ============================================================

def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    tf.keras.utils.set_random_seed(
        seed
    )

    try:

        tf.config.experimental.enable_op_determinism()

    except Exception:

        pass


# ============================================================
# MODEL BUILDER
# ============================================================

def build_full_agent2_model():

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

    return model


# ============================================================
# MODEL COMPILATION
# ============================================================

def compile_model(
    model,
):

    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=(
                LEARNING_RATE
            )
        )
    )

    loss = (
        tf.keras.losses
        .SparseCategoricalCrossentropy()
    )

    model.compile(

        optimizer=optimizer,

        loss=loss,

        metrics=[
            tf.keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )


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

    predicted = (
        probabilities
        >=
        DECISION_THRESHOLD
    ).astype(
        np.int32
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predicted,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    specificity = (

        tn
        /
        (
            tn
            +
            fp
        )

        if
        (
            tn
            +
            fp
        )
        >
        0

        else
        np.nan
    )

    metrics = {

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
                predicted.mean()
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predicted,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    predicted,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    predicted,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predicted,
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

    return metrics


# ============================================================
# CREATE GENERATOR
# ============================================================

def create_generator(
    feature_matrix,
    rows,
    batch_size,
    shuffle,
    class_weights=None,
    return_targets=True,
):

    return DynamicSequence(

        feature_matrix=(
            feature_matrix
        ),

        sequence_rows=(
            rows
        ),

        batch_size=(
            batch_size
        ),

        shuffle=(
            shuffle
        ),

        class_weights=(
            class_weights
        ),

        return_targets=(
            return_targets
        ),
    )


# ============================================================
# PREDICT
# ============================================================

def generate_predictions(
    model,
    feature_matrix,
    rows,
):

    generator = create_generator(

        feature_matrix=(
            feature_matrix
        ),

        rows=(
            rows
        ),

        batch_size=(
            PREDICTION_BATCH_SIZE
        ),

        shuffle=False,

        class_weights=None,

        return_targets=False,
    )

    predictions = model.predict(
        generator,
        verbose=1,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    if predictions.ndim != 2:

        raise ValueError(
            "Prediction output must be "
            "2-dimensional."
        )

    if predictions.shape[1] != 2:

        raise ValueError(
            "Prediction output must contain "
            "exactly two class probabilities."
        )

    if len(predictions) != len(
        rows
    ):

        raise ValueError(
            "Prediction count does not match "
            "sequence count."
        )

    if not np.isfinite(
        predictions
    ).all():

        raise ValueError(
            "Prediction probabilities contain "
            "NaN or infinity."
        )

    return predictions[
        :,
        1,
    ]


# ============================================================
# YEAR-BY-YEAR METRICS
# ============================================================

def calculate_year_metrics(
    seed,
    result,
):

    output = []

    result = result.copy()

    result[
        "year"
    ] = (
        result[
            "prediction_date"
        ]
        .dt
        .year
    )

    for year, group in result.groupby(
        "year",
        sort=True,
    ):

        metrics = calculate_metrics(

            group[
                "target"
            ].to_numpy(),

            group[
                "top30_probability"
            ].to_numpy(),
        )

        record = {

            "seed":
                int(
                    seed
                ),

            "year":
                int(
                    year
                ),

            "stocks":
                int(
                    group[
                        "symbol"
                    ].nunique()
                ),

            **metrics,
        }

        output.append(
            record
        )

    return output


# ============================================================
# RUN ONE SEED
# ============================================================

def train_seed(
    seed,
    feature_matrix,
    fitting,
    internal_validation,
    backtest,
    class_weights,
):

    section(
        f"SEED {seed}"
    )

    print(
        f"Random seed       : "
        f"{seed}"
    )

    print(
        f"Architecture      : "
        f"full_paper_model"
    )

    print(
        f"Maximum epochs    : "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Learning rate     : "
        f"{LEARNING_RATE}"
    )

    print()

    # --------------------------------------------------------
    # Reset TensorFlow state
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    gc.collect()

    set_seed(
        seed
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = (
        build_full_agent2_model()
    )

    compile_model(
        model
    )

    print(
        f"Model input       : "
        f"{model.input_shape}"
    )

    print(
        f"Model output      : "
        f"{model.output_shape}"
    )

    print(
        f"Parameters        : "
        f"{model.count_params():,}"
    )

    if model.count_params() != 210052:

        print()

        print(
            "[WARNING]"
        )

        print(
            "Expected approximately "
            "210,052 parameters."
        )

    # --------------------------------------------------------
    # Training generators
    # --------------------------------------------------------

    training_generator = (
        create_generator(

            feature_matrix=(
                feature_matrix
            ),

            rows=(
                fitting
            ),

            batch_size=(
                BATCH_SIZE
            ),

            shuffle=True,

            class_weights=(
                class_weights
            ),

            return_targets=True,
        )
    )

    internal_validation_generator = (
        create_generator(

            feature_matrix=(
                feature_matrix
            ),

            rows=(
                internal_validation
            ),

            batch_size=(
                BATCH_SIZE
            ),

            shuffle=False,

            class_weights=None,

            return_targets=True,
        )
    )

    # --------------------------------------------------------
    # Seed-specific model path
    # --------------------------------------------------------

    weights_file = (

        SEED_MODEL_DIR

        /

        (
            f"full_paper_model_seed_"
            f"{seed}.weights.h5"
        )
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                weights_file
            ),

            monitor="val_loss",

            mode="min",

            save_best_only=True,

            save_weights_only=True,

            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            mode="min",

            patience=(
                EARLY_STOPPING_PATIENCE
            ),

            restore_best_weights=True,

            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            mode="min",

            factor=0.5,

            patience=1,

            min_lr=1e-6,

            verbose=1,
        ),
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    section(
        f"TRAINING SEED {seed}"
    )

    start_time = time.time()

    history = model.fit(

        training_generator,

        validation_data=(
            internal_validation_generator
        ),

        epochs=(
            MAX_EPOCHS
        ),

        callbacks=(
            callbacks
        ),

        verbose=1,
    )

    training_seconds = (
        time.time()
        -
        start_time
    )

    # --------------------------------------------------------
    # Explicitly load best weights
    # --------------------------------------------------------

    if not weights_file.exists():

        raise FileNotFoundError(
            f"Best weights not found:\n"
            f"{weights_file}"
        )

    model.load_weights(
        weights_file
    )

    history_df = pd.DataFrame(
        history.history
    )

    best_epoch = int(

        history_df[
            "val_loss"
        ]
        .idxmin()

        +
        1
    )

    best_val_loss = float(

        history_df[
            "val_loss"
        ]
        .min()
    )

    best_val_accuracy = float(

        history_df.loc[
            best_epoch - 1,
            "val_accuracy",
        ]
    )

    print()

    print(
        f"Training time     : "
        f"{training_seconds / 60:.2f} min"
    )

    print(
        f"Best epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Best val loss     : "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best val accuracy : "
        f"{best_val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Backtest prediction
    # --------------------------------------------------------

    section(
        f"SEED {seed} - 2019-2024 BACKTEST"
    )

    prediction_start = time.time()

    probabilities = (
        generate_predictions(

            model=(
                model
            ),

            feature_matrix=(
                feature_matrix
            ),

            rows=(
                backtest
            ),
        )
    )

    prediction_seconds = (
        time.time()
        -
        prediction_start
    )

    metrics = calculate_metrics(

        backtest[
            "target"
        ].to_numpy(),

        probabilities,
    )

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
        f"F1                   : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"Specificity          : "
        f"{metrics['specificity']:.6f}"
    )

    print(
        f"Actual TOP30 rate    : "
        f"{metrics['actual_top30_rate']:.2%}"
    )

    print(
        f"Predicted TOP30 rate : "
        f"{metrics['predicted_top30_rate']:.2%}"
    )

    print(
        f"Prediction time      : "
        f"{prediction_seconds / 60:.2f} min"
    )

    # --------------------------------------------------------
    # Save prediction frame
    # --------------------------------------------------------

    prediction_frame = (

        backtest[
            [
                "sequence_id",
                "symbol",
                "prediction_date",
                "target_date",
                "future_return",
                "cross_section_percentile",
                "target",
            ]
        ]

        .copy()
    )

    prediction_frame[
        "seed"
    ] = int(
        seed
    )

    prediction_frame[
        "top30_probability"
    ] = probabilities

    prediction_frame[
        "predicted_class"
    ] = (

        probabilities
        >=
        DECISION_THRESHOLD

    ).astype(
        np.int8
    )

    # --------------------------------------------------------
    # Year metrics
    # --------------------------------------------------------

    year_records = (
        calculate_year_metrics(

            seed=seed,

            result=(
                prediction_frame
            ),
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "seed":
            int(
                seed
            ),

        "parameters":
            int(
                model.count_params()
            ),

        "epochs_completed":
            int(
                len(
                    history_df
                )
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_internal_val_loss":
            float(
                best_val_loss
            ),

        "best_internal_val_accuracy":
            float(
                best_val_accuracy
            ),

        "training_seconds":
            float(
                training_seconds
            ),

        "training_minutes":
            float(
                training_seconds
                /
                60
            ),

        "prediction_seconds":
            float(
                prediction_seconds
            ),

        "weights_file":
            str(
                weights_file
            ),

        **metrics,
    }

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model

    del training_generator

    del internal_validation_generator

    tf.keras.backend.clear_session()

    gc.collect()

    return (
        summary,
        year_records,
        prediction_frame,
    )


# ============================================================
# STABILITY ANALYSIS
# ============================================================

def calculate_stability(
    summary_df,
    year_df,
):

    auc_values = (

        summary_df[
            "roc_auc"
        ]

        .to_numpy(
            dtype=np.float64
        )
    )

    mean_auc = float(
        np.mean(
            auc_values
        )
    )

    std_auc = float(
        np.std(
            auc_values,
            ddof=1,
        )
    )

    min_auc = float(
        np.min(
            auc_values
        )
    )

    max_auc = float(
        np.max(
            auc_values
        )
    )

    auc_range = float(
        max_auc
        -
        min_auc
    )

    # --------------------------------------------------------
    # Year stability across all seed/year combinations
    # --------------------------------------------------------

    yearly_mean = (

        year_df

        .groupby(
            "year"
        )[
            "roc_auc"
        ]

        .mean()
    )

    yearly_min = (

        year_df

        .groupby(
            "year"
        )[
            "roc_auc"
        ]

        .min()
    )

    years_mean_above_50 = int(
        (
            yearly_mean
            >
            0.50
        ).sum()
    )

    years_mean_above_52 = int(
        (
            yearly_mean
            >
            0.52
        ).sum()
    )

    worst_seed_year_auc = float(

        year_df[
            "roc_auc"
        ]
        .min()
    )

    # --------------------------------------------------------
    # Practical project-level stability interpretation
    #
    # These are OUR validation rules.
    # They are not thresholds specified by the paper.
    # --------------------------------------------------------

    if (
        mean_auc >= 0.53
        and
        std_auc <= 0.01
        and
        min_auc >= 0.52
    ):

        status = (
            "STABLE"
        )

        recommendation = (
            "Seed stability is acceptable. "
            "Lock the architecture and proceed "
            "to final model selection."
        )

    elif (
        mean_auc >= 0.525
        and
        std_auc <= 0.015
    ):

        status = (
            "ACCEPTABLE_WITH_VARIATION"
        )

        recommendation = (
            "The model contains useful signal "
            "but has noticeable seed variation. "
            "Consider selecting the median seed "
            "or using a small seed ensemble."
        )

    else:

        status = (
            "UNSTABLE"
        )

        recommendation = (
            "The architecture is too sensitive "
            "to initialization. Do not evaluate "
            "2025 yet. Stabilize training or "
            "consider an ensemble."
        )

    return {

        "seed_count":
            int(
                len(
                    auc_values
                )
            ),

        "mean_backtest_auc":
            mean_auc,

        "std_backtest_auc":
            std_auc,

        "min_backtest_auc":
            min_auc,

        "max_backtest_auc":
            max_auc,

        "backtest_auc_range":
            auc_range,

        "years_mean_auc_above_50":
            years_mean_above_50,

        "years_mean_auc_above_52":
            years_mean_above_52,

        "worst_seed_year_auc":
            worst_seed_year_auc,

        "status":
            status,

        "recommendation":
            recommendation,
    }


# ============================================================
# MAIN
# ============================================================

def run_seed_stability():

    overall_start = time.time()

    section(
        "AGENT 2 - SEED STABILITY EXPERIMENT"
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
        "Relative TOP30 based on "
        "21-day future return"
    )

    print()

    print(
        f"Seeds              : "
        f"{SEEDS}"
    )

    print(
        f"Sequence shape     : "
        f"({SEQUENCE_LENGTH}, "
        f"{FEATURE_COUNT})"
    )

    print(
        f"Epoch limit        : "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Early stop patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "2025 validation will NOT be "
        "loaded into model evaluation."
    )

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

    (
        fitting,
        internal_validation,
        embargoed,

    ) = create_internal_validation(
        sequence_index
    )

    class_weights = (
        calculate_class_weights(
            fitting
        )
    )

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

    protected_validation = (

        sequence_index[
            sequence_index[
                "dataset_split"
            ]
            ==
            "validation"
        ]
    )

    section(
        "DATA PARTITIONS"
    )

    print(
        f"Model fitting       : "
        f"{len(fitting):,}"
    )

    print(
        f"Internal validation : "
        f"{len(internal_validation):,}"
    )

    print(
        f"Embargoed           : "
        f"{len(embargoed):,}"
    )

    print(
        f"2019-2024 backtest  : "
        f"{len(backtest):,}"
    )

    print(
        f"Protected 2025      : "
        f"{len(protected_validation):,}"
    )

    # ========================================================
    # RUN SEEDS
    # ========================================================

    summary_records = []

    year_records = []

    prediction_frames = []

    for seed in SEEDS:

        (
            summary,
            seed_year_records,
            predictions,

        ) = train_seed(

            seed=(
                seed
            ),

            feature_matrix=(
                feature_matrix
            ),

            fitting=(
                fitting
            ),

            internal_validation=(
                internal_validation
            ),

            backtest=(
                backtest
            ),

            class_weights=(
                class_weights
            ),
        )

        summary_records.append(
            summary
        )

        year_records.extend(
            seed_year_records
        )

        prediction_frames.append(
            predictions
        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    summary_df = pd.DataFrame(
        summary_records
    )

    year_df = pd.DataFrame(
        year_records
    )

    prediction_df = pd.concat(

        prediction_frames,

        ignore_index=True,
    )

    # ========================================================
    # RANK SEEDS
    # ========================================================

    summary_df = (
        summary_df

        .sort_values(
            "roc_auc",
            ascending=False,
        )

        .reset_index(
            drop=True
        )
    )

    summary_df.insert(

        0,

        "rank",

        np.arange(
            1,
            len(summary_df)
            +
            1
        ),
    )

    # ========================================================
    # STABILITY ANALYSIS
    # ========================================================

    stability = (
        calculate_stability(

            summary_df,

            year_df,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    year_df.to_csv(
        YEAR_METRICS_FILE,
        index=False,
    )

    prediction_df.to_parquet(
        PREDICTIONS_FILE,
        index=False,
    )

    detail = {

        "experiment":
            "agent2_seed_stability",

        "target":
            "relative_21d_top30",

        "architecture":
            (
                "dilated_lstm_transformer_"
                "progressive_attention"
            ),

        "seeds":
            SEEDS,

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_count":
            FEATURE_COUNT,

        "max_epochs":
            MAX_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "fitting_sequences":
            int(
                len(
                    fitting
                )
            ),

        "internal_validation_sequences":
            int(
                len(
                    internal_validation
                )
            ),

        "embargoed_sequences":
            int(
                len(
                    embargoed
                )
            ),

        "backtest_sequences":
            int(
                len(
                    backtest
                )
            ),

        "protected_2025_sequences":
            int(
                len(
                    protected_validation
                )
            ),

        "validation_2025_used":
            False,

        "stability":
            stability,

        "seed_results":
            summary_df.to_dict(
                orient="records"
            ),
    }

    with open(
        DETAIL_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            detail,
            handle,
            indent=4,
            default=str,
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    section(
        "SEED STABILITY RESULTS"
    )

    columns = [

        "rank",

        "seed",

        "roc_auc",

        "average_precision",

        "accuracy",

        "f1",

        "best_epoch",

        "best_internal_val_loss",

        "training_minutes",
    ]

    print(
        summary_df[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # YEAR TABLE
    # ========================================================

    section(
        "YEAR-BY-YEAR ROC-AUC"
    )

    year_pivot = (

        year_df

        .pivot(
            index="year",
            columns="seed",
            values="roc_auc",
        )

        .sort_index()
    )

    year_pivot[
        "mean"
    ] = (
        year_pivot.mean(
            axis=1
        )
    )

    year_pivot[
        "min"
    ] = (
        year_pivot[
            SEEDS
        ]
        .min(
            axis=1
        )
    )

    year_pivot[
        "max"
    ] = (
        year_pivot[
            SEEDS
        ]
        .max(
            axis=1
        )
    )

    print(
        year_pivot.to_string()
    )

    # ========================================================
    # FINAL STABILITY
    # ========================================================

    section(
        "STABILITY SUMMARY"
    )

    print(
        f"Mean backtest AUC : "
        f"{stability['mean_backtest_auc']:.6f}"
    )

    print(
        f"Std backtest AUC  : "
        f"{stability['std_backtest_auc']:.6f}"
    )

    print(
        f"Minimum AUC       : "
        f"{stability['min_backtest_auc']:.6f}"
    )

    print(
        f"Maximum AUC       : "
        f"{stability['max_backtest_auc']:.6f}"
    )

    print(
        f"AUC range         : "
        f"{stability['backtest_auc_range']:.6f}"
    )

    print()

    print(
        f"Years mean AUC > 0.50 : "
        f"{stability['years_mean_auc_above_50']}"
        f"/6"
    )

    print(
        f"Years mean AUC > 0.52 : "
        f"{stability['years_mean_auc_above_52']}"
        f"/6"
    )

    print(
        f"Worst seed/year AUC   : "
        f"{stability['worst_seed_year_auc']:.6f}"
    )

    print()

    print(
        f"Stability status:"
    )

    print(
        stability[
            "status"
        ]
    )

    print()

    print(
        stability[
            "recommendation"
        ]
    )

    # ========================================================
    # BEST / MEDIAN SEED
    # ========================================================

    best_seed = int(
        summary_df.iloc[
            0
        ][
            "seed"
        ]
    )

    median_seed_row = (

        summary_df

        .assign(
            distance_from_median=
            (
                summary_df[
                    "roc_auc"
                ]
                -
                summary_df[
                    "roc_auc"
                ]
                .median()
            )
            .abs()
        )

        .sort_values(
            "distance_from_median"
        )

        .iloc[
            0
        ]
    )

    median_seed = int(
        median_seed_row[
            "seed"
        ]
    )

    print()

    print(
        f"Best backtest seed   : "
        f"{best_seed}"
    )

    print(
        f"Median-behavior seed : "
        f"{median_seed}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Do not automatically select the "
        "highest-AUC seed as the final model."
    )

    print(
        "Seed choice will be decided after "
        "checking stability and yearly behavior."
    )

    # ========================================================
    # RUNTIME
    # ========================================================

    elapsed = (
        time.time()
        -
        overall_start
    )

    section(
        "SEED STABILITY EXPERIMENT COMPLETE"
    )

    print(
        f"Total runtime : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()

    print(
        "2025 validation evaluated : NO"
    )

    print()

    print(
        "Saved summary:"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "Saved yearly metrics:"
    )

    print(
        YEAR_METRICS_FILE
    )

    print()

    print(
        "Saved predictions:"
    )

    print(
        PREDICTIONS_FILE
    )

    print()

    print(
        "Saved JSON:"
    )

    print(
        DETAIL_FILE
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_seed_stability()