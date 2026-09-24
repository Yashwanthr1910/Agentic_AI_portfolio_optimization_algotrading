"""
Agent 2 - Architecture Ablation Experiment
===========================================

Target:
    Relative 21-day TOP30 classification

Purpose:
    Determine whether Agent 2's poor out-of-sample performance
    comes from model complexity / architecture rather than the
    target or data pipeline.

Models compared:
    1. current_step_mlp
    2. plain_lstm
    3. dilated_lstm
    4. dilated_lstm_transformer
    5. full_paper_model

IMPORTANT:
    - Uses the exact same data and labels as trainer.py
    - Uses the exact same internal chronological split
    - Uses target-date embargo
    - Uses 2019-2024 ONLY as architecture-selection backtest
    - DOES NOT evaluate 2025 validation
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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
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
    create_internal_validation,
    calculate_class_weights,
)

from src.trend_model import build_trend_model


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = int(
    getattr(
        cfg,
        "RANDOM_SEED",
        42,
    )
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# CONFIG
# ============================================================

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

# ------------------------------------------------------------
# Ablation-specific settings
#
# Intentionally smaller than final training.
# We only need enough training to compare architectures.
# ------------------------------------------------------------

ABLATION_EPOCHS = 6

ABLATION_PATIENCE = 2

ABLATION_BATCH_SIZE = 64

PREDICTION_BATCH_SIZE = 256

DECISION_THRESHOLD = 0.50


# ============================================================
# OUTPUT PATHS
# ============================================================

REPORT_DIR = Path(
    cfg.REPORT_DIR
)

RESULTS_DIR = (
    REPORT_DIR
    / "ablation"
)

MODEL_DIR = (
    Path(cfg.MODEL_DIR)
    / "ablation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_FILE = (
    RESULTS_DIR
    / "architecture_ablation_summary.csv"
)

YEAR_METRICS_FILE = (
    RESULTS_DIR
    / "architecture_ablation_year_metrics.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "architecture_ablation_predictions.parquet"
)

DETAIL_FILE = (
    RESULTS_DIR
    / "architecture_ablation_summary.json"
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
# COMMON CLASSIFIER HEAD
# ============================================================

def classifier_head(
    x,
    name_prefix,
):

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name=f"{name_prefix}_dense",
    )(x)

    x = tf.keras.layers.Dropout(
        0.20,
        name=f"{name_prefix}_dropout",
    )(x)

    logits = tf.keras.layers.Dense(
        2,
        name=f"{name_prefix}_logits",
    )(x)

    output = tf.keras.layers.Softmax(
        name=f"{name_prefix}_probability",
    )(logits)

    return output


# ============================================================
# MODEL 1 - CURRENT STEP MLP
# ============================================================

def build_current_step_mlp():

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ),
        name="sequence_input",
    )

    # Only information available at prediction date.
    x = tf.keras.layers.Lambda(
        lambda tensor: tensor[:, -1, :],
        name="current_timestep",
    )(inputs)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="mlp_dense_1",
    )(x)

    x = tf.keras.layers.Dropout(
        0.20,
        name="mlp_dropout_1",
    )(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        name="mlp_dense_2",
    )(x)

    output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="trend_probability",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="current_step_mlp",
    )


# ============================================================
# MODEL 2 - PLAIN LSTM
# ============================================================

def build_plain_lstm():

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ),
        name="sequence_input",
    )

    x = tf.keras.layers.Dense(
        64,
        name="feature_embedding",
    )(inputs)

    x = tf.keras.layers.LSTM(
        64,
        return_sequences=False,
        dropout=0.20,
        name="plain_lstm",
    )(x)

    output = classifier_head(
        x,
        "plain_lstm",
    )

    return tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="plain_lstm_model",
    )


# ============================================================
# BUILD FULL ARCHITECTURE ONCE
# ============================================================

def build_full_base():

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
# MODEL 3 - DILATED LSTM ONLY
# ============================================================

def build_dilated_lstm_only():

    base = build_full_base()

    dilated_output = base.get_layer(
        "dilated_lstm_encoder"
    ).output

    x = tf.keras.layers.GlobalAveragePooling1D(
        name="dilated_global_pool"
    )(dilated_output)

    output = classifier_head(
        x,
        "dilated",
    )

    model = tf.keras.Model(
        inputs=base.input,
        outputs=output,
        name="dilated_lstm_only",
    )

    return model


# ============================================================
# MODEL 4 - DILATED LSTM + TRANSFORMER
# ============================================================

def build_dilated_transformer():

    base = build_full_base()

    transformer_output = base.get_layer(
        "transformer_encoder"
    ).output

    x = tf.keras.layers.GlobalAveragePooling1D(
        name="transformer_global_pool"
    )(transformer_output)

    output = classifier_head(
        x,
        "dilated_transformer",
    )

    model = tf.keras.Model(
        inputs=base.input,
        outputs=output,
        name="dilated_lstm_transformer_ablation",
    )

    return model


# ============================================================
# MODEL 5 - COMPLETE PAPER-INSPIRED MODEL
# ============================================================

def build_full_model():

    return build_trend_model(
        compile_model=False
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_BUILDERS = {

    "current_step_mlp":
        build_current_step_mlp,

    "plain_lstm":
        build_plain_lstm,

    "dilated_lstm":
        build_dilated_lstm_only,

    "dilated_lstm_transformer":
        build_dilated_transformer,

    "full_paper_model":
        build_full_model,
}


# ============================================================
# COMPILE
# ============================================================

def compile_model(model):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss=(
            tf.keras.losses
            .SparseCategoricalCrossentropy()
        ),

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

    predicted = (
        probability
        >=
        DECISION_THRESHOLD
    ).astype(
        np.int32
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {

        "samples":
            int(len(y_true)),

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
                    probability,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    y_true,
                    probability,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probability,
                )
            ),

        "mean_probability":
            float(
                probability.mean()
            ),
    }


# ============================================================
# PREDICTION GENERATOR
# ============================================================

def predict_model(
    model,
    feature_matrix,
    data,
):

    generator = DynamicSequence(

        feature_matrix=(
            feature_matrix
        ),

        sequence_rows=data,

        batch_size=(
            PREDICTION_BATCH_SIZE
        ),

        shuffle=False,

        class_weights=None,

        return_targets=False,
    )

    prediction = model.predict(
        generator,
        verbose=1,
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
            f"Unexpected prediction shape: "
            f"{prediction.shape}"
        )

    return prediction[:, 1]


# ============================================================
# YEAR METRICS
# ============================================================

def calculate_year_metrics(
    model_name,
    result,
):

    records = []

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

            group["target"],

            group["top30_probability"],
        )

        records.append(
            {
                "model":
                    model_name,

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
# TRAIN ONE MODEL
# ============================================================

def run_single_model(
    model_name,
    builder,
    feature_matrix,
    fitting,
    internal_validation,
    backtest,
    class_weights,
):

    section(
        f"ABLATION MODEL: {model_name}"
    )

    # --------------------------------------------------------
    # Make each experiment independent.
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    gc.collect()

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    tf.random.set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = builder()

    compile_model(
        model
    )

    print(
        f"Model name       : {model.name}"
    )

    print(
        f"Parameters       : "
        f"{model.count_params():,}"
    )

    print(
        f"Input shape      : "
        f"{model.input_shape}"
    )

    print(
        f"Output shape     : "
        f"{model.output_shape}"
    )

    # --------------------------------------------------------
    # Generators
    # --------------------------------------------------------

    training_generator = DynamicSequence(

        feature_matrix=(
            feature_matrix
        ),

        sequence_rows=(
            fitting
        ),

        batch_size=(
            ABLATION_BATCH_SIZE
        ),

        shuffle=True,

        class_weights=(
            class_weights
        ),

        return_targets=True,
    )

    validation_generator = DynamicSequence(

        feature_matrix=(
            feature_matrix
        ),

        sequence_rows=(
            internal_validation
        ),

        batch_size=(
            ABLATION_BATCH_SIZE
        ),

        shuffle=False,

        class_weights=None,

        return_targets=True,
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    weights_file = (
        MODEL_DIR
        /
        f"{model_name}.weights.h5"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            str(
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
                ABLATION_PATIENCE
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
    # Training
    # --------------------------------------------------------

    start_time = time.time()

    history = model.fit(

        training_generator,

        validation_data=(
            validation_generator
        ),

        epochs=(
            ABLATION_EPOCHS
        ),

        callbacks=callbacks,

        verbose=1,
    )

    training_seconds = (
        time.time()
        -
        start_time
    )

    if weights_file.exists():

        model.load_weights(
            weights_file
        )

    history_df = pd.DataFrame(
        history.history
    )

    best_epoch = int(
        history_df[
            "val_loss"
        ].idxmin()
        +
        1
    )

    best_val_loss = float(
        history_df[
            "val_loss"
        ].min()
    )

    best_val_accuracy = float(
        history_df.loc[
            best_epoch - 1,
            "val_accuracy"
        ]
    )

    # --------------------------------------------------------
    # 2019-2024 BACKTEST ONLY
    # --------------------------------------------------------

    section(
        f"{model_name} - 2019-2024 BACKTEST"
    )

    probability = predict_model(

        model,

        feature_matrix,

        backtest,
    )

    metrics = calculate_metrics(

        backtest[
            "target"
        ].to_numpy(),

        probability,
    )

    print(
        f"ROC-AUC           : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision : "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        f"Accuracy          : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"F1                : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Specificity       : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"Predicted TOP30   : "
        f"{metrics['predicted_top30_rate']:.2%}"
    )

    # --------------------------------------------------------
    # Prediction data
    # --------------------------------------------------------

    result = backtest[
        [
            "sequence_id",
            "symbol",
            "prediction_date",
            "target_date",
            "future_return",
            "cross_section_percentile",
            "target",
        ]
    ].copy()

    result[
        "model"
    ] = model_name

    result[
        "top30_probability"
    ] = probability

    result[
        "predicted_class"
    ] = (
        probability
        >=
        DECISION_THRESHOLD
    ).astype(
        np.int8
    )

    year_records = (
        calculate_year_metrics(
            model_name,
            result,
        )
    )

    summary = {

        "model":
            model_name,

        "parameters":
            int(
                model.count_params()
            ),

        "epochs_completed":
            int(
                len(history_df)
            ),

        "best_epoch":
            best_epoch,

        "best_internal_val_loss":
            best_val_loss,

        "best_internal_val_accuracy":
            best_val_accuracy,

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

        **metrics,
    }

    # --------------------------------------------------------
    # Free memory before next architecture
    # --------------------------------------------------------

    del model

    del training_generator

    del validation_generator

    tf.keras.backend.clear_session()

    gc.collect()

    return (
        summary,
        year_records,
        result,
    )


# ============================================================
# RANK RESULTS
# ============================================================

def rank_results(
    summary_df,
    year_df,
):

    stability = (

        year_df

        .groupby(
            "model"
        )["roc_auc"]

        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )

        .reset_index()
    )

    stability.columns = [

        "model",

        "mean_year_auc",

        "std_year_auc",

        "min_year_auc",

        "max_year_auc",
    ]

    count_above_50 = (

        year_df

        .assign(
            above_50=lambda x:
            x[
                "roc_auc"
            ]
            >
            0.50
        )

        .groupby(
            "model"
        )[
            "above_50"
        ]

        .sum()

        .reset_index(
            name="years_auc_above_50"
        )
    )

    count_above_52 = (

        year_df

        .assign(
            above_52=lambda x:
            x[
                "roc_auc"
            ]
            >
            0.52
        )

        .groupby(
            "model"
        )[
            "above_52"
        ]

        .sum()

        .reset_index(
            name="years_auc_above_52"
        )
    )

    ranked = (

        summary_df

        .merge(
            stability,
            on="model",
            how="left",
        )

        .merge(
            count_above_50,
            on="model",
            how="left",
        )

        .merge(
            count_above_52,
            on="model",
            how="left",
        )
    )

    # --------------------------------------------------------
    # Primary ranking:
    #
    # 1. aggregate backtest AUC
    # 2. mean annual AUC
    # 3. minimum annual AUC
    # --------------------------------------------------------

    ranked = ranked.sort_values(

        [
            "roc_auc",
            "mean_year_auc",
            "min_year_auc",
        ],

        ascending=[
            False,
            False,
            False,
        ],
    )

    ranked.insert(
        0,
        "rank",
        np.arange(
            1,
            len(ranked)
            +
            1
        ),
    )

    return ranked


# ============================================================
# MAIN
# ============================================================

def run_ablation():

    overall_start = time.time()

    section(
        "AGENT 2 - ARCHITECTURE ABLATION EXPERIMENT"
    )

    print(
        "Target             : "
        "relative_21d_top30"
    )

    print(
        f"Sequence shape     : "
        f"({SEQUENCE_LENGTH}, "
        f"{FEATURE_COUNT})"
    )

    print(
        f"Maximum epochs     : "
        f"{ABLATION_EPOCHS}"
    )

    print(
        f"Early-stop patience: "
        f"{ABLATION_PATIENCE}"
    )

    print()

    print(
        "2025 VALIDATION WILL NOT BE USED."
    )

    # ========================================================
    # LOAD EXISTING DATA
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

    validation = (
        sequence_index[
            sequence_index[
                "dataset_split"
            ]
            ==
            "validation"
        ]
    )

    print()

    print(
        f"Fitting sequences       : "
        f"{len(fitting):,}"
    )

    print(
        f"Internal validation     : "
        f"{len(internal_validation):,}"
    )

    print(
        f"Embargoed               : "
        f"{len(embargoed):,}"
    )

    print(
        f"Backtest sequences      : "
        f"{len(backtest):,}"
    )

    print(
        f"Protected 2025 sequences: "
        f"{len(validation):,}"
    )

    # ========================================================
    # EXPERIMENTS
    # ========================================================

    summaries = []

    year_records = []

    prediction_frames = []

    for model_name, builder in (
        MODEL_BUILDERS.items()
    ):

        try:

            (
                summary,
                model_year_records,
                predictions,

            ) = run_single_model(

                model_name=(
                    model_name
                ),

                builder=builder,

                feature_matrix=(
                    feature_matrix
                ),

                fitting=fitting,

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

            summaries.append(
                summary
            )

            year_records.extend(
                model_year_records
            )

            prediction_frames.append(
                predictions
            )

        except Exception as exc:

            print()

            print(
                f"[ERROR] {model_name} failed:"
            )

            print(
                exc
            )

            print()

            print(
                "Continuing with remaining "
                "architectures..."
            )

            summaries.append(
                {
                    "model":
                        model_name,

                    "error":
                        str(exc),
                }
            )

            tf.keras.backend.clear_session()

            gc.collect()

    # ========================================================
    # SAVE SUCCESSFUL RESULTS
    # ========================================================

    successful = [

        record

        for record in summaries

        if "roc_auc" in record
    ]

    if not successful:

        raise RuntimeError(
            "Every ablation architecture failed."
        )

    summary_df = pd.DataFrame(
        successful
    )

    year_df = pd.DataFrame(
        year_records
    )

    ranked = rank_results(

        summary_df,

        year_df,
    )

    ranked.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    year_df.to_csv(
        YEAR_METRICS_FILE,
        index=False,
    )

    if prediction_frames:

        predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        predictions.to_parquet(
            PREDICTIONS_FILE,
            index=False,
        )

    # ========================================================
    # JSON
    # ========================================================

    details = {

        "experiment":
            "agent2_architecture_ablation",

        "target":
            "relative_21d_top30",

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_count":
            FEATURE_COUNT,

        "ablation_epochs":
            ABLATION_EPOCHS,

        "patience":
            ABLATION_PATIENCE,

        "fitting_sequences":
            len(fitting),

        "internal_validation_sequences":
            len(
                internal_validation
            ),

        "embargoed_sequences":
            len(embargoed),

        "backtest_sequences":
            len(backtest),

        "protected_validation_sequences":
            len(validation),

        "validation_2025_used":
            False,

        "ranking":
            ranked.to_dict(
                orient="records"
            ),

        "failed_models":
            [
                x
                for x in summaries
                if "error" in x
            ],
    }

    with open(
        DETAIL_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            details,
            handle,
            indent=4,
            default=str,
        )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    section(
        "ARCHITECTURE ABLATION RESULTS"
    )

    display_columns = [

        "rank",

        "model",

        "parameters",

        "roc_auc",

        "average_precision",

        "mean_year_auc",

        "std_year_auc",

        "min_year_auc",

        "years_auc_above_50",

        "years_auc_above_52",

        "best_epoch",

        "training_minutes",
    ]

    print(
        ranked[
            display_columns
        ].to_string(
            index=False
        )
    )

    winner = ranked.iloc[
        0
    ]

    section(
        "ABLATION WINNER"
    )

    print(
        f"Model                 : "
        f"{winner['model']}"
    )

    print(
        f"2019-2024 ROC-AUC     : "
        f"{winner['roc_auc']:.4f}"
    )

    print(
        f"Mean yearly ROC-AUC   : "
        f"{winner['mean_year_auc']:.4f}"
    )

    print(
        f"Worst yearly ROC-AUC  : "
        f"{winner['min_year_auc']:.4f}"
    )

    print(
        f"Years > 0.50 AUC      : "
        f"{int(winner['years_auc_above_50'])}/6"
    )

    print(
        f"Years > 0.52 AUC      : "
        f"{int(winner['years_auc_above_52'])}/6"
    )

    print()

    print(
        "Reference benchmarks:"
    )

    print(
        "Full deep model previous backtest AUC : "
        "0.5014"
    )

    print(
        "Relative-target Logistic baseline      : "
        "~0.5530"
    )

    print()

    print(
        "2025 remains protected."
    )

    elapsed = (
        time.time()
        -
        overall_start
    )

    print()

    print(
        f"Total ablation runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()

    print(
        "Saved:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        YEAR_METRICS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        DETAIL_FILE
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_ablation()