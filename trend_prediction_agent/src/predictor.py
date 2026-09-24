"""
Agent 2 - Final Trend Predictor
===============================

Uses the locked final ensemble:

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

Purpose:
    Take the stocks selected by Agent 1,
    obtain their latest 60 preprocessed observations,
    run all three trained Agent 2 models,
    average their P(TOP30),
    rank the selected stocks,
    and save the final Agent 2 predictions.

IMPORTANT:
    No retraining.
    No scaler fitting.
    No threshold tuning.
"""

from pathlib import Path

import gc
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# PATH SETUP
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
# IMPORT PROJECT CONFIG / MODEL
# ============================================================

from config import config as cfg

from src.trend_model import (
    build_trend_model,
)


# ============================================================
# FINAL LOCKED CONFIGURATION
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


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path(
    cfg.MODEL_DIR
)

SEED_MODEL_DIR = (
    MODEL_DIR
    / "seed_stability"
)

PROCESSED_DIR = (
    TREND_AGENT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    TREND_AGENT_ROOT
    / "outputs"
)

REPORT_DIR = (
    TREND_AGENT_ROOT
    / "reports"
    / "final_inference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Broad data already transformed using the TRAIN-fitted scaler.
PREPROCESSED_DATA_FILE = (
    PROCESSED_DIR
    / "preprocessed_trend_data.parquet"
)

# Agent-1 selected-stock universe.
INFERENCE_UNIVERSE_FILE = (
    PROCESSED_DIR
    / "inference_universe.parquet"
)


OUTPUT_CSV = (
    OUTPUT_DIR
    / "trend_predictions.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "trend_predictions.parquet"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "prediction_summary.json"
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
# VALIDATE FILES
# ============================================================

def validate_files():

    section(
        "VALIDATING FINAL AGENT 2 FILES"
    )

    required = [
        PREPROCESSED_DATA_FILE,
        INFERENCE_UNIVERSE_FILE,
    ]

    for path in required:

        print(
            f"{path.name:<40} : "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    print()

    for seed in FINAL_SEEDS:

        weights = (
            SEED_MODEL_DIR
            /
            f"full_paper_model_seed_{seed}.weights.h5"
        )

        print(
            f"Seed {seed:<4} weights"
            f"{'':<20}: "
            f"{'FOUND' if weights.exists() else 'MISSING'}"
        )

        if not weights.exists():

            raise FileNotFoundError(
                f"Missing final model weights:\n"
                f"{weights}"
            )


# ============================================================
# LOAD AGENT 1 SELECTED SYMBOLS
# ============================================================

def load_selected_symbols():

    section(
        "LOADING AGENT 1 SELECTED STOCKS"
    )

    data = pd.read_parquet(
        INFERENCE_UNIVERSE_FILE
    )

    if "symbol" not in data.columns:

        raise ValueError(
            "inference_universe.parquet "
            "does not contain a symbol column."
        )

    symbols = (
        data[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    symbols = sorted(
        symbols
    )

    if len(symbols) == 0:

        raise ValueError(
            "No selected symbols found."
        )

    print(
        f"Selected stocks : "
        f"{len(symbols)}"
    )

    print()

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        print(
            f"{index:>2}. {symbol}"
        )

    return symbols


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data(
    selected_symbols,
):

    section(
        "LOADING PREPROCESSED TREND DATA"
    )

    data = pd.read_parquet(
        PREPROCESSED_DATA_FILE
    )

    required_columns = (
        [
            "symbol",
            "date",
        ]
        +
        FEATURE_COLUMNS
    )

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns in "
            "preprocessed trend data:\n"
            +
            "\n".join(
                missing
            )
        )

    data[
        "symbol"
    ] = (
        data[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

    data[
        "date"
    ] = pd.to_datetime(
        data[
            "date"
        ],
        errors="coerce",
    )

    data = data[
        data[
            "symbol"
        ].isin(
            selected_symbols
        )
    ].copy()

    data = (
        data
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

    print(
        f"Rows loaded       : "
        f"{len(data):,}"
    )

    print(
        f"Stocks available  : "
        f"{data['symbol'].nunique()}"
    )

    print(
        f"Date range        : "
        f"{data['date'].min()} "
        f"-> "
        f"{data['date'].max()}"
    )

    return data


# ============================================================
# BUILD LATEST SEQUENCES
# ============================================================

def build_latest_sequences(
    data,
    selected_symbols,
):

    section(
        "BUILDING LATEST 60-DAY SEQUENCES"
    )

    sequences = []

    metadata = []

    skipped = []

    for symbol in selected_symbols:

        stock = (
            data[
                data[
                    "symbol"
                ]
                ==
                symbol
            ]
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(stock) < SEQUENCE_LENGTH:

            skipped.append(
                {
                    "symbol":
                        symbol,

                    "reason":
                        (
                            "insufficient_history"
                        ),

                    "rows":
                        int(
                            len(stock)
                        ),
                }
            )

            continue

        latest = (
            stock
            .tail(
                SEQUENCE_LENGTH
            )
            .copy()
        )

        feature_values = (
            latest[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        if feature_values.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ):

            skipped.append(
                {
                    "symbol":
                        symbol,

                    "reason":
                        (
                            "incorrect_sequence_shape"
                        ),

                    "rows":
                        int(
                            len(latest)
                        ),
                }
            )

            continue

        if not np.isfinite(
            feature_values
        ).all():

            skipped.append(
                {
                    "symbol":
                        symbol,

                    "reason":
                        (
                            "non_finite_features"
                        ),

                    "rows":
                        int(
                            len(latest)
                        ),
                }
            )

            continue

        sequences.append(
            feature_values
        )

        metadata.append(
            {
                "symbol":
                    symbol,

                "sequence_start_date":
                    latest[
                        "date"
                    ].iloc[0],

                "prediction_date":
                    latest[
                        "date"
                    ].iloc[-1],

                "observations":
                    int(
                        len(latest)
                    ),
            }
        )

        print(
            f"{symbol:<20} "
            f"{latest['date'].iloc[0].date()} "
            f"-> "
            f"{latest['date'].iloc[-1].date()}"
        )

    if len(sequences) == 0:

        raise ValueError(
            "No valid inference sequences "
            "could be generated."
        )

    X = np.stack(
        sequences,
        axis=0,
    ).astype(
        np.float32
    )

    metadata_df = pd.DataFrame(
        metadata
    )

    print()

    print(
        f"Sequences created : "
        f"{len(X)}"
    )

    print(
        f"Sequence shape     : "
        f"{X.shape}"
    )

    print(
        f"Skipped stocks     : "
        f"{len(skipped)}"
    )

    return (
        X,
        metadata_df,
        skipped,
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_final_model():

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
            "[WARNING]"
        )

        print(
            f"Expected model parameters: "
            f"210,052"
        )

        print(
            f"Actual model parameters  : "
            f"{model.count_params():,}"
        )

    return model


# ============================================================
# PREDICT ONE SEED
# ============================================================

def predict_seed(
    seed,
    X,
):

    tf.keras.backend.clear_session()

    gc.collect()

    model = build_final_model()

    weights_file = (
        SEED_MODEL_DIR
        /
        f"full_paper_model_seed_{seed}.weights.h5"
    )

    model.load_weights(
        weights_file
    )

    probabilities = model.predict(
        X,
        batch_size=64,
        verbose=0,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if probabilities.shape != (
        len(X),
        2,
    ):

        raise ValueError(
            f"Seed {seed} returned "
            f"unexpected output shape: "
            f"{probabilities.shape}"
        )

    top30_probability = (
        probabilities[
            :,
            1
        ]
    )

    if not np.isfinite(
        top30_probability
    ).all():

        raise ValueError(
            f"Seed {seed} generated "
            "non-finite probabilities."
        )

    del model

    tf.keras.backend.clear_session()

    gc.collect()

    return top30_probability


# ============================================================
# RUN FINAL ENSEMBLE
# ============================================================

def predict_ensemble(
    X,
):

    section(
        "RUNNING FINAL 3-SEED ENSEMBLE"
    )

    predictions = {}

    for seed in FINAL_SEEDS:

        print(
            f"Predicting seed {seed}..."
        )

        probabilities = predict_seed(
            seed,
            X,
        )

        predictions[
            seed
        ] = probabilities

        print(
            f"Mean P(TOP30): "
            f"{probabilities.mean():.6f}"
        )

    stacked = np.column_stack(
        [
            predictions[
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

    ensemble_std = (
        stacked.std(
            axis=1
        )
    )

    return (
        predictions,
        ensemble_probability,
        ensemble_std,
    )


# ============================================================
# CREATE FINAL OUTPUT
# ============================================================

def build_output(
    metadata,
    seed_predictions,
    ensemble_probability,
    ensemble_std,
):

    result = metadata.copy()

    for seed in FINAL_SEEDS:

        result[
            f"probability_seed_{seed}"
        ] = (
            seed_predictions[
                seed
            ]
        )

    result[
        "top30_probability"
    ] = (
        ensemble_probability
    )

    result[
        "ensemble_std"
    ] = (
        ensemble_std
    )

    result[
        "trend_class"
    ] = np.where(
        result[
            "top30_probability"
        ]
        >=
        DECISION_THRESHOLD,
        "TOP30",
        "NOT_TOP30",
    )

    # --------------------------------------------------------
    # Cross-sectional ranking among Agent-1 candidates
    # --------------------------------------------------------

    result[
        "agent2_rank"
    ] = (
        result[
            "top30_probability"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    result = (
        result
        .sort_values(
            [
                "agent2_rank",
                "symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# DISPLAY
# ============================================================

def display_predictions(
    result,
):

    section(
        "FINAL AGENT 2 TREND PREDICTIONS"
    )

    display = result[
        [
            "agent2_rank",
            "symbol",
            "prediction_date",
            "top30_probability",
            "ensemble_std",
            "trend_class",
        ]
    ].copy()

    display[
        "top30_probability"
    ] = (
        display[
            "top30_probability"
        ]
        .map(
            lambda x:
            f"{x:.4f}"
        )
    )

    display[
        "ensemble_std"
    ] = (
        display[
            "ensemble_std"
        ]
        .map(
            lambda x:
            f"{x:.4f}"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_predictions(
    result,
    skipped,
):

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    result.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    summary = {

        "agent":
            "trend_prediction_agent",

        "status":
            "final_inference",

        "architecture":
            (
                "Dilated LSTM -> Transformer -> "
                "Progressive Attention -> Softmax"
            ),

        "target":
            "relative_21d_top30",

        "prediction_horizon_days":
            21,

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_count":
            FEATURE_COUNT,

        "ensemble_seeds":
            FINAL_SEEDS,

        "ensemble_method":
            "mean_probability",

        "decision_threshold":
            DECISION_THRESHOLD,

        "selected_stocks_received":
            int(
                len(result)
                +
                len(skipped)
            ),

        "predictions_generated":
            int(
                len(result)
            ),

        "skipped_stocks":
            skipped,

        "prediction_date_min":
            str(
                result[
                    "prediction_date"
                ].min()
            ),

        "prediction_date_max":
            str(
                result[
                    "prediction_date"
                ].max()
            ),

        "mean_top30_probability":
            float(
                result[
                    "top30_probability"
                ].mean()
            ),

        "top30_predictions":
            int(
                (
                    result[
                        "trend_class"
                    ]
                    ==
                    "TOP30"
                )
                .sum()
            ),

        "output_csv":
            str(
                OUTPUT_CSV
            ),

        "output_parquet":
            str(
                OUTPUT_PARQUET
            ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=4,
            default=str,
        )


# ============================================================
# MAIN
# ============================================================

def run_predictor():

    section(
        "AGENT 2 - FINAL TREND PREDICTOR"
    )

    print(
        "Locked architecture:"
    )

    print(
        "Dilated LSTM -> Transformer -> "
        "Progressive Attention -> Softmax"
    )

    print()

    print(
        f"Ensemble seeds : "
        f"{FINAL_SEEDS}"
    )

    print(
        f"Input shape    : "
        f"({SEQUENCE_LENGTH}, "
        f"{FEATURE_COUNT})"
    )

    print(
        f"Target         : "
        "relative_21d_top30"
    )

    print()

    print(
        "No training will occur."
    )

    print(
        "No scaler will be fitted."
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_files()

    # --------------------------------------------------------
    # Agent 1 symbols
    # --------------------------------------------------------

    selected_symbols = (
        load_selected_symbols()
    )

    # --------------------------------------------------------
    # Already standardized broad data
    # --------------------------------------------------------

    data = load_preprocessed_data(
        selected_symbols
    )

    # --------------------------------------------------------
    # Latest 60-day sequences
    # --------------------------------------------------------

    (
        X,
        metadata,
        skipped,

    ) = build_latest_sequences(
        data,
        selected_symbols,
    )

    # --------------------------------------------------------
    # Ensemble prediction
    # --------------------------------------------------------

    (
        seed_predictions,
        ensemble_probability,
        ensemble_std,

    ) = predict_ensemble(
        X
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = build_output(

        metadata=metadata,

        seed_predictions=(
            seed_predictions
        ),

        ensemble_probability=(
            ensemble_probability
        ),

        ensemble_std=(
            ensemble_std
        ),
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_predictions(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_predictions(
        result,
        skipped,
    )

    section(
        "AGENT 2 FINAL PREDICTION COMPLETE"
    )

    print(
        f"Stocks received  : "
        f"{len(selected_symbols)}"
    )

    print(
        f"Predictions made : "
        f"{len(result)}"
    )

    print(
        f"Stocks skipped   : "
        f"{len(skipped)}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        OUTPUT_PARQUET
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Validate Agent 1 -> Agent 2 integration."
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_predictor()