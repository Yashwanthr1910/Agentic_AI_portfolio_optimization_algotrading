"""
trainer.py

Agent 2 - Memory-Efficient Trend Model Trainer
===============================================

CURRENT EXPERIMENT
------------------
Target:
    Relative cross-sectional TOP-30% classification.

Class 0:
    NOT_TOP30

Class 1:
    TOP30

A stock receives class 1 when its future 21-trading-day return
belongs to the top 30% of eligible stocks on the same prediction date.

The neural architecture is intentionally unchanged:

    60 x 30 input
        ->
    Feature embedding
        ->
    Dilated LSTM
        ->
    Transformer
        ->
    Progressive Attention
        ->
    Dense(2)
        ->
    Softmax

Purpose
-------
Run a clean target-only A/B experiment against the previous:

    21-day future return >= +2%

experiment.

IMPORTANT RESEARCH RULES
------------------------
1. Scaling has already been fitted on TRAIN only.
2. No scaler is refitted here.
3. Internal validation comes only from TRAIN.
4. Target-date embargo prevents overlap across the internal cutoff.
5. 2019-2024 BACKTEST is never used for early stopping.
6. 2025 VALIDATION is never used for training or early stopping.
7. Decision threshold remains fixed at 0.50.
8. New model files are saved separately from the old absolute-target model.
"""

from pathlib import Path
import json
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

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
# PROJECT PATH SETUP
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

from src.trend_model import (
    build_trend_model,
)


# ============================================================
# WARNING CONTROL
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# ============================================================
# RANDOM REPRODUCIBILITY
# ============================================================

RANDOM_SEED = int(
    cfg.RANDOM_SEED
)

os.environ[
    "PYTHONHASHSEED"
] = str(
    RANDOM_SEED
)

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# INPUT PATHS
# ============================================================

FEATURE_FILE = Path(
    cfg.PREPROCESSED_TREND_DATA_FILE
)

SEQUENCE_INDEX_FILE = Path(
    cfg.SEQUENCE_INDEX_FILE
)

SEQUENCE_METADATA_FILE = Path(
    cfg.SEQUENCE_METADATA_FILE
)


# ============================================================
# MODEL OUTPUT PATHS
# ============================================================

MODEL_DIR = Path(
    cfg.MODEL_DIR
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Do NOT overwrite previous absolute-target model.
# ------------------------------------------------------------

BEST_WEIGHTS_FILE = (
    MODEL_DIR
    / "best_trend_model_relative_21d_top30.weights.h5"
)

FINAL_MODEL_FILE = (
    MODEL_DIR
    / "trend_model_relative_21d_top30.keras"
)


# ============================================================
# REPORT PATHS
# ============================================================

REPORT_DIR = Path(
    cfg.REPORT_DIR
)

RESULTS_DIR = Path(
    cfg.RESULTS_DIR
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAINING_HISTORY_FILE = (
    REPORT_DIR
    / "training_history_relative_21d_top30.csv"
)

TREND_METRICS_FILE = (
    REPORT_DIR
    / "trend_metrics_relative_21d_top30.csv"
)

TRAINING_SUMMARY_FILE = (
    REPORT_DIR
    / "training_summary_relative_21d_top30.json"
)

EVALUATION_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "trend_evaluation_predictions_relative_21d_top30.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_COLUMNS = list(
    cfg.TREND_FEATURE_COLUMNS
)

SEQUENCE_LENGTH = int(
    cfg.SEQUENCE_LENGTH
)

PREDICTION_HORIZON = int(
    cfg.PREDICTION_HORIZON
)

TARGET_MODE = str(
    cfg.TARGET_MODE
)

RELATIVE_TOP_FRACTION = float(
    cfg.RELATIVE_TOP_FRACTION
)

TARGET_CLASS_NAMES = dict(
    cfg.TARGET_CLASS_NAMES
)

INTERNAL_VALIDATION_CUTOFF = pd.Timestamp(
    cfg.INTERNAL_VALIDATION_CUTOFF
)


BATCH_SIZE = int(
    cfg.BATCH_SIZE
)

EPOCHS = int(
    cfg.EPOCHS
)

LEARNING_RATE = float(
    cfg.LEARNING_RATE
)

EARLY_STOPPING_PATIENCE = int(
    cfg.EARLY_STOPPING_PATIENCE
)

REDUCE_LR_PATIENCE = int(
    cfg.REDUCE_LR_PATIENCE
)

REDUCE_LR_FACTOR = float(
    getattr(
        cfg,
        "REDUCE_LR_FACTOR",
        0.50,
    )
)

MIN_LEARNING_RATE = float(
    cfg.MIN_LEARNING_RATE
)

CLASSIFICATION_THRESHOLD = float(
    getattr(
        cfg,
        "CLASSIFICATION_THRESHOLD",
        0.50,
    )
)

USE_CLASS_WEIGHTS = bool(
    getattr(
        cfg,
        "USE_CLASS_WEIGHTS",
        True,
    )
)


# ============================================================
# DISPLAY
# ============================================================

def print_section(
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


def print_subsection(
    title,
):

    print()

    print(
        "-" * 72
    )

    print(
        title
    )

    print(
        "-" * 72
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_files():

    print_section(
        "VALIDATING TRAINER INPUT FILES"
    )

    required_files = [

        FEATURE_FILE,

        SEQUENCE_INDEX_FILE,

        SEQUENCE_METADATA_FILE,
    ]

    for path in required_files:

        print(
            path
        )

        if not path.exists():

            raise FileNotFoundError(
                "\nRequired trainer input "
                "file was not found:\n"
                f"{path}"
            )

    print()

    print(
        "All trainer input files found."
    )


# ============================================================
# METADATA
# ============================================================

def load_sequence_metadata():

    print_section(
        "LOADING SEQUENCE METADATA"
    )

    with open(
        SEQUENCE_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    required_keys = [

        "storage",

        "sequence_length",

        "prediction_horizon",

        "feature_count",

        "feature_columns",

        "target_mode",

        "total_sequences",
    ]

    missing = [

        key

        for key in required_keys

        if key not in metadata
    ]

    if missing:

        raise KeyError(
            "\nSequence metadata missing "
            f"required keys:\n{missing}"
        )

    stored_sequence_length = int(
        metadata[
            "sequence_length"
        ]
    )

    stored_horizon = int(
        metadata[
            "prediction_horizon"
        ]
    )

    stored_feature_count = int(
        metadata[
            "feature_count"
        ]
    )

    stored_features = list(
        metadata[
            "feature_columns"
        ]
    )

    stored_target_mode = str(
        metadata[
            "target_mode"
        ]
    )

    total_sequences = int(
        metadata[
            "total_sequences"
        ]
    )

    print(
        f"Storage             : "
        f"{metadata['storage']}"
    )

    print(
        f"Sequence length     : "
        f"{stored_sequence_length}"
    )

    print(
        f"Prediction horizon  : "
        f"{stored_horizon}"
    )

    print(
        f"Features            : "
        f"{stored_feature_count}"
    )

    print(
        f"Target mode         : "
        f"{stored_target_mode}"
    )

    if (
        "relative_top_fraction"
        in metadata
    ):

        print(
            f"Relative top share  : "
            f"{float(metadata['relative_top_fraction']):.2%}"
        )

    if (
        "relative_percentile_cutoff"
        in metadata
    ):

        print(
            f"Percentile cutoff   : "
            f"{float(metadata['relative_percentile_cutoff']):.2f}"
        )

    print(
        f"Class 0             : "
        f"{metadata.get('class_0', 'NOT_TOP30')}"
    )

    print(
        f"Class 1             : "
        f"{metadata.get('class_1', 'TOP30')}"
    )

    print(
        f"Total sequences     : "
        f"{total_sequences:,}"
    )

    # ========================================================
    # STRICT METADATA CHECKS
    # ========================================================

    if (
        stored_sequence_length
        !=
        SEQUENCE_LENGTH
    ):

        raise ValueError(
            "\nSequence-length mismatch.\n"
            f"Config   : {SEQUENCE_LENGTH}\n"
            f"Metadata : {stored_sequence_length}"
        )

    if (
        stored_horizon
        !=
        PREDICTION_HORIZON
    ):

        raise ValueError(
            "\nPrediction-horizon mismatch.\n"
            f"Config   : {PREDICTION_HORIZON}\n"
            f"Metadata : {stored_horizon}"
        )

    if (
        stored_feature_count
        !=
        len(
            FEATURE_COLUMNS
        )
    ):

        raise ValueError(
            "\nFeature-count mismatch.\n"
            f"Config   : "
            f"{len(FEATURE_COLUMNS)}\n"
            f"Metadata : "
            f"{stored_feature_count}"
        )

    if (
        stored_features
        !=
        FEATURE_COLUMNS
    ):

        raise ValueError(
            "\nFeature order mismatch between "
            "config.py and sequence_metadata.json."
        )

    if (
        stored_target_mode
        !=
        TARGET_MODE
    ):

        raise ValueError(
            "\nTarget-mode mismatch.\n"
            f"Config   : {TARGET_MODE}\n"
            f"Metadata : {stored_target_mode}"
        )

    if TARGET_MODE != (
        "relative_top_fraction"
    ):

        raise ValueError(
            "\nThis trainer is intended for the "
            "relative TOP30 experiment.\n"
            f"Current TARGET_MODE: "
            f"{TARGET_MODE}"
        )

    stored_fraction = float(
        metadata.get(
            "relative_top_fraction",
            -1,
        )
    )

    if not np.isclose(
        stored_fraction,
        RELATIVE_TOP_FRACTION,
    ):

        raise ValueError(
            "\nRelative target fraction mismatch.\n"
            f"Config   : "
            f"{RELATIVE_TOP_FRACTION}\n"
            f"Metadata : "
            f"{stored_fraction}"
        )

    print()

    print(
        "Sequence metadata validation PASSED."
    )

    return metadata


# ============================================================
# LOAD PREPROCESSED FEATURE MATRIX
# ============================================================

def load_feature_matrix(
    metadata,
):

    print_section(
        "LOADING PREPROCESSED FEATURE MATRIX"
    )

    required_columns = [

        "symbol",

        "date",

        "dataset_split",

        *FEATURE_COLUMNS,
    ]

    feature_data = pd.read_parquet(

        FEATURE_FILE,

        columns=required_columns,
    )

    feature_data[
        "symbol"
    ] = (

        feature_data[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    feature_data[
        "dataset_split"
    ] = (

        feature_data[
            "dataset_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    feature_data[
        "date"
    ] = pd.to_datetime(
        feature_data[
            "date"
        ]
    )

    # --------------------------------------------------------
    # MUST match sequence_builder.py source-row ordering.
    # --------------------------------------------------------

    feature_data = (

        feature_data

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

    feature_matrix = (

        feature_data[
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

    source_rows = metadata.get(
        "source_rows"
    )

    if (
        source_rows
        is not None
    ):

        if int(
            source_rows
        ) != len(
            feature_data
        ):

            raise ValueError(
                "\nMetadata source_rows mismatch.\n"
                f"Metadata : {source_rows:,}\n"
                f"Actual   : "
                f"{len(feature_data):,}\n\n"
                "Rerun sequence_builder.py after "
                "correcting source_rows metadata."
            )

    ram_mb = (
        feature_matrix.nbytes
        /
        1024
        /
        1024
    )

    print(
        f"Rows              : "
        f"{len(feature_data):,}"
    )

    print(
        f"Stocks            : "
        f"{feature_data['symbol'].nunique():,}"
    )

    print(
        f"Feature matrix    : "
        f"{feature_matrix.shape}"
    )

    print(
        f"Feature dtype     : "
        f"{feature_matrix.dtype}"
    )

    print(
        f"Feature RAM usage : "
        f"{ram_mb:.2f} MB"
    )

    return (
        feature_data,
        feature_matrix,
    )


# ============================================================
# LOAD SEQUENCE INDEX
# ============================================================

def load_sequence_index(
    metadata,
):

    print_section(
        "LOADING RELATIVE-TARGET SEQUENCE INDEX"
    )

    sequence_index = pd.read_parquet(
        SEQUENCE_INDEX_FILE
    )

    required_columns = [

        "sequence_id",

        "symbol",

        "dataset_split",

        "start_row",

        "end_row",

        "prediction_row",

        "target_row",

        "prediction_date",

        "target_date",

        "future_return",

        "cross_section_percentile",

        "target",
    ]

    missing = [

        column

        for column in required_columns

        if column
        not in sequence_index.columns
    ]

    if missing:

        raise KeyError(
            "\nMissing sequence-index columns:\n"
            f"{missing}"
        )

    sequence_index[
        "symbol"
    ] = (

        sequence_index[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    sequence_index[
        "dataset_split"
    ] = (

        sequence_index[
            "dataset_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sequence_index[
        "prediction_date"
    ] = pd.to_datetime(
        sequence_index[
            "prediction_date"
        ]
    )

    sequence_index[
        "target_date"
    ] = pd.to_datetime(
        sequence_index[
            "target_date"
        ]
    )

    sequence_index[
        "target"
    ] = (

        sequence_index[
            "target"
        ]
        .astype(
            np.int32
        )
    )

    if (
        len(
            sequence_index
        )
        !=
        int(
            metadata[
                "total_sequences"
            ]
        )
    ):

        raise ValueError(
            "\nSequence count mismatch.\n"
            f"Metadata : "
            f"{metadata['total_sequences']:,}\n"
            f"Index    : "
            f"{len(sequence_index):,}"
        )

    if sequence_index[
        "sequence_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate sequence IDs detected."
        )

    if not sequence_index[
        "target"
    ].isin(
        [
            0,
            1,
        ]
    ).all():

        raise ValueError(
            "Targets must contain only 0 and 1."
        )

    print(
        f"Sequences : "
        f"{len(sequence_index):,}"
    )

    for split in [

        "train",

        "backtest",

        "validation",

    ]:

        subset = sequence_index[
            sequence_index[
                "dataset_split"
            ]
            ==
            split
        ]

        if subset.empty:

            continue

        print(
            f"{split.upper():10s}: "
            f"{len(subset):10,d} | "
            f"TOP30 "
            f"{subset['target'].mean():7.2%}"
        )

    print()

    print(
        "Sequence-index validation PASSED."
    )

    return sequence_index


# ============================================================
# VALIDATE ROW MAPPING
# ============================================================

def validate_row_mapping(
    sequence_index,
    feature_data,
):

    print_section(
        "VALIDATING SEQUENCE ROW MAPPING"
    )

    sample_size = min(
        2000,
        len(
            sequence_index
        ),
    )

    sample = sequence_index.sample(

        n=sample_size,

        random_state=RANDOM_SEED,
    )

    failures = 0

    for row in sample.itertuples(
        index=False
    ):

        start = int(
            row.start_row
        )

        end = int(
            row.end_row
        )

        if (
            start < 0
            or
            end >= len(
                feature_data
            )
        ):

            failures += 1
            continue

        window = feature_data.iloc[
            start:
            end + 1
        ]

        if len(
            window
        ) != SEQUENCE_LENGTH:

            failures += 1
            continue

        if (
            window[
                "symbol"
            ].nunique()
            !=
            1
        ):

            failures += 1
            continue

        if (
            window[
                "dataset_split"
            ].nunique()
            !=
            1
        ):

            failures += 1
            continue

        if (
            window.iloc[-1][
                "symbol"
            ]
            !=
            row.symbol
        ):

            failures += 1
            continue

        if (
            window.iloc[-1][
                "date"
            ]
            !=
            row.prediction_date
        ):

            failures += 1

    print(
        f"Mappings checked : "
        f"{sample_size:,}"
    )

    print(
        f"Failures         : "
        f"{failures:,}"
    )

    if failures > 0:

        raise ValueError(
            "Sequence row mapping failed."
        )

    print()

    print(
        "Sequence row mapping PASSED."
    )


# ============================================================
# INTERNAL CHRONOLOGICAL VALIDATION
# ============================================================

def create_internal_validation(
    sequence_index,
):

    print_section(
        "CREATING INTERNAL CHRONOLOGICAL VALIDATION"
    )

    train = sequence_index[
        sequence_index[
            "dataset_split"
        ]
        ==
        "train"
    ].copy()

    cutoff = pd.Timestamp(
        INTERNAL_VALIDATION_CUTOFF
    )

    # --------------------------------------------------------
    # MODEL FITTING
    #
    # Both prediction date and target date must occur
    # strictly before the internal validation cutoff.
    # --------------------------------------------------------

    fitting_mask = (
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
    )

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    #
    # Starts at the cutoff date.
    # --------------------------------------------------------

    internal_validation_mask = (
        train[
            "prediction_date"
        ]
        >=
        cutoff
    )

    # --------------------------------------------------------
    # TARGET-DATE EMBARGO
    #
    # Prediction happened before cutoff, but its future
    # target reaches into the internal-validation period.
    # --------------------------------------------------------

    embargo_mask = (
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
    )

    fitting = train[
        fitting_mask
    ].copy()

    internal_validation = train[
        internal_validation_mask
    ].copy()

    embargoed = train[
        embargo_mask
    ].copy()

    # --------------------------------------------------------
    # CHECK PARTITIONS
    # --------------------------------------------------------

    total_partitioned = (
        len(fitting)
        +
        len(internal_validation)
        +
        len(embargoed)
    )

    print(
        f"Internal validation cutoff : "
        f"{cutoff.date()}"
    )

    print()

    print(
        f"Original TRAIN sequences   : "
        f"{len(train):,}"
    )

    print(
        f"Model-fitting sequences    : "
        f"{len(fitting):,}"
    )

    print(
        f"Internal validation        : "
        f"{len(internal_validation):,}"
    )

    print(
        f"Embargoed sequences        : "
        f"{len(embargoed):,}"
    )

    print(
        f"Partitioned total          : "
        f"{total_partitioned:,}"
    )

    # --------------------------------------------------------
    # VALIDATE PARTITION COUNTS
    # --------------------------------------------------------

    if total_partitioned != len(
        train
    ):

        raise ValueError(
            "\nInternal split does not "
            "cover the entire TRAIN set.\n"
            f"TRAIN total       : "
            f"{len(train):,}\n"
            f"Partitioned total : "
            f"{total_partitioned:,}"
        )

    if fitting.empty:

        raise ValueError(
            "Model-fitting partition is empty."
        )

    if internal_validation.empty:

        raise ValueError(
            "Internal-validation partition "
            "is empty."
        )

    # --------------------------------------------------------
    # STRONG LEAKAGE CHECKS
    # --------------------------------------------------------

    if not (
        fitting[
            "prediction_date"
        ]
        <
        cutoff
    ).all():

        raise ValueError(
            "Fitting prediction dates cross "
            "the internal validation cutoff."
        )

    if not (
        fitting[
            "target_date"
        ]
        <
        cutoff
    ).all():

        raise ValueError(
            "Fitting target dates cross "
            "the internal validation cutoff."
        )

    if not (
        internal_validation[
            "prediction_date"
        ]
        >=
        cutoff
    ).all():

        raise ValueError(
            "Internal validation contains "
            "prediction dates before cutoff."
        )

    if not embargoed.empty:

        valid_embargo = (
            (
                embargoed[
                    "prediction_date"
                ]
                <
                cutoff
            )
            &
            (
                embargoed[
                    "target_date"
                ]
                >=
                cutoff
            )
        )

        if not valid_embargo.all():

            raise ValueError(
                "Invalid rows detected inside "
                "target-date embargo."
            )

    # --------------------------------------------------------
    # DATE SUMMARY
    # --------------------------------------------------------

    print()

    print(
        "Model-fitting period:"
    )

    print(
        f"  Prediction dates : "
        f"{fitting['prediction_date'].min().date()} "
        f"-> "
        f"{fitting['prediction_date'].max().date()}"
    )

    print(
        f"  Last target date : "
        f"{fitting['target_date'].max().date()}"
    )

    print()

    print(
        "Internal validation period:"
    )

    print(
        f"  Prediction dates : "
        f"{internal_validation['prediction_date'].min().date()} "
        f"-> "
        f"{internal_validation['prediction_date'].max().date()}"
    )

    if not embargoed.empty:

        print()

        print(
            "Embargo period:"
        )

        print(
            f"  Prediction dates : "
            f"{embargoed['prediction_date'].min().date()} "
            f"-> "
            f"{embargoed['prediction_date'].max().date()}"
        )

        print(
            f"  Target dates     : "
            f"{embargoed['target_date'].min().date()} "
            f"-> "
            f"{embargoed['target_date'].max().date()}"
        )

    print()

    print(
        "Chronological internal split PASSED."
    )

    return (
        fitting.reset_index(
            drop=True
        ),

        internal_validation.reset_index(
            drop=True
        ),

        embargoed.reset_index(
            drop=True
        ),
    )
# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    fitting,
):

    print_section(
        "CALCULATING TRAINING CLASS WEIGHTS"
    )

    y = fitting[
        "target"
    ].to_numpy(
        dtype=np.int32
    )

    class_0_count = int(
        np.sum(
            y == 0
        )
    )

    class_1_count = int(
        np.sum(
            y == 1
        )
    )

    total = int(
        len(
            y
        )
    )

    if (
        class_0_count == 0
        or
        class_1_count == 0
    ):

        raise ValueError(
            "Both target classes are required."
        )

    class_0_weight = (

        total

        /

        (
            2.0
            *
            class_0_count
        )
    )

    class_1_weight = (

        total

        /

        (
            2.0
            *
            class_1_count
        )
    )

    class_weights = {

        0:
            float(
                class_0_weight
            ),

        1:
            float(
                class_1_weight
            ),
    }

    print(
        f"{TARGET_CLASS_NAMES[0]} samples : "
        f"{class_0_count:,}"
    )

    print(
        f"{TARGET_CLASS_NAMES[1]} samples     : "
        f"{class_1_count:,}"
    )

    print(
        f"TOP30 rate        : "
        f"{class_1_count / total:.2%}"
    )

    print()

    if USE_CLASS_WEIGHTS:

        print(
            f"Class 0 weight : "
            f"{class_weights[0]:.6f}"
        )

        print(
            f"Class 1 weight : "
            f"{class_weights[1]:.6f}"
        )

    else:

        print(
            "Class weighting is disabled."
        )

        class_weights = {

            0: 1.0,

            1: 1.0,
        }

    return class_weights


# ============================================================
# MEMORY-EFFICIENT KERAS SEQUENCE
# ============================================================

class DynamicSequence(
    tf.keras.utils.Sequence
):
    """
    Construct 60 x 30 windows dynamically from row indices.

    No full 3D training tensor is stored in memory.
    """

    def __init__(
        self,
        feature_matrix,
        sequence_rows,
        batch_size,
        shuffle=False,
        class_weights=None,
        return_targets=True,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.feature_matrix = (
            feature_matrix
        )

        self.sequence_rows = (
            sequence_rows
            .reset_index(
                drop=True
            )
        )

        self.batch_size = int(
            batch_size
        )

        self.shuffle = bool(
            shuffle
        )

        self.class_weights = (
            class_weights
        )

        self.return_targets = bool(
            return_targets
        )

        self.order = np.arange(
            len(
                self.sequence_rows
            ),
            dtype=np.int64,
        )

        self.sequence_offsets = np.arange(
            SEQUENCE_LENGTH,
            dtype=np.int64,
        )

        self.on_epoch_end()

    def __len__(
        self,
    ):

        return math.ceil(

            len(
                self.sequence_rows
            )

            /

            self.batch_size
        )

    def on_epoch_end(
        self,
    ):

        if self.shuffle:

            np.random.shuffle(
                self.order
            )

    def __getitem__(
        self,
        batch_index,
    ):

        start = (

            batch_index
            *
            self.batch_size
        )

        stop = min(

            start
            +
            self.batch_size,

            len(
                self.order
            ),
        )

        order_indices = (
            self.order[
                start:
                stop
            ]
        )

        batch_rows = (
            self.sequence_rows
            .iloc[
                order_indices
            ]
        )

        start_rows = (
            batch_rows[
                "start_row"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        # ----------------------------------------------------
        # Vectorized row-index construction:
        #
        # batch x 60
        # ----------------------------------------------------

        row_indices = (

            start_rows[
                :,
                None
            ]

            +

            self.sequence_offsets[
                None,
                :
            ]
        )

        X = (
            self.feature_matrix[
                row_indices
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )

        if not self.return_targets:

            return X

        y = (
            batch_rows[
                "target"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        if self.class_weights is None:

            return (
                X,
                y,
            )

        sample_weights = np.where(

            y == 0,

            self.class_weights[
                0
            ],

            self.class_weights[
                1
            ],

        ).astype(
            np.float32
        )

        return (
            X,
            y,
            sample_weights,
        )


# ============================================================
# GENERATOR VALIDATION
# ============================================================

def validate_generators(
    training_generator,
    validation_generator,
):

    print_section(
        "VALIDATING DYNAMIC GENERATORS"
    )

    train_batch = (
        training_generator[
            0
        ]
    )

    if len(
        train_batch
    ) == 3:

        X_train, y_train, w_train = (
            train_batch
        )

    else:

        X_train, y_train = (
            train_batch
        )

        w_train = None

    validation_batch = (
        validation_generator[
            0
        ]
    )

    if len(
        validation_batch
    ) == 3:

        X_val, y_val, _ = (
            validation_batch
        )

    else:

        X_val, y_val = (
            validation_batch
        )

    print(
        f"Training batches            : "
        f"{len(training_generator):,}"
    )

    print(
        f"Internal validation batches : "
        f"{len(validation_generator):,}"
    )

    print()

    print(
        f"Sample X shape : "
        f"{X_train.shape}"
    )

    print(
        f"Sample y shape : "
        f"{y_train.shape}"
    )

    if w_train is not None:

        print(
            f"Sample weights : "
            f"{w_train.shape}"
        )

    if X_train.shape[
        1:
    ] != (

        SEQUENCE_LENGTH,

        len(
            FEATURE_COLUMNS
        ),

    ):

        raise ValueError(
            "Training generator produced "
            "incorrect input shape."
        )

    if X_val.shape[
        1:
    ] != (

        SEQUENCE_LENGTH,

        len(
            FEATURE_COLUMNS
        ),

    ):

        raise ValueError(
            "Validation generator produced "
            "incorrect input shape."
        )

    if not np.isfinite(
        X_train
    ).all():

        raise ValueError(
            "Training batch contains "
            "NaN/Inf."
        )

    if not np.isfinite(
        X_val
    ).all():

        raise ValueError(
            "Validation batch contains "
            "NaN/Inf."
        )

    print()

    print(
        "Dynamic generators ready."
    )


# ============================================================
# BUILD MODEL
# ============================================================

def create_model():

    print_section(
        "BUILDING TREND MODEL"
    )

    print(
        "Model builder found: "
        "build_trend_model"
    )

    # --------------------------------------------------------
    # Keep EXACT same model architecture as previous run.
    # --------------------------------------------------------

    model = build_trend_model(
        compile_model=False
    )

    dummy_input = np.zeros(
        (
            1,

            SEQUENCE_LENGTH,

            len(
                FEATURE_COLUMNS
            ),
        ),
        dtype=np.float32,
    )

    _ = model(
        dummy_input,
        training=False,
    )

    expected_input = (

        None,

        SEQUENCE_LENGTH,

        len(
            FEATURE_COLUMNS
        ),
    )

    if tuple(
        model.input_shape
    ) != expected_input:

        raise ValueError(
            "\nModel input shape mismatch.\n"
            f"Expected: {expected_input}\n"
            f"Actual  : {model.input_shape}"
        )

    if int(
        model.output_shape[
            -1
        ]
    ) != 2:

        raise ValueError(
            "\nAgent 2 model must have exactly "
            "2 output classes."
        )

    print()

    print(
        f"Model input  : "
        f"{model.input_shape}"
    )

    print(
        f"Model output : "
        f"{model.output_shape}"
    )

    print(
        f"Parameters   : "
        f"{model.count_params():,}"
    )

    return model


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_model(
    model,
):

    print_section(
        "COMPILING TREND MODEL"
    )

    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=(
                LEARNING_RATE
            )
        )
    )

    loss = (
        tf.keras.losses.SparseCategoricalCrossentropy()
    )

    model.compile(

        optimizer=optimizer,

        loss=loss,

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    print(
        "Optimizer     : Adam"
    )

    print(
        f"Learning rate : "
        f"{LEARNING_RATE}"
    )

    print(
        "Loss          : "
        "SparseCategoricalCrossentropy"
    )

    print()

    model.summary()

    return model


# ============================================================
# CALLBACKS
# ============================================================

def create_callbacks():

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                BEST_WEIGHTS_FILE
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

            factor=(
                REDUCE_LR_FACTOR
            ),

            patience=(
                REDUCE_LR_PATIENCE
            ),

            min_lr=(
                MIN_LEARNING_RATE
            ),

            verbose=1,
        ),
    ]

    return callbacks


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    training_generator,
    internal_validation_generator,
):

    print_section(
        "TRAINING RELATIVE-TARGET TREND MODEL"
    )

    print(
        f"Epoch limit : "
        f"{EPOCHS}"
    )

    print(
        f"Batch size  : "
        f"{BATCH_SIZE}"
    )

    print()

    print(
        "Target:"
    )

    print(
        "TOP 30% cross-sectional "
        "21-day future return."
    )

    print()

    print(
        "Early stopping uses ONLY the "
        "internal chronological portion "
        "of TRAIN."
    )

    print()

    start_time = time.time()

    history = model.fit(

        training_generator,

        validation_data=(
            internal_validation_generator
        ),

        epochs=EPOCHS,

        callbacks=(
            create_callbacks()
        ),

        verbose=1,
    )

    training_seconds = (
        time.time()
        -
        start_time
    )

    print()

    print(
        f"Training time : "
        f"{training_seconds:.2f} sec"
    )

    print(
        f"Training time : "
        f"{training_seconds / 60:.2f} min"
    )

    # --------------------------------------------------------
    # Explicitly reload checkpoint.
    # --------------------------------------------------------

    if not BEST_WEIGHTS_FILE.exists():

        raise FileNotFoundError(
            "\nBest-model checkpoint was "
            "not created."
        )

    model.load_weights(
        BEST_WEIGHTS_FILE
    )

    print()

    print(
        "Best checkpoint restored from:"
    )

    print(
        BEST_WEIGHTS_FILE
    )

    # --------------------------------------------------------
    # Save complete final model separately.
    # --------------------------------------------------------

    model.save(
        FINAL_MODEL_FILE
    )

    print()

    print(
        "Full relative-target model saved:"
    )

    print(
        FINAL_MODEL_FILE
    )

    history_df = pd.DataFrame(
        history.history
    )

    history_df[
        "epoch"
    ] = np.arange(
        1,
        len(
            history_df
        )
        +
        1
    )

    history_df.to_csv(
        TRAINING_HISTORY_FILE,
        index=False,
    )

    print()

    print(
        "Training history saved:"
    )

    print(
        TRAINING_HISTORY_FILE
    )

    return (
        model,
        history,
        training_seconds,
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

        CLASSIFICATION_THRESHOLD
    ).astype(
        np.int32
    )

    matrix = confusion_matrix(

        y_true,

        predicted,

        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        matrix.ravel()
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

        brier = brier_score_loss(
            y_true,
            probability,
        )

    except ValueError:

        brier = np.nan

    return {

        "samples":
            int(
                len(
                    y_true
                )
            ),

        "actual_positive_rate":
            float(
                np.mean(
                    y_true
                )
            ),

        "predicted_positive_rate":
            float(
                np.mean(
                    predicted
                )
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

        "specificity":
            float(
                specificity
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    predicted,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc
            )
            if np.isfinite(
                roc_auc
            )
            else np.nan,

        "average_precision":
            float(
                average_precision
            )
            if np.isfinite(
                average_precision
            )
            else np.nan,

        "brier_score":
            float(
                brier
            )
            if np.isfinite(
                brier
            )
            else np.nan,

        "mean_positive_probability":
            float(
                np.mean(
                    probability
                )
            ),

        "min_positive_probability":
            float(
                np.min(
                    probability
                )
            ),

        "max_positive_probability":
            float(
                np.max(
                    probability
                )
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
# PREDICT ONE SPLIT
# ============================================================

def predict_split(
    model,
    feature_matrix,
    split_data,
    split_name,
):

    print()

    print(
        f"Predicting "
        f"{split_name.upper()}..."
    )

    prediction_generator = (
        DynamicSequence(

            feature_matrix=(
                feature_matrix
            ),

            sequence_rows=(
                split_data
            ),

            batch_size=(
                BATCH_SIZE
            ),

            shuffle=False,

            class_weights=None,

            return_targets=False,
        )
    )

    start_time = time.time()

    probabilities = model.predict(

        prediction_generator,

        verbose=1,
    )

    inference_seconds = (
        time.time()
        -
        start_time
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if probabilities.ndim != 2:

        raise ValueError(
            "\nModel prediction output "
            "must be 2-dimensional.\n"
            f"Received: "
            f"{probabilities.shape}"
        )

    if probabilities.shape[
        1
    ] != 2:

        raise ValueError(
            "\nExpected 2 probability "
            "columns from Softmax."
        )

    if probabilities.shape[
        0
    ] != len(
        split_data
    ):

        raise ValueError(
            "\nPrediction row-count mismatch."
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Predictions contain NaN/Inf."
        )

    probability_top30 = (
        probabilities[
            :,
            1
        ]
    )

    actual = (
        split_data[
            "target"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    metrics = calculate_metrics(

        actual,

        probability_top30,
    )

    print(
        f"{split_name.upper()} "
        f"inference time: "
        f"{inference_seconds:.2f} sec"
    )

    print_subsection(
        split_name.upper()
    )

    print(
        f"Samples              : "
        f"{metrics['samples']:,}"
    )

    print(
        f"Actual TOP30 rate    : "
        f"{metrics['actual_positive_rate']:.2%}"
    )

    print(
        f"Predicted TOP30 rate : "
        f"{metrics['predicted_positive_rate']:.2%}"
    )

    print(
        f"Accuracy             : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision            : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall               : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1                   : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Specificity          : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"ROC-AUC              : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision    : "
        f"{metrics['average_precision']:.4f}"
    )

    print(
        f"Brier score          : "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"Mean P(TOP30)        : "
        f"{metrics['mean_positive_probability']:.4f}"
    )

    print(
        f"Min P(TOP30)         : "
        f"{metrics['min_positive_probability']:.4f}"
    )

    print(
        f"Max P(TOP30)         : "
        f"{metrics['max_positive_probability']:.4f}"
    )

    print(
        f"TN / FP / FN / TP    : "
        f"{metrics['tn']} / "
        f"{metrics['fp']} / "
        f"{metrics['fn']} / "
        f"{metrics['tp']}"
    )

    result = split_data.copy()

    result[
        "actual_class"
    ] = actual

    result[
        "actual_label"
    ] = np.where(

        actual == 1,

        TARGET_CLASS_NAMES[
            1
        ],

        TARGET_CLASS_NAMES[
            0
        ],
    )

    result[
        "not_top30_probability"
    ] = probabilities[
        :,
        0
    ]

    result[
        "top30_probability"
    ] = probability_top30

    result[
        "predicted_class"
    ] = (

        probability_top30

        >=

        CLASSIFICATION_THRESHOLD
    ).astype(
        np.int32
    )

    result[
        "predicted_label"
    ] = np.where(

        result[
            "predicted_class"
        ]
        ==
        1,

        TARGET_CLASS_NAMES[
            1
        ],

        TARGET_CLASS_NAMES[
            0
        ],
    )

    result[
        "year"
    ] = (

        result[
            "prediction_date"
        ]
        .dt
        .year
    )

    return (
        result,
        metrics,
        inference_seconds,
    )


# ============================================================
# EVALUATE ALL SPLITS
# ============================================================

def evaluate_model(
    model,
    feature_matrix,
    sequence_index,
):

    print_section(
        "FINAL MODEL EVALUATION"
    )

    print(
        "Decision threshold : "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "2025 is evaluated only after "
        "model training has completely finished."
    )

    all_results = []

    metric_records = []

    inference_times = {}

    for split_name in [

        "train",

        "backtest",

        "validation",

    ]:

        split_data = (

            sequence_index[
                sequence_index[
                    "dataset_split"
                ]
                ==
                split_name
            ]

            .reset_index(
                drop=True
            )
        )

        if split_data.empty:

            continue

        (
            result,
            metrics,
            inference_seconds,

        ) = predict_split(

            model=model,

            feature_matrix=(
                feature_matrix
            ),

            split_data=(
                split_data
            ),

            split_name=(
                split_name
            ),
        )

        all_results.append(
            result
        )

        inference_times[
            split_name
        ] = float(
            inference_seconds
        )

        metric_records.append(
            {

                "model":
                    "dilated_lstm_transformer_progressive_attention",

                "experiment":
                    "relative_21d_top30",

                "target_mode":
                    TARGET_MODE,

                "prediction_horizon":
                    PREDICTION_HORIZON,

                "relative_top_fraction":
                    RELATIVE_TOP_FRACTION,

                "dataset_split":
                    split_name,

                "decision_threshold":
                    CLASSIFICATION_THRESHOLD,

                "inference_seconds":
                    float(
                        inference_seconds
                    ),

                **metrics,
            }
        )

    predictions = pd.concat(
        all_results,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        metric_records
    )

    return (
        predictions,
        metrics_df,
        inference_times,
    )


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation(
    predictions,
    metrics_df,
):

    print_section(
        "SAVING RELATIVE-TARGET EVALUATION RESULTS"
    )

    metrics_df.to_csv(
        TREND_METRICS_FILE,
        index=False,
    )

    predictions.to_parquet(

        EVALUATION_PREDICTIONS_FILE,

        index=False,
    )

    print(
        "Metrics:"
    )

    print(
        TREND_METRICS_FILE
    )

    print()

    print(
        "Predictions:"
    )

    print(
        EVALUATION_PREDICTIONS_FILE
    )


# ============================================================
# YEAR-BY-YEAR METRIC HELPER
# ============================================================

def calculate_year_metrics(
    predictions,
):

    records = []

    for (
        dataset_split,
        year,
    ), group in predictions.groupby(

        [
            "dataset_split",
            "year",
        ],

        sort=True,
    ):

        metrics = calculate_metrics(

            group[
                "actual_class"
            ],

            group[
                "top30_probability"
            ],
        )

        records.append(
            {

                "dataset_split":
                    dataset_split,

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
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# TRAINING SUMMARY
# ============================================================

def save_training_summary(
    metadata,
    fitting,
    internal_validation,
    embargoed,
    history,
    training_seconds,
    metrics_df,
    predictions,
    inference_times,
):

    history_df = pd.DataFrame(
        history.history
    )

    best_epoch = None

    if (
        "val_loss"
        in history_df.columns
        and
        not history_df.empty
    ):

        best_epoch = int(

            history_df[
                "val_loss"
            ]
            .idxmin()

            +

            1
        )

    year_metrics = calculate_year_metrics(
        predictions
    )

    year_metrics_file = (
        REPORT_DIR
        / "trend_year_metrics_relative_21d_top30.csv"
    )

    year_metrics.to_csv(
        year_metrics_file,
        index=False,
    )

    summary = {

        "experiment":
            "relative_21d_top30",

        "target": {

            "target_mode":
                TARGET_MODE,

            "prediction_horizon":
                PREDICTION_HORIZON,

            "relative_top_fraction":
                RELATIVE_TOP_FRACTION,

            "class_0":
                TARGET_CLASS_NAMES[
                    0
                ],

            "class_1":
                TARGET_CLASS_NAMES[
                    1
                ],
        },

        "architecture": {

            "sequence_length":
                SEQUENCE_LENGTH,

            "feature_count":
                len(
                    FEATURE_COLUMNS
                ),

            "feature_columns":
                FEATURE_COLUMNS,

            "architecture_name":
                (
                    "Dilated LSTM + Transformer + "
                    "Progressive Attention"
                ),
        },

        "chronology": {

            "internal_validation_cutoff":
                str(
                    INTERNAL_VALIDATION_CUTOFF.date()
                ),

            "model_fitting_sequences":
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

            "fitting_prediction_start":
                str(
                    fitting[
                        "prediction_date"
                    ]
                    .min()
                    .date()
                ),

            "fitting_prediction_end":
                str(
                    fitting[
                        "prediction_date"
                    ]
                    .max()
                    .date()
                ),

            "fitting_last_target_date":
                str(
                    fitting[
                        "target_date"
                    ]
                    .max()
                    .date()
                ),

            "internal_validation_start":
                str(
                    internal_validation[
                        "prediction_date"
                    ]
                    .min()
                    .date()
                ),

            "internal_validation_end":
                str(
                    internal_validation[
                        "prediction_date"
                    ]
                    .max()
                    .date()
                ),
        },

        "training": {

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                EPOCHS,

            "epochs_completed":
                int(
                    len(
                        history_df
                    )
                ),

            "best_epoch":
                best_epoch,

            "learning_rate":
                LEARNING_RATE,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

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
        },

        "evaluation": (

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

        "inference_seconds":
            inference_times,

        "files": {

            "best_weights":
                str(
                    BEST_WEIGHTS_FILE
                ),

            "final_model":
                str(
                    FINAL_MODEL_FILE
                ),

            "training_history":
                str(
                    TRAINING_HISTORY_FILE
                ),

            "metrics":
                str(
                    TREND_METRICS_FILE
                ),

            "predictions":
                str(
                    EVALUATION_PREDICTIONS_FILE
                ),

            "year_metrics":
                str(
                    year_metrics_file
                ),
        },

        "comparison_baseline": {

            "old_target":
                "absolute_21d_2pct",

            "old_backtest_roc_auc":
                0.5093485296695143,

            "old_validation_roc_auc":
                0.46511226244690207,

            "selection_rule":
                (
                    "New model should primarily be "
                    "judged by backtest ROC-AUC and "
                    "temporal stability, not by tuning "
                    "against 2025."
                ),
        },

        "sequence_metadata":
            metadata,
    }

    with open(

        TRAINING_SUMMARY_FILE,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

            default=str,
        )

    print()

    print(
        "Training summary:"
    )

    print(
        TRAINING_SUMMARY_FILE
    )

    print()

    print(
        "Year metrics:"
    )

    print(
        year_metrics_file
    )

    return year_metrics


# ============================================================
# OLD-VS-NEW COMPARISON
# ============================================================

def print_old_new_comparison(
    metrics_df,
):

    print_section(
        "OLD TARGET VS NEW TARGET"
    )

    old_backtest_auc = (
        0.5093485296695143
    )

    old_validation_auc = (
        0.46511226244690207
    )

    print(
        "OLD DEEP MODEL"
    )

    print(
        "Target:"
    )

    print(
        "  future 21-day return >= +2%"
    )

    print(
        f"Backtest ROC-AUC   : "
        f"{old_backtest_auc:.4f}"
    )

    print(
        f"Validation ROC-AUC : "
        f"{old_validation_auc:.4f}"
    )

    print()

    print(
        "NEW DEEP MODEL"
    )

    print(
        "Target:"
    )

    print(
        "  relative 21-day TOP30"
    )

    for split in [

        "backtest",

        "validation",

    ]:

        subset = metrics_df[
            metrics_df[
                "dataset_split"
            ]
            ==
            split
        ]

        if subset.empty:

            continue

        auc = float(
            subset.iloc[
                0
            ][
                "roc_auc"
            ]
        )

        old_auc = (

            old_backtest_auc

            if split
            ==
            "backtest"

            else
            old_validation_auc
        )

        change = (

            auc
            -
            old_auc
        )

        print(
            f"{split.upper():10s} "
            f"ROC-AUC : "
            f"{auc:.4f} "
            f"(change "
            f"{change:+.4f})"
        )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training():

    overall_start = time.time()

    print_section(
        "AGENT 2 - RELATIVE TOP-30% "
        "MEMORY-EFFICIENT TRAINER"
    )

    print(
        "Training design:"
    )

    print(
        "  Broad historical stock universe"
    )

    print(
        "  Dynamic 60 x 30 sequence batches"
    )

    print(
        "  Relative TOP30 target"
    )

    print(
        "  Chronological internal validation"
    )

    print(
        "  Target-date embargo"
    )

    print(
        "  2019-2024 used only for backtest"
    )

    print(
        "  2025 final validation untouched "
        "until evaluation"
    )

    print()

    print(
        "This is a TARGET-ONLY A/B experiment."
    )

    print(
        "The neural architecture is unchanged."
    )

    # ========================================================
    # 1. FILE VALIDATION
    # ========================================================

    validate_input_files()

    # ========================================================
    # 2. METADATA
    # ========================================================

    metadata = (
        load_sequence_metadata()
    )

    # ========================================================
    # 3. FEATURES
    # ========================================================

    (
        feature_data,
        feature_matrix,

    ) = load_feature_matrix(
        metadata
    )

    # ========================================================
    # 4. SEQUENCE INDEX
    # ========================================================

    sequence_index = (
        load_sequence_index(
            metadata
        )
    )

    # ========================================================
    # 5. MAPPING VALIDATION
    # ========================================================

    validate_row_mapping(

        sequence_index,

        feature_data,
    )

    # ========================================================
    # 6. INTERNAL VALIDATION
    # ========================================================

    (
        fitting,

        internal_validation,

        embargoed,

    ) = create_internal_validation(
        sequence_index
    )

    # ========================================================
    # 7. CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            fitting
        )
    )

    # ========================================================
    # 8. GENERATORS
    # ========================================================

    print_section(
        "CREATING DYNAMIC TRAINING GENERATORS"
    )

    training_generator = (
        DynamicSequence(

            feature_matrix=(
                feature_matrix
            ),

            sequence_rows=(
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
        DynamicSequence(

            feature_matrix=(
                feature_matrix
            ),

            sequence_rows=(
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

    validate_generators(

        training_generator,

        internal_validation_generator,
    )

    # ========================================================
    # 9. MODEL
    # ========================================================

    model = create_model()

    model = compile_model(
        model
    )

    # ========================================================
    # 10. TRAINING
    # ========================================================

    (
        model,
        history,
        training_seconds,

    ) = train_model(

        model,

        training_generator,

        internal_validation_generator,
    )

    # ========================================================
    # 11. FINAL EVALUATION
    # ========================================================

    (
        predictions,
        metrics_df,
        inference_times,

    ) = evaluate_model(

        model,

        feature_matrix,

        sequence_index,
    )

    # ========================================================
    # 12. SAVE
    # ========================================================

    save_evaluation(

        predictions,

        metrics_df,
    )

    # ========================================================
    # 13. SUMMARY
    # ========================================================

    year_metrics = (
        save_training_summary(

            metadata=metadata,

            fitting=fitting,

            internal_validation=(
                internal_validation
            ),

            embargoed=embargoed,

            history=history,

            training_seconds=(
                training_seconds
            ),

            metrics_df=metrics_df,

            predictions=predictions,

            inference_times=(
                inference_times
            ),
        )
    )

    # ========================================================
    # 14. A/B COMPARISON
    # ========================================================

    print_old_new_comparison(
        metrics_df
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    elapsed = (
        time.time()
        -
        overall_start
    )

    print_section(
        "AGENT 2 RELATIVE-TARGET TRAINING COMPLETE"
    )

    print(
        f"Overall runtime : "
        f"{elapsed:.2f} sec "
        f"({elapsed / 60:.2f} min)"
    )

    print()

    print(
        "Best relative-target weights:"
    )

    print(
        BEST_WEIGHTS_FILE
    )

    print()

    print(
        "Relative-target metrics:"
    )

    print(
        TREND_METRICS_FILE
    )

    print()

    print(
        "Relative-target training history:"
    )

    print(
        TRAINING_HISTORY_FILE
    )

    print()

    print(
        "Relative-target predictions:"
    )

    print(
        EVALUATION_PREDICTIONS_FILE
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT tune the 0.50 threshold "
        "against the 2025 validation set."
    )

    print()

    print(
        "First compare:"
    )

    print(
        "  1. Backtest ROC-AUC"
    )

    print(
        "  2. Year-by-year ROC-AUC"
    )

    print(
        "  3. Validation ROC-AUC"
    )

    print(
        "  4. Train -> backtest gap"
    )

    print(
        "  5. Probability separation"
    )

    return {

        "model":
            model,

        "history":
            history,

        "metrics":
            metrics_df,

        "year_metrics":
            year_metrics,

        "predictions":
            predictions,
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_training()