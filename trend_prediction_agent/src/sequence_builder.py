"""
sequence_builder.py

Agent 2 - Memory-Efficient Relative Target Sequence Builder
===========================================================

Current target
--------------
For each prediction date:

    TARGET = 1

if the stock is within the TOP 30% of eligible stocks ranked
by future 21-trading-day return.

Otherwise:

    TARGET = 0

Architecture input
------------------
60 historical rows x 30 features.

Prediction horizon
------------------
21 trading rows.

Important safeguards
--------------------
1. Input sequences never cross symbols.
2. Input sequences never cross dataset splits.
3. Future returns never cross dataset splits.
4. Cross-sectional labels are generated separately for each
   prediction date and dataset split.
5. No dense sequence tensor is created.
6. Only sequence row indices and metadata are persisted.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PROJECT IMPORT SETUP
# ============================================================

CURRENT_FILE = Path(
    __file__
).resolve()

TREND_AGENT_ROOT = (
    CURRENT_FILE.parents[1]
)


if str(
    TREND_AGENT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            TREND_AGENT_ROOT
        ),
    )


# ============================================================
# CONFIGURATION
# ============================================================

from config.config import (

    PREPROCESSED_TREND_DATA_FILE,

    SEQUENCE_INDEX_FILE,

    SEQUENCE_METADATA_FILE,

    SEQUENCE_SUMMARY_FILE,

    SEQUENCE_STOCK_SUMMARY_FILE,

    TREND_FEATURE_COLUMNS,

    SEQUENCE_LENGTH,

    PREDICTION_HORIZON,

    TARGET_MODE,

    RELATIVE_TOP_FRACTION,

    RELATIVE_PERCENTILE_CUTOFF,

    TARGET_CLASS_NAMES,

    HIGH_RETURN_THRESHOLD,

    RANDOM_SEED,
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


# ============================================================
# PARQUET HELPERS
# ============================================================

def get_parquet_columns(
    path,
):

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
# LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data():

    print_section(
        "LOADING PREPROCESSED TREND DATA"
    )

    print(
        f"Input:\n"
        f"{PREPROCESSED_TREND_DATA_FILE}"
    )

    if not (
        PREPROCESSED_TREND_DATA_FILE.exists()
    ):

        raise FileNotFoundError(
            "\nPreprocessed trend data "
            "was not found.\n"
            f"{PREPROCESSED_TREND_DATA_FILE}"
        )

    available_columns = (
        get_parquet_columns(
            PREPROCESSED_TREND_DATA_FILE
        )
    )

    symbol_column = find_column(
        available_columns,
        [
            "symbol",
            "Symbol",
            "SYMBOL",
        ],
    )

    date_column = find_column(
        available_columns,
        [
            "date",
            "Date",
            "DATE",
        ],
    )

    split_column = find_column(
        available_columns,
        [
            "dataset_split",
            "split",
        ],
    )

    close_column = find_column(
        available_columns,
        [
            "close",
            "Close",
            "CLOSE",
        ],
    )

    if symbol_column is None:

        raise KeyError(
            "Symbol column not found."
        )

    if date_column is None:

        raise KeyError(
            "Date column not found."
        )

    if split_column is None:

        raise KeyError(
            "dataset_split column not found."
        )

    if close_column is None:

        raise KeyError(
            "Close-price column not found."
        )

    missing_features = [

        feature

        for feature
        in TREND_FEATURE_COLUMNS

        if feature
        not in available_columns
    ]

    if missing_features:

        raise KeyError(
            "\nMissing model features:\n"
            f"{missing_features}"
        )

    required_columns = [

        symbol_column,

        date_column,

        split_column,

        close_column,

        *TREND_FEATURE_COLUMNS,
    ]

    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    data = pd.read_parquet(

        PREPROCESSED_TREND_DATA_FILE,

        columns=required_columns,
    )

    rename_map = {

        symbol_column:
            "symbol",

        date_column:
            "date",

        split_column:
            "dataset_split",

        close_column:
            "close",
    }

    data = data.rename(
        columns=rename_map
    )

    data[
        "symbol"
    ] = (

        data[
            "symbol"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    data[
        "dataset_split"
    ] = (

        data[
            "dataset_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data[
        "date"
    ] = pd.to_datetime(
        data[
            "date"
        ],
        errors="coerce",
    )

    data[
        "close"
    ] = pd.to_numeric(
        data[
            "close"
        ],
        errors="coerce",
    )

    if data[
        "date"
    ].isna().any():

        raise ValueError(
            "Invalid dates detected."
        )

    if data[
        "close"
    ].isna().any():

        raise ValueError(
            "Invalid close prices detected."
        )

    # --------------------------------------------------------
    # Exact deterministic source-row ordering.
    #
    # Trainer must use the same symbol/date sorting.
    # --------------------------------------------------------

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

    data[
        "source_row"
    ] = np.arange(
        len(data),
        dtype=np.int64,
    )

    duplicate_count = (
        data.duplicated(
            subset=[
                "symbol",
                "date",
            ]
        )
        .sum()
    )

    if duplicate_count > 0:

        raise ValueError(
            f"Duplicate symbol/date rows: "
            f"{duplicate_count:,}"
        )

    feature_matrix = (
        data[
            TREND_FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(
        feature_matrix
    ).all():

        raise ValueError(
            "NaN/Inf found in feature matrix."
        )

    if not np.isfinite(
        data[
            "close"
        ].to_numpy(
            dtype=np.float64
        )
    ).all():

        raise ValueError(
            "NaN/Inf found in close prices."
        )

    print()

    print(
        f"Rows              : "
        f"{len(data):,}"
    )

    print(
        f"Stocks            : "
        f"{data['symbol'].nunique():,}"
    )

    print(
        f"Features          : "
        f"{len(TREND_FEATURE_COLUMNS)}"
    )

    print(
        f"Date range        : "
        f"{data['date'].min().date()} -> "
        f"{data['date'].max().date()}"
    )

    print()

    for split in [

        "train",
        "backtest",
        "validation",

    ]:

        part = data[
            data[
                "dataset_split"
            ]
            ==
            split
        ]

        if part.empty:

            continue

        print(
            f"{split.upper():10s}: "
            f"{len(part):10,d} rows | "
            f"{part['symbol'].nunique():4,d} stocks"
        )

    print()

    print(
        "Preprocessed data validation PASSED."
    )

    return data


# ============================================================
# GENERATE FUTURE RETURNS
# ============================================================

def generate_future_returns(
    data,
):

    print_section(
        "GENERATING FUTURE RETURNS"
    )

    result = data.copy()

    grouped = result.groupby(

        [
            "symbol",
            "dataset_split",
        ],

        sort=False,

        observed=True,
    )

    result[
        "future_close"
    ] = (

        grouped[
            "close"
        ]
        .shift(
            -PREDICTION_HORIZON
        )
    )

    result[
        "target_date"
    ] = (

        grouped[
            "date"
        ]
        .shift(
            -PREDICTION_HORIZON
        )
    )

    result[
        "target_row"
    ] = (

        grouped[
            "source_row"
        ]
        .shift(
            -PREDICTION_HORIZON
        )
    )

    result[
        "future_return"
    ] = (

        result[
            "future_close"
        ]

        /

        result[
            "close"
        ]

        -

        1.0
    )

    valid_target = (

        result[
            "future_return"
        ].notna()

        &

        result[
            "target_date"
        ].notna()

        &

        result[
            "target_row"
        ].notna()
    )

    print(
        f"Prediction horizon : "
        f"{PREDICTION_HORIZON}"
    )

    print(
        f"Valid target rows  : "
        f"{valid_target.sum():,}"
    )

    print(
        f"No future target   : "
        f"{(~valid_target).sum():,}"
    )

    return result


# ============================================================
# GENERATE CROSS-SECTIONAL TARGET
# ============================================================

def generate_relative_labels(
    data,
):

    print_section(
        "GENERATING RELATIVE TOP-30% LABEL"
    )

    if TARGET_MODE != (
        "relative_top_fraction"
    ):

        raise ValueError(
            "\nThis sequence builder requires:\n"
            "TARGET_MODE = "
            "'relative_top_fraction'"
        )

    result = data.copy()

    valid = (

        result[
            "future_return"
        ].notna()

        &

        result[
            "target_date"
        ].notna()

        &

        result[
            "target_row"
        ].notna()
    )

    result[
        "cross_section_percentile"
    ] = np.nan

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Stocks are ranked against all other eligible stocks
    # available on the SAME prediction date.
    #
    # Split is explicitly included so no data from another
    # chronological partition can influence the label.
    # --------------------------------------------------------

    percentiles = (

        result.loc[
            valid
        ]
        .groupby(
            [
                "dataset_split",
                "date",
            ],
            observed=True,
        )[
            "future_return"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    result.loc[
        valid,
        "cross_section_percentile",
    ] = percentiles

    result[
        "target"
    ] = np.nan

    result.loc[
        valid,
        "target",
    ] = (

        result.loc[
            valid,
            "cross_section_percentile",
        ]

        >

        RELATIVE_PERCENTILE_CUTOFF
    ).astype(
        np.int8
    )

    print(
        f"Target mode       : "
        f"{TARGET_MODE}"
    )

    print(
        f"Future horizon    : "
        f"{PREDICTION_HORIZON}"
    )

    print(
        f"Top fraction      : "
        f"{RELATIVE_TOP_FRACTION:.2%}"
    )

    print(
        f"Percentile cutoff : "
        f"{RELATIVE_PERCENTILE_CUTOFF:.2f}"
    )

    print()

    for split in [

        "train",
        "backtest",
        "validation",

    ]:

        subset = result[
            (
                result[
                    "dataset_split"
                ]
                ==
                split
            )
            &
            (
                result[
                    "target"
                ].notna()
            )
        ]

        if subset.empty:

            continue

        print(
            f"{split.upper():10s}: "
            f"{len(subset):10,d} labels | "
            f"TOP30="
            f"{subset['target'].mean():7.2%}"
        )

    return result


# ============================================================
# BUILD INDEXED SEQUENCES
# ============================================================

def build_sequence_index(
    data,
):

    print_section(
        "BUILDING SEQUENCE INDEX"
    )

    records = []

    sequence_id = 0

    groups_processed = 0

    groups_skipped = 0

    minimum_rows = (

        SEQUENCE_LENGTH

        +

        PREDICTION_HORIZON
    )

    groups = data.groupby(

        [
            "symbol",
            "dataset_split",
        ],

        sort=False,

        observed=True,
    )

    for (
        symbol,
        split,
    ), group in groups:

        groups_processed += 1

        group = (

            group
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            group
        ) < minimum_rows:

            groups_skipped += 1

            continue

        # ----------------------------------------------------
        # Prediction position is final row of 60-row window.
        # ----------------------------------------------------

        for prediction_position in range(

            SEQUENCE_LENGTH - 1,

            len(group),
        ):

            row = group.iloc[
                prediction_position
            ]

            if pd.isna(
                row[
                    "target"
                ]
            ):

                continue

            if pd.isna(
                row[
                    "future_return"
                ]
            ):

                continue

            if pd.isna(
                row[
                    "target_date"
                ]
            ):

                continue

            if pd.isna(
                row[
                    "target_row"
                ]
            ):

                continue

            start_position = (

                prediction_position

                -

                SEQUENCE_LENGTH

                +

                1
            )

            start_row = int(

                group.iloc[
                    start_position
                ][
                    "source_row"
                ]
            )

            end_row = int(
                row[
                    "source_row"
                ]
            )

            target_row = int(
                row[
                    "target_row"
                ]
            )

            actual_sequence_length = (

                end_row

                -

                start_row

                +

                1
            )

            if (
                actual_sequence_length
                !=
                SEQUENCE_LENGTH
            ):

                raise ValueError(
                    "\nSequence mapping error.\n"
                    f"Symbol      : {symbol}\n"
                    f"Split       : {split}\n"
                    f"Start row   : {start_row}\n"
                    f"End row     : {end_row}\n"
                    f"Expected    : "
                    f"{SEQUENCE_LENGTH}\n"
                    f"Mapped      : "
                    f"{actual_sequence_length}"
                )

            records.append(
                {

                    "sequence_id":
                        sequence_id,

                    "symbol":
                        symbol,

                    "dataset_split":
                        split,

                    "start_row":
                        start_row,

                    "end_row":
                        end_row,

                    "prediction_row":
                        end_row,

                    "target_row":
                        target_row,

                    "prediction_date":
                        row[
                            "date"
                        ],

                    "target_date":
                        row[
                            "target_date"
                        ],

                    "future_return":
                        float(
                            row[
                                "future_return"
                            ]
                        ),

                    "cross_section_percentile":
                        float(
                            row[
                                "cross_section_percentile"
                            ]
                        ),

                    "target":
                        int(
                            row[
                                "target"
                            ]
                        ),
                }
            )

            sequence_id += 1

    sequence_index = pd.DataFrame(
        records
    )

    if sequence_index.empty:

        raise ValueError(
            "No sequences generated."
        )

    print(
        f"Groups processed : "
        f"{groups_processed:,}"
    )

    print(
        f"Groups skipped   : "
        f"{groups_skipped:,}"
    )

    print(
        f"Total sequences  : "
        f"{len(sequence_index):,}"
    )

    return sequence_index


# ============================================================
# VALIDATE INDEX
# ============================================================

def validate_sequence_index(
    sequence_index,
    source_data,
):

    print_section(
        "VALIDATING SEQUENCE INDEX"
    )

    duplicate_ids = (
        sequence_index[
            "sequence_id"
        ]
        .duplicated()
        .sum()
    )

    invalid_lengths = (

        (
            sequence_index[
                "end_row"
            ]

            -

            sequence_index[
                "start_row"
            ]

            +

            1
        )

        !=

        SEQUENCE_LENGTH

    ).sum()

    invalid_labels = (

        ~sequence_index[
            "target"
        ]
        .isin(
            [
                0,
                1,
            ]
        )

    ).sum()

    invalid_returns = (

        ~np.isfinite(

            sequence_index[
                "future_return"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

    ).sum()

    invalid_percentiles = (

        ~sequence_index[
            "cross_section_percentile"
        ]
        .between(
            0.0,
            1.0,
            inclusive="both",
        )

    ).sum()

    invalid_target_order = (

        sequence_index[
            "target_date"
        ]

        <=

        sequence_index[
            "prediction_date"
        ]

    ).sum()

    print(
        f"Duplicate IDs          : "
        f"{duplicate_ids:,}"
    )

    print(
        f"Invalid lengths        : "
        f"{invalid_lengths:,}"
    )

    print(
        f"Invalid labels         : "
        f"{invalid_labels:,}"
    )

    print(
        f"Invalid returns        : "
        f"{invalid_returns:,}"
    )

    print(
        f"Invalid percentiles    : "
        f"{invalid_percentiles:,}"
    )

    print(
        f"Invalid target dates   : "
        f"{invalid_target_order:,}"
    )

    if any(
        [
            duplicate_ids,
            invalid_lengths,
            invalid_labels,
            invalid_returns,
            invalid_percentiles,
            invalid_target_order,
        ]
    ):

        raise ValueError(
            "Sequence validation failed."
        )

    # ========================================================
    # RANDOM WINDOW VALIDATION
    # ========================================================

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

    mapping_failures = 0

    symbol_failures = 0

    split_failures = 0

    prediction_date_failures = 0

    for row in sample.itertuples(
        index=False
    ):

        window = source_data.iloc[

            int(
                row.start_row
            )
            :
            int(
                row.end_row
            )
            +
            1

        ]

        if len(
            window
        ) != SEQUENCE_LENGTH:

            mapping_failures += 1

            continue

        if window[
            "symbol"
        ].nunique() != 1:

            symbol_failures += 1

        if window[
            "dataset_split"
        ].nunique() != 1:

            split_failures += 1

        if (
            window.iloc[
                -1
            ][
                "symbol"
            ]
            !=
            row.symbol
        ):

            mapping_failures += 1

        if (
            window.iloc[
                -1
            ][
                "date"
            ]
            !=
            row.prediction_date
        ):

            prediction_date_failures += 1

    print()

    print(
        f"Windows checked          : "
        f"{sample_size:,}"
    )

    print(
        f"Mapping failures         : "
        f"{mapping_failures:,}"
    )

    print(
        f"Cross-symbol failures    : "
        f"{symbol_failures:,}"
    )

    print(
        f"Cross-split failures     : "
        f"{split_failures:,}"
    )

    print(
        f"Prediction-date failures : "
        f"{prediction_date_failures:,}"
    )

    if (
        mapping_failures > 0
        or
        symbol_failures > 0
        or
        split_failures > 0
        or
        prediction_date_failures > 0
    ):

        raise ValueError(
            "Input-window mapping failed."
        )

    # ========================================================
    # TARGET ROW VALIDATION
    # ========================================================

    target_rows = (

        source_data
        .iloc[
            sequence_index[
                "target_row"
            ]
            .astype(
                np.int64
            )
            .to_numpy()
        ]
        .reset_index(
            drop=True
        )
    )

    expected_symbols = (

        sequence_index[
            "symbol"
        ]
        .reset_index(
            drop=True
        )
    )

    expected_splits = (

        sequence_index[
            "dataset_split"
        ]
        .reset_index(
            drop=True
        )
    )

    expected_dates = (

        pd.to_datetime(
            sequence_index[
                "target_date"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    symbol_target_failures = (

        target_rows[
            "symbol"
        ]
        .reset_index(
            drop=True
        )
        !=
        expected_symbols

    ).sum()

    split_target_failures = (

        target_rows[
            "dataset_split"
        ]
        .reset_index(
            drop=True
        )
        !=
        expected_splits

    ).sum()

    target_date_failures = (

        pd.to_datetime(
            target_rows[
                "date"
            ]
        )
        .reset_index(
            drop=True
        )
        !=
        expected_dates

    ).sum()

    print()

    print(
        f"Target symbol failures : "
        f"{symbol_target_failures:,}"
    )

    print(
        f"Target split failures  : "
        f"{split_target_failures:,}"
    )

    print(
        f"Target date failures   : "
        f"{target_date_failures:,}"
    )

    if (
        symbol_target_failures > 0
        or
        split_target_failures > 0
        or
        target_date_failures > 0
    ):

        raise ValueError(
            "Target-row mapping failed."
        )

    print()

    print(
        "Sequence-index validation PASSED."
    )


# ============================================================
# TARGET-DISTRIBUTION VALIDATION
# ============================================================

def validate_cross_sectional_distribution(
    sequence_index,
):

    print_section(
        "VALIDATING CROSS-SECTIONAL TARGET DISTRIBUTION"
    )

    grouped = (

        sequence_index
        .groupby(
            [
                "dataset_split",
                "prediction_date",
            ],
            observed=True,
        )
        .agg(
            stocks=(
                "symbol",
                "nunique",
            ),
            top30_rate=(
                "target",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        f"Prediction-date groups : "
        f"{len(grouped):,}"
    )

    print()

    for split in [

        "train",
        "backtest",
        "validation",

    ]:

        subset = grouped[
            grouped[
                "dataset_split"
            ]
            ==
            split
        ]

        if subset.empty:

            continue

        print(
            f"{split.upper():10s}: "
            f"mean TOP30="
            f"{subset['top30_rate'].mean():.4f} | "
            f"min="
            f"{subset['top30_rate'].min():.4f} | "
            f"max="
            f"{subset['top30_rate'].max():.4f}"
        )


# ============================================================
# CREATE SUMMARIES
# ============================================================

def create_summaries(
    sequence_index,
):

    print_section(
        "FINAL SEQUENCE DISTRIBUTION"
    )

    summary_records = []

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

        positive_count = int(
            subset[
                "target"
            ].sum()
        )

        negative_count = int(

            len(
                subset
            )

            -

            positive_count
        )

        positive_rate = float(
            subset[
                "target"
            ].mean()
        )

        print(
            f"{split.upper():10s}: "
            f"{len(subset):10,d} sequences | "
            f"{subset['symbol'].nunique():4,d} stocks | "
            f"TOP30="
            f"{positive_rate:7.2%}"
        )

        summary_records.append(
            {

                "dataset_split":
                    split,

                "sequences":
                    int(
                        len(
                            subset
                        )
                    ),

                "stocks":
                    int(
                        subset[
                            "symbol"
                        ].nunique()
                    ),

                "class_0_not_top30":
                    negative_count,

                "class_1_top30":
                    positive_count,

                "top30_rate":
                    positive_rate,

                "prediction_start":
                    str(
                        subset[
                            "prediction_date"
                        ].min().date()
                    ),

                "prediction_end":
                    str(
                        subset[
                            "prediction_date"
                        ].max().date()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_records
    )

    stock_summary = (

        sequence_index

        .groupby(
            [
                "dataset_split",
                "symbol",
            ],
            observed=True,
        )

        .agg(

            sequences=(
                "sequence_id",
                "size",
            ),

            top30_count=(
                "target",
                "sum",
            ),

            top30_rate=(
                "target",
                "mean",
            ),

            mean_future_return=(
                "future_return",
                "mean",
            ),

            mean_cross_section_percentile=(
                "cross_section_percentile",
                "mean",
            ),

            first_prediction=(
                "prediction_date",
                "min",
            ),

            last_prediction=(
                "prediction_date",
                "max",
            ),
        )

        .reset_index()
    )

    return (
        summary_df,
        stock_summary,
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    sequence_index,
    summary_df,
    stock_summary,
):

    print_section(
        "SAVING NEW SEQUENCE DATA"
    )

    sequence_index.to_parquet(

        SEQUENCE_INDEX_FILE,

        index=False,
    )

    summary_df.to_csv(

        SEQUENCE_SUMMARY_FILE,

        index=False,
    )

    stock_summary.to_csv(

        SEQUENCE_STOCK_SUMMARY_FILE,

        index=False,
    )

    split_counts = {

        split:
            int(
                (
                    sequence_index[
                        "dataset_split"
                    ]
                    ==
                    split
                ).sum()
            )

        for split in [

            "train",
            "backtest",
            "validation",

        ]
    }

    split_positive_rates = {

        split:
            float(
                sequence_index.loc[
                    sequence_index[
                        "dataset_split"
                    ]
                    ==
                    split,
                    "target",
                ].mean()
            )

        for split in [

            "train",
            "backtest",
            "validation",

        ]

        if (
            sequence_index[
                "dataset_split"
            ]
            ==
            split
        ).any()
    }

    metadata = {

        "storage":
            "indexed_dynamic_sequences",

        "source_file":
            str(
                PREPROCESSED_TREND_DATA_FILE
            ),

            "source_rows":
             int(
                 pq.ParquetFile(
                  PREPROCESSED_TREND_DATA_FILE
                ).metadata.num_rows
            ),


        "sequence_length":
            int(
                SEQUENCE_LENGTH
            ),

        "prediction_horizon":
            int(
                PREDICTION_HORIZON
            ),

        "feature_count":
            int(
                len(
                    TREND_FEATURE_COLUMNS
                )
            ),

        "feature_columns":
            list(
                TREND_FEATURE_COLUMNS
            ),

        "target_mode":
            TARGET_MODE,

        "target_type":
            "relative_cross_sectional",

        "target_description":
            (
                "Binary classification of whether "
                "a stock belongs to the top 30 percent "
                "of eligible stocks ranked by "
                "21-trading-day future return on the "
                "same prediction date."
            ),

        "relative_top_fraction":
            float(
                RELATIVE_TOP_FRACTION
            ),

        "relative_percentile_cutoff":
            float(
                RELATIVE_PERCENTILE_CUTOFF
            ),

        "class_0":
            TARGET_CLASS_NAMES[
                0
            ],

        "class_1":
            TARGET_CLASS_NAMES[
                1
            ],

        "total_sequences":
            int(
                len(
                    sequence_index
                )
            ),

        "split_counts":
            split_counts,

        "split_positive_rates":
            split_positive_rates,

        "target_formula":
            (
                "future_return = "
                "close[t+21] / close[t] - 1; "
                "target=1 when cross-sectional "
                "percentile > 0.70"
            ),

        "cross_section_grouping":
            [
                "dataset_split",
                "prediction_date",
            ],

        "future_return_grouping":
            [
                "symbol",
                "dataset_split",
            ],

        "sequence_grouping":
            [
                "symbol",
                "dataset_split",
            ],

        "split_safety":
            (
                "Input windows and future targets "
                "remain within the same symbol and "
                "dataset split."
            ),

        "legacy_absolute_target":
            {

                "prediction_horizon":
                    21,

                "high_return_threshold":
                    float(
                        HIGH_RETURN_THRESHOLD
                    ),

                "status":
                    "NOT_USED",
            },
    }

    with open(

        SEQUENCE_METADATA_FILE,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            metadata,

            file,

            indent=4,
        )

    print(
        f"Sequence index:\n"
        f"{SEQUENCE_INDEX_FILE}"
    )

    print()

    print(
        f"Metadata:\n"
        f"{SEQUENCE_METADATA_FILE}"
    )

    print()

    print(
        f"Summary:\n"
        f"{SEQUENCE_SUMMARY_FILE}"
    )

    print()

    print(
        f"Stock summary:\n"
        f"{SEQUENCE_STOCK_SUMMARY_FILE}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_sequences():

    print_section(
        "AGENT 2 - RELATIVE TOP-30% SEQUENCE BUILDER"
    )

    print(
        "Experiment configuration:"
    )

    print()

    print(
        f"Sequence length      : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Prediction horizon   : "
        f"{PREDICTION_HORIZON}"
    )

    print(
        f"Feature count        : "
        f"{len(TREND_FEATURE_COLUMNS)}"
    )

    print(
        f"Target mode          : "
        f"{TARGET_MODE}"
    )

    print(
        f"Top fraction         : "
        f"{RELATIVE_TOP_FRACTION:.0%}"
    )

    print()

    print(
        "Class definitions:"
    )

    print(
        f"  0 = "
        f"{TARGET_CLASS_NAMES[0]}"
    )

    print(
        f"  1 = "
        f"{TARGET_CLASS_NAMES[1]}"
    )

    # ========================================================
    # STAGE 1
    # ========================================================

    data = load_preprocessed_data()

    # ========================================================
    # STAGE 2
    # ========================================================

    data = generate_future_returns(
        data
    )

    # ========================================================
    # STAGE 3
    # ========================================================

    data = generate_relative_labels(
        data
    )

    # ========================================================
    # STAGE 4
    # ========================================================

    sequence_index = build_sequence_index(
        data
    )

    # ========================================================
    # STAGE 5
    # ========================================================

    validate_sequence_index(
        sequence_index,
        data,
    )

    # ========================================================
    # STAGE 6
    # ========================================================

    validate_cross_sectional_distribution(
        sequence_index
    )

    # ========================================================
    # STAGE 7
    # ========================================================

    (
        summary_df,
        stock_summary,

    ) = create_summaries(
        sequence_index
    )

    # ========================================================
    # STAGE 8
    # ========================================================

    save_outputs(

        sequence_index,

        summary_df,

        stock_summary,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print_section(
        "RELATIVE-TARGET SEQUENCE BUILD COMPLETE"
    )

    print(
        "The active Agent 2 label is now:"
    )

    print()

    print(
        "TOP 30% cross-sectional "
        "21-trading-day future return."
    )

    print()

    print(
        "The 60 x 30 model input architecture "
        "has NOT changed."
    )

    print()

    print(
        "Do NOT run trainer.py until its "
        "target-metadata handling has been updated."
    )

    return sequence_index


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    build_sequences()