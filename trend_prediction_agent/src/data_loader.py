"""
Agent 2 - Trend Prediction Data Loader
======================================

Purpose
-------
Prepare data for Agent 2 using two separate universes:

1. TRAINING UNIVERSE
   Broad historical NIFTY-500 feature data from Agent 1.

2. INFERENCE UNIVERSE
   Only the stocks currently selected by Agent 1.

This separation is important because the deep trend model needs
a much larger historical sample than only the latest selected stocks.
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

TREND_AGENT_ROOT = CURRENT_FILE.parents[1]

PROJECT_ROOT = TREND_AGENT_ROOT.parent


if str(TREND_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TREND_AGENT_ROOT))


# ============================================================
# INPUT PATHS
# ============================================================

SELECTED_STOCKS_FILE = (
    PROJECT_ROOT
    / "stock_selection_agent"
    / "data"
    / "outputs"
    / "selected_stocks.csv"
)


HISTORICAL_FEATURE_FILE = (
    PROJECT_ROOT
    / "stock_selection_agent"
    / "data"
    / "processed"
    / "features.parquet"
)


# ============================================================
# OUTPUT PATHS
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


INFERENCE_UNIVERSE_FILE = (
    PROCESSED_DIR
    / "inference_universe.parquet"
)


SUMMARY_FILE = (
    PROCESSED_DIR
    / "data_loader_summary.csv"
)


# Legacy compatibility file.
# We keep this temporarily because the current preprocessor
# may still expect trend_data.parquet.

TREND_DATA_FILE = (
    PROCESSED_DIR
    / "trend_data.parquet"
)


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

    "dataset_split",
}


# ============================================================
# PRINT HELPER
# ============================================================

def print_section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)
    print()


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_input_files():

    print_section(
        "VALIDATING AGENT 2 INPUT FILES"
    )

    print(
        "Selected stocks file:"
    )

    print(
        SELECTED_STOCKS_FILE
    )

    print()

    print(
        "Historical feature file:"
    )

    print(
        HISTORICAL_FEATURE_FILE
    )

    print()


    if not SELECTED_STOCKS_FILE.exists():

        raise FileNotFoundError(
            "Agent 1 selected stocks file not found:\n"
            f"{SELECTED_STOCKS_FILE}"
        )


    if not HISTORICAL_FEATURE_FILE.exists():

        raise FileNotFoundError(
            "Agent 1 features.parquet not found:\n"
            f"{HISTORICAL_FEATURE_FILE}"
        )


    print(
        "Input files found successfully."
    )


# ============================================================
# LOAD SELECTED STOCKS
# ============================================================

def load_selected_stocks():

    print_section(
        "LOADING AGENT 1 SELECTED STOCKS"
    )


    print(
        f"File: {SELECTED_STOCKS_FILE}"
    )


    selected = pd.read_csv(
        SELECTED_STOCKS_FILE
    )


    if selected.empty:

        raise ValueError(
            "Selected stocks file is empty."
        )


    print()

    print(
        "Columns received from Agent 1:"
    )

    print(
        list(selected.columns)
    )


    # --------------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # --------------------------------------------------------

    selected.columns = [
        str(column).strip().lower()
        for column in selected.columns
    ]


    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    if "symbol" not in selected.columns:

        raise KeyError(
            "selected_stocks.csv does not contain "
            "a 'symbol' column."
        )


    selected["symbol"] = (
        selected["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # --------------------------------------------------------
    # SELECTION DATE
    # --------------------------------------------------------

    date_source = None


    if "selection_date" in selected.columns:

        date_source = "selection_date"


    elif "date" in selected.columns:

        selected = selected.rename(
            columns={
                "date": "selection_date"
            }
        )

        date_source = "date"


    elif "as_of_date" in selected.columns:

        selected = selected.rename(
            columns={
                "as_of_date": "selection_date"
            }
        )

        date_source = "as_of_date"


    else:

        raise KeyError(
            "Could not find selection date column.\n"
            "Expected one of:\n"
            "selection_date, date, as_of_date"
        )


    selected["selection_date"] = pd.to_datetime(
        selected["selection_date"],
        errors="coerce",
    )


    selected = selected[
        selected["selection_date"].notna()
    ].copy()


    if selected.empty:

        raise ValueError(
            "No valid selection dates found."
        )


    # Use latest Agent 1 selection date.

    selection_date = (
        selected["selection_date"].max()
    )


    selected = selected[
        selected["selection_date"]
        ==
        selection_date
    ].copy()


    # --------------------------------------------------------
    # RANK SORTING
    # --------------------------------------------------------

    if "selection_rank" in selected.columns:

        selected = selected.sort_values(
            "selection_rank"
        )


    selected = (
        selected
        .drop_duplicates(
            subset=["symbol"],
            keep="first",
        )
        .reset_index(drop=True)
    )


    print()

    print(
        f"Selection date source : "
        f"{date_source} -> selection_date"
    )

    print()

    print(
        f"Selected stocks loaded : "
        f"{len(selected):,}"
    )

    print(
        f"Selection date         : "
        f"{selection_date.date()}"
    )

    print()

    print(
        "Selected symbols:"
    )


    for index, row in selected.iterrows():

        rank = (
            row["selection_rank"]
            if "selection_rank" in selected.columns
            else index + 1
        )

        print(
            f"  Rank {int(rank):2d}: "
            f"{row['symbol']}"
        )


    return selected, selection_date


# ============================================================
# LOAD HISTORICAL FEATURES
# ============================================================

def load_historical_features():

    print_section(
        "LOADING AGENT 1 HISTORICAL FEATURES"
    )


    print(
        f"File: {HISTORICAL_FEATURE_FILE}"
    )


    features = pd.read_parquet(
        HISTORICAL_FEATURE_FILE
    )


    if features.empty:

        raise ValueError(
            "features.parquet is empty."
        )


    features.columns = [
        str(column).strip().lower()
        for column in features.columns
    ]


    required_columns = [
        "symbol",
        "date",
    ]


    missing = [
        column
        for column in required_columns
        if column not in features.columns
    ]


    if missing:

        raise KeyError(
            "Missing required columns in "
            "features.parquet:\n"
            f"{missing}"
        )


    features["symbol"] = (
        features["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    )


    invalid_date_rows = (
        features["date"].isna().sum()
    )


    if invalid_date_rows > 0:

        print()

        print(
            f"[WARNING] Removing "
            f"{invalid_date_rows:,} "
            f"rows with invalid dates."
        )

        features = features[
            features["date"].notna()
        ].copy()


    # Remove duplicate symbol/date rows if any.

    duplicate_rows = (
        features
        .duplicated(
            subset=[
                "symbol",
                "date",
            ]
        )
        .sum()
    )


    if duplicate_rows > 0:

        print()

        print(
            f"[WARNING] Removing "
            f"{duplicate_rows:,} duplicate "
            f"symbol/date rows."
        )


        features = (
            features
            .drop_duplicates(
                subset=[
                    "symbol",
                    "date",
                ],
                keep="last",
            )
        )


    features = (
        features
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


    print()

    print(
        f"Historical rows loaded : "
        f"{len(features):,}"
    )

    print(
        f"Stocks available       : "
        f"{features['symbol'].nunique():,}"
    )

    print(
        f"Date range             : "
        f"{features['date'].min().date()} "
        f"-> "
        f"{features['date'].max().date()}"
    )

    print(
        f"Columns                : "
        f"{len(features.columns)}"
    )


    return features


# ============================================================
# REMOVE LEAKAGE COLUMNS
# ============================================================

def remove_leakage_columns(dataframe):

    dataframe = dataframe.copy()


    columns_to_remove = [
        column
        for column in dataframe.columns
        if column.lower()
        in LEAKAGE_COLUMNS
    ]


    if columns_to_remove:

        print()

        print(
            "Removing leakage columns:"
        )

        print(
            columns_to_remove
        )


        dataframe = dataframe.drop(
            columns=columns_to_remove,
            errors="ignore",
        )


    return dataframe


# ============================================================
# BUILD BROAD TRAINING UNIVERSE
# ============================================================

def build_training_universe(
    features,
    selection_date,
):

    print_section(
        "BUILDING BROAD TRAINING UNIVERSE"
    )


    original_rows = len(features)


    # --------------------------------------------------------
    # CRITICAL DESIGN
    # --------------------------------------------------------
    #
    # Keep ALL stocks.
    #
    # Only remove observations after the Agent 1 selection date.
    #
    # This is different from the old implementation, which first
    # filtered to the 10 selected stocks.
    #

    training = features[
        features["date"]
        <=
        selection_date
    ].copy()


    future_rows_removed = (
        original_rows
        -
        len(training)
    )


    training = remove_leakage_columns(
        training
    )


    training["data_role"] = (
        "training_universe"
    )


    training["selection_date"] = (
        selection_date
    )


    training = (
        training
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


    print(
        f"Rows before date filter : "
        f"{original_rows:,}"
    )

    print(
        f"Historical rows retained: "
        f"{len(training):,}"
    )

    print(
        f"Future rows removed     : "
        f"{future_rows_removed:,}"
    )

    print(
        f"Historical stocks       : "
        f"{training['symbol'].nunique():,}"
    )

    print(
        f"Start date              : "
        f"{training['date'].min().date()}"
    )

    print(
        f"End date                : "
        f"{training['date'].max().date()}"
    )


    return training


# ============================================================
# BUILD CURRENT INFERENCE UNIVERSE
# ============================================================

def build_inference_universe(
    features,
    selected_stocks,
    selection_date,
):

    print_section(
        "BUILDING CURRENT INFERENCE UNIVERSE"
    )


    selected_symbols = (
        selected_stocks["symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )


    inference = features[
        features["symbol"].isin(
            selected_symbols
        )
    ].copy()


    rows_before_date_filter = (
        len(inference)
    )


    inference = inference[
        inference["date"]
        <=
        selection_date
    ].copy()


    future_rows_removed = (
        rows_before_date_filter
        -
        len(inference)
    )


    inference = remove_leakage_columns(
        inference
    )


    inference["data_role"] = (
        "inference_universe"
    )


    inference["selection_date"] = (
        selection_date
    )


    inference = (
        inference
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


    available_symbols = set(
        inference["symbol"].unique()
    )


    requested_symbols = set(
        selected_symbols
    )


    missing_symbols = sorted(
        requested_symbols
        -
        available_symbols
    )


    print(
        f"Selected stocks requested : "
        f"{len(requested_symbols):,}"
    )

    print(
        f"Selected stocks available : "
        f"{len(available_symbols):,}"
    )

    print(
        f"Inference rows retained   : "
        f"{len(inference):,}"
    )

    print(
        f"Future rows removed       : "
        f"{future_rows_removed:,}"
    )


    if not inference.empty:

        print(
            f"Start date                : "
            f"{inference['date'].min().date()}"
        )

        print(
            f"End date                  : "
            f"{inference['date'].max().date()}"
        )


    print()

    print(
        "Historical observations "
        "for selected stocks:"
    )


    grouped = inference.groupby(
        "symbol"
    )


    for symbol, group in grouped:

        print(
            f"  {symbol:15s} "
            f"{len(group):6,d} rows   "
            f"{group['date'].min().date()} "
            f"-> "
            f"{group['date'].max().date()}"
        )


    if missing_symbols:

        print()

        print(
            "[WARNING] Selected stocks "
            "without historical data:"
        )

        print(
            ", ".join(missing_symbols)
        )


    return inference


# ============================================================
# VALIDATE OUTPUTS
# ============================================================

def validate_outputs(
    training,
    inference,
    selected_stocks,
    selection_date,
):

    print_section(
        "VALIDATING IMPROVED AGENT 2 DATASETS"
    )


    # --------------------------------------------------------
    # TRAINING UNIVERSE
    # --------------------------------------------------------

    train_duplicates = (
        training
        .duplicated(
            subset=[
                "symbol",
                "date",
            ]
        )
        .sum()
    )


    inference_duplicates = (
        inference
        .duplicated(
            subset=[
                "symbol",
                "date",
            ]
        )
        .sum()
    )


    training_future_rows = (
        training["date"]
        >
        selection_date
    ).sum()


    inference_future_rows = (
        inference["date"]
        >
        selection_date
    ).sum()


    leakage_training = [
        column
        for column in training.columns
        if column.lower()
        in LEAKAGE_COLUMNS
    ]


    leakage_inference = [
        column
        for column in inference.columns
        if column.lower()
        in LEAKAGE_COLUMNS
    ]


    requested_symbols = set(
        selected_stocks["symbol"].unique()
    )


    inference_symbols = set(
        inference["symbol"].unique()
    )


    missing_inference_symbols = (
        requested_symbols
        -
        inference_symbols
    )


    print(
        "TRAINING UNIVERSE"
    )

    print(
        "-" * 60
    )

    print(
        f"Rows                    : "
        f"{len(training):,}"
    )

    print(
        f"Stocks                  : "
        f"{training['symbol'].nunique():,}"
    )

    print(
        f"Duplicate symbol/date   : "
        f"{train_duplicates:,}"
    )

    print(
        f"Post-selection rows     : "
        f"{training_future_rows:,}"
    )

    print(
        f"Leakage columns present : "
        f"{leakage_training}"
    )


    print()

    print(
        "INFERENCE UNIVERSE"
    )

    print(
        "-" * 60
    )

    print(
        f"Rows                    : "
        f"{len(inference):,}"
    )

    print(
        f"Stocks                  : "
        f"{inference['symbol'].nunique():,}"
    )

    print(
        f"Duplicate symbol/date   : "
        f"{inference_duplicates:,}"
    )

    print(
        f"Post-selection rows     : "
        f"{inference_future_rows:,}"
    )

    print(
        f"Leakage columns present : "
        f"{leakage_inference}"
    )

    print(
        f"Selected stocks missing : "
        f"{len(missing_inference_symbols):,}"
    )


    if train_duplicates > 0:

        raise ValueError(
            "Training universe contains "
            "duplicate symbol/date rows."
        )


    if inference_duplicates > 0:

        raise ValueError(
            "Inference universe contains "
            "duplicate symbol/date rows."
        )


    if training_future_rows > 0:

        raise ValueError(
            "Training universe contains "
            "post-selection data."
        )


    if inference_future_rows > 0:

        raise ValueError(
            "Inference universe contains "
            "post-selection data."
        )


    if leakage_training:

        raise ValueError(
            "Leakage columns remain in "
            "training universe."
        )


    if leakage_inference:

        raise ValueError(
            "Leakage columns remain in "
            "inference universe."
        )


    if missing_inference_symbols:

        print()

        print(
            "[WARNING] Some Agent 1 stocks "
            "have no Agent 2 history:"
        )

        print(
            sorted(
                missing_inference_symbols
            )
        )


    if (
        training["symbol"].nunique()
        <=
        inference["symbol"].nunique()
    ):

        raise ValueError(
            "Broad training universe was "
            "not created correctly.\n"
            "Training universe must contain "
            "more stocks than the current "
            "inference universe."
        )


    print()

    print(
        "Improved Agent 2 data validation PASSED."
    )


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_summary(
    training,
    inference,
    selection_date,
):

    summary = pd.DataFrame(
        [
            {
                "dataset": (
                    "training_universe"
                ),

                "rows": (
                    len(training)
                ),

                "stocks": (
                    training["symbol"]
                    .nunique()
                ),

                "start_date": (
                    training["date"]
                    .min()
                ),

                "end_date": (
                    training["date"]
                    .max()
                ),

                "selection_date": (
                    selection_date
                ),
            },

            {
                "dataset": (
                    "inference_universe"
                ),

                "rows": (
                    len(inference)
                ),

                "stocks": (
                    inference["symbol"]
                    .nunique()
                ),

                "start_date": (
                    inference["date"]
                    .min()
                ),

                "end_date": (
                    inference["date"]
                    .max()
                ),

                "selection_date": (
                    selection_date
                ),
            },
        ]
    )


    return summary


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

def save_outputs(
    training,
    inference,
    summary,
):

    print_section(
        "SAVING IMPROVED AGENT 2 DATASETS"
    )


    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    training.to_parquet(
        TRAINING_UNIVERSE_FILE,
        index=False,
    )


    inference.to_parquet(
        INFERENCE_UNIVERSE_FILE,
        index=False,
    )


    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    # --------------------------------------------------------
    # LEGACY COMPATIBILITY
    # --------------------------------------------------------
    #
    # Temporarily write the BROAD training universe to
    # trend_data.parquet.
    #
    # In the next step we will update data_preprocessor.py
    # to read training_universe.parquet explicitly.
    #

    training.to_parquet(
        TREND_DATA_FILE,
        index=False,
    )


    print(
        "Training universe saved:"
    )

    print(
        TRAINING_UNIVERSE_FILE
    )

    print()

    print(
        "Inference universe saved:"
    )

    print(
        INFERENCE_UNIVERSE_FILE
    )

    print()

    print(
        "Summary saved:"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "Legacy trend_data.parquet updated:"
    )

    print(
        TREND_DATA_FILE
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_data_loader():

    print_section(
        "AGENT 2 - IMPROVED TREND PREDICTION DATA LOADER"
    )


    print(
        "Training universe : "
        "Broad historical NIFTY-500 feature universe"
    )

    print(
        "Inference universe: "
        "Current Agent 1 selected stocks"
    )


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    validate_input_files()


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    selected_stocks, selection_date = (
        load_selected_stocks()
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    historical_features = (
        load_historical_features()
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    training_universe = (
        build_training_universe(
            historical_features,
            selection_date,
        )
    )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    inference_universe = (
        build_inference_universe(
            historical_features,
            selected_stocks,
            selection_date,
        )
    )


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    validate_outputs(
        training_universe,
        inference_universe,
        selected_stocks,
        selection_date,
    )


    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    summary = create_summary(
        training_universe,
        inference_universe,
        selection_date,
    )


    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    save_outputs(
        training_universe,
        inference_universe,
        summary,
    )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_section(
        "DATA LOADER COMPLETED SUCCESSFULLY"
    )


    print(
        "TRAINING UNIVERSE"
    )

    print(
        "-" * 60
    )

    print(
        f"Rows   : "
        f"{len(training_universe):,}"
    )

    print(
        f"Stocks : "
        f"{training_universe['symbol'].nunique():,}"
    )


    print()

    print(
        "INFERENCE UNIVERSE"
    )

    print(
        "-" * 60
    )

    print(
        f"Rows   : "
        f"{len(inference_universe):,}"
    )

    print(
        f"Stocks : "
        f"{inference_universe['symbol'].nunique():,}"
    )


    print()

    print(
        "Expected behavior:"
    )

    print(
        "  Training universe should contain "
        "hundreds of stocks."
    )

    print(
        "  Inference universe should contain "
        "the 10 Agent 1 selected stocks."
    )


    return {
        "training_universe": (
            training_universe
        ),

        "inference_universe": (
            inference_universe
        ),

        "selected_stocks": (
            selected_stocks
        ),

        "selection_date": (
            selection_date
        ),

        "summary": (
            summary
        ),
    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_data_loader()