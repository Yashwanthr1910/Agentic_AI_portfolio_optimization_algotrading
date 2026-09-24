"""
label_generator.py

Creates supervised learning labels for the Decision Tree
stock-selection stage.

Input:
    data/processed/features.parquet

Output:
    data/processed/labeled_features.parquet

Target:
    future_return_21d

Label:
    1 = BUY
    0 = NON_BUY
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_features.parquet"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

LOOKAHEAD_DAYS = 21

BUY_THRESHOLD = 0.02


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {INPUT_FILE}"
        )

    logger.info("Loading feature dataset...")

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)

    logger.info(
        "Loaded %d rows and %d stocks.",
        len(df),
        df["symbol"].nunique()
    )

    return df


# ============================================================
# FUTURE RETURN
# ============================================================

def add_future_return(
    df: pd.DataFrame,
    lookahead: int = LOOKAHEAD_DAYS
) -> pd.DataFrame:

    logger.info(
        "Calculating %d-day future returns...",
        lookahead
    )

    df = df.copy()

    future_price = (
        df.groupby("symbol")["adj_close"]
        .shift(-lookahead)
    )

    df[f"future_price_{lookahead}d"] = future_price

    df[f"future_return_{lookahead}d"] = (
        future_price / df["adj_close"] - 1
    )

    return df


# ============================================================
# CREATE BUY LABEL
# ============================================================

def create_labels(
    df: pd.DataFrame,
    threshold: float = BUY_THRESHOLD
) -> pd.DataFrame:

    logger.info(
        "Creating BUY labels with threshold %.2f%%...",
        threshold * 100
    )

    target_col = f"future_return_{LOOKAHEAD_DAYS}d"

    df = df.copy()

    df["target"] = np.where(
        df[target_col] > threshold,
        1,
        0
    )

    # The final LOOKAHEAD_DAYS observations
    # for each stock do not have a valid future target.
    invalid_target = df[target_col].isna()

    df.loc[
        invalid_target,
        "target"
    ] = np.nan

    return df


# ============================================================
# DATASET SPLITS
# ============================================================

def add_dataset_split(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Matches the chronological periods used in the base paper.

    Train:
        2015-2018

    Backtest:
        2019-2024

    Validation:
        2025
    """

    logger.info(
        "Assigning chronological dataset splits..."
    )

    df = df.copy()

    df["dataset_split"] = "unused"

    train_mask = (
        (df["date"] >= "2015-01-01")
        & (df["date"] <= "2018-12-31")
    )

    backtest_mask = (
        (df["date"] >= "2019-01-01")
        & (df["date"] <= "2024-12-31")
    )

    validation_mask = (
        (df["date"] >= "2025-01-01")
        & (df["date"] <= "2025-12-31")
    )

    df.loc[
        train_mask,
        "dataset_split"
    ] = "train"

    df.loc[
        backtest_mask,
        "dataset_split"
    ] = "backtest"

    df.loc[
        validation_mask,
        "dataset_split"
    ] = "validation"

    return df


# ============================================================
# TARGET-LEAKAGE CHECK
# ============================================================

def validate_no_target_leakage(
    df: pd.DataFrame
) -> None:

    """
    Basic safety checks to ensure future information
    is not accidentally being used as model input.
    """

    forbidden_feature_names = [
        f"future_price_{LOOKAHEAD_DAYS}d",
        f"future_return_{LOOKAHEAD_DAYS}d",
        "target",
    ]

    logger.info(
        "Target columns created: %s",
        forbidden_feature_names
    )

    logger.info(
        "These columns MUST NOT be used as Decision Tree inputs."
    )


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

def remove_invalid_labels(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Removing rows with unavailable future targets..."
    )

    before = len(df)

    df = df.dropna(
        subset=["target"]
    ).copy()

    df["target"] = df["target"].astype("int8")

    removed = before - len(df)

    logger.info(
        "Removed %d rows without future labels.",
        removed
    )

    return df


# ============================================================
# QUALITY REPORT
# ============================================================

def label_report(
    df: pd.DataFrame
) -> None:

    logger.info(
        "========== LABEL REPORT =========="
    )

    logger.info(
        "Rows: %d",
        len(df)
    )

    logger.info(
        "Stocks: %d",
        df["symbol"].nunique()
    )

    logger.info(
        "Target distribution:\n%s",
        df["target"].value_counts()
    )

    logger.info(
        "Target percentage:\n%s",
        (
            df["target"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )

    logger.info(
        "Dataset split distribution:\n%s",
        df["dataset_split"].value_counts()
    )

    split_target = (
        df.groupby(
            ["dataset_split", "target"]
        )
        .size()
        .unstack(fill_value=0)
    )

    logger.info(
        "Target distribution by split:\n%s",
        split_target
    )

    logger.info(
        "Date range: %s -> %s",
        df["date"].min().date(),
        df["date"].max().date()
    )

    logger.info(
        "=================================="
    )


# ============================================================
# SAVE
# ============================================================

def save_labeled_data(
    df: pd.DataFrame
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        "Labeled dataset saved to: %s",
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting label generation..."
    )

    df = load_features()

    df = add_future_return(df)

    df = create_labels(df)

    df = add_dataset_split(df)

    validate_no_target_leakage(df)

    df = remove_invalid_labels(df)

    label_report(df)

    save_labeled_data(df)

    logger.info(
        "Label generation completed successfully."
    )


if __name__ == "__main__":
    main()