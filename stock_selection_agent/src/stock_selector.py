"""
stock_selector.py

V2.4 - Stock Selection Agent

Purpose:
    Combine the Decision Tree and BGSTO stages into one
    reusable stock-selection pipeline.

Pipeline:
    labeled_features.parquet
            ↓
    Decision Tree Selector
            ↓
    decision_tree_predictions.parquet
            ↓
    BGSTO Optimizer
            ↓
    bgsto_selected_stocks.csv
            ↓
    Stock Selection Agent
            ↓
    selected_stocks.csv
    selected_stocks.parquet

Final V2 Output:
    data/outputs/selected_stocks.csv
"""

from pathlib import Path
import logging
import subprocess
import sys

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

DATA_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
)


# ------------------------------------------------------------
# Scripts
# ------------------------------------------------------------

DECISION_TREE_SCRIPT = (
    SRC_DIR
    / "decision_tree_selector.py"
)

BGSTO_SCRIPT = (
    SRC_DIR
    / "bgsto_optimizer.py"
)


# ------------------------------------------------------------
# BGSTO intermediate output
# ------------------------------------------------------------

BGSTO_SELECTION_FILE = (
    DATA_OUTPUT_DIR
    / "bgsto_selected_stocks.csv"
)


# ------------------------------------------------------------
# Final Stock Selection Agent outputs
# ------------------------------------------------------------

FINAL_CSV_FILE = (
    DATA_OUTPUT_DIR
    / "selected_stocks.csv"
)

FINAL_PARQUET_FILE = (
    DATA_OUTPUT_DIR
    / "selected_stocks.parquet"
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
# RUN PYTHON SCRIPT
# ============================================================

def run_script(
    script_path: Path,
    stage_name: str
) -> None:

    """
    Execute another Python script belonging to the
    Stock Selection Agent.
    """

    if not script_path.exists():

        raise FileNotFoundError(
            f"{stage_name} script not found: "
            f"{script_path}"
        )

    logger.info(
        "--------------------------------------------------"
    )

    logger.info(
        "Starting %s...",
        stage_name
    )

    logger.info(
        "Script: %s",
        script_path
    )

    logger.info(
        "--------------------------------------------------"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_ROOT,
        check=False
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{stage_name} failed with "
            f"exit code {result.returncode}"
        )

    logger.info(
        "%s completed successfully.",
        stage_name
    )


# ============================================================
# RUN DECISION TREE
# ============================================================

def run_decision_tree() -> None:

    """
    Run V2.2 Decision Tree stock-selection stage.
    """

    run_script(
        DECISION_TREE_SCRIPT,
        "Decision Tree stock selection"
    )


# ============================================================
# RUN BGSTO
# ============================================================

def run_bgsto() -> None:

    """
    Run V2.3 BGSTO optimization stage.
    """

    run_script(
        BGSTO_SCRIPT,
        "BGSTO optimization"
    )


# ============================================================
# LOAD BGSTO RESULT
# ============================================================

def load_bgsto_selected_stocks() -> pd.DataFrame:

    """
    Load the stocks selected by BGSTO.
    """

    if not BGSTO_SELECTION_FILE.exists():

        raise FileNotFoundError(
            "BGSTO selected-stock file not found: "
            f"{BGSTO_SELECTION_FILE}"
        )

    logger.info(
        "Loading BGSTO selected stocks..."
    )

    selected = pd.read_csv(
        BGSTO_SELECTION_FILE
    )

    if selected.empty:

        raise ValueError(
            "BGSTO selected-stock file is empty."
        )

    # --------------------------------------------------------
    # Convert date column
    # --------------------------------------------------------

    if "date" in selected.columns:

        selected["date"] = pd.to_datetime(
            selected["date"]
        )

    logger.info(
        "Loaded %d stocks from BGSTO.",
        len(selected)
    )

    return selected


# ============================================================
# VALIDATE FINAL SELECTION
# ============================================================

def validate_selection(
    selected: pd.DataFrame
) -> None:

    """
    Perform basic checks before saving the final
    Stock Selection Agent output.
    """

    logger.info(
        "Validating final stock selection..."
    )

    required_columns = [
        "symbol",
        "buy_probability",
        "bgsto_stock_score"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in selected.columns
    ]

    if missing_columns:

        raise ValueError(
            "Final stock selection is missing "
            f"required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Check duplicate symbols
    # --------------------------------------------------------

    duplicate_count = (
        selected["symbol"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        logger.warning(
            "Found %d duplicate stock symbols.",
            duplicate_count
        )

    # --------------------------------------------------------
    # Check missing symbols
    # --------------------------------------------------------

    missing_symbols = (
        selected["symbol"]
        .isna()
        .sum()
    )

    if missing_symbols > 0:

        raise ValueError(
            f"Found {missing_symbols} rows "
            "with missing stock symbols."
        )

    logger.info(
        "Final selection validation passed."
    )


# ============================================================
# PREPARE FINAL OUTPUT
# ============================================================

def prepare_final_output(
    selected: pd.DataFrame
) -> pd.DataFrame:

    """
    Prepare BGSTO results as the official output
    of the complete Stock Selection Agent.
    """

    final = selected.copy()

    # --------------------------------------------------------
    # Sort by BGSTO rank
    # --------------------------------------------------------

    if "bgsto_rank" in final.columns:

        final = final.sort_values(
            "bgsto_rank"
        )

    elif "bgsto_stock_score" in final.columns:

        final = final.sort_values(
            "bgsto_stock_score",
            ascending=False
        )

    final = final.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Add final selection rank
    # --------------------------------------------------------

    final["selection_rank"] = (
        range(
            1,
            len(final) + 1
        )
    )

    # --------------------------------------------------------
    # Mark every row as selected
    # --------------------------------------------------------

    final["selected"] = 1

    # --------------------------------------------------------
    # Reorder useful columns
    # --------------------------------------------------------

    preferred_columns = [
        "selection_rank",
        "bgsto_rank",
        "date",
        "symbol",
        "selected",
        "buy_probability",
        "bgsto_stock_score",
        "return_21d",
        "return_63d",
        "volatility_60d",
        "sharpe_252d",
        "rsi_14",
        "bgsto_fitness"
    ]

    existing_preferred = [
        column
        for column in preferred_columns
        if column in final.columns
    ]

    remaining_columns = [
        column
        for column in final.columns
        if column not in existing_preferred
    ]

    final = final[
        existing_preferred
        + remaining_columns
    ]

    return final


# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

def save_final_selection(
    selected: pd.DataFrame
) -> None:

    """
    Save the official V2 Stock Selection Agent output.
    """

    DATA_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    selected.to_csv(
        FINAL_CSV_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Parquet
    # --------------------------------------------------------

    selected.to_parquet(
        FINAL_PARQUET_FILE,
        index=False
    )

    logger.info(
        "Final CSV saved to: %s",
        FINAL_CSV_FILE
    )

    logger.info(
        "Final Parquet saved to: %s",
        FINAL_PARQUET_FILE
    )


# ============================================================
# DISPLAY FINAL REPORT
# ============================================================

def display_selection_report(
    selected: pd.DataFrame
) -> None:

    logger.info(
        ""
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "              STOCK SELECTION AGENT REPORT"
    )

    logger.info(
        "============================================================"
    )

    # --------------------------------------------------------
    # Selection date
    # --------------------------------------------------------

    if "date" in selected.columns:

        selection_date = (
            selected["date"]
            .iloc[0]
        )

        logger.info(
            "Selection date: %s",
            selection_date.date()
            if hasattr(
                selection_date,
                "date"
            )
            else selection_date
        )

    # --------------------------------------------------------
    # Number of selected stocks
    # --------------------------------------------------------

    logger.info(
        "Number of selected stocks: %d",
        len(selected)
    )

    # --------------------------------------------------------
    # Display columns
    # --------------------------------------------------------

    display_columns = [
        "selection_rank",
        "symbol",
        "buy_probability",
        "bgsto_stock_score",
        "return_21d",
        "return_63d",
        "volatility_60d",
        "sharpe_252d"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in selected.columns
    ]

    logger.info(
        "\n%s",
        selected[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    logger.info(
        "------------------------------------------------------------"
    )

    if "buy_probability" in selected.columns:

        logger.info(
            "Average BUY probability : %.4f",
            selected[
                "buy_probability"
            ].mean()
        )

    if "bgsto_stock_score" in selected.columns:

        logger.info(
            "Average BGSTO score      : %.4f",
            selected[
                "bgsto_stock_score"
            ].mean()
        )

    if "return_21d" in selected.columns:

        logger.info(
            "Average 21-day return    : %.4f",
            selected[
                "return_21d"
            ].mean()
        )

    if "return_63d" in selected.columns:

        logger.info(
            "Average 63-day return    : %.4f",
            selected[
                "return_63d"
            ].mean()
        )

    if "volatility_60d" in selected.columns:

        logger.info(
            "Average 60-day volatility: %.4f",
            selected[
                "volatility_60d"
            ].mean()
        )

    if "sharpe_252d" in selected.columns:

        logger.info(
            "Average Sharpe ratio     : %.4f",
            selected[
                "sharpe_252d"
            ].mean()
        )

    if "bgsto_fitness" in selected.columns:

        logger.info(
            "BGSTO portfolio fitness  : %.6f",
            selected[
                "bgsto_fitness"
            ].iloc[0]
        )

    # --------------------------------------------------------
    # Symbols
    # --------------------------------------------------------

    logger.info(
        "------------------------------------------------------------"
    )

    logger.info(
        "Selected symbols: %s",
        ", ".join(
            selected[
                "symbol"
            ].astype(str)
        )
    )

    logger.info(
        "============================================================"
    )


# ============================================================
# STOCK SELECTION AGENT
# ============================================================

def select_stocks(
    retrain_decision_tree: bool = True
) -> pd.DataFrame:

    """
    Execute the complete V2 Stock Selection Agent.

    Parameters
    ----------
    retrain_decision_tree : bool

        True:
            Retrain the Decision Tree and regenerate
            Decision Tree predictions.

        False:
            Reuse existing Decision Tree predictions
            and only rerun BGSTO.

    Returns
    -------
    pandas.DataFrame

        Official final V2 selected-stock dataset.
    """

    logger.info(
        "============================================================"
    )

    logger.info(
        "STARTING V2 STOCK SELECTION AGENT"
    )

    logger.info(
        "============================================================"
    )

    # ========================================================
    # STAGE 1
    # Decision Tree
    # ========================================================

    if retrain_decision_tree:

        logger.info(
            "STAGE 1/4: Decision Tree"
        )

        run_decision_tree()

    else:

        logger.info(
            "STAGE 1/4: Decision Tree skipped "
            "(using existing predictions)"
        )

    # ========================================================
    # STAGE 2
    # BGSTO
    # ========================================================

    logger.info(
        "STAGE 2/4: BGSTO Optimization"
    )

    run_bgsto()

    # ========================================================
    # STAGE 3
    # Load + validate
    # ========================================================

    logger.info(
        "STAGE 3/4: Preparing final selection"
    )

    selected = (
        load_bgsto_selected_stocks()
    )

    validate_selection(
        selected
    )

    selected = (
        prepare_final_output(
            selected
        )
    )

    # ========================================================
    # STAGE 4
    # Save official V2 output
    # ========================================================

    logger.info(
        "STAGE 4/4: Saving final V2 output"
    )

    save_final_selection(
        selected
    )

    # ========================================================
    # Report
    # ========================================================

    display_selection_report(
        selected
    )

    logger.info(
        "V2 Stock Selection Agent completed successfully."
    )

    logger.info(
        "Official V2 output:"
    )

    logger.info(
        "%s",
        FINAL_CSV_FILE
    )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    select_stocks(
        retrain_decision_tree=True
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()