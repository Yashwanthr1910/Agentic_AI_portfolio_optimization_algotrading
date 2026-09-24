"""
agent.py

Central controller for Agent 1:
Stock Selection Agent

Core Stock Selection Pipeline
-----------------------------
1. Verify processed data
2. Label generation
3. Decision Tree filtering
4. BGSTO optimization
5. Final stock selection
6. Stock ranking

Optional Evaluation
-------------------
7. Equal-weight evaluation portfolio
8. Forward backtest
9. Performance metrics

IMPORTANT:
The evaluation section is NOT the Portfolio Rebalancing Agent
and is NOT the Trade Execution Agent.

Run:
    python src/agent.py

or from the project root:
    python main.py
"""

from pathlib import Path
import os
import subprocess
import sys
import time


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
BACKTEST_DIR = PROJECT_ROOT / "backtesting"


# ============================================================
# SCRIPT LOCATIONS
# ============================================================

SCRIPTS = {

    "data_loader":
        SRC_DIR / "data_loader.py",

    "data_cleaner":
        SRC_DIR / "data_cleaner.py",

    "feature_engineering":
        SRC_DIR / "feature_engineering.py",

    "label_generator":
        SRC_DIR / "label_generator.py",

    "decision_tree_selector":
        SRC_DIR / "decision_tree_selector.py",

    "bgsto_optimizer":
        SRC_DIR / "bgsto_optimizer.py",

    "stock_selector":
        SRC_DIR / "stock_selector.py",

    "stock_scorer":
        SRC_DIR / "stock_scorer.py",

    # Evaluation utilities only
    "portfolio":
        BACKTEST_DIR / "portfolio.py",

    "backtest":
        BACKTEST_DIR / "backtest.py",
}


# ============================================================
# OUTPUT LOCATIONS
# ============================================================

OUTPUTS = {

    "cleaned_data":
        PROJECT_ROOT
        / "data"
        / "processed"
        / "nifty500_daily.parquet",

    "features":
        PROJECT_ROOT
        / "data"
        / "processed"
        / "features.parquet",

    "labeled_features":
        PROJECT_ROOT
        / "data"
        / "processed"
        / "labeled_features.parquet",

    "decision_tree_predictions":
        PROJECT_ROOT
        / "data"
        / "outputs"
        / "decision_tree_predictions.parquet",

    "bgsto_selected":
        PROJECT_ROOT
        / "data"
        / "outputs"
        / "bgsto_selected_stocks.csv",

    "selected_stocks":
        PROJECT_ROOT
        / "data"
        / "outputs"
        / "selected_stocks.csv",

    "rankings":
        PROJECT_ROOT
        / "outputs"
        / "rankings.csv",

    # Evaluation outputs only
    "portfolio":
        PROJECT_ROOT
        / "outputs"
        / "portfolio.csv",

    "backtest":
        PROJECT_ROOT
        / "outputs"
        / "backtest_results.csv",

    "performance":
        PROJECT_ROOT
        / "outputs"
        / "performance.csv",
}


# ============================================================
# EXECUTION CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

# Keep False because these files are already prepared.
RUN_DATA_LOADER = False
RUN_DATA_CLEANER = False
RUN_FEATURE_ENGINEERING = False

# ------------------------------------------------------------
# Stock Selection Agent
# ------------------------------------------------------------

# Turn True if you want labels regenerated.
RUN_LABEL_GENERATOR = False

RUN_DECISION_TREE = True
RUN_BGSTO = True
RUN_STOCK_SELECTOR = True
RUN_STOCK_SCORER = True

# ------------------------------------------------------------
# Evaluation utilities
# ------------------------------------------------------------

# These are NOT separate AI agents.
#
# True  -> also create simple evaluation portfolio/backtest.
# False -> stop after selected/ranked stocks.
#
RUN_EVALUATION = False


# ============================================================
# CONSOLE HELPERS
# ============================================================

def print_line():
    print(
        "=" * 72,
        flush=True,
    )


def print_section(title: str):

    print(flush=True)

    print_line()

    print(
        title,
        flush=True,
    )

    print_line()


def print_status(message: str):

    print(
        f"[AGENT] {message}",
        flush=True,
    )


# ============================================================
# RUN CHILD SCRIPT
# ============================================================

def run_script(
    script_name: str,
    script_path: Path,
) -> None:

    if not script_path.exists():

        raise FileNotFoundError(
            f"{script_name} not found:\n"
            f"{script_path}"
        )

    print_section(
        f"RUNNING: {script_name.upper()}"
    )

    print_status(
        f"Script: {script_path}"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Make child Python processes unbuffered.
    # --------------------------------------------------------

    env = os.environ.copy()

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    result = subprocess.run(

        [
            sys.executable,
            "-u",
            str(script_path),
        ],

        cwd=str(PROJECT_ROOT),

        env=env,

        # Do not capture stdout/stderr.
        # Child output goes directly to PowerShell.
        stdout=None,
        stderr=None,

        check=False,
    )

    elapsed_time = (
        time.time()
        - start_time
    )

    if result.returncode != 0:

        print_status(
            f"ERROR: {script_name} failed."
        )

        print_status(
            f"Exit code: {result.returncode}"
        )

        raise RuntimeError(
            f"Pipeline stopped because "
            f"{script_name} failed."
        )

    print_status(
        f"{script_name} completed successfully."
    )

    print_status(
        f"Execution time: {elapsed_time:.2f} seconds"
    )


# ============================================================
# VERIFY FILE
# ============================================================

def verify_file(
    description: str,
    path: Path,
    required: bool = True,
) -> bool:

    if path.exists():

        file_size_mb = (
            path.stat().st_size
            / (
                1024
                * 1024
            )
        )

        print_status(
            f"[OK] {description}: {path}"
        )

        print_status(
            f"     Size: {file_size_mb:.2f} MB"
        )

        return True

    if required:

        print_status(
            f"[MISSING] {description}: {path}"
        )

        return False

    print_status(
        f"[OPTIONAL MISSING] {description}: {path}"
    )

    return False


# ============================================================
# VERIFY INPUT DATA
# ============================================================

def verify_initial_files():

    print_section(
        "VERIFYING INPUT DATA"
    )

    required_files = [

        (
            "Cleaned Data",
            OUTPUTS[
                "cleaned_data"
            ],
        ),

        (
            "Feature Data",
            OUTPUTS[
                "features"
            ],
        ),
    ]

    missing = []

    for description, path in required_files:

        if not verify_file(
            description,
            path,
        ):

            missing.append(
                path
            )

    if missing:

        raise FileNotFoundError(
            "Required processed data is missing.\n"
            "Run the data preparation scripts first."
        )


# ============================================================
# OPTIONAL DATA PREPARATION
# ============================================================

def run_data_preparation():

    print_section(
        "DATA PREPARATION"
    )

    if RUN_DATA_LOADER:

        run_script(
            "data_loader",
            SCRIPTS[
                "data_loader"
            ],
        )

    else:

        print_status(
            "Skipping data_loader.py"
        )

    if RUN_DATA_CLEANER:

        run_script(
            "data_cleaner",
            SCRIPTS[
                "data_cleaner"
            ],
        )

    else:

        print_status(
            "Skipping data_cleaner.py"
        )

    if RUN_FEATURE_ENGINEERING:

        run_script(
            "feature_engineering",
            SCRIPTS[
                "feature_engineering"
            ],
        )

    else:

        print_status(
            "Skipping feature_engineering.py"
        )


# ============================================================
# LABEL GENERATION
# ============================================================

def run_label_generation():

    if RUN_LABEL_GENERATOR:

        run_script(
            "label_generator",
            SCRIPTS[
                "label_generator"
            ],
        )

    else:

        print_status(
            "Skipping label_generator.py"
        )

        verify_file(
            "Existing labeled features",
            OUTPUTS[
                "labeled_features"
            ],
        )


# ============================================================
# DECISION TREE
# ============================================================

def run_decision_tree():

    if RUN_DECISION_TREE:

        run_script(
            "decision_tree_selector",
            SCRIPTS[
                "decision_tree_selector"
            ],
        )

    else:

        print_status(
            "Skipping decision_tree_selector.py"
        )


# ============================================================
# BGSTO
# ============================================================

def run_bgsto():

    if RUN_BGSTO:

        run_script(
            "bgsto_optimizer",
            SCRIPTS[
                "bgsto_optimizer"
            ],
        )

    else:

        print_status(
            "Skipping bgsto_optimizer.py"
        )


# ============================================================
# FINAL STOCK SELECTION
# ============================================================

def run_stock_selector():

    if RUN_STOCK_SELECTOR:

        run_script(
            "stock_selector",
            SCRIPTS[
                "stock_selector"
            ],
        )

    else:

        print_status(
            "Skipping stock_selector.py"
        )


# ============================================================
# STOCK RANKING
# ============================================================

def run_stock_scorer():

    if RUN_STOCK_SCORER:

        run_script(
            "stock_scorer",
            SCRIPTS[
                "stock_scorer"
            ],
        )

    else:

        print_status(
            "Skipping stock_scorer.py"
        )


# ============================================================
# OPTIONAL EVALUATION
# ============================================================

def run_evaluation():

    print_section(
        "OPTIONAL STOCK-SELECTION EVALUATION"
    )

    if not RUN_EVALUATION:

        print_status(
            "Evaluation disabled."
        )

        print_status(
            "No portfolio construction or "
            "backtesting will be executed."
        )

        return

    print_status(
        "Running evaluation portfolio."
    )

    print_status(
        "NOTE: This is NOT the Portfolio "
        "Rebalancing Agent."
    )

    run_script(
        "portfolio",
        SCRIPTS[
            "portfolio"
        ],
    )

    print_status(
        "Running simple forward backtest."
    )

    print_status(
        "NOTE: This is NOT the DQN "
        "Trade Execution Agent."
    )

    run_script(
        "backtest",
        SCRIPTS[
            "backtest"
        ],
    )


# ============================================================
# VERIFY STOCK SELECTION OUTPUTS
# ============================================================

def verify_stock_selection_outputs():

    print_section(
        "VERIFYING STOCK SELECTION OUTPUTS"
    )

    required_outputs = [

        (
            "Decision Tree Predictions",
            OUTPUTS[
                "decision_tree_predictions"
            ],
        ),

        (
            "BGSTO Selection",
            OUTPUTS[
                "bgsto_selected"
            ],
        ),

        (
            "Selected Stocks",
            OUTPUTS[
                "selected_stocks"
            ],
        ),

        (
            "Stock Rankings",
            OUTPUTS[
                "rankings"
            ],
        ),
    ]

    missing = []

    for description, path in required_outputs:

        if not verify_file(
            description,
            path,
        ):

            missing.append(
                path
            )

    if missing:

        raise RuntimeError(
            "One or more Stock Selection Agent "
            "outputs are missing."
        )


# ============================================================
# VERIFY OPTIONAL EVALUATION OUTPUTS
# ============================================================

def verify_evaluation_outputs():

    if not RUN_EVALUATION:

        return

    print_section(
        "VERIFYING EVALUATION OUTPUTS"
    )

    evaluation_outputs = [

        (
            "Evaluation Portfolio",
            OUTPUTS[
                "portfolio"
            ],
        ),

        (
            "Backtest Results",
            OUTPUTS[
                "backtest"
            ],
        ),

        (
            "Performance Metrics",
            OUTPUTS[
                "performance"
            ],
        ),
    ]

    for description, path in evaluation_outputs:

        verify_file(
            description,
            path,
            required=False,
        )


# ============================================================
# FINAL REPORT
# ============================================================

def display_final_report(
    elapsed_time: float
):

    print_section(
        "STOCK SELECTION AGENT COMPLETED"
    )

    print_status(
        f"Total execution time: "
        f"{elapsed_time:.2f} seconds"
    )

    print(flush=True)

    print_status(
        "Agent 1 output:"
    )

    print(
        f"  Selected Stocks : "
        f"{OUTPUTS['selected_stocks']}",
        flush=True,
    )

    print(
        f"  Rankings        : "
        f"{OUTPUTS['rankings']}",
        flush=True,
    )

    print(flush=True)

    print_status(
        "Completed components:"
    )

    print(
        "  [OK] Decision Tree filtering",
        flush=True,
    )

    print(
        "  [OK] BGSTO optimization",
        flush=True,
    )

    print(
        "  [OK] Final stock selection",
        flush=True,
    )

    print(
        "  [OK] Stock ranking",
        flush=True,
    )

    print(flush=True)

    print_status(
        "Not implemented in Agent 1:"
    )

    print(
        "  [ ] Trend Prediction Agent",
        flush=True,
    )

    print(
        "  [ ] Risk Management Agent",
        flush=True,
    )

    print(
        "  [ ] DQN Trade Execution Agent",
        flush=True,
    )

    print(
        "  [ ] Portfolio Rebalancing Agent",
        flush=True,
    )

    if RUN_EVALUATION:

        print(flush=True)

        print_status(
            "Offline evaluation was also executed."
        )

        print_status(
            "It should NOT be interpreted as "
            "Trade Execution or Portfolio Rebalancing."
        )

    print_line()


# ============================================================
# COMPLETE STOCK SELECTION AGENT
# ============================================================

def run_stock_selection_agent():

    start_time = time.time()

    print_section(
        "STARTING AGENT 1: STOCK SELECTION AGENT"
    )

    print_status(
        f"Project root: {PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Verify existing processed data
    # --------------------------------------------------------

    verify_initial_files()

    # --------------------------------------------------------
    # Optional preprocessing
    # --------------------------------------------------------

    run_data_preparation()

    # --------------------------------------------------------
    # Stock Selection Agent
    # --------------------------------------------------------

    print_section(
        "AGENT 1 - STOCK SELECTION PIPELINE"
    )

    run_label_generation()

    run_decision_tree()

    run_bgsto()

    run_stock_selector()

    run_stock_scorer()

    # --------------------------------------------------------
    # Verify Agent 1 output
    # --------------------------------------------------------

    verify_stock_selection_outputs()

    # --------------------------------------------------------
    # Optional offline evaluation
    # --------------------------------------------------------

    run_evaluation()

    verify_evaluation_outputs()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    display_final_report(
        total_time
    )


# ============================================================
# DIRECT SCRIPT EXECUTION
# ============================================================

def main():

    try:

        run_stock_selection_agent()

    except KeyboardInterrupt:

        print(
            "\nStock Selection Agent interrupted by user.",
            flush=True,
        )

        sys.exit(1)

    except Exception as error:

        print(
            "\nSTOCK SELECTION AGENT FAILED",
            flush=True,
        )

        print(
            f"Error: {error}",
            flush=True,
        )

        raise


if __name__ == "__main__":
    main()