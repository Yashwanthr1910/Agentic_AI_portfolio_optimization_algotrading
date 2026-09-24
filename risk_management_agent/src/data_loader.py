"""
Agent 3 - Risk Management Data Loader
===============================================================

Purpose
-------
Prepare all historical data required by the Risk Management
Agent.

Inputs
------
1. Agent 2 final trend predictions.
2. Agent 1 cleaned historical OHLCV data.

Outputs
-------
1. risk_input.parquet
2. return_matrix.parquet

Important leakage rule
----------------------
For every selected stock:

    market_date <= Agent 2 prediction_date

No observation occurring after the prediction date may be used.

The loader also restricts the risk history to the configured
historical lookback period.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

RISK_AGENT_ROOT = CURRENT_FILE.parents[1]

PROJECT_ROOT = RISK_AGENT_ROOT.parent


if str(RISK_AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(RISK_AGENT_ROOT),
    )


from config import config as cfg


# ============================================================
# DISPLAY
# ============================================================

def section(title):

    print()

    print("=" * 95)

    print(title)

    print("=" * 95)

    print()


# ============================================================
# VALIDATE FILES
# ============================================================

def validate_input_files():

    section(
        "VALIDATING AGENT 3 INPUT FILES"
    )

    required_files = {

        "Agent 2 predictions":
            cfg.AGENT2_PREDICTIONS_PARQUET,

        "Historical market data":
            cfg.HISTORICAL_DATA_FILE,

    }

    missing = []

    for name, path in required_files.items():

        exists = Path(
            path
        ).exists()

        print(
            f"{name:<30} : "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        print(
            f"{'':<30}   {path}"
        )

        if not exists:

            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(

            "Missing required Agent 3 files:\n\n"

            +
            "\n".join(
                missing
            )
        )


# ============================================================
# LOAD AGENT 2
# ============================================================

def load_agent2_predictions():

    section(
        "LOADING AGENT 2 FINAL PREDICTIONS"
    )

    data = pd.read_parquet(
        cfg.AGENT2_PREDICTIONS_PARQUET
    )

    missing_columns = [

        column

        for column
        in cfg.REQUIRED_AGENT2_COLUMNS

        if column
        not in data.columns
    ]

    if missing_columns:

        raise ValueError(

            "Agent 2 prediction file is missing:\n\n"

            +
            "\n".join(
                missing_columns
            )
        )

    data = data.copy()

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
        "prediction_date"
    ] = pd.to_datetime(

        data[
            "prediction_date"
        ],

        errors="coerce",
    )

    if data[
        "prediction_date"
    ].isna().any():

        raise ValueError(
            "Invalid Agent 2 prediction date."
        )

    duplicate_count = (

        data[
            "symbol"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        raise ValueError(

            "Agent 2 contains duplicate stock predictions: "
            f"{duplicate_count}"
        )

    print(
        f"Predictions : "
        f"{len(data)}"
    )

    print(
        f"Stocks      : "
        f"{data['symbol'].nunique()}"
    )

    print(
        f"Date        : "
        f"{data['prediction_date'].min().date()} "
        f"-> "
        f"{data['prediction_date'].max().date()}"
    )

    print()

    display = (

        data[
            [
                "agent2_rank",
                "symbol",
                "top30_probability",
                "ensemble_std",
                "trend_class",
            ]
        ]
        .sort_values(
            "agent2_rank"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )

    return data


# ============================================================
# LOAD HISTORICAL MARKET DATA
# ============================================================

def load_historical_market_data():

    section(
        "LOADING HISTORICAL MARKET DATA"
    )

    data = pd.read_parquet(
        cfg.HISTORICAL_DATA_FILE
    )

    # Normalize column names.

    data.columns = [

        str(column)
        .strip()
        .lower()

        for column
        in data.columns
    ]

    missing_columns = [

        column

        for column
        in cfg.REQUIRED_MARKET_COLUMNS

        if column
        not in data.columns
    ]

    if missing_columns:

        print(
            "Available columns:"
        )

        print(
            list(
                data.columns
            )
        )

        raise ValueError(

            "Historical data is missing required columns:\n\n"

            +
            "\n".join(
                missing_columns
            )
        )

    data = data.copy()

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

    # Numeric OHLCV conversion.

    numeric_columns = [

        "open",
        "high",
        "low",
        "close",
        "volume",

    ]

    for column in numeric_columns:

        data[
            column
        ] = pd.to_numeric(

            data[
                column
            ],

            errors="coerce",
        )

    data = data.dropna(

        subset=[

            "symbol",
            "date",
            "close",

        ]
    )

    data = data[

        np.isfinite(
            data[
                "close"
            ]
        )

    ]

    data = data[

        data[
            "close"
        ]
        >
        0

    ]

    data = (

        data
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .drop_duplicates(

            subset=[

                "symbol",
                "date",

            ],

            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Rows       : "
        f"{len(data):,}"
    )

    print(
        f"Stocks     : "
        f"{data['symbol'].nunique():,}"
    )

    print(
        f"Date range : "
        f"{data['date'].min().date()} "
        f"-> "
        f"{data['date'].max().date()}"
    )

    return data


# ============================================================
# BUILD RISK HISTORY
# ============================================================

def build_risk_history(
    predictions,
    market,
):

    section(
        "BUILDING HISTORICAL RISK DATA"
    )

    selected_symbols = (

        predictions[
            "symbol"
        ]
        .tolist()
    )

    market = market[

        market[
            "symbol"
        ].isin(
            selected_symbols
        )

    ].copy()

    pieces = []

    skipped = []

    for _, prediction in predictions.iterrows():

        symbol = (
            prediction[
                "symbol"
            ]
        )

        prediction_date = (
            prediction[
                "prediction_date"
            ]
        )

        stock = market[

            market[
                "symbol"
            ]
            ==
            symbol

        ].copy()

        # ----------------------------------------------------
        # LEAKAGE CONTROL
        # ----------------------------------------------------

        stock = stock[

            stock[
                "date"
            ]
            <=
            prediction_date

        ].copy()

        stock = (

            stock
            .sort_values(
                "date"
            )
        )

        # ----------------------------------------------------
        # Keep configured historical lookback.
        #
        # We keep one additional row so that pct_change can
        # produce approximately RISK_LOOKBACK_DAYS returns.
        # ----------------------------------------------------

        required_rows = (
            cfg.RISK_LOOKBACK_DAYS
            +
            1
        )

        if len(stock) > required_rows:

            stock = stock.tail(
                required_rows
            )

        if len(stock) < (
            cfg.MIN_HISTORY_DAYS
            +
            1
        ):

            skipped.append(
                symbol
            )

            print(
                f"{symbol:<20} "
                f"SKIPPED - "
                f"only {len(stock)} historical rows"
            )

            continue

        # ----------------------------------------------------
        # DAILY SIMPLE RETURN
        # ----------------------------------------------------

        stock[
            cfg.RETURN_COLUMN
        ] = (

            stock[
                "close"
            ]
            .pct_change()
        )

        # ----------------------------------------------------
        # Attach Agent-2 signal information.
        # ----------------------------------------------------

        stock[
            "prediction_date"
        ] = prediction_date

        stock[
            "top30_probability"
        ] = prediction[
            "top30_probability"
        ]

        stock[
            "ensemble_std"
        ] = prediction[
            "ensemble_std"
        ]

        stock[
            "trend_class"
        ] = prediction[
            "trend_class"
        ]

        stock[
            "agent2_rank"
        ] = prediction[
            "agent2_rank"
        ]

        pieces.append(
            stock
        )

        valid_returns = (

            stock[
                cfg.RETURN_COLUMN
            ]
            .notna()
            .sum()
        )

        print(
            f"{symbol:<20} "
            f"rows={len(stock):>4} | "
            f"returns={valid_returns:>4} | "
            f"{stock['date'].min().date()} "
            f"-> "
            f"{stock['date'].max().date()}"
        )

    if not pieces:

        raise ValueError(

            "No stocks contained sufficient "
            "historical data for Agent 3."
        )

    risk_data = pd.concat(

        pieces,

        ignore_index=True,
    )

    risk_data = (

        risk_data
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

    print()

    print(
        f"Risk-data rows : "
        f"{len(risk_data):,}"
    )

    print(
        f"Stocks         : "
        f"{risk_data['symbol'].nunique()}"
    )

    print(
        f"Skipped        : "
        f"{len(skipped)}"
    )

    return (
        risk_data,
        skipped,
    )


# ============================================================
# BUILD RETURN MATRIX
# ============================================================

def build_return_matrix(
    risk_data,
):

    section(
        "BUILDING RETURN MATRIX"
    )

    returns = risk_data.dropna(

        subset=[
            cfg.RETURN_COLUMN
        ]

    ).copy()

    return_matrix = returns.pivot(

        index="date",

        columns="symbol",

        values=cfg.RETURN_COLUMN,

    )

    return_matrix = return_matrix.sort_index()

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We do NOT fill missing returns with zero.
    #
    # Zero would mean "stock had exactly zero return",
    # which is not equivalent to missing data.
    # --------------------------------------------------------

    print(
        f"Matrix shape : "
        f"{return_matrix.shape}"
    )

    print(
        f"Dates        : "
        f"{return_matrix.index.min().date()} "
        f"-> "
        f"{return_matrix.index.max().date()}"
    )

    print(
        f"Stocks       : "
        f"{return_matrix.shape[1]}"
    )

    print()

    print(
        "Observations per stock:"
    )

    print(
        return_matrix
        .count()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    return return_matrix


# ============================================================
# VALIDATE RISK DATA
# ============================================================

def validate_risk_data(
    risk_data,
    return_matrix,
    predictions,
):

    section(
        "VALIDATING AGENT 3 DATA"
    )

    failures = []

    expected_symbols = set(

        predictions[
            "symbol"
        ]
    )

    actual_symbols = set(

        risk_data[
            "symbol"
        ]
    )

    symbol_coverage = (

        expected_symbols
        ==
        actual_symbols
    )

    print(
        f"All Agent 2 stocks present : "
        f"{symbol_coverage}"
    )

    if not symbol_coverage:

        missing = sorted(

            expected_symbols
            -
            actual_symbols
        )

        print(
            f"Missing symbols             : "
            f"{missing}"
        )

        failures.append(
            "symbol coverage"
        )


    future_rows = risk_data[

        risk_data[
            "date"
        ]
        >
        risk_data[
            "prediction_date"
        ]

    ]

    no_future_leakage = (

        len(
            future_rows
        )
        ==
        0
    )

    print(
        f"No future-data leakage      : "
        f"{no_future_leakage}"
    )

    if not no_future_leakage:

        failures.append(
            "future-data leakage"
        )


    close_valid = (

        risk_data[
            "close"
        ]
        .notna()
        .all()

        and

        np.isfinite(
            risk_data[
                "close"
            ]
        ).all()

        and

        (
            risk_data[
                "close"
            ]
            >
            0
        ).all()

    )

    print(
        f"Valid closing prices        : "
        f"{close_valid}"
    )

    if not close_valid:

        failures.append(
            "closing prices"
        )


    probability_valid = (

        risk_data[
            "top30_probability"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    print(
        f"Valid Agent 2 probabilities : "
        f"{probability_valid}"
    )

    if not probability_valid:

        failures.append(
            "Agent 2 probabilities"
        )


    return_matrix_valid = (

        return_matrix.shape[1]
        ==
        len(
            expected_symbols
        )
    )

    print(
        f"Return matrix stock count   : "
        f"{return_matrix_valid}"
    )

    if not return_matrix_valid:

        failures.append(
            "return matrix"
        )


    if failures:

        print()

        print(
            "VALIDATION RESULT: FAIL"
        )

        print()

        for failure in failures:

            print(
                f"  - {failure}"
            )

        raise ValueError(

            "Agent 3 data validation failed."
        )

    print()

    print(
        "VALIDATION RESULT: PASS"
    )

    return True


# ============================================================
# SAVE FILES
# ============================================================

def save_outputs(
    risk_data,
    return_matrix,
):

    section(
        "SAVING AGENT 3 PROCESSED DATA"
    )

    risk_data.to_parquet(

        cfg.RISK_INPUT_FILE,

        index=False,
    )


    return_matrix.to_parquet(

        cfg.RETURN_MATRIX_FILE
    )


    print(
        "Risk input:"
    )

    print(
        cfg.RISK_INPUT_FILE
    )

    print()

    print(
        "Return matrix:"
    )

    print(
        cfg.RETURN_MATRIX_FILE
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_data_loader():

    section(
        "AGENT 3 - RISK MANAGEMENT DATA PREPARATION"
    )

    print(
        "Paper-aligned core:"
    )

    print(
        "Historical Volatility + VaR + Sharpe Ratio"
    )

    print()

    print(
        f"Historical lookback : "
        f"{cfg.RISK_LOOKBACK_DAYS} trading days"
    )

    print(
        f"VaR confidence      : "
        f"{cfg.VAR_CONFIDENCE_LEVEL:.0%}"
    )

    print(
        f"Risk-free rate      : "
        f"{cfg.ANNUAL_RISK_FREE_RATE:.2%}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Lookback, confidence level and risk-free rate "
        "are explicit implementation choices."
    )


    validate_input_files()


    predictions = (
        load_agent2_predictions()
    )


    market = (
        load_historical_market_data()
    )


    risk_data, skipped = (
        build_risk_history(
            predictions,
            market,
        )
    )


    return_matrix = (
        build_return_matrix(
            risk_data
        )
    )


    validate_risk_data(

        risk_data,

        return_matrix,

        predictions,
    )


    save_outputs(

        risk_data,

        return_matrix,
    )


    section(
        "AGENT 3 DATA PREPARATION COMPLETE"
    )

    print(
        f"Agent 2 stocks received : "
        f"{len(predictions)}"
    )

    print(
        f"Agent 3 stocks prepared : "
        f"{risk_data['symbol'].nunique()}"
    )

    print(
        f"Stocks skipped          : "
        f"{len(skipped)}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Historical Volatility calculation"
    )

    return {

        "predictions":
            predictions,

        "risk_data":
            risk_data,

        "return_matrix":
            return_matrix,

        "skipped":
            skipped,

    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_data_loader()