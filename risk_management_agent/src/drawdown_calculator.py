"""
Agent 3 - Drawdown Diagnostics
===============================================================

Purpose
-------
Calculate stock-level and portfolio-level drawdown diagnostics.

Paper relationship
------------------
The paper evaluates max drawdown as an important downside and
portfolio-stability measure.

However:

    Historical Volatility
    VaR
    Sharpe Ratio

remain the explicit core Risk Management Agent measures.

Therefore drawdown is treated here as a supporting diagnostic and
NOT as an exact paper-core input.

Definitions
-----------
Running peak:

    Peak_t = max(P_1 ... P_t)

Drawdown:

    DD_t = P_t / Peak_t - 1

Maximum Drawdown:

    MDD = min(DD_t)

For the portfolio, a wealth index is created from the provisional
portfolio return series:

    Wealth_t = product(1 + R_p,t)

and the same drawdown procedure is applied.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

RISK_AGENT_ROOT = CURRENT_FILE.parents[1]


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
# INPUT PATHS
# ============================================================

def get_input_paths():

    return {

        "risk_input":
            cfg.RISK_INPUT_FILE,

        "combined_risk_metrics":
            cfg.PROCESSED_DIR
            / "combined_risk_metrics.parquet",

        "portfolio_returns":
            cfg.PROCESSED_DIR
            / "portfolio_returns.parquet",

    }


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_input_files():

    section(
        "VALIDATING DRAWDOWN INPUT FILES"
    )

    missing = []

    for name, path in get_input_paths().items():

        exists = Path(
            path
        ).exists()

        print(
            f"{name:<25} : "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:

            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(

            "Drawdown calculation cannot continue. "
            "Missing files:\n\n"

            +
            "\n".join(
                missing
            )
        )


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    section(
        "LOADING DRAWDOWN INPUTS"
    )

    risk_data = pd.read_parquet(
        cfg.RISK_INPUT_FILE
    )

    combined_metrics = pd.read_parquet(

        cfg.PROCESSED_DIR
        / "combined_risk_metrics.parquet"
    )

    portfolio_returns = pd.read_parquet(

        cfg.PROCESSED_DIR
        / "portfolio_returns.parquet"
    )

    if (
        "portfolio_return"
        not in portfolio_returns.columns
    ):

        raise ValueError(
            "portfolio_returns.parquet does not contain "
            "'portfolio_return'."
        )

    print(
        f"Risk rows          : "
        f"{len(risk_data):,}"
    )

    print(
        f"Combined stocks    : "
        f"{len(combined_metrics)}"
    )

    print(
        f"Portfolio returns  : "
        f"{len(portfolio_returns)}"
    )

    return {

        "risk_data":
            risk_data,

        "combined_metrics":
            combined_metrics,

        "portfolio_returns":
            portfolio_returns,

    }


# ============================================================
# STOCK DRAWDOWN
# ============================================================

def calculate_stock_drawdowns(
    risk_data,
):

    section(
        "CALCULATING STOCK DRAWDOWN"
    )

    results = []

    for symbol, stock in risk_data.groupby(
        "symbol"
    ):

        stock = (

            stock
            .sort_values(
                "date"
            )
            .copy()
        )

        close = (

            stock[
                "close"
            ]
            .astype(float)
        )

        running_peak = (
            close.cummax()
        )

        drawdown = (

            close
            /
            running_peak

            -
            1.0
        )

        max_drawdown = (
            drawdown.min()
        )

        max_drawdown_date = (

            stock.loc[
                drawdown.idxmin(),
                "date"
            ]
        )

        current_drawdown = (
            drawdown.iloc[-1]
        )

        latest = stock.iloc[-1]

        results.append({

            "symbol":
                symbol,

            "max_drawdown":
                float(
                    max_drawdown
                ),

            "max_drawdown_percent":
                float(
                    max_drawdown
                    *
                    100.0
                ),

            "max_drawdown_date":
                max_drawdown_date,

            "current_drawdown":
                float(
                    current_drawdown
                ),

            "current_drawdown_percent":
                float(
                    current_drawdown
                    *
                    100.0
                ),

            "prediction_date":
                latest[
                    "prediction_date"
                ],

            "agent2_rank":
                int(
                    latest[
                        "agent2_rank"
                    ]
                ),

        })

    result = pd.DataFrame(
        results
    )

    result = (

        result
        .sort_values(
            "agent2_rank"
        )
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        result[
            [
                "agent2_rank",
                "symbol",
                "max_drawdown_percent",
                "current_drawdown_percent",
            ]
        ]
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.2f}",
        )
    )

    return result


# ============================================================
# PORTFOLIO DRAWDOWN
# ============================================================

def calculate_portfolio_drawdown(
    portfolio_returns,
):

    section(
        "CALCULATING PORTFOLIO DRAWDOWN"
    )

    returns = (

        portfolio_returns[
            "portfolio_return"
        ]
        .astype(float)
    )

    wealth = (

        1.0
        +
        returns
    ).cumprod()

    running_peak = (
        wealth.cummax()
    )

    drawdown = (

        wealth
        /
        running_peak

        -
        1.0
    )

    max_drawdown = (
        drawdown.min()
    )

    max_drawdown_date = (
        drawdown.idxmin()
    )

    current_drawdown = (
        drawdown.iloc[-1]
    )

    peak_wealth = (
        running_peak.max()
    )

    ending_wealth = (
        wealth.iloc[-1]
    )

    metrics = {

        "max_drawdown":
            float(
                max_drawdown
            ),

        "max_drawdown_percent":
            float(
                max_drawdown
                *
                100.0
            ),

        "max_drawdown_date":
            str(
                max_drawdown_date
            ),

        "current_drawdown":
            float(
                current_drawdown
            ),

        "current_drawdown_percent":
            float(
                current_drawdown
                *
                100.0
            ),

        "peak_wealth":
            float(
                peak_wealth
            ),

        "ending_wealth":
            float(
                ending_wealth
            ),

    }

    print(
        f"Portfolio max drawdown     : "
        f"{metrics['max_drawdown_percent']:.2f}%"
    )

    print(
        f"Portfolio current drawdown : "
        f"{metrics['current_drawdown_percent']:.2f}%"
    )

    print(
        f"Peak wealth index          : "
        f"{metrics['peak_wealth']:.4f}"
    )

    print(
        f"Ending wealth index        : "
        f"{metrics['ending_wealth']:.4f}"
    )

    return metrics


# ============================================================
# MERGE
# ============================================================

def merge_drawdown_metrics(
    combined_metrics,
    stock_drawdowns,
):

    section(
        "MERGING DRAWDOWN WITH RISK METRICS"
    )

    drawdown_columns = [

        "symbol",

        "max_drawdown",

        "max_drawdown_percent",

        "max_drawdown_date",

        "current_drawdown",

        "current_drawdown_percent",

    ]

    final_metrics = (
        combined_metrics.merge(

            stock_drawdowns[
                drawdown_columns
            ],

            on="symbol",

            how="left",

            validate="one_to_one",
        )
    )

    final_metrics = (

        final_metrics
        .sort_values(
            "agent2_rank"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Stocks merged : "
        f"{len(final_metrics)}"
    )

    return final_metrics


# ============================================================
# VALIDATION
# ============================================================

def validate_drawdown_results(
    stock_drawdowns,
    portfolio_metrics,
    final_metrics,
):

    section(
        "VALIDATING DRAWDOWN RESULTS"
    )

    failures = []

    finite_stock = (

        stock_drawdowns[
            "max_drawdown"
        ]
        .notna()
        .all()

        and

        np.isfinite(

            stock_drawdowns[
                "max_drawdown"
            ]

        ).all()
    )

    print(
        f"Finite stock drawdowns       : "
        f"{finite_stock}"
    )

    if not finite_stock:

        failures.append(
            "stock drawdown"
        )

    stock_non_positive = (

        stock_drawdowns[
            "max_drawdown"
        ]
        <=
        0
    ).all()

    print(
        f"Stock MDD <= 0               : "
        f"{stock_non_positive}"
    )

    if not stock_non_positive:

        failures.append(
            "stock MDD sign"
        )

    portfolio_valid = (

        np.isfinite(
            portfolio_metrics[
                "max_drawdown"
            ]
        )

        and

        portfolio_metrics[
            "max_drawdown"
        ]
        <=
        0
    )

    print(
        f"Valid portfolio drawdown     : "
        f"{portfolio_valid}"
    )

    if not portfolio_valid:

        failures.append(
            "portfolio drawdown"
        )

    stock_count_valid = (

        len(
            stock_drawdowns
        )
        ==
        len(
            final_metrics
        )
    )

    print(
        f"Stock count consistent        : "
        f"{stock_count_valid}"
    )

    if not stock_count_valid:

        failures.append(
            "stock count"
        )

    if failures:

        print()

        print(
            "VALIDATION RESULT: FAIL"
        )

        for failure in failures:

            print(
                f"  - {failure}"
            )

        raise ValueError(
            "Drawdown validation failed."
        )

    print()

    print(
        "VALIDATION RESULT: PASS"
    )

    return True


# ============================================================
# SAVE
# ============================================================

def save_results(
    stock_drawdowns,
    portfolio_metrics,
    final_metrics,
):

    section(
        "SAVING DRAWDOWN RESULTS"
    )

    stock_parquet = (

        cfg.PROCESSED_DIR
        / "drawdown_metrics.parquet"
    )

    stock_csv = (

        cfg.PROCESSED_DIR
        / "drawdown_metrics.csv"
    )

    final_parquet = (

        cfg.PROCESSED_DIR
        / "risk_metrics_with_drawdown.parquet"
    )

    final_csv = (

        cfg.PROCESSED_DIR
        / "risk_metrics_with_drawdown.csv"
    )

    portfolio_json = (

        cfg.PROCESSED_DIR
        / "portfolio_drawdown.json"
    )

    stock_drawdowns.to_parquet(

        stock_parquet,

        index=False,
    )

    stock_drawdowns.to_csv(

        stock_csv,

        index=False,
    )

    final_metrics.to_parquet(

        final_parquet,

        index=False,
    )

    final_metrics.to_csv(

        final_csv,

        index=False,
    )

    with open(

        portfolio_json,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            portfolio_metrics,

            file,

            indent=4,
        )

    print(
        stock_parquet
    )

    print(
        stock_csv
    )

    print(
        final_parquet
    )

    print(
        final_csv
    )

    print(
        portfolio_json
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_drawdown_calculator():

    section(
        "AGENT 3 - DRAWDOWN DIAGNOSTICS"
    )

    print(
        "Supporting risk diagnostic:"
    )

    print()

    print(
        "Drawdown = Price / RunningPeak - 1"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Drawdown is retained as a supporting diagnostic."
    )

    print(
        "The explicit paper-core Agent 3 metrics remain "
        "Volatility + VaR + Sharpe."
    )


    validate_input_files()


    inputs = load_inputs()


    stock_drawdowns = (
        calculate_stock_drawdowns(

            inputs[
                "risk_data"
            ]
        )
    )


    portfolio_metrics = (
        calculate_portfolio_drawdown(

            inputs[
                "portfolio_returns"
            ]
        )
    )


    final_metrics = (
        merge_drawdown_metrics(

            inputs[
                "combined_metrics"
            ],

            stock_drawdowns,
        )
    )


    validate_drawdown_results(

        stock_drawdowns,

        portfolio_metrics,

        final_metrics,
    )


    save_results(

        stock_drawdowns,

        portfolio_metrics,

        final_metrics,
    )


    section(
        "DRAWDOWN DIAGNOSTICS COMPLETE"
    )

    worst = (

        stock_drawdowns
        .sort_values(
            "max_drawdown"
        )
        .iloc[0]
    )

    print(
        f"Worst stock drawdown : "
        f"{worst['symbol']} "
        f"({worst['max_drawdown_percent']:.2f}%)"
    )

    print(
        f"Portfolio MDD        : "
        f"{portfolio_metrics['max_drawdown_percent']:.2f}%"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Combined risk scoring and risk-adjusted allocation"
    )

    return {

        "stock_drawdowns":
            stock_drawdowns,

        "portfolio_drawdown":
            portfolio_metrics,

        "risk_metrics":
            final_metrics,

    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_drawdown_calculator()