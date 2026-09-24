"""
Agent 3 - Sharpe Ratio Calculator
===============================================================

Purpose
-------
Calculate stock-level and portfolio-level Sharpe ratios.

Paper alignment
---------------
The base paper states that the Risk Management Agent evaluates:

    1. Historical Volatility
    2. Value at Risk
    3. Sharpe Ratio

Conceptually:

                Rp - Rf
    Sharpe = -------------
                  sigma

where:

    Rp      = return
    Rf      = risk-free return
    sigma   = return volatility

For daily data:

    Sharpe_annual =
        ((mean_daily_return - daily_risk_free_rate)
        / std_daily_return)
        * sqrt(252)

The portfolio Sharpe ratio uses the provisional portfolio weights
created during the volatility stage.

IMPORTANT
---------
The paper does not clearly specify the exact risk-free rate or
lookback window.

Current implementation choices:

    Risk-free rate = 0.0 annual
    Lookback       = 252 trading days
    Annualization  = sqrt(252)

These are implementation choices, not claimed paper hyperparameters.
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

        "return_matrix":
            cfg.RETURN_MATRIX_FILE,

        "volatility_metrics":
            cfg.PROCESSED_DIR
            / "volatility_metrics.parquet",

        "var_metrics":
            cfg.PROCESSED_DIR
            / "var_metrics.parquet",

        "provisional_weights":
            cfg.PROCESSED_DIR
            / "provisional_weights.csv",

    }


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_input_files():

    section(
        "VALIDATING SHARPE INPUT FILES"
    )

    paths = get_input_paths()

    missing = []

    for name, path in paths.items():

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

            "Sharpe calculation cannot continue. "
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
        "LOADING SHARPE INPUTS"
    )

    risk_data = pd.read_parquet(
        cfg.RISK_INPUT_FILE
    )

    return_matrix = pd.read_parquet(
        cfg.RETURN_MATRIX_FILE
    )

    volatility = pd.read_parquet(

        cfg.PROCESSED_DIR
        / "volatility_metrics.parquet"
    )

    var_metrics = pd.read_parquet(

        cfg.PROCESSED_DIR
        / "var_metrics.parquet"
    )

    weights_df = pd.read_csv(

        cfg.PROCESSED_DIR
        / "provisional_weights.csv",

        index_col=0,
    )

    # --------------------------------------------------------
    # Normalize symbols
    # --------------------------------------------------------

    return_matrix.columns = (

        return_matrix.columns
        .astype(str)
        .str.strip()
    )

    weights_df.index = (

        weights_df.index
        .astype(str)
        .str.strip()
    )

    if "weight" not in weights_df.columns:

        if len(
            weights_df.columns
        ) == 1:

            weights_df.columns = [
                "weight"
            ]

        else:

            raise ValueError(
                "Unable to identify weight column."
            )

    weights = weights_df[
        "weight"
    ].astype(float)

    # Defensive re-normalization.

    weights = (
        weights
        /
        weights.sum()
    )

    print(
        f"Risk rows          : "
        f"{len(risk_data):,}"
    )

    print(
        f"Return matrix      : "
        f"{return_matrix.shape}"
    )

    print(
        f"Volatility records : "
        f"{len(volatility)}"
    )

    print(
        f"VaR records        : "
        f"{len(var_metrics)}"
    )

    print(
        f"Portfolio weights  : "
        f"{len(weights)}"
    )

    return {

        "risk_data":
            risk_data,

        "return_matrix":
            return_matrix,

        "volatility":
            volatility,

        "var_metrics":
            var_metrics,

        "weights":
            weights,

    }


# ============================================================
# RISK-FREE RATE
# ============================================================

def calculate_daily_risk_free_rate():

    # --------------------------------------------------------
    # Convert annual risk-free rate to equivalent daily rate.
    #
    # rf_daily =
    #     (1 + rf_annual)^(1/252) - 1
    # --------------------------------------------------------

    annual_rf = (
        cfg.ANNUAL_RISK_FREE_RATE
    )

    daily_rf = (

        (
            1.0
            +
            annual_rf
        )
        **
        (
            1.0
            /
            cfg.TRADING_DAYS_PER_YEAR
        )

        -
        1.0
    )

    return float(
        daily_rf
    )


# ============================================================
# STOCK-LEVEL SHARPE
# ============================================================

def calculate_asset_sharpe(
    risk_data,
):

    section(
        "CALCULATING STOCK-LEVEL SHARPE RATIOS"
    )

    daily_rf = (
        calculate_daily_risk_free_rate()
    )

    print(
        f"Annual risk-free rate : "
        f"{cfg.ANNUAL_RISK_FREE_RATE:.4%}"
    )

    print(
        f"Daily risk-free rate  : "
        f"{daily_rf:.8f}"
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

        returns = (

            stock[
                cfg.RETURN_COLUMN
            ]
            .dropna()
            .astype(float)
        )

        if len(returns) < cfg.MIN_HISTORY_DAYS:

            print(
                f"{symbol:<20} "
                f"SKIPPED - insufficient history"
            )

            continue

        # ----------------------------------------------------
        # Daily statistics
        # ----------------------------------------------------

        mean_daily_return = (
            returns.mean()
        )

        std_daily_return = (
            returns.std(
                ddof=1
            )
        )

        # ----------------------------------------------------
        # Annualized arithmetic return
        #
        # Used consistently with traditional annualized Sharpe.
        # ----------------------------------------------------

        annualized_return = (

            mean_daily_return
            *
            cfg.TRADING_DAYS_PER_YEAR
        )

        annualized_volatility = (

            std_daily_return
            *
            np.sqrt(
                cfg.TRADING_DAYS_PER_YEAR
            )
        )

        # ----------------------------------------------------
        # Sharpe
        #
        # SR =
        # ((mean daily return - rf daily)
        # / daily volatility) * sqrt(252)
        # ----------------------------------------------------

        if (
            np.isfinite(
                std_daily_return
            )
            and
            std_daily_return
            >
            0
        ):

            sharpe_ratio = (

                (
                    mean_daily_return
                    -
                    daily_rf
                )

                /
                std_daily_return

                *
                np.sqrt(
                    cfg.TRADING_DAYS_PER_YEAR
                )
            )

        else:

            sharpe_ratio = np.nan

        # ----------------------------------------------------
        # Geometric realized return
        #
        # Supporting diagnostic only.
        # ----------------------------------------------------

        cumulative_return = (

            np.prod(
                1.0
                +
                returns.values
            )

            -
            1.0
        )

        # Annualized geometric return.

        years = (

            len(returns)
            /
            cfg.TRADING_DAYS_PER_YEAR
        )

        if (
            years
            >
            0
            and
            (
                1.0
                +
                cumulative_return
            )
            >
            0
        ):

            geometric_annual_return = (

                (
                    1.0
                    +
                    cumulative_return
                )
                **
                (
                    1.0
                    /
                    years
                )

                -
                1.0
            )

        else:

            geometric_annual_return = np.nan

        latest = stock.iloc[-1]

        results.append({

            "symbol":
                symbol,

            "observations":
                int(
                    len(returns)
                ),

            "mean_daily_return":
                float(
                    mean_daily_return
                ),

            "std_daily_return":
                float(
                    std_daily_return
                ),

            "annualized_return_arithmetic":
                float(
                    annualized_return
                ),

            "annualized_return_geometric":
                float(
                    geometric_annual_return
                ),

            "annualized_volatility":
                float(
                    annualized_volatility
                ),

            "risk_free_rate_annual":
                float(
                    cfg.ANNUAL_RISK_FREE_RATE
                ),

            "sharpe_ratio":
                float(
                    sharpe_ratio
                ),

            "prediction_date":
                latest[
                    "prediction_date"
                ],

            "top30_probability":
                float(
                    latest[
                        "top30_probability"
                    ]
                ),

            "ensemble_std":
                float(
                    latest[
                        "ensemble_std"
                    ]
                ),

            "trend_class":
                latest[
                    "trend_class"
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
                "annualized_return_arithmetic",
                "annualized_volatility",
                "sharpe_ratio",
            ]
        ]
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# ============================================================
# PORTFOLIO RETURN SERIES
# ============================================================

def calculate_portfolio_returns(
    return_matrix,
    weights,
):

    section(
        "CALCULATING PORTFOLIO RETURN SERIES"
    )

    common_symbols = [

        symbol

        for symbol
        in weights.index

        if symbol
        in return_matrix.columns
    ]

    if not common_symbols:

        raise ValueError(
            "No common symbols between weights and returns."
        )

    returns = (

        return_matrix[
            common_symbols
        ]
        .dropna(
            how="any"
        )
        .copy()
    )

    aligned_weights = (

        weights
        .loc[
            common_symbols
        ]
        .copy()
    )

    aligned_weights = (

        aligned_weights
        /
        aligned_weights.sum()
    )

    # --------------------------------------------------------
    # Portfolio return:
    #
    # Rp,t = sum(w_i * R_i,t)
    # --------------------------------------------------------

    portfolio_returns = (

        returns
        .mul(
            aligned_weights,
            axis=1,
        )
        .sum(
            axis=1
        )
    )

    portfolio_returns.name = (
        "portfolio_return"
    )

    print(
        f"Common stocks        : "
        f"{len(common_symbols)}"
    )

    print(
        f"Return observations  : "
        f"{len(portfolio_returns)}"
    )

    print(
        f"Weight sum           : "
        f"{aligned_weights.sum():.10f}"
    )

    return (
        portfolio_returns,
        aligned_weights,
    )


# ============================================================
# PORTFOLIO SHARPE
# ============================================================

def calculate_portfolio_sharpe(
    portfolio_returns,
    weights,
):

    section(
        "CALCULATING PORTFOLIO SHARPE RATIO"
    )

    daily_rf = (
        calculate_daily_risk_free_rate()
    )

    mean_daily = (
        portfolio_returns.mean()
    )

    std_daily = (
        portfolio_returns.std(
            ddof=1
        )
    )

    if (
        not np.isfinite(
            std_daily
        )
        or
        std_daily
        <=
        0
    ):

        raise ValueError(
            "Portfolio daily volatility is invalid."
        )

    annualized_return = (

        mean_daily
        *
        cfg.TRADING_DAYS_PER_YEAR
    )

    annualized_volatility = (

        std_daily
        *
        np.sqrt(
            cfg.TRADING_DAYS_PER_YEAR
        )
    )

    sharpe_ratio = (

        (
            mean_daily
            -
            daily_rf
        )

        /
        std_daily

        *
        np.sqrt(
            cfg.TRADING_DAYS_PER_YEAR
        )
    )

    cumulative_return = (

        np.prod(
            1.0
            +
            portfolio_returns.values
        )

        -
        1.0
    )

    years = (

        len(
            portfolio_returns
        )
        /
        cfg.TRADING_DAYS_PER_YEAR
    )

    if (
        years
        >
        0
        and
        (
            1.0
            +
            cumulative_return
        )
        >
        0
    ):

        geometric_annual_return = (

            (
                1.0
                +
                cumulative_return
            )
            **
            (
                1.0
                /
                years
            )

            -
            1.0
        )

    else:

        geometric_annual_return = np.nan

    metrics = {

        "observations":
            int(
                len(
                    portfolio_returns
                )
            ),

        "number_of_assets":
            int(
                len(
                    weights
                )
            ),

        "weighting_method":
            str(
                cfg.INITIAL_WEIGHT_METHOD
            ),

        "risk_free_rate_annual":
            float(
                cfg.ANNUAL_RISK_FREE_RATE
            ),

        "risk_free_rate_daily":
            float(
                daily_rf
            ),

        "mean_daily_return":
            float(
                mean_daily
            ),

        "daily_volatility":
            float(
                std_daily
            ),

        "annualized_return_arithmetic":
            float(
                annualized_return
            ),

        "annualized_return_geometric":
            float(
                geometric_annual_return
            ),

        "annualized_volatility":
            float(
                annualized_volatility
            ),

        "sharpe_ratio":
            float(
                sharpe_ratio
            ),

        "cumulative_return":
            float(
                cumulative_return
            ),

    }

    print(
        f"Mean daily return          : "
        f"{mean_daily:.6f}"
    )

    print(
        f"Daily volatility           : "
        f"{std_daily:.6f}"
    )

    print(
        f"Annualized return          : "
        f"{annualized_return:.4f}"
    )

    print(
        f"Annualized volatility      : "
        f"{annualized_volatility:.4f}"
    )

    print(
        f"Portfolio Sharpe ratio     : "
        f"{sharpe_ratio:.4f}"
    )

    print(
        f"Cumulative realized return : "
        f"{cumulative_return:.4f}"
    )

    return metrics


# ============================================================
# MERGE PREVIOUS AGENT 3 METRICS
# ============================================================

def merge_risk_metrics(
    sharpe_metrics,
    var_metrics,
):

    section(
        "MERGING VOLATILITY + VaR + SHARPE METRICS"
    )

    # --------------------------------------------------------
    # var_metrics already contains the volatility columns
    # from the previous stages.
    # --------------------------------------------------------

    sharpe_columns = [

        "symbol",

        "mean_daily_return",

        "annualized_return_arithmetic",

        "annualized_return_geometric",

        "sharpe_ratio",

    ]

    final_metrics = (
        var_metrics.merge(

            sharpe_metrics[
                sharpe_columns
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

def validate_sharpe_results(
    sharpe_metrics,
    portfolio_metrics,
    final_metrics,
    weights,
):

    section(
        "VALIDATING SHARPE RESULTS"
    )

    failures = []

    # --------------------------------------------------------
    # Asset Sharpe ratios
    # --------------------------------------------------------

    asset_sharpe_valid = (

        sharpe_metrics[
            "sharpe_ratio"
        ]
        .notna()
        .all()

        and

        np.isfinite(

            sharpe_metrics[
                "sharpe_ratio"
            ]

        ).all()
    )

    print(
        f"Finite stock Sharpe ratios   : "
        f"{asset_sharpe_valid}"
    )

    if not asset_sharpe_valid:

        failures.append(
            "stock Sharpe ratios"
        )

    # --------------------------------------------------------
    # Portfolio Sharpe
    # --------------------------------------------------------

    portfolio_sharpe_valid = (

        np.isfinite(

            portfolio_metrics[
                "sharpe_ratio"
            ]
        )
    )

    print(
        f"Finite portfolio Sharpe      : "
        f"{portfolio_sharpe_valid}"
    )

    if not portfolio_sharpe_valid:

        failures.append(
            "portfolio Sharpe"
        )

    # --------------------------------------------------------
    # Portfolio volatility
    # --------------------------------------------------------

    portfolio_vol_valid = (

        portfolio_metrics[
            "annualized_volatility"
        ]
        >
        0

        and

        np.isfinite(

            portfolio_metrics[
                "annualized_volatility"
            ]
        )
    )

    print(
        f"Valid portfolio volatility   : "
        f"{portfolio_vol_valid}"
    )

    if not portfolio_vol_valid:

        failures.append(
            "portfolio volatility"
        )

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------

    weights_valid = np.isclose(

        weights.sum(),

        1.0,

        atol=1e-10,
    )

    print(
        f"Portfolio weights sum to 1   : "
        f"{weights_valid}"
    )

    if not weights_valid:

        failures.append(
            "portfolio weights"
        )

    # --------------------------------------------------------
    # Stock count
    # --------------------------------------------------------

    stock_count_valid = (

        len(
            sharpe_metrics
        )
        ==
        len(
            final_metrics
        )
        ==
        len(
            weights
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

    # --------------------------------------------------------
    # Main three risk metrics exist
    # --------------------------------------------------------

    required_final_columns = [

        "historical_volatility_annual",

        "individual_var_percent",

        "sharpe_ratio",

    ]

    metrics_present = all(

        column
        in final_metrics.columns

        for column
        in required_final_columns
    )

    print(
        f"All 3 core metrics present    : "
        f"{metrics_present}"
    )

    if not metrics_present:

        failures.append(
            "core risk metrics"
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
            "Sharpe validation failed."
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
    sharpe_metrics,
    portfolio_returns,
    portfolio_metrics,
    final_metrics,
):

    section(
        "SAVING SHARPE RESULTS"
    )

    sharpe_parquet = (

        cfg.PROCESSED_DIR
        / "sharpe_metrics.parquet"
    )

    sharpe_csv = (

        cfg.PROCESSED_DIR
        / "sharpe_metrics.csv"
    )

    combined_parquet = (

        cfg.PROCESSED_DIR
        / "combined_risk_metrics.parquet"
    )

    combined_csv = (

        cfg.PROCESSED_DIR
        / "combined_risk_metrics.csv"
    )

    portfolio_return_file = (

        cfg.PROCESSED_DIR
        / "portfolio_returns.parquet"
    )

    portfolio_json = (

        cfg.PROCESSED_DIR
        / "portfolio_sharpe.json"
    )

    # Stock Sharpe metrics

    sharpe_metrics.to_parquet(

        sharpe_parquet,

        index=False,
    )

    sharpe_metrics.to_csv(

        sharpe_csv,

        index=False,
    )

    # Combined Volatility + VaR + Sharpe

    final_metrics.to_parquet(

        combined_parquet,

        index=False,
    )

    final_metrics.to_csv(

        combined_csv,

        index=False,
    )

    # Portfolio returns

    portfolio_return_df = (

        portfolio_returns
        .rename(
            "portfolio_return"
        )
        .to_frame()
    )

    portfolio_return_df.to_parquet(
        portfolio_return_file
    )

    # Portfolio summary

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
        sharpe_parquet
    )

    print(
        sharpe_csv
    )

    print(
        combined_parquet
    )

    print(
        combined_csv
    )

    print(
        portfolio_return_file
    )

    print(
        portfolio_json
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_sharpe_calculator():

    section(
        "AGENT 3 - SHARPE RATIO"
    )

    print(
        "Paper-aligned metric:"
    )

    print()

    print(
        "Sharpe Ratio"
    )

    print()

    print(
        "SR = (Return - Risk-free Rate) / Volatility"
    )

    print()

    print(
        f"Annual risk-free rate : "
        f"{cfg.ANNUAL_RISK_FREE_RATE:.2%}"
    )

    print(
        f"Trading days/year     : "
        f"{cfg.TRADING_DAYS_PER_YEAR}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Risk-free rate and 252-day implementation "
        "are explicit implementation choices."
    )


    validate_input_files()


    inputs = (
        load_inputs()
    )


    asset_sharpe = (
        calculate_asset_sharpe(

            inputs[
                "risk_data"
            ]
        )
    )


    (
        portfolio_returns,
        aligned_weights,

    ) = calculate_portfolio_returns(

        inputs[
            "return_matrix"
        ],

        inputs[
            "weights"
        ],
    )


    portfolio_metrics = (
        calculate_portfolio_sharpe(

            portfolio_returns,

            aligned_weights,
        )
    )


    final_metrics = (
        merge_risk_metrics(

            asset_sharpe,

            inputs[
                "var_metrics"
            ],
        )
    )


    validate_sharpe_results(

        asset_sharpe,

        portfolio_metrics,

        final_metrics,

        aligned_weights,
    )


    save_results(

        asset_sharpe,

        portfolio_returns,

        portfolio_metrics,

        final_metrics,
    )


    section(
        "SHARPE RATIO COMPLETE"
    )

    print(
        f"Stocks evaluated : "
        f"{len(asset_sharpe)}"
    )

    print(
        f"Portfolio Sharpe : "
        f"{portfolio_metrics['sharpe_ratio']:.4f}"
    )

    print()

    best_stock = (

        asset_sharpe
        .sort_values(

            "sharpe_ratio",

            ascending=False,

        )
        .iloc[0]
    )

    worst_stock = (

        asset_sharpe
        .sort_values(

            "sharpe_ratio",

            ascending=True,

        )
        .iloc[0]
    )

    print(
        "Highest stock Sharpe:"
    )

    print(
        f"{best_stock['symbol']} "
        f"({best_stock['sharpe_ratio']:.4f})"
    )

    print()

    print(
        "Lowest stock Sharpe:"
    )

    print(
        f"{worst_stock['symbol']} "
        f"({worst_stock['sharpe_ratio']:.4f})"
    )

    print()

    print(
        "PAPER CORE RISK METRICS:"
    )

    print(
        "Historical Volatility : COMPLETE"
    )

    print(
        "Value at Risk         : COMPLETE"
    )

    print(
        "Sharpe Ratio          : COMPLETE"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Drawdown diagnostics + combined risk scoring"
    )

    return {

        "asset_sharpe":
            asset_sharpe,

        "portfolio_returns":
            portfolio_returns,

        "portfolio_metrics":
            portfolio_metrics,

        "combined_risk_metrics":
            final_metrics,

        "weights":
            aligned_weights,

    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_sharpe_calculator()