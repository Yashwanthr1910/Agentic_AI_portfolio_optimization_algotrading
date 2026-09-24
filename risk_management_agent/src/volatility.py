"""
Agent 3 - Historical Volatility Calculator
===============================================================

Purpose
-------
Calculate historical volatility for each Agent-2-selected stock
and calculate portfolio-level volatility.

Paper alignment
---------------
The base paper states that the Risk Management Agent analyses:

    - Historical Volatility
    - Value at Risk
    - Sharpe Ratio

Historical volatility represents the variability of historical
returns.

Asset volatility:

    sigma_i = std(R_i)

Annualized asset volatility:

    sigma_i,annual =
        std(R_i) * sqrt(252)

Portfolio volatility:

    sigma_p =
        sqrt(w.T @ Sigma @ w)

where:

    w     = portfolio weight vector
    Sigma = covariance matrix

Because Agent 5 has not yet performed final portfolio optimization,
Agent 3 uses equal weights as a neutral provisional portfolio.

IMPORTANT
---------
The exact volatility lookback windows are implementation choices.
The paper does not clearly specify 20/60/252-day windows.
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
# LOAD DATA
# ============================================================

def load_risk_data():

    section(
        "LOADING AGENT 3 RISK DATA"
    )

    if not cfg.RISK_INPUT_FILE.exists():

        raise FileNotFoundError(

            "Risk input file not found:\n"
            f"{cfg.RISK_INPUT_FILE}\n\n"
            "Run data_loader.py first."
        )

    if not cfg.RETURN_MATRIX_FILE.exists():

        raise FileNotFoundError(

            "Return matrix file not found:\n"
            f"{cfg.RETURN_MATRIX_FILE}\n\n"
            "Run data_loader.py first."
        )

    risk_data = pd.read_parquet(
        cfg.RISK_INPUT_FILE
    )

    return_matrix = pd.read_parquet(
        cfg.RETURN_MATRIX_FILE
    )

    return_matrix.index = pd.to_datetime(
        return_matrix.index
    )

    print(
        f"Risk rows       : "
        f"{len(risk_data):,}"
    )

    print(
        f"Stocks          : "
        f"{risk_data['symbol'].nunique()}"
    )

    print(
        f"Return matrix   : "
        f"{return_matrix.shape}"
    )

    return (
        risk_data,
        return_matrix,
    )


# ============================================================
# CALCULATE ASSET VOLATILITY
# ============================================================

def calculate_asset_volatility(
    risk_data,
):

    section(
        "CALCULATING HISTORICAL STOCK VOLATILITY"
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
                f"SKIPPED - "
                f"insufficient return history"
            )

            continue

        row = {

            "symbol":
                symbol,

            "observations":
                len(returns),

        }

        # ----------------------------------------------------
        # MULTIPLE HISTORICAL VOLATILITY WINDOWS
        # ----------------------------------------------------

        for window in cfg.VOLATILITY_WINDOWS:

            if len(returns) >= window:

                sample = returns.tail(
                    window
                )

                daily_vol = sample.std(
                    ddof=1
                )

                annual_vol = (

                    daily_vol
                    *
                    np.sqrt(
                        cfg.TRADING_DAYS_PER_YEAR
                    )

                )

                row[
                    f"volatility_{window}d_daily"
                ] = float(
                    daily_vol
                )

                row[
                    f"volatility_{window}d_annual"
                ] = float(
                    annual_vol
                )

            else:

                row[
                    f"volatility_{window}d_daily"
                ] = np.nan

                row[
                    f"volatility_{window}d_annual"
                ] = np.nan

        # ----------------------------------------------------
        # FULL LOOKBACK VOLATILITY
        # ----------------------------------------------------

        full_daily_vol = returns.std(
            ddof=1
        )

        full_annual_vol = (

            full_daily_vol
            *
            np.sqrt(
                cfg.TRADING_DAYS_PER_YEAR
            )
        )

        row[
            "historical_volatility_daily"
        ] = float(
            full_daily_vol
        )

        row[
            "historical_volatility_annual"
        ] = float(
            full_annual_vol
        )

        # ----------------------------------------------------
        # AGENT 2 INFORMATION
        # ----------------------------------------------------

        latest = stock.iloc[-1]

        row[
            "prediction_date"
        ] = latest[
            "prediction_date"
        ]

        row[
            "top30_probability"
        ] = float(
            latest[
                "top30_probability"
            ]
        )

        row[
            "ensemble_std"
        ] = float(
            latest[
                "ensemble_std"
            ]
        )

        row[
            "trend_class"
        ] = latest[
            "trend_class"
        ]

        row[
            "agent2_rank"
        ] = int(
            latest[
                "agent2_rank"
            ]
        )

        results.append(
            row
        )

    volatility = pd.DataFrame(
        results
    )

    volatility = (

        volatility
        .sort_values(
            "agent2_rank"
        )
        .reset_index(
            drop=True
        )
    )

    print()

    display_columns = [

        "agent2_rank",

        "symbol",

        "volatility_20d_annual",

        "volatility_60d_annual",

        "volatility_252d_annual",

    ]

    print(
        volatility[
            display_columns
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return volatility


# ============================================================
# CALCULATE COVARIANCE MATRIX
# ============================================================

def calculate_covariance_matrix(
    return_matrix,
):

    section(
        "CALCULATING RETURN COVARIANCE MATRIX"
    )

    # --------------------------------------------------------
    # We require overlapping return observations for covariance.
    #
    # Because all 10 current stocks contain the same 252 return
    # dates, this should retain the full matrix.
    # --------------------------------------------------------

    clean_returns = (
        return_matrix
        .dropna(
            how="any"
        )
        .copy()
    )

    if clean_returns.empty:

        raise ValueError(
            "No overlapping return observations available."
        )

    covariance_daily = (
        clean_returns
        .cov()
    )

    covariance_annual = (

        covariance_daily
        *
        cfg.TRADING_DAYS_PER_YEAR
    )

    print(
        f"Common return observations : "
        f"{len(clean_returns)}"
    )

    print(
        f"Covariance matrix shape     : "
        f"{covariance_daily.shape}"
    )

    print()

    print(
        "Daily covariance matrix:"
    )

    print()

    print(
        covariance_daily
        .round(
            6
        )
        .to_string()
    )

    return (
        covariance_daily,
        covariance_annual,
        clean_returns,
    )


# ============================================================
# BUILD PROVISIONAL PORTFOLIO WEIGHTS
# ============================================================

def build_equal_weights(
    symbols,
):

    section(
        "BUILDING PROVISIONAL PORTFOLIO WEIGHTS"
    )

    symbols = list(
        symbols
    )

    n_assets = len(
        symbols
    )

    if n_assets == 0:

        raise ValueError(
            "Cannot create portfolio with zero assets."
        )

    weight = (
        1.0
        /
        n_assets
    )

    weights = pd.Series(

        weight,

        index=symbols,

        name="weight",
    )

    print(
        f"Weighting method : "
        f"{cfg.INITIAL_WEIGHT_METHOD}"
    )

    print(
        f"Assets           : "
        f"{n_assets}"
    )

    print(
        f"Weight each      : "
        f"{weight:.4f}"
    )

    print(
        f"Weight sum       : "
        f"{weights.sum():.6f}"
    )

    print()

    print(
        weights
        .to_string()
    )

    return weights


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def calculate_portfolio_volatility(
    covariance_daily,
    covariance_annual,
    weights,
):

    section(
        "CALCULATING PORTFOLIO VOLATILITY"
    )

    symbols = list(
        weights.index
    )

    covariance_daily = covariance_daily.loc[
        symbols,
        symbols,
    ]

    covariance_annual = covariance_annual.loc[
        symbols,
        symbols,
    ]

    w = weights.values.reshape(
        -1,
        1,
    )

    # --------------------------------------------------------
    # Portfolio variance:
    #
    #     sigma^2_p = w.T Sigma w
    # --------------------------------------------------------

    variance_daily = float(

        (
            w.T
            @ covariance_daily.values
            @ w
        )[0, 0]
    )

    variance_annual = float(

        (
            w.T
            @ covariance_annual.values
            @ w
        )[0, 0]
    )

    # Numerical protection.

    variance_daily = max(
        variance_daily,
        0.0,
    )

    variance_annual = max(
        variance_annual,
        0.0,
    )

    volatility_daily = np.sqrt(
        variance_daily
    )

    volatility_annual = np.sqrt(
        variance_annual
    )

    print(
        f"Daily portfolio variance    : "
        f"{variance_daily:.10f}"
    )

    print(
        f"Daily portfolio volatility  : "
        f"{volatility_daily:.6f}"
    )

    print(
        f"Annual portfolio variance   : "
        f"{variance_annual:.10f}"
    )

    print(
        f"Annual portfolio volatility : "
        f"{volatility_annual:.6f}"
    )

    return {

        "portfolio_variance_daily":
            variance_daily,

        "portfolio_volatility_daily":
            float(
                volatility_daily
            ),

        "portfolio_variance_annual":
            variance_annual,

        "portfolio_volatility_annual":
            float(
                volatility_annual
            ),

    }


# ============================================================
# VALIDATION
# ============================================================

def validate_volatility_results(
    volatility,
    covariance_daily,
    weights,
    portfolio_metrics,
):

    section(
        "VALIDATING VOLATILITY RESULTS"
    )

    failures = []

    # --------------------------------------------------------
    # Asset volatility
    # --------------------------------------------------------

    asset_vol_valid = (

        volatility[
            "historical_volatility_annual"
        ]
        .notna()
        .all()

        and

        np.isfinite(

            volatility[
                "historical_volatility_annual"
            ]

        ).all()

        and

        (
            volatility[
                "historical_volatility_annual"
            ]
            >=
            0
        ).all()
    )

    print(
        f"Valid asset volatility      : "
        f"{asset_vol_valid}"
    )

    if not asset_vol_valid:

        failures.append(
            "asset volatility"
        )

    # --------------------------------------------------------
    # Covariance matrix
    # --------------------------------------------------------

    covariance_finite = (

        np.isfinite(
            covariance_daily.values
        )
        .all()
    )

    print(
        f"Finite covariance matrix    : "
        f"{covariance_finite}"
    )

    if not covariance_finite:

        failures.append(
            "covariance matrix"
        )

    # --------------------------------------------------------
    # Symmetry
    # --------------------------------------------------------

    covariance_symmetric = (

        np.allclose(

            covariance_daily.values,

            covariance_daily.values.T,

            atol=1e-12,
        )
    )

    print(
        f"Covariance symmetric        : "
        f"{covariance_symmetric}"
    )

    if not covariance_symmetric:

        failures.append(
            "covariance symmetry"
        )

    # --------------------------------------------------------
    # Weight sum
    # --------------------------------------------------------

    weights_valid = np.isclose(

        weights.sum(),

        1.0,

        atol=1e-10,
    )

    print(
        f"Portfolio weights sum to 1  : "
        f"{weights_valid}"
    )

    if not weights_valid:

        failures.append(
            "portfolio weights"
        )

    # --------------------------------------------------------
    # Portfolio volatility
    # --------------------------------------------------------

    portfolio_vol_valid = (

        np.isfinite(

            portfolio_metrics[
                "portfolio_volatility_annual"
            ]

        )

        and

        portfolio_metrics[
            "portfolio_volatility_annual"
        ]
        >=
        0
    )

    print(
        f"Valid portfolio volatility  : "
        f"{portfolio_vol_valid}"
    )

    if not portfolio_vol_valid:

        failures.append(
            "portfolio volatility"
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
            "Volatility validation failed."
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
    volatility,
    covariance_daily,
    covariance_annual,
    weights,
):

    section(
        "SAVING VOLATILITY RESULTS"
    )

    volatility_file = (

        cfg.PROCESSED_DIR
        / "volatility_metrics.parquet"
    )

    volatility_csv = (

        cfg.PROCESSED_DIR
        / "volatility_metrics.csv"
    )

    covariance_daily_file = (

        cfg.PROCESSED_DIR
        / "covariance_matrix_daily.csv"
    )

    covariance_annual_file = (

        cfg.PROCESSED_DIR
        / "covariance_matrix_annual.csv"
    )

    weights_file = (

        cfg.PROCESSED_DIR
        / "provisional_weights.csv"
    )

    volatility.to_parquet(

        volatility_file,

        index=False,
    )

    volatility.to_csv(

        volatility_csv,

        index=False,
    )

    covariance_daily.to_csv(
        covariance_daily_file
    )

    covariance_annual.to_csv(
        covariance_annual_file
    )

    weights.to_csv(

        weights_file,

        header=True,
    )

    print(
        volatility_file
    )

    print(
        volatility_csv
    )

    print(
        covariance_daily_file
    )

    print(
        covariance_annual_file
    )

    print(
        weights_file
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_volatility_calculator():

    section(
        "AGENT 3 - HISTORICAL VOLATILITY"
    )

    print(
        "Paper-aligned metric:"
    )

    print(
        "Historical Volatility"
    )

    print()

    print(
        "Asset volatility:"
    )

    print(
        "sigma_i = std(daily returns)"
    )

    print()

    print(
        "Portfolio volatility:"
    )

    print(
        "sigma_p = sqrt(w.T @ Sigma @ w)"
    )


    risk_data, return_matrix = (
        load_risk_data()
    )


    volatility = (
        calculate_asset_volatility(
            risk_data
        )
    )


    (
        covariance_daily,
        covariance_annual,
        clean_returns,

    ) = calculate_covariance_matrix(
        return_matrix
    )


    weights = build_equal_weights(

        covariance_daily.columns
    )


    portfolio_metrics = (
        calculate_portfolio_volatility(

            covariance_daily,

            covariance_annual,

            weights,
        )
    )


    validate_volatility_results(

        volatility,

        covariance_daily,

        weights,

        portfolio_metrics,
    )


    save_results(

        volatility,

        covariance_daily,

        covariance_annual,

        weights,
    )


    section(
        "HISTORICAL VOLATILITY COMPLETE"
    )

    print(
        f"Stocks evaluated : "
        f"{len(volatility)}"
    )

    print(
        f"Common dates     : "
        f"{len(clean_returns)}"
    )

    print(
        f"Portfolio annual volatility : "
        f"{portfolio_metrics['portfolio_volatility_annual']:.4f}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Paper-style portfolio Value at Risk (VaR)"
    )

    return {

        "asset_volatility":
            volatility,

        "covariance_daily":
            covariance_daily,

        "covariance_annual":
            covariance_annual,

        "weights":
            weights,

        "portfolio_metrics":
            portfolio_metrics,

    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_volatility_calculator()