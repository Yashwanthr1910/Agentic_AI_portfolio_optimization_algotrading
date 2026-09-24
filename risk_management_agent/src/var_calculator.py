"""
Agent 3 - Value at Risk Calculator
===============================================================

Purpose
-------
Calculate paper-style parametric portfolio Value at Risk (VaR)
using:

    VaR = z * sqrt(w.T @ Sigma @ w)

where:

    z       = normal-distribution confidence multiplier
    w       = portfolio weight vector
    Sigma   = daily return covariance matrix

Paper alignment
---------------
The base paper explicitly gives the portfolio VaR structure using
portfolio weights and the covariance matrix.

This implementation therefore uses parametric variance-covariance
VaR rather than historical percentile VaR.

Additional diagnostics
----------------------
To make the Agent useful downstream, this file also calculates:

1. Individual asset VaR
2. Marginal contribution to portfolio volatility
3. Component contribution to portfolio volatility
4. Component contribution to portfolio VaR
5. Percentage contribution to portfolio risk

These contribution metrics are implementation enhancements and are
not claimed as exact paper components.

IMPORTANT
---------
The 95% confidence level is an implementation choice because the
paper does not clearly specify the exact confidence level.
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
# INPUT FILES
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

        "covariance_daily":
            cfg.PROCESSED_DIR
            / "covariance_matrix_daily.csv",

        "provisional_weights":
            cfg.PROCESSED_DIR
            / "provisional_weights.csv",

    }


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_input_files():

    section(
        "VALIDATING VaR INPUT FILES"
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

            "VaR calculation cannot continue. "
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
        "LOADING VaR INPUTS"
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

    covariance_daily = pd.read_csv(

        cfg.PROCESSED_DIR
        / "covariance_matrix_daily.csv",

        index_col=0,
    )

    weights_df = pd.read_csv(

        cfg.PROCESSED_DIR
        / "provisional_weights.csv",

        index_col=0,
    )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    return_matrix.columns = (

        return_matrix.columns
        .astype(str)
        .str.strip()
    )

    covariance_daily.index = (

        covariance_daily.index
        .astype(str)
        .str.strip()
    )

    covariance_daily.columns = (

        covariance_daily.columns
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
                "Unable to identify portfolio weight column."
            )

    weights = weights_df[
        "weight"
    ].astype(float)

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
        f"Covariance matrix  : "
        f"{covariance_daily.shape}"
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

        "covariance_daily":
            covariance_daily,

        "weights":
            weights,

    }


# ============================================================
# ALIGN PORTFOLIO INPUTS
# ============================================================

def align_portfolio_inputs(
    covariance,
    weights,
):

    section(
        "ALIGNING PORTFOLIO INPUTS"
    )

    covariance_symbols = set(
        covariance.columns
    )

    weight_symbols = set(
        weights.index
    )

    common_symbols = sorted(

        covariance_symbols
        &
        weight_symbols
    )

    if not common_symbols:

        raise ValueError(
            "No common symbols between covariance and weights."
        )

    if covariance_symbols != weight_symbols:

        print(
            "WARNING:"
        )

        print(
            "Covariance and portfolio symbols "
            "are not identical."
        )

        print()

        print(
            "Using common symbols only."
        )

    covariance = covariance.loc[
        common_symbols,
        common_symbols,
    ]

    weights = weights.loc[
        common_symbols
    ].copy()

    # --------------------------------------------------------
    # Re-normalize defensively.
    # --------------------------------------------------------

    weight_sum = weights.sum()

    if weight_sum <= 0:

        raise ValueError(
            "Portfolio weights must sum to a positive value."
        )

    weights = (

        weights
        /
        weight_sum
    )

    print(
        f"Aligned assets : "
        f"{len(common_symbols)}"
    )

    print(
        f"Weight sum     : "
        f"{weights.sum():.10f}"
    )

    return (
        covariance,
        weights,
    )


# ============================================================
# PORTFOLIO VOLATILITY
# ============================================================

def calculate_daily_portfolio_volatility(
    covariance,
    weights,
):

    w = weights.values.reshape(
        -1,
        1,
    )

    variance = float(

        (
            w.T
            @ covariance.values
            @ w
        )[0, 0]
    )

    variance = max(
        variance,
        0.0,
    )

    volatility = np.sqrt(
        variance
    )

    return (
        variance,
        float(
            volatility
        ),
    )


# ============================================================
# PORTFOLIO VaR
# ============================================================

def calculate_portfolio_var(
    covariance,
    weights,
):

    section(
        "CALCULATING PAPER-STYLE PORTFOLIO VaR"
    )

    variance, daily_volatility = (
        calculate_daily_portfolio_volatility(

            covariance,

            weights,
        )
    )

    z = cfg.VAR_Z_SCORE

    horizon = (
        cfg.VAR_HORIZON_DAYS
    )

    # --------------------------------------------------------
    # Paper-style:
    #
    # VaR = z * sqrt(w.T Sigma w)
    #
    # For multi-day horizon:
    #
    # VaR_h = z * sigma_p * sqrt(h)
    # --------------------------------------------------------

    var_fraction = (

        z

        *
        daily_volatility

        *
        np.sqrt(
            horizon
        )
    )

    var_percent = (

        var_fraction
        *
        100.0
    )

    print(
        f"Confidence level        : "
        f"{cfg.VAR_CONFIDENCE_LEVEL:.2%}"
    )

    print(
        f"Z-score                 : "
        f"{z:.6f}"
    )

    print(
        f"VaR horizon             : "
        f"{horizon} trading day(s)"
    )

    print(
        f"Daily portfolio variance: "
        f"{variance:.10f}"
    )

    print(
        f"Daily portfolio sigma   : "
        f"{daily_volatility:.6f}"
    )

    print()

    print(
        f"Portfolio VaR fraction  : "
        f"{var_fraction:.6f}"
    )

    print(
        f"Portfolio VaR percent   : "
        f"{var_percent:.4f}%"
    )

    return {

        "confidence_level":
            float(
                cfg.VAR_CONFIDENCE_LEVEL
            ),

        "z_score":
            float(
                z
            ),

        "horizon_days":
            int(
                horizon
            ),

        "portfolio_variance_daily":
            float(
                variance
            ),

        "portfolio_volatility_daily":
            float(
                daily_volatility
            ),

        "portfolio_var_fraction":
            float(
                var_fraction
            ),

        "portfolio_var_percent":
            float(
                var_percent
            ),

    }


# ============================================================
# ASSET-LEVEL VaR
# ============================================================

def calculate_asset_var(
    volatility,
):

    section(
        "CALCULATING INDIVIDUAL STOCK VaR"
    )

    data = volatility.copy()

    required_column = (
        "historical_volatility_daily"
    )

    if required_column not in data.columns:

        raise ValueError(

            f"Missing column: "
            f"{required_column}"
        )

    # --------------------------------------------------------
    # Individual Gaussian VaR:
    #
    # VaR_i = z * sigma_i * sqrt(h)
    #
    # This is a diagnostic extension.
    # --------------------------------------------------------

    data[
        "individual_var_fraction"
    ] = (

        cfg.VAR_Z_SCORE

        *
        data[
            required_column
        ]

        *
        np.sqrt(
            cfg.VAR_HORIZON_DAYS
        )
    )

    data[
        "individual_var_percent"
    ] = (

        data[
            "individual_var_fraction"
        ]

        *
        100.0
    )

    display_columns = [

        "agent2_rank",

        "symbol",

        "top30_probability",

        "historical_volatility_daily",

        "individual_var_percent",

    ]

    print(
        data[
            display_columns
        ]
        .sort_values(
            "agent2_rank"
        )
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    return data


# ============================================================
# RISK CONTRIBUTIONS
# ============================================================

def calculate_risk_contributions(
    covariance,
    weights,
    portfolio_var,
):

    section(
        "CALCULATING PORTFOLIO RISK CONTRIBUTIONS"
    )

    symbols = list(
        weights.index
    )

    sigma = (
        covariance
        .loc[
            symbols,
            symbols,
        ]
        .values
    )

    w = (
        weights
        .loc[
            symbols
        ]
        .values
    )

    portfolio_sigma = (

        portfolio_var[
            "portfolio_volatility_daily"
        ]
    )

    if portfolio_sigma <= 0:

        raise ValueError(
            "Portfolio volatility must be positive."
        )

    # --------------------------------------------------------
    # Sigma @ w
    # --------------------------------------------------------

    covariance_times_weights = (

        sigma
        @
        w
    )

    # --------------------------------------------------------
    # Marginal contribution to volatility:
    #
    # MRC_i = (Sigma w)_i / sigma_p
    # --------------------------------------------------------

    marginal_contribution = (

        covariance_times_weights
        /
        portfolio_sigma
    )

    # --------------------------------------------------------
    # Component contribution:
    #
    # CRC_i = w_i * MRC_i
    #
    # Sum(CRC_i) = portfolio volatility
    # --------------------------------------------------------

    component_volatility = (

        w
        *
        marginal_contribution
    )

    # --------------------------------------------------------
    # Percentage contribution:
    #
    # PCR_i = CRC_i / sigma_p
    # --------------------------------------------------------

    percentage_contribution = (

        component_volatility
        /
        portfolio_sigma
    )

    # --------------------------------------------------------
    # Component VaR.
    #
    # Because parametric VaR is proportional to sigma_p:
    #
    # ComponentVaR_i =
    #       percentage contribution * total VaR
    # --------------------------------------------------------

    component_var_fraction = (

        percentage_contribution

        *
        portfolio_var[
            "portfolio_var_fraction"
        ]
    )

    component_var_percent = (

        component_var_fraction
        *
        100.0
    )

    results = pd.DataFrame({

        "symbol":
            symbols,

        "weight":
            w,

        "marginal_volatility":
            marginal_contribution,

        "component_volatility":
            component_volatility,

        "risk_contribution_fraction":
            percentage_contribution,

        "risk_contribution_percent":
            percentage_contribution
            *
            100.0,

        "component_var_fraction":
            component_var_fraction,

        "component_var_percent":
            component_var_percent,

    })

    results[
        "risk_contribution_rank"
    ] = (

        results[
            "risk_contribution_fraction"
        ]
        .rank(

            method="dense",

            ascending=False,
        )
        .astype(int)
    )

    results = (

        results
        .sort_values(
            "risk_contribution_rank"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        results[
            [
                "risk_contribution_rank",
                "symbol",
                "weight",
                "risk_contribution_percent",
                "component_var_percent",
            ]
        ]
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    return results


# ============================================================
# MERGE STOCK METRICS
# ============================================================

def merge_asset_risk_metrics(
    asset_var,
    contributions,
):

    section(
        "MERGING STOCK RISK METRICS"
    )

    result = asset_var.merge(

        contributions,

        on="symbol",

        how="left",

        validate="one_to_one",
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

    print(
        f"Final stock risk rows : "
        f"{len(result)}"
    )

    return result


# ============================================================
# VALIDATE
# ============================================================

def validate_var_results(
    portfolio_var,
    asset_metrics,
    contributions,
    weights,
):

    section(
        "VALIDATING VaR RESULTS"
    )

    failures = []

    # --------------------------------------------------------
    # Portfolio VaR
    # --------------------------------------------------------

    portfolio_var_valid = (

        np.isfinite(

            portfolio_var[
                "portfolio_var_fraction"
            ]
        )

        and

        portfolio_var[
            "portfolio_var_fraction"
        ]
        >=
        0
    )

    print(
        f"Valid portfolio VaR          : "
        f"{portfolio_var_valid}"
    )

    if not portfolio_var_valid:

        failures.append(
            "portfolio VaR"
        )

    # --------------------------------------------------------
    # Individual VaR
    # --------------------------------------------------------

    asset_var_valid = (

        asset_metrics[
            "individual_var_fraction"
        ]
        .notna()
        .all()

        and

        np.isfinite(

            asset_metrics[
                "individual_var_fraction"
            ]

        ).all()

        and

        (
            asset_metrics[
                "individual_var_fraction"
            ]
            >=
            0
        ).all()
    )

    print(
        f"Valid individual VaR         : "
        f"{asset_var_valid}"
    )

    if not asset_var_valid:

        failures.append(
            "individual VaR"
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
        f"Portfolio weights sum to 1   : "
        f"{weights_valid}"
    )

    if not weights_valid:

        failures.append(
            "weights"
        )

    # --------------------------------------------------------
    # Risk contribution sum
    # --------------------------------------------------------

    contribution_sum = (

        contributions[
            "risk_contribution_fraction"
        ]
        .sum()
    )

    contribution_valid = np.isclose(

        contribution_sum,

        1.0,

        atol=1e-8,
    )

    print(
        f"Risk contributions sum to 1  : "
        f"{contribution_valid}"
    )

    print(
        f"Risk contribution sum        : "
        f"{contribution_sum:.10f}"
    )

    if not contribution_valid:

        failures.append(
            "risk contributions"
        )

    # --------------------------------------------------------
    # Component VaR decomposition
    # --------------------------------------------------------

    component_var_sum = (

        contributions[
            "component_var_fraction"
        ]
        .sum()
    )

    component_var_valid = np.isclose(

        component_var_sum,

        portfolio_var[
            "portfolio_var_fraction"
        ],

        atol=1e-8,
    )

    print(
        f"Component VaR sums to VaR    : "
        f"{component_var_valid}"
    )

    print(
        f"Component VaR sum            : "
        f"{component_var_sum:.10f}"
    )

    if not component_var_valid:

        failures.append(
            "component VaR decomposition"
        )

    # --------------------------------------------------------
    # Expected stock count
    # --------------------------------------------------------

    stock_count_valid = (

        len(asset_metrics)
        ==
        len(weights)
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

        print()

        for failure in failures:

            print(
                f"  - {failure}"
            )

        raise ValueError(
            "VaR validation failed."
        )

    print()

    print(
        "VALIDATION RESULT: PASS"
    )

    return True


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    asset_metrics,
    contributions,
    portfolio_var,
):

    section(
        "SAVING VaR RESULTS"
    )

    asset_parquet = (

        cfg.PROCESSED_DIR
        / "var_metrics.parquet"
    )

    asset_csv = (

        cfg.PROCESSED_DIR
        / "var_metrics.csv"
    )

    contribution_csv = (

        cfg.PROCESSED_DIR
        / "risk_contributions.csv"
    )

    portfolio_json = (

        cfg.PROCESSED_DIR
        / "portfolio_var.json"
    )

    asset_metrics.to_parquet(

        asset_parquet,

        index=False,
    )

    asset_metrics.to_csv(

        asset_csv,

        index=False,
    )

    contributions.to_csv(

        contribution_csv,

        index=False,
    )

    with open(

        portfolio_json,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            portfolio_var,

            file,

            indent=4,
        )

    print(
        asset_parquet
    )

    print(
        asset_csv
    )

    print(
        contribution_csv
    )

    print(
        portfolio_json
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_var_calculator():

    section(
        "AGENT 3 - VALUE AT RISK"
    )

    print(
        "Paper-style equation:"
    )

    print()

    print(
        "VaR = z * sqrt(w.T @ Sigma @ w)"
    )

    print()

    print(
        f"Confidence level : "
        f"{cfg.VAR_CONFIDENCE_LEVEL:.2%}"
    )

    print(
        f"Z-score          : "
        f"{cfg.VAR_Z_SCORE:.6f}"
    )

    print(
        f"Horizon          : "
        f"{cfg.VAR_HORIZON_DAYS} day(s)"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Confidence level is an explicit "
        "implementation choice."
    )


    validate_input_files()


    inputs = load_inputs()


    covariance, weights = (
        align_portfolio_inputs(

            inputs[
                "covariance_daily"
            ],

            inputs[
                "weights"
            ],
        )
    )


    portfolio_var = (
        calculate_portfolio_var(

            covariance,

            weights,
        )
    )


    asset_var = (
        calculate_asset_var(

            inputs[
                "volatility"
            ]
        )
    )


    contributions = (
        calculate_risk_contributions(

            covariance,

            weights,

            portfolio_var,
        )
    )


    asset_metrics = (
        merge_asset_risk_metrics(

            asset_var,

            contributions,
        )
    )


    validate_var_results(

        portfolio_var,

        asset_metrics,

        contributions,

        weights,
    )


    save_results(

        asset_metrics,

        contributions,

        portfolio_var,
    )


    section(
        "VALUE AT RISK COMPLETE"
    )

    print(
        f"Stocks evaluated : "
        f"{len(asset_metrics)}"
    )

    print(
        f"Portfolio VaR    : "
        f"{portfolio_var['portfolio_var_percent']:.4f}%"
    )

    print()

    highest_risk = (

        contributions
        .sort_values(
            "risk_contribution_fraction",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        "Highest portfolio risk contributor:"
    )

    print(
        f"{highest_risk['symbol']} "
        f"({highest_risk['risk_contribution_percent']:.2f}%)"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Sharpe Ratio calculation"
    )

    return {

        "portfolio_var":
            portfolio_var,

        "asset_metrics":
            asset_metrics,

        "risk_contributions":
            contributions,

        "weights":
            weights,

    }


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_var_calculator()