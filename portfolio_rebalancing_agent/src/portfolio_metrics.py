"""
Agent 5 - Portfolio Metrics
===============================================================================

PROJECT
-------
Agentic AI Portfolio Optimization and Algorithmic Trading

AGENT
-----
Agent 5 - Portfolio Rebalancing Agent

PURPOSE
-------
Provide reusable portfolio-level mathematical calculations for:

    - Markowitz MPT optimizer
    - RL feedback
    - Portfolio rebalancing agent
    - Portfolio evaluator

Core calculations
-----------------
1. Portfolio expected return
2. Portfolio variance
3. Portfolio volatility
4. Portfolio Sharpe ratio
5. Daily portfolio returns
6. Cumulative return
7. Annualized return
8. Annualized volatility
9. Maximum drawdown
10. Historical Value at Risk
11. Turnover
12. Complete portfolio-performance summary

REFERENCE-PAPER ALIGNMENT
-------------------------
The paper defines portfolio expected return using weighted asset returns
and portfolio variance using covariance between assets.

The paper also evaluates performance using:

    - cumulative return
    - annual return
    - annual volatility
    - maximum drawdown
    - Sharpe ratio
    - daily VaR
    - turnover

IMPORTANT
---------
This file does NOT optimize portfolio weights.

Weight optimization is performed later by:

    mpt_optimizer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# IMPORT PATH
# =============================================================================

AGENT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(AGENT_ROOT),
    )


from config import config as cfg


# =============================================================================
# WEIGHT VALIDATION
# =============================================================================

def validate_weights(
    weights: np.ndarray,
    expected_size: int | None = None,
    require_sum_to_one: bool = True,
) -> np.ndarray:
    """
    Validate a portfolio-weight vector.

    Parameters
    ----------
    weights:
        Portfolio weights.

    expected_size:
        Expected number of assets.

    require_sum_to_one:
        If True, weights must sum approximately to one.

    Returns
    -------
    np.ndarray
        Validated one-dimensional float weight vector.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    ).reshape(
        -1
    )

    if weights.size == 0:

        raise ValueError(
            "Portfolio weight vector is empty."
        )

    if expected_size is not None:

        if weights.size != expected_size:

            raise ValueError(
                "Portfolio weight size mismatch.\n"
                f"Expected: {expected_size}\n"
                f"Received: {weights.size}"
            )

    if not np.isfinite(
        weights
    ).all():

        raise ValueError(
            "Portfolio weights contain NaN or infinity."
        )

    if not cfg.ALLOW_SHORT_SELLING:

        if np.any(
            weights
            <
            (
                cfg.MIN_WEIGHT
                -
                cfg.FLOAT_TOLERANCE
            )
        ):

            raise ValueError(
                "Negative portfolio weight detected "
                "while short selling is disabled."
            )

    if np.any(
        weights
        >
        (
            cfg.MAX_WEIGHT
            +
            cfg.FLOAT_TOLERANCE
        )
    ):

        raise ValueError(
            "Portfolio weight exceeds MAX_WEIGHT."
        )

    if require_sum_to_one:

        weight_sum = float(
            np.sum(
                weights
            )
        )

        if not np.isclose(
            weight_sum,
            cfg.WEIGHT_SUM_TARGET,
            atol=cfg.WEIGHT_SUM_TOLERANCE,
        ):

            raise ValueError(
                "Portfolio weights do not sum to target.\n"
                f"Weight sum: {weight_sum:.12f}\n"
                f"Required  : {cfg.WEIGHT_SUM_TARGET}"
            )

    return weights


# =============================================================================
# EXPECTED RETURN
# =============================================================================

def portfolio_expected_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
) -> float:
    """
    Calculate portfolio expected return.

    Markowitz equation:

        Rp = Σ wi * Ri

    where:

        wi = portfolio weight of asset i
        Ri = expected return of asset i
    """

    expected_returns = np.asarray(
        expected_returns,
        dtype=float,
    ).reshape(
        -1
    )

    weights = validate_weights(
        weights=weights,
        expected_size=len(
            expected_returns
        ),
    )

    if not np.isfinite(
        expected_returns
    ).all():

        raise ValueError(
            "Expected returns contain NaN or infinity."
        )

    result = float(
        np.dot(
            weights,
            expected_returns,
        )
    )

    return result


# =============================================================================
# PORTFOLIO VARIANCE
# =============================================================================

def portfolio_variance(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Calculate Markowitz portfolio variance.

    Equation:

        variance = w.T @ covariance @ w
    """

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    if covariance_matrix.ndim != 2:

        raise ValueError(
            "Covariance matrix must be two-dimensional."
        )

    if (
        covariance_matrix.shape[0]
        !=
        covariance_matrix.shape[1]
    ):

        raise ValueError(
            "Covariance matrix must be square."
        )

    weights = validate_weights(
        weights=weights,
        expected_size=covariance_matrix.shape[
            0
        ],
    )

    if not np.isfinite(
        covariance_matrix
    ).all():

        raise ValueError(
            "Covariance matrix contains NaN or infinity."
        )

    if not np.allclose(
        covariance_matrix,
        covariance_matrix.T,
        atol=1e-10,
    ):

        raise ValueError(
            "Covariance matrix is not symmetric."
        )

    variance = float(
        weights.T
        @
        covariance_matrix
        @
        weights
    )

    # Small numerical negatives may occur because of floating-point error.
    if (
        variance < 0
        and
        abs(
            variance
        )
        <=
        cfg.FLOAT_TOLERANCE
    ):

        variance = 0.0

    if variance < 0:

        raise ValueError(
            f"Portfolio variance is negative: {variance}"
        )

    return variance


# =============================================================================
# PORTFOLIO VOLATILITY
# =============================================================================

def portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Calculate portfolio standard deviation.
    """

    variance = portfolio_variance(
        weights=weights,
        covariance_matrix=covariance_matrix,
    )

    return float(
        np.sqrt(
            variance
        )
    )


# =============================================================================
# SHARPE RATIO
# =============================================================================

def portfolio_sharpe_ratio(
    portfolio_return: float,
    portfolio_volatility_value: float,
    risk_free_rate: float = cfg.RISK_FREE_RATE,
) -> float:
    """
    Calculate Sharpe ratio.

        Sharpe =
            (portfolio return - risk-free rate)
            /
            portfolio volatility
    """

    portfolio_return = float(
        portfolio_return
    )

    portfolio_volatility_value = float(
        portfolio_volatility_value
    )

    risk_free_rate = float(
        risk_free_rate
    )

    if not np.isfinite(
        portfolio_return
    ):

        raise ValueError(
            "Portfolio return is not finite."
        )

    if not np.isfinite(
        portfolio_volatility_value
    ):

        raise ValueError(
            "Portfolio volatility is not finite."
        )

    if portfolio_volatility_value <= 0:

        return 0.0

    return float(
        (
            portfolio_return
            -
            risk_free_rate
        )
        /
        portfolio_volatility_value
    )


# =============================================================================
# DAILY PORTFOLIO RETURNS
# =============================================================================

def portfolio_return_series(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Convert an asset-return matrix into a portfolio-return series.

    Each row:
        portfolio return at time t
        =
        sum(asset return * asset weight)
    """

    if returns.empty:

        raise ValueError(
            "Return matrix is empty."
        )

    numeric_returns = (
        returns
        .astype(
            float
        )
    )

    if not np.isfinite(
        numeric_returns.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Return matrix contains NaN or infinity."
        )

    weights = validate_weights(
        weights=weights,
        expected_size=numeric_returns.shape[
            1
        ],
    )

    values = (
        numeric_returns
        .to_numpy(
            dtype=float
        )
        @
        weights
    )

    return pd.Series(
        values,
        index=numeric_returns.index,
        name="portfolio_return",
    )


# =============================================================================
# CUMULATIVE RETURN
# =============================================================================

def cumulative_return(
    portfolio_returns: pd.Series,
) -> float:
    """
    Calculate cumulative return.

        cumulative return =
            product(1 + daily_return) - 1
    """

    if portfolio_returns.empty:

        raise ValueError(
            "Portfolio return series is empty."
        )

    values = (
        pd.to_numeric(
            portfolio_returns,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if values.size == 0:

        raise ValueError(
            "Portfolio return series contains no valid observations."
        )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "Portfolio return series contains invalid values."
        )

    result = float(
        np.prod(
            1.0
            +
            values
        )
        -
        1.0
    )

    return result


# =============================================================================
# ANNUALIZED RETURN
# =============================================================================

def annualized_return(
    portfolio_returns: pd.Series,
    trading_days_per_year: int = cfg.TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate geometric annualized portfolio return.
    """

    if trading_days_per_year <= 0:

        raise ValueError(
            "trading_days_per_year must be positive."
        )

    valid_returns = (
        pd.to_numeric(
            portfolio_returns,
            errors="coerce",
        )
        .dropna()
    )

    observations = len(
        valid_returns
    )

    if observations == 0:

        raise ValueError(
            "No portfolio returns available."
        )

    cumulative = cumulative_return(
        valid_returns
    )

    growth_factor = (
        1.0
        +
        cumulative
    )

    if growth_factor <= 0:

        # Portfolio has effectively lost all capital.
        return -1.0

    result = float(
        growth_factor
        **
        (
            trading_days_per_year
            /
            observations
        )
        -
        1.0
    )

    return result


# =============================================================================
# ANNUALIZED VOLATILITY
# =============================================================================

def annualized_volatility(
    portfolio_returns: pd.Series,
    trading_days_per_year: int = cfg.TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate annualized standard deviation.
    """

    valid_returns = (
        pd.to_numeric(
            portfolio_returns,
            errors="coerce",
        )
        .dropna()
    )

    if len(
        valid_returns
    ) < 2:

        return 0.0

    daily_std = float(
        valid_returns.std(
            ddof=1
        )
    )

    return float(
        daily_std
        *
        np.sqrt(
            trading_days_per_year
        )
    )


# =============================================================================
# WEALTH CURVE
# =============================================================================

def portfolio_wealth_curve(
    portfolio_returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    Convert portfolio returns into a wealth/index curve.
    """

    if initial_value <= 0:

        raise ValueError(
            "initial_value must be positive."
        )

    values = (
        pd.to_numeric(
            portfolio_returns,
            errors="coerce",
        )
        .fillna(
            0.0
        )
    )

    wealth = (
        (
            1.0
            +
            values
        )
        .cumprod()
        *
        float(
            initial_value
        )
    )

    wealth.name = (
        "portfolio_value"
    )

    return wealth


# =============================================================================
# MAXIMUM DRAWDOWN
# =============================================================================

def maximum_drawdown(
    portfolio_returns: pd.Series,
) -> float:
    """
    Calculate maximum drawdown.

    Drawdown:

        (current wealth - previous peak wealth)
        /
        previous peak wealth

    Output is normally negative.

    Example:

        -0.20 = -20% maximum drawdown
    """

    wealth = portfolio_wealth_curve(
        portfolio_returns=portfolio_returns,
        initial_value=1.0,
    )

    running_peak = (
        wealth.cummax()
    )

    drawdowns = (
        wealth
        /
        running_peak
        -
        1.0
    )

    return float(
        drawdowns.min()
    )


# =============================================================================
# HISTORICAL VALUE AT RISK
# =============================================================================

def historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Historical one-period Value at Risk.

    Returned as a positive loss magnitude.

    Example:

        0.025

    means approximately 2.5% daily VaR.
    """

    if not (
        0.0
        <
        confidence_level
        <
        1.0
    ):

        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    valid_returns = (
        pd.to_numeric(
            portfolio_returns,
            errors="coerce",
        )
        .dropna()
    )

    if valid_returns.empty:

        raise ValueError(
            "Portfolio return series is empty."
        )

    tail_probability = (
        1.0
        -
        confidence_level
    )

    return_quantile = float(
        valid_returns.quantile(
            tail_probability
        )
    )

    return float(
        max(
            0.0,
            -return_quantile,
        )
    )


# =============================================================================
# PARAMETRIC VALUE AT RISK
# =============================================================================

def parametric_var(
    portfolio_return: float,
    portfolio_volatility_value: float,
    z_score: float = 1.6448536269514722,
) -> float:
    """
    One-period parametric VaR using normal approximation.

    Default z-score corresponds approximately to 95% confidence.
    """

    portfolio_return = float(
        portfolio_return
    )

    portfolio_volatility_value = float(
        portfolio_volatility_value
    )

    if portfolio_volatility_value < 0:

        raise ValueError(
            "Portfolio volatility cannot be negative."
        )

    value = (
        z_score
        *
        portfolio_volatility_value
        -
        portfolio_return
    )

    return float(
        max(
            0.0,
            value,
        )
    )


# =============================================================================
# PORTFOLIO TURNOVER
# =============================================================================

def portfolio_turnover(
    previous_weights: np.ndarray,
    new_weights: np.ndarray,
) -> float:
    """
    Calculate one-period portfolio turnover.

        turnover =
            sum(abs(new weight - previous weight))

    This follows the absolute weight-change interpretation.

    It is useful for estimating how aggressively the portfolio was
    rebalanced.
    """

    previous_weights = np.asarray(
        previous_weights,
        dtype=float,
    ).reshape(
        -1
    )

    new_weights = np.asarray(
        new_weights,
        dtype=float,
    ).reshape(
        -1
    )

    if (
        previous_weights.size
        !=
        new_weights.size
    ):

        raise ValueError(
            "Previous and new portfolio weight vectors "
            "must have the same size."
        )

    if not np.isfinite(
        previous_weights
    ).all():

        raise ValueError(
            "Previous weights contain invalid values."
        )

    if not np.isfinite(
        new_weights
    ).all():

        raise ValueError(
            "New weights contain invalid values."
        )

    return float(
        np.sum(
            np.abs(
                new_weights
                -
                previous_weights
            )
        )
    )


# =============================================================================
# CONCENTRATION METRIC
# =============================================================================

def herfindahl_index(
    weights: np.ndarray,
) -> float:
    """
    Calculate portfolio concentration using the Herfindahl index.

        HHI = sum(w_i^2)

    Lower values generally indicate greater diversification.
    """

    weights = validate_weights(
        weights=weights,
    )

    return float(
        np.sum(
            weights
            **
            2
        )
    )


# =============================================================================
# EFFECTIVE NUMBER OF ASSETS
# =============================================================================

def effective_number_of_assets(
    weights: np.ndarray,
) -> float:
    """
    Effective number of equally weighted assets.

        effective N = 1 / HHI
    """

    hhi = herfindahl_index(
        weights
    )

    if hhi <= 0:

        return 0.0

    return float(
        1.0
        /
        hhi
    )


# =============================================================================
# COMPLETE PORTFOLIO METRICS
# =============================================================================

def calculate_portfolio_metrics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = cfg.RISK_FREE_RATE,
    previous_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calculate complete portfolio-performance statistics.
    """

    weights = validate_weights(
        weights=weights,
        expected_size=returns.shape[
            1
        ],
    )

    portfolio_returns = (
        portfolio_return_series(
            returns=returns,
            weights=weights,
        )
    )

    cumulative = cumulative_return(
        portfolio_returns
    )

    annual_return = annualized_return(
        portfolio_returns
    )

    annual_vol = annualized_volatility(
        portfolio_returns
    )

    max_dd = maximum_drawdown(
        portfolio_returns
    )

    sharpe = portfolio_sharpe_ratio(
        portfolio_return=annual_return,
        portfolio_volatility_value=annual_vol,
        risk_free_rate=risk_free_rate,
    )

    hist_var = historical_var(
        portfolio_returns=portfolio_returns,
        confidence_level=0.95,
    )

    daily_mean = float(
        portfolio_returns.mean()
    )

    daily_vol = float(
        portfolio_returns.std(
            ddof=1
        )
    )

    annual_arithmetic_mean = float(
        daily_mean
        *
        cfg.TRADING_DAYS_PER_YEAR
    )

    hhi = herfindahl_index(
        weights
    )

    effective_assets = (
        effective_number_of_assets(
            weights
        )
    )

    if previous_weights is not None:

        turnover = portfolio_turnover(
            previous_weights=previous_weights,
            new_weights=weights,
        )

    else:

        turnover = 0.0

    return {
        "observations":
            int(
                len(
                    portfolio_returns
                )
            ),

        "daily_mean_return":
            daily_mean,

        "daily_volatility":
            daily_vol,

        "cumulative_return":
            cumulative,

        "annualized_return":
            annual_return,

        "annualized_arithmetic_mean_return":
            annual_arithmetic_mean,

        "annualized_volatility":
            annual_vol,

        "sharpe_ratio":
            sharpe,

        "max_drawdown":
            max_dd,

        "historical_var_95":
            hist_var,

        "turnover":
            turnover,

        "herfindahl_index":
            hhi,

        "effective_number_of_assets":
            effective_assets,

        "weight_sum":
            float(
                weights.sum()
            ),

        "minimum_weight":
            float(
                weights.min()
            ),

        "maximum_weight":
            float(
                weights.max()
            ),
    }


# =============================================================================
# READ AGENT-5 RETURNS
# =============================================================================

def load_returns_matrix() -> pd.DataFrame:
    """
    Load the return matrix generated by data_loader.py.
    """

    if cfg.RETURNS_MATRIX_PARQUET.exists():

        dataframe = pd.read_parquet(
            cfg.RETURNS_MATRIX_PARQUET
        )

        source = "PARQUET"

    elif cfg.RETURNS_MATRIX_CSV.exists():

        dataframe = pd.read_csv(
            cfg.RETURNS_MATRIX_CSV
        )

        source = "CSV"

    else:

        raise FileNotFoundError(
            "Agent-5 returns matrix was not found.\n"
            "Run data_loader.py first."
        )

    if "date" not in dataframe.columns:

        raise ValueError(
            "Returns matrix does not contain a date column."
        )

    dataframe[
        "date"
    ] = pd.to_datetime(
        dataframe[
            "date"
        ],
        errors="coerce",
    )

    if dataframe[
        "date"
    ].isna().any():

        raise ValueError(
            "Returns matrix contains invalid dates."
        )

    dataframe = (
        dataframe
        .set_index(
            "date"
        )
        .sort_index()
    )

    dataframe = dataframe.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if dataframe.isna().any().any():

        raise ValueError(
            "Returns matrix contains missing/non-numeric values."
        )

    if not np.isfinite(
        dataframe.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Returns matrix contains invalid numerical values."
        )

    print(
        f"Returns matrix          : {source}"
    )

    return dataframe


# =============================================================================
# READ REBALANCING INPUT
# =============================================================================

def load_rebalancing_input() -> pd.DataFrame:
    """
    Load Agent-5 rebalancing input produced by data_loader.py.
    """

    if cfg.REBALANCING_INPUT_PARQUET.exists():

        dataframe = pd.read_parquet(
            cfg.REBALANCING_INPUT_PARQUET
        )

        source = "PARQUET"

    elif cfg.REBALANCING_INPUT_CSV.exists():

        dataframe = pd.read_csv(
            cfg.REBALANCING_INPUT_CSV,
            low_memory=False,
        )

        source = "CSV"

    else:

        raise FileNotFoundError(
            "Agent-5 rebalancing input was not found.\n"
            "Run data_loader.py first."
        )

    required_columns = {
        "symbol",
        "trade_action",
        "eligible_for_mpt",
    }

    missing = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            "Rebalancing input is missing columns:\n"
            +
            "\n".join(
                sorted(
                    missing
                )
            )
        )

    print(
        f"Rebalancing input      : {source}"
    )

    return dataframe


# =============================================================================
# SMOKE TEST
# =============================================================================

def run_portfolio_metrics_smoke_test() -> dict[str, Any]:
    """
    Validate portfolio-metrics calculations using an equal-weight
    portfolio across the current MPT-eligible assets.

    IMPORTANT:
    This is NOT the final optimized portfolio.

    It exists only to verify portfolio mathematics before implementing
    mpt_optimizer.py.
    """

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 5 - PORTFOLIO METRICS"
    )

    print(
        "=" * 100
    )

    print()

    returns = load_returns_matrix()

    rebalancing_input = (
        load_rebalancing_input()
    )

    # Convert CSV-loaded strings safely when necessary.
    if (
        rebalancing_input[
            "eligible_for_mpt"
        ]
        .dtype
        !=
        bool
    ):

        eligible_mask = (
            rebalancing_input[
                "eligible_for_mpt"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

    else:

        eligible_mask = (
            rebalancing_input[
                "eligible_for_mpt"
            ]
        )

    eligible_symbols = (
        rebalancing_input.loc[
            eligible_mask,
            "symbol",
        ]
        .astype(str)
        .tolist()
    )

    if not eligible_symbols:

        raise ValueError(
            "No MPT-eligible assets are available."
        )

    missing_return_symbols = [
        symbol

        for symbol
        in eligible_symbols

        if symbol
        not in returns.columns
    ]

    if missing_return_symbols:

        raise ValueError(
            "Eligible symbols missing from returns matrix:\n"
            +
            "\n".join(
                missing_return_symbols
            )
        )

    eligible_returns = (
        returns[
            eligible_symbols
        ]
        .copy()
    )

    number_of_assets = len(
        eligible_symbols
    )

    equal_weights = np.full(
        number_of_assets,
        1.0
        /
        number_of_assets,
        dtype=float,
    )

    # Annualized arithmetic expected returns.
    expected_returns = (
        eligible_returns.mean()
        *
        cfg.TRADING_DAYS_PER_YEAR
    )

    covariance_matrix = (
        eligible_returns.cov()
        *
        cfg.TRADING_DAYS_PER_YEAR
    )

    expected_portfolio_return = (
        portfolio_expected_return(
            weights=equal_weights,
            expected_returns=expected_returns.to_numpy(
                dtype=float
            ),
        )
    )

    expected_portfolio_variance = (
        portfolio_variance(
            weights=equal_weights,
            covariance_matrix=covariance_matrix.to_numpy(
                dtype=float
            ),
        )
    )

    expected_portfolio_volatility = (
        portfolio_volatility(
            weights=equal_weights,
            covariance_matrix=covariance_matrix.to_numpy(
                dtype=float
            ),
        )
    )

    expected_sharpe = (
        portfolio_sharpe_ratio(
            portfolio_return=expected_portfolio_return,
            portfolio_volatility_value=expected_portfolio_volatility,
        )
    )

    realized_metrics = (
        calculate_portfolio_metrics(
            returns=eligible_returns,
            weights=equal_weights,
        )
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    checks = {
        "eligible_assets_exist":
            number_of_assets
            >
            0,

        "equal_weights_finite":
            bool(
                np.isfinite(
                    equal_weights
                ).all()
            ),

        "equal_weights_nonnegative":
            bool(
                (
                    equal_weights
                    >=
                    0
                )
                .all()
            ),

        "equal_weights_sum_to_one":
            bool(
                np.isclose(
                    equal_weights.sum(),
                    1.0,
                    atol=cfg.WEIGHT_SUM_TOLERANCE,
                )
            ),

        "expected_return_finite":
            bool(
                np.isfinite(
                    expected_portfolio_return
                )
            ),

        "variance_finite":
            bool(
                np.isfinite(
                    expected_portfolio_variance
                )
            ),

        "variance_nonnegative":
            bool(
                expected_portfolio_variance
                >=
                0
            ),

        "volatility_finite":
            bool(
                np.isfinite(
                    expected_portfolio_volatility
                )
            ),

        "sharpe_finite":
            bool(
                np.isfinite(
                    expected_sharpe
                )
            ),

        "cumulative_return_finite":
            bool(
                np.isfinite(
                    realized_metrics[
                        "cumulative_return"
                    ]
                )
            ),

        "annualized_return_finite":
            bool(
                np.isfinite(
                    realized_metrics[
                        "annualized_return"
                    ]
                )
            ),

        "annualized_volatility_finite":
            bool(
                np.isfinite(
                    realized_metrics[
                        "annualized_volatility"
                    ]
                )
            ),

        "max_drawdown_valid":
            bool(
                realized_metrics[
                    "max_drawdown"
                ]
                <=
                cfg.FLOAT_TOLERANCE
            ),

        "historical_var_valid":
            bool(
                realized_metrics[
                    "historical_var_95"
                ]
                >=
                0
            ),
    }

    overall_pass = all(
        checks.values()
    )

    # =========================================================================
    # DISPLAY
    # =========================================================================

    print(
        f"Eligible assets         : "
        f"{number_of_assets}"
    )

    print(
        f"Return observations     : "
        f"{len(eligible_returns)}"
    )

    print()

    print(
        "Equal-weight portfolio"
    )

    print(
        "-" * 100
    )

    for symbol, weight in zip(
        eligible_symbols,
        equal_weights,
    ):

        print(
            f"{symbol:<20} "
            f"{weight:>12.6%}"
        )

    print()

    print(
        "MPT EXPECTED METRICS"
    )

    print(
        "-" * 100
    )

    print(
        f"Expected annual return  : "
        f"{expected_portfolio_return:.6%}"
    )

    print(
        f"Portfolio variance      : "
        f"{expected_portfolio_variance:.8f}"
    )

    print(
        f"Portfolio volatility    : "
        f"{expected_portfolio_volatility:.6%}"
    )

    print(
        f"Expected Sharpe ratio   : "
        f"{expected_sharpe:.6f}"
    )

    print()

    print(
        "REALIZED 105-DAY METRICS"
    )

    print(
        "-" * 100
    )

    print(
        f"Cumulative return       : "
        f"{realized_metrics['cumulative_return']:.6%}"
    )

    print(
        f"Annualized return       : "
        f"{realized_metrics['annualized_return']:.6%}"
    )

    print(
        f"Annualized volatility   : "
        f"{realized_metrics['annualized_volatility']:.6%}"
    )

    print(
        f"Sharpe ratio            : "
        f"{realized_metrics['sharpe_ratio']:.6f}"
    )

    print(
        f"Max drawdown            : "
        f"{realized_metrics['max_drawdown']:.6%}"
    )

    print(
        f"Historical VaR 95%      : "
        f"{realized_metrics['historical_var_95']:.6%}"
    )

    print(
        f"Effective assets        : "
        f"{realized_metrics['effective_number_of_assets']:.4f}"
    )

    print()

    print(
        "=" * 100
    )

    print(
        "PORTFOLIO METRICS VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<55}: "
            f"{passed}"
        )

    print()

    print(
        "VALIDATION RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    if not overall_pass:

        failed = [
            name

            for name, passed
            in checks.items()

            if not passed
        ]

        raise ValueError(
            "Portfolio-metrics validation failed:\n"
            +
            "\n".join(
                f"  - {name}"

                for name
                in failed
            )
        )

    # =========================================================================
    # SAVE SMOKE-TEST REPORT
    # =========================================================================

    report = {
        "status":
            "PASS",

        "portfolio_type":
            "equal_weight_smoke_test",

        "methodology_note":
            (
                "Equal-weight portfolio used only to validate Agent-5 "
                "portfolio mathematics before Markowitz optimization. "
                "This is not the final Agent-5 allocation."
            ),

        "eligible_symbols":
            eligible_symbols,

        "weights": {
            symbol:
                float(
                    weight
                )

            for symbol, weight
            in zip(
                eligible_symbols,
                equal_weights,
            )
        },

        "expected_metrics": {
            "annualized_expected_return":
                float(
                    expected_portfolio_return
                ),

            "annualized_variance":
                float(
                    expected_portfolio_variance
                ),

            "annualized_volatility":
                float(
                    expected_portfolio_volatility
                ),

            "sharpe_ratio":
                float(
                    expected_sharpe
                ),
        },

        "realized_metrics":
            {
                key:
                    (
                        int(value)
                        if isinstance(
                            value,
                            (
                                int,
                                np.integer,
                            ),
                        )
                        else
                        float(value)
                    )

                for key, value
                in realized_metrics.items()
            },

        "validation": {
            name:
                bool(
                    passed
                )

            for name, passed
            in checks.items()
        },
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        cfg.PORTFOLIO_METRICS_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()

    print(
        f"Metrics report          : "
        f"{cfg.PORTFOLIO_METRICS_REPORT}"
    )

    print()

    print(
        "=" * 100
    )

    print(
        "PORTFOLIO METRICS STATUS: COMPLETE"
    )

    print(
        "=" * 100
    )

    return {
        "returns":
            eligible_returns,

        "symbols":
            eligible_symbols,

        "weights":
            equal_weights,

        "expected_return":
            expected_portfolio_return,

        "variance":
            expected_portfolio_variance,

        "volatility":
            expected_portfolio_volatility,

        "sharpe":
            expected_sharpe,

        "realized_metrics":
            realized_metrics,

        "checks":
            checks,
    }


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_portfolio_metrics_smoke_test()