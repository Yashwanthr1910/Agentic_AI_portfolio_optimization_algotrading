"""
performance.py

Portfolio Performance Metrics Module

Purpose:
    Calculate portfolio performance and risk metrics
    from a historical portfolio return series.

Expected Input:
    A pandas DataFrame containing at least:

        date
        portfolio_return

Optional:
        portfolio_value

Main Metrics:
    - Cumulative Return
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio
    - Maximum Drawdown
    - Calmar Ratio
    - Sortino Ratio
    - VaR
    - Omega Ratio
    - Tail Ratio

This module is used by:
    backtesting/backtest.py

Output:
    outputs/performance.csv
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

PERFORMANCE_FILE = (
    OUTPUT_DIR
    / "performance.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRADING_DAYS_PER_YEAR = 252

# Annual risk-free rate.
#
# Keep at zero for now unless you explicitly want
# to use another assumption.
RISK_FREE_RATE = 0.0

# Value at Risk confidence level
VAR_CONFIDENCE_LEVEL = 0.95


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_returns(
    returns: pd.Series
) -> pd.Series:

    """
    Clean and validate portfolio returns.
    """

    if returns is None:

        raise ValueError(
            "Return series cannot be None."
        )

    returns = pd.Series(
        returns
    ).copy()

    returns = pd.to_numeric(
        returns,
        errors="coerce"
    )

    returns = returns.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    returns = returns.dropna()

    if returns.empty:

        raise ValueError(
            "Portfolio return series is empty "
            "after cleaning."
        )

    return returns


# ============================================================
# CUMULATIVE RETURN
# ============================================================

def calculate_cumulative_return(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    cumulative_return = (
        (1 + returns)
        .prod()
        - 1
    )

    return float(
        cumulative_return
    )


# ============================================================
# ANNUALIZED RETURN
# ============================================================

def calculate_annualized_return(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    number_of_days = len(
        returns
    )

    cumulative_growth = (
        1 + returns
    ).prod()

    if cumulative_growth <= 0:

        return np.nan

    annualized_return = (
        cumulative_growth
        ** (
            TRADING_DAYS_PER_YEAR
            / number_of_days
        )
        - 1
    )

    return float(
        annualized_return
    )


# ============================================================
# ANNUALIZED VOLATILITY
# ============================================================

def calculate_annualized_volatility(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    volatility = (
        returns.std(
            ddof=1
        )
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    return float(
        volatility
    )


# ============================================================
# SHARPE RATIO
# ============================================================

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE
) -> float:

    returns = validate_returns(
        returns
    )

    daily_risk_free_rate = (
        risk_free_rate
        / TRADING_DAYS_PER_YEAR
    )

    excess_returns = (
        returns
        - daily_risk_free_rate
    )

    std = excess_returns.std(
        ddof=1
    )

    if np.isclose(
        std,
        0
    ):

        return np.nan

    sharpe = (
        excess_returns.mean()
        / std
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    return float(
        sharpe
    )


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    wealth_index = (
        1 + returns
    ).cumprod()

    running_peak = (
        wealth_index
        .cummax()
    )

    drawdown = (
        wealth_index
        / running_peak
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    return float(
        max_drawdown
    )


# ============================================================
# CALMAR RATIO
# ============================================================

def calculate_calmar_ratio(
    returns: pd.Series
) -> float:

    annualized_return = (
        calculate_annualized_return(
            returns
        )
    )

    max_drawdown = (
        calculate_max_drawdown(
            returns
        )
    )

    if (
        pd.isna(
            annualized_return
        )
        or np.isclose(
            max_drawdown,
            0
        )
    ):

        return np.nan

    calmar = (
        annualized_return
        / abs(
            max_drawdown
        )
    )

    return float(
        calmar
    )


# ============================================================
# SORTINO RATIO
# ============================================================

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE
) -> float:

    returns = validate_returns(
        returns
    )

    daily_risk_free_rate = (
        risk_free_rate
        / TRADING_DAYS_PER_YEAR
    )

    excess_returns = (
        returns
        - daily_risk_free_rate
    )

    downside_returns = (
        excess_returns[
            excess_returns < 0
        ]
    )

    if downside_returns.empty:

        return np.nan

    downside_deviation = (
        np.sqrt(
            np.mean(
                downside_returns ** 2
            )
        )
    )

    if np.isclose(
        downside_deviation,
        0
    ):

        return np.nan

    sortino = (
        excess_returns.mean()
        / downside_deviation
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    return float(
        sortino
    )


# ============================================================
# VALUE AT RISK
# ============================================================

def calculate_var(
    returns: pd.Series,
    confidence_level: float = VAR_CONFIDENCE_LEVEL
) -> float:

    returns = validate_returns(
        returns
    )

    percentile = (
        1
        - confidence_level
    )

    var = np.quantile(
        returns,
        percentile
    )

    return float(
        var
    )


# ============================================================
# OMEGA RATIO
# ============================================================

def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:

    returns = validate_returns(
        returns
    )

    gains = (
        returns[
            returns > threshold
        ]
        - threshold
    )

    losses = (
        threshold
        - returns[
            returns < threshold
        ]
    )

    total_gains = (
        gains.sum()
    )

    total_losses = (
        losses.sum()
    )

    if np.isclose(
        total_losses,
        0
    ):

        return np.nan

    omega = (
        total_gains
        / total_losses
    )

    return float(
        omega
    )


# ============================================================
# TAIL RATIO
# ============================================================

def calculate_tail_ratio(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    upper_tail = np.quantile(
        returns,
        0.95
    )

    lower_tail = np.quantile(
        returns,
        0.05
    )

    if np.isclose(
        lower_tail,
        0
    ):

        return np.nan

    tail_ratio = (
        abs(
            upper_tail
        )
        /
        abs(
            lower_tail
        )
    )

    return float(
        tail_ratio
    )


# ============================================================
# VARIANCE
# ============================================================

def calculate_variance(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    variance = returns.var(
        ddof=1
    )

    return float(
        variance
    )


# ============================================================
# WIN RATE
# ============================================================

def calculate_win_rate(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    positive_days = (
        returns > 0
    ).sum()

    win_rate = (
        positive_days
        / len(
            returns
        )
    )

    return float(
        win_rate
    )


# ============================================================
# LOSS RATE
# ============================================================

def calculate_loss_rate(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    negative_days = (
        returns < 0
    ).sum()

    loss_rate = (
        negative_days
        / len(
            returns
        )
    )

    return float(
        loss_rate
    )


# ============================================================
# BEST DAILY RETURN
# ============================================================

def calculate_best_day(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    return float(
        returns.max()
    )


# ============================================================
# WORST DAILY RETURN
# ============================================================

def calculate_worst_day(
    returns: pd.Series
) -> float:

    returns = validate_returns(
        returns
    )

    return float(
        returns.min()
    )


# ============================================================
# COMPLETE PERFORMANCE METRICS
# ============================================================

def calculate_performance_metrics(
    returns: pd.Series
) -> dict:

    logger.info(
        "Calculating portfolio performance metrics..."
    )

    returns = validate_returns(
        returns
    )

    metrics = {

        "observations":
            len(
                returns
            ),

        "cumulative_return":
            calculate_cumulative_return(
                returns
            ),

        "annualized_return":
            calculate_annualized_return(
                returns
            ),

        "annualized_volatility":
            calculate_annualized_volatility(
                returns
            ),

        "sharpe_ratio":
            calculate_sharpe_ratio(
                returns
            ),

        "max_drawdown":
            calculate_max_drawdown(
                returns
            ),

        "calmar_ratio":
            calculate_calmar_ratio(
                returns
            ),

        "sortino_ratio":
            calculate_sortino_ratio(
                returns
            ),

        "daily_var_95":
            calculate_var(
                returns,
                confidence_level=0.95
            ),

        "omega_ratio":
            calculate_omega_ratio(
                returns
            ),

        "tail_ratio":
            calculate_tail_ratio(
                returns
            ),

        "variance":
            calculate_variance(
                returns
            ),

        "win_rate":
            calculate_win_rate(
                returns
            ),

        "loss_rate":
            calculate_loss_rate(
                returns
            ),

        "best_daily_return":
            calculate_best_day(
                returns
            ),

        "worst_daily_return":
            calculate_worst_day(
                returns
            )
    }

    return metrics


# ============================================================
# DISPLAY PERFORMANCE REPORT
# ============================================================

def display_performance_report(
    metrics: dict
) -> None:

    logger.info(
        ""
    )

    logger.info(
        "=============================================================="
    )

    logger.info(
        "                  PORTFOLIO PERFORMANCE"
    )

    logger.info(
        "=============================================================="
    )

    logger.info(
        "Observations          : %d",
        metrics[
            "observations"
        ]
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    logger.info(
        "Cumulative Return     : %.2f%%",
        metrics[
            "cumulative_return"
        ]
        * 100
    )

    logger.info(
        "Annualized Return     : %.2f%%",
        metrics[
            "annualized_return"
        ]
        * 100
    )

    logger.info(
        "Annualized Volatility : %.2f%%",
        metrics[
            "annualized_volatility"
        ]
        * 100
    )

    logger.info(
        "Sharpe Ratio          : %.4f",
        metrics[
            "sharpe_ratio"
        ]
    )

    logger.info(
        "Sortino Ratio         : %.4f",
        metrics[
            "sortino_ratio"
        ]
    )

    logger.info(
        "Maximum Drawdown      : %.2f%%",
        metrics[
            "max_drawdown"
        ]
        * 100
    )

    logger.info(
        "Calmar Ratio          : %.4f",
        metrics[
            "calmar_ratio"
        ]
    )

    logger.info(
        "Omega Ratio           : %.4f",
        metrics[
            "omega_ratio"
        ]
    )

    logger.info(
        "Tail Ratio            : %.4f",
        metrics[
            "tail_ratio"
        ]
    )

    logger.info(
        "Daily VaR (95%%)       : %.2f%%",
        metrics[
            "daily_var_95"
        ]
        * 100
    )

    logger.info(
        "Variance              : %.8f",
        metrics[
            "variance"
        ]
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    logger.info(
        "Win Rate              : %.2f%%",
        metrics[
            "win_rate"
        ]
        * 100
    )

    logger.info(
        "Loss Rate             : %.2f%%",
        metrics[
            "loss_rate"
        ]
        * 100
    )

    logger.info(
        "Best Daily Return     : %.2f%%",
        metrics[
            "best_daily_return"
        ]
        * 100
    )

    logger.info(
        "Worst Daily Return    : %.2f%%",
        metrics[
            "worst_daily_return"
        ]
        * 100
    )

    logger.info(
        "=============================================================="
    )


# ============================================================
# SAVE PERFORMANCE
# ============================================================

def save_performance_metrics(
    metrics: dict
) -> pd.DataFrame:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    performance_df = pd.DataFrame(
        [
            metrics
        ]
    )

    performance_df.to_csv(
        PERFORMANCE_FILE,
        index=False
    )

    logger.info(
        "Performance metrics saved to:"
    )

    logger.info(
        "%s",
        PERFORMANCE_FILE
    )

    return performance_df


# ============================================================
# EVALUATE PERFORMANCE
# ============================================================

def evaluate_performance(
    returns: pd.Series
) -> pd.DataFrame:

    """
    Main reusable performance function.

    Called by backtest.py.
    """

    metrics = (
        calculate_performance_metrics(
            returns
        )
    )

    display_performance_report(
        metrics
    )

    performance_df = (
        save_performance_metrics(
            metrics
        )
    )

    return performance_df


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "performance.py is a reusable metrics module."
    )

    logger.info(
        "Run backtesting/backtest.py to generate "
        "portfolio returns and calculate performance."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()