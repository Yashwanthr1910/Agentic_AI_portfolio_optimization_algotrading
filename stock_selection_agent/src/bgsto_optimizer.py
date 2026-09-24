"""
bgsto_optimizer.py

V2.3 - Binary Genetic Siberian Tiger Optimization (BGSTO)

Inputs:
    data/outputs/decision_tree_predictions.parquet
    data/processed/features.parquet

Outputs:
    data/outputs/bgsto_selected_stocks.csv
    data/outputs/bgsto_selected_stocks.parquet
    reports/results/bgsto_optimization_history.csv

Purpose:
    Optimize the subset of BUY candidates produced by the
    Decision Tree while considering:

    - Decision Tree BUY probability
    - Momentum / returns
    - Sharpe ratio
    - Volatility
    - Portfolio size

IMPORTANT:
    Future-return columns and target labels are NOT used
    in optimization to avoid look-ahead leakage.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "decision_tree_predictions.parquet"
)

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
)

CSV_OUTPUT = (
    OUTPUT_DIR
    / "bgsto_selected_stocks.csv"
)

PARQUET_OUTPUT = (
    OUTPUT_DIR
    / "bgsto_selected_stocks.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
)

HISTORY_FILE = (
    REPORT_DIR
    / "bgsto_optimization_history.csv"
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
# BGSTO CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Candidate pool before optimization
MAX_CANDIDATES = 60

# Final number of stocks
MIN_SELECTED_STOCKS = 10
MAX_SELECTED_STOCKS = 20

# Optimizer
POPULATION_SIZE = 50
NUM_ITERATIONS = 100

# Genetic Algorithm parameters
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.05

# ------------------------------------------------------------
# Fitness weights
# ------------------------------------------------------------

WEIGHT_BUY_PROBABILITY = 0.40
WEIGHT_RETURN = 0.25
WEIGHT_SHARPE = 0.20
WEIGHT_VOLATILITY = 0.15


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Decision Tree predictions not found: "
            f"{PREDICTIONS_FILE}"
        )

    if not FEATURES_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{FEATURES_FILE}"
        )

    logger.info(
        "Loading Decision Tree predictions..."
    )

    predictions = pd.read_parquet(
        PREDICTIONS_FILE
    )

    logger.info(
        "Loading feature dataset..."
    )

    features = pd.read_parquet(
        FEATURES_FILE
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    features["date"] = pd.to_datetime(
        features["date"]
    )

    logger.info(
        "Prediction rows: %d",
        len(predictions)
    )

    logger.info(
        "Feature rows: %d",
        len(features)
    )

    return predictions, features


# ============================================================
# BUILD CANDIDATE DATASET
# ============================================================

def build_candidate_dataset(
    predictions,
    features
):

    logger.info(
        "Building BGSTO candidate dataset..."
    )

    # --------------------------------------------------------
    # Use validation period for current/final selection
    # --------------------------------------------------------

    validation_predictions = predictions[
        predictions["dataset_split"]
        == "validation"
    ].copy()

    if validation_predictions.empty:

        raise ValueError(
            "No validation predictions found."
        )

    # --------------------------------------------------------
    # Find latest date that has predictions
    # --------------------------------------------------------

    latest_date = (
        validation_predictions["date"]
        .max()
    )

    logger.info(
        "Initial latest prediction date: %s",
        latest_date.date()
    )

    # --------------------------------------------------------
    # Get BUY candidates
    # --------------------------------------------------------

    candidates = validation_predictions[
        (
            validation_predictions["date"]
            == latest_date
        )
        &
        (
            validation_predictions[
                "predicted_target"
            ] == 1
        )
    ].copy()

    # --------------------------------------------------------
    # If final day does not contain enough candidates,
    # search backwards for a usable date.
    # --------------------------------------------------------

    if len(candidates) < MIN_SELECTED_STOCKS:

        logger.warning(
            "Latest date has only %d BUY candidates.",
            len(candidates)
        )

        available_dates = (
            validation_predictions[
                "date"
            ]
            .drop_duplicates()
            .sort_values(
                ascending=False
            )
        )

        found = False

        for current_date in available_dates:

            current_candidates = (
                validation_predictions[
                    (
                        validation_predictions["date"]
                        == current_date
                    )
                    &
                    (
                        validation_predictions[
                            "predicted_target"
                        ] == 1
                    )
                ]
                .copy()
            )

            if (
                len(current_candidates)
                >= MIN_SELECTED_STOCKS
            ):

                latest_date = current_date
                candidates = current_candidates

                found = True
                break

        if not found:

            raise ValueError(
                "Could not find a date containing enough "
                "Decision Tree BUY candidates."
            )

    logger.info(
        "Selection date: %s",
        latest_date.date()
    )

    logger.info(
        "Decision Tree BUY candidates: %d",
        len(candidates)
    )

    # --------------------------------------------------------
    # Limit candidate universe
    # --------------------------------------------------------

    candidates = (
        candidates
        .sort_values(
            "buy_probability",
            ascending=False
        )
        .head(MAX_CANDIDATES)
        .copy()
    )

    # --------------------------------------------------------
    # Feature columns required by BGSTO
    # --------------------------------------------------------

    required_features = [
        "date",
        "symbol",
        "return_21d",
        "return_63d",
        "volatility_60d",
        "sharpe_252d",
        "rsi_14",
    ]

    available_features = [
        column
        for column in required_features
        if column in features.columns
    ]

    date_features = features[
        features["date"] == latest_date
    ][
        available_features
    ].copy()

    # --------------------------------------------------------
    # Merge Decision Tree information with features
    # --------------------------------------------------------

    merge_columns = [
        column
        for column in available_features
        if column not in [
            "date",
            "symbol"
        ]
    ]

    candidates = candidates.merge(
        date_features[
            [
                "date",
                "symbol",
                *merge_columns
            ]
        ],
        on=[
            "date",
            "symbol"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Remove stocks missing essential optimizer features
    # --------------------------------------------------------

    essential = [
        "buy_probability",
        "return_21d",
        "return_63d",
        "volatility_60d",
        "sharpe_252d",
    ]

    essential = [
        column
        for column in essential
        if column in candidates.columns
    ]

    before = len(candidates)

    candidates = candidates.dropna(
        subset=essential
    ).copy()

    logger.info(
        "Removed %d candidates with missing "
        "optimizer features.",
        before - len(candidates)
    )

    if len(candidates) < MIN_SELECTED_STOCKS:

        raise ValueError(
            "Not enough valid stocks remain for BGSTO. "
            f"Available: {len(candidates)}"
        )

    candidates = candidates.reset_index(
        drop=True
    )

    logger.info(
        "Final BGSTO candidate pool: %d stocks",
        len(candidates)
    )

    return candidates, latest_date


# ============================================================
# NORMALIZATION
# ============================================================

def min_max_normalize(
    series
):

    minimum = series.min()
    maximum = series.max()

    if np.isclose(
        maximum,
        minimum
    ):

        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# PREPARE FITNESS DATA
# ============================================================

def prepare_fitness_features(
    candidates
):

    df = candidates.copy()

    # --------------------------------------------------------
    # Higher is better
    # --------------------------------------------------------

    df["norm_buy_probability"] = (
        min_max_normalize(
            df["buy_probability"]
        )
    )

    df["norm_return"] = (
        min_max_normalize(
            df["return_63d"]
        )
    )

    df["norm_sharpe"] = (
        min_max_normalize(
            df["sharpe_252d"]
        )
    )

    # --------------------------------------------------------
    # Lower volatility is better
    # --------------------------------------------------------

    normalized_volatility = (
        min_max_normalize(
            df["volatility_60d"]
        )
    )

    df["norm_low_volatility"] = (
        1 - normalized_volatility
    )

    return df


# ============================================================
# REPAIR BINARY SOLUTION
# ============================================================

def repair_solution(
    solution,
    scores
):

    solution = solution.copy()

    selected_count = int(
        solution.sum()
    )

    # --------------------------------------------------------
    # Too few selected stocks
    # --------------------------------------------------------

    if (
        selected_count
        < MIN_SELECTED_STOCKS
    ):

        not_selected = np.where(
            solution == 0
        )[0]

        ranked = sorted(
            not_selected,
            key=lambda i: scores[i],
            reverse=True
        )

        needed = (
            MIN_SELECTED_STOCKS
            - selected_count
        )

        for index in ranked[:needed]:

            solution[index] = 1

    # --------------------------------------------------------
    # Too many selected stocks
    # --------------------------------------------------------

    selected_count = int(
        solution.sum()
    )

    if (
        selected_count
        > MAX_SELECTED_STOCKS
    ):

        selected = np.where(
            solution == 1
        )[0]

        ranked = sorted(
            selected,
            key=lambda i: scores[i],
            reverse=True
        )

        keep = ranked[
            :MAX_SELECTED_STOCKS
        ]

        solution[:] = 0

        solution[keep] = 1

    return solution


# ============================================================
# INDIVIDUAL STOCK SCORE
# ============================================================

def calculate_stock_scores(
    candidate_df
):

    scores = (

        WEIGHT_BUY_PROBABILITY
        * candidate_df[
            "norm_buy_probability"
        ].to_numpy()

        +

        WEIGHT_RETURN
        * candidate_df[
            "norm_return"
        ].to_numpy()

        +

        WEIGHT_SHARPE
        * candidate_df[
            "norm_sharpe"
        ].to_numpy()

        +

        WEIGHT_VOLATILITY
        * candidate_df[
            "norm_low_volatility"
        ].to_numpy()

    )

    return scores


# ============================================================
# FITNESS FUNCTION
# ============================================================

def fitness(
    solution,
    stock_scores,
    candidate_df
):

    selected = np.where(
        solution == 1
    )[0]

    if len(selected) == 0:

        return -np.inf

    # --------------------------------------------------------
    # Core quality score
    # --------------------------------------------------------

    quality = float(
        np.mean(
            stock_scores[
                selected
            ]
        )
    )

    # --------------------------------------------------------
    # Additional portfolio-level return/risk behaviour
    # --------------------------------------------------------

    selected_returns = (
        candidate_df
        .iloc[selected][
            "return_63d"
        ]
    )

    selected_volatility = (
        candidate_df
        .iloc[selected][
            "volatility_60d"
        ]
    )

    avg_return = (
        selected_returns.mean()
    )

    avg_volatility = (
        selected_volatility.mean()
    )

    # Small reward-risk adjustment
    return_risk_score = (
        avg_return
        -
        0.25 * avg_volatility
    )

    # --------------------------------------------------------
    # Mild portfolio-size penalty
    # --------------------------------------------------------

    size_ratio = (
        len(selected)
        /
        len(solution)
    )

    size_penalty = (
        0.02 * size_ratio
    )

    final_fitness = (
        quality
        +
        0.10 * return_risk_score
        -
        size_penalty
    )

    return float(
        final_fitness
    )


# ============================================================
# INITIAL POPULATION
# ============================================================

def initialize_population(
    num_stocks,
    scores
):

    population = []

    for _ in range(
        POPULATION_SIZE
    ):

        solution = rng.integers(
            0,
            2,
            size=num_stocks
        )

        solution = repair_solution(
            solution,
            scores
        )

        population.append(
            solution
        )

    return np.array(
        population,
        dtype=np.int8
    )


# ============================================================
# TOURNAMENT SELECTION
# ============================================================

def tournament_selection(
    population,
    fitness_values,
    tournament_size=3
):

    indexes = rng.choice(
        len(population),
        size=tournament_size,
        replace=False
    )

    best_index = indexes[
        np.argmax(
            fitness_values[indexes]
        )
    ]

    return population[
        best_index
    ].copy()


# ============================================================
# GENETIC CROSSOVER
# ============================================================

def crossover(
    parent1,
    parent2
):

    if (
        rng.random()
        > CROSSOVER_RATE
    ):

        return (
            parent1.copy(),
            parent2.copy()
        )

    if len(parent1) <= 2:

        return (
            parent1.copy(),
            parent2.copy()
        )

    point = rng.integers(
        1,
        len(parent1)
    )

    child1 = np.concatenate(
        [
            parent1[:point],
            parent2[point:]
        ]
    )

    child2 = np.concatenate(
        [
            parent2[:point],
            parent1[point:]
        ]
    )

    return child1, child2


# ============================================================
# GENETIC MUTATION
# ============================================================

def mutate(
    solution
):

    solution = solution.copy()

    mutation_mask = (
        rng.random(
            len(solution)
        )
        < MUTATION_RATE
    )

    solution[
        mutation_mask
    ] = (
        1
        -
        solution[
            mutation_mask
        ]
    )

    return solution


# ============================================================
# SIBERIAN TIGER EXPLORATION
# ============================================================

def tiger_exploration(
    solution,
    population
):

    new_solution = (
        solution.copy()
    )

    random_tiger = population[
        rng.integers(
            0,
            len(population)
        )
    ]

    # --------------------------------------------------------
    # Explore dimensions where another tiger differs
    # --------------------------------------------------------

    difference = np.where(
        solution != random_tiger
    )[0]

    if len(difference) > 0:

        number_to_change = max(
            1,
            int(
                0.20
                * len(difference)
            )
        )

        selected_dimensions = (
            rng.choice(
                difference,
                size=min(
                    number_to_change,
                    len(difference)
                ),
                replace=False
            )
        )

        new_solution[
            selected_dimensions
        ] = random_tiger[
            selected_dimensions
        ]

    return new_solution


# ============================================================
# SIBERIAN TIGER EXPLOITATION
# ============================================================

def tiger_exploitation(
    solution,
    best_solution,
    iteration
):

    new_solution = (
        solution.copy()
    )

    progress = (
        iteration
        /
        NUM_ITERATIONS
    )

    # Probability of moving toward best tiger increases
    # as optimization progresses.
    follow_probability = (
        0.30
        +
        0.60 * progress
    )

    mask = (
        rng.random(
            len(solution)
        )
        < follow_probability
    )

    new_solution[
        mask
    ] = best_solution[
        mask
    ]

    return new_solution


# ============================================================
# BGSTO OPTIMIZER
# ============================================================

def run_bgsto(
    candidate_df
):

    logger.info(
        "Starting BGSTO optimization..."
    )

    num_stocks = len(
        candidate_df
    )

    stock_scores = (
        calculate_stock_scores(
            candidate_df
        )
    )

    population = (
        initialize_population(
            num_stocks,
            stock_scores
        )
    )

    best_solution = None

    best_fitness = -np.inf

    history = []

    # ========================================================
    # ITERATIONS
    # ========================================================

    for iteration in range(
        1,
        NUM_ITERATIONS + 1
    ):

        fitness_values = np.array(
            [
                fitness(
                    solution,
                    stock_scores,
                    candidate_df
                )
                for solution
                in population
            ]
        )

        iteration_best_index = int(
            np.argmax(
                fitness_values
            )
        )

        iteration_best_fitness = float(
            fitness_values[
                iteration_best_index
            ]
        )

        if (
            iteration_best_fitness
            > best_fitness
        ):

            best_fitness = (
                iteration_best_fitness
            )

            best_solution = (
                population[
                    iteration_best_index
                ].copy()
            )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history.append(
            {
                "iteration": iteration,
                "best_fitness": best_fitness,
                "iteration_best_fitness":
                    iteration_best_fitness,
                "average_fitness":
                    float(
                        np.mean(
                            fitness_values
                        )
                    ),
                "selected_stocks":
                    int(
                        best_solution.sum()
                    ),
            }
        )

        if (
            iteration == 1
            or iteration % 10 == 0
            or iteration == NUM_ITERATIONS
        ):

            logger.info(
                "Iteration %d/%d | "
                "Best fitness: %.6f | "
                "Selected stocks: %d",
                iteration,
                NUM_ITERATIONS,
                best_fitness,
                int(
                    best_solution.sum()
                )
            )

        # ====================================================
        # GENERATE NEW POPULATION
        # ====================================================

        new_population = []

        # ----------------------------------------------------
        # Elitism
        # ----------------------------------------------------

        new_population.append(
            best_solution.copy()
        )

        # ----------------------------------------------------
        # Hybrid Genetic + Tiger Optimization
        # ----------------------------------------------------

        while (
            len(new_population)
            < POPULATION_SIZE
        ):

            parent1 = (
                tournament_selection(
                    population,
                    fitness_values
                )
            )

            parent2 = (
                tournament_selection(
                    population,
                    fitness_values
                )
            )

            child1, child2 = (
                crossover(
                    parent1,
                    parent2
                )
            )

            child1 = mutate(
                child1
            )

            child2 = mutate(
                child2
            )

            # -----------------------------------------------
            # Early iterations:
            # stronger exploration
            #
            # Later iterations:
            # stronger exploitation
            # -----------------------------------------------

            exploration_probability = (
                1
                -
                iteration
                / NUM_ITERATIONS
            )

            if (
                rng.random()
                < exploration_probability
            ):

                child1 = tiger_exploration(
                    child1,
                    population
                )

            else:

                child1 = tiger_exploitation(
                    child1,
                    best_solution,
                    iteration
                )

            if (
                rng.random()
                < exploration_probability
            ):

                child2 = tiger_exploration(
                    child2,
                    population
                )

            else:

                child2 = tiger_exploitation(
                    child2,
                    best_solution,
                    iteration
                )

            # -----------------------------------------------
            # Repair portfolio-size constraints
            # -----------------------------------------------

            child1 = repair_solution(
                child1,
                stock_scores
            )

            child2 = repair_solution(
                child2,
                stock_scores
            )

            new_population.append(
                child1
            )

            if (
                len(new_population)
                < POPULATION_SIZE
            ):

                new_population.append(
                    child2
                )

        population = np.array(
            new_population,
            dtype=np.int8
        )

    logger.info(
        "BGSTO optimization completed."
    )

    logger.info(
        "Final best fitness: %.6f",
        best_fitness
    )

    return (
        best_solution,
        best_fitness,
        stock_scores,
        pd.DataFrame(history)
    )


# ============================================================
# BUILD FINAL SELECTED STOCK TABLE
# ============================================================

def build_selected_stocks(
    candidates,
    solution,
    stock_scores,
    best_fitness
):

    selected_indices = np.where(
        solution == 1
    )[0]

    selected = candidates.iloc[
        selected_indices
    ].copy()

    selected[
        "bgsto_stock_score"
    ] = stock_scores[
        selected_indices
    ]

    selected[
        "bgsto_fitness"
    ] = best_fitness

    selected = (
        selected
        .sort_values(
            [
                "bgsto_stock_score",
                "buy_probability"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    selected[
        "bgsto_rank"
    ] = (
        np.arange(
            1,
            len(selected) + 1
        )
    )

    # --------------------------------------------------------
    # Useful final columns
    # --------------------------------------------------------

    preferred_columns = [
        "bgsto_rank",
        "date",
        "symbol",
        "buy_probability",
        "bgsto_stock_score",
        "return_21d",
        "return_63d",
        "volatility_60d",
        "sharpe_252d",
        "rsi_14",
        "bgsto_fitness",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in selected.columns
    ]

    return selected[
        columns
    ]


# ============================================================
# REPORT
# ============================================================

def print_report(
    selected,
    candidate_count,
    best_fitness,
    selection_date
):

    logger.info(
        "========== BGSTO REPORT =========="
    )

    logger.info(
        "Selection date: %s",
        selection_date.date()
    )

    logger.info(
        "Candidate stocks: %d",
        candidate_count
    )

    logger.info(
        "Selected stocks: %d",
        len(selected)
    )

    logger.info(
        "Best fitness: %.6f",
        best_fitness
    )

    logger.info(
        "Average BUY probability: %.4f",
        selected[
            "buy_probability"
        ].mean()
    )

    if (
        "return_63d"
        in selected.columns
    ):

        logger.info(
            "Average 63-day return: %.4f",
            selected[
                "return_63d"
            ].mean()
        )

    if (
        "volatility_60d"
        in selected.columns
    ):

        logger.info(
            "Average volatility: %.4f",
            selected[
                "volatility_60d"
            ].mean()
        )

    if (
        "sharpe_252d"
        in selected.columns
    ):

        logger.info(
            "Average Sharpe: %.4f",
            selected[
                "sharpe_252d"
            ].mean()
        )

    logger.info(
        "Selected symbols:\n%s",
        selected[
            [
                "bgsto_rank",
                "symbol",
                "buy_probability",
                "bgsto_stock_score"
            ]
        ].to_string(
            index=False
        )
    )

    logger.info(
        "=================================="
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_results(
    selected,
    history
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    selected.to_csv(
        CSV_OUTPUT,
        index=False
    )

    selected.to_parquet(
        PARQUET_OUTPUT,
        index=False
    )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

    logger.info(
        "Selected stocks CSV saved to: %s",
        CSV_OUTPUT
    )

    logger.info(
        "Selected stocks Parquet saved to: %s",
        PARQUET_OUTPUT
    )

    logger.info(
        "Optimization history saved to: %s",
        HISTORY_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting BGSTO stock optimization..."
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    predictions, features = (
        load_data()
    )

    # --------------------------------------------------------
    # Build candidate universe
    # --------------------------------------------------------

    candidates, selection_date = (
        build_candidate_dataset(
            predictions,
            features
        )
    )

    # --------------------------------------------------------
    # Normalize fitness features
    # --------------------------------------------------------

    candidates = (
        prepare_fitness_features(
            candidates
        )
    )

    # --------------------------------------------------------
    # Run optimizer
    # --------------------------------------------------------

    (
        best_solution,
        best_fitness,
        stock_scores,
        history
    ) = run_bgsto(
        candidates
    )

    # --------------------------------------------------------
    # Extract final stocks
    # --------------------------------------------------------

    selected = (
        build_selected_stocks(
            candidates,
            best_solution,
            stock_scores,
            best_fitness
        )
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        selected,
        len(candidates),
        best_fitness,
        selection_date
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        selected,
        history
    )

    logger.info(
        "BGSTO stock optimization "
        "completed successfully."
    )


if __name__ == "__main__":
    main()