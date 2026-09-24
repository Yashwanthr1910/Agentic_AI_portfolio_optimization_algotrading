"""
decision_tree_selector.py

V2.2 - Decision Tree based stock selection.

Input:
    data/processed/labeled_features.parquet

Outputs:
    models/decision_tree/decision_tree.pkl
    data/outputs/decision_tree_predictions.parquet
    reports/results/decision_tree_metrics.csv

Purpose:
    Train a Decision Tree classifier to identify potential
    BUY candidates before BGSTO optimization.
"""

from pathlib import Path
import logging
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_features.parquet"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "decision_tree"
)

MODEL_FILE = (
    MODEL_DIR
    / "decision_tree.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "decision_tree_predictions.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
)

METRICS_FILE = (
    REPORT_DIR
    / "decision_tree_metrics.csv"
)

FEATURE_IMPORTANCE_FILE = (
    REPORT_DIR
    / "decision_tree_feature_importance.csv"
)

FEATURE_LIST_FILE = (
    MODEL_DIR
    / "feature_columns.json"
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
# MODEL CONFIG
# ============================================================

RANDOM_STATE = 42

MAX_DEPTH = 8

MIN_SAMPLES_SPLIT = 100

MIN_SAMPLES_LEAF = 50

CLASS_WEIGHT = "balanced"


# ============================================================
# COLUMNS THAT MUST NOT BE MODEL FEATURES
# ============================================================

EXCLUDED_COLUMNS = {
    "date",
    "symbol",

    # Original price columns can remain excluded initially
    # because engineered features already describe the price action.
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",

    # Future information - NEVER use as model input.
    "future_price_21d",
    "future_return_21d",

    # Target / metadata.
    "target",
    "dataset_split",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {INPUT_FILE}"
        )

    logger.info("Loading labeled feature dataset...")

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    logger.info(
        "Loaded %d rows and %d stocks.",
        len(df),
        df["symbol"].nunique()
    )

    return df


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

def get_feature_columns(
    df: pd.DataFrame
) -> list[str]:

    feature_columns = []

    for column in df.columns:

        if column in EXCLUDED_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    logger.info(
        "Selected %d model features.",
        len(feature_columns)
    )

    return feature_columns


# ============================================================
# SPLIT DATA
# ============================================================

def split_dataset(
    df: pd.DataFrame
):

    train_df = df[
        df["dataset_split"] == "train"
    ].copy()

    backtest_df = df[
        df["dataset_split"] == "backtest"
    ].copy()

    validation_df = df[
        df["dataset_split"] == "validation"
    ].copy()

    logger.info(
        "Train rows: %d",
        len(train_df)
    )

    logger.info(
        "Backtest rows: %d",
        len(backtest_df)
    )

    logger.info(
        "Validation rows: %d",
        len(validation_df)
    )

    return (
        train_df,
        backtest_df,
        validation_df
    )


# ============================================================
# PREPARE MODEL DATA
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str]
):

    X = df[
        feature_columns
    ].copy()

    y = df[
        "target"
    ].astype(int).copy()

    # DecisionTreeClassifier does not accept NaN values.
    #
    # We intentionally DO NOT fill missing values with zero,
    # because zero has a financial meaning.
    #
    # For this V2 baseline, rows that do not yet have enough
    # historical data for rolling features are removed.

    valid_mask = X.notna().all(axis=1)

    X = X.loc[
        valid_mask
    ].copy()

    y = y.loc[
        valid_mask
    ].copy()

    metadata = df.loc[
        valid_mask,
        [
            "date",
            "symbol",
            "adj_close",
            "future_return_21d",
            "target",
            "dataset_split",
        ]
    ].copy()

    return (
        X,
        y,
        metadata
    )


# ============================================================
# TRAIN DECISION TREE
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
):

    logger.info(
        "Training Decision Tree..."
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "Decision Tree training completed."
    )

    return model


# ============================================================
# PREDICT
# ============================================================

def make_predictions(
    model,
    X: pd.DataFrame,
    metadata: pd.DataFrame
) -> pd.DataFrame:

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    output = metadata.copy()

    output["predicted_target"] = predictions

    # Probability for class 1 = BUY
    if 1 in model.classes_:

        buy_index = list(
            model.classes_
        ).index(1)

        output["buy_probability"] = (
            probabilities[:, buy_index]
        )

    else:

        output["buy_probability"] = 0.0

    return output


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    dataset_name: str,
    y_true: pd.Series,
    prediction_df: pd.DataFrame
) -> dict:

    y_pred = prediction_df[
        "predicted_target"
    ]

    buy_probability = prediction_df[
        "buy_probability"
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            buy_probability
        )

    except ValueError:

        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    metrics = {
        "dataset": dataset_name,
        "rows": len(y_true),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    logger.info(
        "========== %s METRICS ==========",
        dataset_name.upper()
    )

    logger.info(
        "Accuracy: %.4f",
        accuracy
    )

    logger.info(
        "Precision: %.4f",
        precision
    )

    logger.info(
        "Recall: %.4f",
        recall
    )

    logger.info(
        "Specificity: %.4f",
        specificity
    )

    logger.info(
        "F1 Score: %.4f",
        f1
    )

    logger.info(
        "ROC-AUC: %.4f",
        roc_auc
    )

    logger.info(
        "\n%s",
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return metrics


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_columns
):

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    logger.info(
        "Feature importance saved to: %s",
        FEATURE_IMPORTANCE_FILE
    )

    logger.info(
        "Top 10 Decision Tree features:\n%s",
        importance_df.head(10)
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_columns
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    with open(
        FEATURE_LIST_FILE,
        "w"
    ) as file:

        json.dump(
            feature_columns,
            file,
            indent=4
        )

    logger.info(
        "Decision Tree model saved to: %s",
        MODEL_FILE
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    predictions
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_parquet(
        PREDICTION_FILE,
        index=False
    )

    logger.info(
        "Predictions saved to: %s",
        PREDICTION_FILE
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    logger.info(
        "Metrics saved to: %s",
        METRICS_FILE
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    logger.info(
        "Starting Decision Tree stock selection..."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    feature_columns = get_feature_columns(
        df
    )

    # --------------------------------------------------------
    # Chronological splits
    # --------------------------------------------------------

    (
        train_df,
        backtest_df,
        validation_df
    ) = split_dataset(df)

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        train_meta
    ) = prepare_xy(
        train_df,
        feature_columns
    )

    (
        X_backtest,
        y_backtest,
        backtest_meta
    ) = prepare_xy(
        backtest_df,
        feature_columns
    )

    (
        X_validation,
        y_validation,
        validation_meta
    ) = prepare_xy(
        validation_df,
        feature_columns
    )

    logger.info(
        "Usable training rows: %d",
        len(X_train)
    )

    logger.info(
        "Usable backtest rows: %d",
        len(X_backtest)
    )

    logger.info(
        "Usable validation rows: %d",
        len(X_validation)
    )

    # --------------------------------------------------------
    # Train ONLY on 2015-2018
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_predictions = make_predictions(
        model,
        X_train,
        train_meta
    )

    backtest_predictions = make_predictions(
        model,
        X_backtest,
        backtest_meta
    )

    validation_predictions = make_predictions(
        model,
        X_validation,
        validation_meta
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = []

    metrics.append(
        evaluate_model(
            "train",
            y_train,
            train_predictions
        )
    )

    metrics.append(
        evaluate_model(
            "backtest",
            y_backtest,
            backtest_predictions
        )
    )

    metrics.append(
        evaluate_model(
            "validation",
            y_validation,
            validation_predictions
        )
    )

    # --------------------------------------------------------
    # Combine prediction files
    # --------------------------------------------------------

    all_predictions = pd.concat(
        [
            train_predictions,
            backtest_predictions,
            validation_predictions,
        ],
        ignore_index=True
    )

    all_predictions = all_predictions.sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------

    logger.info(
        "========== BUY CANDIDATES =========="
    )

    for split in [
        "train",
        "backtest",
        "validation"
    ]:

        split_data = all_predictions[
            all_predictions[
                "dataset_split"
            ] == split
        ]

        candidates = split_data[
            split_data[
                "predicted_target"
            ] == 1
        ]

        logger.info(
            "%s: %d BUY predictions from %d rows (%.2f%%)",
            split,
            len(candidates),
            len(split_data),
            (
                len(candidates)
                / len(split_data)
                * 100
            )
            if len(split_data) > 0
            else 0
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        model,
        feature_columns
    )

    save_predictions(
        all_predictions
    )

    save_metrics(
        metrics
    )

    save_feature_importance(
        model,
        feature_columns
    )

    logger.info(
        "Decision Tree stock selection completed successfully."
    )


if __name__ == "__main__":
    main()