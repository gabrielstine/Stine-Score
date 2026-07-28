from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {
    "session",
    "stream",
    "region",
    "phy_dir",
    "cluster_id",
    "label",
    "y",
    "session_duration_s",
    "time_bin_count",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
        and frame[column].nunique(dropna=True) > 1
    ]


def make_logistic(random_state: int = 0) -> BaseEstimator:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_boosted_additive(random_state: int = 0) -> BaseEstimator:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_depth=1,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_boosted_interactions(random_state: int = 0) -> BaseEstimator:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_depth=3,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


MODEL_FACTORIES: dict[str, Callable[[int], BaseEstimator]] = {
    "logistic": make_logistic,
    "boosted_additive": make_boosted_additive,
    "boosted_interactions": make_boosted_interactions,
}


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1.0 - probability)).reshape(-1, 1)


@dataclass
class PlattCalibrator:
    model: LogisticRegression

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit(raw_probability))[:, 1]


def fit_platt(raw_probability: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    model = LogisticRegression(C=1e6, max_iter=2_000)
    model.fit(_logit(raw_probability), y)
    return PlattCalibrator(model)


def _inner_oof_predictions(
    factory: Callable[[int], BaseEstimator],
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    n_splits = min(5, unique_groups.size)
    if n_splits < 2:
        raise ValueError("At least two training sessions are required for calibration")
    predictions = np.full(y.size, np.nan)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups)):
        model = factory(random_state + fold)
        model.fit(x.iloc[train_index], y[train_index])
        predictions[test_index] = model.predict_proba(x.iloc[test_index])[:, 1]
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Failed to generate all inner out-of-session predictions")
    return predictions


def nested_group_predictions(
    factory: Callable[[int], BaseEstimator],
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Honest outer-session raw and calibrated predictions."""
    raw_predictions = np.full(y.size, np.nan)
    calibrated_predictions = np.full(y.size, np.nan)
    outer = LeaveOneGroupOut()
    for fold, (train_index, test_index) in enumerate(outer.split(x, y, groups)):
        x_train, y_train = x.iloc[train_index], y[train_index]
        groups_train = groups[train_index]
        inner_raw = _inner_oof_predictions(
            factory,
            x_train,
            y_train,
            groups_train,
            random_state + fold * 100,
        )
        calibrator = fit_platt(inner_raw, y_train)
        model = factory(random_state + fold)
        model.fit(x_train, y_train)
        raw = model.predict_proba(x.iloc[test_index])[:, 1]
        raw_predictions[test_index] = raw
        calibrated_predictions[test_index] = calibrator.predict(raw)
    return raw_predictions, calibrated_predictions


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "precision_at_0_5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, predicted, zero_division=0)),
    }


def reliability_table(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, bins - 1)
    records = []
    for index in range(bins):
        selected = bin_id == index
        records.append(
            {
                "bin": index,
                "lower": edges[index],
                "upper": edges[index + 1],
                "n": int(selected.sum()),
                "mean_predicted_probability": float(np.mean(probability[selected]))
                if selected.any()
                else math.nan,
                "observed_good_fraction": float(np.mean(y[selected]))
                if selected.any()
                else math.nan,
            }
        )
    return pd.DataFrame(records)


def fit_final_calibrated_model(
    factory: Callable[[int], BaseEstimator],
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int = 0,
) -> dict[str, object]:
    calibration_predictions = _inner_oof_predictions(
        factory, x, y, groups, random_state
    )
    calibrator = fit_platt(calibration_predictions, y)
    model = factory(random_state)
    model.fit(x, y)
    return {"model": model, "calibrator": calibrator}
