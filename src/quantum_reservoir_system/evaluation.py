"""Forecast metrics and diagnostic evaluation.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def forecast_metrics(
    model: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    persistence: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
) -> dict[str, Any]:
    actual_values = np.asarray(actual, dtype=float)
    predictions = np.asarray(predicted, dtype=float)
    baseline = np.asarray(persistence, dtype=float)
    if actual_values.shape != predictions.shape or actual_values.shape != baseline.shape:
        raise ValueError("actual, predicted, and persistence must share a shape")
    if actual_values.ndim != 1 or len(actual_values) < 2:
        raise ValueError("forecast arrays must be one-dimensional with at least two rows")
    if not all(np.isfinite(values).all() for values in (actual_values, predictions, baseline)):
        raise ValueError("forecast arrays must be finite")
    rmse = float(np.sqrt(mean_squared_error(actual_values, predictions)))
    mae = float(mean_absolute_error(actual_values, predictions))
    persistence_rmse = float(np.sqrt(mean_squared_error(actual_values, baseline)))
    actual_direction = np.sign(actual_values[1:] - actual_values[:-1])
    predicted_direction = np.sign(predictions[1:] - actual_values[:-1])
    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "nrmse_std": float(rmse / max(np.std(actual_values), 1e-12)),
        "mape_percent": float(100 * mean_absolute_percentage_error(actual_values, predictions)),
        "r2": float(r2_score(actual_values, predictions)),
        "bias": float(np.mean(predictions - actual_values)),
        "directional_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "skill_vs_persistence_rmse": float(1.0 - rmse / max(persistence_rmse, 1e-12)),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_sample": float(1_000 * inference_seconds / len(actual_values)),
    }


def select_ridge_alpha(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in alphas:
        start = perf_counter()
        model = Ridge(alpha=alpha).fit(train_features, train_targets)
        prediction = model.predict(validation_features)
        rows.append(
            {
                "alpha": float(alpha),
                "validation_rmse": float(
                    np.sqrt(mean_squared_error(validation_targets, prediction))
                ),
                "fit_seconds": float(perf_counter() - start),
            }
        )
    history = pd.DataFrame(rows).sort_values(["validation_rmse", "alpha"]).reset_index(drop=True)
    return float(history.iloc[0]["alpha"]), history


def memory_capacity_curve(
    feature_sets: dict[str, np.ndarray],
    inputs: np.ndarray,
    train_end: int,
    validation_end: int,
    max_delay: int = 20,
    alpha: float = 1e-3,
) -> pd.DataFrame:
    """Estimate delayed-input reconstruction R² on the held-out test block."""

    values = np.asarray(inputs, dtype=float)
    rows = []
    for name, features in feature_sets.items():
        matrix = np.asarray(features, dtype=float)
        if len(matrix) != len(values):
            raise ValueError("feature and input lengths must match")
        for delay in range(1, max_delay + 1):
            indices = np.arange(delay, len(values))
            delayed = values[indices - delay]
            train_mask = indices < validation_end
            test_mask = indices >= validation_end
            model = Ridge(alpha=alpha).fit(matrix[indices[train_mask]], delayed[train_mask])
            prediction = model.predict(matrix[indices[test_mask]])
            score = float(r2_score(delayed[test_mask], prediction))
            rows.append(
                {
                    "reservoir": name,
                    "delay": delay,
                    "test_r2": score,
                    "positive_capacity": max(0.0, score),
                    "fit_rows": int(np.sum(train_mask)),
                    "train_boundary": train_end,
                }
            )
    return pd.DataFrame(rows)


def timed_predict(function: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = perf_counter()
    values = np.asarray(function(), dtype=float)
    return values, float(perf_counter() - start)
