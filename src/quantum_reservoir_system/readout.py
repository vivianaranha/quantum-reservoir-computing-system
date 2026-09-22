"""Validated linear readout shared by quantum and classical reservoirs.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .evaluation import select_ridge_alpha


@dataclass(slots=True)
class LinearReadout:
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.coefficients):
            raise ValueError(f"features must have shape (samples, {len(self.coefficients)})")
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        scaled = (values - self.feature_mean) / self.feature_scale
        return scaled @ self.coefficients + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinearReadout:
        return cls(
            alpha=float(payload["alpha"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            intercept=float(payload["intercept"]),
        )


def fit_linear_readout(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[LinearReadout, pd.DataFrame]:
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(targets, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(values):
        raise ValueError("features must be two-dimensional and align with targets")
    train_scaler = StandardScaler().fit(matrix[train_indices])
    train_features = train_scaler.transform(matrix[train_indices])
    validation_features = train_scaler.transform(matrix[validation_indices])
    alpha, history = select_ridge_alpha(
        train_features,
        values[train_indices],
        validation_features,
        values[validation_indices],
        alphas,
    )
    fit_indices = np.concatenate([train_indices, validation_indices])
    final_scaler = StandardScaler().fit(matrix[fit_indices])
    final_features = final_scaler.transform(matrix[fit_indices])
    model = Ridge(alpha=alpha).fit(final_features, values[fit_indices])
    scale = np.asarray(final_scaler.scale_, dtype=float)
    scale[scale < 1e-12] = 1.0
    return (
        LinearReadout(
            alpha=alpha,
            feature_mean=np.asarray(final_scaler.mean_, dtype=float),
            feature_scale=scale,
            coefficients=np.asarray(model.coef_, dtype=float),
            intercept=float(model.intercept_),
        ),
        history,
    )
