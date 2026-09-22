"""Train-only scaling for reservoir inputs and forecast targets.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(slots=True)
class SeriesPreprocessor:
    input_scaler: MinMaxScaler = field(
        default_factory=lambda: MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
    )
    target_scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False

    def fit(self, train_inputs: np.ndarray, train_targets: np.ndarray) -> SeriesPreprocessor:
        inputs = _column(train_inputs, "train_inputs")
        targets = _column(train_targets, "train_targets")
        if len(inputs) != len(targets):
            raise ValueError("train inputs and targets must contain the same number of rows")
        self.input_scaler.fit(inputs)
        self.target_scaler.fit(targets)
        self.fitted = True
        return self

    def transform_inputs(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.input_scaler.transform(_column(values, "inputs")).ravel()

    def transform_targets(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.target_scaler.transform(_column(values, "targets")).ravel()

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.target_scaler.inverse_transform(_column(values, "normalized targets")).ravel()

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("preprocessor must be fitted before transformation")


def _column(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array
