"""Saved-model recursive forecasting for history CSV files.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .forecasting import recursive_forecasts
from .reservoir import QuantumReservoirModel


def validate_history_frame(frame: pd.DataFrame, minimum_rows: int) -> np.ndarray:
    if "value" not in frame.columns:
        raise ValueError("Input CSV must contain a value column")
    values = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=float)
    if len(values) < minimum_rows:
        raise ValueError(f"Input history must contain at least {minimum_rows} rows")
    if not np.isfinite(values).all():
        raise ValueError("Input history values must be finite")
    return values


def forecast_frame(
    artifact_directory: str | Path,
    history_frame: pd.DataFrame,
    horizon: int = 24,
) -> pd.DataFrame:
    artifact_path = Path(artifact_directory)
    quantum_model = QuantumReservoirModel.from_dict(
        json.loads((artifact_path / "quantum_reservoir.json").read_text(encoding="utf-8"))
    )
    preprocessor = joblib.load(artifact_path / "preprocessor.joblib")
    classical_models = joblib.load(artifact_path / "classical_models.joblib")
    history = validate_history_frame(history_frame, len(classical_models.lag_names))
    return recursive_forecasts(
        quantum_model,
        classical_models,
        preprocessor,
        history,
        horizon,
    )


def forecast_csv(
    artifact_directory: str | Path,
    input_csv: str | Path,
    horizon: int = 24,
) -> pd.DataFrame:
    return forecast_frame(artifact_directory, pd.read_csv(input_csv), horizon)
