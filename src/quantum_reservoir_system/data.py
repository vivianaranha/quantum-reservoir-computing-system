"""Deterministic chaotic time-series generation and causal framing.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    train_end: int
    validation_end: int
    total: int

    @property
    def train_indices(self) -> np.ndarray:
        return np.arange(0, self.train_end)

    @property
    def validation_indices(self) -> np.ndarray:
        return np.arange(self.train_end, self.validation_end)

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(self.validation_end, self.total)


def generate_mackey_glass(
    sample_count: int = 1_600,
    burn_in: int = 400,
    delay: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: int = 10,
    initial_value: float = 1.2,
) -> pd.DataFrame:
    """Generate the discrete Mackey–Glass delay system without random noise."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if burn_in < 0:
        raise ValueError("burn_in cannot be negative")
    if delay < 1:
        raise ValueError("delay must be positive")
    if beta <= 0 or gamma <= 0 or exponent < 1 or initial_value <= 0:
        raise ValueError("Mackey–Glass parameters must be positive")
    total = sample_count + burn_in + delay + 1
    values = np.full(total, float(initial_value), dtype=float)
    values[: delay + 1] += 0.01 * np.sin(np.arange(delay + 1))
    for position in range(delay, total - 1):
        delayed = values[position - delay]
        derivative = beta * delayed / (1.0 + delayed**exponent) - gamma * values[position]
        values[position + 1] = values[position] + derivative
    selected = values[burn_in + delay + 1 : burn_in + delay + 1 + sample_count]
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise RuntimeError("Mackey–Glass integration produced invalid values")
    return pd.DataFrame(
        {
            "time_index": np.arange(sample_count, dtype=int),
            "value": selected,
        }
    )


def lag_columns(lag_count: int) -> tuple[str, ...]:
    if lag_count < 1:
        raise ValueError("lag_count must be positive")
    return tuple(f"lag_{lag}" for lag in range(lag_count))


def build_supervised_frame(series: pd.DataFrame, lag_count: int) -> pd.DataFrame:
    """Create one-step targets using current and strictly past values only."""

    if not {"time_index", "value"}.issubset(series.columns):
        raise ValueError("series must contain time_index and value columns")
    values = pd.to_numeric(series["value"], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("series values must be finite")
    frame = series.loc[:, ["time_index", "value"]].copy()
    for lag, column in enumerate(lag_columns(lag_count)):
        frame[column] = values.shift(lag)
    frame["target"] = values.shift(-1)
    frame["target_time_index"] = series["time_index"].shift(-1)
    return frame.dropna().reset_index(drop=True)


def chronological_boundaries(
    row_count: int,
    train_fraction: float,
    validation_fraction: float,
) -> SplitBoundaries:
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError("fractions must leave non-empty train, validation, and test blocks")
    return SplitBoundaries(train_end, validation_end, row_count)


def split_assignments(frame: pd.DataFrame, boundaries: SplitBoundaries) -> pd.DataFrame:
    split = np.full(len(frame), "test", dtype=object)
    split[: boundaries.train_end] = "train"
    split[boundaries.train_end : boundaries.validation_end] = "validation"
    return pd.DataFrame(
        {
            "time_index": frame["time_index"].astype(int),
            "target_time_index": frame["target_time_index"].astype(int),
            "split": split,
        }
    )
