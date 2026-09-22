"""Evaluation tests. Created by School of AI and School of QC."""

import numpy as np
import pytest

from quantum_reservoir_system.evaluation import (
    forecast_metrics,
    memory_capacity_curve,
    select_ridge_alpha,
)


def test_perfect_forecast_metrics():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    persistence = np.array([1.0, 1.0, 2.0, 3.0])
    metrics = forecast_metrics("perfect", actual, actual, persistence, 0.1, 0.01)
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["skill_vs_persistence_rmse"] == 1.0


def test_forecast_metric_validation():
    with pytest.raises(ValueError, match="shape"):
        forecast_metrics("bad", np.ones(3), np.ones(2), np.ones(3), 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        forecast_metrics("bad", np.array([1.0, np.nan]), np.ones(2), np.ones(2), 0.0, 0.0)


def test_ridge_alpha_history_is_sorted():
    x = np.arange(30, dtype=float)[:, None]
    y = 2 * x.ravel()
    alpha, history = select_ridge_alpha(x[:20], y[:20], x[20:], y[20:], (1.0, 0.01))
    assert alpha in {1.0, 0.01}
    assert history["validation_rmse"].is_monotonic_increasing


def test_memory_curve_recovers_delayed_signal():
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=200)
    features = np.column_stack([inputs, np.roll(inputs, 1), np.roll(inputs, 2)])
    memory = memory_capacity_curve(
        {"synthetic": features},
        inputs,
        train_end=100,
        validation_end=150,
        max_delay=2,
    )
    assert len(memory) == 2
    assert (memory["test_r2"] > 0.95).all()


def test_memory_curve_validates_lengths():
    with pytest.raises(ValueError, match="lengths"):
        memory_capacity_curve(
            {"bad": np.ones((10, 2))},
            np.ones(11),
            5,
            8,
        )
