"""Classical reservoir and autoregressive forecasting controls.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .data import SplitBoundaries, lag_columns
from .evaluation import select_ridge_alpha
from .readout import LinearReadout, fit_linear_readout


@dataclass(slots=True)
class EchoStateModel:
    input_weights: np.ndarray
    recurrent_weights: np.ndarray
    leak_rate: float
    readout: LinearReadout

    @property
    def nodes(self) -> int:
        return int(self.recurrent_weights.shape[0])

    def advance(self, input_value: float, state: np.ndarray) -> np.ndarray:
        previous = np.asarray(state, dtype=float)
        if previous.shape != (self.nodes,):
            raise ValueError(f"state must have shape ({self.nodes},)")
        if not np.isfinite(input_value):
            raise ValueError("input_value must be finite")
        drive = self.input_weights @ np.array([1.0, float(input_value)])
        proposal = np.tanh(drive + self.recurrent_weights @ previous)
        return (1.0 - self.leak_rate) * previous + self.leak_rate * proposal

    def simulate(
        self, inputs: np.ndarray, initial_state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(inputs, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("inputs must be a non-empty finite one-dimensional sequence")
        state = (
            np.zeros(self.nodes, dtype=float)
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        features = np.empty((len(values), self.nodes), dtype=float)
        for position, value in enumerate(values):
            state = self.advance(float(value), state)
            features[position] = state
        return features, state

    def predict_from_states(self, states: np.ndarray) -> np.ndarray:
        return self.readout.predict(states)


@dataclass(slots=True)
class ClassicalModels:
    echo_state: EchoStateModel
    autoregressive_ridge: Ridge
    random_forest: RandomForestRegressor
    lag_names: tuple[str, ...]


@dataclass(slots=True)
class ClassicalTrainingOutput:
    models: ClassicalModels
    echo_features: np.ndarray
    alpha_history: pd.DataFrame
    training_seconds: dict[str, float]


def initialize_echo_state(
    nodes: int,
    spectral_radius: float,
    leak_rate: float,
    input_scale: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if nodes < 2:
        raise ValueError("nodes must be at least 2")
    if spectral_radius <= 0 or not 0 < leak_rate <= 1 or input_scale <= 0:
        raise ValueError("spectral radius, leak rate, and input scale must be positive")
    rng = np.random.default_rng(random_seed)
    input_weights = rng.uniform(-input_scale, input_scale, size=(nodes, 2))
    recurrent = rng.normal(0.0, 1.0, size=(nodes, nodes))
    mask = rng.random((nodes, nodes)) < 0.15
    recurrent *= mask
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
    if radius < 1e-12:
        recurrent[0, 0] = 1.0
        radius = 1.0
    recurrent *= spectral_radius / radius
    return input_weights, recurrent


def train_classical_models(
    frame: pd.DataFrame,
    scaled_inputs: np.ndarray,
    normalized_targets: np.ndarray,
    boundaries: SplitBoundaries,
    reservoir_feature_count: int,
    washout: int,
    ridge_alphas: tuple[float, ...],
    spectral_radius: float,
    leak_rate: float,
    input_scale: float,
    forest_estimators: int,
    random_seed: int,
) -> ClassicalTrainingOutput:
    train_indices = np.arange(washout, boundaries.train_end)
    validation_indices = boundaries.validation_indices
    fit_indices = np.arange(washout, boundaries.validation_end)
    histories = []
    training_seconds: dict[str, float] = {}

    start = perf_counter()
    input_weights, recurrent = initialize_echo_state(
        reservoir_feature_count,
        spectral_radius,
        leak_rate,
        input_scale,
        random_seed + 101,
    )
    placeholder = LinearReadout(
        1.0,
        np.zeros(reservoir_feature_count),
        np.ones(reservoir_feature_count),
        np.zeros(reservoir_feature_count),
        0.0,
    )
    echo_model = EchoStateModel(input_weights, recurrent, leak_rate, placeholder)
    echo_features, _ = echo_model.simulate(scaled_inputs)
    echo_readout, echo_history = fit_linear_readout(
        echo_features,
        normalized_targets,
        train_indices,
        validation_indices,
        ridge_alphas,
    )
    echo_model.readout = echo_readout
    echo_history.insert(0, "model", "echo_state_network")
    histories.append(echo_history)
    training_seconds["echo_state_network"] = float(perf_counter() - start)

    lag_names = lag_columns(len([column for column in frame if column.startswith("lag_")]))
    lag_matrix = frame.loc[:, lag_names].to_numpy(dtype=float)
    start = perf_counter()
    alpha, ridge_history = select_ridge_alpha(
        lag_matrix[train_indices],
        normalized_targets[train_indices],
        lag_matrix[validation_indices],
        normalized_targets[validation_indices],
        ridge_alphas,
    )
    ridge_model = Ridge(alpha=alpha).fit(lag_matrix[fit_indices], normalized_targets[fit_indices])
    ridge_history.insert(0, "model", "autoregressive_ridge")
    histories.append(ridge_history)
    training_seconds["autoregressive_ridge"] = float(perf_counter() - start)

    start = perf_counter()
    forest = RandomForestRegressor(
        n_estimators=forest_estimators,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=random_seed,
        n_jobs=-1,
    ).fit(lag_matrix[fit_indices], normalized_targets[fit_indices])
    training_seconds["random_forest"] = float(perf_counter() - start)
    return ClassicalTrainingOutput(
        models=ClassicalModels(echo_model, ridge_model, forest, lag_names),
        echo_features=echo_features,
        alpha_history=pd.concat(histories, ignore_index=True),
        training_seconds=training_seconds,
    )


def classical_diagnostics(models: ClassicalModels) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvals(models.echo_state.recurrent_weights)
    return {
        "echo_state_nodes": models.echo_state.nodes,
        "echo_state_spectral_radius": float(np.max(np.abs(eigenvalues))),
        "echo_state_leak_rate": models.echo_state.leak_rate,
        "echo_state_readout_alpha": models.echo_state.readout.alpha,
        "autoregressive_lags": len(models.lag_names),
        "autoregressive_alpha": float(models.autoregressive_ridge.alpha),
        "random_forest_estimators": int(models.random_forest.n_estimators),
    }
