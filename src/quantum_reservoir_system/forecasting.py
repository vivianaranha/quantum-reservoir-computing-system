"""Recursive multi-step forecasting for saved reservoir models.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .classical import ClassicalModels
from .preprocessing import SeriesPreprocessor
from .reservoir import QuantumReservoirModel, ReservoirDynamics, simulate_reservoir


def recursive_forecasts(
    quantum_model: QuantumReservoirModel,
    classical_models: ClassicalModels,
    preprocessor: SeriesPreprocessor,
    history: np.ndarray,
    horizon: int,
    quantum_initial_density: np.ndarray | None = None,
    echo_initial_state: np.ndarray | None = None,
) -> pd.DataFrame:
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or len(values) < len(classical_models.lag_names):
        raise ValueError(
            f"history must contain at least {len(classical_models.lag_names)} finite values"
        )
    if not np.isfinite(values).all():
        raise ValueError("history values must be finite")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    q_history = values.tolist()
    e_history = values.tolist()
    ridge_history = values.tolist()
    forest_history = values.tolist()
    persistence_value = float(values[-1])

    if quantum_initial_density is None:
        q_context = preprocessor.transform_inputs(values[:-1])
        quantum_state = simulate_reservoir(q_context, quantum_model.dynamics).final_density
    else:
        quantum_state = np.asarray(quantum_initial_density, dtype=complex).copy()

    if echo_initial_state is None:
        e_context = preprocessor.transform_inputs(values[:-1])
        _, echo_state = classical_models.echo_state.simulate(e_context)
    else:
        echo_state = np.asarray(echo_initial_state, dtype=float).copy()

    rows = []
    for step in range(1, horizon + 1):
        q_scaled = preprocessor.transform_inputs(np.asarray([q_history[-1]]))
        q_trace = simulate_reservoir(
            q_scaled,
            quantum_model.dynamics,
            initial_density=quantum_state,
        )
        quantum_state = q_trace.final_density
        q_forecast = float(
            preprocessor.inverse_targets(quantum_model.predict_from_features(q_trace.features))[0]
        )
        q_history.append(q_forecast)

        e_scaled = float(preprocessor.transform_inputs(np.asarray([e_history[-1]]))[0])
        echo_state = classical_models.echo_state.advance(e_scaled, echo_state)
        e_forecast = float(
            preprocessor.inverse_targets(
                classical_models.echo_state.predict_from_states(echo_state[None, :])
            )[0]
        )
        e_history.append(e_forecast)

        ridge_lags = np.asarray(
            [ridge_history[-offset - 1] for offset in range(len(classical_models.lag_names))]
        )[None, :]
        ridge_forecast = float(
            preprocessor.inverse_targets(classical_models.autoregressive_ridge.predict(ridge_lags))[
                0
            ]
        )
        ridge_history.append(ridge_forecast)

        forest_lags = np.asarray(
            [forest_history[-offset - 1] for offset in range(len(classical_models.lag_names))]
        )[None, :]
        forest_forecast = float(
            preprocessor.inverse_targets(classical_models.random_forest.predict(forest_lags))[0]
        )
        forest_history.append(forest_forecast)

        rows.append(
            {
                "horizon_step": step,
                "quantum_reservoir": q_forecast,
                "echo_state_network": e_forecast,
                "autoregressive_ridge": ridge_forecast,
                "random_forest": forest_forecast,
                "persistence": persistence_value,
            }
        )
    return pd.DataFrame(rows)


def context_density(
    dynamics: ReservoirDynamics,
    preprocessor: SeriesPreprocessor,
    history_without_current: np.ndarray,
) -> np.ndarray:
    scaled = preprocessor.transform_inputs(history_without_current)
    return simulate_reservoir(scaled, dynamics).final_density
