"""End-to-end quantum reservoir benchmark.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import create_run_directory, save_all_artifacts
from .classical import (
    ClassicalModels,
    classical_diagnostics,
    train_classical_models,
)
from .config import ExperimentConfig
from .data import (
    SplitBoundaries,
    build_supervised_frame,
    chronological_boundaries,
    generate_mackey_glass,
    split_assignments,
)
from .evaluation import (
    forecast_metrics,
    memory_capacity_curve,
    timed_predict,
)
from .forecasting import recursive_forecasts
from .preprocessing import SeriesPreprocessor
from .readout import fit_linear_readout
from .reservoir import (
    QuantumReservoirModel,
    dynamics_diagnostics,
    random_dynamics,
    simulate_reservoir,
    trotter_visualization_circuit,
)

ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class ExperimentResult:
    output_directory: Path
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    rollout: pd.DataFrame
    memory_capacity: pd.DataFrame
    shot_sensitivity: pd.DataFrame
    series: pd.DataFrame
    supervised: pd.DataFrame
    boundaries: SplitBoundaries
    preprocessor: SeriesPreprocessor
    quantum_model: QuantumReservoirModel
    classical_models: ClassicalModels
    quantum_features: np.ndarray
    echo_features: np.ndarray
    diagnostics: dict[str, Any]

    def forecast(self, history: np.ndarray, horizon: int = 24) -> pd.DataFrame:
        return recursive_forecasts(
            self.quantum_model,
            self.classical_models,
            self.preprocessor,
            history,
            horizon,
        )


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _shot_sensitivity(
    exact_features: np.ndarray,
    actual: np.ndarray,
    persistence: np.ndarray,
    model: QuantumReservoirModel,
    preprocessor: SeriesPreprocessor,
    shot_counts: tuple[int, ...],
    random_seed: int,
) -> pd.DataFrame:
    rows = []
    exact_prediction = preprocessor.inverse_targets(model.predict_from_features(exact_features))
    rows.append(
        {
            "shots": "exact",
            **forecast_metrics(
                "quantum_reservoir",
                actual,
                exact_prediction,
                persistence,
                0.0,
                0.0,
            ),
        }
    )
    rng = np.random.default_rng(random_seed + 909)
    probabilities = np.clip((exact_features + 1.0) / 2.0, 0.0, 1.0)
    for shots in shot_counts:
        counts = rng.binomial(shots, probabilities)
        noisy_features = 2.0 * counts / shots - 1.0
        prediction = preprocessor.inverse_targets(model.predict_from_features(noisy_features))
        rows.append(
            {
                "shots": int(shots),
                **forecast_metrics(
                    "quantum_reservoir",
                    actual,
                    prediction,
                    persistence,
                    0.0,
                    0.0,
                ),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    config: ExperimentConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ExperimentResult:
    resolved = (config or ExperimentConfig()).validate()
    destination = create_run_directory(resolved.output_root)

    _notify(progress_callback, "Generating the deterministic Mackey–Glass series")
    series = generate_mackey_glass(resolved.sample_count, resolved.burn_in)
    supervised = build_supervised_frame(series, resolved.lag_count)
    boundaries = chronological_boundaries(
        len(supervised), resolved.train_fraction, resolved.validation_fraction
    )
    assignments = split_assignments(supervised, boundaries)
    train_slice = slice(0, boundaries.train_end)
    inputs = supervised["value"].to_numpy(dtype=float)
    targets = supervised["target"].to_numpy(dtype=float)

    _notify(progress_callback, "Fitting input and target scaling on the training block only")
    preprocessor = SeriesPreprocessor().fit(inputs[train_slice], targets[train_slice])
    scaled_inputs = preprocessor.transform_inputs(inputs)
    normalized_targets = preprocessor.transform_targets(targets)
    train_indices = np.arange(resolved.washout, boundaries.train_end)
    validation_indices = boundaries.validation_indices
    test_indices = boundaries.test_indices

    _notify(progress_callback, "Driving the fixed Ising quantum reservoir")
    dynamics = random_dynamics(
        resolved.qubits,
        resolved.virtual_nodes,
        resolved.time_delta,
        resolved.coupling_scale,
        resolved.field_scale,
        resolved.random_seed,
    )
    q_train_start = perf_counter()
    prefix_trace = simulate_reservoir(
        scaled_inputs[: boundaries.validation_end],
        dynamics,
    )
    readout, q_alpha_history = fit_linear_readout(
        prefix_trace.features,
        normalized_targets[: boundaries.validation_end],
        train_indices,
        validation_indices,
        resolved.ridge_alphas,
    )
    quantum_model = QuantumReservoirModel(dynamics, readout)
    quantum_training_seconds = float(perf_counter() - q_train_start)
    q_inference_start = perf_counter()
    test_trace = simulate_reservoir(
        scaled_inputs[test_indices],
        dynamics,
        initial_density=prefix_trace.final_density,
    )
    quantum_test_normalized = quantum_model.predict_from_features(test_trace.features)
    quantum_inference_seconds = float(perf_counter() - q_inference_start)
    quantum_features = np.vstack([prefix_trace.features, test_trace.features])
    q_alpha_history.insert(0, "model", "quantum_reservoir")

    _notify(progress_callback, "Training matched and strong classical controls")
    classical_output = train_classical_models(
        supervised,
        scaled_inputs,
        normalized_targets,
        boundaries,
        dynamics.feature_count,
        resolved.washout,
        resolved.ridge_alphas,
        resolved.esn_spectral_radius,
        resolved.esn_leak_rate,
        resolved.esn_input_scale,
        resolved.random_forest_estimators,
        resolved.random_seed,
    )
    classical_models = classical_output.models
    lag_matrix = supervised.loc[:, classical_models.lag_names].to_numpy(dtype=float)

    predictions: dict[str, np.ndarray] = {}
    inference_seconds: dict[str, float] = {}
    predictions["quantum_reservoir"] = preprocessor.inverse_targets(quantum_test_normalized)
    inference_seconds["quantum_reservoir"] = quantum_inference_seconds
    echo_prediction, inference_seconds["echo_state_network"] = timed_predict(
        lambda: preprocessor.inverse_targets(
            classical_models.echo_state.predict_from_states(
                classical_output.echo_features[test_indices]
            )
        )
    )
    predictions["echo_state_network"] = echo_prediction
    ridge_prediction, inference_seconds["autoregressive_ridge"] = timed_predict(
        lambda: preprocessor.inverse_targets(
            classical_models.autoregressive_ridge.predict(lag_matrix[test_indices])
        )
    )
    predictions["autoregressive_ridge"] = ridge_prediction
    forest_prediction, inference_seconds["random_forest"] = timed_predict(
        lambda: preprocessor.inverse_targets(
            classical_models.random_forest.predict(lag_matrix[test_indices])
        )
    )
    predictions["random_forest"] = forest_prediction
    predictions["persistence"] = inputs[test_indices]
    inference_seconds["persistence"] = 0.0

    actual = targets[test_indices]
    persistence = predictions["persistence"]
    training_times = {
        "quantum_reservoir": quantum_training_seconds,
        **classical_output.training_seconds,
        "persistence": 0.0,
    }
    metrics = (
        pd.DataFrame(
            [
                forecast_metrics(
                    name,
                    actual,
                    values,
                    persistence,
                    training_times[name],
                    inference_seconds[name],
                )
                for name, values in predictions.items()
            ]
        )
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    prediction_frame = supervised.loc[
        test_indices, ["time_index", "target_time_index", "value", "target"]
    ].reset_index(drop=True)
    for name, values in predictions.items():
        prediction_frame[name] = values
        prediction_frame[f"{name}_error"] = values - actual

    _notify(progress_callback, "Running recursive rollout and memory diagnostics")
    history = np.asarray(
        [
            supervised.iloc[boundaries.validation_end][f"lag_{lag}"]
            for lag in reversed(range(resolved.lag_count))
        ],
        dtype=float,
    )
    _, echo_prefix_state = classical_models.echo_state.simulate(
        scaled_inputs[: boundaries.validation_end]
    )
    rollout = recursive_forecasts(
        quantum_model,
        classical_models,
        preprocessor,
        history,
        resolved.rollout_horizon,
        quantum_initial_density=prefix_trace.final_density,
        echo_initial_state=echo_prefix_state,
    )
    rollout["actual"] = targets[
        boundaries.validation_end : boundaries.validation_end + resolved.rollout_horizon
    ]
    for name in (
        "quantum_reservoir",
        "echo_state_network",
        "autoregressive_ridge",
        "random_forest",
        "persistence",
    ):
        rollout[f"{name}_error"] = rollout[name] - rollout["actual"]

    memory = memory_capacity_curve(
        {
            "quantum_reservoir": quantum_features,
            "echo_state_network": classical_output.echo_features,
        },
        scaled_inputs,
        boundaries.train_end,
        boundaries.validation_end,
        max_delay=min(20, resolved.lag_count),
    )
    shot_sensitivity = _shot_sensitivity(
        test_trace.features,
        actual,
        persistence,
        quantum_model,
        preprocessor,
        resolved.shot_counts,
        resolved.random_seed,
    )

    diagnostics = {
        "quantum": {
            **dynamics_diagnostics(dynamics),
            "maximum_trace_error": float(
                max(prefix_trace.trace_errors.max(), test_trace.trace_errors.max())
            ),
            "minimum_purity": float(min(prefix_trace.purities.min(), test_trace.purities.min())),
            "maximum_purity": float(max(prefix_trace.purities.max(), test_trace.purities.max())),
            "readout_alpha": quantum_model.readout.alpha,
            "trained_quantum_parameters": 0,
            "trained_readout_parameters": dynamics.feature_count + 1,
        },
        "classical": classical_diagnostics(classical_models),
    }
    alpha_history = pd.concat([q_alpha_history, classical_output.alpha_history], ignore_index=True)
    circuit = trotter_visualization_circuit(dynamics)
    _notify(progress_callback, "Saving models, forecasts, diagnostics, and plots")
    save_all_artifacts(
        destination=destination,
        config=resolved.to_dict(),
        series=series,
        supervised=supervised,
        assignments=assignments,
        boundaries=boundaries,
        preprocessor=preprocessor,
        quantum_model=quantum_model,
        classical_models=classical_models,
        metrics=metrics,
        predictions=prediction_frame,
        rollout=rollout,
        memory_capacity=memory,
        shot_sensitivity=shot_sensitivity,
        alpha_history=alpha_history,
        quantum_features=quantum_features,
        echo_features=classical_output.echo_features,
        diagnostics=diagnostics,
        circuit=circuit,
    )
    _notify(progress_callback, f"Complete: {destination}")
    return ExperimentResult(
        destination,
        metrics,
        prediction_frame,
        rollout,
        memory,
        shot_sensitivity,
        series,
        supervised,
        boundaries,
        preprocessor,
        quantum_model,
        classical_models,
        quantum_features,
        classical_output.echo_features,
        diagnostics,
    )
