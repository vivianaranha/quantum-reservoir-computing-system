"""Artifact persistence and experiment reporting.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .reservoir import ising_hamiltonian
from .visualization import (
    plot_alpha_search,
    plot_couplings,
    plot_error_distributions,
    plot_memory_capacity,
    plot_metric_comparison,
    plot_one_step_forecasts,
    plot_readout_coefficients,
    plot_recursive_rollout,
    plot_reservoir_heatmap,
    plot_series_splits,
    plot_shot_sensitivity,
)


def create_run_directory(output_root: str | Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_root) / f"run-{timestamp}"
    suffix = 1
    while destination.exists():
        destination = Path(output_root) / f"run-{timestamp}-{suffix}"
        suffix += 1
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def environment_details() -> dict[str, Any]:
    packages = {}
    for package in (
        "qiskit",
        "scikit-learn",
        "scipy",
        "numpy",
        "pandas",
        "streamlit",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    rollout: pd.DataFrame,
    memory: pd.DataFrame,
    diagnostics: dict[str, Any],
    row_counts: dict[str, int],
) -> None:
    quantum = diagnostics["quantum"]
    lines = [
        "# Quantum Reservoir Computing System — Experiment Report",
        "",
        "**Created by School of AI and School of QC**",
        "",
        "## Protocol",
        "",
        f"- Supervised rows: {row_counts['total']}",
        f"- Training rows: {row_counts['train']}",
        f"- Validation rows: {row_counts['validation']}",
        f"- Test rows: {row_counts['test']}",
        "- Split policy: chronological train, validation, then untouched test",
        "",
        "## Held-out one-step forecasts",
        "",
    ]
    for row in metrics.to_dict(orient="records"):
        lines.extend(
            [
                f"### {str(row['model']).replace('_', ' ').title()}",
                "",
                f"- MAE: {row['mae']:.6f}",
                f"- RMSE: {row['rmse']:.6f}",
                f"- Normalized RMSE: {row['nrmse_std']:.6f}",
                f"- R-squared: {row['r2']:.6f}",
                f"- Directional accuracy: {row['directional_accuracy']:.4f}",
                f"- Skill versus persistence: {row['skill_vs_persistence_rmse']:.4f}",
                f"- Training seconds: {row['training_seconds']:.4f}",
                f"- Inference milliseconds per sample: {row['inference_ms_per_sample']:.4f}",
                "",
            ]
        )
    rollout_rmse = {}
    for name in (
        "quantum_reservoir",
        "echo_state_network",
        "autoregressive_ridge",
        "random_forest",
        "persistence",
    ):
        rollout_rmse[name] = float(np.sqrt(np.mean(np.square(rollout[f"{name}_error"]))))
    capacity = memory.groupby("reservoir")["positive_capacity"].sum().to_dict()
    lines.extend(
        [
            "## Quantum reservoir",
            "",
            f"- Qubits: {quantum['qubits']}",
            f"- Hilbert-space dimension: {quantum['hilbert_dimension']}",
            f"- Virtual nodes: {quantum['virtual_nodes']}",
            f"- Measured features per time step: {quantum['feature_count']}",
            f"- Fixed Hamiltonian terms: {quantum['hamiltonian_terms']}",
            f"- Trained quantum parameters: {quantum['trained_quantum_parameters']}",
            f"- Trained linear-readout parameters: {quantum['trained_readout_parameters']}",
            f"- Maximum density-trace error: {quantum['maximum_trace_error']:.3e}",
            "",
            "## Recursive rollout RMSE",
            "",
        ]
    )
    for name, value in rollout_rmse.items():
        lines.append(f"- {name.replace('_', ' ').title()}: {value:.6f}")
    lines.extend(["", "## Positive delayed-input memory capacity", ""])
    for name, value in capacity.items():
        lines.append(f"- {name.replace('_', ' ').title()}: {value:.6f}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Only the classical linear readout is trained for the quantum reservoir. The internal "
            "Hamiltonian is fixed after seeded initialization. Classical controls receive either "
            "a matched reservoir-state dimension or explicit lag features. The data are synthetic, "
            "the dynamics are exactly simulated, and these results do not establish quantum "
            "advantage or production forecasting performance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_all_artifacts(
    destination: Path,
    config: dict[str, Any],
    series: pd.DataFrame,
    supervised: pd.DataFrame,
    assignments: pd.DataFrame,
    boundaries: Any,
    preprocessor: Any,
    quantum_model: Any,
    classical_models: Any,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    rollout: pd.DataFrame,
    memory_capacity: pd.DataFrame,
    shot_sensitivity: pd.DataFrame,
    alpha_history: pd.DataFrame,
    quantum_features: np.ndarray,
    echo_features: np.ndarray,
    diagnostics: dict[str, Any],
    circuit: Any,
) -> None:
    save_json(destination / "config.json", config)
    save_json(destination / "environment.json", environment_details())
    save_json(destination / "diagnostics.json", diagnostics)
    save_json(
        destination / "dataset_summary.json",
        {
            "generator": "deterministic Mackey-Glass delay system",
            "raw_samples": len(series),
            "supervised_rows": len(supervised),
            "train_rows": boundaries.train_end,
            "validation_rows": boundaries.validation_end - boundaries.train_end,
            "test_rows": boundaries.total - boundaries.validation_end,
            "target": "one-step-ahead value",
        },
    )
    save_json(
        destination / "metrics.json",
        {"models": metrics.to_dict(orient="records")},
    )
    save_json(destination / "quantum_reservoir.json", quantum_model.to_dict())
    metrics.to_csv(destination / "metrics.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    rollout.to_csv(destination / "recursive_rollout.csv", index=False)
    memory_capacity.to_csv(destination / "memory_capacity.csv", index=False)
    shot_sensitivity.to_csv(destination / "shot_sensitivity.csv", index=False)
    alpha_history.to_csv(destination / "alpha_search.csv", index=False)
    series.to_csv(destination / "mackey_glass_series.csv", index=False)
    supervised.to_csv(destination / "supervised_frame.csv", index=False)
    assignments.to_csv(destination / "split_assignments.csv", index=False)
    pd.DataFrame(quantum_model.dynamics.couplings).to_csv(
        destination / "hamiltonian_couplings.csv", index=False
    )
    pd.DataFrame(
        {"qubit": np.arange(quantum_model.dynamics.qubits), "field": quantum_model.dynamics.fields}
    ).to_csv(destination / "hamiltonian_fields.csv", index=False)
    np.savez_compressed(
        destination / "reservoir_features.npz",
        quantum_features=quantum_features,
        echo_state_features=echo_features,
    )
    joblib.dump(preprocessor, destination / "preprocessor.joblib")
    joblib.dump(classical_models, destination / "classical_models.joblib")
    (destination / "hamiltonian.txt").write_text(
        str(ising_hamiltonian(quantum_model.dynamics)) + "\n", encoding="utf-8"
    )
    (destination / "reservoir_step_circuit.txt").write_text(
        str(circuit.draw(output="text", fold=120)) + "\n", encoding="utf-8"
    )

    plot_series_splits(
        destination / "series_splits.png",
        supervised,
        boundaries.train_end,
        boundaries.validation_end,
    )
    plot_one_step_forecasts(destination / "one_step_forecasts.png", predictions)
    plot_error_distributions(destination / "error_distributions.png", predictions)
    plot_metric_comparison(destination / "metric_comparison.png", metrics)
    plot_recursive_rollout(destination / "recursive_rollout.png", rollout)
    plot_reservoir_heatmap(destination / "quantum_reservoir_features.png", quantum_features)
    plot_memory_capacity(destination / "memory_capacity.png", memory_capacity)
    plot_shot_sensitivity(destination / "shot_sensitivity.png", shot_sensitivity)
    plot_couplings(
        destination / "hamiltonian_dynamics.png",
        quantum_model.dynamics.couplings,
        quantum_model.dynamics.fields,
    )
    plot_readout_coefficients(
        destination / "readout_coefficients.png",
        quantum_model.readout.coefficients,
        classical_models.echo_state.readout.coefficients,
    )
    plot_alpha_search(destination / "alpha_search.png", alpha_history)
    _write_report(
        destination / "experiment_report.md",
        metrics,
        rollout,
        memory_capacity,
        diagnostics,
        {
            "total": len(supervised),
            "train": boundaries.train_end,
            "validation": boundaries.validation_end - boundaries.train_end,
            "test": boundaries.total - boundaries.validation_end,
        },
    )
