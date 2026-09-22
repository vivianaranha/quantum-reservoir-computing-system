"""Command-line interface.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qrc-train",
        description="Run a quantum reservoir time-series benchmark.",
    )
    parser.add_argument("--config", type=Path, help="Optional JSON configuration")
    parser.add_argument("--samples", type=int, help="Generated Mackey–Glass sample count")
    parser.add_argument("--seed", type=int, help="Random seed for fixed reservoir dynamics")
    parser.add_argument("--qubits", type=int, help="Quantum reservoir qubits")
    parser.add_argument("--virtual-nodes", type=int, help="Measurements per input interval")
    parser.add_argument("--time-delta", type=float, help="Hamiltonian evolution time per input")
    parser.add_argument("--washout", type=int, help="Initial training states to discard")
    parser.add_argument("--lags", type=int, help="Explicit lags for classical controls")
    parser.add_argument("--rollout-horizon", type=int, help="Recursive forecast horizon")
    parser.add_argument("--output", type=str, help="Artifact root directory")
    return parser


def resolve_config(arguments: argparse.Namespace) -> ExperimentConfig:
    config = (
        ExperimentConfig.from_json(arguments.config)
        if arguments.config
        else ExperimentConfig().validate()
    )
    return config.with_overrides(
        sample_count=arguments.samples,
        random_seed=arguments.seed,
        qubits=arguments.qubits,
        virtual_nodes=arguments.virtual_nodes,
        time_delta=arguments.time_delta,
        washout=arguments.washout,
        lag_count=arguments.lags,
        rollout_horizon=arguments.rollout_horizon,
        output_root=arguments.output,
    )


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_experiment(
        resolve_config(arguments),
        progress_callback=lambda message: print(f"[qrc] {message}"),
    )
    columns = [
        "model",
        "mae",
        "rmse",
        "nrmse_std",
        "r2",
        "directional_accuracy",
        "skill_vs_persistence_rmse",
    ]
    print("\nHeld-out one-step metrics")
    print(result.metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nReservoir features: {result.quantum_model.dynamics.feature_count}")
    print(
        f"Trained quantum parameters: {result.diagnostics['quantum']['trained_quantum_parameters']}"
    )
    print(f"Artifacts: {result.output_directory.resolve()}")
    print("\nCreated by School of AI and School of QC")


if __name__ == "__main__":
    main()
