"""Diagnostic plots for quantum reservoir forecasting.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLORS = {
    "quantum_reservoir": "#6D28D9",
    "echo_state_network": "#0F766E",
    "autoregressive_ridge": "#2563EB",
    "random_forest": "#DC2626",
    "persistence": "#6B7280",
}


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_series_splits(
    path: Path, supervised: pd.DataFrame, train_end: int, validation_end: int
) -> None:
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.plot(supervised["time_index"], supervised["value"], color="#334155", linewidth=1)
    axis.axvspan(
        supervised.iloc[0]["time_index"],
        supervised.iloc[train_end - 1]["time_index"],
        color="#DBEAFE",
        alpha=0.7,
        label="Train",
    )
    axis.axvspan(
        supervised.iloc[train_end]["time_index"],
        supervised.iloc[validation_end - 1]["time_index"],
        color="#FEF3C7",
        alpha=0.7,
        label="Validation",
    )
    axis.axvspan(
        supervised.iloc[validation_end]["time_index"],
        supervised.iloc[-1]["time_index"],
        color="#FCE7F3",
        alpha=0.7,
        label="Test",
    )
    axis.set_title("Mackey–Glass Series and Chronological Splits")
    axis.set_xlabel("Time index")
    axis.set_ylabel("Series value")
    axis.legend(ncol=3)
    axis.grid(alpha=0.2)
    _save(figure, path)


def plot_one_step_forecasts(path: Path, predictions: pd.DataFrame) -> None:
    display = predictions.iloc[: min(240, len(predictions))]
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.plot(display["target_time_index"], display["target"], color="#111827", label="Actual")
    for name in ("quantum_reservoir", "echo_state_network", "random_forest"):
        axis.plot(
            display["target_time_index"],
            display[name],
            color=COLORS[name],
            linewidth=1.1,
            alpha=0.85,
            label=name.replace("_", " ").title(),
        )
    axis.set_title("Held-Out One-Step Forecasts")
    axis.set_xlabel("Time index")
    axis.set_ylabel("Value")
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    _save(figure, path)


def plot_error_distributions(path: Path, predictions: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    for name, color in COLORS.items():
        sns.kdeplot(
            predictions[f"{name}_error"],
            label=name.replace("_", " ").title(),
            color=color,
            ax=axis,
            fill=False,
        )
    axis.axvline(0, color="#111827", linestyle="--")
    axis.set_title("Held-Out Forecast Error Distributions")
    axis.set_xlabel("Prediction error")
    axis.legend()
    axis.grid(alpha=0.2)
    _save(figure, path)


def plot_metric_comparison(path: Path, metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("rmse")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ordered["model"].str.replace("_", " ").str.title()
    colors = [COLORS[name] for name in ordered["model"]]
    axes[0].barh(labels, ordered["rmse"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_title("Held-Out RMSE")
    axes[0].set_xlabel("Lower is better")
    axes[1].barh(labels, ordered["r2"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("Held-Out R²")
    axes[1].set_xlabel("Higher is better")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    _save(figure, path)


def plot_recursive_rollout(path: Path, rollout: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.plot(rollout["horizon_step"], rollout["actual"], color="#111827", label="Actual")
    for name in (
        "quantum_reservoir",
        "echo_state_network",
        "autoregressive_ridge",
        "random_forest",
        "persistence",
    ):
        axis.plot(
            rollout["horizon_step"],
            rollout[name],
            color=COLORS[name],
            alpha=0.82,
            linewidth=1.1,
            label=name.replace("_", " ").title(),
        )
    axis.set_title("Recursive Multi-Step Forecast from the Test Boundary")
    axis.set_xlabel("Horizon step")
    axis.set_ylabel("Value")
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    _save(figure, path)


def plot_reservoir_heatmap(path: Path, quantum_features: np.ndarray) -> None:
    width = min(180, len(quantum_features))
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        quantum_features[-width:].T,
        cmap="coolwarm",
        center=0.0,
        cbar_kws={"label": "Expectation value"},
        ax=axis,
    )
    axis.set_title("Quantum Reservoir Virtual-Node Features")
    axis.set_xlabel("Recent time steps")
    axis.set_ylabel("Feature index")
    _save(figure, path)


def plot_memory_capacity(path: Path, memory: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5))
    for name, group in memory.groupby("reservoir"):
        axis.plot(
            group["delay"],
            group["test_r2"],
            marker="o",
            markersize=3,
            color=COLORS[name],
            label=name.replace("_", " ").title(),
        )
    axis.axhline(0, color="#6B7280", linestyle="--")
    axis.set_title("Held-Out Delayed-Input Memory Curve")
    axis.set_xlabel("Delay")
    axis.set_ylabel("Reconstruction R²")
    axis.legend()
    axis.grid(alpha=0.2)
    _save(figure, path)


def plot_shot_sensitivity(path: Path, sensitivity: pd.DataFrame) -> None:
    finite = sensitivity.loc[sensitivity["shots"] != "exact"].copy()
    finite["shots_numeric"] = pd.to_numeric(finite["shots"])
    exact_rmse = float(sensitivity.loc[sensitivity["shots"] == "exact", "rmse"].iloc[0])
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.plot(
        finite["shots_numeric"],
        finite["rmse"],
        marker="o",
        color=COLORS["quantum_reservoir"],
    )
    axis.axhline(exact_rmse, color="#111827", linestyle="--", label="Exact expectation")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_title("Approximate Finite-Shot Sensitivity")
    axis.set_xlabel("Shots per observable")
    axis.set_ylabel("Held-out RMSE")
    axis.legend()
    axis.grid(alpha=0.2)
    _save(figure, path)


def plot_couplings(path: Path, couplings: np.ndarray, fields: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    sns.heatmap(
        couplings,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=axes[0],
    )
    axes[0].set_title("XX Couplings")
    axes[0].set_xlabel("Qubit")
    axes[0].set_ylabel("Qubit")
    axes[1].bar(np.arange(len(fields)), fields, color="#7C3AED")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_title("Local Z Fields")
    axes[1].set_xlabel("Qubit")
    axes[1].set_ylabel("Field strength")
    axes[1].grid(axis="y", alpha=0.2)
    _save(figure, path)


def plot_readout_coefficients(
    path: Path, quantum_coefficients: np.ndarray, echo_coefficients: np.ndarray
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.8))
    axes[0].bar(
        np.arange(len(quantum_coefficients)),
        quantum_coefficients,
        color=COLORS["quantum_reservoir"],
    )
    axes[0].set_title("Quantum Reservoir Linear Readout")
    axes[1].bar(
        np.arange(len(echo_coefficients)),
        echo_coefficients,
        color=COLORS["echo_state_network"],
    )
    axes[1].set_title("Echo-State Linear Readout")
    for axis in axes:
        axis.axhline(0, color="#111827", linewidth=0.8)
        axis.set_ylabel("Coefficient")
        axis.grid(axis="y", alpha=0.2)
    axes[1].set_xlabel("Reservoir feature")
    _save(figure, path)


def plot_alpha_search(path: Path, history: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5))
    for name, group in history.groupby("model"):
        axis.plot(
            group["alpha"],
            group["validation_rmse"],
            marker="o",
            label=name.replace("_", " ").title(),
            color=COLORS[name],
        )
    axis.set_xscale("log")
    axis.set_title("Validation Selection of Ridge Regularization")
    axis.set_xlabel("Ridge alpha")
    axis.set_ylabel("Validation RMSE (normalized target)")
    axis.legend()
    axis.grid(alpha=0.2)
    _save(figure, path)
