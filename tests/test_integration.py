"""Integration tests. Created by School of AI and School of QC."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from quantum_reservoir_system.config import ExperimentConfig
from quantum_reservoir_system.experiment import run_experiment
from quantum_reservoir_system.inference import forecast_frame


def small_config(tmp_path):
    return ExperimentConfig(
        sample_count=600,
        burn_in=100,
        qubits=2,
        virtual_nodes=1,
        washout=10,
        lag_count=5,
        rollout_horizon=8,
        random_forest_estimators=50,
        shot_counts=(64, 128),
        output_root=str(tmp_path),
    ).validate()


def test_experiment_saves_complete_reloadable_artifacts(tmp_path):
    result = run_experiment(small_config(tmp_path))
    assert len(result.metrics) == 5
    assert len(result.rollout) == 8
    assert result.quantum_model.dynamics.feature_count == 3
    required = [
        "quantum_reservoir.json",
        "classical_models.joblib",
        "preprocessor.joblib",
        "metrics.csv",
        "predictions.csv",
        "recursive_rollout.csv",
        "memory_capacity.csv",
        "shot_sensitivity.csv",
        "experiment_report.md",
    ]
    assert all((result.output_directory / name).exists() for name in required)
    assert len(list(result.output_directory.glob("*.png"))) == 11


def test_saved_inference_matches_in_memory_shape(tmp_path):
    result = run_experiment(small_config(tmp_path))
    history = result.series.tail(30)[["value"]]
    saved = forecast_frame(result.output_directory, history, horizon=6)
    direct = result.forecast(history["value"].to_numpy(), horizon=6)
    pd.testing.assert_frame_equal(saved, direct)
    assert np.isfinite(saved.drop(columns="horizon_step")).to_numpy().all()


def test_saved_json_contains_fixed_dynamics_and_readout(tmp_path):
    result = run_experiment(small_config(tmp_path))
    payload = json.loads((result.output_directory / "quantum_reservoir.json").read_text())
    assert payload["dynamics"]["qubits"] == 2
    assert payload["dynamics"]["feature_count"] == 3
    assert len(payload["readout"]["coefficients"]) == 3


def test_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "quantum_reservoir_system.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "quantum reservoir" in completed.stdout.lower()


def test_app_starts_headlessly():
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py").run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Quantum Reservoir Computing System"


def test_quickstart_notebook_is_valid_and_branded():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "quickstart.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 10
    assert "Created by School of AI and School of QC" in source
    assert "run_experiment" in source
