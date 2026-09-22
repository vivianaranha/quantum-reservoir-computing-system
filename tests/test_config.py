"""Configuration tests. Created by School of AI and School of QC."""

import json

import pytest

from quantum_reservoir_system.config import ExperimentConfig


def test_default_config_is_valid():
    config = ExperimentConfig().validate()
    assert config.qubits == 4
    assert config.virtual_nodes == 4


def test_json_round_trip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(ExperimentConfig().to_dict()))
    assert ExperimentConfig.from_json(path) == ExperimentConfig()


def test_overrides_ignore_none():
    assert ExperimentConfig().with_overrides(sample_count=None).sample_count == 1_600


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_count", 599),
        ("burn_in", 99),
        ("train_fraction", 0.3),
        ("validation_fraction", 0.35),
        ("qubits", 1),
        ("qubits", 6),
        ("virtual_nodes", 0),
        ("time_delta", 0.0),
        ("coupling_scale", 0.0),
        ("field_scale", -1.0),
        ("washout", 9),
        ("lag_count", 2),
        ("rollout_horizon", 7),
        ("ridge_alphas", ()),
        ("ridge_alphas", (-1.0,)),
        ("esn_spectral_radius", 0.0),
        ("esn_leak_rate", 1.1),
        ("esn_input_scale", 0.0),
        ("random_forest_estimators", 49),
        ("shot_counts", (16,)),
        ("output_root", ""),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        ExperimentConfig().with_overrides(**{field: value})


def test_split_fraction_sum_is_validated():
    with pytest.raises(ValueError, match="cannot exceed"):
        ExperimentConfig(train_fraction=0.7, validation_fraction=0.25).validate()
