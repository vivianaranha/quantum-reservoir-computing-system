"""Data tests. Created by School of AI and School of QC."""

import numpy as np
import pandas as pd
import pytest

from quantum_reservoir_system.data import (
    build_supervised_frame,
    chronological_boundaries,
    generate_mackey_glass,
    lag_columns,
    split_assignments,
)


def test_generator_is_deterministic_and_nonconstant():
    first = generate_mackey_glass(600, 100)
    second = generate_mackey_glass(600, 100)
    pd.testing.assert_frame_equal(first, second)
    assert first["value"].std() > 0.1
    assert np.isfinite(first["value"]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_count": 1},
        {"burn_in": -1},
        {"delay": 0},
        {"beta": 0},
        {"gamma": -1},
        {"exponent": 0},
        {"initial_value": 0},
    ],
)
def test_invalid_generator_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_mackey_glass(**kwargs)


def test_supervised_frame_is_strictly_causal():
    series = generate_mackey_glass(600, 100)
    frame = build_supervised_frame(series, 10)
    assert len(frame) == 590
    assert np.allclose(frame["lag_0"], frame["value"])
    assert np.allclose(frame["target"].iloc[:-1], frame["value"].iloc[1:])
    original = series.set_index("time_index")["value"]
    row = frame.iloc[20]
    assert row["lag_4"] == pytest.approx(original.loc[int(row["time_index"]) - 4])


def test_supervised_frame_validates_schema_and_values():
    with pytest.raises(ValueError, match="contain"):
        build_supervised_frame(pd.DataFrame({"value": [1.0]}), 3)
    bad = pd.DataFrame({"time_index": [0, 1], "value": [1.0, np.nan]})
    with pytest.raises(ValueError, match="finite"):
        build_supervised_frame(bad, 1)


def test_lag_columns_validation():
    assert lag_columns(3) == ("lag_0", "lag_1", "lag_2")
    with pytest.raises(ValueError):
        lag_columns(0)


def test_chronological_boundaries_and_assignments():
    frame = build_supervised_frame(generate_mackey_glass(600, 100), 10)
    boundaries = chronological_boundaries(len(frame), 0.6, 0.2)
    assignments = split_assignments(frame, boundaries)
    assert boundaries.train_end < boundaries.validation_end < boundaries.total
    assert assignments["split"].value_counts().to_dict() == {
        "train": 354,
        "test": 118,
        "validation": 118,
    }


def test_invalid_boundaries_are_rejected():
    with pytest.raises(ValueError):
        chronological_boundaries(10, 0.0, 0.0)
