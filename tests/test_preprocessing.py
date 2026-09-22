"""Preprocessing tests. Created by School of AI and School of QC."""

import numpy as np
import pytest

from quantum_reservoir_system.preprocessing import SeriesPreprocessor


def test_train_scaling_and_inverse_target_round_trip():
    inputs = np.array([1.0, 2.0, 3.0])
    targets = np.array([2.0, 4.0, 6.0])
    preprocessor = SeriesPreprocessor().fit(inputs, targets)
    assert np.allclose(preprocessor.transform_inputs(inputs), [0.0, 0.5, 1.0])
    transformed = preprocessor.transform_targets(targets)
    assert np.allclose(preprocessor.inverse_targets(transformed), targets)


def test_input_scaling_clips_future_extremes():
    preprocessor = SeriesPreprocessor().fit(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert np.array_equal(preprocessor.transform_inputs(np.array([0.0, 3.0])), [0.0, 1.0])


def test_transform_before_fit_is_rejected():
    with pytest.raises(RuntimeError, match="fitted"):
        SeriesPreprocessor().transform_inputs(np.array([1.0]))


@pytest.mark.parametrize(
    ("inputs", "targets"),
    [
        (np.array([]), np.array([])),
        (np.array([1.0, 2.0]), np.array([1.0])),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0])),
        (np.ones((2, 2)), np.array([1.0, 2.0])),
    ],
)
def test_invalid_fit_values_are_rejected(inputs, targets):
    with pytest.raises(ValueError):
        SeriesPreprocessor().fit(inputs, targets)
