"""Readout tests. Created by School of AI and School of QC."""

import numpy as np
import pytest

from quantum_reservoir_system.readout import LinearReadout, fit_linear_readout


def test_readout_fit_selects_alpha_and_predicts():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(100, 5))
    targets = 0.7 * features[:, 0] - 0.2 * features[:, 2]
    readout, history = fit_linear_readout(
        features,
        targets,
        np.arange(0, 60),
        np.arange(60, 80),
        (1e-6, 0.01, 1.0),
    )
    assert readout.alpha in {1e-6, 0.01, 1.0}
    assert len(history) == 3
    assert np.mean((readout.predict(features[80:]) - targets[80:]) ** 2) < 0.01


def test_readout_serialization_round_trip():
    original = LinearReadout(0.1, np.arange(3), np.ones(3), np.arange(3), 0.4)
    restored = LinearReadout.from_dict(original.to_dict())
    values = np.ones((2, 3))
    assert np.array_equal(restored.predict(values), original.predict(values))


@pytest.mark.parametrize(
    "features",
    [np.ones(3), np.ones((2, 2)), np.full((2, 3), np.nan)],
)
def test_readout_rejects_invalid_features(features):
    model = LinearReadout(0.1, np.zeros(3), np.ones(3), np.ones(3), 0.0)
    with pytest.raises(ValueError):
        model.predict(features)
