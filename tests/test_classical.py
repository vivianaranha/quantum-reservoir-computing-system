"""Classical control tests. Created by School of AI and School of QC."""

import numpy as np
import pytest

from quantum_reservoir_system.classical import EchoStateModel, initialize_echo_state
from quantum_reservoir_system.readout import LinearReadout


def placeholder(nodes):
    return LinearReadout(
        0.1,
        np.zeros(nodes),
        np.ones(nodes),
        np.ones(nodes),
        0.0,
    )


def test_echo_state_initialization_has_requested_radius():
    input_weights, recurrent = initialize_echo_state(20, 0.85, 0.3, 0.6, 42)
    assert input_weights.shape == (20, 2)
    assert recurrent.shape == (20, 20)
    assert np.max(np.abs(np.linalg.eigvals(recurrent))) == pytest.approx(0.85)


def test_echo_state_simulation_is_deterministic_and_stateful():
    input_weights, recurrent = initialize_echo_state(10, 0.9, 0.4, 0.6, 42)
    model = EchoStateModel(input_weights, recurrent, 0.4, placeholder(10))
    first, final = model.simulate(np.linspace(0, 1, 8))
    second, second_final = model.simulate(np.linspace(0, 1, 8))
    assert np.allclose(first, second)
    assert np.allclose(final, second_final)
    continued, _ = model.simulate(np.array([0.5]), initial_state=final)
    fresh, _ = model.simulate(np.array([0.5]))
    assert not np.allclose(continued, fresh)


def test_echo_state_prediction_uses_readout():
    input_weights, recurrent = initialize_echo_state(5, 0.9, 0.4, 0.6, 42)
    model = EchoStateModel(input_weights, recurrent, 0.4, placeholder(5))
    states, _ = model.simulate(np.array([0.1, 0.2]))
    assert np.allclose(model.predict_from_states(states), states.sum(axis=1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodes": 1},
        {"spectral_radius": 0.0},
    ],
)
def test_invalid_echo_initialization_is_rejected(kwargs):
    values = {
        "nodes": 10,
        "spectral_radius": 0.9,
        "leak_rate": 0.4,
        "input_scale": 0.6,
        "random_seed": 42,
    }
    values.update(kwargs)
    with pytest.raises((ValueError, FloatingPointError)):
        initialize_echo_state(**values)


def test_echo_state_rejects_bad_input_or_state():
    input_weights, recurrent = initialize_echo_state(5, 0.9, 0.4, 0.6, 42)
    model = EchoStateModel(input_weights, recurrent, 0.4, placeholder(5))
    with pytest.raises(ValueError):
        model.simulate(np.array([]))
    with pytest.raises(ValueError, match="state"):
        model.advance(0.2, np.zeros(4))
