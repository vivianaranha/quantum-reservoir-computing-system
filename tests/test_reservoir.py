"""Quantum reservoir tests. Created by School of AI and School of QC."""

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

from quantum_reservoir_system.readout import LinearReadout
from quantum_reservoir_system.reservoir import (
    QuantumReservoirModel,
    ReservoirDynamics,
    dynamics_diagnostics,
    initial_density_matrix,
    ising_hamiltonian,
    observable_operators,
    random_dynamics,
    reservoir_unitary,
    reset_input_qubit,
    simulate_reservoir,
)


def test_random_dynamics_is_seeded_and_symmetric():
    first = random_dynamics(4, 3, 0.8, 1.0, 0.7, 42)
    second = random_dynamics(4, 3, 0.8, 1.0, 0.7, 42)
    assert np.array_equal(first.couplings, second.couplings)
    assert np.array_equal(first.fields, second.fields)
    assert np.allclose(first.couplings, first.couplings.T)
    assert np.allclose(np.diag(first.couplings), 0.0)
    assert first.feature_count == 30


@pytest.mark.parametrize(
    "dynamics",
    [
        ReservoirDynamics(2, 1, 0.5, np.zeros((2, 2)), np.ones(2)),
        ReservoirDynamics(3, 2, 0.8, np.zeros((3, 3)), np.arange(3)),
    ],
)
def test_hamiltonian_is_hermitian_and_unitary_is_unitary(dynamics):
    matrix = np.asarray(ising_hamiltonian(dynamics).to_matrix())
    unitary = reservoir_unitary(dynamics)
    assert np.allclose(matrix, matrix.conj().T)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(len(unitary)), atol=1e-12)


def test_input_reset_matches_independent_qiskit_partial_trace():
    rng = np.random.default_rng(9)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state /= np.linalg.norm(state)
    density = np.outer(state, state.conj())
    actual = reset_input_qubit(density, 0.3, 3)
    reduced = np.asarray(partial_trace(DensityMatrix(density), [0]).data)
    expected = np.kron(reduced, np.diag([0.7, 0.3]))
    assert np.allclose(actual, expected)
    assert np.trace(actual) == pytest.approx(1.0)


def test_one_step_evolution_matches_qiskit_density_matrix():
    dynamics = random_dynamics(3, 1, 0.6, 0.7, 0.4, 12)
    injected = reset_input_qubit(initial_density_matrix(3), 0.4, 3)
    unitary = reservoir_unitary(dynamics)
    expected = DensityMatrix(injected).evolve(Operator(unitary)).data
    trace = simulate_reservoir(np.array([0.4]), dynamics)
    assert np.allclose(trace.final_density, expected, atol=1e-12)


def test_observables_have_expected_labels_and_eigenvalues():
    dynamics = random_dynamics(3, 2, 0.6, 0.7, 0.4, 12)
    labels, matrices = observable_operators(dynamics)
    assert labels == ["Z0", "Z1", "Z2", "Z0Z1", "Z0Z2", "Z1Z2"]
    assert matrices.shape == (6, 8, 8)
    for matrix in matrices:
        assert set(np.round(np.linalg.eigvalsh(matrix)).astype(int)) == {-1, 1}


@pytest.mark.parametrize(("qubits", "virtual_nodes"), [(2, 1), (3, 2), (4, 3)])
def test_simulation_preserves_density_and_feature_bounds(qubits, virtual_nodes):
    dynamics = random_dynamics(qubits, virtual_nodes, 0.8, 1.0, 0.7, 42)
    trace = simulate_reservoir(np.linspace(0.0, 1.0, 9), dynamics)
    assert trace.features.shape == (9, dynamics.feature_count)
    assert np.max(np.abs(trace.features)) <= 1.0 + 1e-12
    assert trace.trace_errors.max() < 1e-12
    assert np.all((trace.purities > 0) & (trace.purities <= 1.0 + 1e-12))
    assert np.allclose(trace.final_density, trace.final_density.conj().T)


def test_distinct_input_histories_produce_distinct_features():
    dynamics = random_dynamics(3, 2, 0.8, 1.0, 0.7, 42)
    zeros = simulate_reservoir(np.zeros(8), dynamics).features
    alternating = simulate_reservoir(np.tile([0.0, 1.0], 4), dynamics).features
    assert not np.allclose(zeros, alternating)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([]), "non-empty"),
        (np.ones((2, 2)), "one-dimensional"),
        (np.array([-0.1, 0.2]), r"in \[0, 1\]"),
        (np.array([0.1, np.nan]), "finite"),
    ],
)
def test_invalid_simulation_inputs_are_rejected(values, message):
    dynamics = random_dynamics(2, 1, 0.8, 1.0, 0.7, 42)
    with pytest.raises(ValueError, match=message):
        simulate_reservoir(values, dynamics)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qubits": 1},
        {"virtual_nodes": 0},
        {"time_delta": 0},
        {"couplings": np.ones((4, 4))},
        {"fields": np.ones(3)},
    ],
)
def test_invalid_dynamics_are_rejected(kwargs):
    values = {
        "qubits": 4,
        "virtual_nodes": 2,
        "time_delta": 0.8,
        "couplings": np.zeros((4, 4)),
        "fields": np.ones(4),
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        ReservoirDynamics(**values)


def test_model_serialization_round_trip_and_prediction():
    dynamics = random_dynamics(2, 1, 0.8, 1.0, 0.7, 42)
    readout = LinearReadout(
        alpha=0.1,
        feature_mean=np.zeros(dynamics.feature_count),
        feature_scale=np.ones(dynamics.feature_count),
        coefficients=np.arange(dynamics.feature_count, dtype=float),
        intercept=0.2,
    )
    original = QuantumReservoirModel(dynamics, readout)
    restored = QuantumReservoirModel.from_dict(original.to_dict())
    features = simulate_reservoir(np.array([0.2, 0.4]), dynamics).features
    assert np.allclose(
        restored.predict_from_features(features),
        original.predict_from_features(features),
    )


def test_diagnostics_report_zero_trainable_quantum_path_properties():
    diagnostics = dynamics_diagnostics(random_dynamics(4, 4, 0.8, 1.0, 0.7, 42))
    assert diagnostics["hilbert_dimension"] == 16
    assert diagnostics["feature_count"] == 40
    assert diagnostics["unitarity_error_frobenius"] < 1e-12
