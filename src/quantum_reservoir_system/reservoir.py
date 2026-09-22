"""Fixed quantum dynamics and trainable linear readout.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.linalg import expm

from .readout import LinearReadout


@dataclass(frozen=True, slots=True)
class ReservoirDynamics:
    qubits: int
    virtual_nodes: int
    time_delta: float
    couplings: np.ndarray
    fields: np.ndarray

    def __post_init__(self) -> None:
        if not 2 <= self.qubits <= 5:
            raise ValueError("qubits must be between 2 and 5")
        if self.virtual_nodes < 1:
            raise ValueError("virtual_nodes must be positive")
        if self.time_delta <= 0:
            raise ValueError("time_delta must be positive")
        couplings = np.asarray(self.couplings, dtype=float)
        fields = np.asarray(self.fields, dtype=float)
        if couplings.shape != (self.qubits, self.qubits):
            raise ValueError("couplings must be a square matrix matching qubits")
        if fields.shape != (self.qubits,):
            raise ValueError("fields must contain one value per qubit")
        if not np.isfinite(couplings).all() or not np.isfinite(fields).all():
            raise ValueError("dynamics values must be finite")
        if not np.allclose(couplings, couplings.T) or not np.allclose(np.diag(couplings), 0.0):
            raise ValueError("couplings must be symmetric with a zero diagonal")
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "fields", fields)

    @property
    def observable_count(self) -> int:
        return self.qubits + self.qubits * (self.qubits - 1) // 2

    @property
    def feature_count(self) -> int:
        return self.virtual_nodes * self.observable_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "virtual_nodes": self.virtual_nodes,
            "time_delta": self.time_delta,
            "couplings": self.couplings.tolist(),
            "fields": self.fields.tolist(),
            "observable_count": self.observable_count,
            "feature_count": self.feature_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReservoirDynamics:
        return cls(
            qubits=int(payload["qubits"]),
            virtual_nodes=int(payload["virtual_nodes"]),
            time_delta=float(payload["time_delta"]),
            couplings=np.asarray(payload["couplings"], dtype=float),
            fields=np.asarray(payload["fields"], dtype=float),
        )


@dataclass(slots=True)
class ReservoirTrace:
    features: np.ndarray
    final_density: np.ndarray
    purities: np.ndarray
    trace_errors: np.ndarray


@dataclass(slots=True)
class QuantumReservoirModel:
    dynamics: ReservoirDynamics
    readout: LinearReadout

    def predict_from_features(self, features: np.ndarray) -> np.ndarray:
        return self.readout.predict(features)

    def to_dict(self) -> dict[str, Any]:
        return {"dynamics": self.dynamics.to_dict(), "readout": self.readout.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QuantumReservoirModel:
        return cls(
            ReservoirDynamics.from_dict(payload["dynamics"]),
            LinearReadout.from_dict(payload["readout"]),
        )


def random_dynamics(
    qubits: int,
    virtual_nodes: int,
    time_delta: float,
    coupling_scale: float,
    field_scale: float,
    random_seed: int,
) -> ReservoirDynamics:
    rng = np.random.default_rng(random_seed)
    couplings = np.zeros((qubits, qubits), dtype=float)
    for first, second in combinations(range(qubits), 2):
        value = rng.uniform(-coupling_scale, coupling_scale)
        couplings[first, second] = value
        couplings[second, first] = value
    fields = rng.uniform(-field_scale, field_scale, qubits)
    return ReservoirDynamics(qubits, virtual_nodes, time_delta, couplings, fields)


def ising_hamiltonian(dynamics: ReservoirDynamics) -> Any:
    from qiskit.quantum_info import SparsePauliOp

    terms: list[tuple[str, tuple[int, ...], float]] = []
    for first, second in combinations(range(dynamics.qubits), 2):
        terms.append(("XX", (first, second), float(dynamics.couplings[first, second])))
    for qubit, field in enumerate(dynamics.fields):
        terms.append(("Z", (qubit,), float(field)))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=dynamics.qubits)


def observable_operators(dynamics: ReservoirDynamics) -> tuple[list[str], np.ndarray]:
    from qiskit.quantum_info import SparsePauliOp

    labels = [f"Z{qubit}" for qubit in range(dynamics.qubits)]
    terms = [("Z", (qubit,), 1.0) for qubit in range(dynamics.qubits)]
    for first, second in combinations(range(dynamics.qubits), 2):
        labels.append(f"Z{first}Z{second}")
        terms.append(("ZZ", (first, second), 1.0))
    matrices = [
        np.asarray(
            SparsePauliOp.from_sparse_list([term], num_qubits=dynamics.qubits).to_matrix(),
            dtype=complex,
        )
        for term in terms
    ]
    return labels, np.asarray(matrices)


def reservoir_unitary(dynamics: ReservoirDynamics) -> np.ndarray:
    hamiltonian = np.asarray(ising_hamiltonian(dynamics).to_matrix(), dtype=complex)
    return expm(-1j * hamiltonian * dynamics.time_delta / dynamics.virtual_nodes)


def initial_density_matrix(qubits: int) -> np.ndarray:
    dimension = 2**qubits
    density = np.zeros((dimension, dimension), dtype=complex)
    density[0, 0] = 1.0
    return density


def reset_input_qubit(density: np.ndarray, input_value: float, qubits: int) -> np.ndarray:
    """Replace least-significant qubit 0 while preserving the other reduced state."""

    value = float(input_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("input_value must be in [0, 1]")
    matrix = np.asarray(density, dtype=complex)
    dimension = 2**qubits
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"density must have shape ({dimension}, {dimension})")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise ValueError("density must be finite")
    rest_dimension = 2 ** (qubits - 1)
    reshaped = matrix.reshape(rest_dimension, 2, rest_dimension, 2)
    reduced = np.trace(reshaped, axis1=1, axis2=3)
    input_density = np.diag([1.0 - value, value]).astype(complex)
    return np.kron(reduced, input_density)


def simulate_reservoir(
    inputs: np.ndarray,
    dynamics: ReservoirDynamics,
    initial_density: np.ndarray | None = None,
) -> ReservoirTrace:
    values = np.asarray(inputs, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("inputs must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("inputs must contain finite values in [0, 1]")
    density = (
        initial_density_matrix(dynamics.qubits)
        if initial_density is None
        else np.asarray(initial_density, dtype=complex).copy()
    )
    if density.shape != (2**dynamics.qubits, 2**dynamics.qubits):
        raise ValueError("initial density has the wrong shape")
    unitary = reservoir_unitary(dynamics)
    unitary_adjoint = unitary.conj().T
    _, observables = observable_operators(dynamics)
    features = np.empty((len(values), dynamics.feature_count), dtype=float)
    purities = np.empty(len(values), dtype=float)
    trace_errors = np.empty(len(values), dtype=float)
    for row, input_value in enumerate(values):
        density = reset_input_qubit(density, float(input_value), dynamics.qubits)
        position = 0
        for _ in range(dynamics.virtual_nodes):
            density = unitary @ density @ unitary_adjoint
            density = (density + density.conj().T) / 2
            trace_value = np.trace(density)
            density /= trace_value
            for observable in observables:
                features[row, position] = float(np.real(np.trace(density @ observable)))
                position += 1
        purities[row] = float(np.real(np.trace(density @ density)))
        trace_errors[row] = float(abs(np.trace(density) - 1.0))
    return ReservoirTrace(features, density, purities, trace_errors)


def trotter_visualization_circuit(dynamics: ReservoirDynamics) -> Any:
    """Build a first-order circuit diagram for one virtual step.

    The benchmark itself uses exact matrix exponentiation of the full Hamiltonian.
    """

    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(dynamics.qubits, name="QRC virtual step")
    step = dynamics.time_delta / dynamics.virtual_nodes
    for qubit, field in enumerate(dynamics.fields):
        circuit.rz(2 * field * step, qubit)
    for first, second in combinations(range(dynamics.qubits), 2):
        circuit.rxx(2 * dynamics.couplings[first, second] * step, first, second)
    return circuit


def dynamics_diagnostics(dynamics: ReservoirDynamics) -> dict[str, Any]:
    hamiltonian = np.asarray(ising_hamiltonian(dynamics).to_matrix(), dtype=complex)
    eigenvalues = np.linalg.eigvalsh(hamiltonian).real
    unitary = reservoir_unitary(dynamics)
    identity = np.eye(len(unitary))
    labels, _ = observable_operators(dynamics)
    circuit = trotter_visualization_circuit(dynamics)
    return {
        **dynamics.to_dict(),
        "hilbert_dimension": 2**dynamics.qubits,
        "hamiltonian_terms": dynamics.qubits + dynamics.qubits * (dynamics.qubits - 1) // 2,
        "measurement_labels": labels,
        "hamiltonian_min_eigenvalue": float(eigenvalues.min()),
        "hamiltonian_max_eigenvalue": float(eigenvalues.max()),
        "hamiltonian_spectral_width": float(eigenvalues.max() - eigenvalues.min()),
        "unitarity_error_frobenius": float(np.linalg.norm(unitary.conj().T @ unitary - identity)),
        "diagram_circuit_depth": int(circuit.decompose().depth()),
        "diagram_circuit_operations": {
            str(name): int(count) for name, count in circuit.count_ops().items()
        },
        "simulation_method": "exact density matrix with full Hamiltonian exponentiation",
        "diagram_note": "The saved circuit is a first-order visualization, not the simulator path.",
    }
