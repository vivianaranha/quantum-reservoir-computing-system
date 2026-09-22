# Verification Record

Created by School of AI and School of QC.

Verification completed on 2026-09-22 with Python 3.12.14 on Linux. The frozen reference
environment recorded Qiskit 2.5.2, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, SciPy 1.17.0,
and Streamlit 1.64.0. The package supports Python 3.11 and 3.12; CI tests both versions.

Release checks:

- Ruff lint passed.
- Ruff format check passed.
- All 88 tests passed.
- Independent Qiskit partial-trace parity verified the input-reset implementation.
- Independent Qiskit `DensityMatrix.evolve` parity verified one exact evolution step.
- Hamiltonian Hermiticity, unitary evolution, density trace, purity, and feature bounds passed.
- Chronological framing and train-only scaling tests passed.
- All five fitted models produced finite one-step and recursive predictions.
- Saved QRC JSON, joblib models, preprocessing, CSV inference, CLI help, and app startup passed.
- The quickstart notebook parsed as valid notebook format and contains the executable workflow.
- All eleven generated plots were visually inspected together at full-output scale.
- The source distribution and wheel built successfully.
- The final GitHub ZIP passed lint, formatting, tests, and build again from a fresh extraction.

The default seed-42 run generated 1,600 Mackey–Glass points and 1,580 causal supervised rows:
948 training, 316 validation, and 316 held-out test rows. The QRC used four qubits, a
16-dimensional Hilbert space, four virtual nodes, ten observables, 40 features, zero trained
quantum parameters, and 41 fitted readout parameters. Maximum density-trace error was
4.44e-16; unitary error was 5.47e-16.

Held-out one-step RMSE:

- Autoregressive ridge: 0.000495.
- Quantum reservoir: 0.001056.
- Echo-state network: 0.001593.
- Random forest: 0.005204.
- Persistence: 0.034504.

Recursive 64-step RMSE:

- Random forest: 0.010024.
- Echo-state network: 0.011189.
- Autoregressive ridge: 0.057156.
- Quantum reservoir: 0.187603.
- Persistence: 0.211345.

The QRC beat the size-matched ESN on one-step RMSE but not autoregressive ridge, and its
recursive trajectory accumulated substantial error. Positive delayed-input memory was 17.7966
for QRC and 19.9599 for ESN. The exact-trained QRC readout was extremely fragile under the
independent finite-shot sensitivity probe. These outcomes are retained rather than filtered.

The curated machine-readable evidence is in `examples/reference-run`. Results remain specific
to one deterministic synthetic sequence, one chronological split, one reservoir seed, and exact
local simulation; they do not establish quantum advantage or production performance.
