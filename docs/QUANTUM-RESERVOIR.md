# Quantum Reservoir Mathematics

Created by School of AI and School of QC.

## Input injection

Each scalar input is transformed to $u_t\in[0,1]$. Qubit 0 is discarded and replaced with

$$\rho_{in}(u_t)=(1-u_t)|0\rangle\langle0|+u_t|1\rangle\langle1|.$$

Using Qiskit's little-endian convention, the updated full state is

$$\rho'_t=\operatorname{Tr}_0(\rho_{t-1})\otimes\rho_{in}(u_t).$$

This channel injects new information, removes old qubit-0 correlations, and gives the otherwise
unitary system a fading-memory mechanism.

## Fixed dynamics

The reservoir Hamiltonian is

$$H=\sum_{i<j}J_{ij}X_iX_j+\sum_i h_iZ_i,$$

where all couplings and fields are sampled once from a seeded uniform distribution. Nothing in
$H$ is optimized. For $V$ virtual nodes, each substep uses

$$U_V=\exp\left(-iH\Delta t/V\right),\qquad
\rho_{t,v}=U_V\rho_{t,v-1}U_V^\dagger.$$

The default measures four single-qubit $Z_i$ observables and six pairwise $Z_iZ_j$ observables
at each of four virtual nodes, creating 40 real features per input.

## Readout

Only a ridge readout trains:

$$\hat y_{t+1}=b+w^Tz_t.$$

Validation chooses the ridge penalty. The final readout refits on train plus validation features
and is evaluated once on the chronological test block.

## Simulator and circuit diagram

The actual benchmark exponentiates the complete Hamiltonian and evolves a density matrix exactly.
The saved RZ/RXX circuit is a first-order Trotter visualization of one virtual step; it is
explicitly not the numerical simulator path.

Independent tests compare input reset with Qiskit's partial_trace and compare one-step density
evolution with Qiskit DensityMatrix.evolve. Unitarity, Hermiticity, trace, purity, feature bounds,
and serialization are also checked.

## Boundaries

Exact density matrices scale as $4^n$ in storage. Finite-shot and noisy hardware execution would
require observable grouping, repeated state preparation, uncertainty estimates, transpilation,
and backend calibration. The current finite-shot diagnostic perturbs each expectation
independently and must not be interpreted as a full hardware simulation.

References:

- [Original QRC proposal](https://arxiv.org/abs/1602.08159)
- [QRC optimization for time-series prediction](https://arxiv.org/abs/1807.03947)
- [Qiskit quantum information](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info)
