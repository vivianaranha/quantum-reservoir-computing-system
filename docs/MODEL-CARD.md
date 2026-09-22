# Model Card

Created by School of AI and School of QC.

## Intended use

Education, local experimentation, and research prototyping for quantum reservoir computing,
causal time-series evaluation, and quantum/classical benchmarking.

## Inputs and outputs

The benchmark consumes a univariate finite numeric sequence ordered from oldest to newest. It
produces one-step predictions for a held-out chronological block, a recursive future rollout,
and diagnostic metrics and plots. Saved inference accepts a CSV containing a `value` column and
returns forecasts from all five fitted models.

## Model design

The QRC replaces qubit 0 with a mixed state encoding the current train-scaled scalar, evolves the
full density matrix under a fixed seeded XX plus local-Z Hamiltonian, and measures Z and pairwise
ZZ expectations at virtual nodes. Only a ridge readout is fitted. The default four-qubit model
has 40 features, zero trainable quantum parameters, and 41 trainable linear parameters.

## Training and evaluation data

The default data are a deterministic synthetic Mackey–Glass sequence. The causal frame is split
chronologically into 60% training, 20% validation, and 20% test. Input and target scalers fit the
training block only. Validation selects readout regularization. Test rows remain untouched until
final evaluation.

## Risks and limitations

Synthetic performance may not transfer to real systems. Exact simulation omits quantum-device
noise and execution cost. Four qubits do not establish useful scale. One reservoir seed and one
chronological split do not quantify uncertainty. Recursive forecasts can drift even when
one-step error is small. Exact-trained readouts can be extremely sensitive to finite-shot noise.
The classical lag models receive more explicit history than the reservoir models.

## Out of scope

Claims of quantum speedup, autonomous trading, clinical monitoring, critical-infrastructure
control, safety shutdown, or any consequential forecast without representative validation,
uncertainty analysis, monitoring, and accountable human oversight.
