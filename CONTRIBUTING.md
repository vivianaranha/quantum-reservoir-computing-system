# Contributing

Created by School of AI and School of QC.

Open an issue before a large change. Keep pull requests focused, add tests for changed behavior,
and document any change to the generator, chronological split, scaler fitting, reservoir state,
Hamiltonian, observable set, washout, hyperparameter search, or rollout protocol.

Before opening a pull request, run:

~~~bash
ruff check .
ruff format --check .
pytest
python -m build
~~~

Forecasting changes should preserve causal ordering and train-only preprocessing. Report both
one-step and recursive behavior when either changes. Clearly distinguish exact simulation,
finite-shot sensitivity analysis, noisy simulation, and hardware execution. Do not claim quantum
advantage from a single seed or synthetic dataset.

By contributing, you agree that your work is licensed under the MIT License.
