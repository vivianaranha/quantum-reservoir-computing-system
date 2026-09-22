# Curated Reference Run

Created by School of AI and School of QC.

This compact evidence bundle comes from the frozen seed-42 default run generated on 2026-09-22.
It includes configuration, environment, dataset summary, exact metrics, row-level held-out
predictions, recursive rollout, delayed-input memory, finite-shot sensitivity, Hamiltonian
coefficients, QRC dynamics and readout, alpha search, circuit text, diagnostics, and the generated
report.

Large generated files such as joblib models, the full causal frame, reservoir feature matrices,
and all eleven full-resolution plots are intentionally omitted. Reproduce the complete 34-file
run with:

~~~bash
qrc-train --config configs/default.json
~~~

The six selected plots in `docs/assets` come from the same run. See `docs/VERIFICATION.md` for
release checks and `docs/RESULTS.md` for interpretation. The evidence represents exact local
simulation on one deterministic synthetic sequence and does not establish quantum advantage or
production forecasting performance.
