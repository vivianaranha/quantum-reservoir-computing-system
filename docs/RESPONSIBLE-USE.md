# Responsible Use

Created by School of AI and School of QC.

Treat this repository as an educational exact simulator. Before adapting it to real data, obtain
permission, document provenance and sampling, remove unnecessary identifiers, preserve temporal
ordering, establish non-model fallbacks, and evaluate performance across operating regimes and
forecast horizons.

Never infer that a quantum component makes a forecast more accurate, private, secure, fair, or
causal. Report the best classical control, failed comparisons, compute cost, seed sensitivity,
and whether results come from exact simulation, finite shots, a noise model, or physical
hardware. Do not describe the saved circuit diagram as the execution path; the benchmark uses
exact full-Hamiltonian exponentiation.

Consequential use requires calibrated uncertainty, drift detection, rollback procedures, access
controls, versioned data and models, domain review, and human authority to reject predictions.
Do not use one-step test performance as evidence of stable long-horizon behavior.

Only load joblib files from trusted sources. Report security issues through `SECURITY.md`; never
put credentials, private time series, or identifiable records in a public issue.
