# Frozen Reference Results

Created by School of AI and School of QC.

The seed-42 run was generated on 2026-09-22 with 1,600 deterministic Mackey–Glass values and
1,580 supervised rows. Chronological blocks contained 948 train, 316 validation, and 316 test
rows.

## One-step held-out results

- Autoregressive ridge: RMSE 0.000495, MAE 0.000385, R² 0.999995.
- Quantum reservoir: RMSE 0.001056, MAE 0.000793, R² 0.999980.
- Echo-state network: RMSE 0.001593, MAE 0.001258, R² 0.999953.
- Random forest: RMSE 0.005204, MAE 0.003944, R² 0.999503.
- Persistence: RMSE 0.034504, MAE 0.028711, R² 0.978134.

The QRC beat the dimension-matched ESN and persistence but did not beat autoregressive ridge.
This is evidence that the fixed quantum features are useful on this split, not evidence of
quantum advantage.

![One-step forecasts](assets/one_step_forecasts.png)

## Recursive results

Over 64 self-fed steps, random forest RMSE was 0.010024, ESN 0.011189, autoregressive ridge
0.057156, QRC 0.187603, and persistence 0.211345. The QRC trajectory drifts after its strong
early forecasts, showing that one-step scores alone are insufficient.

![Recursive rollout](assets/recursive_rollout.png)

## Reservoir diagnostics

The QRC used four qubits, four virtual nodes, ten observables, 40 features, zero trained quantum
parameters, and 41 readout parameters. Its maximum density-trace error was
$4.44\times10^{-16}$ and unitary error was $5.47\times10^{-16}$.

Positive delayed-input memory across delays 1–20 was 17.7966 for QRC and 19.9599 for ESN.

The frozen exact readout was highly sensitive to independently sampled measurement expectations:
RMSE was 1113.2532 at 128 shots, 546.5150 at 512 shots, and 274.5951 at 2,048 shots, compared
with 0.001056 exactly. Hardware-oriented work should retrain with shot noise, regularize more
strongly, group measurements, and report uncertainty.

![Memory curve](assets/memory_capacity.png)

![Shot sensitivity](assets/shot_sensitivity.png)

All exact values and environment details are preserved in examples/reference-run. Results are
specific to one synthetic sequence, one split, one reservoir seed, and exact simulation.
