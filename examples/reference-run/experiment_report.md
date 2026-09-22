# Quantum Reservoir Computing System — Experiment Report

**Created by School of AI and School of QC**

## Protocol

- Supervised rows: 1580
- Training rows: 948
- Validation rows: 316
- Test rows: 316
- Split policy: chronological train, validation, then untouched test

## Held-out one-step forecasts

### Autoregressive Ridge

- MAE: 0.000385
- RMSE: 0.000495
- Normalized RMSE: 0.002122
- R-squared: 0.999995
- Directional accuracy: 0.9937
- Skill versus persistence: 0.9856
- Training seconds: 0.0055
- Inference milliseconds per sample: 0.0004

### Quantum Reservoir

- MAE: 0.000793
- RMSE: 0.001056
- Normalized RMSE: 0.004524
- R-squared: 0.999980
- Directional accuracy: 0.9968
- Skill versus persistence: 0.9694
- Training seconds: 0.7208
- Inference milliseconds per sample: 0.4874

### Echo State Network

- MAE: 0.001258
- RMSE: 0.001593
- Normalized RMSE: 0.006828
- R-squared: 0.999953
- Directional accuracy: 0.9968
- Skill versus persistence: 0.9538
- Training seconds: 0.0174
- Inference milliseconds per sample: 0.0013

### Random Forest

- MAE: 0.003944
- RMSE: 0.005204
- Normalized RMSE: 0.022304
- R-squared: 0.999503
- Directional accuracy: 0.9714
- Skill versus persistence: 0.8492
- Training seconds: 0.2998
- Inference milliseconds per sample: 0.1454

### Persistence

- MAE: 0.028711
- RMSE: 0.034504
- Normalized RMSE: 0.147871
- R-squared: 0.978134
- Directional accuracy: 0.0000
- Skill versus persistence: 0.0000
- Training seconds: 0.0000
- Inference milliseconds per sample: 0.0000

## Quantum reservoir

- Qubits: 4
- Hilbert-space dimension: 16
- Virtual nodes: 4
- Measured features per time step: 40
- Fixed Hamiltonian terms: 10
- Trained quantum parameters: 0
- Trained linear-readout parameters: 41
- Maximum density-trace error: 4.441e-16

## Recursive rollout RMSE

- Quantum Reservoir: 0.187603
- Echo State Network: 0.011189
- Autoregressive Ridge: 0.057156
- Random Forest: 0.010024
- Persistence: 0.211345

## Positive delayed-input memory capacity

- Echo State Network: 19.959863
- Quantum Reservoir: 17.796640

## Interpretation

Only the classical linear readout is trained for the quantum reservoir. The internal Hamiltonian is fixed after seeded initialization. Classical controls receive either a matched reservoir-state dimension or explicit lag features. The data are synthetic, the dynamics are exactly simulated, and these results do not establish quantum advantage or production forecasting performance.
