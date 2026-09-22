# Evaluation Protocol

Created by School of AI and School of QC.

## One-step evaluation

Every model predicts the next value from information available at the current step. Reported
metrics include MAE, RMSE, RMSE normalized by test standard deviation, MAPE, R², bias,
directional accuracy, skill against persistence, training time, and inference time.

## Recursive rollout

The first test input seeds a 64-step free-running trajectory. Each model receives its own prior
prediction as the next input. This evaluates error accumulation rather than teacher-forced
one-step accuracy. A model can excel one step ahead and fail in rollout; the project reports both.

## Delayed-input memory

A diagnostic ridge model reconstructs scaled inputs delayed by 1–20 steps from frozen QRC and
ESN features. It fits on train plus validation and reports test R². Positive R² values are summed
as a compact descriptive score. It is not a universal memory-capacity theorem.

## Finite-shot sensitivity

For each exact expectation $z\in[-1,1]$, the probe treats $(1+z)/2$ as a Bernoulli probability
and generates independent binomial counts. The exact-trained readout is not recalibrated. This
isolates sampling sensitivity but ignores commuting groups, covariance, circuit repetition,
device noise, and mitigation.

## Fairness of controls

QRC and ESN have 40 features and internal temporal state. Autoregressive ridge and random forest
receive 20 explicit lags. This makes the latter strong practical controls, but not identical
information-processing systems. Conclusions are limited to predictive accuracy under the stated
protocol.
