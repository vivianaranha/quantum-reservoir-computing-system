# Experiment Guide

Created by School of AI and School of QC.

Start with `configs/default.json`. Change one factor at a time and retain the complete timestamped
artifact directory for every run. Useful controlled experiments vary the qubit count, number of
virtual nodes, evolution time, coupling or field scale, washout, readout regularization grid,
echo-state parameters, and reservoir seed.

Run the frozen benchmark:

~~~bash
qrc-train --config configs/default.json
~~~

Run a faster development experiment:

~~~bash
qrc-train --samples 600 --qubits 3 --virtual-nodes 2 \
  --washout 20 --lags 10 --rollout-horizon 16
~~~

Use `notebooks/quickstart.ipynb` for a guided Python workflow. It runs a compact configuration,
inspects generated evidence, and exercises saved-model forecasting.

Do not choose a model solely from teacher-forced one-step RMSE. Inspect recursive rollout,
directional accuracy, skill versus persistence, runtime, memory reconstruction, density-matrix
invariants, and finite-shot sensitivity. For serious conclusions, repeat configurations over
several reservoir seeds and chronological origins, and report the full distribution rather than
only the best run.

Keep comparisons explicit. QRC and ESN receive one scalar per step and store history in their
states. Autoregressive ridge and random forest receive 20 explicit lag values. That makes the
controls useful but not perfectly information matched.

The finite-shot diagnostic perturbs each expectation independently and evaluates the readout
trained on exact features. Treat it as a stress test. A hardware-oriented experiment must model
measurement grouping, state preparation, noise, transpilation, and retraining or calibration
under the intended shot budget.
