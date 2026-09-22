# Architecture

Created by School of AI and School of QC.

The package separates deterministic data generation, causal framing, train-only scaling,
quantum dynamics, classical controls, readout selection, evaluation, persistence, and inference.
The CLI, notebook, tests, and Streamlit application call the same experiment code.

~~~mermaid
flowchart TD
    UI["CLI, notebook, or Streamlit"] --> EX["Experiment orchestrator"]
    EX --> DATA["Mackey–Glass and chronological splits"]
    EX --> Q["Fixed quantum reservoir"]
    EX --> C["Classical controls"]
    Q --> EV["One-step, rollout, memory, shot diagnostics"]
    C --> EV
    EV --> ART["Versioned run artifacts"]
    ART --> INF["Saved CSV forecasting"]
~~~

data.py owns the deterministic delay-system generator and causal lag frame. preprocessing.py
fits both scalers on the training block only. reservoir.py constructs the seeded Ising
Hamiltonian, input-reset channel, exact unitary, observables, virtual-node features, and
serializable QRC. readout.py selects and freezes the linear ridge readout.

classical.py implements a dimension-matched echo-state reservoir, autoregressive ridge, and
random forest. forecasting.py performs separate recursive trajectories for every model.
evaluation.py owns metrics, alpha selection, timing, and delayed-input memory. artifacts.py
creates the full audit bundle, while inference.py reloads frozen artifacts without training.

No model imports the test target during fitting. The only cross-boundary state is the causal
reservoir state created by earlier inputs, which is exactly what a streaming forecaster would
carry forward.
