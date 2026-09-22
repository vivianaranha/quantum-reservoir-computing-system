"""Interactive quantum reservoir forecasting workbench.

Created by School of AI and School of QC.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from quantum_reservoir_system import ExperimentConfig, run_experiment
from quantum_reservoir_system.inference import forecast_frame

st.set_page_config(page_title="Quantum Reservoir Computing System", layout="wide")


def available_runs() -> list[Path]:
    return sorted(
        [path for path in Path("artifacts").glob("run-*") if (path / "metrics.csv").exists()],
        reverse=True,
    )


def render_metrics(metrics: pd.DataFrame) -> None:
    columns = [
        "model",
        "mae",
        "rmse",
        "nrmse_std",
        "r2",
        "directional_accuracy",
        "skill_vs_persistence_rmse",
    ]
    st.dataframe(
        metrics[columns].style.format(precision=5),
        use_container_width=True,
        hide_index=True,
    )


def render_run(run_directory: Path) -> None:
    metrics = pd.read_csv(run_directory / "metrics.csv")
    render_metrics(metrics)
    gallery = [
        ("one_step_forecasts.png", "Held-out one-step forecasts"),
        ("recursive_rollout.png", "Recursive rollout"),
        ("memory_capacity.png", "Delayed-input memory"),
        ("shot_sensitivity.png", "Finite-shot sensitivity"),
        ("quantum_reservoir_features.png", "Quantum reservoir features"),
        ("hamiltonian_dynamics.png", "Fixed Hamiltonian dynamics"),
    ]
    columns = st.columns(2)
    for position, (filename, caption) in enumerate(gallery):
        image_path = run_directory / filename
        if image_path.exists():
            columns[position % 2].image(str(image_path), caption=caption, use_container_width=True)
    report_path = run_directory / "experiment_report.md"
    if report_path.exists():
        with st.expander("Experiment report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    circuit_path = run_directory / "reservoir_step_circuit.txt"
    if circuit_path.exists():
        with st.expander("Reservoir step circuit diagram"):
            st.code(circuit_path.read_text(encoding="utf-8"), language="text")
    st.download_button(
        "Download metrics CSV",
        metrics.to_csv(index=False),
        file_name=f"{run_directory.name}-metrics.csv",
        mime="text/csv",
    )
    st.caption(f"Artifact directory: {run_directory}")


def benchmark_page() -> None:
    with st.sidebar:
        st.header("Benchmark settings")
        samples = st.slider("Series samples", 600, 3_000, 1_200, step=100)
        seed = st.number_input("Reservoir seed", 0, 10_000, 42)
        qubits = st.slider("Reservoir qubits", 2, 5, 4)
        virtual_nodes = st.slider("Virtual nodes", 1, 8, 4)
        time_delta = st.slider("Evolution time", 0.1, 2.0, 0.8, step=0.1)
        washout = st.slider("Washout states", 20, 200, 60, step=10)
        lags = st.slider("Classical lag count", 5, 40, 20)
        horizon = st.slider("Recursive horizon", 8, 128, 48, step=8)
        run_button = st.button("Run benchmark", type="primary", use_container_width=True)

    st.info(
        "This is an educational exact density-matrix simulator using synthetic chaotic data. "
        "It is not a production forecasting or quantum-advantage claim."
    )
    if run_button:
        config = ExperimentConfig(
            sample_count=samples,
            random_seed=int(seed),
            qubits=qubits,
            virtual_nodes=virtual_nodes,
            time_delta=time_delta,
            washout=washout,
            lag_count=lags,
            rollout_horizon=horizon,
            output_root="artifacts",
        ).validate()
        status = st.empty()
        with st.spinner("Driving reservoirs and fitting readouts..."):
            st.session_state["experiment_result"] = run_experiment(
                config,
                progress_callback=status.write,
            )
        status.success("Experiment complete")

    result = st.session_state.get("experiment_result")
    if result is None:
        st.subheader("What the benchmark tests")
        st.write(
            "A fixed quantum system receives one scalar input at a time, evolves under a seeded "
            "Ising Hamiltonian, and emits virtual-node expectation values. Only its linear "
            "readout is trained. Chronological validation selects regularization before an "
            "untouched future block is evaluated."
        )
        return
    st.subheader("Held-out one-step comparison")
    render_metrics(result.metrics)
    left, right = st.columns(2)
    left.image(
        str(result.output_directory / "one_step_forecasts.png"),
        use_container_width=True,
    )
    right.image(
        str(result.output_directory / "recursive_rollout.png"),
        use_container_width=True,
    )
    st.success(f"Experiment artifacts: {result.output_directory}")


def inspect_page() -> None:
    st.subheader("Inspect a saved benchmark")
    runs = available_runs()
    if not runs:
        st.warning("No local runs were found. Run the benchmark first.")
        return
    selected = st.selectbox("Saved run", runs, format_func=lambda path: path.name)
    render_run(selected)


def forecast_page() -> None:
    st.subheader("Forecast from a history CSV")
    runs = available_runs()
    if not runs:
        st.warning("No local runs were found. Run the benchmark first.")
        return
    selected = st.selectbox("Saved model", runs, format_func=lambda path: path.name)
    horizon = st.slider("Future steps", 1, 128, 24)
    template = pd.DataFrame({"value": [1.0 + 0.01 * index for index in range(20)]})
    st.download_button(
        "Download history template",
        template.to_csv(index=False),
        file_name="history-template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload CSV with a value column", type="csv")
    if uploaded is None:
        return
    try:
        forecasts = forecast_frame(selected, pd.read_csv(uploaded), horizon)
    except (ValueError, OSError, KeyError) as error:
        st.error(str(error))
        return
    st.line_chart(
        forecasts.set_index("horizon_step"),
        use_container_width=True,
    )
    st.dataframe(forecasts, use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecasts",
        forecasts.to_csv(index=False),
        file_name="quantum-reservoir-forecasts.csv",
        mime="text/csv",
    )


st.title("Quantum Reservoir Computing System")
st.caption("Created by School of AI and School of QC")
st.write(
    "Use fixed quantum dynamics as a temporal feature reservoir, train a lightweight classical "
    "readout, and compare one-step and recursive forecasts with strong classical controls."
)
with st.sidebar:
    page = st.radio("Workspace", ["Run benchmark", "Inspect saved run", "Forecast CSV"])

if page == "Run benchmark":
    benchmark_page()
elif page == "Inspect saved run":
    inspect_page()
else:
    forecast_page()

st.divider()
st.caption("Created by School of AI and School of QC")
