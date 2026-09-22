"""Validated experiment configuration.

Created by School of AI and School of QC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    sample_count: int = 1_600
    burn_in: int = 400
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    random_seed: int = 42
    qubits: int = 4
    virtual_nodes: int = 4
    time_delta: float = 0.8
    coupling_scale: float = 1.0
    field_scale: float = 0.7
    washout: int = 80
    lag_count: int = 20
    rollout_horizon: int = 64
    ridge_alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 0.1, 1.0)
    esn_spectral_radius: float = 0.90
    esn_leak_rate: float = 0.35
    esn_input_scale: float = 0.60
    random_forest_estimators: int = 250
    shot_counts: tuple[int, ...] = (128, 512, 2048)
    output_root: str = "artifacts"

    def validate(self) -> ExperimentConfig:
        if not 600 <= self.sample_count <= 10_000:
            raise ValueError("sample_count must be between 600 and 10,000")
        if not 100 <= self.burn_in <= 2_000:
            raise ValueError("burn_in must be between 100 and 2,000")
        if not 0.45 <= self.train_fraction <= 0.75:
            raise ValueError("train_fraction must be between 0.45 and 0.75")
        if not 0.10 <= self.validation_fraction <= 0.30:
            raise ValueError("validation_fraction must be between 0.10 and 0.30")
        if self.train_fraction + self.validation_fraction > 0.90:
            raise ValueError("train_fraction plus validation_fraction cannot exceed 0.90")
        if not 2 <= self.qubits <= 5:
            raise ValueError("qubits must be between 2 and 5 for exact density-matrix simulation")
        if not 1 <= self.virtual_nodes <= 10:
            raise ValueError("virtual_nodes must be between 1 and 10")
        if not 0.05 <= self.time_delta <= 3.0:
            raise ValueError("time_delta must be between 0.05 and 3.0")
        if self.coupling_scale <= 0 or self.field_scale <= 0:
            raise ValueError("Hamiltonian scales must be positive")
        if not 10 <= self.washout <= 300:
            raise ValueError("washout must be between 10 and 300")
        if not 3 <= self.lag_count <= 100:
            raise ValueError("lag_count must be between 3 and 100")
        if not 8 <= self.rollout_horizon <= 256:
            raise ValueError("rollout_horizon must be between 8 and 256")
        if not self.ridge_alphas or any(value <= 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain positive values")
        if not 0 < self.esn_spectral_radius < 1.5:
            raise ValueError("esn_spectral_radius must be between 0 and 1.5")
        if not 0 < self.esn_leak_rate <= 1:
            raise ValueError("esn_leak_rate must be between 0 and 1")
        if self.esn_input_scale <= 0:
            raise ValueError("esn_input_scale must be positive")
        if not 50 <= self.random_forest_estimators <= 1_000:
            raise ValueError("random_forest_estimators must be between 50 and 1,000")
        if not self.shot_counts or any(value < 32 for value in self.shot_counts):
            raise ValueError("shot_counts must contain integers of at least 32")
        if not self.output_root.strip():
            raise ValueError("output_root cannot be empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        values = {key: value for key, value in overrides.items() if value is not None}
        if "ridge_alphas" in values:
            values["ridge_alphas"] = tuple(float(value) for value in values["ridge_alphas"])
        if "shot_counts" in values:
            values["shot_counts"] = tuple(int(value) for value in values["shot_counts"])
        return replace(self, **values).validate()

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "ridge_alphas" in payload:
            payload["ridge_alphas"] = tuple(payload["ridge_alphas"])
        if "shot_counts" in payload:
            payload["shot_counts"] = tuple(payload["shot_counts"])
        return cls(**payload).validate()
