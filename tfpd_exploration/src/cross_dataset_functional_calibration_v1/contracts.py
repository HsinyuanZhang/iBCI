"""Frozen interfaces for cross-dataset functional calibration v1.

Coordinator-owned types. Workers implement against these names.
Do not invent estimators or silently equate forward/backward descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from . import plan

CONTRACT_VERSION = plan.CONTRACT_VERSION
PathLike = str | Path

NAMED_ESTIMATORS = {
    "h1_deployment_m3": {
        "direction": plan.H1_ESTIMATOR_DIRECTION,
        "operator": "SPINT-main.src.data.h1_m4_eb_pilot.fit_deployment_carrier",
        "support": "first_3_eval_valid_TrialNum",
        "output": "[N,4] EB-shrunk backward rows",
        "not": (
            "forward physiological tuning, Haufe pattern, q3-AFC4, "
            "or a new PCA/ridge/EB"
        ),
    },
    "h1_frozen_m4": {
        "direction": plan.H1_ESTIMATOR_DIRECTION,
        "operator": "SPINT-main.src.data.h1_m4_eb_pilot.fit_frozen_carrier",
        "support": "first_4_eval_valid_TrialNum",
        "note": "historical M4 audit; E1 uses M3 deployment, not this",
    },
    "h1_forward_unit_ridge_control": {
        "direction": "forward_encoding_ridge",
        "operator": "local diagnostic using H1 unnormalized ridge on (1, velocity)",
        "role": "invariance control only; not a replacement carrier",
    },
    "m1_rsyn3_unit_ridge": {
        "direction": plan.M1_ESTIMATOR_DIRECTION,
        "operator": "tfpd_exploration.src.m1_emg_syn3_fcm_v1.syn3.fit_unit_ridge",
        "output": "[N,4] = [w1,w2,w3,b] on source-frozen rSyn3 scores",
        "not": "direction T4, NNMF refit on target, or a new lambda",
    },
    "m1_rsyn3_dynamic_lag_diagnostic": {
        "direction": plan.M1_ESTIMATOR_DIRECTION,
        "z": "concat(z0(t-5), z0(t), z0(t+5)) with same-trial intersection",
        "role": "E2 feasibility only; not a P-pilot basis",
    },
    "m1_row_normalized_nnmf_nnls_v1": {
        "direction": plan.M1_ESTIMATOR_DIRECTION,
        "operator": "cross_dataset_functional_calibration_v1.basis.RowNormalizedNNMFBasis",
        "output": "z in R^3 then [w1,w2,w3,b] via solve_ridge",
        "role": "named P f_eta; zero perturbation equals bank rSyn3",
        "not": "16-to-3 unconstrained linear, q=8 bottleneck MLP, or E2 lags",
    },
}

UNRESOLVED_P_OPERATORS = {
    "f_eta": {
        "status": "NAMED_REVISION",
        "name": plan.P_F_ETA_NAME,
        "shape": list(plan.P_F_ETA_SHAPE),
        "authority": plan.P_REVISION_RELATIVE,
    },
    "differentiable_solve": {
        "status": "NAMED_REVISION",
        "operator": "carrier_solver.solve_ridge",
        "authority": plan.P_REVISION_RELATIVE,
    },
    "parent_bytes": {
        "status": "NAMED_REVISION",
        "bytes": plan.S_FIX_EPOCH011_SHA256,
        "claim": plan.P_PARENT_CLAIM,
        "authority": plan.P_REVISION_RELATIVE,
        "gpu_eligible": False,
    },
    "normalizer": {
        "status": "PREFLIGHT_NAMED",
        "name": plan.P_NORMALIZER_NAME,
        "authority": plan.P_PREFLIGHT_AUTHORITY_RELATIVE,
        "gpu_eligible": False,
    },
}


@dataclass(frozen=True)
class BoundAsset:
    relative: str
    sha256: str | None
    exists: bool
    role: str
    shape: tuple[int, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class H1SupportView:
    session_name: str
    date: str
    trial_ids: tuple[float, ...]
    n_blocks: int
    n_channels: int
    rates: np.ndarray
    velocity: np.ndarray
    block_indices: np.ndarray


@dataclass(frozen=True)
class M1EncodingView:
    session_name: str
    role: str
    fit_trial_range: tuple[int, int]
    eval_trial_range: tuple[int, int]
    z_static: np.ndarray
    rates: np.ndarray
    trial_ids: np.ndarray
    query_values_read: bool = False


@dataclass(frozen=True)
class AssayStatus:
    name: str
    status: str
    blocker: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class InventoryBuilder(Protocol):
    def build(self, root: PathLike) -> dict[str, Any]:
        """Return inventory.json body. CPU only. No hidden files."""


class H1PopulationAssay(Protocol):
    def run(self, repo_root: PathLike) -> AssayStatus:
        """E1. No decoder R2. Isolated wrapper around fit_deployment_carrier."""


class M1DynamicAssay(Protocol):
    def run(self, repo_root: PathLike) -> AssayStatus:
        """E2. Source-session encoding diagnostic. No target-query fitting."""


class CoverageAssay(Protocol):
    def run(self, repo_root: PathLike, *, prior: Mapping[str, Any]) -> AssayStatus:
        """E3. Reuse receipts; do not infer n_eff from bin count."""


# P interfaces are named so a later worker cannot invent a different API.
# Implementations are not provided in this Stage0 package.


class BehaviorBasis(Protocol):
    name: str

    def encode(self, y_support: np.ndarray) -> np.ndarray:
        """Return Z_S with a frozen column count. Source-only scale/gauge."""


class CarrierSolver(Protocol):
    def solve(self, z_support: np.ndarray, x_support: np.ndarray) -> np.ndarray:
        """Closed-form intercept-unpenalized ridge. Slopes only penalized."""


class StrongConsumer(Protocol):
    def forward(
        self,
        x_query: Any,
        activity: Any,
        carrier: Any,
    ) -> Any:
        """Native 16-D signed EMG. Full residual path preserved."""


def require_named_estimator(name: str) -> dict[str, object]:
    if name not in NAMED_ESTIMATORS:
        raise plan.PlanError(f"undocumented estimator {name!r}")
    return dict(NAMED_ESTIMATORS[name])


def jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)
