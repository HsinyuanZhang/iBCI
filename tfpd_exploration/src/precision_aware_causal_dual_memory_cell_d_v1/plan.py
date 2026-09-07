"""Static no-data contract for precision-aware CDM-D V1."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CELL = "PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_V1"
PHASE = "support_precision_aware_independent_activity_transition"
SCHEMA = "precision_aware_causal_dual_memory_cell_d_v1"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_V1_20260826.md"
# Bound to the complete concise work order.  Closure validation compares this
# literal to a held no-follow read of the document.
WORKORDER_SHA256 = "ced29701e27da700a44e11d8d11370cfb4f3ab8eda66f42d4eec48f32f1f8e36"

BUDGETS = (4, 10, 30)
TRANSITION_BUDGETS = (4, 10)
M30_DEPLOYMENT_NOOP = True
POST_FIRST30_QUERY_START = 30
FIXED_RIDGE_NORMALIZED_LAMBDA = 0.1
RESIDUAL_VARIANCE_FLOOR = 1.0e-12
CREDIBLE_REGION_CHI2_DF2_95 = 5.991464547107979

SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
SEALED_CELL_D_BASELINE_SHA256 = "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"
SEALED_OLS_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
SEALED_OLS_MEAN_FLOAT32 = (
    0.04627712443470955,
    0.4544036388397217,
    1.3432163000106812,
    10.150517463684082,
)
SEALED_OLS_STD_FLOAT32 = (
    1.126278281211853,
    1.284820556640625,
    1.2352101802825928,
    9.115250587463379,
)


class PrecisionAwarePlanError(RuntimeError):
    """Fail closed for a static plan, sealed authority, or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionAwarePlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


@dataclass(frozen=True)
class SealedCellDAuthority:
    """Exact sealed artifacts the mechanism is forbidden to modify or refit."""

    terminal_sha256: str = SEALED_CELL_D_TERMINAL_SHA256
    swa_sha256: str = SEALED_CELL_D_SWA_SHA256
    baseline_sha256: str = SEALED_CELL_D_BASELINE_SHA256
    ordinary_ols_normalizer_sha256: str = SEALED_OLS_NORMALIZER_SHA256
    ordinary_ols_mean_float32: tuple[float, float, float, float] = SEALED_OLS_MEAN_FLOAT32
    ordinary_ols_std_float32: tuple[float, float, float, float] = SEALED_OLS_STD_FLOAT32

    def payload(self) -> dict[str, object]:
        for label, value in (
            ("sealed terminal", self.terminal_sha256),
            ("sealed SWA", self.swa_sha256),
            ("sealed baseline", self.baseline_sha256),
            ("sealed OLS normalizer", self.ordinary_ols_normalizer_sha256),
        ):
            require_sha256(value, label)
        _require(
            tuple(self.ordinary_ols_mean_float32) == SEALED_OLS_MEAN_FLOAT32
            and tuple(self.ordinary_ols_std_float32) == SEALED_OLS_STD_FLOAT32,
            "precision route may only bind the frozen ordinary OLS moments",
        )
        return {
            "schema": "precision_aware_cdmd_sealed_cell_d_authority_v1",
            "terminal_sha256": self.terminal_sha256,
            "swa_sha256": self.swa_sha256,
            "baseline_sha256": self.baseline_sha256,
            "ordinary_ols_normalizer_sha256": self.ordinary_ols_normalizer_sha256,
            "ordinary_ols_mean_float32": list(self.ordinary_ols_mean_float32),
            "ordinary_ols_std_float32": list(self.ordinary_ols_std_float32),
            "model_parameter_changes_allowed": False,
            "normalizer_refit_allowed": False,
            "target_access_allowed": False,
        }


SEALED_CELL_D_AUTHORITY = SealedCellDAuthority()

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v1/plan.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v1/transition.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v1/physical.py",
    "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_v1.py",
    "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_v1.py",
)
# The only executed shared implementation dependency is the independent
# activity state machine.  The later physical route must add its own exact
# source/evaluator closure rather than borrowing a historical current closure.
IMPLEMENTATION_PATHS = (
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    *_OWNED_PATHS,
)


def _read_regular_no_follow(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PrecisionAwarePlanError(f"precision closure path unavailable: {path}") from error
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode), f"precision closure path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return sha256_bytes(b"".join(chunks))
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "precision closure path topology drift")
        rows = [
            {"path": relative, "sha256": require_sha256(self.sha256_by_path[relative], f"closure {relative}")}
            for relative in IMPLEMENTATION_PATHS
        ]
        workorder = next(item["sha256"] for item in rows if item["path"] == WORKORDER_RELATIVE)
        _require(workorder == WORKORDER_SHA256, "precision workorder bytes drift")
        body = {"schema": "precision_aware_cdmd_implementation_closure_v1", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({relative: _read_regular_no_follow(base / relative) for relative in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "precision closure must be a mapping")
    candidate = dict(value)
    _require(
        set(candidate) == {"schema", "paths", "closure_sha256"}
        and candidate.get("schema") == "precision_aware_cdmd_implementation_closure_v1",
        "precision closure schema drift",
    )
    rows = candidate.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "precision closure row count drift")
    rebuilt: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(
            isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
            "precision closure path/order drift",
        )
        rebuilt[expected] = require_sha256(row.get("sha256"), f"closure {expected}")
    result = ImplementationClosure(rebuilt).payload()
    _require(candidate == result, "precision closure canonical payload drift")
    return result


def dry_plan() -> dict[str, object]:
    """Return a public, side-effect-free contract without importing runtime code."""
    sealed = SEALED_CELL_D_AUTHORITY.payload()
    return {
        "schema": "precision_aware_cdmd_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "budgets": list(BUDGETS),
        "transition_budgets": list(TRANSITION_BUDGETS),
        "m30_deployment_noop": M30_DEPLOYMENT_NOOP,
        "post_first30_query_start": POST_FIRST30_QUERY_START,
        "fit": {
            "mode": "fixed_ridge_by_trial",
            "normalized_lambda": FIXED_RIDGE_NORMALIZED_LAMBDA,
            "float64_covariance": True,
            "residual_variance_floor": RESIDUAL_VARIANCE_FLOOR,
            "coefficient_fields": ["a", "c"],
        },
        "credible_region": {
            "statistic": "max_valid_unit_delta_ac_transpose_covariance_inverse_delta_ac",
            "chi2_df2_95_threshold": CREDIBLE_REGION_CHI2_DF2_95,
            "high_precision_monotonically_stricter": True,
            "typed_rejection_reason": "precision_credible_region",
        },
        "decoder_contract": {
            "precision_token_to_decoder": False,
            "posterior_or_target_refit": False,
            "model_parameter_changes": False,
            "normalizer_changes": False,
        },
        "sealed_cell_d_authority": sealed,
        "public_cli_imports_runtime": False,
    }
