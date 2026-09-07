"""Static, no-data contract for Precision-Aware CDM-D V2."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CELL = "PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_V2"
PHASE = "conditional_gaussian_ridge_fwer_frozen_support_transition"
SCHEMA = "precision_aware_causal_dual_memory_cell_d_v2"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_V2_20260826.md"
WORKORDER_SHA256 = "8e25cfd8df7868dbaec2596825211eb8724ccaed98023eaba6731028ba85e879"

BUDGETS = (4, 10, 30)
TRANSITION_BUDGETS = (4, 10)
POST_FIRST30_QUERY_START = 30
FIXED_RIDGE_NORMALIZED_LAMBDA = 0.1
POSTERIOR_VARIANCE_FLOOR = 1.0e-12
FAMILYWISE_ALPHA = 0.05
M30_DEPLOYMENT_NOOP = True

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


class PrecisionAwareV2PlanError(RuntimeError):
    """Fail closed for V2 static contract or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionAwareV2PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value),
        f"precision V2 {label} must be a lowercase SHA256",
    )
    return value


def bonferroni_chi2_df2_threshold(valid_unit_count: int) -> float:
    """Exact chi-square(2) FWER threshold: P(any false reject) <= 0.05."""
    _require(type(valid_unit_count) is int and valid_unit_count >= 4,
             "precision V2 requires at least four valid units for the FWER gate")
    result = -2.0 * math.log(FAMILYWISE_ALPHA / float(valid_unit_count))
    _require(math.isfinite(result) and result > 0.0, "precision V2 Bonferroni threshold drift")
    return result


@dataclass(frozen=True)
class SealedCellDAuthority:
    """Identity-only frozen Cell-D and ordinary-OLS authority."""

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
            "precision V2 may only bind frozen ordinary OLS moments",
        )
        return {
            "schema": "precision_aware_cdmd_v2_sealed_cell_d_authority_v1",
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
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/__init__.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/plan.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/transition.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/physical.py",
    "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_v2.py",
    "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_v2.py",
)
IMPLEMENTATION_PATHS = (
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    *_OWNED_PATHS,
)


def _read_regular_no_follow(path: Path) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    _require(isinstance(no_follow, int) and no_follow != 0,
             "precision V2 closure requires O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PrecisionAwareV2PlanError(f"precision V2 closure path unavailable: {path}") from error
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode), f"precision V2 closure path is not a regular file: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "precision V2 closure path topology drift")
        rows = [
            {"path": relative, "sha256": require_sha256(self.sha256_by_path[relative], f"closure {relative}")}
            for relative in IMPLEMENTATION_PATHS
        ]
        workorder = next(item["sha256"] for item in rows if item["path"] == WORKORDER_RELATIVE)
        _require(workorder == WORKORDER_SHA256, "precision V2 workorder bytes drift")
        body = {"schema": "precision_aware_cdmd_v2_implementation_closure_v1", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({relative: _read_regular_no_follow(base / relative) for relative in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "precision V2 closure must be a mapping")
    candidate = dict(value)
    _require(
        set(candidate) == {"schema", "paths", "closure_sha256"}
        and candidate.get("schema") == "precision_aware_cdmd_v2_implementation_closure_v1",
        "precision V2 closure schema drift",
    )
    rows = candidate.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "precision V2 closure row count drift")
    rebuilt: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(
            isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
            "precision V2 closure path/order drift",
        )
        rebuilt[expected] = require_sha256(row.get("sha256"), f"closure {expected}")
    result = ImplementationClosure(rebuilt).payload()
    _require(candidate == result, "precision V2 closure canonical payload drift")
    return result


def dry_plan() -> dict[str, object]:
    """Public static contract; imports neither Torch nor runtime modules."""
    return {
        "schema": "precision_aware_cdmd_v2_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "budgets": list(BUDGETS),
        "transition_budgets": list(TRANSITION_BUDGETS),
        "m30_literal_deployment_noop": M30_DEPLOYMENT_NOOP,
        "post_first30_query_start": POST_FIRST30_QUERY_START,
        "support_fit": {
            "mode": "fixed_ridge_by_trial",
            "normalized_lambda": FIXED_RIDGE_NORMALIZED_LAMBDA,
            "float64": True,
            "no_pseudo_refit": True,
        },
        "conditional_posterior": {
            "covariance": "max(SSE/(M-tr(XA^-1X')),1e-12)*A^-1",
            "retained_coefficients": ["a", "c"],
            "not_sampling_sandwich_covariance": True,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "threshold": "-2*log(0.05/N_valid)",
            "reference": "frozen_support_only_initial_fixed_ridge_not_current_active",
        },
        "decoder_contract": {
            "precision_token_to_decoder": False,
            "model_parameter_changes": False,
            "normalizer_refit": False,
            "target_access_or_update": False,
        },
        "sealed_cell_d_authority": SEALED_CELL_D_AUTHORITY.payload(),
        "public_cli_imports_runtime": False,
        "execution_capability_issued": False,
    }
