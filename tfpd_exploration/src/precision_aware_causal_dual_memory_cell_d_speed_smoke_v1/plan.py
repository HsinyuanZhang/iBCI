"""Literal contract and no-glob closure for the Precision-V2 speed smoke.

This module intentionally imports no Torch and never touches an authority,
result, source root, checkpoint, or CUDA device.  The descriptor reader and
future physical composition live in sibling modules behind an opaque root
capability.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v1 import plan as base_plan
from src.causal_dual_memory_cell_d_speed_v1 import plan as speed_stage0_plan
from src.precision_aware_causal_dual_memory_cell_d_score_v2 import plan as precision_v2_plan


CELL = "PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_SPEED_SMOKE_V1"
PHASE = "gpu0_single_m4_external_eager_baseline_vs_o1_o2_speed_smoke"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PRECISION_AWARE_CDM_D_SPEED_SMOKE_V1_20260826.md"
WORKORDER_SHA256 = "39619dfa213490f618ab388117851c89cf438fbf427398fcf5f94decab9825ad"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_speed_smoke_v1_authority"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_speed_smoke_v1"

V2_AUTHORITY_ROOT_RELATIVE = precision_v2_plan.AUTHORITY_ROOT_RELATIVE
V2_SCORE_ROOT_RELATIVE = precision_v2_plan.SCORE_ROOT_RELATIVE
V2_AUTHORITY_SHAS = {
    "official_preflight.json": "6c08320054b24ea9281ab521719462c7a03ca023d8e214730f5b78a723b56536",
    "root_authorization.json": "e6bb2ea84806867ba95622ce6265f281e33df9e87c5f5056acd4ea242d81ce80",
}
V2_RESULT_SHAS = {
    "attempt.json": "71897d2026b11c7e7ae18b760c189e8eda9c255094d46350f459b84e201e9061",
    "input_authority.json": "5cbd7376f3fed117de02ce532be20ef09cb314972ce02c2e9e05415b851278df",
    "score.json": "4e06ffc544210fbb7cba68c21fd86831c7d0a0407d43ef379a33b9a05bb2bcb1",
    "terminal.json": "dfff5ebb6dd1cbab6e40547d86e6db25715e7a34262054e0406dd144eb8e441c",
}
V2_ATTEMPT_IDENTITY_SHA256 = "5d56583032dca0a3e6bdda83fc0f579f288a7569174cc965fe23f59876ce52da"
V2_FINAL_CLOSURE_SHA256 = "c6c14b1e1aa9f19639195558604ce6fc2a6517573ca66461ac570b8499e75f08"

GPU0_PROFILE = dict(base_plan.COMPATIBLE_DEVICE_PROFILES["gpu0"])
HISTORICAL_V2_SELECTED_PROFILE = dict(base_plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
CANONICAL_SOURCE_ROOTS = dict(precision_v2_plan.CANONICAL_SOURCE_ROOTS)
LOGICAL_EVAL_BATCH_SIZE = speed_stage0_plan.LOGICAL_EVAL_BATCH_SIZE
R2_ABSOLUTE_TOLERANCE = 1e-7
PREDICTION_MAX_ABSOLUTE_TOLERANCE = 1e-6
BASELINE_PHYSICAL_EVAL_BATCH_SIZE = 128
OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES = (1024, 512, 128)
SMOKE_MIN_SPEEDUP_RATIO = 1.0
RECOMMENDED_SPEEDUP_RATIO = 1.5


class SpeedSmokePlanError(RuntimeError):
    """Fail closed for literal, closure, or launch-environment drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpeedSmokePlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"speed-smoke {label} must be a lowercase SHA256",
    )
    return value


@dataclass(frozen=True)
class SmokeRowWitness:
    """The one historical V2 row whose exact bytes gate one smoke cell."""

    budget: int
    surface: str
    session: str
    canonical_row_sha256: str
    governing_r2: float
    prediction_sha256: str
    input_record_sha256: str
    n_windows: int
    carrier_commits: int
    transition_count: int
    model_state_sha256: str
    sealed_model_load_proof_sha256: str

    def payload(self) -> dict[str, object]:
        _require(self.budget == 4, "speed-smoke row budget drift")
        _require(self.surface in {"within", "external"} and isinstance(self.session, str) and self.session,
                 "speed-smoke row surface/session drift")
        _require(type(self.governing_r2) is float and self.governing_r2 == self.governing_r2,
                 "speed-smoke row R2 drift")
        _require(type(self.n_windows) is int and self.n_windows > 0, "speed-smoke row window count drift")
        _require(type(self.carrier_commits) is int and type(self.transition_count) is int
                 and 0 <= self.carrier_commits <= self.transition_count and self.transition_count > 0,
                 "speed-smoke row carrier-transition count drift")
        for label, value in (
            ("row", self.canonical_row_sha256), ("prediction", self.prediction_sha256),
            ("input", self.input_record_sha256), ("model state", self.model_state_sha256),
            ("sealed load", self.sealed_model_load_proof_sha256),
        ):
            require_sha256(value, f"row {label}")
        return {
            "budget": self.budget,
            "surface": self.surface,
            "session": self.session,
            "canonical_row_sha256": self.canonical_row_sha256,
            "governing_r2": self.governing_r2,
            "prediction_sha256": self.prediction_sha256,
            "input_record_sha256": self.input_record_sha256,
            "n_windows": self.n_windows,
            "carrier_transition_committed_count": self.carrier_commits,
            "transition_count": self.transition_count,
            "model_state_before_and_after_sha256": self.model_state_sha256,
            "sealed_model_load_proof_sha256": self.sealed_model_load_proof_sha256,
        }


SMOKE_ROWS = (
    SmokeRowWitness(
        budget=4, surface="external", session="sub-M_ses-CO-20150615",
        canonical_row_sha256="46247c88f3a4bfd880367ac58890a25ab803ba1b23ea72fda94c8c16c0c3062f",
        governing_r2=0.3249187469482422,
        prediction_sha256="0e8644e163b8e463d36f1fc29ca7da3f87a11f00addccad6e159ba75f78bd1b3",
        input_record_sha256="3c86adf7250da43db5cea5fb5eec2f4fccf8e871332c5a15c7248030ccd34f23",
        n_windows=34370, carrier_commits=5, transition_count=143,
        model_state_sha256="c7a8489d5e17583ac210dd06b09e931ef6ba0d6b98a2767cae26d897725cf8fc",
        sealed_model_load_proof_sha256="0e61cce60cf827b62589fccc2a29356f17d035c279c1beb628e34b1bdebd71b7",
    ),
)
SMOKE_EXECUTION_ORDER = tuple((row.budget, row.surface, row.session) for row in SMOKE_ROWS)


@dataclass(frozen=True)
class CompletedV2Contract:
    """Exact 12-leaf historical input to the current independent smoke."""

    authority_root_relative: str = V2_AUTHORITY_ROOT_RELATIVE
    score_root_relative: str = V2_SCORE_ROOT_RELATIVE
    authority_body_sha256s: Mapping[str, str] = field(default_factory=lambda: dict(V2_AUTHORITY_SHAS))
    score_body_sha256s: Mapping[str, str] = field(default_factory=lambda: dict(V2_RESULT_SHAS))

    def payload(self) -> dict[str, object]:
        authority = dict(self.authority_body_sha256s)
        score = dict(self.score_body_sha256s)
        _require(self.authority_root_relative == V2_AUTHORITY_ROOT_RELATIVE,
                 "speed-smoke V2 authority root literal drift")
        _require(self.score_root_relative == V2_SCORE_ROOT_RELATIVE,
                 "speed-smoke V2 score root literal drift")
        _require(authority == V2_AUTHORITY_SHAS and score == V2_RESULT_SHAS,
                 "speed-smoke V2 SHA literal drift")
        return {
            "schema": "precision_aware_cdmd_speed_smoke_completed_v2_contract_v1",
            "authority_root_relative": self.authority_root_relative,
            "authority_body_sha256s": {name: require_sha256(authority[name], f"V2 authority {name}")
                                       for name in sorted(V2_AUTHORITY_SHAS)},
            "score_root_relative": self.score_root_relative,
            "score_body_sha256s": {name: require_sha256(score[name], f"V2 score {name}")
                                    for name in sorted(V2_RESULT_SHAS)},
            "exact_json_body_count": 6,
            "exact_leaf_count": 12,
            "all_leaves_regular_0444_nlink1": True,
            "canonical_basename_sidecars": True,
            "attempt_identity_sha256": V2_ATTEMPT_IDENTITY_SHA256,
            "historical_final_and_launch_closure_sha256": V2_FINAL_CLOSURE_SHA256,
            "historical_selected_device_profile": dict(HISTORICAL_V2_SELECTED_PROFILE),
            "completed_terminal_required": True,
            "failure_absent": True,
            "target_optimizer_backward_update": 0,
        }


COMPLETED_V2 = CompletedV2Contract()


def canonical_source_environment_payload() -> dict[str, object]:
    return {
        "schema": "precision_aware_cdmd_speed_smoke_source_environment_v1",
        "roots": dict(CANONICAL_SOURCE_ROOTS),
        "literal_absolute_path_equality": True,
        "relative_alias_or_symlink_spelling_forbidden": True,
        "resolve_stat_or_open_before_attempt": False,
    }


def validate_gpu0_profile(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping) and dict(value) == GPU0_PROFILE,
             "speed-smoke current execution requires the exact GPU0 compatible profile")
    return dict(GPU0_PROFILE)


def validate_selected_launch_environment(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Lexically reject GPU1, relative, missing, or swapped variables before write.

    This does not call ``Path.resolve``, stat a source root, or import Torch.
    """
    values = os.environ if environ is None else environ
    _require(values.get("CUDA_VISIBLE_DEVICES") == "0", "speed-smoke requires CUDA_VISIBLE_DEVICES=0")
    _require(values.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "speed-smoke requires PCI_BUS_ID ordering")
    _require(all(values.get(key) == expected for key, expected in CANONICAL_SOURCE_ROOTS.items()),
             "speed-smoke canonical SUBC/SUBM source-root environment drift")
    return {
        "selected_device_profile": dict(GPU0_PROFILE),
        "canonical_source_environment": canonical_source_environment_payload(),
        "gpu1_rejected_for_current_speed_smoke": True,
    }


_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v1/__init__.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v1/plan.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v1/score.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v1/physical.py",
    "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_speed_smoke_v1.py",
    "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_speed_smoke_v1.py",
)
# Explicit transitive closure: accepted V2's current environment wrapper,
# accepted Stage-0 speed runtime, and these successor leaves.  No globbing,
# import discovery, result-byte paths, or ambient module names participate.
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    *precision_v2_plan.IMPLEMENTATION_PATHS,
    *speed_stage0_plan.IMPLEMENTATION_PATHS,
    *_OWNED_PATHS,
)))


def _read_regular_no_follow(path: Path) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    _require(isinstance(no_follow, int) and no_follow != 0, "speed-smoke closure requires O_NOFOLLOW")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow)
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 f"speed-smoke closure leaf is not regular/no-follow: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise SpeedSmokePlanError(f"speed-smoke closure leaf unavailable: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "speed-smoke closure path topology drift")
        rows = [
            {"path": relative, "sha256": require_sha256(self.sha256_by_path[relative], f"closure {relative}")}
            for relative in IMPLEMENTATION_PATHS
        ]
        workorder_sha = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder_sha == WORKORDER_SHA256, "speed-smoke workorder bytes drift")
        body = {"schema": "precision_aware_cdmd_speed_smoke_closure_v1", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({relative: _read_regular_no_follow(base / relative) for relative in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "speed-smoke closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "precision_aware_cdmd_speed_smoke_closure_v1":
        raise SpeedSmokePlanError("speed-smoke closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "speed-smoke closure row count drift")
    rebuilt: dict[str, str] = {}
    for relative, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == relative,
                 "speed-smoke closure path/order drift")
        rebuilt[relative] = require_sha256(row.get("sha256"), f"closure {relative}")
    expected = ImplementationClosure(rebuilt).payload()
    _require(dict(value) == expected, "speed-smoke closure canonical payload drift")
    return expected


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    _require(relative in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}, "speed-smoke root is outside topology")
    candidate = Path(root).absolute() / relative
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SpeedSmokePlanError("speed-smoke prospective root cannot be inspected") from error
    raise SpeedSmokePlanError(f"speed-smoke prospective root already exists: {candidate}")


def dry_plan() -> dict[str, object]:
    """Public literal-only route description; deliberately no Torch or I/O."""
    return {
        "schema": "precision_aware_cdmd_speed_smoke_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "completed_precision_v2": COMPLETED_V2.payload(),
        "historical_v2_gpu1_evidence_only": dict(HISTORICAL_V2_SELECTED_PROFILE),
        "current_gpu0_required": dict(GPU0_PROFILE),
        "canonical_source_environment": canonical_source_environment_payload(),
        "smoke_execution_order": [row.payload() for row in SMOKE_ROWS],
        "m30_rerun": False,
        "sealed_comparator_rerun": False,
        "baseline": {
            "physical_eval_batch_size": BASELINE_PHYSICAL_EVAL_BATCH_SIZE,
            "historical_prediction_sha256_and_r2_exact": True,
        },
        "optimized_o1_o2": {
            "physical_eval_batch_candidates": list(OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES),
            "fallback_order": list(OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES),
            "prediction_sha256_disclosed_not_required_equal": True,
            "prediction_max_absolute_tolerance": PREDICTION_MAX_ABSOLUTE_TOLERANCE,
            "r2_absolute_tolerance": R2_ABSOLUTE_TOLERANCE,
            "transition_sequence_exact": True,
            "smoke_min_speedup_ratio_exclusive": SMOKE_MIN_SPEEDUP_RATIO,
            "recommended_speedup_ratio": RECOMMENDED_SPEEDUP_RATIO,
        },
        "logical_eval_batch_size": LOGICAL_EVAL_BATCH_SIZE,
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_durable_authority_and_opaque_in_process_capability",
        "no_torch_import": True,
        "no_data_checkpoint_cuda_result_or_launch": True,
    }
