"""Literal no-data contract and explicit closure for speed-smoke V2."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v1 import plan as v1_plan


CELL = "PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_SPEED_SMOKE_V2"
PHASE = "gpu0_single_m4_external_diagnostic_numerical_equivalence_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PRECISION_AWARE_CDM_D_SPEED_SMOKE_V2_DIAGNOSTIC_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "f3e4efa75dc998da85c9b6442633617edf7d80742b9952f9227f40369abdd0ba"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_speed_smoke_v2_authority"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_speed_smoke_v2"

V1_SCORE_ROOT_RELATIVE = v1_plan.SCORE_ROOT_RELATIVE
V1_FAILURE_SHAS = {
    "attempt.json": "68419c40f381737c3700b9893cfd7d4fa489c4a887efdaf50ea999a8a98d992b",
    "input_authority.json": "18b58d85cea9099360f7f0f4123ff99e45a258b3ea9a6928f2cf1240e77e111c",
    "failure.json": "dd1f6b9bd4ced42ff24a4a4ef1e43eb031987541282f1e8be638cb1b772f37a4",
}
V1_FAILURE_ERROR_SHA256 = "db39bc6275e036c0fb90da7bdb0d487ed988cab9fae32fbb183bf1b2048d9e81"

GPU0_PROFILE = dict(v1_plan.GPU0_PROFILE)
CANONICAL_SOURCE_ROOTS = dict(v1_plan.CANONICAL_SOURCE_ROOTS)
LOGICAL_EVAL_BATCH_SIZE = v1_plan.LOGICAL_EVAL_BATCH_SIZE
BASELINE_PHYSICAL_EVAL_BATCH_SIZE = v1_plan.BASELINE_PHYSICAL_EVAL_BATCH_SIZE
OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES = v1_plan.OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES

V1_MAX_ABS_PREDICTION_ERROR = v1_plan.PREDICTION_MAX_ABSOLUTE_TOLERANCE
V1_R2_ABS_ERROR = v1_plan.R2_ABSOLUTE_TOLERANCE
V1_SPEEDUP_EXCLUSIVE = v1_plan.SMOKE_MIN_SPEEDUP_RATIO
V2_MAX_ABS_PREDICTION_ERROR = 2e-6
V2_R2_ABS_ERROR = 2e-7
RECOMMENDED_SPEEDUP_RATIO = 1.5
SMOKE_ROWS = v1_plan.SMOKE_ROWS
SMOKE_EXECUTION_ORDER = v1_plan.SMOKE_EXECUTION_ORDER


class SpeedSmokeV2PlanError(RuntimeError):
    """Fail closed for V2 literal, closure, or launch-environment drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpeedSmokeV2PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"speed-smoke V2 {label} must be a lowercase SHA256",
    )
    return value


def canonical_source_environment_payload() -> dict[str, object]:
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_source_environment_v1",
        "roots": dict(CANONICAL_SOURCE_ROOTS),
        "literal_absolute_path_equality": True,
        "relative_alias_or_symlink_spelling_forbidden": True,
        "resolve_stat_or_open_before_attempt": False,
    }


def validate_selected_launch_environment(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    values = os.environ if environ is None else environ
    _require(values.get("CUDA_VISIBLE_DEVICES") == "0", "speed-smoke V2 requires CUDA_VISIBLE_DEVICES=0")
    _require(values.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "speed-smoke V2 requires PCI_BUS_ID ordering")
    _require(
        all(values.get(key) == expected for key, expected in CANONICAL_SOURCE_ROOTS.items()),
        "speed-smoke V2 canonical SUBC/SUBM source-root environment drift",
    )
    return {
        "selected_device_profile": dict(GPU0_PROFILE),
        "canonical_source_environment": canonical_source_environment_payload(),
        "gpu1_rejected_for_current_speed_smoke": True,
    }


@dataclass(frozen=True)
class V1FailureContract:
    """Exact immutable V1 failed score topology used only as predecessor evidence."""

    score_root_relative: str = V1_SCORE_ROOT_RELATIVE
    body_sha256s: Mapping[str, str] = field(default_factory=lambda: dict(V1_FAILURE_SHAS))

    def payload(self) -> dict[str, object]:
        bodies = dict(self.body_sha256s)
        _require(self.score_root_relative == V1_SCORE_ROOT_RELATIVE, "speed-smoke V2 V1 score root drift")
        _require(bodies == V1_FAILURE_SHAS, "speed-smoke V2 V1 failure body SHA literal drift")
        return {
            "schema": "precision_aware_cdmd_speed_smoke_v1_failed_graph_contract_v1",
            "score_root_relative": self.score_root_relative,
            "body_sha256s": {name: require_sha256(bodies[name], f"V1 failure {name}") for name in sorted(V1_FAILURE_SHAS)},
            "exact_json_body_count": 3,
            "exact_leaf_count": 6,
            "canonical_basename_sidecars": True,
            "all_leaves_regular_0444_nlink1": True,
            "failure_stage": "budget_m4",
            "failure_error_class": "SpeedSmokeError",
            "failure_error_sha256": V1_FAILURE_ERROR_SHA256,
            "score_and_terminal_absent": True,
            "target_optimizer_backward_update": 0,
            "failure_is_diagnostic_predecessor_not_retry_authority": True,
        }


V1_FAILURE = V1FailureContract()

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v2/__init__.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v2/plan.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v2/score.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_speed_smoke_v2/physical.py",
    "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_speed_smoke_v2.py",
    "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_speed_smoke_v2.py",
)
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((*v1_plan.IMPLEMENTATION_PATHS, *_OWNED_PATHS)))


def _read_regular_no_follow(path: Path) -> str:
    descriptor = -1
    try:
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        _require(isinstance(no_follow, int) and no_follow != 0, "speed-smoke V2 closure requires O_NOFOLLOW")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow)
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 f"speed-smoke V2 closure leaf is not regular/no-follow: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise SpeedSmokeV2PlanError(f"speed-smoke V2 closure leaf unavailable: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "speed-smoke V2 closure path topology drift")
        rows = [
            {"path": relative, "sha256": require_sha256(self.sha256_by_path[relative], f"closure {relative}")}
            for relative in IMPLEMENTATION_PATHS
        ]
        workorder_sha = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder_sha == WORKORDER_SHA256, "speed-smoke V2 workorder bytes drift")
        body = {"schema": "precision_aware_cdmd_speed_smoke_v2_closure_v1", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({relative: _read_regular_no_follow(base / relative) for relative in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "speed-smoke V2 closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "precision_aware_cdmd_speed_smoke_v2_closure_v1":
        raise SpeedSmokeV2PlanError("speed-smoke V2 closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "speed-smoke V2 closure count drift")
    rebuilt: dict[str, str] = {}
    for relative, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == relative,
                 "speed-smoke V2 closure path/order drift")
        rebuilt[relative] = require_sha256(row.get("sha256"), f"closure {relative}")
    expected = ImplementationClosure(rebuilt).payload()
    _require(dict(value) == expected, "speed-smoke V2 closure canonical payload drift")
    return expected


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    _require(relative in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}, "speed-smoke V2 root is outside topology")
    candidate = Path(root).absolute() / relative
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SpeedSmokeV2PlanError("speed-smoke V2 prospective root cannot be inspected") from error
    raise SpeedSmokeV2PlanError(f"speed-smoke V2 prospective root already exists: {candidate}")


def dry_plan() -> dict[str, object]:
    """Literal-only public contract; no Torch, result, data, or device operation."""
    return {
        "schema": "precision_aware_cdmd_speed_smoke_v2_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "v1_failed_predecessor": V1_FAILURE.payload(),
        "accepted_precision_v2_is_revalidated_through_v1_identity": True,
        "current_gpu0_required": dict(GPU0_PROFILE),
        "canonical_source_environment": canonical_source_environment_payload(),
        "smoke_execution_order": [item.payload() for item in SMOKE_ROWS],
        "baseline": {"physical_eval_batch_size": BASELINE_PHYSICAL_EVAL_BATCH_SIZE, "historical_anchor_exact": True},
        "optimized_o1_o2": {
            "physical_eval_batch_candidates": list(OPTIMIZED_PHYSICAL_EVAL_BATCH_CANDIDATES),
            "v1_thresholds": {
                "prediction_max_abs": V1_MAX_ABS_PREDICTION_ERROR,
                "r2_abs": V1_R2_ABS_ERROR,
                "speedup_exclusive": V1_SPEEDUP_EXCLUSIVE,
            },
            "v2_terminal_thresholds": {
                "prediction_max_abs": V2_MAX_ABS_PREDICTION_ERROR,
                "r2_abs": V2_R2_ABS_ERROR,
                "speedup_is_descriptive": True,
            },
            "recommended_speedup_ratio": RECOMMENDED_SPEEDUP_RATIO,
        },
        "m10_m30_rerun": False,
        "target_optimizer_backward_update": 0,
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_durable_authority_and_opaque_in_process_capability",
        "no_torch_import": True,
        "no_data_checkpoint_cuda_result_or_launch": True,
    }
