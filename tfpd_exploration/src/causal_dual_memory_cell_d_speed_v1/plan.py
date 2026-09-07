"""Static contract and explicit closure for the CDM-D pure-speed path."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.precision_aware_causal_dual_memory_cell_d_score_v1 import plan as precision_score_plan


CELL = "CAUSAL_DUAL_MEMORY_CELL_D_SPEED_V1"
PHASE = "identity_cache_and_sampled_repeat_audit_b128_only"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_SPEED_V1_20260826.md"
WORKORDER_SHA256 = "264468964aa6ec233046e30a2ec5ead20b0e2114ff5e95e94fcae68bf878364f"
OPTIMIZATION_NOTES_RELATIVE = "tfpd_exploration/docs/OPTIMIZATION_NOTES_CDM_SPEED_20260826.md"

LOGICAL_EVAL_BATCH_SIZE = 128
DEFERRED_NUMERIC_BATCH_VARIANTS = (1024, 2048)
REPEAT_MID_SESSION_RULE = "floor(n_query_trials/2)_held_group_0_first_logical_chunk"
IDENTITY_CACHE_KEY_SEMANTICS = (
    "path_plus_contiguous_activity_prefix_normalized_t4_and_held_view_bytes"
)


class CDMDSpeedPlanError(RuntimeError):
    """Fail closed for acceleration closure or static contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CDMDSpeedPlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value),
        f"CDM-D speed {label} must be a lowercase SHA256",
    )
    return value


_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_speed_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_speed_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_speed_v1/accelerator.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_speed_v1/physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_speed_v1.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_speed_v1.py",
)

# These are explicit, closure-bound physical dependencies.  No glob or
# ambient import discovery is accepted by the successor contract.
# The speed mixin dispatches through the accepted Precision-V2 runtime, which
# itself composes V8/V5/V1 and the exact source-execution helper.  Bind that
# already-reviewed explicit no-glob closure wholesale rather than pretending
# that the three direct imports in ``physical.py`` are its complete runtime
# dependency graph.  The speed-specific leaves follow it deterministically.
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    *precision_score_plan.IMPLEMENTATION_PATHS,
    OPTIMIZATION_NOTES_RELATIVE,
    *_OWNED_PATHS,
)))


def _read_regular_no_follow(path: Path) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    _require(isinstance(no_follow, int) and no_follow != 0, "speed closure requires O_NOFOLLOW")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow)
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 f"speed closure leaf is not a regular no-follow file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise CDMDSpeedPlanError(f"speed closure leaf unavailable: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "speed closure path topology drift")
        rows = [
            {"path": relative, "sha256": require_sha256(self.sha256_by_path[relative], f"closure {relative}")}
            for relative in IMPLEMENTATION_PATHS
        ]
        workorder_sha = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder_sha == WORKORDER_SHA256, "speed workorder bytes drift")
        body = {"schema": "causal_dual_memory_cell_d_speed_v1_closure_v1", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({relative: _read_regular_no_follow(base / relative) for relative in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "speed closure must be a mapping")
    candidate = dict(value)
    _require(
        set(candidate) == {"schema", "paths", "closure_sha256"}
        and candidate.get("schema") == "causal_dual_memory_cell_d_speed_v1_closure_v1",
        "speed closure schema drift",
    )
    rows = candidate.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "speed closure row count drift")
    rebuilt: dict[str, str] = {}
    for relative, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"}
                 and row.get("path") == relative, "speed closure path/order drift")
        rebuilt[relative] = require_sha256(row.get("sha256"), f"closure {relative}")
    result = ImplementationClosure(rebuilt).payload()
    _require(candidate == result, "speed closure canonical payload drift")
    return result


def dry_plan() -> dict[str, object]:
    """Return a literal-only contract; this module imports no Torch."""
    return {
        "schema": "causal_dual_memory_cell_d_speed_v1_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "pure_speed_only": True,
        "identity_cache": {
            "enabled": True,
            "semantics": IDENTITY_CACHE_KEY_SEMANTICS,
            "hard_fail_on_identity_encoder_gate_semantics": True,
            "cached_identity_batch_size": 1,
        },
        "repeat_audit": {
            "enabled": True,
            "canonical_full_first_chunk_per_exact_state": True,
            "one_held_group_0_mid_session_first_chunk": True,
            "fixed_mid_session_rule": REPEAT_MID_SESSION_RULE,
            "state_rng_dropout_model_invariants_retained": True,
        },
        "logical_eval_batch_size": LOGICAL_EVAL_BATCH_SIZE,
        "numeric_batch_variants": {
            "candidates": list(DEFERRED_NUMERIC_BATCH_VARIANTS),
            "selected": None,
            "deferred_reason": "real_cpu_prediction_sha256_parity_failed_at_2048",
            "not_selectable_in_pure_speed_successor": True,
        },
        "forbidden": {
            "o3_subsampling_or_interval_gate": True,
            "o5_concurrency_scheduler": True,
            "o6_disk_cache": True,
            "o7_tf32": True,
            "data_checkpoint_cuda_result_or_launch": True,
        },
        "public_cli_imports_torch": False,
        "public_cli_executes": False,
    }
