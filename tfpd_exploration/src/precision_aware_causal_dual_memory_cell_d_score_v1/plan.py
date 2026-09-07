"""Static contract for the Precision-Aware CDM-D V2 matched-score successor.

The predecessor V8 closure is intentionally historical receipt evidence.  The
successor closure below binds the current code that would execute a future
score, including the exact V8 evaluator and V2 transition leaves.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v8 import plan as v8plan
from src.precision_aware_causal_dual_memory_cell_d_v2 import plan as precision_v2_plan


CELL = "PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V1"
PHASE = "precision_v2_m10_m4_v8_reused_sealed_matched_score"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PRECISION_AWARE_CDM_D_MATCHED_SCORE_V1_20260826.md"
WORKORDER_SHA256 = "01a2b6530c85218811dc1e80c66ebe77450f941da6d61555f0fc7d7e101f8021"

WITHIN = v1plan.WITHIN
EXTERNAL = v1plan.EXTERNAL
SURFACES = (WITHIN, EXTERNAL)
BUDGETS = (10, 4)
M30_REFERENCE_BUDGET = 30
EVAL_BATCH_SIZE = 128
SYSTEM_PRECISION_V2 = "precision_aware_independent_activity_v2"
SYSTEM_REUSED_SEALED_V8 = "sealed_cell_d_swa_reused_v8"

AUTHORITY_ROOT_RELATIVE = (
    "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_matched_score_v1_authority"
)
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_matched_score_v1"

V8_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v8"
V8_HISTORICAL_CLOSURE_SHA256 = "62c685fb0d288107ed43d7a0ef469bbb655c32e68b0d7b677aac8b371ab94939"
V8_EXPECTED_SHAS = {
    "attempt.json": "557bf8071094d6512be862b1b56f0d8a949df57f5fe79356d76771d1fea2a376",
    "input_authority.json": "ada5427650d5e612232e74b12159b332721fd475bb218f9894ec3b30ecffdd07",
    "score.json": "98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e",
    "terminal.json": "80b2dff139a1298e1447c9313ae70548191c7406f2643a1cfb45a1867dec8096",
}
V8_EXPECTED_TOPOLOGY = tuple(V8_EXPECTED_SHAS)
V8_TERMINAL_VERDICT = "ADVANCE_SHORT_BUDGET"
PRECISION_V2_STATIC_CLOSURE_SHA256 = "17d966e90c038e8626b82c60b5645d9fb4ce8aeaef58f3b9126966ceefc5f0d0"

PAIRED_BOOTSTRAP_SEED = v1plan.PAIRED_BOOTSTRAP_SEED
PAIRED_BOOTSTRAP_DRAWS = v1plan.PAIRED_BOOTSTRAP_DRAWS


class PrecisionMatchedScorePlanError(RuntimeError):
    """Fail closed for static identity, closure, or root topology drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionMatchedScorePlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        f"precision matched score {label} must be a lowercase SHA256",
    )
    return value


@dataclass(frozen=True)
class V8PredecessorContract:
    """Literal complete V8 graph, not a rebuilt current V8 identity."""

    root_relative: str = V8_ROOT_RELATIVE
    historical_closure_sha256: str = V8_HISTORICAL_CLOSURE_SHA256
    body_sha256s: Mapping[str, str] | None = None
    terminal_verdict: str = V8_TERMINAL_VERDICT

    def payload(self) -> dict[str, object]:
        body = dict(V8_EXPECTED_SHAS if self.body_sha256s is None else self.body_sha256s)
        _require(self.root_relative == V8_ROOT_RELATIVE, "V8 predecessor root drift")
        _require(body == V8_EXPECTED_SHAS, "V8 predecessor body SHA literals drift")
        _require(self.historical_closure_sha256 == V8_HISTORICAL_CLOSURE_SHA256,
                 "V8 predecessor historical closure drift")
        _require(self.terminal_verdict == V8_TERMINAL_VERDICT, "V8 predecessor verdict drift")
        return {
            "schema": "precision_aware_cdmd_matched_score_v8_predecessor_v1",
            "root_relative": self.root_relative,
            "historical_implementation_closure_sha256": self.historical_closure_sha256,
            "body_sha256s": {name: require_sha(body[name], f"V8 {name}") for name in V8_EXPECTED_TOPOLOGY},
            "exact_json_body_count": 4,
            "exact_leaf_count": 8,
            "terminal_status": "TERMINAL",
            "terminal_verdict": self.terminal_verdict,
            "reused_budgets": [30, 10, 4],
            "reused_system": SYSTEM_REUSED_SEALED_V8,
        }


V8_PREDECESSOR = V8PredecessorContract()


def validate_v8_binding_witness(value: object) -> dict[str, object]:
    """Validate the dynamic, descriptor-derived portion of V8 lineage.

    Directory identity and canonical per-cell digests cannot be literals in a
    source file: they are observed only while one held no-follow V8 graph is
    validated.  The identity carries this witness so a later score receipt can
    bind every reused row to the same durable predecessor, while issuers and
    final revalidation exact-compare it to a fresh held-FD reconstruction.
    """
    required = {
        "schema", "contract", "directory_identity", "v8_identity_sha256", "v8_source_gate_binding",
        "input_records_sha256", "sealed_cell_sha256s", "binding_sha256",
    }
    _require(isinstance(value, Mapping) and set(value) == required,
             "V8 predecessor witness schema drift")
    _require(value.get("schema") == "precision_aware_cdmd_v8_binding_witness_v1",
             "V8 predecessor witness type drift")
    _require(value.get("contract") == V8_PREDECESSOR.payload(), "V8 predecessor witness contract drift")
    identity = value.get("directory_identity")
    _require(isinstance(identity, list) and len(identity) == 2
             and all(type(item) is int and item > 0 for item in identity),
             "V8 predecessor witness directory identity drift")
    require_sha(value.get("v8_identity_sha256"), "V8 predecessor identity")
    source_binding = value.get("v8_source_gate_binding")
    _require(isinstance(source_binding, Mapping) and isinstance(source_binding.get("binding_sha256"), str),
             "V8 predecessor witness source-gate binding drift")
    require_sha(source_binding.get("binding_sha256"), "V8 predecessor source-gate binding")
    require_sha(value.get("input_records_sha256"), "V8 predecessor input-record topology")
    cells = value.get("sealed_cell_sha256s")
    expected_cells = {f"m{budget}:{surface}" for budget in (30, 10, 4) for surface in SURFACES}
    _require(isinstance(cells, Mapping) and set(cells) == expected_cells,
             "V8 predecessor sealed-cell topology drift")
    canonical_cells = {key: require_sha(cells[key], f"V8 sealed cell {key}") for key in sorted(expected_cells)}
    body = {
        "schema": "precision_aware_cdmd_v8_binding_witness_v1",
        "contract": V8_PREDECESSOR.payload(),
        "directory_identity": list(identity),
        "v8_identity_sha256": value["v8_identity_sha256"],
        "v8_source_gate_binding": dict(source_binding),
        "input_records_sha256": value["input_records_sha256"],
        "sealed_cell_sha256s": canonical_cells,
    }
    expected = {**body, "binding_sha256": sha256_bytes(canonical_json_bytes(body))}
    _require(value == expected, "V8 predecessor witness canonical drift")
    return expected


@dataclass(frozen=True)
class ScoreSpec:
    """Non-adaptive four-cell successor matrix plus immutable M30 reference."""

    def payload(self) -> dict[str, object]:
        return {
            "schema": "precision_aware_cdmd_matched_score_spec_v1",
            "budgets_executed": list(BUDGETS),
            "m30_reference_only": True,
            "surfaces": list(SURFACES),
            "new_system": SYSTEM_PRECISION_V2,
            "reused_system": SYSTEM_REUSED_SEALED_V8,
            "new_matrix_cell_count": 4,
            "new_cell_order": [
                {"budget": budget, "surface": surface, "system": SYSTEM_PRECISION_V2}
                for budget in BUDGETS for surface in SURFACES
            ],
            "reused_sealed_cell_order": [
                {"budget": budget, "surface": surface, "system": SYSTEM_REUSED_SEALED_V8}
                for budget in (30, 10, 4) for surface in SURFACES
            ],
            "eval_batch_size": EVAL_BATCH_SIZE,
            "metric": "last_bin_variance_weighted_two_output_r2",
            "same_post_first30_query_pool": True,
            "same_input_authority": True,
            "target_optimizer_backward_update": 0,
            "normalizer_refit": False,
            "model_parameter_update": False,
            "precision_decoder_token_used": False,
            "m30_literal_no_proposal_or_state_update": True,
        }


PUBLIC_SPEC = ScoreSpec()

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v1/__init__.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v1/plan.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v1/score.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v1/physical.py",
    "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_matched_score_v1.py",
    "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_matched_score_v1.py",
)
# Explicit/no-glob transitive binding.  V8 provides parser/metric/lifecycle
# composition and V2 provides only the conditional-posterior transition law.
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    *v8plan.IMPLEMENTATION_PATHS,
    *precision_v2_plan.IMPLEMENTATION_PATHS,
    *_OWNED_PATHS,
)))


def _read_regular_no_follow(path: Path) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    _require(isinstance(no_follow, int) and no_follow != 0, "successor closure requires O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode), f"successor closure leaf is not regular: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise PrecisionMatchedScorePlanError(f"successor closure leaf unavailable: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "successor closure path topology drift")
        rows = [
            {"path": relative, "sha256": require_sha(self.sha256_by_path[relative], f"closure {relative}")}
            for relative in IMPLEMENTATION_PATHS
        ]
        workorder = next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE)
        _require(workorder == WORKORDER_SHA256, "successor workorder bytes drift")
        body = {"schema": "precision_aware_cdmd_matched_score_closure_v1", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({relative: _read_regular_no_follow(base / relative) for relative in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "successor closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "precision_aware_cdmd_matched_score_closure_v1":
        raise PrecisionMatchedScorePlanError("successor closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "successor closure row count drift")
    rebuilt: dict[str, str] = {}
    for expected, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == expected,
                 "successor closure path/order drift")
        rebuilt[expected] = require_sha(row.get("sha256"), f"closure {expected}")
    result = ImplementationClosure(rebuilt).payload()
    _require(dict(value) == result, "successor closure canonical payload drift")
    return result


@dataclass(frozen=True)
class ScoreIdentity:
    """Current successor identity plus literal historical V8 predecessor."""

    closure: Mapping[str, object]
    v8_binding: Mapping[str, object]
    selected_device_profile: Mapping[str, object] | None = None
    v8_predecessor: V8PredecessorContract = V8_PREDECESSOR

    def payload(self) -> dict[str, object]:
        closure = validate_implementation_closure(self.closure)
        selected = (
            dict(v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
            if self.selected_device_profile is None
            else v1plan.validate_compatible_device_profile(self.selected_device_profile)
        )
        return {
            "schema": "precision_aware_cdmd_matched_score_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "score_spec": PUBLIC_SPEC.payload(),
            "closure": closure,
            "v8_predecessor": self.v8_predecessor.payload(),
            "v8_predecessor_binding": validate_v8_binding_witness(self.v8_binding),
            "precision_v2_static_closure_sha256": require_sha(
                PRECISION_V2_STATIC_CLOSURE_SHA256, "Precision-V2 static closure"
            ),
            "sealed_cell_d": precision_v2_plan.SEALED_CELL_D_AUTHORITY.payload(),
            "selected_device_profile": selected,
            "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": SCORE_ROOT_RELATIVE,
            "source_only_predecessor": True,
            "target_updates_forbidden": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    _require(relative in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}, "successor root is outside topology")
    candidate = Path(root).absolute() / relative
    try:
        info = os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise PrecisionMatchedScorePlanError("successor prospective root cannot be inspected") from error
    raise PrecisionMatchedScorePlanError(
        f"successor prospective root already exists or aliases a live entry: {candidate} ({info.st_mode:o})"
    )


def assert_fresh_prospective_roots(root: Path) -> None:
    assert_fresh_prospective_root(root, AUTHORITY_ROOT_RELATIVE)
    assert_fresh_prospective_root(root, SCORE_ROOT_RELATIVE)


def dry_plan() -> dict[str, object]:
    """Public static plan; no runtime imports, reads, writes, or CUDA action."""
    return {
        "schema": "precision_aware_cdmd_matched_score_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "spec": PUBLIC_SPEC.payload(),
        "v8_predecessor": V8_PREDECESSOR.payload(),
        "precision_v2_static_closure_sha256": PRECISION_V2_STATIC_CLOSURE_SHA256,
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_durable_authority_and_opaque_in_process_capability",
        "no_torch_import": True,
        "no_data_access": True,
        "no_result_write": True,
    }
