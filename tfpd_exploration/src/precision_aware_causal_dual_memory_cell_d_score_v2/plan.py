"""Static V2 environment-successor identity and explicit implementation closure.

V1 receipt bytes remain historical evidence.  V2 deliberately carries the
V1-compatible science identity as an inherited base while binding its own
closure, roots, failed-predecessor graph, and lexical source-root contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.precision_aware_causal_dual_memory_cell_d_score_v1 import plan as v1plan


CELL = "PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V2"
PHASE = "precision_v2_m10_m4_v8_reused_sealed_matched_score_environment_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PRECISION_AWARE_CDM_D_MATCHED_SCORE_V2_ENVIRONMENT_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "3080d27b8c505576c019816d4b32c0c55d74b3a7c9a18f8c8c8dd0f023a29516"

AUTHORITY_ROOT_RELATIVE = (
    "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_matched_score_v2_authority"
)
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_matched_score_v2"

V1_AUTHORITY_ROOT_RELATIVE = v1plan.AUTHORITY_ROOT_RELATIVE
V1_SCORE_ROOT_RELATIVE = v1plan.SCORE_ROOT_RELATIVE
V1_AUTHORITY_SHAS = {
    "official_preflight.json": "cc51447ff154d57d40504c860114793259ec1083bdb2889c8996badbaba20874",
    "root_authorization.json": "3d00ef126c3c358f77f7ee098623ccf929d5c7277a73990186b7ea8ec6157483",
}
V1_FAILED_SCORE_SHAS = {
    "attempt.json": "59469ac2c250fb22f1df0c94c49d17473f8d1553c1d3ea5a81bf074d6be6c849",
    "failure.json": "435ccf2c300ec4d6180738c89a51737050b768225fcee28bc165d6ae0179e253",
}
V1_FAILURE_ERROR_SHA256 = "058a6e19dd1fee1b1febed44a30014b0b803f74cac6c587f3bf4238f4a1a936d"

CANONICAL_SOURCE_ROOTS = {
    "SUBC_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C",
    "SUBM_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M",
}


class PrecisionMatchedScoreV2PlanError(RuntimeError):
    """Fail closed for successor closure, root, or environment drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionMatchedScoreV2PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        f"Precision score V2 {label} must be a lowercase SHA256",
    )
    return value


def canonical_source_environment_payload() -> dict[str, object]:
    """Literal-only contract: deliberately no source-root path operation."""
    return {
        "schema": "precision_aware_cdmd_matched_score_v2_source_environment_v1",
        "roots": dict(CANONICAL_SOURCE_ROOTS),
        "literal_absolute_path_equality": True,
        "resolve_stat_or_source_open_before_attempt": False,
        "relative_alias_or_symlink_spelling_forbidden": True,
    }


@dataclass(frozen=True)
class V1FailedPredecessorContract:
    """Exact immutable V1 authority plus failed attempt/failure topology."""

    authority_root_relative: str = V1_AUTHORITY_ROOT_RELATIVE
    score_root_relative: str = V1_SCORE_ROOT_RELATIVE
    authority_body_sha256s: Mapping[str, str] = field(default_factory=lambda: dict(V1_AUTHORITY_SHAS))
    failed_score_body_sha256s: Mapping[str, str] = field(default_factory=lambda: dict(V1_FAILED_SCORE_SHAS))
    error_sha256: str = V1_FAILURE_ERROR_SHA256

    def payload(self) -> dict[str, object]:
        authority = dict(self.authority_body_sha256s)
        failed = dict(self.failed_score_body_sha256s)
        _require(self.authority_root_relative == V1_AUTHORITY_ROOT_RELATIVE, "V1 authority root literal drift")
        _require(self.score_root_relative == V1_SCORE_ROOT_RELATIVE, "V1 failed-score root literal drift")
        _require(authority == V1_AUTHORITY_SHAS, "V1 authority SHA literals drift")
        _require(failed == V1_FAILED_SCORE_SHAS, "V1 failed-score SHA literals drift")
        _require(self.error_sha256 == V1_FAILURE_ERROR_SHA256, "V1 failure error SHA literal drift")
        return {
            "schema": "precision_aware_cdmd_matched_score_v1_failed_predecessor_v1",
            "authority_root_relative": self.authority_root_relative,
            "authority_body_sha256s": {
                name: require_sha(authority[name], f"V1 authority {name}") for name in sorted(V1_AUTHORITY_SHAS)
            },
            "authority_exact_json_body_count": 2,
            "authority_exact_leaf_count": 4,
            "score_root_relative": self.score_root_relative,
            "failed_score_body_sha256s": {
                name: require_sha(failed[name], f"V1 failed score {name}") for name in sorted(V1_FAILED_SCORE_SHAS)
            },
            "failed_score_exact_json_body_count": 2,
            "failed_score_exact_leaf_count": 4,
            "all_leaves_regular_0444_nlink1": True,
            "failure_stage": "materialize_inputs",
            "failure_error_class": "PhysicalCDMDScoreError",
            "failure_error_sha256": require_sha(self.error_sha256, "V1 failure error"),
            "within_external_assets_opened": False,
            "full_system_forward_count": 0,
            "group_forward_count": 0,
            "target_optimizer_backward_update": 0,
            "checkpoint_opened": True,
            "cuda_initialized": True,
            "terminal_absent": True,
            "environment_failure_not_performance_result": True,
        }


V1_FAILED_PREDECESSOR = V1FailedPredecessorContract()


def validate_v1_failed_predecessor_binding(value: object) -> dict[str, object]:
    required = {
        "schema", "contract", "authority_directory_identity", "score_directory_identity",
        "historical_v1_identity_sha256", "binding_sha256",
    }
    _require(isinstance(value, Mapping) and set(value) == required, "V1 failed predecessor binding schema drift")
    _require(value.get("schema") == "precision_aware_cdmd_matched_score_v1_failed_binding_v1",
             "V1 failed predecessor binding type drift")
    _require(value.get("contract") == V1_FAILED_PREDECESSOR.payload(), "V1 failed predecessor contract drift")
    for label in ("authority_directory_identity", "score_directory_identity"):
        identity = value.get(label)
        _require(isinstance(identity, list) and len(identity) == 2
                 and all(type(item) is int and item > 0 for item in identity),
                 f"V1 failed predecessor {label} drift")
    require_sha(value.get("historical_v1_identity_sha256"), "V1 historical identity")
    body = {
        "schema": "precision_aware_cdmd_matched_score_v1_failed_binding_v1",
        "contract": V1_FAILED_PREDECESSOR.payload(),
        "authority_directory_identity": list(value["authority_directory_identity"]),
        "score_directory_identity": list(value["score_directory_identity"]),
        "historical_v1_identity_sha256": value["historical_v1_identity_sha256"],
    }
    expected = {**body, "binding_sha256": sha256_bytes(canonical_json_bytes(body))}
    _require(dict(value) == expected, "V1 failed predecessor binding canonical drift")
    return expected


_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v2/__init__.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v2/plan.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v2/score.py",
    "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v2/physical.py",
    "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_matched_score_v2.py",
    "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_matched_score_v2.py",
)
IMPLEMENTATION_PATHS = tuple(dict.fromkeys((*v1plan.IMPLEMENTATION_PATHS, *_OWNED_PATHS)))


def _read_regular_no_follow(path: Path) -> str:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    _require(isinstance(no_follow, int) and no_follow != 0, "V2 closure requires O_NOFOLLOW")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow)
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                 f"V2 closure leaf is not regular: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise PrecisionMatchedScoreV2PlanError(f"V2 closure leaf unavailable: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        _require(set(self.sha256_by_path) == set(IMPLEMENTATION_PATHS), "V2 closure path topology drift")
        rows = [
            {"path": relative, "sha256": require_sha(self.sha256_by_path[relative], f"closure {relative}")}
            for relative in IMPLEMENTATION_PATHS
        ]
        _require(next(row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE) == WORKORDER_SHA256,
                 "V2 workorder bytes drift")
        body = {"schema": "precision_aware_cdmd_matched_score_v2_closure_v1", "paths": rows}
        return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    base = Path(root).absolute()
    return ImplementationClosure({relative: _read_regular_no_follow(base / relative) for relative in IMPLEMENTATION_PATHS})


def validate_implementation_closure(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "V2 closure must be a mapping")
    if set(value) != {"schema", "paths", "closure_sha256"} or value.get("schema") != "precision_aware_cdmd_matched_score_v2_closure_v1":
        raise PrecisionMatchedScoreV2PlanError("V2 closure schema drift")
    rows = value.get("paths")
    _require(isinstance(rows, list) and len(rows) == len(IMPLEMENTATION_PATHS), "V2 closure row count drift")
    rebuilt: dict[str, str] = {}
    for relative, row in zip(IMPLEMENTATION_PATHS, rows, strict=True):
        _require(isinstance(row, Mapping) and set(row) == {"path", "sha256"} and row.get("path") == relative,
                 "V2 closure path/order drift")
        rebuilt[relative] = require_sha(row.get("sha256"), f"closure {relative}")
    result = ImplementationClosure(rebuilt).payload()
    _require(dict(value) == result, "V2 closure canonical payload drift")
    return result


@dataclass(frozen=True)
class ScoreIdentity(v1plan.ScoreIdentity):
    """V1-compatible science identity with successor execution provenance.

    The inherited fields preserve the exact physical V8/V5 composition.  The
    additional closure and failed-V1 binding are what authenticate the V2
    route itself; no historical closure is silently rebuilt as current V2.
    """

    successor_closure: Mapping[str, object] = field(default_factory=dict)
    v1_failed_predecessor_binding: Mapping[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        historical = super().payload()
        closure = validate_implementation_closure(self.successor_closure)
        predecessor = validate_v1_failed_predecessor_binding(self.v1_failed_predecessor_binding)
        _require(historical.get("score_spec") == v1plan.PUBLIC_SPEC.payload(), "V2 inherited science matrix drift")
        return {
            **historical,
            "schema": "precision_aware_cdmd_matched_score_identity_v2",
            "cell": CELL,
            "phase": PHASE,
            "closure": closure,
            "historical_v1_science_identity": historical,
            "v1_failed_predecessor_binding": predecessor,
            "canonical_source_environment": canonical_source_environment_payload(),
            "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
            "score_root_relative": SCORE_ROOT_RELATIVE,
            "v1_science_runtime_compatibility": True,
            "v2_environment_gate_required_before_reserve_publish_capability_attempt_and_final": True,
        }


def assert_fresh_prospective_root(root: Path, relative: str) -> None:
    _require(relative in {AUTHORITY_ROOT_RELATIVE, SCORE_ROOT_RELATIVE}, "V2 root is outside topology")
    candidate = Path(root).absolute() / relative
    try:
        info = os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise PrecisionMatchedScoreV2PlanError("V2 prospective root cannot be inspected") from error
    raise PrecisionMatchedScoreV2PlanError(
        f"V2 prospective root already exists or aliases a live entry: {candidate} ({info.st_mode:o})"
    )


def assert_fresh_prospective_roots(root: Path) -> None:
    assert_fresh_prospective_root(root, AUTHORITY_ROOT_RELATIVE)
    assert_fresh_prospective_root(root, SCORE_ROOT_RELATIVE)


def dry_plan() -> dict[str, object]:
    """Public inert contract; it neither imports Torch nor touches a root."""
    return {
        "schema": "precision_aware_cdmd_matched_score_v2_dry_plan_v1",
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "inherited_science_spec": v1plan.PUBLIC_SPEC.payload(),
        "v1_failed_predecessor": V1_FAILED_PREDECESSOR.payload(),
        "canonical_source_environment": canonical_source_environment_payload(),
        "prospective_authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "prospective_score_root_relative": SCORE_ROOT_RELATIVE,
        "public_execution": "fail_closed__requires_durable_authority_and_opaque_in_process_capability",
        "no_torch_import": True,
        "no_data_access": True,
        "no_result_write": True,
    }
