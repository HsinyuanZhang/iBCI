"""Typed no-data contract for the matched-ERM held-in scorer successor.

The completed producer does not exist at candidate construction time.  Its
body and state digests are therefore mandatory constructor inputs, never
placeholder literals or values inferred from a mutable result directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import plan as cswg_plan
from tfpd_exploration.src.cross_session_worst_group_matched_erm_full_v1 import full_train as matched_full
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1


class MatchedERMHeldInScorePlanError(RuntimeError):
    """Fail closed for successor literals, closure, or input-anchor drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedERMHeldInScorePlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    try:
        return cswg_plan.require_sha(value, label)
    except cswg_plan.HeldInScorePlanError as error:
        raise MatchedERMHeldInScorePlanError(str(error)) from error


def safe_relative(value: object) -> str:
    try:
        return cswg_plan.safe_relative(value)
    except cswg_plan.HeldInScorePlanError as error:
        raise MatchedERMHeldInScorePlanError(str(error)) from error


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "fold20120924_matched_erm_heldin_metric_only_score_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_FOLD20120924_MATCHED_ERM_HELDIN_SCORE_V1_20260827.md"
WORKORDER_SHA256 = "4b9da84fb4f5a9cf08d0bf1cb019f00ef45fbdb28f233fdcf2c02d25d9f70e2e"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_v1"
MATCHED_ERM_FULL_ROOT_RELATIVE = matched_full.MATCHED_ERM_FULL_ROOT_RELATIVE
CSWG_COMPARATOR_ROOT_RELATIVE = cswg_plan.RESULT_ROOT_RELATIVE

TARGET_SESSION = "20120924"
SOURCE_SESSIONS = ("20120926", "20120927", "20120928")
EVAL_BATCH_SIZE = 128
EXPECTED_N_WINDOWS = 54_849
METRIC_LABEL = "torchmetrics.regression.R2Score(multioutput='variance_weighted')"
MODEL_SHAPE = {
    "window": 100,
    "units": 64,
    "outputs": 16,
    "calibration_shape_per_row": [10, 1024, 64],
    "live_parameters_after_lazy_materialization": 15_007_496,
}

FULL_BODY_NAMES = cswg_plan.FULL_BODY_NAMES


@dataclass(frozen=True)
class SameInputAnchor:
    """Exact underlying target-evaluation facts from the accepted CS-WG row."""

    comparator_input_authority_sha256: str = "342c78953a5e7070e3168db7c7644f9d543ee36459243bc9323fe815ce8a9a00"
    comparator_score_sha256: str = "d5a08db493ced3c7284fc3d8b295c9e9609326658cf7deb9ef49f9826d6bb605"
    n_windows: int = EXPECTED_N_WINDOWS
    target_descriptor_sha256: str = "63ee25782c62ff2275dcfbdcaa56552ec4c26fcde00f5a74e5be54785b5c25eb"
    target_descriptor_byte_count: int = 73_077_382
    calibration_sha256: str = "3dfabe28ff6bfdd90f9f866fe6ed9e406a10cd66944b9cc11a54fd8831c4c3cc"
    ordered_window_start_sha256: str = "7f8693db5004525fee536698860e04f9f0cb33a3508011da88bc63f6e3bfc12c"
    ordered_query_identity_sha256: str = "a89144a9d223603d243d9493fe7fe04e3dafa309b6014fab206170ecc7cc4f98"
    ordered_target_evalmask_sha256: str = "85e8ad4acae31373e2917bb6939ae1233c6d2f47be12331480f6cc8c7413e8e0"
    target_sha256: str = "e913d03a972154a4a7ad3eae9963174fb54dbeefd3bc32b542b5cb76bc0ff8aa"
    reader_recipe_sha256: str = "559d86219c24190bc326349239a45a80f466332965b435a68a538a65f905d9b1"
    comparator_root_relative: str = CSWG_COMPARATOR_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(type(self.n_windows) is int and self.n_windows == EXPECTED_N_WINDOWS
                 and type(self.target_descriptor_byte_count) is int and self.target_descriptor_byte_count > 0
                 and safe_relative(self.comparator_root_relative) == self.comparator_root_relative
                 and self.comparator_root_relative == CSWG_COMPARATOR_ROOT_RELATIVE
                 and all(require_sha(value, "same-input anchor") for value in (
                     self.comparator_input_authority_sha256, self.comparator_score_sha256,
                     self.target_descriptor_sha256, self.calibration_sha256,
                     self.ordered_window_start_sha256, self.ordered_query_identity_sha256,
                     self.ordered_target_evalmask_sha256, self.target_sha256,
                     self.reader_recipe_sha256,
                 )), "CS-WG matched-ERM held-in score same-input anchor drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_same_input_anchor_v1",
            "comparator_system": "CS_WG",
            "comparator_root_relative": self.comparator_root_relative,
            "comparator_input_authority_sha256": self.comparator_input_authority_sha256,
            "comparator_score_sha256": self.comparator_score_sha256,
            "target_session": TARGET_SESSION,
            "n_windows": self.n_windows,
            "target_descriptor": {
                "sha256": self.target_descriptor_sha256,
                "byte_count": self.target_descriptor_byte_count,
            },
            "calibration_sha256": self.calibration_sha256,
            "calibration_shape_per_row": list(MODEL_SHAPE["calibration_shape_per_row"]),
            "ordered_window_start_sha256": self.ordered_window_start_sha256,
            "ordered_query_identity_sha256": self.ordered_query_identity_sha256,
            "ordered_target_evalmask_sha256": self.ordered_target_evalmask_sha256,
            "target_sha256": self.target_sha256,
            "reader_recipe_sha256": self.reader_recipe_sha256,
            "whole_input_authority_sha_comparison_forbidden": True,
        }


DEFAULT_SAME_INPUT_ANCHOR = SameInputAnchor()


@dataclass(frozen=True)
class MatchedERMFullBinding:
    """All future terminal facts required before the scorer can be reviewed."""

    attempt_sha256: str
    launch_sha256: str
    source_authority_sha256: str
    training_sha256: str
    identity_sha256: str
    checkpoint_manifest_sha256: str
    best_checkpoint_sha256: str
    best_checkpoint_state_sha256: str
    last_checkpoint_sha256: str
    last_checkpoint_state_sha256: str
    terminal_sha256: str
    best_epoch_index: int
    last_epoch_index: int
    root_relative: str = MATCHED_ERM_FULL_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(safe_relative(self.root_relative) == self.root_relative
                 and self.root_relative == MATCHED_ERM_FULL_ROOT_RELATIVE
                 and all(require_sha(value, "matched-ERM full binding") for value in (
                     self.attempt_sha256, self.launch_sha256, self.source_authority_sha256,
                     self.training_sha256, self.identity_sha256, self.checkpoint_manifest_sha256,
                     self.best_checkpoint_sha256, self.best_checkpoint_state_sha256,
                     self.last_checkpoint_sha256, self.last_checkpoint_state_sha256, self.terminal_sha256,
                 ))
                 and type(self.best_epoch_index) is int and 0 <= self.best_epoch_index < 20
                 and type(self.last_epoch_index) is int and 0 <= self.last_epoch_index < 20,
                 "CS-WG matched-ERM held-in score producer binding is incomplete/drifted")

    def completed_full_expectation(self) -> cswg_plan.CompletedFullExpectation:
        return cswg_plan.CompletedFullExpectation(
            root_relative=self.root_relative,
            terminal_sha256=self.terminal_sha256,
            training_sha256=self.training_sha256,
            identity_sha256=self.identity_sha256,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            best_checkpoint_sha256=self.best_checkpoint_sha256,
            best_checkpoint_state_sha256=self.best_checkpoint_state_sha256,
            last_checkpoint_sha256=self.last_checkpoint_sha256,
            last_checkpoint_state_sha256=self.last_checkpoint_state_sha256,
            best_epoch_index=self.best_epoch_index,
            last_epoch_index=self.last_epoch_index,
            expected_system="MATCHED_ERM",
            expected_objective_lambda=0.0,
            expected_objective_tau=0.01,
            extra_body_sha256={
                "attempt.json": self.attempt_sha256,
                "launch.json": self.launch_sha256,
                "source_authority.json": self.source_authority_sha256,
            },
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_matched_erm_completed_full_binding_v1",
            "system": "MATCHED_ERM",
            "objective_lambda": 0.0,
            "objective_tau": 0.01,
            "completed_full_expectation": self.completed_full_expectation().payload(),
            "attempt_sha256": self.attempt_sha256,
            "launch_sha256": self.launch_sha256,
            "source_authority_sha256": self.source_authority_sha256,
            "training_sha256": self.training_sha256,
            "identity_sha256": self.identity_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "selected_checkpoint": {
                "role": "best_source_train_loss",
                "filename": "checkpoint_best_source_train_loss.pt",
                "body_sha256": self.best_checkpoint_sha256,
                "state_sha256": self.best_checkpoint_state_sha256,
                "best_epoch_index": self.best_epoch_index,
            },
            "last_checkpoint": {
                "filename": "checkpoint_last.pt",
                "body_sha256": self.last_checkpoint_sha256,
                "state_sha256": self.last_checkpoint_state_sha256,
                "last_epoch_index": self.last_epoch_index,
            },
            "terminal_sha256": self.terminal_sha256,
            "exact_body_pair_count": len(FULL_BODY_NAMES),
            "exact_leaf_count": len(FULL_BODY_NAMES) * 2,
            "leaf_mode": "0444",
            "leaf_nlink": 1,
            "canonical_basename_sidecars": True,
            "failure_leaf_forbidden": True,
            "no_swa": True,
        }


@dataclass(frozen=True)
class MatchedERMScoreSpec:
    root_relative: str = RESULT_ROOT_RELATIVE
    target_session: str = TARGET_SESSION
    selected_checkpoint_role: str = "best_source_train_loss"
    eval_batch_size: int = EVAL_BATCH_SIZE

    def __post_init__(self) -> None:
        _require(safe_relative(self.root_relative) == self.root_relative
                 and self.root_relative == RESULT_ROOT_RELATIVE
                 and self.target_session == TARGET_SESSION
                 and self.selected_checkpoint_role == "best_source_train_loss"
                 and self.eval_batch_size == EVAL_BATCH_SIZE,
                 "CS-WG matched-ERM held-in score fixed spec drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_spec_v1",
            "cell": CELL,
            "phase": PHASE,
            "root_relative": self.root_relative,
            "target_session": self.target_session,
            "source_sessions": list(SOURCE_SESSIONS),
            "producer_system": "MATCHED_ERM",
            "producer_objective_lambda": 0.0,
            "producer_objective_tau": 0.01,
            "selected_checkpoint_role": self.selected_checkpoint_role,
            "eval_batch_size": self.eval_batch_size,
            "metric": METRIC_LABEL,
            "last_bin_only": True,
            "eval_no_grad_dropout_off": True,
            "target_metric_only": True,
            "target_optimizer_backward_update": 0,
            "same_input_anchor_required": True,
            "formal_benchmark_verdict": False,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


_RUNTIME_PATHS = (
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/plan.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/score.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/physical.py",
    "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_matched_erm_full_v1/full_train.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py",
)
_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_matched_erm_score_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_matched_erm_score_v1/plan.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_matched_erm_score_v1/score.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_matched_erm_score_v1/physical.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_fold20120924_matched_erm_heldin_score_v1.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_fold20120924_matched_erm_heldin_score_v1.py",
)


def _closure_leaf_sha(root: Path, relative: str) -> str:
    try:
        return v1._read_regular_no_follow(Path(root), relative)
    except BaseException as error:
        raise MatchedERMHeldInScorePlanError(
            f"CS-WG matched-ERM held-in score closure leaf drift: {relative}"
        ) from error


def implementation_closure(root: Path) -> dict[str, object]:
    """Explicit current-byte closure; historical result roots are not read."""
    inherited = cswg_plan.implementation_closure(Path(root))
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list) and inherited_rows,
             "CS-WG matched-ERM held-in score inherited scorer closure drift")
    rows = [dict(row) for row in inherited_rows]
    seen = {row.get("path") for row in rows}
    for relative in (*_RUNTIME_PATHS, *_OWNED_PATHS):
        if relative not in seen:
            rows.append({"path": relative, "sha256": _closure_leaf_sha(Path(root), relative)})
            seen.add(relative)
    workorder = next((row for row in rows if row.get("path") == WORKORDER_RELATIVE), None)
    _require(isinstance(workorder, Mapping) and workorder.get("sha256") == WORKORDER_SHA256,
             "CS-WG matched-ERM held-in score workorder literal/body drift")
    body = {
        "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_closure_v1",
        "current_cswg_heldin_score_closure_sha256": inherited.get("closure_sha256"),
        "current_matched_erm_full_root_relative": MATCHED_ERM_FULL_ROOT_RELATIVE,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG matched-ERM held-in score closure must be a mapping")
    rebuilt = implementation_closure(Path(root))
    _require(dict(value) == rebuilt, "CS-WG matched-ERM held-in score closure/current-byte drift")
    return rebuilt


@dataclass(frozen=True)
class MatchedERMScoreIdentity:
    spec: MatchedERMScoreSpec
    closure: Mapping[str, object]
    producer_binding: MatchedERMFullBinding
    same_input_anchor: SameInputAnchor = DEFAULT_SAME_INPUT_ANCHOR

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, MatchedERMScoreSpec)
                 and isinstance(self.producer_binding, MatchedERMFullBinding)
                 and isinstance(self.same_input_anchor, SameInputAnchor)
                 and isinstance(self.closure, Mapping)
                 and require_sha(self.closure.get("closure_sha256"), "identity closure"),
                 "CS-WG matched-ERM held-in score identity topology drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "closure": dict(self.closure),
            "matched_erm_full_binding": self.producer_binding.payload(),
            "same_input_anchor": self.same_input_anchor.payload(),
            "sealed_m1_metadata_manifest": v1.m1_metadata_manifest_binding_payload(),
            "target_metric_only": True,
            "target_optimizer_backward_update": 0,
            "no_amp_tf32_compile_enablement": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def build_identity(root: Path, *, producer_binding: MatchedERMFullBinding) -> MatchedERMScoreIdentity:
    """Build only with an explicit post-terminal producer binding.

    This function intentionally performs no result-root, target, checkpoint,
    Torch, or CUDA I/O.  Future capability issuance revalidates the sealed
    metadata authority and held producer graph before reservation.
    """
    _require(isinstance(producer_binding, MatchedERMFullBinding),
             "CS-WG matched-ERM held-in score producer binding is required")
    return MatchedERMScoreIdentity(MatchedERMScoreSpec(), implementation_closure(Path(root)), producer_binding)


def dry_plan(root: Path | None = None) -> dict[str, object]:
    del root
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_relative": WORKORDER_RELATIVE,
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_score_root_relative": RESULT_ROOT_RELATIVE,
        "future_matched_erm_root_relative": MATCHED_ERM_FULL_ROOT_RELATIVE,
        "producer_binding_constructor_required": [
            "attempt_sha256", "launch_sha256", "source_authority_sha256", "training_sha256",
            "identity_sha256", "checkpoint_manifest_sha256", "best_checkpoint_sha256",
            "best_checkpoint_state_sha256", "last_checkpoint_sha256", "last_checkpoint_state_sha256",
            "terminal_sha256", "best_epoch_index", "last_epoch_index",
        ],
        "same_input_anchor": DEFAULT_SAME_INPUT_ANCHOR.payload(),
        "score_spec": MatchedERMScoreSpec().payload(),
        "opens_future_result_or_nwb": False,
        "opens_checkpoint_tensor": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


__all__ = (
    "CELL", "PHASE", "WORKORDER_RELATIVE", "WORKORDER_SHA256", "RESULT_ROOT_RELATIVE", "MATCHED_ERM_FULL_ROOT_RELATIVE", "CSWG_COMPARATOR_ROOT_RELATIVE",
    "TARGET_SESSION", "SOURCE_SESSIONS", "EVAL_BATCH_SIZE", "EXPECTED_N_WINDOWS", "METRIC_LABEL",
    "MODEL_SHAPE", "FULL_BODY_NAMES", "MatchedERMHeldInScorePlanError", "SameInputAnchor",
    "DEFAULT_SAME_INPUT_ANCHOR", "MatchedERMFullBinding", "MatchedERMScoreSpec", "MatchedERMScoreIdentity",
    "canonical_json_bytes", "sha256_bytes", "require_sha", "safe_relative", "implementation_closure",
    "validate_current_closure", "build_identity", "dry_plan",
)
