"""Static contract and closure for the M1 held-in vs held-out fold gap.

The frozen producer and the frozen held-in/split anchor receipt are immutable
historical evidence bound by literal digests.  This module deliberately
performs no I/O beyond closure leaf hashing and never rebuilds a historical
route closure as current code.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1 import plan as cswg_plan
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1


class M1GapPlanError(RuntimeError):
    """Fail closed for static gap contract, literals, or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M1GapPlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    try:
        return cswg_plan.require_sha(value, label)
    except cswg_plan.HeldInScorePlanError as error:
        raise M1GapPlanError(str(error)) from error


def safe_relative(value: object) -> str:
    try:
        return cswg_plan.safe_relative(value)
    except cswg_plan.HeldInScorePlanError as error:
        raise M1GapPlanError(str(error)) from error


CELL = cswg_plan.CELL
PHASE = "m1_heldin_heldout_gap_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M1_HELDIN_HELDOUT_GAP_V1_20260831.md"
WORKORDER_SHA256 = "604f4ca30a78f1ff97dd7481513d8f8f61afa66b624b405c48483bcd846457c6"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_heldin_heldout_gap_v1"

PRODUCER_ROOT_RELATIVE = cswg_plan.FULL_ROOT_RELATIVE
PRODUCER_TERMINAL_SHA256 = cswg_plan.FULL_TERMINAL_SHA256
PRODUCER_IDENTITY_SHA256 = cswg_plan.FULL_IDENTITY_SHA256
PRODUCER_TRAINING_SHA256 = cswg_plan.FULL_TRAINING_SHA256
PRODUCER_CHECKPOINT_MANIFEST_SHA256 = cswg_plan.CHECKPOINT_MANIFEST_SHA256
PRODUCER_BEST_CHECKPOINT_SHA256 = cswg_plan.BEST_CHECKPOINT_SHA256
PRODUCER_BEST_CHECKPOINT_STATE_SHA256 = cswg_plan.BEST_CHECKPOINT_STATE_SHA256

ANCHOR_ROOT_RELATIVE = cswg_plan.RESULT_ROOT_RELATIVE
ANCHOR_SESSION = cswg_plan.TARGET_SESSION
ANCHOR_ATTEMPT_SHA256 = "5145133c258f6ae5565a3a6d13cebacd46d9f3b3e93cf6e82341ece56736396e"
ANCHOR_LAUNCH_SHA256 = "120f75a2a63ad3a9e28508331555b0149c2b21e747a7e3129c26d1b0c109e0e5"
ANCHOR_INPUT_AUTHORITY_SHA256 = "342c78953a5e7070e3168db7c7644f9d543ee36459243bc9323fe815ce8a9a00"
ANCHOR_SCORE_SHA256 = "d5a08db493ced3c7284fc3d8b295c9e9609326658cf7deb9ef49f9826d6bb605"
ANCHOR_TERMINAL_SHA256 = "0db36a9a5ee314cf4904b80449febe393ff107e510369b0f2b176b9945caa7e5"
ANCHOR_TERMINAL_STATUS = "COMPLETE_DESCRIPTIVE_HELDIN_R2"
ANCHOR_GOVERNING_R2 = 0.5679166316986084
ANCHOR_PREDICTION_SHA256 = "781bf769021a67aaedd8839b51b90ee65203e28a8ded366612e51fb8c85f336f"
ANCHOR_TARGET_SHA256 = "e913d03a972154a4a7ad3eae9963174fb54dbeefd3bc32b542b5cb76bc0ff8aa"
ANCHOR_CALIBRATION_SHA256 = "3dfabe28ff6bfdd90f9f866fe6ed9e406a10cd66944b9cc11a54fd8831c4c3cc"
ANCHOR_ORDERED_WINDOW_START_SHA256 = "7f8693db5004525fee536698860e04f9f0cb33a3508011da88bc63f6e3bfc12c"
ANCHOR_ORDERED_QUERY_IDENTITY_SHA256 = "a89144a9d223603d243d9493fe7fe04e3dafa309b6014fab206170ecc7cc4f98"
ANCHOR_ORDERED_TARGET_EVALMASK_SHA256 = "85e8ad4acae31373e2917bb6939ae1233c6d2f47be12331480f6cc8c7413e8e0"
ANCHOR_READER_RECIPE_SHA256 = "559d86219c24190bc326349239a45a80f466332965b435a68a538a65f905d9b1"
ANCHOR_DESCRIPTOR_SHA256 = "63ee25782c62ff2275dcfbdcaa56552ec4c26fcde00f5a74e5be54785b5c25eb"
ANCHOR_DESCRIPTOR_BYTE_COUNT = 73_077_382
ANCHOR_N_WINDOWS = 54_849
ANCHOR_FORWARD_BATCHES = 429
ANCHOR_MODEL_STATE_SHA256 = PRODUCER_BEST_CHECKPOINT_STATE_SHA256

HELDOUT_FOLD_SESSIONS = ("20120924",)
HELDIN_TRAINING_SESSIONS = cswg_plan.SOURCE_SESSIONS
SCORE_ORDER = ("20120924", "20120926", "20120927", "20120928")
FOLD_TARGET_OF_SESSION = {"20120924": "fold0", "20120926": "fold1", "20120927": "fold2", "20120928": None}

BUDGET_LABEL = "M10_native"
CALIBRATION_TRIALS = 10
EVAL_BATCH_SIZE = cswg_plan.EVAL_BATCH_SIZE
METRIC_LABEL = cswg_plan.METRIC_LABEL
MODEL_SHAPE = dict(cswg_plan.MODEL_SHAPE)

GAP_TOLERANCE = 0.03
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 10_000

SUPPORTING_DIAGNOSTIC_RELATIVE = "tfpd_exploration/results/m1_h1_activity_headroom_v1/m1_fold20120924.json"
SUPPORTING_DIAGNOSTIC_SHA256 = "5ea74d3131f0b3bfc4b757ca29748285e48b4978cfdd7b07454f380ea48670d8"
SUPPORTING_DIAGNOSTIC_STATIC_R2 = 0.5707439184188843
SUPPORTING_DIAGNOSTIC_GROWING_R2 = 0.5933129191398621

OFFICIAL_HELDOUT_BLOCKED = {
    "sessions": ["20121004", "20121017", "20121024"],
    "reason": "official held-out-calib files have exactly 10 trials each; no query rows after the [0,10) support",
    "recipe_literal": "official_held_out_sessions_forbidden = 3",
    "citations": [
        "sua_exploration/docs/HELDOUT_BP_FREE_PUBLICATION_EVIDENCE_MATRIX_20260805.md",
        "sua_exploration/docs/C1_TO_NATIVE_MUA_CLAIM_BRIDGE_AUDIT_20260804.md",
    ],
    "silently_substituted": False,
}


@dataclass(frozen=True)
class AnchorLiterals:
    """Every exact value the new scorer must reproduce on session 20120924.

    The zero-argument value is the accepted frozen receipt; tests may inject
    an explicit synthetic literal set with the same topology.
    """

    attempt_sha256: str = ANCHOR_ATTEMPT_SHA256
    launch_sha256: str = ANCHOR_LAUNCH_SHA256
    input_authority_sha256: str = ANCHOR_INPUT_AUTHORITY_SHA256
    score_sha256: str = ANCHOR_SCORE_SHA256
    terminal_sha256: str = ANCHOR_TERMINAL_SHA256
    governing_r2: float = ANCHOR_GOVERNING_R2
    prediction_sha256: str = ANCHOR_PREDICTION_SHA256
    target_sha256: str = ANCHOR_TARGET_SHA256
    calibration_sha256: str = ANCHOR_CALIBRATION_SHA256
    ordered_window_start_sha256: str = ANCHOR_ORDERED_WINDOW_START_SHA256
    ordered_query_identity_sha256: str = ANCHOR_ORDERED_QUERY_IDENTITY_SHA256
    ordered_target_evalmask_sha256: str = ANCHOR_ORDERED_TARGET_EVALMASK_SHA256
    reader_recipe_sha256: str = ANCHOR_READER_RECIPE_SHA256
    target_descriptor_sha256: str = ANCHOR_DESCRIPTOR_SHA256
    target_descriptor_byte_count: int = ANCHOR_DESCRIPTOR_BYTE_COUNT
    n_windows: int = ANCHOR_N_WINDOWS
    forward_batches: int = ANCHOR_FORWARD_BATCHES
    model_state_sha256: str = ANCHOR_MODEL_STATE_SHA256

    def payload(self) -> dict[str, object]:
        return {
            "schema": "m1_heldin_heldout_gap_anchor_literals_v1",
            "root_relative": ANCHOR_ROOT_RELATIVE,
            "session_id": ANCHOR_SESSION,
            "attempt_sha256": self.attempt_sha256,
            "launch_sha256": self.launch_sha256,
            "input_authority_sha256": self.input_authority_sha256,
            "score_sha256": self.score_sha256,
            "terminal_sha256": self.terminal_sha256,
            "terminal_status": ANCHOR_TERMINAL_STATUS,
            "governing_r2": self.governing_r2,
            "prediction_sha256": self.prediction_sha256,
            "target_sha256": self.target_sha256,
            "calibration_sha256": self.calibration_sha256,
            "ordered_window_start_sha256": self.ordered_window_start_sha256,
            "ordered_query_identity_sha256": self.ordered_query_identity_sha256,
            "ordered_target_evalmask_sha256": self.ordered_target_evalmask_sha256,
            "reader_recipe_sha256": self.reader_recipe_sha256,
            "target_descriptor_sha256": self.target_descriptor_sha256,
            "target_descriptor_byte_count": self.target_descriptor_byte_count,
            "n_windows": self.n_windows,
            "forward_batches": self.forward_batches,
            "model_state_sha256": self.model_state_sha256,
        }

    def __post_init__(self) -> None:
        item = self.payload()
        _require(all(require_sha(item[name], f"anchor literal {name}") for name in (
            "attempt_sha256", "launch_sha256", "input_authority_sha256", "score_sha256",
            "terminal_sha256", "prediction_sha256", "target_sha256", "calibration_sha256",
            "ordered_window_start_sha256", "ordered_query_identity_sha256",
            "ordered_target_evalmask_sha256", "reader_recipe_sha256",
            "target_descriptor_sha256", "model_state_sha256",
        )) and type(item["n_windows"]) is int and item["n_windows"] > 0
        and type(item["forward_batches"]) is int and item["forward_batches"] > 0
        and type(item["target_descriptor_byte_count"]) is int
        and item["target_descriptor_byte_count"] > 0
        and isinstance(item["governing_r2"], float)
        and math.isfinite(item["governing_r2"]),
            "M1 gap anchor literal drift")


DEFAULT_ANCHOR = AnchorLiterals()


VERDICT_RULE = {
    "schema": "m1_heldin_heldout_gap_preregistered_verdict_rule_v1",
    "gap_definition": "heldout_fold_equal_session_mean - heldin_training_equal_session_mean",
    "tolerance": GAP_TOLERANCE,
    "branches_in_order": [
        {"index": 1,
         "condition": "abs(gap) <= 0.03 and ci_low >= -0.03 and ci_high <= 0.03",
         "verdict": "NO_HEADROOM", "direction": None},
        {"index": 2,
         "condition": "gap < -0.03 or ci_high < -0.03",
         "verdict": "HEADROOM_PRESENT", "direction": "HELD_OUT_WORSE"},
        {"index": 3,
         "condition": "gap > 0.03 or ci_low > 0.03",
         "verdict": "HEADROOM_PRESENT", "direction": "HELD_OUT_BETTER"},
        {"index": 4,
         "condition": "otherwise",
         "verdict": "INDETERMINATE", "direction": None},
    ],
    "aggregation": {"mean": "unweighted per-session mean within arm", "sd": "ddof=0"},
    "bootstrap": {
        "seed": BOOTSTRAP_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "generator": "numpy.random.default_rng(42)",
        "resampling": "sessions with replacement independently within each arm; held-in arm first",
        "ci": "percentile 2.5/97.5",
        "heldout_arm_session_count": len(HELDOUT_FOLD_SESSIONS),
        "degenerate_heldout_arm": len(HELDOUT_FOLD_SESSIONS) == 1,
        "limitation": ("single fold-defined held-out session for this producer; the CI "
                       "quantifies training-session sampling variability only"),
    },
    "decided_before_any_scoring": True,
}


@dataclass(frozen=True)
class GapSpec:
    """The sole paired surface; no training choice remains."""

    root_relative: str = RESULT_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(safe_relative(self.root_relative) == self.root_relative
                 and self.root_relative == RESULT_ROOT_RELATIVE,
                 "M1 gap score fixed root drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "m1_heldin_heldout_gap_spec_v1",
            "cell": CELL,
            "phase": PHASE,
            "root_relative": self.root_relative,
            "budget": BUDGET_LABEL,
            "calibration_trials": CALIBRATION_TRIALS,
            "heldout_fold_sessions": list(HELDOUT_FOLD_SESSIONS),
            "heldin_training_sessions": list(HELDIN_TRAINING_SESSIONS),
            "score_order": list(SCORE_ORDER),
            "fold_target_of_session": dict(FOLD_TARGET_OF_SESSION),
            "selected_checkpoint_role": "best_source_train_loss",
            "eval_batch_size": EVAL_BATCH_SIZE,
            "metric": METRIC_LABEL,
            "last_bin_only": True,
            "eval_no_grad_dropout_off": True,
            "target_metric_only": True,
            "target_optimizer_backward_update": 0,
            "formal_benchmark_verdict": False,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class GapIdentity:
    spec: GapSpec
    closure: Mapping[str, object]
    anchor: AnchorLiterals = DEFAULT_ANCHOR

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, GapSpec) and isinstance(self.anchor, AnchorLiterals)
                 and isinstance(self.closure, Mapping)
                 and require_sha(self.closure.get("closure_sha256"), "gap identity closure"),
                 "M1 gap identity topology drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "m1_heldin_heldout_gap_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "closure": dict(self.closure),
            "producer": {
                "schema": "m1_heldin_heldout_gap_producer_binding_v1",
                "completed_full_expectation": cswg_plan.DEFAULT_COMPLETED_FULL_EXPECTATION.payload(),
                "selected_checkpoint_role": "best_source_train_loss",
            },
            "anchor": self.anchor.payload(),
            "preregistered_verdict_rule": VERDICT_RULE,
            "official_heldout_surface_blocked": dict(OFFICIAL_HELDOUT_BLOCKED),
            "sealed_m1_metadata_manifest": v1.m1_metadata_manifest_binding_payload(),
            "supporting_diagnostic": {
                "relative_path": SUPPORTING_DIAGNOSTIC_RELATIVE,
                "sha256": SUPPORTING_DIAGNOSTIC_SHA256,
                "static_support_r2": SUPPORTING_DIAGNOSTIC_STATIC_R2,
                "causal_growing_cap30_r2": SUPPORTING_DIAGNOSTIC_GROWING_R2,
                "read_only_quote": True,
                "part_of_gap_or_verdict": False,
            },
            "metric_only_target_access": True,
            "target_optimizer_backward_update": 0,
            "no_amp_tf32_compile_enablement": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


_RUNTIME_PATHS: tuple[str, ...] = (
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py",
)
_OWNED_PATHS: tuple[str, ...] = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/m1_heldin_heldout_gap_v1/__init__.py",
    "tfpd_exploration/src/m1_heldin_heldout_gap_v1/plan.py",
    "tfpd_exploration/src/m1_heldin_heldout_gap_v1/score.py",
    "tfpd_exploration/src/m1_heldin_heldout_gap_v1/physical.py",
    "tfpd_exploration/scripts/run_m1_heldin_heldout_gap_v1.py",
    "tfpd_exploration/tests/test_m1_heldin_heldout_gap_v1.py",
)


def _closure_leaf_sha(root: Path, relative: str) -> str:
    try:
        return v1._read_regular_no_follow(Path(root), relative)
    except v1.SourceLifecycleError as error:
        raise M1GapPlanError(f"M1 gap closure leaf drift: {relative}") from error


def implementation_closure(root: Path) -> dict[str, object]:
    """Explicit successor closure over the inherited CS-WG scorer closure."""
    try:
        inherited = cswg_plan.implementation_closure(Path(root))
    except cswg_plan.HeldInScorePlanError as error:
        raise M1GapPlanError("M1 gap inherited CS-WG scorer closure drift") from error
    rows = [dict(row) for row in inherited.get("paths", [])]  # type: ignore[union-attr]
    _require(isinstance(rows, list) and rows, "M1 gap inherited closure topology drift")
    seen = {row.get("path") for row in rows}
    for relative in (*_RUNTIME_PATHS, *_OWNED_PATHS):
        if relative in seen:
            continue
        rows.append({"path": relative, "sha256": _closure_leaf_sha(Path(root), relative)})
        seen.add(relative)
    workorder = next((row for row in rows if row.get("path") == WORKORDER_RELATIVE), None)
    _require(isinstance(workorder, Mapping) and workorder.get("sha256") == WORKORDER_SHA256,
             "M1 gap workorder literal/body drift")
    body = {
        "schema": "m1_heldin_heldout_gap_closure_v1",
        "current_cswg_heldin_score_closure_sha256": inherited.get("closure_sha256"),
        "producer_root_relative": PRODUCER_ROOT_RELATIVE,
        "anchor_root_relative": ANCHOR_ROOT_RELATIVE,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "M1 gap closure must be a mapping")
    rebuilt = implementation_closure(Path(root))
    _require(dict(value) == rebuilt, "M1 gap closure/current-byte drift")
    return rebuilt


def build_identity(root: Path) -> GapIdentity:
    try:
        v1.load_m1_metadata_manifest_authority(Path(root))
    except v1.SourceLifecycleError as error:
        raise M1GapPlanError("M1 gap sealed target metadata authority drift") from error
    return GapIdentity(GapSpec(), implementation_closure(Path(root)))


def dry_plan(root: Path | None = None) -> dict[str, object]:
    del root
    spec = GapSpec()
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_score_root_relative": spec.root_relative,
        "producer_root_relative": PRODUCER_ROOT_RELATIVE,
        "anchor_root_relative": ANCHOR_ROOT_RELATIVE,
        "score_spec": spec.payload(),
        "anchor": DEFAULT_ANCHOR.payload(),
        "preregistered_verdict_rule": VERDICT_RULE,
        "official_heldout_surface_blocked": dict(OFFICIAL_HELDOUT_BLOCKED),
        "public_execution_authorized": False,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
    }


__all__ = (
    "M1GapPlanError", "CELL", "PHASE", "WORKORDER_RELATIVE", "WORKORDER_SHA256",
    "RESULT_ROOT_RELATIVE", "PRODUCER_ROOT_RELATIVE", "PRODUCER_TERMINAL_SHA256",
    "PRODUCER_IDENTITY_SHA256", "PRODUCER_TRAINING_SHA256", "PRODUCER_CHECKPOINT_MANIFEST_SHA256",
    "PRODUCER_BEST_CHECKPOINT_SHA256", "PRODUCER_BEST_CHECKPOINT_STATE_SHA256",
    "ANCHOR_ROOT_RELATIVE", "ANCHOR_SESSION", "HELDOUT_FOLD_SESSIONS", "HELDIN_TRAINING_SESSIONS",
    "SCORE_ORDER", "FOLD_TARGET_OF_SESSION", "BUDGET_LABEL", "CALIBRATION_TRIALS",
    "EVAL_BATCH_SIZE", "METRIC_LABEL", "MODEL_SHAPE", "GAP_TOLERANCE", "BOOTSTRAP_SEED",
    "BOOTSTRAP_DRAWS", "VERDICT_RULE", "OFFICIAL_HELDOUT_BLOCKED", "AnchorLiterals",
    "DEFAULT_ANCHOR", "GapSpec", "GapIdentity", "canonical_json_bytes", "sha256_bytes",
    "require_sha", "safe_relative", "implementation_closure", "validate_current_closure",
    "build_identity", "dry_plan",
)
