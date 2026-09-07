"""Static contract and explicit current-byte closure for the held-in score.

The completed full-training graph is immutable historical evidence.  Its
terminal digest is fixed below; this module deliberately does not rebuild the
old full-route closure in order to treat that historical result as current
code.  The scorer's own closure separately binds every local module it will
execute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_v6


class HeldInScorePlanError(RuntimeError):
    """Fail closed for static held-in-score contract or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldInScorePlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG held-in score {label} must be a lowercase SHA-256")
    return value


def safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG held-in score relative path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG held-in score relative path is unsafe")
    return path.as_posix()


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "fold20120924_heldin_metric_only_score_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_FOLD20120924_HELDIN_SCORE_V1_20260827.md"
WORKORDER_SHA256 = "1ff3faab95f58aa66730d4793b524be6b5370b54863300c700aa734251e44213"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_heldin_score_v1"

FULL_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa/fold_20120924_cswg"
FULL_TERMINAL_SHA256 = "efd084573843e05ada6c28c050c28e8bbf01976bec443d864bc3a0a7921afdc5"
FULL_TRAINING_SHA256 = "0250b689de465b8777485883e9646c5fc64bfa9d5ea710b8670afffece5406b1"
FULL_IDENTITY_SHA256 = "c6b23446bca124b26b47c01485de19354be4d47a42a2bc5e59e43ee0b34b56d3"
CHECKPOINT_MANIFEST_SHA256 = "e190f6c813d5f9403e7d4f505cebbeb4830edb63f1ae2a11d15bdc9907a40550"
BEST_CHECKPOINT_SHA256 = "2cefa5cbeec5653a4fed76cacfb61113ae47bf4546879cf6cc9f6f427890d9e9"
BEST_CHECKPOINT_STATE_SHA256 = "d5d86325e21b5a257a44ee2eff4a9e35ddba9d1d6b0de591e607538bbbb41db7"

TARGET_SESSION = "20120924"
SOURCE_SESSIONS = ("20120926", "20120927", "20120928")
EVAL_BATCH_SIZE = 128
METRIC_LABEL = "torchmetrics.regression.R2Score(multioutput='variance_weighted')"
MODEL_SHAPE = {
    "window": 100,
    "units": 64,
    "outputs": 16,
    "calibration_shape_per_row": [10, 1024, 64],
    "live_parameters_after_lazy_materialization": 15_007_496,
}

FULL_BODY_NAMES = (
    "attempt.json", "launch.json", "source_authority.json",
    *(f"epoch_{index:02d}.json" for index in range(20)),
    "training.json", "checkpoint_best_source_train_loss.pt", "checkpoint_last.pt",
    "checkpoint_manifest.json", "terminal.json",
)

_RUNTIME_PATHS = (
    "tfpd_exploration/src/cross_session_worst_group_full_v1/full_train.py",
    "tfpd_exploration/src/cross_session_worst_group_full_v1/physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py",
)
_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/plan.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/score.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/physical.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_fold20120924_heldin_score_v1.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_fold20120924_heldin_score_v1.py",
)


@dataclass(frozen=True)
class ScoreSpec:
    """The sole metric-only target surface; no source-training choice remains."""

    root_relative: str = RESULT_ROOT_RELATIVE
    target_session: str = TARGET_SESSION
    selected_checkpoint_role: str = "best_source_train_loss"
    eval_batch_size: int = EVAL_BATCH_SIZE

    def __post_init__(self) -> None:
        _require(safe_relative(self.root_relative) == self.root_relative
                 and self.target_session == TARGET_SESSION
                 and self.selected_checkpoint_role == "best_source_train_loss"
                 and self.eval_batch_size == EVAL_BATCH_SIZE,
                 "CS-WG held-in score fixed surface/spec drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_spec_v1",
            "cell": CELL,
            "phase": PHASE,
            "root_relative": self.root_relative,
            "target_session": self.target_session,
            "source_sessions": list(SOURCE_SESSIONS),
            "selected_checkpoint_role": self.selected_checkpoint_role,
            "eval_batch_size": self.eval_batch_size,
            "metric": METRIC_LABEL,
            "last_bin_only": True,
            "eval_no_grad_dropout_off": True,
            "target_metric_only": True,
            "target_optimizer_backward_update": 0,
            "matched_erm_reference_bound": False,
            "formal_benchmark_verdict": False,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


@dataclass(frozen=True)
class CompletedFullExpectation:
    """Literal provenance of a terminal-pinned no-SWA M1 full producer.

    The zero-argument value is the historical accepted CS-WG producer and
    deliberately keeps its original serialized payload.  The optional fields
    are a narrow successor seam for a separately terminal-pinned producer
    with the same 20-epoch/no-SWA artifact topology.  They are intentionally
    explicit rather than inferred from a mutable result directory.
    """

    root_relative: str = FULL_ROOT_RELATIVE
    terminal_sha256: str = FULL_TERMINAL_SHA256
    training_sha256: str = FULL_TRAINING_SHA256
    identity_sha256: str = FULL_IDENTITY_SHA256
    checkpoint_manifest_sha256: str = CHECKPOINT_MANIFEST_SHA256
    best_checkpoint_sha256: str = BEST_CHECKPOINT_SHA256
    best_checkpoint_state_sha256: str = BEST_CHECKPOINT_STATE_SHA256
    last_checkpoint_sha256: str | None = None
    last_checkpoint_state_sha256: str | None = None
    best_epoch_index: int = 19
    last_epoch_index: int = 19
    expected_system: str = "CS_WG"
    expected_objective_lambda: float | None = None
    expected_objective_tau: float | None = None
    extra_body_sha256: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        extras = dict(self.extra_body_sha256)
        optional_sha = tuple(value for value in (
            self.last_checkpoint_sha256, self.last_checkpoint_state_sha256,
        ) if value is not None)
        _require(safe_relative(self.root_relative) == self.root_relative
                 and all(require_sha(value, "completed full expectation") for value in (
                     self.terminal_sha256, self.training_sha256, self.identity_sha256,
                     self.checkpoint_manifest_sha256, self.best_checkpoint_sha256,
                     self.best_checkpoint_state_sha256, *optional_sha,
                 ))
                 and (self.last_checkpoint_sha256 is None) == (self.last_checkpoint_state_sha256 is None)
                 and type(self.best_epoch_index) is int and 0 <= self.best_epoch_index < 20
                 and type(self.last_epoch_index) is int and 0 <= self.last_epoch_index < 20
                 and self.expected_system in {"CS_WG", "MATCHED_ERM"}
                 and (self.expected_objective_lambda is None
                      or (type(self.expected_objective_lambda) in {int, float}
                          and self.expected_objective_lambda >= 0.0))
                 and (self.expected_objective_tau is None
                      or (type(self.expected_objective_tau) in {int, float}
                          and self.expected_objective_tau > 0.0))
                 and tuple(sorted(extras)) == tuple(sorted(set(extras)))
                 and set(extras).issubset({"attempt.json", "launch.json", "source_authority.json"})
                 and all(require_sha(value, f"completed full extra body {name}")
                         for name, value in extras.items()),
                 "CS-WG held-in score full expectation drift")
        object.__setattr__(self, "extra_body_sha256", MappingProxyType(extras))

    @property
    def effective_last_checkpoint_sha256(self) -> str:
        return self.best_checkpoint_sha256 if self.last_checkpoint_sha256 is None else self.last_checkpoint_sha256

    @property
    def effective_last_checkpoint_state_sha256(self) -> str:
        return self.best_checkpoint_state_sha256 if self.last_checkpoint_state_sha256 is None else self.last_checkpoint_state_sha256

    @property
    def is_historical_default(self) -> bool:
        """Whether serializing this expectation must retain historical bytes."""
        return (
            self.root_relative == FULL_ROOT_RELATIVE
            and self.terminal_sha256 == FULL_TERMINAL_SHA256
            and self.training_sha256 == FULL_TRAINING_SHA256
            and self.identity_sha256 == FULL_IDENTITY_SHA256
            and self.checkpoint_manifest_sha256 == CHECKPOINT_MANIFEST_SHA256
            and self.best_checkpoint_sha256 == BEST_CHECKPOINT_SHA256
            and self.best_checkpoint_state_sha256 == BEST_CHECKPOINT_STATE_SHA256
            and self.last_checkpoint_sha256 is None
            and self.last_checkpoint_state_sha256 is None
            and self.best_epoch_index == 19 and self.last_epoch_index == 19
            and self.expected_system == "CS_WG"
            and self.expected_objective_lambda is None
            and self.expected_objective_tau is None
            and not self.extra_body_sha256
        )

    def payload(self) -> dict[str, object]:
        historical_payload = {
            "schema": "cross_session_worst_group_m1_completed_full_expectation_v1",
            "root_relative": self.root_relative,
            "terminal_sha256": self.terminal_sha256,
            "training_sha256": self.training_sha256,
            "identity_sha256": self.identity_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "selected_checkpoint": {
                "role": "best_source_train_loss",
                "filename": "checkpoint_best_source_train_loss.pt",
                "body_sha256": self.best_checkpoint_sha256,
                "state_sha256": self.best_checkpoint_state_sha256,
            },
            "last_checkpoint_must_equal_selected_best": True,
            "best_epoch_index": 19,
            "last_epoch_index": 19,
            "exact_body_pair_count": len(FULL_BODY_NAMES),
            "exact_leaf_count": len(FULL_BODY_NAMES) * 2,
            "leaf_mode": "0444",
            "leaf_nlink": 1,
            "canonical_basename_sidecars": True,
            "failure_leaf_forbidden": True,
            "no_swa": True,
        }
        if self.is_historical_default:
            return historical_payload
        return {
            **historical_payload,
            "last_checkpoint_must_equal_selected_best": (
                self.effective_last_checkpoint_sha256 == self.best_checkpoint_sha256
                and self.effective_last_checkpoint_state_sha256 == self.best_checkpoint_state_sha256
            ),
            "last_checkpoint": {
                "filename": "checkpoint_last.pt",
                "body_sha256": self.effective_last_checkpoint_sha256,
                "state_sha256": self.effective_last_checkpoint_state_sha256,
            },
            "best_epoch_index": self.best_epoch_index,
            "last_epoch_index": self.last_epoch_index,
            "expected_system": self.expected_system,
            "expected_objective_lambda": self.expected_objective_lambda,
            "expected_objective_tau": self.expected_objective_tau,
            "extra_body_sha256": dict(self.extra_body_sha256),
        }


DEFAULT_COMPLETED_FULL_EXPECTATION = CompletedFullExpectation()


def _closure_leaf_sha(root: Path, relative: str) -> str:
    try:
        return source_smoke_v6._read_regular_no_follow(Path(root), relative)
    except BaseException as error:
        raise HeldInScorePlanError(f"CS-WG held-in score closure leaf drift: {relative}") from error


def implementation_closure(root: Path) -> dict[str, object]:
    """Build an explicit successor closure without globbing historical code."""
    try:
        inherited = v1.implementation_closure(Path(root))
    except v1.SourceLifecycleError as error:
        raise HeldInScorePlanError("CS-WG held-in score inherited lifecycle closure drift") from error
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list) and inherited_rows,
             "CS-WG held-in score inherited closure topology drift")
    rows = [dict(row) for row in inherited_rows]
    seen = {row.get("path") for row in rows}
    for relative in (*_RUNTIME_PATHS, *_OWNED_PATHS):
        if relative in seen:
            continue
        rows.append({"path": relative, "sha256": _closure_leaf_sha(Path(root), relative)})
        seen.add(relative)
    workorder = next((row for row in rows if row.get("path") == WORKORDER_RELATIVE), None)
    _require(isinstance(workorder, Mapping) and workorder.get("sha256") == WORKORDER_SHA256,
             "CS-WG held-in score workorder literal/body drift")
    body = {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_closure_v1",
        "current_source_lifecycle_closure_sha256": inherited.get("closure_sha256"),
        "completed_full_terminal_sha256": FULL_TERMINAL_SHA256,
        "completed_full_identity_sha256": FULL_IDENTITY_SHA256,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG held-in score closure must be a mapping")
    rebuilt = implementation_closure(Path(root))
    _require(dict(value) == rebuilt, "CS-WG held-in score closure/current-byte drift")
    return rebuilt


@dataclass(frozen=True)
class ScoreIdentity:
    """Current scorer closure plus immutable selected producer facts."""

    spec: ScoreSpec
    closure: Mapping[str, object]
    completed_full: CompletedFullExpectation = DEFAULT_COMPLETED_FULL_EXPECTATION

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, ScoreSpec)
                 and isinstance(self.completed_full, CompletedFullExpectation)
                 and isinstance(self.closure, Mapping)
                 and require_sha(self.closure.get("closure_sha256"), "identity closure"),
                 "CS-WG held-in score identity topology drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "closure": dict(self.closure),
            "completed_full_expectation": self.completed_full.payload(),
            "sealed_m1_metadata_manifest": v1.m1_metadata_manifest_binding_payload(),
            "metric_only_target_access": True,
            "target_optimizer_backward_update": 0,
            "no_amp_tf32_compile_enablement": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def build_identity(root: Path) -> ScoreIdentity:
    try:
        v1.load_m1_metadata_manifest_authority(Path(root))
    except v1.SourceLifecycleError as error:
        raise HeldInScorePlanError("CS-WG held-in score sealed target metadata authority drift") from error
    return ScoreIdentity(ScoreSpec(), implementation_closure(Path(root)))


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static public declaration.  It intentionally performs no I/O."""
    del root
    spec = ScoreSpec()
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_score_root_relative": spec.root_relative,
        "completed_full_expectation": DEFAULT_COMPLETED_FULL_EXPECTATION.payload(),
        "score_spec": spec.payload(),
        "public_execution_authorized": False,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
    }


__all__ = (
    "HeldInScorePlanError", "CELL", "PHASE", "WORKORDER_RELATIVE", "WORKORDER_SHA256",
    "RESULT_ROOT_RELATIVE", "FULL_ROOT_RELATIVE", "FULL_TERMINAL_SHA256", "FULL_TRAINING_SHA256",
    "FULL_IDENTITY_SHA256", "CHECKPOINT_MANIFEST_SHA256", "BEST_CHECKPOINT_SHA256",
    "BEST_CHECKPOINT_STATE_SHA256", "FULL_BODY_NAMES", "TARGET_SESSION", "SOURCE_SESSIONS",
    "EVAL_BATCH_SIZE", "METRIC_LABEL", "MODEL_SHAPE", "ScoreSpec", "CompletedFullExpectation",
    "DEFAULT_COMPLETED_FULL_EXPECTATION", "ScoreIdentity", "canonical_json_bytes", "sha256_bytes",
    "require_sha", "safe_relative", "implementation_closure", "validate_current_closure",
    "build_identity", "dry_plan",
)
