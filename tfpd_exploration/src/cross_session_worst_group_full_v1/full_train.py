"""Descriptor-safe V6-bound CS-WG M1 fixed-epoch full lifecycle.

This module is intentionally stdlib-only.  It owns the full receipt graph but
delegates all physical parsing/model/optimizer work to ``physical`` only after
an opaque root-reviewed capability, durable attempt, and source authority.
The accepted V6 smoke graph is historical evidence: its terminal SHA pins the
complete 16-leaf graph without rebuilding V6's old current closure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

# Keep one qualified package identity.  The reviewed CS-WG namespace is
# ``tfpd_exploration.src``; importing the sibling as a top-level package would
# duplicate dataclass types and make an otherwise exact SourceRouteSpec fail
# ``isinstance`` at the capability boundary.
from ..cross_session_worst_group_v1 import plan
from ..cross_session_worst_group_v1 import source_lifecycle as v1
from ..cross_session_worst_group_v1 import source_smoke_v5 as v5
from ..cross_session_worst_group_v1 import source_smoke_v6 as v6


class CSWGFullTrainError(RuntimeError):
    """Fail closed for V6-bound full-source lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CSWGFullTrainError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG full {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG full relative path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG full relative path is unsafe")
    return path.as_posix()


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "m1_source_full_v1_no_swa_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_FULL_TRAIN_V1_NO_SWA_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "a71a7402d43f5012028dcd237d42861b647ec327e433f23df9c7a1bb10090179"
FULL_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa/fold_20120924_cswg"
MATCHED_ERM_FULL_ROOT_RELATIVE = (
    "tfpd_exploration/results/cross_session_worst_group_m1_matched_erm_full_v1_no_swa/"
    "fold_20120924_matched_erm"
)

V6_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6"
V6_TERMINAL_SHA256 = "dab4c9cfaad593c76d14b9d300642da31d7545248d4daf22bf2691c2b8cceb01"
V6_IDENTITY_SHA256 = "cd8c387317e27865f2b1bbc2b6594aa7d258dd10c9eb472493df0029fdda8879"
V6_CLOSURE_SHA256 = "4aa89da216fdf401e6c129a12af28cbb45e9f858d5f4db092fb24b1d77c6947c"
V6_BODY_NAMES = (
    "attempt.json", "launch.json", "source_authority.json", "smoke.json",
    "checkpoint_best_source_train_loss.pt", "checkpoint_last.pt",
    "checkpoint_manifest.json", "terminal.json",
)

_RUNTIME_DEPENDENCY_PATHS = (
    "streaming_calibration_exp/configs/callbacks/m1_source_only_fixed_epoch.yaml",
    "streaming_calibration_exp/configs/experiment/m1_afc4_source_decoder_fold0.yaml",
    "streaming_calibration_exp/src/callbacks/override_epoch_step.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v3.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_physical_v4.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v5.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v6.py",
)
_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_full_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_full_v1/full_train.py",
    "tfpd_exploration/src/cross_session_worst_group_full_v1/physical.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_full_v1.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_full_v1.py",
)


@dataclass(frozen=True)
class AcceptedV6GraphExpectation:
    """Terminal-pinned expectation for the completed historical V6 graph."""

    root_relative: str = V6_ROOT_RELATIVE
    terminal_sha256: str = V6_TERMINAL_SHA256
    identity_sha256: str = V6_IDENTITY_SHA256
    closure_sha256: str = V6_CLOSURE_SHA256

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == self.root_relative
                 and all(_require_sha(item, "V6 expectation") for item in (
                     self.terminal_sha256, self.identity_sha256, self.closure_sha256,
                 )), "CS-WG full V6 expectation drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_accepted_v6_terminal_expectation_v1",
            "root_relative": self.root_relative,
            "terminal_sha256": self.terminal_sha256,
            "identity_sha256": self.identity_sha256,
            "closure_sha256": self.closure_sha256,
            "exact_body_pairs": list(V6_BODY_NAMES),
            "exact_leaf_count": len(V6_BODY_NAMES) * 2,
            "leaf_mode": "0444",
            "canonical_basename_sidecars": True,
            "failure_leaf_forbidden": True,
        }


DEFAULT_V6_EXPECTATION = AcceptedV6GraphExpectation()

# The shared full lifecycle has one additional exact objective member: the
# same-fold ordinary ERM reference.  Its autograd derivative with respect to
# the three complete per-session losses is the uniform mean, not the CS-WG
# softmax weighting used by the historical V5 observer.
MATCHED_ERM_DERIVATIVE_SUM_ABS_TOLERANCE = 2.0e-6
MATCHED_ERM_DERIVATIVE_REFERENCE_ABS_TOLERANCE = 2.0e-6
MATCHED_ERM_DERIVATIVE_MINIMUM = -2.0e-6


def matched_erm_derivative_gate_contract_payload() -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_matched_erm_derivative_gate_v1",
        "objective_system": "MATCHED_ERM",
        "objective_lambda": 0.0,
        "objective_tau": v1.CSWG_TAU,
        "reference": "exact_uniform_mean_of_three_complete_session_losses",
        "raw_sum_abs_error_max": MATCHED_ERM_DERIVATIVE_SUM_ABS_TOLERANCE,
        "raw_reference_abs_error_max": MATCHED_ERM_DERIVATIVE_REFERENCE_ABS_TOLERANCE,
        "raw_minimum": MATCHED_ERM_DERIVATIVE_MINIMUM,
        "raw_fp32_autograd_observed": True,
        "live_result_tuning_forbidden": True,
    }


def _leaf_snapshot(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(info.st_dev), int(info.st_ino), int(stat.S_IFMT(info.st_mode)),
        int(info.st_size), int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def _read_held_pair(root_fd: int, name: str, *, expected_sha256: str | None = None) -> bytes:
    """Read one immutable body/sidecar pair through the held root FD only."""
    _require(name in V6_BODY_NAMES and Path(name).name == name, "CS-WG full V6 body name drift")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    _require(isinstance(no_follow, int) and no_follow != 0,
             "CS-WG full V6 no-follow file descriptors are unavailable")
    values: list[bytes] = []
    for leaf in (name, f"{name}.sha256"):
        try:
            named_before = os.stat(leaf, dir_fd=root_fd, follow_symlinks=False)
            descriptor = os.open(leaf, os.O_RDONLY | os.O_CLOEXEC | no_follow, dir_fd=root_fd)
        except OSError as error:
            raise CSWGFullTrainError(f"CS-WG full V6 immutable leaf inaccessible: {leaf}") from error
        try:
            opened = os.fstat(descriptor)
            _require(stat.S_ISREG(named_before.st_mode) and not stat.S_ISLNK(named_before.st_mode)
                     and stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode) == 0o444
                     and opened.st_nlink == 1 and _leaf_snapshot(named_before) == _leaf_snapshot(opened),
                     f"CS-WG full V6 immutable leaf mode/identity drift: {leaf}")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1 << 20):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            named_after = os.stat(leaf, dir_fd=root_fd, follow_symlinks=False)
            _require(_leaf_snapshot(after) == _leaf_snapshot(opened)
                     and _leaf_snapshot(named_after) == _leaf_snapshot(opened),
                     f"CS-WG full V6 immutable leaf changed while read: {leaf}")
            values.append(b"".join(chunks))
        finally:
            os.close(descriptor)
    body, sidecar = values
    digest = _sha(body)
    _require(expected_sha256 is None or digest == _require_sha(expected_sha256, "V6 expected body"),
             f"CS-WG full V6 immutable body digest drift: {name}")
    _require(sidecar == f"{digest}  {name}\n".encode("ascii"),
             f"CS-WG full V6 immutable sidecar drift: {name}")
    return body


def _read_held_json(root_fd: int, name: str, *, expected_sha256: str | None = None) -> tuple[dict[str, object], str]:
    body = _read_held_pair(root_fd, name, expected_sha256=expected_sha256)
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CSWGFullTrainError(f"CS-WG full V6 immutable JSON decode drift: {name}") from error
    _require(isinstance(value, dict), f"CS-WG full V6 immutable JSON object drift: {name}")
    return value, _sha(body)


@dataclass(frozen=True)
class AcceptedV6SmokeGraph:
    expectation: AcceptedV6GraphExpectation
    root_identity: tuple[int, int]
    body_sha256: Mapping[str, str]
    identity: Mapping[str, object] = field(repr=False, compare=False)
    terminal: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        bodies = dict(self.body_sha256)
        _require(isinstance(self.expectation, AcceptedV6GraphExpectation)
                 and len(self.root_identity) == 2 and all(type(value) is int and value >= 0 for value in self.root_identity)
                 and tuple(sorted(bodies)) == tuple(sorted(V6_BODY_NAMES))
                 and all(_require_sha(digest, f"V6 {name}") for name, digest in bodies.items())
                 and _sha(_json_bytes(dict(self.identity))) == self.expectation.identity_sha256
                 and bodies["terminal.json"] == self.expectation.terminal_sha256,
                 "CS-WG full held V6 graph topology drift")
        object.__setattr__(self, "body_sha256", MappingProxyType(bodies))
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))
        object.__setattr__(self, "terminal", MappingProxyType(dict(self.terminal)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_held_v6_smoke_graph_v1",
            "expectation": self.expectation.payload(),
            "root_identity": list(self.root_identity),
            "body_sha256": dict(self.body_sha256),
            "terminal_status": self.terminal.get("status"),
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def historical_v6_audit_spec(graph: AcceptedV6SmokeGraph) -> v1.SourceRouteSpec:
    """Rehydrate V6's historical audit builder spec from its terminal bytes.

    The full route must not call a current V3/V4/V5/V6 identity builder merely
    to rediscover the audit spec: those historical closures are intentionally
    immutable evidence.  The accepted V6 terminal already contains the typed
    V3-audit bridge.  We exact-compare that payload to the sealed V1 audit
    authority and return only the rehydrated typed spec needed by the parser
    seam.
    """
    _require(isinstance(graph, AcceptedV6SmokeGraph), "CS-WG full held V6 graph type drift")
    identity = graph.identity
    nested = identity.get("accepted_v3_identity")
    expected = v1.source_audit_spec()
    _require(identity.get("schema") == "cross_session_worst_group_m1_source_smoke_identity_v6"
             and identity.get("cell") == CELL
             and identity.get("phase") == "m1_source_smoke_v6_audit_spec_successor"
             and isinstance(nested, Mapping)
             and nested.get("schema") == "cross_session_worst_group_m1_historical_v3_audit_identity_v6"
             and nested.get("inherited_v1_audit_spec") == expected.payload()
             and nested.get("historical_v3_identity_sha256") == v6.V3_IDENTITY_SHA256
             and nested.get("historical_v3_closure_sha256") == v6.V3_CLOSURE_SHA256
             and identity.get("accepted_v3_completed_graph") == v6.V3_COMPLETED_EXPECTATION.payload()
             and identity.get("failed_v5_graph") == v6.V5_FAILED_EXPECTATION.payload(),
             "CS-WG full accepted V6 historical audit-spec bridge drift")
    return expected


def load_accepted_v6_smoke_graph(
    root: Path, *, expectation: AcceptedV6GraphExpectation = DEFAULT_V6_EXPECTATION,
) -> AcceptedV6SmokeGraph:
    """Descriptor-load the exact accepted V6 graph without rebuilding V6 code."""
    _require(isinstance(expectation, AcceptedV6GraphExpectation), "CS-WG full V6 expectation type drift")
    candidate = Path(root).absolute() / _safe_relative(expectation.root_relative)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    _require(isinstance(no_follow, int) and no_follow != 0 and isinstance(directory, int) and directory != 0,
             "CS-WG full V6 no-follow root descriptors are unavailable")
    try:
        named_before = os.lstat(candidate)
        root_fd = os.open(candidate, os.O_RDONLY | os.O_CLOEXEC | no_follow | directory)
    except OSError as error:
        raise CSWGFullTrainError("CS-WG full accepted V6 root inaccessible") from error
    try:
        held = os.fstat(root_fd)
        _require(stat.S_ISDIR(named_before.st_mode) and not stat.S_ISLNK(named_before.st_mode)
                 and stat.S_ISDIR(held.st_mode) and (held.st_dev, held.st_ino) == (named_before.st_dev, named_before.st_ino),
                 "CS-WG full accepted V6 root identity drift")
        expected_leaves = tuple(sorted((*V6_BODY_NAMES, *(f"{name}.sha256" for name in V6_BODY_NAMES))))
        _require(tuple(sorted(os.listdir(root_fd))) == expected_leaves,
                 "CS-WG full accepted V6 topology/failure/extra leaf drift")
        terminal, terminal_sha = _read_held_json(root_fd, "terminal.json", expected_sha256=expectation.terminal_sha256)
        identity = terminal.get("identity")
        _require(isinstance(identity, Mapping) and _sha(_json_bytes(dict(identity))) == expectation.identity_sha256
                 and terminal.get("schema") == "cross_session_worst_group_m1_source_smoke_terminal_v6"
                 and terminal.get("status") == "PASS_SOURCE_SMOKE_COMMON_STRATUM_CONSTRUCTIBLE_AUDIT_SPEC_REBOUND"
                 and terminal.get("launch_closure_sha256") == expectation.closure_sha256
                 and terminal.get("final_closure_sha256") == expectation.closure_sha256
                 and terminal.get("source_only") is True
                 and terminal.get("target_optimizer_backward_update") == 0
                 and all(terminal.get(key) is expected for key, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()),
                 "CS-WG full accepted V6 terminal identity/closure/flags drift")
        body_sha = {"terminal.json": terminal_sha}
        terminal_links = {
            "attempt.json": terminal.get("attempt_sha256"),
            "launch.json": terminal.get("launch_sha256"),
            "source_authority.json": terminal.get("source_authority_sha256"),
            "smoke.json": terminal.get("smoke_sha256"),
            "checkpoint_manifest.json": terminal.get("checkpoint_manifest_sha256"),
        }
        json_bodies: dict[str, dict[str, object]] = {"terminal.json": terminal}
        for name, digest in terminal_links.items():
            item, observed = _read_held_json(root_fd, name, expected_sha256=_require_sha(digest, f"V6 terminal {name}"))
            body_sha[name] = observed
            json_bodies[name] = item
        for name in ("attempt.json", "launch.json", "source_authority.json"):
            _require(json_bodies[name].get("identity") == identity,
                     f"CS-WG full accepted V6 {name} identity propagation drift")
        smoke = json_bodies["smoke.json"]
        _require(smoke.get("identity_sha256") == expectation.identity_sha256
                 and smoke.get("optimizer_steps") == v1.SMOKE_STEPS
                 and smoke.get("source_only") is True
                 and smoke.get("target_optimizer_backward_update") == 0
                 and all(smoke.get(key) is expected for key, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()),
                 "CS-WG full accepted V6 smoke semantics drift")
        manifest = json_bodies["checkpoint_manifest.json"]
        inherited_manifest = manifest.get("inherited_v1_checkpoint_manifest")
        _require(manifest.get("schema") == "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v6"
                 and isinstance(inherited_manifest, Mapping)
                 and isinstance(inherited_manifest.get("checkpoints"), Mapping),
                 "CS-WG full accepted V6 checkpoint manifest drift")
        roles = inherited_manifest["checkpoints"]
        _require(tuple(sorted(roles)) == ("best_source_train_loss", "last"),
                 "CS-WG full accepted V6 checkpoint role topology drift")
        for role, name in (("best_source_train_loss", "checkpoint_best_source_train_loss.pt"),
                           ("last", "checkpoint_last.pt")):
            entry = roles[role]
            _require(isinstance(entry, Mapping) and entry.get("filename") == name,
                     "CS-WG full accepted V6 checkpoint filename drift")
            body = _read_held_pair(root_fd, name, expected_sha256=_require_sha(entry.get("sha256"), f"V6 {role} checkpoint"))
            body_sha[name] = _sha(body)
        _require(tuple(sorted(body_sha)) == tuple(sorted(V6_BODY_NAMES)),
                 "CS-WG full accepted V6 body topology drift")
        named_after = os.lstat(candidate)
        _require(stat.S_ISDIR(named_after.st_mode) and (named_after.st_dev, named_after.st_ino) == (held.st_dev, held.st_ino),
                 "CS-WG full accepted V6 root changed while read")
        graph = AcceptedV6SmokeGraph(expectation, (held.st_dev, held.st_ino), body_sha, dict(identity), terminal)
        historical_v6_audit_spec(graph)
        return graph
    finally:
        os.close(root_fd)


def _full_cswg_and_erm_specs() -> tuple[v1.SourceRouteSpec, v1.SourceRouteSpec]:
    pair = v1.build_fold_route_specs(v1.SOURCE_SMOKE_OUTER_TARGET)
    _require(len(pair) == 2 and pair[0].stage0_spec.system == "CS_WG"
             and pair[1].stage0_spec.system == "MATCHED_ERM", "CS-WG full fixed fold-pair drift")
    return pair


@dataclass(frozen=True)
class FullTrainingSpec:
    """Own root plus one exact member of the fixed CS-WG/ERM fold pair.

    The no-argument form is deliberately the historical CS-WG full route.
    ``MATCHED_ERM`` is admitted only as its fixed same-fold counterpart, with
    the same graph, seed, source roster, optimizer recipe, and checkpoint
    policy.  It is not a caller-configurable objective search surface.
    """

    root_relative: str = FULL_ROOT_RELATIVE
    inherited_v1_full_spec: v1.SourceRouteSpec = field(default_factory=lambda: _full_cswg_and_erm_specs()[0])
    paired_matched_erm_spec: v1.SourceRouteSpec = field(default_factory=lambda: _full_cswg_and_erm_specs()[1])

    def __post_init__(self) -> None:
        expected_cswg, expected_erm = _full_cswg_and_erm_specs()
        system = self.inherited_v1_full_spec.stage0_spec.system
        if system == "CS_WG":
            expected_inherited, expected_counterpart = expected_cswg, expected_erm
        elif system == "MATCHED_ERM":
            expected_inherited, expected_counterpart = expected_erm, expected_cswg
        else:
            raise CSWGFullTrainError("CS-WG full system kind drift")
        _require(_safe_relative(self.root_relative) == self.root_relative
                 and self.inherited_v1_full_spec == expected_inherited
                 and self.paired_matched_erm_spec == expected_counterpart
                 and self.inherited_v1_full_spec.stage0_spec.outer_target_session == v1.SOURCE_SMOKE_OUTER_TARGET
                 and self.inherited_v1_full_spec.stage0_spec.source_sessions
                     == v1.source_smoke_spec().stage0_spec.source_sessions,
                 "CS-WG full V6 fold/system pairing drift")
        if system == "CS_WG":
            _require(self.root_relative == FULL_ROOT_RELATIVE,
                     "CS-WG full historical artifact-root drift")
        else:
            _require(self.root_relative == MATCHED_ERM_FULL_ROOT_RELATIVE,
                     "CS-WG matched-ERM full artifact-root drift")
        if system == "CS_WG":
            plan.validate_matched_same_fold_pair(
                self.inherited_v1_full_spec.stage0_spec, self.paired_matched_erm_spec.stage0_spec,
            )
        else:
            plan.validate_matched_same_fold_pair(
                self.paired_matched_erm_spec.stage0_spec, self.inherited_v1_full_spec.stage0_spec,
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_full_training_spec_v1",
            "root_relative": self.root_relative,
            "inherited_v1_full_spec": self.inherited_v1_full_spec.payload(),
            "paired_matched_erm_spec": self.paired_matched_erm_spec.payload(),
            "full_system": self.inherited_v1_full_spec.stage0_spec.system,
            "epoch_count": plan.M1_EPOCH_BUDGET,
            "optimizer": plan.M1_OPTIMIZER_LITERAL,
            "adam_lr": plan.M1_ADAM_LR,
            "adam_weight_decay": plan.M1_ADAM_WEIGHT_DECAY,
            "scheduler": plan.M1_LR_SCHEDULE_LITERAL,
            "checkpoint_monitor": "source_train_loss_epoch_mean",
            "checkpoint_tie_break": "first_strict_minimum_epoch",
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def full_training_spec(*, system: str = "CS_WG", root_relative: str | None = None) -> FullTrainingSpec:
    """Return one fixed member of the sealed fold-20120924 pair.

    Calling without arguments remains byte-semantic with the historical
    CS-WG route.  The only alternate value is the exact same-fold ERM member
    already constructed by ``build_fold_route_specs``.
    """
    _require(system in {"CS_WG", "MATCHED_ERM"}, "CS-WG full requested system drift")
    if system == "CS_WG":
        _require(root_relative is None or root_relative == FULL_ROOT_RELATIVE,
                 "CS-WG full historical root override is forbidden")
        return FullTrainingSpec()
    cswg, erm = _full_cswg_and_erm_specs()
    selected_root = MATCHED_ERM_FULL_ROOT_RELATIVE if root_relative is None else root_relative
    _require(selected_root == MATCHED_ERM_FULL_ROOT_RELATIVE,
             "CS-WG matched-ERM full root override is forbidden")
    return FullTrainingSpec(
        root_relative=selected_root,
        inherited_v1_full_spec=erm,
        paired_matched_erm_spec=cswg,
    )


def implementation_closure(
    root: Path, *, successor_extension_paths: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build an explicit full-route closure with an optional typed successor tail.

    The default empty extension remains the historical CS-WG full route.  A
    successor may add only an explicit, safe, duplicate-free list of its own
    immutable leaves; this avoids silently treating current CS-WG bytes as an
    ERM closure or rebuilding historical V6 code.
    """
    _require(isinstance(successor_extension_paths, tuple)
             and all(isinstance(item, str) and _safe_relative(item) == item
                     for item in successor_extension_paths)
             and len(set(successor_extension_paths)) == len(successor_extension_paths),
             "CS-WG full successor closure-extension topology drift")
    try:
        inherited = v1.implementation_closure(Path(root))
    except v1.SourceLifecycleError as error:
        raise CSWGFullTrainError("CS-WG full current V1 closure drift") from error
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list) and inherited_rows, "CS-WG full inherited closure topology drift")
    rows = [dict(row) for row in inherited_rows]
    existing = {row.get("path") for row in rows}
    for relative in (*_RUNTIME_DEPENDENCY_PATHS, *_OWNED_PATHS, *successor_extension_paths):
        _require(relative not in existing, "CS-WG full closure path duplication")
        try:
            digest = v6._read_regular_no_follow(Path(root), relative)
        except BaseException as error:
            raise CSWGFullTrainError(f"CS-WG full closure leaf drift: {relative}") from error
        rows.append({"path": relative, "sha256": digest})
        existing.add(relative)
    workorder = next((row for row in rows if row["path"] == WORKORDER_RELATIVE), None)
    _require(isinstance(workorder, Mapping) and workorder.get("sha256") == WORKORDER_SHA256,
             "CS-WG full workorder literal/body drift")
    body: dict[str, object] = {
        "schema": "cross_session_worst_group_m1_full_training_closure_v1",
        "current_v1_source_closure_sha256": inherited.get("closure_sha256"),
        "historical_v6_terminal_sha256": V6_TERMINAL_SHA256,
        "historical_v6_identity_sha256": V6_IDENTITY_SHA256,
        "historical_v6_closure_sha256": V6_CLOSURE_SHA256,
        "paths": rows,
    }
    if successor_extension_paths:
        body["successor_extension_paths"] = list(successor_extension_paths)
    return {**body, "closure_sha256": _sha(_json_bytes(body))}

def validate_current_closure(
    root: Path, value: Mapping[str, object], *, successor_extension_paths: tuple[str, ...] = (),
) -> None:
    _require(isinstance(value, Mapping)
             and dict(value) == implementation_closure(Path(root), successor_extension_paths=successor_extension_paths),
             "CS-WG full successor closure/current-byte drift")


@dataclass(frozen=True)
class FullTrainingIdentity:
    spec: FullTrainingSpec
    inherited_v1_full_identity: v1.SourceExecutionIdentity
    closure: Mapping[str, object]
    v6_expectation: AcceptedV6GraphExpectation = DEFAULT_V6_EXPECTATION
    successor_extension_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, FullTrainingSpec)
                 and isinstance(self.inherited_v1_full_identity, v1.SourceExecutionIdentity)
                 and self.inherited_v1_full_identity.spec == self.spec.inherited_v1_full_spec
                 and isinstance(self.closure, Mapping) and _require_sha(self.closure.get("closure_sha256"), "closure")
                 and isinstance(self.v6_expectation, AcceptedV6GraphExpectation)
                 and isinstance(self.successor_extension_paths, tuple)
                 and all(isinstance(item, str) and _safe_relative(item) == item
                         for item in self.successor_extension_paths)
                 and len(set(self.successor_extension_paths)) == len(self.successor_extension_paths),
                 "CS-WG full identity topology drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    @property
    def device(self) -> v1.DeviceProfile:
        return self.inherited_v1_full_identity.device

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "cross_session_worst_group_m1_full_training_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "inherited_v1_full_identity": self.inherited_v1_full_identity.payload(),
            "accepted_v6_smoke_expectation": self.v6_expectation.payload(),
            "closure": dict(self.closure),
            "source_only": True,
            "no_amp_tf32_compile": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }
        # Preserve the historical default CS-WG identity payload exactly.  An
        # additive successor is explicitly visible only when it binds leaves.
        if self.successor_extension_paths:
            payload["successor_extension_paths"] = list(self.successor_extension_paths)
        return payload

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def build_full_training_identity(
    root: Path, *, device: v1.DeviceProfile,
    v6_expectation: AcceptedV6GraphExpectation = DEFAULT_V6_EXPECTATION,
    spec: FullTrainingSpec | None = None,
    successor_extension_paths: tuple[str, ...] = (),
) -> FullTrainingIdentity:
    selected_spec = full_training_spec() if spec is None else spec
    _require(isinstance(selected_spec, FullTrainingSpec), "CS-WG full selected spec type drift")
    try:
        inherited = v1.build_identity(Path(root), spec=selected_spec.inherited_v1_full_spec, device=device)
    except v1.SourceLifecycleError as error:
        raise CSWGFullTrainError("CS-WG full inherited V1 identity drift") from error
    return FullTrainingIdentity(
        selected_spec,
        inherited,
        implementation_closure(Path(root), successor_extension_paths=successor_extension_paths),
        v6_expectation,
        successor_extension_paths,
    )


def validate_full_training_identity_current(root: Path, identity: FullTrainingIdentity) -> None:
    _require(isinstance(identity, FullTrainingIdentity), "CS-WG full identity must be typed")
    try:
        v1.validate_identity_current(Path(root), identity.inherited_v1_full_identity)
    except v1.SourceLifecycleError as error:
        raise CSWGFullTrainError("CS-WG full inherited V1 identity/current closure drift") from error
    validate_current_closure(
        Path(root), identity.closure, successor_extension_paths=identity.successor_extension_paths,
    )
    _require(identity.spec == full_training_spec(
                 system=identity.spec.inherited_v1_full_spec.stage0_spec.system,
                 root_relative=identity.spec.root_relative,
             ),
             "CS-WG full identity spec drift")


class _FullCapabilitySeal:
    pass


_FULL_CAPABILITY_SEAL = _FullCapabilitySeal()


@dataclass(frozen=True)
class FullTrainingCapability:
    identity_sha256: str
    accepted_v6_graph_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(_require_sha(self.identity_sha256, "capability identity")
                 and _require_sha(self.accepted_v6_graph_sha256, "capability V6 graph")
                 and self._seal is _FULL_CAPABILITY_SEAL,
                 "CS-WG full requires an in-process root-reviewed capability")


def assert_prospective_full_root_fresh(root: Path, spec: FullTrainingSpec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise CSWGFullTrainError("CS-WG full prospective root cannot be safely inspected") from error
    raise CSWGFullTrainError("CS-WG full prospective root already exists")


def _issue_root_reviewed_full_capability(
    root: Path, *, identity: FullTrainingIdentity, environ: Mapping[str, str] | None,
    graph_loader: Callable[[Path], AcceptedV6SmokeGraph] = load_accepted_v6_smoke_graph,
) -> FullTrainingCapability:
    validate_full_training_identity_current(Path(root), identity)
    _require(callable(graph_loader), "CS-WG full V6 graph loader absent")
    graph = graph_loader(Path(root))
    _require(isinstance(graph, AcceptedV6SmokeGraph)
             and graph.expectation == identity.v6_expectation,
             "CS-WG full V6 graph/identity expectation drift")
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise CSWGFullTrainError("CS-WG full device environment drift before capability") from error
    assert_prospective_full_root_fresh(Path(root), identity.spec)
    return FullTrainingCapability(identity.sha256, graph.sha256, _FULL_CAPABILITY_SEAL)


def issue_root_reviewed_full_capability(
    root: Path, *, identity: FullTrainingIdentity, environ: Mapping[str, str] | None,
) -> FullTrainingCapability:
    return _issue_root_reviewed_full_capability(Path(root), identity=identity, environ=environ)


def _require_full_capability(capability: object, identity: FullTrainingIdentity) -> FullTrainingCapability:
    _require(isinstance(capability, FullTrainingCapability)
             and capability._seal is _FULL_CAPABILITY_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG full capability/identity drift")
    return capability


def _flags_exact(value: Mapping[str, object]) -> bool:
    return (value.get("source_only") is True
            and value.get("target_optimizer_backward_update") == 0
            and all(value.get(name) is expected for name, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()))


def _reject_swa_claim(value: Mapping[str, object]) -> None:
    keys = set(value)
    swa_keys = {key for key in keys if "swa" in key.lower()}
    # Explicit negative evidence is allowed so receipts can disclose the
    # route's no-SWA contract.  No manifest/proof/state/selection alias is.
    _require(swa_keys <= {"swa_enabled", "swa_artifact_forbidden"}
             and value.get("swa_enabled", False) is False
             and value.get("swa_artifact_forbidden") is True,
             "CS-WG full SWA artifact/claim is forbidden")


def _full_attempt_payload(identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_full_training_attempt_v1",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "accepted_v6_smoke_graph_sha256": graph.sha256,
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_launch_backend(
    value: object, *, graph: AcceptedV6SmokeGraph, system: str = "CS_WG",
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG full physical launch payload absent")
    item = dict(value)
    expected_observer = (
        "FullDerivativeEvidenceCollector" if system == "CS_WG"
        else "MatchedERMDerivativeEvidenceCollector" if system == "MATCHED_ERM"
        else None
    )
    _require(item.get("schema") == "cross_session_worst_group_m1_full_training_physical_launch_v1"
             and item.get("provider") == "V6BoundFullCommonStratumSourceProvider"
             and item.get("runner") == "TorchCSWGFullTrainingRunner"
             and (item.get("derivative_observer", "FullDerivativeEvidenceCollector") == expected_observer
                  if system == "CS_WG" else item.get("derivative_observer") == expected_observer)
             and item.get("accepted_v6_smoke_graph_sha256") == graph.sha256
             and item.get("source_opened") is False and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False and item.get("optimizer_steps_completed") == 0
             and _flags_exact(item), "CS-WG full physical launch schema drift")
    _reject_swa_claim(item)
    return item


def _full_launch_payload(
    identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph, attempt_sha256: str, backend: object,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_full_training_launch_v1",
        "identity": identity.payload(),
        "accepted_v6_smoke_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "launch attempt"),
        "backend": _validate_launch_backend(
            backend, graph=graph, system=identity.inherited_v1_full_identity.spec.stage0_spec.system,
        ),
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _full_source_authority_payload(
    identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph, prepared: object,
) -> dict[str, object]:
    fragment = getattr(prepared, "authority_fragment", None)
    _require(callable(fragment), "CS-WG full prepared source-authority seam drift")
    source = dict(fragment())
    valid_windows = source.get("valid_source_windows")
    _require(isinstance(valid_windows, Mapping)
             and tuple(valid_windows) == identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions,
             "CS-WG full source authority valid-window topology drift")
    expected_steps = v1.paired_epoch_step_count(identity.inherited_v1_full_identity.spec, valid_windows)
    _require(expected_steps > 0 and source.get("paired_cswg_and_matched_erm_steps_per_epoch") == expected_steps,
             "CS-WG full source authority epoch-cardinality drift")
    payload = {
        "schema": "cross_session_worst_group_m1_full_training_source_authority_v1",
        "identity_sha256": identity.sha256,
        "accepted_v6_smoke_graph_sha256": graph.sha256,
        "inherited_v1_full_source_authority": source,
        "training_plan": {
            "epoch_count": plan.M1_EPOCH_BUDGET,
            "steps_per_epoch": expected_steps,
            "total_optimizer_steps": plan.M1_EPOCH_BUDGET * expected_steps,
            "episode_index_scope": "epoch_local",
            "source_train_loss_aggregation": "arithmetic_mean_of_complete_source_objective_per_step",
            "checkpoint_selection": "epoch_mean_source_train_loss_first_minimum",
            "optimizer": plan.M1_OPTIMIZER_LITERAL,
            "adam_lr": plan.M1_ADAM_LR,
            "adam_weight_decay": plan.M1_ADAM_WEIGHT_DECAY,
            "scheduler": plan.M1_LR_SCHEDULE_LITERAL,
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
        },
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }
    _validate_full_source_authority(payload, identity, graph)
    return payload


def _validate_full_source_authority(value: object, identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG full source authority must be a mapping")
    item = dict(value)
    training = item.get("training_plan")
    source = item.get("inherited_v1_full_source_authority")
    _require(item.get("schema") == "cross_session_worst_group_m1_full_training_source_authority_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("accepted_v6_smoke_graph_sha256") == graph.sha256
             and isinstance(training, Mapping) and isinstance(source, Mapping)
             and training.get("epoch_count") == plan.M1_EPOCH_BUDGET
             and type(training.get("steps_per_epoch")) is int and training["steps_per_epoch"] > 0
             and training.get("total_optimizer_steps") == plan.M1_EPOCH_BUDGET * training["steps_per_epoch"]
             and training.get("episode_index_scope") == "epoch_local"
             and training.get("source_train_loss_aggregation") == "arithmetic_mean_of_complete_source_objective_per_step"
             and training.get("checkpoint_selection") == "epoch_mean_source_train_loss_first_minimum"
             and training.get("optimizer") == plan.M1_OPTIMIZER_LITERAL
             and training.get("adam_lr") == plan.M1_ADAM_LR
             and training.get("adam_weight_decay") == plan.M1_ADAM_WEIGHT_DECAY
             and training.get("scheduler") == plan.M1_LR_SCHEDULE_LITERAL
             and source.get("paired_cswg_and_matched_erm_steps_per_epoch") == training["steps_per_epoch"]
             and source.get("accepted_v6_smoke_graph_sha256") == graph.sha256
             and source.get("accepted_v6_historical_audit_spec") == historical_v6_audit_spec(graph).payload()
             and source.get("accepted_v6_historical_audit_spec_sha256") == historical_v6_audit_spec(graph).sha256
             and _flags_exact(item), "CS-WG full source authority schema/science drift")
    _reject_swa_claim(item)
    _reject_swa_claim(training)
    return item


def _validate_epoch_payload(value: object, *, identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph,
                            authority_sha256: str, epoch_index: int, steps_per_epoch: int) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG full epoch payload must be a mapping")
    item = dict(value)
    finite = ("source_train_loss",)
    _require(item.get("schema") == "cross_session_worst_group_m1_full_training_epoch_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("accepted_v6_smoke_graph_sha256") == graph.sha256
             and item.get("source_authority_sha256") == authority_sha256
             and item.get("epoch_index") == epoch_index
             and item.get("steps_in_epoch") == steps_per_epoch
             and item.get("global_optimizer_steps_completed") == (epoch_index + 1) * steps_per_epoch
             and item.get("source_train_loss_aggregation") == "arithmetic_mean_of_complete_source_objective_per_step"
             and item.get("episode_index_scope") == "epoch_local"
             and item.get("one_concatenated_forward_per_step") is True
             and item.get("finite_objective") is True and item.get("finite_model") is True
             and item.get("finite_adam_state") is True
             and all(type(item.get(name)) in {int, float} and math.isfinite(float(item[name])) for name in finite)
             and _require_sha(item.get("model_state_sha256"), "epoch model state")
             and _flags_exact(item), "CS-WG full epoch receipt/schema drift")
    _reject_swa_claim(item)
    return item


def _validate_full_derivative_evidence(value: object, *, identity: FullTrainingIdentity, expected_steps: int) -> dict[str, object]:
    """Validate the objective-specific compact derivative evidence.

    The historical CS-WG path keeps its V5 softmax-reference gate unchanged.
    The exact matched-ERM counterpart instead proves the uniform derivative
    of the arithmetic mean; applying a softmax reference at lambda=0 would be
    a semantic validation error rather than an additional safeguard.
    """
    _require(isinstance(value, Mapping), "CS-WG full derivative evidence must be a mapping")
    item = dict(value)
    sessions = list(identity.inherited_v1_full_identity.spec.stage0_spec.source_sessions)
    system = identity.inherited_v1_full_identity.spec.stage0_spec.system
    if system == "MATCHED_ERM":
        expected_weight = 1.0 / float(len(sessions))
        _require(item.get("schema") == "cross_session_worst_group_m1_matched_erm_full_derivative_numeric_evidence_v1"
                 and item.get("objective_system") == "MATCHED_ERM"
                 and item.get("objective_lambda") == 0.0
                 and item.get("objective_tau") == v1.CSWG_TAU
                 and item.get("session_ids") == sessions
                 and item.get("steps_observed") == expected_steps
                 and item.get("gate") == matched_erm_derivative_gate_contract_payload()
                 and item.get("uniform_reference_weight") == expected_weight
                 and _require_sha(item.get("raw_observation_domain_separated_sha256"), "matched ERM derivative digest")
                 and all(type(item.get(name)) in {int, float} and math.isfinite(float(item[name]))
                         for name in ("raw_global_min", "raw_global_max", "max_sum_abs_error", "max_reference_abs_error"))
                 and float(item["raw_global_min"]) >= MATCHED_ERM_DERIVATIVE_MINIMUM
                 and float(item["raw_global_max"]) <= 1.0 + MATCHED_ERM_DERIVATIVE_REFERENCE_ABS_TOLERANCE
                 and float(item["max_sum_abs_error"]) <= MATCHED_ERM_DERIVATIVE_SUM_ABS_TOLERANCE
                 and float(item["max_reference_abs_error"]) <= MATCHED_ERM_DERIVATIVE_REFERENCE_ABS_TOLERANCE
                 and item.get("all_rows_pass") is True
                 and item.get("retained_gpu_tensors") == 0
                 and item.get("second_forward_or_backward") is False,
                 "CS-WG matched-ERM full derivative numeric evidence drift")
        return item
    _require(system == "CS_WG", "CS-WG full derivative system drift")
    _require(item.get("schema") == "cross_session_worst_group_m1_full_derivative_numeric_evidence_v1"
             and item.get("session_ids") == sessions
             and item.get("steps_observed") == expected_steps
             and item.get("gate") == v5.derivative_gate_contract_payload()
             and _require_sha(item.get("raw_observation_domain_separated_sha256"), "full derivative digest")
             and all(type(item.get(name)) in {int, float} and math.isfinite(float(item[name]))
                     for name in ("raw_global_min", "raw_global_max", "max_sum_abs_error", "max_reference_abs_error"))
             and float(item["raw_global_min"]) >= v5.DERIVATIVE_MIN_TOLERANCE
             and float(item["raw_global_max"]) <= 1.0 + v5.DERIVATIVE_REFERENCE_ABS_TOLERANCE
             and float(item["max_sum_abs_error"]) <= v5.DERIVATIVE_SUM_ABS_TOLERANCE
             and float(item["max_reference_abs_error"]) <= v5.DERIVATIVE_REFERENCE_ABS_TOLERANCE
             and item.get("all_rows_pass") is True
             and item.get("retained_gpu_tensors") == 0
             and item.get("second_forward_or_backward") is False,
             "CS-WG full V5 derivative numeric evidence drift")
    return item


def _full_training_payload(identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph, authority_sha256: str,
                           raw: Mapping[str, object], *, epoch_sha256: Mapping[str, str]) -> dict[str, object]:
    item = dict(raw)
    expected_steps = plan.M1_EPOCH_BUDGET * int(item.get("training_plan", {}).get("steps_per_epoch", -1)) \
        if isinstance(item.get("training_plan"), Mapping) else -1
    _require(item.get("schema") == "cross_session_worst_group_m1_source_full_training_physical_v1"
             and item.get("identity_sha256") == identity.inherited_v1_full_identity.sha256
             and expected_steps > 0 and item.get("optimizer_steps") == expected_steps
             and item.get("one_concatenated_forward_per_step") is True
             and item.get("model_parameter_count") == plan.M1_LIVE_PARAMETERS_AFTER_LAZY1024
             and item.get("model_output_shape") == [plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS]
             and item.get("calibration_shape_per_row") == list(plan.M1_CALIBRATION_SHAPE_PER_ROW)
             and item.get("finite_objective") is True and item.get("finite_model") is True
             and item.get("finite_gradients") is True and item.get("finite_adam_state") is True
             and item.get("checkpoint_reload_strict") is True
             and item.get("best_checkpoint_reload_strict") is True and item.get("last_checkpoint_reload_strict") is True
             and item.get("model_state_changed") is True
             and _require_sha(item.get("initial_model_state_sha256"), "full initial state")
             and _require_sha(item.get("final_model_state_sha256"), "full final state")
             and _require_sha(item.get("best_checkpoint_state_sha256"), "full best state")
             and item.get("initial_model_state_sha256") != item.get("final_model_state_sha256")
             and type(item.get("best_source_train_loss")) in {int, float}
             and math.isfinite(float(item["best_source_train_loss"]))
             and type(item.get("best_source_train_loss_epoch_index")) is int
             and 0 <= item["best_source_train_loss_epoch_index"] < plan.M1_EPOCH_BUDGET
             and isinstance(item.get("epochs"), list) and len(item["epochs"]) == plan.M1_EPOCH_BUDGET
             and _flags_exact(item), "CS-WG full physical training result drift")
    _reject_swa_claim(item)
    coverage = item.get("gradient_coverage")
    _require(isinstance(coverage, Mapping) and type(coverage.get("trainable_parameter_count")) is int
             and coverage.get("trainable_parameter_count", 0) > 0
             and coverage.get("observed_gradient_count") == coverage.get("trainable_parameter_count")
             and coverage.get("observed_gradient_names_sha256") == coverage.get("trainable_parameter_names_sha256")
             and coverage.get("missing_trainable_names") == [] and coverage.get("excluded_trainable_names") == [],
             "CS-WG full trainable-gradient coverage drift")
    resources = item.get("resources")
    try:
        v1._validate_resources(resources)
    except v1.SourceLifecycleError as error:
        raise CSWGFullTrainError("CS-WG full resource evidence drift") from error
    runtime = item.get("runtime_environment")
    _require(isinstance(runtime, Mapping) and runtime.get("tf32_matmul_after") is False
             and runtime.get("tf32_cudnn_after") is False and runtime.get("amp") is False
             and runtime.get("compile") is False, "CS-WG full runtime TF32/AMP/compile evidence drift")
    rng = item.get("rng")
    _require(isinstance(rng, Mapping) and rng.get("seed") == v1.SEED
             and rng.get("post_run_state_restored") is True, "CS-WG full RNG restoration evidence drift")
    _validate_full_derivative_evidence(
        item.get("derivative_numeric_evidence"), identity=identity, expected_steps=expected_steps,
    )
    epoch_rows = item["epochs"]
    min_loss = math.inf
    first_minimum = None
    for index, row in enumerate(epoch_rows):
        _require(isinstance(row, Mapping), "CS-WG full physical epoch row type drift")
        observed = dict(row)
        _require(observed.get("schema") == "cross_session_worst_group_m1_source_training_epoch_v1"
                 and observed.get("epoch_index") == index
                 and observed.get("steps_in_epoch") == expected_steps // plan.M1_EPOCH_BUDGET
                 and observed.get("global_optimizer_steps_completed") == (index + 1) * observed["steps_in_epoch"]
                 and type(observed.get("source_train_loss")) in {int, float}
                 and math.isfinite(float(observed["source_train_loss"]))
                 and _require_sha(observed.get("model_state_sha256"), "physical epoch state"),
                 "CS-WG full physical epoch cardinality/state drift")
        value = float(observed["source_train_loss"])
        if value < min_loss:
            min_loss = value
            first_minimum = index
    _require(first_minimum == item["best_source_train_loss_epoch_index"]
             and float(item["best_source_train_loss"]) == min_loss
             and tuple(epoch_sha256) == tuple(f"epoch_{index:02d}.json" for index in range(plan.M1_EPOCH_BUDGET)),
             "CS-WG full physical first-minimum/epoch receipt topology drift")
    payload = {
        "schema": "cross_session_worst_group_m1_full_training_result_v1",
        "identity_sha256": identity.sha256,
        "accepted_v6_smoke_graph_sha256": graph.sha256,
        "source_authority_sha256": authority_sha256,
        "inherited_shared_training_result": item,
        "inherited_shared_training_result_sha256": _sha(_json_bytes(item)),
        "epoch_sha256": dict(epoch_sha256),
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }
    return payload


def _validate_full_training_payload(value: object, identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph,
                                    authority_sha256: str, epoch_sha256: Mapping[str, str]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG full training receipt must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_shared_training_result")
    _require(item.get("schema") == "cross_session_worst_group_m1_full_training_result_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("accepted_v6_smoke_graph_sha256") == graph.sha256
             and item.get("source_authority_sha256") == authority_sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_shared_training_result_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("epoch_sha256") == dict(epoch_sha256)
             and _flags_exact(item), "CS-WG full training wrapper provenance drift")
    _reject_swa_claim(item)
    # Re-run the strict semantic body validation; all entries were produced by
    # the shared runner and are now receipt-bound before checkpoint links.
    _full_training_payload(identity, graph, authority_sha256, dict(inherited), epoch_sha256=epoch_sha256)
    return item


def _publish_full_checkpoints(
    artifact: v1.ImmutableArtifactRoot, identity: FullTrainingIdentity, training: Mapping[str, object],
    bodies: object,
) -> tuple[dict[str, object], str]:
    _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last"),
             "CS-WG full checkpoint body topology drift")
    inherited = training.get("inherited_shared_training_result")
    _require(isinstance(inherited, Mapping), "CS-WG full inherited result absent before checkpoints")
    roles: dict[str, dict[str, object]] = {}
    for role, name, state_key in (
        ("best_source_train_loss", "checkpoint_best_source_train_loss.pt", "best_checkpoint_state_sha256"),
        ("last", "checkpoint_last.pt", "final_model_state_sha256"),
    ):
        body = bodies[role]
        _require(isinstance(body, bytes) and body, f"CS-WG full {role} checkpoint bytes drift")
        roles[role] = {
            "filename": name,
            "sha256": artifact.publish_bytes(name, body),
            "state_sha256": _require_sha(inherited.get(state_key), f"full {role} state"),
            "strict_reload": True,
        }
    manifest = {
        "schema": "cross_session_worst_group_m1_full_training_checkpoint_manifest_v1",
        "identity_sha256": identity.sha256,
        "checkpoints": roles,
        "monitor": "source_train_loss_epoch_mean",
        "mode": "min",
        "tie_break": "first_strict_minimum_epoch",
        "best_epoch_index": inherited.get("best_source_train_loss_epoch_index"),
        "last_epoch_index": plan.M1_EPOCH_BUDGET - 1,
        "early_stopping": False,
        "validation_or_target_selection": False,
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }
    _validate_full_checkpoint_manifest(manifest, identity)
    return manifest, artifact.publish_json("checkpoint_manifest.json", manifest)


def _validate_full_checkpoint_manifest(value: object, identity: FullTrainingIdentity) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG full checkpoint manifest must be a mapping")
    item = dict(value)
    roles = item.get("checkpoints")
    _require(item.get("schema") == "cross_session_worst_group_m1_full_training_checkpoint_manifest_v1"
             and item.get("identity_sha256") == identity.sha256
             and isinstance(roles, Mapping) and tuple(sorted(roles)) == ("best_source_train_loss", "last")
             and item.get("monitor") == "source_train_loss_epoch_mean" and item.get("mode") == "min"
             and item.get("tie_break") == "first_strict_minimum_epoch"
             and type(item.get("best_epoch_index")) is int and 0 <= item["best_epoch_index"] < plan.M1_EPOCH_BUDGET
             and item.get("last_epoch_index") == plan.M1_EPOCH_BUDGET - 1
             and item.get("early_stopping") is False and item.get("validation_or_target_selection") is False
             and _flags_exact(item), "CS-WG full checkpoint manifest schema/selection drift")
    _reject_swa_claim(item)
    for role, filename in (("best_source_train_loss", "checkpoint_best_source_train_loss.pt"),
                           ("last", "checkpoint_last.pt")):
        entry = roles[role]
        _require(isinstance(entry, Mapping) and entry.get("filename") == filename
                 and _require_sha(entry.get("sha256"), f"manifest {role} body")
                 and _require_sha(entry.get("state_sha256"), f"manifest {role} state")
                 and entry.get("strict_reload") is True,
                 "CS-WG full checkpoint manifest role drift")
    return item


def _full_terminal_payload(identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph, *, attempt_sha256: str,
                           launch_sha256: str, authority_sha256: str, training_sha256: str,
                           epoch_sha256: Mapping[str, str], manifest: Mapping[str, object],
                           manifest_sha256: str) -> dict[str, object]:
    _validate_full_checkpoint_manifest(manifest, identity)
    roles = manifest["checkpoints"]
    assert isinstance(roles, Mapping)
    best = roles["best_source_train_loss"]
    last = roles["last"]
    assert isinstance(best, Mapping) and isinstance(last, Mapping)
    return {
        "schema": "cross_session_worst_group_m1_full_training_terminal_v1",
        "cell": CELL,
        "status": "PASS_SOURCE_FULL_FIXED_20_EPOCH_NO_SWA",
        "identity": identity.payload(),
        "accepted_v6_smoke_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "terminal authority"),
        "training_sha256": _require_sha(training_sha256, "terminal training"),
        "epoch_sha256": dict(epoch_sha256),
        "checkpoint_manifest_sha256": _require_sha(manifest_sha256, "terminal manifest"),
        # The terminal independently pins the two and only two approved full
        # artifacts.  Their strict state links remain in the manifest, so a
        # later scorer never needs an SWA placeholder to recover provenance.
        "checkpoint_best_source_train_loss_sha256": _require_sha(
            best.get("sha256"), "terminal best checkpoint",
        ),
        "checkpoint_last_sha256": _require_sha(last.get("sha256"), "terminal last checkpoint"),
        "checkpoint_best_source_train_loss_state_sha256": _require_sha(
            best.get("state_sha256"), "terminal best checkpoint state",
        ),
        "checkpoint_last_state_sha256": _require_sha(last.get("state_sha256"), "terminal last checkpoint state",
        ),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _failure_payload(identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph, *, attempt_sha256: str,
                     launch_sha256: str | None, authority_sha256: str | None, progress: v1.LifecycleProgress,
                     error: BaseException, epoch_sha256: Mapping[str, str]) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_full_training_failure_v1",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "accepted_v6_smoke_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "failure launch"),
        "source_authority_sha256": None if authority_sha256 is None else _require_sha(authority_sha256, "failure authority"),
        "epoch_sha256": dict(epoch_sha256),
        "progress": progress.payload(),
        "error_class": type(error).__name__,
        "error_sha256": _sha(repr(error).encode("utf-8")),
        "terminal_published": False,
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


@dataclass(frozen=True)
class FullTrainingLifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    training_sha256: str | None
    checkpoint_manifest_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None
    epoch_sha256: Mapping[str, str]


class DeferredFullTrainingBackend(Protocol):
    def launch_payload(self, identity: FullTrainingIdentity) -> Mapping[str, object]: ...
    def prepare_source(self, identity: FullTrainingIdentity) -> object: ...
    def run_full(self, identity: FullTrainingIdentity, epoch_observer: Callable[[Mapping[str, object]], None]) -> Mapping[str, object]: ...
    def checkpoint_bodies(self) -> Mapping[str, bytes]: ...
    def progress(self) -> v1.LifecycleProgress: ...
    def close(self) -> None: ...


def _expected_success_names(epoch_count: int, *, terminal: bool) -> tuple[str, ...]:
    bodies = ["attempt.json", "launch.json", "source_authority.json"]
    bodies.extend(f"epoch_{index:02d}.json" for index in range(epoch_count))
    bodies.extend(("training.json", "checkpoint_best_source_train_loss.pt", "checkpoint_last.pt", "checkpoint_manifest.json"))
    if terminal:
        bodies.append("terminal.json")
    return tuple(item for body in bodies for item in (body, f"{body}.sha256"))


def _revalidate_published_full_graph(
    artifact: v1.ImmutableArtifactRoot, identity: FullTrainingIdentity, graph: AcceptedV6SmokeGraph,
    *, attempt_sha256: str, launch_sha256: str, authority_sha256: str, training_sha256: str,
    epoch_sha256: Mapping[str, str], manifest_sha256: str,
) -> None:
    artifact.validate_live(expected_names=_expected_success_names(plan.M1_EPOCH_BUDGET, terminal=False))
    attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
    launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
    authority = artifact.read_json_pair("source_authority.json", expected_sha256=authority_sha256)
    training = artifact.read_json_pair("training.json", expected_sha256=training_sha256)
    manifest = artifact.read_json_pair("checkpoint_manifest.json", expected_sha256=manifest_sha256)
    _require(attempt == _full_attempt_payload(identity, graph)
             and launch == _full_launch_payload(identity, graph, attempt_sha256, launch.get("backend"))
             and launch.get("attempt_sha256") == attempt_sha256,
             "CS-WG full immutable attempt/launch graph drift")
    _validate_full_source_authority(authority, identity, graph)
    steps = authority["training_plan"]["steps_per_epoch"]
    assert type(steps) is int
    for index in range(plan.M1_EPOCH_BUDGET):
        name = f"epoch_{index:02d}.json"
        body = artifact.read_json_pair(name, expected_sha256=epoch_sha256[name])
        _validate_epoch_payload(body, identity=identity, graph=graph, authority_sha256=authority_sha256,
                                epoch_index=index, steps_per_epoch=steps)
    _validate_full_training_payload(training, identity, graph, authority_sha256, epoch_sha256)
    inherited = training["inherited_shared_training_result"]
    assert isinstance(inherited, Mapping)
    _validate_full_checkpoint_manifest(manifest, identity)
    roles = manifest["checkpoints"]
    assert isinstance(roles, Mapping)
    _require(roles["best_source_train_loss"].get("state_sha256") == inherited.get("best_checkpoint_state_sha256")
             and roles["last"].get("state_sha256") == inherited.get("final_model_state_sha256"),
             "CS-WG full checkpoint/state provenance drift")
    for role in ("best_source_train_loss", "last"):
        entry = roles[role]
        assert isinstance(entry, Mapping)
        artifact.read_bytes_pair(str(entry["filename"]), expected_sha256=str(entry["sha256"]))


def _execute_reviewed_full_training(
    root: Path, *, identity: FullTrainingIdentity, capability: object, backend: DeferredFullTrainingBackend,
    environ: Mapping[str, str] | None, graph_loader: Callable[[Path], AcceptedV6SmokeGraph],
) -> FullTrainingLifecycleResult:
    cap = _require_full_capability(capability, identity)
    validate_full_training_identity_current(Path(root), identity)
    graph = graph_loader(Path(root))
    _require(graph.expectation == identity.v6_expectation and graph.sha256 == cap.accepted_v6_graph_sha256,
             "CS-WG full capability/V6 predecessor drift before reserve")
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise CSWGFullTrainError("CS-WG full device environment drift before reserve") from error
    assert_prospective_full_root_fresh(Path(root), identity.spec)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), identity.spec)  # type: ignore[arg-type]
    attempt_sha256 = artifact.publish_json("attempt.json", _full_attempt_payload(identity, graph))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    training_sha256: str | None = None
    manifest_sha256: str | None = None
    terminal_sha256: str | None = None
    epoch_sha256: dict[str, str] = {}
    try:
        launch_sha256 = artifact.publish_json(
            "launch.json", _full_launch_payload(identity, graph, attempt_sha256, backend.launch_payload(identity)),
        )
        prepared = backend.prepare_source(identity)
        authority = _full_source_authority_payload(identity, graph, prepared)
        authority_sha256 = artifact.publish_json("source_authority.json", authority)
        steps = authority["training_plan"]["steps_per_epoch"]
        assert type(steps) is int

        def publish_epoch(raw: Mapping[str, object]) -> None:
            index = len(epoch_sha256)
            body = {
                **dict(raw),
                # The shared runner emits a generic source-training summary;
                # the durable full receipt owns its distinct schema while
                # retaining every measured scalar/state field verbatim.
                "schema": "cross_session_worst_group_m1_full_training_epoch_v1",
                "identity_sha256": identity.sha256,
                "accepted_v6_smoke_graph_sha256": graph.sha256,
                "source_authority_sha256": authority_sha256,
                "swa_enabled": False,
                "swa_artifact_forbidden": True,
            }
            _validate_epoch_payload(body, identity=identity, graph=graph, authority_sha256=authority_sha256,
                                    epoch_index=index, steps_per_epoch=steps)
            name = f"epoch_{index:02d}.json"
            epoch_sha256[name] = artifact.publish_json(name, body)

        raw = dict(backend.run_full(identity, publish_epoch))
        bodies = raw.pop("_checkpoint_bodies", None)
        _require(len(epoch_sha256) == plan.M1_EPOCH_BUDGET,
                 "CS-WG full did not publish exactly 20 epoch receipts")
        training = _full_training_payload(identity, graph, authority_sha256, raw, epoch_sha256=epoch_sha256)
        _validate_full_training_payload(training, identity, graph, authority_sha256, epoch_sha256)
        training_sha256 = artifact.publish_json("training.json", training)
        if bodies is None:
            bodies = backend.checkpoint_bodies()
        manifest, manifest_sha256 = _publish_full_checkpoints(artifact, identity, training, bodies)
        validate_full_training_identity_current(Path(root), identity)
        graph_now = graph_loader(Path(root))
        _require(graph_now.sha256 == graph.sha256 == cap.accepted_v6_graph_sha256,
                 "CS-WG full immutable V6 predecessor drifted during full training")
        try:
            v1.validate_device_environment(identity.device, environ)
        except v1.SourceLifecycleError as error:
            raise CSWGFullTrainError("CS-WG full device environment drift before terminal") from error
        _revalidate_published_full_graph(
            artifact, identity, graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            authority_sha256=authority_sha256, training_sha256=training_sha256,
            epoch_sha256=epoch_sha256, manifest_sha256=manifest_sha256,
        )
        terminal = _full_terminal_payload(
            identity, graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            authority_sha256=authority_sha256, training_sha256=training_sha256,
            epoch_sha256=epoch_sha256, manifest=manifest, manifest_sha256=manifest_sha256,
        )
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=_expected_success_names(plan.M1_EPOCH_BUDGET, terminal=True))
        _require(artifact.read_json_pair("terminal.json", expected_sha256=terminal_sha256) == terminal,
                 "CS-WG full immutable terminal body/sidecar drift")
        return FullTrainingLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            training_sha256, manifest_sha256, terminal_sha256, None, MappingProxyType(dict(epoch_sha256)),
        )
    except BaseException as error:
        # Once the immutable terminal pair exists, this route must never add
        # a failure pair beside it.  A post-publication descriptor anomaly is
        # surfaced to the root caller for containment; all normal failure
        # receipts are necessarily pre-terminal prefixes.
        if terminal_sha256 is not None:
            raise
        try:
            progress = backend.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG full backend failure-progress type drift")
        except BaseException:
            progress = v1.LifecycleProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json", _failure_payload(
                identity, graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
                authority_sha256=authority_sha256, progress=progress, error=error, epoch_sha256=epoch_sha256,
            ),
        )
        return FullTrainingLifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            training_sha256, manifest_sha256, terminal_sha256, failure_sha256, MappingProxyType(dict(epoch_sha256)),
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def execute_reviewed_full_training(
    root: Path, *, identity: FullTrainingIdentity, capability: object, backend: DeferredFullTrainingBackend,
    environ: Mapping[str, str] | None,
) -> FullTrainingLifecycleResult:
    """Root-only physical entrypoint; public CLI is dry and cannot call this."""
    return _execute_reviewed_full_training(
        Path(root), identity=identity, capability=capability, backend=backend, environ=environ,
        graph_loader=load_accepted_v6_smoke_graph,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static no-I/O public declaration of the prospective full candidate."""
    del root
    spec = full_training_spec()
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_full_root_relative": spec.root_relative,
        "accepted_v6_smoke_expectation": DEFAULT_V6_EXPECTATION.payload(),
        "full_spec": spec.payload(),
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "public_execution_authorized": False,
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
    }


__all__ = (
    "CSWGFullTrainError", "AcceptedV6GraphExpectation", "AcceptedV6SmokeGraph",
    "DEFAULT_V6_EXPECTATION", "FullTrainingSpec", "FullTrainingIdentity", "FullTrainingCapability",
    "FullTrainingLifecycleResult", "load_accepted_v6_smoke_graph", "full_training_spec",
    "historical_v6_audit_spec",
    "implementation_closure", "validate_current_closure", "build_full_training_identity",
    "validate_full_training_identity_current", "issue_root_reviewed_full_capability",
    "assert_prospective_full_root_fresh", "execute_reviewed_full_training", "dry_plan",
)
