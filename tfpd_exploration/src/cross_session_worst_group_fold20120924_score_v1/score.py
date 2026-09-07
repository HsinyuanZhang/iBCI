"""Immutable producer binding and metric-only held-in score lifecycle.

This module is deliberately standard-library-only.  The Torch/parser work is
behind ``physical`` and can happen only after this lifecycle has published a
durable attempt.  The completed full route is never re-executed or treated as
current code: its terminal-pinned immutable receipt graph is descriptor-read
as historical evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1

from . import plan


class HeldInScoreError(RuntimeError):
    """Fail closed for producer, lifecycle, or score evidence drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldInScoreError(message)


def _json_bytes(value: object) -> bytes:
    return plan.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative(value: object) -> str:
    try:
        return plan.safe_relative(value)
    except plan.HeldInScorePlanError as error:
        raise HeldInScoreError(str(error)) from error


def _require_sha(value: object, label: str) -> str:
    try:
        return plan.require_sha(value, label)
    except plan.HeldInScorePlanError as error:
        raise HeldInScoreError(str(error)) from error


def _leaf_snapshot(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(info.st_dev), int(info.st_ino), int(stat.S_IFMT(info.st_mode)),
        int(info.st_size), int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def _directory_identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _no_follow() -> int:
    flag = getattr(os, "O_NOFOLLOW", 0)
    _require(isinstance(flag, int) and flag != 0,
             "CS-WG held-in score requires O_NOFOLLOW")
    return flag


def _open_held_result_directory(root: Path, relative: str) -> tuple[int, list[int], tuple[tuple[str, int, int], ...]]:
    """Open a root-relative directory through a held no-follow chain."""
    base = Path(root).absolute()
    parts = Path(_safe_relative(relative)).parts
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | _no_follow()
    _require(getattr(os, "O_DIRECTORY", 0) != 0, "CS-WG held-in score requires O_DIRECTORY")
    try:
        named_root = os.lstat(base)
        _require(stat.S_ISDIR(named_root.st_mode) and not stat.S_ISLNK(named_root.st_mode),
                 "CS-WG held-in score repository root is noncanonical")
        root_fd = os.open(base, flags)
    except OSError as error:
        raise HeldInScoreError("CS-WG held-in score repository root cannot be opened no-follow") from error
    opened = [root_fd]
    identities: list[tuple[str, int, int]] = [(".", *_directory_identity(named_root))]
    try:
        opened_root = os.fstat(root_fd)
        _require(stat.S_ISDIR(opened_root.st_mode)
                 and _directory_identity(opened_root) == _directory_identity(named_root),
                 "CS-WG held-in score repository root changed during open")
        current_fd = root_fd
        for index, component in enumerate(parts):
            named = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode),
                     "CS-WG held-in score predecessor directory type/symlink drift")
            child_fd = os.open(component, flags, dir_fd=current_fd)
            opened.append(child_fd)
            opened_info = os.fstat(child_fd)
            _require(stat.S_ISDIR(opened_info.st_mode)
                     and _directory_identity(opened_info) == _directory_identity(named),
                     "CS-WG held-in score predecessor directory changed during open")
            identities.append(("/".join(parts[:index + 1]), *_directory_identity(opened_info)))
            current_fd = child_fd
        return root_fd, opened, tuple(identities)
    except OSError as error:
        for descriptor in reversed(opened):
            os.close(descriptor)
        raise HeldInScoreError("CS-WG held-in score predecessor directory open failed") from error
    except BaseException:
        for descriptor in reversed(opened):
            os.close(descriptor)
        raise


def _close_held_chain(opened: list[int]) -> None:
    for descriptor in reversed(opened):
        os.close(descriptor)


def _named_chain_identities(root: Path, relative: str) -> tuple[tuple[str, int, int], ...]:
    _root_fd, opened, identities = _open_held_result_directory(Path(root), relative)
    try:
        return identities
    finally:
        _close_held_chain(opened)


def _read_0444_leaf(directory_fd: int, name: str) -> bytes:
    _require(isinstance(name, str) and Path(name).name == name,
             "CS-WG held-in score immutable leaf name drift")
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
                 f"CS-WG held-in score immutable leaf is nonregular: {name}")
        descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | _no_follow(), dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode) == 0o444
                 and int(opened.st_nlink) == 1 and _leaf_snapshot(opened) == _leaf_snapshot(before),
                 f"CS-WG held-in score immutable leaf mode/identity drift: {name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(_leaf_snapshot(after) == _leaf_snapshot(opened)
                 and _leaf_snapshot(named_after) == _leaf_snapshot(opened),
                 f"CS-WG held-in score immutable leaf changed while read: {name}")
        return b"".join(chunks)
    except HeldInScoreError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise HeldInScoreError(f"CS-WG held-in score immutable leaf symlink drift: {name}") from error
        raise HeldInScoreError(f"CS-WG held-in score immutable leaf inaccessible: {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_pair(directory_fd: int, name: str, *, expected_sha256: str | None = None) -> tuple[bytes, str]:
    body = _read_0444_leaf(directory_fd, name)
    digest = _sha(body)
    if expected_sha256 is not None:
        _require(digest == _require_sha(expected_sha256, f"expected {name}"),
                 f"CS-WG held-in score immutable body digest drift: {name}")
    sidecar = _read_0444_leaf(directory_fd, f"{name}.sha256")
    _require(sidecar == f"{digest}  {name}\n".encode("ascii"),
             f"CS-WG held-in score immutable canonical sidecar drift: {name}")
    return body, digest


def _read_json_pair(directory_fd: int, name: str, *, expected_sha256: str | None = None) -> tuple[dict[str, object], str]:
    body, digest = _read_pair(directory_fd, name, expected_sha256=expected_sha256)
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HeldInScoreError(f"CS-WG held-in score immutable JSON decode drift: {name}") from error
    _require(isinstance(parsed, dict), f"CS-WG held-in score immutable JSON body is not an object: {name}")
    return parsed, digest


def _expected_full_leaf_names() -> tuple[str, ...]:
    return tuple(sorted(item for body in plan.FULL_BODY_NAMES for item in (body, f"{body}.sha256")))


def _identity_fold_semantics(
    identity: Mapping[str, object], expectation: plan.CompletedFullExpectation,
) -> None:
    spec = identity.get("spec")
    _require(isinstance(spec, Mapping), "CS-WG held-in score completed full identity spec absent")
    inherited = spec.get("inherited_v1_full_spec")
    _require(isinstance(inherited, Mapping), "CS-WG held-in score completed full inherited spec absent")
    stage = inherited.get("stage0_run_spec")
    _require(isinstance(stage, Mapping)
             and stage.get("system") == expectation.expected_system
             and stage.get("outer_target_session") == plan.TARGET_SESSION
             and stage.get("source_sessions") == list(plan.SOURCE_SESSIONS)
             and spec.get("swa_enabled") is False and spec.get("swa_artifact_forbidden") is True,
             "CS-WG held-in score completed full fold/no-SWA semantics drift")
    if expectation.expected_objective_lambda is not None:
        _require(stage.get("lambda") == expectation.expected_objective_lambda,
                 "CS-WG held-in score completed full objective-lambda drift")
    if expectation.expected_objective_tau is not None:
        _require(stage.get("tau") == expectation.expected_objective_tau,
                 "CS-WG held-in score completed full objective-tau drift")


@dataclass(frozen=True)
class CompletedFullGraph:
    """Held, terminal-pinned full producer graph for one later score attempt."""

    expectation: plan.CompletedFullExpectation
    root_identity: tuple[int, int]
    named_chain_identities: tuple[tuple[str, int, int], ...]
    body_sha256: Mapping[str, str]
    terminal: Mapping[str, object] = field(repr=False, compare=False)
    manifest: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        body_sha = dict(self.body_sha256)
        _require(isinstance(self.expectation, plan.CompletedFullExpectation)
                 and len(self.root_identity) == 2 and all(type(value) is int and value >= 0 for value in self.root_identity)
                 and self.named_chain_identities
                 and tuple(sorted(body_sha)) == tuple(sorted(plan.FULL_BODY_NAMES))
                 and all(_require_sha(value, f"completed body {key}") for key, value in body_sha.items())
                 and body_sha.get("terminal.json") == self.expectation.terminal_sha256
                 and body_sha.get("training.json") == self.expectation.training_sha256
                 and body_sha.get("checkpoint_manifest.json") == self.expectation.checkpoint_manifest_sha256
                 and body_sha.get("checkpoint_best_source_train_loss.pt") == self.expectation.best_checkpoint_sha256
                 and body_sha.get("checkpoint_last.pt") == self.expectation.effective_last_checkpoint_sha256
                 and all(body_sha.get(name) == digest
                         for name, digest in self.expectation.extra_body_sha256.items()),
                 "CS-WG held-in score completed full graph topology drift")
        object.__setattr__(self, "body_sha256", MappingProxyType(body_sha))
        object.__setattr__(self, "terminal", MappingProxyType(dict(self.terminal)))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    def payload(self) -> dict[str, object]:
        body = {
            "schema": "cross_session_worst_group_m1_completed_full_graph_binding_v1",
            "expectation": self.expectation.payload(),
            "root_identity": list(self.root_identity),
            "named_chain_identities": [list(item) for item in self.named_chain_identities],
            "body_sha256": dict(self.body_sha256),
            "selected_checkpoint_role": "best_source_train_loss",
            "no_swa": True,
            "failure_absent": True,
        }
        return {**body, "binding_sha256": _sha(_json_bytes(body))}

    @property
    def sha256(self) -> str:
        return str(self.payload()["binding_sha256"])


def _validate_completed_full_semantics(
    *, payloads: Mapping[str, Mapping[str, object]], body_sha256: Mapping[str, str],
    expectation: plan.CompletedFullExpectation,
) -> None:
    terminal = payloads["terminal.json"]
    manifest = payloads["checkpoint_manifest.json"]
    training = payloads["training.json"]
    identity = terminal.get("identity")
    _require(terminal.get("schema") == "cross_session_worst_group_m1_full_training_terminal_v1"
             and terminal.get("status") == "PASS_SOURCE_FULL_FIXED_20_EPOCH_NO_SWA"
             and isinstance(identity, Mapping)
             and _sha(_json_bytes(dict(identity))) == expectation.identity_sha256
             and terminal.get("training_sha256") == expectation.training_sha256
             and terminal.get("checkpoint_manifest_sha256") == expectation.checkpoint_manifest_sha256
             and terminal.get("checkpoint_best_source_train_loss_sha256") == expectation.best_checkpoint_sha256
             and terminal.get("checkpoint_last_sha256") == expectation.effective_last_checkpoint_sha256
             and terminal.get("checkpoint_best_source_train_loss_state_sha256") == expectation.best_checkpoint_state_sha256
             and terminal.get("checkpoint_last_state_sha256") == expectation.effective_last_checkpoint_state_sha256
             and terminal.get("swa_enabled") is False and terminal.get("swa_artifact_forbidden") is True
             and terminal.get("source_only") is True and terminal.get("target_optimizer_backward_update") == 0,
             "CS-WG held-in score completed full terminal semantics drift")
    _identity_fold_semantics(identity, expectation)
    _require(training.get("schema") == "cross_session_worst_group_m1_full_training_result_v1"
             and training.get("identity_sha256") == _sha(_json_bytes(dict(identity)))
             and training.get("swa_enabled") is False and training.get("swa_artifact_forbidden") is True
             and training.get("source_only") is True,
             "CS-WG held-in score completed full training semantics drift")
    roles = manifest.get("checkpoints")
    _require(manifest.get("schema") == "cross_session_worst_group_m1_full_training_checkpoint_manifest_v1"
             and isinstance(roles, Mapping) and tuple(sorted(roles)) == ("best_source_train_loss", "last")
             and manifest.get("best_epoch_index") == expectation.best_epoch_index
             and manifest.get("last_epoch_index") == expectation.last_epoch_index
             and manifest.get("swa_enabled") is False and manifest.get("swa_artifact_forbidden") is True,
             "CS-WG held-in score completed full manifest topology drift")
    best = roles["best_source_train_loss"]
    last = roles["last"]
    _require(isinstance(best, Mapping) and isinstance(last, Mapping)
             and best.get("filename") == "checkpoint_best_source_train_loss.pt"
             and last.get("filename") == "checkpoint_last.pt"
             and best.get("sha256") == expectation.best_checkpoint_sha256
             and last.get("sha256") == expectation.effective_last_checkpoint_sha256
             and best.get("state_sha256") == expectation.best_checkpoint_state_sha256
             and last.get("state_sha256") == expectation.effective_last_checkpoint_state_sha256
             and best.get("strict_reload") is True and last.get("strict_reload") is True,
             "CS-WG held-in score selected best/last checkpoint provenance drift")
    epoch_sha = terminal.get("epoch_sha256")
    _require(isinstance(epoch_sha, Mapping)
             and tuple(epoch_sha) == tuple(f"epoch_{index:02d}.json" for index in range(20))
             and all(epoch_sha[name] == body_sha256[name] for name in epoch_sha)
             and terminal.get("attempt_sha256") == body_sha256["attempt.json"]
             and terminal.get("launch_sha256") == body_sha256["launch.json"]
             and terminal.get("source_authority_sha256") == body_sha256["source_authority.json"]
             and terminal.get("training_sha256") == body_sha256["training.json"]
             and terminal.get("checkpoint_manifest_sha256") == body_sha256["checkpoint_manifest.json"],
             "CS-WG held-in score completed full terminal link drift")


def load_completed_full_graph(
    root: Path, *, expectation: plan.CompletedFullExpectation = plan.DEFAULT_COMPLETED_FULL_EXPECTATION,
) -> CompletedFullGraph:
    """Descriptor-read the exact 56-leaf producer graph without tensor loading."""
    _require(isinstance(expectation, plan.CompletedFullExpectation),
             "CS-WG held-in score completed full expectation type drift")
    _base_fd, opened, identities = _open_held_result_directory(Path(root), expectation.root_relative)
    result_fd = opened[-1]
    try:
        expected_names = _expected_full_leaf_names()
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG held-in score completed full exact leaf topology drift")
        fixed = {
            "terminal.json": expectation.terminal_sha256,
            "training.json": expectation.training_sha256,
            "checkpoint_manifest.json": expectation.checkpoint_manifest_sha256,
            "checkpoint_best_source_train_loss.pt": expectation.best_checkpoint_sha256,
            "checkpoint_last.pt": expectation.effective_last_checkpoint_sha256,
            **dict(expectation.extra_body_sha256),
        }
        body_sha: dict[str, str] = {}
        payloads: dict[str, dict[str, object]] = {}
        for name in plan.FULL_BODY_NAMES:
            body, digest = _read_pair(result_fd, name, expected_sha256=fixed.get(name))
            body_sha[name] = digest
            if name.endswith(".json"):
                try:
                    parsed = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise HeldInScoreError(f"CS-WG held-in score completed full JSON decode drift: {name}") from error
                _require(isinstance(parsed, dict),
                         f"CS-WG held-in score completed full JSON type drift: {name}")
                payloads[name] = parsed
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG held-in score completed full topology changed during read")
        _validate_completed_full_semantics(payloads=payloads, body_sha256=body_sha, expectation=expectation)
        _require(_named_chain_identities(Path(root), expectation.root_relative) == identities,
                 "CS-WG held-in score completed full named directory identity drift")
        info = os.fstat(result_fd)
        return CompletedFullGraph(
            expectation, _directory_identity(info), identities, body_sha,
            payloads["terminal.json"], payloads["checkpoint_manifest.json"],
        )
    finally:
        _close_held_chain(opened)


def read_selected_checkpoint_bytes(root: Path, graph: CompletedFullGraph) -> bytes:
    """Descriptor-reload the selected historical best checkpoint after attempt.

    The caller supplies the already-held graph binding.  We reload the full
    producer graph first, then read the selected bytes through a fresh
    no-follow descriptor chain.  This intentionally is a byte/provenance
    operation, not a Torch tensor load; physical code performs the strict
    tensor-state load only after the new score input authority is durable.
    """
    _require(isinstance(graph, CompletedFullGraph),
             "CS-WG held-in score selected checkpoint graph type drift")
    reloaded = load_completed_full_graph(Path(root), expectation=graph.expectation)
    _require(reloaded.sha256 == graph.sha256,
             "CS-WG held-in score completed full graph drift before checkpoint read")
    _base_fd, opened, identities = _open_held_result_directory(Path(root), graph.expectation.root_relative)
    result_fd = opened[-1]
    try:
        body, digest = _read_pair(
            result_fd, "checkpoint_best_source_train_loss.pt",
            expected_sha256=graph.expectation.best_checkpoint_sha256,
        )
        _require(digest == graph.body_sha256["checkpoint_best_source_train_loss.pt"]
                 and _named_chain_identities(Path(root), graph.expectation.root_relative) == identities,
                 "CS-WG held-in score selected checkpoint named provenance drift")
        return body
    finally:
        _close_held_chain(opened)


def assert_prospective_score_root_fresh(root: Path, spec: plan.ScoreSpec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise HeldInScoreError("CS-WG held-in score prospective root cannot be safely inspected") from error
    raise HeldInScoreError("CS-WG held-in score canonical score root already exists")


def validate_identity_current(root: Path, identity: plan.ScoreIdentity) -> None:
    _require(isinstance(identity, plan.ScoreIdentity) and identity.spec == plan.ScoreSpec()
             and identity.completed_full == plan.DEFAULT_COMPLETED_FULL_EXPECTATION,
             "CS-WG held-in score identity/spec provenance drift")
    try:
        plan.validate_current_closure(Path(root), identity.closure)
    except plan.HeldInScorePlanError as error:
        raise HeldInScoreError("CS-WG held-in score current closure drift") from error
    try:
        metadata = v1.load_m1_metadata_manifest_authority(Path(root))
    except v1.SourceLifecycleError as error:
        raise HeldInScoreError("CS-WG held-in score sealed target metadata authority drift") from error
    _require(metadata.get("body_sha256") == v1.M1_METADATA_MANIFEST_SHA256,
             "CS-WG held-in score target metadata body binding drift")


class _ScoreReviewSeal:
    pass


_SCORE_REVIEW_SEAL = _ScoreReviewSeal()


@dataclass(frozen=True)
class ScoreCapability:
    identity_sha256: str
    completed_full_graph_sha256: str
    target_source_root: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "score capability identity")
        _require_sha(self.completed_full_graph_sha256, "score capability full graph")
        _require(isinstance(self.target_source_root, str) and self.target_source_root,
                 "CS-WG held-in score capability target source root is absent")
        candidate = Path(self.target_source_root)
        _require(candidate.is_absolute() and str(candidate) == self.target_source_root,
                 "CS-WG held-in score capability target source root must be lexical absolute")
        _require(self._seal is _SCORE_REVIEW_SEAL,
                 "CS-WG held-in score needs an in-process root-reviewed capability")


def issue_root_reviewed_score_capability(
    root: Path, identity: plan.ScoreIdentity, *, source_root: Path, review_seal: object,
) -> ScoreCapability:
    _require(review_seal is _SCORE_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG held-in score capability")
    validate_identity_current(Path(root), identity)
    graph = load_completed_full_graph(Path(root), expectation=identity.completed_full)
    assert_prospective_score_root_fresh(Path(root), identity.spec)
    # This is intentionally a lexical capability binding only: source-root
    # lstat/open/SHA is deferred until after the durable score attempt.
    _require(Path(source_root).is_absolute(),
             "CS-WG held-in score root reviewer must supply an absolute target source root")
    lexical_source_root = str(Path(source_root))
    return ScoreCapability(identity.sha256, graph.sha256, lexical_source_root, _SCORE_REVIEW_SEAL)


def _require_capability(capability: object, identity: plan.ScoreIdentity) -> ScoreCapability:
    _require(isinstance(capability, ScoreCapability) and capability._seal is _SCORE_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG held-in score exact root-reviewed capability is required")
    return capability


@dataclass(frozen=True)
class ScoreProgress:
    target_resolved_or_opened: bool = False
    checkpoint_opened: bool = False
    model_constructed: bool = False
    cuda_initialized: bool = False
    full_system_forward_count: int = 0
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0
    input_authority_published: bool = False

    def __post_init__(self) -> None:
        _require(all(type(value) is bool for value in (
            self.target_resolved_or_opened, self.checkpoint_opened, self.model_constructed,
            self.cuda_initialized, self.input_authority_published,
        )) and all(type(value) is int and value >= 0 for value in (
            self.full_system_forward_count, self.target_optimizer_steps,
            self.target_backward_calls, self.target_update_calls,
        )), "CS-WG held-in score lifecycle progress drift")

    def payload(self) -> dict[str, object]:
        return {
            "target_resolved_or_opened": self.target_resolved_or_opened,
            "checkpoint_opened": self.checkpoint_opened,
            "model_constructed": self.model_constructed,
            "cuda_initialized": self.cuda_initialized,
            "full_system_forward_count": self.full_system_forward_count,
            "target_optimizer_steps": self.target_optimizer_steps,
            "target_backward_calls": self.target_backward_calls,
            "target_update_calls": self.target_update_calls,
            "input_authority_published": self.input_authority_published,
            "target_metric_only": True,
            "target_labels_used_only_for_metric": True,
            "target_training_forbidden": True,
        }


class DeferredHeldInScoreBackend(Protocol):
    def launch_payload(self, identity: plan.ScoreIdentity, graph: CompletedFullGraph) -> Mapping[str, object]: ...
    def prepare_inputs(self, identity: plan.ScoreIdentity, graph: CompletedFullGraph) -> Mapping[str, object]: ...
    def score(self, identity: plan.ScoreIdentity, graph: CompletedFullGraph, input_authority: Mapping[str, object]) -> Mapping[str, object]: ...
    def progress(self) -> ScoreProgress: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ScoreLifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    input_authority_sha256: str | None
    score_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


def _attempt_payload(identity: plan.ScoreIdentity, graph: CompletedFullGraph) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_attempt_v1",
        "cell": plan.CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "completed_full_graph": graph.payload(),
        **ScoreProgress().payload(),
    }


def _launch_payload(
    identity: plan.ScoreIdentity, graph: CompletedFullGraph, attempt_sha256: str, backend: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_launch_v1",
        "cell": plan.CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "completed_full_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "launch attempt"),
        "backend": dict(backend),
        **ScoreProgress().payload(),
    }


_INPUT_PROTECTED = {
    "schema", "identity_sha256", "completed_full_graph_sha256", "target_session",
    "selected_checkpoint_role", "metric", "last_bin_only", "target_metric_only",
    "target_optimizer_backward_update", "target_labels_used_only_for_metric",
}


def _input_authority_payload(
    identity: plan.ScoreIdentity, graph: CompletedFullGraph, prepared: Mapping[str, object],
) -> dict[str, object]:
    _require(isinstance(prepared, Mapping) and not (set(prepared) & _INPUT_PROTECTED),
             "CS-WG held-in score backend input authority tried to overwrite protected fields")
    result = {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_input_authority_v1",
        "identity_sha256": identity.sha256,
        "completed_full_graph_sha256": graph.sha256,
        "target_session": plan.TARGET_SESSION,
        "selected_checkpoint_role": "best_source_train_loss",
        "metric": plan.METRIC_LABEL,
        "last_bin_only": True,
        "target_metric_only": True,
        "target_labels_used_only_for_metric": True,
        "target_optimizer_backward_update": 0,
    }
    result.update(dict(prepared))
    _validate_input_authority(result, identity, graph)
    return result


def _validate_input_authority(value: object, identity: plan.ScoreIdentity, graph: CompletedFullGraph) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG held-in score input authority must be a mapping")
    item = dict(value)
    required_sha = ("input_record_sha256", "ordered_window_start_sha256", "target_sha256", "calibration_sha256")
    _require(item.get("schema") == "cross_session_worst_group_m1_fold20120924_heldin_input_authority_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("completed_full_graph_sha256") == graph.sha256
             and item.get("target_session") == plan.TARGET_SESSION
             and item.get("selected_checkpoint_role") == "best_source_train_loss"
             and item.get("metric") == plan.METRIC_LABEL and item.get("last_bin_only") is True
             and item.get("target_metric_only") is True and item.get("target_labels_used_only_for_metric") is True
             and item.get("target_optimizer_backward_update") == 0
             and type(item.get("n_windows")) is int and item["n_windows"] > 0
             and item.get("model_input_shape") == [plan.MODEL_SHAPE["window"], plan.MODEL_SHAPE["units"]]
             and item.get("calibration_shape_per_row") == plan.MODEL_SHAPE["calibration_shape_per_row"]
             and all(_require_sha(item.get(name), f"input authority {name}") for name in required_sha),
             "CS-WG held-in score input authority semantics drift")
    return item


def _validate_score_payload(
    value: object, identity: plan.ScoreIdentity, graph: CompletedFullGraph, input_authority_sha256: str,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG held-in score payload must be a mapping")
    item = dict(value)
    numeric = item.get("governing_r2")
    required_sha = ("prediction_sha256", "target_sha256", "model_state_before_sha256", "model_state_after_sha256")
    _require(item.get("schema") == "cross_session_worst_group_m1_fold20120924_heldin_score_payload_v1"
             and item.get("identity_sha256") == identity.sha256
             and item.get("completed_full_graph_sha256") == graph.sha256
             and item.get("input_authority_sha256") == input_authority_sha256
             and item.get("target_session") == plan.TARGET_SESSION
             and item.get("selected_checkpoint_role") == "best_source_train_loss"
             and item.get("metric") == plan.METRIC_LABEL and item.get("last_bin_only") is True
             and item.get("eval_mode") is True and item.get("no_grad") is True
             and item.get("dynamic_dropout_disabled") is True and item.get("target_metric_only") is True
             and item.get("target_labels_used_only_for_metric") is True
             and item.get("target_optimizer_steps") == 0 and item.get("target_backward_calls") == 0
             and item.get("target_update_calls") == 0
             and type(item.get("full_system_forward_count")) is int and item["full_system_forward_count"] > 0
             and item.get("n_windows") == _validate_input_authority(
                 item.get("input_authority"), identity, graph,
             ).get("n_windows")
             and type(numeric) in {int, float} and math.isfinite(float(numeric))
             and all(_require_sha(item.get(name), f"score {name}") for name in required_sha)
             and item.get("model_state_before_sha256") == item.get("model_state_after_sha256")
             and item.get("target_sha256") == _validate_input_authority(
                 item.get("input_authority"), identity, graph,
             ).get("target_sha256"),
             "CS-WG held-in score metric/state/update semantics drift")
    return item


def _terminal_payload(
    identity: plan.ScoreIdentity, graph: CompletedFullGraph, *, attempt_sha256: str, launch_sha256: str,
    input_authority_sha256: str, score_sha256: str, score_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_terminal_v1",
        "cell": plan.CELL,
        "status": "COMPLETE_DESCRIPTIVE_HELDIN_R2",
        "identity": identity.payload(),
        "completed_full_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "terminal launch"),
        "input_authority_sha256": _require_sha(input_authority_sha256, "terminal input authority"),
        "score_sha256": _require_sha(score_sha256, "terminal score"),
        "governing_r2": score_payload["governing_r2"],
        "n_windows": score_payload["n_windows"],
        "prediction_sha256": score_payload["prediction_sha256"],
        "target_sha256": score_payload["target_sha256"],
        "selected_checkpoint_role": "best_source_train_loss",
        "matched_erm_reference_bound": False,
        "formal_benchmark_verdict": False,
        "target_optimizer_backward_update": 0,
        "target_metric_only": True,
    }


def _failure_payload(
    identity: plan.ScoreIdentity, graph: CompletedFullGraph, *, attempt_sha256: str,
    launch_sha256: str | None, input_authority_sha256: str | None, score_sha256: str | None,
    progress: ScoreProgress, error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_fold20120924_heldin_score_failure_v1",
        "cell": plan.CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "completed_full_graph_sha256": graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "failure launch"),
        "input_authority_sha256": None if input_authority_sha256 is None else _require_sha(
            input_authority_sha256, "failure input authority",
        ),
        "score_sha256": None if score_sha256 is None else _require_sha(score_sha256, "failure score"),
        "progress": progress.payload(),
        "error_class": type(error).__name__,
        "error_sha256": _sha(repr(error).encode("utf-8")),
        "terminal_published": False,
        "target_metric_only": True,
        "target_optimizer_backward_update": 0,
    }


def _success_names(*, terminal: bool) -> tuple[str, ...]:
    bodies = ["attempt.json", "launch.json", "input_authority.json", "score.json"]
    if terminal:
        bodies.append("terminal.json")
    return tuple(item for body in bodies for item in (body, f"{body}.sha256"))


def _revalidate_published_score_graph(
    artifact: v1.ImmutableArtifactRoot, identity: plan.ScoreIdentity, graph: CompletedFullGraph,
    *, attempt_sha256: str, launch_sha256: str, input_authority_sha256: str, score_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact.validate_live(expected_names=_success_names(terminal=False))
    attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
    launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
    inputs = artifact.read_json_pair("input_authority.json", expected_sha256=input_authority_sha256)
    scored = artifact.read_json_pair("score.json", expected_sha256=score_sha256)
    _require(attempt == _attempt_payload(identity, graph)
             and launch.get("attempt_sha256") == attempt_sha256
             and launch.get("completed_full_graph_sha256") == graph.sha256,
             "CS-WG held-in score published attempt/launch graph drift")
    _validate_input_authority(inputs, identity, graph)
    _require(scored.get("input_authority") == inputs,
             "CS-WG held-in score published input snapshot drift")
    _validate_score_payload(scored, identity, graph, input_authority_sha256)
    return inputs, scored


@dataclass(frozen=True)
class ProfiledHeldInScoreLifecycleHooks:
    """Typed, narrow lifecycle seam for same-surface successor scorers.

    This owns the one immutable attempt/input/score/terminal loop.  A profile
    supplies only its fixed identity/producer checks and receipt codec.  The
    historical V1 wrapper below supplies the original callbacks, so its
    payloads and evaluation ordering remain unchanged.
    """

    label: str
    pre_reserve: Callable[[Path, object, object, object], object]
    spec_for_identity: Callable[[object], object]
    attempt_payload: Callable[[object, object], Mapping[str, object]]
    launch_payload: Callable[[object, object, str, Mapping[str, object]], Mapping[str, object]]
    input_authority_payload: Callable[[object, object, Mapping[str, object]], Mapping[str, object]]
    validate_score_payload: Callable[[object, object, object, str], Mapping[str, object]]
    revalidate_before_terminal: Callable[
        [Path, object, object, object, v1.ImmutableArtifactRoot, str, str, str, str], Mapping[str, object]
    ]
    terminal_payload: Callable[[object, object, str, str, str, str, Mapping[str, object]], Mapping[str, object]]
    failure_payload: Callable[[object, object, str, str | None, str | None, str | None, ScoreProgress, BaseException], Mapping[str, object]]
    success_names: Callable[[bool], tuple[str, ...]]

    def __post_init__(self) -> None:
        _require(isinstance(self.label, str) and self.label
                 and all(callable(value) for value in (
                     self.pre_reserve, self.spec_for_identity, self.attempt_payload,
                     self.launch_payload, self.input_authority_payload, self.validate_score_payload,
                     self.revalidate_before_terminal, self.terminal_payload, self.failure_payload,
                     self.success_names,
                 )), "CS-WG held-in score profiled lifecycle hook drift")


def execute_profiled_heldin_score(
    root: Path, *, identity: object, capability: object, backend: object,
    hooks: ProfiledHeldInScoreLifecycleHooks,
) -> ScoreLifecycleResult:
    """Execute one reviewed held-in score through a typed receipt profile.

    This generic loop deliberately does not know any producer SHA, target
    session, model, or receipt schema.  Those are sealed in ``hooks`` before
    the held artifact root is reserved.  It is the only lifecycle loop used by
    the V1 wrapper and later same-surface successors.
    """
    _require(isinstance(hooks, ProfiledHeldInScoreLifecycleHooks),
             "CS-WG held-in score profiled lifecycle hooks are absent")
    graph = hooks.pre_reserve(Path(root), identity, capability, backend)
    spec = hooks.spec_for_identity(identity)
    _require(isinstance(getattr(spec, "root_relative", None), str),
             "CS-WG held-in score profiled root spec drift")
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), spec)  # type: ignore[arg-type]
    attempt_payload = dict(hooks.attempt_payload(identity, graph))
    attempt_sha256 = artifact.publish_json("attempt.json", attempt_payload)
    launch_sha256: str | None = None
    input_sha256: str | None = None
    score_sha256: str | None = None
    terminal_sha256: str | None = None
    try:
        launch_method = getattr(backend, "launch_payload", None)
        prepare_method = getattr(backend, "prepare_inputs", None)
        score_method = getattr(backend, "score", None)
        _require(callable(launch_method) and callable(prepare_method) and callable(score_method),
                 "CS-WG held-in score profiled backend lifecycle seam drift")
        launch_backend = launch_method(identity, graph)
        _require(isinstance(launch_backend, Mapping), "CS-WG held-in score launch backend payload drift")
        launch_sha256 = artifact.publish_json(
            "launch.json", dict(hooks.launch_payload(identity, graph, attempt_sha256, launch_backend)),
        )
        prepared = prepare_method(identity, graph)
        _require(isinstance(prepared, Mapping), "CS-WG held-in score prepared input payload type drift")
        inputs = dict(hooks.input_authority_payload(identity, graph, prepared))
        input_sha256 = artifact.publish_json("input_authority.json", inputs)
        raw_score = score_method(identity, graph, inputs)
        _require(isinstance(raw_score, Mapping), "CS-WG held-in score physical payload type drift")
        body = dict(raw_score)
        body["input_authority"] = inputs
        hooks.validate_score_payload(body, identity, graph, input_sha256)
        score_sha256 = artifact.publish_json("score.json", body)
        checked_score = hooks.revalidate_before_terminal(
            Path(root), identity, capability, graph, artifact, attempt_sha256, launch_sha256,
            input_sha256, score_sha256,
        )
        terminal = dict(hooks.terminal_payload(
            identity, graph, attempt_sha256, launch_sha256, input_sha256, score_sha256, checked_score,
        ))
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=hooks.success_names(True))
        _require(artifact.read_json_pair("terminal.json", expected_sha256=terminal_sha256) == terminal,
                 "CS-WG held-in score terminal descriptor reload drift")
        return ScoreLifecycleResult(artifact.root_identity, attempt_sha256, launch_sha256, input_sha256,
                                    score_sha256, terminal_sha256, None)
    except BaseException as error:
        if terminal_sha256 is not None:
            raise
        try:
            progress_method = getattr(backend, "progress", None)
            progress = progress_method() if callable(progress_method) else None
            _require(isinstance(progress, ScoreProgress), "CS-WG held-in score backend progress type drift")
        except BaseException:
            progress = ScoreProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json",
            dict(hooks.failure_payload(
                identity, graph, attempt_sha256, launch_sha256, input_sha256, score_sha256, progress, error,
            )),
        )
        return ScoreLifecycleResult(artifact.root_identity, attempt_sha256, launch_sha256, input_sha256,
                                    score_sha256, None, failure_sha256)
    finally:
        try:
            close_method = getattr(backend, "close", None)
            _require(callable(close_method), "CS-WG held-in score backend close seam drift")
            close_method()
        finally:
            artifact.close()


def execute_reviewed_heldin_score(
    root: Path, *, identity: plan.ScoreIdentity, capability: object, backend: DeferredHeldInScoreBackend,
    graph_loader: Callable[[Path], CompletedFullGraph] = load_completed_full_graph,
) -> ScoreLifecycleResult:
    """Root-only V1 wrapper over the shared profiled lifecycle loop."""

    def pre_reserve(
        checked_root: Path, observed_identity: object, observed_capability: object, observed_backend: object,
    ) -> CompletedFullGraph:
        _require(isinstance(observed_identity, plan.ScoreIdentity),
                 "CS-WG held-in score V1 identity type drift")
        cap = _require_capability(observed_capability, observed_identity)
        validate_identity_current(checked_root, observed_identity)
        graph = graph_loader(checked_root)
        _require(graph.expectation == observed_identity.completed_full
                 and graph.sha256 == cap.completed_full_graph_sha256,
                 "CS-WG held-in score capability/completed full graph drift before reserve")
        backend_source_root = Path(getattr(observed_backend, "source_root", ""))
        _require(backend_source_root.is_absolute() and str(backend_source_root) == cap.target_source_root,
                 "CS-WG held-in score capability/backend target source-root drift before reserve")
        assert_prospective_score_root_fresh(checked_root, observed_identity.spec)
        return graph

    def revalidate_before_terminal(
        checked_root: Path, observed_identity: object, observed_capability: object, graph: object,
        artifact: v1.ImmutableArtifactRoot, attempt_sha: str, launch_sha: str, input_sha: str, score_sha: str,
    ) -> Mapping[str, object]:
        _require(isinstance(observed_identity, plan.ScoreIdentity)
                 and isinstance(graph, CompletedFullGraph),
                 "CS-WG held-in score V1 final identity/graph type drift")
        cap = _require_capability(observed_capability, observed_identity)
        validate_identity_current(checked_root, observed_identity)
        graph_now = graph_loader(checked_root)
        _require(graph_now.sha256 == graph.sha256 == cap.completed_full_graph_sha256,
                 "CS-WG held-in score immutable completed full graph drifted during evaluation")
        _inputs, checked_score = _revalidate_published_score_graph(
            artifact, observed_identity, graph, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
            input_authority_sha256=input_sha, score_sha256=score_sha,
        )
        return checked_score

    hooks = ProfiledHeldInScoreLifecycleHooks(
        label="historical_v1_default",
        pre_reserve=pre_reserve,
        spec_for_identity=lambda observed: observed.spec if isinstance(observed, plan.ScoreIdentity) else None,
        attempt_payload=lambda observed, graph: _attempt_payload(observed, graph),
        launch_payload=lambda observed, graph, attempt_sha, backend_payload: _launch_payload(
            observed, graph, attempt_sha, backend_payload,
        ),
        input_authority_payload=lambda observed, graph, prepared: _input_authority_payload(observed, graph, prepared),
        validate_score_payload=lambda value, observed, graph, input_sha: _validate_score_payload(
            value, observed, graph, input_sha,
        ),
        revalidate_before_terminal=revalidate_before_terminal,
        terminal_payload=lambda observed, graph, attempt_sha, launch_sha, input_sha, score_sha, checked: _terminal_payload(
            observed, graph, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
            input_authority_sha256=input_sha, score_sha256=score_sha, score_payload=checked,
        ),
        failure_payload=lambda observed, graph, attempt_sha, launch_sha, input_sha, score_sha, progress, error: _failure_payload(
            observed, graph, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
            input_authority_sha256=input_sha, score_sha256=score_sha, progress=progress, error=error,
        ),
        success_names=lambda terminal: _success_names(terminal=terminal),
    )
    return execute_profiled_heldin_score(
        Path(root), identity=identity, capability=capability, backend=backend, hooks=hooks,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    return plan.dry_plan(root)


__all__ = (
    "HeldInScoreError", "CompletedFullGraph", "ScoreCapability", "ScoreProgress", "ScoreLifecycleResult",
    "DeferredHeldInScoreBackend", "ProfiledHeldInScoreLifecycleHooks", "load_completed_full_graph", "validate_identity_current",
    "read_selected_checkpoint_bytes",
    "assert_prospective_score_root_fresh", "issue_root_reviewed_score_capability",
    "execute_profiled_heldin_score", "execute_reviewed_heldin_score", "dry_plan",
)
