"""Seed-43 Phase-E replication addendum for TF-SR.

This module deliberately has only standard-library imports at module import
time.  The public CLI can therefore render :func:`dry_plan` without importing
Torch, resolving an evaluation asset, opening CUDA, or reserving an output
directory.  The live path is intentionally narrow: it reuses the immutable
seed-42 Phase-E *input* implementation through a subclass of its physical
backend, but validates and loads only the seed-43 training chain for the
successor model.

The addendum is not a second discovery gate.  Its sole scientific role is a
build-labelled replication after the seed-42 Phase-E terminal is immutable.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


CELL = "TFSR_B3ST4_DDROP_SEED43"
PHASE = "TFSR_PHASE_E_SEED43_REPLICATION_ADDENDUM_V1"
SEED = 43
BUILD_LABEL = "v2_accelerated_jit_scripted_step"
# This label is not inferred from the live seed-43 result.  It identifies the
# immutable eager build that produced the frozen seed-42 Phase-E authority
# which the addendum reuses only as an input/Cell-D replay anchor.
SEED42_BUILD_LABEL = "v1_eager"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_TFSR_SEED43_PHASE_E_SUCCESSOR_20260821.md"
WORKORDER_SHA256 = "73ccb4d95eae65581584fc77f2d72b13082aad235921a568feba79b424123bf1"
SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/score_43.py"
CLI_RELATIVE = "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed43_score_v1.py"
SEED43_INIT_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/__init__.py"
SEED43_CONTRACT_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/contract_43.py"
SEED43_TRAIN_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/train_43.py"
SEED43_ACCELERATED_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/accelerated_forward.py"
# ``train_43`` deliberately reuses this exact frozen generic artifact-root
# implementation.  It does not re-export the private directory helper, so
# consumers must bind this explicit dependency instead of inventing a local
# look-alike or accepting an alias from the live seed-43 module.
FROZEN_TRAIN42_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/train.py"
SEED43_TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_train_v1"
SEED43_AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_score_authority_v1"
SEED43_SCORE_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_matched_score_v1"

SCORE_TOPOLOGY = (
    "attempt.json",
    "input_replay.json",
    "score.json",
    "terminal.json",
    "failure.json",
)
AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")
PUBLIC_FLAGS = frozenset(("--execute", "--i-have-seed43-phase-e-replication-authorization"))
PUBLIC_SCORE_SPEC = {
    "within_count": 6,
    "external_count": 15,
    "window_bins": 50,
    "calibration_trials": 30,
    "batch_size": 128,
    "bootstrap_draws": 10_000,
    "bootstrap_seed": 42,
}


class FailClosedError(RuntimeError):
    """A provenance, boundary, or score-integrity violation."""


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _json_copy(value: object) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _require_sha(value: object, label: str) -> str:
    if not _is_sha(value):
        raise FailClosedError(f"{label} must be an exact lowercase SHA-256")
    return str(value)


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise FailClosedError("path must be a canonical safe relative path")
    return value


def _seed42_score() -> Any:
    """Import the frozen scorer only after a non-static route requests it."""
    from src.tfsr_b3st4_ddrop_v1 import score

    return score


def _seed43_train() -> Any:
    from . import train_43

    return train_43


def _seed43_contract() -> Any:
    from . import contract_43

    return contract_43


@dataclass(frozen=True)
class Seed43TrainingEvidence:
    """The seed-43-only immutable terminal/SWA chain consumed by this addendum."""

    terminal_sha256: str
    swa_sha256: str
    swa_state_digest: str
    checkpoint_sha256: Mapping[str, str]
    launch_closure: Mapping[str, object]
    lineage: Mapping[str, object]
    build_disclosure: Mapping[str, object]
    training_peak_memory_bytes: int
    cell: str = CELL

    def __post_init__(self) -> None:
        for label, value in (
            ("seed43 terminal", self.terminal_sha256),
            ("seed43 SWA", self.swa_sha256),
            ("seed43 SWA state", self.swa_state_digest),
        ):
            _require_sha(value, label)
        if self.cell != CELL:
            raise ValueError("seed42 or foreign training evidence cannot satisfy seed43 validation")
        if (not isinstance(self.checkpoint_sha256, Mapping) or set(self.checkpoint_sha256) != {"44", "45", "46", "47"}
                or not all(_is_sha(item) for item in self.checkpoint_sha256.values())):
            raise ValueError("seed43 final-four checkpoint binding drift")
        if (not isinstance(self.launch_closure, Mapping) or not isinstance(self.lineage, Mapping)
                or not isinstance(self.build_disclosure, Mapping)
                or type(self.training_peak_memory_bytes) is not int or self.training_peak_memory_bytes < 0):
            raise ValueError("seed43 training evidence schema drift")
        if self.build_disclosure.get("build") != BUILD_LABEL:
            raise ValueError("seed43 accelerated build disclosure drift")

    def payload(self) -> dict[str, object]:
        return {
            "cell": CELL,
            "terminal_sha256": self.terminal_sha256,
            "swa_sha256": self.swa_sha256,
            "swa_state_digest": self.swa_state_digest,
            "checkpoint_sha256": dict(self.checkpoint_sha256),
            "launch_closure": _json_copy(self.launch_closure),
            "lineage": _json_copy(self.lineage),
            "build_disclosure": _json_copy(self.build_disclosure),
            "training_peak_memory_bytes": self.training_peak_memory_bytes,
        }


@dataclass(frozen=True)
class UpstreamSeed42Evidence:
    """Descriptor-validated seed-42 Phase-E terminal and replay inputs."""

    input_authority_sha256: str
    score_sha256: str
    terminal_sha256: str
    verdict: str
    input_payload: Mapping[str, object]
    score_payload: Mapping[str, object]
    terminal_payload: Mapping[str, object]
    build_label: str = SEED42_BUILD_LABEL

    def __post_init__(self) -> None:
        for label, value in (
            ("upstream input authority", self.input_authority_sha256),
            ("upstream score", self.score_sha256),
            ("upstream terminal", self.terminal_sha256),
        ):
            _require_sha(value, label)
        if self.verdict not in {"CLEAR_GO", "HOLD", "STOP"}:
            raise ValueError("upstream seed42 verdict drift")
        if self.build_label != SEED42_BUILD_LABEL:
            raise ValueError("upstream seed42 build-label drift")
        if not all(isinstance(item, Mapping) for item in (self.input_payload, self.score_payload, self.terminal_payload)):
            raise ValueError("upstream seed42 payload maps required")
        if (
            _sha(_json_bytes(self.input_payload)) != self.input_authority_sha256
            or _sha(_json_bytes(self.score_payload)) != self.score_sha256
            or _sha(_json_bytes(self.terminal_payload)) != self.terminal_sha256
        ):
            raise ValueError("upstream seed42 payload/body SHA binding drift")

    def binding_payload(self) -> dict[str, object]:
        return {
            "seed42_input_authority_sha256": self.input_authority_sha256,
            "seed42_score_sha256": self.score_sha256,
            "seed42_terminal_sha256": self.terminal_sha256,
            "seed42_verdict": self.verdict,
            "seed42_cell": "TFSR_B3ST4_DDROP_SEED42",
            "seed42_build_label": self.build_label,
            "cell_d_replay_required": True,
        }


@dataclass(frozen=True)
class Seed43CompletedScoreEvidence:
    """Descriptor-validated terminal score pair for a completed seed-43 run.

    This object deliberately retains the exact immutable body SHAs rather
    than accepting a caller's summary statistic.  It is consumed only by the
    non-gating two-seed descriptive report.
    """

    score_sha256: str
    terminal_sha256: str
    score_payload: Mapping[str, object]
    terminal_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_sha(self.score_sha256, "seed43 completed score")
        _require_sha(self.terminal_sha256, "seed43 completed terminal")
        if not isinstance(self.score_payload, Mapping) or not isinstance(self.terminal_payload, Mapping):
            raise ValueError("seed43 completed score/terminal payload maps required")
        if (self.score_payload.get("cell") != CELL
                or self.score_payload.get("status") != "REPLICATION_SCORE_COMPLETE"
                or self.score_payload.get("build") != BUILD_LABEL
                or self.terminal_payload.get("cell") != CELL
                or self.terminal_payload.get("status") != "REPLICATION_SCORE_COMPLETE"):
            raise ValueError("seed43 completed score/terminal identity or build drift")
        if (
            _sha(_json_bytes(self.score_payload)) != self.score_sha256
            or _sha(_json_bytes(self.terminal_payload)) != self.terminal_sha256
        ):
            raise ValueError("seed43 completed score/terminal body SHA binding drift")

    def binding_payload(self) -> dict[str, object]:
        return {
            "seed": SEED,
            "build_label": BUILD_LABEL,
            "score_sha256": self.score_sha256,
            "terminal_sha256": self.terminal_sha256,
        }


@dataclass(frozen=True)
class Seed43ScoreIdentity:
    training: Seed43TrainingEvidence
    upstream: UpstreamSeed42Evidence
    closure: Mapping[str, object]
    authorization: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.closure, Mapping) or not isinstance(self.authorization, Mapping):
            raise ValueError("seed43 score closure/authorization maps required")

    def payload(self) -> dict[str, object]:
        return {
            "seed43_training": self.training.payload(),
            "upstream_seed42": self.upstream.binding_payload(),
            "addendum_closure": _json_copy(self.closure),
            "authorization": _json_copy(self.authorization),
        }


@dataclass(frozen=True)
class Seed43Authorization:
    preflight_sha256: str
    root_authorization_sha256: str
    preflight: Mapping[str, object]
    root_authorization: Mapping[str, object]
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_sha(self.preflight_sha256, "seed43 preflight")
        _require_sha(self.root_authorization_sha256, "seed43 root authorization")
        if not all(isinstance(item, Mapping) for item in (self.preflight, self.root_authorization, self.closure)):
            raise ValueError("seed43 authorization maps required")

    def payload(self) -> dict[str, object]:
        return {
            "preflight_sha256": self.preflight_sha256,
            "root_authorization_sha256": self.root_authorization_sha256,
            "closure": _json_copy(self.closure),
        }


_ROOT_SEAL = object()
_EXECUTION_SEAL = object()


class RootPublicationCapability:
    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_SEAL:
            raise TypeError("seed43 authority publication requires the root capability")
        self._seal = seal


class ExecutionCapability:
    __slots__ = ("authorization", "_seal")

    def __init__(self, authorization: Seed43Authorization, seal: object) -> None:
        if seal is not _EXECUTION_SEAL:
            raise TypeError("seed43 scoring capability cannot be constructed publicly")
        self.authorization = authorization
        self._seal = seal


def issue_root_publication_capability() -> RootPublicationCapability:
    """Root-reviewed callers may mint an authority-pair publication capability."""
    return RootPublicationCapability(_ROOT_SEAL)


def _issue_execution_capability(authorization: Seed43Authorization) -> ExecutionCapability:
    return ExecutionCapability(authorization, _EXECUTION_SEAL)


def _require_execution_capability(capability: object, identity: Seed43ScoreIdentity) -> ExecutionCapability:
    if not isinstance(capability, ExecutionCapability) or capability._seal is not _EXECUTION_SEAL:
        raise FailClosedError("root-reviewed in-process seed43 capability required before target resolution")
    if capability.authorization.payload() != identity.authorization:
        raise FailClosedError("seed43 capability/identity authorization drift")
    return capability


def _addendum_closure_paths(s42: Any) -> tuple[str, ...]:
    """Explicit non-glob closure for direct and inherited runtime code."""
    paths = (
        WORKORDER_RELATIVE,
        SOURCE_RELATIVE,
        CLI_RELATIVE,
        TEST_RELATIVE,
        SEED43_INIT_RELATIVE,
        SEED43_CONTRACT_RELATIVE,
        SEED43_TRAIN_RELATIVE,
        SEED43_ACCELERATED_RELATIVE,
        FROZEN_TRAIN42_RELATIVE,
        "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark_v2.py",
        # ``train_43.validate_epoch_receipt`` calls this public schedule
        # authority at score preflight and SWA/terminal revalidation time.
        # It is a live dependency of this addendum, not merely a dependency of
        # the already-terminal training receipt.
        "tfpd_exploration/src/tfpd_lane/arm_common.py",
        *tuple(s42.PHASE_E_CLOSURE),
    )
    return tuple(dict.fromkeys(paths))


def addendum_closure(root: Path, *, s42: Any | None = None) -> dict[str, object]:
    """Descriptor-hash the complete explicit seed43 addendum closure."""
    s42 = _seed42_score() if s42 is None else s42
    paths = _addendum_closure_paths(s42)
    hashes: dict[str, str] = {}
    for relative in paths:
        body, _identity = s42._canonical_regular_bytes(root.absolute(), relative, expected_mode=0o664)
        hashes[relative] = _sha(body)
    if hashes.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise FailClosedError("seed43 addendum work-order SHA drift")
    encoded = json.dumps({"paths": list(paths), "sha256_by_path": hashes}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"paths": list(paths), "sha256_by_path": hashes, "closure_sha256": _sha(encoded)}


def validate_addendum_closure(value: object, *, s42: Any | None = None) -> dict[str, object]:
    s42 = _seed42_score() if s42 is None else s42
    paths = _addendum_closure_paths(s42)
    if not isinstance(value, Mapping) or set(value) != {"paths", "sha256_by_path", "closure_sha256"}:
        raise FailClosedError("seed43 addendum closure schema drift")
    hashes = value.get("sha256_by_path")
    if value.get("paths") != list(paths) or not isinstance(hashes, Mapping) or set(hashes) != set(paths):
        raise FailClosedError("seed43 addendum closure path map drift")
    if any(not _is_sha(hashes.get(path)) for path in paths) or hashes.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise FailClosedError("seed43 addendum closure file SHA drift")
    encoded = json.dumps({"paths": list(paths), "sha256_by_path": dict(hashes)}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if value.get("closure_sha256") != _sha(encoded):
        raise FailClosedError("seed43 addendum closure aggregate drift")
    return {"paths": list(paths), "sha256_by_path": {path: str(hashes[path]) for path in paths}, "closure_sha256": str(value["closure_sha256"])}


def canonical_authority_parent(root: Path) -> tuple[Path, str]:
    relative = _safe_relative(SEED43_AUTHORITY_ROOT_RELATIVE)
    value = root.absolute() / relative
    return value.parent, value.name


def canonical_score_parent(root: Path) -> tuple[Path, str]:
    relative = _safe_relative(SEED43_SCORE_ROOT_RELATIVE)
    value = root.absolute() / relative
    return value.parent, value.name


def _require_seed43_output_names() -> None:
    s42 = _seed42_score()
    forbidden = {s42.SCORE_ROOT_RELATIVE, s42.AUTHORITY_ROOT_RELATIVE}
    if ({SEED43_SCORE_ROOT_RELATIVE, SEED43_AUTHORITY_ROOT_RELATIVE} & forbidden
            or SEED43_SCORE_ROOT_RELATIVE == SEED43_AUTHORITY_ROOT_RELATIVE):
        raise FailClosedError("seed43 addendum output root aliases seed42 authority/result root")


def reserve_authority_artifact(root: Path, capability: RootPublicationCapability) -> Any:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may reserve seed43 authority root")
    _require_seed43_output_names()
    s42 = _seed42_score()
    parent, name = canonical_authority_parent(root)
    return s42.reserve_artifact_root(parent, name, AUTHORITY_TOPOLOGY)


def reserve_score_artifact(root: Path) -> Any:
    _require_seed43_output_names()
    s42 = _seed42_score()
    parent, name = canonical_score_parent(root)
    return s42.reserve_artifact_root(parent, name, SCORE_TOPOLOGY)


def _training_evidence_from_terminal(
    *,
    train: Any,
    terminal: Mapping[str, object],
    terminal_sha256: str,
    checkpoint_sha256: Mapping[str, str],
    peak_memory: int,
) -> Seed43TrainingEvidence:
    build = terminal.get("build_disclosure")
    lineage = terminal.get("lineage")
    closure = terminal.get("launch_closure")
    if (not isinstance(build, Mapping) or not isinstance(lineage, Mapping) or not isinstance(closure, Mapping)
            or terminal.get("cell") != CELL or build.get("build") != BUILD_LABEL):
        raise FailClosedError("seed43 terminal identity/build disclosure drift")
    return Seed43TrainingEvidence(
        terminal_sha256=terminal_sha256,
        swa_sha256=str(terminal.get("swa_sha256")),
        swa_state_digest=str(terminal.get("swa_state_digest")),
        checkpoint_sha256={str(key): str(value) for key, value in checkpoint_sha256.items()},
        launch_closure=_json_copy(closure),
        lineage=_json_copy(lineage),
        build_disclosure=_json_copy(build),
        training_peak_memory_bytes=peak_memory,
    )


def _frozen_train42_directory_identity(train: Any) -> Callable[[Path], tuple[int, int]]:
    """Return the closure-bound generic artifact identity primitive.

    ``train_43`` intentionally shares ``ArtifactRoot`` with the sealed
    seed-42 lifecycle, but it has no private ``_directory_identity`` alias of
    its own.  Import the exact frozen dependency locally, then prove that the
    supplied seed-43 train module still points at that same module and its
    shared artifact implementation.  This is deliberately not a fallback to
    a caller-provided helper: an alias or monkeypatch must fail before any
    artifact body is opened.
    """
    from src.tfsr_b3st4_ddrop_v1 import train as frozen_train42

    identity = frozen_train42._directory_identity
    if (
        getattr(train, "train42", None) is not frozen_train42
        or getattr(train, "ArtifactRoot", None) is not frozen_train42.ArtifactRoot
        or getattr(train, "_sha", None) is not frozen_train42._sha
        or not callable(identity)
    ):
        raise FailClosedError("seed43 train/frozen train42 artifact-identity binding drift")
    return identity


def _attach_readonly_seed43_artifact(root: Path, train: Any) -> Any:
    directory = root.absolute() / SEED43_TRAIN_ROOT_RELATIVE
    if not directory.exists() or directory.is_symlink():
        raise FailClosedError("canonical seed43 training root is absent or aliased")
    parent = directory.parent
    try:
        directory_identity = _frozen_train42_directory_identity(train)
        identity = directory_identity(directory)
        parent_identity = directory_identity(parent)
    except Exception as error:
        raise FailClosedError("seed43 training artifact root identity drift") from error
    return train.ArtifactRoot(directory, train.PUBLIC_SPEC.topology, identity, parent, parent_identity)


def validate_seed43_training_terminal(root: Path) -> Seed43TrainingEvidence:
    """Reload the seed43-native terminal chain before any target route exists."""
    train = _seed43_train()
    contract = _seed43_contract()
    try:
        contract.verify_frozen_route(root)
        throughput_v2 = contract.verify_throughput_v2_receipt(root)
        if throughput_v2.get("conclusion_fastest_kind") != "jit_scripted_step":
            raise RuntimeError("throughput-v2 build evidence drift")
        artifact = _attach_readonly_seed43_artifact(root, train)
        identity = train.production_identity(root)
        terminal_body = artifact.reload_pair("terminal.json")
        terminal_sha256 = _sha(terminal_body)
        terminal = json.loads(terminal_body)
        if not isinstance(terminal, Mapping):
            raise RuntimeError("seed43 terminal root is not a map")
        if (terminal.get("schema") != "tfsr_b3st4_ddrop_seed43_train_terminal_v1"
                or terminal.get("status") != "TRAINING_COMPLETE"
                or terminal.get("cell") != CELL
                or terminal.get("run_spec") != train.PUBLIC_SPEC.payload()
                or terminal.get("epochs") != train.PUBLIC_SPEC.epochs
                or terminal.get("steps_per_epoch") != train.PUBLIC_SPEC.steps_per_epoch
                or terminal.get("total_optimizer_steps") != train.PUBLIC_SPEC.total_optimizer_steps):
            raise RuntimeError("seed43 terminal header/accounting drift")
        train.validate_terminal_receipt(terminal, train.PUBLIC_SPEC, identity)
        if terminal.get("launch_closure") != terminal.get("final_closure"):
            raise RuntimeError("seed43 launch/final closure mismatch")
        if terminal.get("build_disclosure") != contract.validate_build_disclosure(contract.BUILD_DISCLOSURE):
            raise RuntimeError("seed43 terminal accelerated-build identity drift")
        if terminal.get("boundaries") != {
            "source_only": True, "target_or_formal_opened": False,
            "scientific_result": False, "score": False, "capture_diagnostics": False,
        }:
            raise RuntimeError("seed43 terminal source-only boundary drift")
        attempt = artifact.reload_json("attempt.json", terminal["attempt_sha256"])
        train.validate_attempt_receipt(attempt, train.PUBLIC_SPEC, identity)
        launch = artifact.reload_json("launch.json", terminal["launch_sha256"])
        train.validate_launch_receipt(launch, train.PUBLIC_SPEC, identity)
        throughput_name = f"throughput{train.PUBLIC_SPEC.throughput_probe_steps}.json"
        throughput = artifact.reload_json(throughput_name, terminal["throughput_sha256"])
        train.validate_throughput_receipt(throughput, train.PUBLIC_SPEC, identity)
        epoch_shas = terminal.get("epoch_receipt_sha256")
        if not isinstance(epoch_shas, list) or len(epoch_shas) != train.PUBLIC_SPEC.epochs:
            raise RuntimeError("seed43 epoch receipt cardinality drift")
        peak_memory = 0
        for epoch, digest in enumerate(epoch_shas):
            epoch_payload = artifact.reload_json(f"epoch-{epoch:02d}.json", digest)
            train.validate_epoch_receipt(
                epoch_payload, epoch, (epoch + 1) * train.PUBLIC_SPEC.steps_per_epoch,
                train.PUBLIC_SPEC, identity,
            )
            resources = epoch_payload.get("resources") if isinstance(epoch_payload, Mapping) else None
            if isinstance(resources, Mapping) and type(resources.get("peak_allocated_bytes")) is int:
                peak_memory = max(peak_memory, int(resources["peak_allocated_bytes"]))
        checkpoint_sha256 = terminal.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, Mapping) or set(checkpoint_sha256) != {"44", "45", "46", "47"}:
            raise RuntimeError("seed43 terminal final-four checkpoint map drift")
        expected_binding = {
            "cell": train.CELL,
            "run_spec": train.PUBLIC_SPEC.payload(),
            "launch_sha256": terminal["launch_sha256"],
            "launch_closure": terminal["launch_closure"],
            "lineage": terminal["lineage"],
        }
        backend = train.TrainingBackend43(root)
        for epoch in train.PUBLIC_SPEC.checkpoint_epochs:
            body = artifact.reload_pair(f"checkpoint-{epoch:02d}.pt", checkpoint_sha256[str(epoch)])
            backend.validate_checkpoint(
                body, epoch, (epoch + 1) * train.PUBLIC_SPEC.steps_per_epoch,
                train.PUBLIC_SPEC, expected_binding=expected_binding,
            )
        swa_body = artifact.reload_pair("swa.pt", terminal["swa_sha256"])
        swa = backend.validate_swa(swa_body, train.PUBLIC_SPEC, expected_binding=expected_binding)
        if swa.get("state_digest") != terminal.get("swa_state_digest"):
            raise RuntimeError("seed43 terminal/SWA state digest drift")
        if artifact.has_name("failure.json"):
            raise RuntimeError("seed43 terminal cannot coexist with failed training receipt")
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise FailClosedError("seed43 terminal/SWA chain invalid before score input resolution") from error
    return _training_evidence_from_terminal(
        train=train, terminal=terminal, terminal_sha256=terminal_sha256,
        checkpoint_sha256=checkpoint_sha256, peak_memory=peak_memory,
    )


def _seed43_swa_state_at_use(root: Path, evidence: Seed43TrainingEvidence) -> Mapping[str, object]:
    """Revalidate native seed43 SWA at model-load time; never touch seed42 SWA."""
    train = _seed43_train()
    artifact = _attach_readonly_seed43_artifact(root, train)
    try:
        terminal_body = artifact.reload_pair("terminal.json", evidence.terminal_sha256)
        terminal = json.loads(terminal_body)
        if not isinstance(terminal, Mapping):
            raise RuntimeError("seed43 terminal JSON root drift")
        identity = train.production_identity(root)
        train.validate_terminal_receipt(terminal, train.PUBLIC_SPEC, identity)
        if (terminal.get("swa_sha256") != evidence.swa_sha256
                or terminal.get("swa_state_digest") != evidence.swa_state_digest
                or terminal.get("launch_closure") != evidence.launch_closure
                or terminal.get("lineage") != evidence.lineage
                or terminal.get("build_disclosure") != evidence.build_disclosure):
            raise RuntimeError("seed43 terminal changed after authority validation")
        expected_binding = {
            "cell": train.CELL,
            "run_spec": train.PUBLIC_SPEC.payload(),
            "launch_sha256": terminal["launch_sha256"],
            "launch_closure": terminal["launch_closure"],
            "lineage": terminal["lineage"],
        }
        body = artifact.reload_pair("swa.pt", evidence.swa_sha256)
        swa = train.TrainingBackend43(root).validate_swa(body, train.PUBLIC_SPEC, expected_binding=expected_binding)
        if swa.get("state_digest") != evidence.swa_state_digest or not isinstance(swa.get("state"), Mapping):
            raise RuntimeError("seed43 SWA state at use drift")
        return swa["state"]
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise FailClosedError("seed43 SWA cannot be loaded at physical score point") from error


def _upstream_identity_from_terminal(s42: Any, terminal: Mapping[str, object]) -> Any:
    value = terminal.get("identity")
    required = {
        "fixed_authorities", "training_terminal_sha256", "training_swa_sha256",
        "training_swa_state_digest", "launch_closure", "phase_e_authorization",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FailClosedError("upstream seed42 terminal identity schema drift")
    try:
        return s42.ScoreIdentity(
            fixed_authorities=value["fixed_authorities"],
            training_terminal_sha256=value["training_terminal_sha256"],
            training_swa_sha256=value["training_swa_sha256"],
            training_swa_state_digest=value["training_swa_state_digest"],
            launch_closure=value["launch_closure"],
            phase_e_authorization=value["phase_e_authorization"],
        )
    except (TypeError, ValueError) as error:
        raise FailClosedError("upstream seed42 terminal identity cannot be reconstructed") from error


def validate_upstream_seed42_payloads(
    s42: Any,
    *,
    input_payload: Mapping[str, object],
    input_sha256: str,
    score_payload: Mapping[str, object],
    score_sha256: str,
    terminal_payload: Mapping[str, object],
    terminal_sha256: str,
) -> UpstreamSeed42Evidence:
    """Use the frozen seed42 validators; never reinterpret their receipt schema."""
    _require_sha(input_sha256, "upstream seed42 input authority")
    _require_sha(score_sha256, "upstream seed42 score")
    _require_sha(terminal_sha256, "upstream seed42 terminal")
    if (
        _sha(_json_bytes(input_payload)) != input_sha256
        or _sha(_json_bytes(score_payload)) != score_sha256
        or _sha(_json_bytes(terminal_payload)) != terminal_sha256
    ):
        raise FailClosedError("upstream seed42 descriptor payload/body SHA binding drift")
    identity = _upstream_identity_from_terminal(s42, terminal_payload)
    try:
        s42.validate_input_authority_payload(input_payload, identity)
        s42.validate_score_payload(score_payload, identity=identity, input_authority_sha256=input_sha256)
        s42.validate_terminal_payload(terminal_payload, identity, score_payload, expected_score_sha=score_sha256)
    except Exception as error:
        raise FailClosedError("upstream seed42 Phase-E input/score/terminal binding drift") from error
    verdict = terminal_payload.get("verdict")
    if verdict not in {"CLEAR_GO", "HOLD", "STOP"}:
        raise FailClosedError("upstream seed42 terminal verdict drift")
    return UpstreamSeed42Evidence(
        input_authority_sha256=input_sha256,
        score_sha256=score_sha256,
        terminal_sha256=terminal_sha256,
        verdict=str(verdict),
        input_payload=_json_copy(input_payload),
        score_payload=_json_copy(score_payload),
        terminal_payload=_json_copy(terminal_payload),
    )


def load_upstream_seed42_phase_e(root: Path) -> UpstreamSeed42Evidence:
    """Descriptor-load the immutable seed42 Phase-E terminal before data access."""
    s42 = _seed42_score()
    directory = root.absolute() / s42.SCORE_ROOT_RELATIVE
    try:
        input_payload, input_sha = s42._read_authority_pair(directory, "input_authority.json")
        score_payload, score_sha = s42._read_authority_pair(directory, "score.json")
        terminal_payload, terminal_sha = s42._read_authority_pair(directory, "terminal.json")
    except Exception as error:
        raise FailClosedError("seed43 replication requires descriptor-valid seed42 Phase-E terminal first") from error
    return validate_upstream_seed42_payloads(
        s42,
        input_payload=input_payload, input_sha256=input_sha,
        score_payload=score_payload, score_sha256=score_sha,
        terminal_payload=terminal_payload, terminal_sha256=terminal_sha,
    )


def _validate_upstream_binding(value: object, upstream: UpstreamSeed42Evidence) -> dict[str, object]:
    expected = upstream.binding_payload()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise FailClosedError("seed43 authority upstream seed42 binding drift")
    return expected


def build_target_free_preflight(
    *,
    training: Seed43TrainingEvidence,
    upstream: UpstreamSeed42Evidence,
    closure: Mapping[str, object],
) -> dict[str, object]:
    """Build a target-free addendum authority; no caller can choose assets."""
    s42 = _seed42_score()
    checked_closure = validate_addendum_closure(closure, s42=s42)
    return {
        "schema": "tfsr_seed43_phase_e_replication_preflight_v1",
        "status": "PREFLIGHT_ACCEPTED",
        "cell": CELL,
        "phase": PHASE,
        "seed": SEED,
        "score_spec": dict(PUBLIC_SCORE_SPEC),
        "seed43_training": training.payload(),
        "upstream_seed42": upstream.binding_payload(),
        "addendum_closure": checked_closure,
        "authority_root_relative": SEED43_AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SEED43_SCORE_ROOT_RELATIVE,
        "target_free": True,
        "formal_sessions_inert": list(s42.FORMAL_TEST_SESSION_NAMES),
        "metric_from_upstream": {
            "governing": "last_bin_variance_weighted_r2_equal_session",
            "cell_d_replay": "upstream_seed42_exact_only_no_rerun",
            "modes": ["aligned", "zero", "wrong_pair"],
        },
    }


def validate_target_free_preflight(
    value: Mapping[str, object], *, training: Seed43TrainingEvidence, upstream: UpstreamSeed42Evidence,
) -> dict[str, object]:
    s42 = _seed42_score()
    expected = {
        "schema", "status", "cell", "phase", "seed", "score_spec", "seed43_training", "upstream_seed42",
        "addendum_closure", "authority_root_relative", "score_root_relative", "target_free",
        "formal_sessions_inert", "metric_from_upstream",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_seed43_phase_e_replication_preflight_v1"
            or value.get("status") != "PREFLIGHT_ACCEPTED" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("seed") != SEED
            or value.get("score_spec") != PUBLIC_SCORE_SPEC
            or value.get("seed43_training") != training.payload()
            or value.get("authority_root_relative") != SEED43_AUTHORITY_ROOT_RELATIVE
            or value.get("score_root_relative") != SEED43_SCORE_ROOT_RELATIVE
            or value.get("target_free") is not True
            or value.get("formal_sessions_inert") != list(s42.FORMAL_TEST_SESSION_NAMES)
            or value.get("metric_from_upstream") != {
                "governing": "last_bin_variance_weighted_r2_equal_session",
                "cell_d_replay": "upstream_seed42_exact_only_no_rerun",
                "modes": ["aligned", "zero", "wrong_pair"],
            }):
        raise FailClosedError("seed43 target-free preflight schema/binding drift")
    _validate_upstream_binding(value.get("upstream_seed42"), upstream)
    validate_addendum_closure(value.get("addendum_closure"), s42=s42)
    _require_seed43_output_names()
    return _json_copy(value)


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    _require_sha(official_preflight_sha256, "seed43 official preflight")
    closure = preflight.get("addendum_closure") if isinstance(preflight, Mapping) else None
    if not isinstance(closure, Mapping):
        raise FailClosedError("seed43 root authorization requires a validated preflight closure")
    return {
        "schema": "tfsr_seed43_phase_e_replication_root_authorization_v1",
        "status": "ROOT_AUTHORIZED",
        "cell": CELL,
        "phase": PHASE,
        "official_preflight_sha256": official_preflight_sha256,
        "authority_root_relative": SEED43_AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SEED43_SCORE_ROOT_RELATIVE,
        "addendum_closure": _json_copy(closure),
        "target_free_preflight_required": True,
        "seed42_terminal_mandatory": True,
    }


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str, preflight: Mapping[str, object],
) -> dict[str, object]:
    _require_sha(official_preflight_sha256, "seed43 official preflight")
    expected = {
        "schema", "status", "cell", "phase", "official_preflight_sha256", "authority_root_relative",
        "score_root_relative", "addendum_closure", "target_free_preflight_required", "seed42_terminal_mandatory",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_seed43_phase_e_replication_root_authorization_v1"
            or value.get("status") != "ROOT_AUTHORIZED" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("official_preflight_sha256") != official_preflight_sha256
            or value.get("authority_root_relative") != SEED43_AUTHORITY_ROOT_RELATIVE
            or value.get("score_root_relative") != SEED43_SCORE_ROOT_RELATIVE
            or value.get("target_free_preflight_required") is not True
            or value.get("seed42_terminal_mandatory") is not True
            or value.get("addendum_closure") != preflight.get("addendum_closure")):
        raise FailClosedError("seed43 root authorization schema/binding drift")
    return _json_copy(value)


def publish_target_free_preflight(
    artifact: Any, capability: RootPublicationCapability, payload: Mapping[str, object], *,
    training: Seed43TrainingEvidence, upstream: UpstreamSeed42Evidence,
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may publish seed43 target-free preflight")
    checked = validate_target_free_preflight(payload, training=training, upstream=upstream)
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    artifact: Any, capability: RootPublicationCapability, payload: Mapping[str, object], *,
    training: Seed43TrainingEvidence, upstream: UpstreamSeed42Evidence,
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may publish seed43 root authorization")
    if not artifact.has_name("official_preflight.json"):
        raise FailClosedError("seed43 root authorization requires durable preflight")
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("seed43 durable preflight JSON drift") from error
    if not isinstance(preflight, Mapping):
        raise FailClosedError("seed43 durable preflight root drift")
    checked = validate_target_free_preflight(preflight, training=training, upstream=upstream)
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=_sha(body), preflight=checked,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def _read_seed43_authority_pair(root: Path, name: str) -> tuple[Mapping[str, object], str]:
    s42 = _seed42_score()
    return s42._read_authority_pair(root.absolute() / SEED43_AUTHORITY_ROOT_RELATIVE, name)


def verify_seed43_authorization(
    root: Path, *, training: Seed43TrainingEvidence, upstream: UpstreamSeed42Evidence,
) -> Seed43Authorization:
    try:
        preflight, preflight_sha = _read_seed43_authority_pair(root, "official_preflight.json")
        authorization, authorization_sha = _read_seed43_authority_pair(root, "root_authorization.json")
    except Exception as error:
        raise FailClosedError("seed43 authority pair absent or malformed before score root reserve") from error
    checked_preflight = validate_target_free_preflight(preflight, training=training, upstream=upstream)
    checked_auth = validate_root_authorization(
        authorization, official_preflight_sha256=preflight_sha, preflight=checked_preflight,
    )
    closure = checked_auth.get("addendum_closure")
    if not isinstance(closure, Mapping):
        raise FailClosedError("seed43 authorization closure missing")
    # A durable closure can be internally self-consistent while describing an
    # earlier source tree.  Rebuild the descriptor-bound current closure now,
    # before a physical backend, output root, or evaluation-asset factory can
    # be reached.  The same check is repeated after forwards, but that later
    # check cannot repair a stale pre-data authorization.
    observed = addendum_closure(root)
    if (checked_preflight.get("addendum_closure") != observed
            or closure != observed):
        raise FailClosedError("seed43 authorization closure differs from current implementation before backend")
    return Seed43Authorization(
        preflight_sha256=preflight_sha, root_authorization_sha256=authorization_sha,
        preflight=checked_preflight, root_authorization=checked_auth, closure=observed,
    )


@dataclass
class Score43Flags:
    """Compatibility subset required by the frozen input/mode implementation."""

    stage: str = "non_data_authority"
    source_resolved: bool = False
    within_resolved: bool = False
    external_resolved: bool = False
    formal_resolved: bool = False
    source_opened: bool = False
    within_opened: bool = False
    external_opened: bool = False
    formal_opened: bool = False
    forward_calls: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "within": {"aligned": 0, "zero": 0, "wrong_pair": 0},
        "external": {"aligned": 0, "zero": 0, "wrong_pair": 0},
    })
    backward_calls: int = 0
    optimizer_calls: int = 0
    terminal_published: bool = False

    def record_forward(self, system: str, surface: str, mode: str) -> None:
        if system != "tfsr" or surface not in self.forward_calls or mode not in self.forward_calls[surface]:
            raise FailClosedError("seed43 addendum observed forbidden Cell-D/unknown forward")
        self.forward_calls[surface][mode] += 1

    def validate_forward_calls(self, *, successful: bool = False) -> dict[str, dict[str, int]]:
        expected = {"within", "external"}
        if set(self.forward_calls) != expected:
            raise FailClosedError("seed43 forward-counter surface matrix drift")
        checked: dict[str, dict[str, int]] = {}
        for surface in ("within", "external"):
            row = self.forward_calls[surface]
            if set(row) != {"aligned", "zero", "wrong_pair"}:
                raise FailClosedError("seed43 forward-counter mode matrix drift")
            if any(type(item) is not int or item < 0 for item in row.values()):
                raise FailClosedError("seed43 forward-counter value drift")
            if successful and any(item <= 0 for item in row.values()):
                raise FailClosedError("successful seed43 addendum lacks a required mode forward")
            checked[surface] = {mode: int(row[mode]) for mode in ("aligned", "zero", "wrong_pair")}
        return checked

    def payload(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "resolved": {
                "source": self.source_resolved, "within": self.within_resolved,
                "external": self.external_resolved, "formal": self.formal_resolved,
            },
            "opened": {
                "source": self.source_opened, "within": self.within_opened,
                "external": self.external_opened, "formal": self.formal_opened,
            },
            "tfsr_forward_calls": self.validate_forward_calls(),
            "backward_calls": self.backward_calls,
            "optimizer_calls": self.optimizer_calls,
            "terminal_published": self.terminal_published,
        }


@dataclass
class AcceleratedParityManifest:
    """Exact per-forward build-v2/eager equivalence evidence, not a probe."""

    events: list[dict[str, object]] = field(default_factory=list)
    _next: dict[tuple[str, str], int] = field(default_factory=dict)

    def record(self, *, surface: str, mode: str, eager_bytes: bytes, accelerated_bytes: bytes, shape: Sequence[int]) -> None:
        if surface not in {"within", "external"} or mode not in {"aligned", "zero", "wrong_pair"}:
            raise FailClosedError("accelerated parity event surface/mode drift")
        if eager_bytes != accelerated_bytes:
            raise FailClosedError("seed43 eager/build-v2 output inequality before metric accumulation")
        if tuple(shape)[1:] != (50, 2):
            raise FailClosedError("accelerated parity event output shape drift")
        key = (surface, mode)
        ordinal = self._next.get(key, 0) + 1
        self._next[key] = ordinal
        digest = _sha(eager_bytes)
        self.events.append({
            "surface": surface, "mode": mode, "ordinal": ordinal,
            "eager_prediction_sha256": digest,
            "accelerated_prediction_sha256": _sha(accelerated_bytes),
            "bitwise_equal": True,
            "shape": [int(item) for item in shape],
        })

    def payload(self, flags: Score43Flags) -> dict[str, object]:
        counts = flags.validate_forward_calls(successful=True)
        expected_total = sum(item for row in counts.values() for item in row.values())
        if len(self.events) != expected_total:
            raise FailClosedError("accelerated parity manifest misses a scored forward")
        by_key: dict[tuple[str, str], list[Mapping[str, object]]] = {}
        for event in self.events:
            if not isinstance(event, Mapping):
                raise FailClosedError("accelerated parity event type drift")
            key = (str(event.get("surface")), str(event.get("mode")))
            by_key.setdefault(key, []).append(event)
        for surface, row in counts.items():
            for mode, count in row.items():
                events = by_key.get((surface, mode), [])
                if len(events) != count:
                    raise FailClosedError("accelerated parity manifest counter drift")
                for expected_ordinal, event in enumerate(events, start=1):
                    if (event.get("ordinal") != expected_ordinal or event.get("bitwise_equal") is not True
                            or event.get("eager_prediction_sha256") != event.get("accelerated_prediction_sha256")
                            or not _is_sha(event.get("eager_prediction_sha256"))
                            or event.get("shape") is None):
                        raise FailClosedError("accelerated parity event integrity drift")
        payload_events = [_json_copy(item) for item in self.events]
        body = _json_bytes(payload_events)
        return {
            "schema": "tfsr_seed43_accelerated_eager_parity_manifest_v1",
            "build": BUILD_LABEL,
            "all_scored_batches_bitwise_equal": True,
            "tfsr_forward_calls": counts,
            "events": payload_events,
            "events_sha256": _sha(body),
        }


def validate_parity_manifest(value: object, flags: Score43Flags) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "build", "all_scored_batches_bitwise_equal", "tfsr_forward_calls", "events", "events_sha256",
    }:
        raise FailClosedError("accelerated parity manifest schema drift")
    if (value.get("schema") != "tfsr_seed43_accelerated_eager_parity_manifest_v1"
            or value.get("build") != BUILD_LABEL or value.get("all_scored_batches_bitwise_equal") is not True
            or value.get("tfsr_forward_calls") != flags.validate_forward_calls(successful=True)
            or not isinstance(value.get("events"), list)
            or value.get("events_sha256") != _sha(_json_bytes(value["events"]))):
        raise FailClosedError("accelerated parity manifest header/binding drift")
    rebuilt = AcceleratedParityManifest()
    for event in value["events"]:
        if not isinstance(event, Mapping):
            raise FailClosedError("accelerated parity manifest event drift")
        # Reconstruct only structural checks.  The bytes have deliberately not
        # been persisted; equal digests are the immutable compact evidence.
        required = {"surface", "mode", "ordinal", "eager_prediction_sha256", "accelerated_prediction_sha256", "bitwise_equal", "shape"}
        if set(event) != required or event.get("eager_prediction_sha256") != event.get("accelerated_prediction_sha256"):
            raise FailClosedError("accelerated parity manifest event schema/equality drift")
        if (event.get("surface") not in {"within", "external"} or event.get("mode") not in {"aligned", "zero", "wrong_pair"}
                or type(event.get("ordinal")) is not int or int(event["ordinal"]) <= 0
                or not _is_sha(event.get("eager_prediction_sha256"))
                or event.get("bitwise_equal") is not True
                or not isinstance(event.get("shape"), list) or len(event["shape"]) != 3 or event["shape"][1:] != [50, 2]):
            raise FailClosedError("accelerated parity manifest event value drift")
        key = (str(event["surface"]), str(event["mode"]))
        ordinal = rebuilt._next.get(key, 0) + 1
        if event["ordinal"] != ordinal:
            raise FailClosedError("accelerated parity manifest ordinal drift")
        rebuilt._next[key] = ordinal
        rebuilt.events.append(dict(event))
    rebuilt_payload = rebuilt.payload(flags)
    if rebuilt_payload != dict(value):
        raise FailClosedError("accelerated parity manifest roundtrip drift")
    return rebuilt_payload


def _tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def build_seed43_physical_backend(
    *,
    root: Path,
    fixed_authorities: Mapping[str, Any],
    upstream_training: Any,
    upstream_authorization: Any,
    seed43_training: Seed43TrainingEvidence,
) -> Any:
    """Return a real subclass of the immutable seed42 physical backend.

    ``super`` remains responsible for GPU1 attestation, descriptor-held input
    parsing, T4 controls, the governed metric, and the shared scoring loop.
    The addendum does not mutate any seed42 global and never executes its
    Cell-D path.  Only its model/SWA loader and TF-SR forward are overridden.
    """
    s42 = _seed42_score()

    class Seed43PhysicalMatchedScoreBackend(s42.PhysicalMatchedScoreBackend):
        def __init__(self) -> None:
            super().__init__(
                root=root, fixed_authorities=fixed_authorities,
                training=upstream_training, authorization=upstream_authorization,
            )
            self._seed43_training = seed43_training
            self._scripted_step: Any | None = None
            self._accelerated_eval: Callable[..., Any] | None = None
            self._parity_manifest = AcceleratedParityManifest()

        def _ensure_models(self) -> None:
            if self._tfsr is not None:
                if self._cell_d is not None:
                    raise FailClosedError("seed43 addendum must not construct or reuse Cell-D model")
                return
            runtime = self._load_runtime()
            torch = runtime["torch"]
            state = _seed43_swa_state_at_use(root, self._seed43_training)
            model = runtime["TFSRDecoder"](capture_diagnostics=False).to(runtime["device"])
            model.load_state_dict(state, strict=True)
            model.eval()
            if model.capture_diagnostics is not False:
                raise FailClosedError("seed43 strict-loaded model capture diagnostics drift")
            if not self._all_gradients_none(model, torch):
                raise FailClosedError("seed43 strict-loaded model has preexisting gradients")
            from . import accelerated_forward

            scripted = accelerated_forward.build_scripted_step_binding(torch, model)
            self._tfsr = model
            self._state_at_load = {"tfsr": s42._physical_model_state_digest(model, torch)}
            self._scripted_step = scripted
            self._accelerated_eval = accelerated_forward.accelerated_forward

        def score_cell_d(self, **_: Any) -> Any:
            raise FailClosedError("seed43 replication addendum must never rerun Cell-D")

        def _forward_tfsr(
            self, model: Any, neural: Any, calib: Any, capability: Any, *,
            surface: str, mode: str, flags: Score43Flags, measure_latency: bool = False,
        ) -> Any:
            eager = super()._forward_tfsr(
                model, neural, calib, capability, surface=surface, mode=mode,
                flags=flags, measure_latency=measure_latency,
            )
            runtime = self._load_runtime()
            torch = runtime["torch"]
            if self._scripted_step is None or self._accelerated_eval is None:
                raise FailClosedError("seed43 accelerated evaluation binding absent")
            with torch.no_grad():
                accelerated = self._accelerated_eval(torch, model, self._scripted_step, neural, calib, capability)
            if (not torch.is_tensor(accelerated) or accelerated.shape != eager.shape
                    or not torch.isfinite(accelerated).all().item()):
                raise FailClosedError("seed43 accelerated evaluation output shape/finite drift")
            self._assert_tfsr_eval_no_mask(model, torch)
            self._parity_manifest.record(
                surface=surface, mode=mode,
                eager_bytes=_tensor_bytes(eager), accelerated_bytes=_tensor_bytes(accelerated),
                shape=tuple(int(item) for item in eager.shape),
            )
            return eager

        def build_parity_manifest(self, *, flags: Score43Flags) -> Mapping[str, object]:
            return self._parity_manifest.payload(flags)

        def reverify_after_forwards(self, *, flags: Score43Flags) -> None:
            runtime = self._load_runtime()
            torch = runtime["torch"]
            for held in self._held:
                held.reverify()
            if self._tfsr is None or self._cell_d is not None:
                raise FailClosedError("seed43 addendum physical model topology drift after forwards")
            if (s42._physical_model_state_digest(self._tfsr, torch) != self._state_at_load.get("tfsr")
                    or not self._all_gradients_none(self._tfsr, torch)
                    or flags.backward_calls != 0 or flags.optimizer_calls != 0
                    or flags.formal_resolved or flags.formal_opened):
                raise FailClosedError("seed43 addendum crossed no-update/formal/model-state boundary")

        def resource_disclosure(self) -> Mapping[str, object]:
            runtime = self._load_runtime()
            torch = runtime["torch"]
            import torchmetrics

            if self._tfsr is None or self._cell_d is not None or self._aligned_latency_ms is None:
                raise FailClosedError("seed43 resource disclosure requested before complete TF-SR-only score")
            parameter_count = sum(parameter.numel() for parameter in self._tfsr.parameters())
            return {
                "schema": "tfsr_seed43_phase_e_replication_resources_v1",
                "runtime": {**dict(s42.FROZEN_SCORE_DEVICE), "torchmetrics_version": str(torchmetrics.__version__)},
                "seed43_training_peak_memory_bytes": self._seed43_training.training_peak_memory_bytes,
                "score_peak_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
                "aligned_eager_latency_ms": float(self._aligned_latency_ms),
                "tfsr_parameters": int(parameter_count),
                "build": BUILD_LABEL,
                "cell_d_replayed_upstream_only": True,
                "cell_d_forward_calls": 0,
                "forward_only": True,
                "no_grad": True,
            }

    return Seed43PhysicalMatchedScoreBackend()


class AddendumBackend(Protocol):
    def resolve_inputs(self, **kwargs: Any) -> Any: ...
    def score_tfsr(self, **kwargs: Any) -> Any: ...
    def build_parity_manifest(self, **kwargs: Any) -> Mapping[str, object]: ...
    def reverify_after_forwards(self, **kwargs: Any) -> None: ...
    def resource_disclosure(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


def _upstream_input_evidence_payload(upstream: UpstreamSeed42Evidence) -> dict[str, object]:
    keys = {
        "schema", "cell", "records", "source_normalizer_body_sha256", "t4_normalizer_semantic_sha256",
        "behavior_normalizer_semantic_sha256", "strict_manifest_sha256", "raw_to_normalized_exact",
        "no_cache_readonly_adapter",
    }
    payload = upstream.input_payload
    if not isinstance(payload, Mapping) or not keys.issubset(payload):
        raise FailClosedError("upstream seed42 input authority missing replay evidence")
    return {key: _json_copy(payload[key]) for key in sorted(keys)}


def _validate_input_replay(
    evidence: Any, *, upstream: UpstreamSeed42Evidence, flags: Score43Flags, s42: Any,
) -> dict[str, object]:
    if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
        raise FailClosedError("seed43 input replay crossed formal/update boundary")
    observed = evidence.payload()
    expected = _upstream_input_evidence_payload(upstream)
    if observed != expected:
        raise FailClosedError("seed43 live input reconstruction differs from immutable seed42 input authority")
    return {
        "schema": "tfsr_seed43_phase_e_input_replay_v1",
        "upstream_seed42_input_authority_sha256": upstream.input_authority_sha256,
        "upstream_seed42_score_sha256": upstream.score_sha256,
        "input_evidence": _json_copy(observed),
        "input_evidence_sha256": _sha(_json_bytes(observed)),
        "same_rosters_normalizers_target_masks": True,
        "no_cache_readonly_adapter": True,
    }


def validate_input_replay_payload(
    value: Mapping[str, object], *, upstream: UpstreamSeed42Evidence,
) -> dict[str, object]:
    expected = {
        "schema", "upstream_seed42_input_authority_sha256", "upstream_seed42_score_sha256",
        "input_evidence", "input_evidence_sha256", "same_rosters_normalizers_target_masks", "no_cache_readonly_adapter",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_seed43_phase_e_input_replay_v1"
            or value.get("upstream_seed42_input_authority_sha256") != upstream.input_authority_sha256
            or value.get("upstream_seed42_score_sha256") != upstream.score_sha256
            or value.get("same_rosters_normalizers_target_masks") is not True
            or value.get("no_cache_readonly_adapter") is not True
            or value.get("input_evidence") != _upstream_input_evidence_payload(upstream)
            or value.get("input_evidence_sha256") != _sha(_json_bytes(value.get("input_evidence")))):
        raise FailClosedError("seed43 input replay receipt drift")
    return _json_copy(value)


def _replication_role(upstream_verdict: str) -> str:
    if upstream_verdict == "CLEAR_GO":
        return "CONFIRMATORY_REPLICATION"
    if upstream_verdict in {"HOLD", "STOP"}:
        return "NON_GOVERNING_POST_AUTHORIZATION_REPLICATION"
    raise FailClosedError("unknown upstream seed42 verdict")


def _mode_map(s42: Any, evidence: Sequence[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {"within": {}, "external": {}}
    for item in evidence:
        if item.system != "tfsr" or item.surface not in result or item.mode not in {"aligned", "zero", "wrong_pair"}:
            raise FailClosedError("seed43 addendum TF-SR evidence cell drift")
        if item.mode in result[item.surface]:
            raise FailClosedError("seed43 addendum duplicate evidence cell")
        result[item.surface][item.mode] = item
    if any(set(row) != {"aligned", "zero", "wrong_pair"} for row in result.values()):
        raise FailClosedError("seed43 addendum incomplete TF-SR evidence matrix")
    return result


def _upstream_cell_d_map(s42: Any, upstream: UpstreamSeed42Evidence) -> dict[str, Any]:
    block = upstream.score_payload.get("cell_d")
    if not isinstance(block, Mapping) or set(block) != {"within", "external"}:
        raise FailClosedError("upstream seed42 Cell-D replay evidence missing")
    result: dict[str, Any] = {}
    for surface in ("within", "external"):
        try:
            value = s42.mode_evidence_from_payload(block[surface])
        except Exception as error:
            raise FailClosedError("upstream seed42 Cell-D replay evidence invalid") from error
        if value.system != "cell_d" or value.surface != surface or value.mode != "aligned":
            raise FailClosedError("upstream seed42 Cell-D replay mode drift")
        result[surface] = value
    return result


def _paired_delta(s42: Any, successor: Any, baseline: Any) -> dict[str, object]:
    successor_values = {item.session: item.governing_r2 for item in successor.sessions}
    baseline_values = {item.session: item.governing_r2 for item in baseline.sessions}
    sessions = tuple(sorted(baseline_values))
    if tuple(sorted(successor_values)) != sessions:
        raise FailClosedError("seed43/seed42 Cell-D paired roster drift")
    result = s42.paired_statistics([successor_values[item] - baseline_values[item] for item in sessions])
    result["sessions_sorted"] = list(sessions)
    return result


def _a2_contextual(s42: Any, successor: Any, upstream: UpstreamSeed42Evidence, surface: str) -> dict[str, object]:
    block = upstream.score_payload.get("a2_contextual")
    pooled = block.get("pooled_per_session") if isinstance(block, Mapping) else None
    table = pooled.get(surface) if isinstance(pooled, Mapping) else None
    values = {item.session: item.governing_r2 for item in successor.sessions}
    if not isinstance(table, Mapping) or set(table) != set(values):
        raise FailClosedError("upstream A2 pooled contextual table drift")
    result = s42.paired_statistics([values[name] - float(table[name]) for name in sorted(values)])
    result.update({
        "label": "seed43-accelerated-v2-versus-A2-three-seed-pooled",
        "non_gating": True,
        "sessions_sorted": list(sorted(values)),
        "a2_equal_session_mean": sum(float(table[name]) for name in values) / len(values),
    })
    return result


def _validate_resources(value: Mapping[str, object], *, s42: Any) -> dict[str, object]:
    required = {
        "schema", "runtime", "seed43_training_peak_memory_bytes", "score_peak_memory_bytes",
        "aligned_eager_latency_ms", "tfsr_parameters", "build", "cell_d_replayed_upstream_only",
        "cell_d_forward_calls", "forward_only", "no_grad",
    }
    if (not isinstance(value, Mapping) or set(value) != required
            or value.get("schema") != "tfsr_seed43_phase_e_replication_resources_v1"
            or value.get("runtime") != {**dict(s42.FROZEN_SCORE_DEVICE), "torchmetrics_version": "1.5.1"}
            or value.get("build") != BUILD_LABEL or value.get("cell_d_replayed_upstream_only") is not True
            or value.get("cell_d_forward_calls") != 0 or value.get("forward_only") is not True or value.get("no_grad") is not True):
        raise FailClosedError("seed43 resource disclosure schema/runtime drift")
    for key in ("seed43_training_peak_memory_bytes", "score_peak_memory_bytes", "tfsr_parameters"):
        if type(value.get(key)) is not int or int(value[key]) < 0:
            raise FailClosedError("seed43 resource disclosure integer drift")
    latency = value.get("aligned_eager_latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or float(latency) < 0:
        raise FailClosedError("seed43 resource disclosure latency drift")
    return _json_copy(value)


def build_score_payload(
    *,
    identity: Seed43ScoreIdentity,
    input_replay_sha256: str,
    tfsr_evidence: Sequence[Any],
    parity_manifest: Mapping[str, object],
    resources: Mapping[str, object],
    flags: Score43Flags,
) -> dict[str, object]:
    s42 = _seed42_score()
    _require_sha(input_replay_sha256, "seed43 input replay")
    mode_map = _mode_map(s42, tfsr_evidence)
    cell_d = _upstream_cell_d_map(s42, identity.upstream)
    contrasts: dict[str, object] = {}
    contextual: dict[str, object] = {}
    controls: dict[str, object] = {}
    for surface in ("within", "external"):
        contrasts[surface] = _paired_delta(s42, mode_map[surface]["aligned"], cell_d[surface])
        contextual[surface] = _a2_contextual(s42, mode_map[surface]["aligned"], identity.upstream, surface)
        controls[surface] = {
            "aligned_minus_zero": s42._control_contrast(mode_map[surface]["aligned"], mode_map[surface]["zero"]),
            "aligned_minus_wrong_pair": s42._control_contrast(mode_map[surface]["aligned"], mode_map[surface]["wrong_pair"]),
            "non_rescuing": True,
        }
    if (flags.backward_calls != 0 or flags.optimizer_calls != 0
            or flags.formal_resolved or flags.formal_opened
            or not flags.within_opened or not flags.external_opened):
        raise FailClosedError("seed43 score crossed no-update/formal or omitted read-only evaluation boundary")
    checked_parity = validate_parity_manifest(parity_manifest, flags)
    checked_resources = _validate_resources(resources, s42=s42)
    verdict = s42.decide_verdict(external=contrasts["external"], within=contrasts["within"])
    payload = {
        "schema": "tfsr_seed43_phase_e_replication_addendum_score_v1",
        "status": "REPLICATION_SCORE_COMPLETE",
        "cell": CELL,
        "phase": PHASE,
        "seed": SEED,
        "build": BUILD_LABEL,
        "score_spec": dict(PUBLIC_SCORE_SPEC),
        "identity": identity.payload(),
        "input_replay_sha256": input_replay_sha256,
        "upstream_seed42": identity.upstream.binding_payload(),
        "replication_role": _replication_role(identity.upstream.verdict),
        "upstream_cell_d_replay": {
            surface: identity.upstream.score_payload["cell_d"][surface] for surface in ("within", "external")
        },
        "tfsr": {
            surface: {mode: mode_map[surface][mode].payload() for mode in ("aligned", "zero", "wrong_pair")}
            for surface in ("within", "external")
        },
        "tfsr_minus_upstream_cell_d": contrasts,
        "a2_contextual": {
            "label": "seed43-accelerated-v2-versus-A2-three-seed-pooled",
            "non_gating": True,
            "by_surface": contextual,
        },
        "controls": controls,
        "accelerated_eager_parity": checked_parity,
        "resources": checked_resources,
        "zero_target_optimizer_backward_update_evidence": {
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "within_readonly_opened": flags.within_opened,
            "external_readonly_opened": flags.external_opened,
            "formal_resolved": False,
            "formal_opened": False,
        },
        "boundaries": flags.payload(),
        "seed42_gate_not_revised": True,
        "seed44_authorized": False,
        "three_seed_superiority_claim": False,
        "verdict": verdict,
        "verdict_rule": "inherited_seed42_STOP_then_CLEAR_GO_else_HOLD; addendum_does_not_revise_gate",
    }
    validate_score_payload(payload, identity=identity, input_replay_sha256=input_replay_sha256)
    return payload


def validate_score_payload(
    value: Mapping[str, object], *, identity: Seed43ScoreIdentity | None = None,
    input_replay_sha256: str | None = None,
) -> dict[str, object]:
    s42 = _seed42_score()
    required = {
        "schema", "status", "cell", "phase", "seed", "build", "score_spec", "identity", "input_replay_sha256",
        "upstream_seed42", "replication_role", "upstream_cell_d_replay", "tfsr", "tfsr_minus_upstream_cell_d",
        "a2_contextual", "controls", "accelerated_eager_parity", "resources",
        "zero_target_optimizer_backward_update_evidence", "boundaries", "seed42_gate_not_revised",
        "seed44_authorized", "three_seed_superiority_claim", "verdict", "verdict_rule",
    }
    if (not isinstance(value, Mapping) or set(value) != required
            or value.get("schema") != "tfsr_seed43_phase_e_replication_addendum_score_v1"
            or value.get("status") != "REPLICATION_SCORE_COMPLETE" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("seed") != SEED or value.get("build") != BUILD_LABEL
            or value.get("score_spec") != PUBLIC_SCORE_SPEC or not _is_sha(value.get("input_replay_sha256"))
            or value.get("seed42_gate_not_revised") is not True or value.get("seed44_authorized") is not False
            or value.get("three_seed_superiority_claim") is not False
            or value.get("verdict_rule") != "inherited_seed42_STOP_then_CLEAR_GO_else_HOLD; addendum_does_not_revise_gate"):
        raise FailClosedError("seed43 score receipt schema/header drift")
    if identity is not None:
        if value.get("identity") != identity.payload() or value.get("upstream_seed42") != identity.upstream.binding_payload():
            raise FailClosedError("seed43 score identity/upstream binding drift")
    if input_replay_sha256 is not None and value.get("input_replay_sha256") != input_replay_sha256:
        raise FailClosedError("seed43 score input replay binding drift")
    upstream = identity.upstream if identity is not None else None
    tfsr_block = value.get("tfsr")
    if not isinstance(tfsr_block, Mapping) or set(tfsr_block) != {"within", "external"}:
        raise FailClosedError("seed43 score TF-SR evidence matrix drift")
    modes: dict[str, dict[str, Any]] = {"within": {}, "external": {}}
    for surface in ("within", "external"):
        row = tfsr_block.get(surface)
        if not isinstance(row, Mapping) or set(row) != {"aligned", "zero", "wrong_pair"}:
            raise FailClosedError("seed43 score TF-SR mode matrix drift")
        for mode in ("aligned", "zero", "wrong_pair"):
            try:
                item = s42.mode_evidence_from_payload(row[mode])
            except Exception as error:
                raise FailClosedError("seed43 score persisted mode evidence invalid") from error
            if item.system != "tfsr" or item.surface != surface or item.mode != mode:
                raise FailClosedError("seed43 score persisted mode label drift")
            modes[surface][mode] = item
    boundaries = value.get("boundaries")
    if not isinstance(boundaries, Mapping) or boundaries.get("terminal_published") is not False:
        raise FailClosedError("seed43 score boundary payload drift before terminal publication")
    flags = Score43Flags(
        stage=str(boundaries.get("stage")),
        source_resolved=bool(boundaries.get("resolved", {}).get("source")) if isinstance(boundaries.get("resolved"), Mapping) else False,
        within_resolved=bool(boundaries.get("resolved", {}).get("within")) if isinstance(boundaries.get("resolved"), Mapping) else False,
        external_resolved=bool(boundaries.get("resolved", {}).get("external")) if isinstance(boundaries.get("resolved"), Mapping) else False,
        formal_resolved=bool(boundaries.get("resolved", {}).get("formal")) if isinstance(boundaries.get("resolved"), Mapping) else False,
        source_opened=bool(boundaries.get("opened", {}).get("source")) if isinstance(boundaries.get("opened"), Mapping) else False,
        within_opened=bool(boundaries.get("opened", {}).get("within")) if isinstance(boundaries.get("opened"), Mapping) else False,
        external_opened=bool(boundaries.get("opened", {}).get("external")) if isinstance(boundaries.get("opened"), Mapping) else False,
        formal_opened=bool(boundaries.get("opened", {}).get("formal")) if isinstance(boundaries.get("opened"), Mapping) else False,
        forward_calls=_json_copy(boundaries.get("tfsr_forward_calls")) if isinstance(boundaries.get("tfsr_forward_calls"), Mapping) else {},
        backward_calls=boundaries.get("backward_calls"), optimizer_calls=boundaries.get("optimizer_calls"),
        terminal_published=False,
    )
    if flags.payload() != dict(boundaries):
        raise FailClosedError("seed43 score boundary roundtrip drift")
    validate_parity_manifest(value.get("accelerated_eager_parity"), flags)
    _validate_resources(value.get("resources"), s42=s42)
    zero_update = value.get("zero_target_optimizer_backward_update_evidence")
    if zero_update != {
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "within_readonly_opened": True, "external_readonly_opened": True,
        "formal_resolved": False, "formal_opened": False,
    }:
        raise FailClosedError("seed43 score no-update/formal evidence drift")
    if upstream is not None:
        baseline = _upstream_cell_d_map(s42, upstream)
        if value.get("upstream_cell_d_replay") != {
            surface: upstream.score_payload["cell_d"][surface] for surface in ("within", "external")
        }:
            raise FailClosedError("seed43 score upstream Cell-D replay payload drift")
        contrast = value.get("tfsr_minus_upstream_cell_d")
        if not isinstance(contrast, Mapping):
            raise FailClosedError("seed43 score paired contrast block missing")
        for surface in ("within", "external"):
            if contrast.get(surface) != _paired_delta(s42, modes[surface]["aligned"], baseline[surface]):
                raise FailClosedError("seed43 score paired Cell-D contrast drift")
        expected_role = _replication_role(upstream.verdict)
        if value.get("replication_role") != expected_role:
            raise FailClosedError("seed43 replication role rewrites upstream gate")
        expected_verdict = s42.decide_verdict(external=contrast["external"], within=contrast["within"])
        if value.get("verdict") != expected_verdict:
            raise FailClosedError("seed43 score verdict drift")
    return _json_copy(value)


def _attempt_payload(identity: Seed43ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "tfsr_seed43_phase_e_replication_attempt_v1",
        "cell": CELL,
        "phase": PHASE,
        "identity": identity.payload(),
        "topology": list(SCORE_TOPOLOGY),
        "resolved": {"source": False, "within": False, "external": False, "formal": False},
        "opened": {"source": False, "within": False, "external": False, "formal": False},
        "backward_calls": 0,
        "optimizer_calls": 0,
        "cell_d_rerun": False,
    }


def validate_attempt_payload(value: Mapping[str, object], identity: Seed43ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "cell", "phase", "identity", "topology", "resolved", "opened",
        "backward_calls", "optimizer_calls", "cell_d_rerun",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_seed43_phase_e_replication_attempt_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("identity") != identity.payload()
            or value.get("topology") != list(SCORE_TOPOLOGY)
            or value.get("resolved") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("opened") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("backward_calls") != 0 or value.get("optimizer_calls") != 0 or value.get("cell_d_rerun") is not False):
        raise FailClosedError("seed43 attempt receipt schema/boundary drift")
    return _json_copy(value)


def _failure_payload(
    flags: Score43Flags,
    *,
    identity: Seed43ScoreIdentity,
    attempt_sha256: str,
    input_replay_sha256: str | None,
) -> dict[str, object]:
    """Forensically bind a failed attempt to its immutable identity/input state."""
    _require_sha(attempt_sha256, "seed43 failed-attempt SHA")
    if input_replay_sha256 is not None:
        _require_sha(input_replay_sha256, "seed43 failed input-replay SHA")
    return {
        "schema": "tfsr_seed43_phase_e_replication_failure_v1",
        "cell": CELL,
        "phase": PHASE,
        "identity": identity.payload(),
        "attempt_sha256": attempt_sha256,
        "input_replay_sha256": input_replay_sha256,
        "stage": flags.stage,
        **flags.payload(),
        "cell_d_rerun": False,
        "traceback_sha256": _sha(traceback.format_exc().encode("utf-8")),
    }


def validate_failure_payload(
    value: Mapping[str, object],
    *,
    identity: Seed43ScoreIdentity,
    attempt_sha256: str,
    input_replay_sha256: str | None,
) -> dict[str, object]:
    _require_sha(attempt_sha256, "seed43 failed-attempt SHA")
    if input_replay_sha256 is not None:
        _require_sha(input_replay_sha256, "seed43 failed input-replay SHA")
    required = {
        "schema", "cell", "phase", "identity", "attempt_sha256", "input_replay_sha256",
        "stage", "resolved", "opened", "tfsr_forward_calls",
        "backward_calls", "optimizer_calls", "terminal_published", "cell_d_rerun", "traceback_sha256",
    }
    if (not isinstance(value, Mapping) or set(value) != required
            or value.get("schema") != "tfsr_seed43_phase_e_replication_failure_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("identity") != identity.payload()
            or value.get("attempt_sha256") != attempt_sha256
            or value.get("input_replay_sha256") != input_replay_sha256
            or value.get("terminal_published") is not False or value.get("cell_d_rerun") is not False
            or not _is_sha(value.get("traceback_sha256"))):
        raise FailClosedError("seed43 failure receipt schema drift")
    flags = Score43Flags(
        stage=str(value.get("stage")),
        source_resolved=bool(value.get("resolved", {}).get("source")) if isinstance(value.get("resolved"), Mapping) else False,
        within_resolved=bool(value.get("resolved", {}).get("within")) if isinstance(value.get("resolved"), Mapping) else False,
        external_resolved=bool(value.get("resolved", {}).get("external")) if isinstance(value.get("resolved"), Mapping) else False,
        formal_resolved=bool(value.get("resolved", {}).get("formal")) if isinstance(value.get("resolved"), Mapping) else False,
        source_opened=bool(value.get("opened", {}).get("source")) if isinstance(value.get("opened"), Mapping) else False,
        within_opened=bool(value.get("opened", {}).get("within")) if isinstance(value.get("opened"), Mapping) else False,
        external_opened=bool(value.get("opened", {}).get("external")) if isinstance(value.get("opened"), Mapping) else False,
        formal_opened=bool(value.get("opened", {}).get("formal")) if isinstance(value.get("opened"), Mapping) else False,
        forward_calls=_json_copy(value.get("tfsr_forward_calls")) if isinstance(value.get("tfsr_forward_calls"), Mapping) else {},
        backward_calls=value.get("backward_calls"), optimizer_calls=value.get("optimizer_calls"), terminal_published=False,
    )
    if flags.payload() != {
        "stage": value["stage"], "resolved": value["resolved"], "opened": value["opened"],
        "tfsr_forward_calls": value["tfsr_forward_calls"], "backward_calls": value["backward_calls"],
        "optimizer_calls": value["optimizer_calls"], "terminal_published": False,
    }:
        raise FailClosedError("seed43 failure receipt flag roundtrip drift")
    if flags.formal_resolved or flags.formal_opened:
        raise FailClosedError("seed43 failure receipt crossed formal boundary")
    return _json_copy(value)


def _publish_failure(
    artifact: Any,
    flags: Score43Flags,
    *,
    identity: Seed43ScoreIdentity,
    attempt_sha256: str,
    input_replay_sha256: str | None,
) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json"):
        raise FailClosedError("seed43 failure publication forbidden after terminal")
    if artifact.has_name("failure.json"):
        return
    if input_replay_sha256 is None:
        if artifact.has_name("input_replay.json"):
            raise FailClosedError("seed43 failed replay binding omitted despite durable input receipt")
    elif not artifact.has_name("input_replay.json"):
        raise FailClosedError("seed43 failed replay binding names no durable input receipt")
    payload = _failure_payload(
        flags, identity=identity, attempt_sha256=attempt_sha256,
        input_replay_sha256=input_replay_sha256,
    )
    validate_failure_payload(
        payload, identity=identity, attempt_sha256=attempt_sha256,
        input_replay_sha256=input_replay_sha256,
    )
    digest = artifact.publish_json("failure.json", payload)
    validate_failure_payload(
        artifact.reload_json("failure.json", digest), identity=identity,
        attempt_sha256=attempt_sha256, input_replay_sha256=input_replay_sha256,
    )


def _terminal_payload(
    *, identity: Seed43ScoreIdentity, attempt_sha256: str, input_replay_sha256: str,
    score_sha256: str, score_payload: Mapping[str, object], final_closure: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "tfsr_seed43_phase_e_replication_terminal_v1",
        "status": "REPLICATION_SCORE_COMPLETE",
        "cell": CELL,
        "phase": PHASE,
        "identity": identity.payload(),
        "attempt_sha256": attempt_sha256,
        "input_replay_sha256": input_replay_sha256,
        "score_sha256": score_sha256,
        "launch_closure": _json_copy(identity.closure),
        "final_closure": _json_copy(final_closure),
        "seed42_gate_verdict": identity.upstream.verdict,
        "replication_role": _replication_role(identity.upstream.verdict),
        "seed43_verdict": score_payload["verdict"],
        "seed42_gate_not_revised": True,
        "seed44_authorized": False,
        "three_seed_superiority_claim": False,
        "boundaries": {
            "target_optimizer_steps": 0, "backward_calls": 0, "optimizer_calls": 0,
            "formal_resolved": False, "formal_opened": False, "cell_d_rerun": False,
            "terminal_after_revalidation": True,
        },
    }


def validate_terminal_payload(
    value: Mapping[str, object], *, identity: Seed43ScoreIdentity, score_payload: Mapping[str, object],
    expected_score_sha256: str | None = None, expected_input_replay_sha256: str | None = None,
) -> dict[str, object]:
    required = {
        "schema", "status", "cell", "phase", "identity", "attempt_sha256", "input_replay_sha256", "score_sha256",
        "launch_closure", "final_closure", "seed42_gate_verdict", "replication_role", "seed43_verdict",
        "seed42_gate_not_revised", "seed44_authorized", "three_seed_superiority_claim", "boundaries",
    }
    if (not isinstance(value, Mapping) or set(value) != required
            or value.get("schema") != "tfsr_seed43_phase_e_replication_terminal_v1"
            or value.get("status") != "REPLICATION_SCORE_COMPLETE" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("identity") != identity.payload()
            or value.get("launch_closure") != identity.closure or value.get("final_closure") != identity.closure
            or value.get("seed42_gate_verdict") != identity.upstream.verdict
            or value.get("replication_role") != _replication_role(identity.upstream.verdict)
            or value.get("seed43_verdict") != score_payload.get("verdict")
            or value.get("seed42_gate_not_revised") is not True or value.get("seed44_authorized") is not False
            or value.get("three_seed_superiority_claim") is not False
            or not all(_is_sha(value.get(key)) for key in ("attempt_sha256", "input_replay_sha256", "score_sha256"))):
        raise FailClosedError("seed43 terminal receipt schema/binding drift")
    if expected_score_sha256 is not None and value.get("score_sha256") != expected_score_sha256:
        raise FailClosedError("seed43 terminal score SHA drift")
    if (expected_input_replay_sha256 is not None
            and value.get("input_replay_sha256") != expected_input_replay_sha256):
        raise FailClosedError("seed43 terminal input-replay SHA drift")
    expected_boundaries = {
        "target_optimizer_steps": 0, "backward_calls": 0, "optimizer_calls": 0,
        "formal_resolved": False, "formal_opened": False, "cell_d_rerun": False,
        "terminal_after_revalidation": True,
    }
    if value.get("boundaries") != expected_boundaries:
        raise FailClosedError("seed43 terminal boundary drift")
    return _json_copy(value)


def run_addendum_lifecycle(
    *,
    artifact: Any,
    identity: Seed43ScoreIdentity,
    execution_capability: ExecutionCapability,
    upstream: UpstreamSeed42Evidence,
    within_roster: tuple[str, ...],
    external_sessions: tuple[str, ...],
    within_assets_factory: Callable[[], tuple[Any, ...]],
    external_assets_factory: Callable[[], tuple[Any, ...]],
    backend: AddendumBackend,
    final_reverify: Callable[[], Mapping[str, object]],
) -> Mapping[str, object]:
    """Small independent receipt lifecycle; it never invokes Cell-D scoring."""
    _require_execution_capability(execution_capability, identity)
    if tuple(getattr(artifact, "topology", ())) != SCORE_TOPOLOGY:
        raise FailClosedError("seed43 score artifact topology drift")
    s42 = _seed42_score()
    flags = Score43Flags()
    attempt_written = False
    hashes: dict[str, str] = {}
    try:
        flags.stage = "attempt"
        attempt = _attempt_payload(identity)
        validate_attempt_payload(attempt, identity)
        hashes = {"attempt.json": artifact.publish_json("attempt.json", attempt)}
        validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), identity)
        attempt_written = True

        flags.stage = "resolve_identical_seed42_inputs_after_attempt"
        within_assets = within_assets_factory()
        flags.within_resolved = True
        external_assets = external_assets_factory()
        flags.external_resolved = True
        if tuple(item.session for item in within_assets) != tuple(sorted(within_roster)):
            raise FailClosedError("seed43 within asset roster drift after attempt")
        if tuple(item.session for item in external_assets) != external_sessions:
            raise FailClosedError("seed43 external asset roster drift after attempt")
        evidence = backend.resolve_inputs(
            spec=s42.PUBLIC_SPEC, within_roster=within_roster, within_assets=within_assets,
            external_roster=external_assets, flags=flags,
        )
        s42.validate_input_authority_evidence(
            evidence, within_roster=within_roster, within_assets=within_assets, external_roster=external_assets,
        )
        replay = _validate_input_replay(evidence, upstream=upstream, flags=flags, s42=s42)
        validate_input_replay_payload(replay, upstream=upstream)
        hashes["input_replay.json"] = artifact.publish_json("input_replay.json", replay)
        validate_input_replay_payload(artifact.reload_json("input_replay.json", hashes["input_replay.json"]), upstream=upstream)

        tfsr: list[Any] = []
        for surface, roster in (("within", within_roster), ("external", external_sessions)):
            for mode in ("aligned", "zero", "wrong_pair"):
                flags.stage = f"seed43_tfsr_{surface}_{mode}"
                item = backend.score_tfsr(
                    surface=surface, mode=mode, input_authority_sha256=hashes["input_replay.json"], flags=flags,
                )
                s42._validate_mode_against_roster(
                    item, system="tfsr", surface=surface, mode=mode,
                    expected_sessions=roster, input_authority_sha256=hashes["input_replay.json"],
                )
                tfsr.append(item)
        if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
            raise FailClosedError("seed43 addendum forward path crossed forbidden boundary")
        flags.stage = "post_forward_reverify"
        backend.reverify_after_forwards(flags=flags)
        parity = backend.build_parity_manifest(flags=flags)
        validate_parity_manifest(parity, flags)

        flags.stage = "score"
        score = build_score_payload(
            identity=identity, input_replay_sha256=hashes["input_replay.json"], tfsr_evidence=tfsr,
            parity_manifest=parity, resources=backend.resource_disclosure(), flags=flags,
        )
        score_body = _json_bytes(score)
        score_sha = _sha(score_body)
        flags.stage = "terminal_revalidation"
        final_closure = final_reverify()
        if validate_addendum_closure(final_closure, s42=s42) != identity.closure:
            raise FailClosedError("seed43 launch/final closure differs before terminal publication")
        validate_attempt_payload(artifact.reload_json("attempt.json", hashes["attempt.json"]), identity)
        validate_input_replay_payload(artifact.reload_json("input_replay.json", hashes["input_replay.json"]), upstream=upstream)
        validate_score_payload(score, identity=identity, input_replay_sha256=hashes["input_replay.json"])
        terminal = _terminal_payload(
            identity=identity, attempt_sha256=hashes["attempt.json"], input_replay_sha256=hashes["input_replay.json"],
            score_sha256=score_sha, score_payload=score, final_closure=final_closure,
        )
        validate_terminal_payload(
            terminal, identity=identity, score_payload=score, expected_score_sha256=score_sha,
            expected_input_replay_sha256=hashes["input_replay.json"],
        )
        terminal_body = _json_bytes(terminal)

        def validate_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if (bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body
                    or digests.get("score.json") != score_sha):
                raise FailClosedError("seed43 score/terminal group binding drift")
            grouped_score = json.loads(artifact.reload_pair("score.json", score_sha))
            grouped_terminal = json.loads(artifact.reload_pair("terminal.json", _sha(terminal_body)))
            if not isinstance(grouped_score, Mapping) or not isinstance(grouped_terminal, Mapping):
                raise FailClosedError("seed43 score/terminal group JSON root drift")
            validate_score_payload(grouped_score, identity=identity, input_replay_sha256=hashes["input_replay.json"])
            validate_terminal_payload(
                grouped_terminal, identity=identity, score_payload=grouped_score, expected_score_sha256=score_sha,
                expected_input_replay_sha256=hashes["input_replay.json"],
            )

        hashes.update(artifact.publish_group({"score.json": score_body, "terminal.json": terminal_body}, post_publish=validate_group))
        flags.terminal_published = True
        final_score = artifact.reload_json("score.json", hashes["score.json"])
        validate_score_payload(final_score, identity=identity, input_replay_sha256=hashes["input_replay.json"])
        final_terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
        validate_terminal_payload(
            final_terminal, identity=identity, score_payload=final_score, expected_score_sha256=hashes["score.json"],
            expected_input_replay_sha256=hashes["input_replay.json"],
        )
        if artifact.has_name("failure.json"):
            raise FailClosedError("seed43 terminal cannot coexist with failure receipt")
        return final_terminal
    except BaseException:
        if attempt_written:
            try:
                _publish_failure(
                    artifact, flags, identity=identity, attempt_sha256=hashes["attempt.json"],
                    input_replay_sha256=hashes.get("input_replay.json"),
                )
            except BaseException:
                pass
        raise
    finally:
        backend.close()


def validate_seed43_completed_score_payloads(
    *,
    score_payload: Mapping[str, object],
    score_sha256: str,
    terminal_payload: Mapping[str, object],
    terminal_sha256: str,
    identity: Seed43ScoreIdentity,
) -> Seed43CompletedScoreEvidence:
    """Validate an already descriptor-read seed-43 score/terminal pair.

    ``score_sha256`` and ``terminal_sha256`` must originate from immutable
    body/sidecar pairs (the reader below provides that capability).  Keeping
    their exact values in this typed evidence prevents a later descriptive
    report from being made from bare, caller-supplied means.
    """
    _require_sha(score_sha256, "seed43 completed score body")
    _require_sha(terminal_sha256, "seed43 completed terminal body")
    if (
        _sha(_json_bytes(score_payload)) != score_sha256
        or _sha(_json_bytes(terminal_payload)) != terminal_sha256
    ):
        raise FailClosedError("seed43 completed descriptor payload/body SHA binding drift")
    checked_score = validate_score_payload(score_payload, identity=identity)
    input_replay_sha256 = checked_score.get("input_replay_sha256")
    if not _is_sha(input_replay_sha256):
        raise FailClosedError("seed43 completed score input-replay binding drift")
    checked_terminal = validate_terminal_payload(
        terminal_payload, identity=identity, score_payload=checked_score,
        expected_score_sha256=score_sha256, expected_input_replay_sha256=str(input_replay_sha256),
    )
    return Seed43CompletedScoreEvidence(
        score_sha256=score_sha256, terminal_sha256=terminal_sha256,
        score_payload=checked_score, terminal_payload=checked_terminal,
    )


def load_seed43_completed_score(
    root: Path, *, identity: Seed43ScoreIdentity,
) -> Seed43CompletedScoreEvidence:
    """Descriptor-load the seed-43 score pair for a post-terminal report only."""
    s42 = _seed42_score()
    directory = root.absolute() / SEED43_SCORE_ROOT_RELATIVE
    try:
        score, score_sha = s42._read_authority_pair(directory, "score.json")
        terminal, terminal_sha = s42._read_authority_pair(directory, "terminal.json")
    except Exception as error:
        raise FailClosedError("seed43 completed score/terminal pair absent or malformed") from error
    return validate_seed43_completed_score_payloads(
        score_payload=score, score_sha256=score_sha, terminal_payload=terminal,
        terminal_sha256=terminal_sha, identity=identity,
    )


def _aligned_mean_from_score(s42: Any, payload: Mapping[str, object], *, surface: str, expected_system: str) -> float:
    block = payload.get(expected_system)
    if not isinstance(block, Mapping):
        raise FailClosedError("two-seed summary aligned evidence block missing")
    row = block.get(surface)
    if not isinstance(row, Mapping):
        raise FailClosedError("two-seed summary aligned surface evidence missing")
    try:
        evidence = s42.mode_evidence_from_payload(row["aligned"])
    except Exception as error:
        raise FailClosedError("two-seed summary aligned evidence invalid") from error
    if evidence.system != expected_system or evidence.surface != surface or evidence.mode != "aligned":
        raise FailClosedError("two-seed summary aligned evidence label drift")
    return float(evidence.mean_governing)


def two_seed_descriptive_summary(
    *,
    upstream: UpstreamSeed42Evidence,
    seed43: Seed43CompletedScoreEvidence,
    identity: Seed43ScoreIdentity,
) -> dict[str, object]:
    """Return a strictly descriptive, receipt-bound two-seed comparison.

    This cannot issue a decision: it revalidates both individual completed
    score chains, reports their exact body SHAs and build labels, and averages
    their already-equal-session aligned means only for presentation.
    """
    s42 = _seed42_score()
    # Re-run the frozen seed-42 receipt validators, even though ``upstream``
    # normally arrived from the descriptor loader, so a substituted payload or
    # body SHA cannot silently enter a publication summary.
    checked_upstream = validate_upstream_seed42_payloads(
        s42,
        input_payload=upstream.input_payload, input_sha256=upstream.input_authority_sha256,
        score_payload=upstream.score_payload, score_sha256=upstream.score_sha256,
        terminal_payload=upstream.terminal_payload, terminal_sha256=upstream.terminal_sha256,
    )
    if (checked_upstream.binding_payload() != upstream.binding_payload()
            or identity.upstream.binding_payload() != upstream.binding_payload()
            or upstream.build_label != SEED42_BUILD_LABEL):
        raise FailClosedError("two-seed summary upstream identity/build binding drift")
    checked_seed43 = validate_seed43_completed_score_payloads(
        score_payload=seed43.score_payload, score_sha256=seed43.score_sha256,
        terminal_payload=seed43.terminal_payload, terminal_sha256=seed43.terminal_sha256,
        identity=identity,
    )
    # The addendum score itself is the authoritative build-v2 disclosure.  A
    # summary never gets a caller-provided label that could hide a mismatch.
    if checked_seed43.score_payload.get("build") != BUILD_LABEL:
        raise FailClosedError("two-seed summary seed43 build label drift")
    by_surface: dict[str, dict[str, float]] = {}
    for surface in ("within", "external"):
        seed42_mean = _aligned_mean_from_score(s42, checked_upstream.score_payload, surface=surface, expected_system="tfsr")
        seed43_mean = _aligned_mean_from_score(s42, checked_seed43.score_payload, surface=surface, expected_system="tfsr")
        by_surface[surface] = {
            "seed42_aligned_equal_session_mean": seed42_mean,
            "seed43_aligned_equal_session_mean": seed43_mean,
            "two_seed_descriptive_equal_weight_mean": (seed42_mean + seed43_mean) / 2.0,
        }
    return {
        "schema": "tfsr_seed42_seed43_descriptive_summary_v2",
        "seeds": [42, 43],
        "individual_receipts": {
            "seed42": {
                "build_label": SEED42_BUILD_LABEL,
                "score_sha256": checked_upstream.score_sha256,
                "terminal_sha256": checked_upstream.terminal_sha256,
            },
            "seed43": checked_seed43.binding_payload(),
        },
        "aligned_equal_session_means": by_surface,
        "individual_terminals_required": True,
        "description_only": True,
        "non_inferential": True,
        "does_not_revise_seed42_gate": True,
        "three_seed_superiority_claim": False,
        "seed44_authorized": False,
    }


def dry_plan() -> dict[str, object]:
    """Pure static plan; do not import the seed42 scorer or Torch here."""
    return {
        "cell": CELL,
        "phase": PHASE,
        "seed": SEED,
        "build": BUILD_LABEL,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE",
        "score_spec": dict(PUBLIC_SCORE_SPEC),
        "upstream_requirement": {
            "seed42_phase_e_terminal_required_before_target_resolution": True,
            "bind": ["input_authority.json", "score.json", "terminal.json"],
            "cell_d": "replay upstream exact evidence only; no Cell-D rerun",
        },
        "canonical_roots": {
            "authority": SEED43_AUTHORITY_ROOT_RELATIVE,
            "score": SEED43_SCORE_ROOT_RELATIVE,
            "must_not_alias_seed42": True,
            "must_be_fresh_before_reviewed_execution": True,
        },
        "forward": {
            "metric": "seed42 matched final-bin variance-weighted R2; equal session weighting",
            "modes": ["aligned", "zero", "wrong_pair"],
            "eager_accelerated_v2": "bitwise equality on every scoring forward before metric",
        },
        "execution": "requires both public flags and a root-reviewed in-process capability; CLI provides none",
        "forbidden": [
            "target_optimizer", "backward", "update", "checkpoint_selection", "session_selection", "formal_data",
            "Cell-D_rerun", "seed42_global_mutation", "seed44_authorization", "three_seed_superiority_claim",
        ],
    }


def execute_authorized(
    root: Path,
    *,
    capability: ExecutionCapability | None = None,
    seed43_training_validator: Callable[[Path], Seed43TrainingEvidence] = validate_seed43_training_terminal,
    upstream_loader: Callable[[Path], UpstreamSeed42Evidence] = load_upstream_seed42_phase_e,
    backend_factory: Callable[..., AddendumBackend] | None = None,
) -> Mapping[str, object]:
    """Root-only live path; public CLI cannot manufacture ``capability``."""
    if capability is None:
        raise FailClosedError("root-reviewed seed43 capability required before authority/data/model resolution")
    # Required ordering: terminal score gate and seed43 chain are both checked
    # before a target path or physical backend is created.
    upstream = upstream_loader(root)
    training = seed43_training_validator(root)
    authorization = verify_seed43_authorization(root, training=training, upstream=upstream)
    identity = Seed43ScoreIdentity(training=training, upstream=upstream, closure=authorization.closure,
                                   authorization=authorization.payload())
    _require_execution_capability(capability, identity)
    s42 = _seed42_score()
    fixed = s42.verify_fixed_authorities(root)
    upstream_training = s42.validate_phase_d_training_terminal(root)
    upstream_authorization = s42.verify_phase_e_authorization(root, upstream_training, fixed)
    # The seed42 score terminal has already independently bound its input
    # authority to these seed42 training values.  Repeat that equality here so
    # the base input adapter cannot silently consume a substituted lineage.
    upstream_identity = _upstream_identity_from_terminal(s42, upstream.terminal_payload)
    if (upstream_identity.training_terminal_sha256 != upstream_training.terminal_sha256
            or upstream_identity.training_swa_sha256 != upstream_training.swa_sha256
            or upstream_identity.training_swa_state_digest != upstream_training.swa_state_digest):
        raise FailClosedError("seed42 input backend lineage does not match upstream terminal")
    if backend_factory is None:
        backend = build_seed43_physical_backend(
            root=root, fixed_authorities=fixed, upstream_training=upstream_training,
            upstream_authorization=upstream_authorization, seed43_training=training,
        )
    else:
        backend = backend_factory(
            root=root, fixed_authorities=fixed, upstream_training=upstream_training,
            upstream_authorization=upstream_authorization, seed43_training=training,
        )
    artifact = reserve_score_artifact(root)
    within_roster = s42.extract_within_roster(fixed["strict_manifest"].value or {})
    external_sessions = s42.extract_external_sessions(fixed["external_asset_ledger"].value or {})

    def within_assets_factory() -> tuple[Any, ...]:
        raw_root = os.environ.get("SUBC_DATA_ROOT")
        if not isinstance(raw_root, str) or not raw_root:
            raise FailClosedError("SUBC_DATA_ROOT is required only after durable seed43 attempt")
        return s42.join_within_assets(upstream_authorization.preflight, Path(raw_root))

    def external_assets_factory() -> tuple[Any, ...]:
        raw_root = os.environ.get("SUBM_DATA_ROOT")
        if not isinstance(raw_root, str) or not raw_root:
            raise FailClosedError("SUBM_DATA_ROOT is required only after durable seed43 attempt")
        return s42.join_external_assets(
            fixed["external_asset_ledger"].value or {}, fixed["external_scope"].value or {}, Path(raw_root),
        )

    def final_reverify() -> Mapping[str, object]:
        final_upstream = upstream_loader(root)
        final_training = seed43_training_validator(root)
        final_authorization = verify_seed43_authorization(root, training=final_training, upstream=final_upstream)
        if (final_upstream.binding_payload() != upstream.binding_payload()
                or final_training.payload() != training.payload()
                or final_authorization.payload() != authorization.payload()):
            raise FailClosedError("seed43/upstream authority changed after forwards")
        observed = addendum_closure(root, s42=s42)
        if observed != authorization.closure:
            raise FailClosedError("seed43 addendum closure changed after forwards")
        return observed

    return run_addendum_lifecycle(
        artifact=artifact, identity=identity, execution_capability=capability, upstream=upstream,
        within_roster=within_roster, external_sessions=external_sessions,
        within_assets_factory=within_assets_factory, external_assets_factory=external_assets_factory,
        backend=backend, final_reverify=final_reverify,
    )
