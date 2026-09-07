"""Fresh Phase-B source-smoke-v2 lifecycle after the immutable v1 failure.

This is deliberately additive.  V1 remains the historical failed attempt;
v2 binds that failure before it reserves a fresh output root, reuses the exact
Cell-D/source/optimizer contract, and changes only the same-prefix direction
recovery implemented in :mod:`source_adapter_v2`.

No public CLI imports this module.  Importing it alone performs no source,
CUDA, remote, checkpoint, result-root, or output operation.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch

from . import core, phase_b, source_adapter, source_adapter_v2
from .plan import CELL, HANDOFF_RELATIVE, HANDOFF_SHA256


class PhaseBV2Error(phase_b.PhaseBError):
    """Fail-closed Phase-B-v2 lifecycle error."""


PHASE_B_V2 = "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V2"
SOURCE_SMOKE_V2_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v2"
V1_FAILURE_ROOT_RELATIVE = phase_b.SOURCE_SMOKE_ROOT_RELATIVE
V1_ATTEMPT_BODY_SHA256 = "4d6283534017938de44450a9cd1160496e1efaa1f91728a2c0d82fb2276454c3"
V1_LAUNCH_BODY_SHA256 = "bce0ac19b3517dfeb4298faedcd322efd6f10248768ce93fef782388f21c2229"
V1_FAILURE_BODY_SHA256 = "a7dfaf466642ca92a945dc545ad0ec0228a0eac180dc3e24ec4d93471977738f"
V1_FAILURE_EXPECTED = {
    "schema": "posterior_carrier_source_smoke_failure_v1",
    "cell": CELL,
    "phase": phase_b.PHASE_B,
    "stage": "prepare",
    "source_opened": True,
    "remote_initialized": False,
    "optimizer_steps_completed": 0,
    "source_authority_sha256": None,
    "terminal_published": False,
    "status": "SOURCE_SMOKE_FAILED_HONESTLY",
}

# The base v1 runtime closure is retained as an explicit dependency rather
# than reusing its permissive glob-free validator on a v2 list.
PHASE_B_V2_CLOSURE_PATHS = phase_b.PHASE_B_CLOSURE_PATHS + (
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v2.py",
    "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke_v2.py",
    "tfpd_exploration/tests/test_posterior_carrier_phase_b_v2.py",
)

SMOKE_TOPOLOGY_V2 = (
    "attempt.json", "launch.json", "source_authority.json", "step100.json", "terminal.json", "failure.json",
)

V1_PREDECESSOR_STAGE_ASSETS = (
    ("attempt.json", V1_ATTEMPT_BODY_SHA256),
    ("launch.json", V1_LAUNCH_BODY_SHA256),
    ("failure.json", V1_FAILURE_BODY_SHA256),
)


def _validate_v1_predecessor_stage_assets() -> tuple[tuple[str, str], ...]:
    """Keep the staged v1 receipts an exact literal authority, never a prefix."""
    expected = (
        ("attempt.json", "4d6283534017938de44450a9cd1160496e1efaa1f91728a2c0d82fb2276454c3"),
        ("launch.json", "bce0ac19b3517dfeb4298faedcd322efd6f10248768ce93fef782388f21c2229"),
        ("failure.json", "a7dfaf466642ca92a945dc545ad0ec0228a0eac180dc3e24ec4d93471977738f"),
    )
    if V1_PREDECESSOR_STAGE_ASSETS != expected:
        raise PhaseBV2Error("v1 predecessor stage-asset literal drift")
    return expected


def _sha(value: object, name: str) -> str:
    try:
        return phase_b._sha(value, name)
    except phase_b.PhaseBError as error:
        raise PhaseBV2Error(str(error)) from error


def _json_bytes(value: object) -> bytes:
    return phase_b.canonical_json_bytes(value)


def _json_sha(value: object) -> str:
    return phase_b.sha256_bytes(_json_bytes(value))


def phase_b_v2_closure(root: Path) -> dict[str, object]:
    """Descriptor-read explicit v2 closure without opening source data."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in PHASE_B_V2_CLOSURE_PATHS:
        body, _identity = phase_b.descriptor_read_stage_file(base, relative)
        hashes[relative] = phase_b.sha256_bytes(body)
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PhaseBV2Error("Phase-B-v2 handoff SHA drift")
    body = {"paths": list(PHASE_B_V2_CLOSURE_PATHS), "sha256_by_path": hashes}
    return {**body, "closure_sha256": _json_sha(body)}


def validate_phase_b_v2_closure(value: Mapping[str, object]) -> dict[str, object]:
    expected = {"paths", "sha256_by_path", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhaseBV2Error("Phase-B-v2 closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(PHASE_B_V2_CLOSURE_PATHS) or not isinstance(hashes, Mapping) or set(hashes) != set(PHASE_B_V2_CLOSURE_PATHS):
        raise PhaseBV2Error("Phase-B-v2 closure paths drift")
    for relative in PHASE_B_V2_CLOSURE_PATHS:
        _sha(hashes[relative], f"Phase-B-v2 closure SHA {relative}")
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PhaseBV2Error("Phase-B-v2 closure handoff binding drift")
    body = {"paths": list(PHASE_B_V2_CLOSURE_PATHS), "sha256_by_path": dict(hashes)}
    if value["closure_sha256"] != _json_sha(body):
        raise PhaseBV2Error("Phase-B-v2 closure digest drift")
    return {**body, "closure_sha256": value["closure_sha256"]}


def base_v1_closure_from_v2(v2_closure: Mapping[str, object]) -> dict[str, object]:
    """Extract and validate the exact v1 runtime subset from a v2 closure."""
    stable = validate_phase_b_v2_closure(v2_closure)
    hashes = stable["sha256_by_path"]
    body = {
        "paths": list(phase_b.PHASE_B_CLOSURE_PATHS),
        "sha256_by_path": {relative: hashes[relative] for relative in phase_b.PHASE_B_CLOSURE_PATHS},
    }
    subset = {**body, "closure_sha256": _json_sha(body)}
    return phase_b.validate_phase_b_closure(subset)


@dataclass(frozen=True)
class V1FailedPredecessor:
    """Immutable lineage assertion for the v1 fail-closed attempt."""

    failure_body_sha256: str = V1_FAILURE_BODY_SHA256

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_carrier_source_smoke_v1_failed_predecessor_v2",
            "root_relative": V1_FAILURE_ROOT_RELATIVE,
            "attempt_body_sha256": V1_ATTEMPT_BODY_SHA256,
            "launch_body_sha256": V1_LAUNCH_BODY_SHA256,
            "failure_body_sha256": self.failure_body_sha256,
            **V1_FAILURE_EXPECTED,
        }

    def validate(self) -> None:
        if self.payload() != {
            "schema": "posterior_carrier_source_smoke_v1_failed_predecessor_v2",
            "root_relative": V1_FAILURE_ROOT_RELATIVE,
            "attempt_body_sha256": V1_ATTEMPT_BODY_SHA256,
            "launch_body_sha256": V1_LAUNCH_BODY_SHA256,
            "failure_body_sha256": V1_FAILURE_BODY_SHA256,
            **V1_FAILURE_EXPECTED,
        }:
            raise PhaseBV2Error("v1 failed-predecessor literal drift")


@dataclass(frozen=True)
class RunIdentityV2:
    """V2 identity nests the target-free v1 source identity exactly."""

    base_identity: phase_b.RunIdentity
    closure: Mapping[str, object]
    predecessor: V1FailedPredecessor = V1FailedPredecessor()

    def payload(self) -> dict[str, object]:
        return {
            "cell": CELL,
            "phase": PHASE_B_V2,
            "handoff": {"path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256},
            "v1_base_source_identity": self.base_identity.payload(),
            "v1_failed_predecessor": self.predecessor.payload(),
            "theta_recovery": {
                "schema": source_adapter_v2.THETA_RECOVERY_SCHEMA,
                "fallback_topology": source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY,
                "fallback_topology_sha256": source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY_SHA256,
                "snap_tolerance_rad": source_adapter_v2.THETA_SNAP_TOLERANCE_RAD,
                "same_prefix_only": True,
                "no_later_row_substitution": True,
            },
            "closure": validate_phase_b_v2_closure(self.closure),
            "remote_device": dict(self.base_identity.remote_device),
            "boundaries": {
                "source_only": True,
                "target_opened": False,
                "within_opened": False,
                "external_opened": False,
                "formal_opened": False,
                "h1_opened": False,
                "target_optimizer_steps": 0,
                "target_backward_calls": 0,
                "target_update_calls": 0,
                "scientific_score": False,
                "v1_result_mutated": False,
            },
        }


def validate_run_identity_v2(identity: RunIdentityV2) -> None:
    if not isinstance(identity, RunIdentityV2):
        raise PhaseBV2Error("Phase-B-v2 identity type drift")
    phase_b.validate_run_identity(identity.base_identity)
    identity.predecessor.validate()
    data = identity.payload()
    if data["handoff"] != {"path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256}:
        raise PhaseBV2Error("Phase-B-v2 identity handoff drift")
    if data["remote_device"] != phase_b.REMOTE_TORCH_AUTHORITY:
        raise PhaseBV2Error("Phase-B-v2 identity remote Torch authority drift")
    if data["v1_base_source_identity"] != identity.base_identity.payload():
        raise PhaseBV2Error("Phase-B-v2 base source identity drift")
    theta = data["theta_recovery"]
    expected_theta = {
        "schema": source_adapter_v2.THETA_RECOVERY_SCHEMA,
        "fallback_topology": source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY,
        "fallback_topology_sha256": source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY_SHA256,
        "snap_tolerance_rad": source_adapter_v2.THETA_SNAP_TOLERANCE_RAD,
        "same_prefix_only": True,
        "no_later_row_substitution": True,
    }
    if theta != expected_theta:
        raise PhaseBV2Error("Phase-B-v2 theta-recovery identity drift")
    subset = base_v1_closure_from_v2(data["closure"])
    if subset != phase_b.validate_phase_b_closure(identity.base_identity.closure):
        raise PhaseBV2Error("Phase-B-v2/v1 base closure cross-binding drift")
    if data["boundaries"] != {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "scientific_score": False, "v1_result_mutated": False,
    }:
        raise PhaseBV2Error("Phase-B-v2 boundaries drift")


def build_target_free_source_identity_v2(
    root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability | None = None,
) -> RunIdentityV2:
    """Metadata-only v2 identity builder; it never opens an NWB or CUDA."""
    base = source_adapter.build_target_free_source_identity(root, source_data=source_data)
    return RunIdentityV2(base_identity=base, closure=phase_b_v2_closure(Path(root).absolute()))


def _open_predecessor_directory(directory: Path) -> tuple[int, tuple[int, int]]:
    """Open one immutable predecessor root and bind its named identity."""
    directory = Path(directory).absolute()
    before = os.lstat(directory)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise PhaseBV2Error("predecessor result root must be a non-symlink directory")
    dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = os.fstat(dfd)
    identity = (int(before.st_dev), int(before.st_ino))
    if (not stat.S_ISDIR(opened.st_mode) or (int(opened.st_dev), int(opened.st_ino)) != identity):
        os.close(dfd)
        raise PhaseBV2Error("predecessor result root changed between lstat/open")
    return dfd, identity


def _recheck_predecessor_directory(directory: Path, dfd: int, identity: tuple[int, int]) -> None:
    """Prove the held root still names the same directory after all reads."""
    held = os.fstat(dfd)
    after = os.lstat(Path(directory).absolute())
    if (not stat.S_ISDIR(held.st_mode) or not stat.S_ISDIR(after.st_mode) or stat.S_ISLNK(after.st_mode)
            or (int(held.st_dev), int(held.st_ino)) != identity
            or (int(after.st_dev), int(after.st_ino)) != identity):
        raise PhaseBV2Error("predecessor result root identity drift during held-FD reads")


def _read_immutable_json_pair_from_fd(
    dfd: int,
    name: str,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], str]:
    """Read one exact mode-0444 JSON/sidecar pair from a held root FD."""
    _sha(expected_sha256, "predecessor expected body SHA")

    def read(leaf: str) -> bytes:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        try:
            leaf_info = os.fstat(fd)
            if not stat.S_ISREG(leaf_info.st_mode) or stat.S_IMODE(leaf_info.st_mode) != 0o444:
                raise PhaseBV2Error("predecessor immutable artifact type/mode drift")
            return phase_b._read_all(fd)
        finally:
            os.close(fd)

    body = read(name)
    digest = phase_b.sha256_bytes(body)
    if digest != expected_sha256:
        raise PhaseBV2Error("predecessor immutable body SHA drift")
    sidecar = read(f"{name}.sha256")
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise PhaseBV2Error("predecessor immutable sidecar drift")
    decoded = json.loads(body)
    if not isinstance(decoded, Mapping):
        raise PhaseBV2Error("predecessor JSON root drift")
    return dict(decoded), digest


def _read_v1_predecessor_pairs(directory: Path) -> dict[str, tuple[dict[str, object], str]]:
    """Read all three v1 receipt pairs under exactly one held directory FD."""
    assets = _validate_v1_predecessor_stage_assets()
    directory = Path(directory).absolute()
    dfd, identity = _open_predecessor_directory(directory)
    try:
        # A failed predecessor cannot be silently paired with a later terminal.
        try:
            os.stat("terminal.json", dir_fd=dfd, follow_symlinks=False)
            terminal_exists = True
        except FileNotFoundError:
            terminal_exists = False
        if terminal_exists:
            raise PhaseBV2Error("v1 failed predecessor unexpectedly contains a terminal receipt")
        pairs = {
            name: _read_immutable_json_pair_from_fd(dfd, name, expected_sha256=digest)
            for name, digest in assets
        }
        _recheck_predecessor_directory(directory, dfd, identity)
        return pairs
    finally:
        os.close(dfd)


def validate_v1_failed_predecessor(root: Path) -> dict[str, object]:
    """Validate the actual immutable v1 failure before v2 may reserve output."""
    base = Path(root).absolute() / V1_FAILURE_ROOT_RELATIVE
    pairs = _read_v1_predecessor_pairs(base)
    attempt, attempt_sha = pairs["attempt.json"]
    launch, launch_sha = pairs["launch.json"]
    failure, failure_sha = pairs["failure.json"]
    if failure_sha != V1_FAILURE_BODY_SHA256:
        raise PhaseBV2Error("v1 failure literal SHA drift")
    identity_value = failure.get("identity")
    if not isinstance(identity_value, Mapping):
        raise PhaseBV2Error("v1 failure identity missing")
    try:
        v1_identity = phase_b.RunIdentity(
            source_authority=identity_value["source_authority"],
            closure=identity_value["closure"],
            remote_device=identity_value["remote_device"],
        )
    except (KeyError, TypeError) as error:
        raise PhaseBV2Error("v1 failure identity schema drift") from error
    phase_b.validate_run_identity(v1_identity)
    if attempt.get("identity") != v1_identity.payload() or launch.get("identity") != v1_identity.payload():
        raise PhaseBV2Error("v1 predecessor attempt/launch identity drift")
    if launch.get("attempt_sha256") != attempt_sha or failure.get("attempt_sha256") != attempt_sha:
        raise PhaseBV2Error("v1 predecessor attempt binding drift")
    if failure.get("launch_sha256") != launch_sha:
        raise PhaseBV2Error("v1 predecessor launch binding drift")
    progress = phase_b.SmokeExecutionProgress(
        source_opened=True, remote_initialized=False, optimizer_steps_completed=0, source_authority_sha256=None,
    )
    phase_b.validate_failure_payload(
        failure, identity=v1_identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, progress=progress,
    )
    for key, expected in V1_FAILURE_EXPECTED.items():
        if failure.get(key) != expected:
            raise PhaseBV2Error(f"v1 predecessor failure {key} drift")
    return {
        "predecessor": V1FailedPredecessor().payload(),
        "attempt_sha256": attempt_sha,
        "launch_sha256": launch_sha,
        "failure_sha256": failure_sha,
        "failure": failure,
    }


def validate_source_authority_v2(
    value: Mapping[str, object],
    *,
    identity: RunIdentityV2,
    launch_sha256: str,
) -> dict[str, object]:
    """Validate v1-compatible physical evidence plus v2 recovery evidence."""
    validate_run_identity_v2(identity)
    required = {
        "schema", "cell", "v1_compatible_authority", "theta_recovery_by_session",
        "theta_fallback_topology", "theta_semantics", "closure", "launch_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PhaseBV2Error("Phase-B-v2 source authority schema drift")
    payload = dict(value)
    if (payload["schema"] != "posterior_carrier_source_authority_v2" or payload["cell"] != CELL
            or payload["theta_semantics"] != source_adapter_v2.THETA_SEMANTICS_V2
            or payload["launch_sha256"] != _sha(launch_sha256, "Phase-B-v2 source authority launch SHA")
            or validate_phase_b_v2_closure(payload["closure"]) != validate_phase_b_v2_closure(identity.closure)):
        raise PhaseBV2Error("Phase-B-v2 source authority immutable binding drift")
    nested = payload["v1_compatible_authority"]
    if not isinstance(nested, Mapping):
        raise PhaseBV2Error("Phase-B-v2 nested v1 authority missing")
    # The nested authority includes exactly the same source-only normalizer,
    # direct count/exposure prefix evidence, Cell-D graph policy, and remote
    # runtime attestation that v1 validated.  V2 cannot weaken any of it.
    phase_b.validate_source_authority_for_identity(
        nested, identity=identity.base_identity, launch_sha256=launch_sha256,
    )
    roster = tuple(identity.base_identity.source_authority["roster"])
    recovery = payload["theta_recovery_by_session"]
    if not isinstance(recovery, Mapping) or set(recovery) != set(roster):
        raise PhaseBV2Error("Phase-B-v2 theta-recovery roster drift")
    inputs = nested["posterior_inputs"]
    for session in roster:
        source_input = inputs[session]
        source_adapter_v2.validate_theta_recovery_evidence(
            recovery[session], session=session,
            prefix_row_ids=source_input["prefix_row_ids"], theta_sha256=source_input["theta_m30_sha256"],
        )
    topology = source_adapter_v2.validate_theta_fallback_topology(payload["theta_fallback_topology"])
    aggregate = source_adapter_v2.theta_fallback_topology(roster=roster, recovery_by_session=recovery)
    if aggregate != topology:
        raise PhaseBV2Error("Phase-B-v2 theta-recovery aggregate/topology drift")
    return payload


class _V2CapabilitySeal:
    __slots__ = ()


_V2_CAPABILITY_SEAL = _V2CapabilitySeal()


@dataclass(frozen=True)
class RootReviewedCapabilityV2:
    """Unforgeable in-process execution capability for v2 only."""

    closure_sha256: str
    predecessor_failure_sha256: str
    _seal: object

    def validate(self, *, closure: Mapping[str, object], predecessor: V1FailedPredecessor) -> None:
        if self._seal is not _V2_CAPABILITY_SEAL:
            raise PhaseBV2Error("Phase-B-v2 source smoke requires root-reviewed capability")
        if (self.closure_sha256 != validate_phase_b_v2_closure(closure)["closure_sha256"]
                or self.predecessor_failure_sha256 != predecessor.failure_body_sha256):
            raise PhaseBV2Error("Phase-B-v2 capability binding drift")


def _issue_root_review_capability_for_v2(*, closure: Mapping[str, object]) -> RootReviewedCapabilityV2:
    """Internal root-only factory; no public CLI can create this capability."""
    stable = validate_phase_b_v2_closure(closure)
    predecessor = V1FailedPredecessor()
    predecessor.validate()
    return RootReviewedCapabilityV2(
        closure_sha256=str(stable["closure_sha256"]),
        predecessor_failure_sha256=predecessor.failure_body_sha256,
        _seal=_V2_CAPABILITY_SEAL,
    )


def reserve_source_smoke_root_v2(root: Path) -> phase_b.ArtifactRoot:
    return phase_b.reserve_source_smoke_root(root, relative=SOURCE_SMOKE_V2_ROOT_RELATIVE)


class SourceSmokeBackendV2(Protocol):
    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: RunIdentityV2) -> Any: ...
    def source_authority(self, runtime: Any, identity: RunIdentityV2) -> Mapping[str, object]: ...
    def run_steps(self, runtime: Any, spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary: ...
    def close(self, runtime: Any | None) -> None: ...


def _attempt_payload(identity: RunIdentityV2) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_source_smoke_attempt_v2", "cell": CELL, "phase": PHASE_B_V2,
        "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(),
        "v1_failed_predecessor": identity.predecessor.payload(),
        "boundaries": identity.payload()["boundaries"], "status": "ATTEMPT_STARTED_SOURCE_ONLY_V2",
    }


def _launch_payload(identity: RunIdentityV2, attempt_sha256: str) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_source_smoke_launch_v2", "cell": CELL, "phase": PHASE_B_V2,
        "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "Phase-B-v2 attempt SHA"),
        "v1_failed_predecessor": identity.predecessor.payload(),
        "launch_closure": validate_phase_b_v2_closure(identity.closure),
        "status": "SOURCE_SMOKE_V2_LAUNCHED",
    }


def _failure_payload(
    *, identity: RunIdentityV2, attempt_sha256: str, launch_sha256: str | None,
    stage: str, error: BaseException, progress: phase_b.SmokeExecutionProgress,
) -> dict[str, object]:
    if stage not in {"prepare", "source_authority", "steps", "terminal"}:
        raise PhaseBV2Error("Phase-B-v2 failure stage drift")
    return {
        "schema": "posterior_carrier_source_smoke_failure_v2", "cell": CELL, "phase": PHASE_B_V2,
        "identity": identity.payload(), "attempt_sha256": _sha(attempt_sha256, "Phase-B-v2 failure attempt SHA"),
        "launch_sha256": launch_sha256, "source_authority_sha256": progress.source_authority_sha256,
        "v1_failed_predecessor": identity.predecessor.payload(), "stage": stage,
        "error_class": type(error).__name__, "error_sha256": phase_b.sha256_bytes(repr(error).encode("utf-8")),
        "source_opened": progress.source_opened, "remote_initialized": progress.remote_initialized,
        "optimizer_steps_completed": progress.optimizer_steps_completed,
        "boundaries": identity.payload()["boundaries"], "terminal_published": False,
        "status": "SOURCE_SMOKE_V2_FAILED_HONESTLY",
    }


def validate_failure_payload_v2(
    value: Mapping[str, object], *, identity: RunIdentityV2, attempt_sha256: str,
    launch_sha256: str | None, progress: phase_b.SmokeExecutionProgress,
) -> dict[str, object]:
    validate_run_identity_v2(identity)
    expected = {
        "schema", "cell", "phase", "identity", "attempt_sha256", "launch_sha256", "source_authority_sha256",
        "v1_failed_predecessor", "stage", "error_class", "error_sha256", "source_opened", "remote_initialized",
        "optimizer_steps_completed", "boundaries", "terminal_published", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhaseBV2Error("Phase-B-v2 failure schema drift")
    payload = dict(value)
    if (payload["schema"] != "posterior_carrier_source_smoke_failure_v2" or payload["cell"] != CELL
            or payload["phase"] != PHASE_B_V2 or payload["identity"] != identity.payload()
            or payload["attempt_sha256"] != _sha(attempt_sha256, "Phase-B-v2 failure attempt SHA")
            or payload["launch_sha256"] != launch_sha256
            or payload["source_authority_sha256"] != progress.source_authority_sha256
            or payload["v1_failed_predecessor"] != identity.predecessor.payload()
            or payload["stage"] not in {"prepare", "source_authority", "steps", "terminal"}
            or not isinstance(payload["error_class"], str) or not payload["error_class"]
            or not isinstance(payload["error_sha256"], str) or len(payload["error_sha256"]) != 64
            or payload["source_opened"] is not progress.source_opened
            or payload["remote_initialized"] is not progress.remote_initialized
            or payload["optimizer_steps_completed"] != progress.optimizer_steps_completed
            or payload["boundaries"] != identity.payload()["boundaries"]
            or payload["terminal_published"] is not False
            or payload["status"] != "SOURCE_SMOKE_V2_FAILED_HONESTLY"):
        raise PhaseBV2Error("Phase-B-v2 failure binding drift")
    if launch_sha256 is not None:
        _sha(launch_sha256, "Phase-B-v2 failure launch SHA")
    if progress.source_authority_sha256 is not None:
        _sha(progress.source_authority_sha256, "Phase-B-v2 failure source authority SHA")
    return payload


def _runtime_progress(runtime: Any | None) -> phase_b.SmokeExecutionProgress:
    return phase_b._runtime_progress(runtime)


def run_source_smoke_lifecycle_v2(
    *, backend: SourceSmokeBackendV2, artifact: phase_b.ArtifactRoot, identity: RunIdentityV2,
    stage_root: Path | None = None,
) -> Mapping[str, object]:
    """Run v2 through injected backend; never invoked by static CLI."""
    validate_run_identity_v2(identity)
    runtime: Any | None = None
    stage = "prepare"
    progress = phase_b.SmokeExecutionProgress()
    launch_sha: str | None = None
    terminal_published = False
    if stage_root is not None and phase_b_v2_closure(Path(stage_root).absolute()) != validate_phase_b_v2_closure(identity.closure):
        raise PhaseBV2Error("Phase-B-v2 live closure drift before attempt")
    attempt_sha = artifact.publish_json("attempt.json", _attempt_payload(identity))
    try:
        launch = _launch_payload(identity, attempt_sha)
        launch_sha = artifact.publish_json("launch.json", launch)
        runtime = backend.prepare(phase_b.SMOKE_SPEC, identity)
        progress = progress.merge(_runtime_progress(runtime))
        stage = "source_authority"
        authority = dict(backend.source_authority(runtime, identity))
        authority["launch_sha256"] = launch_sha
        authority["closure"] = validate_phase_b_v2_closure(identity.closure)
        authority = validate_source_authority_v2(authority, identity=identity, launch_sha256=launch_sha)
        authority_sha = artifact.publish_json("source_authority.json", authority)
        progress = progress.merge(phase_b.SmokeExecutionProgress(
            source_opened=progress.source_opened, remote_initialized=progress.remote_initialized,
            optimizer_steps_completed=progress.optimizer_steps_completed, source_authority_sha256=authority_sha,
        ))
        stage = "steps"
        summary = backend.run_steps(runtime, phase_b.SMOKE_SPEC)
        step = {
            "schema": "posterior_carrier_source_smoke_step100_v2", "cell": CELL, "phase": PHASE_B_V2,
            "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(), "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha, "summary": summary.payload(),
            "v1_failed_predecessor": identity.predecessor.payload(),
            "boundaries": identity.payload()["boundaries"], "status": "SOURCE_SMOKE_V2_100_STEPS_COMPLETE",
        }
        step_sha = artifact.publish_json("step100.json", step)
        progress = progress.merge(phase_b.SmokeExecutionProgress(
            source_opened=progress.source_opened, remote_initialized=progress.remote_initialized,
            optimizer_steps_completed=phase_b.SOURCE_SMOKE_STEPS, source_authority_sha256=authority_sha,
        ))
        stage = "terminal"
        final_closure = phase_b_v2_closure(Path(stage_root).absolute()) if stage_root is not None else validate_phase_b_v2_closure(identity.closure)
        if final_closure != launch["launch_closure"]:
            raise PhaseBV2Error("Phase-B-v2 launch/final closure drift")
        terminal = {
            "schema": "posterior_carrier_source_smoke_terminal_v2", "cell": CELL, "phase": PHASE_B_V2,
            "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(),
            "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha, "step100_sha256": step_sha,
            "v1_failed_predecessor": identity.predecessor.payload(),
            "launch_closure": launch["launch_closure"], "final_closure": final_closure,
            "boundaries": identity.payload()["boundaries"],
            "status": "SOURCE_SMOKE_V2_COMPLETE__NON_AUTHORITATIVE",
        }
        terminal_sha = artifact.publish_json("terminal.json", terminal)
        terminal_published = True
        if artifact.reload_json("terminal.json", terminal_sha) != terminal:
            raise PhaseBV2Error("Phase-B-v2 terminal reload drift")
        return terminal
    except BaseException as error:
        if isinstance(error, phase_b.SourceSmokeExecutionError):
            progress = progress.merge(error.progress)
            stage = error.stage
            cause: BaseException = error.cause
        else:
            progress = progress.merge(_runtime_progress(runtime))
            cause = error
        if not terminal_published:
            try:
                failure = _failure_payload(
                    identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                    stage=stage, error=cause, progress=progress,
                )
                failure = validate_failure_payload_v2(
                    failure, identity=identity, attempt_sha256=attempt_sha,
                    launch_sha256=launch_sha, progress=progress,
                )
                if launch_sha is not None:
                    artifact.reload_json("launch.json", launch_sha)
                if progress.source_authority_sha256 is not None:
                    artifact.reload_json("source_authority.json", progress.source_authority_sha256)
                failure_sha = artifact.publish_json("failure.json", failure)
                if artifact.reload_json("failure.json", failure_sha) != failure:
                    raise PhaseBV2Error("Phase-B-v2 failure reload drift")
            except BaseException:
                pass
        raise
    finally:
        backend.close(runtime)


def reviewed_remote_source_smoke_v2(
    *, root: Path, capability: RootReviewedCapabilityV2, backend: SourceSmokeBackendV2,
    identity: RunIdentityV2, source_data: phase_b.SourceDataRootCapability,
) -> Mapping[str, object]:
    """Execution-only v2 route; root review must issue the capability."""
    validate_run_identity_v2(identity)
    if source_data.payload() != identity.base_identity.source_authority.get("source_data_root"):
        raise PhaseBV2Error("Phase-B-v2 external source-data capability drift")
    phase_b.validate_stage_source_separation(stage_root=Path(root).absolute(), source_data=source_data)
    live = phase_b_v2_closure(Path(root).absolute())
    if live != validate_phase_b_v2_closure(identity.closure):
        raise PhaseBV2Error("Phase-B-v2 live closure drift before predecessor validation")
    capability.validate(closure=live, predecessor=identity.predecessor)
    # This read happens before v2 output reservation and before source/CUDA.
    validate_v1_failed_predecessor(Path(root).absolute())
    artifact = reserve_source_smoke_root_v2(Path(root).absolute())
    return run_source_smoke_lifecycle_v2(backend=backend, artifact=artifact, identity=identity, stage_root=Path(root).absolute())


def remote_staging_plan_v2(
    *,
    closure: Mapping[str, object],
    source_authority_sha256: str,
    source_data: phase_b.SourceDataRootCapability,
    remote_root_name: str = "posterior_carrier_budgetmix_d_seed42_stage_v2",
) -> dict[str, object]:
    """Pure non-glob v2 staging manifest; no ssh/socket/output action."""
    if (not isinstance(remote_root_name, str) or not remote_root_name
            or "/" in remote_root_name or ".." in remote_root_name):
        raise PhaseBV2Error("Phase-B-v2 remote staging root name must be one safe component")
    stable = validate_phase_b_v2_closure(closure)
    predecessor_assets = _validate_v1_predecessor_stage_assets()
    for relative, expected_sha, _mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS:
        if stable["sha256_by_path"].get(relative) != expected_sha:
            raise PhaseBV2Error("Phase-B-v2 staging authority asset drift")
    mode_by_path = {relative: mode for relative, _sha_value, mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS}
    stage_files = [
        {
            "relative_path": relative,
            "sha256": stable["sha256_by_path"][relative],
            "role": "immutable_authority" if relative in phase_b.SOURCE_AUTHORITY_ASSET_PATHS else "closure_dependency",
            **({"mode": mode_by_path[relative]} if mode_by_path.get(relative) is not None else {}),
        }
        for relative in PHASE_B_V2_CLOSURE_PATHS
    ]
    for relative, expected_sha, expected_mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS:
        if expected_mode == 0o444:
            stage_files.append({
                "relative_path": f"{relative}.sha256",
                "contents": f"{expected_sha}  {Path(relative).name}\n",
                "mode": 0o444,
                "role": "immutable_authority_sidecar",
            })
    # The v2 reviewed route must descriptor-read/revalidate the v1 failed
    # attempt before it reserves its fresh successor root.  These six files
    # are therefore first-class staged immutable assets, not an out-of-band
    # assumption about a historical remote checkout.
    for name, digest in predecessor_assets:
        relative = f"{V1_FAILURE_ROOT_RELATIVE}/{name}"
        stage_files.append({
            "relative_path": relative,
            "sha256": digest,
            "mode": 0o444,
            "role": "immutable_v1_failed_predecessor_receipt",
        })
        stage_files.append({
            "relative_path": f"{relative}.sha256",
            "contents": f"{digest}  {name}\n",
            "mode": 0o444,
            "role": "immutable_v1_failed_predecessor_sidecar",
        })
    return {
        "schema": "posterior_carrier_remote_staging_plan_v2",
        "cell": CELL,
        "phase": PHASE_B_V2,
        "remote_host": "xinyuan@100.103.97.12",
        "remote_root_name": remote_root_name,
        "closure": stable,
        "source_authority_sha256": _sha(source_authority_sha256, "Phase-B-v2 staging source authority SHA"),
        "v1_failed_predecessor": V1FailedPredecessor().payload(),
        "stage_files": stage_files,
        "external_source_data_root": phase_b.validate_source_data_root_payload(source_data.payload()),
        "nwb_assets_in_stage": False,
        "remote_torch_authority": dict(phase_b.REMOTE_TORCH_AUTHORITY),
        "forbidden": ["target", "within", "external", "formal", "h1", "teacher", "remote_historical_repo_mutation"],
        "launch": "NOT_AUTHORIZED_BY_PLAN; requires a fresh in-process root-reviewed capability",
    }


class RemotePosteriorSourceSmokeBackendV2(source_adapter.RemotePosteriorSourceSmokeBackend):
    """Deferred physical v2 backend; source/cache/CUDA work stays in ``prepare``."""

    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: RunIdentityV2) -> source_adapter._PhysicalSmokeRuntime:
        if spec != phase_b.SMOKE_SPEC:
            raise PhaseBV2Error("physical Phase-B-v2 backend accepts only fixed 100-step source smoke")
        validate_run_identity_v2(identity)
        if self.source_data.payload() != identity.base_identity.source_authority.get("source_data_root"):
            raise PhaseBV2Error("Phase-B-v2 launch/source-data capability drift")
        self.source_data.validate()
        if self.num_workers != 4:
            raise PhaseBV2Error("Phase-B-v2 holds predecessor worker count at four")
        live_v2 = phase_b_v2_closure(self.root)
        if live_v2 != validate_phase_b_v2_closure(identity.closure):
            raise PhaseBV2Error("physical Phase-B-v2 live closure drift")
        base_v1 = base_v1_closure_from_v2(live_v2)
        progress = phase_b.SmokeExecutionProgress()

        def mark_source_opened() -> None:
            nonlocal progress
            progress = progress.merge(phase_b.SmokeExecutionProgress(source_opened=True))

        try:
            adapter = source_adapter_v2.build_physical_source_adapter_v2(
                self.root, source_data=self.source_data, num_workers=self.num_workers,
                on_source_opened=mark_source_opened,
            )
            if tuple(adapter.roster) != tuple(identity.base_identity.source_authority.get("roster", ())):
                raise PhaseBV2Error("Phase-B-v2 source strict-27 roster drift")
            # `source_authority_metadata` remains v1 byte-compatible.  The
            # v2-specific field is the receipt-bound theta recovery only.
            if identity.base_identity.source_authority.get("strict_source_metadata_sha256") != source_adapter._json_sha(adapter.source_authority_metadata):
                raise PhaseBV2Error("Phase-B-v2 strict source metadata drift")
            import numpy as np
            import random
            from torch.utils.data import DataLoader
            arm_common, pop_robust = source_adapter.load_stage_runtime_helpers(self.root, closure=base_v1)
            if not torch.cuda.is_available():
                raise PhaseBV2Error("Phase-B-v2 source smoke requires one visible CUDA device")
            progress = progress.merge(phase_b.SmokeExecutionProgress(
                source_opened=progress.source_opened, remote_initialized=True,
            ))
            attestation = phase_b.attest_remote_torch_only(torch, nvml_status=self.nvml_status)
            if torch.is_autocast_enabled() or bool(torch.backends.cuda.matmul.allow_tf32) or bool(torch.backends.cudnn.allow_tf32):
                raise PhaseBV2Error("Phase-B-v2 forbids AMP/TF32")
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            model = pop_robust.build_population_robustness_model(seed=42, cell="D")
            wrapper = core.CellDPosteriorWrapper(model)
            preservation = wrapper.preservation_audit()
            if (preservation.base_live_parameter_count != 3_510_842
                    or preservation.wrapper_new_parameter_count != 0
                    or preservation.dynamic_dropout is not True):
                raise PhaseBV2Error("Phase-B-v2 Cell-D graph/parameter/dropout preservation drift")
            device = torch.device("cuda:0")
            wrapper.to(device)
            optimizer = torch.optim.Adam(
                wrapper.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=0.0, amsgrad=False,
            )
            cache_started = __import__("time").monotonic()
            adapter.prewarm_epoch(0)
            adapter.materialize_epoch_for_device(epoch=0, device=device)
            posterior_prepare_seconds = adapter.preparation_seconds + (__import__("time").monotonic() - cache_started)
            loader = DataLoader(adapter.dataset, batch_sampler=adapter.sampler, num_workers=self.num_workers, pin_memory=True)
            iterator = iter(loader)
            torch.cuda.reset_peak_memory_stats(0)
            return source_adapter._PhysicalSmokeRuntime(
                torch=torch, adapter=adapter, wrapper=wrapper, optimizer=optimizer,
                arm_common=arm_common, pop_robust=pop_robust, iterator=iterator, device=device,
                remote_device=dict(attestation.payload), posterior_prepare_seconds=posterior_prepare_seconds,
                progress=progress,
            )
        except BaseException as error:
            if isinstance(error, phase_b.SourceSmokeExecutionError):
                raise
            raise phase_b.SourceSmokeExecutionError(stage="prepare", progress=progress, cause=error) from error

    def source_authority(self, runtime: source_adapter._PhysicalSmokeRuntime, identity: RunIdentityV2) -> Mapping[str, object]:
        base_v1 = base_v1_closure_from_v2(identity.closure)
        nested = source_adapter.PhysicalPosteriorSourceAdapter.source_authority_payload(runtime.adapter, closure=base_v1)
        nested.update({
            "posterior_preparation_seconds": runtime.posterior_prepare_seconds,
            "remote_torch_authority": dict(runtime.remote_device),
            "posterior_credibility_statistics": source_adapter._posterior_statistics(runtime.adapter.bank),
            "optimizer": {
                "class": "Adam", "lr_constructor": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0.0, "amsgrad": False, "schedule": "arm_common.lr_at_step(48,33925)",
            },
            "execution_policy": {"amp": False, "tf32": False, "torch_compile": False, "batch_size": 32},
        })
        return source_adapter_v2.wrap_v1_authority_with_theta_recovery(
            adapter=runtime.adapter, nested_v1_authority=nested, v2_closure=identity.closure,
        )

    def run_steps(self, runtime: source_adapter._PhysicalSmokeRuntime, spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary:
        # Frozen v1 step core already holds Cell-D graph, dense loss, Adam/LR,
        # no-inverse-in-loop accounting, and exact progress semantics.  It is
        # invoked only after the v2 carrier adapter has supplied its static
        # epoch-0 posterior view.
        return source_adapter.RemotePosteriorSourceSmokeBackend.run_steps(self, runtime, spec)
