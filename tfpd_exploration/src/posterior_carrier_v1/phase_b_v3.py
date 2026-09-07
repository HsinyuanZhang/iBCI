"""Fresh Phase-B source-smoke-v3 lifecycle with explicit TF32 enforcement.

V1 and V2 are immutable failed predecessors.  V3 is deliberately additive:
it retains the reviewed V2 source adapter and Cell-D graph, but enforces the
two CUDA TF32 switches *inside* the reviewed physical route before V2 can
construct a model or an optimizer.  A true pre-state is disclosed, whereas
acceptance is determined only by the explicitly enforced post-state.

Importing this module is inert: it does not open source data, contact a
remote host, initialize CUDA, reserve an output root, or write a receipt.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch

from . import phase_b, phase_b_v2
from .plan import CELL, HANDOFF_RELATIVE, HANDOFF_SHA256


class PhaseBV3Error(phase_b.PhaseBError):
    """Fail-closed Phase-B-v3 lifecycle error."""


PHASE_B_V3 = "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V3"
SOURCE_SMOKE_V3_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v3"

# V3 may proceed only from the exact accepted V2 code closure.  It must not
# silently turn a later, internally self-consistent V2 checkout into lineage.
ACCEPTED_PHASE_B_V2_CLOSURE_SHA256 = "9a692c704f2c31c470e9ee465307c9bb1ea4ec2b73b0d156218e06b387abb853"

V1_FAILURE_ROOT_RELATIVE = phase_b_v2.V1_FAILURE_ROOT_RELATIVE
V2_FAILURE_ROOT_RELATIVE = phase_b_v2.SOURCE_SMOKE_V2_ROOT_RELATIVE

V1_PREDECESSOR_ASSETS = phase_b_v2.V1_PREDECESSOR_STAGE_ASSETS
V2_ATTEMPT_BODY_SHA256 = "e93e1a59724d06abcc8831c3b0d4f49f81f7eecb451c6e429e96f1f70d6cea24"
V2_LAUNCH_BODY_SHA256 = "fadd0c94a35a416f0c4a6d527be1ee9a85368d845d64ab64a20d2028c8ba3416"
V2_FAILURE_BODY_SHA256 = "d8c5a4eac84d14187163991209b486433a9851496f7a336b9e5a3dcd7243a2af"
V2_PREDECESSOR_ASSETS = (
    ("attempt.json", V2_ATTEMPT_BODY_SHA256),
    ("launch.json", V2_LAUNCH_BODY_SHA256),
    ("failure.json", V2_FAILURE_BODY_SHA256),
)
V2_FAILURE_EXPECTED = {
    "schema": "posterior_carrier_source_smoke_failure_v2",
    "cell": CELL,
    "phase": phase_b_v2.PHASE_B_V2,
    "stage": "prepare",
    "source_opened": True,
    "remote_initialized": True,
    "optimizer_steps_completed": 0,
    "source_authority_sha256": None,
    "terminal_published": False,
    "status": "SOURCE_SMOKE_V2_FAILED_HONESTLY",
    "error_class": "PhaseBV2Error",
    "error_sha256": "935fc4d0b5cb14b8d78fe4f36c919950c3332e4a6d7cec02f306d41d6e538a91",
}

PHASE_B_V3_CLOSURE_PATHS = phase_b_v2.PHASE_B_V2_CLOSURE_PATHS + (
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py",
    "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke_v3.py",
    "tfpd_exploration/tests/test_posterior_carrier_phase_b_v3.py",
)
SMOKE_TOPOLOGY_V3 = (
    "attempt.json", "launch.json", "source_authority.json", "step100.json", "terminal.json", "failure.json",
)

TF32_POLICY_SCHEMA = "posterior_carrier_tf32_enforcement_v3"


def _sha(value: object, name: str) -> str:
    try:
        return phase_b._sha(value, name)
    except phase_b.PhaseBError as error:
        raise PhaseBV3Error(str(error)) from error


def _json_bytes(value: object) -> bytes:
    return phase_b.canonical_json_bytes(value)


def _json_sha(value: object) -> str:
    return phase_b.sha256_bytes(_json_bytes(value))


def phase_b_v3_closure(root: Path) -> dict[str, object]:
    """Descriptor-read the explicit V3 closure, never source data."""
    stage_root = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in PHASE_B_V3_CLOSURE_PATHS:
        body, _identity = phase_b.descriptor_read_stage_file(stage_root, relative)
        hashes[relative] = phase_b.sha256_bytes(body)
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PhaseBV3Error("Phase-B-v3 handoff SHA drift")
    body = {"paths": list(PHASE_B_V3_CLOSURE_PATHS), "sha256_by_path": hashes}
    stable = {**body, "closure_sha256": _json_sha(body)}
    _v2_closure_from_v3(stable)
    return stable


def _v2_closure_from_v3(closure: Mapping[str, object]) -> dict[str, object]:
    """Extract the exact V2 subset and bind it to the reviewed V2 closure."""
    if not isinstance(closure, Mapping):
        raise PhaseBV3Error("Phase-B-v3 closure type drift")
    hashes = closure.get("sha256_by_path")
    if not isinstance(hashes, Mapping):
        raise PhaseBV3Error("Phase-B-v3 closure lacks path hashes")
    body = {
        "paths": list(phase_b_v2.PHASE_B_V2_CLOSURE_PATHS),
        "sha256_by_path": {relative: hashes[relative] for relative in phase_b_v2.PHASE_B_V2_CLOSURE_PATHS},
    }
    v2 = {**body, "closure_sha256": _json_sha(body)}
    v2 = phase_b_v2.validate_phase_b_v2_closure(v2)
    if v2["closure_sha256"] != ACCEPTED_PHASE_B_V2_CLOSURE_SHA256:
        raise PhaseBV3Error("Phase-B-v3 requires the accepted immutable V2 closure")
    return v2


def validate_phase_b_v3_closure(value: Mapping[str, object]) -> dict[str, object]:
    expected = {"paths", "sha256_by_path", "closure_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhaseBV3Error("Phase-B-v3 closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(PHASE_B_V3_CLOSURE_PATHS) or not isinstance(hashes, Mapping) or set(hashes) != set(PHASE_B_V3_CLOSURE_PATHS):
        raise PhaseBV3Error("Phase-B-v3 closure path topology drift")
    for relative in PHASE_B_V3_CLOSURE_PATHS:
        _sha(hashes[relative], f"Phase-B-v3 closure SHA {relative}")
    if hashes[HANDOFF_RELATIVE] != HANDOFF_SHA256:
        raise PhaseBV3Error("Phase-B-v3 closure handoff binding drift")
    body = {"paths": list(PHASE_B_V3_CLOSURE_PATHS), "sha256_by_path": dict(hashes)}
    if value["closure_sha256"] != _json_sha(body):
        raise PhaseBV3Error("Phase-B-v3 closure digest drift")
    stable = {**body, "closure_sha256": value["closure_sha256"]}
    _v2_closure_from_v3(stable)
    return stable


@dataclass(frozen=True)
class V2FailedPredecessor:
    """Exact immutable lineage assertion for the TF32 V2 failure."""

    failure_body_sha256: str = V2_FAILURE_BODY_SHA256

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_carrier_source_smoke_v2_failed_predecessor_v3",
            "root_relative": V2_FAILURE_ROOT_RELATIVE,
            "attempt_body_sha256": V2_ATTEMPT_BODY_SHA256,
            "launch_body_sha256": V2_LAUNCH_BODY_SHA256,
            "failure_body_sha256": self.failure_body_sha256,
            **V2_FAILURE_EXPECTED,
        }

    def validate(self) -> None:
        expected = {
            "schema": "posterior_carrier_source_smoke_v2_failed_predecessor_v3",
            "root_relative": V2_FAILURE_ROOT_RELATIVE,
            "attempt_body_sha256": V2_ATTEMPT_BODY_SHA256,
            "launch_body_sha256": V2_LAUNCH_BODY_SHA256,
            "failure_body_sha256": V2_FAILURE_BODY_SHA256,
            **V2_FAILURE_EXPECTED,
        }
        if self.payload() != expected:
            raise PhaseBV3Error("v2 failed-predecessor literal drift")


@dataclass(frozen=True)
class RunIdentityV3:
    """V3 identity nests the exact reviewed V2 source identity."""

    base_identity: phase_b_v2.RunIdentityV2
    closure: Mapping[str, object]
    predecessor: V2FailedPredecessor = V2FailedPredecessor()

    def payload(self) -> dict[str, object]:
        return {
            "cell": CELL,
            "phase": PHASE_B_V3,
            "handoff": {"path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256},
            "v2_source_identity": self.base_identity.payload(),
            "v1_failed_predecessor": self.base_identity.predecessor.payload(),
            "v2_failed_predecessor": self.predecessor.payload(),
            "closure": validate_phase_b_v3_closure(self.closure),
            "remote_device": dict(self.base_identity.base_identity.remote_device),
            "tf32_contract": {
                "amp": False,
                "cuda_matmul_allow_tf32_enforced": False,
                "cudnn_allow_tf32_enforced": False,
                "enforcement_before_model_and_optimizer": True,
                "pre_state_disclosed_not_gating": True,
                "route_local_restore": True,
            },
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
                "v2_result_mutated": False,
            },
        }


def validate_run_identity_v3(identity: RunIdentityV3) -> None:
    if not isinstance(identity, RunIdentityV3):
        raise PhaseBV3Error("Phase-B-v3 identity type drift")
    phase_b_v2.validate_run_identity_v2(identity.base_identity)
    identity.predecessor.validate()
    payload = identity.payload()
    if payload["handoff"] != {"path": HANDOFF_RELATIVE, "sha256": HANDOFF_SHA256}:
        raise PhaseBV3Error("Phase-B-v3 handoff binding drift")
    if payload["remote_device"] != phase_b.REMOTE_TORCH_AUTHORITY:
        raise PhaseBV3Error("Phase-B-v3 remote Torch authority drift")
    if payload["v2_source_identity"] != identity.base_identity.payload():
        raise PhaseBV3Error("Phase-B-v3 V2 identity binding drift")
    if payload["v1_failed_predecessor"] != identity.base_identity.predecessor.payload():
        raise PhaseBV3Error("Phase-B-v3 V1 predecessor binding drift")
    if payload["v2_failed_predecessor"] != identity.predecessor.payload():
        raise PhaseBV3Error("Phase-B-v3 V2 predecessor binding drift")
    v2 = _v2_closure_from_v3(payload["closure"])
    if v2 != phase_b_v2.validate_phase_b_v2_closure(identity.base_identity.closure):
        raise PhaseBV3Error("Phase-B-v3/V2 closure cross-binding drift")
    if payload["tf32_contract"] != {
        "amp": False,
        "cuda_matmul_allow_tf32_enforced": False,
        "cudnn_allow_tf32_enforced": False,
        "enforcement_before_model_and_optimizer": True,
        "pre_state_disclosed_not_gating": True,
        "route_local_restore": True,
    }:
        raise PhaseBV3Error("Phase-B-v3 TF32 contract drift")
    if payload["boundaries"] != {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "scientific_score": False,
        "v1_result_mutated": False, "v2_result_mutated": False,
    }:
        raise PhaseBV3Error("Phase-B-v3 target-boundary drift")


def build_target_free_source_identity_v3(
    root: Path,
    *,
    source_data: phase_b.SourceDataRootCapability | None = None,
) -> RunIdentityV3:
    """Metadata-only identity builder; source data and CUDA stay unopened."""
    base = phase_b_v2.build_target_free_source_identity_v2(root, source_data=source_data)
    return RunIdentityV3(base_identity=base, closure=phase_b_v3_closure(Path(root).absolute()))


def _read_immutable_pairs_under_held_root(
    directory: Path,
    assets: tuple[tuple[str, str], ...],
) -> dict[str, tuple[dict[str, object], str]]:
    """Read every named predecessor pair through one O_NOFOLLOW root FD."""
    root = Path(directory).absolute()
    try:
        dfd, identity = phase_b_v2._open_predecessor_directory(root)
    except phase_b_v2.PhaseBV2Error as error:
        raise PhaseBV3Error(str(error)) from error
    try:
        try:
            try:
                os.stat("terminal.json", dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise PhaseBV3Error("failed predecessor unexpectedly contains terminal.json")
            pairs = {
                name: phase_b_v2._read_immutable_json_pair_from_fd(dfd, name, expected_sha256=digest)
                for name, digest in assets
            }
            phase_b_v2._recheck_predecessor_directory(root, dfd, identity)
            return pairs
        except phase_b_v2.PhaseBV2Error as error:
            raise PhaseBV3Error(str(error)) from error
    finally:
        os.close(dfd)


def _v1_identity_from_payload(value: Mapping[str, object]) -> phase_b.RunIdentity:
    try:
        identity = phase_b.RunIdentity(
            source_authority=value["source_authority"], closure=value["closure"], remote_device=value["remote_device"],
        )
    except (KeyError, TypeError) as error:
        raise PhaseBV3Error("v1 predecessor identity schema drift") from error
    phase_b.validate_run_identity(identity)
    if dict(value) != identity.payload():
        raise PhaseBV3Error("v1 predecessor identity exact-payload drift")
    return identity


def _v2_identity_from_payload(value: Mapping[str, object]) -> phase_b_v2.RunIdentityV2:
    try:
        base = _v1_identity_from_payload(value["v1_base_source_identity"])
        identity = phase_b_v2.RunIdentityV2(base_identity=base, closure=value["closure"])
    except (KeyError, TypeError) as error:
        raise PhaseBV3Error("v2 predecessor identity schema drift") from error
    phase_b_v2.validate_run_identity_v2(identity)
    if dict(value) != identity.payload():
        raise PhaseBV3Error("v2 predecessor identity exact-payload drift")
    if identity.closure["closure_sha256"] != ACCEPTED_PHASE_B_V2_CLOSURE_SHA256:
        raise PhaseBV3Error("v2 predecessor did not use accepted V2 closure")
    return identity


def _validate_v1_predecessor_pairs(pairs: Mapping[str, tuple[dict[str, object], str]]) -> dict[str, object]:
    attempt, attempt_sha = pairs["attempt.json"]
    launch, launch_sha = pairs["launch.json"]
    failure, failure_sha = pairs["failure.json"]
    identity_value = failure.get("identity")
    if not isinstance(identity_value, Mapping):
        raise PhaseBV3Error("v1 predecessor failure identity missing")
    identity = _v1_identity_from_payload(identity_value)
    phase_b.validate_run_identity(identity)
    if attempt.get("identity") != identity.payload() or launch.get("identity") != identity.payload():
        raise PhaseBV3Error("v1 predecessor attempt/launch identity drift")
    if launch.get("attempt_sha256") != attempt_sha or failure.get("attempt_sha256") != attempt_sha:
        raise PhaseBV3Error("v1 predecessor attempt binding drift")
    if failure.get("launch_sha256") != launch_sha:
        raise PhaseBV3Error("v1 predecessor launch binding drift")
    progress = phase_b.SmokeExecutionProgress(
        source_opened=True, remote_initialized=False, optimizer_steps_completed=0, source_authority_sha256=None,
    )
    phase_b.validate_failure_payload(
        failure, identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, progress=progress,
    )
    for key, expected in phase_b_v2.V1_FAILURE_EXPECTED.items():
        if failure.get(key) != expected:
            raise PhaseBV3Error(f"v1 predecessor failure {key} drift")
    return {"attempt_sha256": attempt_sha, "launch_sha256": launch_sha, "failure_sha256": failure_sha}


def _validate_v2_predecessor_pairs(pairs: Mapping[str, tuple[dict[str, object], str]]) -> dict[str, object]:
    attempt, attempt_sha = pairs["attempt.json"]
    launch, launch_sha = pairs["launch.json"]
    failure, failure_sha = pairs["failure.json"]
    identity_value = failure.get("identity")
    if not isinstance(identity_value, Mapping):
        raise PhaseBV3Error("v2 predecessor failure identity missing")
    identity = _v2_identity_from_payload(identity_value)
    if attempt.get("identity") != identity.payload() or launch.get("identity") != identity.payload():
        raise PhaseBV3Error("v2 predecessor attempt/launch identity drift")
    if launch.get("attempt_sha256") != attempt_sha or failure.get("attempt_sha256") != attempt_sha:
        raise PhaseBV3Error("v2 predecessor attempt binding drift")
    if failure.get("launch_sha256") != launch_sha:
        raise PhaseBV3Error("v2 predecessor launch binding drift")
    progress = phase_b.SmokeExecutionProgress(
        source_opened=True, remote_initialized=True, optimizer_steps_completed=0, source_authority_sha256=None,
    )
    phase_b_v2.validate_failure_payload_v2(
        failure, identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, progress=progress,
    )
    for key, expected in V2_FAILURE_EXPECTED.items():
        if failure.get(key) != expected:
            raise PhaseBV3Error(f"v2 predecessor failure {key} drift")
    return {"attempt_sha256": attempt_sha, "launch_sha256": launch_sha, "failure_sha256": failure_sha}


def validate_failed_predecessor_chain(root: Path) -> dict[str, object]:
    """Bind all six V1/V2 predecessor receipt pairs before V3 reservation."""
    base = Path(root).absolute()
    v1_pairs = _read_immutable_pairs_under_held_root(base / V1_FAILURE_ROOT_RELATIVE, V1_PREDECESSOR_ASSETS)
    v2_pairs = _read_immutable_pairs_under_held_root(base / V2_FAILURE_ROOT_RELATIVE, V2_PREDECESSOR_ASSETS)
    return {
        "schema": "posterior_carrier_source_smoke_failed_predecessor_chain_v3",
        "v1": _validate_v1_predecessor_pairs(v1_pairs),
        "v2": _validate_v2_predecessor_pairs(v2_pairs),
    }


@dataclass(frozen=True)
class TF32Enforcement:
    """Observed and explicitly enforced CUDA backend precision policy."""

    pre_matmul_allow_tf32: bool
    pre_cudnn_allow_tf32: bool
    pre_amp_enabled: bool
    post_matmul_allow_tf32: bool
    post_cudnn_allow_tf32: bool
    post_amp_enabled: bool

    @classmethod
    def enforce(cls, torch_module: Any) -> "TF32Enforcement":
        """Set both TF32 switches before V2 can build model/optimizer."""
        try:
            pre_matmul = bool(torch_module.backends.cuda.matmul.allow_tf32)
            pre_cudnn = bool(torch_module.backends.cudnn.allow_tf32)
            pre_amp = bool(torch_module.is_autocast_enabled())
        except BaseException as error:
            raise PhaseBV3Error("unable to observe pre-enforcement AMP/TF32 state") from error
        # Autocast is not a mutable global knob to reset safely here.  It must
        # already be off; TF32 defaults are explicitly overridden instead.
        if pre_amp:
            raise PhaseBV3Error("Phase-B-v3 forbids AMP/autocast")
        torch_module.backends.cuda.matmul.allow_tf32 = False
        torch_module.backends.cudnn.allow_tf32 = False
        post = cls(
            pre_matmul_allow_tf32=pre_matmul,
            pre_cudnn_allow_tf32=pre_cudnn,
            pre_amp_enabled=pre_amp,
            post_matmul_allow_tf32=bool(torch_module.backends.cuda.matmul.allow_tf32),
            post_cudnn_allow_tf32=bool(torch_module.backends.cudnn.allow_tf32),
            post_amp_enabled=bool(torch_module.is_autocast_enabled()),
        )
        validate_tf32_enforcement(post.payload())
        return post

    def restore(self, torch_module: Any) -> None:
        """Restore only the two route-local switches after this route closes."""
        torch_module.backends.cuda.matmul.allow_tf32 = self.pre_matmul_allow_tf32
        torch_module.backends.cudnn.allow_tf32 = self.pre_cudnn_allow_tf32

    def payload(self) -> dict[str, object]:
        return {
            "schema": TF32_POLICY_SCHEMA,
            "observed_pre_state": {
                "cuda_matmul_allow_tf32": self.pre_matmul_allow_tf32,
                "cudnn_allow_tf32": self.pre_cudnn_allow_tf32,
                "amp_enabled": self.pre_amp_enabled,
            },
            "enforced_post_state": {
                "cuda_matmul_allow_tf32": self.post_matmul_allow_tf32,
                "cudnn_allow_tf32": self.post_cudnn_allow_tf32,
                "amp_enabled": self.post_amp_enabled,
            },
            "enforced_before_model_construction": True,
            "enforced_before_optimizer_construction": True,
            "route_local_restore_on_close": True,
            "acceptance_uses_enforced_post_state": True,
        }


def validate_tf32_enforcement(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema", "observed_pre_state", "enforced_post_state", "enforced_before_model_construction",
        "enforced_before_optimizer_construction", "route_local_restore_on_close", "acceptance_uses_enforced_post_state",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != TF32_POLICY_SCHEMA:
        raise PhaseBV3Error("TF32 enforcement schema drift")
    pre, post = value.get("observed_pre_state"), value.get("enforced_post_state")
    fields = {"cuda_matmul_allow_tf32", "cudnn_allow_tf32", "amp_enabled"}
    if (not isinstance(pre, Mapping) or not isinstance(post, Mapping) or set(pre) != fields or set(post) != fields
            or any(type(pre[item]) is not bool for item in fields)
            or any(type(post[item]) is not bool for item in fields)):
        raise PhaseBV3Error("TF32 enforcement state schema drift")
    # Pre-state is disclosure only.  V3 accepts a true default only after it
    # explicitly proves both switches and AMP are false in the post-state.
    if any(post[item] is not False for item in fields):
        raise PhaseBV3Error("TF32 enforcement post-state must disable AMP and both TF32 paths")
    if (value.get("enforced_before_model_construction") is not True
            or value.get("enforced_before_optimizer_construction") is not True
            or value.get("route_local_restore_on_close") is not True
            or value.get("acceptance_uses_enforced_post_state") is not True):
        raise PhaseBV3Error("TF32 enforcement order/acceptance proof drift")
    return dict(value)


class _V3CapabilitySeal:
    __slots__ = ()


_V3_CAPABILITY_SEAL = _V3CapabilitySeal()


@dataclass(frozen=True)
class RootReviewedCapabilityV3:
    """Unforgeable in-process capability for the fresh V3 route."""

    closure_sha256: str
    v2_failure_sha256: str
    _seal: object

    def validate(self, *, closure: Mapping[str, object], predecessor: V2FailedPredecessor) -> None:
        if self._seal is not _V3_CAPABILITY_SEAL:
            raise PhaseBV3Error("Phase-B-v3 requires an in-process root-reviewed capability")
        if (self.closure_sha256 != validate_phase_b_v3_closure(closure)["closure_sha256"]
                or self.v2_failure_sha256 != predecessor.failure_body_sha256):
            raise PhaseBV3Error("Phase-B-v3 capability binding drift")


def _issue_root_review_capability_for_v3(*, closure: Mapping[str, object]) -> RootReviewedCapabilityV3:
    stable = validate_phase_b_v3_closure(closure)
    predecessor = V2FailedPredecessor()
    predecessor.validate()
    return RootReviewedCapabilityV3(
        closure_sha256=str(stable["closure_sha256"]),
        v2_failure_sha256=predecessor.failure_body_sha256,
        _seal=_V3_CAPABILITY_SEAL,
    )


def reserve_source_smoke_root_v3(root: Path) -> phase_b.ArtifactRoot:
    return phase_b.reserve_source_smoke_root(root, relative=SOURCE_SMOKE_V3_ROOT_RELATIVE)


class SourceSmokeBackendV3(Protocol):
    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: RunIdentityV3) -> Any: ...
    def source_authority(self, runtime: Any, identity: RunIdentityV3) -> Mapping[str, object]: ...
    def run_steps(self, runtime: Any, spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary: ...
    def close(self, runtime: Any | None) -> None: ...


def _attempt_payload(identity: RunIdentityV3) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_source_smoke_attempt_v3", "cell": CELL, "phase": PHASE_B_V3,
        "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(),
        "v2_failed_predecessor": identity.predecessor.payload(), "boundaries": identity.payload()["boundaries"],
        "status": "ATTEMPT_STARTED_SOURCE_ONLY_V3",
    }


def _launch_payload(identity: RunIdentityV3, attempt_sha256: str) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_source_smoke_launch_v3", "cell": CELL, "phase": PHASE_B_V3,
        "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "Phase-B-v3 attempt SHA"),
        "v2_failed_predecessor": identity.predecessor.payload(),
        "launch_closure": validate_phase_b_v3_closure(identity.closure),
        "status": "SOURCE_SMOKE_V3_LAUNCHED",
    }


def validate_source_authority_v3(
    value: Mapping[str, object], *, identity: RunIdentityV3, launch_sha256: str,
) -> dict[str, object]:
    validate_run_identity_v3(identity)
    expected = {
        "schema", "cell", "v2_compatible_authority", "tf32_enforcement", "closure", "launch_sha256",
        "v2_failed_predecessor",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhaseBV3Error("Phase-B-v3 source authority schema drift")
    payload = dict(value)
    if (payload["schema"] != "posterior_carrier_source_authority_v3" or payload["cell"] != CELL
            or payload["launch_sha256"] != _sha(launch_sha256, "Phase-B-v3 source authority launch SHA")
            or payload["v2_failed_predecessor"] != identity.predecessor.payload()
            or validate_phase_b_v3_closure(payload["closure"]) != validate_phase_b_v3_closure(identity.closure)):
        raise PhaseBV3Error("Phase-B-v3 source authority immutable binding drift")
    nested = payload["v2_compatible_authority"]
    if not isinstance(nested, Mapping):
        raise PhaseBV3Error("Phase-B-v3 nested V2 authority missing")
    phase_b_v2.validate_source_authority_v2(nested, identity=identity.base_identity, launch_sha256=launch_sha256)
    validate_tf32_enforcement(payload["tf32_enforcement"])
    return payload


def _bind_launch_to_nested_v2_authority(value: Mapping[str, object], launch_sha256: str) -> dict[str, object]:
    """Bind the one durable V3 launch through both inherited authority layers.

    V2's physical adapter intentionally produces pre-publication authority
    bytes.  Its own lifecycle fills the V2 launch field immediately before
    validation.  V3 preserves that exact pattern and additionally binds the
    nested V1 authority, so neither layer can retain a placeholder digest.
    """
    outer = dict(value)
    nested = outer.get("v2_compatible_authority")
    if not isinstance(nested, Mapping):
        raise PhaseBV3Error("Phase-B-v3 source authority lacks nested V2 mapping")
    v2 = dict(nested)
    v1 = v2.get("v1_compatible_authority")
    if not isinstance(v1, Mapping):
        raise PhaseBV3Error("Phase-B-v3 source authority lacks nested V1 mapping")
    v1_copy = dict(v1)
    v1_copy["launch_sha256"] = _sha(launch_sha256, "Phase-B-v3 inherited V1 launch SHA")
    v2["v1_compatible_authority"] = v1_copy
    v2["launch_sha256"] = _sha(launch_sha256, "Phase-B-v3 inherited V2 launch SHA")
    outer["v2_compatible_authority"] = v2
    return outer


def _tf32_for_failure(backend: Any, runtime: Any | None) -> Mapping[str, object] | None:
    candidate = getattr(runtime, "tf32_enforcement", None) if runtime is not None else None
    if candidate is None:
        candidate = getattr(backend, "last_tf32_enforcement", None)
    if candidate is None:
        return None
    if isinstance(candidate, TF32Enforcement):
        candidate = candidate.payload()
    return validate_tf32_enforcement(candidate)


def _failure_payload(
    *, identity: RunIdentityV3, attempt_sha256: str, launch_sha256: str | None, stage: str,
    error: BaseException, progress: phase_b.SmokeExecutionProgress, tf32_enforcement: Mapping[str, object] | None,
) -> dict[str, object]:
    if stage not in {"prepare", "source_authority", "steps", "terminal"}:
        raise PhaseBV3Error("Phase-B-v3 failure stage drift")
    if tf32_enforcement is not None:
        tf32_enforcement = validate_tf32_enforcement(tf32_enforcement)
    return {
        "schema": "posterior_carrier_source_smoke_failure_v3", "cell": CELL, "phase": PHASE_B_V3,
        "identity": identity.payload(), "attempt_sha256": _sha(attempt_sha256, "Phase-B-v3 failure attempt SHA"),
        "launch_sha256": launch_sha256, "source_authority_sha256": progress.source_authority_sha256,
        "v2_failed_predecessor": identity.predecessor.payload(), "stage": stage,
        "error_class": type(error).__name__, "error_sha256": phase_b.sha256_bytes(repr(error).encode("utf-8")),
        "source_opened": progress.source_opened, "remote_initialized": progress.remote_initialized,
        "optimizer_steps_completed": progress.optimizer_steps_completed,
        "tf32_enforcement": tf32_enforcement, "boundaries": identity.payload()["boundaries"],
        "terminal_published": False, "status": "SOURCE_SMOKE_V3_FAILED_HONESTLY",
    }


def validate_failure_payload_v3(
    value: Mapping[str, object], *, identity: RunIdentityV3, attempt_sha256: str,
    launch_sha256: str | None, progress: phase_b.SmokeExecutionProgress,
) -> dict[str, object]:
    validate_run_identity_v3(identity)
    expected = {
        "schema", "cell", "phase", "identity", "attempt_sha256", "launch_sha256", "source_authority_sha256",
        "v2_failed_predecessor", "stage", "error_class", "error_sha256", "source_opened", "remote_initialized",
        "optimizer_steps_completed", "tf32_enforcement", "boundaries", "terminal_published", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhaseBV3Error("Phase-B-v3 failure schema drift")
    payload = dict(value)
    policy = payload["tf32_enforcement"]
    if policy is not None:
        validate_tf32_enforcement(policy)
    if (payload["schema"] != "posterior_carrier_source_smoke_failure_v3" or payload["cell"] != CELL
            or payload["phase"] != PHASE_B_V3 or payload["identity"] != identity.payload()
            or payload["attempt_sha256"] != _sha(attempt_sha256, "Phase-B-v3 failure attempt SHA")
            or payload["launch_sha256"] != launch_sha256
            or payload["source_authority_sha256"] != progress.source_authority_sha256
            or payload["v2_failed_predecessor"] != identity.predecessor.payload()
            or payload["stage"] not in {"prepare", "source_authority", "steps", "terminal"}
            or not isinstance(payload["error_class"], str) or not payload["error_class"]
            or not isinstance(payload["error_sha256"], str) or len(payload["error_sha256"]) != 64
            or payload["source_opened"] is not progress.source_opened
            or payload["remote_initialized"] is not progress.remote_initialized
            or payload["optimizer_steps_completed"] != progress.optimizer_steps_completed
            or payload["boundaries"] != identity.payload()["boundaries"]
            or payload["terminal_published"] is not False
            or payload["status"] != "SOURCE_SMOKE_V3_FAILED_HONESTLY"):
        raise PhaseBV3Error("Phase-B-v3 failure binding drift")
    if launch_sha256 is not None:
        _sha(launch_sha256, "Phase-B-v3 failure launch SHA")
    if progress.source_authority_sha256 is not None:
        _sha(progress.source_authority_sha256, "Phase-B-v3 failure source authority SHA")
    return payload


def run_source_smoke_lifecycle_v3(
    *, backend: SourceSmokeBackendV3, artifact: phase_b.ArtifactRoot, identity: RunIdentityV3,
    stage_root: Path | None = None,
) -> Mapping[str, object]:
    """Run V3 through an injected backend; static CLI never reaches here."""
    validate_run_identity_v3(identity)
    runtime: Any | None = None
    progress = phase_b.SmokeExecutionProgress()
    stage = "prepare"
    launch_sha: str | None = None
    terminal_published = False
    if stage_root is not None and phase_b_v3_closure(Path(stage_root).absolute()) != validate_phase_b_v3_closure(identity.closure):
        raise PhaseBV3Error("Phase-B-v3 live closure drift before attempt")
    attempt_sha = artifact.publish_json("attempt.json", _attempt_payload(identity))
    try:
        launch = _launch_payload(identity, attempt_sha)
        launch_sha = artifact.publish_json("launch.json", launch)
        runtime = backend.prepare(phase_b.SMOKE_SPEC, identity)
        progress = progress.merge(phase_b._runtime_progress(runtime))
        stage = "source_authority"
        authority = dict(backend.source_authority(runtime, identity))
        authority["launch_sha256"] = launch_sha
        authority["closure"] = validate_phase_b_v3_closure(identity.closure)
        authority = _bind_launch_to_nested_v2_authority(authority, launch_sha)
        authority = validate_source_authority_v3(authority, identity=identity, launch_sha256=launch_sha)
        authority_sha = artifact.publish_json("source_authority.json", authority)
        progress = progress.merge(phase_b.SmokeExecutionProgress(
            source_opened=progress.source_opened, remote_initialized=progress.remote_initialized,
            optimizer_steps_completed=progress.optimizer_steps_completed, source_authority_sha256=authority_sha,
        ))
        stage = "steps"
        summary = backend.run_steps(runtime, phase_b.SMOKE_SPEC)
        step = {
            "schema": "posterior_carrier_source_smoke_step100_v3", "cell": CELL, "phase": PHASE_B_V3,
            "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(), "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha, "summary": summary.payload(),
            "v2_failed_predecessor": identity.predecessor.payload(), "boundaries": identity.payload()["boundaries"],
            "status": "SOURCE_SMOKE_V3_100_STEPS_COMPLETE",
        }
        step_sha = artifact.publish_json("step100.json", step)
        progress = progress.merge(phase_b.SmokeExecutionProgress(
            source_opened=progress.source_opened, remote_initialized=progress.remote_initialized,
            optimizer_steps_completed=phase_b.SOURCE_SMOKE_STEPS, source_authority_sha256=authority_sha,
        ))
        stage = "terminal"
        final_closure = phase_b_v3_closure(Path(stage_root).absolute()) if stage_root is not None else validate_phase_b_v3_closure(identity.closure)
        if final_closure != launch["launch_closure"]:
            raise PhaseBV3Error("Phase-B-v3 launch/final closure drift")
        terminal = {
            "schema": "posterior_carrier_source_smoke_terminal_v3", "cell": CELL, "phase": PHASE_B_V3,
            "spec": phase_b.SMOKE_SPEC.payload(), "identity": identity.payload(), "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha, "source_authority_sha256": authority_sha, "step100_sha256": step_sha,
            "v2_failed_predecessor": identity.predecessor.payload(), "launch_closure": launch["launch_closure"],
            "final_closure": final_closure, "boundaries": identity.payload()["boundaries"],
            "status": "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE",
        }
        terminal_sha = artifact.publish_json("terminal.json", terminal)
        terminal_published = True
        if artifact.reload_json("terminal.json", terminal_sha) != terminal:
            raise PhaseBV3Error("Phase-B-v3 terminal reload drift")
        return terminal
    except BaseException as error:
        if isinstance(error, phase_b.SourceSmokeExecutionError):
            progress = progress.merge(error.progress)
            stage = error.stage
            cause: BaseException = error.cause
        else:
            progress = progress.merge(phase_b._runtime_progress(runtime))
            cause = error
        if not terminal_published:
            try:
                failure = _failure_payload(
                    identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, stage=stage,
                    error=cause, progress=progress, tf32_enforcement=_tf32_for_failure(backend, runtime),
                )
                failure = validate_failure_payload_v3(
                    failure, identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, progress=progress,
                )
                if launch_sha is not None:
                    artifact.reload_json("launch.json", launch_sha)
                if progress.source_authority_sha256 is not None:
                    artifact.reload_json("source_authority.json", progress.source_authority_sha256)
                failure_sha = artifact.publish_json("failure.json", failure)
                if artifact.reload_json("failure.json", failure_sha) != failure:
                    raise PhaseBV3Error("Phase-B-v3 failure reload drift")
            except BaseException:
                pass
        raise
    finally:
        backend.close(runtime)


def reviewed_remote_source_smoke_v3(
    *, root: Path, capability: RootReviewedCapabilityV3, backend: SourceSmokeBackendV3,
    identity: RunIdentityV3, source_data: phase_b.SourceDataRootCapability,
) -> Mapping[str, object]:
    """Execution-only V3 route; public flags cannot manufacture capability."""
    validate_run_identity_v3(identity)
    if source_data.payload() != identity.base_identity.base_identity.source_authority.get("source_data_root"):
        raise PhaseBV3Error("Phase-B-v3 external source-data capability drift")
    phase_b.validate_stage_source_separation(stage_root=Path(root).absolute(), source_data=source_data)
    live = phase_b_v3_closure(Path(root).absolute())
    if live != validate_phase_b_v3_closure(identity.closure):
        raise PhaseBV3Error("Phase-B-v3 live closure drift before predecessor validation")
    capability.validate(closure=live, predecessor=identity.predecessor)
    # All six immutable V1/V2 receipt pairs are read under held no-follow
    # directory FDs before V3 output reservation and before source/CUDA.
    validate_failed_predecessor_chain(Path(root).absolute())
    artifact = reserve_source_smoke_root_v3(Path(root).absolute())
    return run_source_smoke_lifecycle_v3(backend=backend, artifact=artifact, identity=identity, stage_root=Path(root).absolute())


def remote_staging_plan_v3(
    *, closure: Mapping[str, object], source_authority_sha256: str,
    source_data: phase_b.SourceDataRootCapability, remote_root_name: str = "posterior_carrier_budgetmix_d_seed42_stage_v3",
) -> dict[str, object]:
    """Pure explicit V3 stage plan; it does not transfer or launch anything."""
    if (not isinstance(remote_root_name, str) or not remote_root_name or "/" in remote_root_name or ".." in remote_root_name):
        raise PhaseBV3Error("Phase-B-v3 remote stage root must be one safe component")
    stable = validate_phase_b_v3_closure(closure)
    v2_closure = _v2_closure_from_v3(stable)
    mode_by_path = {relative: mode for relative, _sha_value, mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS}
    stage_files = [
        {
            "relative_path": relative,
            "sha256": stable["sha256_by_path"][relative],
            "role": "immutable_authority" if relative in phase_b.SOURCE_AUTHORITY_ASSET_PATHS else "closure_dependency",
            **({"mode": mode_by_path[relative]} if mode_by_path.get(relative) is not None else {}),
        }
        for relative in PHASE_B_V3_CLOSURE_PATHS
    ]
    for relative, expected_sha, mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS:
        if stable["sha256_by_path"].get(relative) != expected_sha:
            raise PhaseBV3Error("Phase-B-v3 immutable authority asset closure drift")
        if mode == 0o444:
            stage_files.append({
                "relative_path": f"{relative}.sha256", "contents": f"{expected_sha}  {Path(relative).name}\n",
                "mode": 0o444, "role": "immutable_authority_sidecar",
            })
    for relative_root, assets, role in (
        (V1_FAILURE_ROOT_RELATIVE, V1_PREDECESSOR_ASSETS, "immutable_v1_failed_predecessor"),
        (V2_FAILURE_ROOT_RELATIVE, V2_PREDECESSOR_ASSETS, "immutable_v2_failed_predecessor"),
    ):
        for name, digest in assets:
            relative = f"{relative_root}/{name}"
            stage_files.extend((
                {"relative_path": relative, "sha256": digest, "mode": 0o444, "role": f"{role}_receipt"},
                {"relative_path": f"{relative}.sha256", "contents": f"{digest}  {name}\n", "mode": 0o444,
                 "role": f"{role}_sidecar"},
            ))
    return {
        "schema": "posterior_carrier_remote_staging_plan_v3", "cell": CELL, "phase": PHASE_B_V3,
        "remote_host": "xinyuan@100.103.97.12", "remote_root_name": remote_root_name,
        "closure": stable, "accepted_v2_closure_sha256": v2_closure["closure_sha256"],
        "source_authority_sha256": _sha(source_authority_sha256, "Phase-B-v3 staging source authority SHA"),
        "v2_failed_predecessor": V2FailedPredecessor().payload(), "stage_files": stage_files,
        "external_source_data_root": phase_b.validate_source_data_root_payload(source_data.payload()),
        "nwb_assets_in_stage": False, "remote_torch_authority": dict(phase_b.REMOTE_TORCH_AUTHORITY),
        "forbidden": ["target", "within", "external", "formal", "h1", "teacher", "remote_historical_repo_mutation"],
        "launch": "NOT_AUTHORIZED_BY_PLAN; requires a fresh in-process root-reviewed capability",
    }


class RemotePosteriorSourceSmokeBackendV3(phase_b_v2.RemotePosteriorSourceSmokeBackendV2):
    """Deferred V3 backend: force TF32 off before V2 reaches model/Adam."""

    def __init__(self, root: Path, *, source_data: phase_b.SourceDataRootCapability, num_workers: int = 4,
                 nvml_status: str = "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH") -> None:
        super().__init__(root, source_data=source_data, num_workers=num_workers, nvml_status=nvml_status)
        self.last_tf32_enforcement: TF32Enforcement | None = None

    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: RunIdentityV3) -> Any:
        # A backend object is normally single-use, but clear prior evidence so
        # a rejected second attempt cannot inherit another attempt's policy.
        self.last_tf32_enforcement = None
        if spec != phase_b.SMOKE_SPEC:
            raise PhaseBV3Error("physical Phase-B-v3 backend accepts only fixed 100-step source smoke")
        validate_run_identity_v3(identity)
        # This is intentionally the first physical precision action.  It is
        # before V2 source/model construction; V2 then independently checks
        # the post-state, so defaults cannot decide acceptance.
        policy = TF32Enforcement.enforce(torch)
        self.last_tf32_enforcement = policy
        try:
            runtime = super().prepare(spec, identity.base_identity)
        except BaseException:
            policy.restore(torch)
            raise
        setattr(runtime, "tf32_enforcement", policy)
        return runtime

    def source_authority(self, runtime: Any, identity: RunIdentityV3) -> Mapping[str, object]:
        policy = _tf32_for_failure(self, runtime)
        if policy is None:
            raise PhaseBV3Error("physical Phase-B-v3 lost TF32 enforcement evidence")
        nested = super().source_authority(runtime, identity.base_identity)
        return {
            "schema": "posterior_carrier_source_authority_v3", "cell": CELL,
            "v2_compatible_authority": dict(nested), "tf32_enforcement": policy,
            "closure": validate_phase_b_v3_closure(identity.closure), "launch_sha256": "0" * 64,
            "v2_failed_predecessor": identity.predecessor.payload(),
        }

    def close(self, runtime: Any | None) -> None:
        policy = getattr(runtime, "tf32_enforcement", None) if runtime is not None else self.last_tf32_enforcement
        try:
            super().close(runtime)
        finally:
            if isinstance(policy, TF32Enforcement):
                policy.restore(torch)
