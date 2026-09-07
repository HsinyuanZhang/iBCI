"""PMC-specific immutable source-training lifecycle.

The Cell-D equal-session route supplies the reviewed scheduler, train-step,
loss and dropout mechanics.  This module owns only PMC identity, independent
result roots, capabilities, and receipts so a posterior-marginalized run can
never publish under the sealed Cell-D result lineage.

It is safe to import for synthetic tests but has no public execution entry
point.  A physical backend is accepted only together with an opaque
in-process root-reviewed capability.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from . import plan


class PMCRunnerError(RuntimeError):
    """Fail closed for PMC lifecycle/capability/receipt drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PMCRunnerError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class PMCExecutionSpec:
    """Exact route lifecycle budget; no shortened public execution spec."""

    kind: str
    epochs: int
    steps_per_epoch: int
    checkpoint_epochs: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = {
            "source_smoke": (1, 1, ()),
            "full_train": (plan.EPOCHS, plan.STEPS_PER_EPOCH, plan.CHECKPOINT_EPOCHS),
        }.get(self.kind)
        _require(expected is not None and (self.epochs, self.steps_per_epoch, self.checkpoint_epochs) == expected,
                 "PMC fixed execution spec drift")

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_marginalized_cell_d_execution_spec_v1",
            "kind": self.kind,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "batch_size": plan.BATCH_SIZE,
            "seed": plan.SEED,
            "source_only": True,
        }


# Frozen values, but ordinary dataclass instances: this avoids manufacturing
# uninitialised objects while still rejecting caller-created shortened specs.
PMC_SOURCE_SMOKE_SPEC = PMCExecutionSpec("source_smoke", 1, 1, ())
PMC_FULL_TRAIN_SPEC = PMCExecutionSpec(
    "full_train", plan.EPOCHS, plan.STEPS_PER_EPOCH, plan.CHECKPOINT_EPOCHS,
)


@dataclass(frozen=True)
class PMCIdentity:
    """Route-specific identity binding Phase-1 + Phase-B-v2 provenance."""

    closure: Mapping[str, object]
    phase_b_v2_authority_sha256: str
    phase_b_v2_closure_sha256: str
    source_binding_sha256: str
    gpu: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        stable = plan.validate_implementation_closure(self.closure)
        for label, value in (
            ("Phase-B-v2 authority", self.phase_b_v2_authority_sha256),
            ("Phase-B-v2 closure", self.phase_b_v2_closure_sha256),
            ("PMC source binding", self.source_binding_sha256),
        ):
            _require(_is_sha(value), f"PMC {label} SHA drift")
        return {
            "schema": "posterior_marginalized_cell_d_identity_v1",
            "cell": plan.CELL,
            "phase": "SOURCE_TRAINING_POSTERIOR_MARGINALIZATION",
            "closure": stable,
            "predecessor_phase1_closure_sha256": plan.PHASE1_ACCEPTED_CLOSURE_SHA256,
            "approved_phase_b_v2_authority_sha256": self.phase_b_v2_authority_sha256,
            "approved_phase_b_v2_closure_sha256": self.phase_b_v2_closure_sha256,
            "pmc_source_binding_sha256": self.source_binding_sha256,
            "sealed_cell_d": {
                "canonical_initial_state_artifact_sha256": plan.CANONICAL_INITIAL_STATE_SHA256,
                "canonical_initial_state_state_sha256": plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
                "ordinary_ols_t4_normalizer_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
                "ordinary_ols_t4_mean_float32": list(plan.SEALED_OLS_T4_MEAN_FLOAT32),
                "ordinary_ols_t4_std_float32": list(plan.SEALED_OLS_T4_STD_FLOAT32),
                "initialized_trainable_parameters": plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
                "uninitialized_lazy_keys": list(plan.SEALED_CELL_D_LAZY_KEYS),
            },
            "sole_intervention": "source_training_side_cached_posterior_sample_only",
            "inference": "deterministic_ordinary_ols_point_t4_only",
            "gpu": plan.validate_compatible_device_profile(self.gpu),
            "boundaries": {
                "source_only": True,
                "within_opened": False,
                "external_opened": False,
                "formal_opened": False,
                "target_opened": False,
                "target_optimizer_steps": 0,
                "target_backward_calls": 0,
                "target_update_calls": 0,
                "posterior_normalizer_used": False,
                "posterior_credibility_or_attention_bias_used": False,
            },
        }


def validate_identity(value: PMCIdentity | Mapping[str, object]) -> dict[str, object]:
    payload = value.payload() if isinstance(value, PMCIdentity) else dict(value)
    required = {
        "schema", "cell", "phase", "closure", "predecessor_phase1_closure_sha256",
        "approved_phase_b_v2_authority_sha256", "approved_phase_b_v2_closure_sha256",
        "pmc_source_binding_sha256", "sealed_cell_d", "sole_intervention", "inference", "gpu", "boundaries",
    }
    _require(set(payload) == required, "PMC identity schema drift")
    closure = plan.validate_implementation_closure(payload["closure"])
    _require(payload["schema"] == "posterior_marginalized_cell_d_identity_v1" and payload["cell"] == plan.CELL
             and payload["phase"] == "SOURCE_TRAINING_POSTERIOR_MARGINALIZATION"
             and payload["predecessor_phase1_closure_sha256"] == plan.PHASE1_ACCEPTED_CLOSURE_SHA256,
             "PMC identity topology/predecessor drift")
    for key in ("approved_phase_b_v2_authority_sha256", "approved_phase_b_v2_closure_sha256", "pmc_source_binding_sha256"):
        _require(_is_sha(payload[key]), f"PMC identity {key} drift")
    held = payload["sealed_cell_d"]
    _require(isinstance(held, Mapping) and held == {
        "canonical_initial_state_artifact_sha256": plan.CANONICAL_INITIAL_STATE_SHA256,
        "canonical_initial_state_state_sha256": plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
        "ordinary_ols_t4_normalizer_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
        "ordinary_ols_t4_mean_float32": list(plan.SEALED_OLS_T4_MEAN_FLOAT32),
        "ordinary_ols_t4_std_float32": list(plan.SEALED_OLS_T4_STD_FLOAT32),
        "initialized_trainable_parameters": plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
        "uninitialized_lazy_keys": list(plan.SEALED_CELL_D_LAZY_KEYS),
    }, "PMC held Cell-D identity drift")
    _require(payload["sole_intervention"] == "source_training_side_cached_posterior_sample_only"
             and payload["inference"] == "deterministic_ordinary_ols_point_t4_only",
             "PMC identity intervention drift")
    payload_gpu = plan.validate_compatible_device_profile(payload["gpu"])
    expected_boundaries = {
        "source_only": True, "within_opened": False, "external_opened": False,
        "formal_opened": False, "target_opened": False, "target_optimizer_steps": 0,
        "target_backward_calls": 0, "target_update_calls": 0,
        "posterior_normalizer_used": False, "posterior_credibility_or_attention_bias_used": False,
    }
    _require(payload["boundaries"] == expected_boundaries, "PMC identity target/posterior boundary drift")
    return {**payload, "closure": closure, "gpu": payload_gpu}


class _CapabilitySeal:
    __slots__ = ()


_CAPABILITY_SEAL = _CapabilitySeal()


@dataclass(frozen=True)
class RootReviewedPMCCapability:
    """Opaque in-process gate; neither CLI nor serialized input can forge it."""

    kind: str
    closure_sha256: str
    _seal: object = field(repr=False, compare=False, default=None)

    def validate(self, *, identity: PMCIdentity, spec: PMCExecutionSpec) -> None:
        _require(self._seal is _CAPABILITY_SEAL and self.kind == spec.kind,
                 "PMC route requires an in-process root-reviewed capability")
        payload = validate_identity(identity)
        _require(self.closure_sha256 == payload["closure"]["closure_sha256"],
                 "PMC capability closure drift")


def _issue_root_review_capability_for_audited_route(
    *, identity: PMCIdentity, spec: PMCExecutionSpec,
) -> RootReviewedPMCCapability:
    """Private root-only issuance; it does not reserve a root or open data."""
    stable = validate_identity(identity)
    return RootReviewedPMCCapability(
        kind=spec.kind, closure_sha256=str(stable["closure"]["closure_sha256"]), _seal=_CAPABILITY_SEAL,
    )


class PMCArtifactRoot:
    """Small O_EXCL/fsync/0444 writer with same-pair rollback semantics."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).absolute()
        self._parent_fd: int | None = None
        self._fd: int | None = None
        self._identity: tuple[int, int] | None = None

    def reserve(self) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        self._parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.mkdir(self.path.name, 0o755, dir_fd=self._parent_fd)
        except FileExistsError as error:
            raise PMCRunnerError("PMC canonical output root is not fresh") from error
        self._fd = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._parent_fd)
        info = os.fstat(self._fd)
        self._identity = (int(info.st_dev), int(info.st_ino))

    def _root_fd(self) -> int:
        _require(self._fd is not None and self._identity is not None, "PMC artifact root not reserved")
        info = os.fstat(self._fd)
        _require((int(info.st_dev), int(info.st_ino)) == self._identity, "PMC artifact root descriptor identity drift")
        return self._fd

    def publish_bytes(self, name: str, body: bytes) -> str:
        _require("/" not in name and name and name.endswith((".json", ".pt")), "PMC artifact name drift")
        root_fd = self._root_fd()
        digest, sidecar = _sha(body), None
        sidecar = f"{digest}  {name}\n".encode("ascii")
        created: list[str] = []
        try:
            for leaf, payload in ((name, body), (f"{name}.sha256", sidecar)):
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444, dir_fd=root_fd)
                created.append(leaf)
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise PMCRunnerError("short PMC artifact write")
                        view = view[written:]
                    os.fsync(fd)
                    os.fchmod(fd, 0o444)
                finally:
                    os.close(fd)
            os.fsync(root_fd)
        except BaseException:
            for leaf in reversed(created):
                try:
                    os.unlink(leaf, dir_fd=root_fd)
                except OSError:
                    pass
            raise
        return digest

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        return self.publish_bytes(name, _json(dict(payload)))

    def reload_bytes(self, name: str, expected_sha256: str) -> bytes:
        """Descriptor-reload one immutable body/sidecar pair exactly.

        Callers use this after every material publication which feeds a later
        provenance edge (for example a checkpoint entering SWA).  Therefore
        the later edge is built from durable bytes, not merely the pre-write
        in-memory value that happened to be hashed.
        """
        root_fd = self._root_fd()

        def read(leaf: str) -> bytes:
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                info = os.fstat(fd)
                _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444,
                         "PMC immutable artifact mode/type drift")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        return b"".join(chunks)
                    chunks.append(chunk)
            finally:
                os.close(fd)
        body = read(name)
        _require(_sha(body) == expected_sha256, "PMC immutable artifact digest drift")
        _require(read(f"{name}.sha256") == f"{expected_sha256}  {name}\n".encode("ascii"),
                 "PMC immutable artifact sidecar drift")
        return body

    def reload_json(self, name: str, expected_sha256: str) -> dict[str, object]:
        body = self.reload_bytes(name, expected_sha256)
        value = json.loads(body)
        _require(isinstance(value, dict), "PMC immutable JSON root drift")
        return value

    def close(self) -> None:
        for descriptor in (self._fd, self._parent_fd):
            if descriptor is not None:
                os.close(descriptor)
        self._fd = self._parent_fd = None


@dataclass
class PMCProgress:
    source_opened: bool = False
    cuda_initialized: bool = False
    optimizer_steps_completed: int = 0
    backward_calls: int = 0
    update_calls: int = 0
    source_authority_sha256: str | None = None
    terminal_published: bool = False
    stage: str = "identity"

    def payload(self) -> dict[str, object]:
        return {
            "source_opened": self.source_opened,
            "cuda_initialized": self.cuda_initialized,
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "backward_calls": self.backward_calls,
            "update_calls": self.update_calls,
            "source_authority_sha256": self.source_authority_sha256,
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "target_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }


class PMCLifecycleBackend(Protocol):
    """Physical or synthetic source-only backend surface."""

    def prepare(self, *, spec: PMCExecutionSpec, identity: PMCIdentity, progress: PMCProgress) -> Any: ...
    def source_authority(self, runtime: Any, *, identity: PMCIdentity) -> Mapping[str, object]: ...
    def begin_epoch(self, runtime: Any, *, epoch: int, progress: PMCProgress) -> Mapping[str, object]: ...
    def train_step(self, runtime: Any, *, epoch: int, global_step: int, require_full_proof: bool,
                   progress: PMCProgress) -> Mapping[str, object]: ...
    def end_epoch(self, runtime: Any, *, epoch: int, rows: Sequence[Mapping[str, object]],
                  progress: PMCProgress) -> Mapping[str, object]: ...
    def checkpoint(self, runtime: Any, *, epoch: int, global_step: int, binding: Mapping[str, object]) -> bytes: ...
    def swa(self, runtime: Any, *, checkpoint_bodies: Mapping[int, bytes], binding: Mapping[str, object]) -> tuple[bytes, Mapping[str, object]]: ...
    def final_reverify(self, runtime: Any, *, identity: PMCIdentity) -> Mapping[str, object]: ...
    def close(self, runtime: Any | None) -> None: ...


def _attempt_payload(spec: PMCExecutionSpec, identity: PMCIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_attempt_v1", "cell": plan.CELL,
        "status": "ATTEMPT_RESERVED", "spec": spec.payload(), "identity": validate_identity(identity),
        "source_only": True, "target_or_formal_or_evaluation_forbidden": True,
        "execution_authorization": "opaque_in_process_root_reviewed_capability",
    }


def _launch_payload(spec: PMCExecutionSpec, identity: PMCIdentity, attempt_sha256: str) -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_launch_v1", "cell": plan.CELL,
        "status": "LAUNCHED", "spec": spec.payload(), "identity": validate_identity(identity),
        "attempt_sha256": attempt_sha256,
        "sole_intervention": "cached_posterior_sample_replaces_source_training_side_features_only",
        "held": {"ordinary_ols_normalizer": True, "b3s_m30": True,
                   "cell_d_train_step_loss_dropout_schedule": True, "posterior_normalizer": False,
                   "posterior_credibility_bias": False},
    }


def _failure_payload(*, spec: PMCExecutionSpec, identity: PMCIdentity, attempt_sha256: str,
                     launch_sha256: str | None, progress: PMCProgress, error: BaseException) -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_failure_v1", "cell": plan.CELL,
        "status": "FAILED", "stage": progress.stage, "spec": spec.payload(),
        "identity": validate_identity(identity), "attempt_sha256": attempt_sha256,
        "launch_sha256": launch_sha256, "progress": progress.payload(),
        "error_class": type(error).__name__, "error_sha256": _sha(traceback.format_exc().encode("utf-8")),
        "terminal_published": False,
    }


def _validate_source_authority(value: Mapping[str, object], identity: PMCIdentity) -> dict[str, object]:
    payload = dict(value)
    required = {
        "schema", "cell", "source_only", "target_opened", "within_opened", "external_opened", "formal_opened",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "source_binding", "cache_policy",
        "normalizer", "gpu", "identity",
    }
    _require(set(payload) == required and payload["schema"] == "posterior_marginalized_cell_d_source_authority_v1"
             and payload["cell"] == plan.CELL and payload["source_only"] is True,
             "PMC source authority schema drift")
    _require(all(payload[key] is False for key in ("target_opened", "within_opened", "external_opened", "formal_opened"))
             and all(payload[key] == 0 for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls")),
             "PMC source authority target boundary drift")
    _require(payload["identity"] == validate_identity(identity), "PMC source authority identity drift")
    _require(isinstance(payload["source_binding"], Mapping), "PMC source authority source-binding type drift")
    source_binding = dict(payload["source_binding"])
    _require(_sha(_json(source_binding)) == identity.source_binding_sha256,
             "PMC source authority canonical source-binding SHA drift")
    _require(source_binding.get("approved_phase_b_v2_authority_sha256") == identity.phase_b_v2_authority_sha256
             and source_binding.get("approved_phase_b_v2_closure_sha256") == identity.phase_b_v2_closure_sha256,
             "PMC source authority Phase-B-v2 authority/closure binding drift")
    _require(payload["normalizer"] == {
        "semantic_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
        "mean_float32": list(plan.SEALED_OLS_T4_MEAN_FLOAT32),
        "std_float32": list(plan.SEALED_OLS_T4_STD_FLOAT32),
        "posterior_specific_normalizer_used": False,
    }, "PMC source authority OLS normalizer literal drift")
    _require(isinstance(payload["cache_policy"], Mapping)
             and payload["cache_policy"].get("batch_loop_inverse_calls") == 0
             and payload["cache_policy"].get("batch_loop_sampling_calls") == 0
             and payload["cache_policy"].get("device_cache_before_iterator") is True,
             "PMC source authority cache policy drift")
    expected_runtime = {
        **plan.validate_compatible_device_profile(identity.gpu),
        "visible_devices": 1,
        "attested": True,
        # These settings are enforced after physical device attestation and
        # before the model/optimizer path.  Keep them in the durable source
        # authority rather than treating them as a transient launch log.
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
    }
    _require(payload["gpu"] == expected_runtime,
             "PMC source authority selected-device/TF32 runtime evidence drift")
    return payload


def _validate_epoch_row(value: Mapping[str, object], *, epoch: int, cumulative: int, spec: PMCExecutionSpec) -> dict[str, object]:
    payload = dict(value)
    required = {
        "epoch", "steps", "cumulative_optimizer_steps", "loss", "lr", "dropout", "cache", "proof",
        "resources", "diagnostic_side",
    }
    _require(set(payload) == required and payload["epoch"] == epoch and payload["steps"] == spec.steps_per_epoch
             and payload["cumulative_optimizer_steps"] == cumulative, "PMC epoch receipt topology drift")
    _require(isinstance(payload["loss"], Mapping) and isinstance(payload["lr"], Mapping)
             and isinstance(payload["dropout"], Mapping) and isinstance(payload["cache"], Mapping)
             and isinstance(payload["proof"], Mapping) and isinstance(payload["resources"], Mapping),
             "PMC epoch receipt nested schema drift")
    _require(payload["cache"].get("batch_loop_inverse_calls") == 0
             and payload["cache"].get("batch_loop_sampling_calls") == 0
             and payload["cache"].get("batch_loop_normalizer_fit_calls") == 0,
             "PMC epoch receipt batch-loop posterior work drift")
    _require(payload["proof"].get("finite_model") is True and payload["proof"].get("finite_optimizer") is True
             and isinstance(payload["proof"].get("critical_gradients"), Mapping)
             and all(item is True for item in payload["proof"]["critical_gradients"].values()),
             "PMC epoch boundary proof drift")
    _require(payload["diagnostic_side"]
             == "ordinary_ols_point_side__held_auxiliary_diagnostic_not_pmc_training_sample",
             "PMC fixed-diagnostic side disclosure drift")
    return payload


def _result_relative(spec: PMCExecutionSpec) -> str:
    return plan.SOURCE_SMOKE_ROOT_RELATIVE if spec.kind == "source_smoke" else plan.FULL_TRAIN_ROOT_RELATIVE


def validate_terminal_payload(
    value: Mapping[str, object], *, identity: PMCIdentity, spec: PMCExecutionSpec,
    attempt_sha256: str, launch_sha256: str, source_authority_sha256: str,
    epoch_sha256: Mapping[str, str], artifacts: Mapping[str, object],
) -> dict[str, object]:
    """Validate terminal graph links without trusting an implied artifact set.

    This small route-local validator is deliberately explicit so a later
    scorer can establish whether a smoke or full result has all the immutable
    leaves it claims.  It never opens tensors itself; exact checkpoint/SWA
    bytes are revalidated by the physical backend before this terminal is
    published.
    """
    payload = dict(value)
    required = {
        "schema", "cell", "status", "spec", "identity", "attempt_sha256", "launch_sha256",
        "source_authority_sha256", "optimizer_steps_completed", "epoch_count",
        "launch_final_closure_equal", "source_only", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "artifacts",
    }
    _require(set(payload) == required and payload["schema"] == "posterior_marginalized_cell_d_terminal_v1"
             and payload["cell"] == plan.CELL and payload["status"] == "TERMINAL"
             and payload["spec"] == spec.payload() and payload["identity"] == validate_identity(identity),
             "PMC terminal schema/identity/spec drift")
    _require(payload["attempt_sha256"] == attempt_sha256 and payload["launch_sha256"] == launch_sha256
             and payload["source_authority_sha256"] == source_authority_sha256
             and payload["optimizer_steps_completed"] == spec.total_steps and payload["epoch_count"] == spec.epochs
             and payload["launch_final_closure_equal"] is True and payload["source_only"] is True
             and payload["target_optimizer_steps"] == 0 and payload["target_backward_calls"] == 0
             and payload["target_update_calls"] == 0, "PMC terminal boundary/progress drift")
    _require(payload["artifacts"] == dict(artifacts), "PMC terminal artifact graph drift")
    actual_epochs = payload["artifacts"].get("epoch_sha256") if isinstance(payload["artifacts"], Mapping) else None
    _require(actual_epochs == dict(epoch_sha256) and set(actual_epochs) == {str(epoch) for epoch in range(spec.epochs)}
             and all(_is_sha(item) for item in actual_epochs.values()), "PMC terminal epoch artifact topology drift")
    if spec.kind == "source_smoke":
        _require(payload["artifacts"] == {
            "epoch_sha256": dict(epoch_sha256),
            "smoke_sha256": artifacts.get("smoke_sha256"),
            "checkpoint_sha256": {}, "swa_sha256": None,
            "swa_manifest_sha256": None, "swa_proof": None,
        } and _is_sha(artifacts.get("smoke_sha256")), "PMC smoke terminal artifact graph drift")
    else:
        checkpoints = artifacts.get("checkpoint_sha256")
        _require(isinstance(checkpoints, Mapping)
                 and set(checkpoints) == {str(epoch) for epoch in spec.checkpoint_epochs}
                 and all(_is_sha(item) for item in checkpoints.values())
                 and _is_sha(artifacts.get("swa_sha256")) and _is_sha(artifacts.get("swa_manifest_sha256"))
                 and isinstance(artifacts.get("swa_proof"), Mapping) and artifacts.get("smoke_sha256") is None,
                 "PMC full terminal final-four/SWA artifact graph drift")
    return payload


def execute_authorized(
    root: Path,
    *,
    spec: PMCExecutionSpec,
    identity: PMCIdentity,
    capability: RootReviewedPMCCapability | None,
    backend: PMCLifecycleBackend,
) -> dict[str, object]:
    """Execute only after a root-owned in-process capability has been issued.

    This function is intentionally not called by the public CLI.  It publishes
    an attempt before backend preparation; all source/CUDA failure paths carry
    honest progress and cannot leave a terminal success receipt.
    """
    stable_identity = validate_identity(identity)
    _require(spec in (PMC_SOURCE_SMOKE_SPEC, PMC_FULL_TRAIN_SPEC), "PMC lifecycle spec must be a frozen singleton")
    _require(isinstance(capability, RootReviewedPMCCapability), "PMC execution requires an in-process capability")
    capability.validate(identity=identity, spec=spec)
    root = Path(root).absolute()
    plan.assert_fresh_prospective_roots(root, spec_kind=spec.kind)
    artifact = PMCArtifactRoot(root / _result_relative(spec))
    progress = PMCProgress()
    runtime: Any | None = None
    attempt_sha: str | None = None
    launch_sha: str | None = None
    try:
        artifact.reserve()
        attempt_payload = _attempt_payload(spec, identity)
        attempt_sha = artifact.publish_json("attempt.json", attempt_payload)
        _require(artifact.reload_json("attempt.json", attempt_sha) == attempt_payload,
                 "PMC durable attempt reload drift")
        progress.stage = "prepare"
        runtime = backend.prepare(spec=spec, identity=identity, progress=progress)
        progress.stage = "launch"
        launch_payload = _launch_payload(spec, identity, attempt_sha)
        launch_sha = artifact.publish_json("launch.json", launch_payload)
        _require(artifact.reload_json("launch.json", launch_sha) == launch_payload,
                 "PMC durable launch reload drift")
        progress.stage = "source_authority"
        source_authority = _validate_source_authority(backend.source_authority(runtime, identity=identity), identity)
        progress.source_authority_sha256 = artifact.publish_json("source_authority.json", source_authority)
        _require(
            _validate_source_authority(
                artifact.reload_json("source_authority.json", progress.source_authority_sha256), identity,
            ) == source_authority,
            "PMC durable source-authority reload drift",
        )
        checkpoint_bodies: dict[int, bytes] = {}
        checkpoint_sha256: dict[str, str] = {}
        epoch_sha256: dict[str, str] = {}
        rows: list[dict[str, object]] = []
        global_step = 0
        for epoch in range(spec.epochs):
            progress.stage = "epoch"
            schedule = backend.begin_epoch(runtime, epoch=epoch, progress=progress)
            _require(isinstance(schedule, Mapping) and schedule.get("epoch") == epoch
                     and schedule.get("one_iterator") is True, "PMC inherited schedule/one-iterator drift")
            step_rows: list[Mapping[str, object]] = []
            for _ in range(spec.steps_per_epoch):
                full_proof = (_ == spec.steps_per_epoch - 1)
                row = backend.train_step(runtime, epoch=epoch, global_step=global_step,
                                         require_full_proof=full_proof, progress=progress)
                _require(isinstance(row, Mapping), "PMC backend step row drift")
                step_rows.append(row)
                global_step += 1
                progress.optimizer_steps_completed += 1
                progress.backward_calls += 1
                progress.update_calls += 1
            summary = backend.end_epoch(runtime, epoch=epoch, rows=step_rows, progress=progress)
            validated = _validate_epoch_row(summary, epoch=epoch, cumulative=global_step, spec=spec)
            row_payload = {"schema": "posterior_marginalized_cell_d_epoch_v1", "cell": plan.CELL,
                           "spec": spec.payload(), "launch_sha256": launch_sha,
                           "source_authority_sha256": progress.source_authority_sha256, **validated}
            rows.append(row_payload)
            epoch_sha256[str(epoch)] = artifact.publish_json(f"epoch_{epoch:02d}.json", row_payload)
            _require(
                artifact.reload_json(f"epoch_{epoch:02d}.json", epoch_sha256[str(epoch)]) == row_payload,
                "PMC durable epoch reload drift",
            )
            if epoch in spec.checkpoint_epochs:
                progress.stage = "checkpoint"
                # Equal-session checkpoints and SWA ingestion require the
                # *same* immutable run binding.  Epoch/global-step belong to
                # the checkpoint body, not this shared provenance map.
                binding = {
                    "cell": plan.CELL, "run_spec": spec.payload(), "launch_sha256": launch_sha,
                    "launch_closure_sha256": stable_identity["closure"]["closure_sha256"],
                    "canonical_initial_state_sha256": plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
                    "schedule_plan_sha256": "17c1ee6d1ddc62e50aa62c5e25ff800a1612631b4aa36f7508805ce6f83fc98b",
                    "source_authority_sha256": progress.source_authority_sha256,
                    "identity": stable_identity,
                }
                body = backend.checkpoint(runtime, epoch=epoch, global_step=global_step, binding=binding)
                _require(isinstance(body, bytes) and body, "PMC checkpoint body drift")
                checkpoint_sha256[str(epoch)] = artifact.publish_bytes(f"checkpoint_epoch_{epoch:02d}.pt", body)
                checkpoint_bodies[epoch] = artifact.reload_bytes(
                    f"checkpoint_epoch_{epoch:02d}.pt", checkpoint_sha256[str(epoch)],
                )
                _require(checkpoint_bodies[epoch] == body, "PMC durable checkpoint reload drift")
        artifact_graph: dict[str, object]
        if spec.kind == "source_smoke":
            smoke_payload = {
                "schema": "posterior_marginalized_cell_d_source_smoke_v1", "cell": plan.CELL,
                "steps": 1, "epoch_sha256": epoch_sha256["0"],
                "source_authority_sha256": progress.source_authority_sha256,
            }
            smoke_sha = artifact.publish_json("smoke.json", smoke_payload)
            _require(artifact.reload_json("smoke.json", smoke_sha) == smoke_payload,
                     "PMC durable smoke reload drift")
            artifact_graph = {
                "epoch_sha256": dict(epoch_sha256), "smoke_sha256": smoke_sha,
                "checkpoint_sha256": {}, "swa_sha256": None, "swa_manifest_sha256": None, "swa_proof": None,
            }
        else:
            progress.stage = "swa"
            binding = {
                "cell": plan.CELL, "run_spec": spec.payload(), "launch_sha256": launch_sha,
                "launch_closure_sha256": stable_identity["closure"]["closure_sha256"],
                "canonical_initial_state_sha256": plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
                "schedule_plan_sha256": "17c1ee6d1ddc62e50aa62c5e25ff800a1612631b4aa36f7508805ce6f83fc98b",
                "source_authority_sha256": progress.source_authority_sha256,
                "identity": stable_identity,
            }
            swa_body, swa_proof = backend.swa(runtime, checkpoint_bodies=checkpoint_bodies, binding=binding)
            _require(isinstance(swa_body, bytes) and swa_body and isinstance(swa_proof, Mapping), "PMC SWA body/proof drift")
            swa_sha = artifact.publish_bytes("swa_final4.pt", swa_body)
            _require(artifact.reload_bytes("swa_final4.pt", swa_sha) == swa_body,
                     "PMC durable SWA reload drift")
            manifest_payload = {
                "schema": "posterior_marginalized_cell_d_swa_manifest_v1", "cell": plan.CELL,
                "swa_sha256": swa_sha, "proof": dict(swa_proof), "binding": binding,
                "checkpoint_sha256": dict(checkpoint_sha256),
            }
            manifest_sha = artifact.publish_json("swa_manifest.json", manifest_payload)
            _require(artifact.reload_json("swa_manifest.json", manifest_sha) == manifest_payload,
                     "PMC durable SWA manifest reload drift")
            artifact_graph = {
                "epoch_sha256": dict(epoch_sha256), "smoke_sha256": None,
                "checkpoint_sha256": dict(checkpoint_sha256), "swa_sha256": swa_sha,
                "swa_manifest_sha256": manifest_sha, "swa_proof": dict(swa_proof),
            }
        progress.stage = "final_reverify"
        final_closure = plan.validate_implementation_closure(backend.final_reverify(runtime, identity=identity))
        _require(final_closure == stable_identity["closure"], "PMC launch/final live closure drift")
        progress.stage = "terminal"
        terminal = {
            "schema": "posterior_marginalized_cell_d_terminal_v1", "cell": plan.CELL,
            "status": "TERMINAL", "spec": spec.payload(), "identity": stable_identity,
            "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
            "source_authority_sha256": progress.source_authority_sha256,
            "optimizer_steps_completed": progress.optimizer_steps_completed,
            "epoch_count": len(rows), "launch_final_closure_equal": True,
            "source_only": True, "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "artifacts": artifact_graph,
        }
        validate_terminal_payload(
            terminal, identity=identity, spec=spec, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
            source_authority_sha256=str(progress.source_authority_sha256), epoch_sha256=epoch_sha256,
            artifacts=artifact_graph,
        )
        terminal_sha = artifact.publish_json("terminal.json", terminal)
        progress.terminal_published = True
        _require(artifact.reload_json("terminal.json", terminal_sha) == terminal,
                 "PMC durable terminal reload drift")
        return {"root": str(artifact.path), "attempt_sha256": attempt_sha, "terminal_sha256": terminal_sha}
    except BaseException as error:
        if attempt_sha is not None and not progress.terminal_published:
            try:
                artifact.publish_json("failure.json", _failure_payload(
                    spec=spec, identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                    progress=progress, error=error,
                ))
            except BaseException:
                pass
        raise
    finally:
        try:
            backend.close(runtime)
        finally:
            artifact.close()
