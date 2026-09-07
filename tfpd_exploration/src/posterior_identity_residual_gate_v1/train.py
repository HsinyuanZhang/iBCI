"""Reviewed source-only PIRG lifecycle and deferred physical backend.

The public CLI never imports this module.  This file contains a small,
route-owned lifecycle rather than modifying Cell-D, the posterior adapter, or
the V3 scorer.  The physical backend composes the audited strict-27 v2 source
adapter for already-fitted directional credibility only.
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


class PIRGTrainError(RuntimeError):
    """Fail closed for PIRG source-only training."""


RESULT_ROOT_RELATIVE = "tfpd_exploration/results/posterior_identity_residual_gate_v1"
REMOTE_STAGE_ROOT = "/home/xinyuan/Work_host/posterior_identity_residual_gate_stage_v1"
SOURCE_STEPS_PER_EPOCH = 33_925
TOTAL_STEPS = plan.EPOCHS * SOURCE_STEPS_PER_EPOCH
# Keep the inherited Cell-D Adam learning-rate scale.  There is no sweep in
# this one-scalar cell; changing it would be a second, unregistered factor.
ALPHA_LR = 1e-4
SEALED_CELL_D_TERMINAL_SHA256 = plan.SEALED_CELL_D_TERMINAL_SHA256

# The score closure is intentionally broader: it composes the immutable V3
# evaluator.  Keep it separate so the source-only route does not stage or
# import target/evaluation machinery.
SCORE_IMPLEMENTATION_CLOSURE = tuple(dict.fromkeys((
    plan.WORKORDER_RELATIVE,
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/__init__.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/plan.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/core.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/train.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/physical.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/score.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/score_physical.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/remote_stage.py",
    "tfpd_exploration/scripts/run_posterior_identity_residual_gate.py",
    "tfpd_exploration/scripts/run_posterior_identity_residual_gate_score.py",
    "tfpd_exploration/tests/test_posterior_identity_residual_gate_v1.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score.py",
    "tfpd_exploration/src/posterior_carrier_v1/full_train.py",
    "tfpd_exploration/src/posterior_carrier_v1/full_result_import.py",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/quick_screen.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/physical.py",
    "tfpd_exploration/src/posterior_carrier_quick_screen_v1/remote_stage.py",
    "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_QUICK_SCREEN_20260822.md",
    "tfpd_exploration/scripts/run_posterior_carrier_quick_screen.py",
    "tfpd_exploration/tests/test_posterior_carrier_quick_screen_v1.py",
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/attribution.py",
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/physical.py",
    "tfpd_exploration/src/posterior_carrier_m30_attribution_v1/remote_stage.py",
    "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_M30_ATTRIBUTION_20260822.md",
    "tfpd_exploration/scripts/run_posterior_carrier_m30_attribution.py",
    "tfpd_exploration/tests/test_posterior_carrier_m30_attribution_v1.py",
    "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_MATCHED_SCORE_20260822.md",
    "tfpd_exploration/scripts/run_posterior_carrier_matched_score.py",
    "tfpd_exploration/scripts/run_posterior_carrier_full_result_import.py",
    "tfpd_exploration/tests/test_posterior_carrier_matched_score.py",
    "tfpd_exploration/tests/test_posterior_carrier_full_result_import.py",
    "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v1/c1_train_val_33_manifest.json",
    "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json",
    "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
)))


IMPLEMENTATION_CLOSURE = (
    plan.WORKORDER_RELATIVE,
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/__init__.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/plan.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/core.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/train.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/physical.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_v1/remote_stage.py",
    "tfpd_exploration/scripts/run_posterior_identity_residual_gate.py",
    "tfpd_exploration/tests/test_posterior_identity_residual_gate_v1.py",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PIRGTrainError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PIRGTrainError(f"{label} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise PIRGTrainError(f"{label} must be hexadecimal") from error
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PIRGTrainError(f"{label} must be a finite numeric scalar")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise PIRGTrainError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        if not isinstance(self.sha256_by_path, Mapping) or set(self.sha256_by_path) != set(IMPLEMENTATION_CLOSURE):
            raise PIRGTrainError("PIRG closure topology drift")
        hashes = {path: _sha(self.sha256_by_path[path], f"PIRG closure {path}") for path in IMPLEMENTATION_CLOSURE}
        body = {"paths": list(IMPLEMENTATION_CLOSURE), "sha256_by_path": hashes}
        return {**body, "closure_sha256": _digest(_json(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    """Descriptor-hash only explicit code and metadata leaves."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_CLOSURE:
        path = base / relative
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise PIRGTrainError(f"PIRG closure leaf missing or aliased: {relative}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
                raise PIRGTrainError("PIRG closure descriptor identity drift")
            pieces: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                pieces.append(chunk)
        finally:
            os.close(descriptor)
        after = os.lstat(path)
        if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise PIRGTrainError("PIRG closure changed during descriptor read")
        hashes[relative] = _digest(b"".join(pieces))
    return ImplementationClosure(hashes)


@dataclass(frozen=True)
class TrainingSpec:
    cell: str = plan.CELL
    seed: int = plan.SEED
    batch_size: int = plan.BATCH_SIZE
    epochs: int = plan.EPOCHS
    steps_per_epoch: int = SOURCE_STEPS_PER_EPOCH
    alpha_lr: float = ALPHA_LR
    optimizer: str = "Adam"
    weight_decay: float = 0.0

    def payload(self) -> dict[str, object]:
        if (
            self.cell != plan.CELL or self.seed != plan.SEED or self.batch_size != plan.BATCH_SIZE
            or self.epochs != plan.EPOCHS or self.steps_per_epoch != SOURCE_STEPS_PER_EPOCH
            or self.alpha_lr != ALPHA_LR or self.optimizer != "Adam" or self.weight_decay != 0.0
        ):
            raise PIRGTrainError("PIRG fixed training spec drift")
        return {
            "cell": self.cell, "seed": self.seed, "batch_size": self.batch_size,
            "epochs": self.epochs, "steps_per_epoch": self.steps_per_epoch,
            "total_steps": TOTAL_STEPS, "alpha_lr": self.alpha_lr, "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "trainable_parameters": ["alpha"],
            "loss": "dense_valid_bin_supervised_mse",
            "source_only": True,
        }


@dataclass(frozen=True)
class PIRGIdentity:
    closure: ImplementationClosure
    spec: TrainingSpec = field(default_factory=TrainingSpec)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_identity_residual_gate_identity_v1",
            "cell": plan.CELL,
            "phase": plan.PHASE,
            "classification": "SOURCE_ONLY_PERFORMANCE_SCREEN_PRECURSOR",
            "closure": self.closure.payload(),
            "spec": self.spec.payload(),
            "sealed_cell_d": {
                "terminal_sha256": SEALED_CELL_D_TERMINAL_SHA256,
                "swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
                "baseline_sha256": plan.SEALED_CELL_D_BASELINE_SHA256,
                "initialization_assets": plan.sealed_cell_d_init_assets_payload(),
                "ordinary_ols_t4_held": True,
                "b3s_activity_held": True,
                "dynamic_whole_unit_dropout_held": True,
            },
            "intervention": {
                "parameter": "alpha", "shape": [], "initial_value": 0.0,
                "gate": "1+0.5*tanh(alpha)*tanh(clamp(log(r)-mean(log(r)),-4,4))",
                "identity_only": True, "activity_gated": False,
                "posterior_mean_used": False, "posterior_normalizer_used": False,
                "posterior_sampling_used": False, "attention_logit_bias_used": False,
            },
            "boundaries": {
                "target_opened": False, "within_opened": False, "external_opened": False,
                "formal_opened": False, "h1_opened": False, "target_optimizer_steps": 0,
                "target_backward_calls": 0, "target_update_calls": 0,
            },
        }


def validate_identity(value: PIRGIdentity | Mapping[str, object]) -> dict[str, object]:
    payload = value.payload() if isinstance(value, PIRGIdentity) else dict(value)
    required = {"schema", "cell", "phase", "classification", "closure", "spec", "sealed_cell_d", "intervention", "boundaries"}
    if set(payload) != required:
        raise PIRGTrainError("PIRG identity schema drift")
    # Reconstructing through a typed object would accept arbitrary closure
    # bytes, so validate closure topology/digest directly and hold every other
    # semantic field to the literal builder.
    closure = payload.get("closure")
    if not isinstance(closure, Mapping) or set(closure) != {"paths", "sha256_by_path", "closure_sha256"}:
        raise PIRGTrainError("PIRG identity closure schema drift")
    if tuple(closure["paths"]) != IMPLEMENTATION_CLOSURE or not isinstance(closure["sha256_by_path"], Mapping):
        raise PIRGTrainError("PIRG identity closure topology drift")
    hashes = {path: _sha(closure["sha256_by_path"].get(path), f"PIRG identity closure {path}") for path in IMPLEMENTATION_CLOSURE}
    body = {"paths": list(IMPLEMENTATION_CLOSURE), "sha256_by_path": hashes}
    if closure["closure_sha256"] != _digest(_json(body)):
        raise PIRGTrainError("PIRG identity closure digest drift")
    expected = PIRGIdentity(ImplementationClosure(hashes)).payload()
    if {**payload, "closure": {**body, "closure_sha256": closure["closure_sha256"]}} != expected:
        raise PIRGTrainError("PIRG identity scientific boundary drift")
    return expected


class _ArtifactRoot:
    """Fresh named-root, O_EXCL/fsync/0444 immutable receipt writer."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        self._parent_fd: int | None = None
        self._dir_fd: int | None = None
        self._identity: tuple[int, int] | None = None

    def reserve(self) -> None:
        parent = self.root.parent
        parent.mkdir(parents=True, exist_ok=True)
        self._parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.mkdir(self.root.name, mode=0o755, dir_fd=self._parent_fd)
        except FileExistsError as error:
            raise PIRGTrainError("PIRG canonical output root already exists") from error
        self._dir_fd = os.open(self.root.name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._parent_fd)
        info = os.fstat(self._dir_fd)
        self._identity = (int(info.st_dev), int(info.st_ino))

    def _require_open(self) -> int:
        if self._dir_fd is None or self._identity is None:
            raise PIRGTrainError("PIRG result root is not reserved")
        info = os.fstat(self._dir_fd)
        if (int(info.st_dev), int(info.st_ino)) != self._identity:
            raise PIRGTrainError("PIRG result root descriptor identity drift")
        return self._dir_fd

    def _publish_bytes(self, name: str, body: bytes) -> str:
        _require(name.endswith((".json", ".pt")) and "/" not in name, "PIRG artifact name drift")
        fd_root = self._require_open()
        digest = _digest(body)
        sidecar = f"{digest}  {name}\n".encode("ascii")
        created: list[str] = []
        try:
            for filename, payload in ((name, body), (f"{name}.sha256", sidecar)):
                fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444, dir_fd=fd_root)
                created.append(filename)
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise PIRGTrainError("PIRG short immutable artifact write")
                        view = view[written:]
                    os.fsync(fd)
                    os.fchmod(fd, 0o444)
                finally:
                    os.close(fd)
            os.fsync(fd_root)
        except BaseException:
            for filename in reversed(created):
                try:
                    os.unlink(filename, dir_fd=fd_root)
                except OSError:
                    pass
            raise
        return digest

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        return self._publish_bytes(name, _json(dict(payload)))

    def publish_binary(self, name: str, body: bytes) -> str:
        return self._publish_bytes(name, body)

    def reload_json(self, name: str, digest: str) -> dict[str, object]:
        fd_root = self._require_open()
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd_root)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                raise PIRGTrainError("PIRG receipt mode/type drift")
            body = b"".join(iter(lambda: os.read(descriptor, 1 << 20), b""))
        finally:
            os.close(descriptor)
        if _digest(body) != digest:
            raise PIRGTrainError("PIRG receipt body digest drift")
        side = os.open(f"{name}.sha256", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd_root)
        try:
            side_info = os.fstat(side)
            if not stat.S_ISREG(side_info.st_mode) or stat.S_IMODE(side_info.st_mode) != 0o444:
                raise PIRGTrainError("PIRG receipt sidecar mode/type drift")
            side_body = b"".join(iter(lambda: os.read(side, 1 << 20), b""))
        finally:
            os.close(side)
        if side_body != f"{digest}  {name}\n".encode("ascii"):
            raise PIRGTrainError("PIRG receipt sidecar drift")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise PIRGTrainError("PIRG receipt JSON root drift")
        return value

    def close(self) -> None:
        for descriptor in (self._dir_fd, self._parent_fd):
            if descriptor is not None:
                os.close(descriptor)
        self._dir_fd = self._parent_fd = None


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    budget_schedule: Sequence[int]
    alpha_before: float
    alpha_after: float
    loss_first: float
    loss_last: float
    loss_min: float
    loss_max: float
    optimizer_steps: int
    only_alpha_gradient: bool
    alpha_gradient_nonzero: bool
    finite_model: bool
    finite_optimizer: bool
    gate_stats_by_budget: Mapping[str, Mapping[str, float]]
    throughput_steps_per_second: float
    cache: Mapping[str, int]

    def payload(self) -> dict[str, object]:
        if type(self.epoch) is not int or not 0 <= self.epoch < plan.EPOCHS:
            raise PIRGTrainError("PIRG epoch-summary epoch drift")
        if list(self.budget_schedule) != list(build_schedule_for_count(len(self.budget_schedule))[self.epoch]):
            raise PIRGTrainError("PIRG epoch-summary budget schedule drift")
        if self.optimizer_steps != SOURCE_STEPS_PER_EPOCH or not self.only_alpha_gradient or not self.alpha_gradient_nonzero:
            raise PIRGTrainError("PIRG epoch-summary optimizer/gradient proof drift")
        if not self.finite_model or not self.finite_optimizer:
            raise PIRGTrainError("PIRG epoch-summary finite-state proof drift")
        stats: dict[str, dict[str, float]] = {}
        if set(self.gate_stats_by_budget) != {str(item) for item in plan.BUDGETS}:
            raise PIRGTrainError("PIRG epoch-summary gate-budget topology drift")
        for budget in plan.BUDGETS:
            row = self.gate_stats_by_budget[str(budget)]
            if not isinstance(row, Mapping) or set(row) != {"min", "mean", "max"}:
                raise PIRGTrainError("PIRG epoch-summary gate stats schema drift")
            minimum, mean, maximum = (_finite(row[key], f"PIRG gate {key}") for key in ("min", "mean", "max"))
            if not 0.5 <= minimum <= mean <= maximum <= 1.5:
                raise PIRGTrainError("PIRG epoch-summary gate bound drift")
            stats[str(budget)] = {"min": minimum, "mean": mean, "max": maximum}
        return {
            "epoch": self.epoch, "budget_schedule": list(self.budget_schedule),
            "alpha_before": _finite(self.alpha_before, "PIRG alpha before"),
            "alpha_after": _finite(self.alpha_after, "PIRG alpha after"),
            "loss_first": _finite(self.loss_first, "PIRG loss first"),
            "loss_last": _finite(self.loss_last, "PIRG loss last"),
            "loss_min": _finite(self.loss_min, "PIRG loss min"),
            "loss_max": _finite(self.loss_max, "PIRG loss max"),
            "optimizer_steps": self.optimizer_steps, "only_alpha_gradient": True,
            "alpha_gradient_nonzero": True, "finite_model": True, "finite_optimizer": True,
            "gate_stats_by_budget": stats,
            "throughput_steps_per_second": _finite(self.throughput_steps_per_second, "PIRG throughput"),
            "cache": {key: int(value) for key, value in self.cache.items()},
        }


def build_schedule_for_count(session_count: int) -> tuple[tuple[int, ...], ...]:
    if type(session_count) is not int or session_count < 1:
        raise PIRGTrainError("PIRG source session count drift")
    return tuple(tuple(plan.budget_for(epoch, index) for index in range(session_count)) for epoch in range(plan.EPOCHS))


def _attempt_payload(identity: PIRGIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_identity_residual_gate_attempt_v1",
        "status": "ATTEMPT_RESERVED_BEFORE_SOURCE_OR_CUDA",
        "identity": validate_identity(identity), "source_opened": False, "cuda_initialized": False,
        "target_opened": False, "within_opened": False, "external_opened": False, "formal_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }


def _failure_payload(*, identity: PIRGIdentity, attempt_sha256: str, source_authority_sha256: str | None,
                     stage: str, source_opened: bool, cuda_initialized: bool, optimizer_steps: int,
                     error: BaseException) -> dict[str, object]:
    _require(stage in {"prepare", "source_authority", "epoch", "final"}, "PIRG failure stage drift")
    return {
        "schema": "posterior_identity_residual_gate_failure_v1", "identity": validate_identity(identity),
        "attempt_sha256": _sha(attempt_sha256, "PIRG failure attempt SHA"),
        "source_authority_sha256": source_authority_sha256,
        "stage": stage, "source_opened": bool(source_opened), "cuda_initialized": bool(cuda_initialized),
        "optimizer_steps_completed": int(optimizer_steps), "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "target_optimizer_steps": 0,
        "target_backward_calls": 0, "target_update_calls": 0,
        "error_class": type(error).__name__, "error_sha256": _digest(repr(error).encode("utf-8")),
        "traceback_sha256": _digest("".join(traceback.format_exception(error)).encode("utf-8")),
        "terminal_published": False,
    }


_CAPABILITY_SEAL = object()


@dataclass(frozen=True)
class PIRGExecutionCapability:
    identity_sha256: str
    _seal: object = field(repr=False, compare=False, default=_CAPABILITY_SEAL)


def _issue_root_review_capability(identity: PIRGIdentity) -> PIRGExecutionCapability:
    return PIRGExecutionCapability(identity_sha256=_digest(_json(validate_identity(identity))))


def _require_capability(value: object, *, identity: PIRGIdentity) -> PIRGExecutionCapability:
    if not isinstance(value, PIRGExecutionCapability) or value._seal is not _CAPABILITY_SEAL or value.identity_sha256 != _digest(_json(validate_identity(identity))):
        raise PIRGTrainError("PIRG root-reviewed in-process capability is required")
    return value


class PIRGBackend(Protocol):
    def prepare(self, *, identity: PIRGIdentity) -> None: ...
    def source_authority(self, *, identity: PIRGIdentity) -> Mapping[str, object]: ...
    def run_epoch(self, *, epoch: int, identity: PIRGIdentity) -> EpochSummary: ...
    def final_artifact(self, *, identity: PIRGIdentity) -> tuple[bytes, Mapping[str, object]]: ...
    def final_reverify(self, *, identity: PIRGIdentity) -> ImplementationClosure: ...
    def progress(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


def _validate_source_authority(value: Mapping[str, object], *, identity: PIRGIdentity) -> dict[str, object]:
    required = {
        "schema", "identity", "source_only", "roster", "budget_schedule", "posterior_credibility_only",
        "ordinary_ols_t4_held", "posterior_mean_used", "posterior_normalizer_used", "posterior_sampling_used",
        "posterior_attention_bias_used", "batch_loop_inverse_calls", "gate_cache", "base_swa_sha256",
        "model_dropout_rng_seed", "control_digest_rows", "source_adapter_metadata_sha256", "remote_torch_authority", "tf32_enforcement",
        "source_opened", "target_opened", "within_opened", "external_opened", "formal_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PIRGTrainError("PIRG source authority schema drift")
    roster = value.get("roster")
    if (value.get("schema") != "posterior_identity_residual_gate_source_authority_v1"
            or value.get("identity") != validate_identity(identity)
            or not isinstance(roster, list) or len(roster) != 27 or len(set(roster)) != 27
            or value.get("budget_schedule") != [list(row) for row in build_schedule_for_count(27)]
            or value.get("source_only") is not True or value.get("posterior_credibility_only") is not True
            or value.get("ordinary_ols_t4_held") is not True or value.get("posterior_mean_used") is not False
            or value.get("posterior_normalizer_used") is not False or value.get("posterior_sampling_used") is not False
            or value.get("posterior_attention_bias_used") is not False or value.get("batch_loop_inverse_calls") != 0
            or value.get("base_swa_sha256") != plan.SEALED_CELL_D_SWA_SHA256
            or value.get("model_dropout_rng_seed") != plan.SEED
            or value.get("source_opened") is not True or any(value.get(key) is not False for key in ("target_opened", "within_opened", "external_opened", "formal_opened"))):
        raise PIRGTrainError("PIRG source authority semantic boundary drift")
    cache = value.get("gate_cache")
    if (not isinstance(cache, Mapping) or cache.get("source_sessions") != 27
            or cache.get("logical_epochs") != plan.EPOCHS
            or cache.get("controls_built") != 27 * plan.EPOCHS
            or cache.get("optimizer_batch_requests") != 0
            or cache.get("optimizer_batch_inverse_calls") != 0
            or cache.get("posterior_mean_view_builds") != 0
            or cache.get("posterior_sampling_view_builds") != 0
            or cache.get("posterior_normalizer_view_builds") != 0):
        raise PIRGTrainError("PIRG source authority cache proof drift")
    rows = value.get("control_digest_rows")
    if not isinstance(rows, list) or len(rows) != 27 * plan.EPOCHS:
        raise PIRGTrainError("PIRG source authority control topology drift")
    expected_rows = []
    for epoch, schedule in enumerate(build_schedule_for_count(27)):
        for index, budget in enumerate(schedule):
            expected_rows.append({"session": roster[index], "session_index": index, "epoch": epoch, "budget": budget})
    for item, expected in zip(rows, expected_rows, strict=True):
        if (not isinstance(item, Mapping) or {key: item.get(key) for key in expected} != expected
                or _sha(item.get("posterior_sha256"), "PIRG source control posterior SHA") is None):
            raise PIRGTrainError("PIRG source authority control row drift")
    if (not isinstance(value.get("remote_torch_authority"), Mapping) or not value["remote_torch_authority"]
            or not isinstance(value.get("tf32_enforcement"), Mapping) or not value["tf32_enforcement"]
            or _sha(value.get("source_adapter_metadata_sha256"), "PIRG source adapter metadata SHA") is None):
        raise PIRGTrainError("PIRG source authority runtime binding drift")
    return dict(value)


def run_training_lifecycle(*, root: Path, identity: PIRGIdentity, backend: PIRGBackend,
                           execution_capability: object) -> dict[str, str]:
    """Future reviewed source-only execution.  Tests use injected CPU mocks."""
    _require_capability(execution_capability, identity=identity)
    validate_identity(identity)
    artifact = _ArtifactRoot(Path(root).absolute() / RESULT_ROOT_RELATIVE)
    attempt_sha: str | None = None
    source_sha: str | None = None
    steps = 0
    stage = "prepare"
    try:
        artifact.reserve()
        attempt_sha = artifact.publish_json("attempt.json", _attempt_payload(identity))
        backend.prepare(identity=identity)
        stage = "source_authority"
        authority = _validate_source_authority(backend.source_authority(identity=identity), identity=identity)
        source_sha = artifact.publish_json("source_authority.json", authority)
        summaries: list[dict[str, object]] = []
        for epoch in range(plan.EPOCHS):
            stage = "epoch"
            summary = backend.run_epoch(epoch=epoch, identity=identity).payload()
            if summary["epoch"] != epoch:
                raise PIRGTrainError("PIRG backend epoch order drift")
            steps += int(summary["optimizer_steps"])
            summary["cumulative_optimizer_steps"] = steps
            epoch_sha = artifact.publish_json(f"epoch-{epoch:02d}.json", summary)
            summaries.append({"epoch": epoch, "sha256": epoch_sha})
        if steps != TOTAL_STEPS:
            raise PIRGTrainError("PIRG total optimizer-step boundary drift")
        stage = "final"
        final_body, final_manifest = backend.final_artifact(identity=identity)
        final_sha = artifact.publish_binary("final_alpha.pt", final_body)
        if not isinstance(final_manifest, Mapping) or final_manifest.get("artifact_sha256") != final_sha:
            raise PIRGTrainError("PIRG final alpha artifact manifest/digest drift")
        final_closure = backend.final_reverify(identity=identity).payload()
        if final_closure != identity.closure.payload():
            raise PIRGTrainError("PIRG launch/final/live closure drift")
        terminal = {
            "schema": "posterior_identity_residual_gate_terminal_v1", "status": "PIRG_SOURCE_TRAINING_COMPLETE",
            "identity": validate_identity(identity), "attempt_sha256": attempt_sha,
            "source_authority_sha256": source_sha, "epochs": summaries,
            "final_alpha_sha256": final_sha, "final_alpha_manifest": dict(final_manifest),
            "optimizer_steps": TOTAL_STEPS, "launch_closure": identity.closure.payload(),
            "final_closure": final_closure, "target_opened": False, "within_opened": False,
            "external_opened": False, "formal_opened": False, "target_optimizer_steps": 0,
            "target_backward_calls": 0, "target_update_calls": 0,
        }
        terminal_sha = artifact.publish_json("terminal.json", terminal)
        return {"attempt_sha256": attempt_sha, "source_authority_sha256": source_sha,
                "final_alpha_sha256": final_sha, "terminal_sha256": terminal_sha}
    except BaseException as error:
        if attempt_sha is not None:
            progress = dict(backend.progress())
            try:
                artifact.publish_json("failure.json", _failure_payload(
                    identity=identity, attempt_sha256=attempt_sha, source_authority_sha256=source_sha,
                    stage=stage, source_opened=bool(progress.get("source_opened", False)),
                    cuda_initialized=bool(progress.get("cuda_initialized", False)), optimizer_steps=steps,
                    error=error,
                ))
            except BaseException:
                pass
        raise
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


class NoLivePIRGBackend:
    """Public dry route guard; fails before source/CUDA/output creation."""

    def prepare(self, *, identity: PIRGIdentity) -> None:
        del identity
        raise PIRGTrainError("PIRG physical backend is unavailable from the dry route")

    def source_authority(self, *, identity: PIRGIdentity) -> Mapping[str, object]:
        raise AssertionError("PIRG dry backend cannot create source authority")

    def run_epoch(self, *, epoch: int, identity: PIRGIdentity) -> EpochSummary:
        raise AssertionError("PIRG dry backend cannot train")

    def final_artifact(self, *, identity: PIRGIdentity) -> tuple[bytes, Mapping[str, object]]:
        raise AssertionError("PIRG dry backend cannot emit an artifact")

    def final_reverify(self, *, identity: PIRGIdentity) -> ImplementationClosure:
        raise AssertionError("PIRG dry backend cannot reverify")

    def progress(self) -> Mapping[str, object]:
        return {"source_opened": False, "cuda_initialized": False}

    def close(self) -> None:
        return None
