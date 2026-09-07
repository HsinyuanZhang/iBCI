"""Fail-closed, source-only seed-43 training lifecycle for TF-SR (build v2).

This is the matched-replication seed-43 route ordered by section 0 of
``HANDOFF_TASK_FRAME_STATEFUL_READIN_20260819.md``.  The recipe is exactly the
frozen seed-42 Phase-D recipe with seed 43:

* identical strict-27 train-only source adapter and authority verification
  (imported from the frozen :mod:`src.tfsr_b3st4_ddrop_v1.source_smoke`;
  only the ``SessionBatchSampler`` draw seed is 43);
* identical frozen model graph, Adam, warmup/cosine schedule, 48 epochs,
  33,925 steps per epoch, final-four-checkpoint SWA, dense valid-bin MSE, and
  Cell-D whole-unit dropout law;
* identical epoch-boundary full-state proof discipline.

Two deliberate differences are both disclosed in every launch/terminal
receipt through the build-disclosure and lineage blocks:

1. the executed forward is the accelerated build v2 (jit-scripted recurrent
   step) whose evidence is the sealed throughput-v2 receipt bound by
   :mod:`.contract_43`;
2. the device authority is physical GPU0 as the sole visible CUDA device
   (``CUDA_VISIBLE_DEVICES=0``).  The live seed-42 run owns GPU1 and this
   route never touches it or its output roots.

Like the frozen route, this module has no top-level torch/data import; those
live strictly inside :class:`TorchTrainingBackend43.prepare`, reachable only
through the two-flag authorized public route.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from src.tfsr_b3st4_ddrop_v1 import source_smoke
from src.tfsr_b3st4_ddrop_v1 import train as train42

from . import contract_43


CELL = contract_43.CELL_43
SEED = contract_43.SEED_43
TRAIN_ROOT_RELATIVE = contract_43.TRAIN_ROOT_RELATIVE
FROZEN_DEVICE_43 = contract_43.FROZEN_DEVICE_43
BUILD_DISCLOSURE = contract_43.BUILD_DISCLOSURE

# Roots this route must never alias or occupy.  The seed-42 v1 root is failed
# immutable evidence, the v2 root belongs to the live GPU1 run, and the two
# throughput roots are sealed engineering receipts.
FORBIDDEN_OUTPUT_ALIASES = (
    "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v1",
    "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v2",
    "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v1",
    "tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v2",
    "tfpd_exploration/results/tfsr_b3st4_ddrop_v1",
)

OPTIMIZER = {
    "cls": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
    "weight_decay": 0.0, "amsgrad": False, "gradient_clipping": None,
}
_CRITICAL_GRADIENT_GROUPS = tuple(source_smoke._CRITICAL_PARAMETER_PREFIXES)
_RESOURCE_KEYS = ("rss_bytes", "peak_allocated_bytes", "peak_reserved_bytes")

# Generic frozen lifecycle helpers reused verbatim (none reference a cell).
_sha = train42._sha
_json = train42._json
_is_sha = train42._is_sha
_finite_number = train42._finite_number
_strict_bool_map = train42._strict_bool_map
_safe_mapping_copy = train42._safe_mapping_copy
ArtifactRoot = train42.ArtifactRoot
reserve_artifact_root = train42.reserve_artifact_root
requires_epoch_proof = train42.requires_epoch_proof
_validate_step_outcome = train42._validate_step_outcome
_quantile = train42._quantile
_resource_payload = train42._resource_payload
_tensor_digest = train42._tensor_digest
_optimizer_digest = train42._optimizer_digest
_epoch_boundary_proof = train42._epoch_boundary_proof
_batch_iter = train42._batch_iter
_capability_for_batch = train42._capability_for_batch
StepOutcome = train42.StepOutcome
CheckpointPayload = train42.CheckpointPayload
SWAPayload = train42.SWAPayload


@dataclass
class LifecycleFlags:
    """Seed-43 lifecycle state; ``lineage`` replaces the frozen route's predecessor."""

    source_opened: bool = False
    gpu_initialized: bool = False
    optimizer_steps_completed: int = 0
    target_or_formal_opened: bool = False
    terminal_published: bool = False
    stage: str = "identity"
    lineage: Mapping[str, Any] | None = None
    identity_closure: Mapping[str, Any] | None = None
    launch_closure: Mapping[str, Any] | None = None

    def boundary_payload(self) -> dict[str, bool]:
        return {
            "source_only": True,
            "target_or_formal_opened": self.target_or_formal_opened,
            "scientific_result": False,
            "score": False,
            "capture_diagnostics": False,
        }

    def lineage_payload(self) -> dict[str, Any]:
        if self.lineage is None:
            raise RuntimeError("lifecycle lineage is not bound")
        contract_43.validate_lineage(self.lineage)
        return _safe_mapping_copy(self.lineage)

    def launch_closure_payload(self) -> dict[str, Any]:
        if self.launch_closure is None:
            raise RuntimeError("lifecycle launch closure is not bound")
        if not isinstance(self.launch_closure, Mapping):
            raise RuntimeError("lifecycle launch closure is malformed")
        return _safe_mapping_copy(self.launch_closure)

    def identity_closure_payload(self) -> dict[str, Any]:
        if self.identity_closure is None or not isinstance(self.identity_closure, Mapping):
            raise RuntimeError("lifecycle identity closure is not bound")
        return _safe_mapping_copy(self.identity_closure)


class TrainingBackend(Protocol):
    """Injected backend: source/CUDA work stays behind this interface."""

    def prepare(self, spec: RunSpec, identity: RunIdentity, flags: LifecycleFlags) -> Any: ...
    def begin_epoch(self, runtime: Any, epoch: int) -> None: ...
    def train_step(self, runtime: Any, lr: float, global_step: int, *, require_epoch_proof: bool) -> StepOutcome: ...
    def after_optimizer_step(self, runtime: Any, global_step: int, flags: LifecycleFlags) -> None: ...
    def resources(self, runtime: Any) -> Mapping[str, int]: ...
    def make_checkpoint(self, runtime: Any, epoch: int, global_step: int, binding: Mapping[str, Any]) -> CheckpointPayload: ...
    def validate_checkpoint(self, body: bytes, epoch: int, global_step: int, spec: RunSpec,
                            *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...
    def build_swa(self, runtime: Any, checkpoints: Mapping[int, bytes], spec: RunSpec,
                  *, expected_binding: Mapping[str, Any]) -> SWAPayload: ...
    def validate_swa(self, body: bytes, spec: RunSpec,
                     *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...
    def close(self, runtime: Any | None) -> None: ...


@dataclass(frozen=True)
class RunSpec:
    """Immutable seed-43 run budget injected into the lifecycle, never parsed from CLI."""

    epochs: int
    batch_size: int
    steps_per_epoch: int
    checkpoint_epochs: tuple[int, ...]
    throughput_probe_steps: int
    seed: int = SEED
    capture_diagnostics: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (self.epochs, self.batch_size, self.steps_per_epoch, self.throughput_probe_steps)):
            raise ValueError("RunSpec integer budgets must be positive exact ints")
        if self.seed != SEED or self.capture_diagnostics is not False:
            raise ValueError("the seed-43 route fixes seed=43 and capture_diagnostics=False")
        if not isinstance(self.checkpoint_epochs, tuple) or not self.checkpoint_epochs:
            raise ValueError("RunSpec must name nonempty checkpoint epochs")
        if tuple(sorted(self.checkpoint_epochs)) != self.checkpoint_epochs or len(set(self.checkpoint_epochs)) != len(self.checkpoint_epochs):
            raise ValueError("checkpoint epochs must be unique sorted tuple")
        if any(isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < self.epochs
               for epoch in self.checkpoint_epochs):
            raise ValueError("checkpoint epoch lies outside run")
        if self.throughput_probe_steps > self.total_optimizer_steps:
            raise ValueError("throughput probe lies outside run")

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def topology(self) -> tuple[str, ...]:
        return (
            "attempt.json", "launch.json", f"throughput{self.throughput_probe_steps}.json", "swa.pt",
            "terminal.json", "failure.json",
            *(f"epoch-{epoch:02d}.json" for epoch in range(self.epochs)),
            *(f"checkpoint-{epoch:02d}.pt" for epoch in self.checkpoint_epochs),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs, "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "total_optimizer_steps": self.total_optimizer_steps,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "throughput_probe_steps": self.throughput_probe_steps,
            "seed": self.seed, "capture_diagnostics": self.capture_diagnostics,
        }


PUBLIC_SPEC = RunSpec(
    epochs=48, batch_size=32, steps_per_epoch=33_925,
    checkpoint_epochs=(44, 45, 46, 47), throughput_probe_steps=100,
)
EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH = PUBLIC_SPEC.epochs, PUBLIC_SPEC.batch_size, PUBLIC_SPEC.steps_per_epoch
TOTAL_STEPS, WARMUP_STEPS = PUBLIC_SPEC.total_optimizer_steps, 2 * PUBLIC_SPEC.steps_per_epoch
TOPOLOGY = PUBLIC_SPEC.topology


def lr_for_step(step: int) -> float:
    """Bitwise live-arm parity for every public optimizer step (seed-independent)."""
    if type(step) is not int or not 0 <= step < PUBLIC_SPEC.total_optimizer_steps:
        raise ValueError("step must be an integer in the exact 48-epoch schedule")
    from tfpd_lane.arm_common import lr_at_step
    return lr_at_step(step, PUBLIC_SPEC.epochs, PUBLIC_SPEC.steps_per_epoch)


def lr_for_spec(spec: RunSpec, step: int) -> float:
    """Use arm-common exactly for public; identical formula for injected mocks."""
    if type(step) is not int or not 0 <= step < spec.total_optimizer_steps:
        raise ValueError("step lies outside RunSpec")
    if spec == PUBLIC_SPEC:
        return lr_for_step(step)
    warm = min(2 * spec.steps_per_epoch, spec.total_optimizer_steps)
    if step < warm:
        return 1e-5 + (1e-4 - 1e-5) * (step / warm)
    span = spec.total_optimizer_steps - warm
    if span <= 0:
        return 1e-4
    progress = (step - warm) / span
    return 1e-6 + 0.5 * (1e-4 - 1e-6) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunIdentity:
    """Receipt identity injected once at launch and recomputed at terminal."""

    phase_c_acceptance: Mapping[str, Any]
    source_authorities: Mapping[str, Any]
    closures: Mapping[str, Any]
    device: Mapping[str, Any]
    lineage: Mapping[str, Any]
    build_disclosure: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "phase_c_acceptance": _safe_mapping_copy(self.phase_c_acceptance),
            "source_authorities": _safe_mapping_copy(self.source_authorities),
            "closures": _safe_mapping_copy(self.closures),
            "device": _safe_mapping_copy(self.device),
            "lineage": _safe_mapping_copy(self.lineage),
            "build_disclosure": _safe_mapping_copy(self.build_disclosure),
        }


def _verify_frozen_phase_d_closure(root: Path) -> dict[str, object]:
    closure = train42.phase_d_closure(root)
    hashes = closure["sha256_by_path"]
    for path, sha in hashes.items():
        expected = contract_43.EXPECTED_FROZEN_ROUTE_SHA.get(path)
        if expected is not None and sha != expected:
            raise RuntimeError(f"frozen seed-42 Phase-D closure drift: {path}")
    return closure


def production_identity(root: Path) -> RunIdentity:
    """Descriptor-only identity binding; opens no NWB, target, or CUDA object."""
    return RunIdentity(
        phase_c_acceptance=train42.verify_phase_c_acceptance(root),
        source_authorities=source_smoke.verify_canonical_source_authorities(root),
        closures={
            **source_smoke.verify_stage0_and_phase_c_closures(root),
            "frozen_phase_d": _verify_frozen_phase_d_closure(root),
            "seed43": contract_43.compute_seed43_closure(root),
        },
        device=dict(FROZEN_DEVICE_43),
        lineage=contract_43.public_lineage(root),
        build_disclosure=contract_43.validate_build_disclosure(BUILD_DISCLOSURE),
    )


def training_plan(root: Path) -> dict[str, object]:
    identity = production_identity(root)
    return {
        "cell": CELL, "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH", "authorization": "none",
        "execution_flags_required_together": ["--execute", "--i-have-48epoch-authorization"],
        "cuda_visible_devices_required_exactly": "0",
        **identity.payload(),
        "training": {
            **PUBLIC_SPEC.payload(), "optimizer": dict(OPTIMIZER),
            "schedule": {"authority": "tfpd_lane.arm_common.lr_at_step", "warmup_epochs": 2,
                         "warmup_start_lr": 1e-5, "warmup_end_lr": 1e-4, "cosine_final_lr": 1e-6},
            "forward": "accelerated build v2 (jit-scripted recurrent step; see build_disclosure)",
            "data_order": "SessionBatchSampler(seed=43) over the frozen train-only strict-27 adapter",
            "validation_or_target": "forbidden", "swa": "arithmetic_checkpoint_state_mean_final_4",
        },
        "canonical_output": {"root_relative": TRAIN_ROOT_RELATIVE, "must_be_fresh_before_execution": True},
    }


def output_gate(root: Path) -> Path:
    """No output collision/alias may survive to source or CUDA imports."""
    target = root / TRAIN_ROOT_RELATIVE
    for forbidden in FORBIDDEN_OUTPUT_ALIASES:
        if TRAIN_ROOT_RELATIVE == forbidden or target.absolute() == (root / forbidden).absolute():
            raise RuntimeError("seed-43 output root may not alias a frozen seed-42 root")
    if target.exists() or target.is_symlink():
        raise RuntimeError("fresh canonical seed-43 output root required")
    ancestor = target.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise RuntimeError("canonical seed-43 output ancestor is invalid")
    return target


def reserve_output_root(root: Path) -> ArtifactRoot:
    output_gate(root)
    return reserve_artifact_root(root, TRAIN_ROOT_RELATIVE, PUBLIC_SPEC.topology)


def _validate_identity(identity: RunIdentity, *, public: bool) -> None:
    if not isinstance(identity, RunIdentity):
        raise RuntimeError("run identity type drift")
    data = identity.payload()
    if set(data) != {"phase_c_acceptance", "source_authorities", "closures", "device", "lineage", "build_disclosure"}:
        raise RuntimeError("run identity key drift")
    if not all(isinstance(data[key], Mapping) for key in data):
        raise RuntimeError("run identity mapping drift")
    contract_43.validate_lineage(data["lineage"])
    contract_43.validate_build_disclosure(data["build_disclosure"])
    if public:
        if data["phase_c_acceptance"].get("body_sha256") != train42.PHASE_C_RECEIPT_SHA:
            raise RuntimeError("public Phase-C acceptance drift")
        if data["device"] != FROZEN_DEVICE_43 or data["device"]["cuda_visible_devices"] != "0":
            raise RuntimeError("public GPU0 device drift")
        if data["source_authorities"].get("manifest_sha256") != source_smoke.MANIFEST_SHA:
            raise RuntimeError("public source authority drift")
        if data["lineage"]["throughput_v2_receipt_sha256"] != contract_43.THROUGHPUT_V2_RECEIPT_SHA256:
            raise RuntimeError("public build-evidence receipt drift")
        if set(data["closures"]) != {"stage0", "phase_c", "frozen_phase_d", "seed43"}:
            raise RuntimeError("public closure map drift")


# --------------------------------------------------------------------------- #
# Receipt payloads and validators
# --------------------------------------------------------------------------- #


def _attempt_payload(spec: RunSpec, identity: RunIdentity) -> dict[str, Any]:
    return {
        "schema": "tfsr_b3st4_ddrop_seed43_train_attempt_v1", "cell": CELL, "run_spec": spec.payload(),
        "phase_c_acceptance": _safe_mapping_copy(identity.phase_c_acceptance),
        "closures": _safe_mapping_copy(identity.closures),
        "lineage": _safe_mapping_copy(identity.lineage),
        "build_disclosure": _safe_mapping_copy(identity.build_disclosure),
        "topology": list(spec.topology),
        "source_opened": False, "gpu_initialized": False, "target_or_formal_opened": False,
    }


def validate_attempt_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "run_spec", "phase_c_acceptance", "closures", "lineage", "build_disclosure",
                "topology", "source_opened", "gpu_initialized", "target_or_formal_opened"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_seed43_train_attempt_v1" or value.get("cell") != CELL):
        raise RuntimeError("attempt receipt schema drift")
    if (value.get("run_spec") != spec.payload() or value.get("topology") != list(spec.topology)
            or value.get("source_opened") is not False or value.get("gpu_initialized") is not False
            or value.get("target_or_formal_opened") is not False):
        raise RuntimeError("attempt receipt state drift")
    contract_43.validate_lineage(value.get("lineage"), exact=(identity.lineage if identity is not None else None))
    if identity is not None and (value.get("phase_c_acceptance") != _safe_mapping_copy(identity.phase_c_acceptance)
                                 or value.get("closures") != _safe_mapping_copy(identity.closures)):
        raise RuntimeError("attempt identity drift")
    contract_43.validate_build_disclosure(value.get("build_disclosure"))


def _launch_payload(spec: RunSpec, identity: RunIdentity) -> dict[str, Any]:
    sample_steps = sorted({0, min(2 * spec.steps_per_epoch - 1, spec.total_optimizer_steps - 1),
                           min(2 * spec.steps_per_epoch, spec.total_optimizer_steps - 1), spec.total_optimizer_steps - 1})
    return {
        "schema": "tfsr_b3st4_ddrop_seed43_train_launch_v1", "cell": CELL, "run_spec": spec.payload(),
        "phase_c_acceptance": _safe_mapping_copy(identity.phase_c_acceptance),
        "source_authorities": _safe_mapping_copy(identity.source_authorities),
        "closures": _safe_mapping_copy(identity.closures), "device": _safe_mapping_copy(identity.device),
        "lineage": _safe_mapping_copy(identity.lineage),
        "build_disclosure": _safe_mapping_copy(identity.build_disclosure),
        "optimizer": dict(OPTIMIZER),
        "schedule_formula": "tfpd_lane.arm_common.lr_at_step (public); mathematically identical injected-spec schedule (mock only)",
        "sampled_lr": {str(step): lr_for_spec(spec, step) for step in sample_steps},
        "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                       "score": False, "capture_diagnostics": False},
    }


def validate_launch_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "run_spec", "phase_c_acceptance", "source_authorities", "closures", "device",
                "lineage", "build_disclosure", "optimizer", "schedule_formula", "sampled_lr", "boundaries"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_seed43_train_launch_v1" or value.get("cell") != CELL):
        raise RuntimeError("launch receipt schema drift")
    if (value.get("run_spec") != spec.payload() or value.get("optimizer") != OPTIMIZER
            or not isinstance(value.get("schedule_formula"), str) or not value["schedule_formula"]):
        raise RuntimeError("launch receipt run/optimizer drift")
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                                   "score": False, "capture_diagnostics": False}:
        raise RuntimeError("launch source-only boundary drift")
    if identity is not None:
        payload = identity.payload()
        if any(value.get(key) != payload[key] for key in ("phase_c_acceptance", "source_authorities", "closures",
                                                          "device", "lineage", "build_disclosure")):
            raise RuntimeError("launch identity drift")
    contract_43.validate_lineage(value.get("lineage"), exact=(identity.lineage if identity is not None else None))
    if spec == PUBLIC_SPEC:
        if (value.get("phase_c_acceptance", {}).get("body_sha256") != train42.PHASE_C_RECEIPT_SHA
                or value.get("device") != FROZEN_DEVICE_43
                or value.get("source_authorities", {}).get("manifest_sha256") != source_smoke.MANIFEST_SHA):
            raise RuntimeError("launch public authority/device drift")
    contract_43.validate_build_disclosure(value.get("build_disclosure"))
    if not isinstance(value.get("sampled_lr"), Mapping) or not value["sampled_lr"] or any(
            not _finite_number(item, positive=True) for item in value["sampled_lr"].values()):
        raise RuntimeError("launch sampled LR drift")


def _epoch_payload(spec: RunSpec, epoch: int, global_step: int, outcomes: Sequence[StepOutcome], elapsed: float,
                   resources: Mapping[str, Any], flags: LifecycleFlags) -> dict[str, Any]:
    if len(outcomes) != spec.steps_per_epoch or not _finite_number(elapsed, positive=True):
        raise RuntimeError("epoch outcome/time accounting drift")
    p = [item.dropout_p for item in outcomes]
    loss = [item.loss for item in outcomes]
    kept, dropped = sum(item.kept for item in outcomes), sum(item.dropped for item in outcomes)
    population = sum(item.population_examples for item in outcomes)
    all_zero = sum(item.all_zero_examples for item in outcomes)
    if kept + dropped <= 0 or population <= 0:
        raise RuntimeError("epoch aggregate mask denominator drift")
    boundary = outcomes[-1]
    if boundary.epoch_boundary_proof is not True or any(item is None for item in (
            boundary.critical_gradients, boundary.finite_model, boundary.finite_optimizer,
            boundary.model_state_digest, boundary.optimizer_state_digest)):
        raise RuntimeError("epoch lacks a final full-state proof")
    if any(item.epoch_boundary_proof for item in outcomes[:-1]):
        raise RuntimeError("epoch has a nonfinal full-state proof")
    return {
        "schema": "tfsr_b3st4_ddrop_seed43_epoch_v1", "cell": CELL, "epoch": epoch,
        "steps": spec.steps_per_epoch, "cumulative_steps": global_step,
        "loss": {"mean": sum(loss) / len(loss), "min": min(loss), "max": max(loss)},
        "lr": {"first": outcomes[0].lr_observed, "last": outcomes[-1].lr_observed,
               "expected_first": lr_for_spec(spec, global_step - spec.steps_per_epoch),
               "expected_last": lr_for_spec(spec, global_step - 1)},
        "dropout": {
            "p_min": min(p), "p_max": max(p), "p_mean": sum(p) / len(p),
            "p_q25": _quantile(p, 0.25), "p_q50": _quantile(p, 0.50), "p_q75": _quantile(p, 0.75),
            "kept": kept, "dropped": dropped, "kept_fraction": kept / (kept + dropped),
            "dropped_fraction": dropped / (kept + dropped), "all_zero_examples": all_zero,
            "population_examples": population, "all_zero_fraction": all_zero / population,
            "max_gain": max(item.max_gain for item in outcomes),
        },
        "critical_gradients": dict(boundary.critical_gradients),
        "finite": {"model": boundary.finite_model, "optimizer": boundary.finite_optimizer},
        "state": {"model_state_digest": boundary.model_state_digest,
                  "optimizer_state_digest": boundary.optimizer_state_digest},
        "lineage": flags.lineage_payload(),
        "launch_closure": flags.launch_closure_payload(),
        "resources": _resource_payload(resources),
        "boundaries": flags.boundary_payload(),
        "progress": {"epoch": epoch, "completed_epochs": epoch + 1, "epochs": spec.epochs,
                     "global_step": global_step, "total_optimizer_steps": spec.total_optimizer_steps},
        "elapsed_seconds": elapsed, "throughput_steps_per_second": spec.steps_per_epoch / elapsed,
    }


def validate_epoch_receipt(value: Mapping[str, Any], epoch: int, cumulative: int,
                           spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "epoch", "steps", "cumulative_steps", "loss", "lr", "dropout",
                "critical_gradients", "finite", "state", "lineage", "launch_closure", "resources", "boundaries",
                "progress", "elapsed_seconds", "throughput_steps_per_second"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_seed43_epoch_v1"):
        raise RuntimeError("epoch receipt schema drift")
    if (value.get("cell") != CELL or value.get("epoch") != epoch or value.get("steps") != spec.steps_per_epoch
            or value.get("cumulative_steps") != cumulative):
        raise RuntimeError("epoch receipt accounting drift")
    loss = value.get("loss")
    if not isinstance(loss, Mapping) or set(loss) != {"mean", "min", "max"} or not all(
            _finite_number(loss[key], nonnegative=True) for key in loss):
        raise RuntimeError("epoch loss payload drift")
    if not loss["min"] <= loss["mean"] <= loss["max"]:
        raise RuntimeError("epoch loss ordering drift")
    lr = value.get("lr")
    if not isinstance(lr, Mapping) or set(lr) != {"first", "last", "expected_first", "expected_last"}:
        raise RuntimeError("epoch LR payload drift")
    first_step, last_step = cumulative - spec.steps_per_epoch, cumulative - 1
    if lr["first"] != lr["expected_first"] or lr["last"] != lr["expected_last"] or lr["first"] != lr_for_spec(spec, first_step) or lr["last"] != lr_for_spec(spec, last_step):
        raise RuntimeError("epoch expected LR drift")
    dropout = value.get("dropout")
    expected_dropout = {"p_min", "p_max", "p_mean", "p_q25", "p_q50", "p_q75", "kept", "dropped",
                        "kept_fraction", "dropped_fraction", "all_zero_examples", "population_examples",
                        "all_zero_fraction", "max_gain"}
    if not isinstance(dropout, Mapping) or set(dropout) != expected_dropout:
        raise RuntimeError("epoch dropout schema drift")
    if not all(_finite_number(dropout[key], nonnegative=True) for key in
               ("p_min", "p_max", "p_mean", "p_q25", "p_q50", "p_q75", "kept_fraction", "dropped_fraction",
                "all_zero_fraction", "max_gain")):
        raise RuntimeError("epoch dropout numeric drift")
    if any(type(dropout[key]) is not int or dropout[key] < 0 for key in
           ("kept", "dropped", "all_zero_examples", "population_examples")):
        raise RuntimeError("epoch dropout count drift")
    if not 0 <= dropout["p_min"] <= dropout["p_q25"] <= dropout["p_q50"] <= dropout["p_q75"] <= dropout["p_max"] <= 1:
        raise RuntimeError("epoch p quantile drift")
    if (dropout["kept"] + dropout["dropped"] <= 0
            or dropout["population_examples"] != spec.steps_per_epoch * spec.batch_size
            or dropout["all_zero_examples"] > dropout["population_examples"]
            or not math.isclose(dropout["kept_fraction"] + dropout["dropped_fraction"], 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(dropout["all_zero_fraction"], dropout["all_zero_examples"] / dropout["population_examples"],
                                rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError("epoch kept/drop fraction drift")
    _strict_bool_map(value.get("critical_gradients"), _CRITICAL_GRADIENT_GROUPS, "epoch critical gradient")
    if value.get("finite") != {"model": True, "optimizer": True}:
        raise RuntimeError("epoch finite proof drift")
    state = value.get("state")
    if (not isinstance(state, Mapping) or set(state) != {"model_state_digest", "optimizer_state_digest"}
            or not all(_is_sha(state[key]) for key in state)):
        raise RuntimeError("epoch state digest drift")
    contract_43.validate_lineage(value.get("lineage"), exact=(identity.lineage if identity is not None else None))
    if not isinstance(value.get("launch_closure"), Mapping):
        raise RuntimeError("epoch launch closure drift")
    if identity is not None and value.get("launch_closure") != _safe_mapping_copy(identity.closures):
        raise RuntimeError("epoch exact launch closure drift")
    _resource_payload(value.get("resources"))
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                                   "score": False, "capture_diagnostics": False}:
        raise RuntimeError("epoch source-only boundary drift")
    expected_progress = {"epoch": epoch, "completed_epochs": epoch + 1, "epochs": spec.epochs,
                         "global_step": cumulative, "total_optimizer_steps": spec.total_optimizer_steps}
    if (value.get("progress") != expected_progress or not _finite_number(value.get("elapsed_seconds"), positive=True)
            or not _finite_number(value.get("throughput_steps_per_second"), positive=True)):
        raise RuntimeError("epoch progress/resource timing drift")


def _throughput_payload(spec: RunSpec, elapsed: float, resources: Mapping[str, Any], flags: LifecycleFlags) -> dict[str, Any]:
    if not _finite_number(elapsed, positive=True):
        raise RuntimeError("throughput elapsed drift")
    rate = spec.throughput_probe_steps / elapsed
    return {
        "schema": "tfsr_b3st4_ddrop_seed43_throughput_v1", "cell": CELL,
        "threshold_steps": spec.throughput_probe_steps, "elapsed_seconds": elapsed,
        "steps_per_second": rate,
        "eta_seconds_estimate_only": (spec.total_optimizer_steps - spec.throughput_probe_steps) / rate,
        "lineage": flags.lineage_payload(), "launch_closure": flags.launch_closure_payload(),
        "resources": _resource_payload(resources), "boundaries": flags.boundary_payload(),
    }


def validate_throughput_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC,
                                identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "threshold_steps", "elapsed_seconds", "steps_per_second", "eta_seconds_estimate_only",
                "lineage", "launch_closure", "resources", "boundaries"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_seed43_throughput_v1" or value.get("cell") != CELL
            or value.get("threshold_steps") != spec.throughput_probe_steps):
        raise RuntimeError("throughput receipt schema drift")
    if (not all(_finite_number(value[key], nonnegative=True) for key in
                ("elapsed_seconds", "steps_per_second", "eta_seconds_estimate_only"))
            or value["elapsed_seconds"] <= 0 or value["steps_per_second"] <= 0):
        raise RuntimeError("throughput receipt numeric drift")
    _resource_payload(value.get("resources"))
    contract_43.validate_lineage(value.get("lineage"), exact=(identity.lineage if identity is not None else None))
    if not isinstance(value.get("launch_closure"), Mapping):
        raise RuntimeError("throughput launch closure drift")
    if identity is not None and value.get("launch_closure") != _safe_mapping_copy(identity.closures):
        raise RuntimeError("throughput exact launch closure drift")
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                                   "score": False, "capture_diagnostics": False}:
        raise RuntimeError("throughput source-only boundary drift")


def validate_failure_receipt(value: Mapping[str, Any], identity: RunIdentity | None = None) -> None:
    expected = {"schema", "cell", "stage", "source_opened", "gpu_initialized", "optimizer_steps_completed",
                "target_or_formal_opened", "terminal_published", "lineage", "closures", "traceback_sha256"}
    allowed_stages = {"identity", "attempt", "backend_prepare", "epoch", "optimizer_step", "checkpoint", "swa", "terminal"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_seed43_train_failure_v1" or value.get("cell") != CELL):
        raise RuntimeError("failure receipt schema drift")
    if (value.get("stage") not in allowed_stages or type(value.get("source_opened")) is not bool
            or type(value.get("gpu_initialized")) is not bool
            or type(value.get("target_or_formal_opened")) is not bool or value.get("terminal_published") is not False):
        raise RuntimeError("failure receipt boundary drift")
    if (type(value.get("optimizer_steps_completed")) is not int or value["optimizer_steps_completed"] < 0
            or not _is_sha(value.get("traceback_sha256"))):
        raise RuntimeError("failure receipt accounting drift")
    contract_43.validate_lineage(value.get("lineage"), exact=(identity.lineage if identity is not None else None))
    if not isinstance(value.get("closures"), Mapping):
        raise RuntimeError("failure receipt closure drift")
    if identity is not None and value.get("closures") != _safe_mapping_copy(identity.closures):
        raise RuntimeError("failure receipt exact closure drift")


def _validate_swa_evaluation_proof(proof: object, spec: RunSpec) -> Mapping[str, Any]:
    expected = {"checkpoint_epochs", "checkpoint_model_state_digests", "fresh_strict_load", "eval_mode",
                "capture_diagnostics", "repeat_bitwise_equal", "state_unchanged", "eval_no_mask",
                "forward_build", "prediction_shape", "prediction_sha256", "state_digest_before_eval",
                "state_digest_after_eval", "boundaries"}
    if (not isinstance(proof, Mapping) or set(proof) != expected
            or proof.get("checkpoint_epochs") != list(spec.checkpoint_epochs)):
        raise RuntimeError("SWA evaluation proof schema drift")
    if (proof.get("fresh_strict_load") is not True or proof.get("eval_mode") is not True
            or proof.get("capture_diagnostics") is not False or proof.get("repeat_bitwise_equal") is not True
            or proof.get("state_unchanged") is not True or proof.get("eval_no_mask") is not True
            or proof.get("forward_build") != contract_43.BUILD_DISCLOSURE["build"]
            or not _is_sha(proof.get("prediction_sha256")) or not _is_sha(proof.get("state_digest_before_eval"))
            or not _is_sha(proof.get("state_digest_after_eval"))
            or proof["state_digest_before_eval"] != proof["state_digest_after_eval"]):
        raise RuntimeError("SWA evaluation proof semantic drift")
    if proof.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                                   "score": False}:
        raise RuntimeError("SWA source-only boundary drift")
    shape = proof.get("prediction_shape")
    if not isinstance(shape, list) or len(shape) != 3 or type(shape[0]) is not int or shape[0] <= 0 or shape[1:] != [50, 2]:
        raise RuntimeError("SWA prediction shape drift")
    expected_epochs = {str(epoch) for epoch in spec.checkpoint_epochs}
    digest_map = proof.get("checkpoint_model_state_digests")
    if (not isinstance(digest_map, Mapping) or set(digest_map) != expected_epochs
            or not all(_is_sha(item) for item in digest_map.values())):
        raise RuntimeError("SWA checkpoint digest proof drift")
    return proof


def _terminal_payload(spec: RunSpec, identity: RunIdentity, final_identity: RunIdentity, *, global_step: int,
                      attempt_sha: str, launch_sha: str, throughput_sha: str, epoch_sha: list[str],
                      checkpoint_sha: Mapping[str, str], swa_sha: str, swa_state_digest: str,
                      swa_proof: Mapping[str, Any], artifact_validation: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": "tfsr_b3st4_ddrop_seed43_train_terminal_v1", "status": "TRAINING_COMPLETE", "cell": CELL,
        "run_spec": spec.payload(), "phase_c_acceptance": _safe_mapping_copy(identity.phase_c_acceptance),
        "source_authorities": _safe_mapping_copy(identity.source_authorities),
        "launch_closure": _safe_mapping_copy(identity.closures), "final_closure": _safe_mapping_copy(final_identity.closures),
        "device": _safe_mapping_copy(identity.device), "lineage": _safe_mapping_copy(identity.lineage),
        "build_disclosure": _safe_mapping_copy(identity.build_disclosure),
        "epochs": spec.epochs, "steps_per_epoch": spec.steps_per_epoch,
        "total_optimizer_steps": global_step, "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
        "throughput_sha256": throughput_sha, "epoch_receipt_sha256": epoch_sha,
        "checkpoint_sha256": dict(checkpoint_sha), "swa_sha256": swa_sha,
        "swa_state_digest": swa_state_digest, "swa_evaluation_proof": _safe_mapping_copy(swa_proof),
        "artifact_validation": dict(artifact_validation),
        "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                       "score": False, "capture_diagnostics": False},
    }


def validate_terminal_receipt(value: Mapping[str, Any], spec: RunSpec = PUBLIC_SPEC, identity: RunIdentity | None = None) -> None:
    expected = {"schema", "status", "cell", "run_spec", "phase_c_acceptance", "source_authorities", "launch_closure",
                "final_closure", "device", "lineage", "build_disclosure", "epochs", "steps_per_epoch",
                "total_optimizer_steps", "attempt_sha256", "launch_sha256", "throughput_sha256",
                "epoch_receipt_sha256", "checkpoint_sha256", "swa_sha256", "swa_state_digest",
                "swa_evaluation_proof", "artifact_validation", "boundaries"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_b3st4_ddrop_seed43_train_terminal_v1"
            or value.get("status") != "TRAINING_COMPLETE" or value.get("cell") != CELL):
        raise RuntimeError("terminal receipt schema drift")
    if (value.get("run_spec") != spec.payload() or value.get("epochs") != spec.epochs
            or value.get("steps_per_epoch") != spec.steps_per_epoch
            or value.get("total_optimizer_steps") != spec.total_optimizer_steps):
        raise RuntimeError("terminal accounting drift")
    if value.get("launch_closure") != value.get("final_closure"):
        raise RuntimeError("terminal launch/final closure drift")
    if value.get("boundaries") != {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                                   "score": False, "capture_diagnostics": False}:
        raise RuntimeError("terminal source-only boundary drift")
    if identity is not None:
        payload = identity.payload()
        if any(value.get(key) != payload[key] for key in ("phase_c_acceptance", "source_authorities", "device",
                                                          "lineage", "build_disclosure")) or value.get("launch_closure") != payload["closures"]:
            raise RuntimeError("terminal identity drift")
    contract_43.validate_lineage(value.get("lineage"), exact=(identity.lineage if identity is not None else None))
    contract_43.validate_build_disclosure(value.get("build_disclosure"))
    direct_hashes = ("attempt_sha256", "launch_sha256", "throughput_sha256", "swa_sha256", "swa_state_digest")
    if not all(_is_sha(value.get(key)) for key in direct_hashes):
        raise RuntimeError("terminal direct SHA drift")
    epochs = value.get("epoch_receipt_sha256")
    if not isinstance(epochs, list) or len(epochs) != spec.epochs or not all(_is_sha(item) for item in epochs):
        raise RuntimeError("terminal epoch SHA list drift")
    checkpoints = value.get("checkpoint_sha256")
    expected_checkpoint_keys = {str(epoch) for epoch in spec.checkpoint_epochs}
    if (not isinstance(checkpoints, Mapping) or set(checkpoints) != expected_checkpoint_keys
            or not all(_is_sha(item) for item in checkpoints.values())):
        raise RuntimeError("terminal checkpoint SHA map drift")
    try:
        _validate_swa_evaluation_proof(value.get("swa_evaluation_proof"), spec)
    except RuntimeError as error:
        raise RuntimeError("terminal SWA evaluation proof drift") from error
    artifacts = value.get("artifact_validation")
    expected_artifacts = {"attempt.json", "launch.json", f"throughput{spec.throughput_probe_steps}.json", "swa.pt",
                          *(f"epoch-{epoch:02d}.json" for epoch in range(spec.epochs)),
                          *(f"checkpoint-{epoch:02d}.pt" for epoch in spec.checkpoint_epochs)}
    if (not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts
            or not all(_is_sha(item) for item in artifacts.values())):
        raise RuntimeError("terminal artifact validation map drift")


def _failure_payload(flags: LifecycleFlags) -> dict[str, Any]:
    return {
        "schema": "tfsr_b3st4_ddrop_seed43_train_failure_v1", "cell": CELL, "stage": flags.stage,
        "source_opened": flags.source_opened, "gpu_initialized": flags.gpu_initialized,
        "optimizer_steps_completed": flags.optimizer_steps_completed,
        "target_or_formal_opened": flags.target_or_formal_opened,
        "terminal_published": flags.terminal_published,
        "lineage": flags.lineage_payload(),
        "closures": flags.identity_closure_payload(),
        "traceback_sha256": _sha(traceback.format_exc().encode()),
    }


def _publish_failure(artifact: ArtifactRoot, flags: LifecycleFlags, identity: RunIdentity | None = None) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json"):
        raise RuntimeError("failure publication is forbidden after terminal")
    if artifact.has_name("failure.json"):
        return
    receipt = _failure_payload(flags)
    validate_failure_receipt(receipt, identity)
    digest = artifact.publish_json("failure.json", receipt)
    validate_failure_receipt(artifact.reload_json("failure.json", digest), identity)


def _validate_all_preterminal_artifacts(artifact: ArtifactRoot, spec: RunSpec, identity: RunIdentity,
                                        backend: TrainingBackend, hashes: Mapping[str, str]) -> dict[str, str]:
    """Reload every body+sidecar and its schema before terminal publication."""
    expected_names = {"attempt.json", "launch.json", f"throughput{spec.throughput_probe_steps}.json", "swa.pt",
                      *(f"epoch-{epoch:02d}.json" for epoch in range(spec.epochs)),
                      *(f"checkpoint-{epoch:02d}.pt" for epoch in spec.checkpoint_epochs)}
    if set(hashes) != expected_names:
        raise RuntimeError("preterminal artifact hash set drift")
    validate_attempt_receipt(artifact.reload_json("attempt.json", hashes["attempt.json"]), spec, identity)
    validate_launch_receipt(artifact.reload_json("launch.json", hashes["launch.json"]), spec, identity)
    throughput_name = f"throughput{spec.throughput_probe_steps}.json"
    validate_throughput_receipt(artifact.reload_json(throughput_name, hashes[throughput_name]), spec, identity)
    for epoch in range(spec.epochs):
        name = f"epoch-{epoch:02d}.json"
        validate_epoch_receipt(artifact.reload_json(name, hashes[name]), epoch, (epoch + 1) * spec.steps_per_epoch,
                               spec, identity)
    expected_binding = {
        "cell": CELL, "run_spec": spec.payload(), "launch_sha256": hashes["launch.json"],
        "launch_closure": _safe_mapping_copy(identity.closures),
        "lineage": _safe_mapping_copy(identity.lineage),
    }
    for epoch in spec.checkpoint_epochs:
        name = f"checkpoint-{epoch:02d}.pt"
        backend.validate_checkpoint(artifact.reload_pair(name, hashes[name]), epoch, (epoch + 1) * spec.steps_per_epoch,
                                    spec, expected_binding=expected_binding)
    backend.validate_swa(artifact.reload_pair("swa.pt", hashes["swa.pt"]), spec, expected_binding=expected_binding)
    if artifact.has_name("failure.json") or artifact.has_name("terminal.json"):
        raise RuntimeError("terminal finalizer sees an illegal prior terminal/failure artifact")
    return dict(hashes)


def run_lifecycle(*, spec: RunSpec, backend: TrainingBackend, artifact: ArtifactRoot,
                  identity_factory: Callable[[], RunIdentity]) -> Mapping[str, Any]:
    """Run the injected lifecycle and leave either terminal or failure evidence."""
    flags = LifecycleFlags()
    runtime: Any | None = None
    identity: RunIdentity | None = None
    try:
        flags.stage = "identity"
        identity = identity_factory()
        _validate_identity(identity, public=(spec == PUBLIC_SPEC))
        flags.lineage = _safe_mapping_copy(identity.lineage)
        flags.identity_closure = _safe_mapping_copy(identity.closures)
        flags.stage = "attempt"
        attempt = _attempt_payload(spec, identity)
        validate_attempt_receipt(attempt, spec, identity)
        hashes: dict[str, str] = {"attempt.json": artifact.publish_json("attempt.json", attempt)}
        validate_attempt_receipt(artifact.reload_json("attempt.json", hashes["attempt.json"]), spec, identity)

        flags.stage = "backend_prepare"
        runtime = backend.prepare(spec, identity, flags)
        if flags.target_or_formal_opened:
            raise RuntimeError("target/formal access is forbidden in the seed-43 route")
        flags.stage = "epoch"
        launch = _launch_payload(spec, identity)
        validate_launch_receipt(launch, spec, identity)
        hashes["launch.json"] = artifact.publish_json("launch.json", launch)
        validate_launch_receipt(artifact.reload_json("launch.json", hashes["launch.json"]), spec, identity)
        flags.launch_closure = _safe_mapping_copy(identity.closures)
        checkpoint_binding = {
            "cell": CELL, "run_spec": spec.payload(), "launch_sha256": hashes["launch.json"],
            "launch_closure": _safe_mapping_copy(identity.closures),
            "lineage": _safe_mapping_copy(identity.lineage),
        }

        global_step = 0
        checkpoint_bodies: dict[int, bytes] = {}
        epoch_sha: list[str] = []
        throughput_written = False
        for epoch in range(spec.epochs):
            backend.begin_epoch(runtime, epoch)
            start = time.perf_counter()
            outcomes: list[StepOutcome] = []
            for _ in range(spec.steps_per_epoch):
                expected_lr = lr_for_spec(spec, global_step)
                boundary_proof = requires_epoch_proof(spec, global_step)
                flags.stage = "optimizer_step"
                outcome = backend.train_step(runtime, expected_lr, global_step, require_epoch_proof=boundary_proof)
                _validate_step_outcome(outcome, expected_lr, require_epoch_proof=boundary_proof)
                outcomes.append(outcome)
                global_step += 1
                flags.optimizer_steps_completed = global_step
                backend.after_optimizer_step(runtime, global_step, flags)
                if global_step == spec.throughput_probe_steps:
                    elapsed = max(time.perf_counter() - start, 1e-12)
                    payload = _throughput_payload(spec, elapsed, backend.resources(runtime), flags)
                    validate_throughput_receipt(payload, spec, identity)
                    name = f"throughput{spec.throughput_probe_steps}.json"
                    hashes[name] = artifact.publish_json(name, payload)
                    validate_throughput_receipt(artifact.reload_json(name, hashes[name]), spec, identity)
                    throughput_written = True
            flags.stage = "epoch"
            elapsed = max(time.perf_counter() - start, 1e-12)
            receipt = _epoch_payload(spec, epoch, global_step, outcomes, elapsed, backend.resources(runtime), flags)
            validate_epoch_receipt(receipt, epoch, global_step, spec, identity)
            name = f"epoch-{epoch:02d}.json"
            digest = artifact.publish_json(name, receipt)
            hashes[name] = digest
            epoch_sha.append(digest)
            validate_epoch_receipt(artifact.reload_json(name, digest), epoch, global_step, spec, identity)
            if epoch in spec.checkpoint_epochs:
                flags.stage = "checkpoint"
                checkpoint = backend.make_checkpoint(runtime, epoch, global_step, checkpoint_binding)
                if not isinstance(checkpoint, CheckpointPayload) or not isinstance(checkpoint.body, bytes) or not _is_sha(checkpoint.model_state_digest):
                    raise RuntimeError("checkpoint backend payload drift")
                backend.validate_checkpoint(checkpoint.body, epoch, global_step, spec, expected_binding=checkpoint_binding)
                name = f"checkpoint-{epoch:02d}.pt"
                digest = artifact.publish_bytes(name, checkpoint.body)
                hashes[name] = digest
                canonical = artifact.reload_pair(name, digest)
                backend.validate_checkpoint(canonical, epoch, global_step, spec, expected_binding=checkpoint_binding)
                checkpoint_bodies[epoch] = canonical
        if global_step != spec.total_optimizer_steps or not throughput_written or len(checkpoint_bodies) != len(spec.checkpoint_epochs):
            raise RuntimeError("terminal training accounting drift")

        flags.stage = "swa"
        swa = backend.build_swa(runtime, checkpoint_bodies, spec, expected_binding=checkpoint_binding)
        if not isinstance(swa, SWAPayload) or not isinstance(swa.body, bytes) or not _is_sha(swa.state_digest) or not isinstance(swa.evaluation_proof, Mapping):
            raise RuntimeError("SWA backend payload drift")
        backend.validate_swa(swa.body, spec, expected_binding=checkpoint_binding)
        hashes["swa.pt"] = artifact.publish_bytes("swa.pt", swa.body)
        backend.validate_swa(artifact.reload_pair("swa.pt", hashes["swa.pt"]), spec,
                             expected_binding=checkpoint_binding)

        flags.stage = "terminal"
        final_identity = identity_factory()
        _validate_identity(final_identity, public=(spec == PUBLIC_SPEC))
        if identity.payload() != final_identity.payload():
            raise RuntimeError("launch/final identity drift")
        validation = _validate_all_preterminal_artifacts(artifact, spec, identity, backend, hashes)
        checkpoint_sha = {str(epoch): hashes[f"checkpoint-{epoch:02d}.pt"] for epoch in spec.checkpoint_epochs}
        terminal = _terminal_payload(spec, identity, final_identity, global_step=global_step,
                                     attempt_sha=hashes["attempt.json"], launch_sha=hashes["launch.json"],
                                     throughput_sha=hashes[f"throughput{spec.throughput_probe_steps}.json"],
                                     epoch_sha=epoch_sha, checkpoint_sha=checkpoint_sha,
                                     swa_sha=hashes["swa.pt"], swa_state_digest=swa.state_digest,
                                     swa_proof=swa.evaluation_proof, artifact_validation=validation)
        validate_terminal_receipt(terminal, spec, identity)
        terminal_sha = artifact.publish_json("terminal.json", terminal)
        validate_terminal_receipt(artifact.reload_json("terminal.json", terminal_sha), spec, identity)
        flags.terminal_published = True
        return terminal
    except BaseException:
        try:
            _publish_failure(artifact, flags, identity)
        finally:
            try:
                backend.close(runtime)
            except BaseException:
                pass
        raise
    finally:
        if flags.terminal_published:
            backend.close(runtime)


# --------------------------------------------------------------------------- #
# Physical GPU0 backend with the accelerated build v2 forward
# --------------------------------------------------------------------------- #


def require_single_visible_gpu0(torch: Any) -> dict[str, object]:
    """Accept exactly physical GPU0 exposed as the sole logical CUDA device.

    The live seed-42 run owns physical GPU1 under ``CUDA_VISIBLE_DEVICES=1``;
    this route refuses every other visibility mask so the two runs cannot
    share a device by accident.
    """
    import subprocess

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("seed-43 route requires CUDA_VISIBLE_DEVICES=0 exactly (physical GPU0)")
    lines = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid,pci.bus_id,name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi/CVD visible-device count mismatch")
    uuid, bdf, name, memory = [part.strip() for part in lines[0].split(",", 3)]
    expected = FROZEN_DEVICE_43
    if (uuid != expected["uuid"] or bdf != expected["bdf"] or name != expected["name"]
            or int(memory) != expected["memory_total_mib"]):
        raise RuntimeError("physical GPU0 UUID/BDF identity mismatch")
    return dict(FROZEN_DEVICE_43)


class _Seed43TrainAdapter:
    """Public wrapper: the frozen verified adapter with a seed-43 batch sampler.

    Everything authoritative (records, raw-T4 digests, theta authority, the
    train-only dataset) is carried over untouched from the frozen verified
    adapter; only ``_sampler`` is a fresh ``SessionBatchSampler(seed=43)`` so
    the matched-replication draw differs from seed 42 exactly through the seed.
    """

    def __init__(self, frozen_adapter: Any, sampler: Any, *, sampler_seed: int) -> None:
        if sampler_seed != SEED:
            raise RuntimeError("seed-43 adapter sampler seed drift")
        self.records = frozen_adapter.records
        self.raw_t4_sha256 = frozen_adapter.raw_t4_sha256
        self.theta_authority = frozen_adapter.theta_authority
        self._dataset = frozen_adapter._dataset
        self._sampler = sampler
        self.sampler_seed = sampler_seed

    def first_seed43_batch(self):
        import torch

        batch = next(iter(self._sampler))
        rows = [self._dataset[index] for index in batch]
        return torch.utils.data.default_collate(rows)


def build_seed43_adapter(root: Path, authorities: Mapping[str, Any]) -> _Seed43TrainAdapter:
    """Frozen train-only strict-27 adapter; data order drawn with seed 43."""
    frozen = source_smoke.build_train_only_adapter(root, authorities)
    from mc_maze.multisession_datamodule import SessionBatchSampler

    sampler = SessionBatchSampler(frozen._dataset, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    if len(sampler) != STEPS_PER_EPOCH:
        raise RuntimeError("seed-43 train-only sampler step drift")
    return _Seed43TrainAdapter(frozen, sampler, sampler_seed=SEED)


class TrainingBackend43:
    """The one physical source-only GPU0 backend; instantiated only after authorization."""

    def __init__(self, root: Path):
        self.root = root

    def prepare(self, spec: RunSpec, identity: RunIdentity, flags: LifecycleFlags) -> dict[str, Any]:
        if spec != PUBLIC_SPEC:
            raise RuntimeError("physical backend accepts only the frozen public seed-43 RunSpec")
        import random

        import numpy as np
        import torch

        from src.tfsr_b3st4_ddrop_v1.model import TFSRDecoder

        from . import accelerated_forward

        authorities = source_smoke.verify_canonical_source_authorities(self.root)
        if _safe_mapping_copy(authorities) != _safe_mapping_copy(identity.source_authorities):
            raise RuntimeError("launch source authorities drift before adapter open")
        require_single_visible_gpu0(torch)
        flags.gpu_initialized = True
        flags.source_opened = True
        adapter = build_seed43_adapter(self.root, authorities)
        # Seed 43 everywhere the seed-42 route used 42: init/dropout RNG here,
        # the batch-sampler draw inside the adapter.
        random.seed(spec.seed); np.random.seed(spec.seed); torch.manual_seed(spec.seed); torch.cuda.manual_seed_all(spec.seed)
        model = TFSRDecoder(capture_diagnostics=False).to("cuda:0")
        scripted_step = accelerated_forward.build_scripted_step_binding(torch, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0.0, amsgrad=False)
        return {"torch": torch, "adapter": adapter, "authorities": authorities, "model": model,
                "scripted_step": scripted_step, "optimizer": optimizer, "iterator": None, "fixed_eval": None}

    def begin_epoch(self, runtime: Mapping[str, Any], epoch: int) -> None:
        torch = runtime["torch"]
        torch.cuda.reset_peak_memory_stats(0)
        runtime["iterator"] = iter(_batch_iter(runtime["adapter"], torch))

    def train_step(self, runtime: Mapping[str, Any], lr: float, global_step: int, *,
                   require_epoch_proof: bool) -> StepOutcome:
        from . import accelerated_forward

        torch, adapter, model, optimizer = runtime["torch"], runtime["adapter"], runtime["model"], runtime["optimizer"]
        batch = next(runtime["iterator"])
        cap, _ = _capability_for_batch(batch, adapter, runtime["authorities"], torch)
        x, behavior, calib = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
        if runtime["fixed_eval"] is None:
            runtime["fixed_eval"] = (x.detach().clone(), calib.detach().clone(), cap)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        prediction = accelerated_forward.accelerated_forward(
            torch, model, runtime["scripted_step"], x, calib, cap)
        valid = (behavior != -1.0).all(dim=-1)
        loss = model.dense_valid_bin_mse(prediction, behavior, valid)
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite source-only training loss")
        loss.backward()
        optimizer.step()
        # Full gradients, full parameter/Adam finite scans, and CPU state
        # digests are paid once at the final optimizer step of each epoch,
        # exactly as in the frozen seed-42 route.
        if require_epoch_proof:
            (gradients, finite_model, finite_optimizer, model_state_digest,
             optimizer_state_digest) = _epoch_boundary_proof(
                 model, optimizer, torch, source_smoke.require_critical_gradients,
             )
        else:
            gradients = None
            finite_model = None
            finite_optimizer = None
            model_state_digest = None
            optimizer_state_digest = None
        gain, survivor, p = model.last_unit_gain_mask, model.last_unit_survivor_mask, model.last_dropout_p
        if gain is None or survivor is None or p is None:
            raise RuntimeError("training dropout accounting missing")
        kept = int(survivor.sum().item())
        return StepOutcome(
            loss=float(loss.item()), lr_observed=lr, dropout_p=float(p.item()), kept=kept,
            dropped=survivor.numel() - kept, all_zero_examples=int((survivor.sum(dim=1) == 0).sum().item()),
            population_examples=int(survivor.shape[0]), max_gain=float(gain.max().item()),
            epoch_boundary_proof=require_epoch_proof, critical_gradients=gradients,
            finite_model=finite_model, finite_optimizer=finite_optimizer,
            model_state_digest=model_state_digest, optimizer_state_digest=optimizer_state_digest,
        )

    def after_optimizer_step(self, runtime: Any, global_step: int, flags: LifecycleFlags) -> None:
        return None

    def resources(self, runtime: Mapping[str, Any]) -> Mapping[str, int]:
        import resource

        torch = runtime["torch"]
        return {"rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}

    @staticmethod
    def _checkpoint_bytes(epoch: int, global_step: int, model: Any, optimizer: Any,
                          binding: Mapping[str, Any], torch: Any) -> CheckpointPayload:
        import io

        state = model.state_dict()
        digest = _tensor_digest(state, torch)
        value = {"schema": "tfsr_b3st4_ddrop_seed43_checkpoint_v1", "epoch": epoch, "global_step": global_step,
                 "model_state": state, "optimizer_state": optimizer.state_dict(), "model_state_digest": digest,
                 "binding": _safe_mapping_copy(binding)}
        stream = io.BytesIO()
        torch.save(value, stream)
        return CheckpointPayload(stream.getvalue(), digest)

    def make_checkpoint(self, runtime: Mapping[str, Any], epoch: int, global_step: int,
                        binding: Mapping[str, Any]) -> CheckpointPayload:
        return self._checkpoint_bytes(epoch, global_step, runtime["model"], runtime["optimizer"], binding, runtime["torch"])

    def validate_checkpoint(self, body: bytes, epoch: int, global_step: int, spec: RunSpec, *,
                            expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        import io

        import torch

        value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        expected = {"schema", "epoch", "global_step", "model_state", "optimizer_state", "model_state_digest", "binding"}
        if (not isinstance(value, Mapping) or set(value) != expected
                or value.get("schema") != "tfsr_b3st4_ddrop_seed43_checkpoint_v1"
                or value.get("epoch") != epoch or value.get("global_step") != global_step):
            raise RuntimeError("checkpoint schema/accounting drift")
        if (not isinstance(value.get("model_state"), Mapping)
                or _tensor_digest(value["model_state"], torch) != value.get("model_state_digest")
                or not _is_sha(value.get("model_state_digest"))):
            raise RuntimeError("checkpoint model-state digest drift")
        binding = value.get("binding")
        expected_keys = {"cell", "run_spec", "launch_sha256", "launch_closure", "lineage"}
        if (not isinstance(binding, Mapping) or set(binding) != expected_keys or binding.get("cell") != CELL
                or binding.get("run_spec") != spec.payload() or not _is_sha(binding.get("launch_sha256"))
                or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("checkpoint binding drift")
        contract_43.validate_lineage(binding.get("lineage"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("checkpoint exact launch binding drift")
        return value

    def build_swa(self, runtime: Mapping[str, Any], checkpoints: Mapping[int, bytes], spec: RunSpec, *,
                  expected_binding: Mapping[str, Any]) -> SWAPayload:
        import io

        torch = runtime["torch"]

        from src.tfsr_b3st4_ddrop_v1.model import TFSRDecoder

        from . import accelerated_forward

        if set(checkpoints) != set(spec.checkpoint_epochs):
            raise RuntimeError("SWA checkpoint epoch set drift")
        states: list[Mapping[str, Any]] = []
        checkpoint_digests: dict[str, str] = {}
        for epoch in spec.checkpoint_epochs:
            item = self.validate_checkpoint(checkpoints[epoch], epoch, (epoch + 1) * spec.steps_per_epoch, spec,
                                            expected_binding=expected_binding)
            states.append(item["model_state"])
            checkpoint_digests[str(epoch)] = item["model_state_digest"]
        keys = tuple(states[0])
        if any(tuple(state) != keys for state in states[1:]):
            raise RuntimeError("SWA checkpoint state key drift")
        swa_state: dict[str, Any] = {}
        for key in keys:
            values = [state[key] for state in states]
            if not all(torch.is_tensor(value) and value.shape == values[0].shape and value.dtype == values[0].dtype for value in values):
                raise RuntimeError("SWA checkpoint tensor type/shape drift")
            if values[0].is_floating_point():
                total = torch.zeros_like(values[0])
                for value in values:
                    total.add_(value)
                swa_state[key] = total.div(len(values))
            elif not all(torch.equal(values[0], value) for value in values[1:]):
                raise RuntimeError("SWA nonfloating checkpoint buffer drift")
            else:
                swa_state[key] = values[0].detach().clone()
        state_digest = _tensor_digest(swa_state, torch)
        fresh = TFSRDecoder(capture_diagnostics=False).to("cuda:0")
        fresh.load_state_dict(swa_state, strict=True)
        if fresh.capture_diagnostics is not False:
            raise RuntimeError("SWA fresh-model capture diagnostics drift")
        fixed = runtime.get("fixed_eval")
        if fixed is None:
            raise RuntimeError("SWA has no fixed verified source capability")
        x, calib, capability = fixed
        # The evaluation proof exercises the same accelerated build the run
        # trained with; the proof discloses that build explicitly.
        scripted = accelerated_forward.build_scripted_step_binding(torch, fresh)
        fresh.eval()
        state_before = _tensor_digest(fresh.state_dict(), torch)
        with torch.no_grad():
            first = accelerated_forward.accelerated_forward(torch, fresh, scripted, x, calib, capability)
            first_p = fresh.last_dropout_p
            first_gain, first_survivor = fresh.last_unit_gain_mask, fresh.last_unit_survivor_mask
            second = accelerated_forward.accelerated_forward(torch, fresh, scripted, x, calib, capability)
            second_p = fresh.last_dropout_p
            second_gain, second_survivor = fresh.last_unit_gain_mask, fresh.last_unit_survivor_mask
        state_after = _tensor_digest(fresh.state_dict(), torch)
        if (not torch.is_tensor(first) or first.shape != (x.shape[0], 50, 2) or not torch.isfinite(first).all().item()
                or not torch.equal(first, second) or first_p is not None or second_p is not None
                or first_gain is None or first_survivor is None or second_gain is None or second_survivor is None
                or not torch.equal(first_gain, torch.ones_like(first_gain)) or not bool(first_survivor.all().item())
                or not torch.equal(second_gain, torch.ones_like(second_gain)) or not bool(second_survivor.all().item())
                or state_before != state_after):
            raise RuntimeError("SWA fixed-source eval/no-mask/state proof drift")
        proof = {
            "checkpoint_epochs": list(spec.checkpoint_epochs), "checkpoint_model_state_digests": checkpoint_digests,
            "fresh_strict_load": True, "eval_mode": True, "capture_diagnostics": False,
            "repeat_bitwise_equal": True, "state_unchanged": True, "eval_no_mask": True,
            "forward_build": contract_43.BUILD_DISCLOSURE["build"],
            "prediction_shape": list(first.shape),
            "prediction_sha256": _sha(first.detach().cpu().contiguous().numpy().tobytes()),
            "state_digest_before_eval": state_before, "state_digest_after_eval": state_after,
            "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                           "score": False},
        }
        _validate_swa_evaluation_proof(proof, spec)
        value = {"schema": "tfsr_b3st4_ddrop_seed43_swa_v1", "state": swa_state, "state_digest": state_digest,
                 "evaluation_proof": proof, "binding": _safe_mapping_copy(expected_binding)}
        stream = io.BytesIO()
        torch.save(value, stream)
        return SWAPayload(stream.getvalue(), state_digest, proof)

    def validate_swa(self, body: bytes, spec: RunSpec,
                     *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        import io

        import torch

        value = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        if (not isinstance(value, Mapping) or set(value) != {"schema", "state", "state_digest", "evaluation_proof", "binding"}
                or value.get("schema") != "tfsr_b3st4_ddrop_seed43_swa_v1"):
            raise RuntimeError("SWA schema drift")
        if (not isinstance(value.get("state"), Mapping)
                or _tensor_digest(value["state"], torch) != value.get("state_digest")
                or not _is_sha(value.get("state_digest"))):
            raise RuntimeError("SWA state digest drift")
        _validate_swa_evaluation_proof(value.get("evaluation_proof"), spec)
        binding = value.get("binding")
        expected_keys = {"cell", "run_spec", "launch_sha256", "launch_closure", "lineage"}
        if (not isinstance(binding, Mapping) or set(binding) != expected_keys or binding.get("cell") != CELL
                or binding.get("run_spec") != spec.payload() or not _is_sha(binding.get("launch_sha256"))
                or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("SWA binding drift")
        contract_43.validate_lineage(binding.get("lineage"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("SWA exact launch binding drift")
        return value

    def close(self, runtime: Any | None) -> None:
        return None


class DeterministicMockBackend:
    """No-data/no-CUDA mock used only by tests to prove the seed-43 lifecycle."""

    def __init__(self, *, failure: str | None = None):
        if failure not in {None, "before_gpu", "during_adapter", "after_step"}:
            raise ValueError("unknown deterministic mock failure")
        self.failure = failure
        self.closed = False
        self.proof_requests: list[bool] = []
        self.expensive_proof_count = 0

    @staticmethod
    def _digest(label: str) -> str:
        return _sha(label.encode())

    def prepare(self, spec: RunSpec, identity: RunIdentity, flags: LifecycleFlags) -> dict[str, Any]:
        if self.failure == "before_gpu":
            raise RuntimeError("synthetic failure before GPU initialization")
        flags.gpu_initialized = True
        if self.failure == "during_adapter":
            flags.source_opened = True
            raise RuntimeError("synthetic failure during source adapter")
        flags.source_opened = True
        return {"epoch": -1, "step": 0}

    def begin_epoch(self, runtime: dict[str, Any], epoch: int) -> None:
        runtime["epoch"] = epoch

    def train_step(self, runtime: dict[str, Any], lr: float, global_step: int, *,
                   require_epoch_proof: bool) -> StepOutcome:
        runtime["step"] += 1
        self.proof_requests.append(require_epoch_proof)
        even = (global_step % 2) == 0
        if require_epoch_proof:
            self.expensive_proof_count += 1
            gradients: Mapping[str, bool] | None = {key: True for key in _CRITICAL_GRADIENT_GROUPS}
            finite_model: bool | None = True
            finite_optimizer: bool | None = True
            model_state_digest: str | None = self._digest(f"model:{global_step}")
            optimizer_state_digest: str | None = self._digest(f"optimizer:{global_step}")
        else:
            gradients = None
            finite_model = None
            finite_optimizer = None
            model_state_digest = None
            optimizer_state_digest = None
        return StepOutcome(
            loss=1.0 + global_step / 10.0, lr_observed=lr, dropout_p=0.25 if even else 0.75,
            kept=6 if even else 4, dropped=2 if even else 4, all_zero_examples=0,
            population_examples=2, max_gain=1.0 / (1.0 - (0.25 if even else 0.75)),
            epoch_boundary_proof=require_epoch_proof, critical_gradients=gradients,
            finite_model=finite_model, finite_optimizer=finite_optimizer,
            model_state_digest=model_state_digest, optimizer_state_digest=optimizer_state_digest,
        )

    def after_optimizer_step(self, runtime: Any, global_step: int, flags: LifecycleFlags) -> None:
        if self.failure == "after_step" and global_step == 1:
            raise RuntimeError("synthetic failure after optimizer step")

    def resources(self, runtime: Any) -> Mapping[str, int]:
        return {"rss_bytes": 1, "peak_allocated_bytes": 2, "peak_reserved_bytes": 3}

    def make_checkpoint(self, runtime: Any, epoch: int, global_step: int, binding: Mapping[str, Any]) -> CheckpointPayload:
        state_digest = self._digest(f"state:{epoch}:{global_step}")
        value = {"schema": "tfsr_b3st4_ddrop_seed43_mock_checkpoint_v1", "epoch": epoch,
                 "global_step": global_step, "state_digest": state_digest, "binding": _safe_mapping_copy(binding)}
        return CheckpointPayload(_json(value), state_digest)

    def validate_checkpoint(self, body: bytes, epoch: int, global_step: int, spec: RunSpec, *,
                            expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        value = json.loads(body)
        if (not isinstance(value, Mapping) or set(value) != {"schema", "epoch", "global_step", "state_digest", "binding"}
                or value.get("schema") != "tfsr_b3st4_ddrop_seed43_mock_checkpoint_v1"
                or value.get("epoch") != epoch or value.get("global_step") != global_step
                or not _is_sha(value.get("state_digest"))):
            raise RuntimeError("mock checkpoint validation drift")
        binding = value.get("binding")
        if (not isinstance(binding, Mapping)
                or set(binding) != {"cell", "run_spec", "launch_sha256", "launch_closure", "lineage"}
                or binding.get("cell") != CELL or binding.get("run_spec") != spec.payload()
                or not _is_sha(binding.get("launch_sha256"))
                or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("mock checkpoint binding drift")
        contract_43.validate_lineage(binding.get("lineage"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("mock checkpoint exact launch binding drift")
        return value

    def build_swa(self, runtime: Any, checkpoints: Mapping[int, bytes], spec: RunSpec, *,
                  expected_binding: Mapping[str, Any]) -> SWAPayload:
        if set(checkpoints) != set(spec.checkpoint_epochs):
            raise RuntimeError("mock SWA checkpoint set drift")
        digests = {str(epoch): self.validate_checkpoint(body, epoch, (epoch + 1) * spec.steps_per_epoch, spec,
                                                        expected_binding=expected_binding)["state_digest"]
                   for epoch, body in checkpoints.items()}
        state_digest = self._digest("mock-swa:" + ",".join(digests.values()))
        proof = {"checkpoint_epochs": list(spec.checkpoint_epochs), "checkpoint_model_state_digests": digests,
                 "fresh_strict_load": True, "eval_mode": True, "capture_diagnostics": False,
                 "repeat_bitwise_equal": True, "state_unchanged": True, "eval_no_mask": True,
                 "forward_build": contract_43.BUILD_DISCLOSURE["build"],
                 "prediction_shape": [2, 50, 2], "prediction_sha256": self._digest("mock-prediction"),
                 "state_digest_before_eval": state_digest, "state_digest_after_eval": state_digest,
                 "boundaries": {"source_only": True, "target_or_formal_opened": False, "scientific_result": False,
                                "score": False}}
        _validate_swa_evaluation_proof(proof, spec)
        value = {"schema": "tfsr_b3st4_ddrop_seed43_mock_swa_v1", "state_digest": state_digest,
                 "evaluation_proof": proof, "binding": _safe_mapping_copy(expected_binding)}
        return SWAPayload(_json(value), state_digest, proof)

    def validate_swa(self, body: bytes, spec: RunSpec,
                     *, expected_binding: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        value = json.loads(body)
        if (not isinstance(value, Mapping) or set(value) != {"schema", "state_digest", "evaluation_proof", "binding"}
                or value.get("schema") != "tfsr_b3st4_ddrop_seed43_mock_swa_v1"
                or not _is_sha(value.get("state_digest"))):
            raise RuntimeError("mock SWA schema drift")
        proof = value.get("evaluation_proof")
        if (not isinstance(proof, Mapping) or proof.get("checkpoint_epochs") != list(spec.checkpoint_epochs)
                or proof.get("fresh_strict_load") is not True or proof.get("eval_no_mask") is not True):
            raise RuntimeError("mock SWA proof drift")
        binding = value.get("binding")
        expected_keys = {"cell", "run_spec", "launch_sha256", "launch_closure", "lineage"}
        if (not isinstance(binding, Mapping) or set(binding) != expected_keys or binding.get("cell") != CELL
                or binding.get("run_spec") != spec.payload() or not _is_sha(binding.get("launch_sha256"))
                or not isinstance(binding.get("launch_closure"), Mapping)):
            raise RuntimeError("mock SWA binding drift")
        contract_43.validate_lineage(binding.get("lineage"))
        if expected_binding is not None and _safe_mapping_copy(binding) != _safe_mapping_copy(expected_binding):
            raise RuntimeError("mock SWA exact launch binding drift")
        return value

    def close(self, runtime: Any | None) -> None:
        self.closed = True


def mock_identity() -> RunIdentity:
    """Explicit synthetic identity for isolated lifecycle tests only."""
    return RunIdentity(
        phase_c_acceptance={"body_sha256": "a" * 64, "path": "synthetic_phase_c.json"},
        source_authorities={"manifest_sha256": "b" * 64, "roster": ["synthetic-source"]},
        closures={"stage0": {"closure_sha256": "c" * 64}, "phase_c": {"closure_sha256": "d" * 64},
                  "frozen_phase_d": {"closure_sha256": "e" * 64}, "seed43": {"closure_sha256": "1" * 64}},
        device=dict(FROZEN_DEVICE_43),
        lineage={
            "supersedes_cell": contract_43.SUPERSEDED_CELL,
            "supersedes_contract_relative": contract_43.SUPERSEDED_CONTRACT_RELATIVE,
            "supersedes_note": contract_43.SUPERSEDES_NOTE,
            "throughput_v2_receipt_relative": contract_43.THROUGHPUT_V2_RECEIPT_RELATIVE,
            "throughput_v2_receipt_sha256": contract_43.THROUGHPUT_V2_RECEIPT_SHA256,
            "frozen_route_closure_sha256": "2" * 64,
            "seed42_run_resumed": False,
        },
        build_disclosure=contract_43.validate_build_disclosure(contract_43.BUILD_DISCLOSURE),
    )


def execute_training(root: Path) -> None:
    """Authorized-only public lifecycle.  There are no CLI budget overrides."""
    output_gate(root)
    # Verify every immutable input before reserving an output root, so an
    # identity failure cannot leave a partial seed-43 alias behind.
    production_identity(root)
    artifact = reserve_output_root(root)
    run_lifecycle(spec=PUBLIC_SPEC, backend=TrainingBackend43(root), artifact=artifact,
                  identity_factory=lambda: production_identity(root))
