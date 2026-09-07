"""Deferred selected-compatible-device physical backend for PMC-D.

The public CLI never imports this module.  The backend composes the reviewed
strict-27 Phase-B-v2 source adapter and the sealed equal-session Cell-D
``begin_epoch``/``train_step``/``end_epoch`` methods.  It owns no alternate
model forward, loss, scheduler, sampler, dropout law, or normalizer.

No source file, checkpoint tensor or CUDA device is accessed at import time.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src import cell_d_equal_session_v1 as equal_session
from src.posterior_carrier_v1 import phase_b

from . import plan
from .core import PosteriorMarginalizedSideCache
from .lifecycle import PosteriorMarginalizedEqualSessionAdapter
from .runner import PMCExecutionSpec, PMCIdentity, PMCProgress, PMCRunnerError
from .source_audit import (
    PMCLiveSourceAdapter,
    build_strict27_pmc_source,
    load_approved_phase_b_v2_authority_from_full_import,
)


class PMCPhysicalError(PMCRunnerError):
    """Fail closed for the future PMC selected-device physical backend."""


# Compatibility aliases are intentionally derived from the route-local static
# plan rather than duplicated authority literals.  ``GPU1_AUTHORITY`` remains
# an import-compatible alias for Phase-2 tests; production receives an
# explicit profile from the root reviewer and can choose either reviewed idle
# device without waiting on a fixed ordinal.
GPU0_AUTHORITY = dict(plan.COMPATIBLE_DEVICE_PROFILES["gpu0"])
GPU1_AUTHORITY = dict(plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PMCPhysicalError(message)


def validate_future_compatible_device_environment(
    profile: Mapping[str, object], environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Static future-launch guard for a root-selected compatible profile."""
    selected = plan.validate_compatible_device_profile(profile)
    values = os.environ if environ is None else environ
    _require(values.get("CUDA_VISIBLE_DEVICES") == selected["cuda_visible_devices"],
             "PMC future selected-device CUDA_VISIBLE_DEVICES drift")
    _require(values.get("CUDA_DEVICE_ORDER") == selected["cuda_device_order"],
             "PMC future selected-device CUDA_DEVICE_ORDER drift")
    return {
        "CUDA_VISIBLE_DEVICES": selected["cuda_visible_devices"],
        "CUDA_DEVICE_ORDER": selected["cuda_device_order"], "logical_device": selected["logical_device"],
        "gpu_authority": selected,
    }


def validate_future_gpu1_environment(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Backward-compatible static GPU1 convenience wrapper for tests only."""
    return validate_future_compatible_device_environment(GPU1_AUTHORITY, environ)


def _attest_exact_compatible_device(
    torch: Any,
    profile: Mapping[str, object],
    *,
    on_cuda_initialized: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Execution-only dual NVML/Torch attestation for the selected profile."""
    selected = plan.validate_compatible_device_profile(profile)
    validate_future_compatible_device_environment(selected)
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "PMC selected-device route requires exactly one visible CUDA device")
    props = torch.cuda.get_device_properties(0)
    # ``get_device_properties`` is the first operation that unquestionably
    # initializes the CUDA runtime.  Mark it before the independent
    # nvidia-smi half of attestation so a later failure is never reported as
    # a pre-CUDA failure.
    if on_cuda_initialized is not None:
        on_cuda_initialized()
    runtime = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version()),
        "name": str(props.name),
        "torch_total_memory_bytes": int(props.total_memory),
    }
    for key in ("torch_version", "torch_cuda_version", "cudnn_version", "name", "torch_total_memory_bytes"):
        _require(runtime[key] == selected[key], f"PMC selected-device Torch authority {key} drift")
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-i", str(selected["cuda_visible_devices"]),
             "--query-gpu=uuid,pci.bus_id,name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PMCPhysicalError("PMC selected-device nvidia-smi attestation failed") from error
    _require(len(output) == 1, "PMC selected-device nvidia-smi row count drift")
    uuid, bdf, name, memory = (item.strip() for item in output[0].split(",", 3))
    _require({"uuid": uuid, "bdf": bdf, "name": name, "nvidia_smi_memory_total_mib": int(memory)}
             == {key: selected[key] for key in ("uuid", "bdf", "name", "nvidia_smi_memory_total_mib")},
             "PMC selected-device nvidia-smi authority drift")
    return {**selected, "visible_devices": 1, "attested": True}


@dataclass
class _PMCRuntime:
    torch: Any
    physical_source: PMCLiveSourceAdapter
    base: Any
    adapter: PosteriorMarginalizedEqualSessionAdapter
    inherited_runtime: dict[str, Any]
    device: Any
    runtime_environment: Mapping[str, object]
    current_epoch: int | None = None
    outcomes: list[Any] | None = None


class PhysicalPMCBackend:
    """Composition-only deferred backend; no shared route is modified.

    ``prepare`` descriptor-loads the exact immutable imported Phase-B-v2
    authority pair before source resolution; callers cannot hand-assemble a
    mutable authority mapping.  Calling it is an execution action and is
    intentionally impossible through the public CLI or ordinary flags.
    """

    def __init__(
        self,
        root: Path,
        *,
        source_data: phase_b.SourceDataRootCapability,
        device_profile: Mapping[str, object],
        num_workers: int = 4,
    ) -> None:
        self.root = Path(root).absolute()
        self.source_data = source_data
        self.device_profile = plan.validate_compatible_device_profile(device_profile)
        self.num_workers = num_workers
        self._runtime: _PMCRuntime | None = None
        self._closed = False

    def _require_runtime(self) -> _PMCRuntime:
        if self._runtime is None or self._closed:
            raise PMCPhysicalError("PMC physical runtime is unavailable")
        return self._runtime

    def prepare(self, *, spec: PMCExecutionSpec, identity: PMCIdentity, progress: PMCProgress) -> _PMCRuntime:
        if spec.kind not in {"source_smoke", "full_train"} or self.num_workers != 4:
            raise PMCPhysicalError("PMC physical lifecycle/worker contract drift")
        if plan.validate_compatible_device_profile(identity.gpu) != self.device_profile:
            raise PMCPhysicalError("PMC identity selected-device authority drift")
        # The v2 source authority is fully validated before *any* source path
        # resolution.  It includes same-prefix recovery, every prefix digest,
        # fallback topology and its own closure cross-binding.
        approved_phase_b_v2 = load_approved_phase_b_v2_authority_from_full_import(self.root)
        approved_phase_b_v2.validate()
        _require(
            approved_phase_b_v2.imported_full_source_authority_sha256
            == plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256,
            "PMC imported Phase-B-v2 source-authority root binding drift",
        )
        # Canonical initial state validation is CPU-only and occurs before the
        # source pass, matching the checked equal-session access-order repair.
        import numpy as np
        import torch
        from torch.nn.parameter import UninitializedParameter
        from torch.utils.data import default_collate
        import lightning.pytorch as pl

        canonical = equal_session._load_canonical_initial_state(self.root, torch)
        progress.stage = "source_adapter"
        physical_source = build_strict27_pmc_source(
            self.root, source_data=self.source_data, approved=approved_phase_b_v2,
            num_workers=self.num_workers,
            # The reviewed train-only adapter invokes this immediately before
            # its shared setup opens the strict source records.  Do not claim
            # source access merely because we are about to construct it.
            on_source_opened=lambda: setattr(progress, "source_opened", True),
        )
        # Bind the actual same-FD strict-27 source material before model/CUDA
        # preparation.  A root capability may not label a prospective or
        # syntactically shaped source binding as this run's actual inputs.
        live_binding = physical_source.bound_authority
        _require(live_binding.sha256() == identity.source_binding_sha256,
                 "PMC live strict-source binding differs from launch identity")
        _require(
            live_binding.phase_b_v2_authority_sha256 == identity.phase_b_v2_authority_sha256
            and live_binding.phase_b_v2_closure_sha256 == identity.phase_b_v2_closure_sha256,
            "PMC live strict-source Phase-B-v2 authority/closure differs from launch identity",
        )
        dataset = physical_source.dataset
        inventory = equal_session._source_inventory_from_dataset(dataset, physical_source.roster)
        _require(inventory.eligible_windows == equal_session.SOURCE_ONLY_EXPECTED_ELIGIBLE_WINDOWS,
                 "PMC strict-27 live inventory window count drift")
        _require(inventory.full_batches(equal_session.PUBLIC_SPEC) == equal_session.PUBLIC_SPEC.steps_per_epoch,
                 "PMC strict-27 full-B32 batch count drift")
        schedule_plan = equal_session.build_plan_evidence(
            spec=equal_session.PUBLIC_SPEC, inventory=inventory,
            snapshot_rng=lambda: equal_session._torch_rng_snapshot(torch, np),
        )
        _require(schedule_plan["full_plan_sha256"] == equal_session.AUDITED_FULL_PLAN_SHA256,
                 "PMC equal-session schedule plan drift")

        # This is the same model-seeding order as Cell-D.  The schedule and
        # posterior bank use only local RNG before the dropout/model stream.
        pl.seed_everything(plan.SEED, workers=True)
        arm_common = equal_session._load_runtime_module(
            "_pmc_arm_common", self.root / "tfpd_exploration/src/tfpd_lane/arm_common.py",
        )
        pop_robust = equal_session._load_runtime_module(
            "_pmc_pop_robust", self.root / "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        )
        model = pop_robust.build_population_robustness_model(seed=plan.SEED, cell="D")
        model.load_state_dict(canonical, strict=True)
        _require(arm_common.state_sha256(model) == plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
                 "PMC canonical Cell-D strict-load state drift")
        live_parameters = sum(int(parameter.numel()) for parameter in model.parameters()
                              if not isinstance(parameter, UninitializedParameter))
        lazy = tuple(sorted(name for name, parameter in model.named_parameters()
                            if isinstance(parameter, UninitializedParameter)))
        _require(live_parameters == plan.SEALED_CELL_D_INITIALIZED_PARAMETERS and lazy == plan.SEALED_CELL_D_LAZY_KEYS,
                 "PMC Cell-D parameter/lazy topology drift")

        progress.stage = "gpu_attestation"
        runtime_environment = _attest_exact_compatible_device(
            torch, self.device_profile,
            on_cuda_initialized=lambda: setattr(progress, "cuda_initialized", True),
        )
        # Explicitly forbid mixed/TF32 math before model batches/optimizer.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _require(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False,
                 "PMC forbids TF32")
        runtime_environment = {
            **runtime_environment,
            "torch_cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch_cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
        device = torch.device("cuda:0")
        model.to(device)
        cache = PosteriorMarginalizedSideCache(
            roster=physical_source.roster,
            prefix_inputs_by_session=physical_source.prefix_inputs(),
            prior=physical_source.prior,
            sealed_ols_normalizer=physical_source.ordinary_ols_normalizer,
            seed=plan.SEED,
        )
        # Build the exact fixed diagnostic batch before the single epoch
        # iterator, just as the inherited equal-session source path does.
        first = next(equal_session.EpochSchedule(
            spec=equal_session.PUBLIC_SPEC, inventory=inventory, epoch=0,
        ).iter_batches())
        fixed = default_collate([dataset[index] for index in first.dataset_indices])
        fixed = tuple(item.to(device) if torch.is_tensor(item) else item for item in fixed[:5])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, amsgrad=False,
        )
        torch.cuda.reset_peak_memory_stats(0)
        inherited_spec = equal_session.SOURCE_SMOKE_SPEC if spec.kind == "source_smoke" else equal_session.FULL_TRAIN_SPEC
        runtime: dict[str, Any] = {
            "torch": torch, "numpy": np, "pl": pl, "arm_common": arm_common, "pop_robust": pop_robust,
            "model": model, "optimizer": optimizer, "dataset": dataset, "inventory": inventory,
            "plan": schedule_plan, "ledger": equal_session.EpochLedger(schedule_plan), "device": device,
            "runtime_environment": runtime_environment, "peak_stats_reset_before_training": True,
            "fixed": fixed, "fixed_batch_schedule": first.payload(), "fixed_diagnostic_input_rng_unchanged": True,
            # The inherited Cell-D diagnostic is an auxiliary, forward-only
            # held-control diagnostic.  Its source-side carrier remains the
            # ordinary OLS point carrier from the fixed Cell-D batch; it is
            # not a second PMC sampled-side forward and is disclosed in every
            # epoch receipt below.
            "fixed_diagnostic_side_semantics": "ordinary_ols_point_side__held_auxiliary_diagnostic_not_pmc_training_sample",
            "behavior_normalizer_semantic_sha256": physical_source.behavior_normalizer_semantic_sha256,
            "t4_normalizer_semantic_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
            "t4_authority_fingerprint": arm_common.t4_authority_fingerprint(dataset.sessions),
            "epoch_iterator": None, "epoch_sampler": None, "epoch_schedule": None,
            "lifecycle_spec": inherited_spec, "pmc_execution_spec": spec,
        }
        base = equal_session.PhysicalEqualSessionBackend(self.root, num_workers=self.num_workers)
        base.lr_for_step = lambda step: float(arm_common.lr_at_step(step, plan.EPOCHS, plan.STEPS_PER_EPOCH))
        adapter = PosteriorMarginalizedEqualSessionAdapter(base=base, cache=cache)
        self._runtime = _PMCRuntime(
            torch=torch, physical_source=physical_source, base=base, adapter=adapter, inherited_runtime=runtime,
            device=device, runtime_environment=runtime_environment,
        )
        return self._runtime

    def source_authority(self, runtime: _PMCRuntime, *, identity: PMCIdentity) -> Mapping[str, object]:
        state = self._require_runtime()
        _require(runtime is state, "PMC source authority runtime identity drift")
        source_binding = state.physical_source.bound_authority.payload()
        return {
            "schema": "posterior_marginalized_cell_d_source_authority_v1", "cell": plan.CELL,
            "source_only": True, "target_opened": False, "within_opened": False,
            "external_opened": False, "formal_opened": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "source_binding": source_binding,
            "cache_policy": {"fits_and_inverses_before_iterator": True, "samples_before_iterator": True,
                             "device_cache_before_iterator": True, "batch_loop_inverse_calls": 0,
                             "batch_loop_sampling_calls": 0, "batch_loop_normalizer_fit_calls": 0},
            "normalizer": source_binding["normalizer"], "gpu": dict(state.runtime_environment),
            "identity": identity.payload(),
        }

    def begin_epoch(self, runtime: _PMCRuntime, *, epoch: int, progress: PMCProgress) -> Mapping[str, object]:
        state = self._require_runtime()
        _require(runtime is state and state.current_epoch is None, "PMC physical epoch ordering drift")
        flags = equal_session.LifecycleFlags(source_train_opened=True, cuda_initialized=True)
        schedule = state.adapter.begin_epoch(state.inherited_runtime, epoch, flags)
        state.current_epoch, state.outcomes = epoch, []
        return {"epoch": epoch, "one_iterator": True, "schedule": dict(schedule),
                "pmc_cache": state.adapter.cache_evidence()}

    def train_step(self, runtime: _PMCRuntime, *, epoch: int, global_step: int, require_full_proof: bool,
                   progress: PMCProgress) -> Mapping[str, object]:
        state = self._require_runtime()
        _require(runtime is state and state.current_epoch == epoch and state.outcomes is not None,
                 "PMC physical step outside bound epoch")
        flags = equal_session.LifecycleFlags(source_train_opened=True, cuda_initialized=True)
        try:
            outcome = state.adapter.train_step(
                state.inherited_runtime, global_step=global_step,
                expected_lr=state.base.lr_for_step(global_step), require_full_proof=require_full_proof, flags=flags,
            )
            state.outcomes.append(outcome)
        except BaseException:
            # The generic runner increments progress only after a successful
            # return.  The inherited step can fail after backward or even
            # after ``optimizer.step`` (for example while checking a receipt
            # invariant), so preserve the truthful partial physical progress
            # in its immutable failure terminal.
            progress.backward_calls += flags.backward_calls
            progress.update_calls += flags.update_calls
            progress.optimizer_steps_completed += flags.update_calls
            raise
        return {
            "loss": outcome.loss, "lr": outcome.lr, "dropout_p": outcome.dropout_p,
            "full_proof": require_full_proof, "batch": dict(outcome.batch_evidence),
        }

    def end_epoch(self, runtime: _PMCRuntime, *, epoch: int, rows: Sequence[Mapping[str, object]],
                  progress: PMCProgress) -> Mapping[str, object]:
        state = self._require_runtime()
        _require(runtime is state and state.current_epoch == epoch and state.outcomes is not None,
                 "PMC physical end epoch ordering drift")
        flags = equal_session.LifecycleFlags(source_train_opened=True, cuda_initialized=True)
        detail = state.base.end_epoch(state.inherited_runtime, epoch, state.outcomes, flags)
        outcomes = state.outcomes
        last = outcomes[-1]
        resources = state.base.resources(state.inherited_runtime)
        loss = [item.loss for item in outcomes]
        probability = [item.dropout_p for item in outcomes]
        cache = state.adapter.cache_evidence()
        state.current_epoch, state.outcomes = None, None
        return {
            "epoch": epoch, "steps": len(outcomes),
            "cumulative_optimizer_steps": (epoch + 1) * state.inherited_runtime["pmc_execution_spec"].steps_per_epoch,
            "loss": {"mean": sum(loss) / len(loss), "min": min(loss), "max": max(loss)},
            "lr": {"first": outcomes[0].lr, "last": outcomes[-1].lr},
            "dropout": {"p_min": min(probability), "p_max": max(probability),
                        "p_mean": sum(probability) / len(probability), "one_call_per_step": True},
            "cache": cache, "proof": {"finite_model": last.finite_model, "finite_optimizer": last.finite_optimizer,
                                           "critical_gradients": last.critical_gradients,
                                           "model_state_sha256": last.model_state_sha256,
                                           "optimizer_state_sha256": last.optimizer_state_sha256,
                                           "inherited_end_epoch": dict(detail)},
            "resources": dict(resources),
            "diagnostic_side": state.inherited_runtime["fixed_diagnostic_side_semantics"],
        }

    def checkpoint(self, runtime: _PMCRuntime, *, epoch: int, global_step: int, binding: Mapping[str, object]) -> bytes:
        state = self._require_runtime()
        return state.base.make_checkpoint(state.inherited_runtime, epoch, global_step, binding).body

    def swa(self, runtime: _PMCRuntime, *, checkpoint_bodies: Mapping[int, bytes], binding: Mapping[str, object]) -> tuple[bytes, Mapping[str, object]]:
        state = self._require_runtime()
        payload = state.base.build_swa(state.inherited_runtime, checkpoint_bodies, equal_session.FULL_TRAIN_SPEC, binding)
        state.base.validate_swa(payload.body, spec=equal_session.FULL_TRAIN_SPEC, binding=binding)
        return payload.body, {"state_sha256": payload.state_sha256, **dict(payload.proof)}

    def final_reverify(self, runtime: _PMCRuntime, *, identity: PMCIdentity) -> Mapping[str, object]:
        state = self._require_runtime()
        _require(runtime is state, "PMC final runtime identity drift")
        observer = state.adapter.cache_evidence()
        _require(observer["posterior_fit_calls"] == plan.STRICT_SOURCE_SESSION_COUNT * len(plan.BUDGETS)
                 and observer["posterior_inverse_calls"] == plan.STRICT_SOURCE_SESSION_COUNT * len(plan.BUDGETS)
                 and observer["batch_loop_inverse_calls"] == 0 and observer["batch_loop_sampling_calls"] == 0,
                 "PMC final cache/batch-loop audit drift")
        closure = plan.implementation_closure(self.root)
        _require(closure == identity.payload()["closure"], "PMC physical final closure drift")
        return closure

    def close(self, runtime: _PMCRuntime | None) -> None:
        if self._closed:
            return
        self._closed = True
        if runtime is not None:
            runtime.base.close(runtime.inherited_runtime)
        self._runtime = None
