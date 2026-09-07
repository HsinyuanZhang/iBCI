"""Armed matched T0/C1 M1 trainer — 50-epoch sibling with the warmup-then-cosine LR.

``ArmedM1Trainer50`` subclasses the sealed 20-epoch trainer and inherits
``prepare()`` unchanged (no LR/epoch coupling).  ``run`` is a faithful copy
of the parent loop with three disclosed deltas: per-step LR application,
five sealed epoch checkpoints, and the LR fields on the return dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import core as cswg_core
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.m1_t0c1_prefix_v1 import hook as hook_module
from tfpd_exploration.src.m1_t0c1_prefix_v1 import schedule
from tfpd_exploration.src.m1_t0c1_prefix_v1.trainer import (
    ArmedM1Trainer,
    epoch_mean_loss,
    _serialize_model,
)

from . import plan


class TrainerError(RuntimeError):
    """Fail closed for the 50-epoch armed matched trainer."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainerError(message)


class ArmedM1Trainer50(ArmedM1Trainer):
    """One arm of the 50-epoch pair; constructs nothing at import time."""

    def run(self, *, epoch_count: int, steps_per_epoch: int, record_steps: int) -> Mapping[str, object]:
        """Run the armed training loop; returns receipt bodies and checkpoint bytes.

        Deltas versus the sealed parent loop, and only these:

        1. ``lr_schedule.apply(optimizer, global_step)`` immediately before
           ``optimizer.step()``; per-step LR stream plus epoch-final LR.
        2. Checkpoint capture at every index in ``CHECKPOINT_EPOCH_INDICES``.
        3. Return dict is a superset of the parent (LR law, epoch checkpoint
           bodies/digests, per-epoch ``lr_at_epoch_final_step``).
        """
        import math
        import random
        import time

        import torch

        from . import lr_schedule

        _require(self._prepared is not None and type(epoch_count) is int and epoch_count > 0
                 and type(steps_per_epoch) is int and steps_per_epoch > 0,
                 "armed trainer run lifecycle drift")
        prepared = self._prepared
        spec = prepared.spec
        total_steps = epoch_count * steps_per_epoch
        started = time.monotonic()
        runtime = base_physical.SelectedCudaRuntime(self.device_profile)
        runtime.__enter__()
        rng_snapshot = base_physical.ProcessRngSnapshot.capture_and_seed(torch, seed=plan.SEED)
        operator = hook_module.M1CalPrefixOperator(self.arm, record_steps=record_steps)
        stream = hook_module.TrainingStreamRecorder()
        try:
            model = base_physical.load_exact_m1_spint_model(Path(self.code_root))
            model = model.to(self.device)
            materialization = base_physical.materialize_exact_m1_model(model, device=self.device)
            operator.attach(model)
            adapter = base_physical.ExactM1ForwardAdapter(model)
            step = cswg_core.RouteOwnedMixedSessionTrainingStep(
                model=adapter, compatibility=prepared.compatibility, run_spec=spec.stage0_spec,
            )
            initial_lr = lr_schedule.lr_at(0)
            _require(initial_lr == plan.ADAM_LR == plan.LR_WARMUP_START,
                     "Adam constructor LR drifted from lr_at(0)")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=plan.ADAM_LR, weight_decay=plan.ADAM_WEIGHT_DECAY,
            )
            model.train(True)
            _require(operator._handle is not None and model.training is True,
                     "armed trainer hook/training-mode drift")
            initial_digest = base_physical._state_digest(model)
            trainable_names = tuple(sorted(
                name for name, parameter in model.named_parameters() if parameter.requires_grad))
            best_loss = math.inf
            best_body: bytes | None = None
            best_digest: str | None = None
            best_epoch_index: int | None = None
            epoch_rows: list[dict[str, object]] = []
            observed_gradients: set[str] = set()
            final_loss = math.nan
            lr_applied: list[float] = []
            checkpoint_epoch_bodies: dict[int, bytes] = {}
            checkpoint_epoch_state_sha256: dict[int, str] = {}
            # The episode stream is a pure deterministic function of the
            # epoch-local step index (``build_balanced_episode`` documents
            # "without global RNG"; stable sha-based offsets; the same episode
            # sequence repeats every epoch).  Memoizing it across epochs
            # preserves the exact frozen episode sequence while removing the
            # O(pool-rows) per-step candidate scan, which alone costs ~0.31
            # s/step and would push one arm past the 12 h per-arm bound.
            episode_cache: dict[int, Any] = {}
            for global_step in range(total_steps):
                epoch_index, epoch_step = divmod(global_step, steps_per_epoch)
                episode = episode_cache.get(epoch_step)
                if episode is None:
                    episode = prepared.episode(epoch_step)
                    episode_cache[epoch_step] = episode
                state_before = random.getstate()
                objective = step.run(
                    episode, lambda_=plan.OBJECTIVE_LAMBDA, tau=plan.OBJECTIVE_TAU,
                )
                optimizer.zero_grad(set_to_none=True)
                objective.loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        _require(parameter.grad is not None
                                 and bool(torch.isfinite(parameter.grad).all()),
                                 f"armed trainer missing/nonfinite gradient: {name}")
                        observed_gradients.add(name)
                applied_lr = lr_schedule.apply(optimizer, global_step)
                lr_applied.append(applied_lr)
                optimizer.step()
                final_loss = float(objective.loss.detach().cpu())
                state_after = random.getstate()
                # EVERY step is recorded: the epoch-mean coverage invariant
                # requires exactly steps_per_epoch loss rows per epoch, and the
                # per-epoch receipts digest each epoch's full row block.  Only
                # the published stream_head.json is limited to record_steps.
                stream.record_step(
                    global_step, before_state=state_before, after_state=state_after,
                    sample_ids=[row.sample_id for row in episode.all_rows], loss=final_loss,
                )
                self._progress = {
                    "source_resolved_or_opened": True,
                    "model_constructed": True,
                    "cuda_initialized": bool(torch.cuda.is_initialized()),
                    "optimizer_steps_completed": global_step + 1,
                }
                if epoch_step == steps_per_epoch - 1:
                    epoch_loss = epoch_mean_loss(stream, global_step, steps_per_epoch)
                    epoch_rows_digest = schedule.sequence_digest(
                        [row["loss"] for row in stream.rows[global_step + 1 - steps_per_epoch:global_step + 1]])
                    elapsed = time.monotonic() - started
                    _require(elapsed <= plan.HARD_TIMEOUT_SECONDS_PER_ARM,
                             f"armed trainer exceeded the {plan.HARD_TIMEOUT_SECONDS_PER_ARM}s per-arm hard timeout")
                    model_digest = base_physical._state_digest(model)
                    epoch_row: dict[str, object] = {
                        "epoch_index": epoch_index,
                        "steps_in_epoch": steps_per_epoch,
                        "global_optimizer_steps_completed": global_step + 1,
                        "epoch_mean_source_train_loss": epoch_loss,
                        "epoch_loss_rows_sha256": epoch_rows_digest,
                        "source_train_loss_aggregation": "arithmetic_mean_of_complete_source_objective_per_step",
                        "model_state_sha256": model_digest,
                        "finite_objective": bool(math.isfinite(final_loss)),
                        "operator_snapshot": operator.snapshot(),
                        "elapsed_seconds": elapsed,
                        "target_optimizer_backward_update": 0,
                        "source_only": True,
                        "lr_at_epoch_final_step": applied_lr,
                    }
                    if epoch_index == 1:
                        # elapsed_seconds on each row is cumulative from start.
                        _require(epoch_rows and int(epoch_rows[0]["epoch_index"]) == 0,
                                 "epoch 1 abort heuristic is missing the epoch-0 receipt")
                        epoch_0_seconds = float(epoch_rows[0]["elapsed_seconds"])
                        epoch_1_seconds = float(elapsed) - epoch_0_seconds
                        estimate = plan.runtime_abort_estimate_seconds(
                            epoch_0_seconds, epoch_1_seconds)
                        epoch_row["epoch_0_seconds"] = epoch_0_seconds
                        epoch_row["epoch_1_seconds"] = epoch_1_seconds
                        epoch_row["runtime_abort_estimate_seconds"] = estimate
                        try:
                            plan.assert_runtime_abort_heuristic(epoch_0_seconds, epoch_1_seconds)
                        except plan.M1T0C150EpPlanError as error:
                            raise TrainerError(str(error)) from error
                    epoch_rows.append(epoch_row)
                    if epoch_index in plan.CHECKPOINT_EPOCH_INDICES:
                        checkpoint_epoch_bodies[epoch_index] = _serialize_model(model)
                        checkpoint_epoch_state_sha256[epoch_index] = model_digest
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_epoch_index = epoch_index
                        best_body = _serialize_model(model)
                        best_digest = model_digest
            expected_epoch_checkpoints = {
                index for index in plan.CHECKPOINT_EPOCH_INDICES if index < epoch_count
            }
            _require(set(checkpoint_epoch_bodies) == expected_epoch_checkpoints
                     and set(checkpoint_epoch_state_sha256) == expected_epoch_checkpoints,
                     "armed trainer epoch-checkpoint coverage drift")
            _require(len(lr_applied) == total_steps, "armed trainer LR stream coverage drift")
            model.eval()
            eval_proof = hook_module.eval_dropout_inactive_proof(model, device=self.device)
            model.train(False)
            final_digest = base_physical._state_digest(model)
            operator.detach()
            last_body = _serialize_model(model)
            best_body = best_body if best_body is not None else last_body
            best_digest = best_digest if best_digest is not None else final_digest
            law = lr_schedule.schedule_law()
            return {
                "arm": self.arm,
                "initial_model_state_sha256": initial_digest,
                "final_model_state_sha256": final_digest,
                "best_checkpoint_state_sha256": best_digest,
                "best_source_train_loss": best_loss if best_loss is not math.inf else None,
                "best_source_train_loss_epoch_index": best_epoch_index,
                "epochs": epoch_rows,
                "trainable_parameter_count": len(trainable_names),
                "observed_gradient_count": len(observed_gradients),
                "model_parameter_count": materialization["live_parameter_count"],
                "operator_snapshot": operator.snapshot(),
                "stream_records": stream.payload(record_steps),
                "eval_dropout_inactive_proof": eval_proof,
                "optimizer_steps": total_steps,
                "elapsed_seconds": time.monotonic() - started,
                "dynamic_dropout_preserved": bool(getattr(model, "dynamic_dropout", False)),
                "runtime_environment": dict(runtime.attestation),
                "source_only": True,
                "swa_enabled": False,
                "lr_per_step": list(lr_applied),
                "lr_stream_digest": lr_schedule.sequence_digest(lr_applied),
                "lr_schedule": law,
                "checkpoint_epoch_state_sha256": dict(checkpoint_epoch_state_sha256),
                "_checkpoint_bodies": {"best": best_body, "last": last_body},
                "_checkpoint_epoch_bodies": dict(checkpoint_epoch_bodies),
            }
        finally:
            operator.detach()
            rng_snapshot.restore()
            runtime.close()


def live_device_profile(torch_module: Any) -> v1.DeviceProfile:
    """Build the single-visible-device profile by querying physical GPU 1 only.

    The sealed parent profiler uses ``nvidia-smi --id 0``, which is physical
    GPU 0 regardless of ``CUDA_VISIBLE_DEVICES``.  Spec §0 / §2.10 forbid that
    query.  This replacement queries ``--id ${CUDA_VISIBLE_DEVICES}`` after
    asserting CVD == "1", which is therefore physical ``--id 1``.  NVML
    ordinals ignore CVD, so passing 0 here would still be GPU 0.
    """
    import os as _os
    import subprocess as _subprocess

    identity = plan.DEVICE_IDENTITY_LAW
    expected_device = plan.LAUNCH_ENVELOPE["expected_device"]
    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(visible == plan.BOUND_GPU["cuda_visible_devices"] == plan.LAUNCH_ENVELOPE["CUDA_VISIBLE_DEVICES"] == "1",
             f"armed trainer requires CUDA_VISIBLE_DEVICES={plan.BOUND_GPU['cuda_visible_devices']}")
    _require(torch_module.cuda.is_available() and torch_module.cuda.device_count() == 1,
             "armed trainer requires exactly one visible CUDA device")
    props = torch_module.cuda.get_device_properties(0)
    # Coupled to the CVD=="1" assert above: --id ${CUDA_VISIBLE_DEVICES} == --id 1.
    physical_nvml_id = visible
    completed = _subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name", "--format=csv,noheader",
         "--id", physical_nvml_id],
        check=True, text=True, capture_output=True, timeout=15,
    )
    _require(completed.stdout.strip(), "nvidia-smi -i 1 returned empty")
    uuid, pci, name = (part.strip() for part in completed.stdout.splitlines()[0].split(","))
    _require(uuid != identity["refused_uuid_gpu0"],
             "armed trainer nvidia-smi -i 1 returned the forbidden GPU-0 uuid")
    _require(uuid == identity["required_uuid"] and pci == identity["required_pci_bus_id"]
             and name == expected_device["name"],
             "armed trainer device is not the bound RTX 3090 GPU 1")
    _require(props.name == expected_device["name"]
             and int(props.total_memory) == int(expected_device["total_memory_bytes"]),
             "armed trainer visible torch device is not the bound RTX 3090")
    capability = tuple(int(item) for item in torch_module.cuda.get_device_capability(0))
    expected_capability = tuple(int(item) for item in expected_device["capability"])
    _require(capability == expected_capability,
             f"armed trainer compute capability drifted: {capability} != {expected_capability}")
    profile = v1.DeviceProfile(
        cuda_visible_devices=visible, torch_device=plan.BOUND_GPU["torch_device"],
        uuid=uuid, pci_bus_id=pci, name=props.name,
        compute_capability=capability,
        total_memory_bytes=int(props.total_memory),
        torch_version=str(torch_module.__version__),
        cuda_version=str(torch_module.version.cuda),
        cudnn_version=int(torch_module.backends.cudnn.version()),
        visible_device_count=1,
    )
    return profile


__all__ = (
    "TrainerError", "ArmedM1Trainer50", "live_device_profile",
)
