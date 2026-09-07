"""Armed matched T0/C1 M1 trainer: one loop, one operator, full receipts data.

The loop mirrors the frozen CS-WG/matched-ERM 20-epoch physical core except
for the disclosed diagnostic-derivative-scan omission; the sole arm difference
is the training-gated calibration-prefix operator registered on the frozen
``SpintModel``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tfpd_exploration.src.cross_session_worst_group_v1 import core as cswg_core
from tfpd_exploration.src.cross_session_worst_group_v1 import plan as cswg_plan
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader

from . import hook as hook_module
from . import plan, schedule


class TrainerError(RuntimeError):
    """Fail closed for the armed matched trainer."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainerError(message)


def matched_erm_fold20120924_route_spec() -> v1.SourceRouteSpec:
    """The exact frozen matched-ERM fold-20120924 recipe spec (read-only reuse)."""
    sessions = plan.HELDIN_TRAINING_SESSIONS
    stage0 = cswg_plan.CSWGRunSpec(
        system="MATCHED_ERM",
        outer_target_session=plan.HELDOUT_FOLD_SESSIONS[0],
        source_sessions=sessions,
        initialization_seed=plan.SEED,
        lambda_=plan.OBJECTIVE_LAMBDA,
        tau=plan.OBJECTIVE_TAU,
        checkpoint_training_sessions=sessions,
    )
    return v1.SourceRouteSpec(
        stage0_spec=stage0,
        run_kind="full",
        root_relative=v1.full_root_relative(stage0.outer_target_session, "MATCHED_ERM"),
    )


@dataclass
class ArmedM1Trainer:
    """One arm of the matched pair; constructs nothing at import time."""

    code_root: Path
    source_root: Path
    arm: str
    device: str
    device_profile: Any = None
    _prepared: Any = field(default=None, init=False, repr=False)
    _progress: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(self.arm in plan.ARMS and self.device
                 and self.code_root.is_absolute() and self.source_root.is_absolute()
                 and isinstance(self.device_profile, v1.DeviceProfile),
                 "armed trainer construction drift")

    @property
    def progress(self) -> dict[str, object]:
        return dict(self._progress)

    def prepare(self) -> Mapping[str, object]:
        """Sealed-metadata source preparation through the frozen common-stratum
        fallback (the exact v6-bound full-run source path); no Torch or CUDA.

        The raw per-session stratum sets are NOT identical across sessions
        (47/51/50 distinct), so the frozen 20-epoch runs trained on the
        deterministic ``common ∩ min-count-2`` pruned pools.  This mirrors
        ``V6BoundFullCommonStratumSourceProvider`` exactly, including the
        accepted-v6 predecessor graph binding.
        """
        _require(self._prepared is None, "armed trainer prepare lifecycle drift")
        from tfpd_exploration.src.cross_session_worst_group_v1 import source_audit_v3 as v3
        from tfpd_exploration.src.cross_session_worst_group_v1 import source_smoke_physical_v4 as v4
        from tfpd_exploration.src.cross_session_worst_group_full_v1 import physical as full_physical
        from tfpd_exploration.src.cross_session_worst_group_v1.source_reader import (
            build_sealed_metadata_descriptor_provider, route_owned_reader_factory,
        )

        metadata = v1.load_m1_metadata_manifest_authority(Path(self.code_root))
        audit_spec = v1.source_audit_spec()
        full_spec = matched_erm_fold20120924_route_spec()
        _require(audit_spec.stage0_spec.source_sessions == full_spec.stage0_spec.source_sessions
                 and audit_spec.stage0_spec.outer_target_session
                 == full_spec.stage0_spec.outer_target_session,
                 "armed trainer audit/full source topology drift")
        resolver = build_sealed_metadata_descriptor_provider(
            source_root=Path(self.source_root), metadata_authority=metadata,
        )
        reader = route_owned_reader_factory(
            code_root=Path(self.code_root), source_root=Path(self.source_root),
        )
        descriptors = tuple(resolver.resolve_exact_sources(audit_spec))
        read_events: list[str] = []
        cached_materials = {}
        for descriptor in descriptors:
            base = reader.read_source_session(descriptor)
            cached_materials[descriptor.session_id] = v4._cache_material(base)
            read_events.append(descriptor.session_id)
        audit = v3.build_common_stratum_prepared_audit(
            physical_module=base_physical, spec=audit_spec,
            descriptors=descriptors, materials=cached_materials,
        )
        prepared = full_physical.rebind_v3_audit_prepared_to_full(
            physical_module=base_physical, audit_prepared=audit, full_spec=full_spec,
            historical_v6_audit_spec=audit_spec,
            accepted_v6_graph_sha256=plan.ACCEPTED_V6_GRAPH_SHA256,
        )
        self._prepared = prepared.inherited_full_prepared
        self._progress = {"source_resolved_or_opened": True, "read_events": read_events}
        return prepared.authority_fragment()

    def run(self, *, epoch_count: int, steps_per_epoch: int, record_steps: int) -> Mapping[str, object]:
        """Run the armed training loop; returns receipt bodies and checkpoint bytes."""
        import torch

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
                    epoch_rows.append({
                        "epoch_index": epoch_index,
                        "steps_in_epoch": steps_per_epoch,
                        "global_optimizer_steps_completed": global_step + 1,
                        "epoch_mean_source_train_loss": epoch_loss,
                        "epoch_loss_rows_sha256": epoch_rows_digest,
                        "source_train_loss_aggregation": "arithmetic_mean_of_complete_source_objective_per_step",
                        "model_state_sha256": base_physical._state_digest(model),
                        "finite_objective": bool(math.isfinite(final_loss)),
                        "operator_snapshot": operator.snapshot(),
                        "elapsed_seconds": elapsed,
                        "target_optimizer_backward_update": 0,
                        "source_only": True,
                    })
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_epoch_index = epoch_index
                        best_body = _serialize_model(model)
                        best_digest = base_physical._state_digest(model)
            model.eval()
            eval_proof = hook_module.eval_dropout_inactive_proof(model, device=self.device)
            model.train(False)
            final_digest = base_physical._state_digest(model)
            operator.detach()
            last_body = _serialize_model(model)
            best_body = best_body if best_body is not None else last_body
            best_digest = best_digest if best_digest is not None else final_digest
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
                "_checkpoint_bodies": {"best": best_body, "last": last_body},
            }
        finally:
            operator.detach()
            rng_snapshot.restore()
            runtime.close()


def stream_rows_tail(stream: hook_module.TrainingStreamRecorder, epoch_step: int) -> list[dict[str, object]]:
    return stream.rows  # retained for schema stability; unused


def epoch_mean_loss(stream: hook_module.TrainingStreamRecorder, global_step: int, steps_per_epoch: int) -> float:
    """Epoch mean over the FULL epoch's per-step losses (the recorder keeps every row)."""
    start = global_step + 1 - steps_per_epoch
    rows = stream.rows[start:global_step + 1]
    _require(len(rows) == steps_per_epoch, "epoch loss row coverage drift")
    return float(np.mean([row["loss"] for row in rows]))


def _serialize_model(model: Any) -> bytes:
    """Serialize the bare state dict (the strict-reload payload format)."""
    import io

    buffer = io.BytesIO()
    import torch

    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def live_device_profile(torch_module: Any) -> v1.DeviceProfile:
    """Build the single-visible-device profile and pin the operator GPU bound."""
    import os as _os
    import subprocess as _subprocess

    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(visible == plan.BOUND_GPU["cuda_visible_devices"],
             f"armed trainer requires CUDA_VISIBLE_DEVICES={plan.BOUND_GPU['cuda_visible_devices']}")
    _require(torch_module.cuda.is_available() and torch_module.cuda.device_count() == 1,
             "armed trainer requires exactly one visible CUDA device")
    props = torch_module.cuda.get_device_properties(0)
    completed = _subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader", "--id", "0"],
        check=True, text=True, capture_output=True, timeout=15,
    )
    uuid, pci = (part.strip() for part in completed.stdout.splitlines()[0].split(","))
    profile = v1.DeviceProfile(
        cuda_visible_devices=visible, torch_device=plan.BOUND_GPU["torch_device"],
        uuid=uuid, pci_bus_id=pci, name=props.name,
        compute_capability=tuple(int(item) for item in torch_module.cuda.get_device_capability(0)),
        total_memory_bytes=int(props.total_memory),
        torch_version=str(torch_module.__version__),
        cuda_version=str(torch_module.version.cuda),
        cudnn_version=int(torch_module.backends.cudnn.version()),
        visible_device_count=1,
    )
    _require(profile.name == "NVIDIA GeForce RTX 3090",
             "armed trainer device is not the bound RTX 3090 GPU 1")
    return profile


__all__ = (
    "TrainerError", "ArmedM1Trainer", "matched_erm_fold20120924_route_spec",
)
