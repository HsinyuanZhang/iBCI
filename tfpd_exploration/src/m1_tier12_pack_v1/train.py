"""Armed matched trainer for the four Tier 1/2 pack arms.

Cloned from m1_t0c1_prefix_v1.trainer / hook (hook imported unchanged from
T0/C1; prefix kind mapped t0_* → t0, c1_* → c1). Changes vs sealed T0/C1:
GPU index + UUID bind, public execute, SWA final-4 on t0_swa/c1_swa, W32
SpintIdentityWidthModel on t0_w32/c1_w32. Recipe otherwise identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tfpd_exploration.src.cross_session_worst_group_v1 import core as cswg_core
from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.m1_t0c1_prefix_v1 import hook as t0c1_hook
from tfpd_exploration.src.m1_t0c1_prefix_v1 import trainer as t0c1_trainer
from tfpd_exploration.src.m1_t0c1_prefix_v1 import schedule as t0c1_schedule

from . import plan


class TrainerError(RuntimeError):
    """Fail closed for the packed matched trainer."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainerError(message)


matched_erm_fold20120924_route_spec = t0c1_trainer.matched_erm_fold20120924_route_spec
epoch_mean_loss = t0c1_trainer.epoch_mean_loss


def apply_swa_final_four(checkpoint_paths: Sequence[Path], out_path: Path) -> dict[str, object]:
    """FP64 mean of the final four wrapped ``state_dict`` checkpoints.

    Loads ``matched_scorer.py`` by file path so ``tfpd_lane/__init__.py`` (which
    imports ``src.tfpd_lane``) is never executed.
    """
    import importlib.util

    paths = [Path(path) for path in checkpoint_paths]
    _require(len(paths) == 4, "SWA window is exactly the final four checkpoints")
    source = Path(__file__).resolve().parent.parent / "tfpd_lane" / "matched_scorer.py"
    _require(source.is_file(), "matched_scorer.py absent")
    spec = importlib.util.spec_from_file_location("_m1_tier12_matched_scorer", source)
    _require(spec is not None and spec.loader is not None, "matched_scorer import spec drift")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_swa_final_four(paths, Path(out_path))


def w32_constructor_kwargs() -> dict[str, object]:
    kwargs = dict(base_physical._exact_spint_constructor_kwargs())
    kwargs["identity_width"] = plan.IDENTITY_WIDTH_W32
    return kwargs


def load_w32_model(code_root: Path) -> Any:
    """Construct SpintIdentityWidthModel with sealed M1 kwargs + identity_width=32."""
    spint_main = Path(code_root).absolute() / "SPINT-main"
    _require(spint_main.is_dir(), "SPINT-main tree absent for W32 constructor")
    inserted = str(spint_main)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    from src.models.components.spint_identity_width import SpintIdentityWidthModel

    model = SpintIdentityWidthModel(**w32_constructor_kwargs())
    last = list(model.fc_id_out.children())[-1]
    _require(int(model.identity_width) == plan.IDENTITY_WIDTH_W32, "W32 identity_width drift")
    _require(int(model.model_dim) == 1024 and int(model.window_size) == 100, "W32 decoder shape drift")
    _require(int(last.out_features) == 100, "W32 fc_id_out last Linear must be window_size=100")
    _require(hasattr(model, "fc_id_in") and hasattr(model, "fc_id_out"), "W32 missing identity MLP")
    return model


def materialize_w32_model(model: Any, *, device: str) -> dict[str, object]:
    """Materialize LazyLinear via one eval forward. Do not use the W1024 helper."""
    import torch

    from tfpd_exploration.src.cross_session_worst_group_v1 import plan as cswg_literals

    _require(isinstance(model, torch.nn.Module), "W32 materialization requires a Torch module")
    x = torch.zeros(
        (1, cswg_literals.M1_WINDOW_SIZE, cswg_literals.M1_UNIT_COUNT),
        dtype=torch.float32, device=device,
    )
    calibration = torch.zeros(
        (1, *cswg_literals.M1_CALIBRATION_SHAPE_PER_ROW), dtype=torch.float32, device=device,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(x, calib_trialized_neural_features=calibration)
    model.train(was_training)
    _require(
        tuple(output.shape) == (1, cswg_literals.M1_WINDOW_SIZE, cswg_literals.M1_RAW_BEHAVIOR_OUTPUTS)
        and bool(torch.isfinite(output).all()),
        "W32 lazy materialization output drift",
    )
    last = list(model.fc_id_out.children())[-1]
    _require(int(last.out_features) == 100, "W32 fc_id_out last Linear out_features drift")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "output_shape": list(output.shape),
        "live_parameter_count": count,
        "identity_width": int(model.identity_width),
        "fc_id_out_last_out_features": int(last.out_features),
        "device": str(output.device),
    }


def _serialize_bare_state(model: Any) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _serialize_wrapped_state(model: Any) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, buffer)
    return buffer.getvalue()


def live_device_profile(torch_module: Any, gpu_index: int) -> v1.DeviceProfile:
    """Build the single-visible-device profile and pin the requested GPU UUID."""
    import os as _os
    import subprocess as _subprocess

    _require(type(gpu_index) is int and gpu_index == plan.GPU1_INDEX, "this cell may only use GPU 1")
    visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(visible == str(gpu_index),
             f"trainer requires CUDA_VISIBLE_DEVICES={gpu_index}, saw {visible!r}")
    _require(torch_module.cuda.is_available() and torch_module.cuda.device_count() == 1,
             "trainer requires exactly one visible CUDA device")
    props = torch_module.cuda.get_device_properties(0)
    completed = _subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader",
         "--id", str(gpu_index)],
        check=True, text=True, capture_output=True, timeout=15,
    )
    uuid, pci = (part.strip() for part in completed.stdout.splitlines()[0].split(","))
    expected = plan.gpu_uuid(gpu_index)
    _require(uuid == expected, f"trainer GPU UUID drift: {uuid} != {expected}")
    profile = v1.DeviceProfile(
        cuda_visible_devices=visible, torch_device="cuda:0",
        uuid=uuid, pci_bus_id=pci, name=props.name,
        compute_capability=tuple(int(item) for item in torch_module.cuda.get_device_capability(0)),
        total_memory_bytes=int(props.total_memory),
        torch_version=str(torch_module.__version__),
        cuda_version=str(torch_module.version.cuda),
        cudnn_version=int(torch_module.backends.cudnn.version()),
        visible_device_count=1,
    )
    _require(profile.name == plan.DEVICE_NAME, "trainer device is not the bound RTX 3090")
    return profile


@dataclass
class ArmedM1Trainer:
    """One train arm of the pack cell; constructs nothing at import time."""

    code_root: Path
    source_root: Path
    arm: str
    device: str
    device_profile: Any = None
    gpu_index: int = plan.GPU1_INDEX
    _prepared: Any = field(default=None, init=False, repr=False)
    _progress: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(self.arm in plan.TRAIN_ARMS and self.device
                 and self.code_root.is_absolute() and self.source_root.is_absolute()
                 and isinstance(self.device_profile, v1.DeviceProfile)
                 and type(self.gpu_index) is int and self.gpu_index == plan.GPU1_INDEX,
                 "armed trainer construction drift")

    @property
    def progress(self) -> dict[str, object]:
        return dict(self._progress)

    def prepare(self) -> Mapping[str, object]:
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
        import torch

        _require(self._prepared is not None and type(epoch_count) is int and epoch_count > 0
                 and type(steps_per_epoch) is int and steps_per_epoch > 0,
                 "armed trainer run lifecycle drift")
        spec_arm = plan.ARM_SPECS[self.arm]
        prefix = plan.prefix_kind(self.arm)
        prepared = self._prepared
        spec = prepared.spec
        total_steps = epoch_count * steps_per_epoch
        started = time.monotonic()
        runtime = base_physical.SelectedCudaRuntime(self.device_profile)
        runtime.__enter__()
        rng_snapshot = base_physical.ProcessRngSnapshot.capture_and_seed(torch, seed=plan.SEED)
        operator = t0c1_hook.M1CalPrefixOperator(prefix, record_steps=record_steps)
        stream = t0c1_hook.TrainingStreamRecorder()
        try:
            if spec_arm["stock_spint"]:
                model = base_physical.load_exact_m1_spint_model(Path(self.code_root))
                model = model.to(self.device)
                materialization = base_physical.materialize_exact_m1_model(model, device=self.device)
            else:
                model = load_w32_model(Path(self.code_root))
                model = model.to(self.device)
                materialization = materialize_w32_model(model, device=self.device)
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
            swa_bodies: dict[int, bytes] = {}
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
                stream.record_step(
                    global_step, before_state=state_before, after_state=state_after,
                    sample_ids=[row.sample_id for row in episode.all_rows], loss=final_loss,
                )
                self._progress = {
                    "source_resolved_or_opened": True,
                    "model_constructed": True,
                    "cuda_initialized": bool(torch.cuda.is_initialized()),
                    "optimizer_steps_completed": global_step + 1,
                    "arm": self.arm,
                }
                if epoch_step == steps_per_epoch - 1:
                    epoch_loss = epoch_mean_loss(stream, global_step, steps_per_epoch)
                    epoch_rows_digest = t0c1_schedule.sequence_digest(
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
                        best_body = _serialize_bare_state(model)
                        best_digest = base_physical._state_digest(model)
                    if spec_arm["swa"] and epoch_index in plan.SWA_EPOCH_INDICES:
                        swa_bodies[epoch_index] = _serialize_wrapped_state(model)
            model.eval()
            eval_proof = t0c1_hook.eval_dropout_inactive_proof(model, device=self.device)
            model.train(False)
            final_digest = base_physical._state_digest(model)
            operator.detach()
            last_body = _serialize_bare_state(model)
            best_body = best_body if best_body is not None else last_body
            best_digest = best_digest if best_digest is not None else final_digest
            checkpoint_bodies: dict[str, bytes] = {"best": best_body, "last": last_body}
            swa_manifest: dict[str, object] | None = None
            if spec_arm["swa"]:
                _require(epoch_count == plan.EPOCHS and set(swa_bodies) == set(plan.SWA_EPOCH_INDICES),
                         "SWA arm missing epoch_16..epoch_19 state_dict snapshots")
                with tempfile.TemporaryDirectory(prefix="tier12_swa_") as tmp:
                    tmp_path = Path(tmp)
                    paths = []
                    for index in plan.SWA_EPOCH_INDICES:
                        path = tmp_path / f"epoch_{index}.pt"
                        path.write_bytes(swa_bodies[index])
                        paths.append(path)
                        checkpoint_bodies[f"epoch_{index}"] = swa_bodies[index]
                    swa_path = tmp_path / "swa_final4.pt"
                    swa_manifest = apply_swa_final_four(paths, swa_path)
                    checkpoint_bodies["swa_final4"] = swa_path.read_bytes()
            return {
                "arm": self.arm,
                "prefix_kind": prefix,
                "identity_width": spec_arm["identity_width"],
                "swa_enabled": bool(spec_arm["swa"]),
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
                "scheduler": plan.SCHEDULER,
                "cosine_50ep_forbidden": plan.COSINE_50EP_FORBIDDEN,
                "gpu_index": self.gpu_index,
                "swa_manifest": swa_manifest,
                "_checkpoint_bodies": checkpoint_bodies,
            }
        finally:
            operator.detach()
            rng_snapshot.restore()
            runtime.close()
