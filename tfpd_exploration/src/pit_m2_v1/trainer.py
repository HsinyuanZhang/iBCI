"""The armed PIT-M2 trainer: the sealed M2 recipe with one hook attach point.

The lane binds to the ONLY trainer that produced the sealed
``m2_t4_activity_budget`` checkpoints (the ``25d7bc72`` family): the hydra
entry ``streaming_calibration_exp/src/train.py`` driving
``StreamingCalibrationLitModule`` (B3S, T4 side features) on
``FalconDataModule`` (task m2, chronological first-33, minival, seed 42,
12 epochs, Adam 1e-4 on the encoder only).  This module reconstructs that
stack from the SEALED resolved config (pinned by sha256) and drives it with
the Lightning ``Trainer`` exactly as the sealed launch did, with the single
arm difference being the training-gated prefix operator registered on
``litmodule.student`` before ``fit``.

Nothing here initializes CUDA on the CPU path; the smoke runs with
``CUDA_VISIBLE_DEVICES`` empty and asserts ``torch.cuda.is_initialized()``
stays False.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from . import hook as hook_module
from . import plan, schedule


class TrainerError(RuntimeError):
    """Fail closed for the armed PIT-M2 trainer."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainerError(message)


def ensure_streaming_paths(repo_root: Path) -> None:
    """Put the repo root and streaming_calibration_exp importable, idempotently."""
    import sys

    base = Path(repo_root).absolute()
    for entry in (str(base), str(base / "streaming_calibration_exp")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def load_sealed_config(repo_root: Path):
    """Load the SEALED resolved config (sha-pinned) with anchors bound."""
    ensure_streaming_paths(Path(repo_root))
    from omegaconf import OmegaConf

    path = Path(repo_root).absolute() / plan.SEALED_RESOLVED_CONFIG_RELATIVE
    digest = plan.sha256_file(path)
    _require(digest == plan.SEALED_RESOLVED_CONFIG_SHA256,
             "pit m2 sealed resolved-config drift")
    config = OmegaConf.load(path)
    config.model.teacher_ckpt_path = str(Path(repo_root).absolute() / plan.TEACHER_CHECKPOINT_RELATIVE)
    config.data.data_dir = str(Path(repo_root).absolute() / plan.M2_DATA_DIR_RELATIVE) + "/"
    _require(int(config.seed) == plan.SEED
             and int(config.data.calibration_n_trials) == plan.TRAINING_CALIBRATION_N_TRIALS
             and bool(config.data.random_calibration) is False
             and str(config.data.side_feature_group) == "t4"
             and int(config.trainer.max_epochs) == plan.EPOCHS
             and float(config.model.optimizer.lr) == plan.ADAM_LR
             and float(config.model.optimizer.weight_decay) == plan.ADAM_WEIGHT_DECAY
             and bool(config.model.freeze_decoder) is True
             and int(config.data.batch_size) == plan.TRAIN_BATCH_SIZE,
             "pit m2 sealed config literals drift against the frozen plan")
    return config


def _normalization_sha256(datamodule: Any) -> str:
    manifest = datamodule.get_split_manifest()
    encoded = manifest.get("native_t4_normalization")
    _require(isinstance(encoded, Mapping) and encoded.get("sha256"),
             "pit m2 datamodule produced no T4 normalization manifest")
    return str(encoded["sha256"])


def _state_digest(module: Any) -> str:
    import numpy as np
    import torch

    state = module.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        array = np.ascontiguousarray(state[name].detach().cpu().numpy())
        digest.update(plan.canonical_json_bytes(
            {"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _serialize_state_dict(module: Any) -> bytes:
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()


def epoch_val_metric(pl_module: Any):
    """The FRESH per-epoch val metric, computed from the module's own R2 states.

    Lightning calls Callback.on_validation_epoch_end BEFORE the module hook
    that logs ``val_heldin/r2_mean`` and resets the metric states, so reading
    ``trainer.callback_metrics`` from a callback yields nothing at epoch 0 and
    the PREVIOUS epoch's value afterwards (the attempt-1 staleness bug).  This
    helper mirrors the module's own formula over the still-populated states
    (sessions with ``total <= 2`` are skipped exactly as the module does).
    Returns None when no session has enough updates (no row is recorded).
    """
    import torch

    values = []
    for metric in pl_module.val_heldin_r2.values():
        if metric.total <= 2:
            continue
        values.append(torch.as_tensor(metric.compute()).detach().cpu())
    if not values:
        return None
    return float(torch.stack(values).mean())


class _StepRecorderCore:
    """Per-step matched-pair digest recorder (device-agnostic, no Lightning)."""

    def __init__(self, runner: "PitM2ArmedRunner") -> None:
        self.runner = runner
        self.stream = hook_module.TrainingStreamRecorder()
        self.optimizer_steps = 0
        self._before_python: tuple | None = None
        self._before_torch: Any = None

    def begin_step(self) -> None:
        self._before_python = random.getstate()
        self._before_torch = self.runner.torch.get_rng_state()

    def end_step(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        _neural, _covariates, _calib, session_names, side = batch
        _require(len(set(session_names)) == 1, "pit m2 training batch spans sessions")
        sample_ids = self.runner.sample_ids_for_batch(batch_idx)
        after_python = random.getstate()
        after_torch = self.runner.torch.get_rng_state()
        loss_value = float(self.runner.torch.as_tensor(loss).detach().cpu())
        self.stream.record_step(
            batch_idx,
            before_python=self._before_python, after_python=after_python,
            before_torch=self._before_torch, after_torch=after_torch,
            sample_ids=sample_ids,
            t4_bytes_digest=schedule.tensor_digest(side.detach().cpu().numpy()),
            loss=loss_value,
        )
        self.optimizer_steps += 1


def make_step_recorder_callback(runner: "PitM2ArmedRunner", core: _StepRecorderCore):
    """Build the Lightning Callback adapter lazily (imports lightning here)."""
    from lightning.pytorch.callbacks import Callback

    class _StepRecorder(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
            core.begin_step()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            core.end_step(outputs, batch, batch_idx)

    return _StepRecorder()


@dataclass
class PitM2ArmedRunner:
    """One arm of the matched pair; constructs nothing at import time."""

    repo_root: Path
    arm: str
    device: str = "cpu"
    record_steps: int = plan.RECORD_STEPS
    _config: Any = field(default=None, init=False, repr=False)
    _datamodule: Any = field(default=None, init=False, repr=False)
    _litmodule: Any = field(default=None, init=False, repr=False)
    _operator: Any = field(default=None, init=False, repr=False)
    _recorder: Any = field(default=None, init=False, repr=False)
    _recorder_core: Any = field(default=None, init=False, repr=False)
    _authority: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    torch: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(self.arm in plan.ARMS, "pit m2 armed runner arm drift")
        _require(self.device in ("cpu", "cuda:0"), "pit m2 armed runner device drift")
        _require(Path(self.repo_root).is_absolute(), "pit m2 armed runner needs absolute root")

    # -- construction ------------------------------------------------------
    def prepare(self, *, attach_operator: bool = True) -> Mapping[str, object]:
        """Build the sealed stack (data + module + optional hook); cpu-safe."""
        _require(self._litmodule is None, "pit m2 armed runner prepare lifecycle drift")
        import lightning as L

        ensure_streaming_paths(Path(self.repo_root))
        import torch

        self.torch = torch
        self._config = load_sealed_config(Path(self.repo_root))
        from hydra.utils import instantiate

        L.seed_everything(plan.SEED, workers=True)
        rng_after_seed = (schedule.rng_state_digest(random.getstate()),
                          schedule.torch_rng_state_digest(torch.get_rng_state()))
        datamodule = instantiate(self._config.data)
        datamodule.prepare_data()
        datamodule.setup("fit")
        normalization_sha = _normalization_sha256(datamodule)
        _require(normalization_sha == plan.NORMALIZATION_SHA256,
                 "pit m2 re-fit T4 normalizer drifted from the sealed d17f5f4c...")
        _require(getattr(datamodule, "val_heldout_dataset", None) is None,
                 "pit m2 fit stage resolved held-out data (target path resolved)")
        sampler_batches = [list(batch) for batch in datamodule.train_batch_sampler.batched_indices]
        _require(len(sampler_batches) == plan.SAMPLER_LAW["batches_per_epoch_sealed"],
                 "pit m2 train batches/epoch drift from the sealed 3618")
        _require(len(datamodule.train_dataset.window_indices)
                 == plan.SAMPLER_LAW["train_windows_sealed"],
                 "pit m2 train window count drift from the sealed 115911")
        rng_after_data = (schedule.rng_state_digest(random.getstate()),
                          schedule.torch_rng_state_digest(torch.get_rng_state()))
        litmodule = instantiate(self._config.model)
        litmodule.setup("fit")
        student = litmodule.student
        _require(student is not None and student._decoder_frozen,
                 "pit m2 student missing or decoder not frozen (sealed recipe)")
        rng_after_model = (schedule.rng_state_digest(random.getstate()),
                           schedule.torch_rng_state_digest(torch.get_rng_state()))
        operator = hook_module.PitM2PrefixOperator(self.arm, record_steps=self.record_steps)
        if attach_operator:
            operator.attach(student)
        self._datamodule = datamodule
        self._litmodule = litmodule
        self._operator = operator
        self._recorder_core = _StepRecorderCore(self)
        self._recorder = make_step_recorder_callback(self, self._recorder_core)
        self._authority = {
            "sealed_run_id": plan.SEALED_RUN_ID,
            "sealed_resolved_config_sha256": plan.SEALED_RESOLVED_CONFIG_SHA256,
            "teacher_checkpoint_sha256": plan.TEACHER_CHECKPOINT_SHA256,
            "normalization_sha256": normalization_sha,
            "train_sessions": list(datamodule.train_session_names),
            "val_heldin_sessions": list(datamodule.val_heldin_session_names),
            "batches_per_epoch": len(sampler_batches),
            "train_window_count": len(datamodule.train_dataset.window_indices),
            "sampler_batch_stream_sha256": schedule.int_list_digest(sampler_batches),
            "heldout_dataset_built": False,
            "rng_digests": {
                "after_seed_everything": rng_after_seed,
                "after_datamodule_setup": rng_after_data,
                "after_model_setup": rng_after_model,
            },
            "initial_student_state_sha256": _state_digest(student),
            "initial_teacher_state_sha256": _state_digest(litmodule.teacher),
            "operator_registered": bool(attach_operator),
        }
        return dict(self._authority)

    def sample_ids_for_batch(self, batch_idx: int) -> list[str]:
        dataset = self._datamodule.train_dataset
        indices = self._datamodule.train_batch_sampler.batched_indices[batch_idx]
        return [
            f"{dataset.window_indices[index][0]}@window_start:{dataset.window_indices[index][1]}"
            for index in indices
        ]

    # -- smoke -------------------------------------------------------------
    def run_smoke(self, steps: int = plan.SMOKE_STEPS) -> Mapping[str, object]:
        """First ``steps`` optimizer steps on the bound device (cpu for the smoke)."""
        _require(self._litmodule is not None and type(steps) is int and steps > 0,
                 "pit m2 smoke lifecycle drift")
        from lightning.pytorch.trainer import Trainer

        started = time.monotonic()
        trainer = Trainer(
            accelerator="cpu" if self.device == "cpu" else "gpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=plan.GRADIENT_CLIP_VAL,
            deterministic=False,
            callbacks=[self._recorder],
        )
        with hook_module.CountingRngProbe() as probe:
            trainer.fit(model=self._litmodule, datamodule=self._datamodule)
        rng_draw_counts = probe.snapshot()
        student = self._litmodule.student
        student.eval()
        eval_proof = hook_module.eval_dropout_inactive_proof(student, device=self.device)
        teacher = self._litmodule.teacher
        teacher.eval()
        teacher_proof = hook_module.teacher_eval_dropout_inert_proof(teacher, device=self.device)
        self._operator.detach()
        _require(self._recorder_core.optimizer_steps == steps,
                 "pit m2 smoke optimizer-step coverage drift")
        _require(rng_draw_counts["python"] == steps,
                 "pit m2 smoke expected exactly one inert teacher dropout-p draw per step")
        return {
            "arm": self.arm,
            "arm_role": plan.ARM_ROLES[self.arm],
            "device": self.device,
            "steps": steps,
            "optimizer_steps": self._recorder_core.optimizer_steps,
            "authority": dict(self._authority),
            "stream_records": self._recorder_core.stream.payload(steps),
            "operator_snapshot": self._operator.snapshot(),
            "rng_draw_counts_during_steps": rng_draw_counts,
            "eval_dropout_inactive_proof": eval_proof,
            "teacher_eval_dropout_inert_proof": teacher_proof,
            "final_student_state_sha256": _state_digest(student),
            "cuda_initialized": bool(self.torch.cuda.is_initialized()),
            "elapsed_seconds": time.monotonic() - started,
        }

    # -- full arm ----------------------------------------------------------
    def run_full(self) -> Mapping[str, object]:
        """The sealed 12-epoch budget with per-epoch val and best-checkpoint law."""
        _require(self._litmodule is not None, "pit m2 full-run lifecycle drift")
        from lightning.pytorch.callbacks import Callback
        from lightning.pytorch.trainer import Trainer

        started = time.monotonic()
        state: dict[str, object] = {"best_score": float("-inf"), "best_epoch": None}
        best_body: list[bytes] = []
        epoch_rows: list[dict[str, object]] = []
        runner = self

        class _ValTracker(Callback):
            def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
                score_value = epoch_val_metric(pl_module)
                if score_value is None:
                    return
                epoch = int(trainer.current_epoch)
                elapsed = time.monotonic() - started
                _require(elapsed <= plan.HARD_TIMEOUT_SECONDS_PER_ARM,
                         f"pit m2 arm exceeded the {plan.HARD_TIMEOUT_SECONDS_PER_ARM}s hard timeout")
                row = {
                    "epoch_index": epoch,
                    "optimizer_steps_completed": runner._recorder_core.optimizer_steps,
                    "val_heldin_r2_mean": score_value,
                    "student_state_sha256": _state_digest(pl_module.student),
                    "operator_snapshot": runner._operator.snapshot(),
                    "elapsed_seconds": elapsed,
                }
                epoch_rows.append(row)
                if score_value > float(state["best_score"]):
                    state["best_score"] = score_value
                    state["best_epoch"] = epoch
                    best_body.clear()
                    best_body.append(_serialize_state_dict(pl_module))

        trainer = Trainer(
            accelerator="cpu" if self.device == "cpu" else "gpu",
            devices=1,
            max_epochs=plan.EPOCHS,
            check_val_every_n_epoch=plan.CHECK_VAL_EVERY_N_EPOCH,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=plan.GRADIENT_CLIP_VAL,
            deterministic=False,
            log_every_n_steps=10,
            callbacks=[self._recorder, _ValTracker()],
        )
        with hook_module.CountingRngProbe() as probe:
            trainer.fit(model=self._litmodule, datamodule=self._datamodule)
        rng_draw_counts = probe.snapshot()
        student = self._litmodule.student
        student.eval()
        eval_proof = hook_module.eval_dropout_inactive_proof(student, device=self.device)
        self._operator.detach()
        last_body = _serialize_state_dict(self._litmodule)
        _require(bool(best_body), "pit m2 full run produced no best checkpoint")
        _require(self._recorder_core.optimizer_steps == plan.EPOCHS * len(
            self._datamodule.train_batch_sampler.batched_indices),
            "pit m2 full-run optimizer-step coverage drift")
        return {
            "arm": self.arm,
            "arm_role": plan.ARM_ROLES[self.arm],
            "device": self.device,
            "epochs": epoch_rows,
            "best_epoch_index": state["best_epoch"],
            "best_checkpoint_metric": plan.CHECKPOINT_MONITOR,
            "best_checkpoint_metric_value": state["best_score"],
            "optimizer_steps": self._recorder_core.optimizer_steps,
            "authority": dict(self._authority),
            "stream_records": self._recorder_core.stream.payload(self.record_steps),
            "operator_snapshot": self._operator.snapshot(),
            "rng_draw_counts_during_run": rng_draw_counts,
            "eval_dropout_inactive_proof": eval_proof,
            "final_student_state_sha256": _state_digest(student),
            "cuda_initialized": bool(self.torch.cuda.is_initialized()),
            "elapsed_seconds": time.monotonic() - started,
            "_checkpoint_bodies": {"best": best_body[0], "last": last_body},
        }


# ---------------------------------------------------------------------------
# GPU authorization (train / phase3 stages).
# ---------------------------------------------------------------------------

def query_gpu_rows(cmdline_runner: Callable[[list[str]], str] | None = None) -> list[dict[str, str]]:
    """Query BOTH physical GPUs; never a bare nvidia-smi call in tests."""
    runner = cmdline_runner or _default_cmdline
    text = runner(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"])
    rows: list[dict[str, str]] = []
    for line in (line.strip() for line in text.strip().splitlines() if line.strip()):
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 4, "pit m2 nvidia-smi row drift")
        rows.append({"index": parts[0], "uuid": parts[1],
                     "memory_used_mib": parts[2], "utilization_gpu": parts[3]})
    _require(bool(rows), "pit m2 nvidia-smi returned no GPUs")
    return rows


def _default_cmdline(arguments: list[str]) -> str:
    completed = subprocess.run(arguments, check=True, text=True, capture_output=True, timeout=30)
    return completed.stdout


def assert_both_gpus_idle(cmdline_runner: Callable[[list[str]], str] | None = None) -> list[dict[str, object]]:
    """Fail closed unless EVERY visible-in-query GPU is idle (the v1 law)."""
    rows = query_gpu_rows(cmdline_runner)
    for row in rows:
        memory = float(row["memory_used_mib"])
        utilization = float(row["utilization_gpu"])
        _require(memory < plan.GPU_IDLE_MEMORY_THRESHOLD_MIB
                 and utilization <= plan.GPU_IDLE_UTILIZATION_PERCENT,
                 f"pit m2 GPU {row['index']} is busy "
                 f"({row['memory_used_mib']} MiB, {row['utilization_gpu']}%); refusing")
    return [dict(row) for row in rows]


def query_compute_apps(cmdline_runner: Callable[[list[str]], str] | None = None
                       ) -> list[dict[str, str]]:
    """Query every running compute app (gpu_uuid, pid, process, memory)."""
    runner = cmdline_runner or _default_cmdline
    text = runner(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                   "--format=csv,noheader,nounits"])
    rows: list[dict[str, str]] = []
    for line in (line.strip() for line in text.strip().splitlines() if line.strip()):
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 4, "pit m2 nvidia-smi compute-app row drift")
        rows.append({"gpu_uuid": parts[0], "pid": parts[1],
                     "process_name": parts[2], "used_memory_mib": parts[3]})
    return rows


def assert_target_gpu_idle(gpu_index: int,
                           cmdline_runner: Callable[[list[str]], str] | None = None
                           ) -> dict[str, object]:
    """The amended (2026-09-02) guard: only the TARGET card must be idle.

    Refuses if the target card has compute apps, or >= 100 MiB used, or >0%
    utilization.  A busy OTHER card does not refuse the run (the relaxation
    the operator ordered); its state is still recorded in the receipt.
    """
    _require(type(gpu_index) is int and gpu_index >= 0, "gpu_index drift")
    rows = query_gpu_rows(cmdline_runner)
    target = next((row for row in rows if row["index"] == str(gpu_index)), None)
    _require(target is not None, f"pit m2 target GPU {gpu_index} absent from the query")
    memory = float(target["memory_used_mib"])
    utilization = float(target["utilization_gpu"])
    _require(memory < plan.GPU_IDLE_MEMORY_THRESHOLD_MIB
             and utilization <= plan.GPU_IDLE_UTILIZATION_PERCENT,
             f"pit m2 target GPU {gpu_index} is busy "
             f"({target['memory_used_mib']} MiB, {target['utilization_gpu']}%); refusing")
    apps = query_compute_apps(cmdline_runner)
    target_apps = [app for app in apps if app["gpu_uuid"] == target["uuid"]]
    _require(not target_apps,
             f"pit m2 target GPU {gpu_index} has compute apps: "
             f"{[app['pid'] for app in target_apps]}; refusing")
    return {
        "gpu_index": int(gpu_index),
        "target_uuid": target["uuid"],
        "target_memory_used_mib": target["memory_used_mib"],
        "target_utilization_gpu": target["utilization_gpu"],
        "target_compute_apps": [],
        "other_cards_recorded": [dict(row) for row in rows if row["index"] != str(gpu_index)],
        "checked_at_epoch_seconds": time.time(),
    }


def live_device_profile(torch_module: Any, gpu_index: int) -> dict[str, object]:
    """Pin the single visible card and record its identity."""
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(visible == str(gpu_index),
             f"pit m2 armed trainer requires CUDA_VISIBLE_DEVICES={gpu_index}")
    _require(torch_module.cuda.is_available() and torch_module.cuda.device_count() == 1,
             "pit m2 armed trainer requires exactly one visible CUDA device")
    props = torch_module.cuda.get_device_properties(0)
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader", "--id", "0"],
        check=True, text=True, capture_output=True, timeout=15,
    )
    uuid, pci = (part.strip() for part in completed.stdout.splitlines()[0].split(","))
    return {
        "cuda_visible_devices": visible,
        "torch_device": "cuda:0",
        "uuid": uuid,
        "pci_bus_id": pci,
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "torch_version": str(torch_module.__version__),
        "visible_device_count": 1,
    }


def require_train_authorization(gpu_authorized: bool, *, gpu_index: int | None = None,
                                cmdline_runner: Callable[[list[str]], str] | None = None
                                ) -> dict[str, object]:
    """The train/phase3 refusal law: flag AND the target GPU idle.

    ``gpu_index=None`` keeps the original v1 BOTH-idle law (used by tests and
    any legacy caller); an explicit index applies the amended target-idle law.
    """
    _require(bool(gpu_authorized),
             "pit m2 train/phase3 REFUSED: pass --gpu-authorized (operator decision required)")
    if gpu_index is None:
        rows = assert_both_gpus_idle(cmdline_runner)
        return {
            "gpu_authorized": True,
            "law": "both_gpus_idle_v1",
            "both_gpus_idle": True,
            "gpu_rows": rows,
            "checked_at_epoch_seconds": time.time(),
        }
    target = assert_target_gpu_idle(gpu_index, cmdline_runner)
    return {
        "gpu_authorized": True,
        "law": "target_gpu_idle_v2_20260902",
        **target,
    }


__all__ = (
    "TrainerError", "ensure_streaming_paths", "load_sealed_config", "PitM2ArmedRunner",
    "epoch_val_metric", "query_gpu_rows", "assert_both_gpus_idle", "query_compute_apps",
    "assert_target_gpu_idle", "live_device_profile", "require_train_authorization",
)
