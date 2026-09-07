"""Coordinated single-loader, three-arm post-fusion training runtime.

This is intentionally a route-owned coordinator rather than three Lightning
``PitM2ArmedRunner.run_full`` calls: that method has a fixed 12-epoch, one-arm
topology and constructs a DataModule per runner.  We still compose its sealed
config/data/module preparation primitives and do not copy them.
"""
from __future__ import annotations

import copy
import hashlib
import io
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from . import controller, plan, selection, variants


class RunnerError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


@dataclass
class _RngSnapshot:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _capture_rng(device: torch.device) -> _RngSnapshot:
    cuda = tuple(torch.cuda.get_rng_state_all()) if device.type == "cuda" else ()
    return _RngSnapshot(random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda)


def _restore_rng(snapshot: _RngSnapshot, device: torch.device) -> None:
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))


def _rng_digests(snapshot: _RngSnapshot) -> dict[str, str]:
    return {
        "python": _digest_bytes(repr(snapshot.python).encode("utf-8")),
        "numpy": _digest_bytes(repr(snapshot.numpy).encode("utf-8")),
        "torch": _digest_bytes(snapshot.torch_cpu.detach().cpu().numpy().tobytes()),
        "cuda": _digest_bytes(b"".join(item.detach().cpu().numpy().tobytes() for item in snapshot.torch_cuda)),
    }


def _same_rng(left: _RngSnapshot, right: _RngSnapshot) -> bool:
    return (left.python == right.python and left.numpy[0] == right.numpy[0]
            and np.array_equal(left.numpy[1], right.numpy[1]) and left.numpy[2:] == right.numpy[2:]
            and torch.equal(left.torch_cpu, right.torch_cpu)
            and len(left.torch_cuda) == len(right.torch_cuda)
            and all(torch.equal(a, b) for a, b in zip(left.torch_cuda, right.torch_cuda)))


def _to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=False)
    if isinstance(batch, tuple):
        return tuple(_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [_to_device(item, device) for item in batch]
    return batch


def _selected_batch(batch: tuple[Any, ...], decision: controller.PoolDecision) -> tuple[Any, ...]:
    _require(len(batch) in (5, 6), "sealed M2 batch tuple arity drift")
    calib = batch[2]
    _require(int(calib.shape[1]) >= plan.POOL_CYCLE[0], "M2 batch lacks 30 valid members")
    selected = calib[:, list(decision.member_indices)]
    return tuple([batch[0], batch[1], selected, *batch[3:]])


def _state_digest(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(plan.canonical_json_bytes({"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}))
        digest.update(array.tobytes())
    return digest.hexdigest()


class PostFusionVariantScreenRunner:
    """One datamodule/loader; three independent models stepped on each batch."""

    def __init__(self, *, repo_root, device: str = "cuda:0") -> None:
        self.repo_root = repo_root
        self.device = torch.device(device)
        self.datamodule: Any = None
        self.train_loader: Any = None
        self.monitor_loader: Any = None
        self.models: dict[str, Any] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.adapters: dict[str, variants.PostFusionIdentityAdapter] = {}
        self.pool_controller = controller.ChronologicalPoolController()
        self.global_step = 0
        self.source_authority: dict[str, object] = {}
        self._canonical_rng: _RngSnapshot | None = None
        self._epoch_rows: dict[str, list[dict[str, object]]] = {arm: [] for arm in plan.ARMS}
        self._best_source: dict[str, dict[str, object]] = {}

    @staticmethod
    def _window_indices_from_loader(loader: Any, *, label: str) -> tuple[object, ...]:
        dataset = getattr(loader, "dataset", None)
        indices = getattr(dataset, "window_indices", None)
        _require(isinstance(indices, (tuple, list)),
                 f"{label} loader has no materialized window_indices authority")
        return tuple(indices)

    def prepare(self) -> Mapping[str, object]:
        """Construct source data exactly once using PIT's sealed preparation."""
        _require(self.datamodule is None, "postfusion runner prepare lifecycle drift")
        from tfpd_exploration.src.pit_m2_v1 import trainer as pit_trainer

        # This call is the sole DataModule construction/materialization point.
        source = pit_trainer.PitM2ArmedRunner(self.repo_root, "t0m", device=str(self.device))
        authority = dict(source.prepare(attach_operator=False))
        self.datamodule = source._datamodule
        base = source._litmodule
        _require(self.datamodule is not None and base is not None, "PIT source preparation returned no stack")
        train_groups = tuple(str(item) for item in self.datamodule.train_session_names)
        monitor_groups = tuple(str(item) for item in self.datamodule.val_heldin_session_names)
        # Construct loaders once; all arms share the same resident batch from
        # each iterator position.  No arm owns a loader or sampler.
        self.train_loader = self.datamodule.train_dataloader()
        monitor = self.datamodule.val_dataloader()
        self.monitor_loader = monitor[0] if isinstance(monitor, (list, tuple)) else monitor
        monitor_authority = selection.validate_source_heldin_in_sample_monitor(
            train_groups, monitor_groups,
            self._window_indices_from_loader(self.train_loader, label="source train"),
            self._window_indices_from_loader(self.monitor_loader, label="source monitor"),
        )
        initial_state = copy.deepcopy(base.state_dict())
        initial_sha = _state_digest(base.student)
        for arm in plan.ARMS:
            module = copy.deepcopy(base)
            module.load_state_dict(initial_state, strict=True)
            adapter = variants.install_variant(module.student, arm)
            # Install residual alpha before the only device transfer: otherwise
            # PF-R1/PF-R50 would carry a CPU alpha into a CUDA forward.
            module.to(self.device)
            # The decoder stays frozen; the adapter contains the sealed
            # pre/post objects plus only the explicit residual alpha.
            _require(all(not parameter.requires_grad for parameter in module.student.decoder.parameters()),
                     "postfusion screen decoder became trainable")
            configured = module.configure_optimizers()
            _require(set(configured) == {"optimizer"}, "postfusion V1 forbids scheduler drift")
            optimizer = configured["optimizer"]
            _require(isinstance(optimizer, torch.optim.Adam), "postfusion route requires sealed Adam optimizer")
            _require(len(optimizer.param_groups) == 1
                     and float(optimizer.param_groups[0]["lr"]) == plan.ADAM_LR
                     and float(optimizer.param_groups[0]["weight_decay"]) == plan.ADAM_WEIGHT_DECAY,
                     "sealed Adam hyperparameter drift")
            self.models[arm] = module
            self.adapters[arm] = adapter
            self.optimizers[arm] = optimizer
        _require(len({id(model) for model in self.models.values()}) == len(plan.ARMS),
                 "postfusion arms share mutable model state")
        self._canonical_rng = _capture_rng(self.device)
        self.source_authority = {
            **authority,
            "source_heldin_in_sample_monitor": monitor_authority,
            "data_module_constructions": 1,
            "shared_train_loader_object_id": id(self.train_loader),
            "shared_monitor_loader_object_id": id(self.monitor_loader),
            "initial_native_student_state_sha256": initial_sha,
            "arms": {arm: variants.parameter_topology(adapter) for arm, adapter in self.adapters.items()},
            "pool_cycle": list(plan.POOL_CYCLE),
        }
        return dict(self.source_authority)

    def _step_group(self, raw_batch: tuple[Any, ...], *, detailed: bool = False) -> dict[str, object]:
        _require(self._canonical_rng is not None, "missing canonical paired RNG state")
        # Receipt hashes must be computed against the one CPU-materialized
        # batch, before the single transfer below.  Ordinary steps intentionally
        # do not hash full tensors at all.
        decision = self.pool_controller.decide(self.global_step, available_members=int(raw_batch[2].shape[1]))
        before = self._canonical_rng
        before_digests = _rng_digests(before)
        common: dict[str, object] = {
            "global_step": self.global_step, "member_count": decision.member_count,
            "member_indices": list(decision.member_indices), "batch_object_id": id(raw_batch),
            "paired_rng_before": before_digests,
        }
        if detailed:
            common.update(controller.shared_batch_evidence(
                batch=raw_batch, decision=decision, python_rng_sha256=before_digests["python"],
                numpy_rng_sha256=before_digests["numpy"], torch_rng_sha256=before_digests["torch"],
            ))
        batch = _to_device(raw_batch, self.device)
        selected = _selected_batch(tuple(batch), decision)
        arm_records: dict[str, dict[str, object]] = {}
        after_states: dict[str, _RngSnapshot] = {}
        for arm in plan.ARMS:
            _restore_rng(before, self.device)
            model = self.models[arm]
            optimizer = self.optimizers[arm]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            state_before = _state_digest(model.student) if detailed else None
            out = model.model_step(selected)
            loss = out["loss"]
            if detailed:
                _require(bool(torch.isfinite(loss).item()), f"{arm}: nonfinite loss")
            loss.backward()
            adapter = self.adapters[arm]
            def _grad_l1(parameters: Any) -> torch.Tensor:
                values = [parameter.grad.detach().abs().sum() for parameter in parameters if parameter.grad is not None]
                return torch.stack(values).sum() if values else loss.new_zeros(())

            alpha_grad_tensor = (None if adapter.alpha is None or not detailed
                                 else adapter.alpha.grad.detach().abs().sum())
            pre_grad_tensor = _grad_l1(adapter.native.pre_pool.parameters()) if detailed else None
            post_grad_tensor = _grad_l1(adapter.native.post_pool.parameters()) if detailed else None
            if detailed:
                assert pre_grad_tensor is not None and post_grad_tensor is not None
                required_grads = [pre_grad_tensor, post_grad_tensor]
                if arm != "PF-MEAN":
                    _require(alpha_grad_tensor is not None, f"{arm}: alpha gradient is absent")
                    required_grads.append(alpha_grad_tensor)
                grad_vector = torch.stack(required_grads)
                # One device-to-host boolean per arm/detailed step only.
                _require(bool((torch.isfinite(grad_vector) & (grad_vector > 0)).all().item()),
                         f"{arm}: identity branch/alpha gradient is zero or nonfinite")
            optimizer.step()
            after = _capture_rng(self.device)
            after_states[arm] = after
            if detailed:
                arm_records[arm] = {
                    "loss": float(loss.detach().cpu()), "state_before_sha256": state_before,
                    "state_after_sha256": _state_digest(model.student),
                    "pre_pool_grad_l1": float(pre_grad_tensor.detach().cpu()),
                    "post_pool_grad_l1": float(post_grad_tensor.detach().cpu()),
                    "alpha_grad_l1": None if alpha_grad_tensor is None else float(alpha_grad_tensor.detach().cpu()),
                    "alpha_l1_after": None if adapter.alpha is None else float(adapter.alpha.detach().abs().sum().cpu()),
                    "rng_after": _rng_digests(after),
                }
            else:
                arm_records[arm] = {"ordinary_step": True}
        canonical_after = after_states[plan.ARMS[0]]
        _require(all(_same_rng(canonical_after, after_states[arm]) for arm in plan.ARMS[1:]),
                 "paired arms consumed different Python/NumPy/Torch/CUDA RNG streams")
        self._canonical_rng = canonical_after
        _restore_rng(canonical_after, self.device)
        self.global_step += 1
        return {"shared": common, "arms": arm_records, "paired_rng_exact": True, "detailed": detailed}

    @staticmethod
    def _require_before_deadline(deadline: float | None) -> None:
        if deadline is not None:
            _require(time.monotonic() <= deadline,
                     "postfusion 12-epoch screen exceeded the 180-minute hard cap")

    def _finish_epoch(self, epoch: int, iterator: Any, *, completed_steps: int = 0,
                      detailed_records: list[dict[str, object]] | None = None,
                      deadline: float | None = None) -> dict[str, object]:
        _require(self.train_loader is not None and 1 <= int(epoch) <= plan.SOURCE_EPOCHS_SCREEN,
                 "postfusion run_epoch lifecycle drift")
        started = time.monotonic()
        step_count = int(completed_steps)
        smoke_records = [] if detailed_records is None else list(detailed_records)
        for raw_batch in iterator:
            # This is a real wall-clock stop condition, not merely the smoke
            # projection.  It runs before every shared resident-batch group.
            self._require_before_deadline(deadline)
            # Detailed receipts are limited to the mandatory first 12 paired
            # steps and one sentinel on each source milestone.  No ordinary
            # step transfers full tensors/model state back to CPU.
            detailed = self.global_step < plan.SMOKE_SHARED_STEPS or (epoch in plan.SCREEN_MILESTONES and step_count == 0)
            record = self._step_group(tuple(raw_batch), detailed=detailed)
            if detailed:
                smoke_records.append(record)
            step_count += 1
        self._require_before_deadline(deadline)
        metrics = {arm: self.source_heldin_in_sample_monitor_metric(arm) for arm in plan.ARMS}
        rows: dict[str, dict[str, object]] = {}
        for arm in plan.ARMS:
            finite_parameters = torch.stack([
                torch.isfinite(parameter.detach()).all() for parameter in self.models[arm].student.parameters()
            ]).all()
            _require(bool(finite_parameters.item()), f"{arm}: nonfinite student parameter at epoch boundary")
            row = {
                "epoch": int(epoch), "optimizer_steps_completed": int(self.global_step),
                "source_heldin_in_sample_monitor_mean": metrics[arm]["equal_session_mean"],
                "source_heldin_in_sample_monitor_per_session": metrics[arm]["per_session_r2"],
                "student_state_sha256": _state_digest(self.models[arm].student),
                "elapsed_seconds": time.monotonic() - started,
            }
            self._epoch_rows[arm].append(row)
            prior = self._best_source.get(arm)
            if prior is None or (float(row["source_heldin_in_sample_monitor_mean"])
                                 > float(prior["source_heldin_in_sample_monitor_mean"])):
                buffer = io.BytesIO()
                torch.save({"model": self.models[arm].state_dict(), "optimizer": self.optimizers[arm].state_dict()}, buffer)
                self._best_source[arm] = {
                    "epoch": int(epoch),
                    "source_heldin_in_sample_monitor_mean": row["source_heldin_in_sample_monitor_mean"],
                    "student_state_sha256": row["student_state_sha256"], "body": buffer.getvalue(),
                }
            rows[arm] = row
        return {"epoch": int(epoch), "step_count": step_count, "detailed_step_records": smoke_records,
                "arms": rows}

    def run_epoch(self, epoch: int, *, deadline: float | None = None) -> dict[str, object]:
        _require(self.train_loader is not None, "postfusion train loader is absent")
        return self._finish_epoch(epoch, iter(self.train_loader), deadline=deadline)

    @staticmethod
    def _validate_smoke_records(records: list[dict[str, object]]) -> None:
        _require(len(records) == plan.SMOKE_SHARED_STEPS, "postfusion smoke must contain exactly 12 paired steps")
        _require(all(record.get("paired_rng_exact") is True for record in records),
                 "postfusion paired RNG smoke law failed")
        for record in records:
            arms = record["arms"]
            for arm in plan.ARMS:
                entry = arms[arm]
                _require(entry.get("state_before_sha256") != entry.get("state_after_sha256"),
                         f"{arm}: no state change in paired smoke")
                if arm != "PF-MEAN":
                    _require(float(entry["alpha_l1_after"]) > 0.0,
                             f"{arm}: alpha did not move during paired smoke")

    def _runtime_smoke_gate(self, records: list[dict[str, object]], elapsed_seconds: float) -> dict[str, object]:
        """GPU0-only throughput/VRAM gate after the exact 12 shared steps."""
        self._validate_smoke_records(records)
        _require(elapsed_seconds > 0.0, "postfusion smoke elapsed time is nonpositive")
        batches_per_epoch = len(self.train_loader)
        projected = elapsed_seconds / plan.SMOKE_SHARED_STEPS * batches_per_epoch * plan.SOURCE_EPOCHS_SCREEN
        payload: dict[str, object] = {
            "steps": plan.SMOKE_SHARED_STEPS, "elapsed_seconds": elapsed_seconds,
            "projected_screen_seconds": projected, "screen_hard_cap_seconds": plan.SCREEN_HARD_CAP_SECONDS,
            "projected_within_hard_cap": bool(projected <= plan.SCREEN_HARD_CAP_SECONDS),
        }
        _require(projected <= plan.SCREEN_HARD_CAP_SECONDS, "postfusion 12-epoch screen projection exceeds 180-minute cap")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            payload.update({
                "gpu0_free_bytes": int(free_bytes), "gpu0_total_bytes": int(total_bytes),
                "peak_allocated_bytes": int(peak_allocated), "peak_reserved_bytes": int(peak_reserved),
                "minimum_headroom_bytes": int(plan.MIN_FREE_VRAM_GIB * (1024 ** 3)),
            })
            _require(int(free_bytes) >= plan.MIN_FREE_VRAM_GIB * (1024 ** 3),
                     "postfusion GPU0 has less than 4 GiB physical VRAM headroom")
        return payload

    def source_heldin_in_sample_monitor_metric(self, arm: str) -> dict[str, object]:
        _require(self.monitor_loader is not None and arm in self.models, "source monitor metric lifecycle drift")
        model = self.models[arm]
        model.eval()
        grouped: dict[str, list[torch.Tensor]] = {}
        targets: dict[str, list[torch.Tensor]] = {}
        with torch.inference_mode():
            for raw_batch in self.monitor_loader:
                batch = _to_device(tuple(raw_batch), self.device)
                # The inherited monitor uses the full source block: pool-size
                # is a training exposure law, not a monitor hyperparameter.
                out = model.model_step(tuple(batch))
                names = out["session_name"]
                _require(len(set(names)) == 1, "source monitor batch spans session groups")
                name = str(names[0])
                grouped.setdefault(name, []).append(out["behavior_pred"].detach().cpu())
                targets.setdefault(name, []).append(out["behavior_target"].detach().cpu())
        per_session: dict[str, float] = {}
        for name in sorted(grouped):
            prediction = torch.cat(grouped[name], dim=0).to(torch.float64)
            target = torch.cat(targets[name], dim=0).to(torch.float64)
            residual = torch.sum((prediction - target) ** 2)
            total = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2)
            _require(float(total) > 0.0, f"source monitor target variance is zero: {name}")
            per_session[name] = float(1.0 - residual / total)
        monitor = self.source_authority["source_heldin_in_sample_monitor"]
        _require(tuple(sorted(per_session)) == tuple(monitor["monitor_groups"]),
                 "source held-in monitor roster drift")
        return {"per_session_r2": per_session, "equal_session_mean": selection.equal_group_mean(per_session)}

    def run_fixed_screen(self, *, smoke_gate: Any = None) -> dict[str, object]:
        _require(self.datamodule is not None, "prepare must precede postfusion screen")
        _require(self.train_loader is not None, "postfusion train loader is absent")
        started = time.monotonic()
        deadline = started + plan.SCREEN_HARD_CAP_SECONDS
        epoch1 = iter(self.train_loader)
        smoke_records: list[dict[str, object]] = []
        for _ in range(plan.SMOKE_SHARED_STEPS):
            try:
                raw_batch = next(epoch1)
            except StopIteration as error:
                raise RunnerError("source epoch ended before 12 paired smoke steps") from error
            smoke_records.append(self._step_group(tuple(raw_batch), detailed=True))
        smoke = self._runtime_smoke_gate(smoke_records, time.monotonic() - started) if smoke_gate is None else None
        if smoke_gate is not None:
            smoke_gate(smoke_records)
        # The exact SAME iterator continues, so the first 12 optimizer updates
        # are neither replayed nor skipped before source epoch one completes.
        first_epoch = self._finish_epoch(1, epoch1, completed_steps=plan.SMOKE_SHARED_STEPS,
                                         detailed_records=smoke_records, deadline=deadline)
        screen_epochs = [first_epoch] + [
            self.run_epoch(epoch, deadline=deadline)
            for epoch in range(2, plan.SOURCE_EPOCHS_SCREEN + 1)
        ]
        self._require_before_deadline(deadline)
        winner = selection.choose_screen_winner(self._epoch_rows)
        winner_arm = str(winner["winner"])
        final = selection.choose_screen_checkpoint(self._epoch_rows[winner_arm])
        best = self._best_source.get(winner_arm)
        _require(best is not None and int(best["epoch"]) == int(final["epoch"]),
                 "winner checkpoint/source selector drift")
        endpoint_slope_limitations: dict[str, dict[str, object]] = {}
        for arm in plan.ARMS:
            endpoint = self._epoch_rows[arm]
            _require(len(endpoint) == plan.SOURCE_EPOCHS_SCREEN
                     and int(endpoint[-2]["epoch"]) == plan.SOURCE_EPOCHS_SCREEN - 1
                     and int(endpoint[-1]["epoch"]) == plan.SOURCE_EPOCHS_SCREEN,
                     f"{arm}: endpoint slope epoch topology drift")
            slope = (float(endpoint[-1]["source_heldin_in_sample_monitor_mean"])
                     - float(endpoint[-2]["source_heldin_in_sample_monitor_mean"]))
            endpoint_slope_limitations[arm] = {
                "epoch11_to_epoch12_monitor_slope": slope,
                "possible_horizon_limitation": bool(slope > 0.0),
            }
        total_wall_seconds = time.monotonic() - started
        _require(total_wall_seconds > 0.0, "postfusion screen total wall time is nonpositive")
        runtime: dict[str, object] = {
            "total_wall_seconds": total_wall_seconds,
            "hard_cap_seconds": plan.SCREEN_HARD_CAP_SECONDS,
            "within_hard_cap": True,
            "shared_optimizer_step_groups": self.global_step,
            "shared_step_groups_per_second": self.global_step / total_wall_seconds,
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            runtime.update({
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
            })
        checkpoints: dict[str, dict[str, object]] = {}
        checkpoint_bodies: dict[str, bytes] = {}
        for arm in plan.ARMS:
            selected = selection.choose_screen_checkpoint(self._epoch_rows[arm])
            arm_best = self._best_source.get(arm)
            _require(arm_best is not None and int(arm_best["epoch"]) == int(selected["epoch"]),
                     f"{arm}: source-best checkpoint selector drift")
            checkpoints[arm] = {key: value for key, value in arm_best.items() if key != "body"}
            checkpoint_bodies[arm] = arm_best["body"]
        return {"source_authority": dict(self.source_authority), "screen_epochs": screen_epochs,
                "winner": winner, "selected_checkpoint": final,
                "endpoint_slope_limitations": endpoint_slope_limitations,
                "smoke": smoke, "runtime": runtime,
                "matched_prefusion_control_trained": False,
                "_source_best_checkpoint_bodies": checkpoint_bodies,
                "source_best_checkpoints": checkpoints}


def validate_gpu0_only(torch_module: Any) -> dict[str, object]:
    """Live-only physical-device check.  It never enumerates or queries GPU1."""
    import os

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "postfusion screen requires CUDA_VISIBLE_DEVICES=0")
    _require(torch_module.cuda.is_available() and torch_module.cuda.device_count() == 1,
             "postfusion screen needs exactly one visible GPU0")
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
    _require(int(free_bytes) >= plan.MIN_FREE_VRAM_GIB * (1024 ** 3),
             "postfusion GPU0 has less than 4 GiB physical headroom before source access")
    torch_module.cuda.reset_peak_memory_stats(0)
    return {"cuda_visible_devices": "0", "visible_device_count": 1, "torch_device": "cuda:0",
            "gpu0_free_bytes_pre_source": int(free_bytes), "gpu0_total_bytes": int(total_bytes),
            "minimum_headroom_bytes": int(plan.MIN_FREE_VRAM_GIB * (1024 ** 3))}


__all__ = ("RunnerError", "PostFusionVariantScreenRunner", "validate_gpu0_only")
