"""The M1 calibration-prefix operator and the stream recorders.

The operator is a training-gated ``forward_pre_hook(with_kwargs=True)`` on the
frozen ``SpintModel`` rewriting ``kwargs["calib_trialized_neural_features"] =
kwargs["calib_trialized_neural_features"][:, :M]``.  T0 registers the hook but
returns the kwargs unchanged (operator disabled, same runner); every eval-mode
forward is untouched and unrecorded by the operator.
"""
from __future__ import annotations

from typing import Any

from . import plan, schedule


class HookError(RuntimeError):
    """Fail closed for operator or recorder drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HookError(message)


class M1CalPrefixOperator:
    """Route-owned training-gated prefix operator for one arm of the pair."""

    def __init__(self, arm: str, record_steps: int = plan.RECORD_STEPS) -> None:
        _require(arm in plan.ARMS, f"m1 t0c1 unknown arm {arm!r}")
        _require(type(record_steps) is int and record_steps >= 0, "record_steps drift")
        self.arm = arm
        self.cycle = schedule.validate_cycle(plan.CYCLE)
        self.record_steps = int(record_steps)
        self.training_invocations = 0
        self.eval_invocations = 0
        self.effective_prefixes: list[int] = []
        self.records: list[dict[str, object]] = []
        self._handle: Any = None

    def m_for_index(self, index: int) -> int:
        return schedule.m_at(index, self.cycle)

    def forward_pre_hook(self, module, args, kwargs):
        """``register_forward_pre_hook(..., with_kwargs=True)`` target."""
        if not module.training:
            self.eval_invocations += 1
            return None
        index = self.training_invocations
        scheduled_m = self.m_for_index(index)
        self.training_invocations += 1
        calib = kwargs.get("calib_trialized_neural_features")
        if calib is None:
            if self.arm == "c1":
                raise HookError("C1 operator requires calib_trialized_neural_features in the forward kwargs")
            return None
        available = int(calib.shape[1])
        _require(available == 10, f"training calibration block must be the native M10, saw {available}")
        if self.arm == "c1":
            effective_m = schedule.effective_prefix_length(scheduled_m, available)
            visible = calib[:, :effective_m]
        else:
            effective_m = available
            visible = calib
        self.effective_prefixes.append(effective_m)
        if index < self.record_steps:
            import torch

            with torch.no_grad():
                self.records.append({
                    "step": index,
                    "arm": self.arm,
                    "scheduled_m": int(scheduled_m),
                    "effective_m": int(effective_m),
                    "operator_applied": self.arm == "c1",
                    "visible_slice_sha256": schedule.tensor_digest(
                        visible.detach().cpu().numpy()),
                    "full_block_sha256": schedule.tensor_digest(calib.detach().cpu().numpy()),
                })
        if self.arm == "c1":
            kwargs = dict(kwargs)
            kwargs["calib_trialized_neural_features"] = visible
            return args, kwargs
        return None

    def attach(self, model) -> Any:
        _require(self._handle is None, "operator already attached")
        self._handle = model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        return self._handle

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def snapshot(self) -> dict[str, object]:
        prefixes = [record["scheduled_m"] for record in self.records]
        return {
            "arm": self.arm,
            "cycle": list(self.cycle),
            "cycle_sha256": schedule.sequence_digest(self.cycle),
            "record_steps": self.record_steps,
            "training_invocations": self.training_invocations,
            "eval_invocations": self.eval_invocations,
            "n_recorded": len(self.records),
            "effective_prefixes": list(self.effective_prefixes),
            "effective_prefix_sequence_sha256": schedule.sequence_digest(self.effective_prefixes),
            "records": [dict(record) for record in self.records],
            "recorded_prefix_sequence": prefixes,
            "recorded_prefix_sequence_sha256": schedule.sequence_digest(prefixes),
        }


class TrainingStreamRecorder:
    """Per-training-step batch and Python-RNG state digests.

    The frozen ``SpintModel`` draws its whole-unit dropout probability from the
    Python global ``random`` stream (one ``random.uniform`` per training
    forward and nothing else consumes that stream inside a step), so equality
    of the per-step ``random.getstate()`` digests across arms IS equality of
    the dropout-p stream.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def record_step(self, step: int, *, before_state, after_state, sample_ids, loss: float) -> None:
        _require(type(step) is int and step >= 0 and isinstance(sample_ids, (list, tuple)),
                 "stream recorder row drift")
        self.rows.append({
            "step": step,
            "rng_state_before_sha256": schedule.rng_state_digest(before_state),
            "rng_state_after_sha256": schedule.rng_state_digest(after_state),
            "batch_sha256": schedule.batch_digest(list(sample_ids)),
            "loss": float(loss),
        })

    def payload(self, limit: int) -> dict[str, object]:
        _require(type(limit) is int and limit >= 0, "stream recorder limit drift")
        rows = [dict(row) for row in self.rows[:limit]]
        return {
            "recorded_steps": len(rows),
            "rows": rows,
            "rng_stream_digest": schedule.sequence_digest(
                [row["rng_state_after_sha256"] for row in rows]),
            "batch_stream_digest": schedule.sequence_digest(
                [row["batch_sha256"] for row in rows]),
        }


def eval_dropout_inactive_proof(model, *, device: str, window: int = 100, units: int = 64) -> dict[str, object]:
    """Prove eval-mode dropout inactivity: repeated forwards are bit-identical."""
    import numpy as np
    import torch

    _require(model.training is False, "eval proof requires model.eval()")
    rng_state = __import__("random").getstate()
    generator = torch.Generator().manual_seed(7)
    probe = torch.randn(3, window, units, generator=generator, dtype=torch.float32)
    calibration = torch.rand(3, 10, 1024, units, generator=generator, dtype=torch.float32)
    with torch.no_grad():
        first = model(probe.to(device), calib_trialized_neural_features=calibration.to(device))
        second = model(probe.to(device), calib_trialized_neural_features=calibration.to(device))
    first_digest = schedule.tensor_digest(first.detach().cpu().numpy())
    second_digest = schedule.tensor_digest(second.detach().cpu().numpy())
    _require(first_digest == second_digest, "eval forward is not deterministic (dropout active?)")
    return {
        "model_training_flag": False,
        "forward_1_sha256": first_digest,
        "forward_2_sha256": second_digest,
        "bit_identical": True,
        "dynamic_dropout_config": bool(getattr(model, "dynamic_dropout", False)),
        "dropout_inactive_in_eval": True,
    }


__all__ = (
    "HookError", "M1CalPrefixOperator", "TrainingStreamRecorder", "eval_dropout_inactive_proof",
)
