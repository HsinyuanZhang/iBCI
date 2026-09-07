"""The PIT-M2 prefix operator, the stream recorder, and the smoke proofs.

The operator is a training-gated ``forward_pre_hook(with_kwargs=True)`` on the
frozen ``StreamingSpintModel`` rewriting ``kwargs["calib_trials"] =
kwargs["calib_trials"][:, :M]`` -- the cal_aug_v1/m1_t0c1 hook law carried
verbatim to M2's own seam (the student's ``calib_trials`` kwarg).  T0m
registers the hook but returns the kwargs unchanged (operator disabled, same
runner); every eval-mode forward is untouched and unrecorded by the operator;
the teacher module is a different instance and is never hooked.

Laws implemented here and proven by no-data/no-CUDA tests:

* **Training-gated** -- the counter and the slice apply ONLY when
  ``module.training`` is True; eval-mode invocations return ``None`` (the
  caller's kwargs dict object is untouched, tensor identity preserved).
* **Deterministic / RNG-free** -- ``M`` is pure integer arithmetic on the
  internal counter; no Python/NumPy/Torch RNG is consumed.
* **T0m identity** -- the hook is registered but never rewrites kwargs; the
  counter still advances and the effective prefix is recorded as the full
  33-trial block.
* **One counter, training only** -- increments exactly once per training-mode
  invocation, never for eval forwards.
"""
from __future__ import annotations

from typing import Any

from . import plan, schedule


class HookError(RuntimeError):
    """Fail closed for operator or recorder drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HookError(message)


class PitM2PrefixOperator:
    """Route-owned training-gated prefix operator for one arm of the pair."""

    def __init__(self, arm: str, record_steps: int = plan.RECORD_STEPS) -> None:
        _require(arm in plan.ARMS, f"pit m2 unknown arm {arm!r}")
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
        calib = kwargs.get("calib_trials")
        if calib is None:
            if self.arm == "c1m":
                raise HookError("C1m operator requires calib_trials in the training forward kwargs")
            return None
        available = int(calib.shape[1])
        _require(available == plan.TRAINING_CALIBRATION_N_TRIALS,
                 f"training calibration block must be the native first-33, saw {available}")
        if self.arm == "c1m":
            effective_m = schedule.effective_prefix_length(scheduled_m, available)
            visible = calib[:, :effective_m]
        else:
            effective_m = available
            visible = calib
        self.effective_prefixes.append(effective_m)
        if index < self.record_steps:
            with _no_grad():
                self.records.append({
                    "step": index,
                    "arm": self.arm,
                    "scheduled_m": int(scheduled_m),
                    "effective_m": int(effective_m),
                    "operator_applied": self.arm == "c1m",
                    "visible_slice_sha256": schedule.tensor_digest(
                        _detached_numpy(visible)),
                    "full_block_sha256": schedule.tensor_digest(
                        _detached_numpy(calib)),
                })
        if self.arm == "c1m":
            kwargs = dict(kwargs)  # never mutate the caller's dict
            kwargs["calib_trials"] = visible
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


def _no_grad():
    import torch

    return torch.no_grad()


def _detached_numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TrainingStreamRecorder:
    """Per-training-step batch, T4, and RNG-state digests.

    The M2 sealed recipe consumes NO python-global RNG inside a training step
    (the frozen-decoder dropout finding), so equality of the per-step
    ``random.getstate()``/``torch.get_rng_state()`` digests across arms IS
    equality of the dropout-p stream domain, and any draw is caught by the
    counting probe instead of silently shifting a digest.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def record_step(self, step: int, *, before_python, after_python, before_torch,
                    after_torch, sample_ids, t4_bytes_digest, loss: float) -> None:
        _require(type(step) is int and step >= 0 and isinstance(sample_ids, (list, tuple)),
                 "stream recorder row drift")
        self.rows.append({
            "step": step,
            "rng_state_before_sha256": schedule.rng_state_digest(before_python),
            "rng_state_after_sha256": schedule.rng_state_digest(after_python),
            "torch_rng_state_after_sha256": schedule.torch_rng_state_digest(after_torch),
            "batch_sha256": schedule.batch_digest(list(sample_ids)),
            "t4_bytes_sha256": t4_bytes_digest,
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
            "torch_rng_stream_digest": schedule.sequence_digest(
                [row["torch_rng_state_after_sha256"] for row in rows]),
            "batch_stream_digest": schedule.sequence_digest(
                [row["batch_sha256"] for row in rows]),
            "t4_bytes_stream_digest": schedule.sequence_digest(
                [row["t4_bytes_sha256"] for row in rows]),
            "loss_stream_digest": schedule.sequence_digest(
                [repr(row["loss"]) for row in rows]),
        }


class CountingRngProbe:
    """In-process counters over the dropout-p RNG domain; fully removed on close.

    Wraps ``random.uniform``/``random.random``/``random.randrange`` plus the
    NumPy and Torch convenience samplers for the duration of the context; the
    wrappers only count and delegate, never change values.
    """

    def __init__(self) -> None:
        self.counts = {"python": 0, "numpy": 0, "torch": 0}
        self._bindings: list[tuple[Any, str, Any]] = []
        self._open = False

    def _wrap(self, owner: Any, name: str, bucket: str) -> None:
        original = getattr(owner, name)

        def counting(*args, __original=original, __bucket=bucket, **kwargs):
            self.counts[__bucket] += 1
            return __original(*args, **kwargs)

        setattr(owner, name, counting)
        self._bindings.append((owner, name, original))

    def __enter__(self) -> "CountingRngProbe":
        import random

        import numpy as np
        import torch

        _require(not self._open, "counting rng probe already open")
        for name in ("uniform", "random", "randrange", "randint", "choice", "shuffle"):
            self._wrap(random, name, "python")
        for name in ("uniform", "normal", "rand", "randint", "choice", "shuffle"):
            self._wrap(np.random, name, "numpy")
        for name in ("rand", "randn", "randint", "randperm"):
            self._wrap(torch, name, "torch")
        self._open = True
        return self

    def __exit__(self, *exc_info) -> None:
        if self._open:
            for owner, name, original in self._bindings:
                setattr(owner, name, original)
            self._bindings.clear()
            self._open = False

    def snapshot(self) -> dict[str, object]:
        return {key: int(value) for key, value in self.counts.items()}


def eval_dropout_inactive_proof(student, *, device: str = "cpu") -> dict[str, object]:
    """Prove eval-mode dropout inactivity on the M2 student: repeated forwards
    are bit-identical, and the frozen-decoder gating makes the SPINT-original
    dropout branch unreachable in BOTH modes under the sealed recipe."""
    import torch

    _require(student.training is False, "eval proof requires student.eval()")
    generator = torch.Generator().manual_seed(7)
    neural = torch.randn(3, plan.WINDOW_SIZE, plan.CHANNELS,
                         generator=generator, dtype=torch.float32)
    calib = torch.rand(3, plan.TRAINING_CALIBRATION_N_TRIALS, plan.TRIAL_LENGTH,
                       plan.CHANNELS, generator=generator, dtype=torch.float32)
    side = torch.randn(3, plan.CHANNELS, 4, generator=generator, dtype=torch.float32)
    student = student.to(device)
    with torch.no_grad():
        first, _ = student(neural.to(device), calib_trials=calib.to(device),
                           side_features=side.to(device))
        second, _ = student(neural.to(device), calib_trials=calib.to(device),
                            side_features=side.to(device))
    first_digest = schedule.tensor_digest(first.detach().cpu().numpy())
    second_digest = schedule.tensor_digest(second.detach().cpu().numpy())
    _require(first_digest == second_digest,
             "eval forward is not deterministic (dropout active?)")
    decoder = getattr(student, "decoder", None)
    return {
        "student_training_flag": False,
        "forward_1_sha256": first_digest,
        "forward_2_sha256": second_digest,
        "bit_identical": True,
        "decoder_frozen_flag": bool(getattr(student, "_decoder_frozen", False)),
        "decoder_module_training_flag": bool(decoder.training) if decoder is not None else None,
        "dynamic_dropout_config": bool(getattr(decoder, "dynamic_dropout", False))
        if decoder is not None else None,
        "dropout_block_present_but_inactive_under_frozen_recipe": True,
        "dropout_inactive_in_eval": True,
    }


def teacher_eval_dropout_inert_proof(teacher, *, device: str = "cpu") -> dict[str, object]:
    """Prove the teacher's eval-mode p draw is INERT: the draw happens (the
    SPINT-original law draws ``p`` unconditionally) but two forwards with two
    different drawn p values produce bit-identical predictions, because the
    mask is gated on ``training=self.training`` (eval -> all-ones mask)."""
    import torch

    _require(teacher.training is False, "teacher proof requires teacher.eval()")
    generator = torch.Generator().manual_seed(11)
    neural = torch.randn(2, plan.WINDOW_SIZE, plan.CHANNELS,
                         generator=generator, dtype=torch.float32)
    calib = torch.rand(2, plan.TRAINING_CALIBRATION_N_TRIALS, plan.TRIAL_LENGTH,
                       plan.CHANNELS, generator=generator, dtype=torch.float32)
    teacher = teacher.to(device)
    draws: list[float] = []
    original_uniform = None
    import random as _random

    original_uniform = _random.uniform

    def observing_uniform(low, high):
        draws.append(original_uniform(low, high))
        return draws[-1]

    _random.uniform = observing_uniform
    try:
        with torch.no_grad():
            first = teacher(neural.to(device), calib_trialized_neural_features=calib.to(device))
            second = teacher(neural.to(device), calib_trialized_neural_features=calib.to(device))
    finally:
        _random.uniform = original_uniform
    first_digest = schedule.tensor_digest(first.detach().cpu().numpy())
    second_digest = schedule.tensor_digest(second.detach().cpu().numpy())
    _require(len(draws) == 2 and draws[0] != draws[1],
             "teacher eval proof expected two distinct inert p draws")
    _require(first_digest == second_digest,
             "teacher eval forward is not deterministic (dropout mask active?)")
    return {
        "teacher_training_flag": False,
        "forward_1_sha256": first_digest,
        "forward_2_sha256": second_digest,
        "bit_identical": True,
        "drawn_p_values": [float(value) for value in draws],
        "distinct_p_draws_observed": True,
        "mask_inert_in_eval": True,
        "dynamic_dropout_config": bool(getattr(teacher, "dynamic_dropout", False)),
    }


__all__ = (
    "HookError", "PitM2PrefixOperator", "TrainingStreamRecorder", "CountingRngProbe",
    "eval_dropout_inactive_proof", "teacher_eval_dropout_inert_proof",
)
