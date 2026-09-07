"""The CAL-AUG prefix operator: a training-gated forward-pre-hook.

Work order section 2 (the seam, frozen bytes stay authoritative):

    a ``forward_pre_hook(with_kwargs=True)`` on the ``StreamingSpintModel``
    rewrites ``kwargs["calib_trials"] = kwargs["calib_trials"][:, :M]``,
    gated on ``module.training`` so every eval/diagnostic forward is untouched.

Laws implemented here and proven by tests (no data, no CUDA):

* **Training-gated** — the counter and the slice apply ONLY when
  ``module.training`` is True; eval-mode invocations return ``None`` (the
  kwargs dict object itself is untouched, tensor identity preserved).
* **Deterministic** — ``M`` is pure integer arithmetic on the internal
  counter (``schedule.m_at``); no Python/NumPy/Torch RNG is consumed.
* **T0 identity** — with ``arm="t0"`` the hook is registered but returns the
  kwargs unchanged; the counter still advances (the schedule is auditable) and
  the effective prefix is recorded as the full 30-trial block.
* **One counter, training only** — the counter increments exactly once per
  training-mode invocation, never for eval forwards.

Also here: the read-only query-neural digest hook (pre-dropout input evidence
for the smoke) and the encoder ``push_trial`` probe that proves the B3S
encoder's ``trial_count`` equals the declared ``M`` at every step.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import schedule


class HookError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HookError(message)


class CalPrefixOperator:
    """Route-owned training-gated prefix operator for one arm of the pair.

    ``arm``: ``"c1"`` rewrites ``kwargs["calib_trials"]`` to the scheduled
    chronological prefix; ``"t0"`` returns the kwargs unchanged (operator
    disabled, same runner).  ``record_steps`` bounds the per-step recording
    (scheduled/effective ``M`` plus digests of the visible calibration rows);
    the counter itself always advances.
    """

    def __init__(self, arm: str, cycle, record_steps: int = 200):
        _require(arm in ("t0", "c1"), f"unknown arm {arm!r}")
        self.arm = arm
        self.cycle = schedule.validate_cycle(cycle)
        self.record_steps = int(record_steps)
        self.training_invocations = 0
        self.eval_invocations = 0
        self.records: list[dict] = []

    # -- schedule ----------------------------------------------------------
    def m_for_index(self, index: int) -> int:
        return schedule.m_at(index, self.cycle)

    # -- the hook ----------------------------------------------------------
    def forward_pre_hook(self, module, args, kwargs):
        """``register_forward_pre_hook(..., with_kwargs=True)`` target."""
        if not module.training:
            self.eval_invocations += 1
            return None  # eval/diagnostic forward: untouched, unrecorded

        index = self.training_invocations
        scheduled_m = self.m_for_index(index)
        self.training_invocations += 1

        calib = kwargs.get("calib_trials")
        if calib is None:
            if self.arm == "c1":
                raise HookError(
                    "C1 operator requires calib_trials in the training forward kwargs"
                )
            return None

        available = int(calib.shape[1])
        if self.arm == "c1":
            effective_m = schedule.effective_prefix_length(scheduled_m, available)
            visible = calib[:, :effective_m]
        else:
            # T0: the operator is DISABLED — return None so the forward sees
            # the caller's original args/kwargs objects, exactly as with no
            # hook registered; the effective prefix is the full block.
            effective_m = available
            visible = calib

        if index < self.record_steps:
            with torch.no_grad():
                self.records.append(
                    {
                        "step": index,
                        "arm": self.arm,
                        "scheduled_m": int(scheduled_m),
                        "effective_m": int(effective_m),
                        "operator_applied": self.arm == "c1",
                        "visible_slice_sha256": schedule.visible_slice_digest(visible),
                        "full_block_sha256": schedule.visible_slice_digest(calib),
                    }
                )
        if self.arm == "c1":
            kwargs = dict(kwargs)  # never mutate the caller's dict
            kwargs["calib_trials"] = visible
            return args, kwargs
        return None

    # -- lifecycle ---------------------------------------------------------
    def attach(self, model):
        handle = model.register_forward_pre_hook(self.forward_pre_hook, with_kwargs=True)
        self._handle = handle
        return handle

    def detach(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.remove()
            self._handle = None

    # -- receipts ----------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "arm": self.arm,
            "cycle": list(self.cycle),
            "cycle_sha256": schedule.cycle_digest(self.cycle),
            "record_steps": self.record_steps,
            "training_invocations": self.training_invocations,
            "eval_invocations": self.eval_invocations,
            "n_recorded": len(self.records),
            "records": [dict(record) for record in self.records],
            "recorded_prefix_sequence": [record["scheduled_m"] for record in self.records],
            "recorded_prefix_sequence_sha256": schedule.sequence_digest(
                [record["scheduled_m"] for record in self.records]
            ),
        }


def make_read_only_neural_digest_hook(store: list, limit: Optional[int] = None):
    """Read-only pre-hook recording sha256 of the query-neural bytes per call.

    The hook fires BEFORE the model body (hence pre-dropout: the Cell-D unit
    dropout is applied inside ``decode_with_identity``).  It returns ``None``
    so both args and kwargs pass through untouched.  Because T0/C1 query
    activity is identical, these digests must be equal across arms.
    """

    def hook(module, args, kwargs):
        neural = kwargs.get("neural")
        if neural is None and args:
            neural = args[0]
        if neural is None:
            return None
        if limit is not None and len(store) >= int(limit):
            return None
        with torch.no_grad():
            store.append(
                {
                    "call": len(store),
                    "training": bool(module.training),
                    "neural_sha256": schedule.tensor_digest(neural),
                }
            )
        return None

    return hook


class EncoderTrialProbe:
    """Counts the B3S encoder's ``push_trial`` calls (= its ``trial_count``).

    ``SideFeatureEarlyPoolEncoder.forward_batch`` loops ``push_trial`` once per
    calibration trial and accumulates ``state["trial_count"]``, so the number
    of ``push_trial`` calls in one forward IS the trial count the identity
    consumed.  The probe wraps the bound method on the instance only (the same
    route-owned instrumentation pattern as ``pop_robust.per_head_attention_
    summary``); no sealed file is touched and the wrapper is fully removed on
    close.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.count = 0
        self.calls_per_forward: list[int] = []
        self._open = False

    def __enter__(self) -> "EncoderTrialProbe":
        _require(not self._open, "encoder trial probe already open")
        original = self.encoder.push_trial
        original_forward_batch = self.encoder.forward_batch

        def counting_push_trial(state, trial, *rest, **kwargs):
            self.count += 1
            return original(state, trial, *rest, **kwargs)

        def counting_forward_batch(*args, **kwargs):
            start = self.count
            output = original_forward_batch(*args, **kwargs)
            self.calls_per_forward.append(self.count - start)
            return output

        self._patched_original = original
        self.encoder.push_trial = counting_push_trial
        self.encoder.forward_batch = counting_forward_batch
        self._open = True
        return self

    def __exit__(self, *exc_info) -> None:
        if self._open:
            self.encoder.__dict__.pop("push_trial", None)
            self.encoder.__dict__.pop("forward_batch", None)
            self._open = False

    def mark_forward_boundary(self) -> int:
        """Trials consumed since the last boundary (one training forward)."""
        previous = getattr(self, "_last_mark", 0)
        self._last_mark = self.count
        delta = self.count - previous
        self.calls_per_forward.append(delta)
        return delta
