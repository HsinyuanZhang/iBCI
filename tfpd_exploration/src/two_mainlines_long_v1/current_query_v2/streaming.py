"""Model-level finite-window streaming for H1/M1 current-query decoders.

Inputs are one binned neural observation, in exactly the model training input
space. This layer does not smooth, normalize, drop masked time bins, or infer
session boundaries. A trial ``on_done`` is not a session reset by default.
Startup consists of W-1 raw zero bins passed through the actual biased frontend.
"""
from __future__ import annotations

import math
from typing import Hashable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..latency_opt_v2 import FrontendWindowCache
from .core import QueryMemoryCache, QueryTemporalStack


def _tensor_token(value: torch.Tensor) -> tuple:
    # In-place optimizer/copy_ changes are detected. Mutating through .data or a
    # NumPy alias bypasses PyTorch versioning and is not a supported update API.
    try:
        version = value._version
    except RuntimeError:
        # Banks copied to GPU inside inference_mode have no version counter.
        # The stream snapshots/compares those tensors explicitly below.
        version = None
    return (value.data_ptr(), version, tuple(value.shape), value.dtype, value.device)


class _FiniteWindowStreamBase:
    """Independent stream(s) sharing one bank, with explicit session identity.

    A batch may contain independent trajectories ONLY with the same calibration
    bank. Different sessions use different instances; there is no hidden global
    cache. Returned values use native task units (H1 /20, M1 /1).
    """

    def __init__(self, model, bank, *, task: str, session_id: Hashable,
                 unit_ids: Sequence[Hashable], batch_size: int = 1,
                 prediction_divisor: float | None = None, kernel: int = 5,
                 temporal_mode: str):
        if task not in {"h1", "m1"}:
            raise ValueError("task must be h1 or m1")
        if temporal_mode == "current_query":
            if not isinstance(model.temporal, QueryTemporalStack):
                raise TypeError("current-query memory cache requires QueryTemporalStack")
        elif temporal_mode == "full_window_exact":
            if isinstance(model.temporal, QueryTemporalStack) or not hasattr(model.temporal, "pe"):
                raise TypeError("exact full-window path requires the matched original temporal stack")
        else:
            raise ValueError("unrecognized temporal mode")
        self.temporal_mode = temporal_mode
        expected = 20.0 if task == "h1" else 1.0
        divisor = expected if prediction_divisor is None else float(prediction_divisor)
        if not math.isfinite(divisor) or divisor != expected:
            raise ValueError(f"{task} V2 output contract requires divisor {expected}")
        self.model, self.bank, self.task = model, bank, task
        if any(_tensor_token(p)[1] is None for p in model.parameters()):
            raise ValueError("construct/load model parameters outside torch.inference_mode()")
        self.divisor, self.batch_size = divisor, int(batch_size)
        if self.batch_size < 1:
            raise ValueError("positive batch_size required")
        self.session_id, self.unit_ids = session_id, tuple(unit_ids)
        self.window = int(model.cfg.window)
        if kernel != getattr(model.cfg, "conv_kernel", kernel):
            raise ValueError("cache kernel must match the frontend receptive field")
        self.front = FrontendWindowCache(self._frontend, window=self.window, kernel=kernel)
        self.memory = QueryMemoryCache(model.temporal) if temporal_mode == "current_query" else None
        self.token = None
        self.n_observations = 0
        self.last_rebuild_reason = None
        self.versionless_snapshots = []
        self.reset(bank=bank, session_id=session_id, unit_ids=unit_ids)

    def _require_eval(self):
        if self.model.training:
            raise RuntimeError("CurrentQueryStream requires model.eval()")

    def _parameter(self):
        return next(self.model.parameters())

    def _validate_bank(self):
        n = len(self.unit_ids)
        if n == 0 or len(set(self.unit_ids)) != n:
            raise ValueError("unit_ids must be nonempty and unique")
        if self.bank.E0.shape[-2] != n or self.bank.T.shape[-2] != n:
            raise ValueError("bank and explicit unit roster disagree")
        if self.bank.unit_mask.shape[-1] != n:
            raise ValueError("bank mask and unit roster disagree")
        if self.bank.unit_mask.ndim not in {1, 2}:
            raise ValueError("unit mask must be [N] or [B,N]")
        if self.bank.unit_mask.ndim == 2 and self.bank.unit_mask.shape[0] != self.batch_size:
            raise ValueError("bank mask batch mismatch")
        if not bool(self.bank.unit_mask.bool().any(dim=-1).all()):
            raise ValueError("at least one unit must be present in each stream")

    def _state_token(self):
        return (
            tuple((name, _tensor_token(p)) for name, p in self.model.named_parameters()),
            tuple((name, _tensor_token(b)) for name, b in self.model.named_buffers()),
            _tensor_token(self.bank.E0), _tensor_token(self.bank.T),
            _tensor_token(self.bank.unit_mask), self.unit_ids,
            float(getattr(self.model, "activity_scale", 1.0)), self.divisor,
        )

    def _bank_and_buffers(self):
        return (self.bank.E0, self.bank.T, self.bank.unit_mask, *self.model.buffers())

    def _frontend(self, x):
        if self.task == "h1":
            encode = getattr(self.model, "encode_frontend", None)
            if encode is None:
                encode = self.model._z
            return encode(x, self.bank)
        return self.model._fuse(x, self.bank, self.bank.unit_mask.to(x.device))

    @torch.no_grad()
    def _rebuild(self, raw, reason):
        self._require_eval()
        self._validate_bank()
        ref = self._parameter()
        raw = raw.to(device=ref.device, dtype=ref.dtype)
        self.front.rebuild(raw)
        if self.memory is not None:
            self.memory.reset(self.front.current, reason=reason)
        self.token = self._state_token()
        self.versionless_snapshots = [(t, t.detach().clone()) for t in self._bank_and_buffers()
                                      if _tensor_token(t)[1] is None]
        self.last_rebuild_reason = reason

    @torch.no_grad()
    def reset(self, *, bank=None, session_id=None, unit_ids=None, history=None):
        """Start a session; optional history is already in training input units.

        Short histories are LEFT padded in raw input space. Rebinding a bank or
        roster does not silently carry observations from the preceding session.
        """
        if bank is not None:
            self.bank = bank
        if session_id is not None:
            self.session_id = session_id
        if unit_ids is not None:
            self.unit_ids = tuple(unit_ids)
        self._validate_bank()
        ref = self._parameter()
        raw = torch.zeros(self.batch_size, self.window, len(self.unit_ids),
                          device=ref.device, dtype=ref.dtype)
        count = 0
        if history is not None:
            source = torch.as_tensor(history, device=ref.device, dtype=ref.dtype)
            if source.ndim != 3 or source.shape[0] != self.batch_size or source.shape[2] != len(self.unit_ids):
                raise ValueError("history must be [B,T,N] with the explicit roster")
            if not bool(torch.isfinite(source).all()):
                raise ValueError("history contains nonfinite inputs")
            count = min(self.window, source.shape[1])
            if count:
                raw[:, -count:] = source[:, -count:]
        self.n_observations = count
        self._rebuild(raw, "session_reset")

    @torch.no_grad()
    def set_bank(self, bank):
        """Recalibrate the SAME roster/session, preserving its raw observations.

        Roster/order changes must instead use reset(unit_ids=..., bank=...).
        """
        self.bank = bank
        self._rebuild(self.front.raw, "bank_rebound")

    @torch.no_grad()
    def _ensure_current(self):
        self._require_eval()
        if (self.token != self._state_token()
                or any(not torch.equal(t, snapshot) for t, snapshot in self.versionless_snapshots)):
            self._rebuild(self.front.raw, "model_bank_or_mask_changed")

    @torch.no_grad()
    def observe(self, observation, *, session_id=None):
        """Advance every observed bin, even when its behavior scoring mask is false."""
        if session_id is not None and session_id != self.session_id:
            raise ValueError("session changed; explicit reset is required")
        self._ensure_current()
        ref = self._parameter()
        x = torch.as_tensor(observation, device=ref.device, dtype=ref.dtype)
        if x.ndim != 2 or x.shape != (self.batch_size, len(self.unit_ids)):
            raise ValueError("one observation must be [B,N]")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("observation contains nonfinite inputs")
        self.front.advance(x[:, None, :])
        if self.memory is not None:
            self.memory.advance(self.front.current, self.front.changed_indices)
        self.n_observations += 1

    @torch.no_grad()
    def current_prediction(self) -> torch.Tensor:
        self._ensure_current()
        hidden = self.memory.predict() if self.memory is not None else self._full_window_last_hidden()
        prediction = self.model.readout(self.model.final_norm(hidden))[:, -1] / self.divisor
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("nonfinite prediction")
        return prediction

    @torch.no_grad()
    def predict(self, observation, *, session_id=None) -> np.ndarray:
        self.observe(observation, session_id=session_id)
        # Conversion/host transfer is deliberately within the timed public API.
        return self.current_prediction().detach().cpu().numpy().astype(np.float32, copy=True)

    def on_done(self, *, reset_session: bool = False):
        """Trial boundaries retain finite history; an explicit session reset clears it."""
        if reset_session:
            self.reset()

    @property
    def state_bytes(self):
        return (self.memory.state_bytes if self.memory is not None else 0) + sum(
            t.numel() * t.element_size() for t in (self.front.raw, self.front.current))

    def _full_window_last_hidden(self):
        """Exact E control: no temporal cross-window KV is ever retained."""
        z = self.front.current
        temporal = self.model.temporal
        hidden = z + temporal.pe[:z.size(1)].to(z).unsqueeze(0)
        for block in temporal.blocks[:-1]:
            hidden = block(hidden)
        block = temporal.blocks[-1]
        normalized = block.norm1(hidden)
        b, w, d = normalized.shape
        attention = block.attn
        qkv = attention.qkv(normalized).reshape(b, w, 3, attention.n_heads, attention.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q[:, -1:].transpose(1, 2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        # Query is the final position, so every finite-window key is legal.
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
        result = attention.proj(result.transpose(1, 2).reshape(b, 1, d))
        last = hidden[:, -1:] + result
        return last + block.ffn(block.norm2(last))


class CurrentQueryStream(_FiniteWindowStreamBase):
    """New T operator with boundary-correct frontend and independent-memory KV reuse."""
    def __init__(self, model, bank, **kwargs):
        super().__init__(model, bank, temporal_mode="current_query", **kwargs)


class ExactFullWindowStream(_FiniteWindowStreamBase):
    """Equivalent E execution of a trained FULL control, with no temporal KV reuse.

    Only frontend outputs survive a window shift. Temporal layers 1..L-1 are
    fully recomputed, and only final-layer query/output/FFN rows are sliced.
    """
    def __init__(self, model, bank, **kwargs):
        super().__init__(model, bank, temporal_mode="full_window_exact", **kwargs)
