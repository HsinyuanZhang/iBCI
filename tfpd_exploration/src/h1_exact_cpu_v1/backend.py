"""Exact, isolated CPU backend for the trained V4 FULL temporal operator.

The parent stream retains all public API, calibration/session guards, frontend
window caching, native-unit conversion, and finite checks.  This module changes
only how the already-legal full-window temporal computation is dispatched.
It neither caches contextualized temporal state across calls nor changes W.
"""
from __future__ import annotations

import time
import torch
from torch import nn
from torch.nn import functional as F

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import (
    ExactFullWindowStream,
)
from .static_frontend_qonly import StaticCarrierQOnlyExactFullWindowStream


class ExactTemporalLastRow(nn.Module):
    """The reference E temporal calculation, factored into an nn.Module.

    This is intentionally a transcription of
    ``ExactFullWindowStream._full_window_last_hidden``.  Layers before the
    final layer still produce every position.  In the final layer only the
    final query/output/FFN rows are calculated, exactly as the reference E
    path does.  No output from a previous window is retained.
    """

    def __init__(self, temporal: nn.Module) -> None:
        super().__init__()
        self.temporal = temporal

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal
        hidden = z + temporal.pe[:z.size(1)].to(z).unsqueeze(0)
        for block in temporal.blocks[:-1]:
            hidden = block(hidden)
        block = temporal.blocks[-1]
        normalized = block.norm1(hidden)
        batch, width, dim = normalized.shape
        attention = block.attn
        qkv = attention.qkv(normalized).reshape(
            batch, width, 3, attention.n_heads, attention.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query[:, -1:].transpose(1, 2)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        result = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        result = attention.proj(result.transpose(1, 2).reshape(batch, 1, dim))
        last = hidden[:, -1:] + result
        return last + block.ffn(block.norm2(last))


class QOnlyTemporalLastRow(nn.Module):
    """Exact final-row temporal operator with unused final-layer Q rows omitted.

    This is the module form of
    :class:`StaticCarrierQOnlyExactFullWindowStream`'s existing eager override.
    In particular, it retains all K/V rows and all earlier temporal blocks; it
    only slices the final block's Q projection to the output row E consumes.
    """

    def __init__(self, temporal: nn.Module) -> None:
        super().__init__()
        self.temporal = temporal

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal
        hidden = z + temporal.pe[:z.size(1)].to(z).unsqueeze(0)
        for block in temporal.blocks[:-1]:
            hidden = block(hidden)
        block = temporal.blocks[-1]
        normalized = block.norm1(hidden)
        batch, width, dim = normalized.shape
        attention = block.attn
        weight, bias = attention.qkv.weight, attention.qkv.bias
        query = F.linear(normalized[:, -1:], weight[:dim], bias[:dim])
        kv = F.linear(normalized, weight[dim:], bias[dim:]).reshape(
            batch, width, 2, attention.n_heads, attention.head_dim
        )
        key, value = kv.unbind(dim=2)
        query = query.reshape(batch, 1, attention.n_heads, attention.head_dim).transpose(1, 2)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        result = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        result = attention.proj(result.transpose(1, 2).reshape(batch, 1, dim))
        last = hidden[:, -1:] + result
        return last + block.ffn(block.norm2(last))


class CompiledExactFullWindowStream(ExactFullWindowStream):
    """Exact E stream with a locally compiled temporal-last-row backend.

    ``torch.compile`` is deliberately optional so parity tests can exercise
    the algebraic backend on installations where an inductor backend is not
    available.  Compiling this module does not compile, replace, or mutate the
    frozen model; all parameters remain the same module objects.
    """

    def __init__(self, model, bank, *, compile_backend: bool = True, **kwargs) -> None:
        super().__init__(model, bank, **kwargs)
        self.exact_temporal = ExactTemporalLastRow(model.temporal)
        self.compile_backend = bool(compile_backend)
        self.compile_error: str | None = None
        self.compile_ms: float | None = None
        self.backend_kind = "eager"
        self._temporal_last = self.exact_temporal
        if self.compile_backend:
            self._compile_temporal_backend()

    def _compile_temporal_backend(self) -> None:
        # ``reduce-overhead`` is appropriate to the fixed B/W CPU stream.  The
        # first call remains an explicit cold cost in the isolated receipt.
        compile_failure = "torch.compile unavailable"
        if hasattr(torch, "compile"):
            started = time.perf_counter_ns()
            try:
                self._temporal_last = torch.compile(self.exact_temporal, mode="reduce-overhead")
                self.compile_ms = (time.perf_counter_ns() - started) / 1e6
                self.backend_kind = "torch_compile_reduce_overhead"
                return
            except Exception as error:  # environment capability, not model fallback
                compile_failure = f"{type(error).__name__}: {error}"
        # Trace is a static-shape CPU dispatch alternative.  It observes the
        # same current zero-padded [B,W,D] tensor that the parent stream has
        # just constructed; the module still owns the original parameters.
        started = time.perf_counter_ns()
        try:
            self._temporal_last = torch.jit.trace(
                self.exact_temporal, self.front.current, check_trace=True
            )
            self.compile_ms = (time.perf_counter_ns() - started) / 1e6
            self.backend_kind = "torchscript_trace"
            self.compile_error = compile_failure
        except Exception as error:
            self.compile_error = f"{compile_failure}; trace {type(error).__name__}: {error}"

    def _full_window_last_hidden(self) -> torch.Tensor:
        return self._temporal_last(self.front.current)


class CompiledStaticCarrierQOnlyExactFullWindowStream(
        StaticCarrierQOnlyExactFullWindowStream):
    """Static-carrier final-Q-only E with an optional compiled temporal call.

    The superclass continues to own public API validation, bank/model/mask
    mutation detection and static-carrier invalidation.  Compilation wraps only
    the pure current-window temporal operator; it introduces no cross-call
    temporal state.
    """

    def __init__(self, model, bank, *, compile_backend: bool = True, **kwargs) -> None:
        # ``_FiniteWindowStreamBase.__init__`` calls the virtual ``_rebuild``.
        # Set these sentinels before that parent initialization so the override
        # below is safe during the initial static-carrier rebuild.
        self.compile_backend = bool(compile_backend)
        self._compiled_backend_initialized = False
        super().__init__(model, bank, **kwargs)
        self.compile_error: str | None = None
        self.compile_ms: float | None = None
        self.backend_kind = "eager"
        self._compiled_backend_initialized = True
        self._refresh_temporal_backend()

    @torch.no_grad()
    def _rebuild(self, raw: torch.Tensor, reason: str) -> None:
        # The parent detects parameter/buffer/bank/roster/dtype/device changes
        # before calling this virtual method.  Rebuilding the compiled callable
        # for every such invalidation is deliberately conservative: a compiled
        # graph can never retain a replaced temporal submodule, parameter
        # object, device/dtype specialization, or a prior frontend shape.
        super()._rebuild(raw, reason)
        if self._compiled_backend_initialized:
            self._refresh_temporal_backend()

    def _refresh_temporal_backend(self) -> None:
        self.exact_temporal = QOnlyTemporalLastRow(self.model.temporal)
        self._temporal_last = self.exact_temporal
        self.compile_error = None
        self.compile_ms = None
        self.backend_kind = "eager"
        if self.compile_backend:
            self._compile_temporal_backend()

    def _compile_temporal_backend(self) -> None:
        compile_failure = "torch.compile unavailable"
        if hasattr(torch, "compile"):
            started = time.perf_counter_ns()
            try:
                self._temporal_last = torch.compile(self.exact_temporal, mode="reduce-overhead")
                self.compile_ms = (time.perf_counter_ns() - started) / 1e6
                self.backend_kind = "torch_compile_reduce_overhead"
                return
            except Exception as error:
                compile_failure = f"{type(error).__name__}: {error}"
        started = time.perf_counter_ns()
        try:
            self._temporal_last = torch.jit.trace(
                self.exact_temporal, self.front.current, check_trace=True
            )
            self.compile_ms = (time.perf_counter_ns() - started) / 1e6
            self.backend_kind = "torchscript_trace"
            self.compile_error = compile_failure
        except Exception as error:
            self.compile_error = f"{compile_failure}; trace {type(error).__name__}: {error}"

    def _full_window_last_hidden(self) -> torch.Tensor:
        return self._temporal_last(self.front.current)
