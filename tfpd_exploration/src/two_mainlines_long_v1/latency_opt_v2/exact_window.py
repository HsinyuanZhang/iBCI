"""Safe reusable pieces for fixed-window, last-output inference.

The adapter owns *frontend* outputs only.  It never reuses self-attention KV
between shifted windows.  On a one-bin shift a causal-k frontend has exactly
five changed output positions: 0..k-2 (new zero-padded left edge) and W-1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


Frontend = Callable[[torch.Tensor], torch.Tensor]
TemporalLast = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class WindowCacheState:
    raw: torch.Tensor
    frontend: torch.Tensor


class ExactWindowInferenceAdapter:
    """Cache a local causal frontend while recomputing temporal history exactly.

    ``frontend`` must be time-separable after its causal receptive field and
    preserve a leading-zero padding contract.  ``temporal_last`` receives the
    complete frontend sequence; it is intentionally recomputed every call.
    """

    def __init__(self, frontend: Frontend, temporal_last: TemporalLast, *, window: int, kernel: int = 5):
        if window < kernel or kernel < 2:
            raise ValueError("require window >= kernel >= 2 (kernel=1 has no left-boundary cache path)")
        self.frontend, self.temporal_last = frontend, temporal_last
        self.window, self.kernel = int(window), int(kernel)
        self.state: WindowCacheState | None = None

    def reset(self) -> None:
        self.state = None

    def rebuild(self, raw_window: torch.Tensor) -> torch.Tensor:
        if raw_window.ndim != 3 or raw_window.size(1) != self.window:
            raise ValueError(f"expected [B,{self.window},N] window, got {tuple(raw_window.shape)}")
        z = self.frontend(raw_window)
        self.state = WindowCacheState(raw=raw_window.detach().clone(), frontend=z.detach().clone())
        return self.temporal_last(z)

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("advance requires rebuild after reset or invalidation")
        old = self.state
        if next_bin.ndim != 3 or next_bin.shape[1:] != (1, old.raw.size(2)):
            raise ValueError("next_bin must be [B,1,N] with the compiled unit roster")
        raw = torch.cat((old.raw[:, 1:], next_bin), dim=1)
        left = self.frontend(raw[:, : self.kernel - 1])
        right = self.frontend(raw[:, -self.kernel :])[:, -1:]
        z = torch.cat((left, old.frontend[:, self.kernel :], right), dim=1)
        self.state = WindowCacheState(raw=raw.detach().clone(), frontend=z.detach().clone())
        return self.temporal_last(z)


class FrontendWindowCache:
    """Reusable, temporal-model-agnostic causal frontend sliding cache.

    ``frontend`` maps ``[B,T,N]`` to ``[B,T,D]`` and may encapsulate a bank.
    After ``advance`` its ``current`` property is the full, correctly indexed
    z-window.  The exposed changed indices make it safe for a query-memory
    stack to refresh only K/V entries 0..k-2 and W-1.  This class makes no
    assertion about temporal transformer caches because it has none.
    """

    def __init__(self, frontend: Frontend, *, window: int, kernel: int = 5):
        if window < kernel or kernel < 2:
            raise ValueError("require window >= kernel >= 2 (kernel=1 has no left-boundary cache path)")
        self.frontend, self.window, self.kernel = frontend, int(window), int(kernel)
        self.state: WindowCacheState | None = None
        self.changed_indices: tuple[int, ...] = tuple(range(window))

    @property
    def current(self) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("cache is empty; call rebuild")
        return self.state.frontend

    @property
    def raw(self) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("cache is empty; call rebuild")
        return self.state.raw

    def reset(self) -> None:
        self.state = None
        self.changed_indices = tuple(range(self.window))

    def rebuild(self, raw_window: torch.Tensor) -> torch.Tensor:
        if raw_window.ndim != 3 or raw_window.size(1) != self.window:
            raise ValueError(f"expected [B,{self.window},N] window, got {tuple(raw_window.shape)}")
        z = self.frontend(raw_window)
        self.state = WindowCacheState(raw=raw_window.detach().clone(), frontend=z.detach().clone())
        self.changed_indices = tuple(range(self.window))
        return z

    def advance(self, next_bin: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("advance requires rebuild after reset or invalidation")
        old = self.state
        if next_bin.ndim != 3 or next_bin.shape[1:] != (1, old.raw.size(2)):
            raise ValueError("next_bin must be [B,1,N] with the compiled unit roster")
        raw = torch.cat((old.raw[:, 1:], next_bin), dim=1)
        left = self.frontend(raw[:, : self.kernel - 1])
        right = self.frontend(raw[:, -self.kernel :])[:, -1:]
        z = torch.cat((left, old.frontend[:, self.kernel :], right), dim=1)
        self.state = WindowCacheState(raw=raw.detach().clone(), frontend=z.detach().clone())
        self.changed_indices = tuple(range(self.kernel - 1)) + (self.window - 1,)
        return z
