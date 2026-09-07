"""Additive CPU-only Stage-0 core for Causal Dual-Memory Cell D (CDM-D).

The public package initializer intentionally imports no numerical or model
runtime.  The static CLI imports only :mod:`plan`; callers that need the
synthetic core must import :mod:`core` explicitly.
"""

from __future__ import annotations

__all__ = ()
