"""Additive, model-free Stage-0 helpers for Cross-Session Worst-Group SPINT.

The package initializer intentionally imports neither Torch nor any M1 model,
dataset, datamodule, checkpoint, data authority, or CUDA runtime.  Numerical
helpers live in :mod:`core`; the static CLI imports only :mod:`plan`.
"""
from __future__ import annotations

__all__ = ()
