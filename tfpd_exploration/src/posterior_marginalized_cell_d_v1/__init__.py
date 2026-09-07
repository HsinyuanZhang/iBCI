"""Posterior-marginalized Cell-D route.

The package import surface intentionally stays free of Torch, data, CUDA,
checkpoint, result-root, and authority side effects.  Tensor-bearing cache
primitives live in :mod:`core` and are imported only by an explicitly
non-dry caller.
"""

from .plan import dry_plan

__all__ = ("dry_plan",)
