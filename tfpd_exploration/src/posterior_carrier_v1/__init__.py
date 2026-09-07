"""Posterior-carrier Stage-0 package.

Importing this package intentionally stays free of Torch, data, CUDA, result,
or artifact side effects.  Tensor-bearing posterior primitives live in
``core`` and are imported only by an explicitly non-dry caller.
"""

from .plan import dry_plan

__all__ = ("dry_plan",)

