"""CPU-only, whole-pool correction for immutable Post-Fusion checkpoints.

Importing this package is deliberately stdlib-only.  Torch and every source or
held-out reader are deferred until after a root-only attempt receipt exists.
"""

from . import plan

__all__ = ("plan",)
