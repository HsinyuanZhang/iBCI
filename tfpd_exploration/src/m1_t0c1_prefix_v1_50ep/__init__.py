"""M1 matched T0/C1 calibration-prefix pair — 50-epoch extension (additive).

Importing this package loads only the torch-free plan module.  The LR
wrapper, trainer, probe, Phase 3, and drivers are imported from their own
modules so a plan-only import cannot initialize CUDA or pull Torch.
"""
from . import plan  # noqa: F401

__all__ = ("plan",)
