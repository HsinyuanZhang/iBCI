"""CPU-only Stage-0 contract for budget-matched posterior CAL-AUG.

The package initializer intentionally imports no NumPy or Torch.  Dry planning
can therefore run under ``python -S``; numerical submodules are explicit.
"""

__all__ = ["contract", "plan", "posterior"]
