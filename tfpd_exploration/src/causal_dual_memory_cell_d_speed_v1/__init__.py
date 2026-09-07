"""Pure-speed, opt-in CDM-D evaluator composition.

This package is intentionally inert at import time: it owns no result root,
capability, data parser, checkpoint loader, CUDA initialization, or launch
path.  The future physical adapter is reachable only by an explicitly reviewed
consumer after its own authority lifecycle has completed.
"""

from .plan import CELL, PHASE

__all__ = ("CELL", "PHASE")
