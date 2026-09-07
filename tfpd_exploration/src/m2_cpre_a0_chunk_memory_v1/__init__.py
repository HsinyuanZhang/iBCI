"""M2 C-Pre metadata and trial-free chunk-memory A0 route.

This package is deliberately inert at import time: it does not import Torch,
open a dataset/checkpoint/result root, or inspect CUDA.  Runtime admission is
separate and requires an opaque in-process capability.
"""

from .plan import ROUTE_SCHEMA

__all__ = ("ROUTE_SCHEMA",)
