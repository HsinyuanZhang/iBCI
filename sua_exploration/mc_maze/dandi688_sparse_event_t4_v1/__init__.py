"""DANDI 000688 sparse-event T4 / conditional-FiLM V1 route.

This package is deliberately descriptor-first.  Importing it does not open an
NWB, construct a decoder, initialize CUDA, or create a results directory.
"""

from .plan import ROUTE_NAME

__all__ = ["ROUTE_NAME"]
