"""Exact (within the documented floating point gate) sliding-window helpers.

This revision deliberately contains no ordinary temporal KV cache: a causal
transformer token changes when its left context changes after a window shift.
"""

from .exact_window import ExactWindowInferenceAdapter, FrontendWindowCache, WindowCacheState

__all__ = ["ExactWindowInferenceAdapter", "FrontendWindowCache", "WindowCacheState"]
