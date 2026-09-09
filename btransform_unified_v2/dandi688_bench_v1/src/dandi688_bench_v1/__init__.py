"""dandi688_bench_v1: local-benchmark skeleton for DANDI 000688 (CO).

Independent of the rift_v1 runner code; re-uses its immutable prepared cache
read-only.  Pure-CPU discipline, no training in this skeleton, formal-test
split sealed.  See docs/README.md in this package.
"""
from . import plan

__version__ = "1.0.0-skeleton"

__all__ = ["plan", "__version__"]
