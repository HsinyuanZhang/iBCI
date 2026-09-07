"""Isolated exact-E M2 runtime revision.

This package deliberately does not alter any EvalAI submission or accepted
runtime.  It loads their sealed payload format read-only for parity work.
"""

from .runtime import RuntimeV3Decoder, RuntimeV3Error

__all__ = ["RuntimeV3Decoder", "RuntimeV3Error"]
