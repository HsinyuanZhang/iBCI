"""Post-fusion M2 variant screen V1.

Importing this package is deliberately inert: the public CLI imports only the
pure plan until a root-owned live capability admits the runtime.
"""

from . import plan

__all__ = ("plan",)
