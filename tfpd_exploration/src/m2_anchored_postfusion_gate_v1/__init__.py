"""No-data contract for the M2 anchored post-fusion scalar gate.

The package initializer deliberately imports neither Torch nor a data/model
stack.  Runtime modules are reached only by a future, reviewed admission
layer after an immutable attempt exists.
"""

from .plan import CELL, SCHEMA

__all__ = ("CELL", "SCHEMA")
