"""Posterior Carrier non-governing quick-screen package.

The package root is intentionally static: importing it does not import Torch,
remote transport, data readers, a checkpoint loader, or the physical scorer.
"""

from .quick_screen import dry_plan

__all__ = ("dry_plan",)
