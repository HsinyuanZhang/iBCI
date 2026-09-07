"""Support-Anchored Causal T4 Memory, Stage O (oracle headroom), V1.

Additive score-only successor over the frozen activity-only CDM runtime; no
frozen package is edited.  See ``plan.py`` for the binding pre-registrations.
"""

from . import anchor, block_refit, gates, plan, trust_region

__all__ = ["anchor", "block_refit", "gates", "plan", "trust_region"]
