"""Support-Anchored Causal T4 Memory, Stage P (deployable pseudo-direction), V1.

Additive score-only successor over the frozen activity-only CDM runtime and the
proven Stage-O machinery; no frozen package is edited.  See ``plan.py`` for the
binding pre-registrations (design §5--§8, work order 20260830).
"""

from . import direction_estimator, gate, gates, plan, replay

__all__ = ["direction_estimator", "gate", "gates", "plan", "replay"]
