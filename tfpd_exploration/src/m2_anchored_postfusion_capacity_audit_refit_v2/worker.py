"""Post-V1 GPU0-only hook; deliberately refuses execution until literals bind."""
from __future__ import annotations
from . import plan
def require_bound_v1_graph():
 if plan.V1_BODIES is None: raise RuntimeError('APFC V2 V1 terminal literals are not bound')
 return dict(plan.V1_BODIES)
