"""Review-only CRST-B4 M2 SMALL family contract.

This package deliberately contains no training entry point, checkpoint loader, or
submission code.  It exists to make the one permitted FLAT/ROUTE difference
inspectable before an experiment is authorized.
"""

from .routing import M2RoutingConfig, M2SlotRoutingBias, add_route_bias

__all__ = ["M2RoutingConfig", "M2SlotRoutingBias", "add_route_bias"]
