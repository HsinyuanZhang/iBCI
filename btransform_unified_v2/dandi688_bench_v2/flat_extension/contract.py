"""Immutable Flat-control contract; this module never mutates base recipe state."""
from __future__ import annotations

from pathlib import Path

from learnable_recency_v1.flat_control import flat_config

from .. import protocol
from ..common import RECIPE as BASE_RECIPE, SCHEMA as BASE_SCHEMA, sha256, source_hashes

FLAT_SCHEMA = BASE_SCHEMA + "_flat_recency"
FLAT_RECIPE = {**BASE_RECIPE, "tier": "fixed", "half_life_seconds": [None] * 8,
               "effective_bias": "allzero", "temporal_variant": "flat",
               "recency_config": {"task": "m2", "tier": "fixed", "half_life_seconds": [None] * 8,
                                  "layers": 4, "heads": 8, "effective_bias": "allzero"}}
_PACKAGE = Path(__file__).resolve().parent
_OWNED = ("__init__.py", "contract.py", "model.py", "training.py")


def flat_config_metadata() -> dict:
    cfg = flat_config("m2")
    if cfg.tier != "fixed" or tuple(cfg.half_life_seconds) != (None,) * 8 or cfg.layers != 4:
        raise RuntimeError("flat m2 configuration drift")
    return {"task": cfg.task, "tier": cfg.tier, "half_life_seconds": list(cfg.half_life_seconds),
            "layers": cfg.layers, "context_bins": cfg.context_bins, "effective_bias": "allzero"}


def flat_training_source_hashes() -> dict[str, str]:
    """Freeze base execution sources plus only the four training-extension files."""
    hashes = dict(source_hashes())
    for name in _OWNED:
        hashes[str((_PACKAGE / name).relative_to(protocol.WORKSPACE_ROOT))] = sha256(_PACKAGE / name)
    return hashes


__all__ = ["FLAT_SCHEMA", "FLAT_RECIPE", "flat_config", "flat_config_metadata", "flat_training_source_hashes"]
