"""Independent Flat-recency extension for the completed CONCAT campaign."""
from __future__ import annotations

import sys
from pathlib import Path

# ``python -m btransform_unified_v2.dandi688_bench_v2.flat_extension.training``
# starts with the workspace namespace package only.  Keep the extension
# executable in that clean environment without changing any base package.
_EXTENSION = Path(__file__).resolve().parent
_BENCH, _ROOT, _WORKSPACE = _EXTENSION.parent, _EXTENSION.parents[1], _EXTENSION.parents[2]
for _entry in (_WORKSPACE, _ROOT, _ROOT / "src", _ROOT / "learnable_recency_v1" / "src",
               _WORKSPACE / "btransform_unified_v1" / "src"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))
_top = sys.modules.get("btransform_unified_v2")
_source_package = str(_ROOT / "src" / "btransform_unified_v2")
if _top is not None and hasattr(_top, "__path__") and _source_package not in _top.__path__:
    _top.__path__.append(_source_package)

from .contract import FLAT_RECIPE, FLAT_SCHEMA
from .model import FlatDandiRiftDecoder, make_model, validate_flat_model

__all__ = ["FLAT_SCHEMA", "FLAT_RECIPE", "FlatDandiRiftDecoder", "make_model", "validate_flat_model"]
