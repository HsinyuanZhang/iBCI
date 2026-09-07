"""Pytest path setup for btransform_unified_v1 tests.

Runs from the repository root or the package directory. Phase 0 is strictly
CPU-only: CUDA is masked before torch is imported anywhere in the suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hard constraint: no CUDA init

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # .../btransform_unified_v1
_SRC_ROOT = _PACKAGE_ROOT / "src"

for path in (_SRC_ROOT,):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
