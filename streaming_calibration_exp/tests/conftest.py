"""Pytest path setup so tests run from the repository root or package directory."""
from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_SUA_ROOT = _REPO_ROOT / "sua_exploration"

for path in (_PACKAGE_ROOT, _SUA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
