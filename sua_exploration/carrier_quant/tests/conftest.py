"""Pytest path setup for carrier_quant tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUA_EXPLORATION = Path(__file__).resolve().parents[2]
_SPINT_SRC = _REPO_ROOT / "SPINT-main" / "src"
if str(_SUA_EXPLORATION) not in sys.path:
  sys.path.insert(0, str(_SUA_EXPLORATION))
if str(_SPINT_SRC) not in sys.path:
  sys.path.insert(0, str(_SPINT_SRC))
