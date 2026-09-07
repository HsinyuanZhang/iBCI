"""T4 project import shim for the exact Phase-C source runtime callback."""
from __future__ import annotations

from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[3]
SPINT_CALLBACKS = WORKSPACE / "SPINT-main/src"
if str(SPINT_CALLBACKS) not in sys.path:
    sys.path.append(str(SPINT_CALLBACKS))

# Load by file path to avoid replacing this project's top-level ``src`` package.
import importlib.util

SOURCE = SPINT_CALLBACKS / "callbacks/post33_source_cost_runtime_v4.py"
SPEC = importlib.util.spec_from_file_location("phase_c_shared_source_cost_callback", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Post33SourceCostRuntimeV4 = MODULE.Post33SourceCostRuntimeV4

__all__ = ["Post33SourceCostRuntimeV4"]
