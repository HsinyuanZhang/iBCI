#!/usr/bin/env python3
"""Static Phase-E plan by default; scoring needs both explicit authorization flags.

The loader intentionally executes ``score.py`` as a tiny synthetic package so
the package's existing ``__init__.py`` (which exposes the Torch model) cannot
run on a zero-argument dry invocation.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


FLAGS = {"--execute", "--i-have-phase-e-root-authorization"}
GIVEN = set(sys.argv[1:])
ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_repository_imports() -> None:
    """Install both fixed repository roots before Phase-D receipt validation.

    The static scorer shim deliberately avoids importing the package's
    Torch-bearing ``__init__`` on a zero-argument dry invocation.  The live
    Phase-D receipt validator nonetheless evaluates the frozen LR formula,
    whose import is top-level ``tfpd_lane.arm_common`` while that package's
    initializer imports ``src.tfpd_lane``.  Those two canonical package names
    require both roots below.  Do not depend on an operator-provided
    ``PYTHONPATH`` and do not discover any path dynamically.
    """
    for path in (ROOT / "tfpd_exploration" / "src", ROOT / "tfpd_exploration"):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


def _load_static() -> ModuleType:
    _bootstrap_repository_imports()
    package = ModuleType("_tfsr_phase_e_static")
    package.__path__ = [str(ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1")]
    sys.modules[package.__name__] = package
    path = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/score.py"
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.score", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("static Phase-E loader failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if GIVEN and (GIVEN != FLAGS or len(sys.argv[1:]) != len(FLAGS)):
        raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE")
    route = _load_static()
    if not GIVEN:
        print(json.dumps(route.dry_plan(), sort_keys=True, indent=2))
        return
    # This reaches the fixed physical no-cache backend only after the
    # independently published Phase-E authority pair and terminal validation.
    # There are no CLI data/device/budget overrides or backend substitutions.
    route.execute_authorized(ROOT)


if __name__ == "__main__":
    main()
