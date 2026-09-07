#!/usr/bin/env python3
"""Static seed43 Phase-E V3 plan; execution needs both reviewed flags."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


FLAGS = {"--execute", "--i-have-seed43-phase-e-v3-authorization"}
GIVEN = set(sys.argv[1:])
ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_repository_imports() -> None:
    """Match reviewed score CLIs without relying on ambient ``PYTHONPATH``."""
    for path in (ROOT / "tfpd_exploration" / "src", ROOT / "tfpd_exploration"):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


def _load_static() -> ModuleType:
    _bootstrap_repository_imports()
    package = ModuleType("_tfsr_seed43_phase_e_v3_static")
    package.__path__ = [str(ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v3")]
    sys.modules[package.__name__] = package
    source = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v3/score_v3.py"
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.score_v3", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("static seed43 V3 loader failure")
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
    # Public flags intentionally cannot manufacture the opaque in-process
    # capability required by this call.  Root uses the reviewed module API.
    route.execute_authorized_v3(ROOT)


if __name__ == "__main__":
    main()
