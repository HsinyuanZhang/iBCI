#!/usr/bin/env python3
"""Static Phase-E V2 plan by default; execution needs the exact two flags."""
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
    """Install deterministic repository roots without ambient PYTHONPATH."""
    for path in (ROOT / "tfpd_exploration" / "src", ROOT / "tfpd_exploration"):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


def _load_static() -> ModuleType:
    _bootstrap_repository_imports()
    package = ModuleType("_tfsr_phase_e_v2_static")
    package.__path__ = [str(ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2")]
    sys.modules[package.__name__] = package
    source = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2/score_v2.py"
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.score_v2", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("static Phase-E V2 loader failure")
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
    # Public flags are necessary but not sufficient: this routine must reload
    # V1/V2 immutable authority and receive its internal reviewed capability
    # before it can reserve an attempt root or resolve an evaluation asset.
    route.execute_authorized_v2(ROOT)


if __name__ == "__main__":
    main()
