#!/usr/bin/env python3
"""Static/fail-closed entrypoint for equal-session matched scoring.

No public command line can manufacture the in-process root capability required
by the scorer.  Thus both visible flags are merely an acknowledgement; without
the separately reviewed capability the route stops before an authority root,
target pathname, Torch import, CUDA initialisation, or output directory exists.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
EXECUTE_FLAGS = {"--execute", "--i-have-root-reviewed-equal-session-score-authorization"}


def _load_static() -> ModuleType:
    path = ROOT / "tfpd_exploration/src/cell_d_equal_session_score_v1.py"
    spec = importlib.util.spec_from_file_location("_cell_d_equal_session_score_static", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("equal-session matched-score static loader failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    given = sys.argv[1:]
    given_set = set(given)
    if given and (len(given) != len(given_set) or given_set != EXECUTE_FLAGS):
        raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE")
    route = _load_static()
    if not given:
        print(json.dumps(route.dry_plan(), sort_keys=True, indent=2))
        return
    # There is deliberately no CLI path for an opaque root capability or a
    # backend injection.  The function fails before target resolution.
    route.execute_authorized(ROOT, capability=None, backend=None)


if __name__ == "__main__":
    main()
