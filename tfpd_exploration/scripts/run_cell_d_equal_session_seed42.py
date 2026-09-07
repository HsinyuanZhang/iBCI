#!/usr/bin/env python3
"""Dry/fail-closed entrypoint for ``CELL_D_EQUAL_SESSION_SEED42``.

The zero-argument route imports only the additive standard-library schedule
module and prints a plan.  It cannot discover source/target data, import a
model, initialize CUDA, create a result root, or launch training.  A distinct
explicit source-only audit route is available for the Stage-0 window census;
the eventual training flags still fail closed because this implementation pass
does not carry a root-review capability.  The route-owned backend is only
reachable through that in-process capability.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
EXECUTE_FLAGS = {"--execute", "--i-have-root-reviewed-equal-session-authorization"}
SOURCE_SMOKE_FLAGS = {"--source-smoke", "--i-have-root-reviewed-equal-session-smoke-authorization"}
SOURCE_AUDIT_FLAGS = {"--audit-source-only", "--i-acknowledge-source-only-read"}


def _load_static() -> ModuleType:
    path = ROOT / "tfpd_exploration/src/cell_d_equal_session_v1.py"
    spec = importlib.util.spec_from_file_location("_cell_d_equal_session_static", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("equal-session static route loader failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    given = sys.argv[1:]
    given_set = set(given)
    if given and (len(given) != len(given_set)
                  or (given_set != EXECUTE_FLAGS and given_set != SOURCE_SMOKE_FLAGS
                      and given_set != SOURCE_AUDIT_FLAGS)):
        raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH")
    route = _load_static()
    if not given:
        print(json.dumps(route.dry_plan(), sort_keys=True, indent=2))
        return
    if given_set == SOURCE_AUDIT_FLAGS:
        print(json.dumps(route.source_only_window_audit(ROOT), sort_keys=True, indent=2))
        return
    # There is intentionally no CLI mechanism for supplying an in-process
    # root-reviewed capability.  Both routes therefore reject before any
    # source/Torch/CUDA/output-root action.
    route.execute_reviewed_training(
        root=ROOT,
        execute_flag=True,
        acknowledgement_flag=True,
        capability=None,
        source_smoke=given_set == SOURCE_SMOKE_FLAGS,
    )


if __name__ == "__main__":
    main()
