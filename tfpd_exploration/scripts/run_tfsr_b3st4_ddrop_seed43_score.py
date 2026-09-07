#!/usr/bin/env python3
"""Static/fail-closed entrypoint for the seed-43 Phase-E replication addendum.

The zero-argument route loads only standard-library code from ``score_43.py``.
It intentionally does not import the frozen seed-42 package (whose package
initializer imports the model), Torch, an NWB parser, or CUDA.  The public
flags cannot supply the opaque root-reviewed capability accepted by the live
function, so they fail before any authority, data, or output path is touched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
FLAGS = {"--execute", "--i-have-seed43-phase-e-replication-authorization"}


def _bootstrap_route_import_paths(*, live: bool) -> None:
    """Install the exact route roots without inheriting an accidental order.

    The static path needs only ``tfpd_exploration`` so that ``src`` resolves.
    The reviewed live path additionally validates seed-43 epoch receipts,
    whose public learning-rate authority is ``tfpd_lane.arm_common`` under
    ``tfpd_exploration/src``.  This mirrors the seed-43 train CLI rather than
    relying on an ambient ``PYTHONPATH``.  Adding either directory does not
    import Torch or touch data/CUDA.
    """
    project = str(ROOT / "tfpd_exploration")
    lane = str(ROOT / "tfpd_exploration/src")
    managed = {project, lane}
    sys.path[:] = [item for item in sys.path if item not in managed]
    sys.path.insert(0, project)
    if live:
        # Keep the live ordering identical to the train route: lane first,
        # project second, so both ``tfpd_lane`` and ``src`` are unambiguous.
        sys.path.insert(0, lane)


def _load_static() -> ModuleType:
    """Load the additive module under a synthetic torch-free package."""
    _bootstrap_route_import_paths(live=False)
    package_name = "src.tfsr_b3st4_ddrop_seed43_v1"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1")]
        sys.modules[package_name] = package
    path = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1/score_43.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.score_43", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("static seed43 replication scorer loader failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    given = sys.argv[1:]
    given_set = set(given)
    if given and (len(given) != len(given_set) or given_set != FLAGS):
        raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE")
    route = _load_static()
    if not given:
        print(json.dumps(route.dry_plan(), sort_keys=True, indent=2))
        return
    # The CLI has no way to create the opaque root-reviewed in-process
    # capability, so this stops before authority or target resolution.
    _bootstrap_route_import_paths(live=True)
    route.execute_authorized(ROOT, capability=None)


if __name__ == "__main__":
    main()
