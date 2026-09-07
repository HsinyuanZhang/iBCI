#!/usr/bin/env python3
"""Static/default seed-43 gate; execution requires the exact two public flags.

Zero arguments print the seed-43 training plan without importing torch: the
frozen seed-42 package modules and this route's modules are loaded by file
path as synthetic packages, so the frozen package ``__init__`` (which imports
torch through ``model.py``) never executes on the static path.

The two flags together authorize the one 48-epoch seed-43 run on physical GPU0
(``CUDA_VISIBLE_DEVICES=0``).  The live seed-42 run owns GPU1; this route
refuses every other visibility mask.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

FLAGS = {"--execute", "--i-have-48epoch-authorization"}
given = set(sys.argv[1:])
if given and (given != FLAGS or len(given) != len(sys.argv[1:])):
    raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH")
ROOT = Path(__file__).resolve().parents[2]

FROZEN_PACKAGE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_v1"
SEED43_PACKAGE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1"
FROZEN_DOTTED = "src.tfsr_b3st4_ddrop_v1"


def _load_synthetic_package(dotted: str, package_root: Path, leaves: tuple[str, ...]) -> ModuleType:
    """Load route modules by file path, bypassing any torch-importing __init__."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    package = ModuleType(dotted)
    package.__path__ = [str(package_root)]
    sys.modules[dotted] = package
    for leaf in leaves:
        spec = importlib.util.spec_from_file_location(f"{dotted}.{leaf}", package_root / f"{leaf}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"static seed-43 loader failure for {leaf}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(package, leaf, module)
    return package


def _load_static() -> ModuleType:
    # tfpd_exploration must be importable so the plain ``src`` package resolves;
    # its __init__ is a docstring only.  The frozen route modules are then
    # loaded under their real dotted name WITHOUT executing the frozen package
    # __init__, keeping the dry plan torch-free.
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))
    _load_synthetic_package(FROZEN_DOTTED, ROOT / FROZEN_PACKAGE_RELATIVE, ("contract", "source_smoke", "train"))
    package = _load_synthetic_package("_tfsr_seed43_static", ROOT / SEED43_PACKAGE_RELATIVE, ("contract_43", "train_43"))
    return sys.modules[f"{package.__name__}.train_43"]


def main() -> None:
    route = _load_static()
    if not given:
        print(json.dumps(route.training_plan(ROOT), sort_keys=True, indent=2))
        return
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("SEED43_TRAIN_REQUIRES_CUDA_VISIBLE_DEVICES_0_EXACTLY")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("SEED43_TRAIN_REQUIRES_PYTHONNOUSERSITE_1")
    # The physical route needs the lane helpers (tfpd_lane.arm_common) and the
    # native dependency root (src.tfpd) importable; the dry plan never did.
    sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))
    route.output_gate(ROOT)
    route.execute_training(ROOT)


if __name__ == "__main__":
    main()
