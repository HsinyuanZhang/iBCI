#!/usr/bin/env python3
"""Static/default Phase-D gate; execution requires the exact two public flags."""
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

def _load_static():
    package = ModuleType("_tfsr_phase_d_static"); package.__path__ = [str(ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1")]; sys.modules[package.__name__] = package
    for leaf in ("contract", "source_smoke", "train"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{leaf}", ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1" / f"{leaf}.py")
        if spec is None or spec.loader is None: raise RuntimeError("static Phase-D loader failure")
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return sys.modules[f"{package.__name__}.train"]

def main() -> None:
    route = _load_static()
    if not given:
        print(json.dumps(route.training_plan(ROOT), sort_keys=True, indent=2)); return
    route.output_gate(ROOT)
    route.execute_training(ROOT)
if __name__ == "__main__": main()
