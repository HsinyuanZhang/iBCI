#!/usr/bin/env python3
"""v3 TF-SR throughput engineering benchmark CLI.

Zero arguments print the sealed static plan without importing Torch.  The two
explicit flags together execute the reviewed GPU0 matrix and publish exactly
one engineering-only receipt under
``tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v3``.

With ``--venv-cells-only`` added, the SAME flags run only the triton-pinned
venv cells inside this interpreter (used when the CLI itself is executed by
the throwaway venv python) and print the cells JSON between markers instead
of publishing anything.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
FLAGS = {"--execute", "--i-have-root-throughput-authorization-v3"}
VENV_FLAG = "--venv-cells-only"
GIVEN = sys.argv[1:]


def _load_static() -> ModuleType:
    """Load the v3 benchmark metadata (and the v1/v2 modules it imports)."""
    package_root = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1"
    package = ModuleType("_tfsr_throughput_v3_static")
    package.__path__ = [str(package_root)]
    sys.modules[package.__name__] = package
    for leaf in ("throughput_benchmark.py", "throughput_benchmark_v2.py", "throughput_benchmark_v3.py"):
        source = package_root / leaf
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{leaf[:-3]}", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"static v3 throughput loader failure for {leaf}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[package.__name__ + ".throughput_benchmark_v3"]


def main() -> None:
    venv_mode = VENV_FLAG in GIVEN
    given = [flag for flag in GIVEN if flag != VENV_FLAG]
    if given and (set(given) != FLAGS or len(given) != len(FLAGS)):
        raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK")
    route = _load_static()
    if not given:
        print(json.dumps(route.dry_plan(), sort_keys=True, indent=2))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("V3_BENCHMARK_REQUIRES_CUDA_VISIBLE_DEVICES_0_EXACTLY")
    # The frozen model imports its native dependency as ``src.tfpd...``; the
    # physical route therefore needs the tfpd_exploration package root on the
    # import path.  The dry plan above never needed it.
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))
    if venv_mode:
        result = route.run_venv_cells_only(ROOT)
        print("__V3_VENV_CELLS_JSON__")
        print(json.dumps(result, sort_keys=True))
        return
    result = route.run_reviewed_gpu_benchmark_v3(ROOT)
    print(json.dumps({
        "status": "ENGINEERING_BENCHMARK_COMPLETE",
        "receipt_sha256": result["receipt_sha256"],
        "conclusion": result["receipt"]["conclusion"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
