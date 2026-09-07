#!/usr/bin/env python3
"""v2 TF-SR throughput engineering benchmark CLI.

Zero arguments print the sealed static plan without importing Torch.  The two
explicit flags together execute the reviewed GPU0 matrix and publish exactly
one engineering-only receipt under
``tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v2``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
FLAGS = {"--execute", "--i-have-root-throughput-authorization-v2"}
GIVEN = sys.argv[1:]


def _load_static() -> ModuleType:
    """Load the v2 benchmark metadata (and the v1 module it imports) as a package."""
    package_root = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1"
    package = ModuleType("_tfsr_throughput_v2_static")
    package.__path__ = [str(package_root)]
    sys.modules[package.__name__] = package
    for leaf in ("throughput_benchmark.py", "throughput_benchmark_v2.py"):
        source = package_root / leaf
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{leaf[:-3]}", source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"static v2 throughput loader failure for {leaf}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[package.__name__ + ".throughput_benchmark_v2"]


def main() -> None:
    # Reject any partial/unrecognized flag combination before even loading the
    # static module: a one-flag command can never import Torch or reach CUDA,
    # a dataset, a checkpoint, or an output path.
    if GIVEN and (set(GIVEN) != FLAGS or len(GIVEN) != len(FLAGS)):
        raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK")
    route = _load_static()
    if not GIVEN:
        print(json.dumps(route.dry_plan(), sort_keys=True, indent=2))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("V2_BENCHMARK_REQUIRES_CUDA_VISIBLE_DEVICES_0_EXACTLY")
    # The frozen model imports its native dependency as ``src.tfpd...``; the
    # physical route therefore needs the tfpd_exploration package root on the
    # import path.  The dry plan above never needed it.
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))
    result = route.run_reviewed_gpu_benchmark_v2(ROOT)
    print(json.dumps({
        "status": "ENGINEERING_BENCHMARK_COMPLETE",
        "receipt_sha256": result["receipt_sha256"],
        "conclusion": result["receipt"]["conclusion"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
