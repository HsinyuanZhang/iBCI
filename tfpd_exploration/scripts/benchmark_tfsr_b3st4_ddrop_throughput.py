#!/usr/bin/env python3
"""Static TF-SR throughput plan; GPU execution remains root-audit fail-closed."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
FLAGS = {"--execute", "--i-have-root-throughput-authorization"}
GIVEN = sys.argv[1:]


def _load_static() -> ModuleType:
    """Load only stdlib-only benchmark metadata, never package ``__init__``."""
    package = ModuleType("_tfsr_throughput_static")
    package.__path__ = [str(ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1")]
    sys.modules[package.__name__] = package
    source = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark.py"
    spec = importlib.util.spec_from_file_location(package.__name__ + ".throughput_benchmark", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("static throughput benchmark loader failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    # Reject any partial/unrecognized flag combination before even loading the
    # static module.  This proves a one-flag command cannot import Torch or
    # reach CUDA, a dataset, a checkpoint, or an output path.
    if GIVEN and (set(GIVEN) != FLAGS or len(GIVEN) != len(FLAGS)):
        raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_BENCHMARK")
    route = _load_static()
    if not GIVEN:
        print(json.dumps(route.dry_plan(), sort_keys=True, indent=2))
        return
    # The two explicit flags prove intent, not root acceptance.  The current
    # work order intentionally stops before any physical import or CUDA call.
    route.reject_unreviewed_public_execution()


if __name__ == "__main__":
    main()
