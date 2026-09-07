#!/usr/bin/env python3
"""Run a non-authorising, synthetic source-only vendored-CEBRA selector smoke.

This entrypoint has no NWB/target/formal path.  It intentionally cannot mint an
official selector; root review must supply a settled source-fold authority to a
future invocation of the core API.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path[:] = [entry for entry in sys.path if "/.local/lib/python" not in entry]

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
VENDOR = REPO_ROOT / "cebra_exploration" / "third_party" / "cebra"
for entry in (str(SRC), str(VENDOR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import track_b_v2_actual_cpu_route as route  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1,
                        help="Engineering-only shortened CEBRA iterations; never official.")
    parser.add_argument("--output-dimension", type=int, default=3)
    parser.add_argument("--normalized-lambda", type=float, default=1.0e-2)
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    sidecar = output.with_name(f"{output.name}.sha256")
    route.require(not os.path.lexists(output) and not os.path.lexists(sidecar),
                  "selector output body/sidecar must be fresh before CEBRA import")
    fold = route.make_synthetic_control_fold()
    authority = route.build_source_authority([fold], engineering=True)
    spec = route.SelectorSpec(d_grid=(args.output_dimension,),
                              iteration_grid=(args.iterations,),
                              lambda_grid=(args.normalized_lambda,),
                              cebra_seed=42, mode="engineering_smoke")
    backend = route.VendoredCebra061Backend()
    payload = route.execute_source_only_selector(
        folds=[fold], spec=spec, backend=backend, source_authority=authority,
        source_authority_binding={"kind": "in_memory_deterministic_synthetic_engineering_authority",
                                  "payload_sha256": route.sha256_bytes(route.canonical_json_bytes(authority))})
    entrypoint = Path(__file__).resolve()
    payload["entrypoint"] = str(entrypoint)
    payload["entrypoint_sha256"] = route.sha256_bytes(entrypoint.read_bytes())
    payload["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    published = route.write_immutable_pair(output, payload)
    print(json.dumps({"status": payload["status"], "published": published,
                      "cebra_fit_call_count": payload["cebra_fit_call_count"],
                      "geometry_runs": payload["geometry_runs"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
