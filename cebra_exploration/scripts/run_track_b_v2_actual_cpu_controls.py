#!/usr/bin/env python3
"""Run non-authorising actual vendored-CEBRA synthetic Track-B v2 controls.

Geometry is loaded only from an immutable selector receipt.  There are no
geometry override arguments and no NWB/target/formal paths.  The default is a
one-seed engineering runtime smoke; the eight-seed option still cannot freeze a
threshold or mint official authority without a later reviewed root action.
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
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapt-iterations", type=int, default=1,
                        help="Engineering-only shortened adaptation iterations.")
    parser.add_argument("--eight-seed-smoke", action="store_true",
                        help="Still non-authorising and does not freeze a threshold.")
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    sidecar = output.with_name(f"{output.name}.sha256")
    route.require(not os.path.lexists(output) and not os.path.lexists(sidecar),
                  "control output body/sidecar must be fresh before selector/CEBRA access")
    seeds = route.CONTROL_SEEDS if args.eight_seed_smoke else (0,)
    backend = route.VendoredCebra061Backend()
    payload = route.execute_synthetic_controls(
        selector_path=args.selector.expanduser().absolute(), backend=backend,
        fold=route.make_synthetic_control_fold(), seeds=seeds,
        adapt_iterations=args.adapt_iterations, allow_engineering_selector=True,
        enforce_positive_gate=False)
    entrypoint = Path(__file__).resolve()
    payload["entrypoint"] = str(entrypoint)
    payload["entrypoint_sha256"] = route.sha256_bytes(entrypoint.read_bytes())
    payload["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    payload["threshold_freeze_performed"] = False
    published = route.write_immutable_pair(output, payload)
    print(json.dumps({"status": payload["status"], "published": published,
                      "cebra_fit_call_count": payload["cebra_fit_call_count"],
                      "fit_calls": payload["fit_calls"],
                      "hard_null_threshold_status": payload["hard_null_threshold_status"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
