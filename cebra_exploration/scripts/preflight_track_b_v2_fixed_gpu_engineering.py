#!/usr/bin/env python3
"""No-data preflight for the fixed-canonical Track-B GPU cost gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["PYTHONNOUSERSITE"] = "1"
sys.path[:] = [entry for entry in sys.path if "/.local/lib/python" not in entry]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
VENDOR = REPO_ROOT / "cebra_exploration" / "third_party" / "cebra"
for entry in (str(SRC), str(VENDOR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import track_b_v2_actual_cpu_route as route  # noqa: E402
import track_b_v2_fixed_gpu_engineering as gate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-cuda-visible-device", required=True)
    parser.add_argument("--probe-cuda", action="store_true",
                        help="Query the selected CUDA runtime without fitting a model or opening data.")
    parser.add_argument("--output", type=Path,
                        help="Immutable preflight receipt; permitted only with --probe-cuda.")
    args = parser.parse_args(argv)
    if args.output is not None and not args.probe_cuda:
        parser.error("--output requires --probe-cuda; dry-run does not publish")
    if args.output is not None:
        output = args.output.expanduser().absolute()
        sidecar = output.with_name(f"{output.name}.sha256")
        route.require(not os.path.lexists(output) and not os.path.lexists(sidecar),
                      "preflight output body/sidecar must be fresh before CUDA probe")
    runtime = None
    if args.probe_cuda:
        import torch
        nvidia = gate.nvidia_smi_identity(args.expected_cuda_visible_device)
        runtime = gate.live_cuda_probe(torch, nvidia_identity=nvidia)
    payload = gate.no_data_preflight(
        entrypoint=Path(__file__).resolve(),
        expected_visible_device=args.expected_cuda_visible_device,
        runtime_probe=runtime)
    if args.output is not None:
        gate.route.require_file_closure_unchanged(payload["implementation_closure_at_preflight"])
        published = gate.route.write_immutable_pair(args.output.expanduser().absolute(), payload)
    else:
        published = None
    print(json.dumps({"status": payload["status"], "runtime_probe": payload["runtime_probe"],
                      "vendored_cuda_static_audit": payload["vendored_cuda_static_audit"],
                      "published": published}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
