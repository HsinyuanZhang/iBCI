#!/usr/bin/env python3
"""Run one authorised d8/it250 strict-source GPU engineering cost fit.

This runner has no target, formal, geometry, iteration, seed, decoder, metric,
or winner option.  Root review must separately authorise its invocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


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
    parser.add_argument("--output", type=Path, default=gate.CANONICAL_COST_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-have-authorization", action="store_true")
    args = parser.parse_args(argv)
    entrypoint = Path(__file__).resolve()
    route.require(args.expected_cuda_visible_device == gate.CANONICAL_PHYSICAL_GPU_INDEX,
                  "fixed GPU cost route requires frozen physical GPU index 1")
    output = gate.validate_canonical_output(args.output)
    route.require(args.execute == args.i_have_authorization,
                  "real GPU fit requires both --execute and --i-have-authorization")
    if not args.execute:
        dry = gate.no_data_preflight(
            entrypoint=entrypoint, expected_visible_device=args.expected_cuda_visible_device)
        print(json.dumps({
            "status": "DRY_PLAN_ONLY__NO_FIT__NO_WRITE",
            "preflight_status": dry["status"],
            "canonical_output": str(output),
            "execute_requested": False,
            "authorization_asserted": False,
            "torch_imported": "torch" in sys.modules,
            "source_data_opened": False,
            "model_fit_called": False,
        }, indent=2, sort_keys=True))
        return 0

    launch_closure = route.snapshot_file_closure(gate.implementation_paths(entrypoint))
    preflight_dry = gate.no_data_preflight(
        entrypoint=entrypoint, expected_visible_device=args.expected_cuda_visible_device)

    import torch
    import cebra
    nvidia = gate.nvidia_smi_identity(args.expected_cuda_visible_device)
    runtime_probe = gate.live_cuda_probe(torch, nvidia_identity=nvidia)
    preflight = gate.no_data_preflight(
        entrypoint=entrypoint, expected_visible_device=args.expected_cuda_visible_device,
        runtime_probe=runtime_probe)
    route.require(preflight_dry["implementation_closure_at_preflight"] == launch_closure ==
                  preflight["implementation_closure_at_preflight"], "preflight closure drift")

    started = time.monotonic()
    authority_started = time.monotonic()
    payloads, bindings = gate.load_authorities()
    authority_seconds = time.monotonic() - authority_started
    materialize_started = time.monotonic()
    fold, boundary = gate.materialize_first_fold(payloads)
    materialize_seconds = time.monotonic() - materialize_started
    backend = gate.VendoredCebra061GpuBackend(torch=torch, cebra=cebra)
    fit_started = time.monotonic()
    run = backend.run_cost_fit(fold)
    fit_transform_seconds = time.monotonic() - fit_started
    validation = gate.validate_cost_run(run=run, fold=fold, boundary=boundary)
    runtime = {
        "authority_validation_wall_clock_s": authority_seconds,
        "strict27_materialization_wall_clock_s": materialize_seconds,
        "gpu_fit_plus_transform_wall_clock_s": fit_transform_seconds,
        "total_wall_clock_s": time.monotonic() - started,
        "python_executable": sys.executable,
    }
    route.require_file_closure_unchanged(launch_closure)
    receipt = gate.cost_receipt(
        preflight=preflight, authority_bindings=bindings, boundary=boundary,
        validation=validation, runtime_seconds=runtime, runtime_probe=runtime_probe,
        backend_identity=backend.identity, launch_closure=launch_closure, live_closure_equal=True)
    route.require_file_closure_unchanged(launch_closure)
    published = route.write_immutable_pair(output, receipt)
    print(json.dumps({"status": receipt["status"], "gpu_fit_call_count": 1,
                      "runtime": receipt["runtime"], "published": published},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
