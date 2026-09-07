#!/usr/bin/env python3
"""No-write, CPU-only TF-SR Stage-0 preflight; never a launch entry point."""
from __future__ import annotations

import sys

# This gate intentionally precedes Path, contract, model, torch, evidence, and any write-capable work.
if len(sys.argv) != 1:
    raise SystemExit("NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_LAUNCH")

import json
import os
import random
from pathlib import Path

sys.dont_write_bytecode = True

from src.tfsr_b3st4_ddrop_v1.contract import compute_live_closure, dry_plan, verify_canonical_evidence


ROOT = Path(__file__).resolve().parents[2]


def _require_cpu_isolation() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES='' ")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
        raise RuntimeError("CPU preflight requires PYTHONNOUSERSITE=1")


def cpu_no_data_resource_audit() -> dict[str, object]:
    """Measure only the synthetic CPU forward after all evidence/closure gates pass."""
    _require_cpu_isolation()
    # Delayed imports make invalid CLI arguments and evidence failures model/torch-free.
    import torch

    from src.tfsr_b3st4_ddrop_v1.model import T4Normalizer, TFSRDecoder

    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    try:
        random.seed(42)
        torch.manual_seed(42)
        batch, units = 1, 128
        x = torch.randn(batch, 50, units)
        calibration = torch.randn(batch, 30, 100, units)
        raw_t4 = torch.randn(batch, units, 4)
        normalizer = T4Normalizer(torch.zeros(4), torch.ones(4), "a" * 64, "b" * 64)
        t4 = normalizer(
            raw_t4,
            roster_digest="c" * 64,
            ordered_unit_ids=tuple(f"synthetic-unit-{index}" for index in range(units)),
            lineage=("stage0_cpu_no_data",),
        )
        model = TFSRDecoder(capture_diagnostics=False)
        model.eval()
        accounting = model.accounting(representative_n=units)
        latency_ms = model.cpu_latency_ms(x, calibration, t4, warmup=1, repeats=3)
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
    parameter_bytes = int(accounting["trainable_params"]) * 4
    token_tensor_bytes = 50 * units * 256 * 4
    return {
        "scope": "synthetic_CPU_no_data",
        "gpu": False,
        "torch_cuda_called": False,
        "synthetic_shape": {"batch": batch, "time": 50, "units": units, "calibration_trials": 30, "calibration_features": 100},
        "trainable_params": int(accounting["trainable_params"]),
        "params_by_block": {name.removeprefix("params_"): int(value) for name, value in accounting.items() if name.startswith("params_")},
        "analytic_mac_estimate_n128": int(accounting["analytic_mac_estimate_n"]),
        "persistent_state_bytes": int(accounting["persistent_state_bytes"]),
        "analytic_one_token_tensor_bytes": token_tensor_bytes,
        "cpu_forward_latency_median_ms": latency_ms,
        "cpu_latency_protocol": {"warmup": 1, "repeats": 3, "statistic": "median", "mode_restored": True},
        "analytic_gpu_envelope_fp32_B1_N128": {
            "parameter_bytes": parameter_bytes,
            "one_token_tensor_bytes": token_tensor_bytes,
            "inference_peak": "NOT_MEASURED",
            "training_peak": "NOT_MEASURED",
            "excluded": ["autograd", "optimizer", "allocator", "workspaces"],
            "required_full_training_launch_blocker": "measure_training_peak_during_authorized_source_only_smoke_before_48_epoch_training_launch",
        },
    }


def run_preflight(root: Path = ROOT) -> dict[str, object]:
    evidence = verify_canonical_evidence(root)
    closure = compute_live_closure(root)
    plan = dry_plan(evidence, closure)
    plan["cpu_no_data_resource_audit"] = cpu_no_data_resource_audit()
    return plan


def main() -> None:
    print(json.dumps(run_preflight(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
