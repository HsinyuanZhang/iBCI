#!/usr/bin/env python3
"""No-write, CPU-only seed-43 preflight; never a launch entry point.

Mirrors ``preflight_tfsr_b3st4_ddrop_seed42.py`` and adds the two seed-43
specific gates: the build-disclosure binding against the sealed throughput-v2
receipt (including a measured CPU first-forward equivalence spot-check of the
accelerated build versus the frozen forward) and the fresh canonical output
root check for ``results/tfsr_b3st4_ddrop_seed43_train_v1``.
"""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration"))

from src.tfsr_b3st4_ddrop_seed43_v1 import contract_43, train_43  # noqa: E402


def _require_cpu_isolation() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES='' ")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
        raise RuntimeError("CPU preflight requires PYTHONNOUSERSITE=1")


def _fresh_root_check() -> dict[str, object]:
    target = ROOT / contract_43.TRAIN_ROOT_RELATIVE
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("canonical seed-43 output ancestor is invalid")
    if target.exists() or target.is_symlink():
        raise RuntimeError("fresh canonical seed-43 output root required")
    return {"root_relative": contract_43.TRAIN_ROOT_RELATIVE, "fresh": True, "write_performed": False}


def _synthetic_capability(torch, model_module, batch: int, units: int, dtype):
    generator = torch.Generator()
    generator.manual_seed(contract_43.SEED_43)
    x = torch.randn((batch, 50, units), generator=generator, dtype=dtype)
    calibration = torch.randn((batch, 30, 100, units), generator=generator, dtype=dtype)
    raw_t4 = torch.randn((batch, units, 4), generator=generator, dtype=dtype)
    normalizer = model_module.T4Normalizer(
        torch.zeros((4,), dtype=dtype), torch.ones((4,), dtype=dtype), "a" * 64, "b" * 64
    )
    return x, calibration, normalizer(
        raw_t4,
        roster_digest="c" * 64,
        ordered_unit_ids=tuple(f"synthetic-unit-{index}" for index in range(units)),
        lineage=("seed43_cpu_no_data_preflight",),
    )


def cpu_no_data_resource_audit() -> dict[str, object]:
    """Measure the synthetic CPU forward and bind the accelerated build to it."""
    _require_cpu_isolation()
    import statistics
    import time

    import torch

    from src.tfsr_b3st4_ddrop_seed43_v1 import accelerated_forward
    from src.tfsr_b3st4_ddrop_v1 import model as model_module

    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    try:
        random.seed(contract_43.SEED_43)
        torch.manual_seed(contract_43.SEED_43)
        batch, units = 1, 128
        x, calibration, t4 = _synthetic_capability(torch, model_module, batch, units, torch.float32)
        model = model_module.TFSRDecoder(capture_diagnostics=False)
        model.eval()
        accounting = model.accounting(representative_n=units)
        latency_ms = model.cpu_latency_ms(x, calibration, t4, warmup=1, repeats=3)
        scripted = accelerated_forward.build_scripted_step_binding(torch, model)
        with torch.no_grad():
            frozen_prediction = model(x, calibration, t4)
            accelerated_prediction = accelerated_forward.accelerated_forward(torch, model, scripted, x, calibration, t4)
            forward_max_abs = float((frozen_prediction - accelerated_prediction).abs().max().item())
            accelerated_ms: list[float] = []
            for _ in range(1 + 3):
                start = time.perf_counter()
                accelerated_forward.accelerated_forward(torch, model, scripted, x, calibration, t4)
                accelerated_ms.append((time.perf_counter() - start) * 1000.0)
        if forward_max_abs != 0.0:
            raise RuntimeError("CPU preflight accelerated-build forward is not bitwise-equal to the frozen forward")
        if not torch.isfinite(accelerated_prediction).all().item():
            raise RuntimeError("CPU preflight accelerated-build nonfinite prediction")
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
    parameter_bytes = int(accounting["trainable_params"]) * 4
    token_tensor_bytes = 50 * units * 256 * 4
    return {
        "scope": "synthetic_CPU_no_data",
        "gpu": False,
        "torch_cuda_called": False,
        "synthetic_shape": {"batch": batch, "time": 50, "units": units,
                            "calibration_trials": 30, "calibration_features": 100, "seed": contract_43.SEED_43},
        "trainable_params": int(accounting["trainable_params"]),
        "params_by_block": {name.removeprefix("params_"): int(value) for name, value in accounting.items()
                            if name.startswith("params_")},
        "analytic_mac_estimate_n128": int(accounting["analytic_mac_estimate_n"]),
        "persistent_state_bytes": int(accounting["persistent_state_bytes"]),
        "analytic_one_token_tensor_bytes": token_tensor_bytes,
        "frozen_forward_cpu_latency_median_ms": latency_ms,
        "accelerated_forward_cpu_latency_median_ms": float(statistics.median(accelerated_ms[1:])),
        "accelerated_build_forward_max_abs_vs_frozen": forward_max_abs,
        "accelerated_build_first_step_forward_bitwise_equal": forward_max_abs == 0.0,
        "cpu_latency_protocol": {"warmup": 1, "repeats": 3, "statistic": "median", "mode_restored": True},
        "analytic_gpu_envelope_fp32_B1_N128": {
            "parameter_bytes": parameter_bytes,
            "one_token_tensor_bytes": token_tensor_bytes,
            "inference_peak": "NOT_MEASURED",
            "training_peak": "NOT_MEASURED",
            "excluded": ["autograd", "optimizer", "allocator", "workspaces"],
            "required_full_training_launch_blocker": "throughput_v2_receipt_evidence_and_live_gpu0_smoke",
        },
    }


def run_preflight(root: Path = ROOT) -> dict[str, object]:
    evidence = contract_43.verify_canonical_evidence_43(root)
    closure = contract_43.compute_seed43_closure(root)
    plan = contract_43.dry_plan_43(evidence, closure)
    # The full route identity (Phase-C acceptance, frozen closures, device
    # declaration) must also verify before any launch request is drafted.
    identity = train_43.production_identity(root)
    plan["route_identity_verified"] = {
        "phase_c_acceptance_body_sha256": identity.phase_c_acceptance["body_sha256"],
        "closures": {name: closure_map["closure_sha256"] for name, closure_map in identity.closures.items()},
        "device": dict(identity.device),
        "lineage": dict(identity.lineage),
    }
    plan["fresh_output_root"] = _fresh_root_check()
    plan["cpu_no_data_resource_audit"] = cpu_no_data_resource_audit()
    return plan


def main() -> None:
    print(json.dumps(run_preflight(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
