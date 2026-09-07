#!/usr/bin/env python3
"""Seal the non-overriding Phase-C cost supplement after audits/benchmarks exist."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
    static_forward_macs,
    training_macs_per_sample,
    validate_capacity_benchmark,
    validate_deep_source_audit_receipt,
    validate_source_batch_audit,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import file_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cost-receipt", type=Path, required=True)
    parser.add_argument("--source-batch-audit", type=Path, required=True)
    parser.add_argument("--deep-source-audit-receipt", type=Path, required=True)
    parser.add_argument("--spint-benchmark", type=Path, required=True)
    parser.add_argument("--t4-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base_cost_receipt.resolve(strict=True)
    audit_path = args.source_batch_audit.resolve(strict=True)
    deep_audit_path = args.deep_source_audit_receipt.resolve(strict=True)
    benchmark_paths = {
        "spint": args.spint_benchmark.resolve(strict=True),
        "t4": args.t4_benchmark.resolve(strict=True),
    }
    base = json.loads(base_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_source_batch_audit(audit)
    deep_audit = json.loads(deep_audit_path.read_text(encoding="utf-8"))
    validate_deep_source_audit_receipt(
        deep_audit, source_batch_audit_path=audit_path, deep_verify=False
    )
    for arm, path in benchmark_paths.items():
        validate_capacity_benchmark(json.loads(path.read_text(encoding="utf-8")), arm=arm)
    forward = static_forward_macs(base)
    training = training_macs_per_sample(base)
    epochs = {"spint": 35, "t4": 12}
    plan = {}
    for arm in ("spint", "t4"):
        plan[arm] = {}
        for fold in range(7):
            row = audit["arms"][arm][str(fold)]
            per_seed = {}
            for seed in (42, 43, 44):
                train_batches = row["train_full_batches_per_epoch"] * epochs[arm]
                validation_batches = row["source_validation_full_batches_per_epoch"] * epochs[arm]
                per_seed[str(seed)] = {
                    "epochs": epochs[arm],
                    "batch_size": 32,
                    "train_batches_per_epoch": row["train_full_batches_per_epoch"],
                    "source_validation_batches_per_epoch": row["source_validation_full_batches_per_epoch"],
                    "expected_train_batch_executions": train_batches,
                    "expected_validation_batch_executions_excluding_sanity": validation_batches,
                    "expected_train_samples": train_batches * 32,
                    "expected_validation_samples": validation_batches * 32,
                    "source_train_forward_backward_macs": (
                        train_batches * 32 * training[f"{arm}_training_forward_backward_per_sample"]
                    ),
                    "source_validation_forward_macs": (
                        validation_batches * 32 * training[f"{arm}_source_validation_forward_per_sample"]
                    ),
                }
            plan[arm][str(fold)] = per_seed
    payload = {
        "schema": "m2_post33_phase_c_cost_supplement_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "base_receipt_overridden": False,
        "score_data_accessed": False,
        "formal_data_accessed": False,
        "base_cost_receipt": file_metadata(base_path),
        "source_batch_audit": file_metadata(audit_path),
        "deep_source_audit_receipt": file_metadata(deep_audit_path),
        "capacity_benchmarks": {
            arm: file_metadata(path) for arm, path in benchmark_paths.items()
        },
        "backward_counting_convention": {
            "trainable_graph": "3F = forward + input-gradient equivalent + weight-gradient equivalent",
            "frozen_on_gradient_path": "2F = forward + input-gradient equivalent",
            "frozen_no_grad_teacher": "1F = forward only",
            "loss_activation_normalization_metric_optimizer_elementwise_ops": "excluded from analytical MACs; included in measured wall/peak",
        },
        "static_forward_macs_per_sample": forward,
        "training_and_validation_macs_per_sample": training,
        "source_plan_by_arm_fold_seed": plan,
        "deployment_static_costs": {
            "common_online_macs_per_window": base["common_decoder"]["online_macs_per_window"],
            "spint_support_calibration_macs": base["arms"]["spint"]["calibration_network_macs"],
            "t4_encoder_support_macs": base["arms"]["t4"]["encoder_cost_profile"]["mac_per_session"],
            "t4_ac4_fit_macs": base["arms"]["t4"]["ac4_fit_macs"],
            "spint_peak_stream_calibration_live_state_bytes_fp32": base["arms"]["spint"]["peak_stream_calibration_live_state_bytes_fp32"],
            "t4_peak_stream_calibration_live_state_bytes_fp32": base["arms"]["t4"]["peak_stream_calibration_live_state_bytes_fp32"],
        },
        "gate_split": {
            "prelaunch_accuracy_cost_gate": "PASS_STATIC_EXACT_AND_SYNTHETIC_CAPACITY_ONLY",
            "production_efficiency_claim_gate": "BLOCKED_UNTIL_ALL_42_CELLS_HAVE_VALID_SOURCE_AND_DEPLOYMENT_RUNTIME_EVIDENCE",
            "accuracy_interpretation_if_run": "runtime evidence does not read or alter endpoint scores",
        },
        "source_bindings": {
            "writer": file_metadata(Path(__file__).resolve()),
        },
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
