#!/usr/bin/env python3
"""Write-once CPU-only r10 receipt after the complete source chain is frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
BASE_RECEIPT = SUA / (
    "results/t4_m30_experiment_a_descriptor_prelaunch_v3_r5_20260802/receipt.json"
)
PUBLIC_KEY = SUA / "configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem"
EXPECTED_PUBLIC_KEY_SHA256 = (
    "ff9d1b2b985c9cd8c2697cfb0e1e5353b8c19d1f3ff094d2987aa1537c79a375"
)

# This list is intentionally explicit.  Adding or changing a runtime import
# requires changing this writer and the independent closure test before a new
# receipt can be emitted.
RUNTIME_SOURCES = (
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/metrics/__init__.py",
    "streaming_calibration_exp/src/metrics/gate2_matrix.py",
    "streaming_calibration_exp/src/metrics/run_artifacts.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/neuron_dropout.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_spint_t4_logit_residual_adapter.py",
    "streaming_calibration_exp/src/models/falcon_module.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/t4_logit_residual_module.py",
    "streaming_calibration_exp/src/utils/__init__.py",
    "streaming_calibration_exp/src/utils/clean_teacher_validation.py",
    "streaming_calibration_exp/src/utils/instantiators.py",
    "streaming_calibration_exp/src/utils/logging_utils.py",
    "streaming_calibration_exp/src/utils/pylogger.py",
    "streaming_calibration_exp/src/utils/rich_utils.py",
    "streaming_calibration_exp/src/utils/utils.py",
    "sua_exploration/configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/sua_auxiliary_stage0.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/scripts/aggregate_t4_m30_experiment_a_r10.py",
    "sua_exploration/scripts/aggregate_t4_m30_experiment_a_r5.py",
    "sua_exploration/scripts/aggregate_t4_m30_experiment_a_v3.py",
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py",
    "sua_exploration/scripts/eval_adaptation_dandi688.py",
    "sua_exploration/scripts/eval_epoch_window_generic_dandi688.py",
    "sua_exploration/scripts/eval_t4_m30_experiment_a.py",
    "sua_exploration/scripts/run_t4_m30_experiment_a_r10_one_cell.sh",
    "sua_exploration/scripts/schedule_t4_m30_experiment_a_r10_2gpu.sh",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
    "sua_exploration/scripts/t4_m30_experiment_a_r10_authorization.py",
    "sua_exploration/scripts/t4_m30_experiment_a_r4_authorization.py",
    "sua_exploration/scripts/t4_m30_experiment_a_r5_authorization.py",
    "sua_exploration/scripts/train_variant_dandi688.py",
    "sua_exploration/scripts/verify_t4_m30_experiment_a_r10_authorization.py",
    "sua_exploration/scripts/write_t4_m30_experiment_a_prelaunch_r10.py",
    "sua_exploration/tests/test_t4_m30_experiment_a_r5_disk_fixture.py",
    "sua_exploration/tests/test_t4_m30_experiment_a_r10_closure.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    expected_output = (
        SUA
        / "results/t4_m30_experiment_a_descriptor_prelaunch_v3_r10_20260803"
    ).resolve()
    if output_dir != expected_output:
        raise ValueError(f"r10 receipt path must be {expected_output}")
    if sha256(PUBLIC_KEY) != EXPECTED_PUBLIC_KEY_SHA256:
        raise ValueError("public key drift")
    missing = [relative for relative in RUNTIME_SOURCES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing r10 source dependencies: {missing}")

    receipt = json.loads(BASE_RECEIPT.read_text())
    source_hashes = {
        relative: sha256(ROOT / relative) for relative in RUNTIME_SOURCES
    }
    receipt.update(
        {
            "schema_version": 10,
            "status": "PASS",
            "source_sha256": source_hashes,
            "dependency_closure": list(RUNTIME_SOURCES),
            "execution": {
                "gpu_used": False,
                "training_started": False,
                "formal_sua_opened": False,
            },
            "authorization": {
                "gpu_launch_authorized": False,
                "trust_anchor": {
                    "algorithm": "Ed25519",
                    "public_key_path": str(PUBLIC_KEY.resolve()),
                    "public_key_sha256": sha256(PUBLIC_KEY),
                    "fixed_authorization_path": str(
                        (output_dir / "root_gpu_authorization.json").resolve()
                    ),
                    "fixed_signature_path": str(
                        (output_dir / "root_gpu_authorization.sig").resolve()
                    ),
                },
            },
            "matrix": {
                **receipt["matrix"],
                "arms": ["Z4", "PH4", "AC4", "MB4", "B4", "LS4"],
                "seeds": [42, 43, 44],
                "cells": 18,
                "support_pool": 30,
                "epochs": 12,
                "epoch_window": list(range(5, 13)),
                "cache_dir": "sua_exploration/cache/t4_m30_experiment_a_v10",
                "runtime_screen": "sua_t4_m30_component_attribution_v10",
                "managed_execution": (
                    "managed foreground exec session only; "
                    "no nohup/disown/short-lived wrapper"
                ),
            },
            "recovery_incidents": {
                "r5": {
                    "all_cells_exit_code": 126,
                    "cause": "scheduler direct-executed non-executable runner",
                    "gpu_used": False,
                    "checkpoints_created": False,
                    "results_created": False,
                    "evidence_preserved": True,
                },
                "r6": {
                    "cause": "short-lived nohup wrapper reaped scheduler and children",
                    "python_entered_for_two_cells": True,
                    "epochs_completed": False,
                    "gpu_result_created": False,
                    "evidence_preserved": True,
                },
                "r7_r8_r9": {
                    "gpu_authorized": False,
                    "gpu_launched": False,
                    "reason": "prelaunch receipt/source dependency closure rejected",
                    "evidence_preserved": True,
                },
            },
            "r10_execution_contract": {
                "scheduler": "schedule_t4_m30_experiment_a_r10_2gpu.sh",
                "runner_invocation": 'bash "$RUNNER"',
                "status_hashes": [
                    "result_sha256",
                    "metadata_sha256",
                    "cost_sha256",
                ],
                "pre_aggregate_sweep": (
                    "all 18 status receipts completed with exit_code 0 and all "
                    "three artifact hashes present"
                ),
                "scheduler_transport": "persistent managed foreground exec session",
            },
        }
    )

    # Receipt creation is deliberately the final mutation in this program.
    output_dir.mkdir(parents=True)
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "RECEIPT.md").write_text(
        "# Experiment A r10 prelaunch\n\n"
        "Status: **PASS (CPU-only prelaunch)**. GPU authorization remains false. "
        "Launch is permitted only after a detached Ed25519 authorization is "
        "issued and only in a persistent managed foreground execution session.\n"
    )


if __name__ == "__main__":
    main()
