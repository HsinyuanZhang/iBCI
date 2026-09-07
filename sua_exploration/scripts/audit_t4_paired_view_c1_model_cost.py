#!/usr/bin/env python3
"""CPU-only exact parameter/MAC/state audit for fresh paired-view C1.

The audit constructs the exact jointly trained B3S/T4 architecture used by C1,
but opens no neural dataset and performs no optimization.  It records configured
Linear/attention MAC counts and separates persistent deployment state from
transient calibration buffers so the two quantities cannot be conflated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))

from src.models.paired_view_c1_module import PairedViewC1LitModule  # noqa: E402


REFERENCE_UNITS = 64
TRIAL_LENGTH = 100
WINDOW_SIZE = 50
ACTIVITY_TRIALS = 10
T4_POOL_TRIALS = 50
SIDE_DIM = 4
HIDDEN_DIM = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_model(teacher: Path) -> PairedViewC1LitModule:
    optimizer = partial(torch.optim.Adam, lr=1.0e-4, weight_decay=0.0)
    model = PairedViewC1LitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path=str(teacher),
        window_size=WINDOW_SIZE,
        trial_length=TRIAL_LENGTH,
        id_hidden_dim=128,
        hidden_dim=HIDDEN_DIM,
        pad_value=-1.0,
        freeze_decoder=False,
        loss_mode="task_only",
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        identity_mode="calibrated",
        side_dim=SIDE_DIM,
        electrode_embed_dim=0,
        num_electrodes=0,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
        lambda_consistency=0.0,
        sua_task_loss_weight=0.5,
        pseudo_mua_task_loss_weight=0.5,
    )
    model.setup("fit")
    if model.student is None:
        raise RuntimeError("C1 cost audit constructed no student model")
    if any(parameter.device.type != "cpu" for parameter in model.student.parameters()):
        raise RuntimeError("C1 cost audit must remain CPU-only")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    teacher = args.teacher.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not teacher.is_file():
        raise FileNotFoundError(f"teacher checkpoint missing: {teacher}")
    if output.exists():
        raise FileExistsError(f"write-once C1 cost audit exists: {output}")

    model = build_model(teacher)
    assert model.student is not None
    student = model.student
    parameter_count = int(sum(parameter.numel() for parameter in student.parameters()))
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
    )
    encoder_parameter_count = int(sum(parameter.numel() for parameter in student.id_encoder.parameters()))
    decoder_parameter_count = int(sum(parameter.numel() for parameter in student.decoder.parameters()))
    if parameter_count != encoder_parameter_count + decoder_parameter_count:
        raise RuntimeError("C1 has an unaccounted student parameter path")
    if trainable_parameter_count != parameter_count:
        raise RuntimeError("C1 cost audit expected joint source training of encoder and decoder")

    encoder = asdict(
        student.id_encoder.cost_profile(
            num_neurons=REFERENCE_UNITS,
            trial_length=TRIAL_LENGTH,
            num_trials=ACTIVITY_TRIALS,
        )
    )
    decoder = student.decoder_cost_comparison_receipt(
        batch_size=1, num_neurons=REFERENCE_UNITS
    )
    decoder_macs = int(decoder["coupled"]["total"])
    descriptor_bytes = REFERENCE_UNITS * SIDE_DIM * 4
    identity_bytes = REFERENCE_UNITS * WINDOW_SIZE * 4
    neural_window_bytes = REFERENCE_UNITS * WINDOW_SIZE * 4
    persistent_identity_plus_descriptor = identity_bytes + descriptor_bytes

    payload = {
        "schema_version": 1,
        "status": "passed",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scope": "CPU architecture audit only; no train, development, or formal NWB opened",
        "formal_sua_files_opened": False,
        "formal_sua_paths_resolved": False,
        "neural_data_files_opened": 0,
        "teacher": {"path": str(teacher), "sha256": sha256_file(teacher)},
        "architecture": {
            "variant": "B3S",
            "side_dim": SIDE_DIM,
            "freeze_decoder_during_source_training": False,
            "lambda_consistency": 0.0,
            "task_loss_weights": {"sua": 0.5, "pseudo_mua": 0.5},
            "shared_optimizer": True,
            "shared_optimizer_steps_per_paired_batch": 1,
            "backward_passes_per_paired_batch": 2,
            "view_specific_heads": False,
        },
        "reference_shape": {
            "batch_size": 1,
            "units_or_channels": REFERENCE_UNITS,
            "trial_length_bins": TRIAL_LENGTH,
            "online_window_bins": WINDOW_SIZE,
            "activity_calibration_trials": ACTIVITY_TRIALS,
            "t4_label_rate_pool_trials": T4_POOL_TRIALS,
            "fp32_bytes_per_value": 4,
        },
        "exact_parameter_receipt": {
            "student_parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "encoder_parameter_count": encoder_parameter_count,
            "decoder_parameter_count": decoder_parameter_count,
            "student_fp32_weight_bytes": parameter_count * 4,
            "deployment_weight_copies_shared": 1,
            "dual_view_separate_comparator_weight_copies": 2,
        },
        "configured_mac_receipt": {
            "definition": "multiply-accumulates executed by configured Linear/attention/FFN paths; exclusions are carried in the decoder receipt",
            "b3s_encoder_activity_calibration": encoder,
            "coupled_decoder_per_online_window": decoder,
            "encoder_macs_per_activity_calibration_session": int(encoder["mac_per_session"]),
            "decoder_macs_per_online_window": decoder_macs,
            "shared_training_multiplier": {
                "view_forwards_per_optimizer_step": 2,
                "view_backwards_per_optimizer_step": 2,
                "optimizer_steps_per_paired_batch": 1,
            },
            "t4_closed_form_fit_macs": None,
            "t4_closed_form_fit_note": "descriptor fit is outside the neural model MAC counter and must not be silently folded into encoder MACs",
        },
        "fp32_state_receipt_reference_n64": {
            "persistent_t4_descriptor_bytes": descriptor_bytes,
            "persistent_identity_E_bytes": identity_bytes,
            "conservative_persistent_identity_plus_descriptor_bytes": persistent_identity_plus_descriptor,
            "activity_encoder_support_state_bytes": int(encoder["support_state_bytes"]),
            "activity_encoder_trial_buffer_bytes": int(encoder["trial_buffer_bytes"]),
            "activity_encoder_peak_live_state_bytes": int(encoder["peak_live_state_bytes"]),
            "online_neural_window_input_bytes": neural_window_bytes,
            "decoder_receipt_persistent_identity_bytes": int(
                decoder["coupled"]["persistent_state_bytes_fp32"]
            ),
            "deployment_extra_state_shared_vs_one_separate_model_bytes": 0,
            "state_definition": {
                "persistent": "session state retained across online windows",
                "transient": "calibration accumulator/trial input that may be released after E is finalized",
                "online_input": "current decoder window, reported separately and not called persistent calibration state",
            },
        },
        "no_heldout_backprop_contract": {
            "source_train_sessions": 27,
            "development_heldout_sessions": 6,
            "formal_test_sessions": 6,
            "optimizer_and_backward_scope": "source_train_27_only",
            "development_use": "post-training forward-only fixed epoch-5-through-12 scoring",
            "development_enters_train_dataloader": False,
            "development_enters_loss": False,
            "development_enters_optimizer": False,
            "development_uses_backward_gradients": False,
            "formal_paths_resolved": False,
            "formal_files_opened": False,
        },
    }
    write_json_exclusive(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "status": "passed",
                "student_parameter_count": parameter_count,
                "encoder_macs_per_session": encoder["mac_per_session"],
                "decoder_macs_per_window": decoder_macs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
