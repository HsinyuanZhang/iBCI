#!/usr/bin/env python3
"""Recompute score-free paired-arm deployment costs for Phase-C v4."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "streaming_calibration_exp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-date", required=True)
    args = parser.parse_args()
    # Reject timestamps and ambiguous locale-specific dates. Receipts use one
    # explicit ISO calendar day supplied by the launch owner.
    try:
        receipt_date = date.fromisoformat(args.receipt_date).isoformat()
    except ValueError as error:
        raise ValueError("--receipt-date must be canonical YYYY-MM-DD") from error
    if receipt_date != args.receipt_date:
        raise ValueError("--receipt-date must be canonical YYYY-MM-DD")
    # Keep the caller's lexical final component intact so O_NOFOLLOW below can
    # reject a pre-existing output symlink.  Path.resolve() here would follow
    # that symlink before os.open and silently defeat the protection.
    output = args.output.expanduser().absolute()

    sys.path.insert(0, str(STREAMING))
    import torch
    from src.models.components.spint import SpintModel
    from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
    from src.models.components.streaming_spint import StreamingSpintModel

    num_units, trial_length, support_trials, window = 96, 100, 33, 50
    model_dim, num_id_layers, hidden_dim, side_dim = 512, 3, 64, 4
    decoder = SpintModel(
        model_dim=model_dim,
        num_covariates=2,
        window_size=window,
        num_heads=64,
        num_layers=1,
        num_id_layers=num_id_layers,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=True,
        dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
        tf_drop_rate=0.1,
        readin_layer_type="mlp",
    )
    # Materialize the one LazyLinear with the exact calibration trial width.
    decoder.fc_id_in(torch.zeros(1, trial_length, dtype=torch.float32))
    encoder = SideFeatureEarlyPoolEncoder(
        trial_length=trial_length,
        window_size=window,
        hidden_dim=hidden_dim,
        side_dim=side_dim,
        electrode_embed_dim=0,
        num_electrodes=0,
        num_post_layers=3,
    )
    student = StreamingSpintModel(decoder=decoder, id_encoder=encoder)
    decoder_parameters = sum(parameter.numel() for parameter in decoder.parameters())
    decoder_identity_parameters = sum(parameter.numel() for parameter in decoder.fc_id_in.parameters()) + sum(
        parameter.numel() for parameter in decoder.fc_id_out.parameters()
    )
    decoder_online_parameters = decoder_parameters - decoder_identity_parameters
    encoder_profile = asdict(encoder.cost_profile(num_units, trial_length, support_trials))
    decoder_cost = student.decoder_cost_comparison_receipt(
        batch_size=1, num_neurons=num_units
    )["coupled"]

    spint_per_trial_identity_mac = num_units * (
        trial_length * model_dim + (num_id_layers - 1) * model_dim * model_dim
    )
    spint_finalize_identity_mac = num_units * (
        (num_id_layers - 1) * model_dim * model_dim + model_dim * window
    )
    spint_calibration_mac = support_trials * spint_per_trial_identity_mac + spint_finalize_identity_mac
    raw_support_bytes = support_trials * trial_length * num_units * 4
    spint_support_state = num_units * model_dim * 4
    trial_buffer = trial_length * num_units * 4
    identity_state = num_units * window * 4
    neural_history_state = num_units * window * 4

    # AC4/T4 normal equations use a common p=3 direction design and N RHSs.
    p = 3
    ac4_normal_equation_mac = support_trials * p * p + num_units * support_trials * p
    ac4_common_inverse_and_apply_mac = p**3 + num_units * p * p
    ac4_fit_mac = ac4_normal_equation_mac + ac4_common_inverse_and_apply_mac
    ac4_sufficient_state_bytes = (p * p + num_units * p + num_units + 1) * 4
    t4_descriptor_bytes = num_units * side_dim * 4

    reference = {
        "num_units": num_units,
        "trial_length_bins": trial_length,
        "support_trials": support_trials,
        "window_bins": window,
        "model_dim": model_dim,
        "hidden_dim": hidden_dim,
        "side_dim": side_dim,
        "float_bytes": 4,
        "batch_size_online": 1,
    }
    receipt = {
        "schema": "m2_post33_paired_arm_cost_receipt_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "date": receipt_date,
        "reference_shape": reference,
        "gpu_used": False,
        "score_data_accessed": False,
        "formal_data_accessed": False,
        "counting_convention": {
            "mac": "one multiply-accumulate counted as one MAC",
            "included": ["Linear", "attention matmul", "FFN", "AC4 normal equations and apply"],
            "excluded": ["activation", "normalization", "softmax", "dropout", "sqrt", "spike accumulation additions"],
            "dtype_for_weight_and_state_bytes": "fp32",
        },
        "common_decoder": {
            "stored_parameter_count": decoder_parameters,
            "stored_weight_bytes_fp32": decoder_parameters * 4,
            "identity_subnetwork_parameter_count": decoder_identity_parameters,
            "online_active_parameter_count_after_identity": decoder_online_parameters,
            "online_macs_per_window": decoder_cost["total"],
            "online_mac_breakdown": decoder_cost,
            "online_neural_history_state_bytes_fp32": neural_history_state,
            "persistent_identity_state_bytes_fp32": identity_state,
            "combined_online_history_plus_identity_bytes_fp32": neural_history_state + identity_state,
        },
        "arms": {
            "spint": {
                "stored_parameter_count": decoder_parameters,
                "trainable_source_parameter_count": decoder_parameters,
                "calibration_network_macs": spint_calibration_mac,
                "calibration_mac_per_trial": spint_per_trial_identity_mac,
                "calibration_finalize_macs": spint_finalize_identity_mac,
                "reference_batch_support_tensor_bytes_fp32": raw_support_bytes,
                "deployable_stream_support_state_bytes_fp32": spint_support_state,
                "trial_buffer_bytes_fp32": trial_buffer,
                "peak_stream_calibration_live_state_bytes_fp32": spint_support_state + trial_buffer,
                "persistent_identity_state_bytes_fp32": identity_state,
                "online_macs_per_window": decoder_cost["total"],
                "target_labels_used": False,
            },
            "t4": {
                "stored_parameter_count": decoder_parameters + encoder_profile["parameter_count"],
                "frozen_decoder_parameter_count": decoder_parameters,
                "trainable_source_parameter_count": encoder_profile["parameter_count"],
                "encoder_cost_profile": encoder_profile,
                "ac4_fit_macs": ac4_fit_mac,
                "ac4_sufficient_state_bytes_fp32": ac4_sufficient_state_bytes,
                "t4_descriptor_bytes_fp32": t4_descriptor_bytes,
                "peak_stream_calibration_live_state_bytes_fp32": encoder_profile["peak_live_state_bytes"] + ac4_sufficient_state_bytes,
                "persistent_identity_state_bytes_fp32": identity_state,
                "online_macs_per_window": decoder_cost["total"],
                "target_labels_used": True,
                "target_label_scope": "eligible directional labels from first 33 calibration trials only",
            },
        },
        "paired_interpretation": {
            "neural_exposure_matched": True,
            "label_information_matched": False,
            "claim": "supervised-versus-neural-only deployment utility; not an information-matched mechanism comparison",
        },
        "not_yet_measured_for_go": {
            "source_training_forward_backward_macs_by_fold_seed": "requires production dataloader batch cardinalities and backward-count convention",
            "source_training_wall_time": "requires authorized production run",
            "deployment_calibration_wall_time": "requires authorized production run",
            "streaming_inference_wall_time": "requires authorized production run",
            "peak_host_and_device_memory": "requires authorized production run",
        },
        "source_bindings": {
            "spint_model_config": source(ROOT / "SPINT-main/configs/model/falcon_m2_post33_confirm_v4.yaml"),
            "t4_model_config": source(STREAMING / "configs/model/streaming_b3s_t4_post33_exact_v4.yaml"),
            "t4_encoder_source": source(STREAMING / "src/models/components/streaming_encoders.py"),
            "decoder_cost_source": source(STREAMING / "src/models/components/streaming_spint.py"),
            "writer": source(Path(__file__).resolve()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        cursor = 0
        while cursor < len(data):
            cursor += os.write(fd, data[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)
    print(output)
    print(hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
