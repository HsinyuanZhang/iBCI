"""Write analytic deployment costs for the fixed-slot NeuronID interface."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_calibration_exp"))


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("Expected a comma-separated list of positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--slots", default="16,32")
    parser.add_argument("--unit-counts", default="38,64,91")
    parser.add_argument(
        "--teacher-ckpt",
        default=(
            "sua_exploration/checkpoints/teacher_mc_maze/"
            "best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
        ),
    )
    parser.add_argument("--state-bytes", type=int, default=2)
    args = parser.parse_args()

    slot_counts = parse_ints(args.slots)
    unit_counts = parse_ints(args.unit_counts)
    if args.state_bytes <= 0:
        raise ValueError("--state-bytes must be positive")

    checkpoint = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    first_weight = state_dict["net.fc_in.0.weight"]
    second_weight = state_dict["net.fc_in.2.weight"]
    model_dim, window_size = first_weight.shape
    if second_weight.shape != (model_dim, model_dim):
        raise ValueError("Unexpected teacher fc_in shape")

    readin_mac_per_token = window_size * model_dim + model_dim * model_dim
    profile = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "analytic_fixed_slot_router_hardware_cost_profile",
        "assumptions": {
            "teacher_checkpoint": str(Path(args.teacher_ckpt).resolve()),
            "window_size": window_size,
            "decoder_model_dim": model_dim,
            "state_precision_bytes": args.state_bytes,
            "router": "calibration-frozen dense N-to-K assignment with FiLM slot state",
            "online_scope": "one W-bin decoder window; calibration encoder/routing derivation excluded",
        },
        "readin_mac_per_token": readin_mac_per_token,
        "unit_count_profiles": {},
    }
    for unit_count in unit_counts:
        variable_readin_mac = unit_count * readin_mac_per_token
        variable_decoder_input_bytes = unit_count * window_size * args.state_bytes
        variable_readin_activation_bytes = unit_count * model_dim * args.state_bytes
        fixed_profiles = {}
        for slot_count in slot_counts:
            router_mac = unit_count * slot_count * window_size
            fixed_readin_mac = slot_count * readin_mac_per_token
            fixed_decoder_input_bytes = slot_count * window_size * args.state_bytes
            fixed_readin_activation_bytes = slot_count * model_dim * args.state_bytes
            routing_state_bytes = (
                unit_count * slot_count
                + 2 * slot_count * window_size
                + slot_count
            ) * args.state_bytes
            fixed_profiles[f"K{slot_count}"] = {
                "router_mac_per_window": router_mac,
                "decoder_readin_mac_per_window": fixed_readin_mac,
                "total_router_plus_readin_mac_per_window": router_mac + fixed_readin_mac,
                "readin_mac_reduction_vs_variable": variable_readin_mac / fixed_readin_mac,
                "decoder_input_bytes": fixed_decoder_input_bytes,
                "readin_activation_bytes": fixed_readin_activation_bytes,
                "calibration_cached_state_bytes": routing_state_bytes,
                "fixed_shape_after_router": [slot_count, window_size],
            }
        profile["unit_count_profiles"][f"N{unit_count}"] = {
            "variable_decoder_input_shape": [unit_count, window_size],
            "variable_decoder_readin_mac_per_window": variable_readin_mac,
            "variable_decoder_input_bytes": variable_decoder_input_bytes,
            "variable_readin_activation_bytes": variable_readin_activation_bytes,
            "fixed_slot_profiles": fixed_profiles,
        }

    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / args.pilot_id
        / "hardware_profile.json"
    )
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
