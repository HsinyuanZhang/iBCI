"""Aggregate the low-temperature K=32 fixed-slot router pilot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


CONFIG_NAME = "gradient_free_calibrated_first_n30"
REFERENCE_B15P_MEAN_R2 = 0.3598937069


def load_router_diagnostics(result_dir: Path, records: dict[int, dict]) -> dict:
    diagnostics = {}
    missing = []
    for seed in (42, 43):
        path = result_dir / f"router_diagnostic_k32_soft_t010_s{seed}.json"
        if not path.is_file():
            missing.append(path.name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("purpose") != "validation_only_spike_only_fixed_slot_routing_diagnostic":
            raise ValueError(f"{path}: unexpected diagnostic purpose")
        if payload.get("no_test_files_accessed") is not True:
            raise ValueError(f"{path}: test-session isolation contract failed")
        if payload.get("behavior_data_read") is not False or payload.get("behavior_labels_used") is not False:
            raise ValueError(f"{path}: diagnostic must not read behavior")
        if payload.get("decoder_outputs_evaluated") is not False:
            raise ValueError(f"{path}: diagnostic must not evaluate decoder outputs")
        if payload.get("ckpt") != records[seed]["ckpt"]:
            raise ValueError(f"{path}: checkpoint mismatch")
        diagnostics[seed] = payload
    if missing:
        return {"complete": False, "missing_artifacts": missing}

    return {
        "complete": True,
        "protocol": {"selection_mode": "first", "calibration_n": 30, "pool_size": 50},
        "mean_across_seeds": {
            metric: sum(
                diagnostics[seed]["mean_across_sessions"][metric] for seed in (42, 43)
            ) / 2.0
            for metric in (
                "mean_assignment_normalized_entropy",
                "mean_assignment_max_probability",
                "slot_mass_coefficient_of_variation",
                "effective_slot_count_from_mass",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-id", required=True)
    args = parser.parse_args()
    result_dir = Path(__file__).resolve().parents[1] / "results" / args.pilot_id
    records = {}
    for seed in (42, 43):
        path = result_dir / f"fsr_k32_soft_t010_s{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pilot result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata_path = Path(payload["training_run_metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fixed_slot = metadata.get("fixed_slot", {})
        if payload.get("purpose") != "validation_only_fixed_gradient_free_evaluation":
            raise ValueError(f"{path}: unexpected evaluation purpose")
        if payload.get("no_test_files_evaluated") is not True:
            raise ValueError(f"{path}: test-session isolation failed")
        if payload.get("selected_protocol", {}) != {
            "selection_mode": "first",
            "calibration_n": 30,
            "pool_size": 50,
            "validation_mean_r2": payload["mean_r2"][CONFIG_NAME],
            "validation_paired_delta_vs_zero_identity_no_calibration": (
                payload["mean_paired_delta_vs_zero_identity_no_calibration"][CONFIG_NAME]
            ),
        }:
            raise ValueError(f"{path}: validation protocol mismatch")
        if metadata.get("status") != "completed" or metadata.get("seed") != seed:
            raise ValueError(f"{path}: incomplete or mismatched training metadata")
        if fixed_slot.get("enabled") is not True or fixed_slot.get("count") != 32:
            raise ValueError(f"{path}: fixed-slot count mismatch")
        if fixed_slot.get("router_dim") != 32 or fixed_slot.get("routing_mode") != "soft":
            raise ValueError(f"{path}: router architecture mismatch")
        if fixed_slot.get("fusion") != "film" or fixed_slot.get("temperature") != 0.1:
            raise ValueError(f"{path}: router temperature/fusion mismatch")
        records[seed] = payload

    per_seed = {str(seed): records[seed]["mean_r2"][CONFIG_NAME] for seed in (42, 43)}
    per_session = {
        session: sum(records[seed]["per_session_r2"][session][CONFIG_NAME] for seed in (42, 43)) / 2.0
        for session in records[42]["per_session_r2"]
    }
    mean_r2 = sum(per_seed.values()) / 2.0
    aggregate = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "validation_only_low_temperature_fixed_slot_router_pilot",
        "pilot_id": args.pilot_id,
        "no_formal_test_sessions_evaluated": True,
        "protocol": {"selection_mode": "first", "calibration_n": 30, "pool_size": 50},
        "architecture": {"encoder": "B3", "slot_count": 32, "routing": "soft", "fusion": "film", "temperature": 0.1},
        "mean_r2": mean_r2,
        "per_seed_mean_r2": per_seed,
        "per_session_seed_mean_r2": per_session,
        "delta_vs_b15p_reference": mean_r2 - REFERENCE_B15P_MEAN_R2,
        "accuracy_feasible": all(value > 0.0 for value in per_seed.values()) and mean_r2 >= REFERENCE_B15P_MEAN_R2 - 0.03,
        "router_diagnostics": load_router_diagnostics(result_dir, records),
    }
    output = result_dir / "aggregate.json"
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
