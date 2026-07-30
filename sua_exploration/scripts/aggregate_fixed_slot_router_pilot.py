"""Aggregate the validation-only NeuronID fixed-slot feasibility pilot."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


REFERENCE_B15P_MEAN_R2 = 0.3598937069
CONFIG_NAME = "gradient_free_calibrated_first_n30"
EXPECTED_CASES = {
    (16, 42),
    (16, 43),
    (32, 42),
    (32, 43),
}


def load_router_diagnostics(result_dir: Path, records: dict[tuple[int, int], dict]) -> dict:
    diagnostics: dict[tuple[int, int], dict] = {}
    missing = []
    for slot_count, seed in sorted(EXPECTED_CASES):
        path = result_dir / f"router_diagnostic_k{slot_count}_soft_s{seed}.json"
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
        if payload.get("ckpt") != records[(slot_count, seed)]["ckpt"]:
            raise ValueError(f"{path}: checkpoint mismatch")
        diagnostics[(slot_count, seed)] = payload
    if missing:
        return {"complete": False, "missing_artifacts": missing}

    metrics = (
        "mean_assignment_normalized_entropy",
        "mean_assignment_max_probability",
        "slot_mass_coefficient_of_variation",
        "effective_slot_count_from_mass",
    )
    return {
        "complete": True,
        "protocol": {"selection_mode": "first", "calibration_n": 30, "pool_size": 50},
        "by_slot_count": {
            f"K{slot_count}": {
                metric: sum(
                    diagnostics[(slot_count, seed)]["mean_across_sessions"][metric]
                    for seed in (42, 43)
                ) / 2.0
                for metric in metrics
            }
            for slot_count in (16, 32)
        },
    }


def load_cached_decode_verifications(result_dir: Path, records: dict[tuple[int, int], dict]) -> dict:
    verifications: dict[tuple[int, int], dict] = {}
    missing = []
    for slot_count, seed in sorted(EXPECTED_CASES):
        path = result_dir / f"cached_decode_k{slot_count}_soft_s{seed}.json"
        if not path.is_file():
            missing.append(path.name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("purpose") != "validation_only_spike_only_fixed_slot_cached_decode_verification":
            raise ValueError(f"{path}: unexpected cached-decode verification purpose")
        if payload.get("no_test_files_accessed") is not True:
            raise ValueError(f"{path}: test-session isolation contract failed")
        if payload.get("behavior_data_read") is not False or payload.get("behavior_labels_used") is not False:
            raise ValueError(f"{path}: cached decode must not read behavior")
        if payload.get("weights_updated") is not False or payload.get("backward_gradients_used") is not False:
            raise ValueError(f"{path}: cached decode must keep weights frozen")
        if payload.get("decoder_outputs_evaluated") is not True:
            raise ValueError(f"{path}: cached decode must evaluate decoder equivalence")
        if payload.get("ckpt") != records[(slot_count, seed)]["ckpt"]:
            raise ValueError(f"{path}: checkpoint mismatch")
        if payload.get("all_sessions_match") is not True:
            raise ValueError(f"{path}: cached decode does not match normal forward")
        verifications[(slot_count, seed)] = payload
    if missing:
        return {"complete": False, "missing_artifacts": missing}

    return {
        "complete": True,
        "protocol": {
            "selection_mode": "first",
            "calibration_n": 30,
            "pool_size": 50,
            "windows_per_session": 8,
        },
        "by_slot_count": {
            f"K{slot_count}": {
                "max_absolute_difference": max(
                    verifications[(slot_count, seed)]["max_absolute_difference_across_sessions"]
                    for seed in (42, 43)
                ),
                "verified_session_count": sum(
                    len(verifications[(slot_count, seed)]["per_session"]) for seed in (42, 43)
                ),
            }
            for slot_count in (16, 32)
        },
    }


def parse_case_name(path: Path) -> tuple[int, int]:
    stem = path.stem
    parts = stem.split("_")
    slot_part = next(part for part in parts if part.startswith("k"))
    seed_part = next(part for part in parts if part.startswith("s") and part[1:].isdigit())
    return int(slot_part.removeprefix("k")), int(seed_part.removeprefix("s"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-id", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results" / args.pilot_id
    records: dict[tuple[int, int], dict] = {}
    for path in sorted(result_dir.glob("fsr_k*_soft_s*.json")):
        slot_count, seed = parse_case_name(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("purpose") != "validation_only_fixed_gradient_free_evaluation":
            raise ValueError(f"{path}: unexpected evaluation purpose")
        if payload.get("no_test_files_evaluated") is not True:
            raise ValueError(f"{path}: test-session isolation contract failed")
        if payload.get("selected_protocol", {}).get("selection_mode") != "first":
            raise ValueError(f"{path}: unexpected calibration selection mode")
        if payload.get("selected_protocol", {}).get("calibration_n") != 30:
            raise ValueError(f"{path}: unexpected calibration count")
        if payload.get("selected_protocol", {}).get("pool_size") != 50:
            raise ValueError(f"{path}: unexpected calibration pool")
        metadata_path = Path(payload["training_run_metadata"])
        if not metadata_path.is_file():
            raise ValueError(f"{path}: missing training metadata {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fixed_slot = metadata.get("fixed_slot", {})
        if metadata.get("status") != "completed":
            raise ValueError(f"{path}: training run did not complete")
        if metadata.get("seed") != seed or payload.get("seed") != seed:
            raise ValueError(f"{path}: seed provenance mismatch")
        if metadata.get("variant") != "B3" or payload.get("variant") != "B3":
            raise ValueError(f"{path}: unexpected encoder base variant")
        if fixed_slot != {
            "enabled": True,
            "count": slot_count,
            "router_dim": 32,
            "routing_mode": "soft",
            "fusion": "film",
            "temperature": 1.0,
            "deployment_contract": (
                "Calibration derives session-local routing; online decoding projects all "
                "variable unit windows into fixed slot_count tokens before decoder.fc_in."
            ),
        }:
            raise ValueError(f"{path}: fixed-slot provenance mismatch")
        records[(slot_count, seed)] = payload

    if set(records) != EXPECTED_CASES:
        missing = sorted(EXPECTED_CASES - set(records))
        extra = sorted(set(records) - EXPECTED_CASES)
        raise ValueError(f"Pilot artifacts incomplete; missing={missing}, extra={extra}")

    summary: dict[str, dict] = {}
    for slot_count in (16, 32):
        per_seed = {
            str(seed): records[(slot_count, seed)]["mean_r2"][CONFIG_NAME]
            for seed in (42, 43)
        }
        per_session = {}
        for session_name in records[(slot_count, 42)]["per_session_r2"]:
            per_session[session_name] = sum(
                records[(slot_count, seed)]["per_session_r2"][session_name][CONFIG_NAME]
                for seed in (42, 43)
            ) / 2.0
        mean_r2 = sum(per_seed.values()) / len(per_seed)
        summary[f"K{slot_count}"] = {
            "mean_r2": mean_r2,
            "per_seed_mean_r2": per_seed,
            "per_session_seed_mean_r2": per_session,
            "delta_vs_b15p_reference": mean_r2 - REFERENCE_B15P_MEAN_R2,
            "positive_sessions": sum(value > 0.0 for value in per_session.values()),
            "pilot_accuracy_feasible": (
                all(value > 0.0 for value in per_seed.values())
                and mean_r2 >= REFERENCE_B15P_MEAN_R2 - 0.03
            ),
            "readin_token_reduction_at_n64": 64 / slot_count,
        }

    cached_decode_verification = load_cached_decode_verifications(result_dir, records)
    aggregate = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "validation_only_fixed_slot_router_feasibility_pilot",
        "pilot_id": args.pilot_id,
        "no_formal_test_sessions_evaluated": True,
        "reference": {
            "variant": "B15P",
            "mean_r2": REFERENCE_B15P_MEAN_R2,
            "source": "attention_arch_screen_v3 aggregate",
        },
        "protocol": {
            "selection_mode": "first",
            "calibration_n": 30,
            "pool_size": 50,
            "evaluation_trials": "trials[50:]",
        },
        "results": summary,
        "router_diagnostics": load_router_diagnostics(result_dir, records),
        "cached_decode_verification": cached_decode_verification,
        "gates": {
            "k32_accuracy_feasible": summary["K32"]["pilot_accuracy_feasible"],
            "k16_accuracy_feasible": summary["K16"]["pilot_accuracy_feasible"],
            "advance_to_hard_routing_controls": summary["K32"]["pilot_accuracy_feasible"],
            "cached_deployment_equivalent": cached_decode_verification.get("complete") is True,
        },
    }
    output_path = result_dir / "aggregate.json"
    output_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
