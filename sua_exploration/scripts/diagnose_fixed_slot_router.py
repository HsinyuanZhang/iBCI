"""Measure calibration-derived fixed-slot routing without decoding behavior.

This diagnostic deliberately reads only SUA spike times and rewarded-trial
metadata.  It never reads cursor behavior, evaluates no decoder outputs, and
does not access formal held-out test sessions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pynwb import NWBHDF5IO


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dandi688_gradient_free_protocol import (
    select_calibration_trial_indices,
    sha256_file,
    validate_training_run_metadata,
)
from eval_adaptation_dandi688 import (
    DEFAULT_TEACHER,
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    build_calib_trials_for_indices,
    parse_split_counts,
)
from mc_maze.datamodule import bin_spikes
from mc_maze.multisession_datamodule import (
    chronological_session_split,
    discover_nwb_files,
    session_name_from_path,
)
from select_gradient_free_protocol_dandi688 import load_frozen_model


def load_spike_only_record(path: Path, bin_size_ms: int) -> dict:
    """Build the calibration inputs without opening the behavior stream."""
    bin_size_s = bin_size_ms / 1000.0
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        all_spikes = np.concatenate(units["spike_times"].values)
        bin_edges = np.arange(
            float(all_spikes.min()), float(all_spikes.max()) + bin_size_s, bin_size_s
        )
        binned_spikes = np.zeros((len(bin_edges) - 1, len(units)), dtype=np.float32)
        for unit_index, (_, unit) in enumerate(units.iterrows()):
            binned_spikes[:, unit_index] = bin_spikes(unit["spike_times"], bin_edges)

        trials = []
        for _, trial in nwb.intervals["trials"].to_dataframe().iterrows():
            if trial["result"] != "R":
                continue
            start = max(0, int(np.searchsorted(bin_edges, trial["start_time"])))
            stop = min(len(binned_spikes), int(np.searchsorted(bin_edges, trial["stop_time"])))
            if stop - start >= WINDOW_SIZE:
                trials.append(
                    {
                        "start": start,
                        "stop": stop,
                        "trial_index": int(trial.name),
                        "target_dir": trial.get("target_dir"),
                        "target_id": trial.get("target_id"),
                    }
                )
    return {
        "name": session_name_from_path(path),
        "n_units": len(units),
        "neural": binned_spikes,
        "trials": trials,
    }


def summarize_state(calibration_state: dict[str, torch.Tensor]) -> dict[str, float | int]:
    assignment = calibration_state["assignment"].squeeze(0)
    slot_mass = calibration_state["slot_mass"].squeeze(0)
    slot_count = assignment.shape[1]
    assignment_entropy = -(assignment * assignment.clamp_min(1.0e-12).log()).sum(dim=-1)
    normalized_entropy = assignment_entropy / math.log(slot_count)
    mass_distribution = slot_mass / slot_mass.sum().clamp_min(1.0e-12)
    effective_slots = torch.exp(
        -(mass_distribution * mass_distribution.clamp_min(1.0e-12).log()).sum()
    )
    return {
        "unit_count": int(assignment.shape[0]),
        "slot_count": int(slot_count),
        "mean_assignment_normalized_entropy": float(normalized_entropy.mean().item()),
        "mean_assignment_max_probability": float(assignment.max(dim=-1).values.mean().item()),
        "slot_mass_min": float(slot_mass.min().item()),
        "slot_mass_max": float(slot_mass.max().item()),
        "slot_mass_coefficient_of_variation": float(
            (slot_mass.std(unbiased=False) / slot_mass.mean().clamp_min(1.0e-12)).item()
        ),
        "effective_slot_count_from_mass": float(effective_slots.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--teacher_ckpt", default=str(DEFAULT_TEACHER))
    parser.add_argument("--variant", default="B3", choices=["B3", "B15P", "B15D", "B15", "B16"])
    parser.add_argument("--data_dir", default="sua_exploration/data/dandi_000688/sub-C")
    parser.add_argument("--task", default="CO", choices=["CO", "RT"])
    parser.add_argument("--split_counts", default="27,6,6")
    parser.add_argument("--max_units_exclusive", type=int, default=100)
    parser.add_argument("--calibration_n", type=int, default=30)
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--selection_mode", choices=["first", "direction_coverage"], default="first")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    if args.calibration_n <= 0 or args.pool_size < args.calibration_n:
        raise ValueError("Require 0 < calibration_n <= pool_size")
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    if not ckpt_path.is_file() or not teacher_ckpt.is_file():
        raise FileNotFoundError("--ckpt and --teacher_ckpt must name existing files")

    split_counts = parse_split_counts(args.split_counts)
    metadata_path, metadata = validate_training_run_metadata(
        ckpt_path, teacher_ckpt, args.variant, data_dir, args.task,
        split_counts, args.max_units_exclusive, args.seed,
    )
    fixed_slot = metadata.get("fixed_slot", {})
    if not fixed_slot.get("enabled"):
        raise ValueError("Checkpoint does not contain a fixed-slot router")

    all_files = discover_nwb_files(data_dir, args.task, args.max_units_exclusive)
    _, validation_files, _ = chronological_session_split(
        all_files, split_counts, max_units_exclusive=args.max_units_exclusive
    )
    if not validation_files:
        raise ValueError("No validation sessions selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(ckpt_path, teacher_ckpt, args.variant, device)
    per_session: dict[str, dict[str, float | int | list[int]]] = {}
    with torch.no_grad():
        for path in validation_files:
            record = load_spike_only_record(path, bin_size_ms=20)
            indices = select_calibration_trial_indices(
                record["trials"], args.calibration_n, args.pool_size, args.selection_mode
            )
            calibration_trials = build_calib_trials_for_indices(
                record, indices, args.calibration_n
            )
            identity = model.student.compute_identity(
                torch.from_numpy(calibration_trials).unsqueeze(0).to(device)
            )
            summary = summarize_state(
                model.student.derive_fixed_slot_state(identity, record["n_units"])
            )
            summary["usable_trial_list_indices"] = indices
            summary["original_trial_indices"] = [
                int(record["trials"][index]["trial_index"]) for index in indices
            ]
            per_session[record["name"]] = summary

    metric_names = [
        "mean_assignment_normalized_entropy",
        "mean_assignment_max_probability",
        "slot_mass_coefficient_of_variation",
        "effective_slot_count_from_mass",
    ]
    summary = {
        metric: float(sum(row[metric] for row in per_session.values()) / len(per_session))
        for metric in metric_names
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "validation_only_spike_only_fixed_slot_routing_diagnostic",
        "ckpt": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_ckpt_sha256": sha256_file(teacher_ckpt),
        "training_run_metadata": str(metadata_path),
        "training_run_metadata_sha256": sha256_file(metadata_path),
        "fixed_slot": fixed_slot,
        "device": str(device),
        "session_split": "validation",
        "session_names": list(per_session),
        "no_test_files_accessed": True,
        "behavior_data_read": False,
        "behavior_labels_used": False,
        "decoder_outputs_evaluated": False,
        "weights_updated": False,
        "backward_gradients_used": False,
        "protocol": {
            "selection_mode": args.selection_mode,
            "calibration_n": args.calibration_n,
            "pool_size": args.pool_size,
            "spike_bin_size_ms": 20,
            "trial_length": TRIAL_LENGTH,
            "pad_value": PAD_VALUE,
        },
        "per_session": per_session,
        "mean_across_sessions": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
