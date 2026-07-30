"""Verify cached fixed-slot decoding on validation spikes without behavior labels."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dandi688_gradient_free_protocol import (
    select_calibration_trial_indices,
    sha256_file,
    validate_training_run_metadata,
)
from diagnose_fixed_slot_router import load_spike_only_record
from eval_adaptation_dandi688 import (
    DEFAULT_TEACHER,
    WINDOW_SIZE,
    build_calib_trials_for_indices,
    parse_split_counts,
)
from mc_maze.multisession_datamodule import chronological_session_split, discover_nwb_files
from select_gradient_free_protocol_dandi688 import load_frozen_model


def collect_online_windows(record: dict, pool_size: int, windows_per_session: int) -> np.ndarray:
    """Return fixed-width spike windows after the shared calibration pool."""
    windows: list[np.ndarray] = []
    for trial in record["trials"][pool_size:]:
        start = int(trial["start"])
        stop = int(trial["stop"])
        if stop - start < WINDOW_SIZE:
            continue
        windows.append(record["neural"][start : start + WINDOW_SIZE])
        if len(windows) == windows_per_session:
            break
    if not windows:
        raise ValueError(f"{record['name']}: no usable online spike windows after calibration pool")
    return np.stack(windows).astype(np.float32, copy=False)


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
    parser.add_argument("--windows_per_session", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-5,
        help="FP32 absolute tolerance for normal versus cached batched matmuls.",
    )
    parser.add_argument("--out_path", required=True)
    args = parser.parse_args()

    if args.calibration_n <= 0 or args.pool_size < args.calibration_n:
        raise ValueError("Require 0 < calibration_n <= pool_size")
    if args.windows_per_session <= 0:
        raise ValueError("--windows_per_session must be positive")
    if args.atol <= 0.0:
        raise ValueError("--atol must be positive")

    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    teacher_checkpoint = Path(args.teacher_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.out_path).expanduser().resolve()
    if not checkpoint_path.is_file() or not teacher_checkpoint.is_file():
        raise FileNotFoundError("--ckpt and --teacher_ckpt must name existing files")

    split_counts = parse_split_counts(args.split_counts)
    metadata_path, metadata = validate_training_run_metadata(
        checkpoint_path,
        teacher_checkpoint,
        args.variant,
        data_dir,
        args.task,
        split_counts,
        args.max_units_exclusive,
        args.seed,
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

    device = torch.device("cpu")
    model = load_frozen_model(checkpoint_path, teacher_checkpoint, args.variant, device)
    per_session: dict[str, dict[str, float | int | list[int] | list[list[int]]]] = {}
    with torch.no_grad():
        for path in validation_files:
            record = load_spike_only_record(path, bin_size_ms=20)
            calibration_indices = select_calibration_trial_indices(
                record["trials"], args.calibration_n, args.pool_size, args.selection_mode
            )
            calibration_trials = torch.from_numpy(
                build_calib_trials_for_indices(record, calibration_indices, args.calibration_n)
            ).unsqueeze(0)
            identity = model.student.compute_identity(calibration_trials)
            calibration_state = model.student.derive_fixed_slot_state(
                identity, record["n_units"]
            )
            neural_windows = torch.from_numpy(
                collect_online_windows(record, args.pool_size, args.windows_per_session)
            )
            repeated_identity = identity.expand(neural_windows.shape[0], -1, -1)
            normal_output = model.student.decode_with_identity(neural_windows, repeated_identity)
            cached_output = model.student.decode_with_fixed_slot_state(
                neural_windows, calibration_state
            )
            difference = (normal_output - cached_output).abs()
            max_absolute_difference = float(difference.max().item())
            per_session[record["name"]] = {
                "unit_count": record["n_units"],
                "online_window_count": int(neural_windows.shape[0]),
                "online_neural_shape": list(neural_windows.shape),
                "fixed_decoder_input_shape": [
                    int(neural_windows.shape[0]),
                    int(calibration_state["gain"].shape[1]),
                    int(calibration_state["gain"].shape[2]),
                ],
                "decoder_output_shape": list(cached_output.shape),
                "max_absolute_difference": max_absolute_difference,
                "cached_path_matches_normal_forward": bool(
                    torch.allclose(normal_output, cached_output, atol=args.atol, rtol=0.0)
                ),
            }

    per_session_maxima = [record["max_absolute_difference"] for record in per_session.values()]
    output_payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "validation_only_spike_only_fixed_slot_cached_decode_verification",
        "ckpt": str(checkpoint_path),
        "ckpt_sha256": sha256_file(checkpoint_path),
        "teacher_ckpt": str(teacher_checkpoint),
        "teacher_ckpt_sha256": sha256_file(teacher_checkpoint),
        "training_run_metadata": str(metadata_path),
        "training_run_metadata_sha256": sha256_file(metadata_path),
        "fixed_slot": fixed_slot,
        "device": str(device),
        "session_split": "validation",
        "session_names": list(per_session),
        "no_test_files_accessed": True,
        "behavior_data_read": False,
        "behavior_labels_used": False,
        "decoder_outputs_evaluated": True,
        "weights_updated": False,
        "backward_gradients_used": False,
        "protocol": {
            "selection_mode": args.selection_mode,
            "calibration_n": args.calibration_n,
            "pool_size": args.pool_size,
            "windows_per_session": args.windows_per_session,
            "spike_bin_size_ms": 20,
            "window_size": WINDOW_SIZE,
            "comparison": "normal forward with repeated identity versus batch-1 cached state",
            "absolute_tolerance": args.atol,
        },
        "per_session": per_session,
        "max_absolute_difference_across_sessions": float(max(per_session_maxima)),
        "all_sessions_match": bool(
            all(record["cached_path_matches_normal_forward"] for record in per_session.values())
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
