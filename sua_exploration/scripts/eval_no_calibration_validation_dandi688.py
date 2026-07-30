"""Validation-only no-calibration control for DANDI 000688."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dandi688_gradient_free_protocol import sha256_file, validate_training_run_metadata
from eval_adaptation_dandi688 import (
    DEFAULT_TEACHER,
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    eval_r2_with_learned_prior,
    eval_r2_with_zero_identity,
    load_session_with_trials,
    make_subset_dataset,
    parse_split_counts,
)
from select_gradient_free_protocol_dandi688 import load_frozen_model
from mc_maze.multisession_datamodule import (
    chronological_session_split, discover_nwb_files, fit_behavior_stats,
    nwb_unit_count, session_name_from_path,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--teacher_ckpt", default=str(DEFAULT_TEACHER))
    parser.add_argument("--variant", default="B3", choices=["B3", "B15P", "B15D", "B15", "B16"])
    parser.add_argument("--data_dir", default="sua_exploration/data/dandi_000688/sub-C")
    parser.add_argument("--task", default="CO", choices=["CO", "RT"])
    parser.add_argument("--split_counts", default="27,6,6")
    parser.add_argument("--max_units_exclusive", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument(
        "--control_mode",
        type=str,
        default="zero_identity",
        choices=["zero_identity", "learned_prior"],
        help="No-calibration control mode.",
    )
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()
    if args.pool_size <= 0 or (args.max_sessions is not None and args.max_sessions <= 0):
        raise ValueError("pool_size and max_sessions must be positive")
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    if not ckpt_path.is_file() or not teacher_ckpt.is_file():
        raise FileNotFoundError("--ckpt and --teacher_ckpt must name existing files")
    split_counts = parse_split_counts(args.split_counts)
    data_dir = Path(args.data_dir).expanduser().resolve()
    metadata_path, _ = validate_training_run_metadata(
        ckpt_path, teacher_ckpt, args.variant, data_dir, args.task, split_counts,
        args.max_units_exclusive, args.seed,
    )
    all_files = discover_nwb_files(data_dir, args.task, args.max_units_exclusive)
    train_files, val_files, test_files = chronological_session_split(
        all_files, split_counts, max_units_exclusive=args.max_units_exclusive
    )
    validation_complete = args.max_sessions is None
    if args.max_sessions is not None:
        val_files = val_files[:args.max_sessions]
    mean, std = fit_behavior_stats(train_files, 20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(
        ckpt_path,
        teacher_ckpt,
        args.variant,
        device,
        identity_mode="learned_prior" if args.control_mode == "learned_prior" else "calibrated",
    )
    per_session: dict[str, float] = {}
    window_counts: dict[str, int] = {}
    with torch.no_grad():
        for path in val_files:
            record = load_session_with_trials(path, 20, WINDOW_SIZE, args.pool_size,
                                              TRIAL_LENGTH, PAD_VALUE, mean, std)
            if len(record["trials"]) <= args.pool_size:
                raise ValueError(
                    f"{record['name']}: no evaluation trial remains after pool_size={args.pool_size}"
                )
            evaluation_trials = record["trials"][args.pool_size:]
            dataset = make_subset_dataset(record, evaluation_trials, record["name"])
            if not len(dataset):
                raise ValueError(f"{record['name']}: no usable windows after pool_size={args.pool_size}")
            if args.control_mode == "zero_identity":
                per_session[record["name"]] = eval_r2_with_zero_identity(model, dataset, device)
            elif args.control_mode == "learned_prior":
                per_session[record["name"]] = eval_r2_with_learned_prior(model, dataset, device)
            else:
                raise ValueError(f"Unsupported control_mode: {args.control_mode}")
            window_counts[record["name"]] = len(dataset)
    scores = list(per_session.values())
    if not scores:
        raise ValueError("No validation sessions produced no-calibration scores")
    payload = {
        "schema_version": 1,
        "purpose": "validation_only_no_calibration",
        "created_at": datetime.now().astimezone().isoformat(),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_ckpt_sha256": sha256_file(teacher_ckpt),
        "training_run_metadata": str(metadata_path),
        "training_run_metadata_sha256": sha256_file(metadata_path),
        "variant": args.variant,
        "task": args.task,
        "seed": args.seed,
        "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "pool_size": args.pool_size,
        "control_mode": args.control_mode,
        "common_evaluation_start_index": args.pool_size,
        "session_splits": {
            "train": [session_name_from_path(p) for p in train_files],
            "val": [session_name_from_path(p) for p in val_files],
            "test": [session_name_from_path(p) for p in test_files],
        },
        "session_unit_counts": {session_name_from_path(p): nwb_unit_count(p) for p in all_files},
        "per_session_r2": per_session,
        "mean_r2": sum(scores) / len(scores),
        "median_r2": float(statistics.median(scores)),
        "positive_session_count": sum(score > 0 for score in scores),
        "window_counts": window_counts,
        "validation_complete": validation_complete,
        "no_test_files_evaluated": True,
        "calibration_spikes_used_by_model": False,
        "identity_encoder_called": False,
        "evaluation_trials_reserved_after_pool": True,
        "no_calibration_control_description": (
            "non-learned all-zero identity control"
            if args.control_mode == "zero_identity"
            else "learned population-prior identity control"
        ),
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if args.control_mode == "zero_identity":
        default_name = f"p3_no_calibration_validation_{args.variant.lower()}_s{args.seed}.json"
    else:
        default_name = (
            f"p3_no_calibration_validation_{args.variant.lower()}_{args.control_mode}_s{args.seed}.json"
        )
    out_path = Path(args.out_path) if args.out_path else Path(__file__).resolve().parents[1] / "results" / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Validation-only no-calibration result: {out_path}")


if __name__ == "__main__":
    main()
