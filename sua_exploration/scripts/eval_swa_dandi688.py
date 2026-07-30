"""E1: build and evaluate post-hoc SWA (weight-averaged) checkpoints for one DANDI 000688
run trained with ``train_variant_dandi688.py --checkpoint_every_epoch``.

sua_exploration/ROADMAP.md "当前实验计划（2026-07-26 起）" E1: attacks sigma_seed (the
dominant variance component measured in side_feature_ablation_v2, see
sua_exploration/docs/CURRENT_RESULTS.md section I) by averaging trained *weights* -- not
measurements -- across a trailing window of per-epoch checkpoints from one run.

For a given ``--run_dir`` and a set of trailing-window sizes (default 5, 10, 20), this:
  1. builds one merged checkpoint per window via
     ``swa_utils_dandi688.build_swa_checkpoint`` (averages the trailing ``window`` epochs,
     i.e. epochs ``[max_epoch-window+1, max_epoch]``; see that module for the exact
     float-only / teacher-excluded averaging rule);
  2. evaluates each merged checkpoint with the exact same fixed protocol metric as every
     other DANDI 000688 eval script, via
     ``select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions``
     (reused, not reimplemented -- this script never recomputes R2 itself).

This is post-hoc only: it requires no retraining, just the per-epoch checkpoints a
``--checkpoint_every_epoch`` run already saved.

Hard constraint: only ever loads validation-session spike/behavior/trial data (same
discover_nwb_files / chronological_session_split / evaluate_fixed_protocol_over_validation_sessions
path as every other DANDI 000688 eval script here). Test sessions are only ever touched for
their names and NWB unit-table row counts, never spike/behavior/trial data.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dandi688_gradient_free_protocol import sha256_file  # noqa: E402
from eval_epoch_window_dandi688 import (  # noqa: E402
    FIXED_CALIBRATION_N,
    FIXED_POOL_SIZE,
    FIXED_SELECTION_MODE,
)
from select_gradient_free_protocol_dandi688 import (  # noqa: E402
    evaluate_fixed_protocol_over_validation_sessions,
)
from swa_utils_dandi688 import build_swa_checkpoint  # noqa: E402


def parse_positive_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError("windows must be a comma-separated list of positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir", required=True, type=str,
        help="Checkpoint output directory from train_variant_dandi688.py --checkpoint_every_epoch.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--max_epoch", type=int, required=True,
        help="Last trained protocol epoch of this run (e.g. 40); trailing windows are "
        "[max_epoch-window+1, max_epoch].",
    )
    parser.add_argument("--windows", type=str, default="5,10,20")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    windows = parse_positive_int_list(args.windows)
    for window in windows:
        if window > args.max_epoch:
            raise ValueError(f"window {window} exceeds --max_epoch {args.max_epoch}")

    run_dir = Path(args.run_dir).expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint provenance metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "completed":
        raise ValueError(f"run_metadata status must be 'completed', found {metadata.get('status')!r}")
    if metadata.get("held_out_test_evaluated") is not False:
        raise ValueError(
            "run_metadata.held_out_test_evaluated must be false for a validation-only SWA "
            "evaluation"
        )
    training = metadata.get("training", {})
    if training.get("checkpoint_every_epoch") is not True:
        raise ValueError(
            "SWA requires the run to have been trained with --checkpoint_every_epoch "
            f"(found training.checkpoint_every_epoch={training.get('checkpoint_every_epoch')!r})"
        )

    variant = metadata["variant"]
    seed = metadata["seed"]
    task = metadata["task"]
    split_counts = tuple(metadata["split_counts"])
    max_units_exclusive = metadata["max_units_exclusive"]
    signal_view = metadata.get("signal_view", "sua")

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else Path(metadata["data_dir"])
    teacher_ckpt = (
        Path(args.teacher_ckpt).expanduser().resolve() if args.teacher_ckpt
        else Path(metadata["teacher_checkpoint"])
    )
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve() if args.cache_dir
        else (Path(metadata["cache_dir"]) if metadata.get("cache_dir") else None)
    )
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_ckpt}")
    if sha256_file(teacher_ckpt) != metadata["teacher_sha256"]:
        raise ValueError("Teacher checkpoint SHA-256 does not match run_metadata.json")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    device = torch.device(args.device) if args.device else None
    swa_ckpt_dir = run_dir / "swa_ckpts"

    per_window: dict[str, dict] = {}
    session_splits = None
    session_unit_counts = None
    for window in windows:
        epochs = list(range(args.max_epoch - window + 1, args.max_epoch + 1))
        out_ckpt = swa_ckpt_dir / f"swa_last{window}.ckpt"
        build_swa_checkpoint(run_dir, epochs, out_ckpt)
        result = evaluate_fixed_protocol_over_validation_sessions(
            ckpt_path=out_ckpt,
            teacher_ckpt=teacher_ckpt,
            variant=variant,
            data_dir=data_dir,
            task=task,
            split_counts=split_counts,
            max_units_exclusive=max_units_exclusive,
            cache_dir=cache_dir,
            pool_size=FIXED_POOL_SIZE,
            selection_mode=FIXED_SELECTION_MODE,
            calibration_n=FIXED_CALIBRATION_N,
            signal_view=signal_view,
            device=device,
        )
        if session_splits is None:
            session_splits = result["session_splits"]
            session_unit_counts = result["session_unit_counts"]
        elif result["session_splits"] != session_splits:
            raise ValueError(f"Session split drifted at window {window}")
        per_window[str(window)] = {
            "epochs_averaged": epochs,
            "num_epochs_averaged": len(epochs),
            "swa_checkpoint_path": str(out_ckpt.resolve()),
            "swa_checkpoint_sha256": sha256_file(out_ckpt),
            "per_session_r2": result["per_session_r2"],
            "mean_r2": result["mean_r2"],
        }
        print(
            f"window last-{window} (epochs {epochs[0]}-{epochs[-1]}): "
            f"mean_r2={result['mean_r2']:.4f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "purpose": "swa_trailing_window_evaluation",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "run_metadata_path": str(metadata_path.resolve()),
        "run_metadata_sha256": sha256_file(metadata_path),
        "variant": variant,
        "seed": seed,
        "task": task,
        "max_epoch": args.max_epoch,
        "windows": windows,
        "data_dir": str(data_dir),
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_ckpt_sha256": sha256_file(teacher_ckpt),
        "split_counts": list(split_counts),
        "max_units_exclusive": max_units_exclusive,
        "signal_view": signal_view,
        "protocol": {
            "name": "fixed_forward_calibration_protocol_swa_checkpoint",
            "description": (
                "E1 (sua_exploration/ROADMAP.md): weight-average "
                "(swa_utils_dandi688.average_student_state_dicts) the trailing `window` "
                "per-epoch checkpoints of this run into one merged checkpoint, then "
                "evaluate it with the same fixed first/n=30/pool=50 forward-calibration "
                "protocol (evaluate_fixed_protocol_over_validation_sessions, reused not "
                "reimplemented) as every other DANDI 000688 eval script here."
            ),
            "selection_mode": FIXED_SELECTION_MODE,
            "calibration_n": FIXED_CALIBRATION_N,
            "pool_size": FIXED_POOL_SIZE,
            "protocol_metric_source": (
                "select_gradient_free_protocol_dandi688."
                "evaluate_fixed_protocol_over_validation_sessions"
            ),
            "swa_averaging_source": "swa_utils_dandi688.average_student_state_dicts",
        },
        "per_window": per_window,
        "session_splits": session_splits,
        "session_unit_counts": session_unit_counts,
        "calibration_trial_selection_uses_behavior_labels": False,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
    }
    results_dir = Path(__file__).resolve().parents[1] / "results"
    out_path = (
        Path(args.out_path) if args.out_path
        else results_dir / "convergence_swa_v1" / f"swa_s{seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote SWA evaluation: {out_path}")


if __name__ == "__main__":
    main()
