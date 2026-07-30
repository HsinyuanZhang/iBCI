"""E2: per-epoch convergence curve for one DANDI 000688 run trained with
``--no_early_stopping --checkpoint_every_epoch`` for more than 12 epochs (here, 40).

sua_exploration/ROADMAP.md "当前实验计划（2026-07-26 起）" E2: the F0 baseline's M3 window
(epochs 5-12) still had a within-window slope of +0.0024/epoch at epoch 12 -- the model was
still improving when training stopped. This script evaluates the SAME fixed gradient-free
protocol metric ``eval_epoch_window_dandi688.py`` uses (first / n=30 / pool=50, over the 6
validation sessions) at EVERY epoch in ``[--epoch_start, --epoch_end]`` (default 5..40) via
``select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions``.
That function performs the actual R2 computation and is called once per epoch checkpoint --
it is reused unmodified, never reimplemented, exactly as ``eval_epoch_window_dandi688.py``
already does for its own (fixed 12-epoch, epochs 5-12) window.

This is a separate, new script rather than an edit to ``eval_epoch_window_dandi688.py``:
that script hardcodes ``TOTAL_TRAINING_EPOCHS = 12`` / ``PROTOCOL_EPOCHS = (5..12)`` and
actively rejects (``_validate_run_metadata_for_epoch_window``) any run whose
``training.max_epochs != 12`` -- it is not applicable to a 40-epoch run and per task
constraints must not be modified. The Lightning 0-indexed-vs-1-indexed "protocol epoch"
checkpoint filename convention is reused unmodified from that module's
``epoch_checkpoint_path`` / ``lightning_epoch_index`` so the mapping cannot drift between
the two scripts.

This script only ever produces the RAW per-epoch curve for one run (one seed); slope
computation and cross-seed aggregation live in ``aggregate_convergence_swa_v1.py``, matching
this repo's existing separation between per-run eval scripts and aggregators.

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
    epoch_checkpoint_path,
)
from select_gradient_free_protocol_dandi688 import (  # noqa: E402
    evaluate_fixed_protocol_over_validation_sessions,
)


def validate_run_metadata_for_convergence_curve(metadata: dict, *, expected_max_epochs: int) -> None:
    if metadata.get("status") != "completed":
        raise ValueError(f"run_metadata status must be 'completed', found {metadata.get('status')!r}")
    if metadata.get("held_out_test_evaluated") is not False:
        raise ValueError(
            "run_metadata.held_out_test_evaluated must be false for a validation-only "
            "convergence-curve evaluation"
        )
    training = metadata.get("training", {})
    if training.get("max_epochs") != expected_max_epochs:
        raise ValueError(
            f"expected training.max_epochs == {expected_max_epochs}, found "
            f"{training.get('max_epochs')!r} (pass --expected_max_epochs to override)"
        )
    if training.get("no_early_stopping") is not True:
        raise ValueError(
            "convergence-curve evaluation requires the run to have been trained with "
            f"--no_early_stopping (found training.no_early_stopping="
            f"{training.get('no_early_stopping')!r})"
        )
    if training.get("checkpoint_every_epoch") is not True:
        raise ValueError(
            "convergence-curve evaluation requires the run to have been trained with "
            f"--checkpoint_every_epoch (found training.checkpoint_every_epoch="
            f"{training.get('checkpoint_every_epoch')!r})"
        )


def select_epoch_range_checkpoints(run_dir: Path, epochs: list[int]) -> dict[int, Path]:
    epoch_ckpt_dir = Path(run_dir) / "epoch_ckpts"
    if not epoch_ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"{epoch_ckpt_dir} does not exist. --run_dir must be a checkpoint directory "
            "produced by train_variant_dandi688.py --checkpoint_every_epoch."
        )
    selected: dict[int, Path] = {}
    missing: list[str] = []
    for epoch in epochs:
        path = epoch_checkpoint_path(epoch_ckpt_dir, epoch)
        if path.is_file():
            selected[epoch] = path
        else:
            missing.append(path.name)
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint(s) for requested epoch range {epochs[0]}-{epochs[-1]} in "
            f"{epoch_ckpt_dir}: {missing}. If the run crashed before finishing, this must be "
            "reported, not silently worked around with a shorter range."
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir", required=True, type=str,
        help="Checkpoint output directory from train_variant_dandi688.py --checkpoint_every_epoch.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--epoch_start", type=int, default=5)
    parser.add_argument("--epoch_end", type=int, default=40)
    parser.add_argument(
        "--expected_max_epochs", type=int, default=40,
        help="run_metadata.json training.max_epochs must equal this (integrity check).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()
    if args.epoch_start < 1 or args.epoch_end < args.epoch_start:
        raise ValueError("--epoch_start must be >= 1 and --epoch_end must be >= --epoch_start")

    run_dir = Path(args.run_dir).expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint provenance metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    validate_run_metadata_for_convergence_curve(metadata, expected_max_epochs=args.expected_max_epochs)

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

    epochs = list(range(args.epoch_start, args.epoch_end + 1))
    checkpoints = select_epoch_range_checkpoints(run_dir, epochs)
    device = torch.device(args.device) if args.device else None

    per_epoch: dict[str, dict] = {}
    session_splits = None
    session_unit_counts = None
    for epoch in epochs:
        ckpt_path = checkpoints[epoch]
        result = evaluate_fixed_protocol_over_validation_sessions(
            ckpt_path=ckpt_path,
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
            raise ValueError(
                f"Session split drifted at epoch {epoch}; data_dir or split_counts may have "
                "changed mid-run"
            )
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(ckpt_path.resolve()),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "per_session_r2": result["per_session_r2"],
            "mean_r2": result["mean_r2"],
        }
        print(f"epoch {epoch:3d}: mean_r2={result['mean_r2']:.4f}", flush=True)

    per_epoch_mean_r2 = {str(epoch): per_epoch[str(epoch)]["mean_r2"] for epoch in epochs}

    payload = {
        "schema_version": 1,
        "purpose": "convergence_curve_epoch_5_to_40",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "run_metadata_path": str(metadata_path.resolve()),
        "run_metadata_sha256": sha256_file(metadata_path),
        "variant": variant,
        "seed": seed,
        "task": task,
        "data_dir": str(data_dir),
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_ckpt_sha256": sha256_file(teacher_ckpt),
        "split_counts": list(split_counts),
        "max_units_exclusive": max_units_exclusive,
        "signal_view": signal_view,
        "protocol": {
            "name": "fixed_forward_calibration_protocol_full_epoch_range",
            "description": (
                "E2 (sua_exploration/ROADMAP.md): same fixed first/n=30/pool=50 "
                "forward-calibration protocol as M3's eval_epoch_window_dandi688.py "
                "(evaluate_fixed_protocol_over_validation_sessions, reused not "
                "reimplemented), evaluated at every epoch in epoch_start..epoch_end instead "
                "of only the trailing 8-epoch M3 window."
            ),
            "expected_max_epochs": args.expected_max_epochs,
            "epoch_start": args.epoch_start,
            "epoch_end": args.epoch_end,
            "selection_mode": FIXED_SELECTION_MODE,
            "calibration_n": FIXED_CALIBRATION_N,
            "pool_size": FIXED_POOL_SIZE,
            "protocol_metric_source": (
                "select_gradient_free_protocol_dandi688."
                "evaluate_fixed_protocol_over_validation_sessions"
            ),
        },
        "epoch_list": epochs,
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": per_epoch_mean_r2,
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
        else results_dir / "convergence_swa_v1" / f"curve_s{seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote convergence curve: {out_path}")


if __name__ == "__main__":
    main()
