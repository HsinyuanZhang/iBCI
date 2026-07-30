"""M3: deterministic, pre-declared epoch-window checkpoint scoring for DANDI 000688 SUA runs.

Replaces selecting a checkpoint by argmax over a noisy per-epoch validation metric
(sua_exploration/docs/CURRENT_RESULTS.md section H.1-H.3: epoch-to-epoch sigma=0.0388 on
SUA against measured effects of 0.006-0.04, plus an unequal max-of-N selection bias from
variants training different numbers of epochs). Instead:

1. The run must have trained exactly ``TOTAL_TRAINING_EPOCHS`` (12) epochs with
   ``--no_early_stopping`` and saved a checkpoint at every epoch with
   ``--checkpoint_every_epoch`` (both flags added to train_variant_dandi688.py for M2/M3).
2. The variant score is the unweighted mean of the protocol metric evaluated at epochs
   5,6,7,8,9,10,11,12 (an 8-epoch trailing average with a 4-epoch burn-in) -- see
   ``PROTOCOL_EPOCHS`` below. These values are hardcoded, not CLI-configurable, because
   the whole point of a pre-declared estimator is that it cannot be tuned after the fact.
3. The protocol metric is the existing fixed forward-calibration evaluation
   (first / n=30 / pool=50) over the 6 validation sessions, computed by
   ``select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions``
   -- this script calls that function once per epoch checkpoint; it does not
   reimplement the R2 computation.

Hard constraint: this script only ever loads validation-session spike/behavior/trial
data (via the same ``discover_nwb_files`` / ``chronological_session_split`` /
``evaluate_fixed_protocol_over_validation_sessions`` path used by
select_gradient_free_protocol_dandi688.py). Test sessions are only ever touched for
their names and NWB unit-table row counts (via ``nwb_unit_count``), never spike,
behavior, or trial data -- the one access this repo's protocol documents as permitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dandi688_gradient_free_protocol import sha256_file
from select_gradient_free_protocol_dandi688 import (
    evaluate_fixed_protocol_over_validation_sessions,
)

# --- Pre-declared estimator (hardcoded; not CLI-overridable). ---------------------
# Human/1-indexed "protocol epoch" N means "the checkpoint saved after N completed
# training epochs" (i.e. after the Nth pass through the training data; protocol epoch 1
# is the first completed epoch, protocol epoch 12 is the last of a 12-epoch run). Every
# JSON field this script writes (epoch_list, per_epoch keys, etc.) uses this convention.
TOTAL_TRAINING_EPOCHS = 12
BURN_IN_EPOCHS = 4
PROTOCOL_EPOCHS: tuple[int, ...] = tuple(range(BURN_IN_EPOCHS + 1, TOTAL_TRAINING_EPOCHS + 1))
assert PROTOCOL_EPOCHS == (5, 6, 7, 8, 9, 10, 11, 12)
assert len(PROTOCOL_EPOCHS) == TOTAL_TRAINING_EPOCHS - BURN_IN_EPOCHS == 8

FIXED_SELECTION_MODE = "first"
FIXED_CALIBRATION_N = 30
FIXED_POOL_SIZE = 50


def lightning_epoch_index(protocol_epoch: int) -> int:
    """Map a 1-indexed protocol epoch to Lightning's native 0-indexed ``{epoch}``.

    Lightning's ModelCheckpoint filenames use ``trainer.current_epoch``, which is still
    0 when the first epoch's checkpoint is written. This is the single place that
    translation happens; both ``select_epoch_window_checkpoints`` below and this
    module's tests use it, so the convention cannot silently drift between the training
    script (which just uses Lightning's native filenames) and this evaluation script.
    """
    if protocol_epoch < 1:
        raise ValueError(f"protocol_epoch must be >= 1, got {protocol_epoch}")
    return protocol_epoch - 1


def epoch_checkpoint_path(epoch_ckpt_dir: Path, protocol_epoch: int) -> Path:
    return epoch_ckpt_dir / f"epoch_{lightning_epoch_index(protocol_epoch):03d}.ckpt"


def select_epoch_window_checkpoints(
    run_dir: Path, epochs: Sequence[int] = PROTOCOL_EPOCHS
) -> dict[int, Path]:
    """Resolve one checkpoint file per pre-declared window epoch, or raise.

    Never falls back to a "closest available" epoch or a different count: the M3
    protocol is exactly these epochs, saved by train_variant_dandi688.py
    ``--checkpoint_every_epoch``, or nothing.
    """
    epoch_ckpt_dir = Path(run_dir) / "epoch_ckpts"
    if not epoch_ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"{epoch_ckpt_dir} does not exist. --run_dir must be a checkpoint directory "
            "produced by train_variant_dandi688.py --checkpoint_every_epoch."
        )
    selected: dict[int, Path] = {}
    missing: list[str] = []
    for protocol_epoch in epochs:
        path = epoch_checkpoint_path(epoch_ckpt_dir, protocol_epoch)
        if path.is_file():
            selected[protocol_epoch] = path
        else:
            missing.append(path.name)
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint(s) for the pre-declared epoch window {list(epochs)} in "
            f"{epoch_ckpt_dir}: {missing}. The run must complete exactly "
            f"{TOTAL_TRAINING_EPOCHS} epochs with --no_early_stopping "
            "--checkpoint_every_epoch (sua_exploration/docs/CURRENT_RESULTS.md section H.3)."
        )
    return selected


def compute_variant_score(
    per_epoch_mean_r2: Mapping[int, float], epochs: Sequence[int] = PROTOCOL_EPOCHS
) -> float:
    """Unweighted mean of the protocol metric over exactly ``epochs``.

    Raises if the input does not carry exactly the pre-declared epoch set -- this
    estimator is not allowed to silently average over a different window (e.g. because a
    checkpoint went missing) or double-count an epoch.
    """
    expected = list(epochs)
    observed = sorted(per_epoch_mean_r2.keys())
    if observed != sorted(expected):
        raise ValueError(
            f"variant score requires exactly epochs {expected}, got {observed}"
        )
    values = [per_epoch_mean_r2[epoch] for epoch in expected]
    return sum(values) / len(values)


def _validate_run_metadata_for_epoch_window(metadata: dict) -> None:
    if metadata.get("status") != "completed":
        raise ValueError(
            f"run_metadata status must be 'completed', found {metadata.get('status')!r}"
        )
    if metadata.get("held_out_test_evaluated") is not False:
        raise ValueError(
            "run_metadata.held_out_test_evaluated must be false for a validation-only "
            "epoch-window evaluation"
        )
    training = metadata.get("training", {})
    if training.get("max_epochs") != TOTAL_TRAINING_EPOCHS:
        raise ValueError(
            f"M3 requires training.max_epochs == {TOTAL_TRAINING_EPOCHS}, found "
            f"{training.get('max_epochs')!r}. Re-run train_variant_dandi688.py with "
            f"--max_epochs {TOTAL_TRAINING_EPOCHS} --no_early_stopping "
            "--checkpoint_every_epoch."
        )
    if training.get("no_early_stopping") is not True:
        raise ValueError(
            "M3 requires the run to have been trained with --no_early_stopping "
            f"(found training.no_early_stopping={training.get('no_early_stopping')!r}); "
            f"otherwise epochs {list(PROTOCOL_EPOCHS)} are not guaranteed to exist."
        )
    if training.get("checkpoint_every_epoch") is not True:
        raise ValueError(
            "M3 requires the run to have been trained with --checkpoint_every_epoch "
            f"(found training.checkpoint_every_epoch={training.get('checkpoint_every_epoch')!r})."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help=(
            "Checkpoint output directory from train_variant_dandi688.py "
            "--checkpoint_every_epoch (e.g. sua_exploration/checkpoints/<out_name>); "
            "must contain run_metadata.json and epoch_ckpts/."
        ),
    )
    parser.add_argument(
        "--teacher_ckpt", type=str, default=None,
        help="Defaults to the teacher_checkpoint recorded in run_dir/run_metadata.json.",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Defaults to the data_dir recorded in run_dir/run_metadata.json.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Defaults to the cache_dir recorded in run_dir/run_metadata.json (if any).",
    )
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint provenance metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    _validate_run_metadata_for_epoch_window(metadata)

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

    checkpoints = select_epoch_window_checkpoints(run_dir, PROTOCOL_EPOCHS)

    per_epoch: dict[str, dict] = {}
    session_splits = None
    session_unit_counts = None
    for protocol_epoch in PROTOCOL_EPOCHS:
        ckpt_path = checkpoints[protocol_epoch]
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
        )
        if session_splits is None:
            session_splits = result["session_splits"]
            session_unit_counts = result["session_unit_counts"]
        elif result["session_splits"] != session_splits:
            raise ValueError(
                f"Session split drifted at epoch {protocol_epoch}; data_dir or "
                "split_counts may have changed mid-run"
            )
        per_epoch[str(protocol_epoch)] = {
            "checkpoint_path": str(ckpt_path.resolve()),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "per_session_r2": result["per_session_r2"],
            "mean_r2": result["mean_r2"],
        }

    per_epoch_mean_r2 = {epoch: per_epoch[str(epoch)]["mean_r2"] for epoch in PROTOCOL_EPOCHS}
    variant_score = compute_variant_score(per_epoch_mean_r2, PROTOCOL_EPOCHS)

    payload = {
        "schema_version": 1,
        "purpose": "epoch_window_deterministic_checkpoint_selection",
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
            "name": "fixed_epoch_window_deterministic_checkpoint_rule",
            "description": (
                "M3 (sua_exploration/docs/CURRENT_RESULTS.md section H.3): replaces "
                "argmax-over-noisy-validation-metric checkpoint selection. Train exactly "
                f"{TOTAL_TRAINING_EPOCHS} epochs; the variant score is the unweighted "
                f"mean of the protocol metric over the trailing {len(PROTOCOL_EPOCHS)} "
                f"epochs (epochs {list(PROTOCOL_EPOCHS)}), i.e. a {BURN_IN_EPOCHS}-epoch "
                "burn-in."
            ),
            "total_epochs": TOTAL_TRAINING_EPOCHS,
            "epoch_window": list(PROTOCOL_EPOCHS),
            "burn_in_epochs": BURN_IN_EPOCHS,
            "selection_mode": FIXED_SELECTION_MODE,
            "calibration_n": FIXED_CALIBRATION_N,
            "pool_size": FIXED_POOL_SIZE,
            "protocol_metric_source": (
                "select_gradient_free_protocol_dandi688."
                "evaluate_fixed_protocol_over_validation_sessions"
            ),
        },
        "epoch_list": list(PROTOCOL_EPOCHS),
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": {str(epoch): per_epoch_mean_r2[epoch] for epoch in PROTOCOL_EPOCHS},
        "variant_score": variant_score,
        "variant_score_definition": (
            "unweighted mean over epoch_window of per-epoch mean validation R2 under the "
            f"fixed {FIXED_SELECTION_MODE}/n={FIXED_CALIBRATION_N}/pool={FIXED_POOL_SIZE} "
            "forward-calibration protocol across the 6 validation sessions"
        ),
        "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
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
        else results_dir / f"p3_epoch_window_{variant.lower()}_s{seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote epoch-window evaluation: {out_path}")
    print(f"Variant score ({variant}, seed={seed}): {variant_score:.4f}")


if __name__ == "__main__":
    main()
