"""M3 estimator, generalized: CLI-configurable epoch-window checkpoint scoring for DANDI
000688 SUA runs, for screens whose epoch budget is not the frozen 12 that
``eval_epoch_window_dandi688.py`` hardcodes.

Why this file exists instead of a flag on the frozen script (E3_E4_ENCODER_PROGRAM.md
section 0): ``eval_epoch_window_dandi688.py`` is the M3 estimator already used by a
published screen (``side_feature_ablation_v2``) and is frozen deliberately -- it must not
change out from under an already-reported result. E3 (tuning-feature ablation) and E4
(encoder architecture variants) cannot reuse it as-is because their epoch budget is set by
E2's convergence measurement and their seed count by E1's measured ``sigma_seed``; neither
is known at the time this file is written, and hardcoding a guess for either is exactly the
failure mode this project has hit twice (attention_arch_screen_v3's ``+0.005`` threshold and
side_feature_ablation_v2's original ``sigma_variant = sigma_run/sqrt(3)`` formula, both
documented in MEASUREMENT_PROTOCOL_V4.md section 4.1). So this script takes the epoch budget
(``--total_epochs``) and the burn-in (``--burn_in``) as explicit, required CLI arguments and
computes the protocol window as ``burn_in+1 .. total_epochs`` inclusive, instead of the
frozen script's hardcoded ``range(5, 13)``.

Everything else is identical to the frozen script on purpose:

1. The run must have trained exactly ``--total_epochs`` epochs with ``--no_early_stopping``
   and saved a checkpoint at every epoch with ``--checkpoint_every_epoch``.
2. The variant score is the unweighted mean of the protocol metric evaluated at every epoch
   in the window ``burn_in+1 .. total_epochs`` -- an ``(total_epochs - burn_in)``-epoch
   trailing average with a ``burn_in``-epoch burn-in (MEASUREMENT_PROTOCOL_V4.md section 2.3
   generalizes the frozen script's "4-epoch burn-in + 8-epoch tail" to an arbitrary budget;
   the burn-in *count* is a CLI argument precisely so it is never silently re-guessed here).
3. The protocol metric is the existing fixed forward-calibration evaluation
   (chronological first prefix, with explicit calibration and disjoint-pool sizes) over
   the 6 validation sessions, computed by
   ``select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions``
   -- this script calls that function once per epoch checkpoint; it does not reimplement the
   R2 computation, and does not modify that function.

Output JSON schema (top-level keys, ``protocol`` sub-keys, ``per_epoch`` shape) is kept
field-for-field identical to ``eval_epoch_window_dandi688.py``'s, so
``aggregate_e3_tuning_ablation.py`` / ``aggregate_e4_encoder_variants.py`` -- or, in
principle, any consumer written against the frozen script's artifacts -- can read either
file's output without a schema branch. The only new field is ``generated_by``, which is
purely additive.

Hard constraint: this script only ever loads validation-session spike/behavior/trial data.
When ``--train_val_manifest`` is supplied, it uses its frozen 27-train/6-validation path and
the six formal-test entries are receipt names only: no test NWB path is resolved, opened, or
unit-counted. Without that strict-manifest flag, it retains the frozen evaluator's historical
discovery path.
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

# --- Default sub-protocol (explicit CLI values are recorded in every artifact). ----
FIXED_SELECTION_MODE = "first"
FIXED_CALIBRATION_N = 30
FIXED_POOL_SIZE = 50


def compute_protocol_epochs(total_epochs: int, burn_in: int) -> tuple[int, ...]:
    """The 1-indexed protocol-epoch window ``burn_in+1 .. total_epochs`` (inclusive).

    "Protocol epoch" N means "the checkpoint saved after N completed training epochs", same
    convention as the frozen script (there hardcoded to ``total_epochs=12, burn_in=4`` ->
    ``(5, 6, ..., 12)``). Both numbers are CLI-supplied here instead: E2 sets
    ``total_epochs`` (the convergence-derived epoch budget), and ``burn_in`` generalizes the
    frozen script's 4-epoch burn-in so it is never silently re-hardcoded either.
    """
    if total_epochs < 1:
        raise ValueError(f"--total_epochs must be >= 1, got {total_epochs}")
    if burn_in < 0:
        raise ValueError(f"--burn_in must be >= 0, got {burn_in}")
    if burn_in >= total_epochs:
        raise ValueError(
            f"--burn_in ({burn_in}) must be strictly less than --total_epochs "
            f"({total_epochs}); otherwise the protocol window burn_in+1..total_epochs is "
            "empty"
        )
    epochs = tuple(range(burn_in + 1, total_epochs + 1))
    assert epochs[0] == burn_in + 1
    assert epochs[-1] == total_epochs
    assert len(epochs) == total_epochs - burn_in
    return epochs


def lightning_epoch_index(protocol_epoch: int) -> int:
    """Map a 1-indexed protocol epoch to Lightning's native 0-indexed ``{epoch}``.

    Identical to the frozen script's helper of the same name: Lightning's ModelCheckpoint
    filenames use ``trainer.current_epoch``, which is still 0 when the first epoch's
    checkpoint is written.
    """
    if protocol_epoch < 1:
        raise ValueError(f"protocol_epoch must be >= 1, got {protocol_epoch}")
    return protocol_epoch - 1


def epoch_checkpoint_path(epoch_ckpt_dir: Path, protocol_epoch: int) -> Path:
    return epoch_ckpt_dir / f"epoch_{lightning_epoch_index(protocol_epoch):03d}.ckpt"


def select_epoch_window_checkpoints(
    run_dir: Path, epochs: Sequence[int], total_epochs: int
) -> dict[int, Path]:
    """Resolve one checkpoint file per pre-declared window epoch, or raise.

    ``epochs`` and ``total_epochs`` must be supplied explicitly by the caller (computed from
    ``--total_epochs``/``--burn_in`` via ``compute_protocol_epochs``) -- unlike the frozen
    script, there is no module-level default, since there is no single frozen window to
    default to. Never falls back to a "closest available" epoch or a different count: the
    protocol is exactly ``epochs``, saved by train_variant_dandi688.py
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
            f"{epoch_ckpt_dir}: {missing}. The run must complete exactly {total_epochs} "
            "epochs with --no_early_stopping --checkpoint_every_epoch."
        )
    return selected


def compute_variant_score(
    per_epoch_mean_r2: Mapping[int, float], epochs: Sequence[int]
) -> float:
    """Unweighted mean of the protocol metric over exactly ``epochs``.

    ``epochs`` has no default (unlike the frozen script's ``PROTOCOL_EPOCHS``-defaulted
    version): every caller must pass the window explicitly so it is always traceable to the
    ``--total_epochs``/``--burn_in`` that produced it. Raises if the input does not carry
    exactly the declared epoch set -- this estimator is not allowed to silently average over
    a different window (e.g. because a checkpoint went missing) or double-count an epoch.
    """
    expected = list(epochs)
    observed = sorted(per_epoch_mean_r2.keys())
    if observed != sorted(expected):
        raise ValueError(
            f"variant score requires exactly epochs {expected}, got {observed}"
        )
    values = [per_epoch_mean_r2[epoch] for epoch in expected]
    return sum(values) / len(values)


def validate_prefix_budget(calibration_n: int, pool_size: int) -> None:
    """Require a positive calibration prefix fully contained in the excluded pool."""
    if calibration_n <= 0:
        raise ValueError("--calibration_n must be positive")
    if pool_size < calibration_n:
        raise ValueError("--pool_size must be >= --calibration_n")


def _validate_run_metadata_for_epoch_window(metadata: dict, *, total_epochs: int) -> None:
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
    if training.get("max_epochs") != total_epochs:
        raise ValueError(
            f"--total_epochs {total_epochs} requires training.max_epochs == {total_epochs}, "
            f"found {training.get('max_epochs')!r}. Either re-run train_variant_dandi688.py "
            f"with --max_epochs {total_epochs} --no_early_stopping --checkpoint_every_epoch, "
            "or pass the --total_epochs this run actually used."
        )
    if training.get("no_early_stopping") is not True:
        raise ValueError(
            "M3 requires the run to have been trained with --no_early_stopping "
            f"(found training.no_early_stopping={training.get('no_early_stopping')!r})."
        )
    if training.get("checkpoint_every_epoch") is not True:
        raise ValueError(
            "M3 requires the run to have been trained with --checkpoint_every_epoch "
            f"(found training.checkpoint_every_epoch={training.get('checkpoint_every_epoch')!r})."
        )


def validate_relation_calibration_contract(metadata: dict, evaluation_calibration_n: int) -> None:
    """Fail closed on the intentionally distinct relation train/eval budgets."""
    side = metadata.get("side_features") or {}
    relation_tokens = {"t4rel", "t4rel_membership_shuffled", "t4rel_nogroup"}
    relation_variants = {"B3SER", "B3SERN"}
    is_relation_run = (
        side.get("group") in relation_tokens
        or metadata.get("variant") in relation_variants
        # Backward-compatible support for pre-no-group metadata produced during
        # the initial relation wiring pass.
        or side.get("uses_equality_only_relation_membership", False)
    )
    if not is_relation_run:
        return
    training_n = (metadata.get("training") or {}).get("calibration_n_trials")
    if training_n != 10:
        raise ValueError(
            "same-electrode relation contract requires training.activity_calibration_n_trials=10"
        )
    if evaluation_calibration_n != 30:
        raise ValueError(
            "same-electrode relation epoch-window contract requires evaluation.forward_calibration_n=30"
        )


def validate_strict_manifest_provenance(
    metadata: dict, manifest_path: Path | None
) -> None:
    """Require the exact manifest bytes that training recorded for strict isolation."""
    metadata_manifest = metadata.get("train_val_manifest")
    if metadata_manifest is None:
        return
    if manifest_path is None:
        raise ValueError("run metadata requires --train_val_manifest for strict isolation")
    if str(manifest_path) != metadata_manifest:
        raise ValueError("--train_val_manifest does not match run_metadata.json")
    expected_hash = metadata.get("train_val_manifest_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(
            "strict run metadata is missing a valid train_val_manifest_sha256 provenance hash"
        )
    observed_hash = sha256_file(manifest_path)
    if observed_hash != expected_hash:
        raise ValueError(
            "--train_val_manifest SHA-256 does not match run_metadata.json; "
            "the strict split contents drifted"
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
        "--total_epochs",
        required=True,
        type=int,
        help=(
            "Total training epoch budget this run used (E2's convergence-derived value; "
            "no default -- see E3_E4_ENCODER_PROGRAM.md section 0). Must equal the run's "
            "own run_metadata.json training.max_epochs."
        ),
    )
    parser.add_argument(
        "--burn_in",
        required=True,
        type=int,
        help=(
            "Number of leading epochs excluded from the trailing-average window (frozen "
            "script's M3 estimator used 4 for a 12-epoch budget). The scored window is "
            "epochs burn_in+1 .. total_epochs inclusive. No default -- must be supplied "
            "explicitly every time (E3_E4_ENCODER_PROGRAM.md section 0)."
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
    parser.add_argument(
        "--calibration_n", type=int, default=FIXED_CALIBRATION_N,
        help=(
            "Forward-calibration trials for validation scoring. Default preserves the existing "
            f"generic evaluator protocol ({FIXED_CALIBRATION_N}). Same-electrode relation "
            "runs deliberately train activity with n=10 but must evaluate forward calibration "
            "with the frozen n=30 protocol."
        ),
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=FIXED_POOL_SIZE,
        help=(
            "Chronological prefix excluded from evaluation windows. It must be at least "
            "--calibration_n. The matched SPINT/T4 mainline sets both values to 30 so all "
            "methods receive the same trials and scoring starts immediately afterward."
        ),
    )
    parser.add_argument(
        "--train_val_manifest", type=str, default=None,
        help="Strict train/validation manifest; when supplied no formal-test NWB may be opened.",
    )
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()
    validate_prefix_budget(args.calibration_n, args.pool_size)

    protocol_epochs = compute_protocol_epochs(args.total_epochs, args.burn_in)

    run_dir = Path(args.run_dir).expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint provenance metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    _validate_run_metadata_for_epoch_window(metadata, total_epochs=args.total_epochs)
    validate_relation_calibration_contract(metadata, args.calibration_n)

    variant = metadata["variant"]
    seed = metadata["seed"]
    task = metadata["task"]
    split_counts = tuple(metadata["split_counts"])
    max_units_exclusive = metadata["max_units_exclusive"]
    signal_view = metadata.get("signal_view", "sua")
    side_feature_group = (metadata.get("side_features") or {}).get("group", "none")
    label_derived_side_groups = {
        "t4", "t8", "ts4", "ts8",
        "t4e", "t4e_shuffled",
        "t4gate", "t4gate_shuffled",
        "t4anchor", "t4anchor_shuffled",
        "t4rel", "t4rel_membership_shuffled", "t4rel_nogroup",
        "t4cf", "t4cf_ts4", "t4cf_confidence_shuffled",
    }
    label_feature_pool_size = (
        (metadata.get("side_features") or {}).get("pool_size")
        if side_feature_group in label_derived_side_groups
        else None
    )

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
    manifest_path = Path(args.train_val_manifest).expanduser().resolve() if args.train_val_manifest else None
    validate_strict_manifest_provenance(metadata, manifest_path)

    checkpoints = select_epoch_window_checkpoints(run_dir, protocol_epochs, args.total_epochs)

    per_epoch: dict[str, dict] = {}
    session_splits = None
    session_unit_counts = None
    for protocol_epoch in protocol_epochs:
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
            pool_size=args.pool_size,
            selection_mode=FIXED_SELECTION_MODE,
            calibration_n=args.calibration_n,
            signal_view=signal_view,
            train_val_manifest=manifest_path,
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

    per_epoch_mean_r2 = {epoch: per_epoch[str(epoch)]["mean_r2"] for epoch in protocol_epochs}
    variant_score = compute_variant_score(per_epoch_mean_r2, protocol_epochs)

    payload = {
        "schema_version": 1,
        "purpose": "epoch_window_deterministic_checkpoint_selection",
        "generated_by": "eval_epoch_window_generic_dandi688.py",
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
        "train_val_manifest": str(manifest_path) if manifest_path else None,
        "train_val_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path else None
        ),
        "protocol": {
            "name": "fixed_epoch_window_deterministic_checkpoint_rule",
            "description": (
                "M3, generalized (E3_E4_ENCODER_PROGRAM.md section 0): replaces "
                "argmax-over-noisy-validation-metric checkpoint selection. Train exactly "
                f"{args.total_epochs} epochs; the variant score is the unweighted mean of "
                f"the protocol metric over the trailing {len(protocol_epochs)} epochs "
                f"(epochs {list(protocol_epochs)}), i.e. a {args.burn_in}-epoch burn-in."
            ),
            "total_epochs": args.total_epochs,
            "epoch_window": list(protocol_epochs),
            "burn_in_epochs": args.burn_in,
            "selection_mode": FIXED_SELECTION_MODE,
            "calibration_n": args.calibration_n,
            "train_activity_calibration_n": metadata.get("training", {}).get("calibration_n_trials"),
            "evaluation_forward_calibration_n": args.calibration_n,
            "label_feature_calibration_n": label_feature_pool_size,
            "pool_size": args.pool_size,
            "protocol_metric_source": (
                "select_gradient_free_protocol_dandi688."
                "evaluate_fixed_protocol_over_validation_sessions"
            ),
        },
        "epoch_list": list(protocol_epochs),
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": {str(epoch): per_epoch_mean_r2[epoch] for epoch in protocol_epochs},
        "variant_score": variant_score,
        "variant_score_definition": (
            "unweighted mean over epoch_window of per-epoch mean validation R2 under the "
            f"fixed {FIXED_SELECTION_MODE}/n={args.calibration_n}/pool={args.pool_size} "
            "forward-calibration protocol across the 6 validation sessions"
        ),
        "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
        "session_splits": session_splits,
        "session_unit_counts": session_unit_counts,
        "calibration_trial_selection_uses_behavior_labels": False,
        "calibration_features_use_behavior_labels": (
            side_feature_group in label_derived_side_groups
        ),
        "calibration_feature_label_scope": (
            f"chronological_rewarded_trials[0:{label_feature_pool_size}]"
            if side_feature_group in label_derived_side_groups
            else None
        ),
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
    }
    results_dir = Path(__file__).resolve().parents[1] / "results"
    out_path = (
        Path(args.out_path) if args.out_path
        else results_dir / f"p3_epoch_window_generic_{variant.lower()}_s{seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote epoch-window evaluation: {out_path}")
    print(f"Variant score ({variant}, seed={seed}): {variant_score:.4f}")


if __name__ == "__main__":
    main()
