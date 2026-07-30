"""Validation-only selection and locking of a gradient-free calibration protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dandi688_gradient_free_protocol import (
    canonical_direction_key,
    select_calibration_trial_indices,
    sha256_file,
)
from eval_adaptation_dandi688 import (
    BEHAVIOR_SCALING_FACTOR,
    DEFAULT_TEACHER,
    HIDDEN_DIM,
    ID_HIDDEN_DIM,
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    attach_side_features,
    build_calib_trials_for_indices,
    eval_r2,
    eval_r2_with_zero_identity,
    load_session_with_trials,
    load_side_feature_stats_for_run_metadata,
    make_subset_dataset,
    parse_split_counts,
)
from mc_maze.multisession_datamodule import (
    chronological_session_split,
    discover_nwb_files,
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    nwb_unit_count,
    session_name_from_path,
)

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",")})
    if not values or any(item <= 0 for item in values):
        raise ValueError("calibration_ns must contain positive integers")
    return values


_LEARNED_PRIOR_KEY = "population_identity"
_LEARNED_PRIOR_ERROR = (
    "Checkpoint was not trained with --identity_mode learned_prior: "
    f"state_dict must contain a non-zero '{_LEARNED_PRIOR_KEY}' tensor"
)


def validate_learned_prior_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    missing_keys: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Reject calibrated checkpoints masquerading as learned-prior controls."""
    if _LEARNED_PRIOR_KEY not in state_dict:
        raise ValueError(_LEARNED_PRIOR_ERROR)
    population_identity = state_dict[_LEARNED_PRIOR_KEY]
    if torch.count_nonzero(population_identity).item() == 0:
        raise ValueError(_LEARNED_PRIOR_ERROR)
    if missing_keys and _LEARNED_PRIOR_KEY in missing_keys:
        raise ValueError(_LEARNED_PRIOR_ERROR)


def load_frozen_model(
    ckpt_path: Path,
    teacher_ckpt: Path,
    variant: str,
    device: torch.device,
    identity_mode: str = "calibrated",
):
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    model = StreamingCalibrationLitModule(
        task="mc_maze", variant=variant, teacher_ckpt_path=str(teacher_ckpt),
        window_size=WINDOW_SIZE, trial_length=TRIAL_LENGTH, id_hidden_dim=ID_HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM, pad_value=PAD_VALUE, freeze_decoder=False,
        loss_mode="task_only", decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=BEHAVIOR_SCALING_FACTOR,
        identity_mode=identity_mode,
        fixed_slot_count=int(hyper_parameters.get("fixed_slot_count", 0)),
        fixed_slot_dim=int(hyper_parameters.get("fixed_slot_dim", 32)),
        fixed_slot_mode=str(hyper_parameters.get("fixed_slot_mode", "soft")),
        fixed_slot_fusion=str(hyper_parameters.get("fixed_slot_fusion", "film")),
        fixed_slot_temperature=float(hyper_parameters.get("fixed_slot_temperature", 1.0)),
        side_dim=int(hyper_parameters.get("side_dim", 0)),
        electrode_embed_dim=int(hyper_parameters.get("electrode_embed_dim", 0)),
        num_electrodes=int(hyper_parameters.get("num_electrodes", 0)),
        compile=False,
    )
    model.setup("fit")
    strict_load = identity_mode != "learned_prior"
    if identity_mode == "learned_prior":
        validate_learned_prior_checkpoint(checkpoint["state_dict"])
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=strict_load)
    if identity_mode == "learned_prior":
        validate_learned_prior_checkpoint(
            checkpoint["state_dict"],
            missing_keys=getattr(load_result, "missing_keys", ()),
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    return model


def evaluate_session_configs(
    rec: dict,
    configs: list[tuple[str, int]],
    pool_size: int,
    model,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Evaluate one loaded session record under each (selection_mode, calibration_n) config.

    This is the exact per-session protocol computation `main()` has always used (fixed
    or swept over multiple configs); it is factored out so
    ``evaluate_fixed_protocol_over_validation_sessions`` below and
    ``eval_epoch_window_dandi688.py`` (M3 deterministic epoch-window checkpoint rule,
    sua_exploration/docs/CURRENT_RESULTS.md section H.3) can reuse it verbatim instead of
    recomputing R2 with a second, potentially-drifting implementation. Behavior is
    unchanged from the inline loop this was extracted from.
    """
    if len(rec["trials"]) <= pool_size:
        raise ValueError(f"{rec['name']}: no trial remains for common evaluation after pool_size={pool_size}")
    eval_trials = rec["trials"][pool_size:]
    session_r2: dict[str, float] = {}
    session_selections: dict[str, dict] = {}
    for mode, calibration_n in configs:
        indices = select_calibration_trial_indices(rec["trials"], calibration_n, pool_size, mode)
        rec["calib_trials"] = build_calib_trials_for_indices(rec, indices, calibration_n)
        eval_ds = make_subset_dataset(rec, eval_trials, rec["name"])
        if not len(eval_ds):
            raise ValueError(f"{rec['name']}: common evaluation trials provide no usable windows")
        config_name = f"gradient_free_calibrated_{mode}_n{calibration_n}"
        session_r2[config_name] = eval_r2(model, eval_ds, device)
        session_selections[config_name] = {
            "usable_trial_list_indices": indices,
            "original_trial_indices": [rec["trials"][index]["trial_index"] for index in indices],
            "direction_keys": [repr(canonical_direction_key(rec["trials"][index])) for index in indices],
        }
    return session_r2, session_selections


def evaluate_fixed_protocol_over_validation_sessions(
    *,
    ckpt_path: Path,
    teacher_ckpt: Path,
    variant: str,
    data_dir: Path,
    task: str,
    split_counts: tuple[int, int, int],
    max_units_exclusive: int | None,
    cache_dir: Path | None,
    pool_size: int,
    selection_mode: str,
    calibration_n: int,
    signal_view: str = "sua",
    train_val_manifest: Path | None = None,
    device: torch.device | None = None,
) -> dict:
    """Evaluate ONE checkpoint's fixed forward-calibration protocol over the validation
    sessions only (never loads test-session spike/behavior/trial data; test files are
    only used to report their names and NWB unit-table row counts, matching the access
    pattern already used elsewhere in this repo).

    This performs the same computation as this module's ``main()`` with
    ``--fixed_selection_mode``/``--fixed_calibration_n`` (both delegate to
    ``evaluate_session_configs`` above), minus main()'s CLI concerns: writing a JSON
    artifact, the zero-identity control, and the single-"selected"-checkpoint
    run_metadata provenance gate (``best_checkpoint`` cross-check), which does not apply
    when the caller intentionally evaluates many checkpoints from the same run directory
    (M3 epoch-window scoring, see eval_epoch_window_dandi688.py).

    B3S side features (UNIT_SIDE_FEATURE_ABLATION.md): if the checkpoint's own
    run_metadata.json records a real ``side_features.group`` (i.e. this is a B3S run with
    f1/f2/fs1/fs2, not "none"), per-unit side features are computed with train-only
    normalization stats and attached to each validation session before scoring -- without
    this, ``SideFeatureEarlyPoolEncoder.finalize_identity`` raises ``"B3S requires
    side_features when side_dim > 0"`` the moment such a checkpoint is evaluated, because
    ``load_session_with_trials``/``MCMazeSessionDataset`` never computed them on their own.
    Every non-B3S checkpoint (and B3S with ``--side_features none``) is unaffected: this
    function's own signature and its one caller (eval_epoch_window_dandi688.py) are
    unchanged, and the side-feature resolution is a no-op whenever
    ``run_metadata["side_features"]["group"] == "none"``.
    """
    if calibration_n > pool_size:
        raise ValueError("pool_size must be at least calibration_n")
    if train_val_manifest is not None:
        train_files, val_files, test_names = load_frozen_train_val_manifest(
            train_val_manifest, data_dir
        )
        if split_counts != (27, 6, 6) or max_units_exclusive != 100:
            raise ValueError("strict train_val_manifest requires fixed 27/6/6 and units<100")
        counted_files = train_files + val_files
    else:
        all_files = discover_nwb_files(data_dir, task, max_units_exclusive)
        train_files, val_files, test_files = chronological_session_split(
            all_files, split_counts, max_units_exclusive=max_units_exclusive
        )
        test_names = [session_name_from_path(path) for path in test_files]
        counted_files = all_files
    if not val_files:
        raise ValueError("No validation sessions selected for protocol evaluation")
    mean, std = fit_behavior_stats(train_files, 20, cache_dir=cache_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(ckpt_path, teacher_ckpt, variant, device)

    # ckpt_path.parent.parent is the run directory for both layouts this function's caller
    # ever passes: M3 epoch-window checkpoints (run_dir/epoch_ckpts/epoch_NNN.ckpt) and
    # Lightning's best-*.ckpt ModelCheckpoint output (run_dir/best-.../metric.ckpt, where
    # Lightning turned the "/" in the val_heldin/r2_mean metric name into a real
    # subdirectory) -- the same derivation main() below uses for its own provenance check.
    # Soft-missing (no raise) so this can never newly break a caller whose checkpoint has no
    # run_metadata.json at that path; a side_dim>0 model with no attached side_features
    # still fails loudly downstream in SideFeatureEarlyPoolEncoder.finalize_identity.
    run_metadata_path = Path(ckpt_path).parent.parent / "run_metadata.json"
    side_feature_config = None
    if run_metadata_path.is_file():
        run_metadata = json.loads(run_metadata_path.read_text())
        side_feature_config = load_side_feature_stats_for_run_metadata(
            run_metadata, train_files, cache_dir
        )

    configs = [(selection_mode, calibration_n)]
    config_name = f"gradient_free_calibrated_{selection_mode}_n{calibration_n}"
    per_session_r2: dict[str, float] = {}
    selections: dict[str, dict] = {}
    with torch.no_grad():
        for path in val_files:
            rec = load_session_with_trials(
                path, 20, WINDOW_SIZE, pool_size, TRIAL_LENGTH, PAD_VALUE, mean, std,
                cache_dir=cache_dir, signal_view=signal_view,
            )
            if side_feature_config is not None:
                (
                    side_feature_group,
                    waveform_feature_group,
                    side_pool_size,
                    permutation_seed,
                    side_mean,
                    side_std,
                ) = side_feature_config
                rec = attach_side_features(
                    rec,
                    path,
                    side_feature_group=side_feature_group,
                    waveform_feature_group=waveform_feature_group,
                    pool_size=side_pool_size,
                    permutation_seed=permutation_seed,
                    mean=side_mean,
                    std=side_std,
                    cache_dir=cache_dir,
                )
            session_r2, session_selections = evaluate_session_configs(rec, configs, pool_size, model, device)
            per_session_r2[rec["name"]] = session_r2[config_name]
            selections[rec["name"]] = session_selections[config_name]
    mean_r2 = sum(per_session_r2.values()) / len(per_session_r2)
    return {
        "per_session_r2": per_session_r2,
        "mean_r2": mean_r2,
        "trial_selections": selections,
        "session_splits": {
            "train": [session_name_from_path(p) for p in train_files],
            "val": [session_name_from_path(p) for p in val_files],
            "test": test_names,
        },
        "session_unit_counts": {
            session_name_from_path(path): nwb_unit_count(path) for path in counted_files
        },
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
    parser.add_argument(
        "--signal_view",
        choices=["sua", "pseudo_mua"],
        default="sua",
        help="Evaluate sorted units directly or the electrode-pooled pseudo-MUA view.",
    )
    parser.add_argument("--calibration_ns", default="10,20,30,50")
    parser.add_argument("--selection_modes", default="first,direction_coverage")
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional shared disk cache for validation session tensors.",
    )
    parser.add_argument(
        "--fixed_selection_mode",
        choices=["first", "direction_coverage"],
        default=None,
        help="Evaluate one predeclared calibration selection mode without protocol tuning.",
    )
    parser.add_argument(
        "--fixed_calibration_n",
        type=int,
        default=None,
        help="Evaluate one predeclared calibration trial count without protocol tuning.",
    )
    parser.add_argument(
        "--no_formal_lock",
        action="store_true",
        help="Do not emit a formal-test lock; use for development-only fixed evaluations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--lock_path", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    if not ckpt_path.is_file() or not teacher_ckpt.is_file():
        raise FileNotFoundError("--ckpt and --teacher_ckpt must name existing files")
    split_counts = parse_split_counts(args.split_counts)
    data_dir = Path(args.data_dir).expanduser().resolve()
    run_metadata_path = ckpt_path.parent.parent / "run_metadata.json"
    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint provenance metadata is missing: {run_metadata_path}")
    run_metadata = json.loads(run_metadata_path.read_text())
    expected_provenance = {
        "status": "completed",
        "best_checkpoint": str(ckpt_path),
        "best_checkpoint_sha256": sha256_file(ckpt_path),
        "variant": args.variant,
        "teacher_sha256": sha256_file(teacher_ckpt),
        "data_dir": str(data_dir),
        "task": args.task,
        "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "seed": args.seed,
        "held_out_test_evaluated": False,
    }
    provenance_mismatches = [
        key for key, value in expected_provenance.items()
        if run_metadata.get(key) != value
    ]
    if provenance_mismatches:
        raise ValueError("Checkpoint run_metadata provenance mismatch: " + ", ".join(provenance_mismatches))
    if run_metadata.get("signal_view", "sua") != args.signal_view:
        raise ValueError(
            "Checkpoint run_metadata signal_view mismatch: "
            f"expected {args.signal_view!r}, found {run_metadata.get('signal_view', 'sua')!r}"
        )
    fixed_protocol = args.fixed_selection_mode is not None or args.fixed_calibration_n is not None
    if fixed_protocol and (args.fixed_selection_mode is None or args.fixed_calibration_n is None):
        raise ValueError("--fixed_selection_mode and --fixed_calibration_n must be supplied together")
    if fixed_protocol:
        if args.fixed_calibration_n <= 0:
            raise ValueError("--fixed_calibration_n must be positive")
        calibration_ns = [args.fixed_calibration_n]
        modes = [args.fixed_selection_mode]
    else:
        calibration_ns = parse_int_list(args.calibration_ns)
        modes = list(dict.fromkeys(item.strip() for item in args.selection_modes.split(",") if item.strip()))
        if not modes or any(mode not in {"first", "direction_coverage"} for mode in modes):
            raise ValueError("selection_modes must use first and/or direction_coverage")
    if max(calibration_ns) > args.pool_size:
        raise ValueError("pool_size must be at least max(calibration_ns)")

    all_files = discover_nwb_files(data_dir, args.task, args.max_units_exclusive)
    train_files, val_files, test_files = chronological_session_split(
        all_files, split_counts, max_units_exclusive=args.max_units_exclusive
    )
    if args.max_sessions is not None and args.max_sessions <= 0:
        raise ValueError("max_sessions must be positive when provided")
    validation_complete = args.max_sessions is None
    if args.max_sessions is not None:
        val_files = val_files[:args.max_sessions]
    if not val_files:
        raise ValueError("No validation sessions selected for protocol selection")
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else (
            Path(__file__).resolve().parents[1]
            / "cache"
            / f"dandi688_{data_dir.name.lower().replace('-', '')}_{args.task.lower()}_v1"
        )
    )
    mean, std = fit_behavior_stats(train_files, 20, cache_dir=cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(ckpt_path, teacher_ckpt, args.variant, device)
    configs = [(mode, calibration_n) for mode in modes for calibration_n in calibration_ns]
    config_names = [
        f"gradient_free_calibrated_{mode}_n{calibration_n}"
        for mode, calibration_n in configs
    ]
    per_session: dict[str, dict[str, float]] = {}
    no_calibration_per_session: dict[str, float] = {}
    selections: dict[str, dict[str, dict]] = {}
    with torch.no_grad():
        for path in val_files:
            rec = load_session_with_trials(
                path,
                20,
                WINDOW_SIZE,
                args.pool_size,
                TRIAL_LENGTH,
                PAD_VALUE,
                mean,
                std,
                cache_dir=cache_dir,
                signal_view=args.signal_view,
            )
            if len(rec["trials"]) <= args.pool_size:
                raise ValueError(f"{rec['name']}: no trial remains for common evaluation after pool_size={args.pool_size}")
            eval_trials = rec["trials"][args.pool_size:]
            no_calibration_ds = make_subset_dataset(rec, eval_trials, rec["name"])
            if not len(no_calibration_ds):
                raise ValueError(f"{rec['name']}: common evaluation trials provide no usable windows")
            no_calibration_per_session[rec["name"]] = eval_r2_with_zero_identity(
                model, no_calibration_ds, device
            )
            # Reused by eval_epoch_window_dandi688.py (M3) via
            # evaluate_fixed_protocol_over_validation_sessions / evaluate_session_configs
            # above -- do not reimplement this per-config R2 computation here.
            per_session[rec["name"]], selections[rec["name"]] = evaluate_session_configs(
                rec, configs, args.pool_size, model, device
            )
    means = {name: sum(row[name] for row in per_session.values()) / len(per_session) for name in per_session[next(iter(per_session))]}
    chosen_name, chosen_r2 = sorted(
        means.items(), key=lambda item: (-item[1], int(item[0].rsplit("n", 1)[1]), item[0].split("_n")[0])
    )[0]
    chosen_mode, chosen_n = chosen_name.rsplit("_n", 1)
    chosen_mode = chosen_mode.removeprefix("gradient_free_calibrated_")
    paired_deltas = {
        config: {session: values[config] - no_calibration_per_session[session]
                 for session, values in per_session.items()}
        for config in means
    }
    mean_paired_deltas = {
        config: sum(values.values()) / len(values) for config, values in paired_deltas.items()
    }
    session_unit_counts = {session_name_from_path(path): nwb_unit_count(path) for path in all_files}
    formal_scope = {
        "data_dir": str(data_dir), "task": args.task, "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "test_sessions": [session_name_from_path(path) for path in test_files],
        "test_session_unit_counts": {
            session_name_from_path(path): session_unit_counts[session_name_from_path(path)]
            for path in test_files
        },
    }
    formal_test_scope_id = hashlib.sha256(
        json.dumps(formal_scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
        "purpose": (
            "validation_only_fixed_gradient_free_evaluation"
            if fixed_protocol
            else "validation_only_gradient_free_protocol_selection"
        ),
        "ckpt": str(ckpt_path), "ckpt_sha256": sha256_file(ckpt_path), "teacher_ckpt": str(teacher_ckpt), "teacher_ckpt_sha256": sha256_file(teacher_ckpt), "data_dir": str(data_dir), "cache_dir": str(cache_dir) if cache_dir else None, "signal_view": args.signal_view, "variant": args.variant,
        "training_run_metadata": str(run_metadata_path.resolve()), "training_run_metadata_sha256": sha256_file(run_metadata_path),
        "task": args.task, "seed": args.seed, "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "session_splits": {"train": [session_name_from_path(p) for p in train_files], "val": [session_name_from_path(p) for p in val_files], "test": [session_name_from_path(p) for p in test_files]},
        "session_unit_counts": session_unit_counts,
        "formal_test_scope_id": formal_test_scope_id,
        "configs": config_names, "per_session_r2": per_session,
        "mean_r2": means, "trial_selections": selections, "common_evaluation_start_index": args.pool_size,
        "zero_identity_no_calibration": {"per_session_r2": no_calibration_per_session, "mean_r2": sum(no_calibration_per_session.values()) / len(no_calibration_per_session), "description": "Non-learned all-zero identity control; no calibration spikes or identity encoder."},
        "paired_delta_vs_zero_identity_no_calibration": paired_deltas,
        "mean_paired_delta_vs_zero_identity_no_calibration": mean_paired_deltas,
        "calibration_trial_selection_uses_behavior_labels": False,
        "protocol_selection_uses_validation_behavior_labels": not fixed_protocol,
        "fixed_protocol": fixed_protocol,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
        "validation_complete": validation_complete,
        "selected_protocol": {"selection_mode": chosen_mode, "calibration_n": int(chosen_n), "pool_size": args.pool_size, "validation_mean_r2": chosen_r2, "validation_paired_delta_vs_zero_identity_no_calibration": mean_paired_deltas[chosen_name]},
        "outcome_interpretation": {
            "primary_estimand": "unweighted mean across formal test sessions of paired R2(calibrated - zero identity) on common trials[pool_size:]",
            "supports_gradient_free_calibration_rule": "mean paired delta > 0 and positive-session count >= 4 of 6",
            "supports_usable_cross_session_decoding_rule": "calibrated mean R2 > 0",
        },
    }
    results = Path(__file__).resolve().parents[1] / "results"
    out_path = Path(args.out_path) if args.out_path else results / f"p3_gradient_free_protocol_selection_{args.variant.lower()}_s{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lock = {key: payload[key] for key in ("ckpt", "ckpt_sha256", "teacher_ckpt", "teacher_ckpt_sha256", "data_dir", "variant", "task", "seed", "split_counts", "max_units_exclusive", "selected_protocol", "training_run_metadata", "training_run_metadata_sha256", "outcome_interpretation", "formal_test_scope_id")}
    lock.update({"schema_version": 1, "purpose": "locked_gradient_free_formal_test_protocol", "validation_complete": validation_complete, "source_validation_result": str(out_path.resolve()), "source_validation_result_sha256": sha256_file(out_path)})
    lock_path = Path(args.lock_path) if args.lock_path else out_path.with_name(out_path.stem + "_lock.json")
    if validation_complete and not args.no_formal_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"Validation-only evaluation: {out_path}")
    if validation_complete and not args.no_formal_lock:
        print(f"Locked protocol: {lock_path}")
    else:
        print("No formal protocol lock written")


if __name__ == "__main__":
    main()
