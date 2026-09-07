#!/usr/bin/env python3
"""Forward-only CPU screen for A5 transferred/aged carrier and A3 breakage dose response.

This script emits deterministic JSON receipts only.  It does not launch training,
GPU jobs, background workers, or watchers.  Real NWB paths are optional and
should remain behind explicit flags; unit tests use synthetic fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dandi688_gradient_free_protocol import sha256_file
from eval_adaptation_dandi688 import (
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    attach_side_features,
    build_calib_trials_for_indices,
    eval_r2,
    load_session_with_trials,
    load_side_feature_stats_for_run_metadata,
    parse_split_counts,
)
from mc_maze.carrier_transfer_and_breakage import (
    ACTIVITY_CALIBRATION_N,
    ADMISSIBLE_TRANSFER_PAIRS,
    AGED_TRANSFER_PAIRS,
    BASE_RANDOMIZATION_SEED,
    BREAKAGE_LEVELS,
    BREAKAGE_MODES,
    EVALUATION_START_TRIAL,
    PROTOCOL_ID,
    ROW_MATCHING_RULE,
    SCHEMA_VERSION,
    SEALED_FORMAL_TEST_SESSIONS,
    T4_LABEL_POOL_N,
    UNIT_SUBSET_SIZES,
    VALIDATION_SESSIONS,
    apply_electrode_pooling_consistent,
    apply_unit_dropout_consistent,
    assert_session_allowed,
    assert_transfer_pair_admissible,
    breakage_interaction_statistic,
    build_transfer_arm_carriers,
    calendar_gap_days,
    compute_transfer_deltas,
    derived_seed,
    pooled_mean,
    transfer_pair_subset,
    unit_dropout_indices,
    unit_subset_indices,
    zero_carrier,
)
from mc_maze.multisession_datamodule import (
    chronological_session_split,
    discover_nwb_files,
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    session_name_from_path,
)
from select_gradient_free_protocol_dandi688 import (
    evaluate_session_configs,
    load_frozen_model,
)

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))


def _force_cpu() -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return torch.device("cpu")


def _write_receipt(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _session_path_map(data_dir: Path, *, task: str, max_units_exclusive: int | None) -> dict[str, Path]:
    files = discover_nwb_files(data_dir, task=task, max_units_exclusive=max_units_exclusive)
    mapping = {session_name_from_path(path): path for path in files}
    for session in VALIDATION_SESSIONS:
        if session not in mapping:
            raise FileNotFoundError(f"validation session missing from data_dir: {session}")
    for sealed in SEALED_FORMAL_TEST_SESSIONS:
        if sealed in mapping:
            raise ValueError(f"refusing to proceed while sealed session is discoverable: {sealed}")
    return mapping


def _load_recipient_record(
    *,
    nwb_path: Path,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    cache_dir: Path | None,
    signal_view: str,
    side_feature_config,
) -> dict:
    rec = load_session_with_trials(
        nwb_path,
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        calib_n=ACTIVITY_CALIBRATION_N,
        max_trial_length=TRIAL_LENGTH,
        pad_value=PAD_VALUE,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        cache_dir=cache_dir,
        signal_view=signal_view,
    )
    if side_feature_config is None:
        raise ValueError("carrier screen requires a B3S/T4 checkpoint with side features")
    (
        side_feature_group,
        waveform_feature_group,
        side_pool_size,
        permutation_seed,
        side_mean,
        side_std,
    ) = side_feature_config
    return attach_side_features(
        rec,
        nwb_path,
        side_feature_group=side_feature_group,
        waveform_feature_group=waveform_feature_group,
        pool_size=side_pool_size,
        permutation_seed=permutation_seed,
        mean=side_mean,
        std=side_std,
        cache_dir=cache_dir,
    )


def _load_normalized_t4(
    nwb_path: Path,
    *,
    side_feature_config,
    cache_dir: Path | None,
    signal_view: str,
) -> np.ndarray:
    (
        _side_feature_group,
        waveform_feature_group,
        side_pool_size,
        _permutation_seed,
        side_mean,
        side_std,
    ) = side_feature_config
    from mc_maze.unit_side_features import load_unit_side_features

    features, _ = load_unit_side_features(
        nwb_path,
        feature_group="t4",
        pool_size=side_pool_size,
        mean=side_mean,
        std=side_std,
        cache_dir=cache_dir,
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
        signal_view=signal_view,
    )
    return np.asarray(features, dtype=np.float32)


def _apply_side_override(rec: dict, side_features: np.ndarray) -> dict:
    side = np.asarray(side_features, dtype=np.float32)
    if side.shape != (rec["n_units"], 4):
        raise ValueError(
            f"{rec['name']}: side override shape {side.shape} != ({rec['n_units']}, 4)"
        )
    return {**rec, "side_features": side}


def run_transfer_pair(
    *,
    donor_session: str,
    recipient_session: str,
    model,
    device: torch.device,
    session_paths: dict[str, Path],
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    cache_dir: Path | None,
    signal_view: str,
    side_feature_config,
    pool_size: int,
    selection_mode: str,
    calibration_n: int,
    checkpoint_sha256: str,
    base_seed: int,
) -> dict[str, Any]:
    assert_transfer_pair_admissible(donor_session, recipient_session)
    donor_path = session_paths[donor_session]
    recipient_path = session_paths[recipient_session]
    donor_carrier = _load_normalized_t4(
        donor_path,
        side_feature_config=side_feature_config,
        cache_dir=cache_dir,
        signal_view=signal_view,
    )
    rec = _load_recipient_record(
        nwb_path=recipient_path,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        cache_dir=cache_dir,
        signal_view=signal_view,
        side_feature_config=side_feature_config,
    )
    own_carrier = np.asarray(rec["side_features"], dtype=np.float32)
    arms_bundle = build_transfer_arm_carriers(
        own_carrier,
        donor_carrier,
        recipient_n_units=rec["n_units"],
        rule=ROW_MATCHING_RULE,
    )
    matching = arms_bundle.matching
    arms = {
        "own_carrier": arms_bundle.own_full,
        "own_truncated": arms_bundle.own_truncated,
        "transferred_carrier": arms_bundle.transferred,
        "zero_carrier": arms_bundle.zero,
    }
    per_arm_scores: dict[str, float] = {}
    for arm_name, side in arms.items():
        trial_r2, _ = evaluate_session_configs(
            _apply_side_override(rec, side),
            configs=[(selection_mode, calibration_n)],
            pool_size=pool_size,
            model=model,
            device=device,
        )
        per_arm_scores[arm_name] = float(next(iter(trial_r2.values())))
    deltas = compute_transfer_deltas(per_arm_scores)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "experiment": "A5_transferred_and_aged_carrier",
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint_sha256": checkpoint_sha256,
        "donor_session": donor_session,
        "recipient_session": recipient_session,
        "pair_subset": transfer_pair_subset(donor_session, recipient_session),
        "calendar_gap_days": calendar_gap_days(donor_session, recipient_session),
        "row_matching_rule": matching.rule,
        "row_matching_rule_version": matching.rule_version,
        "zero_fill_pattern_digest": arms_bundle.pattern.digest(),
        "matched_rows": matching.matched_rows,
        "donor_rows_dropped": matching.donor_rows_dropped,
        "recipient_rows_zero_filled": matching.recipient_rows_zero_filled,
        "donor_unit_count": matching.donor_rows,
        "recipient_unit_count": matching.recipient_rows,
        "N_donor": matching.donor_rows,
        "N_recipient": matching.recipient_rows,
        "zero_fill_fraction": arms_bundle.zero_fill_fraction,
        "breakage_level": 0.0,
        "breakage_mode": None,
        "randomization_seed": derived_seed(
            family="transfer",
            base_seed=base_seed,
            donor=donor_session,
            recipient=recipient_session,
        ),
        "query_window": {
            "pool_size": pool_size,
            "calibration_n": calibration_n,
            "evaluation_start_trial": pool_size,
            "activity_calibration_n": ACTIVITY_CALIBRATION_N,
            "t4_label_pool_n": T4_LABEL_POOL_N,
            "registered_evaluation_start_trial": EVALUATION_START_TRIAL,
        },
        "per_session_scores": per_arm_scores,
        "deltas": deltas.as_dict(),
        "primary_statistic": deltas.transferred_minus_own_truncated,
        "secondary_statistic": deltas.transferred_minus_own_full,
        "nuisance_statistic": deltas.own_truncated_minus_own_full,
        "pooled_score": pooled_mean(per_arm_scores),
        "sealed_test_sessions_opened": False,
    }


def _score_arms_on_record(
    model,
    base_rec: dict,
    *,
    keep: np.ndarray,
    device: torch.device,
    pool_size: int,
    selection_mode: str,
    calibration_n: int,
) -> tuple[float, float]:
    neural = base_rec["neural"][:, keep]
    calib = base_rec["calib_trials"][..., keep]
    carrier = np.asarray(base_rec["side_features"], dtype=np.float32)[keep]
    control = zero_carrier(int(keep.size))
    if calib.shape[-1] != neural.shape[1] or carrier.shape[0] != neural.shape[1]:
        raise ValueError("breakage left neural/calib/side streams inconsistent")
    scores: dict[str, float] = {}
    for arm_name, side in (("carrier", carrier), ("control", control)):
        trial_r2, _ = evaluate_session_configs(
            {
                **base_rec,
                "neural": neural,
                "calib_trials": calib,
                "side_features": side,
                "n_units": int(keep.size),
            },
            configs=[(selection_mode, calibration_n)],
            pool_size=pool_size,
            model=model,
            device=device,
        )
        scores[arm_name] = float(next(iter(trial_r2.values())))
    return scores["carrier"], scores["control"]


def run_breakage_session(
    *,
    session_name: str,
    breakage_mode: str,
    model,
    device: torch.device,
    session_paths: dict[str, Path],
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    cache_dir: Path | None,
    signal_view: str,
    side_feature_config,
    pool_size: int,
    selection_mode: str,
    calibration_n: int,
    checkpoint_sha256: str,
    base_seed: int,
) -> dict[str, Any]:
    assert_session_allowed(session_name)
    if breakage_mode not in BREAKAGE_MODES:
        raise ValueError(f"unsupported breakage mode {breakage_mode!r}")
    rec = _load_recipient_record(
        nwb_path=session_paths[session_name],
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        cache_dir=cache_dir,
        signal_view=signal_view,
        side_feature_config=side_feature_config,
    )
    from mc_maze.unit_side_features import load_session_electrode_ids

    electrode_ids = load_session_electrode_ids(session_paths[session_name])
    ladder: list[dict[str, Any]] = []
    carrier_scores: list[float] = []
    control_scores: list[float] = []
    interaction_levels: list[float] = []

    if breakage_mode == "unit_dropout":
        for level in BREAKAGE_LEVELS:
            seed = derived_seed(
                family=f"breakage:{breakage_mode}",
                base_seed=base_seed,
                session=session_name,
                level=level,
            )
            keep = unit_dropout_indices(rec["n_units"], dropout_p=level, seed=seed)
            carrier_score, control_score = _score_arms_on_record(
                model,
                rec,
                keep=keep,
                device=device,
                pool_size=pool_size,
                selection_mode=selection_mode,
                calibration_n=calibration_n,
            )
            carrier_scores.append(carrier_score)
            control_scores.append(control_score)
            interaction_levels.append(float(level))
            ladder.append(
                {
                    "breakage_level": level,
                    "seed": seed,
                    "kept_units": int(keep.size),
                }
            )
    elif breakage_mode == "electrode_pooling":
        full_keep = np.arange(rec["n_units"], dtype=np.int64)
        carrier_full, control_full = _score_arms_on_record(
            model,
            rec,
            keep=full_keep,
            device=device,
            pool_size=pool_size,
            selection_mode=selection_mode,
            calibration_n=calibration_n,
        )
        pooled_neural, pooled_side, pooled_calib, channel_ids = apply_electrode_pooling_consistent(
            neural=rec["neural"],
            side_features=np.asarray(rec["side_features"], dtype=np.float32),
            calib_trials=rec["calib_trials"],
            electrode_ids=electrode_ids,
        )
        pooled_control = zero_carrier(channel_ids.size)
        pooled_scores: dict[str, float] = {}
        for arm_name, side in (("carrier", pooled_side), ("control", pooled_control)):
            trial_r2, _ = evaluate_session_configs(
                {
                    **rec,
                    "neural": pooled_neural,
                    "calib_trials": pooled_calib,
                    "side_features": side,
                    "n_units": int(channel_ids.size),
                    "electrode_ids": channel_ids,
                },
                configs=[(selection_mode, calibration_n)],
                pool_size=pool_size,
                model=model,
                device=device,
            )
            pooled_scores[arm_name] = float(next(iter(trial_r2.values())))
        carrier_scores.extend([carrier_full, pooled_scores["carrier"]])
        control_scores.extend([control_full, pooled_scores["control"]])
        interaction_levels.extend([0.0, 1.0])
        ladder.extend(
            [
                {"breakage_level": 0.0, "kept_units": int(rec["n_units"]), "pooled_channels": None},
                {
                    "breakage_level": 1.0,
                    "kept_units": int(rec["n_units"]),
                    "pooled_channels": int(channel_ids.size),
                    "seed": derived_seed(
                        family="breakage:electrode_pooling",
                        base_seed=base_seed,
                        session=session_name,
                    ),
                },
            ]
        )
    elif breakage_mode == "unit_subset":
        subset_sizes = (rec["n_units"], *UNIT_SUBSET_SIZES)
        for subset_size in subset_sizes:
            if subset_size > rec["n_units"]:
                continue
            seed = derived_seed(
                family=f"breakage:{breakage_mode}",
                base_seed=base_seed,
                session=session_name,
                subset_size=subset_size,
            )
            if subset_size == rec["n_units"]:
                keep = np.arange(rec["n_units"], dtype=np.int64)
            else:
                keep = unit_subset_indices(rec["n_units"], subset_size=subset_size, seed=seed)
            carrier_score, control_score = _score_arms_on_record(
                model,
                rec,
                keep=keep,
                device=device,
                pool_size=pool_size,
                selection_mode=selection_mode,
                calibration_n=calibration_n,
            )
            carrier_scores.append(carrier_score)
            control_scores.append(control_score)
            interaction_levels.append(float(subset_size))
            ladder.append(
                {
                    "subset_size": int(subset_size),
                    "seed": seed,
                    "kept_units": int(keep.size),
                }
            )
    else:
        raise ValueError(f"unhandled breakage mode {breakage_mode!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "experiment": "A3_correspondence_breakage_dose_response",
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint_sha256": checkpoint_sha256,
        "session": session_name,
        "breakage_mode": breakage_mode,
        "breakage_ladder": ladder,
        "randomization_seed": derived_seed(
            family=f"breakage_session:{breakage_mode}",
            base_seed=base_seed,
            session=session_name,
        ),
        "query_window": {
            "pool_size": pool_size,
            "calibration_n": calibration_n,
            "evaluation_start_trial": pool_size,
            "activity_calibration_n": ACTIVITY_CALIBRATION_N,
            "t4_label_pool_n": T4_LABEL_POOL_N,
            "registered_evaluation_start_trial": EVALUATION_START_TRIAL,
        },
        "carrier_scores": carrier_scores,
        "control_scores": control_scores,
        "interaction_statistic": breakage_interaction_statistic(
            carrier_scores,
            control_scores,
            breakage_levels=interaction_levels,
        ),
        "pooled_carrier_score": pooled_mean({str(i): v for i, v in enumerate(carrier_scores)}),
        "pooled_control_score": pooled_mean({str(i): v for i, v in enumerate(control_scores)}),
        "sealed_test_sessions_opened": False,
    }


def _add_transfer_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("transfer", help="A5 transferred and aged carrier screen")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="B3S")
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="CO")
    parser.add_argument("--split-counts", type=str, default="27,6,6")
    parser.add_argument("--max-units-exclusive", type=int, default=100)
    parser.add_argument("--train-val-manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--pool-size", type=int, default=T4_LABEL_POOL_N)
    parser.add_argument("--calibration-n", type=int, default=ACTIVITY_CALIBRATION_N)
    parser.add_argument("--selection-mode", type=str, default="first")
    parser.add_argument("--donor-session", type=str, default=None)
    parser.add_argument("--recipient-session", type=str, default=None)
    parser.add_argument("--aged-sweep", action="store_true")
    parser.add_argument("--base-seed", type=int, default=BASE_RANDOMIZATION_SEED)
    parser.add_argument("--out", type=Path, required=True)


def _add_breakage_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("breakage", help="A3 correspondence-breakage dose response")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="B3S")
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="CO")
    parser.add_argument("--split-counts", type=str, default="27,6,6")
    parser.add_argument("--max-units-exclusive", type=int, default=100)
    parser.add_argument("--train-val-manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--pool-size", type=int, default=T4_LABEL_POOL_N)
    parser.add_argument("--calibration-n", type=int, default=ACTIVITY_CALIBRATION_N)
    parser.add_argument("--selection-mode", type=str, default="first")
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument(
        "--breakage-mode",
        type=str,
        choices=BREAKAGE_MODES,
        required=True,
    )
    parser.add_argument("--base-seed", type=int, default=BASE_RANDOMIZATION_SEED)
    parser.add_argument("--out", type=Path, required=True)


def _load_screen_context(args: argparse.Namespace):
    device = _force_cpu()
    ckpt_path = args.ckpt.expanduser().resolve()
    teacher_ckpt = args.teacher_ckpt.expanduser().resolve()
    run_metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
    split_counts = parse_split_counts(args.split_counts)
    data_dir = args.data_dir.expanduser().resolve()
    if args.train_val_manifest is not None:
        manifest = load_frozen_train_val_manifest(args.train_val_manifest)
        train_files = [data_dir / f"{name}_behavior+ecephys.nwb" for name in manifest["session_splits"]["train"]]
        val_names = manifest["session_splits"]["val"]
    else:
        all_files = discover_nwb_files(
            data_dir,
            task=args.task,
            max_units_exclusive=args.max_units_exclusive,
        )
        train_files, _val_files, _test_files = chronological_session_split(
            all_files,
            split_counts,
            max_units_exclusive=args.max_units_exclusive,
        )
        val_names = list(VALIDATION_SESSIONS)
    for name in val_names:
        assert_session_allowed(name)
    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, bin_size_ms=20, cache_dir=args.cache_dir
    )
    side_feature_config = load_side_feature_stats_for_run_metadata(
        run_metadata, train_files, args.cache_dir
    )
    model = load_frozen_model(
        ckpt_path,
        teacher_ckpt,
        args.variant,
        device,
        identity_mode="calibrated",
    )
    session_paths = _session_path_map(
        data_dir,
        task=args.task,
        max_units_exclusive=args.max_units_exclusive,
    )
    return {
        "device": device,
        "ckpt_path": ckpt_path,
        "model": model,
        "behavior_mean": behavior_mean,
        "behavior_std": behavior_std,
        "cache_dir": args.cache_dir,
        "signal_view": str(run_metadata.get("signal_view", "sua")),
        "side_feature_config": side_feature_config,
        "session_paths": session_paths,
        "checkpoint_sha256": sha256_file(ckpt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_transfer_parser(subparsers)
    _add_breakage_parser(subparsers)
    args = parser.parse_args()
    ctx = _load_screen_context(args)
    if args.command == "transfer":
        pairs: list[tuple[str, str]]
        if args.aged_sweep:
            pairs = list(AGED_TRANSFER_PAIRS)
        elif args.donor_session and args.recipient_session:
            pairs = [(args.donor_session, args.recipient_session)]
        else:
            raise ValueError("transfer requires --donor-session/--recipient-session or --aged-sweep")
        receipts = [
            run_transfer_pair(
                donor_session=donor,
                recipient_session=recipient,
                model=ctx["model"],
                device=ctx["device"],
                session_paths=ctx["session_paths"],
                behavior_mean=ctx["behavior_mean"],
                behavior_std=ctx["behavior_std"],
                cache_dir=ctx["cache_dir"],
                signal_view=ctx["signal_view"],
                side_feature_config=ctx["side_feature_config"],
                pool_size=args.pool_size,
                selection_mode=args.selection_mode,
                calibration_n=args.calibration_n,
                checkpoint_sha256=ctx["checkpoint_sha256"],
                base_seed=args.base_seed,
            )
            for donor, recipient in pairs
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "experiment": "A5_transferred_and_aged_carrier",
            "created_at": datetime.now().astimezone().isoformat(),
            "checkpoint_sha256": ctx["checkpoint_sha256"],
            "row_matching_rule": ROW_MATCHING_RULE,
            "pairs": receipts,
            "sealed_test_sessions_opened": False,
        }
        digest = _write_receipt(args.out, payload)
        print(json.dumps({"receipt": str(args.out), "sha256": digest, "pairs": len(receipts)}))
        return
    if args.command == "breakage":
        payload = run_breakage_session(
            session_name=args.session,
            breakage_mode=args.breakage_mode,
            model=ctx["model"],
            device=ctx["device"],
            session_paths=ctx["session_paths"],
            behavior_mean=ctx["behavior_mean"],
            behavior_std=ctx["behavior_std"],
            cache_dir=ctx["cache_dir"],
            signal_view=ctx["signal_view"],
            side_feature_config=ctx["side_feature_config"],
            pool_size=args.pool_size,
            selection_mode=args.selection_mode,
            calibration_n=args.calibration_n,
            checkpoint_sha256=ctx["checkpoint_sha256"],
            base_seed=args.base_seed,
        )
        digest = _write_receipt(args.out, payload)
        print(json.dumps({"receipt": str(args.out), "sha256": digest}))
        return
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
