#!/usr/bin/env python3
"""Fixed epoch-window C1 scoring for one view of a shared paired checkpoint.

The generic evaluator intentionally trusts a run's single ``signal_view`` and
single side-feature normalizer.  A C1 checkpoint has two legitimate views and
two separately fitted source-only T4 normalizers, so this small wrapper uses
the same lower-level frozen scorer/model routines while selecting the
view-specific configuration explicitly.  It opens only the 27 source and six
development files named by the strict C1 manifest; formal files are never
resolved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from dandi688_gradient_free_protocol import sha256_file
from eval_adaptation_dandi688 import (
    BEHAVIOR_SCALING_FACTOR,
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    attach_side_features,
    load_session_with_trials,
)
from eval_epoch_window_generic_dandi688 import (
    compute_protocol_epochs,
    compute_variant_score,
    select_epoch_window_checkpoints,
)
from mc_maze.multisession_datamodule import (
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    nwb_unit_count,
    session_name_from_path,
)
from mc_maze.unit_side_features import base_feature_group, fit_side_feature_stats, side_feature_stats_sha256
from select_gradient_free_protocol_dandi688 import (
    evaluate_session_configs,
    load_frozen_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--signal-view", choices=["sua", "pseudo_mua"], required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--total-epochs", type=int, default=12)
    parser.add_argument("--burn-in", type=int, default=4)
    parser.add_argument("--calibration-n", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--train-val-manifest", required=True)
    return parser.parse_args()


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing C1 run metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError("C1 epoch evaluation requires completed run metadata")
    if payload.get("training_kind") != "shared_paired_view":
        raise ValueError("C1 shared evaluator refuses a non-paired run")
    if payload.get("held_out_test_evaluated") is not False:
        raise ValueError("C1 run metadata must state held_out_test_evaluated=false")
    if payload.get("formal_sua_files_opened") is not False:
        raise ValueError("C1 run metadata must state formal_sua_files_opened=false")
    return payload


def _view_feature_config(
    metadata: dict[str, Any], view: str, train_files: list[Path]
) -> tuple[str, int, int | None, object, object, str, Path]:
    views = metadata.get("view_configs")
    if not isinstance(views, dict) or view not in views:
        raise ValueError(f"C1 metadata has no config for {view!r}")
    config = views[view]
    if not isinstance(config, dict) or config.get("signal_view") != view:
        raise ValueError(f"C1 metadata view config mismatch for {view!r}")
    side = config.get("side_features")
    if not isinstance(side, dict):
        raise ValueError(f"C1 metadata side features missing for {view!r}")
    group = side.get("group")
    if group not in {"t4", "ts4"}:
        raise ValueError(f"C1 evaluator only accepts T4/TS4, got {group!r}")
    pool_size = int(side.get("pool_size", -1))
    if pool_size != 50 or int(side.get("side_dim", -1)) != 4:
        raise ValueError("C1 requires four-dimensional T4/TS4 fit from pool=50")
    cache_dir = Path(str(config.get("cache_dir", ""))).expanduser().resolve()
    if not str(cache_dir):
        raise ValueError(f"C1 metadata cache path missing for {view!r}")
    mean, std = fit_side_feature_stats(
        train_files,
        feature_group=base_feature_group(group),
        pool_size=pool_size,
        cache_dir=cache_dir,
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
        signal_view=view,
    )
    observed = side_feature_stats_sha256(mean, std)
    if observed != side.get("normalization_sha256"):
        raise ValueError(
            f"C1 {view} train-only normalizer hash drifted: "
            f"expected {side.get('normalization_sha256')!r}, got {observed!r}"
        )
    return group, pool_size, side.get("permutation_seed"), mean, std, observed, cache_dir


def main() -> None:
    args = parse_args()
    if (args.total_epochs, args.burn_in, args.calibration_n, args.pool_size) != (12, 4, 30, 50):
        raise ValueError("C1 freezes total=12, burn_in=4, forward Q=30, pool=50")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output = Path(args.out_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"C1 evaluator refuses to overwrite {output}")
    metadata = _load_metadata(run_dir)
    manifest = Path(args.train_val_manifest).expanduser().resolve()
    if sha256_file(manifest) != metadata.get("train_val_manifest_sha256"):
        raise ValueError("C1 train/validation manifest SHA drifted")
    train_files, val_files, test_names = load_frozen_train_val_manifest(
        manifest, Path(metadata["data_dir"])
    )
    # The loader returns test *names* only.  Assert its behavior rather than
    # resolving any formal paths.
    if len(train_files) != 27 or len(val_files) != 6 or len(test_names) != 6:
        raise ValueError("C1 strict manifest no longer describes 27/6/6")
    group, feature_pool, permutation_seed, side_mean, side_std, normalizer_sha, cache_dir = (
        _view_feature_config(metadata, args.signal_view, train_files)
    )
    behavior_mean, behavior_std = fit_behavior_stats(train_files, 20, cache_dir=cache_dir)
    protocol_epochs = compute_protocol_epochs(args.total_epochs, args.burn_in)
    checkpoints = select_epoch_window_checkpoints(run_dir, protocol_epochs, args.total_epochs)
    teacher = Path(metadata["teacher_checkpoint"]).expanduser().resolve()
    if sha256_file(teacher) != metadata.get("teacher_sha256"):
        raise ValueError("C1 teacher SHA drifted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_epoch: dict[str, dict[str, Any]] = {}
    session_splits = {
        "train": [session_name_from_path(path) for path in train_files],
        "val": [session_name_from_path(path) for path in val_files],
        "test": test_names,
    }
    for epoch, checkpoint in checkpoints.items():
        model = load_frozen_model(checkpoint, teacher, "B3S", device)
        per_session: dict[str, float] = {}
        selections: dict[str, Any] = {}
        with torch.no_grad():
            for path in val_files:
                record = load_session_with_trials(
                    path,
                    20,
                    WINDOW_SIZE,
                    args.pool_size,
                    TRIAL_LENGTH,
                    PAD_VALUE,
                    behavior_mean,
                    behavior_std,
                    cache_dir=cache_dir,
                    signal_view=args.signal_view,
                )
                record = attach_side_features(
                    record,
                    path,
                    side_feature_group=group,
                    waveform_feature_group=base_feature_group(group),
                    pool_size=feature_pool,
                    permutation_seed=permutation_seed,
                    mean=side_mean,
                    std=side_std,
                    cache_dir=cache_dir,
                )
                session_result, session_selection = evaluate_session_configs(
                    record,
                    [("first", args.calibration_n)],
                    args.pool_size,
                    model,
                    device,
                )
                name = "gradient_free_calibrated_first_n30"
                per_session[record["name"]] = float(session_result[name])
                selections[record["name"]] = session_selection[name]
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "per_session_r2": per_session,
            "mean_r2": sum(per_session.values()) / len(per_session),
            "trial_selections": selections,
        }
    score_values = {int(epoch): row["mean_r2"] for epoch, row in per_epoch.items()}
    payload = {
        "schema_version": 1,
        "purpose": "paired_view_c1_fixed_epoch_window_evaluation",
        "generated_by": "eval_paired_view_c1_epoch_window.py",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "run_metadata_path": str(run_dir / "run_metadata.json"),
        "run_metadata_sha256": sha256_file(run_dir / "run_metadata.json"),
        "variant": "B3S",
        "seed": metadata["seed"],
        "task": "CO",
        "signal_view": args.signal_view,
        "shared_weights": True,
        "shared_training_side_features": group,
        "teacher_ckpt": str(teacher),
        "teacher_ckpt_sha256": sha256_file(teacher),
        "train_val_manifest": str(manifest),
        "train_val_manifest_sha256": sha256_file(manifest),
        "data_manifest": metadata["data_manifest"],
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "session_splits": session_splits,
        "session_unit_counts": {
            session_name_from_path(path): nwb_unit_count(path)
            for path in train_files + val_files
        },
        "protocol": {
            "total_epochs": 12,
            "burn_in_epochs": 4,
            "epoch_window": list(protocol_epochs),
            "selection_mode": "first",
            "calibration_n": 30,
            "train_activity_calibration_n": 10,
            "evaluation_forward_calibration_n": 30,
            "label_feature_calibration_n": 50,
            "pool_size": 50,
            "score_start": 50,
        },
        "epoch_list": list(protocol_epochs),
        "per_epoch": per_epoch,
        "per_epoch_mean_r2": {str(epoch): score_values[epoch] for epoch in protocol_epochs},
        "variant_score": compute_variant_score(score_values, protocol_epochs),
        "checkpoint_selection_rule": "pre_declared_fixed_epoch_window_no_argmax",
        "view_normalizer_sha256": normalizer_sha,
        "calibration_features_use_behavior_labels": True,
        "calibration_feature_label_scope": "chronological_rewarded_trials[0:50]",
        "calibration_trial_selection_uses_behavior_labels": False,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "no_test_files_evaluated": True,
        "formal_sua_files_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "score": payload["variant_score"]}, sort_keys=True))


if __name__ == "__main__":
    main()

