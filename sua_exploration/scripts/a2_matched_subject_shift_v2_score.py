#!/usr/bin/env python3
"""Score one unchanged A2 v2 source run on one matched scoring domain.

This evaluator is intentionally explicit about all transfer invariants:

* it loads a source-trained `source_z4` or `source_t4` run exactly once;
* it scores the predeclared epoch 5--12 checkpoint window, never selecting
  a target-domain checkpoint;
* both sub-C development and external sub-M use M30 / trial-30 chronology;
* behavior and T4 normalizers are refit from the strict 27-session source
  roster only, then reused unchanged on the target domain; and
* it never resolves or opens a formal sub-C test NWB.

The module has a dry-run mode that reads only source metadata/checkpoint file
hashes.  Actual model/NWB work requires `--launch`; the enclosing runner
requires explicit GPU authorization before calling that mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))
SCRIPTS_ROOT = SUA_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from mc_maze import a2_matched_subject_shift_v2_core as core


class A2V2ScoreError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A2V2ScoreError(message)


def _write_result_once(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """Write an O_EXCL/fsync/0444 body plus required SHA-256 sidecar."""
    return core.write_immutable_json(path, payload)


def _source_bindings(source_arm: str, seed: int, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, str]]:
    arm = core.source_arm_for_name(source_arm)
    expected_name = core.source_run_name(source_arm, seed)
    run_dir = run_dir.expanduser().resolve()
    _require(run_dir.name == expected_name, f"run-dir name must be exactly {expected_name}")
    fingerprint = core.source_run_receipt_fingerprint(run_dir, source_arm=source_arm, seed=seed)
    metadata = core.load_json_object(run_dir / "run_metadata.json")
    return fingerprint, metadata, arm


def build_dry_run(
    source_arm: str,
    seed: int,
    domain: str,
    run_dir: Path,
    out_path: Path,
    *,
    official_preflight: Path | None = None,
    result_root: Path = core.RESULT_ROOT,
) -> dict[str, Any]:
    fingerprint, metadata, arm = _source_bindings(source_arm, seed, run_dir)
    sessions = core.expected_domain_sessions(domain)
    preflight_digest = None
    if official_preflight is not None:
        _preflight, preflight_digest = core.load_verified_official_preflight(
            official_preflight,
            result_root=result_root,
        )
    return {
        "schema_version": 2,
        "mode": "dry_run",
        "status": "NOT_AUTHORIZED_FOR_A2_V2_DOMAIN_SCORING",
        "screen_id": core.SCREEN_ID,
        "source_arm": source_arm,
        "arm": arm["arm"],
        "seed": seed,
        "domain": domain,
        "run_dir": str(run_dir.expanduser().resolve()),
        "out_path": str(out_path.expanduser().resolve()),
        "contract_path": str(core.CONTRACT_PATH),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "implementation_bindings": core.current_implementation_bindings(),
        "implementation_bindings_sha256": core.implementation_bindings_sha256(core.current_implementation_bindings()),
        "official_preflight_path": str(official_preflight.expanduser().resolve()) if official_preflight is not None else None,
        "official_preflight_sha256": preflight_digest,
        "source_run": fingerprint,
        "source_run_status": metadata.get("status"),
        "expected_sessions": list(sessions),
        "query_policy": core.frozen_query_policy(),
        "normalizer_policy": core.frozen_normalizer_policy(),
        "source_checkpoint_scored_unchanged_on_both_domains_required": True,
        "target_domain_normalizer_fit_forbidden": True,
        "formal_subc_test_nwb_opened": False,
        "checkpoint_loaded": False,
        "nwb_loaded": False,
        "model_forward_performed": False,
    }


def _normalizer_pair_equal(left: tuple[Any, Any], right: tuple[Any, Any]) -> bool:
    import numpy as np

    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(left, right))


def _load_training_cache_pair(path: Path, *, label: str) -> tuple[Any, Any]:
    """Read a pre-existing trainer cache without repairing or rewriting it."""
    import numpy as np

    _require(path.is_file(), f"training-path {label} normalizer cache is missing")
    try:
        with np.load(path, allow_pickle=False) as cache:
            mean = cache["mean"].astype(np.float32, copy=False)
            std = cache["std"].astype(np.float32, copy=False)
    except (KeyError, OSError, ValueError) as exc:
        raise A2V2ScoreError(f"cannot read immutable training-path {label} normalizer cache: {path}") from exc
    _require(mean.shape == std.shape and mean.ndim >= 1, f"invalid training-path {label} normalizer cache shape")
    return mean, std


def _fit_source_normalizers(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[Any, Any], tuple[Any, Any], tuple[Any, ...], list[Path]]:
    """Recompute source stats and prove cached training-path values are identical.

    The scorer itself never adopts a target-derived cache.  It performs a
    cache-free source recomputation, separately loads the exact cache root the
    trainer used, and rejects any bitwise value mismatch before a target NWB
    is opened.  This makes the cache an audited acceleration artifact rather
    than an unverified normalizer authority.
    """
    import numpy as np
    from mc_maze.multisession_datamodule import _behavior_stats_cache_path, fit_behavior_stats, session_name_from_path
    from mc_maze.unit_side_features import _side_stats_cache_path, base_feature_group, fit_side_feature_stats

    train_paths, _val_paths, formal_test_names = core.active_source_session_paths()
    # A score artifact must state that those names remained strings.  They are
    # intentionally not resolved as paths by active_source_session_paths().
    _require(len(formal_test_names) == 6, "strict source test-name receipt drift")
    cache_dir = None  # Target scoring never writes or fits through a cache.
    behavior_mean, behavior_std = fit_behavior_stats(train_paths, core.BIN_SIZE_MS, cache_dir=cache_dir)
    side_group = str((metadata.get("side_features") or {}).get("group"))
    waveform_group = base_feature_group(side_group)
    _require(waveform_group == "t4", "A2 v2 source side normalizer must use the ordinary T4 substrate")
    side_mean, side_std = fit_side_feature_stats(
        train_paths,
        feature_group=waveform_group,
        pool_size=core.SIDE_FEATURE_POOL_TRIALS,
        cache_dir=cache_dir,
        bin_size_ms=core.BIN_SIZE_MS,
        window_size=core.WINDOW_SIZE_BINS,
        trial_result_filter="R",
        signal_view="sua",
    )
    training_cache_dir = Path(str(metadata.get("cache_dir", ""))).expanduser().resolve()
    _require(training_cache_dir == core.SOURCE_CACHE_ROOT.resolve(), "source run cache root drift")
    behavior_cache_path = _behavior_stats_cache_path(training_cache_dir, train_paths, core.BIN_SIZE_MS)
    side_cache_path = _side_stats_cache_path(
        training_cache_dir,
        train_paths,
        feature_group=waveform_group,
        pool_size=core.SIDE_FEATURE_POOL_TRIALS,
        bin_size_ms=core.BIN_SIZE_MS,
        window_size=core.WINDOW_SIZE_BINS,
        trial_result_filter="R",
        signal_view="sua",
    )
    cached_behavior = _load_training_cache_pair(behavior_cache_path, label="behavior")
    cached_side = _load_training_cache_pair(side_cache_path, label="T4")
    _require(_normalizer_pair_equal((behavior_mean, behavior_std), cached_behavior),
             "cached training-path behavior normalizer differs from uncached source recomputation")
    _require(_normalizer_pair_equal((side_mean, side_std), cached_side),
             "cached training-path T4 normalizer differs from uncached source recomputation")
    _require(np.isfinite(behavior_mean).all() and np.isfinite(behavior_std).all() and np.all(behavior_std > 0),
             "invalid source behavior normalizer")
    _require(np.isfinite(side_mean).all() and np.isfinite(side_std).all() and np.all(side_std > 0),
             "invalid source T4 normalizer")
    train_sessions = [session_name_from_path(path) for path in train_paths]
    _require(train_sessions == core.load_strict_manifest()["train"], "source normalizer roster drift")
    expected_side_hash = (metadata.get("side_features") or {}).get("normalization_sha256")
    actual_side_hash = core.normalizer_value_sha256(side_mean, side_std)
    _require(expected_side_hash == actual_side_hash, "source T4 normalizer no longer matches training metadata")
    behavior_digest = core.normalizer_value_sha256(behavior_mean, behavior_std)
    side_digest = core.normalizer_value_sha256(side_mean, side_std)
    authority = {
        "policy": core.frozen_normalizer_policy(),
        "source_train_sessions": train_sessions,
        "source_train_session_count": len(train_sessions),
        "behavior_normalizer_value_sha256": behavior_digest,
        "side_normalizer_value_sha256": side_digest,
        "source_training_side_normalizer_value_sha256": expected_side_hash,
        "cached_training_path_behavior_normalizer_value_sha256": core.normalizer_value_sha256(*cached_behavior),
        "cached_training_path_side_normalizer_value_sha256": core.normalizer_value_sha256(*cached_side),
        "cached_vs_uncached_behavior_values_bitwise_identical": True,
        "cached_vs_uncached_side_values_bitwise_identical": True,
        "training_cache_root": str(training_cache_dir),
        "training_path_behavior_cache": str(behavior_cache_path),
        "training_path_behavior_cache_sha256": core.sha256_file(behavior_cache_path),
        "training_path_side_cache": str(side_cache_path),
        "training_path_side_cache_sha256": core.sha256_file(side_cache_path),
        "target_domain_normalizer_refit_performed": False,
        "target_domain_normalizer_refit_forbidden": True,
        "formal_test_sessions_resolved_or_opened": False,
        "formal_test_session_names_only": list(formal_test_names),
        "normalizer_cache_dir": None,
    }
    return authority, (behavior_mean, behavior_std), (side_mean, side_std), tuple(formal_test_names), train_paths


def _load_domain_paths(domain: str) -> list[Path]:
    if domain == "within_subject":
        _train, val_paths, _test = core.active_source_session_paths()
        return val_paths
    if domain == "external_subject_M":
        return core.external_session_paths()
    raise A2V2ScoreError(f"unsupported domain: {domain}")


def _evaluate_epoch(
    *,
    checkpoint_path: Path,
    source_arm: str,
    metadata: Mapping[str, Any],
    domain_paths: list[Path],
    behavior_stats: tuple[Any, Any],
    side_stats: tuple[Any, Any],
    device: Any,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Evaluate one source epoch using an unmodified source checkpoint.

    The code deliberately uses the same evaluation-record, side-feature,
    model-loading, dataset, and R² helpers as the source-subject evaluator,
    with only the domain roster and pool size changed to the predeclared M30
    contract.  No optimizer, backward, or target-parameter update is present.
    """
    import torch
    from mc_maze.multisession_datamodule import session_name_from_path
    from scripts.eval_adaptation_dandi688 import (
        PAD_VALUE,
        TRIAL_LENGTH,
        WINDOW_SIZE,
        attach_side_features,
        build_calib_trials_for_indices,
        eval_r2,
        load_session_with_trials,
        make_subset_dataset,
    )
    from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model

    _require(WINDOW_SIZE == core.WINDOW_SIZE_BINS and TRIAL_LENGTH == core.TRIAL_LENGTH_BINS,
             "shared evaluator window/trial constants drift")
    behavior_mean, behavior_std = behavior_stats
    side_mean, side_std = side_stats
    side_group = str((metadata.get("side_features") or {}).get("group"))
    from mc_maze.unit_side_features import base_feature_group

    waveform_group = base_feature_group(side_group)
    _require(waveform_group == "t4", "A2 v2 source side normalizer must use the ordinary T4 substrate")
    model = load_frozen_model(
        checkpoint_path,
        core.TEACHER_PATH,
        "B3S",
        device,
        identity_mode="calibrated",
    )
    per_session: dict[str, float] = {}
    session_receipts: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for nwb_path in domain_paths:
            expected_name = session_name_from_path(nwb_path)
            record = load_session_with_trials(
                nwb_path,
                core.BIN_SIZE_MS,
                core.WINDOW_SIZE_BINS,
                core.ACTIVITY_CALIBRATION_TRIALS,
                core.TRIAL_LENGTH_BINS,
                PAD_VALUE,
                behavior_mean,
                behavior_std,
                trial_result_filter="R",
                cache_dir=None,
                signal_view="sua",
            )
            _require(record["name"] == expected_name, "domain session identity drift")
            semantics = core.trial30_semantics_from_trials(record["trials"], require_target_labels=True, session=expected_name)
            # The exact first-30 selection constructs a session-local carrier
            # for both T4 and Z4-before-mask.  It neither refits a source
            # normalizer nor adapts/selects the frozen decoder.
            indices = list(range(core.ACTIVITY_CALIBRATION_TRIALS))
            record["calib_trials"] = build_calib_trials_for_indices(record, indices, core.ACTIVITY_CALIBRATION_TRIALS)
            record = attach_side_features(
                record,
                nwb_path,
                side_feature_group=side_group,
                waveform_feature_group=waveform_group,
                pool_size=core.SIDE_FEATURE_POOL_TRIALS,
                permutation_seed=None,
                mean=side_mean,
                std=side_std,
                cache_dir=None,
            )
            query_trials = record["trials"][core.EVALUATION_START_TRIAL_INDEX:]
            dataset = make_subset_dataset(record, query_trials, expected_name)
            _require(len(dataset) == semantics["post30_query_window_count"], "M30/trial-30 query window count drift")
            _require(len(dataset) > 0, "empty M30/trial-30 query dataset")
            per_session[expected_name] = float(eval_r2(model, dataset, device))
            session_receipts[expected_name] = {
                **semantics,
                "dataset_query_window_count": len(dataset),
                "activity_calibration_trial_indices": indices,
                "side_feature_label_pool_trial_indices": list(range(core.SIDE_FEATURE_POOL_TRIALS)),
                "behavior_normalizer_authority": "strict_subc_source_train_27_only",
                "side_normalizer_authority": "strict_subc_source_train_27_only",
                "target_session_carrier_fit_performed": True,
                "target_direction_labels_used_for_carrier": True,
                "target_velocity_labels_used_for_weight_updates": False,
                "backward_gradients": False,
                "decoder_weight_updates": False,
            }
    return per_session, session_receipts


def execute_score(
    source_arm: str,
    seed: int,
    domain: str,
    run_dir: Path,
    out_path: Path,
    *,
    device_name: str,
    official_preflight: Path,
    result_root: Path,
    cell_launch_receipt: Path,
) -> dict[str, Any]:
    official_payload, official_preflight_sha = core.load_verified_official_preflight(
        official_preflight,
        result_root=result_root,
    )
    cell_launch, cell_launch_sha = core.load_verified_immutable_json(cell_launch_receipt, label="A2 v2 cell launch receipt")
    _require(cell_launch.get("receipt_kind") == core.CELL_LAUNCH_KIND, "cell launch receipt kind drift")
    _require(cell_launch.get("source_arm") == source_arm and cell_launch.get("seed") == seed,
             "cell launch receipt source-cell drift")
    _require(cell_launch.get("official_preflight_sha256") == official_preflight_sha,
             "cell launch receipt official-preflight digest drift")
    _require(cell_launch.get("implementation_bindings") == official_payload.get("implementation_bindings"),
             "cell launch receipt implementation binding drift")
    core.verify_implementation_bindings(cell_launch.get("implementation_bindings"))
    core.verify_python_isolation_binding(cell_launch.get("python_isolation"))
    current_runtime = core.torch_runtime_binding(visible_device_index=0)
    _require(cell_launch.get("torch_runtime") == current_runtime,
             "cell launch Torch/CUDA runtime drift before domain scoring")
    import torch
    _require(torch.cuda.is_available(), "A2 v2 score execution requires CUDA; this entrypoint refuses CPU fallback")
    device = torch.device(device_name)
    _require(device.type == "cuda", "A2 v2 score execution must use a CUDA device")
    fingerprint, metadata, arm = _source_bindings(source_arm, seed, run_dir)
    authority, behavior_stats, side_stats, formal_test_names, _source_train_paths = _fit_source_normalizers(metadata)
    domain_paths = _load_domain_paths(domain)
    expected_sessions = core.expected_domain_sessions(domain)
    from mc_maze.multisession_datamodule import session_name_from_path

    _require(tuple(session_name_from_path(path) for path in domain_paths) == expected_sessions,
             "domain path roster differs from frozen session ordering")
    checkpoints = core.source_epoch_checkpoint_paths(run_dir)
    per_epoch: dict[str, dict[str, Any]] = {}
    reference_session_receipts: dict[str, dict[str, Any]] | None = None
    for epoch in core.EPOCH_WINDOW:
        checkpoint_path = checkpoints[epoch]
        per_session_r2, session_receipts = _evaluate_epoch(
            checkpoint_path=checkpoint_path,
            source_arm=source_arm,
            metadata=metadata,
            domain_paths=domain_paths,
            behavior_stats=behavior_stats,
            side_stats=side_stats,
            device=device,
        )
        _require(tuple(per_session_r2) == expected_sessions, "per-session R² roster/order drift")
        if reference_session_receipts is None:
            reference_session_receipts = session_receipts
        else:
            _require(session_receipts == reference_session_receipts, "query semantics drifted across source epochs")
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": fingerprint["source_checkpoint_sha256_bundle"][str(epoch)],
            "per_session_r2": per_session_r2,
            "mean_r2": sum(per_session_r2.values()) / len(per_session_r2),
        }
    _require(reference_session_receipts is not None, "empty epoch window")
    mean_by_session = {
        session: sum(per_epoch[str(epoch)]["per_session_r2"][session] for epoch in core.EPOCH_WINDOW) / len(core.EPOCH_WINDOW)
        for session in expected_sessions
    }
    payload = {
        "schema_version": 3,
        "purpose": "a2_matched_subject_shift_v2_unchanged_source_checkpoint_domain_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "contract_path": str(core.CONTRACT_PATH),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "implementation_bindings": official_payload["implementation_bindings"],
        "implementation_bindings_sha256": official_payload["implementation_bindings_sha256"],
        "official_preflight_path": str(official_preflight.expanduser().resolve()),
        "official_preflight_sha256": official_preflight_sha,
        "cell_launch_receipt_path": str(cell_launch_receipt.expanduser().resolve()),
        "cell_launch_receipt_sha256": cell_launch_sha,
        "source_arm": source_arm,
        "arm": arm["arm"],
        "variant": arm["variant"],
        "seed": seed,
        "domain": domain,
        "domain_sessions": list(expected_sessions),
        "domain_session_count": len(expected_sessions),
        "source_run": fingerprint,
        "source_run_metadata_path": fingerprint["source_run_metadata_path"],
        "source_run_metadata_sha256": fingerprint["source_run_metadata_sha256"],
        "source_checkpoint_scored_unchanged_on_both_domains_required": True,
        "source_checkpoint_sha256_bundle": fingerprint["source_checkpoint_sha256_bundle"],
        "source_checkpoint_sha256_bundle_sha256": fingerprint["source_checkpoint_sha256_bundle_sha256"],
        "query_policy": core.frozen_query_policy(),
        "normalizer_authority": authority,
        "protocol": {
            "total_epochs": core.TOTAL_EPOCHS,
            "epoch_window": list(core.EPOCH_WINDOW),
            "epoch_score_rule": "unweighted mean session R2 over exactly source epochs 5..12",
            "activity_calibration_n": core.ACTIVITY_CALIBRATION_TRIALS,
            "pool_size": core.SIDE_FEATURE_POOL_TRIALS,
            "selection_mode": "first",
            "evaluation_start_trial_index": core.EVALUATION_START_TRIAL_INDEX,
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
            "signal_view": "sua",
        },
        "per_epoch": per_epoch,
        "per_session_mean_r2": mean_by_session,
        "mean_r2": sum(mean_by_session.values()) / len(mean_by_session),
        "session_query_receipts": reference_session_receipts,
        "no_test_files_evaluated": True,
        "formal_subc_test_nwb_opened": False,
        "formal_subc_test_session_names_only": list(formal_test_names),
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_velocity_labels_used_for_weight_updates": False,
        "backward_gradients": False,
        "decoder_weight_updates": False,
        "target_domain_normalizer_refit_performed": False,
        "execution_device": str(device),
    }
    _write_result_once(out_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-arm", choices=tuple(core.SOURCE_ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=core.SEEDS, required=True)
    parser.add_argument("--domain", choices=core.DOMAINS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-preflight", type=Path, default=None, help="One immutable official CPU preflight shared by all six cells.")
    parser.add_argument("--result-root", type=Path, default=core.RESULT_ROOT)
    parser.add_argument("--cell-launch-receipt", type=Path, default=None, help="Immutable per-cell launch environment receipt created before trainer start.")
    parser.add_argument("--launch", action="store_true", help="Execute the CUDA evaluator; otherwise print an inert dry-run receipt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("FAIL_CLOSED: PYTHONNOUSERSITE=1 is required", file=sys.stderr)
        return 2
    if not args.launch:
        try:
            print(json.dumps(build_dry_run(
                args.source_arm, args.seed, args.domain, args.run_dir, args.out_path,
                official_preflight=args.official_preflight,
                result_root=args.result_root.expanduser().resolve(),
            ), indent=2, sort_keys=True))
            return 0
        except (A2V2ScoreError, core.A2V2ContractError, FileNotFoundError, ValueError) as exc:
            print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
            return 2
    if os.environ.get(core.GPU_AUTH_ENV) != core.GPU_AUTH_VALUE:
        print(f"Refusing A2 v2 score launch: set {core.GPU_AUTH_ENV}={core.GPU_AUTH_VALUE}", file=sys.stderr)
        return 3
    if args.official_preflight is None or args.cell_launch_receipt is None:
        print("FAIL_CLOSED: --official-preflight and --cell-launch-receipt are required for A2 v2 scoring", file=sys.stderr)
        return 2
    try:
        payload = execute_score(
            args.source_arm, args.seed, args.domain, args.run_dir, args.out_path,
            device_name=args.device,
            official_preflight=args.official_preflight,
            result_root=args.result_root.expanduser().resolve(),
            cell_launch_receipt=args.cell_launch_receipt,
        )
    except (A2V2ScoreError, core.A2V2ContractError, FileNotFoundError, ValueError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "COMPLETE", "out_path": str(args.out_path), "mean_r2": payload["mean_r2"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
