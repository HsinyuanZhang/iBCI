#!/usr/bin/env python3
"""Source-only value-weighted attention gate for the carrier mask route."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for value in (REPO_ROOT, SUA_ROOT, SUA_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mc_maze import a2_matched_subject_shift_v2_core as a2  # noqa: E402
from mc_maze import carrier_value_mask_core as core  # noqa: E402
from mc_maze.carrier_value_mask import (  # noqa: E402
    first_layer_value_weighted_importance,
    masked_share,
    unit_mask,
)
from scripts import a2_matched_subject_shift_v2_score as a2_score  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise core.CarrierValueMaskContractError(message)


def _fresh(path: Path) -> None:
    require(not os.path.lexists(path), f"source-gate output exists: {path}")
    require(not os.path.lexists(Path(str(path) + ".sha256")),
            f"source-gate sidecar exists: {path}.sha256")


def _require_canonical_output(path: Path) -> None:
    require(path.resolve() == core.SOURCE_GATE.resolve(),
            f"source-gate output must use canonical path: {core.SOURCE_GATE.resolve()}")


def _source_authority() -> tuple[dict[str, Any], dict[str, Any], Path]:
    official, official_sha = a2.load_verified_immutable_json(core.A2_PREFLIGHT, label="sealed A2 preflight")
    require(official_sha == core.EXPECTED_A2_PREFLIGHT_SHA256, "sealed A2 preflight SHA drift")
    a2.verify_implementation_bindings(official.get("implementation_bindings"))
    baseline, _ = a2.load_verified_immutable_json(core.A2_T4_WITHIN, label="sealed A2 T4 seed42 receipt")
    require(baseline.get("source_arm") == "source_t4" and baseline.get("seed") == core.SEED
            and baseline.get("domain") == "within_subject",
            "A2 T4 parent drift")
    require(baseline.get("official_preflight_sha256") == official_sha,
            "A2 T4 parent/preflight linkage drift")
    require(baseline.get("query_policy") == a2.frozen_query_policy(), "A2 T4 parent query-policy drift")
    require(isinstance(baseline.get("normalizer_authority"), dict), "A2 T4 parent normalizer authority absent")
    run_dir = Path(str(baseline["source_run"]["source_run_dir"])).resolve()
    fingerprint, metadata, _arm = a2_score._source_bindings("source_t4", core.SEED, run_dir)
    require(fingerprint == baseline["source_run"], "live A2 T4 source bundle drift")
    checkpoint = a2.source_epoch_checkpoint_paths(run_dir)[core.SOURCE_EPOCH]
    require(a2.sha256_file(checkpoint) == baseline["source_checkpoint_sha256_bundle"][str(core.SOURCE_EPOCH)],
            "A2 T4 source checkpoint byte drift")
    return baseline, metadata, checkpoint


def execute(output: Path) -> dict[str, Any]:
    _require_canonical_output(output)
    _fresh(output)
    import torch
    from torch.utils.data import DataLoader, Subset
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import base_feature_group
    from scripts.eval_adaptation_dandi688 import (
        PAD_VALUE,
        _unpack_loader_batch,
        attach_side_features,
        build_calib_trials_for_indices,
        load_session_with_trials,
        make_subset_dataset,
    )
    from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model

    require(not torch.cuda.is_available(), "source mechanism probe is CPU-only; clear CUDA visibility")
    baseline, metadata, checkpoint = _source_authority()
    authority, behavior_stats, side_stats, formal_names, source_paths = a2_score._fit_source_normalizers(metadata)
    require(authority == baseline["normalizer_authority"],
            "source normalizer authority differs from sealed A2 T4 parent")
    require(len(source_paths) == 27, "source mechanism probe requires all 27 source sessions")
    formal = set(formal_names)
    require(not formal.intersection(path.name for path in source_paths), "formal/source path intersection")
    behavior_mean, behavior_std = behavior_stats
    side_mean, side_std = side_stats
    side_group = str(metadata["side_features"]["group"])
    require(side_group == "t4" and base_feature_group(side_group) == "t4", "probe requires T4 substrate")
    model = load_frozen_model(checkpoint, a2.TEACHER_PATH, "B3S", torch.device("cpu"), identity_mode="calibrated")
    model.eval()
    before = tuple((name, parameter.detach().cpu().numpy().tobytes()) for name, parameter in model.named_parameters())
    sessions: dict[str, Any] = {}
    with torch.no_grad():
        for path in source_paths:
            name = session_name_from_path(path)
            require(name not in formal, f"formal session reached source probe: {name}")
            record = load_session_with_trials(
                path,
                a2.BIN_SIZE_MS,
                a2.WINDOW_SIZE_BINS,
                a2.ACTIVITY_CALIBRATION_TRIALS,
                a2.TRIAL_LENGTH_BINS,
                PAD_VALUE,
                behavior_mean,
                behavior_std,
                trial_result_filter="R",
                cache_dir=None,
                signal_view="sua",
            )
            production_trials = list_datamodule_rewarded_trials(
                path,
                bin_size_ms=a2.BIN_SIZE_MS,
                window_size=a2.WINDOW_SIZE_BINS,
                trial_result_filter="R",
            )
            semantics = core.source_trial_semantics(
                record["trials"], production_trials, session=name
            )
            indices = list(range(a2.ACTIVITY_CALIBRATION_TRIALS))
            record["calib_trials"] = build_calib_trials_for_indices(
                record, indices, a2.ACTIVITY_CALIBRATION_TRIALS
            )
            record = attach_side_features(
                record,
                path,
                side_feature_group=side_group,
                waveform_feature_group="t4",
                pool_size=a2.SIDE_FEATURE_POOL_TRIALS,
                permutation_seed=None,
                mean=side_mean,
                std=side_std,
                cache_dir=None,
            )
            dataset = make_subset_dataset(record, record["trials"][a2.EVALUATION_START_TRIAL_INDEX :], name)
            count = min(core.WINDOW_CAP_PER_SESSION, len(dataset))
            require(count == core.WINDOW_CAP_PER_SESSION, f"source window cap unavailable: {name}")
            loader = DataLoader(Subset(dataset, range(count)), batch_size=32, shuffle=False, num_workers=0)
            low_attention = total_attention = low_value = total_value = 0.0
            for batch in loader:
                neural, _behavior, calib, side, _electrodes = _unpack_loader_batch(batch)
                require(side is not None, "source probe side features missing")
                low = unit_mask(side, fraction=core.MASK_FRACTION, mode="low_gain")
                importance = first_layer_value_weighted_importance(model.student, neural, calib, side)
                attention_share = masked_share(importance.attention_mass, low)
                value_share = masked_share(importance.value_weighted_mass, low)
                low_attention += float(attention_share.sum().item())
                total_attention += int(attention_share.numel())
                low_value += float(value_share.sum().item())
                total_value += int(value_share.numel())
            require(total_attention == count and total_value == count, "source probe sample accounting drift")
            sessions[name] = {
                **semantics,
                "query_window_cap": count,
                "available_post30_query_windows": len(dataset),
                "low_gain_fraction": core.MASK_FRACTION,
                "mean_low_gain_attention_mass_share": low_attention / total_attention,
                "mean_low_gain_value_weighted_mass_share": low_value / total_value,
            }
            core.validate_source_session_receipt(sessions[name], session=name)
    after = tuple((name, parameter.detach().cpu().numpy().tobytes()) for name, parameter in model.named_parameters())
    require(before == after, "model parameters changed during source probe")
    values = [row["mean_low_gain_value_weighted_mass_share"] for row in sessions.values()]
    require(len(values) == 27 and all(math.isfinite(value) for value in values), "invalid source gate values")
    median = statistics.median(values)
    gate = median >= core.MIN_MEDIAN_VALUE_WEIGHTED_SHARE
    bindings = core.current_bindings()
    payload = {
        "schema_version": 1,
        "receipt_kind": "carrier_value_mask_source_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "status": "SOURCE_GATE_PASS__TARGET_FORWARD_ALLOWED" if gate else "SOURCE_GATE_STOP__NO_TARGET_SCORE",
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.canonical_sha256(bindings),
        "a2_parent_receipt_path": str(core.A2_T4_WITHIN.resolve()),
        "a2_parent_receipt_sha256": core.sha256_file(core.A2_T4_WITHIN),
        "a2_official_preflight_sha256": core.EXPECTED_A2_PREFLIGHT_SHA256,
        "source_checkpoint_path": str(checkpoint.resolve()),
        "source_checkpoint_sha256": core.sha256_file(checkpoint),
        "source_epoch": core.SOURCE_EPOCH,
        "source_session_count": len(sessions),
        "window_cap_per_session": core.WINDOW_CAP_PER_SESSION,
        "mask_fraction": core.MASK_FRACTION,
        "minimum_median_value_weighted_share": core.MIN_MEDIAN_VALUE_WEIGHTED_SHARE,
        "median_low_gain_value_weighted_mass_share": median,
        "mean_low_gain_value_weighted_mass_share": statistics.fmean(values),
        "gate_passed": gate,
        "sessions": sessions,
        "normalizer_authority": authority,
        "source_query_policy": core.frozen_source_query_policy(),
        "model_state_unchanged": True,
        "source_nwb_opened": True,
        "target_development_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "cuda_used": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
    }
    core.write_immutable(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=core.SOURCE_GATE)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        bindings = core.current_bindings()
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA_NO_MODEL_NO_GPU",
            "output": str(args.output),
            "source_sessions": 27,
            "source_epoch": core.SOURCE_EPOCH,
            "window_cap_per_session": core.WINDOW_CAP_PER_SESSION,
            "implementation_bindings_sha256": core.canonical_sha256(bindings),
        }, indent=2, sort_keys=True))
        return 0
    payload = execute(args.output)
    print(json.dumps({
        "status": payload["status"],
        "median_low_gain_value_weighted_mass_share": payload["median_low_gain_value_weighted_mass_share"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
