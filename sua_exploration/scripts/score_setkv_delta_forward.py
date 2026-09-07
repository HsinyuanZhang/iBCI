#!/usr/bin/env python3
"""Score one frozen SetKV-delta intervention on one A2 development domain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for value in (REPO_ROOT, SUA_ROOT, SUA_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mc_maze import a2_matched_subject_shift_v2_core as a2  # noqa: E402
from mc_maze import setkv_delta_forward_core as core  # noqa: E402
from mc_maze.setkv_delta import decode_setkv_delta, setkv_delta_cost_receipt  # noqa: E402
from scripts import a2_matched_subject_shift_v2_score as a2_score  # noqa: E402


class SetKVForwardScoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SetKVForwardScoreError(message)


def _require_fresh_output(path: Path) -> None:
    """Refuse before loading source/target data if either receipt path exists."""
    sidecar = Path(str(path) + ".sha256")
    require(not path.exists(), f"score output already exists: {path}")
    require(not sidecar.exists(), f"score output sidecar already exists: {sidecar}")


def _require_canonical_output(name: str, domain: str, path: Path) -> None:
    expected = core.score_path(name, domain).resolve()
    require(path.resolve() == expected, f"score output must use canonical cell path: {expected}")


def _source_authority(name: str, domain: str) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any], Path]:
    spec = core.intervention(name)
    source_arm = str(spec["source_arm"])
    baseline, _digest = core.load_immutable(
        core.a2_baseline_receipt_path(source_arm, domain), "sealed A2 baseline domain receipt"
    )
    require(baseline.get("seed") == core.SEED and baseline.get("source_arm") == source_arm,
            "sealed A2 baseline source cell drift")
    require(baseline.get("domain") == domain, "sealed A2 baseline domain drift")
    require(baseline.get("formal_subc_test_nwb_opened") is False, "sealed baseline opened formal data")
    require(baseline.get("official_preflight_sha256") == core.EXPECTED_A2_PREFLIGHT_SHA256,
            "sealed A2 baseline official-preflight binding drift")
    require(baseline.get("query_policy") == a2.frozen_query_policy(), "sealed A2 baseline query-policy drift")
    require(isinstance(baseline.get("normalizer_authority"), Mapping),
            "sealed A2 baseline normalizer authority absent")
    run_dir = Path(str(baseline["source_run"]["source_run_dir"])).resolve()
    fingerprint, metadata, arm = a2_score._source_bindings(source_arm, core.SEED, run_dir)
    require(fingerprint == baseline["source_run"], "live A2 source bundle differs from sealed domain receipt")
    return spec, baseline, metadata, run_dir


def _evaluate_dataset(model: Any, dataset: Any, device: Any, *, name: str, session: str) -> tuple[float, float, int]:
    import torch
    from torch.utils.data import DataLoader
    from torchmetrics.regression import R2Score
    from scripts.eval_adaptation_dandi688 import _unpack_loader_batch, decode_last_behavior

    spec = core.intervention(name)
    model.eval()
    metric = R2Score(multioutput="variance_weighted").to(device)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    total = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            neural, behavior, calib, side_features, _electrode_ids = _unpack_loader_batch(batch)
            neural = neural.to(device)
            behavior = behavior.to(device)
            calib = calib.to(device)
            require(side_features is not None, "SetKV requires side features")
            side_features = side_features.to(device)
            permutation_seed = (
                core.session_permutation_seed(session)
                if spec["carrier_mode"] == "row_shuffle" else None
            )
            output = decode_setkv_delta(
                model.student,
                neural,
                calib,
                side_features,
                decode_mode=str(spec["decode_mode"]),
                carrier_mode="aligned" if spec["carrier_mode"] is None else str(spec["carrier_mode"]),
                permutation_seed=permutation_seed,
            )
            prediction = decode_last_behavior(output.prediction)
            target = behavior[:, -1:, :]
            metric.update(
                prediction.flatten(start_dim=0, end_dim=1),
                target.flatten(start_dim=0, end_dim=1),
            )
            total += int(neural.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return float(metric.compute().item()), float(elapsed), total


def _evaluate_epoch(
    *,
    checkpoint_path: Path,
    name: str,
    metadata: Mapping[str, Any],
    domain_paths: list[Path],
    behavior_stats: tuple[Any, Any],
    side_stats: tuple[Any, Any],
    device: Any,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, Any]],
    dict[str, float],
    int,
    dict[str, int],
    int,
]:
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.unit_side_features import base_feature_group
    from scripts.eval_adaptation_dandi688 import (
        PAD_VALUE,
        attach_side_features,
        build_calib_trials_for_indices,
        load_session_with_trials,
        make_subset_dataset,
    )
    from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model

    spec = core.intervention(name)
    source_arm = str(spec["source_arm"])
    side_group = str((metadata.get("side_features") or {}).get("group"))
    expected_side_group = str(a2.SOURCE_ARMS[source_arm]["side_feature_group"])
    require(side_group == expected_side_group, "source side group drift")
    waveform_group = base_feature_group(side_group)
    require(waveform_group == "t4", "SetKV source substrate must be ordinary T4")
    model = load_frozen_model(
        checkpoint_path, a2.TEACHER_PATH, "B3S", device, identity_mode="calibrated"
    )
    decoder = model.student.decoder
    require((decoder.model_dim, decoder.num_covariates, decoder.num_layers) == (512, 2, 1),
            "SetKV analytic cost topology drift")
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())
    behavior_mean, behavior_std = behavior_stats
    side_mean, side_std = side_stats
    per_session: dict[str, float] = {}
    receipts: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    unit_counts: dict[str, int] = {}
    samples = 0
    for nwb_path in domain_paths:
        expected_name = session_name_from_path(nwb_path)
        record = load_session_with_trials(
            nwb_path,
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
        require(record["name"] == expected_name, "target session identity drift")
        semantics = a2.trial30_semantics_from_trials(
            record["trials"], require_target_labels=True, session=expected_name
        )
        indices = list(range(a2.ACTIVITY_CALIBRATION_TRIALS))
        record["calib_trials"] = build_calib_trials_for_indices(
            record, indices, a2.ACTIVITY_CALIBRATION_TRIALS
        )
        record = attach_side_features(
            record,
            nwb_path,
            side_feature_group=side_group,
            waveform_feature_group=waveform_group,
            pool_size=a2.SIDE_FEATURE_POOL_TRIALS,
            permutation_seed=None,
            mean=side_mean,
            std=side_std,
            cache_dir=None,
        )
        query_trials = record["trials"][a2.EVALUATION_START_TRIAL_INDEX :]
        dataset = make_subset_dataset(record, query_trials, expected_name)
        require(len(dataset) == semantics["post30_query_window_count"], "query-window count drift")
        score, elapsed, count = _evaluate_dataset(
            model, dataset, device, name=name, session=expected_name
        )
        per_session[expected_name] = score
        timings[expected_name] = elapsed
        unit_counts[expected_name] = int(record["neural"].shape[1])
        samples += count
        receipts[expected_name] = {
            **semantics,
            "dataset_query_window_count": len(dataset),
            "activity_calibration_trial_indices": indices,
            "side_feature_label_pool_trial_indices": list(range(a2.SIDE_FEATURE_POOL_TRIALS)),
            "behavior_normalizer_authority": "strict_subc_source_train_27_only",
            "side_normalizer_authority": "strict_subc_source_train_27_only",
            "target_session_carrier_fit_performed": True,
            "target_direction_labels_used_for_carrier": True,
            "target_velocity_labels_used_for_weight_updates": False,
            "backward_gradients": False,
            "decoder_weight_updates": False,
        }
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
    require(parameter_count_after == parameter_count_before, "SetKV registered or removed parameters")
    return per_session, receipts, timings, samples, unit_counts, parameter_count_before


def execute(name: str, domain: str, output: Path, *, device_name: str, preflight_path: Path) -> dict[str, Any]:
    _require_canonical_output(name, domain, output)
    _require_fresh_output(output)
    preflight, preflight_sha = core.load_official_preflight(preflight_path)
    spec, baseline, metadata, run_dir = _source_authority(name, domain)
    authority, behavior_stats, side_stats, formal_names, _ = a2_score._fit_source_normalizers(metadata)
    # This is deliberately before resolving the scored-domain paths: all source
    # normalizer bytes and their source-only derivation must remain exactly the
    # ones recorded by the matched sealed A2 parent.
    require(authority == baseline["normalizer_authority"],
            "live source normalizer authority differs from sealed A2 baseline")
    domain_paths = a2_score._load_domain_paths(domain)
    expected_sessions = a2.expected_domain_sessions(domain)
    from mc_maze.multisession_datamodule import session_name_from_path
    require(tuple(session_name_from_path(path) for path in domain_paths) == expected_sessions,
            "domain roster drift before model loading")

    import torch
    require(torch.cuda.is_available(), "SetKV execution requires CUDA")
    device = torch.device(device_name)
    require(device.type == "cuda", "SetKV refuses CPU execution fallback")
    checkpoints = a2.source_epoch_checkpoint_paths(run_dir)
    source_bundle = baseline["source_checkpoint_sha256_bundle"]
    per_epoch: dict[str, Any] = {}
    reference_receipts = None
    reference_unit_counts = None
    reference_model_parameter_count = None
    reference_cost_topology = None
    total_seconds = 0.0
    total_samples = 0
    for epoch in a2.EPOCH_WINDOW:
        checkpoint = checkpoints[epoch]
        require(a2.sha256_file(checkpoint) == source_bundle[str(epoch)], "A2 checkpoint byte drift")
        scores, receipts, timings, samples, unit_counts, model_parameter_count = _evaluate_epoch(
            checkpoint_path=checkpoint,
            name=name,
            metadata=metadata,
            domain_paths=domain_paths,
            behavior_stats=behavior_stats,
            side_stats=side_stats,
            device=device,
        )
        require(tuple(scores) == expected_sessions, "score session order drift")
        # A score body is never minted unless its query/support chronology is
        # already an exact match to the sealed A2 parent, rather than relying
        # on the later receipt-only aggregate to discover a mismatch.
        require(receipts == baseline["session_query_receipts"],
                "SetKV session query receipts differ from sealed A2 baseline")
        if reference_receipts is None:
            reference_receipts = receipts
            reference_unit_counts = unit_counts
            reference_model_parameter_count = model_parameter_count
            reference_cost_topology = {"model_dim": 512, "num_queries": 2, "num_layers": 1}
        else:
            require(receipts == reference_receipts, "query semantics drift across epochs")
            require(unit_counts == reference_unit_counts, "target unit counts drift across epochs")
            require(model_parameter_count == reference_model_parameter_count,
                    "model parameter count drift across epochs")
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": source_bundle[str(epoch)],
            "per_session_r2": scores,
            "mean_r2": sum(scores.values()) / len(scores),
            "forward_wall_seconds_by_session": timings,
        }
        total_seconds += sum(timings.values())
        total_samples += samples
    require(reference_receipts is not None, "empty epoch score window")
    require(reference_unit_counts is not None and reference_model_parameter_count is not None,
            "missing engineering-cost authority")
    require(reference_cost_topology is not None, "missing analytic-cost topology")
    per_session_mean = {
        session: sum(per_epoch[str(epoch)]["per_session_r2"][session] for epoch in a2.EPOCH_WINDOW)
        / len(a2.EPOCH_WINDOW)
        for session in expected_sessions
    }
    cost_by_session = {
        session: setkv_delta_cost_receipt(num_units=units, num_queries=2, model_dim=512)
        for session, units in reference_unit_counts.items()
    }
    payload = {
        "schema_version": 1,
        "receipt_kind": "setkv_delta_forward_domain_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "intervention": name,
        "intervention_spec": dict(spec),
        "seed": core.SEED,
        "domain": domain,
        "official_preflight_path": str(preflight_path.resolve()),
        "official_preflight_sha256": preflight_sha,
        "implementation_bindings": preflight["implementation_bindings"],
        "a2_baseline_receipt_path": str(core.a2_baseline_receipt_path(str(spec["source_arm"]), domain).resolve()),
        "a2_baseline_receipt_sha256": core.sha256_file(core.a2_baseline_receipt_path(str(spec["source_arm"]), domain)),
        "source_run": baseline["source_run"],
        "source_checkpoint_sha256_bundle": source_bundle,
        "source_checkpoint_sha256_bundle_sha256": baseline["source_checkpoint_sha256_bundle_sha256"],
        "normalizer_authority": authority,
        "source_decoder_architecture_and_cost": metadata.get("decoder_architecture"),
        "source_encoder_cost_profile_reference": metadata.get("encoder_cost_profile_reference"),
        "query_policy": a2.frozen_query_policy(),
        "domain_sessions": list(expected_sessions),
        "per_epoch": per_epoch,
        "per_session_mean_r2": per_session_mean,
        "mean_r2": sum(per_session_mean.values()) / len(per_session_mean),
        "session_query_receipts": reference_receipts,
        "row_permutation_seed_by_session": (
            {session: core.session_permutation_seed(session) for session in expected_sessions}
            if spec["carrier_mode"] == "row_shuffle" else {}
        ),
        "model_parameter_count": reference_model_parameter_count,
        "engineering_cost_topology": reference_cost_topology,
        "unit_count_by_session": reference_unit_counts,
        "engineering_cost_by_session": cost_by_session,
        "measured_forward_wall_seconds": total_seconds,
        "measured_forward_sample_count_across_epochs": total_samples,
        "measured_forward_seconds_per_sample": total_seconds / total_samples,
        "parameter_delta": 0,
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "backward_gradients": False,
        "decoder_weight_updates": False,
        "target_normalizer_refit": False,
        "target_domain_normalizer_refit_performed": False,
        "formal_subc_test_nwb_opened": False,
        "formal_subc_test_session_names_only": list(formal_names),
        "no_test_files_evaluated": True,
        "checkpoint_selection_from_target_scores": False,
        "source_checkpoint_scored_unchanged": True,
        "execution_device": str(device),
    }
    core.write_immutable(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention", choices=tuple(core.INTERVENTIONS), required=True)
    parser.add_argument("--domain", choices=core.DOMAINS, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-preflight", type=Path, default=core.OFFICIAL_PREFLIGHT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output or core.score_path(args.intervention, args.domain)
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA_NO_MODEL_NO_GPU",
            "intervention": args.intervention,
            "domain": args.domain,
            "output": str(output),
            "official_preflight": str(args.official_preflight),
        }, indent=2, sort_keys=True))
        return 0
    payload = execute(
        args.intervention,
        args.domain,
        output,
        device_name=args.device,
        preflight_path=args.official_preflight,
    )
    print(json.dumps({"status": "COMPLETE", "mean_r2": payload["mean_r2"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
