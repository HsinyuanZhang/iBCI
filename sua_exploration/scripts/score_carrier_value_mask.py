#!/usr/bin/env python3
"""Score one frozen carrier value-mask intervention on one A2 domain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
from mc_maze import carrier_value_mask_core as core  # noqa: E402
from mc_maze.carrier_value_mask import decode_with_unit_mask, unit_mask  # noqa: E402
from scripts import a2_matched_subject_shift_v2_score as a2_score  # noqa: E402


class CarrierValueMaskScoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierValueMaskScoreError(message)


def _fresh(path: Path) -> None:
    require(not os.path.lexists(path), f"mask score output exists: {path}")
    require(not os.path.lexists(Path(str(path) + ".sha256")),
            f"mask score sidecar exists: {path}.sha256")


def _require_canonical_output(name: str, domain: str, path: Path) -> None:
    expected = core.score_path(name, domain).resolve()
    require(path.resolve() == expected, f"mask score output must use canonical path: {expected}")


def _source_authority(name: str, domain: str):
    spec = core.intervention(name)
    source_arm = str(spec["source_arm"])
    official, official_sha = a2.load_verified_immutable_json(core.A2_PREFLIGHT, label="sealed A2 preflight")
    require(official_sha == core.EXPECTED_A2_PREFLIGHT_SHA256, "sealed A2 preflight SHA drift")
    a2.verify_implementation_bindings(official.get("implementation_bindings"))
    baseline, baseline_sha = a2.load_verified_immutable_json(
        core.a2_receipt_path(source_arm, domain), label="sealed A2 mask parent"
    )
    require(baseline.get("source_arm") == source_arm and baseline.get("domain") == domain,
            "A2 mask parent cell drift")
    require(baseline.get("seed") == core.SEED, "A2 mask parent seed drift")
    require(baseline.get("formal_subc_test_nwb_opened") is False, "formal parent data opened")
    require(baseline.get("official_preflight_sha256") == official_sha,
            "A2 mask parent/preflight linkage drift")
    require(baseline.get("query_policy") == a2.frozen_query_policy(), "A2 mask parent query-policy drift")
    require(isinstance(baseline.get("normalizer_authority"), Mapping),
            "A2 mask parent normalizer authority absent")
    run_dir = Path(str(baseline["source_run"]["source_run_dir"])).resolve()
    fingerprint, metadata, _arm = a2_score._source_bindings(source_arm, core.SEED, run_dir)
    require(fingerprint == baseline["source_run"], "live A2 source bundle drift")
    return spec, baseline, baseline_sha, metadata, run_dir


def _evaluate_dataset(model: Any, dataset: Any, mask_features: Any, device: Any, *, mode: str, seed: int):
    import torch
    from torch.utils.data import DataLoader
    from torchmetrics.regression import R2Score
    from scripts.eval_adaptation_dandi688 import _unpack_loader_batch, decode_last_behavior

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    metric = R2Score(multioutput="variance_weighted").to(device)
    total = 0
    model.eval()
    static_features = torch.as_tensor(mask_features, device=device).unsqueeze(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_started = time.perf_counter()
    session_mask = unit_mask(
        static_features,
        fraction=core.MASK_FRACTION,
        mode=mode,
        seed=seed if mode == "random" else None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    require(session_mask.shape[0] == 1 and session_mask.dtype == torch.bool,
            "persistent session mask shape/type drift")
    masked_count = int(session_mask[0].sum().item())
    remaining_count = int(session_mask.shape[1] - masked_count)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            neural, behavior, calib, visible_side, _electrodes = _unpack_loader_batch(batch)
            require(visible_side is not None, "mask scorer visible side features missing")
            neural = neural.to(device)
            behavior = behavior.to(device)
            calib = calib.to(device)
            visible_side = visible_side.to(device)
            require(visible_side.shape[1] == session_mask.shape[1], "persistent mask/unit shape drift")
            mask = session_mask.expand(neural.shape[0], -1)
            current_masked = int(masked_count)
            require((mask.sum(dim=1) == current_masked).all().item(), "mask count differs within session batch")
            require(masked_count == current_masked, "mask count drift across batches")
            prediction = decode_last_behavior(
                decode_with_unit_mask(model.student, neural, calib, visible_side, mask)
            )
            target = behavior[:, -1:, :]
            metric.update(prediction.flatten(0, 1), target.flatten(0, 1))
            total += int(neural.shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    require(total > 0, "empty mask dataset")
    return (
        float(metric.compute().item()), elapsed, total, masked_count, remaining_count,
        int(session_mask.numel() * session_mask.element_size()), setup_seconds,
    )


def _evaluate_epoch(
    *,
    checkpoint: Path,
    name: str,
    metadata: Mapping[str, Any],
    paths: list[Path],
    behavior_stats: tuple[Any, Any],
    side_stats: tuple[Any, Any],
    device: Any,
):
    import numpy as np
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.unit_side_features import load_unit_side_features
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
    visible_group = str(metadata["side_features"]["group"])
    require(visible_group == str(a2.SOURCE_ARMS[source_arm]["side_feature_group"]),
            "mask scorer visible side-group drift")
    model = load_frozen_model(checkpoint, a2.TEACHER_PATH, "B3S", device, identity_mode="calibrated")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    behavior_mean, behavior_std = behavior_stats
    side_mean, side_std = side_stats
    per_session: dict[str, float] = {}
    query_receipts: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    for path in paths:
        session = session_name_from_path(path)
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
        semantics = a2.trial30_semantics_from_trials(record["trials"], require_target_labels=True, session=session)
        indices = list(range(a2.ACTIVITY_CALIBRATION_TRIALS))
        record["calib_trials"] = build_calib_trials_for_indices(record, indices, a2.ACTIVITY_CALIBRATION_TRIALS)
        match_features, _metadata = load_unit_side_features(
            path,
            feature_group="t4",
            pool_size=a2.SIDE_FEATURE_POOL_TRIALS,
            mean=side_mean,
            std=side_std,
            cache_dir=None,
            bin_size_ms=a2.BIN_SIZE_MS,
            window_size=a2.WINDOW_SIZE_BINS,
            trial_result_filter="R",
            signal_view="sua",
        )
        record = attach_side_features(
            record,
            path,
            side_feature_group=visible_group,
            waveform_feature_group="t4",
            pool_size=a2.SIDE_FEATURE_POOL_TRIALS,
            permutation_seed=None,
            mean=side_mean,
            std=side_std,
            cache_dir=None,
        )
        visible = np.asarray(record["side_features"])
        require(match_features.shape == visible.shape, "mask/visible side shape drift")
        if source_arm == "source_t4":
            require(np.array_equal(match_features, visible), "T4 mask features differ from model-visible T4")
        else:
            require(np.count_nonzero(visible) == 0, "Z4 model-visible carrier is not exact zero")
        dataset = make_subset_dataset(record, record["trials"][a2.EVALUATION_START_TRIAL_INDEX :], session)
        require(len(dataset) == semantics["post30_query_window_count"], "mask query-window count drift")
        score, seconds, samples, masked, remaining, mask_state_bytes, mask_setup_seconds = _evaluate_dataset(
            model,
            dataset,
            match_features,
            device,
            mode=str(spec["mask_mode"]),
            seed=core.session_seed(session),
        )
        per_session[session] = score
        runtime[session] = {
            "forward_wall_seconds": seconds,
            "samples": samples,
            "unit_count": masked + remaining,
            "masked_unit_count": masked,
            "remaining_unit_count": remaining,
            "persistent_boolean_mask_state_bytes": mask_state_bytes,
            "persistent_boolean_mask_element_bytes": mask_state_bytes // (masked + remaining),
            "mask_setup_wall_seconds": mask_setup_seconds,
        }
        query_receipts[session] = {
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
    return per_session, query_receipts, runtime, parameter_count


def execute(name: str, domain: str, output: Path, *, device_name: str) -> dict[str, Any]:
    _require_canonical_output(name, domain, output)
    _fresh(output)
    gate, gate_sha = core.load_source_gate()
    require(gate.get("gate_passed") is True and gate.get("status") == "SOURCE_GATE_PASS__TARGET_FORWARD_ALLOWED",
            "source value-weighted gate did not permit target scoring")
    spec, baseline, baseline_sha, metadata, run_dir = _source_authority(name, domain)
    authority, behavior_stats, side_stats, formal_names, _source_paths = a2_score._fit_source_normalizers(metadata)
    # This check is deliberately before resolving any target-domain paths.
    require(authority == baseline["normalizer_authority"],
            "live source normalizer authority differs from sealed A2 parent")
    paths = a2_score._load_domain_paths(domain)
    expected_sessions = a2.expected_domain_sessions(domain)
    from mc_maze.multisession_datamodule import session_name_from_path
    require(tuple(session_name_from_path(path) for path in paths) == expected_sessions, "mask domain roster drift")
    import torch
    require(torch.cuda.is_available(), "mask target score requires CUDA")
    device = torch.device(device_name)
    require(device.type == "cuda", "mask target score refuses CPU fallback")
    checkpoints = a2.source_epoch_checkpoint_paths(run_dir)
    source_bundle = baseline["source_checkpoint_sha256_bundle"]
    per_epoch: dict[str, Any] = {}
    reference_query = reference_runtime_shape = None
    parameter_count = None
    total_seconds = 0.0
    total_samples = 0
    total_mask_setup_seconds = 0.0
    for epoch in a2.EPOCH_WINDOW:
        checkpoint = checkpoints[epoch]
        require(a2.sha256_file(checkpoint) == source_bundle[str(epoch)], "mask source checkpoint drift")
        scores, query, runtime, current_parameter_count = _evaluate_epoch(
            checkpoint=checkpoint,
            name=name,
            metadata=metadata,
            paths=paths,
            behavior_stats=behavior_stats,
            side_stats=side_stats,
            device=device,
        )
        require(tuple(scores) == expected_sessions, "mask score session order drift")
        runtime_shape = {
            session: {
                key: row[key] for key in (
                    "samples", "unit_count", "masked_unit_count", "remaining_unit_count",
                    "persistent_boolean_mask_state_bytes", "persistent_boolean_mask_element_bytes",
                )
            }
            for session, row in runtime.items()
        }
        require(query == baseline["session_query_receipts"],
                "mask session query receipts differ from sealed A2 parent")
        if reference_query is None:
            reference_query = query
            reference_runtime_shape = runtime_shape
            parameter_count = current_parameter_count
        else:
            require(query == reference_query, "mask query semantics drift across epochs")
            require(runtime_shape == reference_runtime_shape, "mask shape/count drift across epochs")
            require(current_parameter_count == parameter_count, "mask model parameter count drift")
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": source_bundle[str(epoch)],
            "per_session_r2": scores,
            "mean_r2": sum(scores.values()) / len(scores),
            "forward_wall_seconds_by_session": {session: row["forward_wall_seconds"] for session, row in runtime.items()},
        }
        total_seconds += sum(row["forward_wall_seconds"] for row in runtime.values())
        total_samples += sum(row["samples"] for row in runtime.values())
        total_mask_setup_seconds += sum(row["mask_setup_wall_seconds"] for row in runtime.values())
    require(reference_query is not None and reference_runtime_shape is not None and parameter_count is not None,
            "empty mask epoch window")
    mean_by_session = {
        session: sum(per_epoch[str(epoch)]["per_session_r2"][session] for epoch in a2.EPOCH_WINDOW)
        / len(a2.EPOCH_WINDOW)
        for session in expected_sessions
    }
    bindings = core.current_bindings()
    payload = {
        "schema_version": 1,
        "receipt_kind": "carrier_value_mask_domain_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "intervention": name,
        "intervention_spec": dict(spec),
        "seed": core.SEED,
        "domain": domain,
        "source_gate_path": str(core.SOURCE_GATE.resolve()),
        "source_gate_sha256": gate_sha,
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.canonical_sha256(bindings),
        "a2_baseline_receipt_path": str(core.a2_receipt_path(str(spec["source_arm"]), domain).resolve()),
        "a2_baseline_receipt_sha256": baseline_sha,
        "a2_official_preflight_sha256": core.EXPECTED_A2_PREFLIGHT_SHA256,
        "source_run": baseline["source_run"],
        "source_checkpoint_sha256_bundle": source_bundle,
        "source_checkpoint_sha256_bundle_sha256": baseline["source_checkpoint_sha256_bundle_sha256"],
        "normalizer_authority": authority,
        "source_decoder_architecture_and_cost": metadata.get("decoder_architecture"),
        "source_encoder_cost_profile_reference": metadata.get("encoder_cost_profile_reference"),
        "query_policy": a2.frozen_query_policy(),
        "domain_sessions": list(expected_sessions),
        "per_epoch": per_epoch,
        "per_session_mean_r2": mean_by_session,
        "mean_r2": sum(mean_by_session.values()) / len(mean_by_session),
        "session_query_receipts": reference_query,
        "mask_runtime_shape_by_session": reference_runtime_shape,
        "random_mask_seed_by_session": (
            {session: core.session_seed(session) for session in expected_sessions}
            if spec["mask_mode"] == "random" else {}
        ),
        "model_parameter_count": parameter_count,
        "parameter_delta": 0,
        "persistent_mask_state_bytes_by_session": {
            session: row["persistent_boolean_mask_state_bytes"]
            for session, row in reference_runtime_shape.items()
        },
        "persistent_state_delta_bytes_max": max(
            row["persistent_boolean_mask_state_bytes"] for row in reference_runtime_shape.values()
        ),
        "persistent_mask_state_definition": "one cached torch.bool [N] mask per target session",
        "analytic_dense_mha_mac_delta_current_key_padding_mask_path": 0,
        "masked_tokens_are_not_physically_compacted_in_current_pytorch_path": True,
        "measured_forward_wall_seconds": total_seconds,
        "measured_forward_samples_across_epochs": total_samples,
        "measured_forward_seconds_per_sample": total_seconds / total_samples,
        "measured_mask_setup_wall_seconds": total_mask_setup_seconds,
        "measured_forward_wall_seconds_definition": (
            "query forward only after one persistent per-session boolean mask is built; "
            "one-time mask setup is reported separately"
        ),
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "backward_gradients": False,
        "decoder_weight_updates": False,
        "target_domain_normalizer_refit_performed": False,
        "formal_subc_test_nwb_opened": False,
        "formal_subc_test_session_names_only": list(formal_names),
        "no_test_files_evaluated": True,
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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output or core.score_path(args.intervention, args.domain)
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA_NO_MODEL_NO_GPU",
            "intervention": args.intervention,
            "domain": args.domain,
            "source_gate": str(core.SOURCE_GATE),
            "output": str(output),
        }, indent=2, sort_keys=True))
        return 0
    payload = execute(args.intervention, args.domain, output, device_name=args.device)
    print(json.dumps({"status": "COMPLETE", "mean_r2": payload["mean_r2"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
