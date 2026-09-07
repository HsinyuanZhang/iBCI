#!/usr/bin/env python3
"""A2-compatible M30 scorer for clean/swapped swap-v2 diagnostics.

Dry-run is inert.  Live modes require explicit authorization and never perform
optimizer, backward, target adaptation, or formal-test access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import misleading_identity_swap_v2_core as core

TARGET_AUTH_ENV = "SWAP_V2_TARGET_ACCESS_AUTHORIZATION"
TARGET_AUTH_VALUE = "I_AUTHORIZE_SWAP_V2_DEVELOPMENT_TARGET_ACCESS"
GPU_AUTH_ENV = "SWAP_V2_STAGE_P_GPU_AUTHORIZATION"
GPU_AUTH_VALUE = "I_AUTHORIZE_SWAP_V2_STAGE_P_GPU"


def require_exact_a2_normalizer_authority(observed: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    core.require(dict(observed) == dict(parent),
                 "live A2 source normalizer authority differs from sealed matched parent")


def require_exact_a2_trial30_semantics(
    observed: Mapping[str, Any], parent_by_session: Mapping[str, Any], *, session: str,
) -> None:
    core.require(session in parent_by_session, f"{session}: absent from sealed matched A2 parent")
    core.require(dict(observed) == dict(parent_by_session[session]),
                 f"{session}: full trial-30 semantics differ from sealed matched A2 parent")


def build_synthetic_score_receipt(
    *,
    cell: str,
    domain: str,
    evaluation_input_mode: str,
    pooled_r2: float,
    authority_sha256: str,
) -> dict[str, Any]:
    """Build an explicitly synthetic receipt for unit tests only."""
    core.require(cell in core.CELLS, "score cell drift")
    core.require(domain in core.DOMAINS, "score domain drift")
    core.require(evaluation_input_mode in core.EVAL_INPUT_MODES, "score input mode drift")
    value = float(pooled_r2)
    core.require(value == value and abs(value) != float("inf"), "pooled R2 must be finite")
    core.require(isinstance(authority_sha256, str) and len(authority_sha256) == 64,
                 "authority SHA malformed")
    return {
        "schema_version": 1,
        "receipt_kind": "misleading_identity_swap_v2_synthetic_score",
        "status": "SYNTHETIC_TEST_ONLY__NOT_A_RESULT",
        "screen_id": core.SCREEN_ID,
        "cell": cell,
        "domain": domain,
        "evaluation_input_mode": evaluation_input_mode,
        "pooled_r2": value,
        "matching_authority_sha256": authority_sha256,
        "query_permuted": False,
        "visible_carrier_permuted": False,
        "hidden_descriptor_forwarded_to_model": False,
        "target_optimizer_or_backward_steps": 0,
        "target_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "gpu_used": False,
    }


def validate_synthetic_score_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_synthetic_score_receipt(
        cell=str(payload.get("cell")),
        domain=str(payload.get("domain")),
        evaluation_input_mode=str(payload.get("evaluation_input_mode")),
        pooled_r2=float(payload.get("pooled_r2")),
        authority_sha256=str(payload.get("matching_authority_sha256")),
    )
    core.require(dict(payload) == expected, "synthetic score receipt drift")
    return expected


def _source_stats(lineage: Mapping[str, Any]):
    import numpy as np
    descriptor = lineage.get("descriptor")
    core.require(isinstance(descriptor, Mapping), "source lineage descriptor missing")
    mean = np.asarray(descriptor.get("normalizer_mean_float32"), dtype=np.float32)
    std = np.asarray(descriptor.get("normalizer_std_float32"), dtype=np.float32)
    core.require(mean.shape == (4,) and std.shape == (4,), "source T4 normalizer shape drift")
    core.require((std > 0).all() and np.isfinite(mean).all() and np.isfinite(std).all(),
                 "source T4 normalizer invalid")
    from mc_maze.unit_side_features import side_feature_stats_sha256
    core.require(side_feature_stats_sha256(mean, std) == descriptor.get("normalizer_value_sha256"),
                 "source T4 normalizer value SHA drift")
    return mean, std


def _domain_paths(domain: str) -> list[Path]:
    from mc_maze import a2_matched_subject_shift_v2_core as a2
    if domain == "within_subject":
        manifest = a2.load_strict_manifest()
        root = a2.SUBC_DATA_ROOT.resolve()
        return [(root / f"{session}_behavior+ecephys.nwb").resolve() for session in manifest["val"]]
    if domain == "external_subject_M":
        return a2.external_session_paths()
    raise core.SwapV2ContractError(f"unsupported scoring domain: {domain}")


def _reconcile_a2_source_authority(*, domain: str, carrier: str):
    """Rebuild through the sealed A2 helper before any target path is resolved."""
    from mc_maze import a2_matched_subject_shift_v2_core as a2
    from scripts import a2_matched_subject_shift_v2_score as a2_score

    parent = core.load_a2_parent_domain_bindings(domain)
    core.require(parent["query_policy"] == a2.frozen_query_policy(),
                 "sealed A2 parent query policy differs from live A2 helper")
    core.require(tuple(parent["domain_sessions"]) == a2.expected_domain_sessions(domain),
                 "sealed A2 parent domain roster differs from live A2 roster")
    metadata = {
        "side_features": {
            "group": carrier,
            "normalization_sha256": parent["normalizer_authority"]["side_normalizer_value_sha256"],
        },
        "cache_dir": str(a2.SOURCE_CACHE_ROOT.resolve()),
    }
    authority, behavior_stats, side_stats, formal_names, train_paths = (
        a2_score._fit_source_normalizers(metadata)
    )
    require_exact_a2_normalizer_authority(authority, parent["normalizer_authority"])
    return parent, authority, behavior_stats, side_stats, formal_names, train_paths


def prepare_target_authority(
    *, domain: str, source_lineage_path: Path, authority_out: Path, lineage_out: Path,
    official_preflight_path: Path,
) -> dict[str, Any]:
    """Open only the declared development domain and freeze one shared mapping."""
    core.assert_immutable_pair_fresh(authority_out, label="target matching authority")
    core.assert_immutable_pair_fresh(lineage_out, label="target authority lineage")
    from scripts.misleading_identity_swap_v2_preflight import load_verified_official_preflight
    official, official_sha = load_verified_official_preflight(official_preflight_path)
    import torch
    from mc_maze import a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.unit_side_features import load_unit_side_features

    source_lineage, source_lineage_sha = core.load_verified_immutable_json(source_lineage_path)
    core.require(isinstance(source_lineage.get("implementation_bindings"), Mapping) and
                 bool(source_lineage.get("implementation_bindings")),
                 "source lineage construction closure missing")
    core.require(source_lineage_path.resolve() == Path(official["source_lineage_path"]).resolve() and
                 source_lineage_sha == official["source_lineage_sha256"],
                 "target-authority source lineage differs from official preflight")
    source_authority_path = Path(official["matching_authority_path"])
    source_authority = core.load_verified_authority(source_authority_path)
    core.require(source_authority.sha256 == official["matching_authority_sha256"],
                 "target-authority source matching authority differs from official preflight")
    mean, std = _source_stats(source_lineage)
    parent, normalizer_authority, _behavior_stats, reconciled_side_stats, _formal, _train = (
        _reconcile_a2_source_authority(domain=domain, carrier="t4")
    )
    core.require(np.array_equal(mean, reconciled_side_stats[0]) and
                 np.array_equal(std, reconciled_side_stats[1]),
                 "source lineage normalizer differs from sealed A2 reconciliation")
    paths = _domain_paths(domain)
    expected = a2.expected_domain_sessions(domain)
    core.require(tuple(session_name_from_path(path) for path in paths) == expected,
                 "target diagnostic roster/order drift")
    descriptors: dict[str, torch.Tensor] = {}
    files: list[dict[str, Any]] = []
    for session, path in zip(expected, paths, strict=True):
        values, metadata = load_unit_side_features(
            path, feature_group="t4", pool_size=30, mean=mean, std=std,
            cache_dir=None, bin_size_ms=20, window_size=50,
            trial_result_filter="R", signal_view="sua",
        )
        descriptors[session] = torch.as_tensor(values, dtype=torch.float64)
        files.append({"session": session, "path": str(path), "sha256": core.sha256_file(path),
                      "size_bytes": path.stat().st_size, "unit_count": int(values.shape[0]),
                      "feature_version": int(metadata.feature_version)})
    authority = core.build_target_diagnostic_authority(descriptors, domain=domain)
    body, _side, authority_sha = core.write_immutable_json_pair(authority_out, authority)
    from scripts.misleading_identity_swap_v2_preflight import current_implementation_bindings
    target_lineage = {
        "schema_version": 1, "receipt_kind": "misleading_identity_swap_v2_target_diagnostic_lineage",
        "status": "DEVELOPMENT_TARGET_DIAGNOSTIC_AUTHORITY_BUILT__NOT_A_RESULT",
        "created_at": datetime.now(timezone.utc).isoformat(), "screen_id": core.SCREEN_ID,
        "domain": domain, "domain_sessions": list(expected), "target_files": files,
        "support": "chronological_first_30_rewarded_trials",
        "source_lineage_path": str(source_lineage_path.resolve()),
        "source_lineage_sha256": source_lineage_sha,
        "source_matching_authority_path": str(source_authority_path.resolve()),
        "source_matching_authority_sha256": source_authority.sha256,
        "source_normalizer_value_sha256": source_lineage["descriptor"]["normalizer_value_sha256"],
        "matching_authority_path": str(body), "matching_authority_sha256": authority_sha,
        "same_bytes_consumed_by_cells": list(core.CELLS),
        "same_bytes_consumed_by_carriers": ["t4", "z4"],
        "target_support_direction_used_for_hidden_descriptor": True,
        "target_query_velocity_used_for_weight_updates": False,
        "target_optimizer_or_backward_steps": 0, "formal_subc_test_nwb_opened": False,
        "gpu_used": False,
        "implementation_bindings": current_implementation_bindings(),
        "official_preflight_path": str(core.OFFICIAL_PREFLIGHT_PATH.resolve()),
        "official_preflight_sha256": official_sha,
        "a2_parent_domain_bindings": parent,
        "a2_normalizer_authority": normalizer_authority,
        "a2_query_policy": parent["query_policy"],
    }
    lineage_body, _lineage_side, lineage_sha = core.write_immutable_json_pair(lineage_out, target_lineage)
    return {"authority": str(body), "authority_sha256": authority_sha,
            "lineage": str(lineage_body), "lineage_sha256": lineage_sha}


def _load_cell_model(checkpoint: Path, authority_path: Path, identity: str, mode: str, epoch0: int, device):
    import torch
    from mc_maze import a2_matched_subject_shift_v2_core as a2
    from mc_maze.misleading_identity_swap_v2_trainer import MisleadingIdentitySwapV2LitModule
    raw = checkpoint.read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    core.require(isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict),
                 "swap-v2 checkpoint payload malformed")
    model = MisleadingIdentitySwapV2LitModule(
        task="mc_maze", variant="B3S", teacher_ckpt_path=str(a2.TEACHER_PATH.resolve()),
        window_size=50, trial_length=100, id_hidden_dim=128, hidden_dim=64, pad_value=-1.0,
        freeze_decoder=False, freeze_encoder_base=False, loss_mode="task_only",
        decode_last_timestep_only=True, predict_scaled_behavior=True, behavior_scaling_factor=5.0,
        identity_mode="calibrated", side_dim=4, electrode_embed_dim=0, num_electrodes=0,
        optimizer=None, scheduler=None, compile=False,
        matching_authority_path=str(authority_path.resolve()),
        matching_authority_kind="target_diagnostic", identity_training_mode=identity,
        evaluation_input_mode=mode,
    )
    model.setup("fit"); model.load_state_dict(payload["state_dict"], strict=True)
    model.set_evaluation_input_mode(mode, checkpoint_lightning_current_epoch=epoch0)
    for parameter in model.parameters(): parameter.requires_grad = False
    model.to(device).eval()
    return model, hashlib.sha256(raw).hexdigest()


def execute_score(*, cell: str, domain: str, mode: str, terminal_receipt_path: Path,
                  target_authority_path: Path, target_lineage_path: Path,
                  source_lineage_path: Path, out_path: Path, device_name: str,
                  official_preflight_path: Path) -> dict[str, Any]:
    core.assert_immutable_pair_fresh(out_path, label="domain score")
    from scripts.misleading_identity_swap_v2_preflight import load_verified_official_preflight
    official, official_sha = load_verified_official_preflight(official_preflight_path)
    import torch
    from mc_maze import a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import session_name_from_path
    from scripts.eval_adaptation_dandi688 import (
        PAD_VALUE, attach_side_features, build_calib_trials_for_indices, eval_r2,
        load_session_with_trials, make_subset_dataset,
    )
    core.require(torch.cuda.is_available(), "live swap-v2 scorer refuses CPU fallback")
    device = torch.device(device_name); core.require(device.type == "cuda", "score device must be CUDA")
    identity, carrier = cell.split("_", 1)
    terminal, terminal_sha = core.load_verified_immutable_json(terminal_receipt_path)
    core.require(terminal.get("cell") == cell and terminal.get("status") == "CELL_TRAINING_COMPLETE__DEVELOPMENT_NOT_OFFICIAL",
                 "cell terminal receipt drift")
    core.require(terminal.get("official_preflight_path") == str(core.OFFICIAL_PREFLIGHT_PATH.resolve()) and
                 terminal.get("official_preflight_sha256") == official_sha,
                 "cell terminal official-preflight binding drift")
    core.require(Path(str(terminal.get("cell_output_path", ""))).resolve() ==
                 Path(official["cell_output_paths"][cell]).resolve(),
                 "cell terminal output topology drift")
    from scripts.misleading_identity_swap_v2_preflight import current_implementation_bindings
    current_bindings = current_implementation_bindings()
    core.require(terminal.get("implementation_bindings") == current_bindings,
                 "cell terminal implementation closure drift")
    target_authority = core.load_verified_runtime_authority(target_authority_path, expected_kind="target_diagnostic")
    target_lineage, target_lineage_sha = core.load_verified_immutable_json(target_lineage_path)
    core.require(target_lineage.get("matching_authority_sha256") == target_authority.sha256,
                 "target lineage/authority binding drift")
    core.require(target_lineage.get("domain") == domain, "target lineage domain drift")
    core.require(target_lineage.get("official_preflight_path") == str(core.OFFICIAL_PREFLIGHT_PATH.resolve()) and
                 target_lineage.get("official_preflight_sha256") == official_sha,
                 "target lineage official-preflight binding drift")
    core.require(target_lineage.get("implementation_bindings") == current_bindings,
                 "target lineage implementation closure drift")
    source_lineage, source_lineage_sha = core.load_verified_immutable_json(source_lineage_path)
    core.require(isinstance(source_lineage.get("implementation_bindings"), Mapping) and
                 bool(source_lineage.get("implementation_bindings")),
                 "source lineage construction closure missing")
    core.require(source_lineage_path.resolve() == Path(official["source_lineage_path"]).resolve() and
                 source_lineage_sha == official["source_lineage_sha256"],
                 "score source lineage differs from official preflight")
    source_authority_path = Path(official["matching_authority_path"])
    source_authority = core.load_verified_authority(source_authority_path)
    core.require(source_authority.sha256 == official["matching_authority_sha256"],
                 "score source matching authority differs from official preflight")
    core.require(target_lineage.get("source_lineage_path") == str(source_lineage_path.resolve()) and
                 target_lineage.get("source_lineage_sha256") == source_lineage_sha and
                 target_lineage.get("source_matching_authority_path") == str(source_authority_path.resolve()) and
                 target_lineage.get("source_matching_authority_sha256") == source_authority.sha256,
                 "target lineage source scientific bindings differ from official preflight")
    core.require(target_lineage.get("source_lineage_sha256") == source_lineage_sha,
                 "target/source lineage binding drift")
    side_mean, side_std = _source_stats(source_lineage)
    parent, normalizer_authority, behavior_stats, reconciled_side_stats, formal_names, _train_paths = (
        _reconcile_a2_source_authority(domain=domain, carrier=carrier)
    )
    behavior_mean, behavior_std = behavior_stats
    core.require(np.array_equal(side_mean, reconciled_side_stats[0]) and
                 np.array_equal(side_std, reconciled_side_stats[1]),
                 "score source lineage normalizer differs from sealed A2 reconciliation")
    core.require(target_lineage.get("a2_parent_domain_bindings") == parent and
                 target_lineage.get("a2_normalizer_authority") == normalizer_authority and
                 target_lineage.get("a2_query_policy") == parent["query_policy"],
                 "target lineage sealed A2 parent binding drift")
    paths = _domain_paths(domain); expected = a2.expected_domain_sessions(domain)
    core.require(list(expected) == parent["domain_sessions"], "score domain roster differs from sealed A2 parent")
    checkpoint_paths = terminal.get("checkpoint_paths_by_epoch_one_based")
    checkpoint_shas = terminal.get("checkpoint_sha256_by_epoch_one_based")
    core.require(isinstance(checkpoint_paths, Mapping) and isinstance(checkpoint_shas, Mapping),
                 "terminal checkpoint bundle missing")
    per_epoch: dict[str, Any] = {}
    query_receipts: dict[str, Any] | None = None
    for epoch1 in range(5, 13):
        checkpoint = Path(str(checkpoint_paths[str(epoch1)]))
        model, checkpoint_sha = _load_cell_model(checkpoint, target_authority_path, identity, mode, epoch1 - 1, device)
        core.require(checkpoint_sha == checkpoint_shas[str(epoch1)], "checkpoint byte SHA drift")
        scores: dict[str, float] = {}; epoch_queries: dict[str, Any] = {}
        for session, path in zip(expected, paths, strict=True):
            record = load_session_with_trials(path, 20, 50, 30, 100, PAD_VALUE,
                                              behavior_mean, behavior_std, trial_result_filter="R",
                                              cache_dir=None, signal_view="sua")
            core.require(record["name"] == session, "scoring session identity drift")
            semantics = a2.trial30_semantics_from_trials(
                record["trials"], require_target_labels=True, session=session
            )
            record["calib_trials"] = build_calib_trials_for_indices(record, list(range(30)), 30)
            record = attach_side_features(record, path, side_feature_group=carrier,
                                          waveform_feature_group="t4", pool_size=30,
                                          permutation_seed=None, mean=side_mean, std=side_std, cache_dir=None)
            query = record["trials"][30:]; dataset = make_subset_dataset(record, query, session)
            core.require(len(dataset) > 0, f"{session}: empty post-M30 query")
            encoder = model.student.id_encoder
            encoder.configure_runtime(session=session, lightning_current_epoch=epoch1 - 1,
                                      phase="eval_swapped_diagnostic" if mode == "swapped_diagnostic" else "eval_clean")
            scores[session] = float(eval_r2(model, dataset, device))
            epoch_queries[session] = {
                **semantics,
                "dataset_query_window_count": len(dataset),
                "activity_calibration_trial_indices": list(range(30)),
                "side_feature_label_pool_trial_indices": list(range(30)),
                "behavior_normalizer_authority": "strict_subc_source_train_27_only",
                "side_normalizer_authority": "strict_subc_source_train_27_only",
                "target_session_carrier_fit_performed": True,
                "target_direction_labels_used_for_carrier": True,
                "target_velocity_labels_used_for_weight_updates": False,
                "backward_gradients": False,
                "decoder_weight_updates": False,
            }
            require_exact_a2_trial30_semantics(
                epoch_queries[session], parent["session_query_receipts"], session=session
            )
        if query_receipts is None: query_receipts = epoch_queries
        else: core.require(query_receipts == epoch_queries, "query semantics drift across epochs")
        per_epoch[str(epoch1)] = {"checkpoint_sha256": checkpoint_sha, "per_session_r2": scores,
                                  "mean_r2": sum(scores.values()) / len(scores)}
    per_session = {session: sum(per_epoch[str(epoch)]["per_session_r2"][session] for epoch in range(5, 13)) / 8
                   for session in expected}
    payload = {
        "schema_version": 1, "receipt_kind": "misleading_identity_swap_v2_domain_score",
        "status": "DEVELOPMENT_SCORE_COMPLETE__NOT_OFFICIAL", "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID, "cell": cell, "domain": domain, "evaluation_input_mode": mode,
        "cell_terminal_receipt_path": str(terminal_receipt_path.resolve()), "cell_terminal_receipt_sha256": terminal_sha,
        "target_matching_authority_path": str(target_authority_path.resolve()),
        "target_matching_authority_sha256": target_authority.sha256,
        "target_lineage_path": str(target_lineage_path.resolve()), "target_lineage_sha256": target_lineage_sha,
        "source_lineage_path": str(source_lineage_path.resolve()), "source_lineage_sha256": source_lineage_sha,
        "implementation_bindings": current_bindings,
        "official_preflight_path": str(core.OFFICIAL_PREFLIGHT_PATH.resolve()),
        "official_preflight_sha256": official_sha,
        "source_matching_authority_path": terminal["matching_authority_path"],
        "source_matching_authority_sha256": terminal["matching_authority_sha256"],
        "a2_parent_domain_bindings": parent,
        "a2_parent_query_policy": parent["query_policy"],
        "a2_parent_normalizer_authority": normalizer_authority,
        "a2_parent_domain_sessions": parent["domain_sessions"],
        "epoch_window_one_based": list(range(5, 13)), "per_epoch": per_epoch,
        "per_session_mean_r2": per_session, "mean_r2": sum(per_session.values()) / len(per_session),
        "session_query_receipts": query_receipts, "query_permuted": False,
        "visible_carrier_permuted": False, "electrode_ids_permuted": False,
        "target_support_direction_used_for_hidden_descriptor_and_carrier": True,
        "target_query_velocity_used_for_metric": True, "target_query_velocity_used_for_weight_updates": False,
        "target_optimizer_or_backward_steps": 0, "formal_subc_test_nwb_opened": False,
        "formal_test_session_names_only": formal_names, "execution_device": str(device),
    }
    core.write_immutable_json_pair(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=core.CELLS)
    parser.add_argument("--domain", choices=core.DOMAINS)
    parser.add_argument("--evaluation-input-mode", choices=core.EVAL_INPUT_MODES)
    parser.add_argument("--source-lineage", type=Path)
    parser.add_argument("--target-authority", type=Path)
    parser.add_argument("--target-lineage", type=Path)
    parser.add_argument("--cell-terminal-receipt", type=Path)
    parser.add_argument("--out-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-preflight", type=Path, default=core.OFFICIAL_PREFLIGHT_PATH)
    parser.add_argument("--prepare-target-authority", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.prepare_target_authority:
        core.require(os.environ.get(TARGET_AUTH_ENV) == TARGET_AUTH_VALUE,
                     "explicit development target-access authorization missing")
        core.require(args.domain and args.source_lineage and args.target_authority and args.target_lineage,
                     "target-authority preparation arguments missing")
        print(json.dumps(prepare_target_authority(domain=args.domain,
            source_lineage_path=args.source_lineage, authority_out=args.target_authority,
            lineage_out=args.target_lineage, official_preflight_path=args.official_preflight), indent=2, sort_keys=True)); return 0
    if args.launch:
        core.require(os.environ.get(TARGET_AUTH_ENV) == TARGET_AUTH_VALUE, "target authorization missing")
        core.require(os.environ.get(GPU_AUTH_ENV) == GPU_AUTH_VALUE, "GPU authorization missing")
        core.require(all((args.cell, args.domain, args.evaluation_input_mode, args.source_lineage,
                          args.target_authority, args.target_lineage, args.cell_terminal_receipt, args.out_path)),
                     "live scorer arguments missing")
        payload = execute_score(cell=args.cell, domain=args.domain, mode=args.evaluation_input_mode,
            terminal_receipt_path=args.cell_terminal_receipt, target_authority_path=args.target_authority,
            target_lineage_path=args.target_lineage, source_lineage_path=args.source_lineage,
            out_path=args.out_path, device_name=args.device,
            official_preflight_path=args.official_preflight)
        print(json.dumps({"status": payload["status"], "mean_r2": payload["mean_r2"]}, indent=2)); return 0
    print(json.dumps({
        "status": "DRY_RUN__NO_CHECKPOINT_NO_DATA_NO_GPU",
        "screen_id": core.SCREEN_ID,
        "live_scorer_enabled": True,
        "required_cells": list(core.CELLS),
        "domains": list(core.DOMAINS),
        "evaluation_input_modes": list(core.EVAL_INPUT_MODES),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
