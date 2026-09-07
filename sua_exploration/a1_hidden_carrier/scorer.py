"""Score one A1 H/T4 epoch bundle and its same-checkpoint H/TS4 control."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
STREAMING = REPO / "streaming_calibration_exp"
SCRIPTS = SUA / "scripts"
for root in (SUA, STREAMING, SCRIPTS):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from a1_hidden_carrier.a2_anchors import (  # noqa: E402
    DEV_SESSIONS,
    FORMAL_SESSIONS,
    MANIFEST,
    TEACHER,
    T4_NORMALIZER_SHA256,
)
from a1_hidden_carrier.artifacts import (  # noqa: E402
    canonical_json_sha256,
    load_verified_immutable_json,
    require,
    sha256_file,
    write_immutable_json,
)
from a1_hidden_carrier.contract import PILOT_SEED, SCREEN_ID  # noqa: E402
from a1_hidden_carrier.evidence import load_verified_preflight  # noqa: E402


def _load_metadata(run_dir: Path, *, preflight_sha: str, launch_sha: str) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    require(path.is_file(), f"A1 run metadata missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "A1 run metadata malformed")
    expected = {
        "schema_version": 2,
        "status": "completed",
        "screen_id": SCREEN_ID,
        "seed": PILOT_SEED,
        "task": "CO",
        "variant": "B3S",
        "held_out_test_evaluated": False,
        "formal_subc_test_nwb_opened": False,
        "official_preflight_sha256": preflight_sha,
        "launch_receipt_sha256": launch_sha,
    }
    for key, value in expected.items():
        require(type(payload.get(key)) is type(value) and payload.get(key) == value, f"A1 run metadata {key} drift")
    training = payload.get("training") or {}
    for key, value in {
        "max_epochs": 12, "no_early_stopping": True,
        "checkpoint_every_epoch": True, "calibration_n_trials": 30,
        "random_calibration": False, "calibration_selection": "chronological_first_n",
        "loss_mode": "task_only", "lambda_y": 0.0, "lambda_E": 0.0,
        "identity_mode": "calibrated", "freeze_decoder": False,
        "freeze_encoder_base": False, "wiring_smoke": False,
    }.items():
        require(type(training.get(key)) is type(value) and training.get(key) == value, f"A1 training {key} drift")
    require(tuple((payload.get("session_splits") or {}).get("val") or ()) == DEV_SESSIONS, "A1 val roster drift")
    require(tuple((payload.get("session_splits") or {}).get("test") or ()) == FORMAL_SESSIONS, "A1 formal exclusion roster drift")
    side = payload.get("side_features") or {}
    require(side.get("group") == "t4" and side.get("pool_size") == 30 and side.get("side_dim") == 4, "A1 T4 metadata drift")
    require(side.get("normalization_sha256") == T4_NORMALIZER_SHA256, "A1 T4 normalizer drift")
    a1 = payload.get("a1_hidden_carrier") or {}
    require(a1.get("add_site") == "hidden" and a1.get("activity_identity_carrier") == "z4", "A1 interface drift")
    coverage = a1.get("optimizer_coverage") or {}
    require(coverage.get("hidden_carrier_parameter_present") is True, "P absent from source optimizer")
    require(coverage.get("all_trainable_parameters_exactly_once") is True, "A1 optimizer coverage failed")
    require(coverage.get("duplicate_parameter_count") == 0, "A1 source optimizer contains duplicates")
    require(coverage.get("trainable_tensor_count") == 40, "A1 source optimizer must cover exactly 40 tensors")
    require(coverage.get("optimizer_tensor_count") == 40, "A1 optimizer tensor count drift")
    require(coverage.get("decoder_tensor_count") == 31, "A1 decoder optimizer coverage drift")
    require(coverage.get("identity_encoder_tensor_count") == 8, "A1 identity optimizer coverage drift")
    require(coverage.get("hidden_carrier_tensor_count") == 1, "A1 P optimizer coverage drift")
    return payload


def _load_a1_model(checkpoint_path: Path, device: Any):
    """Safe load: torch payload -> construct -> setup(test) -> strict state load."""
    import torch
    from scripts.eval_adaptation_dandi688 import (
        BEHAVIOR_SCALING_FACTOR,
        HIDDEN_DIM,
        ID_HIDDEN_DIM,
        PAD_VALUE,
        TRIAL_LENGTH,
        WINDOW_SIZE,
        checkpoint_architecture_kwargs,
    )
    from src.models.a1_hidden_carrier_module import A1HiddenCarrierLitModule

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping), f"A1 checkpoint malformed: {checkpoint_path}")
    state = checkpoint.get("state_dict")
    hyper = checkpoint.get("hyper_parameters")
    require(isinstance(state, Mapping) and isinstance(hyper, Mapping), "A1 checkpoint state/hparams missing")
    for key, expected in {
        "variant": "B3S", "freeze_decoder": False, "loss_mode": "task_only",
        "lambda_y": 0.0, "lambda_E": 0.0, "identity_mode": "calibrated",
        "side_dim": 4, "decoder_mode": "coupled", "fixed_slot_count": 0,
        "a1_attachment_mode": "aligned", "a1_carrier_dim": 4,
    }.items():
        require(type(hyper.get(key)) is type(expected) and hyper.get(key) == expected, f"A1 checkpoint hparam {key} drift")
    architecture = checkpoint_architecture_kwargs(dict(checkpoint))
    model = A1HiddenCarrierLitModule(
        task="mc_maze", variant="B3S", teacher_ckpt_path=str(TEACHER.resolve()),
        window_size=WINDOW_SIZE, trial_length=TRIAL_LENGTH,
        id_hidden_dim=ID_HIDDEN_DIM, hidden_dim=HIDDEN_DIM, pad_value=PAD_VALUE,
        freeze_decoder=False, freeze_encoder_base=False, loss_mode="task_only",
        lambda_y=0.0, lambda_E=0.0, decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=BEHAVIOR_SCALING_FACTOR,
        identity_mode="calibrated", **architecture, optimizer=None, scheduler=None,
        compile=False, a1_attachment_mode="aligned",
        a1_attachment_permutation_seed=None, a1_carrier_dim=4,
    )
    model.setup("test")
    result = model.load_state_dict(state, strict=True)
    require(not result.missing_keys and not result.unexpected_keys, "A1 strict checkpoint load was not exact")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    return model, {
        "strict_load_missing_keys": list(result.missing_keys),
        "strict_load_unexpected_keys": list(result.unexpected_keys),
        "hidden_carrier_key_present": "student.hidden_carrier_map.weight" in state,
        "hidden_carrier_shape": list(state["student.hidden_carrier_map.weight"].shape),
    }


def _evaluate_epoch(
    *, checkpoint: Path, domain_paths: list[Path], behavior_stats: tuple[Any, Any],
    side_stats: tuple[Any, Any], device: Any, attachment_mode: str,
    permutation_seed: int | None,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, object]]:
    import torch
    from mc_maze import a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import session_name_from_path
    from scripts.eval_adaptation_dandi688 import (
        PAD_VALUE, attach_side_features, build_calib_trials_for_indices,
        eval_r2, load_session_with_trials, make_subset_dataset,
    )

    model, load_evidence = _load_a1_model(checkpoint, device)
    model.student.set_attachment_control_for_evaluation(
        attachment_mode=attachment_mode, permutation_seed=permutation_seed
    )
    scores: dict[str, float] = {}
    queries: dict[str, dict[str, Any]] = {}
    behavior_mean, behavior_std = behavior_stats
    side_mean, side_std = side_stats
    with torch.no_grad():
        for nwb_path in domain_paths:
            session = session_name_from_path(nwb_path)
            record = load_session_with_trials(
                nwb_path, 20, 50, 30, 100, PAD_VALUE,
                behavior_mean, behavior_std, trial_result_filter="R",
                cache_dir=None, signal_view="sua",
            )
            require(record["name"] == session, "A1 session identity drift")
            semantics = a2.trial30_semantics_from_trials(
                record["trials"], require_target_labels=True, session=session
            )
            indices = list(range(30))
            record["calib_trials"] = build_calib_trials_for_indices(record, indices, 30)
            record = attach_side_features(
                record, nwb_path, side_feature_group="t4", waveform_feature_group="t4",
                pool_size=30, permutation_seed=None, mean=side_mean, std=side_std,
                cache_dir=None,
            )
            dataset = make_subset_dataset(record, record["trials"][30:], session)
            require(len(dataset) == semantics["post30_query_window_count"] and len(dataset) > 0, "A1 query trace drift")
            scores[session] = float(eval_r2(model, dataset, device))
            queries[session] = {
                **semantics, "dataset_query_window_count": len(dataset),
                "activity_calibration_trial_indices": indices,
                "side_feature_label_pool_trial_indices": indices,
                "backward_gradients": False, "decoder_weight_updates": False,
                "target_velocity_labels_used_for_weight_updates": False,
            }
    require(tuple(scores) == DEV_SESSIONS, "A1 score roster/order drift")
    return scores, queries, load_evidence


def execute(
    *, run_dir: Path, preflight_path: Path, launch_path: Path, out_path: Path,
    device_name: str, permutation_seed: int,
) -> dict[str, Any]:
    import torch
    from scripts import a2_matched_subject_shift_v2_score as a2_score
    from mc_maze import a2_matched_subject_shift_v2_core as a2

    preflight, preflight_sha = load_verified_preflight(preflight_path)
    launch, launch_sha = load_verified_immutable_json(launch_path, label="A1 pretraining launch receipt")
    for key, expected in {
        "schema_version": 2,
        "receipt_kind": "a1_hidden_space_carrier_pretraining_launch",
        "screen_id": SCREEN_ID,
        "status": "ROOT_GO_AND_GPU_AUTHORIZED_PRETRAINING_ENVIRONMENT_VERIFIED",
        "cell": "H/T4",
        "seed": PILOT_SEED,
        "implementation_bindings_sha256": preflight["implementation_bindings_sha256"],
        "training_started": False,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
    }.items():
        require(type(launch.get(key)) is type(expected) and launch.get(key) == expected, f"A1 launch {key} drift")
    require(launch.get("official_preflight_sha256") == preflight_sha, "A1 launch/preflight pairing drift")
    require(torch.cuda.is_available(), "A1 production scorer requires CUDA")
    device = torch.device(device_name)
    require(device.type == "cuda", "A1 scorer refuses CPU fallback")
    run_dir = run_dir.resolve()
    metadata = _load_metadata(run_dir, preflight_sha=preflight_sha, launch_sha=launch_sha)
    authority, behavior_stats, side_stats, formal_names, _ = a2_score._fit_source_normalizers(metadata)
    require(authority["side_normalizer_value_sha256"] == T4_NORMALIZER_SHA256, "A1 scorer normalizer drift")
    require(tuple(formal_names) == FORMAL_SESSIONS, "formal exclusion names drift")
    _train, domain_paths, formal = a2.active_source_session_paths(MANIFEST)
    require(tuple(formal) == FORMAL_SESSIONS, "formal exclusion roster drift")
    from mc_maze.multisession_datamodule import session_name_from_path
    require(tuple(session_name_from_path(path) for path in domain_paths) == DEV_SESSIONS, "development path roster drift")
    checkpoints = [Path(path) for path in metadata["epoch_checkpoints"]]
    require(len(checkpoints) == 12, "A1 source run must contain exactly 12 epochs")
    bundle = {str(i + 1): sha256_file(path) for i, path in enumerate(checkpoints)}
    require(bundle == metadata["epoch_checkpoint_sha256_bundle"], "A1 checkpoint bundle byte drift")
    require(canonical_json_sha256(bundle) == metadata["epoch_checkpoint_sha256_bundle_sha256"], "A1 bundle digest drift")

    per_epoch: dict[str, dict[str, object]] = {}
    reference_queries: dict[str, dict[str, Any]] | None = None
    load_evidence: dict[str, object] | None = None
    for epoch in range(5, 13):
        path = checkpoints[epoch - 1]
        aligned, queries_a, loaded_a = _evaluate_epoch(
            checkpoint=path, domain_paths=domain_paths, behavior_stats=behavior_stats,
            side_stats=side_stats, device=device, attachment_mode="aligned",
            permutation_seed=None,
        )
        shuffled, queries_s, loaded_s = _evaluate_epoch(
            checkpoint=path, domain_paths=domain_paths, behavior_stats=behavior_stats,
            side_stats=side_stats, device=device, attachment_mode="shuffled",
            permutation_seed=permutation_seed,
        )
        require(queries_a == queries_s, "aligned/TS4 query trace differs")
        require(loaded_a == loaded_s, "aligned/TS4 strict-load evidence differs")
        if reference_queries is None:
            reference_queries = queries_a
            load_evidence = loaded_a
        else:
            require(queries_a == reference_queries, "A1 query trace drifted across epochs")
            require(loaded_a == load_evidence, "A1 strict-load evidence drifted across epochs")
        per_epoch[str(epoch)] = {
            "checkpoint_path": str(path.resolve()), "checkpoint_sha256": bundle[str(epoch)],
            "aligned_per_session_r2": aligned, "shuffled_per_session_r2": shuffled,
        }
    assert reference_queries is not None and load_evidence is not None
    aligned_mean = {
        session: sum(per_epoch[str(epoch)]["aligned_per_session_r2"][session] for epoch in range(5, 13)) / 8
        for session in DEV_SESSIONS
    }
    shuffled_mean = {
        session: sum(per_epoch[str(epoch)]["shuffled_per_session_r2"][session] for epoch in range(5, 13)) / 8
        for session in DEV_SESSIONS
    }
    query_trace_sha = canonical_json_sha256(reference_queries)
    payload = {
        "schema_version": 2,
        "receipt_kind": "a1_hidden_space_carrier_h_t4_score",
        "screen_id": SCREEN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "H_T4_ALIGNED_AND_ATTACHMENT_SCORE_COMPLETED",
        "cell": "H/T4", "seed": PILOT_SEED, "fresh_training_family": True,
        "new_teacher_required": False, "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True, "target_backward_gradients": False,
        "target_weight_updates": False,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_normalizer_refit_performed": False,
        "official_preflight_sha256": preflight_sha, "launch_receipt_sha256": launch_sha,
        "run_metadata_path": str((run_dir / "run_metadata.json").resolve()),
        "run_metadata_sha256": sha256_file(run_dir / "run_metadata.json"),
        "source_training": {
            "task": "CO", "variant": "B3S", "side_dim": 4,
            "decoder_mode": "coupled", "fixed_slot_count": 0,
            "add_site": "hidden", "carrier": "t4", "activity_identity_carrier": "z4",
            "freeze_decoder": False, "freeze_encoder_base": False,
            "loss_mode": "task_only", "lambda_y": 0.0, "lambda_E": 0.0,
            "total_epochs": 12, "epoch_window": list(range(5, 13)),
            "no_early_stopping": True,
            "optimizer_coverage": metadata["a1_hidden_carrier"]["optimizer_coverage"],
        },
        "protocol": {
            "activity_calibration_n": 30, "pool_size": 30,
            "evaluation_start_trial_index": 30, "selection_mode": "first",
            "chronological_calibration": True, "total_epochs": 12,
            "epoch_window": list(range(5, 13)),
            "epoch_score_rule": "unweighted mean session R2 over exactly source epochs 5..12",
            "signal_view": "sua",
        },
        "aligned": {"attachment_mode": "aligned", "per_session_mean_r2": aligned_mean},
        "attachment_control": {
            "attachment_mode": "shuffled", "training_run_created": False,
            "same_h_t4_checkpoint": True, "permutation_seed": permutation_seed,
            "per_session_mean_r2": shuffled_mean,
        },
        "same_checkpoint_attachment_evidence": {
            "verified": True,
            "aligned_checkpoint_bundle_sha256": metadata["epoch_checkpoint_sha256_bundle_sha256"],
            "shuffled_checkpoint_bundle_sha256": metadata["epoch_checkpoint_sha256_bundle_sha256"],
            "aligned_query_trace_sha256": query_trace_sha,
            "shuffled_query_trace_sha256": query_trace_sha,
        },
        "checkpoint_load_evidence": load_evidence,
        "per_epoch": per_epoch, "session_query_receipts": reference_queries,
        "normalizer_authority": authority, "execution_device": str(device),
    }
    write_immutable_json(out_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--official-preflight", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--permutation-seed", type=int, default=20260813)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps({"status": "DRY_RUN_NO_CHECKPOINT_OR_NWB_OPEN", "screen_id": SCREEN_ID}, sort_keys=True))
        return
    payload = execute(
        run_dir=args.run_dir, preflight_path=args.official_preflight,
        launch_path=args.launch_receipt, out_path=args.out,
        device_name=args.device, permutation_seed=args.permutation_seed,
    )
    print(json.dumps({"status": payload["status"], "out": str(args.out.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
