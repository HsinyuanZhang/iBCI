#!/usr/bin/env python3
"""Score a completed shared-Z4 C1 matrix on reused development sessions only.

This evaluator is intentionally not called by the source runner or scheduler.
It is a separately gated, future operation that can start only after all three
fixed source seeds have terminal checkpoints.  Unlike the legacy generic Z4
path, it never imports or invokes a T4 feature fitter: each record receives a
direct ``float32 zeros[N,4]`` side tensor in the already-standardized B3S
coordinate.

The common deployment timeline is retained: first 30 rewarded trials provide
neural activity identity, trials 30--49 provide neither Z4 descriptor nor
identity, and query scoring starts at trial 50.  The terminal checkpoint is
fixed as epoch_011; it is not an epoch-window argmax or a first-cell selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from eval_adaptation_dandi688 import (
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    load_session_with_trials,
)
from mc_maze.multisession_datamodule import (
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    nwb_unit_count,
    session_name_from_path,
)
from mc_maze.paired_view_c1_shared_zero4 import (
    DIRECT_ZERO_CONSTRUCTION,
    SIDE_DIM,
    STANDARDIZED_COORDINATE_NAME,
    attach_standardized_zero4_to_evaluation_record,
)
from select_gradient_free_protocol_dandi688 import (
    evaluate_session_configs,
    load_frozen_model,
)
import shared_zero4_terminal_completion_bridge as completion_bridge


EXPECTED_SEEDS = (42, 43, 44)
# Lightning writes the twelfth protocol epoch as zero-based ``epoch_011``.
# Keep both labels explicit so an old C1 scorer's per_epoch["11"] cannot be
# accidentally compared (that key maps to epoch_010, not epoch_011).
CHECKPOINT_EPOCH_INDEX = 11
PROTOCOL_EPOCH_NUMBER = 12
EXPECTED_STRICT_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
SEALED_FORMAL_TEST_NAMES = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)


def compute_fixed_terminal_score(per_epoch_mean_r2: Mapping[int, float]) -> float:
    """Score the one fixed terminal epoch without importing generic E3/M3 code.

    The generic epoch-window module imports B3S feature-policy helpers at
    module load, including legacy side-feature behaviour that is deliberately
    outside this direct-Z4 evaluator.  The frozen terminal contract has a
    single declared protocol epoch, so this local, exact key check is both
    smaller and easier to audit.
    """

    if set(per_epoch_mean_r2) != {PROTOCOL_EPOCH_NUMBER}:
        raise ValueError("shared_zero4 terminal score requires protocol epoch 12 only")
    return float(per_epoch_mean_r2[PROTOCOL_EPOCH_NUMBER])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)


def _require_sealed_regular_file(path: Path, *, label: str) -> Path:
    """Reject raw symlinks and require a 0444 sealed regular artifact."""

    if path.is_symlink():
        raise ValueError(f"shared_zero4 {label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or resolved.is_symlink():
        raise ValueError(f"shared_zero4 {label} is not a regular file: {resolved}")
    if stat.S_IMODE(info.st_mode) != 0o444:
        raise ValueError(f"shared_zero4 {label} is not sealed 0444: {resolved}")
    return resolved


def _require_sealed_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"shared_zero4 {label} must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISDIR(info.st_mode) or resolved.is_symlink():
        raise ValueError(f"shared_zero4 {label} is not a directory: {resolved}")
    if stat.S_IMODE(info.st_mode) != 0o555:
        raise ValueError(f"shared_zero4 {label} is not sealed 0555: {resolved}")
    return resolved


def _load_completed_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    metadata = load_json(metadata_path)
    training = metadata.get("training") or {}
    if (
        metadata.get("status"),
        metadata.get("training_kind"),
        metadata.get("variant"),
        metadata.get("held_out_test_evaluated"),
        metadata.get("formal_sua_files_opened"),
        metadata.get("subm_nwb_files_opened"),
        training.get("terminal_checkpoint"),
        training.get("checkpoint_selection"),
        training.get("future_development_activity_calibration_n_trials"),
        training.get("future_development_query_start_trial"),
        training.get("future_development_trials_30_49_enter_zero4_identity_or_descriptor"),
        training.get("development_score_invoked"),
        training.get("development_score_artifact_paths"),
    ) != (
        "completed",
        "shared_paired_view_direct_standardized_zero4",
        "B3S",
        False,
        False,
        False,
        "epoch_011.ckpt",
        "fixed_terminal_epoch_011_no_selection",
        30,
        50,
        False,
        False,
        [],
    ):
        raise ValueError("shared_zero4 completed metadata contract drift")
    views = metadata.get("view_configs")
    if not isinstance(views, dict) or set(views) != {"sua", "pseudo_mua"}:
        raise ValueError("shared_zero4 view metadata drift")
    for view, config in views.items():
        side = config.get("side_features") if isinstance(config, dict) else None
        if not isinstance(side, dict) or (
            side.get("group"),
            side.get("side_dim"),
            side.get("coordinate"),
            side.get("construction"),
            side.get("target_direction_label_reads_for_descriptor"),
            side.get("t4_trial_rate_reads_for_descriptor"),
            side.get("label_access_scope"),
            side.get("target_t4_rate_fit_calls"),
            side.get("raw_t4_constructed"),
            side.get("source_t4_normalizer_arithmetic_performed"),
        ) != (
            "shared_zero4_direct_standardized",
            SIDE_DIM,
            STANDARDIZED_COORDINATE_NAME,
            DIRECT_ZERO_CONSTRUCTION,
            0,
            0,
            "descriptor_only",
            0,
            False,
            False,
        ):
            raise ValueError(f"shared_zero4 {view} descriptor metadata drift")
    return metadata


def _require_complete_three_seed_matrix(
    completion_path: Path,
    *,
    program_receipt_sha256: str,
) -> dict[str, Any]:
    _require_sealed_regular_file(completion_path, label="matrix-completion receipt")
    completion = load_json(completion_path)
    rows = completion.get("completed_seeds")
    observed = {
        int(row.get("seed"))
        for row in rows
        if isinstance(row, dict) and row.get("status") == "completed"
    } if isinstance(rows, list) else set()
    if (
        completion.get("status") != "completed_all_fixed_source_seeds_score_blind"
        or completion.get("program_receipt_sha256") != program_receipt_sha256
        or observed != set(EXPECTED_SEEDS)
        or completion.get("scheduler_development_score_invocations") != 0
        or completion.get("evaluator_development_score_invocations_before_completion") != 0
        or completion.get("development_score_authorized_by_this_receipt") is not False
    ):
        raise ValueError("shared_zero4 development evaluator requires the completed score-blind 3-seed matrix")
    return completion


def _load_supported_completion_receipt(
    completion_path: Path,
    *,
    program_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept legacy, V3-adapter, or content-bound direct completion.

    Launch capability time windows are deliberately absent here: this is a
    post-completion verifier over sealed terminal files, not a launch gate.
    """

    _require_sealed_regular_file(completion_path, label="matrix-completion receipt")
    probe = load_json(completion_path)
    if completion_bridge.is_direct_recovery_payload(probe):
        verified = completion_bridge.verify_direct_recovery_receipt(completion_path)
        return probe, verified
    if completion_bridge.is_v3_adapter_payload(probe):
        verified = completion_bridge.verify_v3_adapter_receipt(completion_path)
        return probe, verified
    completion = _require_complete_three_seed_matrix(
        completion_path,
        program_receipt_sha256=program_receipt_sha256,
    )
    return completion, {
        "kind": "legacy_v1_matrix_completion",
        "schema": completion.get("schema"),
        "status": completion["status"],
        "path": str(completion_path.resolve(strict=True)),
        "sha256": sha256_file(completion_path),
        "fixed_seeds": list(EXPECTED_SEEDS),
    }


def _require_exact_completion_binding(
    completion: Mapping[str, Any],
    *,
    run_dir: Path,
    metadata_path: Path,
    terminal_checkpoint: Path,
    seed: int,
) -> None:
    """Re-validate the scheduler's exact terminal tuple before any scoring.

    A matrix-completion receipt is not merely a statement that three seed
    labels appeared.  It is a binding to one live run directory, one metadata
    file, one fixed epoch-011 checkpoint, one sealed source status, one
    closure manifest, and one single-use authorization claim for each seed.
    Rechecking all of these here prevents a later caller from swapping a
    run directory under a valid-looking completion document.
    """

    rows = [
        row for row in completion.get("completed_seeds", [])
        if isinstance(row, dict) and row.get("seed") == seed
    ]
    if len(rows) != 1:
        raise ValueError(f"shared_zero4 completion has {len(rows)} rows for seed {seed}")
    row = rows[0]
    required = {
        "seed": seed,
        "status": "completed",
        "run_dir": str(run_dir),
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": sha256_file(metadata_path),
        "terminal_checkpoint": str(terminal_checkpoint),
        "terminal_checkpoint_sha256": sha256_file(terminal_checkpoint),
    }
    for key, value in required.items():
        if row.get(key) != value:
            raise ValueError(f"shared_zero4 completion row binding drift: {key}")
    _require_sealed_regular_file(metadata_path, label="live run metadata")
    _require_sealed_regular_file(terminal_checkpoint, label="live terminal checkpoint")
    status_path = Path(str(row.get("source_status_path", ""))).expanduser().resolve()
    closure_path = Path(str(row.get("closure_manifest_path", ""))).expanduser().resolve()
    nonce_path = Path(str(row.get("authorization_nonce_claim_path", ""))).expanduser().resolve()
    for label, path, expected_sha in (
        ("source status", status_path, row.get("source_status_sha256")),
        ("closure manifest", closure_path, row.get("closure_manifest_sha256")),
        ("authorization nonce claim", nonce_path, row.get("authorization_nonce_claim_sha256")),
    ):
        _require_sealed_regular_file(path, label=label)
        if expected_sha != sha256_file(path):
            raise ValueError(f"shared_zero4 {label} completion binding drift")
    status = load_json(status_path)
    if (
        status.get("status"),
        status.get("seed"),
        status.get("checkpoint_dir"),
        status.get("terminal_checkpoint"),
        status.get("terminal_checkpoint_sha256"),
        status.get("run_metadata_sha256"),
        status.get("closure_manifest"),
        status.get("closure_manifest_sha256"),
        status.get("authorization_nonce_claim"),
        status.get("authorization_nonce_claim_sha256"),
        status.get("authorization_id"),
        status.get("authorization_single_use_nonce"),
        status.get("observed_host_id"),
        status.get("physical_gpu_uuid"),
        status.get("development_scoring_invoked"),
    ) != (
        "completed_source_only_score_blind",
        seed,
        str(run_dir),
        str(terminal_checkpoint),
        sha256_file(terminal_checkpoint),
        sha256_file(metadata_path),
        str(closure_path),
        sha256_file(closure_path),
        str(nonce_path),
        sha256_file(nonce_path),
        row.get("authorization_id"),
        row.get("authorization_single_use_nonce"),
        row.get("observed_host_id"),
        row.get("physical_gpu_uuid"),
        False,
    ):
        raise ValueError("shared_zero4 source-status binding drift")
    closure = load_json(closure_path)
    if (
        closure.get("status"),
        closure.get("seed"),
        closure.get("terminal_checkpoint"),
        closure.get("terminal_checkpoint_sha256"),
        closure.get("development_scoring_invoked"),
    ) != (
        "completed_source_only_score_blind",
        seed,
        str(terminal_checkpoint),
        sha256_file(terminal_checkpoint),
        False,
    ):
        raise ValueError("shared_zero4 closure binding drift")
    closure_dir = _require_sealed_directory(closure_path.parent, label="closure directory")
    closure_files = closure.get("files")
    if not isinstance(closure_files, list):
        raise ValueError("shared_zero4 closure file list is malformed")
    expected_closure_files = {
        "run_metadata.json": sha256_file(metadata_path),
        "initial_state_digest.json": sha256_file(run_dir / "initial_state_digest.json"),
        "post_run_cost_receipt.json": sha256_file(run_dir / "post_run_cost_receipt.json"),
    }
    closure_by_name = {
        Path(str(item.get("copy", ""))).name: item
        for item in closure_files
        if isinstance(item, dict)
    }
    if set(closure_by_name) != set(expected_closure_files):
        raise ValueError("shared_zero4 closure file set drift")
    for filename, expected_sha in expected_closure_files.items():
        copied = closure_dir / filename
        _require_sealed_regular_file(copied, label=f"closure/{filename}")
        source_row = closure_by_name[filename]
        if (
            source_row.get("sha256") != expected_sha
            or sha256_file(copied) != expected_sha
            or str(source_row.get("copy")) != str(copied)
        ):
            raise ValueError(f"shared_zero4 closure file binding drift: {filename}")
    nonce = load_json(nonce_path)
    nonce_identity = (
        nonce.get("status"),
        nonce.get("seed"),
        nonce.get("host_id"),
        nonce.get("physical_gpu_uuid"),
        nonce.get("authorization_id"),
        nonce.get("single_use_nonce"),
        row.get("observed_host_id"),
        row.get("physical_gpu_uuid"),
        row.get("authorization_id"),
        row.get("authorization_single_use_nonce"),
    )
    if nonce_identity != (
        "claimed_before_started_status",
        seed,
        row.get("observed_host_id"),
        row.get("physical_gpu_uuid"),
        row.get("authorization_id"),
        row.get("authorization_single_use_nonce"),
        row.get("observed_host_id"),
        row.get("physical_gpu_uuid"),
        row.get("authorization_id"),
        row.get("authorization_single_use_nonce"),
    ):
        raise ValueError("shared_zero4 nonce identity binding drift")


def _first_mode_label_access_receipt(record: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    """Disclose the non-causal direction-key receipt in the shared scorer.

    ``evaluate_session_configs(..., [("first", 30)], ...)`` selects the
    chronological indices before it constructs the optional ``direction_keys``
    audit field.  That field is useful provenance, but it must not be
    misrepresented as descriptor fitting, calibration selection, or model
    prediction input.
    """

    selected = selection.get("usable_trial_list_indices")
    if selected != list(range(30)):
        raise ValueError("shared_zero4 first-mode selection ceased to be chronological")
    trials = record.get("trials")
    if not isinstance(trials, list):
        raise ValueError("shared_zero4 evaluator record lacks trial list")
    return {
        "direction_labels_present_in_owner_record": any(
            trial.get("target_dir") is not None or trial.get("target_id") is not None
            for trial in trials
            if isinstance(trial, Mapping)
        ),
        "direction_labels_used_for_noncausal_selection_receipt": True,
        "direction_labels_used_for_calibration_selection": False,
        "direction_labels_used_for_prediction": False,
        "chronological_selection_rule": "first 30 usable rewarded trials by record order",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--signal-view", choices=["sua", "pseudo_mua"], required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--matrix-completion-receipt", type=Path, required=True)
    parser.add_argument("--train-val-manifest", type=Path, required=True)
    parser.add_argument("--calibration-n", type=int, default=30)
    parser.add_argument("--query-start-trial", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.calibration_n, args.query_start_trial, args.pool_size) != (30, 50, 50):
        raise ValueError("shared_zero4 freezes development activity=30, query_start=50, pool=50")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    run_dir = args.run_dir.expanduser().resolve()
    output = args.out_path.expanduser().resolve()
    # Preserve the raw CLI path for the sealed-receipt symlink check.
    completion_path = args.matrix_completion_receipt.expanduser()
    manifest = args.train_val_manifest.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"shared_zero4 evaluator refuses to overwrite {output}")
    metadata = _load_completed_metadata(run_dir)
    program_receipt_path = Path(metadata["program_receipt"]).expanduser().resolve()
    program_receipt = load_json(program_receipt_path)
    completion, completion_binding = _load_supported_completion_receipt(
        completion_path,
        program_receipt_sha256=sha256_file(program_receipt_path),
    )
    completion_path = completion_path.resolve(strict=True)
    if (
        sha256_file(manifest) != EXPECTED_STRICT_MANIFEST_SHA256
        or sha256_file(manifest) != metadata.get("train_val_manifest_sha256")
        or program_receipt.get("strict_loader_manifest", {}).get("sha256") != EXPECTED_STRICT_MANIFEST_SHA256
    ):
        raise ValueError("shared_zero4 train/validation manifest SHA drifted")
    train_files, val_files, test_names = load_frozen_train_val_manifest(
        manifest, Path(metadata["data_dir"])
    )
    if len(train_files) != 27 or len(val_files) != 6 or len(test_names) != 6:
        raise ValueError("shared_zero4 strict manifest no longer describes 27/6/6")
    # The frozen loader exposes formal identifiers only.  Never concatenate a
    # test filename or resolve an NWB path from this name list.
    if test_names != list(SEALED_FORMAL_TEST_NAMES) or program_receipt.get("sealed_formal_test_names") != list(SEALED_FORMAL_TEST_NAMES):
        raise ValueError("shared_zero4 sealed formal test-name hard pin drift")
    expected_splits = {
        "train": [session_name_from_path(path) for path in train_files],
        "val": [session_name_from_path(path) for path in val_files],
        "test": list(test_names),
    }
    if metadata.get("session_splits") != expected_splits:
        raise ValueError("shared_zero4 metadata/strict-manifest session split drift")
    view_config = metadata["view_configs"][args.signal_view]
    cache_dir = Path(view_config["cache_dir"]).expanduser().resolve()
    behavior_mean, behavior_std = fit_behavior_stats(train_files, 20, cache_dir=cache_dir)
    terminal_checkpoint = run_dir / "epoch_ckpts" / f"epoch_{CHECKPOINT_EPOCH_INDEX:03d}.ckpt"
    if not terminal_checkpoint.is_file():
        raise FileNotFoundError(f"missing fixed terminal checkpoint: {terminal_checkpoint}")
    if metadata.get("terminal_checkpoint_sha256") != sha256_file(terminal_checkpoint):
        raise ValueError("shared_zero4 terminal checkpoint SHA drift")
    if completion_binding["kind"] in {
        "v3_external_v7_adapter", "direct_recovery_content_bound_v1"
    }:
        completion_bridge.require_supported_run_binding(
            completion_binding,
            run_dir=run_dir,
            metadata_path=run_dir / "run_metadata.json",
            terminal_checkpoint=terminal_checkpoint,
            seed=int(metadata["seed"]),
        )
    else:
        _require_exact_completion_binding(
            completion,
            run_dir=run_dir,
            metadata_path=run_dir / "run_metadata.json",
            terminal_checkpoint=terminal_checkpoint,
            seed=int(metadata["seed"]),
        )
    teacher = Path(metadata["teacher_checkpoint"]).expanduser().resolve()
    if sha256_file(teacher) != metadata.get("teacher_sha256"):
        raise ValueError("shared_zero4 teacher SHA drift")
    device = torch.device(args.device)
    model = load_frozen_model(terminal_checkpoint, teacher, "B3S", device)
    per_session: dict[str, float] = {}
    selections: dict[str, Any] = {}
    zero4_rows: dict[str, Any] = {}
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
            # This call does not receive path/trials/direction/rate/normalizer.
            record = attach_standardized_zero4_to_evaluation_record(record)
            session_result, session_selection = evaluate_session_configs(
                record,
                [("first", args.calibration_n)],
                args.pool_size,
                model,
                device,
            )
            name = "gradient_free_calibrated_first_n30"
            per_session[record["name"]] = float(session_result[name])
            selection = dict(session_selection[name])
            selection["label_access"] = _first_mode_label_access_receipt(record, selection)
            selections[record["name"]] = selection
            zero4_rows[record["name"]] = record["zero4_descriptor_receipt"]
    score = compute_fixed_terminal_score(
        {PROTOCOL_EPOCH_NUMBER: sum(per_session.values()) / len(per_session)}
    )
    payload = {
        "schema_version": 1,
        "purpose": "shared_zero4_terminal_fixed_development_evaluation",
        "generated_by": "eval_paired_view_c1_shared_zero4_terminal.py",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "run_metadata_sha256": sha256_file(run_dir / "run_metadata.json"),
        "matrix_completion_receipt": str(completion_path),
        "matrix_completion_receipt_sha256": sha256_file(completion_path),
        "matrix_completion_kind": completion_binding["kind"],
        "matrix_completion_schema": completion_binding["schema"],
        "matrix_completion_status": completion_binding["status"],
        "variant": "B3S",
        "seed": metadata["seed"],
        "task": "CO",
        "signal_view": args.signal_view,
        "shared_weights": True,
        "side_feature_group": "shared_zero4_direct_standardized",
        "checkpoint": str(terminal_checkpoint),
        "checkpoint_sha256": sha256_file(terminal_checkpoint),
        "checkpoint_selection_rule": "fixed_terminal_epoch_011_no_selection",
        "checkpoint_epoch_index": CHECKPOINT_EPOCH_INDEX,
        "protocol_epoch_number": PROTOCOL_EPOCH_NUMBER,
        "teacher_ckpt": str(teacher),
        "teacher_ckpt_sha256": sha256_file(teacher),
        "train_val_manifest": str(manifest),
        "train_val_manifest_sha256": sha256_file(manifest),
        "split_counts": [27, 6, 6],
        "session_splits": {
            "train": [session_name_from_path(path) for path in train_files],
            "val": [session_name_from_path(path) for path in val_files],
            "test": test_names,
        },
        "session_unit_counts": {
            session_name_from_path(path): nwb_unit_count(path) for path in train_files + val_files
        },
        "protocol": {
            "source_activity_calibration_n": 10,
            "development_activity_calibration_n": 30,
            "descriptor_coordinate": STANDARDIZED_COORDINATE_NAME,
            "descriptor_construction": DIRECT_ZERO_CONSTRUCTION,
            "descriptor_label_pool_n": None,
            "query_start_trial": 50,
            "trials_30_49_enter_zero4_identity_or_descriptor": False,
        },
        "descriptor_access": {
            "target_direction_label_reads_for_descriptor": 0,
            "t4_trial_rate_reads_for_descriptor": 0,
            "label_access_scope": "descriptor_only",
            "target_t4_rate_fit_calls": 0,
            "raw_t4_constructed": False,
            "source_t4_normalizer_arithmetic_performed": False,
            "all_development_records_bitwise_float32_zero": all(
                row["bitwise_float32_zero"] is True for row in zero4_rows.values()
            ),
        },
        "per_session_r2": per_session,
        "mean_r2": sum(per_session.values()) / len(per_session),
        "variant_score": score,
        "trial_selections": selections,
        "direction_label_access": {
            "direction_labels_present_in_owner_record": True,
            "direction_labels_used_for_noncausal_selection_receipt": True,
            "direction_labels_used_for_calibration_selection": False,
            "direction_labels_used_for_prediction": False,
            "scope": "generic first-mode direction_keys receipt only; descriptor remains direct Z4",
        },
        "zero4_records": zero4_rows,
        "uses_backward_gradients": False,
        "development_uses_backward_gradients": False,
        "no_test_files_evaluated": True,
        "formal_sua_files_opened": False,
        "subm_nwb_files_opened": False,
    }
    write_json_exclusive(output, payload)
    print(json.dumps({"out": str(output), "score": payload["variant_score"]}, sort_keys=True))


if __name__ == "__main__":
    main()
