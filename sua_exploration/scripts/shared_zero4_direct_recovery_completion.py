#!/usr/bin/env python3
"""Write and verify a content-bound shared-Z4 direct-recovery completion.

This receipt is deliberately not an authorization repair.  Seeds 42 and 43
remain the already sealed V2 terminal chains.  Seed 44 is accepted only as a
fresh, complete, score-blind twelve-epoch trainer run with the same program,
teacher, manifests, and frozen training contract.  Both predecessor V2
completed receipts record pre-seal modes (terminal 0664 and run metadata 0644)
while the same live files are sealed 0444 with equal path/size/SHA.  Those V2
metadata rows caused a later V3 verifier rejection, but they are not V3
receipts and contribute no execution authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration/scripts"))

terminal_bridge = sys.modules.get(
    "sua_exploration.scripts.shared_zero4_terminal_completion_bridge"
) or sys.modules.get("shared_zero4_terminal_completion_bridge")
if terminal_bridge is None:
    import shared_zero4_terminal_completion_bridge as terminal_bridge


SCHEMA = "t4_paired_view_c1_shared_zero4_direct_recovery_content_bound_v1"
STATUS = "completed_fixed_seed_terminal_content_bound_score_blind"
KIND = "direct_recovery_content_bound_v1"
FIXED_SEEDS = (42, 43, 44)
SEALED_MODE = 0o444
EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"


class DirectRecoveryError(ValueError):
    """The direct-recovery content closure is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_file(path: Path, *, role: str) -> Path:
    if path.is_symlink():
        raise DirectRecoveryError(f"{role} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DirectRecoveryError(f"{role} is missing: {path}") from exc
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or resolved.is_symlink():
        raise DirectRecoveryError(f"{role} is not a regular file: {resolved}")
    return resolved


def _sealed_file(path: Path, *, role: str) -> Path:
    resolved = _regular_file(path, role=role)
    if stat.S_IMODE(resolved.stat().st_mode) != SEALED_MODE:
        raise DirectRecoveryError(f"{role} mode must be 0444: {resolved}")
    return resolved


def _load_json(path: Path, *, role: str) -> tuple[dict[str, Any], Path]:
    resolved = _sealed_file(path, role=role)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectRecoveryError(f"{role} is not valid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise DirectRecoveryError(f"{role} must contain a JSON object")
    return value, resolved


def file_metadata(path: Path, *, role: str) -> dict[str, Any]:
    resolved = _sealed_file(path, role=role)
    return {
        "canonical_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "mode": "0444",
    }


def _metadata_path(value: object, *, role: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "canonical_path", "size_bytes", "sha256", "mode"
    }:
        raise DirectRecoveryError(f"{role} metadata shape drift")
    raw = value.get("canonical_path")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise DirectRecoveryError(f"{role} canonical path must be absolute")
    path = _sealed_file(Path(raw), role=role)
    if dict(value) != file_metadata(path, role=role):
        raise DirectRecoveryError(f"{role} path/size/SHA/mode binding drift")
    return path


def _path_and_sha(metadata: Mapping[str, Any], path_key: str, sha_key: str, *, role: str) -> dict[str, Any]:
    raw = metadata.get(path_key)
    claimed = metadata.get(sha_key)
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise DirectRecoveryError(f"{role} path missing")
    path = _regular_file(Path(raw), role=role)
    observed = sha256_file(path)
    if claimed != observed:
        raise DirectRecoveryError(f"{role} SHA drift")
    return {
        "canonical_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": observed,
        "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
    }


def _training_contract(metadata: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    training = metadata.get("training")
    boundary = metadata.get("no_heldout_backprop_contract")
    if not isinstance(training, Mapping) or not isinstance(boundary, Mapping):
        raise DirectRecoveryError(f"seed{seed} training/boundary contract missing")
    identity = (
        metadata.get("schema_version"), metadata.get("status"), metadata.get("seed"),
        metadata.get("experiment"), metadata.get("training_kind"), metadata.get("variant"),
        metadata.get("task"), metadata.get("signal_view"), metadata.get("split_counts"),
        metadata.get("max_units_exclusive"), metadata.get("held_out_test_evaluated"),
        metadata.get("formal_sua_files_opened"), metadata.get("subm_nwb_files_opened"),
        training.get("max_epochs"), training.get("no_early_stopping"),
        training.get("checkpoint_every_epoch"), training.get("terminal_checkpoint"),
        training.get("checkpoint_selection"), training.get("learning_rate"),
        training.get("batch_size"), training.get("source_activity_calibration_n_trials"),
        training.get("future_development_activity_calibration_n_trials"),
        training.get("future_development_query_start_trial"),
        training.get("future_development_trials_30_49_enter_zero4_identity_or_descriptor"),
        training.get("loss_mode"), training.get("freeze_decoder"),
        training.get("shared_optimizer_steps"), training.get("random_calibration"),
        training.get("development_score_invoked"), training.get("development_score_artifact_paths"),
    )
    expected = (
        2, "completed", seed, "paired_view_c1_shared_zero4_source",
        "shared_paired_view_direct_standardized_zero4", "B3S", "CO",
        "paired_sua_pseudo_mua", [27, 6, 6], 100, False, False, False,
        12, True, True, "epoch_011.ckpt", "fixed_terminal_epoch_011_no_selection",
        1e-4, 32, 10, 30, 50, False, "task_only", False, True, False, False, [],
    )
    if identity != expected:
        raise DirectRecoveryError(f"seed{seed} frozen training configuration drift")
    if (
        boundary.get("source_train_sessions"), boundary.get("development_heldout_sessions"),
        boundary.get("formal_test_sessions"), boundary.get("optimizer_and_backward_scope"),
        boundary.get("development_enters_train_dataloader"), boundary.get("development_enters_loss"),
        boundary.get("development_enters_optimizer"), boundary.get("development_uses_backward_gradients"),
        boundary.get("development_scoring_invoked_by_this_run"), boundary.get("formal_paths_resolved"),
        boundary.get("formal_files_opened"),
    ) != (27, 6, 6, "source_train_27_only", False, False, False, False, False, False, False):
        raise DirectRecoveryError(f"seed{seed} held-out data boundary drift")
    materialized = metadata.get("materialized_split_scope")
    session_files = metadata.get("session_files")
    if not isinstance(materialized, Mapping) or (
        materialized.get("formal_paths_resolved"),
        materialized.get("subm_nwb_files_opened"),
    ) != (False, False):
        raise DirectRecoveryError(f"seed{seed} materialized formal/sub-M boundary drift")
    if not isinstance(session_files, Mapping) or session_files.get("test") != []:
        raise DirectRecoveryError(f"seed{seed} formal session-file boundary drift")
    views = metadata.get("view_configs")
    if not isinstance(views, Mapping) or set(views) != {"sua", "pseudo_mua"}:
        raise DirectRecoveryError(f"seed{seed} paired-view configuration drift")
    for view in ("sua", "pseudo_mua"):
        side = views[view].get("side_features") if isinstance(views[view], Mapping) else None
        if not isinstance(side, Mapping) or (
            side.get("group"), side.get("side_dim"), side.get("coordinate"),
            side.get("construction"), side.get("target_direction_label_reads_for_descriptor"),
            side.get("t4_trial_rate_reads_for_descriptor"), side.get("target_t4_rate_fit_calls"),
            side.get("raw_t4_constructed"), side.get("source_t4_normalizer_arithmetic_performed"),
        ) != (
            "shared_zero4_direct_standardized", 4, "source_only_t4_standardized_coordinate",
            "np.zeros((N,4), dtype=np.float32) directly in standardized coordinate; "
            "no raw T4 descriptor and no normalizer arithmetic",
            0, 0, 0, False, False,
        ):
            raise DirectRecoveryError(f"seed{seed}/{view} zero4 descriptor boundary drift")
    return {
        "schema_version": 2,
        "experiment": "paired_view_c1_shared_zero4_source",
        "training_kind": "shared_paired_view_direct_standardized_zero4",
        "variant": "B3S",
        "task": "CO",
        "signal_view": "paired_sua_pseudo_mua",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "max_epochs": 12,
        "terminal_checkpoint": "epoch_011.ckpt",
        "checkpoint_selection": "fixed_terminal_epoch_011_no_selection",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "source_activity_calibration_n_trials": 10,
        "future_development_activity_calibration_n_trials": 30,
        "future_development_query_start_trial": 50,
        "loss_mode": "task_only",
        "freeze_decoder": False,
        "shared_optimizer_steps": True,
        "random_calibration": False,
    }


def _verify_metadata_and_files(
    run_dir: Path, *, seed: int, require_all_epochs: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if run_dir.is_symlink():
        raise DirectRecoveryError(f"seed{seed} run directory must not be a symlink")
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir():
        raise DirectRecoveryError(f"seed{seed} run path is not a directory")
    metadata, metadata_path = _load_json(run_dir / "run_metadata.json", role=f"seed{seed} run metadata")
    contract = _training_contract(metadata, seed=seed)
    if Path(str(metadata.get("output_dir", ""))).resolve() != run_dir:
        raise DirectRecoveryError(f"seed{seed} output directory binding drift")
    sources = {
        "teacher": _path_and_sha(metadata, "teacher_checkpoint", "teacher_sha256", role=f"seed{seed} teacher"),
        "train_val_manifest": _path_and_sha(
            metadata, "train_val_manifest", "train_val_manifest_sha256", role=f"seed{seed} train-val manifest"
        ),
        "data_manifest": _path_and_sha(
            metadata, "data_manifest", "data_manifest_sha256", role=f"seed{seed} data manifest"
        ),
        "program_receipt": _path_and_sha(
            metadata, "program_receipt", "program_receipt_sha256", role=f"seed{seed} program receipt"
        ),
        "data_dir": str(Path(str(metadata.get("data_dir", ""))).resolve(strict=True)),
        "cache_dirs": {
            view: str(
                Path(str(metadata["view_configs"][view].get("cache_dir", ""))).resolve(
                    strict=True
                )
            )
            for view in ("sua", "pseudo_mua")
        },
    }
    if sources["train_val_manifest"]["sha256"] != EXPECTED_MANIFEST_SHA256:
        raise DirectRecoveryError(f"seed{seed} strict train-val manifest SHA drift")
    initial = metadata.get("initial_state")
    if not isinstance(initial, Mapping) or initial.get("matches_prelaunch") is not True:
        raise DirectRecoveryError(f"seed{seed} initial-state contract drift")
    initial_path = _sealed_file(Path(str(initial.get("path", ""))), role=f"seed{seed} initial digest")
    if initial.get("sha256") != sha256_file(initial_path):
        raise DirectRecoveryError(f"seed{seed} initial-state file SHA drift")
    initial_digest = initial.get("digest")
    if not isinstance(initial_digest, Mapping) or not isinstance(initial_digest.get("sha256"), str):
        raise DirectRecoveryError(f"seed{seed} initial-state digest missing")
    program, _ = _load_json(Path(sources["program_receipt"]["canonical_path"]), role=f"seed{seed} program receipt")
    expected_initial = (program.get("initial_state_digests") or {}).get(str(seed))
    if not isinstance(expected_initial, Mapping) or expected_initial.get("sha256") != initial_digest.get("sha256"):
        raise DirectRecoveryError(f"seed{seed} initial-state digest/program binding drift")
    terminal = _sealed_file(run_dir / "epoch_ckpts/epoch_011.ckpt", role=f"seed{seed} terminal checkpoint")
    if metadata.get("terminal_checkpoint") != str(terminal) or metadata.get("terminal_checkpoint_sha256") != sha256_file(terminal):
        raise DirectRecoveryError(f"seed{seed} terminal checkpoint binding drift")
    checkpoint_paths = [run_dir / "epoch_ckpts" / f"epoch_{index:03d}.ckpt" for index in range(12)]
    if require_all_epochs:
        claimed = metadata.get("epoch_checkpoints")
        if claimed != [str(path.resolve()) for path in checkpoint_paths]:
            raise DirectRecoveryError(f"seed{seed} epoch checkpoint list drift")
        for index, path in enumerate(checkpoint_paths):
            _sealed_file(path, role=f"seed{seed} epoch{index} checkpoint")
    cost, cost_path = _load_json(run_dir / "post_run_cost_receipt.json", role=f"seed{seed} post-run cost receipt")
    if (
        cost.get("schema_version"), cost.get("status"), cost.get("run_metadata_path"),
        cost.get("run_metadata_sha256"), cost.get("terminal_checkpoint"),
        cost.get("terminal_checkpoint_sha256"), cost.get("formal_sua_files_opened"),
        cost.get("subm_nwb_files_opened"), cost.get("development_score_invoked"),
    ) != (
        1, "completed", str(metadata_path), sha256_file(metadata_path), str(terminal),
        sha256_file(terminal), False, False, False,
    ):
        raise DirectRecoveryError(f"seed{seed} post-run cost receipt drift")
    return metadata, {
        "run_dir": str(run_dir),
        "run_metadata": file_metadata(metadata_path, role=f"seed{seed} run metadata"),
        "post_run_cost_receipt": file_metadata(cost_path, role=f"seed{seed} post-run cost receipt"),
        "initial_state": file_metadata(initial_path, role=f"seed{seed} initial digest"),
        "terminal_checkpoint": file_metadata(terminal, role=f"seed{seed} terminal checkpoint"),
        "epoch_checkpoints": [
            file_metadata(path, role=f"seed{seed} epoch{index} checkpoint")
            for index, path in enumerate(checkpoint_paths)
        ] if require_all_epochs else None,
        "training_contract_sha256": canonical_json_sha256(contract),
        "source_identity": sources,
    }


def _v2_terminal_from_completed(
    path: Path, *, seed: int, predecessor_incident: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    completed, completed_path = _load_json(path, role=f"seed{seed} V2 completed status")
    if seed not in (42, 43) or (
        completed.get("schema"), completed.get("status"), completed.get("seed"),
        completed.get("formal_or_subm_endpoint_access"),
    ) != (
        "t4_paired_view_c1_shared_zero4_remote_v2_completed_v1",
        "completed_source_only_score_blind", seed, False,
    ):
        raise DirectRecoveryError(f"seed{seed} V2 completed identity drift")
    row = {
        "seed": seed,
        "origin": "V2",
        "completed_status": file_metadata(completed_path, role=f"seed{seed} V2 completed status"),
        "run_metadata": completed.get("run_metadata"),
        "terminal_checkpoint": completed.get("terminal_checkpoint"),
        "closure_manifest": completed.get("closure_manifest"),
        "initial_state": completed.get("initial_state"),
        "authorization_nonce_claim": completed.get("authorization_nonce_claim"),
        "formal_or_subm_endpoint_access": False,
        "score_read_or_evaluated_by_v3": False,
    }
    if predecessor_incident is None:
        try:
            normalized, _program_identity = terminal_bridge._verify_terminal_row(row, seed=seed)
        except Exception as exc:
            raise DirectRecoveryError(f"seed{seed} V2 terminal chain invalid: {exc}") from exc
    else:
        incident_receipt = predecessor_incident.get("incident_evidence_receipt")
        if not isinstance(incident_receipt, Mapping) or incident_receipt.get("canonical_path") != str(completed_path):
            raise DirectRecoveryError(f"seed{seed} completed receipt is not its predecessor incident evidence")
        mode_rows = predecessor_incident.get("preseal_mode_rows")
        if not isinstance(mode_rows, list) or len(mode_rows) != 3:
            raise DirectRecoveryError(f"seed{seed} predecessor must bind exactly three preseal mode rows")
        by_role = {
            (item.get("source_kind"), item.get("role")): item
            for item in mode_rows if isinstance(item, Mapping)
        }
        expected_roles = {
            ("v2_completed_receipt", "run_metadata"),
            ("v2_completed_receipt", "terminal_checkpoint"),
            ("v2_closure_manifest", "terminal_checkpoint"),
        }
        if set(by_role) != expected_roles:
            raise DirectRecoveryError(f"seed{seed} predecessor mode-role set drift")
        live_metadata = by_role[("v2_completed_receipt", "run_metadata")].get("live_metadata")
        live_terminal = by_role[("v2_completed_receipt", "terminal_checkpoint")].get("live_metadata")
        closure_live_terminal = by_role[("v2_closure_manifest", "terminal_checkpoint")].get(
            "live_metadata"
        )
        if not isinstance(live_metadata, Mapping) or not isinstance(live_terminal, Mapping):
            raise DirectRecoveryError(f"seed{seed} predecessor live metadata evidence missing")
        if closure_live_terminal != live_terminal:
            raise DirectRecoveryError(f"seed{seed} closure/completed live terminal identity drift")
        if completed.get("run_metadata") != {**dict(live_metadata), "mode": "0644"}:
            raise DirectRecoveryError(f"seed{seed} completed run metadata is not exact recorded 0644 row")
        if completed.get("terminal_checkpoint") != {**dict(live_terminal), "mode": "0664"}:
            raise DirectRecoveryError(f"seed{seed} completed terminal is not exact recorded 0664 row")
        normalized = dict(row)
        normalized["run_metadata"] = dict(live_metadata)
        normalized["terminal_checkpoint"] = dict(live_terminal)
        # The existing bridge's content verifiers remain authoritative for
        # run metadata and closure.  Only the two completed-receipt mode
        # strings are normalized in memory; the predecessor file is untouched.
        metadata_path = terminal_bridge._metadata_path(
            normalized["run_metadata"], role=f"seed{seed} run metadata"
        )
        terminal = terminal_bridge._metadata_path(
            normalized["terminal_checkpoint"], role=f"seed{seed} terminal checkpoint"
        )
        closure_path = terminal_bridge._metadata_path(
            normalized["closure_manifest"], role=f"seed{seed} closure"
        )
        initial_path = terminal_bridge._metadata_path(
            normalized["initial_state"], role=f"seed{seed} initial state"
        )
        terminal_bridge._metadata_path(
            normalized["authorization_nonce_claim"], role=f"seed{seed} nonce claim"
        )
        for key in ("initial_state", "closure_manifest", "authorization_nonce_claim"):
            if completed.get(key) != normalized[key]:
                raise DirectRecoveryError(f"seed{seed} completed {key} binding drift")
        for key in ("started_status", "real_cuda_smoke"):
            terminal_bridge._metadata_path(completed.get(key), role=f"seed{seed} completed {key}")
        if completed.get("checkpoint_dir") != str(terminal.parent.parent):
            raise DirectRecoveryError(f"seed{seed} completed checkpoint directory drift")
        terminal_bridge._reject_failure_siblings(completed_path=completed_path, seed=seed)
        terminal_bridge._verify_run_metadata(
            normalized, seed=seed, checkpoint=terminal, metadata_path=metadata_path
        )
        terminal_bridge._verify_closure(
            normalized, seed=seed, origin="V2", checkpoint=terminal,
            metadata_path=metadata_path, closure_path=closure_path, initial_path=initial_path,
            permitted_preseal_modes={
                "$.terminal_checkpoint": {
                    "recorded_mode": "0664",
                    "live_mode": "0444",
                    "live_metadata": dict(live_terminal),
                }
            },
        )
    metadata_path = _metadata_path(normalized["run_metadata"], role=f"seed{seed} run metadata")
    metadata, content = _verify_metadata_and_files(metadata_path.parent, seed=seed, require_all_epochs=False)
    return normalized, metadata, content


def _preseal_mode_row(
    source: Mapping[str, Any], *, field: str, recorded_mode: str, seed: int,
    source_kind: str, source_file: Mapping[str, Any],
) -> dict[str, Any]:
    value = source.get(field)
    if not isinstance(value, Mapping) or set(value) != {
        "canonical_path", "size_bytes", "sha256", "mode"
    }:
        raise DirectRecoveryError(f"seed{seed} predecessor {field} metadata shape drift")
    if value.get("mode") != recorded_mode:
        raise DirectRecoveryError(
            f"seed{seed} predecessor {field} recorded mode must be {recorded_mode}"
        )
    raw = value.get("canonical_path")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise DirectRecoveryError(f"seed{seed} predecessor {field} path drift")
    live = file_metadata(Path(raw), role=f"seed{seed} predecessor live {field}")
    if (
        value.get("canonical_path"), value.get("size_bytes"), value.get("sha256")
    ) != (live["canonical_path"], live["size_bytes"], live["sha256"]):
        raise DirectRecoveryError(f"seed{seed} predecessor {field} path/size/SHA drift")
    return {
        "role": field,
        "source_kind": source_kind,
        "source_file": dict(source_file),
        "json_pointer": f"$.{field}",
        "recorded_mode": recorded_mode,
        "live_mode": "0444",
        "path_size_sha_equal": True,
        "live_metadata": live,
    }


def _one_predecessor_incident(receipt_path: Path, *, seed: int) -> dict[str, Any]:
    incident, incident_receipt = _load_json(
        receipt_path, role=f"seed{seed} predecessor V2 completed receipt"
    )
    expected_keys = {
        "schema", "status", "seed", "started_status", "authorization_nonce_claim",
        "real_cuda_smoke", "checkpoint_dir", "terminal_checkpoint", "run_metadata",
        "initial_state", "closure_manifest", "physical_gpu_uuid",
        "formal_or_subm_endpoint_access",
    }
    if set(incident) != expected_keys or (
        incident.get("schema"), incident.get("status"), incident.get("seed")
    ) != (
        "t4_paired_view_c1_shared_zero4_remote_v2_completed_v1",
        "completed_source_only_score_blind", seed,
    ):
        raise DirectRecoveryError(
            f"predecessor mode incident must be the seed{seed} V2 completed receipt"
        )
    completed_metadata = file_metadata(
        incident_receipt, role=f"seed{seed} predecessor V2 completed receipt"
    )
    closure_path = _metadata_path(
        incident.get("closure_manifest"), role=f"seed{seed} predecessor closure manifest"
    )
    closure, _ = _load_json(
        closure_path, role=f"seed{seed} predecessor closure manifest"
    )
    if (
        closure.get("schema"), closure.get("status"), closure.get("seed")
    ) != (
        "t4_paired_view_c1_shared_zero4_remote_v2_closure_v1",
        "completed_source_only_score_blind", seed,
    ):
        raise DirectRecoveryError(f"seed{seed} predecessor closure identity drift")
    closure_metadata = file_metadata(
        closure_path, role=f"seed{seed} predecessor closure manifest"
    )
    return {
        "incident_evidence_receipt": completed_metadata,
        "closure_manifest": closure_metadata,
        "incident_origin_schema": "t4_paired_view_c1_shared_zero4_remote_v2_completed_v1",
        "incident_origin_status": "completed_source_only_score_blind",
        "incident_origin_seed": seed,
        "preseal_mode_rows": [
            _preseal_mode_row(
                incident, field="run_metadata", recorded_mode="0644", seed=seed,
                source_kind="v2_completed_receipt", source_file=completed_metadata,
            ),
            _preseal_mode_row(
                incident, field="terminal_checkpoint", recorded_mode="0664", seed=seed,
                source_kind="v2_completed_receipt", source_file=completed_metadata,
            ),
            _preseal_mode_row(
                closure, field="terminal_checkpoint", recorded_mode="0664", seed=seed,
                source_kind="v2_closure_manifest", source_file=closure_metadata,
            ),
        ],
    }


def _predecessor_incidents(
    *, seed42_receipt: Path, seed43_receipt: Path, seed42_checkpoint: Path,
    direct_terminal: Path,
) -> dict[str, Any]:
    incidents = {
        "42": _one_predecessor_incident(seed42_receipt, seed=42),
        "43": _one_predecessor_incident(seed43_receipt, seed=43),
    }
    seed42_live_terminal = incidents["42"]["preseal_mode_rows"][1]["live_metadata"]
    supplied_seed42 = file_metadata(
        seed42_checkpoint, role="supplied seed42 predecessor terminal checkpoint"
    )
    if supplied_seed42 != seed42_live_terminal:
        raise DirectRecoveryError("supplied mode-incident checkpoint is not seed42 live terminal")
    for seed in ("42", "43"):
        terminal = incidents[seed]["preseal_mode_rows"][1]["live_metadata"]
        if Path(terminal["canonical_path"]) == direct_terminal.resolve(strict=True):
            raise DirectRecoveryError(
                f"fresh direct seed44 terminal must not reuse predecessor seed{seed} checkpoint"
            )
    return {
        "classification": "predecessor_v2_completed_preseal_mode_metadata_incidents",
        "fixed_seed_order": [42, 43],
        "incidents_by_seed": incidents,
        "normalized_in_memory_only": True,
        "predecessor_files_rewritten": False,
        "unexpected_additional_mode_drift_allowed": False,
        "later_v3_verifier_rejected_these_mode_mismatches": True,
        "incident_evidence_is_a_v3_receipt": False,
        "incident_used_as_authorization": False,
        "incident_checkpoint_used_for_seed44_direct_recovery": False,
    }


def _trainer_exit_observation(trainer_pid: int) -> dict[str, Any]:
    if not isinstance(trainer_pid, int) or trainer_pid <= 1:
        raise DirectRecoveryError("seed44 trainer PID must be an integer greater than one")
    proc_path = Path("/proc") / str(trainer_pid)
    if proc_path.exists() or proc_path.is_symlink():
        raise DirectRecoveryError(
            f"seed44 trainer PID {trainer_pid} is still live; content verification is forbidden"
        )
    return {
        "trainer_pid": trainer_pid,
        "proc_path_checked": str(proc_path),
        "trainer_pid_absent_before_any_terminal_content_verification": True,
        "caller_presealed_all_terminal_content": True,
        "writer_source_chmod_calls": 0,
        "writer_source_mutations": 0,
        "required_source_file_mode_before_verification": "0444",
        "pid_absence_is_exit_observation_not_authorization_evidence": True,
    }


def build_payload(
    *, seed42_completed: Path, seed43_completed: Path, seed44_run_dir: Path,
    mode_incident_evidence_receipt: Path, mode_incident_checkpoint: Path,
    seed44_trainer_pid: int,
) -> dict[str, Any]:
    # This must happen before any terminal content is opened or hashed.  The
    # writer never chmods a source artifact; the caller must seal the complete
    # run only after the trainer process has exited.
    trainer_exit = _trainer_exit_observation(seed44_trainer_pid)
    # The incident receipt is the same predecessor seed42 V2 completed
    # receipt.  Establish this honest origin before using the narrow in-memory
    # terminal-mode normalization in the V2 content verifier.
    if seed42_completed.resolve(strict=True) != mode_incident_evidence_receipt.resolve(strict=True):
        raise DirectRecoveryError(
            "seed42 completed and mode-incident evidence receipt must be the same file"
        )
    incident = _predecessor_incidents(
        seed42_receipt=mode_incident_evidence_receipt,
        seed43_receipt=seed43_completed,
        seed42_checkpoint=mode_incident_checkpoint,
        direct_terminal=seed44_run_dir.resolve(strict=True) / "epoch_ckpts/epoch_011.ckpt",
    )
    rows: dict[str, Any] = {}
    metadata_by_seed: dict[int, dict[str, Any]] = {}
    content_by_seed: dict[str, Any] = {}
    for seed, path in ((42, seed42_completed), (43, seed43_completed)):
        try:
            row, metadata, content = _v2_terminal_from_completed(
                path,
                seed=seed,
                predecessor_incident=incident["incidents_by_seed"][str(seed)],
            )
        except terminal_bridge.TerminalAdapterError as exc:
            raise DirectRecoveryError(f"seed{seed} V2 terminal chain invalid: {exc}") from exc
        rows[str(seed)] = row
        metadata_by_seed[seed] = metadata
        content_by_seed[str(seed)] = content
    metadata44, content44 = _verify_metadata_and_files(
        seed44_run_dir, seed=44, require_all_epochs=True
    )
    metadata_by_seed[44] = metadata44
    content_by_seed["44"] = content44
    rows["44"] = {
        "seed": 44,
        "origin": "direct_recovery_fresh_run",
        "run_metadata": content44["run_metadata"],
        "terminal_checkpoint": content44["terminal_checkpoint"],
        "post_run_cost_receipt": content44["post_run_cost_receipt"],
        "initial_state": content44["initial_state"],
        "epoch_checkpoints": content44["epoch_checkpoints"],
        "formal_or_subm_endpoint_access": False,
        "score_read_or_evaluated": False,
    }
    common_contracts = {content_by_seed[str(seed)]["training_contract_sha256"] for seed in FIXED_SEEDS}
    common_sources = [content_by_seed[str(seed)]["source_identity"] for seed in FIXED_SEEDS]
    if len(common_contracts) != 1 or any(item != common_sources[0] for item in common_sources[1:]):
        raise DirectRecoveryError("three seeds do not share one training/source content identity")
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "status": STATUS,
        "fixed_seed_order": list(FIXED_SEEDS),
        "terminal_rows": rows,
        "content_by_seed": content_by_seed,
        "common_training_contract_sha256": next(iter(common_contracts)),
        "common_source_identity": common_sources[0],
        "seed44_trainer_exit_observation": trainer_exit,
        "predecessor_v2_completed_preseal_mode_metadata_incidents": incident,
        "recovery_statement": {
            "seed42_seed43_origin": "existing_verified_v2_terminal_and_closure",
            "seed44_origin": "fresh_direct_complete_twelve_epoch_run",
            "v3_authorization_claimed_or_repaired": False,
            "old_v3_authorization_chain_used_for_seed44_direct_run": False,
            "run_metadata_authorization_binding_flag_treated_as_authority": False,
            "scientific_data_boundary_changed": False,
            "score_or_metric_read_during_completion": False,
        },
        "formal_or_subm_endpoint_access": False,
        "development_score_invoked": False,
        "score_read_or_evaluated": False,
    }


def write_payload_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, SEALED_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, SEALED_MODE)
    return path.resolve(strict=True)


def verify_receipt(path: Path) -> dict[str, Any]:
    payload, receipt = _load_json(path, role="direct-recovery completion receipt")
    expected_keys = {
        "schema", "kind", "status", "fixed_seed_order", "terminal_rows", "content_by_seed",
        "common_training_contract_sha256", "common_source_identity", "seed44_trainer_exit_observation",
        "predecessor_v2_completed_preseal_mode_metadata_incidents",
        "recovery_statement", "formal_or_subm_endpoint_access", "development_score_invoked",
        "score_read_or_evaluated",
    }
    if set(payload) != expected_keys or (
        payload.get("schema"), payload.get("kind"), payload.get("status"),
        payload.get("fixed_seed_order"), payload.get("formal_or_subm_endpoint_access"),
        payload.get("development_score_invoked"), payload.get("score_read_or_evaluated"),
    ) != (SCHEMA, KIND, STATUS, list(FIXED_SEEDS), False, False, False):
        raise DirectRecoveryError("direct-recovery receipt identity/scope drift")
    rows = payload.get("terminal_rows")
    content = payload.get("content_by_seed")
    if not isinstance(rows, Mapping) or set(rows) != {"42", "43", "44"}:
        raise DirectRecoveryError("direct-recovery terminal seed set drift")
    if not isinstance(content, Mapping) or set(content) != {"42", "43", "44"}:
        raise DirectRecoveryError("direct-recovery content seed set drift")
    row44_probe = rows.get("44")
    if not isinstance(row44_probe, Mapping):
        raise DirectRecoveryError("seed44 direct terminal row missing")
    direct_terminal_probe = _metadata_path(
        row44_probe.get("terminal_checkpoint"), role="seed44 terminal checkpoint"
    )
    incident = payload.get("predecessor_v2_completed_preseal_mode_metadata_incidents")
    if not isinstance(incident, Mapping):
        raise DirectRecoveryError("predecessor V2 preseal mode incidents missing")
    incidents_by_seed = incident.get("incidents_by_seed")
    if not isinstance(incidents_by_seed, Mapping) or set(incidents_by_seed) != {"42", "43"}:
        raise DirectRecoveryError("predecessor V2 incident seed set drift")
    incident_receipts = {
        seed: _metadata_path(
            incidents_by_seed[str(seed)].get("incident_evidence_receipt"),
            role=f"seed{seed} predecessor V2 completed receipt",
        )
        for seed in (42, 43)
    }
    seed42_rows = incidents_by_seed["42"].get("preseal_mode_rows")
    if not isinstance(seed42_rows, list) or len(seed42_rows) != 3:
        raise DirectRecoveryError("seed42 predecessor preseal mode rows drift")
    seed42_checkpoint = _metadata_path(
        seed42_rows[1].get("live_metadata") if isinstance(seed42_rows[1], Mapping) else None,
        role="seed42 predecessor live terminal checkpoint",
    )
    rebuilt_incident = _predecessor_incidents(
        seed42_receipt=incident_receipts[42],
        seed43_receipt=incident_receipts[43],
        seed42_checkpoint=seed42_checkpoint,
        direct_terminal=direct_terminal_probe,
    )
    if dict(incident) != rebuilt_incident:
        raise DirectRecoveryError("predecessor mode-metadata incident binding drift")
    rebuilt_rows: dict[str, Any] = {}
    rebuilt_content: dict[str, Any] = {}
    common_sources = []
    contracts = set()
    for seed in (42, 43):
        row = rows[str(seed)]
        if not isinstance(row, Mapping):
            raise DirectRecoveryError(f"seed{seed} terminal row missing")
        completed_path = _metadata_path(row.get("completed_status"), role=f"seed{seed} V2 completed status")
        try:
            rebuilt, _metadata, seed_content = _v2_terminal_from_completed(
                completed_path,
                seed=seed,
                predecessor_incident=rebuilt_incident["incidents_by_seed"][str(seed)],
            )
        except terminal_bridge.TerminalAdapterError as exc:
            raise DirectRecoveryError(f"seed{seed} V2 terminal chain invalid: {exc}") from exc
        if dict(row) != rebuilt or content[str(seed)] != seed_content:
            raise DirectRecoveryError(f"seed{seed} receipt/live content binding drift")
        rebuilt_rows[str(seed)] = rebuilt
        rebuilt_content[str(seed)] = seed_content
        contracts.add(seed_content["training_contract_sha256"])
        common_sources.append(seed_content["source_identity"])
    row44 = rows["44"]
    if not isinstance(row44, Mapping) or set(row44) != {
        "seed", "origin", "run_metadata", "terminal_checkpoint", "post_run_cost_receipt",
        "initial_state", "epoch_checkpoints", "formal_or_subm_endpoint_access", "score_read_or_evaluated",
    } or (row44.get("seed"), row44.get("origin"), row44.get("formal_or_subm_endpoint_access"),
          row44.get("score_read_or_evaluated")) != (44, "direct_recovery_fresh_run", False, False):
        raise DirectRecoveryError("seed44 direct terminal row identity drift")
    metadata_path = _metadata_path(row44["run_metadata"], role="seed44 run metadata")
    _metadata44, seed44_content = _verify_metadata_and_files(
        metadata_path.parent, seed=44, require_all_epochs=True
    )
    expected_row44 = {
        "seed": 44, "origin": "direct_recovery_fresh_run",
        "run_metadata": seed44_content["run_metadata"],
        "terminal_checkpoint": seed44_content["terminal_checkpoint"],
        "post_run_cost_receipt": seed44_content["post_run_cost_receipt"],
        "initial_state": seed44_content["initial_state"],
        "epoch_checkpoints": seed44_content["epoch_checkpoints"],
        "formal_or_subm_endpoint_access": False, "score_read_or_evaluated": False,
    }
    if dict(row44) != expected_row44 or content["44"] != seed44_content:
        raise DirectRecoveryError("seed44 receipt/live content binding drift")
    rebuilt_rows["44"] = expected_row44
    rebuilt_content["44"] = seed44_content
    contracts.add(seed44_content["training_contract_sha256"])
    common_sources.append(seed44_content["source_identity"])
    if len(contracts) != 1 or any(item != common_sources[0] for item in common_sources[1:]):
        raise DirectRecoveryError("direct-recovery common source/config identity drift")
    if payload.get("common_training_contract_sha256") != next(iter(contracts)) or payload.get("common_source_identity") != common_sources[0]:
        raise DirectRecoveryError("direct-recovery common identity receipt drift")
    exit_observation = payload.get("seed44_trainer_exit_observation")
    if not isinstance(exit_observation, Mapping) or not isinstance(
        exit_observation.get("trainer_pid"), int
    ) or exit_observation.get("trainer_pid") <= 1 or set(exit_observation) != {
        "trainer_pid", "proc_path_checked",
        "trainer_pid_absent_before_any_terminal_content_verification",
        "caller_presealed_all_terminal_content", "writer_source_chmod_calls",
        "writer_source_mutations", "required_source_file_mode_before_verification",
        "pid_absence_is_exit_observation_not_authorization_evidence",
    } or (
        exit_observation.get("proc_path_checked"),
        exit_observation.get("trainer_pid_absent_before_any_terminal_content_verification"),
        exit_observation.get("caller_presealed_all_terminal_content"),
        exit_observation.get("writer_source_chmod_calls"),
        exit_observation.get("writer_source_mutations"),
        exit_observation.get("required_source_file_mode_before_verification"),
        exit_observation.get("pid_absence_is_exit_observation_not_authorization_evidence"),
    ) != (
        f"/proc/{exit_observation.get('trainer_pid')}", True, True, 0, 0, "0444", True,
    ):
        raise DirectRecoveryError("seed44 trainer-exit observation drift")
    statement = payload.get("recovery_statement")
    if not isinstance(statement, Mapping) or (
        statement.get("seed42_seed43_origin"), statement.get("seed44_origin"),
        statement.get("v3_authorization_claimed_or_repaired"),
        statement.get("old_v3_authorization_chain_used_for_seed44_direct_run"),
        statement.get("run_metadata_authorization_binding_flag_treated_as_authority"),
        statement.get("scientific_data_boundary_changed"),
        statement.get("score_or_metric_read_during_completion"),
    ) != (
        "existing_verified_v2_terminal_and_closure", "fresh_direct_complete_twelve_epoch_run",
        False, False, False, False, False,
    ):
        raise DirectRecoveryError("direct-recovery disclosure statement drift")
    return {
        "kind": KIND,
        "schema": SCHEMA,
        "status": STATUS,
        "path": str(receipt),
        "sha256": sha256_file(receipt),
        "fixed_seeds": list(FIXED_SEEDS),
        "rows": rebuilt_rows,
        "payload": payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42-completed", type=Path, required=True)
    parser.add_argument("--seed43-completed", type=Path, required=True)
    parser.add_argument("--seed44-run-dir", type=Path, required=True)
    parser.add_argument("--mode-incident-evidence-receipt", type=Path, required=True)
    parser.add_argument("--mode-incident-checkpoint", type=Path, required=True)
    parser.add_argument("--seed44-trainer-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(
        seed42_completed=args.seed42_completed,
        seed43_completed=args.seed43_completed,
        seed44_run_dir=args.seed44_run_dir,
        mode_incident_evidence_receipt=args.mode_incident_evidence_receipt,
        mode_incident_checkpoint=args.mode_incident_checkpoint,
        seed44_trainer_pid=args.seed44_trainer_pid,
    )
    receipt = write_payload_exclusive(args.out.expanduser().resolve(), payload)
    verified = verify_receipt(receipt)
    print(json.dumps({
        "status": verified["status"], "kind": verified["kind"],
        "receipt": verified["path"], "receipt_sha256": verified["sha256"],
        "score_read_or_evaluated": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
