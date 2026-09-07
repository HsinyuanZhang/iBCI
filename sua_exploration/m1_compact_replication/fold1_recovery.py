#!/usr/bin/env python3
"""Append-only recovery of the fold-1 B0 terminal and B3S continuation.

The original fold-1 v2 executor failed closed after B0 because it reused the
fold-0 batch cardinality (4951 batches/epoch) as a literal contract.  Fold 1
has a different source-session set and its immutable source-fit equivalence
receipt records 4963 batches/epoch.  This module never edits that receipt or
the failed executor log.  It validates the already-written B0 checkpoint,
derives its expected step count from the bound per-fold sampler receipt, and
then launches only the still-missing B3S-Zero4 arm.

No target, validation, formal, or evaluator path is opened by preparation or
execution.  The independent evaluator may consume the recovered state only
after both arms are terminal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping

import torch
import yaml

from sua_exploration.m1_compact_replication import fold1_runner as base
from sua_exploration.m1_compact_replication import fold1_runner_v2 as v2_runner
from sua_exploration.m1_compact_replication import fold1_v2_contract as v2_contract


ROOT = base.ROOT
V2_RECEIPT = v2_contract.V2_RECEIPT
EQUIVALENCE_RECEIPT = v2_contract.EQUIVALENCE_RECEIPT
EQUIVALENCE_BINDING = v2_contract.EQUIVALENCE_BINDING
RECOVERY_RECEIPT = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F1_S42_B0_TERMINAL_RECOVERY_v1.json"
)
RECOVERED_STATE = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F1_S42_EXECUTION_RECOVERED_v1.json"
)
INCIDENT_LOG = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F1_S42_EXECUTOR_V2_RUN.log"
)
RECOVERY_LOG_DIR = ROOT / "sua_exploration/m1_compact_replication/results/fold1_recovery_logs"
RECOVERY_SCHEMA = "m1_compact_b3s_f1_s42_b0_terminal_recovery_v1"
RECOVERY_STATUS = "PASS_M1_COMPACT_B3S_F1_S42_B0_TERMINAL_RECOVERABLE_PENDING_B3S"
STATE_SCHEMA = "m1_compact_b3s_f1_s42_execution_recovered_v1"
STATE_PASS = "PASS_M1_COMPACT_B3S_F1_S42_RECOVERED_PAIR_TERMINAL"
STATE_FAIL = "FAIL_M1_COMPACT_B3S_F1_S42_RECOVERED_EXECUTION"
EXPECTED_EPOCH = 11
SOURCE_SESSIONS = list(v2_contract.SOURCE_SESSIONS)
TARGET_SESSION = v2_contract.TARGET_SESSION


def need(condition: bool, message: str) -> None:
    if not condition:
        raise base.Fold1ContractError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} is not an object")
    return value


def immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        temporary.replace(path)
        return sha(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_immutable(path: Path, schema: str, status: str, label: str) -> dict[str, Any]:
    value = read(path, label)
    need(path.stat().st_mode & 0o777 == 0o444, f"{label} is mutable")
    need(value.get("schema") == schema and value.get("status") == status, f"{label} schema/status drift")
    need(
        value.get("canonical_content_sha256")
        == canonical({k: v for k, v in value.items() if k != "canonical_content_sha256"}),
        f"{label} canonical hash drift",
    )
    return value


def _staged() -> dict[str, Any]:
    staged = v2_contract._read_immutable(
        V2_RECEIPT, "fold1 v2 staged receipt", v2_contract.RECEIPT_SCHEMA, v2_contract.RECEIPT_STATUS
    )
    v2_runner._validate_staged(staged, V2_RECEIPT)
    return staged


def _equivalence(staged: Mapping[str, Any]) -> dict[str, Any]:
    bound = v2_contract.validate_equivalence_binding(V2_RECEIPT)
    need(bound["path"] == str(EQUIVALENCE_BINDING.resolve()), "equivalence binding path drift")
    equivalence = read_immutable(
        EQUIVALENCE_RECEIPT,
        "m1_compact_b3s_f1_s42_source_fit_equivalence_v1",
        "PASS_M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENT",
        "source-fit equivalence receipt",
    )
    need(
        equivalence.get("parent_v2_receipt")
        == {"path": str(V2_RECEIPT.resolve()), "sha256": sha(V2_RECEIPT)},
        "source-fit equivalence parent receipt drift",
    )
    sampler = equivalence.get("sampler", {})
    need(sampler.get("equal") is True and sampler.get("source_windows") == sampler.get("legacy_windows"), "source sampler equivalence drift")
    return equivalence


def _single_output_dir(run_id: str) -> Path:
    candidates = sorted((ROOT / "outputs/streaming_calibration").glob(f"{run_id}_f1_s42_*"))
    need(len(candidates) == 1 and candidates[0].is_dir(), f"expected one output directory for {run_id}, found {len(candidates)}")
    return candidates[0]


def _resolved_config(run_id: str, arm: str, teacher_path: str | None = None) -> tuple[Path, dict[str, Any]]:
    output = _single_output_dir(run_id)
    path = output / "resolved_config.yaml"
    need(path.is_file() and not path.is_symlink(), f"resolved config missing for {run_id}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"resolved config is not an object: {run_id}")
    data = value.get("data", {})
    trainer = value.get("trainer", {})
    need(value.get("run_id") == run_id, f"resolved run_id drift: {run_id}")
    need(value.get("test") is False and value.get("optimized_metric") is None, f"resolved test/metric policy drift: {run_id}")
    need(data.get("_target_") == v2_contract.SOURCE_ONLY_TARGET, f"resolved source-only class drift: {run_id}")
    need(data.get("loso_fold") == 1 and data.get("source_session_names") == SOURCE_SESSIONS, f"resolved source scope drift: {run_id}")
    need(data.get("afc4_arm") == ("none" if arm == "B0" else "zero4"), f"resolved carrier arm drift: {run_id}")
    need(trainer.get("min_epochs") == 12 and trainer.get("max_epochs") == 12, f"resolved epoch policy drift: {run_id}")
    need(trainer.get("limit_val_batches") == 0 and trainer.get("num_sanity_val_steps") == 0, f"resolved validation policy drift: {run_id}")
    need(int(data.get("batch_size", 0)) > 0, f"resolved batch size missing: {run_id}")
    if teacher_path is not None:
        need(value.get("model", {}).get("teacher_ckpt_path") == teacher_path, f"resolved teacher binding drift: {run_id}")
    return path, value


def derive_sampler_contract(equivalence: Mapping[str, Any], *, batch_size: int, epochs: int) -> dict[str, Any]:
    """Derive terminal steps from the bound per-fold source sampler receipt."""
    sampler = equivalence.get("sampler", {})
    counts = sampler.get("source_batch_counts")
    need(isinstance(counts, dict) and set(counts) == set(SOURCE_SESSIONS), "source sampler session counts drift")
    counts = {str(name): int(value) for name, value in counts.items()}
    need(all(value > 0 for value in counts.values()), "source sampler contains nonpositive count")
    total_batches = int(sum(counts.values()))
    source_windows = int(sampler.get("source_windows", 0))
    legacy_windows = int(sampler.get("legacy_windows", -1))
    need(source_windows > 0 and source_windows == legacy_windows, "source sampler window count drift")
    need(source_windows == total_batches * int(batch_size), "source sampler windows do not equal full batches")
    need(int(batch_size) > 0 and int(epochs) > 0, "invalid batch/epoch cardinality")
    return {
        "source_batch_counts": counts,
        "source_batches_per_epoch": total_batches,
        "batch_size": int(batch_size),
        "source_scored_windows_per_epoch": source_windows,
        "epochs": int(epochs),
        "expected_global_step": total_batches * int(epochs),
        "derivation": "sum(bound source_batch_counts) * resolved trainer.max_epochs; no fold-global literal",
        "sampler_sha256": sampler.get("source_sha256"),
        "legacy_sampler_sha256": sampler.get("legacy_sha256"),
    }


def _find_checkpoint(run_id: str) -> Path:
    candidates = sorted(path for path in (ROOT / "logs").rglob("*.ckpt") if run_id in str(path) and "fixed_last" in str(path))
    need(len(candidates) == 1, f"expected one terminal checkpoint for {run_id}, found {len(candidates)}")
    return candidates[0]


def _source_manifest(run_id: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(path for path in (ROOT / "logs").rglob("m1_version_b_source_only_fit_manifest.json") if run_id in str(path))
    need(len(candidates) == 1, f"expected one source-only manifest for {run_id}, found {len(candidates)}")
    path = candidates[0]
    body = read(path, f"source-only fit manifest {run_id}")
    need(body.get("schema") == "m1_version_b_source_only_fit_v2", f"source-only manifest schema drift: {run_id}")
    need(body.get("source_only") is True and body.get("target_path_resolved_during_fit") is False, f"source-only target policy drift: {run_id}")
    need(body.get("validation_sessions") == [] and body.get("target_query_values_read_by_fit") is False, f"source-only validation drift: {run_id}")
    for forbidden in ("target_file", "target_path", "query_window_audit", "query_sampler_sha256", "query_scored_windows"):
        need(forbidden not in body, f"source-only manifest leaked {forbidden}: {run_id}")
    need(body.get("train_sessions") == SOURCE_SESSIONS, f"source-only source order drift: {run_id}")
    for session in SOURCE_SESSIONS:
        spec = body.get("source_files", {}).get(session, {})
        source = Path(str(spec.get("path", "")))
        need(source.is_file() and sha(source) == spec.get("sha256"), f"source NWB drift: {run_id}/{session}")
    return path, body


def _validate_terminal(
    run_id: str,
    variant: str,
    log_path: Path,
    expected_steps: int,
    teacher_path: str,
) -> dict[str, Any]:
    checkpoint = _find_checkpoint(run_id)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == EXPECTED_EPOCH, f"{variant} terminal epoch drift")
    need(payload.get("global_step") == expected_steps, f"{variant} terminal step drift")
    need(payload.get("pytorch-lightning_version") == "2.6.5", f"{variant} Lightning drift")
    optimizer_states = payload.get("optimizer_states")
    need(isinstance(optimizer_states, list) and len(optimizer_states) == 1, f"{variant} optimizer count drift")
    optimizer = optimizer_states[0]
    groups = optimizer.get("param_groups", [])
    need(len(groups) == 1 and float(groups[0].get("lr")) == 1.0e-4 and float(groups[0].get("weight_decay")) == 0.0, f"{variant} optimizer hyperparameter drift")
    state_steps = set()
    for slot in optimizer.get("state", {}).values():
        step = slot.get("step")
        if hasattr(step, "detach"):
            step = step.detach().cpu().item()
        state_steps.add(int(step))
    need(state_steps == {expected_steps}, f"{variant} optimizer step drift")
    fit = payload.get("loops", {}).get("fit_loop", {})
    need(fit.get("epoch_loop.batch_progress", {}).get("total", {}).get("completed") == expected_steps, f"{variant} fit-loop batch drift")
    need(fit.get("epoch_progress", {}).get("total", {}).get("processed") == 12, f"{variant} fit-loop epoch drift")
    callbacks = payload.get("callbacks", {})
    need(len(callbacks) == 1 and "every_n_epochs': 12" in str(next(iter(callbacks))), f"{variant} callback drift")
    need(log_path.is_file(), f"{variant} training log missing")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    need("Starting training!" in log_text and "max_epochs=12" in log_text, f"{variant} training completion evidence missing")
    manifest_path, manifest = _source_manifest(run_id)
    config_path, config = _resolved_config(run_id, variant, teacher_path)
    resolved_summary = v2_runner._resolved_config_for_run(run_id, variant, teacher_path)
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha(checkpoint),
        "variant": variant,
        "epoch": EXPECTED_EPOCH,
        "global_step": expected_steps,
        "optimizer": {"count": 1, "state_steps": sorted(state_steps), "lr": 1.0e-4, "weight_decay": 0.0},
        "fit_loop": {"batch_completed": expected_steps, "epoch_processed": 12},
        "source_only_manifest": {"path": str(manifest_path.resolve()), "sha256": sha(manifest_path), "body": manifest},
        "resolved_config": resolved_summary,
        "restored_teacher": {"path": teacher_path, "config_sha256": sha(config_path)},
        "log": {"path": str(log_path.resolve()), "sha256": sha(log_path)},
    }


def build_recovery_receipt() -> dict[str, Any]:
    staged = _staged()
    equivalence = _equivalence(staged)
    inventory = v2_contract.source_only_inventory()
    need(inventory == staged["inventory"], "source/data inventory changed before recovery")
    config_path, config = _resolved_config("m1_compact_b0_f1_s42_fresh_e11", "B0")
    sampler = derive_sampler_contract(
        equivalence,
        batch_size=int(config["data"]["batch_size"]),
        epochs=int(config["trainer"]["max_epochs"]),
    )
    need(sampler["expected_global_step"] == 59556, "fold-1 source cardinality did not derive 59556")
    need(INCIDENT_LOG.is_file(), "v2 fail-closed incident log missing")
    incident_text = INCIDENT_LOG.read_text(encoding="utf-8", errors="replace")
    need("B0 terminal epoch/step drift" in incident_text and "EXECUTOR_EXIT=1" in incident_text, "expected fail-closed incident evidence missing")
    b0 = _validate_terminal(
        "m1_compact_b0_f1_s42_fresh_e11",
        "B0",
        ROOT / "sua_exploration/m1_compact_replication/results/fold1_v2_logs/b0.log",
        sampler["expected_global_step"],
        staged["teacher"]["checkpoint"]["path"],
    )
    return {
        "schema": RECOVERY_SCHEMA,
        "status": RECOVERY_STATUS,
        "parent_v2_receipt": {"path": str(V2_RECEIPT.resolve()), "sha256": sha(V2_RECEIPT)},
        "equivalence": {"path": str(EQUIVALENCE_RECEIPT.resolve()), "sha256": sha(EQUIVALENCE_RECEIPT), "binding": v2_contract.validate_equivalence_binding(V2_RECEIPT)},
        "scope": {"task": "m1", "fold": 1, "seed": 42, "source_sessions": SOURCE_SESSIONS, "outer_target": TARGET_SESSION, "target_opened_by_fit": False, "target_backward_steps": 0, "evaluator_started": False},
        "source_inventory": inventory,
        "source_sampler": sampler,
        "b0_terminal": b0,
        "incident": {"path": str(INCIDENT_LOG.resolve()), "sha256": sha(INCIDENT_LOG), "classification": "contract_cardinality_mismatch_before_B3S; preserved_fail_closed"},
        "recovery_policy": {"retrain_b0": False, "launch_only": "b3s_zero4", "target_metrics_read_by_runner": False, "target_opened_by_evaluator_after_pair": True, "checkpoint_unchanged": True},
    }


def _validate_recovery_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    need(body.get("schema") == RECOVERY_SCHEMA and body.get("status") == RECOVERY_STATUS, "recovery receipt schema/status drift")
    need(body.get("canonical_content_sha256") == canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"}), "recovery receipt canonical drift")
    staged = _staged()
    need(body.get("parent_v2_receipt") == {"path": str(V2_RECEIPT.resolve()), "sha256": sha(V2_RECEIPT)}, "recovery parent receipt drift")
    need(body.get("source_inventory") == staged["inventory"] == v2_contract.source_only_inventory(), "recovery source inventory drift")
    equivalence = _equivalence(staged)
    sampler = body.get("source_sampler", {})
    config_path, config = _resolved_config("m1_compact_b0_f1_s42_fresh_e11", "B0")
    expected = derive_sampler_contract(equivalence, batch_size=int(config["data"]["batch_size"]), epochs=int(config["trainer"]["max_epochs"]))
    need(sampler == expected, "recovery sampler derivation drift")
    b0 = _validate_terminal(
        "m1_compact_b0_f1_s42_fresh_e11",
        "B0",
        ROOT / "sua_exploration/m1_compact_replication/results/fold1_v2_logs/b0.log",
        expected["expected_global_step"],
        staged["teacher"]["checkpoint"]["path"],
    )
    need(body.get("b0_terminal", {}).get("sha256") == b0["sha256"], "B0 checkpoint changed after recovery receipt")
    need(body.get("incident", {}).get("sha256") == sha(INCIDENT_LOG), "incident log changed after recovery receipt")
    return staged


def prepare() -> dict[str, Any]:
    body = build_recovery_receipt()
    immutable(RECOVERY_RECEIPT, body)
    return read(RECOVERY_RECEIPT, "recovery receipt")


def execute(receipt_path: Path, state_path: Path, authorization: str | None) -> int:
    need(authorization == "FOLD1_APPROVED_BY_ROOT_RECOVERY", "fold1 recovery requires explicit root recovery token")
    body = read_immutable(RECOVERY_RECEIPT, RECOVERY_SCHEMA, RECOVERY_STATUS, "recovery receipt")
    staged = _validate_recovery_receipt(body)
    need(not state_path.exists() and not state_path.is_symlink(), "recovered execution state already exists")
    need(not base.prior_artifact_exists(ROOT / "outputs/streaming_calibration", "m1_compact_b3s_zero4_f1_s42_fresh_e11"), "prior B3S artifact exists")
    command = list(staged["commands"]["b3s_zero4"])
    RECOVERY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RECOVERY_LOG_DIR / "b3s_zero4.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, stdout=handle, stderr=subprocess.STDOUT, check=False)
    exit_code = int(process.returncode)
    terminals: dict[str, Any] = {"b0": body["b0_terminal"]}
    if exit_code == 0:
        sampler = body["source_sampler"]
        terminals["b3s_zero4"] = _validate_terminal(
            "m1_compact_b3s_zero4_f1_s42_fresh_e11",
            "B3S",
            log_path,
            int(sampler["expected_global_step"]),
            staged["teacher"]["checkpoint"]["path"],
        )
    complete = exit_code == 0 and set(terminals) == {"b0", "b3s_zero4"}
    state_body = {
        "schema": STATE_SCHEMA,
        "status": STATE_PASS if complete else STATE_FAIL,
        "receipt_sha256": body["parent_v2_receipt"]["sha256"],
        "recovery_receipt_sha256": sha(RECOVERY_RECEIPT),
        "parent_v2_receipt_sha256": body["parent_v2_receipt"]["sha256"],
        "source_sampler": body["source_sampler"],
        "inventory_before": staged["inventory"],
        "inventory_after": staged["inventory"],
        "exit_codes": {"b3s_zero4": exit_code},
        "fixed_order": ["b0_validated_existing", "b3s_zero4_launched"],
        "terminal_checkpoints": terminals,
        "b3s_command": command,
        "b3s_log": {"path": str(log_path.resolve()), "sha256": sha(log_path)},
        "target_metrics_not_read_by_runner": True,
        "target_opened_by_training": False,
        "retrained_b0": False,
        "created_at_epoch": time.time(),
    }
    immutable(state_path.resolve(), state_body)
    return 0 if complete else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt", type=Path, default=RECOVERY_RECEIPT)
    parser.add_argument("--state", type=Path, default=RECOVERED_STATE)
    parser.add_argument("--allow-root-launch")
    args = parser.parse_args()
    need(args.prepare ^ args.execute, "choose exactly one of --prepare/--execute")
    if args.prepare:
        result = prepare()
        print(json.dumps({"status": result["status"], "receipt": str(RECOVERY_RECEIPT.resolve()), "sha256": sha(RECOVERY_RECEIPT), "expected_global_step": result["source_sampler"]["expected_global_step"]}, sort_keys=True))
    else:
        raise SystemExit(execute(args.receipt.resolve(), args.state.resolve(), args.allow_root_launch))


if __name__ == "__main__":
    main()
