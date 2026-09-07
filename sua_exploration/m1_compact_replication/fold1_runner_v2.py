#!/usr/bin/env python3
"""Source-only-fit runner for the append-only M1 fold-1 v2 proposal.

This module is intentionally separate from the stale v1 runner.  It is a
CPU-only preparation by default; ``--execute`` requires an explicit root
token.  The v2 commands bind the strict source-only-fit data-module class and
``test=false``.  Only the independent evaluator may later instantiate the
ordinary LOSO data module and open the left-out query.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml

from sua_exploration.m1_compact_replication import fold1_runner as base
from sua_exploration.m1_compact_replication import fold1_v2_contract as contract


ROOT = base.ROOT
DEFAULT_RECEIPT = contract.V2_RECEIPT
DEFAULT_STATE = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_EXECUTION_v2.json"
STATUS_STARTED = "PASS_M1_COMPACT_B3S_F1_S42_V2_PAIR_TERMINAL"
STATUS_FAILED = "FAIL_M1_COMPACT_B3S_F1_S42_V2_EXECUTION"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise base.Fold1ContractError(message)


def _source_manifest_for_run(run_id: str) -> dict[str, Any]:
    candidates = sorted(
        path
        for path in (ROOT / "logs").rglob("m1_version_b_source_only_fit_manifest.json")
        if run_id in str(path)
    )
    # train.py writes to the Hydra run directory.  A duplicate means the
    # output is not attributable to one exact launch and must fail closed.
    need(len(candidates) == 1, f"expected one source-only manifest for {run_id}, found {len(candidates)}")
    path = candidates[0]
    body = base.read(path, f"source-only manifest {run_id}")
    need(body.get("schema") == "m1_version_b_source_only_fit_v2", f"source-only manifest schema drift: {run_id}")
    need(body.get("source_only") is True and body.get("target_path_resolved_during_fit") is False, f"target fit policy drift: {run_id}")
    need(body.get("validation_sessions") == [], f"validation session escaped source-only fit: {run_id}")
    for forbidden in ("target_file", "target_path", "query_window_audit", "query_sampler_sha256", "query_scored_windows"):
        need(forbidden not in body, f"source-only manifest leaked {forbidden}: {run_id}")
    sources = ["ses-20120924", "ses-20120927", "ses-20120928"]
    need(body.get("train_sessions") == sources, f"source session list drift: {run_id}")
    source_files = {}
    for session in sources:
        spec = body.get("source_files", {}).get(session, {})
        source_path = Path(str(spec.get("path", "")))
        need(source_path.is_file() and base.sha(source_path) == spec.get("sha256"), f"source manifest NWB drift: {run_id}/{session}")
        source_files[session] = {"path": str(source_path.resolve()), "sha256": str(spec["sha256"]), "bytes": source_path.stat().st_size}
    return {"path": str(path.resolve()), "sha256": base.sha(path), "source_files": source_files, "schema": body["schema"], "source_only": True}


def _resolved_config_for_run(run_id: str, arm: str, teacher_path: str) -> dict[str, Any]:
    # Hydra writes the resolved config beside the artifact under
    # ``outputs/streaming_calibration`` for the compact runs.  Older
    # continuation receipts used a log-root search only, which made terminal
    # validation fail closed even when the bound output config existed.
    candidates = sorted(
        {
            path
            for root in (ROOT / "logs", ROOT / "outputs/streaming_calibration")
            for path in root.rglob("resolved_config.yaml")
            if run_id in str(path)
        }
    )
    need(len(candidates) == 1, f"expected one resolved config for {run_id}, found {len(candidates)}")
    path = candidates[0]
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"resolved config is not an object: {run_id}")
    need(value.get("run_id") == run_id and value.get("test") is False and value.get("optimized_metric") is None, f"resolved test/metric policy drift: {run_id}")
    data = value.get("data", {})
    need(data.get("_target_") == contract.SOURCE_ONLY_TARGET, f"resolved data module drift: {run_id}")
    need(data.get("loso_fold") == 1 and data.get("source_session_names") == list(contract.SOURCE_SESSIONS), f"resolved source scope drift: {run_id}")
    need(data.get("afc4_arm") == ("none" if arm == "B0" else "zero4"), f"resolved arm drift: {run_id}")
    trainer = value.get("trainer", {})
    need(trainer.get("limit_val_batches") == 0 and trainer.get("num_sanity_val_steps") == 0 and trainer.get("min_epochs") == 12 and trainer.get("max_epochs") == 12, f"resolved trainer policy drift: {run_id}")
    model = value.get("model", {})
    need(model.get("teacher_ckpt_path") == teacher_path, f"resolved teacher binding drift: {run_id}")
    return {"path": str(path.resolve()), "sha256": base.sha(path), "source_only_target": data["_target_"], "test": False, "optimized_metric": None, "loso_fold": 1, "source_sessions": list(contract.SOURCE_SESSIONS), "arm": arm}


def _terminal(run_id: str, variant: str, log_path: Path, teacher_path: str) -> dict[str, Any]:
    result = base.terminal_checkpoint(run_id, variant, log_path, teacher_path)
    result["source_only_manifest"] = _source_manifest_for_run(run_id)
    result["resolved_config"] = _resolved_config_for_run(run_id, variant, teacher_path)
    return result


def prepare() -> dict[str, Any]:
    # Contract generation is append-only and idempotent.  It validates the
    # current source/data inventory before creating a read-only receipt.
    return contract.build_receipt()


def _validate_staged(staged: dict[str, Any], receipt_path: Path) -> None:
    need(staged.get("status") == contract.RECEIPT_STATUS, "v2 staged receipt is not dry/not-launched")
    need(receipt_path.stat().st_mode & 0o777 == 0o444, "v2 staged receipt is mutable")
    need(staged.get("canonical_content_sha256") == base.canonical({k: v for k, v in staged.items() if k != "canonical_content_sha256"}), "v2 staged receipt canonical drift")
    proposal_path = Path(staged["proposal"]["path"])
    proposal = contract._read_immutable(proposal_path, "v2 proposal", contract.V2_SCHEMA, contract.V2_STATUS)
    contract._validate_v2_proposal(proposal)
    need(base.sha(proposal_path) == staged["proposal"]["sha256"], "v2 proposal SHA changed after prepare")
    expected_proposal = contract.build_proposal()
    need(proposal["canonical_content_sha256"] == base.canonical(expected_proposal), "v2 proposal source drift")
    need(staged["commands"] == {"b0": proposal["cells"][0]["argv"], "b3s_zero4": proposal["cells"][1]["argv"]}, "v2 command vector changed after prepare")
    need(all("test=false" in argv and "test=true" not in argv for argv in staged["commands"].values()), "v2 command opens test path")
    contract.validate_equivalence_binding(receipt_path)


def execute(receipt_path: Path, state_path: Path, authorization: str | None) -> int:
    need(authorization == "FOLD1_APPROVED_BY_ROOT", "fold1 v2 launch requires explicit root authorization token")
    staged = base.read(receipt_path, "v2 staged receipt")
    _validate_staged(staged, receipt_path)
    need(base.validate_gate()["sha256"] == staged["gate"]["sha256"], "fold0/e23 gate changed after prepare")
    need(base.validate_teacher() == staged["teacher"], "fold1 teacher inputs changed after prepare")
    need(base.validate_preflight() == staged["preflight"], "fold1 preflight changed after prepare")
    inventory_before = contract.source_only_inventory()
    need(inventory_before == staged["inventory"], "source/data inventory changed after prepare")
    artifact = ROOT / "outputs/streaming_calibration"
    for base_name in ("m1_compact_b0_f1_s42_fresh_e11", "m1_compact_b3s_zero4_f1_s42_fresh_e11"):
        need(not base.prior_artifact_exists(artifact, base_name), f"prior fold1 artifact exists: {base_name}_f1_s42_")
    need(not state_path.exists() and not state_path.is_symlink(), "v2 execution state already exists")

    log_dir = state_path.parent / "fold1_v2_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    exit_codes: dict[str, int] = {}
    terminal: dict[str, Any] = {}
    for key in ("b0", "b3s_zero4"):
        log_path = log_dir / f"{key}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                staged["commands"][key], cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        exit_codes[key] = proc.returncode
        if proc.returncode != 0:
            break
        terminal[key] = _terminal(
            "m1_compact_b0_f1_s42_fresh_e11" if key == "b0" else "m1_compact_b3s_zero4_f1_s42_fresh_e11",
            "B0" if key == "b0" else "B3S", log_path,
            staged["teacher"]["checkpoint"]["path"],
        )

    inventory_after = contract.source_only_inventory()
    need(inventory_after == inventory_before, "source/data inventory changed during v2 execution")
    complete = exit_codes == {"b0": 0, "b3s_zero4": 0} and set(terminal) == {"b0", "b3s_zero4"}
    state_body = {
        "schema": "m1_compact_b3s_f1_s42_execution_v2",
        "status": STATUS_STARTED if complete else STATUS_FAILED,
        "receipt_sha256": base.sha(receipt_path),
        "proposal_sha256": staged["proposal"]["sha256"],
        "exit_codes": exit_codes,
        "commands": staged["commands"],
        "fixed_order": ["b0", "b3s_zero4"],
        "terminal_checkpoints": terminal,
        "target_metrics_not_read_by_runner": True,
        "target_opened_by_training": False,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "teacher": base.validate_teacher(),
        "preflight": base.validate_preflight(),
        "created_at_epoch": time.time(),
    }
    base.immutable(state_path.resolve(), state_body)
    return 0 if complete else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--allow-root-launch")
    args = parser.parse_args()
    need(args.prepare ^ args.execute, "choose exactly one of --prepare/--execute")
    if args.prepare:
        body = prepare()
        print(json.dumps({"status": body["status"], "receipt": str(DEFAULT_RECEIPT.resolve()), "sha256": base.sha(DEFAULT_RECEIPT)}, sort_keys=True))
    else:
        raise SystemExit(execute(args.receipt.resolve(), args.state.resolve(), args.allow_root_launch))


if __name__ == "__main__":
    main()
