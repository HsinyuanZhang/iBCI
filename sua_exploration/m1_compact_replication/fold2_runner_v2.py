#!/usr/bin/env python3
"""Source-only-fit runner for the M1 fold-2 v2 continuation.

Fold 2 is deliberately a waiting contract until the independent fold-1
evaluator has written an immutable PASS gate.  The runner validates that gate
before it creates a log directory or starts a GPU process.  It then runs only
the two fixed source-only cells, B0 followed by B3S-Zero4.  Target evaluation
belongs to :mod:`fold2_evaluate` and is never performed by this runner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import yaml
import torch

from sua_exploration.m1_compact_replication import fold1_runner as base
from sua_exploration.m1_compact_replication import fold1_v2_contract as fold1_contract
from sua_exploration.m1_compact_replication import fold2_sampler_audit as sampler_audit
from sua_exploration.m1_compact_replication import fold2_v2_contract as contract


ROOT = base.ROOT
DEFAULT_RECEIPT = contract.V2_RECEIPT
DEFAULT_STATE = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F2_S42_EXECUTION_v2.json"
)
STATUS_TERMINAL = "PASS_M1_COMPACT_B3S_F2_S42_V2_PAIR_TERMINAL"
STATUS_FAILED = "FAIL_M1_COMPACT_B3S_F2_S42_V2_EXECUTION"
FOLD1_GATE_SCHEMA = "m1_compact_b3s_f1_s42_gate_v2"
FOLD1_GATE_STATUS = "PASS_M1_COMPACT_B3S_F1_S42_NONINFERIORITY"
FOLD1_GATE_THRESHOLD = -0.03


class Fold2ContractError(RuntimeError):
    """Raised whenever the append-only fold-2 launch contract is violated."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Fold2ContractError(message)


def _source_manifest_for_run(run_id: str) -> dict[str, Any]:
    candidates = sorted(
        path
        for path in (ROOT / "logs").rglob("m1_version_b_source_only_fit_manifest.json")
        if run_id in str(path)
    )
    need(
        len(candidates) == 1,
        f"expected one fold-2 source-only manifest for {run_id}, found {len(candidates)}",
    )
    path = candidates[0]
    body = base.read(path, f"fold-2 source-only manifest {run_id}")
    need(body.get("schema") == "m1_version_b_source_only_fit_v2", f"source-only manifest schema drift: {run_id}")
    need(
        body.get("source_only") is True
        and body.get("target_path_resolved_during_fit") is False
        and body.get("target_query_values_read_by_fit") is False,
        f"target fit policy drift: {run_id}",
    )
    need(body.get("outer_fold") == 2 and body.get("outer_left_out") == contract.TARGET, f"fold-2 target drift: {run_id}")
    need(body.get("validation_sessions") == [], f"validation session escaped source-only fit: {run_id}")
    for forbidden in (
        "target_file",
        "target_path",
        "query_window_audit",
        "query_sampler_sha256",
        "query_scored_windows",
    ):
        need(forbidden not in body, f"source-only manifest leaked {forbidden}: {run_id}")
    need(body.get("train_sessions") == list(contract.SOURCES), f"fold-2 source session list drift: {run_id}")
    source_files = {}
    for session in contract.SOURCES:
        spec = body.get("source_files", {}).get(session, {})
        source_path = Path(str(spec.get("path", "")))
        need(
            source_path.is_file()
            and not source_path.is_symlink()
            and base.sha(source_path) == spec.get("sha256"),
            f"source manifest NWB drift: {run_id}/{session}",
        )
        source_files[session] = {
            "path": str(source_path.resolve()),
            "sha256": str(spec["sha256"]),
            "bytes": source_path.stat().st_size,
        }
    return {
        "path": str(path.resolve()),
        "sha256": base.sha(path),
        "source_files": source_files,
        "schema": body["schema"],
        "source_only": True,
    }


def _resolved_config_for_run(run_id: str, arm: str, teacher_path: str) -> dict[str, Any]:
    candidates = sorted(
        {
            path
            for root in (ROOT / "logs", ROOT / "outputs/streaming_calibration")
            for path in root.rglob("resolved_config.yaml")
            if run_id in str(path)
        }
    )
    need(len(candidates) == 1, f"expected one fold-2 resolved config for {run_id}, found {len(candidates)}")
    path = candidates[0]
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"resolved config is not an object: {run_id}")
    need(
        value.get("run_id") == run_id
        and value.get("test") is False
        and value.get("optimized_metric") is None,
        f"resolved test/metric policy drift: {run_id}",
    )
    data = value.get("data", {})
    need(
        data.get("_target_") == contract.SOURCE_ONLY_TARGET,
        f"resolved data module drift: {run_id}",
    )
    need(
        data.get("loso_fold") == 2
        and data.get("source_session_names") == list(contract.SOURCES),
        f"resolved source scope drift: {run_id}",
    )
    need(data.get("afc4_arm") == ("none" if arm == "B0" else "zero4"), f"resolved arm drift: {run_id}")
    trainer = value.get("trainer", {})
    need(
        trainer.get("limit_val_batches") == 0
        and trainer.get("num_sanity_val_steps") == 0
        and trainer.get("min_epochs") == 12
        and trainer.get("max_epochs") == 12,
        f"resolved trainer policy drift: {run_id}",
    )
    model = value.get("model", {})
    need(model.get("teacher_ckpt_path") == teacher_path, f"resolved teacher binding drift: {run_id}")
    return {
        "path": str(path.resolve()),
        "sha256": base.sha(path),
        "source_only_target": data["_target_"],
        "test": False,
        "optimized_metric": None,
        "loso_fold": 2,
        "source_sessions": list(contract.SOURCES),
        "arm": arm,
    }


def _terminal(run_id: str, variant: str, log_path: Path, teacher_path: str, expected_steps: int) -> dict[str, Any]:
    """Validate a fresh terminal checkpoint using the bound fold-2 step count."""
    candidates = sorted(
        path for path in (ROOT / "logs").rglob("*.ckpt")
        if run_id in str(path) and "fixed_last" in str(path)
    )
    need(len(candidates) == 1, f"expected one terminal checkpoint for {run_id}, found {len(candidates)}")
    path = candidates[0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == 11 and payload.get("global_step") == expected_steps, f"{variant} terminal checkpoint drift")
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
    text = log_path.read_text(encoding="utf-8", errors="replace")
    need("Starting training!" in text and "max_epochs=12" in text, f"{variant} training completion evidence missing")
    source_manifest = _source_manifest_for_run(run_id)
    resolved_config = _resolved_config_for_run(run_id, variant, teacher_path)
    return {
        "path": str(path.resolve()),
        "sha256": base.sha(path),
        "variant": variant,
        "epoch": 11,
        "global_step": expected_steps,
        "optimizer": {"count": 1, "state_steps": sorted(state_steps), "lr": 1.0e-4, "weight_decay": 0.0},
        "fit_loop": {"batch_completed": expected_steps, "epoch_processed": 12},
        "source_only_manifest": source_manifest,
        "resolved_config": resolved_config,
        "log": {"path": str(log_path.resolve()), "sha256": base.sha(log_path)},
    }


def _read_gate(path: Path) -> dict[str, Any]:
    """Read and validate the immutable PASS gate emitted by fold-1 evaluator."""
    need(path.is_file() and not path.is_symlink(), f"fold-1 gate missing: {path}")
    need(path.stat().st_mode & 0o777 == 0o444, "fold-1 gate must be immutable")
    body = base.read(path, "fold-1 gate")
    need(body.get("schema") == FOLD1_GATE_SCHEMA, "fold-1 gate schema drift")
    need(body.get("status") == FOLD1_GATE_STATUS, "fold-1 gate is not PASS")
    need(
        body.get("canonical_content_sha256")
        == base.canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"}),
        "fold-1 gate canonical hash drift",
    )
    scope = body.get("scope", {})
    need(
        scope.get("task") == "m1"
        and scope.get("fold") == 1
        and scope.get("target_session") == "ses-20120926"
        and scope.get("formal_opened") is False
        and scope.get("minival_opened") is False
        and scope.get("heldout_opened") is False
        and scope.get("target_backward_steps") == 0
        and scope.get("target_optimizer_steps") == 0
        and scope.get("target_checkpoint_selection") is False,
        "fold-1 gate scope drift",
    )
    # Bind the gate to the exact immutable fold-1 staged receipt.  A PASS file
    # copied from another receipt/run must not unlock fold 2 merely because it
    # has the right schema and metric.
    fold1_receipt = fold1_contract.V2_RECEIPT.resolve()
    need(
        body.get("bindings", {}).get("receipt")
        == {"path": str(fold1_receipt), "sha256": base.sha(fold1_receipt)},
        "fold-1 gate receipt binding drift",
    )
    metrics = body.get("metrics", {})
    delta = float(metrics.get("b3s_zero4_minus_b0"))
    need(delta >= FOLD1_GATE_THRESHOLD, "fold-1 gate non-inferiority threshold failed")
    return {
        "path": str(path.resolve()),
        "sha256": base.sha(path),
        "schema": body["schema"],
        "status": body["status"],
        "delta": delta,
        "threshold": FOLD1_GATE_THRESHOLD,
    }


def validate_fold1_gate_file(path: Path) -> dict[str, Any]:
    """Public testable gate validator; missing/invalid gates fail closed."""
    return _read_gate(path.resolve())


def validate_fold1_gate(staged: Mapping[str, Any]) -> dict[str, Any]:
    spec = staged.get("fold1_gate", {})
    expected = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_GATE_v2.json"
    path = Path(str(spec.get("path", ""))).resolve()
    need(path == expected.resolve(), "fold-1 gate path is not the frozen fold-1 v2 gate")
    return validate_fold1_gate_file(path)


def _validate_staged(staged: Mapping[str, Any], receipt_path: Path) -> None:
    need(staged.get("status") == contract.RECEIPT_STATUS, "fold-2 receipt is not waiting/not-launched")
    need(receipt_path.stat().st_mode & 0o777 == 0o444, "fold-2 receipt is mutable")
    need(
        staged.get("canonical_content_sha256")
        == base.canonical({k: v for k, v in staged.items() if k != "canonical_content_sha256"}),
        "fold-2 receipt canonical drift",
    )
    proposal_path = Path(staged["proposal"]["path"])
    proposal = fold1_contract._read_immutable(
        proposal_path,
        "fold-2 proposal",
        contract.V2_SCHEMA,
        contract.V2_STATUS,
    )
    contract._validate_proposal(proposal)
    need(base.sha(proposal_path) == staged["proposal"]["sha256"], "fold-2 proposal SHA changed after prepare")
    expected_proposal = contract.build_proposal()
    need(
        proposal["canonical_content_sha256"] == base.canonical(expected_proposal),
        "fold-2 proposal source drift",
    )
    need(
        staged["commands"] == {
            "b0": proposal["cells"][0]["argv"],
            "b3s_zero4": proposal["cells"][1]["argv"],
        },
        "fold-2 command vector changed after prepare",
    )
    need(
        all("test=false" in argv and "test=true" not in argv for argv in staged["commands"].values()),
        "fold-2 command opens test path",
    )
    gate = staged.get("fold1_gate", {})
    expected_gate = str((ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_GATE_v2.json").resolve())
    need(gate.get("required") is True and gate.get("available") is False and gate.get("launch_authorized") is False, "fold-1 gate waiting policy drift")
    need(gate.get("path") == expected_gate, "fold-1 gate binding drift")
    need(staged.get("inventory") == contract.source_only_inventory(), "fold-2 source/data inventory changed after prepare")


def _validate_sampler_audit(staged: Mapping[str, Any]) -> dict[str, Any]:
    audit = sampler_audit.read_immutable()
    need(
        audit.get("staged_receipt")
        == {"path": str(DEFAULT_RECEIPT.resolve()), "sha256": base.sha(DEFAULT_RECEIPT)},
        "fold-2 sampler audit staged receipt binding drift",
    )
    need(audit.get("source_inventory") == staged.get("inventory") == contract.source_only_inventory(), "fold-2 sampler audit source inventory drift")
    scope = audit.get("scope", {})
    need(scope.get("fold") == 2 and scope.get("outer_target") == contract.TARGET and scope.get("target_opened") is False, "fold-2 sampler audit scope drift")
    sampler = audit.get("sampler", {})
    trainer = audit.get("trainer", {})
    need(int(sampler.get("source_batches_per_epoch", 0)) > 0 and int(trainer.get("epochs", 0)) == 12, "fold-2 sampler audit cardinality drift")
    need(int(trainer.get("expected_global_step", 0)) == int(sampler["source_batches_per_epoch"]) * int(trainer["epochs"]), "fold-2 sampler audit step derivation drift")
    return audit


def prepare() -> dict[str, Any]:
    # Contract generation is append-only and does not inspect target values.
    return contract.build_receipt()


def execute(receipt_path: Path, state_path: Path, authorization: str | None) -> int:
    need(authorization == "FOLD2_APPROVED_BY_ROOT", "fold-2 launch requires explicit root authorization token")
    staged = base.read(receipt_path, "fold-2 staged receipt")
    _validate_staged(staged, receipt_path)
    # This is deliberately before log creation, artifact checks, and the first
    # subprocess.  A missing/stale/STOP fold-1 gate cannot consume a GPU.
    fold1_gate = validate_fold1_gate(staged)
    sampler = _validate_sampler_audit(staged)
    teacher = contract.validate_teacher()
    need(teacher == staged["teacher"], "fold-2 teacher inputs changed after prepare")
    inventory_before = contract.source_only_inventory()
    need(inventory_before == staged["inventory"], "fold-2 source/data inventory changed at launch")
    artifact = ROOT / "outputs/streaming_calibration"
    for base_name in (
        "m1_compact_b0_f2_s42_fresh_e11",
        "m1_compact_b3s_zero4_f2_s42_fresh_e11",
    ):
        need(not base.prior_artifact_exists(artifact, base_name), f"prior fold-2 artifact exists: {base_name}")
    need(not state_path.exists() and not state_path.is_symlink(), "fold-2 execution state already exists")

    log_dir = state_path.parent / "fold2_v2_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    exit_codes: dict[str, int] = {}
    terminal: dict[str, Any] = {}
    for key in ("b0", "b3s_zero4"):
        # The gate is immutable, but revalidate it before each arm to bind the
        # entire pair to the same fold-1 PASS receipt.
        need(validate_fold1_gate(staged) == fold1_gate, "fold-1 gate changed during pair")
        log_path = log_dir / f"{key}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                staged["commands"][key],
                cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_codes[key] = proc.returncode
        if proc.returncode != 0:
            break
        terminal[key] = _terminal(
            "m1_compact_b0_f2_s42_fresh_e11" if key == "b0" else "m1_compact_b3s_zero4_f2_s42_fresh_e11",
            "B0" if key == "b0" else "B3S",
            log_path,
            staged["teacher"]["checkpoint"]["path"],
            int(sampler["trainer"]["expected_global_step"]),
        )

    inventory_after = contract.source_only_inventory()
    need(inventory_after == inventory_before, "fold-2 source/data inventory changed during execution")
    complete = exit_codes == {"b0": 0, "b3s_zero4": 0} and set(terminal) == {"b0", "b3s_zero4"}
    state_body = {
        "schema": "m1_compact_b3s_f2_s42_execution_v2",
        "status": STATUS_TERMINAL if complete else STATUS_FAILED,
        "receipt_sha256": base.sha(receipt_path),
        "proposal_sha256": staged["proposal"]["sha256"],
        "fold1_gate": fold1_gate,
        "source_sampler_audit": {"path": str(sampler_audit.RECEIPT.resolve()), "sha256": base.sha(sampler_audit.RECEIPT), "body": sampler},
        "source_sampler": sampler,
        "exit_codes": exit_codes,
        "commands": staged["commands"],
        "fixed_order": ["b0", "b3s_zero4"],
        "terminal_checkpoints": terminal,
        "target_metrics_not_read_by_runner": True,
        "target_opened_by_training": False,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "teacher": teacher,
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
