#!/usr/bin/env python3
"""Execute exactly one reviewed B1 factorial cell, only with ``--execute``.

The default mode is a no-write dry run.  The explicit execution path checks an
immutable preflight against live B1 bindings, creates an O_EXCL pre-execution
contract, records start/completion separately, runs exactly ``train=true`` /
``test=false`` under a single explicit CUDA device, then binds the retained
epoch-004--011 artifacts.  It never chooses a best validation checkpoint.

This executor is deliberately not a batch scheduler.  A caller must name one
predeclared cell, one CUDA index, and (for Stage F) the verified Stage-P
routing aggregate.  It is intended for a future reviewed run, not for this
implementation task.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.metrics import b1_m2_factorial as core


DEFAULT_RESULT_ROOT = REPO_ROOT / "sua_exploration/results/m2_carrier_distillation_interaction_v2"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise core.B1ContractError(message)


def _load_official_preflight(path: Path) -> tuple[dict[str, Any], str]:
    preflight, digest = core.load_verified_immutable_json(path)
    core.validate_preflight_payload(preflight)
    core.validate_live_source_bindings(REPO_ROOT, preflight["implementation_bindings"])
    return preflight, digest


def _validate_stage_f_gate(path: Path | None, *, official_preflight_sha256: str) -> tuple[Path | None, str | None]:
    if path is None:
        return None, None
    aggregate, digest = core.load_verified_immutable_json(path)
    core.validate_stage_p_aggregate(aggregate, official_preflight_sha256=official_preflight_sha256)
    gate = aggregate.get("stage_f_predeclared_gate")
    _need(isinstance(gate, Mapping) and gate.get("stage_f_authorized_by_evidence") is True,
          "Stage F blocked: immutable Stage-P routing gate did not pass")
    return path.resolve(), digest


@dataclass(frozen=True)
class CellPaths:
    log_dir: Path
    artifact_parent: Path
    launch_receipt: Path
    execution_start_receipt: Path
    execution_completion_receipt: Path
    execution_log: Path
    post_training_binding: Path
    score_receipt: Path


def cell_paths(result_root: Path, cell: core.CellSpec) -> CellPaths:
    stem = f"stage{cell.stage}_fold{cell.fold}_seed{cell.seed}_{cell.carrier}_{cell.loss_mode}"
    root = result_root.resolve()
    return CellPaths(
        log_dir=root / "logs" / stem,
        artifact_parent=root / "artifacts" / stem,
        launch_receipt=root / "launch_receipts" / f"{stem}.json",
        execution_start_receipt=root / "execution_receipts" / f"{stem}.start.json",
        execution_completion_receipt=root / "execution_receipts" / f"{stem}.completion.json",
        execution_log=root / "execution_logs" / f"{stem}.train.log",
        post_training_binding=root / "post_training_bindings" / f"{stem}.json",
        score_receipt=root / "score_receipts" / f"{stem}.json",
    )


def run_id_prefix(cell: core.CellSpec) -> str:
    return f"b1_v2_{cell.stage}_f{cell.fold}_s{cell.seed}_{cell.carrier}_{cell.loss_mode}"


def train_command(cell: core.CellSpec, paths: CellPaths) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str((PROJECT / "src/train.py").resolve()),
        f"experiment={core.expected_config_name(cell.carrier, cell.loss_mode)}",
        f"seed={cell.seed}",
        f"data.loso_fold={cell.fold}",
        "train=true",
        "test=false",
        f"run_id={run_id_prefix(cell)}",
        f"hydra.run.dir={paths.log_dir}",
        f"paths.artifact_dir={paths.artifact_parent}",
    ]


@dataclass(frozen=True)
class ExecutionPlan:
    cell: core.CellSpec
    official_preflight_path: Path
    official_preflight_sha256: str
    stage_p_aggregate_path: Path | None
    stage_p_aggregate_sha256: str | None
    cuda_visible_devices: str
    paths: CellPaths
    command: tuple[str, ...]
    working_dir: Path


def build_plan(
    *,
    stage: str,
    fold: int,
    seed: int,
    carrier: str,
    loss_mode: str,
    cuda_visible_devices: str,
    official_preflight: Path,
    stage_p_aggregate: Path | None,
    result_root: Path = DEFAULT_RESULT_ROOT,
) -> ExecutionPlan:
    _need(cuda_visible_devices.isdecimal(), "B1 one-cell executor requires exactly one decimal CUDA device index")
    cell = core.CellSpec(stage, fold, seed, carrier, loss_mode)
    _need(cell in core.cells_for_stage(stage), f"B1 cell is not in the predeclared {stage} lattice: {cell.key}")
    _preflight, preflight_sha = _load_official_preflight(official_preflight)
    if stage == "P":
        _need(stage_p_aggregate is None, "Stage P does not accept a result-contingent aggregate")
        aggregate_path, aggregate_sha = None, None
    else:
        _need(stage_p_aggregate is not None, "Stage F requires immutable Stage-P aggregate")
        aggregate_path, aggregate_sha = _validate_stage_f_gate(
            stage_p_aggregate, official_preflight_sha256=preflight_sha,
        )
    paths = cell_paths(result_root, cell)
    command = tuple(train_command(cell, paths))
    return ExecutionPlan(
        cell=cell,
        official_preflight_path=official_preflight.resolve(),
        official_preflight_sha256=preflight_sha,
        stage_p_aggregate_path=aggregate_path,
        stage_p_aggregate_sha256=aggregate_sha,
        cuda_visible_devices=cuda_visible_devices,
        paths=paths,
        command=command,
        working_dir=PROJECT.resolve(),
    )


def _execution_environment(device: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(core.execution_environment_contract(device))
    return env


def _execution_contract(plan: ExecutionPlan) -> str:
    return core.write_future_launch_receipt(
        plan.paths.launch_receipt,
        spec=plan.cell,
        official_preflight_sha256=plan.official_preflight_sha256,
        command=plan.command,
        explicit_log_dir=plan.paths.log_dir,
        artifact_parent=plan.paths.artifact_parent,
        run_id_prefix=run_id_prefix(plan.cell),
        future_post_training_binding=plan.paths.post_training_binding,
        future_score_receipt=plan.paths.score_receipt,
        interpreter=Path(sys.executable),
        script=PROJECT / "src/train.py",
        working_dir=plan.working_dir,
        cuda_visible_devices=plan.cuda_visible_devices,
        execution_start_receipt=plan.paths.execution_start_receipt,
        execution_completion_receipt=plan.paths.execution_completion_receipt,
        execution_log=plan.paths.execution_log,
    )


def _find_unique_artifact(plan: ExecutionPlan) -> Path:
    parent = plan.paths.artifact_parent
    _need(parent.is_dir(), "B1 successful subprocess produced no committed artifact parent")
    candidates = sorted(
        child.resolve() for child in parent.iterdir()
        if child.is_dir() and child.name.startswith(run_id_prefix(plan.cell))
    )
    _need(len(candidates) == 1,
          f"B1 successful subprocess needs exactly one matching artifact; found {len(candidates)}")
    return candidates[0]


def binder_command(plan: ExecutionPlan, artifact: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str((PROJECT / "scripts/bind_b1_m2_factorial_post_training.py").resolve()),
        "--future-launch-receipt", str(plan.paths.launch_receipt.resolve()),
        "--artifact", str(artifact.resolve()),
        "--checkpoint-run-dir", str(plan.paths.log_dir.resolve()),
        "--execution-completion-receipt", str(plan.paths.execution_completion_receipt.resolve()),
        "--out", str(plan.paths.post_training_binding.resolve()),
    ]


def scorer_command(plan: ExecutionPlan, artifact: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str((PROJECT / "scripts/score_b1_m2_factorial_epochs.py").resolve()),
        "--stage", plan.cell.stage,
        "--fold", str(plan.cell.fold),
        "--seed", str(plan.cell.seed),
        "--carrier", plan.cell.carrier,
        "--loss-mode", plan.cell.loss_mode,
        "--official-preflight", str(plan.official_preflight_path),
        "--checkpoint-run-dir", str(plan.paths.log_dir.resolve()),
        "--post-training-binding", str(plan.paths.post_training_binding.resolve()),
        "--artifact", str(artifact.resolve()),
        "--out", str(plan.paths.score_receipt.resolve()),
        "--execute",
        "--device", "cuda",
    ]


Runner = Callable[..., subprocess.CompletedProcess[Any]]


def execute_plan(
    plan: ExecutionPlan,
    *,
    execute: bool,
    score_after_success: bool = False,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Execute one cell only when explicitly requested; otherwise make no writes."""
    payload: dict[str, Any] = {
        "screen_id": core.SCREEN_ID,
        "cell": plan.cell.key,
        "official_preflight_sha256": plan.official_preflight_sha256,
        "stage_p_aggregate_sha256": plan.stage_p_aggregate_sha256,
        "command": list(plan.command),
        "working_dir": str(plan.working_dir),
        "cuda_visible_devices": plan.cuda_visible_devices,
        "gpu_subprocess_started": False,
        "default_dry_run": not execute,
    }
    if not execute:
        return payload

    # Re-check at the irreversible boundary: a source edit between dry-plan
    # construction and explicit execution invalidates the reviewed preflight.
    _preflight, live_sha = _load_official_preflight(plan.official_preflight_path)
    _need(live_sha == plan.official_preflight_sha256,
          "B1 immutable official-preflight digest changed after plan construction")
    if plan.cell.stage == "F":
        _need(plan.stage_p_aggregate_path is not None, "Stage-F execution plan lacks Stage-P aggregate")
        gate_path, gate_sha = _validate_stage_f_gate(
            plan.stage_p_aggregate_path, official_preflight_sha256=live_sha,
        )
        _need(gate_path == plan.stage_p_aggregate_path and gate_sha == plan.stage_p_aggregate_sha256,
              "Stage-F immutable routing aggregate changed after plan construction")

    contract_sha = _execution_contract(plan)
    start_sha = core.write_execution_start_receipt(
        plan.paths.execution_start_receipt,
        future_launch_receipt=plan.paths.launch_receipt,
        cuda_visible_devices=plan.cuda_visible_devices,
    )
    plan.paths.execution_log.parent.mkdir(parents=True, exist_ok=True)
    invocation_environment = _execution_environment(plan.cuda_visible_devices)
    try:
        with plan.paths.execution_log.open("xb") as log_handle:
            completed = runner(
                list(plan.command), cwd=str(plan.working_dir), env=invocation_environment,
                stdout=log_handle, stderr=subprocess.STDOUT, check=False,
            )
    except BaseException as exc:
        # A Python-level spawn failure has no exit code, so preserve the start
        # receipt and propagate rather than fabricating a completion receipt.
        raise core.B1ContractError(f"B1 subprocess could not be started after immutable start receipt: {exc}") from exc
    exit_code = int(completed.returncode)
    completion_sha = core.write_execution_completion_receipt(
        plan.paths.execution_completion_receipt,
        future_launch_receipt=plan.paths.launch_receipt,
        execution_start_receipt=plan.paths.execution_start_receipt,
        invoked_command=plan.command,
        invoked_working_dir=plan.working_dir,
        invoked_environment=invocation_environment,
        subprocess_started=True,
        exit_code=exit_code,
    )
    payload.update({
        "gpu_subprocess_started": True,
        "future_launch_contract_sha256": contract_sha,
        "execution_start_sha256": start_sha,
        "execution_completion_sha256": completion_sha,
        "subprocess_exit_code": exit_code,
        "training_completed_successfully": exit_code == 0,
    })
    if exit_code != 0:
        payload["status"] = "subprocess_failed_no_binding_or_scoring"
        return payload

    artifact = _find_unique_artifact(plan)
    bind = runner(
        binder_command(plan, artifact), cwd=str(plan.working_dir), env=_execution_environment(plan.cuda_visible_devices),
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False,
    )
    if int(bind.returncode) != 0 or not plan.paths.post_training_binding.is_file():
        raise core.B1ContractError("B1 train succeeded but post-training binder did not complete")
    payload.update({
        "artifact": str(artifact),
        "post_training_binding": str(plan.paths.post_training_binding),
        "post_training_binder_exit_code": int(bind.returncode),
    })
    if score_after_success:
        scored = runner(
            scorer_command(plan, artifact), cwd=str(plan.working_dir), env=_execution_environment(plan.cuda_visible_devices),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False,
        )
        if int(scored.returncode) != 0 or not plan.paths.score_receipt.is_file():
            raise core.B1ContractError("B1 scorer did not complete after successful binding")
        payload.update({"scorer_exit_code": int(scored.returncode), "score_receipt": str(plan.paths.score_receipt)})
    payload["status"] = "subprocess_and_binding_completed" if not score_after_success else "subprocess_binding_and_scoring_completed"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("P", "F"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--carrier", choices=core.CARRIERS, required=True)
    parser.add_argument("--loss-mode", choices=core.LOSS_MODES, required=True)
    parser.add_argument("--cuda-device", required=True, help="one physical GPU index, e.g. 0")
    parser.add_argument("--official-preflight", required=True, type=Path)
    parser.add_argument("--stage-p-aggregate", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--execute", action="store_true", help="required before any subprocess is started")
    parser.add_argument("--score-after-success", action="store_true", help="after binding, explicitly score fixed epochs 5--12")
    args = parser.parse_args()
    plan = build_plan(
        stage=args.stage, fold=args.fold, seed=args.seed, carrier=args.carrier,
        loss_mode=args.loss_mode, cuda_visible_devices=args.cuda_device,
        official_preflight=args.official_preflight, stage_p_aggregate=args.stage_p_aggregate,
        result_root=args.result_root,
    )
    print(json.dumps(execute_plan(plan, execute=args.execute, score_after_success=args.score_after_success), indent=2))


if __name__ == "__main__":
    main()
