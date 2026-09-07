#!/usr/bin/env python3
"""Static two-lane runner and receipt-only pilot gate for RT R4.

This file is executable infrastructure, not launch authorization.  ``plan``
only prints the frozen 12-cell matrix.  ``run-lane`` requires a separately
reviewed immutable execution-preflight receipt, checks the selected physical
GPU and active RT writers, then runs one static six-cell lane serially.  No
background process or tmux session is created by this script.

The pilot gate is computed independently for M6 and M12 from folds 0, 7, and
14.  A budget expands only when all three Full-minus-MB4 deltas are positive
and their mean is at least +0.03 R2.  The aggregator reports decisions but
never launches expansion folds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
TRAIN_ENTRY = PROJECT / "src/train.py"
EVAL_ENTRY = PROJECT / "src/rt_r4_budget_response_eval.py"
PREPARE_RECEIPT = (
    REPO
    / "sua_exploration/results/rt_r4_budget_response_common_q24_v1/"
      "RT_R4_COMMON_Q24_SOURCE_ONLY_PREPARE_v1.json"
)
PREPARE_RECEIPT_SHA256 = (
    "f54ca085822391cfd3afb3868e0e5f85b7b2219502354be7a72db65dc7b29421"
)
EXECUTION_PREFLIGHT_SCHEMA = "rt_r4_execution_preflight_v1"
EXECUTION_PREFLIGHT_STATUS = "PASS_R4_EXECUTION_PREFLIGHT_NOT_LAUNCHED"
OUTER_SCHEMA = "rt_r4_budget_response_outer_eval_v1"
OUTER_STATUS = "PASS_R4_ONE_SHOT_OUTER_TARGET_NO_BACKPROP"
CELL_SCHEMA = "rt_r4_pilot_cell_terminal_v1"
CELL_STATUS = "PASS_R4_PILOT_CELL_ONE_SHOT_NO_BACKPROP_STATE_IDENTICAL"
AGGREGATE_SCHEMA = "rt_r4_budgetwise_three_fold_pilot_gate_v1"
AGGREGATE_STATUS = "PASS_R4_PILOT_COMPLETE_BUDGET_DECISIONS_RECORDED"
ROOT_SCHEMA = "rt_r4_pilot_work_root_v1"

SEED = 42
PILOT_FOLDS = (0, 7, 14)
BUDGETS = (6, 12)
ARMS = ("afc4_vel", "afc4_mb4")
EXPANSION_FOLDS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13)
MEAN_DELTA_GATE = 0.03
# Each matched Full/MB4 pair stays on one GPU.  Budgets and folds are crossed
# across the two equal-length lanes so a physical GPU is not confounded with
# one carrier budget.
STATIC_LANES: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((6, 0), (12, 7), (6, 14)),
    1: ((12, 0), (6, 7), (12, 14)),
}


class RtR4PilotError(RuntimeError):
    """A pilot execution, identity, or aggregation contract failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RtR4PilotError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"missing {label}: {path}")
    _need((path.stat().st_mode & 0o777) == 0o444, f"{label} is not mode 0444: {path}")


def _write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    _need(not path.exists(), f"refusing to overwrite immutable R4 artifact: {path}")
    payload = json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"file binding source missing: {path}")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "sha256": _sha256(path),
    }


def _verify_binding(value: Mapping[str, Any], *, label: str) -> Path:
    _need(isinstance(value.get("path"), str), f"{label} path absent")
    path = Path(str(value["path"])).resolve()
    _need(path.is_file(), f"{label} path missing: {path}")
    _need(_sha256(path) == value.get("sha256"), f"{label} SHA drift: {path}")
    _need(path.stat().st_size == int(value.get("size", -1)), f"{label} size drift: {path}")
    return path


@dataclass(frozen=True)
class CellPaths:
    cell: Path
    fit: Path
    config: Path
    split: Path
    selection: Path
    outer: Path
    terminal: Path
    log: Path


def cell_paths(root: Path, *, budget: int, arm: str, fold: int) -> CellPaths:
    _need(budget in BUDGETS, f"unsupported primary R4 budget: {budget}")
    _need(arm in ARMS, f"unsupported R4 arm: {arm}")
    _need(fold in PILOT_FOLDS or fold in EXPANSION_FOLDS, f"invalid R4 fold: {fold}")
    cell = root / f"m{budget}" / arm / f"fold_{fold:02d}" / f"seed_{SEED}"
    fit = cell / "fit"
    return CellPaths(
        cell=cell,
        fit=fit,
        config=fit / ".hydra/config.yaml",
        split=fit / "split_manifest.json",
        selection=fit / "rt_nested_selection_receipt.json",
        outer=cell / "outer_target_eval.json",
        terminal=cell / "cell_terminal.json",
        log=cell / "supervisor.log",
    )


def run_id(*, budget: int, arm: str, fold: int) -> str:
    return f"rt_r4_common_q24_m{budget}_{arm}_f{fold}_s{SEED}"


def train_command(*, root: Path, budget: int, arm: str, fold: int) -> list[str]:
    paths = cell_paths(root, budget=budget, arm=arm, fold=fold)
    return [
        sys.executable,
        str(TRAIN_ENTRY),
        "experiment=rt_r4_budget_response_common_q24",
        f"run_id={run_id(budget=budget, arm=arm, fold=fold)}",
        f"data.side_feature_calibration_n_trials={budget}",
        f"data.side_feature_group={arm}",
        f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}",
        f"seed={SEED}",
        "ckpt_path=null",
        "train=true",
        "test=false",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        f"hydra.run.dir={paths.fit}",
        f"paths.root_dir={PROJECT}",
        f"paths.log_dir={root / '_hydra_logs'}",
        f"paths.artifact_dir={root / '_artifacts'}",
    ]


def eval_command(
    *, root: Path, budget: int, arm: str, fold: int, checkpoint: Path, device: str = "cuda"
) -> list[str]:
    paths = cell_paths(root, budget=budget, arm=arm, fold=fold)
    return [
        sys.executable,
        str(EVAL_ENTRY),
        "--config", str(paths.config.resolve()),
        "--checkpoint", str(checkpoint.resolve()),
        "--split-manifest", str(paths.split.resolve()),
        "--selection-receipt", str(paths.selection.resolve()),
        "--output", str(paths.outer.resolve()),
        "--outer-fold", str(fold),
        "--device", str(device),
    ]


def _lane_cells(lane: int) -> list[dict[str, Any]]:
    _need(lane in STATIC_LANES, f"R4 lane must be one of {tuple(STATIC_LANES)}")
    return [
        {"budget": budget, "fold": fold, "arm": arm, "seed": SEED}
        for budget, fold in STATIC_LANES[lane]
        for arm in ARMS
    ]


def build_plan(*, work_root: Path, execution_preflight: Path | None = None) -> dict[str, Any]:
    root = work_root.resolve()
    lanes: dict[str, Any] = {}
    all_cells: list[dict[str, Any]] = []
    for lane in sorted(STATIC_LANES):
        cells = []
        for spec in _lane_cells(lane):
            command = train_command(root=root, budget=spec["budget"], arm=spec["arm"], fold=spec["fold"])
            row = {**spec, "lane": lane, "physical_gpu": lane, "train_command": command}
            cells.append(row)
            all_cells.append(row)
        lanes[str(lane)] = {
            "physical_gpu": lane,
            "requires_gpu_name_contains": "3090",
            "serial_cells": cells,
            "run_lane_command_not_executed": [
                sys.executable, str(Path(__file__).resolve()), "run-lane",
                "--work-root", str(root), "--lane", str(lane), "--gpu", str(lane),
                "--execution-preflight", str(execution_preflight.resolve())
                if execution_preflight is not None else "<REVIEWED_EXECUTION_PREFLIGHT>",
            ],
        }
    matrix = {(row["budget"], row["fold"], row["arm"]) for row in all_cells}
    expected = {(budget, fold, arm) for budget in BUDGETS for fold in PILOT_FOLDS for arm in ARMS}
    _need(matrix == expected and len(all_cells) == 12, "R4 static lane matrix is not exact")
    return {
        "schema": "rt_r4_static_dual_3090_pilot_plan_v1",
        "status": "DRY_RUN_ONLY_NOT_LAUNCHED",
        "work_root": str(root),
        "seed": SEED,
        "activity_calibration_trials": 24,
        "carrier_budgets": list(BUDGETS),
        "common_query_start_trial": 24,
        "pilot_folds": list(PILOT_FOLDS),
        "arms": list(ARMS),
        "lanes": lanes,
        "pilot_cell_count": len(all_cells),
        "budget_gate": {
            "paired_delta": "afc4_vel_minus_afc4_mb4",
            "positive_folds_required": "3/3",
            "mean_delta_at_least_r2": MEAN_DELTA_GATE,
            "evaluated_independently_per_budget": True,
            "expansion_folds_for_passing_budget_only": list(EXPANSION_FOLDS),
            "aggregator_never_launches_expansion": True,
        },
        "scope": {
            "nwb_files_opened": 0,
            "outer_target_payloads_opened": 0,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "cuda_queried": False,
            "gpu_processes_started": 0,
            "tmux_sessions_created": 0,
            "commands_executed": 0,
        },
    }


def validate_execution_preflight(path: Path) -> dict[str, Any]:
    _immutable(path, "R4 execution preflight")
    body = _json(path)
    _need(
        body.get("schema") == EXECUTION_PREFLIGHT_SCHEMA
        and body.get("status") == EXECUTION_PREFLIGHT_STATUS,
        "R4 execution preflight schema/status mismatch",
    )
    parent = body.get("supersedes_without_mutating")
    _need(isinstance(parent, Mapping), "R4 execution preflight lacks prepare-receipt binding")
    prepare = _verify_binding(parent, label="R4 original prepare receipt")
    _need(prepare == PREPARE_RECEIPT.resolve(), "R4 execution preflight binds another prepare receipt")
    _need(parent.get("sha256") == PREPARE_RECEIPT_SHA256,
          "R4 original prepare receipt SHA mismatch")
    closure = body.get("execution_code_closure_sha256")
    _need(isinstance(closure, Mapping) and bool(closure), "R4 execution code closure absent")
    for relative, expected in closure.items():
        source = PROJECT / str(relative)
        _need(source.is_file() and _sha256(source) == expected,
              f"R4 execution code closure drift: {relative}")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and all(int(scope.get(key, -1)) == 0 for key in (
        "nwb_files_opened", "outer_target_payloads_opened", "trainer_constructed",
        "optimizer_constructed", "cuda_queried", "gpu_processes_started",
        "tmux_sessions_created", "pilot_commands_executed",
    )), "R4 execution preflight is not a no-launch receipt")
    return body


def _selected_checkpoint(receipt_path: Path) -> Path:
    receipt = _json(receipt_path)
    _need(receipt.get("schema") == "rt_clean_nested_loso_selection_receipt_v1"
          and receipt.get("status") == "PASS_FIT_INNER_SELECTION_ONLY",
          f"R4 cell lacks a passing selection receipt: {receipt_path}")
    checkpoint = Path(str(receipt.get("best_model_path", ""))).resolve()
    _need(checkpoint.is_file() and _sha256(checkpoint) == receipt.get("best_model_sha256"),
          f"R4 selected checkpoint is missing or SHA-drifted: {checkpoint}")
    return checkpoint


def _active_gpu_and_rt_commands(gpu: int) -> tuple[str, list[str], list[str]]:
    try:
        table = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
        ps_rows = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RtR4PilotError("cannot safely inspect R4 GPU/RT writers") from error
    gpu_rows: dict[int, tuple[str, str]] = {}
    for line in table.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit():
            gpu_rows[int(fields[0])] = (fields[1], fields[2])
    _need(gpu in gpu_rows, f"requested physical GPU {gpu} is not visible")
    name, uuid = gpu_rows[gpu]
    owners: list[str] = []
    for line in apps.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1] == uuid:
            try:
                owners.append(
                    Path(f"/proc/{int(fields[0])}/cmdline").read_bytes()
                    .replace(b"\0", b" ").decode("utf-8", errors="replace")
                )
            except OSError:
                owners.append(f"<unreadable-gpu-pid:{fields[0]}>")
    rt_writers = [row for row in ps_rows if "rt_r4_common_q24" in row]
    return name, owners, rt_writers


def _same_cell_writer(commands: Sequence[str], *, budget: int, arm: str, fold: int) -> list[str]:
    required = {
        f"data.side_feature_calibration_n_trials={budget}",
        f"data.side_feature_group={arm}",
        f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}",
    }
    collisions = []
    for command in commands:
        try:
            tokens = set(shlex.split(command))
        except ValueError:
            collisions.append(command)
            continue
        if required.issubset(tokens):
            collisions.append(command)
    return collisions


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _run(command: Sequence[str], *, env: Mapping[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + _command_text(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(command), cwd=PROJECT, env=dict(env), stdout=handle, stderr=subprocess.STDOUT
        )
        handle.write(f"[exit={completed.returncode}]\n")
    return int(completed.returncode)


def validate_outer_receipt(
    path: Path, *, budget: int, arm: str, fold: int, require_immutable: bool = True
) -> dict[str, Any]:
    if require_immutable:
        _immutable(path, "R4 outer receipt")
    else:
        _need(path.is_file(), f"missing R4 outer receipt: {path}")
    value = _json(path)
    _need(value.get("schema") == OUTER_SCHEMA and value.get("status") == OUTER_STATUS,
          f"R4 outer receipt schema/status drift: {path}")
    _need(value.get("arm") == arm and int(value.get("carrier_calibration_trials", -1)) == budget,
          f"R4 outer arm/budget drift: {path}")
    _need(int(value.get("outer_loso_fold", -1)) == fold and int(value.get("seed", -1)) == SEED,
          f"R4 outer fold/seed drift: {path}")
    _need(int(value.get("activity_calibration_trials", -1)) == 24
          and int(value.get("query_start_trial", -1)) == 24,
          f"R4 outer activity/query boundary drift: {path}")
    _need(value.get("target_backpropagation") is False
          and value.get("optimizer_present") is False
          and value.get("trainer_present") is False
          and value.get("model_training_mode") is False,
          f"R4 outer receipt does not prove forward-only target evaluation: {path}")
    _need(value.get("model_state_unchanged") is True
          and value.get("model_state_sha256_before") == value.get("model_state_sha256_after")
          and isinstance(value.get("model_state_sha256_before"), str)
          and len(value["model_state_sha256_before"]) == 64,
          f"R4 outer model state identity failed: {path}")
    _need(isinstance(value.get("query_window_identity_sha256"), str)
          and len(value["query_window_identity_sha256"]) == 64
          and int(value.get("query_windows_evaluated", 0)) > 0,
          f"R4 outer query identity/count invalid: {path}")
    target_fit = value.get("target_carrier_prefix_fit")
    _need(isinstance(target_fit, Mapping)
          and target_fit.get("status") == "PASS_TARGET_ACTIVITY_M24_CARRIER_PREFIX_QUERY_Q24"
          and int(target_fit.get("carrier_calibration_trials", -1)) == budget
          and target_fit.get("prefix_fit", {}).get("trial_index_range") == [0, budget],
          f"R4 target did not use the requested carrier prefix: {path}")
    _need(isinstance(value.get("r2_variance_weighted"), (int, float))
          and math.isfinite(float(value["r2_variance_weighted"])),
          f"R4 outer R2 is absent/non-finite: {path}")
    return value


def _write_cell_terminal(
    *, paths: CellPaths, budget: int, arm: str, fold: int, outer: Mapping[str, Any]
) -> str:
    checkpoint = Path(str(outer["checkpoint_path"])).resolve()
    files = {
        "checkpoint": _file_binding(checkpoint),
        "config": _file_binding(paths.config),
        "split_manifest": _file_binding(paths.split),
        "selection_receipt": _file_binding(paths.selection),
        "outer_receipt": _file_binding(paths.outer),
    }
    body = {
        "schema": CELL_SCHEMA,
        "status": CELL_STATUS,
        "budget": budget,
        "arm": arm,
        "fold": fold,
        "seed": SEED,
        "outer_target_session": outer["outer_target_session"],
        "query_window_identity_sha256": outer["query_window_identity_sha256"],
        "query_windows_evaluated": outer["query_windows_evaluated"],
        "r2_variance_weighted": outer["r2_variance_weighted"],
        "target_carrier_raw_prefix_sha256": outer["target_carrier_prefix_fit"][
            "raw_prefix_descriptor_sha256"
        ],
        "source_only_normalizer_sha256": outer["source_only_normalizer_sha256"],
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_state_sha256_before": outer["model_state_sha256_before"],
        "model_state_sha256_after": outer["model_state_sha256_after"],
        "model_state_unchanged": True,
        "files": files,
    }
    return _write_immutable(paths.terminal, body)


def validate_cell_terminal(
    *, root: Path, budget: int, arm: str, fold: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = cell_paths(root, budget=budget, arm=arm, fold=fold)
    _immutable(paths.terminal, "R4 cell terminal")
    terminal = _json(paths.terminal)
    _need(terminal.get("schema") == CELL_SCHEMA and terminal.get("status") == CELL_STATUS,
          f"R4 cell terminal schema/status drift: {paths.terminal}")
    _need(int(terminal.get("budget", -1)) == budget and terminal.get("arm") == arm
          and int(terminal.get("fold", -1)) == fold and int(terminal.get("seed", -1)) == SEED,
          f"R4 cell terminal identity drift: {paths.terminal}")
    files = terminal.get("files")
    _need(isinstance(files, Mapping), f"R4 cell terminal file bindings absent: {paths.terminal}")
    expected_paths = {
        "checkpoint": None,
        "config": paths.config.resolve(),
        "split_manifest": paths.split.resolve(),
        "selection_receipt": paths.selection.resolve(),
        "outer_receipt": paths.outer.resolve(),
    }
    for key, expected in expected_paths.items():
        value = files.get(key)
        _need(isinstance(value, Mapping), f"R4 terminal lacks {key} binding")
        actual = _verify_binding(value, label=f"R4 terminal {key}")
        if expected is not None:
            _need(actual == expected, f"R4 terminal {key} path drift")
    outer = validate_outer_receipt(paths.outer, budget=budget, arm=arm, fold=fold)
    _need(float(terminal.get("r2_variance_weighted")) == float(outer["r2_variance_weighted"])
          and terminal.get("query_window_identity_sha256") == outer["query_window_identity_sha256"]
          and terminal.get("source_only_normalizer_sha256") == outer["source_only_normalizer_sha256"]
          and terminal.get("target_carrier_raw_prefix_sha256")
          == outer["target_carrier_prefix_fit"]["raw_prefix_descriptor_sha256"],
          f"R4 cell terminal summary drift: {paths.terminal}")
    return terminal, outer


def _prepare_work_root(root: Path, *, execution_preflight: Path) -> None:
    _need(root.resolve() != PROJECT.resolve() and root.resolve() != REPO.resolve(),
          "R4 work root may not be a project/repository root")
    marker = root / "R4_PILOT_ROOT_v1.json"
    expected = {
        "schema": ROOT_SCHEMA,
        "seed": SEED,
        "execution_preflight_path": str(execution_preflight.resolve()),
        "execution_preflight_sha256": _sha256(execution_preflight),
    }
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        _write_immutable(marker, expected)
        return
    _immutable(marker, "R4 work-root marker")
    _need(_json(marker) == expected, f"R4 work-root marker drift: {marker}")


def run_cell(
    *, root: Path, budget: int, arm: str, fold: int, gpu: int, env: Mapping[str, str]
) -> str:
    paths = cell_paths(root, budget=budget, arm=arm, fold=fold)
    if paths.terminal.exists():
        validate_cell_terminal(root=root, budget=budget, arm=arm, fold=fold)
        return "skipped_complete"
    _need(not paths.outer.exists(), f"R4 one-shot outer receipt already exists without terminal: {paths.outer}")
    _need(not paths.fit.exists() or paths.selection.exists(),
          f"R4 partial fit lacks selection receipt; inspect, do not resume: {paths.fit}")

    if not paths.selection.exists():
        command = train_command(root=root, budget=budget, arm=arm, fold=fold)
        code = _run(command, env=env, log=paths.log)
        _need(code == 0, f"R4 fit failed with exit {code}: M{budget} {arm} fold{fold}")
    checkpoint = _selected_checkpoint(paths.selection)
    _need(paths.config.is_file() and paths.split.is_file(),
          "R4 passing selection receipt lacks config/split manifest")
    command = eval_command(
        root=root, budget=budget, arm=arm, fold=fold, checkpoint=checkpoint, device="cuda"
    )
    code = _run(command, env=env, log=paths.log)
    _need(code == 0, f"R4 outer evaluator failed with exit {code}: M{budget} {arm} fold{fold}")
    outer = validate_outer_receipt(paths.outer, budget=budget, arm=arm, fold=fold)
    _write_cell_terminal(paths=paths, budget=budget, arm=arm, fold=fold, outer=outer)
    validate_cell_terminal(root=root, budget=budget, arm=arm, fold=fold)
    return "passed"


def run_lane(
    *, work_root: Path, lane: int, gpu: int, execution_preflight: Path
) -> dict[str, Any]:
    _need(lane in STATIC_LANES and gpu == lane,
          "R4 static plan binds lane0->GPU0 and lane1->GPU1")
    validate_execution_preflight(execution_preflight)
    name, owners, _rt_writers = _active_gpu_and_rt_commands(gpu)
    _need("3090" in name, f"R4 pilot requires an RTX 3090, got {name!r}")
    _need(not owners, f"R4 requested GPU{gpu} has active compute owners: {owners}")
    root = work_root.resolve()
    _prepare_work_root(root, execution_preflight=execution_preflight)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    summary: dict[str, str] = {}
    for spec in _lane_cells(lane):
        current_name, current_owners, rt_writers = _active_gpu_and_rt_commands(gpu)
        _need(current_name == name, f"R4 GPU{gpu} identity changed during the static lane")
        _need(not current_owners, f"R4 GPU{gpu} acquired another compute owner: {current_owners}")
        collisions = _same_cell_writer(
            rt_writers, budget=spec["budget"], arm=spec["arm"], fold=spec["fold"]
        )
        _need(not collisions, f"active same-cell R4 writer: {collisions}")
        key = f"m{spec['budget']}:{spec['arm']}:fold{spec['fold']}:seed{SEED}"
        summary[key] = run_cell(
            root=root, budget=spec["budget"], arm=spec["arm"], fold=spec["fold"],
            gpu=gpu, env=env,
        )
    terminal = root / f"lane_{lane}_terminal.json"
    body = {
        "schema": "rt_r4_static_lane_terminal_v1",
        "status": "PASS_R4_STATIC_LANE_COMPLETE",
        "lane": lane,
        "physical_gpu": gpu,
        "gpu_name": name,
        "cells": _lane_cells(lane),
        "summary": summary,
        "execution_preflight": _file_binding(execution_preflight),
    }
    digest = _write_immutable(terminal, body)
    return {"status": body["status"], "output": str(terminal), "sha256": digest}


def aggregate_pilot(*, work_root: Path, output: Path) -> dict[str, Any]:
    _need(not output.exists(), f"refusing to overwrite R4 pilot aggregate: {output}")
    rows: list[dict[str, Any]] = []
    per_fold_query_hash: dict[int, str] = {}
    decisions: dict[str, Any] = {}
    for budget in BUDGETS:
        deltas: list[float] = []
        budget_rows: list[dict[str, Any]] = []
        for fold in PILOT_FOLDS:
            full_terminal, full = validate_cell_terminal(
                root=work_root, budget=budget, arm="afc4_vel", fold=fold
            )
            mb_terminal, mb4 = validate_cell_terminal(
                root=work_root, budget=budget, arm="afc4_mb4", fold=fold
            )
            for key in (
                "outer_target_session", "query_window_identity_sha256",
                "query_windows_evaluated", "source_only_normalizer_sha256",
            ):
                _need(full.get(key) == mb4.get(key),
                      f"R4 M{budget} fold{fold} Full/MB4 mismatch: {key}")
            full_raw = full["target_carrier_prefix_fit"]["raw_prefix_descriptor_sha256"]
            mb4_raw = mb4["target_carrier_prefix_fit"]["raw_prefix_descriptor_sha256"]
            _need(full_raw == mb4_raw,
                  f"R4 M{budget} fold{fold} Full/MB4 raw target carrier mismatch")
            query_hash = str(full["query_window_identity_sha256"])
            if fold in per_fold_query_hash:
                _need(per_fold_query_hash[fold] == query_hash,
                      f"R4 fold{fold} query hash differs across M6/M12")
            else:
                per_fold_query_hash[fold] = query_hash
            delta = float(full["r2_variance_weighted"]) - float(mb4["r2_variance_weighted"])
            row = {
                "budget": budget,
                "fold": fold,
                "seed": SEED,
                "outer_target_session": full["outer_target_session"],
                "query_window_identity_sha256": query_hash,
                "query_windows_evaluated": full["query_windows_evaluated"],
                "source_only_normalizer_sha256": full["source_only_normalizer_sha256"],
                "target_carrier_raw_prefix_sha256": full_raw,
                "full_r2": float(full["r2_variance_weighted"]),
                "mb4_r2": float(mb4["r2_variance_weighted"]),
                "full_minus_mb4": delta,
                "full_terminal_sha256": _sha256(cell_paths(
                    work_root, budget=budget, arm="afc4_vel", fold=fold
                ).terminal),
                "mb4_terminal_sha256": _sha256(cell_paths(
                    work_root, budget=budget, arm="afc4_mb4", fold=fold
                ).terminal),
                "state_identity_verified": (
                    full_terminal["model_state_sha256_before"]
                    == full_terminal["model_state_sha256_after"]
                    and mb_terminal["model_state_sha256_before"]
                    == mb_terminal["model_state_sha256_after"]
                ),
            }
            rows.append(row)
            budget_rows.append(row)
            deltas.append(delta)
        mean_delta = sum(deltas) / len(deltas)
        positive = sum(value > 0.0 for value in deltas)
        passed = positive == len(PILOT_FOLDS) and mean_delta >= MEAN_DELTA_GATE
        decisions[str(budget)] = {
            "budget": budget,
            "folds": list(PILOT_FOLDS),
            "signed_full_minus_mb4": deltas,
            "positive_folds": positive,
            "mean_full_minus_mb4": mean_delta,
            "gate": {
                "requires_positive_folds": "3/3",
                "requires_mean_at_least_r2": MEAN_DELTA_GATE,
                "passed": passed,
            },
            "decision": "EXPAND_THIS_BUDGET_TO_REMAINING_12_FOLDS" if passed else "STOP_THIS_BUDGET",
            "expansion_folds": list(EXPANSION_FOLDS) if passed else [],
            "launch_performed_by_aggregator": False,
            "rows": budget_rows,
        }
    body = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "scope": "receipt_only_no_nwb_no_trainer_no_optimizer_no_cuda_no_launch",
        "seed": SEED,
        "pilot_folds": list(PILOT_FOLDS),
        "activity_calibration_trials": 24,
        "carrier_budgets": list(BUDGETS),
        "common_query_start_trial": 24,
        "rows": rows,
        "per_fold_common_query_sha256_across_budgets_and_arms": {
            str(key): value for key, value in sorted(per_fold_query_hash.items())
        },
        "budget_decisions": decisions,
        "failed_budget_policy": "stop; no added arm, seed, fold, or fusion path",
        "pilot_retention_policy": "pilot folds remain in any later 15-fold aggregate",
        "m18_policy": "not authorized or launched by this aggregator",
    }
    digest = _write_immutable(output, body)
    return {"status": AGGREGATE_STATUS, "output": str(output), "sha256": digest,
            "budget_decisions": decisions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan", help="print static 12-cell plan; no launch")
    plan.add_argument("--work-root", type=Path, required=True)
    plan.add_argument("--execution-preflight", type=Path, default=None)

    lane = sub.add_parser("run-lane", help="execute one reviewed static six-cell lane")
    lane.add_argument("--work-root", type=Path, required=True)
    lane.add_argument("--lane", type=int, choices=tuple(STATIC_LANES), required=True)
    lane.add_argument("--gpu", type=int, choices=tuple(STATIC_LANES), required=True)
    lane.add_argument("--execution-preflight", type=Path, required=True)

    aggregate = sub.add_parser("aggregate", help="receipt-only budget-wise three-fold gate")
    aggregate.add_argument("--work-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        result = build_plan(
            work_root=args.work_root,
            execution_preflight=args.execution_preflight,
        )
    elif args.mode == "run-lane":
        result = run_lane(
            work_root=args.work_root,
            lane=int(args.lane),
            gpu=int(args.gpu),
            execution_preflight=args.execution_preflight,
        )
    else:
        result = aggregate_pilot(work_root=args.work_root.resolve(), output=args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
