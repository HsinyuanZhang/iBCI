#!/usr/bin/env python3
"""Fail-closed R4 M6/M12 expansion and final 15-fold aggregation.

The immutable three-fold pilot gate authorizes exactly the remaining twelve
folds for both carrier budgets.  This module freezes that 48-cell matrix,
splits matched Full/MB4 pairs evenly over the two local RTX 3090s, executes
each lane serially, and later produces one receipt-only 15-fold aggregate.
It never adds M18, arms, seeds, folds, or score-dependent work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    from scripts import rt_r4_pilot as pilot
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import rt_r4_pilot as pilot


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
RESULT_ROOT = REPO / "sua_exploration/results/rt_r4_budget_response_common_q24_v1"
WORK_ROOT = RESULT_ROOT / "gpu_runs"
PILOT_AGGREGATE = RESULT_ROOT / "RT_R4_BUDGETWISE_THREE_FOLD_PILOT_GATE_v1.json"
PILOT_AGGREGATE_SHA256 = "a9f38bb5eaa599031742c49120053068f07e869d918ffe987770c82bb2577689"
EXECUTION_PREFLIGHT = RESULT_ROOT / "RT_R4_EXECUTION_PREFLIGHT_v1.json"
EXECUTION_PREFLIGHT_SHA256 = "178d90a2787540af801136a069a6729411cb5a097075b261abe489dbaa32ef10"
DEFAULT_PREFLIGHT = RESULT_ROOT / "RT_R4_EXPANSION_PREFLIGHT_v1.json"
DEFAULT_LAUNCH_RECEIPT = RESULT_ROOT / "RT_R4_EXPANSION_LAUNCH_RECEIPT_v1.json"
DEFAULT_AGGREGATE = RESULT_ROOT / "RT_R4_M6_M12_FULL15_AGGREGATE_v1.json"

PREFLIGHT_SCHEMA = "rt_r4_m6_m12_expansion_preflight_v1"
PREFLIGHT_STATUS = "PASS_R4_M6_M12_EXPANSION_PREFLIGHT_NOT_LAUNCHED"
LAUNCH_SCHEMA = "rt_r4_m6_m12_static_dual3090_launch_receipt_v1"
LAUNCH_STATUS = "PASS_R4_M6_M12_STATIC_EXPANSION_LAUNCH_AUTHORIZED_NOT_STARTED"
LANE_SCHEMA = "rt_r4_m6_m12_expansion_lane_terminal_v1"
LANE_STATUS = "PASS_R4_M6_M12_EXPANSION_LANE_COMPLETE"
FINAL_SCHEMA = "rt_r4_m6_m12_full15_aggregate_v1"
FINAL_STATUS = "PASS_R4_M6_M12_FULL15_RECEIPT_ONLY_AGGREGATE"

FOLDS = tuple(range(15))
EXPANSION_FOLDS = tuple(pilot.EXPANSION_FOLDS)
BUDGETS = tuple(pilot.BUDGETS)
ARMS = tuple(pilot.ARMS)
SEED = pilot.SEED
TMUX = {0: "rt_r4_expansion_lane0_v1", 1: "rt_r4_expansion_lane1_v1"}


class RtR4ExpansionError(RuntimeError):
    """The frozen expansion, execution, or final aggregation contract failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RtR4ExpansionError(message)


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
    _need(not os.path.lexists(path), f"refusing to overwrite immutable R4 expansion artifact: {path}")
    payload = json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"binding source missing: {path}")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "sha256": _sha256(path),
    }


def _verify_binding(value: Mapping[str, Any], *, label: str) -> Path:
    _need(isinstance(value, Mapping) and isinstance(value.get("path"), str), f"{label} binding absent")
    path = Path(str(value["path"])).resolve()
    _need(path.is_file(), f"{label} missing: {path}")
    _need(_sha256(path) == value.get("sha256"), f"{label} SHA drift: {path}")
    _need(path.stat().st_size == int(value.get("size", -1)), f"{label} size drift: {path}")
    return path


def _lane_pairs() -> dict[int, tuple[tuple[int, int], ...]]:
    lanes: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for index, fold in enumerate(EXPANSION_FOLDS):
        lanes[index % 2].append((6, fold))
        lanes[1 - (index % 2)].append((12, fold))
    result = {lane: tuple(rows) for lane, rows in lanes.items()}
    expected = {(budget, fold) for budget in BUDGETS for fold in EXPANSION_FOLDS}
    actual = {row for rows in result.values() for row in rows}
    _need(actual == expected and sum(map(len, result.values())) == 24, "expansion pair grid is not exact")
    _need(not (set(result[0]) & set(result[1])), "expansion lanes overlap")
    for lane in (0, 1):
        _need(len(result[lane]) == 12, f"lane{lane} does not contain twelve matched pairs")
        _need(sum(budget == 6 for budget, _ in result[lane]) == 6, f"lane{lane} M6 imbalance")
        _need(sum(budget == 12 for budget, _ in result[lane]) == 6, f"lane{lane} M12 imbalance")
    return result


STATIC_LANES = _lane_pairs()


def lane_cells(lane: int) -> list[dict[str, Any]]:
    _need(lane in STATIC_LANES, f"invalid R4 expansion lane: {lane}")
    return [
        {"budget": budget, "fold": fold, "arm": arm, "seed": SEED}
        for budget, fold in STATIC_LANES[lane]
        for arm in ARMS
    ]


def lane_terminal(work_root: Path, lane: int) -> Path:
    return work_root.resolve() / f"expansion_lane_{lane}_terminal.json"


def run_lane_command(
    *, work_root: Path, lane: int, gpu: int, preflight: Path, launch_receipt: Path
) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "run-lane",
        "--work-root", str(work_root.resolve()), "--lane", str(lane), "--gpu", str(gpu),
        "--preflight", str(preflight.resolve()), "--launch-receipt", str(launch_receipt.resolve()),
    ]


def validate_pilot_aggregate(path: Path = PILOT_AGGREGATE) -> dict[str, Any]:
    _immutable(path, "R4 pilot aggregate")
    _need(path.resolve() == PILOT_AGGREGATE.resolve(), "another R4 pilot aggregate was supplied")
    _need(_sha256(path) == PILOT_AGGREGATE_SHA256, "R4 pilot aggregate SHA drift")
    body = _json(path)
    _need(body.get("schema") == pilot.AGGREGATE_SCHEMA and body.get("status") == pilot.AGGREGATE_STATUS,
          "R4 pilot aggregate schema/status drift")
    _need(body.get("scope") == "receipt_only_no_nwb_no_trainer_no_optimizer_no_cuda_no_launch",
          "R4 pilot aggregate exceeded receipt-only scope")
    decisions = body.get("budget_decisions")
    _need(isinstance(decisions, Mapping) and set(decisions) == {"6", "12"},
          "R4 pilot lacks exactly the two budget decisions")
    for budget in BUDGETS:
        row = decisions[str(budget)]
        _need(row.get("gate", {}).get("passed") is True
              and row.get("positive_folds") == 3
              and float(row.get("mean_full_minus_mb4", -math.inf)) >= pilot.MEAN_DELTA_GATE,
              f"M{budget} did not pass the frozen pilot gate")
        _need(tuple(row.get("folds", ())) == pilot.PILOT_FOLDS
              and tuple(row.get("expansion_folds", ())) == EXPANSION_FOLDS
              and row.get("decision") == "EXPAND_THIS_BUDGET_TO_REMAINING_12_FOLDS"
              and row.get("launch_performed_by_aggregator") is False,
              f"M{budget} expansion decision drift")
    return body


def _code_closure() -> dict[str, str]:
    relatives = (
        "scripts/rt_r4_expansion.py",
        "scripts/rt_r4_pilot.py",
        "src/rt_r4_budget_response_eval.py",
        "src/data/rt_r4_budget_response_datamodule.py",
        "tests/test_rt_r4_expansion.py",
    )
    closure: dict[str, str] = {}
    for relative in relatives:
        path = PROJECT / relative
        _need(path.is_file(), f"R4 expansion code closure missing: {relative}")
        closure[relative] = _sha256(path)
    return closure


def build_preflight(*, output: Path, work_root: Path = WORK_ROOT) -> dict[str, Any]:
    validate_pilot_aggregate()
    _immutable(EXECUTION_PREFLIGHT, "R4 execution preflight")
    _need(_sha256(EXECUTION_PREFLIGHT) == EXECUTION_PREFLIGHT_SHA256,
          "R4 execution preflight SHA drift")
    pilot.validate_execution_preflight(EXECUTION_PREFLIGHT)
    lanes: dict[str, Any] = {}
    matrix: list[dict[str, Any]] = []
    for lane in (0, 1):
        cells = lane_cells(lane)
        matrix.extend({**row, "lane": lane, "physical_gpu": lane} for row in cells)
        lanes[str(lane)] = {
            "physical_gpu": lane,
            "requires_gpu_name_contains": "3090",
            "matched_pairs": [list(row) for row in STATIC_LANES[lane]],
            "serial_cells": cells,
            "cell_count": len(cells),
        }
    expected = {
        (budget, fold, arm, SEED)
        for budget in BUDGETS for fold in EXPANSION_FOLDS for arm in ARMS
    }
    actual = {(row["budget"], row["fold"], row["arm"], row["seed"]) for row in matrix}
    _need(actual == expected and len(matrix) == 48, "R4 expansion matrix is not the exact 48 cells")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "objective": "execute_both_passing_R4_budgets_on_the_remaining_twelve_folds",
        "pilot_aggregate": _binding(PILOT_AGGREGATE),
        "execution_preflight": _binding(EXECUTION_PREFLIGHT),
        "execution_code_closure_sha256": _code_closure(),
        "work_root": str(work_root.resolve()),
        "seed": SEED,
        "activity_calibration_trials": 24,
        "common_query_start_trial": 24,
        "carrier_budgets": list(BUDGETS),
        "arms": list(ARMS),
        "pilot_folds_retained": list(pilot.PILOT_FOLDS),
        "expansion_folds": list(EXPANSION_FOLDS),
        "expansion_cell_count": 48,
        "final_fold_count_per_budget": 15,
        "static_dual_3090_lanes": lanes,
        "invariants": {
            "matched_full_mb4_pair_stays_on_one_gpu": True,
            "source_only_normalizer_contract": "verbatim RT R4 execution preflight",
            "target_carrier_contract": "activity M24; carrier prefix M6 or M12 only",
            "query_contract": "common q24; identical query-window identity within fold",
            "target_backpropagation": False,
            "model_state_sha256_before_equals_after_required": True,
            "no_score_dependent_stopping": True,
            "m18_authorized": False,
            "new_arms_seeds_or_folds_authorized": False,
            "aggregator_launches_additional_work": False,
        },
        "scope": {
            "nwb_files_opened": 0,
            "outer_target_payloads_opened": 0,
            "trainer_constructed": 0,
            "optimizer_constructed": 0,
            "cuda_queried": 0,
            "gpu_processes_started": 0,
            "tmux_sessions_created": 0,
            "commands_executed": 0,
        },
        "output": str(output.resolve()),
    }


def validate_preflight(path: Path) -> dict[str, Any]:
    _immutable(path, "R4 expansion preflight")
    body = _json(path)
    _need(body.get("schema") == PREFLIGHT_SCHEMA and body.get("status") == PREFLIGHT_STATUS,
          "R4 expansion preflight schema/status drift")
    pilot_path = _verify_binding(body.get("pilot_aggregate", {}), label="R4 pilot aggregate")
    _need(pilot_path == PILOT_AGGREGATE.resolve()
          and body["pilot_aggregate"]["sha256"] == PILOT_AGGREGATE_SHA256,
          "R4 expansion preflight binds another pilot aggregate")
    validate_pilot_aggregate(pilot_path)
    execution = _verify_binding(body.get("execution_preflight", {}), label="R4 execution preflight")
    _need(execution == EXECUTION_PREFLIGHT.resolve()
          and body["execution_preflight"]["sha256"] == EXECUTION_PREFLIGHT_SHA256,
          "R4 expansion preflight binds another execution preflight")
    pilot.validate_execution_preflight(execution)
    closure = body.get("execution_code_closure_sha256")
    _need(isinstance(closure, Mapping) and bool(closure), "R4 expansion code closure absent")
    for relative, expected in closure.items():
        source = PROJECT / str(relative)
        _need(source.is_file() and _sha256(source) == expected,
              f"R4 expansion code closure drift: {relative}")
    lanes = body.get("static_dual_3090_lanes")
    _need(isinstance(lanes, Mapping) and set(lanes) == {"0", "1"}, "R4 expansion lanes absent")
    for lane in (0, 1):
        _need(lanes[str(lane)].get("serial_cells") == lane_cells(lane)
              and int(lanes[str(lane)].get("physical_gpu", -1)) == lane,
              f"R4 expansion lane{lane} grid drift")
    _need(body.get("expansion_cell_count") == 48
          and tuple(body.get("expansion_folds", ())) == EXPANSION_FOLDS
          and body.get("invariants", {}).get("m18_authorized") is False,
          "R4 expansion scope drift")
    return body


def _gpu_snapshot() -> dict[str, Any]:
    try:
        table = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RtR4ExpansionError("cannot inspect local expansion GPUs") from error
    rows: dict[int, dict[str, Any]] = {}
    for line in table.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit():
            rows[int(fields[0])] = {"index": int(fields[0]), "name": fields[1], "uuid": fields[2]}
    for gpu in (0, 1):
        _need(gpu in rows and "3090" in rows[gpu]["name"], f"GPU{gpu} is not the required RTX 3090")
        _name, owners, rt_writers = pilot._active_gpu_and_rt_commands(gpu)
        _need(not owners, f"GPU{gpu} has active compute owners: {owners}")
        _need(not [row for row in rt_writers if "rt_r4_expansion.py" not in row],
              f"GPU{gpu} sees an unexpected active R4 writer")
        rows[gpu]["active_compute_owners"] = []
    return {str(gpu): rows[gpu] for gpu in (0, 1)}


def build_launch_receipt(
    *, output: Path, preflight: Path, work_root: Path = WORK_ROOT
) -> dict[str, Any]:
    body = validate_preflight(preflight)
    _need(Path(str(body["work_root"])).resolve() == work_root.resolve(),
          "R4 expansion launch work root differs from preflight")
    commands = {
        str(lane): run_lane_command(
            work_root=work_root, lane=lane, gpu=lane, preflight=preflight, launch_receipt=output
        )
        for lane in (0, 1)
    }
    for lane in (0, 1):
        _need(not os.path.lexists(lane_terminal(work_root, lane)),
              f"R4 expansion lane{lane} terminal already exists before launch receipt")
    return {
        "schema": LAUNCH_SCHEMA,
        "status": LAUNCH_STATUS,
        "authorization_source": "root_pre_registered_both_passing_budgets_expansion_instruction",
        "expansion_preflight": _binding(preflight),
        "pilot_aggregate": _binding(PILOT_AGGREGATE),
        "work_root": str(work_root.resolve()),
        "gpu_snapshot_before_launch": _gpu_snapshot(),
        "tmux_sessions": {str(lane): TMUX[lane] for lane in (0, 1)},
        "commands": commands,
        "static_cell_count": 48,
        "lane_cell_counts": {"0": 24, "1": 24},
        "prohibitions": {
            "m18": True,
            "new_arm": True,
            "new_seed": True,
            "fold_outside_remaining_twelve": True,
            "result_conditioned_tuning": True,
            "automatic_post_aggregate_launch": True,
        },
        "scope": {
            "result_values_read": False,
            "gpu_processes_started": 0,
            "tmux_sessions_created": 0,
            "aggregate_started": False,
        },
        "output": str(output.resolve()),
    }


def validate_launch_receipt(path: Path, *, preflight: Path) -> dict[str, Any]:
    _immutable(path, "R4 expansion launch receipt")
    body = _json(path)
    _need(body.get("schema") == LAUNCH_SCHEMA and body.get("status") == LAUNCH_STATUS,
          "R4 expansion launch receipt schema/status drift")
    bound = _verify_binding(body.get("expansion_preflight", {}), label="R4 expansion preflight")
    _need(bound == preflight.resolve(), "R4 expansion launch receipt binds another preflight")
    validate_preflight(bound)
    pilot_path = _verify_binding(body.get("pilot_aggregate", {}), label="R4 pilot aggregate")
    _need(pilot_path == PILOT_AGGREGATE.resolve()
          and body["pilot_aggregate"]["sha256"] == PILOT_AGGREGATE_SHA256,
          "R4 expansion launch receipt lost pilot gate binding")
    _need(body.get("static_cell_count") == 48
          and body.get("lane_cell_counts") == {"0": 24, "1": 24}
          and body.get("scope", {}).get("result_values_read") is False,
          "R4 expansion launch scope drift")
    for lane in (0, 1):
        expected = run_lane_command(
            work_root=Path(str(body["work_root"])), lane=lane, gpu=lane,
            preflight=preflight, launch_receipt=path,
        )
        _need(body.get("commands", {}).get(str(lane)) == expected,
              f"R4 expansion lane{lane} command drift")
        snap = body.get("gpu_snapshot_before_launch", {}).get(str(lane), {})
        _need(int(snap.get("index", -1)) == lane and "3090" in str(snap.get("name", ""))
              and snap.get("active_compute_owners") == [],
              f"R4 expansion GPU{lane} launch snapshot drift")
    return body


def _process_rows() -> list[str]:
    return subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()


def _tmux_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def launch(*, preflight: Path, launch_receipt: Path) -> dict[str, Any]:
    receipt = validate_launch_receipt(launch_receipt, preflight=preflight)
    work_root = Path(str(receipt["work_root"])).resolve()
    rows = _process_rows()
    for lane in (0, 1):
        _need(not _tmux_alive(TMUX[lane]), f"R4 expansion tmux already exists: {TMUX[lane]}")
        _need(not any("rt_r4_expansion.py run-lane" in row and f"--lane {lane}" in row for row in rows),
              f"R4 expansion lane{lane} process already exists")
        _need(not os.path.lexists(lane_terminal(work_root, lane)),
              f"R4 expansion lane{lane} terminal already exists")
    snapshot = _gpu_snapshot()
    for lane in (0, 1):
        _need(snapshot[str(lane)]["uuid"]
              == receipt["gpu_snapshot_before_launch"][str(lane)]["uuid"],
              f"R4 expansion GPU{lane} identity changed after launch receipt")
    started: dict[str, Any] = {}
    for lane in (0, 1):
        command = receipt["commands"][str(lane)]
        log = RESULT_ROOT / f"rt_r4_expansion_lane{lane}_v1.log"
        shell = f"exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
        completed = subprocess.run(
            ["tmux", "new-session", "-d", "-s", TMUX[lane], shell],
            text=True, capture_output=True,
        )
        _need(completed.returncode == 0,
              f"failed to launch R4 expansion lane{lane}: {completed.stderr.strip()}")
        started[str(lane)] = {"tmux": TMUX[lane], "log": str(log), "command": command}
    return {
        "status": "PASS_R4_M6_M12_BOTH_STATIC_EXPANSION_LANES_STARTED",
        "launch_receipt": _binding(launch_receipt),
        "started": started,
        "m18_started": False,
        "cell_count": 48,
    }


def _validate_lane_terminal(path: Path, *, lane: int, preflight: Path, launch_receipt: Path) -> dict[str, Any]:
    _immutable(path, f"R4 expansion lane{lane} terminal")
    body = _json(path)
    _need(body.get("schema") == LANE_SCHEMA and body.get("status") == LANE_STATUS
          and int(body.get("lane", -1)) == lane and int(body.get("physical_gpu", -1)) == lane,
          f"R4 expansion lane{lane} terminal identity drift")
    _need(body.get("cells") == lane_cells(lane) and len(body.get("cells", ())) == 24,
          f"R4 expansion lane{lane} terminal cell grid drift")
    _need(_verify_binding(body.get("expansion_preflight", {}), label="R4 expansion preflight")
          == preflight.resolve(), f"R4 expansion lane{lane} preflight drift")
    _need(_verify_binding(body.get("launch_receipt", {}), label="R4 expansion launch receipt")
          == launch_receipt.resolve(), f"R4 expansion lane{lane} launch receipt drift")
    return body


def run_lane(
    *, work_root: Path, lane: int, gpu: int, preflight: Path, launch_receipt: Path
) -> dict[str, Any]:
    _need(lane in (0, 1) and gpu == lane, "R4 expansion statically binds lane0/GPU0 and lane1/GPU1")
    preflight_body = validate_preflight(preflight)
    receipt = validate_launch_receipt(launch_receipt, preflight=preflight)
    root = work_root.resolve()
    _need(root == Path(str(preflight_body["work_root"])).resolve()
          and root == Path(str(receipt["work_root"])).resolve(), "R4 expansion work-root drift")
    terminal = lane_terminal(root, lane)
    if terminal.exists():
        body = _validate_lane_terminal(
            terminal, lane=lane, preflight=preflight, launch_receipt=launch_receipt
        )
        return {"status": body["status"], "output": str(terminal), "sha256": _sha256(terminal)}
    pilot._prepare_work_root(root, execution_preflight=EXECUTION_PREFLIGHT)
    name, owners, _writers = pilot._active_gpu_and_rt_commands(gpu)
    _need("3090" in name and not owners, f"R4 expansion GPU{gpu} is not exclusively available: {owners}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    summary: dict[str, str] = {}
    for spec in lane_cells(lane):
        current_name, current_owners, writers = pilot._active_gpu_and_rt_commands(gpu)
        _need(current_name == name and not current_owners,
              f"R4 expansion GPU{gpu} identity/ownership changed")
        collisions = pilot._same_cell_writer(
            writers, budget=spec["budget"], arm=spec["arm"], fold=spec["fold"]
        )
        _need(not collisions, f"duplicate active R4 expansion cell writer: {collisions}")
        key = f"m{spec['budget']}:{spec['arm']}:fold{spec['fold']}:seed{SEED}"
        summary[key] = pilot.run_cell(
            root=root, budget=spec["budget"], arm=spec["arm"], fold=spec["fold"],
            gpu=gpu, env=env,
        )
    body = {
        "schema": LANE_SCHEMA,
        "status": LANE_STATUS,
        "lane": lane,
        "physical_gpu": gpu,
        "gpu_name": name,
        "cells": lane_cells(lane),
        "summary": summary,
        "expansion_preflight": _binding(preflight),
        "launch_receipt": _binding(launch_receipt),
        "pilot_aggregate": _binding(PILOT_AGGREGATE),
        "scope": {"m18_started": False, "added_cells": 0, "score_conditioned_actions": 0},
    }
    digest = _write_immutable(terminal, body)
    _validate_lane_terminal(terminal, lane=lane, preflight=preflight, launch_receipt=launch_receipt)
    return {"status": LANE_STATUS, "output": str(terminal), "sha256": digest}


def _sign_test_two_sided(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    tail = sum(math.comb(n, index) for index in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def aggregate_final(
    *, work_root: Path, preflight: Path, launch_receipt: Path, output: Path
) -> dict[str, Any]:
    _need(not os.path.lexists(output), f"refusing to overwrite final R4 aggregate: {output}")
    validate_preflight(preflight)
    validate_launch_receipt(launch_receipt, preflight=preflight)
    for lane in (0, 1):
        _validate_lane_terminal(
            lane_terminal(work_root, lane), lane=lane,
            preflight=preflight, launch_receipt=launch_receipt,
        )
    per_fold_query: dict[int, str] = {}
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        rows: list[dict[str, Any]] = []
        deltas: list[float] = []
        for fold in FOLDS:
            full_terminal, full = pilot.validate_cell_terminal(
                root=work_root, budget=budget, arm="afc4_vel", fold=fold
            )
            mb_terminal, mb4 = pilot.validate_cell_terminal(
                root=work_root, budget=budget, arm="afc4_mb4", fold=fold
            )
            for key in (
                "outer_target_session", "query_window_identity_sha256",
                "query_windows_evaluated", "source_only_normalizer_sha256",
            ):
                _need(full.get(key) == mb4.get(key), f"M{budget} fold{fold} Full/MB4 mismatch: {key}")
            full_raw = full["target_carrier_prefix_fit"]["raw_prefix_descriptor_sha256"]
            mb_raw = mb4["target_carrier_prefix_fit"]["raw_prefix_descriptor_sha256"]
            _need(full_raw == mb_raw, f"M{budget} fold{fold} raw carrier mismatch")
            query_hash = str(full["query_window_identity_sha256"])
            if fold in per_fold_query:
                _need(per_fold_query[fold] == query_hash, f"fold{fold} query hash differs across budgets")
            else:
                per_fold_query[fold] = query_hash
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
                "full_terminal_sha256": _sha256(pilot.cell_paths(
                    work_root, budget=budget, arm="afc4_vel", fold=fold
                ).terminal),
                "mb4_terminal_sha256": _sha256(pilot.cell_paths(
                    work_root, budget=budget, arm="afc4_mb4", fold=fold
                ).terminal),
                "state_identity_verified": (
                    full_terminal["model_state_sha256_before"] == full_terminal["model_state_sha256_after"]
                    and mb_terminal["model_state_sha256_before"] == mb_terminal["model_state_sha256_after"]
                ),
            }
            _need(row["state_identity_verified"], f"M{budget} fold{fold} state identity failed")
            rows.append(row)
            all_rows.append(row)
            deltas.append(delta)
        positive = sum(value > 0 for value in deltas)
        negative = sum(value < 0 for value in deltas)
        standard_deviation = statistics.stdev(deltas)
        summaries[str(budget)] = {
            "budget": budget,
            "folds": list(FOLDS),
            "n_folds": len(FOLDS),
            "signed_full_minus_mb4": deltas,
            "mean_full_minus_mb4": statistics.fmean(deltas),
            "median_full_minus_mb4": statistics.median(deltas),
            "sample_standard_deviation": standard_deviation,
            "paired_standard_error": standard_deviation / math.sqrt(len(deltas)),
            "positive_folds": positive,
            "negative_folds": negative,
            "zero_folds": len(deltas) - positive - negative,
            "two_sided_exact_sign_p": _sign_test_two_sided(positive, negative),
            "minimum_delta": min(deltas),
            "maximum_delta": max(deltas),
            "rows": rows,
        }
    body = {
        "schema": FINAL_SCHEMA,
        "status": FINAL_STATUS,
        "scope": "receipt_only_no_nwb_no_trainer_no_optimizer_no_cuda_no_launch",
        "pilot_aggregate": _binding(PILOT_AGGREGATE),
        "expansion_preflight": _binding(preflight),
        "launch_receipt": _binding(launch_receipt),
        "expansion_lane_terminals": {
            str(lane): _binding(lane_terminal(work_root, lane)) for lane in (0, 1)
        },
        "seed": SEED,
        "activity_calibration_trials": 24,
        "common_query_start_trial": 24,
        "carrier_budgets": list(BUDGETS),
        "folds": list(FOLDS),
        "arms": list(ARMS),
        "total_cell_count": 60,
        "pilot_cell_count_retained": 12,
        "expansion_cell_count": 48,
        "all_cells_complete": True,
        "per_fold_common_query_sha256_across_budgets_and_arms": {
            str(fold): per_fold_query[fold] for fold in FOLDS
        },
        "budget_summaries": summaries,
        "rows": all_rows,
        "post_aggregate_action": {
            "automatic_launch": False,
            "m18_started": False,
            "new_arms_or_seeds_started": False,
            "requires_root_review": True,
        },
    }
    digest = _write_immutable(output, body)
    return {"status": FINAL_STATUS, "output": str(output), "sha256": digest,
            "budget_summaries": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    pre = sub.add_parser("prepare-preflight")
    pre.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    pre.add_argument("--work-root", type=Path, default=WORK_ROOT)
    launch_receipt_parser = sub.add_parser("prepare-launch-receipt")
    launch_receipt_parser.add_argument("--output", type=Path, default=DEFAULT_LAUNCH_RECEIPT)
    launch_receipt_parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    launch_receipt_parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    launch_parser.add_argument("--launch-receipt", type=Path, default=DEFAULT_LAUNCH_RECEIPT)
    lane_parser = sub.add_parser("run-lane")
    lane_parser.add_argument("--work-root", required=True, type=Path)
    lane_parser.add_argument("--lane", required=True, type=int, choices=(0, 1))
    lane_parser.add_argument("--gpu", required=True, type=int, choices=(0, 1))
    lane_parser.add_argument("--preflight", required=True, type=Path)
    lane_parser.add_argument("--launch-receipt", required=True, type=Path)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    aggregate_parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    aggregate_parser.add_argument("--launch-receipt", type=Path, default=DEFAULT_LAUNCH_RECEIPT)
    aggregate_parser.add_argument("--output", type=Path, default=DEFAULT_AGGREGATE)
    args = parser.parse_args()
    if args.mode == "prepare-preflight":
        body = build_preflight(output=args.output.resolve(), work_root=args.work_root.resolve())
        digest = _write_immutable(args.output.resolve(), body)
        result = {"status": PREFLIGHT_STATUS, "output": str(args.output.resolve()), "sha256": digest}
    elif args.mode == "prepare-launch-receipt":
        body = build_launch_receipt(
            output=args.output.resolve(), preflight=args.preflight.resolve(),
            work_root=args.work_root.resolve(),
        )
        digest = _write_immutable(args.output.resolve(), body)
        result = {"status": LAUNCH_STATUS, "output": str(args.output.resolve()), "sha256": digest}
    elif args.mode == "launch":
        result = launch(preflight=args.preflight.resolve(), launch_receipt=args.launch_receipt.resolve())
    elif args.mode == "run-lane":
        result = run_lane(
            work_root=args.work_root.resolve(), lane=args.lane, gpu=args.gpu,
            preflight=args.preflight.resolve(), launch_receipt=args.launch_receipt.resolve(),
        )
    else:
        result = aggregate_final(
            work_root=args.work_root.resolve(), preflight=args.preflight.resolve(),
            launch_receipt=args.launch_receipt.resolve(), output=args.output.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
