#!/usr/bin/env python3
"""Receipt-bound source-only executor for the six-arm H1-EST4 grid.

The default mode is plan-only: it validates the immutable five-date prepared
receipt and prints one exact source-training command.  It never starts tmux,
chooses a device, opens a target recording, or selects a date/arm from a
metric.  Training can start only with ``--execute-source-training`` and one
explicit ``CUDA_VISIBLE_DEVICES`` value supplied by the operator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_est4_launch_receipt import (
    DATES,
    EST4_ARMS,
    EST4_PREFLIGHT_SCHEMA,
    EST4_PREFLIGHT_STATUS,
    LAUNCH_RECEIPT_SCHEMA,
    LAUNCH_RECEIPT_STATUS,
    ROUTE,
    _immutable_json,
)
from src.h1_m4_cce_contract import sha256_file


class Est4SourceExecutorError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4SourceExecutorError(message)


def _experiment_name(arm: str) -> str:
    return f"h1_carrierid_date_lodo_est4_{arm.lower().replace('-', '_')}"


def build_command(
    *, launch_receipt: Path, outer_date: str, arm: str, run_dir: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Validate the prepared receipt and return one non-executed command."""

    date, normalized_arm = str(outer_date), str(arm).upper()
    _need(date in DATES, "EST4 executor date is outside the frozen five-date grid")
    _need(normalized_arm in EST4_ARMS, "EST4 executor arm is outside the frozen six-arm grid")
    launch_path, launch, launch_sha = _immutable_json(
        Path(launch_receipt), schema=LAUNCH_RECEIPT_SCHEMA, status=LAUNCH_RECEIPT_STATUS,
    )
    contract = launch.get("training_contract")
    _need(
        launch.get("route") == ROUTE
        and launch.get("explicit_operator_route") == ROUTE
        and launch.get("not_a_gpu_launcher") is True
        and launch.get("launch_authorized") is False
        and launch.get("proposed_arms_per_date") == list(EST4_ARMS)
        and isinstance(contract, Mapping)
        and contract.get("fresh_seed") == 42
        and contract.get("epochs") == 50
        and contract.get("fixed_terminal_epoch_zero_based") == 49
        and contract.get("warm_start_forbidden") is True
        and contract.get("target_score_selection") == "FORBIDDEN"
        and contract.get("deployment_target_optimizer_steps") == 0
        and contract.get("deployment_target_backward_steps") == 0,
        "EST4 prepared receipt lost its frozen source-only training contract",
    )
    _need(tuple(launch.get("est4_source_preflights", {})) == DATES,
          "EST4 prepared receipt does not bind exactly all five dates")
    row = launch["est4_source_preflights"].get(date)
    _need(isinstance(row, Mapping), "EST4 prepared receipt lacks this date preflight")
    preflight_path, preflight, preflight_sha = _immutable_json(
        Path(str(row.get("path", ""))), schema=EST4_PREFLIGHT_SCHEMA, status=EST4_PREFLIGHT_STATUS,
    )
    _need(preflight_sha == row.get("sha256") and preflight.get("outer_date") == date,
          "EST4 date preflight changed after the prepared receipt")
    _need(preflight.get("source_binding_sha256") == row.get("source_binding_sha256"),
          "EST4 prepared/source binding drift")
    controls = preflight.get("source_controls")
    _need(isinstance(controls, Mapping) and controls.get("all_arms") == list(EST4_ARMS)
          and controls.get("same_hs_hc_source_partition") is True
          and controls.get("same_source_windows") is True
          and controls.get("same_source_schedule") is True
          and controls.get("same_source_normalizer") is True
          and controls.get("same_fresh_seed") == 42
          and controls.get("fixed_terminal_epoch_zero_based") == 49,
          "EST4 source preflight lost the matched six-arm contract")
    config_path = ROOT / "configs" / "experiment" / f"{_experiment_name(normalized_arm)}.yaml"
    configured = preflight.get("configuration", {}).get(normalized_arm)
    _need(isinstance(configured, Mapping) and config_path.is_file()
          and configured.get("path") == str(config_path)
          and configured.get("sha256") == sha256_file(config_path),
          "EST4 arm configuration changed after source preflight")
    code = preflight.get("code_sha256")
    expected_code = {
        "data": ROOT / "src/data/h1_carrierid_date_lodo_est4.py",
        "model": ROOT / "src/models/h1_carrierid_date_lodo_est4_module.py",
        "component": ROOT / "src/models/components/h1_carrierid_est4_spint.py",
        "preflight": ROOT / "scripts/h1_carrierid_date_lodo_est4_preflight.py",
    }
    _need(isinstance(code, Mapping)
          and all(code.get(name) == sha256_file(path) for name, path in expected_code.items()),
          "EST4 source implementation changed after source preflight")
    frozen = preflight.get("frozen_estimator_initialization")
    _need(isinstance(frozen, Mapping), "EST4 source preflight lacks frozen estimator initialization")
    frozen_path = Path(str(frozen.get("path", ""))).resolve()
    _need(frozen_path.is_file() and frozen.get("sha256") == sha256_file(frozen_path)
          and row.get("frozen_estimator_initialization") == frozen,
          "EST4 frozen source plan changed after prepared receipt")
    source = preflight.get("source_binding")
    _need(isinstance(source, Mapping), "EST4 source preflight lacks source binding")
    phase1_path = Path(str(source.get("preflight_path", ""))).resolve()
    _need(phase1_path.is_file(), "EST4 Phase-1 source preflight is missing")
    aggregate = launch.get("five_date_aggregate")
    _need(isinstance(aggregate, Mapping), "EST4 prepared receipt lacks five-date aggregate")
    aggregate_path = Path(str(aggregate.get("path", ""))).resolve()
    _need(aggregate_path.is_file() and aggregate.get("sha256") == sha256_file(aggregate_path)
          and aggregate.get("source_date_screen_complete") is True
          and aggregate.get("numeric_results_interpreted") is False,
          "EST4 five-date prerequisite changed after prepared receipt")
    resolved_run_dir = Path(run_dir).resolve()
    _need(not resolved_run_dir.exists() and not resolved_run_dir.is_symlink()
          and not os.path.lexists(str(resolved_run_dir)),
          "EST4 source run directory must be new")
    command = [
        str(python_executable), str(ROOT / "src/train.py"),
        f"experiment={_experiment_name(normalized_arm)}",
        f"phase_est4.outer_date={date}",
        f"phase_est4.phase1_preflight_path={phase1_path}",
        f"phase_est4.est4_preflight_path={preflight_path}",
        f"phase_est4.five_date_aggregate_path={aggregate_path}",
        f"phase_est4.frozen_plan_path={frozen_path}",
        f"hydra.run.dir={resolved_run_dir}",
        "seed=42", "ckpt_path=null", "train=true", "test=false",
    ]
    return {
        "route": ROUTE, "outer_date": date, "arm": normalized_arm,
        "command": command, "run_dir": str(resolved_run_dir),
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "five_date_aggregate": {"path": str(aggregate_path), "sha256": aggregate["sha256"]},
        "source_binding_sha256": preflight["source_binding_sha256"],
        "contract": {"fresh_seed": 42, "epochs": 50, "terminal_epoch_zero_based": 49,
                     "warm_start": False, "target_optimizer_steps": 0,
                     "target_backward_steps": 0, "target_score_selection": "FORBIDDEN"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--outer-date", required=True, choices=DATES)
    parser.add_argument("--arm", required=True, choices=EST4_ARMS)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--execute-source-training", action="store_true")
    args = parser.parse_args()
    plan = build_command(
        launch_receipt=args.launch_receipt, outer_date=args.outer_date, arm=args.arm,
        run_dir=args.run_dir, python_executable=args.python_executable,
    )
    if not args.execute_source_training:
        print(json.dumps({**plan, "executed": False, "trainer_constructed": False,
                          "cuda_constructed": False}, sort_keys=True))
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _need(visible not in (None, "") and "," not in str(visible),
          "explicit EST4 source execution requires exactly one CUDA_VISIBLE_DEVICES value")
    completed = subprocess.run(plan["command"], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(json.dumps({**plan, "executed": True, "returncode": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
