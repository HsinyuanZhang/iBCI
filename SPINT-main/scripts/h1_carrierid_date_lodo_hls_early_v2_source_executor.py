#!/usr/bin/env python3
"""Static two-GPU dry-run planner and explicit H-LS early-v2 executor.

The default path prints a complete plan and starts nothing.  Explicit source
execution atomically publishes one permanent writer claim per date before it
starts two owner workers.  Each owner runs its non-overlapping dates serially;
the two owners may run concurrently on distinct physical GPUs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES, EARLY_LAUNCH_SCHEMA, EARLY_LAUNCH_STATUS, EARLY_PREFLIGHT_SCHEMA,
    EARLY_PREFLIGHT_STATUS, STATIC_PARTITIONS, need, read_immutable_json,
    sha256_file, write_immutable_json,
)


def validate_assignments(assignments: Mapping[str, Mapping[str, Any]]) -> None:
    """Require two distinct GPU owners and exactly one writer per date."""

    need(tuple(assignments) == tuple(STATIC_PARTITIONS),
         "early-v2 executor requires the two frozen GPU owners")
    devices: list[str] = []
    dates: list[str] = []
    for owner, expected_dates in STATIC_PARTITIONS.items():
        row = assignments[owner]
        need(isinstance(row, Mapping), f"early-v2 owner assignment is malformed: {owner}")
        device = str(row.get("physical_gpu", ""))
        need(device != "", f"early-v2 owner has no physical GPU: {owner}")
        devices.append(device); dates.extend(str(value) for value in row.get("dates", ()))
    need(len(devices) == len(set(devices)),
         "early-v2 GPU owners cannot share the same physical GPU")
    need(len(dates) == len(set(dates)) == len(DATES) and set(dates) == set(DATES),
         "early-v2 same-date writer conflict or incomplete five-date partition")
    for owner, expected_dates in STATIC_PARTITIONS.items():
        need(tuple(assignments[owner].get("dates", ())) == expected_dates,
             f"early-v2 owner partition drift: {owner}")


def build_plan(*, launch_receipt: Path, run_root: Path, claim_root: Path,
               gpu0_device: str, gpu1_device: str,
               python_executable: str = sys.executable) -> dict[str, Any]:
    launch_path, launch, launch_sha = read_immutable_json(
        launch_receipt, schema=EARLY_LAUNCH_SCHEMA, status=EARLY_LAUNCH_STATUS,
    )
    need(tuple(launch.get("fixed_grid", ())) == DATES
         and launch.get("does_not_claim_upstream_complete") is True
         and launch.get("target_gate") == "CLOSED_UNTIL_POST_UPSTREAM_BINDER",
         "early-v2 launch receipt grid/upstream/target policy drift")
    need(launch.get("static_two_gpu_partitions")
         == {owner: list(dates) for owner, dates in STATIC_PARTITIONS.items()},
         "early-v2 launch receipt static partitions drift")
    assignments = {
        "local3090_gpu0": {"physical_gpu": str(gpu0_device), "dates": STATIC_PARTITIONS["local3090_gpu0"]},
        "local3090_gpu1": {"physical_gpu": str(gpu1_device), "dates": STATIC_PARTITIONS["local3090_gpu1"]},
    }
    validate_assignments(assignments)
    run_root, claim_root = Path(run_root).resolve(), Path(claim_root).resolve()
    rows: dict[str, Any] = {}
    for owner, assignment in assignments.items():
        for date in assignment["dates"]:
            receipt_row = launch.get("source_preflights", {}).get(date)
            need(isinstance(receipt_row, Mapping), f"early-v2 launch lacks source preflight: {date}")
            preflight_path, preflight, preflight_sha = read_immutable_json(
                receipt_row.get("path", ""), schema=EARLY_PREFLIGHT_SCHEMA, status=EARLY_PREFLIGHT_STATUS,
            )
            need(preflight_sha == receipt_row.get("sha256") and preflight.get("outer_date") == date,
                 f"early-v2 source preflight changed after launch receipt: {date}")
            need(preflight.get("source_binding_sha256") == receipt_row.get("source_binding_sha256"),
                 f"early-v2 source binding drift after launch receipt: {date}")
            closure = preflight.get("code_sha256")
            expected = {
                "early_data": ROOT / "src/data/h1_carrierid_date_lodo_hls_early_v2.py",
                "hls_data": ROOT / "src/data/h1_carrierid_date_lodo_hls.py",
                "model": ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py",
                "component": ROOT / "src/models/components/h1_carrierid_spint.py",
                "experiment": ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_early_v2.yaml",
                "data_config": ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls_early_v2.yaml",
                "model_config": ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml",
                "terminal_callback": ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml",
            }
            need(isinstance(closure, Mapping)
                 and all(closure.get(name) == sha256_file(path) for name, path in expected.items()),
                 f"early-v2 code/config changed after source preflight: {date}")
            run_dir = run_root / date
            claim_path = claim_root / f"H1_HLS_EARLY_V2_{date}_WRITER_CLAIM_v1.json"
            need(not run_dir.exists() and not run_dir.is_symlink() and not os.path.lexists(str(run_dir)),
                 f"early-v2 run directory already exists: {date}")
            need(not claim_path.exists() and not claim_path.is_symlink() and not os.path.lexists(str(claim_path)),
                 f"early-v2 same-date writer claim already exists: {date}")
            phase1 = Path(str(preflight["phase1_preflight"]["path"])).resolve()
            command = [
                str(python_executable), str(ROOT / "src/train.py"),
                "experiment=h1_carrierid_date_lodo_hls_early_v2",
                f"phase2.outer_date={date}", f"phase2.phase1_preflight_path={phase1}",
                f"phase2.hls_source_preflight_path={preflight_path}",
                f"hydra.run.dir={run_dir}", "seed=42", "ckpt_path=null", "train=true", "test=false",
            ]
            rows[date] = {
                "owner": owner, "physical_gpu": assignment["physical_gpu"],
                "run_dir": str(run_dir), "writer_claim": str(claim_path),
                "source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
                "source_binding_sha256": preflight["source_binding_sha256"],
                "command": command,
            }
    need(tuple(rows) == tuple(date for owner in STATIC_PARTITIONS for date in STATIC_PARTITIONS[owner]),
         "early-v2 plan row order differs from static owner partitions")
    return {
        "schema": "h1_carrierid_date_lodo_hls_early_v2_static_two_gpu_execution_plan_v1",
        "status": "DRY_RUN_NOT_EXECUTED", "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "assignments": assignments, "dates": rows,
        "contract": {"fresh_each_cell": True, "warm_start": False, "seed": 42,
                     "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
                     "one_writer_per_date": True, "target_opened": 0, "target_bytes_read": 0},
    }


def _publish_claims(plan: Mapping[str, Any]) -> None:
    # Publish every claim before any subprocess starts.  A collision leaves no
    # ambiguous second writer; already-published claims remain forensic state.
    for date, row in plan["dates"].items():
        write_immutable_json(row["writer_claim"], {
            "schema": "h1_carrierid_date_lodo_hls_early_v2_writer_claim_v1",
            "status": "CLAIMED_BEFORE_SOURCE_PROCESS_START",
            "outer_date": date, "owner": row["owner"], "physical_gpu": row["physical_gpu"],
            "run_dir": row["run_dir"], "source_preflight": row["source_preflight"],
            "source_binding_sha256": row["source_binding_sha256"],
            "target_recordings_opened": 0, "target_bytes_read": 0,
        })


def _run_owner(owner: str, assignment: Mapping[str, Any], rows: Mapping[str, Any]) -> None:
    for date in assignment["dates"]:
        row = rows[date]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(assignment["physical_gpu"])
        completed = subprocess.run(row["command"], cwd=ROOT, env=environment, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"early-v2 source training failed: {owner}/{date}/{completed.returncode}")


def execute(plan: Mapping[str, Any]) -> None:
    need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""),
         "early-v2 two-GPU executor requires parent CUDA_VISIBLE_DEVICES unset")
    _publish_claims(plan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_owner, owner, assignment, plan["dates"])
                   for owner, assignment in plan["assignments"].items()]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--claim-root", required=True, type=Path)
    parser.add_argument("--gpu0-device", default="0")
    parser.add_argument("--gpu1-device", default="1")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--execute-source-training", action="store_true")
    args = parser.parse_args()
    plan = build_plan(launch_receipt=args.launch_receipt, run_root=args.run_root,
                      claim_root=args.claim_root, gpu0_device=args.gpu0_device,
                      gpu1_device=args.gpu1_device, python_executable=args.python_executable)
    if not args.execute_source_training:
        print(json.dumps(plan, sort_keys=True)); return
    execute(plan)
    print(json.dumps({**plan, "status": "SOURCE_TRAINING_COMPLETED"}, sort_keys=True))


if __name__ == "__main__":
    main()
