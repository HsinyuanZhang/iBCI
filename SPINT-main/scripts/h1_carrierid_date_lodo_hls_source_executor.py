#!/usr/bin/env python3
"""Receipt-bound source-only H-LS command builder and explicit executor.

Without ``--execute-source-training`` this program only prints the exact
command.  The explicit path launches one fresh date run, never tmux and never
a target evaluator.  Parallel/device assignment remains an operator decision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, LAUNCH_SCHEMA, LAUNCH_STATUS, SOURCE_PREFLIGHT_SCHEMA, SOURCE_PREFLIGHT_STATUS,
    read_immutable_json, sha256_file,
)


class HlsSourceExecutorError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsSourceExecutorError(message)


def build_command(*, launch_receipt: Path, outer_date: str, run_dir: Path,
                  python_executable: str = sys.executable) -> dict[str, Any]:
    _need(str(outer_date) in DATES, "H-LS executor date is outside the frozen grid")
    launch_path, launch, launch_sha = read_immutable_json(
        launch_receipt, schema=LAUNCH_SCHEMA, status=LAUNCH_STATUS,
    )
    _need(tuple(launch.get("fixed_grid", ())) == DATES,
          "H-LS launch receipt does not bind all five dates")
    contract = launch.get("training_contract")
    _need(launch.get("route") == "H1-HLS-FIVEDATE"
          and launch.get("not_a_gpu_launcher") is True
          and launch.get("launch_authorized") is False
          and isinstance(contract, Mapping) and contract.get("arm") == "H-LS"
          and contract.get("fresh_seed") == 42 and contract.get("epochs") == 50
          and contract.get("fixed_terminal_epoch_zero_based") == 49
          and contract.get("checkpoint_warm_start") is False,
          "H-LS prepared receipt lost its explicit fresh source-training contract")
    row = launch.get("source_preflights", {}).get(str(outer_date))
    _need(isinstance(row, Mapping), "H-LS launch receipt lacks this date preflight")
    preflight_path, preflight, preflight_sha = read_immutable_json(
        row.get("path", ""), schema=SOURCE_PREFLIGHT_SCHEMA, status=SOURCE_PREFLIGHT_STATUS,
    )
    _need(preflight_sha == row.get("sha256") and preflight.get("outer_date") == str(outer_date),
          "H-LS date preflight changed after five-date launch receipt")
    _need(preflight.get("source_binding_sha256") == row.get("source_binding_sha256"),
          "H-LS launch/source binding drift")
    code = preflight.get("code_sha256")
    expected = {
        "data": ROOT / "src/data/h1_carrierid_date_lodo_hls.py",
        "model": ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py",
        "component": ROOT / "src/models/components/h1_carrierid_spint.py",
        "experiment": ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_phase2.yaml",
        "data_config": ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls.yaml",
        "model_config": ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml",
        "terminal_callback": ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml",
    }
    _need(isinstance(code, Mapping)
          and all(code.get(name) == sha256_file(path) for name, path in expected.items()),
          "H-LS implementation/config changed after source preflight")
    phase1 = Path(str(preflight["source_binding"]["preflight_path"])).resolve()
    run_dir = Path(run_dir).resolve()
    _need(not run_dir.exists() and not run_dir.is_symlink() and not os.path.lexists(str(run_dir)),
          "H-LS source run directory must be new")
    command = [
        str(python_executable), str(ROOT / "src/train.py"),
        "experiment=h1_carrierid_date_lodo_hls_phase2",
        f"phase2.outer_date={outer_date}",
        f"phase2.phase1_preflight_path={phase1}",
        f"phase2.hls_source_preflight_path={preflight_path}",
        f"hydra.run.dir={run_dir}",
        "seed=42", "ckpt_path=null", "train=true", "test=false",
    ]
    return {
        "outer_date": str(outer_date), "arm": "H-LS", "command": command,
        "run_dir": str(run_dir), "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "source_binding_sha256": preflight["source_binding_sha256"],
        "contract": {"fresh_seed": 42, "epochs": 50, "terminal_epoch_zero_based": 49,
                     "warm_start": False, "target_optimizer_steps": 0, "target_backward_steps": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--outer-date", required=True, choices=DATES)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--execute-source-training", action="store_true")
    args = parser.parse_args()
    plan = build_command(launch_receipt=args.launch_receipt, outer_date=args.outer_date,
                         run_dir=args.run_dir, python_executable=args.python_executable)
    if not args.execute_source_training:
        print(json.dumps({**plan, "executed": False, "trainer_constructed": False,
                          "cuda_constructed": False}, sort_keys=True))
        return
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""),
          "explicit H-LS source execution requires one selected CUDA_VISIBLE_DEVICES value")
    completed = subprocess.run(plan["command"], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(json.dumps({**plan, "executed": True, "returncode": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
