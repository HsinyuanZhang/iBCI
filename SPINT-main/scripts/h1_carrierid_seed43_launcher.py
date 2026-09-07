#!/usr/bin/env python3
"""Build, but never execute, the three fixed-epoch H1 seed-43 commands.

The launcher is intentionally plan-only.  Root audit must happen after the
local three-arm source preflight, the remote H-C0-only source preflight, and
the predecessor H-RS/H-LS terminal audit.  It performs no target access and
does not allocate a GPU or create a tmux session.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    sha256_file,
)
from scripts.h1_carrierid_paired_launcher import _verify_source_closure
from scripts.h1_carrierid_seed43_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS


def _train_command(*, root: Path, python_bin: Path, experiment: str, output_dir: Path, run_id: str) -> list[str]:
    return [
        str(python_bin),
        str(root / "src/train.py"),
        f"experiment={experiment}",
        f"hydra.run.dir={output_dir}",
        f"paths.root_dir={root}",
        f"paths.work_dir={root}",
        f"paths.data_dir={root / 'data'}",
        f"pilot.shared_cache_dir={output_dir.parent / 'shared_source_cache' / run_id}",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.max_epochs=50",
        "trainer.min_epochs=50",
        "trainer.precision=32-true",
        "model.optimizer.lr=5e-5",
        "seed=43",
        "test=false",
        "ckpt_path=null",
    ]


def prepare_plan(
    *,
    local_preflight_path: str | Path,
    local_output_root: str | Path,
    local_python_bin: str | Path,
    remote_root: str | Path,
    remote_output_root: str | Path,
    remote_python_bin: str | Path,
    remote_preflight_path: str | Path,
) -> dict[str, Any]:
    """Return the fixed command plan after exact local source-closure binding."""

    local_preflight_path = Path(local_preflight_path).resolve()
    receipt = assert_immutable_receipt(local_preflight_path, PREFLIGHT_STATUS)
    if receipt.get("schema") != PREFLIGHT_SCHEMA or receipt.get("launch", {}).get("launch_authorized") is not False:
        raise NormalizedV2ContractError("seed43 launcher requires immutable local nonlaunch preflight")
    if receipt.get("fixed_protocol", {}).get("requested_arms") != ["hs", "hc", "hc0"]:
        raise NormalizedV2ContractError("local seed43 preflight must cover H-S, H-C and H-C0")
    closure = _verify_source_closure(ROOT, receipt.get("source_sha256", {}))

    local_root = ROOT
    local_output = Path(local_output_root).resolve()
    local_python = Path(local_python_bin).resolve()
    remote_root = Path(remote_root)
    remote_output = Path(remote_output_root)
    remote_python = Path(remote_python_bin)
    if not local_python.is_file():
        raise FileNotFoundError(local_python)
    if local_output.exists():
        raise FileExistsError(f"seed43 local output root must be fresh: {local_output}")

    commands = {
        "h_s_local_gpu0": _train_command(
            root=local_root, python_bin=local_python, experiment="h1_carrierid_hs_seed43",
            output_dir=local_output / "hs", run_id="hs",
        ),
        "h_c_local_gpu1": _train_command(
            root=local_root, python_bin=local_python, experiment="h1_carrierid_hc_seed43",
            output_dir=local_output / "hc", run_id="hc",
        ),
        "h_c0_remote_gpu0": _train_command(
            root=remote_root, python_bin=remote_python, experiment="h1_carrierid_hc0_seed43",
            output_dir=remote_output / "hc0", run_id="hc0",
        ),
    }
    return {
        "schema": "h1_carrierid_h32_seed43_three_arm_launch_plan_v1",
        "status": "PREPARED_NOT_LAUNCHED",
        "local_preflight": {
            "path": str(local_preflight_path),
            "sha256": sha256_file(local_preflight_path),
            "source_closure": closure,
        },
        "remote_required_preflight": {
            "path": str(remote_preflight_path),
            "required_status": PREFLIGHT_STATUS,
            "required_requested_arms": ["hc0"],
            "must_be_immutable_and_audited_before_launch": True,
        },
        "fixed_protocol": {
            "fold_date": "19250101",
            "seed": 43,
            "source_scope": "exactly 11 fold0 source held-in calibration recordings",
            "target_minival_formal_evalai": "forbidden during fit",
            "support_trials": 4,
            "epochs": 50,
            "checkpoint": "only epoch_049 after exactly 50 epochs",
            "selection": "none; both seeds reported without selection",
            "precision": "32-true",
            "learning_rate": 5.0e-5,
        },
        "placement": {
            "h_s": "local GPU0",
            "h_c": "local GPU1",
            "h_c0": "remote 5070Ti GPU0",
        },
        "commands": {name: shlex.join(command) for name, command in commands.items()},
        "tmux_names": {
            "h_s_local_gpu0": "h1_seed43_hs_local_gpu0",
            "h_c_local_gpu1": "h1_seed43_hc_local_gpu1",
            "h_c0_remote_gpu0": "h1_seed43_hc0_remote_gpu0",
        },
        "launch_authorized": False,
        "required_before_execute": [
            "root reviews this command plan",
            "H-RS/H-LS terminal evaluator audit completes",
            "remote H-C0 source-only preflight is rerun against remote bytes",
            "all three output directories and tmux names are fresh",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-preflight-path", required=True, type=Path)
    parser.add_argument("--local-output-root", required=True, type=Path)
    parser.add_argument("--local-python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--remote-root", required=True, type=Path)
    parser.add_argument("--remote-output-root", required=True, type=Path)
    parser.add_argument("--remote-python-bin", required=True, type=Path)
    parser.add_argument("--remote-preflight-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"refusing to overwrite seed43 launch plan: {args.output_path}")
    plan = prepare_plan(**{key: value for key, value in vars(args).items() if key != "output_path"})
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
