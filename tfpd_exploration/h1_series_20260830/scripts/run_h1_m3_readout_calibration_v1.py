#!/usr/bin/env python3
"""Run H1-M3RC frozen C1 five-fold screen on GPU0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
for value in (REPO_ROOT, REPO_ROOT / "SPINT-main", REPO_ROOT / "tfpd_exploration/h1_series_20260830/src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from h1_m3_readout_calibration_v1.plan import SCHEMA

RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_m3_readout_calibration_v1"
PREDECESSOR = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_m3_cross_record_joint_postpool_v1/terminal.json"
PREDECESSOR_SHA256 = "5decf64d428c8c88eef72138c46c51210931376f4dbc770f497b2249546bdd81"
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
DESIGN = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/DESIGN_H1_M3_READOUT_CALIBRATION_V1_20260903.md"
WORKORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_M3_READOUT_CALIBRATION_V1_20260903.md"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(path: Path, value: dict) -> str:
    from src.h1_m4_cce_contract import write_immutable_json
    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        raise RuntimeError("--execute and --dry-run are mutually exclusive")
    if not args.execute:
        print(json.dumps({"schema": f"{SCHEMA}_dry", "status": "INERT_USE_EXECUTE"}, sort_keys=True))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if gpu != GPU0_UUID or _sha(PREDECESSOR) != PREDECESSOR_SHA256:
        raise RuntimeError("GPU0 or predecessor authority drift")
    if RESULT_ROOT.exists():
        raise RuntimeError("result root exists; no retry")
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    attempt = _publish(RESULT_ROOT / "attempt.json", {
        "schema": f"{SCHEMA}_attempt", "status": "ATTEMPT_SOURCE_ONLY_FROZEN_C1_M3RC",
        "gpu0_uuid": gpu, "predecessor_terminal_sha256": PREDECESSOR_SHA256,
        "design_sha256": _sha(DESIGN), "workorder_sha256": _sha(WORKORDER),
        "formal_heldout_opened": False, "evalai_opened": False, "gpu1_touched": False,
    })
    try:
        from h1_m3_readout_calibration_v1.evaluate import run
        score = run(REPO_ROOT, device="cuda:0", receipt_root=RESULT_ROOT)
        score["attempt_sha256"] = attempt
        score_sha = _publish(RESULT_ROOT / "score.json", score)
        terminal = {"schema": f"{SCHEMA}_terminal", "status": score["status"], "attempt_sha256": attempt,
                    "score_sha256": score_sha, "decision": score["decision"], "gpu0_uuid": gpu,
                    "formal_heldout_opened": False, "evalai_opened": False, "gpu1_touched": False}
        terminal_sha = _publish(RESULT_ROOT / "terminal.json", terminal)
        print(json.dumps({"terminal_sha256": terminal_sha, "decision": score["decision"]}, sort_keys=True))
        return 0
    except BaseException as error:
        _publish(RESULT_ROOT / "failure.json", {"schema": f"{SCHEMA}_failure", "status": "FAIL_NO_RETRY",
                 "attempt_sha256": attempt, "error_type": type(error).__name__, "error": str(error),
                 "formal_heldout_opened": False, "evalai_opened": False, "gpu1_touched": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
