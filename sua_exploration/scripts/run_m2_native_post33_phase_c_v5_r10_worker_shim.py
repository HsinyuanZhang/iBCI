#!/usr/bin/env python3
"""Run an audited r9 v5 worker with only its authorization import redirected.

The r9 trainer/evaluator bodies (including the verified pre-Hydra and v5 device
repairs) are executed byte-for-byte.  Before their import occurs, this shim
places r10's static guard under the legacy import name in ``sys.modules``.
No source file is edited and no r9 capability is accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import m2_native_post33_phase_c_v5_r10_static as static  # noqa: E402


LEGACY_AUTH_MODULE = "sua_exploration.mc_maze.m2_native_post33_authorization_v5_r9"
WORKERS = {
    ("spint", "train"): ROOT / "SPINT-main/src/train_post33_phase_c_v5_r9.py",
    ("spint", "evaluate"): ROOT / "SPINT-main/src/evaluate_post33_phase_c_v5_r9.py",
    ("t4", "train"): ROOT / "streaming_calibration_exp/src/train_post33_phase_c_v5_r9.py",
    ("t4", "evaluate"): ROOT / "streaming_calibration_exp/src/evaluate_post33_phase_c_v5_r9.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=("spint", "t4"), required=True)
    parser.add_argument("--role", choices=("train", "evaluate"), required=True)
    parser.add_argument("--verify-r10-static-import-redirection-only", action="store_true")
    args, remainder = parser.parse_known_args()
    sys.modules[LEGACY_AUTH_MODULE] = static
    if args.verify_r10_static_import_redirection_only:
        print(json.dumps({
            "schema": "m2_post33_phase_c_v5_r10_worker_import_redirect_smoke_v1",
            "status": "PASS",
            "legacy_auth_module": LEGACY_AUTH_MODULE,
            "redirect_target": static.__name__,
            "r9_worker": str(WORKERS[(args.arm, args.role)]),
            "formal_data_accessed": False,
            "cuda_initialized": False,
            "score_or_r2_accessed": False,
        }, sort_keys=True))
        return
    source = WORKERS[(args.arm, args.role)]
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"audited r9 worker missing/noncanonical: {source}")
    if sys.modules.get(LEGACY_AUTH_MODULE) is not static:
        raise RuntimeError("r10 worker authorization import redirection failed")
    sys.argv = [str(source), *remainder]
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
