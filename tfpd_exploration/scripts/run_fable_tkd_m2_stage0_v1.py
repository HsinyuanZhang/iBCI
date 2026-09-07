#!/usr/bin/env python3
"""Stage 0 runner for FABLE TKD M2 v1 (CPU only, fail-closed, receipt law).

Default invocation prints the plan (dry).  ``--execute`` writes
``results/fable_tkd_m2_v1/attempt.json`` FIRST (O_EXCL semantics: refuses to
overwrite an existing attempt), then runs A1-A6 and publishes
``stage0[/<run_tag>]/terminal.json``.  Any failure writes ``failure.json``
inside the run dir and re-raises.

Authorized re-runs (workorder section 9: A1 in [0.90, 0.99) allows adjusting
PV_BETA / merge bin masses only) use ``--run-tag`` to target a fresh
``stage0_<tag>`` directory; the shared ``stage0/t4_authority.json`` is
recomputed and required to match bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "tfpd_exploration" / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# CPU-only hard pin: set before numpy/torch are imported anywhere below.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")


def _sha_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="run Stage 0 (default: print the plan only)")
    parser.add_argument("--only-a1", action="store_true",
                        help="re-verify A1 only (planner-directed re-checks; "
                             "writes a1_pv_anchor.json + a1_only_terminal.json)")
    parser.add_argument("--baselines", action="store_true",
                        help="ADDENDUM-4 STAGE A controls: PV raw/calibrated + "
                             "ridge rate10 baselines -> stage0_<tag>/baselines.json")
    parser.add_argument("--run-tag", default="",
                        help="write receipts to stage0_<tag>/ (authorized re-runs)")
    parser.add_argument("--pv-beta", type=float, default=None,
                        help="PV anchor softmax temperature override (plan default 4.0)")
    parser.add_argument("--bin-masses", choices=("authority", "uniform"),
                        default="uniform",
                        help="merge init bin masses (ADDENDUM-1 default: uniform)")
    args = parser.parse_args()

    from tfpd_exploration.src.fable_tkd_m2_v1 import plan, stage0

    pv_beta = float(plan.PV_BETA if args.pv_beta is None else args.pv_beta)
    run_tag = str(args.run_tag)

    if not args.execute:
        print(json.dumps({
            "mode": "dry",
            "schema": plan.SCHEMA,
            "repo_root": str(REPO_ROOT),
            "result_root": str(plan.result_root(REPO_ROOT)),
            "stages": ["attempt.json", "A1 pv-anchor", "A2 parity", "A3 construction",
                       "A4 mac", "A5 t-shift", "A6 rho", "terminal.json"],
            "pv_beta": pv_beta,
            "bin_masses": args.bin_masses,
            "cpu_only": True,
        }, indent=2))
        return 0

    import torch

    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA was initialized before Stage 0; refusing (CPU only)")

    root = plan.result_root(REPO_ROOT)
    attempt_path = root / "attempt.json"
    if not attempt_path.exists():
        attempt = {
            "schema": f"{plan.SCHEMA}:attempt",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "spec": {
                "path": str((REPO_ROOT / plan.SPEC_RELATIVE).relative_to(REPO_ROOT)),
                "sha256": _sha_file(REPO_ROOT / plan.SPEC_RELATIVE),
            },
            "workorder": {
                "path": str((REPO_ROOT / plan.WORKORDER_RELATIVE).relative_to(REPO_ROOT)),
                "sha256": _sha_file(REPO_ROOT / plan.WORKORDER_RELATIVE),
            },
            "cpu_only": {
                "cuda_visible_devices": "",
                "blas_threads": 4,
                "host": platform.node(),
                "python": sys.executable,
            },
            "receipt_law": plan.RECEIPT_LAW,
            "initial_run": {"run_tag": run_tag, "pv_beta": pv_beta,
                            "bin_masses": args.bin_masses},
        }
        plan.atomic_receipt(attempt_path, attempt, exclusive=True)
    else:
        plan.verify_sidecar(attempt_path)
        if not run_tag and (root / "stage0" / "terminal.json").exists():
            raise SystemExit(
                "stage0/terminal.json already sealed; authorized re-runs require "
                "--run-tag (workorder section 9)"
            )

    run_dir = root / (f"stage0_{run_tag}" if run_tag else "stage0")
    try:
        if args.baselines:
            terminal = stage0.run_baselines(REPO_ROOT, run_tag=run_tag or "r5")
        elif args.only_a1:
            terminal = stage0.execute_a1_only(
                REPO_ROOT, run_tag=run_tag, pv_beta=pv_beta,
                bin_masses_mode=args.bin_masses,
            )
        else:
            terminal = stage0.execute(
                REPO_ROOT, run_tag=run_tag, pv_beta=pv_beta,
                bin_masses_mode=args.bin_masses,
            )
    except Exception as error:  # noqa: BLE001 - fail-closed receipt then re-raise
        failure = {
            "schema": f"{plan.SCHEMA}:failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir.relative_to(root)),
            "pv_beta": pv_beta,
            "bin_masses": args.bin_masses,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        plan.atomic_receipt(run_dir / "failure.json", failure)
        raise
    summary = {
        "status": terminal["status"],
        "run_dir": str(run_dir),
        "elapsed_seconds": terminal["elapsed_seconds"],
    }
    if args.baselines:
        summary["pv_raw"] = {
            surface: terminal["pv_raw"][surface]["equal_session_mean"]
            for surface in ("within", "external")
        }
        summary["pv_calibrated"] = {
            surface: terminal["pv_calibrated"][surface]["equal_session_mean"]
            for surface in ("within", "external")
        }
        summary["ridge"] = {
            surface: terminal["ridge"][surface]["equal_session_mean"]
            for surface in ("within", "external")
        }
        summary["ridge_lambda"] = terminal["ridge"]["selected_lambda_normalized"]
        summary["decision"] = terminal["decision"]
    else:
        summary["a1_verdict"] = terminal["a1"]["verdict"]
        summary["a1_min_flattened_pearson"] = terminal["a1"]["min_flattened_pearson"]
        summary["a1_per_session"] = terminal["a1"]["per_session"]
        if not args.only_a1:
            summary["a2"] = terminal["a2"]
            summary["a4"] = terminal["a4"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
