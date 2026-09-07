#!/usr/bin/env python3
"""FABLE TKD M1 v1 runner: A1-M1 anchor (CPU) then fold-local pilot (GPU1).

Dry by default.  ``--execute``: attempt receipt first, then the A1-M1
generative-anchor assertion on CPU (>= 0.90 required before any GPU run),
then the pilot -- A' seed 42 and SHUF seed 42 on the fold-local LOSO fold-0
face (verbatim mirror) -- with the preregistered kill-gates
K1: A' R2 >= 0.5374 (Z-Fix - 0.10) and K2: A' - SHUF >= +0.02.
The grid is NEVER launched by this script (workorder section 4).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "tfpd_exploration" / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def gpu1_preflight(stage: str = "launch") -> dict[str, object]:
    import os

    def query(arguments: list[str]) -> str:
        completed = subprocess.run(arguments, check=True, text=True,
                                   capture_output=True, timeout=30)
        return completed.stdout

    rows = query(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                  "--format=csv,noheader,nounits"])
    target = None
    for line in rows.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts[0] == "1":
            target = {"uuid": parts[1], "util": float(parts[2]), "mem": float(parts[3])}
    if target is None:
        raise SystemExit("GPU index 1 absent")
    if target["uuid"] != "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86":
        raise SystemExit(f"GPU1 UUID drift: {target['uuid']}")
    apps = query(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                  "--format=csv,noheader"])
    pids = [[p.strip() for p in line.split(",")]
            for line in apps.strip().splitlines() if line.strip()]
    foreign = [row for row in pids
               if row[1] == target["uuid"] and int(row[0]) != os.getpid()]
    if foreign:
        raise SystemExit(f"GPU1 has foreign compute apps: {[r[0] for r in foreign]}")
    if stage == "launch" and (target["util"] > 5.0 or target["mem"] > 500.0):
        raise SystemExit(f"GPU1 busy: util {target['util']}% mem {target['mem']}MiB")
    return {"stage": stage, "gpu1_uuid": target["uuid"],
            "util": target["util"], "mem_mib": target["mem"],
            "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from tfpd_exploration.src.fable_tkd_m1_v1 import plan

    if not args.execute:
        print(json.dumps({
            "dry": True,
            "schema": plan.SCHEMA,
            "steps": ["attempt.json (O_EXCL)", "A1-M1 anchor (CPU, >=0.90 required)",
                      "pilot A' s42 + SHUF s42 (GPU1, fold-local fold 0)",
                      "K1: A' R2 >= 0.5374; K2: A'-SHUF >= +0.02",
                      "STOP (grid never launched)"],
            "ref_z_fix": plan.REF_Z_FIX,
            "anchor_bands": [plan.ANCHOR_DISCLOSURE, plan.ANCHOR_PASS],
            "schedule": {"epochs": plan.EPOCHS, "lr": plan.LR, "batch": plan.BATCH,
                         "distill_lambda": plan.DISTILL_LAMBDA},
        }, indent=2))
        return 0

    root = plan.result_root(REPO_ROOT)
    attempt_path = root / "attempt.json"
    if not attempt_path.exists():
        plan.atomic_receipt(attempt_path, {
            "schema": f"{plan.SCHEMA}:attempt",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workorder": {
                "path": str((REPO_ROOT / plan.WORKORDER_RELATIVE).relative_to(REPO_ROOT)),
            },
            "cpu_anchor_first": True,
            "pilot_arms": ["A_PRIME", "SHUF"],
        }, exclusive=True)
    else:
        plan.require(
            (root / "stage0" / "a1_m1_anchor.json").exists()
            or (root / "failure.json").exists(),
            "attempt exists without an anchor receipt or failure record",
        )

    try:
        # ---- Stage-0-style anchor on CPU, BEFORE any GPU work -------------
        anchor_dir = root / "stage0"
        if not (anchor_dir / "a1_m1_anchor.json").exists():
            from tfpd_exploration.src.fable_tkd_m1_v1 import anchor as anchor_module

            anchor_receipt = anchor_module.run_anchor(REPO_ROOT, run_dir=anchor_dir)
        else:
            import json as _json

            anchor_receipt = _json.loads(
                (anchor_dir / "a1_m1_anchor.json").read_text(encoding="utf-8")
            )
        print(json.dumps({
            "anchor_verdict": anchor_receipt["verdict"],
            "anchor_min_corr": anchor_receipt["min_flattened_pearson_rho_ones"],
        }))
        plan.require(
            anchor_receipt["verdict"] != "FAIL",
            f"A1-M1 anchor below {plan.ANCHOR_DISCLOSURE}: "
            f"{anchor_receipt['min_flattened_pearson_rho_ones']}",
        )

        # ---- Pilot on GPU1 -------------------------------------------------
        preflight = gpu1_preflight(stage="launch")
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        import torch

        plan.require(torch.cuda.is_available(), "GPU1 unavailable")
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        device = torch.device("cuda:0")

        from tfpd_exploration.src.fable_tkd_m1_v1 import data as m1_data
        from tfpd_exploration.src.fable_tkd_m1_v1 import train as train_module

        plane = m1_data.load_plane(REPO_ROOT)
        runs = {}
        for arm in ("A_PRIME", "SHUF"):
            pre = gpu1_preflight(stage="pre_run")
            receipt = train_module.train_arm(REPO_ROOT, arm, device, plane=plane)
            runs[arm] = {
                "r2": receipt["face_at_fixed_last"]["governing_r2"],
                "selected_epoch": receipt["selected_epoch"],
                "run_dir": receipt["run_dir"],
                "train_seconds": receipt["elapsed_seconds"],
                "gpu_preflight": pre,
            }
        k1 = runs["A_PRIME"]["r2"] >= plan.K1_FLOOR
        k2 = (runs["A_PRIME"]["r2"] - runs["SHUF"]["r2"]) >= plan.K2_FLOOR
        payload = {
            "schema": f"{plan.SCHEMA}:pilot",
            "anchor": {
                "verdict": anchor_receipt["verdict"],
                "min_corr": anchor_receipt["min_flattened_pearson_rho_ones"],
            },
            "gpu_preflight": preflight,
            "runs": runs,
            "gates": {
                "K1": {"rule": f"A' R2 >= {plan.K1_FLOOR}", "value": runs["A_PRIME"]["r2"],
                       "pass": bool(k1)},
                "K2": {"rule": "A' - SHUF >= +0.02",
                       "value": runs["A_PRIME"]["r2"] - runs["SHUF"]["r2"],
                       "pass": bool(k2)},
                "K1_and_K2": bool(k1 and k2),
                "grid_launched": False,
                "note": "workorder section 4: grid decision belongs to the planner",
            },
            "refs": {"Z-Fix": plan.REF_Z_FIX, "S-Fix": plan.REF_S_FIX,
                     "S-Acyc": plan.REF_S_ACYC},
            "deviations": [d["id"] for d in plan.DEVIATIONS],
        }
        plan.atomic_receipt(root / "pilot.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001
        plan.atomic_receipt(root / "failure.json", {
            "schema": f"{plan.SCHEMA}:failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
