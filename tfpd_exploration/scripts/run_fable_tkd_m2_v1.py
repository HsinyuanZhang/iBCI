#!/usr/bin/env python3
"""FABLE TKD M2 v1 Wave 2 runner: pilot / grid / eval modes.

Dry by default.  ``--execute`` runs GPU1 preflight (Wave 2 spec section 0)
BEFORE importing torch, pins ``CUDA_VISIBLE_DEVICES=1``, then executes the
requested mode with the Wave 1 receipt law (atomic 0444 + sha256 sidecars,
attempt-first, fail-closed).

Modes:
  pilot  train arm A2 seed 42 + evaluate + sanity gate -> stage1_pilot.json
  grid   serial (arm, seed) runs (A2_GRU seed 42 only) -> stage1_grid.json
  eval   re-evaluate one existing run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "tfpd_exploration" / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def gpu1_preflight(stage: str = "launch") -> dict[str, object]:
    """GPU1 ownership check BEFORE torch import (Wave 2 spec section 0).

    Mirrors m1_emg_rsyn3_fold_local_v1/gpu.py fail-closed style with the
    Wave 2 thresholds: exact UUID, zero FOREIGN compute pids on GPU1,
    util <= 5%, memory <= 500 MiB.  ``stage="launch"`` runs before this
    process owns a CUDA context, so the thresholds apply to anyone;
    ``stage="pre_run"`` re-checks between runs while our own context is
    resident, so util/mem are recorded but only FOREIGN pids fail the check
    (the gpu.py own/foreign occupancy law).  GPU0 is never touched.
    """
    import os

    def query(arguments: list[str]) -> str:
        completed = subprocess.run(arguments, check=True, text=True,
                                   capture_output=True, timeout=30)
        return completed.stdout

    gpu_rows = query(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"]
    )
    target = None
    for line in gpu_rows.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise SystemExit(f"nvidia-smi row drift: {line}")
        if parts[0] == "1":
            target = {"uuid": parts[1], "util": float(parts[2]), "mem_mib": float(parts[3])}
    if target is None:
        raise SystemExit("GPU index 1 absent")
    if target["uuid"] != "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86":
        raise SystemExit(f"GPU1 UUID drift: {target['uuid']}")
    apps = query(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                  "--format=csv,noheader"])
    pids = [
        [part.strip() for part in line.split(",")]
        for line in apps.strip().splitlines() if line.strip()
    ]
    own_pid = os.getpid()
    occupants = [row for row in pids if row[1] == target["uuid"]]
    foreign = [row for row in occupants if int(row[0]) != own_pid]
    if foreign:
        raise SystemExit(f"GPU1 has foreign compute apps: {[row[0] for row in foreign]}")
    own_occupancy = any(int(row[0]) == own_pid for row in occupants)
    if stage == "launch":
        if target["util"] > 5.0:
            raise SystemExit(f"GPU1 utilization {target['util']}% > 5")
        if target["mem_mib"] > 500.0:
            raise SystemExit(f"GPU1 memory {target['mem_mib']} MiB > 500")
    return {
        "stage": stage,
        "gpu1_uuid": target["uuid"],
        "gpu1_utilization_gpu": target["util"],
        "gpu1_memory_used_mib": target["mem_mib"],
        "foreign_compute_pids": [],
        "own_context_resident": bool(own_occupancy),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _monotonic_improvement(history: list[dict[str, object]]) -> dict[str, object]:
    means = [
        float(row.get("minival_equal_session_mean", row["within_equal_session_mean"]))
        for row in history
    ]
    non_improving = sum(1 for i in range(len(means) - 1) if means[i + 1] <= means[i])
    return {
        "first_epoch_mean": means[0],
        "last_epoch_mean": means[-1],
        "improvement": means[-1] - means[0],
        "non_improving_steps": int(non_improving),
        "monotone_pass": bool(non_improving <= 2 and means[-1] > means[0]),
    }


def _next_receipt_path(root: Path, base: str, failure_base: str) -> Path:
    """O_EXCL receipt slot; retries allowed only after a recorded failure.

    The first attempt uses ``base``; if that is already sealed, a retry is
    authorized only when ``failure_base`` exists (the previous attempt failed
    fail-closed), and it uses ``base_r<n>`` (the Wave 1 D4 convention).
    """
    first = root / base
    if not first.exists():
        return first
    if not (root / failure_base).exists():
        raise SystemExit(f"refusing: {first} already sealed without a failure record")
    index = 1
    while (root / f"{base[:-5]}_r{index}.json").exists():
        index += 1
    return root / f"{base[:-5]}_r{index}.json"


def _next_failure_path(root: Path, failure_base: str) -> Path:
    if not (root / failure_base).exists():
        return root / failure_base
    index = 1
    while (root / f"{failure_base[:-5]}_r{index}.json").exists():
        index += 1
    return root / f"{failure_base[:-5]}_r{index}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("pilot", "grid", "eval"), default="pilot")
    parser.add_argument("--arms", default="A2,A,SHUF,POOL,B,A2_GRU")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--arm", default=None, help="eval mode: single arm")
    parser.add_argument("--seed", type=int, default=None, help="eval mode: seed")
    args = parser.parse_args()

    from tfpd_exploration.src.fable_tkd_m2_v1 import plan

    def _grid_plan() -> list[dict[str, object]]:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        runs: list[dict[str, object]] = []
        for arm in arms:
            for seed in seeds:
                if arm == "A2_GRU" and seed != 42:
                    continue
                runs.append({"arm": arm, "seed": seed})
        return runs

    if not args.execute:
        if args.mode == "pilot":
            plan_summary = {
                "mode": "pilot",
                "steps": ["gpu1 preflight", "stage1_attempt.json (O_EXCL)",
                          "train A2 s42", "evaluate", "sanity gate",
                          "stage1_pilot.json"],
                "sanity": {
                    "external_equal_session_mean_range": list(plan.SANITY_EXTERNAL_RANGE),
                    "max_non_monotonic_epochs": plan.SANITY_MAX_NONMONOTONIC_EPOCHS,
                },
            }
        elif args.mode == "grid":
            plan_summary = {
                "mode": "grid",
                "runs": _grid_plan(),
                "run_count": len(_grid_plan()),
                "note": "serial, resumable; A2_GRU restricted to seed 42",
            }
        else:
            plan_summary = {"mode": "eval", "arm": args.arm, "seed": args.seed}
        print(json.dumps({"dry": True, "schema": plan.SCHEMA, **plan_summary}, indent=2))
        return 0

    # --execute: preflight BEFORE torch import, then pin GPU1.
    preflight = gpu1_preflight()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    import torch

    torch.set_num_threads(4)
    plan.require(torch.cuda.is_available(), "GPU1 unavailable after preflight")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    from tfpd_exploration.src.fable_tkd_m2_v1 import evaluate as evaluate_module
    from tfpd_exploration.src.fable_tkd_m2_v1 import train as train_module

    root = plan.result_root(REPO_ROOT)

    def _failure(error: BaseException, extra: dict[str, object]) -> Path:
        target = _next_failure_path(root, "stage1_failure.json")
        plan.atomic_receipt(
            target,
            {
                "schema": f"{plan.SCHEMA}:stage1_failure",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": args.mode,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                **extra,
            },
        )
        return target

    if args.mode == "pilot":
        attempt_path = _next_receipt_path(root, "stage1_attempt.json", "stage1_failure.json")
        try:
            plan.atomic_receipt(attempt_path, {
                "schema": f"{plan.SCHEMA}:stage1_attempt",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "pilot",
                "gpu_preflight": preflight,
                "constants": {
                    "epochs": plan.EPOCHS, "lr": plan.LR, "wd": plan.WD,
                    "batch": plan.BATCH, "grad_clip": plan.GRAD_CLIP,
                    "pv_init_config": plan.PV_INIT_CONFIG,
                },
            }, exclusive=True)
            plane = train_module.load_session_plane(REPO_ROOT)
            pre_run = gpu1_preflight(stage="pre_run")
            # ADDENDUM-6 pilot r10: A2_GRU under D16 minival selection, two
            # distillation weights (0.1 champion convention, 1.0 pure
            # compression), identical schedule, seed 42.
            pilot_runs = []
            overall = True
            for teacher_weight, tag in ((0.1, "_le1"), (1.0, "_le10")):
                train_receipt = train_module.train_one(
                    REPO_ROOT, "A2_GRU", 42, device, plane=plane,
                    teacher_weight=teacher_weight, run_tag=tag,
                )
                eval_receipt = evaluate_module.evaluate_run(
                    REPO_ROOT, "A2_GRU", 42, device, plane=plane,
                    run_dir=Path(train_receipt["run_dir"]),
                )
                external_mean = float(eval_receipt["external"]["equal_session_mean"])
                gate = _monotonic_improvement(train_receipt["history"])
                low, high = plan.SANITY_EXTERNAL_RANGE
                external_pass = low <= external_mean <= high
                run_pass = bool(external_pass and gate["monotone_pass"])
                overall = overall and run_pass
                pilot_runs.append({
                    "arm": "A2_GRU",
                    "teacher_weight": teacher_weight,
                    "run_dir": train_receipt["run_dir"],
                    "external_equal_session_mean": external_mean,
                    "external_per_session_r2": eval_receipt["external"]["per_session_r2"],
                    "within_equal_session_mean": eval_receipt["within"]["equal_session_mean"],
                    "sanity": {
                        "external_in_range": external_pass,
                        "external_range": [low, high],
                        **gate,
                    },
                    "selected_epoch": train_receipt["selected_epoch"],
                    "selected_minival_mean":
                        train_receipt["selected_minival_equal_session_mean"],
                    "epsilon_value": eval_receipt["static_alpha"]["epsilon_value"],
                    "train_seconds": train_receipt["elapsed_seconds"],
                    "eval_seconds": eval_receipt["elapsed_seconds"],
                    "verdict": "PASS" if run_pass else "FAIL",
                })
            continuation = {
                "rule": (
                    "ADDENDUM-6 preregistered: if either variant's "
                    "minival-selected external >= 0.10 the M2 line continues "
                    "(GRU becomes the main time model, SSM a negative "
                    "result); if both < 0.10 the M2 line downgrades per "
                    "ADDENDUM-5 and the program pivots to M1"
                ),
                "any_variant_external_ge_0_10":
                    bool(any(r["external_equal_session_mean"] >= 0.10
                             for r in pilot_runs)),
            }
            payload = {
                "schema": f"{plan.SCHEMA}:stage1_pilot",
                "seed": 42,
                "selection_law": "D16_minival_80_20",
                "gpu_preflight_initial": preflight,
                "gpu_preflight_pre_run": pre_run,
                "runs": pilot_runs,
                "continuation_rule": continuation,
                "verdict": "PASS" if overall else "FAIL",
            }
            if not overall:
                summary_line = "; ".join(
                    f"{r['arm']}: ext {r['external_equal_session_mean']:.4f}, "
                    f"monotone {r['sanity']['monotone_pass']}" for r in pilot_runs
                )
                _failure(RuntimeError(f"pilot sanity gate failed: {summary_line}"),
                         {"pilot": payload})
                print(json.dumps(payload, indent=2))
                return 1
            plan.atomic_receipt(root / "stage1_pilot.json", payload)
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as error:  # noqa: BLE001
            _failure(error, {"mode": "pilot"})
            raise
    if args.mode == "grid":
        attempt_path = _next_receipt_path(
            root, "stage1_grid_attempt.json", "stage1_grid_failure.json"
        )
        try:
            runs = _grid_plan()
            plan.require(len(runs) == 16, f"grid must list 16 runs, got {len(runs)}")
            plan.atomic_receipt(attempt_path, {
                "schema": f"{plan.SCHEMA}:stage1_grid_attempt",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "grid",
                "runs": runs,
                "gpu_preflight": preflight,
            }, exclusive=True)
            plane = train_module.load_session_plane(REPO_ROOT)
            results: list[dict[str, object]] = []
            skipped: list[dict[str, object]] = []
            for run in runs:
                arm, seed = run["arm"], run["seed"]
                run_dir = evaluate_module.run_dir_for(REPO_ROOT, arm, seed)
                if (run_dir / "train.json").is_file() and (run_dir / "eval.json").is_file():
                    try:
                        plan.verify_sidecar(run_dir / "train.json")
                        plan.verify_sidecar(run_dir / "eval.json")
                        skipped.append({**run, "status": "skipped_sha_chain_intact"})
                        continue
                    except plan.TKDError:
                        skipped.append({**run, "status": "broken_chain_rerun"})
                pre_run = gpu1_preflight(stage="pre_run")
                train_receipt = train_module.train_one(
                    REPO_ROOT, arm, seed, device, plane=plane
                )
                eval_receipt = evaluate_module.evaluate_run(
                    REPO_ROOT, arm, seed, device, plane=plane
                )
                results.append({
                    **run,
                    "external_equal_session_mean":
                        eval_receipt["external"]["equal_session_mean"],
                    "external_per_session_r2": eval_receipt["external"]["per_session_r2"],
                    "within_equal_session_mean":
                        eval_receipt["within"]["equal_session_mean"],
                    "epsilon_value": eval_receipt["static_alpha"]["epsilon_value"],
                    "selected_epoch": train_receipt["selected_epoch"],
                    "train_seconds": train_receipt["elapsed_seconds"],
                    "eval_seconds": eval_receipt["elapsed_seconds"],
                    "gpu_preflight": pre_run,
                })
            plan.atomic_receipt(root / "stage1_grid.json", {
                "schema": f"{plan.SCHEMA}:stage1_grid",
                "runs": runs,
                "results": results,
                "skipped": skipped,
                "verdict": "TERMINAL",
            })
            print(json.dumps({"grid_done": len(results), "skipped": len(skipped)}))
            return 0
        except Exception as error:  # noqa: BLE001
            _failure(error, {"mode": "grid"})
            raise
    # eval mode
    try:
        plan.require(args.arm in plan.ARMS, f"unknown arm {args.arm}")
        plan.require(args.seed is not None, "eval mode needs --seed")
        plane = train_module.load_session_plane(REPO_ROOT)
        pre_run = gpu1_preflight(stage="pre_run")
        run_dir = evaluate_module.run_dir_for(REPO_ROOT, args.arm, args.seed)
        name = "eval.json"
        if (run_dir / name).exists():
            index = 1
            while (run_dir / f"eval_reeval_{index}.json").exists():
                index += 1
            name = f"eval_reeval_{index}.json"
        receipt = evaluate_module.evaluate_run(
            REPO_ROOT, args.arm, args.seed, device, plane=plane, receipt_name=name
        )
        print(json.dumps({
            "arm": args.arm, "seed": args.seed, "receipt": name,
            "external_equal_session_mean": receipt["external"]["equal_session_mean"],
            "within_equal_session_mean": receipt["within"]["equal_session_mean"],
            "gpu_preflight": pre_run,
        }, indent=2))
        return 0
    except Exception as error:  # noqa: BLE001
        _failure(error, {"mode": "eval", "arm": args.arm, "seed": args.seed})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
