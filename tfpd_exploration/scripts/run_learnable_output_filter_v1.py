#!/usr/bin/env python3
"""Runner for the learnable-causal-output-filter frozen-output route (P0-P4).

Lane: learnable_output_filter_v1 (frozen-weight, inference-only, ZERO
training).  Stages:

  plan          print the frozen pre-registration (dry, default)
  attempt       reserve attempt.json (0444+sidecar) BEFORE any data
  cache-static  materialize the frozen static stream once, CPU-only
                (CUDA_VISIBLE_DEVICES must be empty)
  cache-cdm     materialize the activity-only CDM raw stream once, GPU
                INFERENCE only (brief; <10 min GPU compute), frozen weights
  p0            runtime/contract audit on the cached streams
  p1            four-cell fixed-filter factorial (static A0/A1 fresh with
                TRIAL_RESET; CDM B0/B1 bound from the sealed P2' stage A)
  p2            source-learned fixed filter (F2 alpha grid CV, F3 FIR-K4,
                §10.2 selection gate)
  p3            §8 adaptive oracle on the B0 streams, leakage-labelled
  p4            source-fit F4 adaptive scalar gain (only if P3 allows)
  finalize      terminal receipt

Boundaries enforced here:
- PYTHONNOUSERSITE via tfpd_lane.receipt.enforce_environment on every CPU stage;
- sealed receipts SHA-verified at load;
- cache-static requires CUDA_VISIBLE_DEVICES="" and idle GPUs;
- cache-cdm requires the P2' environment (CVD=gpu, PCI order, data roots) and
  performs inference ONLY (no optimizer, no backward, no training);
- every later stage performs ZERO decoder forwards (cached streams only).

Usage (spint env):
  PYTHONNOUSERSITE=1 python scripts/run_learnable_output_filter_v1.py --stage plan
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_learnable_output_filter_v1.py \\
      --execute --stage attempt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

sys.path.insert(0, str(REPO_ROOT / "streaming_calibration_exp"))
sys.path.insert(0, str(REPO_ROOT / "sua_exploration"))
sys.path.insert(0, str(ROOT))


def _gpu_snapshot(label: str) -> dict:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,compute_mode",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    rows = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    compute = [row for row in rows if row.split(",")[2].strip() not in ("0", "0 %")]
    return {"label": label, "rows": rows, "any_gpu_compute_active": bool(compute),
            "compute_rows": compute}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", default="plan", choices=(
        "plan", "attempt", "cache-static", "cache-cdm", "p0", "p1", "p2", "p3", "p4",
        "finalize",
    ))
    parser.add_argument("--gpu-index", type=int, default=1, choices=(0, 1))
    args = parser.parse_args()

    import os

    os.environ.setdefault("SUBC_DATA_ROOT", str(REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"))
    os.environ.setdefault("SUBM_DATA_ROOT", str(REPO_ROOT / "sua_exploration/data/dandi_000688/sub-M"))

    from src.learnable_output_filter_v1 import plan, receipts, stages, streams

    if args.stage == "plan":
        print(json.dumps(plan.pre_registration_payload(), sort_keys=True, indent=2))
        return 0
    if not args.execute:
        print(json.dumps({"stage": args.stage, "note": "dry; pass --execute"}, sort_keys=True))
        return 0

    from src.tfpd_lane import receipt as lane_receipt

    results = ROOT / "results/learnable_output_filter_v1"
    # cache-cdm runs under the P2' GPU environment (CVD=<gpu>, PCI order, data
    # roots), validated inside streams.materialize_cdm; every other stage is
    # CPU-only under the standard empty-CVD lane gate.
    environment = (
        None if args.stage == "cache-cdm" else lane_receipt.enforce_environment()
    )

    if args.stage == "attempt":
        receipts.require_fresh(results / "attempt.json")
        results.mkdir(mode=0o755, parents=True, exist_ok=True)
        payload = {
            "schema": "learnable_output_filter_v1_attempt_v1",
            "status": "ATTEMPT_RESERVED",
            "route": "frozen_output_P0_P4",
            "design_authority": plan.DESIGN_RELATIVE,
            "environment": environment,
            "owned_sha256s": plan.owned_sha256s(REPO_ROOT),
            "pre_registration": plan.pre_registration_payload(),
            "stages": ["cache-static", "cache-cdm", "p0", "p1", "p2", "p3", "p4", "finalize"],
            "inference_only": True,
            "training_authorized": False,
            "j2_j3_training": "NOT AUTHORIZED; spec only (docs/SPEC_J_TRAINING_INTEGRATED_GATES_20260829.md)",
            "target_optimizer_backward_update": 0,
        }
        digest = receipts.publish(results / "attempt.json", payload)
        print(json.dumps({"stage": "attempt", "attempt_sha256": digest}, sort_keys=True))
        return 0

    attempt = receipts.read_receipt(results / "attempt.json")
    owned = plan.owned_sha256s(REPO_ROOT)
    if owned != attempt["owned_sha256s"]:
        drifted = [key for key in owned if owned[key] != attempt["owned_sha256s"][key]]
        raise SystemExit(f"route-owned module bytes drifted after attempt reservation: {drifted}")

    if args.stage == "cache-static":
        import time

        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise SystemExit("cache-static requires CUDA_VISIBLE_DEVICES to be empty (CPU-only)")
        before = _gpu_snapshot("before")
        if before["any_gpu_compute_active"]:
            raise SystemExit(f"GPU compute active before CPU-only stage: {before['compute_rows']}")
        started = time.perf_counter()
        result = streams.materialize_static(ROOT)
        after = _gpu_snapshot("after")
        if after["any_gpu_compute_active"]:
            raise SystemExit(f"GPU compute active after CPU-only stage: {after['compute_rows']}")
        result["gpu_disclosure"] = {"before": before, "after": after}
        result["environment"] = environment
        result["wall_seconds"] = time.perf_counter() - started
        receipts.publish(results / "static_stream_cache.json", result)
        print(json.dumps({
            "stage": "cache-static", "sessions": result["sessions"],
            "max_abs_r2_drift": result["max_abs_r2_drift"],
            "all_prediction_shas_bitexact": result["all_prediction_shas_bitexact"],
            "wall_seconds": result["wall_seconds"],
        }, sort_keys=True))
        return 0

    if args.stage == "cache-cdm":
        result = streams.materialize_cdm(ROOT, gpu_index=args.gpu_index)
        receipts.publish(results / "cdm_stream_cache.json", result)
        print(json.dumps({
            "stage": "cache-cdm", "sessions": result["sessions"],
            "all_matrix_r2_equal": result["all_matrix_r2_equal"],
            "all_raw_shas_bitexact": result["all_raw_shas_bitexact"],
            "model_state_unchanged_by_wrapper": result["model_state_unchanged_by_wrapper"],
            "wall_seconds": result["wall_seconds"],
        }, sort_keys=True))
        return 0

    if args.stage == "p0":
        payload = stages.run_p0(ROOT)
        print(json.dumps({
            "stage": "p0", "n_proofs": payload["n_proofs"],
            "f0_bypass_bitwise_equal": payload["f0_bypass_bitwise_equal"],
            "boundary_tampering_fails_closed": payload["boundary_tampering_fails_closed"]["all_pass"],
        }, sort_keys=True))
        return 0

    if args.stage == "p1":
        payload = stages.run_p1(ROOT)
        print(json.dumps({
            "stage": "p1",
            "primary_readings": payload["primary_readings_summary"],
        }, sort_keys=True, indent=2))
        return 0

    if args.stage == "p2":
        payload = stages.run_p2(ROOT)
        print(json.dumps({
            "stage": "p2", "selected_by_budget": payload["selected_by_budget"],
        }, sort_keys=True))
        return 0

    if args.stage == "p3":
        payload = stages.run_p3(ROOT)
        print(json.dumps({
            "stage": "p3", "disposition_summary": payload["disposition_summary"],
            "f4_allowed_by_budget": payload["f4_allowed_by_budget"],
        }, sort_keys=True, indent=2))
        return 0

    if args.stage == "p4":
        payload = stages.run_p4(ROOT)
        summary = {
            budget: {
                "oof_gain_over_fixed": body["cv"]["oof_gain_over_reference"],
                "oof_gain_over_fixed_paired_mean": body["cv"]["oof_gain_over_reference_paired"]["mean"],
                "within_recovery_vs_raw_mean": body["cv"]["within_recovery_vs_raw_paired"]["mean"],
            } for budget, body in payload["budgets"].items()
        }
        print(json.dumps({
            "stage": "p4", "budgets": summary,
            "skipped_by_disposition": payload["skipped_by_disposition"],
        }, sort_keys=True, indent=2))
        return 0

    if args.stage == "finalize":
        names = (
            "static_stream_cache.json", "cdm_stream_cache.json", "p0_audit.json",
            "p1_factorial.json", "p2_fixed_filter.json", "p3_oracle.json", "p4_adaptive.json",
        )
        digests = {}
        for name in names:
            path = results / name
            if not path.exists():
                raise SystemExit(f"finalize: missing stage receipt {name}")
            body = receipts.read_receipt(path)
            digests[name] = body.get("schema")
        import hashlib

        terminal = {
            "schema": "learnable_output_filter_v1_terminal_v1",
            "status": "TERMINAL",
            "attempt_sha256": hashlib.sha256(
                (results / "attempt.json").read_bytes()
            ).hexdigest(),
            "stage_receipts": digests,
            "p2_selected_by_budget": _terminal_summary(results),
            "target_optimizer_backward_update": 0,
            "inference_only": True,
        }
        receipts.publish(results / "terminal.json", terminal)
        print(json.dumps({"stage": "finalize", "status": terminal["status"]}, sort_keys=True))
        return 0

    raise SystemExit(f"unhandled stage {args.stage}")


def _terminal_summary(results: Path) -> dict:
    body = receipts.read_receipt(results / "p2_fixed_filter.json")
    return body["selected_by_budget"]


if __name__ == "__main__":
    raise SystemExit(main())
