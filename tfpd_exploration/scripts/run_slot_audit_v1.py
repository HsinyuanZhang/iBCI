"""Runner for SLOT-AUDIT V1 — the 50 already-supervised output slots (CPU-only).

Work order: ``docs/WORKORDER_SLOT_AUDIT_V1_20260829.md`` (binding); guidance
sections 2.2/5.  This runner never trains and never updates a checkpoint: it
strict-loads the sealed Cell-D SWA through the Z1 provenance harness, decodes
every (surface, session, budget) once with the frozen static deployment
recipe, anchors each decode to the SEALED continuity-probe baseline rows
(R2 tolerance 1e-12 + bit-exact full-tensor prediction SHAs), caches the full
[W, 50, 2] tensors + the session behavior-bin target source under
``cache/slot_audit_v1/`` (digest-bound manifest), then runs the pure-numpy
audit (per-slot calibration, per-delta residual curves, slot-subset
ensembles, trajalign re-measurement through the frozen probe law, latency,
digests) and publishes the pre-registered reading classification.

Receipt sequence (work order section 6): attempt BEFORE any source/model/data
access -> materialize.json (with the cache manifest digest) -> terminal.json;
every receipt is transactional 0444 + sha256 sidecar.  A failure writes
failure.json and stops.

Boundaries (mirror of run_continuity_probe_v1.py):
- CUDA_VISIBLE_DEVICES must be empty; nvidia-smi captured before AND after,
  both GPUs must show zero compute processes;
- PYTHONNOUSERSITE via tfpd_lane.receipt.enforce_environment;
- sealed model state digest compared before/after;
- the consumed sealed receipts are SHA-verified at load.

Usage (spint env):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
    python scripts/run_slot_audit_v1.py [--budgets 4 10 30]
        [--surfaces external within] [--smoke-sessions N] [--stage all|materialize|audit]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RESULT_DIR = ROOT / "results/slot_audit_v1"
ATTEMPT = RESULT_DIR / "attempt.json"
MATERIALIZE = RESULT_DIR / "materialize.json"
TERMINAL = RESULT_DIR / "terminal.json"
FAILURE = RESULT_DIR / "failure.json"


def _nvidia_smi_snapshot(label: str) -> dict:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,compute_mode",
             "--format=csv,noheader,nounits"],
            check=True, text=True, capture_output=True,
        )
        rows = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    except Exception as error:  # pragma: no cover - environment dependent
        raise SystemExit(f"nvidia-smi snapshot failed ({label}): {error}")
    compute = [row for row in rows if row.split(",")[2].strip() not in ("0", "0 %")]
    return {
        "label": label,
        "rows": rows,
        "any_gpu_compute_active": bool(compute),
        "compute_rows": compute,
    }


def _require_idle(snapshot: dict) -> None:
    if snapshot["any_gpu_compute_active"]:
        raise SystemExit(
            f"GPU compute is active during a CPU-only run: {snapshot['compute_rows']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 10, 30])
    parser.add_argument("--surfaces", nargs="+", default=["external", "within"],
                        choices=["external", "within"])
    parser.add_argument("--smoke-sessions", type=int, default=None,
                        help="limit the roster per surface (never governing)")
    parser.add_argument("--stage", choices=["all", "materialize", "audit"],
                        default="all")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "streaming_calibration_exp"))
    sys.path.insert(0, str(REPO_ROOT / "sua_exploration"))
    sys.path.insert(0, str(ROOT))
    import os

    os.environ.setdefault("SUBC_DATA_ROOT", str(REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"))
    os.environ.setdefault("SUBM_DATA_ROOT", str(REPO_ROOT / "sua_exploration/data/dandi_000688/sub-M"))

    from src.tfpd_lane import receipt as lane_receipt
    from src.slot_audit_v1 import audit as slot_audit
    from src.slot_audit_v1 import materialize as slot_materialize
    from src.slot_audit_v1 import plan

    environment = lane_receipt.enforce_environment()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("CUDA_VISIBLE_DEVICES must be the empty string (CPU-only lane)", file=sys.stderr)
        return 3
    before = _nvidia_smi_snapshot("before")
    _require_idle(before)

    # fresh result root (immutable-receipt discipline: this run's receipts
    # must not pre-exist; a prior failure receipt blocks any re-run)
    will_write = []
    if args.stage in ("all", "materialize"):
        will_write += [ATTEMPT, MATERIALIZE]
    if args.stage in ("all", "audit"):
        will_write += [TERMINAL]
    existing = [path for path in will_write if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing receipts: {existing}")
    if FAILURE.exists():
        raise SystemExit(f"a failure receipt is already present: {FAILURE}")
    if args.stage == "audit" and not MATERIALIZE.exists():
        raise SystemExit(f"audit stage needs the materialize receipt: {MATERIALIZE}")

    started = time.perf_counter()

    def progress(surface, session, parse_s, rows):
        last = rows[-1]
        print(
            f"[slot] {surface:>8} {session} M{last['budget']:<2}..M{rows[0]['budget']:<2} "
            f"baseline r2 {last['probe_baseline_r2']:+.6f} drift "
            f"{last['delta_r2']:+.1e} ({last['n_windows']} windows, parse "
            f"{parse_s:.1f}s, decode {sum(r['decode_wall_seconds'] for r in rows):.1f}s)",
            flush=True,
        )

    try:
        if args.stage in ("all", "materialize"):
            closure = lane_receipt.source_closure(ROOT, plan.CLOSURE_PATTERNS)
            lane_receipt.write_receipt_transactionally(ATTEMPT, {
                "schema": "slot_audit_v1_attempt_v1",
                "status": "ATTEMPT_RESERVED",
                "cell": "SLOT_AUDIT_V1",
                "work_order": "docs/WORKORDER_SLOT_AUDIT_V1_20260829.md",
                "guidance_sections": ["2.2 continuity evidence", "5 output-slot audit"],
                "question": (
                    "are some already-supervised output slots systematically "
                    "better calibrated than the last bin, or is the trajalign "
                    "gain only covariance reduction from averaging equivalent "
                    "slots?"
                ),
                "surfaces": list(args.surfaces),
                "budgets": [int(b) for b in args.budgets],
                "smoke_sessions_per_surface": args.smoke_sessions,
                "pre_registration": {
                    "materialization_parity": (
                        "per (surface, budget, session): last-bin house R2 == "
                        "sealed probe baseline row (tol 1e-12), n_windows equal, "
                        "full [W,50,2] prediction SHA bit-exact"
                    ),
                    "per_slot_law": (
                        "window w slot s speaks about absolute bin w+s; targets "
                        "from the session behavior-bin array; house "
                        "variance-weighted R2 per session, equal-session means"
                    ),
                    "delta_curve": (
                        "residual pairs at slot separation delta in 1..49 over "
                        "shared absolute bins, centered per session per slot "
                        "column, pooled correlation/covariance/stds"
                    ),
                    "subsets": (
                        "{49}, {45..49}, {40..49}, all-50 plus one "
                        "within-source-selected best single slot per budget "
                        "(the only selection), scored vs the governing "
                        "last-bin target"
                    ),
                    "trajalign": (
                        "the frozen probe law by import, K in {2,4,8,16}, mean "
                        "weights; must reproduce the sealed probe rows "
                        "(tol 1e-9)"
                    ),
                    "reading": (
                        "(a) redundancy_ensembling / (b) stable_better_subset "
                        ">= 80% sessions both surfaces / (c) session_dependent; "
                        "decided mechanically, no gate, no training authority"
                    ),
                },
                "runtime_bound_seconds": plan.RUNTIME_BOUND_SECONDS,
                "stop_rule": (
                    "hard failure on any anchor/parity/reproduction drift; "
                    "named limitation recorded and the sub-part stopped if the "
                    "frozen path blocks it"
                ),
                "cpu_only": True,
                "target_optimizer_backward_update": 0,
                "closure": closure,
            })
            print(f"[slot] attempt receipt -> {ATTEMPT}", flush=True)

            payload = slot_materialize.run_materialization(
                ROOT, budgets=tuple(args.budgets), surfaces=tuple(args.surfaces),
                on_progress=progress, smoke_sessions=args.smoke_sessions,
            )
            payload["environment"] = environment
            payload["gpu_disclosure"] = {"before": before}
            lane_receipt.write_receipt_transactionally(MATERIALIZE, payload)
            print(
                f"[slot] materialize receipt -> {MATERIALIZE}\n"
                f"[slot] parity {payload['decodes']} decodes, max |dR2| "
                f"{payload['max_abs_r2_drift']:.1e}, SHAs bit-exact "
                f"{payload['all_prediction_shas_bitexact']}, wall "
                f"{payload['throughput']['wall_seconds_total']:.1f}s",
                flush=True,
            )
        if args.stage in ("all", "audit"):
            terminal = slot_audit.run_audit(
                ROOT, budgets=tuple(args.budgets), surfaces=tuple(args.surfaces),
            )
            after = _nvidia_smi_snapshot("after")
            _require_idle(after)
            terminal["environment"] = environment
            terminal["gpu_disclosure"] = {"after": after}
            lane_receipt.write_receipt_transactionally(TERMINAL, terminal)
            print(
                f"[slot] terminal receipt -> {TERMINAL}\n"
                + slot_audit.headline(terminal)
                + f"\n[slot] GPUs idle before/after; total wall "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    except BaseException as error:  # preserve an atomic failure receipt
        if not FAILURE.exists():
            lane_receipt.write_receipt_transactionally(FAILURE, {
                "schema": "slot_audit_v1_failure_v1",
                "status": "FAILED",
                "error_class": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "stage": args.stage,
                "target_optimizer_backward_update": 0,
                "model_or_checkpoint_updated": False,
            })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
