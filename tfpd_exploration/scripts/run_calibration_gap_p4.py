"""Runner for P4 — label-free eval-stream activity statistics (CPU-only).

Lane: calibration_gap_v1 (HANDOFF_CALIBRATION_GAP_DECOMPOSITION_20260824.md
§3 P4; Z6-Q1 gate PERMITTED with binding conditions — causal accumulation and
a mandatory FSU/TTA class declaration, never mixed with the strict
total-calibration column).  This runner never trains: it strict-loads the
sealed Cell-D SWA, reproduces the strict baseline + the Z1 anchor cells, runs
the two pre-registered transductive variants (P4a streaming normalization
statistics, P4b streaming identity update) at M4/M10/M30 on external-15 and
within-6, and publishes a transactional 0444+sidecar receipt under
``results/calibration_gap_v1/``.

Boundaries enforced here:
- CUDA_VISIBLE_DEVICES must be the empty string; nvidia-smi is captured
  before AND after and both GPUs must show zero compute processes;
- PYTHONNOUSERSITE (via tfpd_lane.receipt.enforce_environment);
- zero target optimizer/backward/update; sealed model state digest compared
  before/after;
- every consumed receipt is SHA-verified at load time (ledger + the Z1
  receipt pin inside p4_stream_stats);
- carrier fit unchanged (frozen deployment recipe); P4 cells are labelled
  transductive/TTA everywhere and never overwrite the strict column.

Usage (spint env):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
    python scripts/run_calibration_gap_p4.py [--budgets 4 10 30] [--surfaces external within]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RESULT_ROOT = ROOT / "results/calibration_gap_v1"
P4_RESULT = RESULT_ROOT / "p4_stream_stats.json"
CLOSURE_PATTERNS = (
    "src/calibration_gap_v1/p4_stream_stats.py",
    "src/calibration_gap_v1/z1_oracle_cells.py",
    "src/calibration_gap_v1/contract_notes.md",
    "scripts/run_calibration_gap_p4.py",
    "tests/test_calibration_gap_p4.py",
)


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
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "streaming_calibration_exp"))
    sys.path.insert(0, str(REPO_ROOT / "sua_exploration"))
    sys.path.insert(0, str(ROOT))
    import os

    os.environ.setdefault("SUBC_DATA_ROOT", str(REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"))
    os.environ.setdefault("SUBM_DATA_ROOT", str(REPO_ROOT / "sua_exploration/data/dandi_000688/sub-M"))

    from src.tfpd_lane import receipt as lane_receipt

    environment = lane_receipt.enforce_environment()
    import os as _os

    if _os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("CUDA_VISIBLE_DEVICES must be the empty string (CPU-only lane)", file=sys.stderr)
        return 3
    before = _nvidia_smi_snapshot("before")
    _require_idle(before)
    closure = lane_receipt.source_closure(ROOT, CLOSURE_PATTERNS)

    from src.calibration_gap_v1 import ledger as gap_ledger
    from src.calibration_gap_v1 import p4_stream_stats as p4
    from src.calibration_gap_v1 import z1_oracle_cells as z1

    started = time.perf_counter()
    ledger = gap_ledger.load_ledger(ROOT)
    z1_receipt_sha = lane_receipt.sha256_file(ROOT / p4.Z1_RECEIPT_REL)
    if z1_receipt_sha != p4.Z1_RECEIPT_SHA256:
        print(f"Z1 receipt SHA drift: {z1_receipt_sha}", file=sys.stderr)
        return 4

    runtime = z1.HonestOracleRuntime(REPO_ROOT)

    def progress(inputs, rows):
        for row in rows:
            extra = ""
            if row["variant"] == "p4b_stream_identity":
                extra = f" pool={row['identity_pool_size_final']} blocks={row['n_identity_refreshes']}"
            elif row["variant"] == "p4a_stream_norm":
                extra = (
                    f" blocks={row['n_identity_refreshes']} "
                    f"gain_med={row['final_block_gain_diagnostics']['gain_median']:.3f}"
                )
            print(
                f"[p4] {row['surface']:>8} M{row['budget']:<2} {row['variant']:<28} "
                f"{row['session']} r2={row['variance_weighted_r2']:+.6f} "
                f"margin={row['min_prefix_margin_bins']} ({row['wall_seconds']:.1f}s){extra}",
                flush=True,
            )

    payload = p4.run_p4(
        ledger, runtime, budgets=tuple(args.budgets), surfaces=tuple(args.surfaces),
        on_progress=progress,
    )
    state_after = runtime.state_digest()
    payload["boundaries"]["sealed_state_unchanged"] = (
        state_after == payload["sealed_state_sha256"]
    )
    payload["boundaries"]["sealed_state_sha256_after"] = state_after
    runtime.close()
    if payload["boundaries"]["sealed_state_unchanged"] is not True:
        raise SystemExit("sealed Cell-D state changed during the P4 run")
    payload["environment"] = environment
    payload["ledger_receipt_sha256s"] = {
        rel: gap_ledger.RECEIPTS[rel] for rel in sorted(gap_ledger.RECEIPTS)
    }
    payload["z1_receipt_sha256"] = z1_receipt_sha
    payload["closure"] = closure

    after = _nvidia_smi_snapshot("after")
    _require_idle(after)
    payload["gpu_disclosure"] = {"before": before, "after": after}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lane_receipt.write_receipt_transactionally(P4_RESULT, payload)

    primary = payload["readings"]["p4b_external_M4"]
    print(
        f"[p4] receipt -> {P4_RESULT}\n"
        f"[p4] z1 anchor max |dR2| = {payload['anchors']['z1_cpu_reproduction']['max_abs_delta_r2']:.2e} "
        f"({payload['anchors']['z1_cpu_reproduction']['prediction_sha256_bitexact_matches']}/"
        f"{payload['anchors']['z1_cpu_reproduction']['sessions_checked']} prediction SHAs bit-exact)\n"
        f"[p4] strict anchor max |dR2| = {payload['anchors']['strict_receipt_reproduction']['max_abs_delta_r2']:.2e}\n"
        f"[p4] PRIMARY P4b external M4 paired delta = {primary['mean_delta_r2']:+.4f} "
        f"(median {primary['median_delta_r2']:+.4f}, {primary['positive_sessions']}/{primary['session_count']} positive, "
        f"boot95 {primary['bootstrap_95']}) -> recovered share of Z1 activity term "
        f"{primary['recovered_share_of_z1_activity_ceiling']:.1%} / deployable ceiling "
        f"{primary['recovered_share_of_deployable_activity_ceiling']:.1%}\n"
        f"[p4] verdict: {payload['readings']['verdicts']['primary_p4b_external_M4']}\n"
        f"[p4] mechanism: {payload['readings']['verdicts']['mechanism_monotonicity']}\n"
        f"[p4] attribution: {payload['readings']['verdicts']['p4a_vs_p4b_attribution']}\n"
        f"[gate] GPUs idle before/after; total wall {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
