"""Runner for the continuity probe v1 — output-level smoothing arms (CPU-only).

Lane: continuity_probe_v1 (frozen-weight, inference-only).  This runner never
trains: it strict-loads the sealed Cell-D SWA through the Z1 provenance
harness, reproduces the strict static deployment baselines at M4/M10/M30 on
external-15 and within-6 (anchored to the P4 CPU receipt AND the sealed GPU
receipts), applies the pre-registered causal / trajectory-aligned smoothing
arms to the raw per-window predictions, asserts causality per session, and
publishes a transactional 0444+sidecar receipt under
``results/continuity_probe_v1/``.

Boundaries enforced here (mirror of run_calibration_gap_p4.py):
- CUDA_VISIBLE_DEVICES must be the empty string; nvidia-smi captured before
  AND after, both GPUs must show zero compute processes;
- PYTHONNOUSERSITE (via tfpd_lane.receipt.enforce_environment);
- zero target optimizer/backward/update; sealed model state digest compared
  before/after;
- every consumed receipt is SHA-verified at load (ledger + the pinned P4
  receipt inside continuity_probe_v1);
- the deployment carrier recipe is reproduced unchanged (ridge-T4 0.1;
  D-opt-first-30 @M4, chronological @M10/M30); smoothing touches OUTPUTS only.

Usage (spint env):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
    python scripts/run_continuity_probe_v1.py [--budgets 4 10 30] [--surfaces external within]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RESULT_DIR = ROOT / "results/continuity_probe_v1"
RESULT = RESULT_DIR / "continuity_probe_v1.json"
CLOSURE_PATTERNS = (
    "src/continuity_probe_v1.py",
    "src/calibration_gap_v1/p4_stream_stats.py",
    "src/calibration_gap_v1/z1_oracle_cells.py",
    "scripts/run_continuity_probe_v1.py",
    "tests/test_continuity_probe_v1.py",
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


def _print_table(payload: dict) -> str:
    lines = []
    for surface in ("external", "within"):
        for budget in (4, 10, 30):
            cell = payload["readings"]["table"][f"{surface}_M{budget}"]
            lines.append(
                f"[table] {surface:>8} M{budget:<2} baseline "
                f"{cell['baseline_mean_r2']:+.4f}"
            )
            for arm in sorted(cell["arms"]):
                stats = cell["arms"][arm]
                lines.append(
                    f"[table] {surface:>8} M{budget:<2} {arm:<24} "
                    f"{stats['mean_delta']:+.4f} median {stats['median_delta']:+.4f} "
                    f"{stats['n_positive']:>2}/{stats['n_total']:<2} + "
                    f"boot95 [{stats['bootstrap_95_interval'][0]:+.4f}, "
                    f"{stats['bootstrap_95_interval'][1]:+.4f}]"
                )
    return "\n".join(lines)


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
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("CUDA_VISIBLE_DEVICES must be the empty string (CPU-only lane)", file=sys.stderr)
        return 3
    before = _nvidia_smi_snapshot("before")
    _require_idle(before)
    closure = lane_receipt.source_closure(ROOT, CLOSURE_PATTERNS)

    from src.calibration_gap_v1 import ledger as gap_ledger
    from src.calibration_gap_v1 import z1_oracle_cells as z1
    from src import continuity_probe_v1 as probe

    started = time.perf_counter()
    ledger = gap_ledger.load_ledger(ROOT)
    runtime = z1.HonestOracleRuntime(REPO_ROOT)

    def progress(inputs, rows, audit):
        base = rows[0]
        audit_ok = all(
            a.get("invariant_upto_cut_bitexact", True)
            and a.get("invariant_before_support_bitexact", True)
            for a in audit["arms"]
        )
        print(
            f"[probe] {base['surface']:>8} M{base['budget']:<2} baseline "
            f"{base['session']} r2={base['variance_weighted_r2']:+.6f} "
            f"({base['wall_seconds']:.1f}s) arms={len(rows) - 1} "
            f"causality_audit={'PASS' if audit_ok else 'FAIL'}",
            flush=True,
        )

    payload = probe.run_probe(
        ledger, runtime, budgets=tuple(args.budgets),
        surfaces=tuple(args.surfaces), on_progress=progress,
    )
    state_after = runtime.state_digest()
    payload["boundaries"]["sealed_state_unchanged"] = (
        state_after == payload["sealed_state_sha256"]
    )
    payload["boundaries"]["sealed_state_sha256_after"] = state_after
    runtime.close()
    if payload["boundaries"]["sealed_state_unchanged"] is not True:
        raise SystemExit("sealed Cell-D state changed during the continuity probe")
    payload["environment"] = environment
    payload["ledger_receipt_sha256s"] = {
        rel: gap_ledger.RECEIPTS[rel] for rel in sorted(gap_ledger.RECEIPTS)
    }
    payload["closure"] = closure

    after = _nvidia_smi_snapshot("after")
    _require_idle(after)
    payload["gpu_disclosure"] = {"before": before, "after": after}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    lane_receipt.write_receipt_transactionally(RESULT, payload)

    p4_anchor = payload["anchors"]["p4_strict_cpu_reproduction"]
    sealed_anchor = payload["anchors"]["sealed_receipt_reproduction"]
    print(
        f"[probe] receipt -> {RESULT}\n"
        f"[probe] p4 strict anchor max |dR2| = {p4_anchor['max_abs_delta_r2']:.2e} "
        f"({p4_anchor['prediction_sha256_bitexact_matches']}/"
        f"{p4_anchor['sessions_checked']} prediction SHAs bit-exact)\n"
        f"[probe] sealed GPU anchor max |dR2| = {sealed_anchor['max_abs_delta_r2']:.2e} "
        f"(tol {sealed_anchor['tolerance']:.0e})\n"
        + _print_table(payload) + "\n"
        + "\n".join(
            f"[probe] verdict {key}: {value['verdict']}"
            for key, value in payload["readings"]["verdicts"].items()
        ) + f"\n[gate] GPUs idle before/after; total wall {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
