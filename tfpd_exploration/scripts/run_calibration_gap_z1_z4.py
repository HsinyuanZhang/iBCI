"""Runner for the zero-cost gate batch Z1/Z4 (CPU-only, frozen weights).

Lane: calibration_gap_v1 (HANDOFF_CALIBRATION_GAP_DECOMPOSITION_20260824.md
§2).  This runner never trains: it strict-loads the sealed Cell-D SWA, runs
the honest-M oracle forward cells (Z1) and the pure-linear-algebra U-stability
diagnostic (Z4), and publishes transactional 0444+sidecar receipts under
``results/calibration_gap_v1/``.

Boundaries enforced here:
- CUDA_VISIBLE_DEVICES must be the empty string; nvidia-smi is captured
  before AND after and both GPUs must show zero compute processes;
- PYTHONNOUSERSITE (via tfpd_lane.receipt.enforce_environment);
- zero target optimizer/backward/update; sealed model state digest compared
  before/after;
- C3-support cells are labelled leakage-diagnostic everywhere;
- every consumed receipt is SHA-verified by the ledger at load time.

Usage (spint env):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
    python scripts/run_calibration_gap_z1_z4.py [--z4-only|--z1-only]
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
Z1_RESULT = RESULT_ROOT / "z1_oracle_cells.json"
Z4_RESULT = RESULT_ROOT / "z4_subspace.json"
CLOSURE_PATTERNS = (
    "src/calibration_gap_v1/*.py",
    "scripts/run_calibration_gap_z1_z4.py",
    "tests/test_calibration_gap_z1_z4.py",
)

# Z1 runtime discipline (handoff §2): if the batch projects beyond this, the
# interim payload path engages (priority order M4 -> M10 -> M30).
Z1_MAX_TOTAL_SECONDS = 2.5 * 3600.0


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


def _write_with_gpu_after(path: Path, payload: dict, before: dict) -> None:
    """Capture the after-snapshot, require idle GPUs, then write once."""
    from src.tfpd_lane import receipt as lane_receipt

    after = _nvidia_smi_snapshot("after")
    _require_idle(after)
    payload["gpu_disclosure"] = {"before": before, "after": after}
    lane_receipt.write_receipt_transactionally(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z4-only", action="store_true")
    parser.add_argument("--z1-only", action="store_true")
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
    from src.calibration_gap_v1 import z1_oracle_cells as z1
    from src.calibration_gap_v1 import z4_subspace as z4

    started = time.perf_counter()
    ledger = gap_ledger.load_ledger(ROOT)
    z4_calibs: dict[str, object] = {}

    if not args.z4_only:
        runtime = z1.HonestOracleRuntime(REPO_ROOT)

        def collect(inputs, rows):
            if inputs.surface == "external":
                z4_calibs[inputs.session] = inputs.calib.numpy().copy()

        def progress(inputs, rows):
            for row in rows:
                print(
                    f"[z1] {row['surface']:>8} M{row['budget']:<2} {row['session']} "
                    f"r2={row['variance_weighted_r2']:+.6f} ({row['wall_seconds']:.1f}s)",
                    flush=True,
                )

        first_cell_started = time.perf_counter()

        def measure_first(inputs, rows):
            nonlocal first_cell_started
            if first_cell_started is not None:
                elapsed = time.perf_counter() - first_cell_started
                sessions = len(runtime.external_roster) + len(runtime.within_roster)
                projected = elapsed * sessions
                print(
                    f"[z1] first session took {elapsed:.1f}s -> projected batch "
                    f"{projected / 60:.1f} min over {sessions} sessions",
                    flush=True,
                )
                first_cell_started = None

        def combined(inputs, rows):
            collect(inputs, rows)
            progress(inputs, rows)
            measure_first(inputs, rows)

        payload = z1.run_z1(
            ledger, runtime, max_total_seconds=Z1_MAX_TOTAL_SECONDS, on_progress=combined,
        )
        state_after = runtime.state_digest()
        payload["boundaries"]["sealed_state_unchanged"] = (
            state_after == payload["sealed_state_sha256"]
        )
        payload["boundaries"]["sealed_state_sha256_after"] = state_after
        runtime.close()
        if not payload["boundaries"]["sealed_state_unchanged"]:
            raise SystemExit("sealed Cell-D state changed during the Z1 run")
        payload["environment"] = environment
        payload["ledger_receipt_sha256s"] = {
            rel: gap_ledger.RECEIPTS[rel] for rel in sorted(gap_ledger.RECEIPTS)
        }
        payload["closure"] = closure
        payload["readings"] = z1.readings(payload)
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        _write_with_gpu_after(Z1_RESULT, payload, before)
        print(
            f"[z1] receipt -> {Z1_RESULT} | anchor max |dR2| = "
            f"{payload['anchor_check']['max_abs_delta_r2']:.2e} over "
            f"{payload['anchor_check']['sessions_checked']} sessions | reading: "
            f"{payload['readings'].get('pre_registered_reading')}",
            flush=True,
        )

    if not args.z1_only:
        if not z4_calibs:
            # Standalone Z4: materialize the external-15 calibration streams
            # through the same reviewed loader (no model needed afterwards).
            runtime = z1.HonestOracleRuntime(REPO_ROOT)
            for session_name in runtime.external_roster:
                inputs = runtime.materialize_session("external", session_name)
                z4_calibs[session_name] = inputs.calib.numpy().copy()
                print(f"[z4] materialized {session_name}", flush=True)
            runtime.close()
        z4_payload = z4.run_z4(z4_calibs)
        z4_payload["environment"] = environment
        z4_payload["closure"] = closure
        z4_payload["boundaries"] = {
            "cpu_only": True,
            "labels_used": False,
            "training": False,
            "activity_source_read_only": True,
        }
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        _write_with_gpu_after(Z4_RESULT, z4_payload, before)
        verdict = z4_payload["verdict"]
        print(
            f"[z4] receipt -> {Z4_RESULT} | k=6 median theta_max: "
            f"{verdict['k6_median_max_angle_deg']} | reading: {verdict['reading']}",
            flush=True,
        )

    after = _nvidia_smi_snapshot("after")
    _require_idle(after)
    print(
        f"[gate] GPUs idle before/after (compute processes: "
        f"{before['any_gpu_compute_active']}/{after['any_gpu_compute_active']}); "
        f"total wall {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
