#!/usr/bin/env python3
"""P5 row R3: external matched score of the source-selected fixed filter F2(alpha=0.7).

Post-processing ONLY on the cached CDM raw streams (zero decoder forwards).
Successor receipt root: results/learnable_output_filter_v1_p5/ (the P0-P4 root
is terminal and immutable). Reports, per budget on external-15 AND within-6:
raw (B0) vs F2(alpha=0.25, the P2' pre-registered value -> must reproduce the
sealed stage-A B1 numbers) vs F2(alpha=0.7, the P2 source-CV unanimous pick),
with paired per-session deltas, sign counts, and session bootstrap CIs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane.receipt import enforce_environment, sha256_file, write_receipt_transactionally
from src.learnable_output_filter_v1 import ladder, metrics, plan, streams

CACHE = ROOT / "cache/learnable_output_filter_v1"
OUT = ROOT / "results/learnable_output_filter_v1_p5"
ALPHAS = (0.25, 0.7)


def main() -> int:
    enforce_environment()
    if OUT.exists():
        raise SystemExit(f"fresh successor root required: {OUT}")
    OUT.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    cdm = streams.load_all(CACHE, "cdm")
    specs = {
        "raw": ladder.FilterSpec(level="F0"),
        **{f"alpha{a}": ladder.FilterSpec(level="F2", alpha=float(a)) for a in ALPHAS},
    }
    summary: dict[str, dict] = {}
    per_session_rows: list[dict] = []
    for surface in plan.SURFACES:
        for budget in plan.BUDGETS:
            keys = [k for k in cdm if k[0] == surface and k[2] == budget]
            if not keys:
                continue
            rows: dict[str, dict[str, float]] = {}
            for key in keys:
                stream = cdm[key]
                entry: dict[str, float] = {}
                for name, spec in specs.items():
                    result = ladder.apply_filter(stream, spec)
                    entry[name] = float(metrics.matrix_r2(result.blocks, stream))
                rows[stream.session] = entry
                per_session_rows.append({
                    "surface": surface, "budget": budget, "session": stream.session, **entry,
                })
            raw_mean = metrics.equal_session_mean([v["raw"] for v in rows.values()])
            cell: dict = {"n_sessions": len(rows), "raw_mean_r2": float(raw_mean)}
            for alpha in ALPHAS:
                name = f"alpha{alpha}"
                vals = [v[name] for v in rows.values()]
                deltas = [a - b for a, b in zip(vals, [v["raw"] for v in rows.values()])]
                rng = np.random.default_rng(42)
                n = len(deltas)
                arr = np.asarray(deltas, dtype=np.float64)
                idx_draws = rng.integers(0, n, size=(10000, n))
                boots = arr[idx_draws].mean(axis=1)
                cell[name] = {
                    "mean_r2": float(np.mean(vals)),
                    "raw_minus": float(np.mean(deltas)),
                    "positive_sessions": int(sum(d > 0 for d in deltas)),
                    "boot95": [float(np.quantile(boots, 0.025)),
                               float(np.quantile(boots, 0.975))],
                }
            summary[f"{surface}_m{budget}"] = cell

    body = {
        "schema": "learnable_output_filter_v1_p5_r3_v1",
        "status": "P5_R3_SCORED",
        "design_authority": ("tfpd_exploration/docs/DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_"
                             "20260829.md §9 P5 row R3 (fixed-filter activity-only)"),
        "authorization": "operator release 2026-08-29 (cached-stream post-processing only)",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_manifest_sha256": sha256_file(streams.manifest_path(CACHE)),
        "decoder_forwards_this_run": 0,
        "target_optimizer_backward_update": 0,
        "model_or_checkpoint_updated": False,
        "filter_specs": {"raw": "F0", "alpha0.25": "F2 pre-registered (P2' stage A)",
                         "alpha0.7": "F2 source-CV unanimous (P2)"},
        "reset_policy": "TRIAL_RESET",
        "summary": summary,
        "per_session": per_session_rows,
        "note": ("alpha0.25 rows must reproduce the sealed stage-A B1 matrix values "
                 "(anchor); alpha0.7 is the first external reading of the "
                 "source-selected filter"),
    }
    write_receipt_transactionally(OUT / "r3_external.json", body)
    for key, cell in sorted(summary.items()):
        print(key, "raw", round(cell["raw_mean_r2"], 4), end=" | ")
        for alpha in ALPHAS:
            row = cell[f"alpha{alpha}"]
            print(f"a{alpha} {row['raw_minus']:+.4f} ({row['positive_sessions']}/{cell['n_sessions']})", end=" ")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
