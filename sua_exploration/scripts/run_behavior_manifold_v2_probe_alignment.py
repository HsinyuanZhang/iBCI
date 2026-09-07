#!/usr/bin/env python3
"""Run the representation-manifold alignment probe (the aux-head gate).

Frozen-weight, inference-only: re-verify each of the three published
source-only full-SPINT fold decoders against the threefold receipt, extract
the penultimate representation (the pooled cross-attention token states
feeding fc_out) on each fold's own three training sessions, and ask whether
the frozen q8 manifold codes are already linearly present there, against raw
16-D behaviour and PCA-8 controls, under the lane's closed-form ridge and
nested lambda discipline.  Zero target backward/optimizer steps; no formal or
minival surface opened; no frozen artifact modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "SPINT-main"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sua_exploration.behavior_manifold_v2.frozen import rebuild_frozen_deployment
from sua_exploration.behavior_manifold_v2.probe_alignment import (
    KILL_RATIO,
    GREENLIGHT_RATIO,
    experiment_probe_alignment,
)
from sua_exploration.behavior_manifold_v2.protocol import write_receipt
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions


def _print_table(body: dict) -> None:
    print("=" * 98, flush=True)
    print("PROBE ALIGNMENT TABLE (pooled variance-weighted query R2, strict post-M10)", flush=True)
    print("=" * 98, flush=True)
    header = (
        f"{'fold':>4} {'session':>14} {'sup':>5} {'qry':>6} "
        f"{'lat0':>8} {'lat1':>8} {'lat2':>8} {'latMean':>8} {'raw16':>8} {'pca8':>8} {'ratio':>7}"
    )
    print(header, flush=True)
    for fold in ("0", "1", "2"):
        block = body["folds"][fold]
        for name in sorted(block["sessions"]):
            row = block["sessions"][name]
            r2 = row["r2"]
            print(
                f"{fold:>4} {name:>14} {row['support_bins']:>5} {row['query_bins']:>6} "
                f"{r2['latent_seed0']:>8.4f} {r2['latent_seed1']:>8.4f} {r2['latent_seed2']:>8.4f} "
                f"{row['latent_seed_mean_r2']:>8.4f} {r2['raw16']:>8.4f} {r2['pca8']:>8.4f} "
                f"{row['ratio_latent_mean_vs_raw16']:>7.3f}",
                flush=True,
            )
        means = block["fold_mean_r2"]
        print(
            f"{fold:>4} {'FOLD-MEAN':>14} {'':>5} {'':>6} "
            f"{means['latent_seed0']:>8.4f} {means['latent_seed1']:>8.4f} {means['latent_seed2']:>8.4f} "
            f"{block['fold_latent_seed_mean_r2']:>8.4f} {means['raw16']:>8.4f} {means['pca8']:>8.4f} "
            f"{block['fold_ratio_latent_mean_vs_raw16']:>7.3f}",
            flush=True,
        )
    overall = body["equal_cell"]
    print("-" * 98, flush=True)
    print(
        f"{'':>4} {'EQUAL-CELL':>14} {'':>5} {'':>6} "
        f"{overall['mean_r2']['latent_seed0']:>8.4f} {overall['mean_r2']['latent_seed1']:>8.4f} "
        f"{overall['mean_r2']['latent_seed2']:>8.4f} {overall['latent_seed_mean_r2']:>8.4f} "
        f"{overall['mean_r2']['raw16']:>8.4f} {overall['mean_r2']['pca8']:>8.4f} "
        f"{overall['ratio_latent_mean_vs_raw16']:>7.3f}",
        flush=True,
    )
    verdict = body["preregistered_reading"]
    print(
        f"primary ratio = {verdict['ratio']:.6f} (kill >= {KILL_RATIO}, green-light <= {GREENLIGHT_RATIO}); "
        f"mean of per-cell ratios = {overall['mean_of_per_cell_ratios']:.6f} "
        f"(range {overall['per_cell_ratio_range'][0]:.4f}..{overall['per_cell_ratio_range'][1]:.4f})",
        flush=True,
    )
    print(f"PRE-REGISTERED VERDICT: {verdict['verdict']}", flush=True)
    print(f"reading: {verdict['reading']}", flush=True)
    print("=" * 98, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.perf_counter()
    sessions = load_m1_sessions()
    print(f"loaded {len(sessions)} M1 held-in-calibration sessions", flush=True)

    frozen = rebuild_frozen_deployment(sessions, device=args.device)
    print(
        "frozen q8 deployment rebuilt and SHA-verified: "
        + json.dumps(frozen.evidence["equal_session"], sort_keys=True),
        flush=True,
    )

    body = experiment_probe_alignment(frozen, sessions=sessions, device=args.device)
    _print_table(body)
    for fold in ("0", "1", "2"):
        block = body["folds"][fold]
        for name in sorted(block["sessions"]):
            decoder = block["extraction"]["sessions"][name]["decoder_query_metrics"]
            print(
                f"fold {fold} {name}: decoder last-timestep in-sample query R2 "
                f"{decoder['pooled_variance_weighted_r2']:.4f} (probe raw16 "
                f"{block['sessions'][name]['r2']['raw16']:.4f})",
                flush=True,
            )
    body["total_elapsed_seconds"] = time.perf_counter() - started
    digest = write_receipt("probe_alignment.json", body)
    print(
        json.dumps(
            {
                "status": body["status"],
                "verdict": body["preregistered_reading"]["verdict"],
                "primary_ratio": body["preregistered_reading"]["ratio"],
                "receipt_sha256": digest,
                "elapsed_seconds": round(body["total_elapsed_seconds"], 1),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
