#!/usr/bin/env python3
"""Run E8: the deployment-legal penultimate-to-latent readout cell.

Frozen-weight, inference-only: re-verify each published threefold replay fold,
extract the penultimate representation on the fold's training sessions and on
the fold target's own held-in-calib bins, fit the M10-support h -> z = enc(y)
probe (the frozen q8 route's exact label budget), decode through each frozen
seed manifold, and score behaviour on the published replay query against the
full decoder's own raw head.  Diagnostics (raw-head mix, probe-raw-decode) are
descriptive only.  Zero target backward/optimizer steps; no formal or minival
surface opened; no frozen artifact modified.
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

from sua_exploration.behavior_manifold_v2.e8_readout import experiment_e8
from sua_exploration.behavior_manifold_v2.frozen import rebuild_frozen_deployment
from sua_exploration.behavior_manifold_v2.protocol import write_receipt
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions


def _print_table(body: dict) -> None:
    print("=" * 100, flush=True)
    print("E8 PROBE-LATENT-DECODE (deployment-legal; pooled variance-weighted R2 on the published replay query)", flush=True)
    print("=" * 100, flush=True)
    header = (
        f"{'fold':>4} {'target':>14} {'sup':>5} {'replay':>7} "
        f"{'seed0':>8} {'seed1':>8} {'seed2':>8} {'ens':>8} {'rawHead':>8} {'d(ens)':>8} {'mix(a)':>8} {'rawDec(b)':>9}"
    )
    print(header, flush=True)
    for fold in ("0", "1", "2"):
        block = body["folds"][fold]
        seeds = block["seeds"]
        print(
            f"{fold:>4} {block['target_session']:>14} {seeds[0]['support_bins']:>5} {seeds[0]['replay_bins']:>7} "
            f"{seeds[0]['replay_r2']:>8.4f} {seeds[1]['replay_r2']:>8.4f} {seeds[2]['replay_r2']:>8.4f} "
            f"{block['ensemble']['r2']:>8.4f} {block['raw_head']['r2']:>8.4f} {block['ensemble']['delta_vs_raw_head']:>+8.4f} "
            f"{block['diagnostics_descriptive_only']['mix_raw_head_and_ensemble']['r2']:>8.4f} "
            f"{block['diagnostics_descriptive_only']['probe_raw_decode']['r2']:>9.4f}",
            flush=True,
        )
    equal = body["equal_fold"]
    print("-" * 100, flush=True)
    print(
        f"{'':>4} {'EQUAL-FOLD':>14} {'':>5} {'':>7} "
        f"{equal['per_seed']['0']:>8.4f} {equal['per_seed']['1']:>8.4f} {equal['per_seed']['2']:>8.4f} "
        f"{equal['probe_latent_decode_ensemble']:>8.4f} {equal['raw_head']:>8.4f} {equal['delta']:>+8.4f} "
        f"{equal['mix_diagnostic']:>8.4f} {equal['probe_raw_decode_diagnostic']:>9.4f}",
        flush=True,
    )
    verdict = body["preregistered_reading"]
    print(
        f"probe-latent-decode ensemble equal-fold {equal['probe_latent_decode_ensemble']:.6f} vs raw head "
        f"{equal['raw_head']:.6f} (delta {equal['delta']:+.6f}; improvement needs >= +0.01; "
        f"frozen q8 M10-route reference {body['references']['frozen_q8_m10_route_equal_session']:.4f})",
        flush=True,
    )
    print(f"PRE-REGISTERED VERDICT: {verdict['verdict']}", flush=True)
    print(f"reading: {verdict['reading']}", flush=True)
    print("=" * 100, flush=True)


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

    body = experiment_e8(frozen, sessions=sessions, device=args.device)
    _print_table(body)
    body["total_elapsed_seconds"] = time.perf_counter() - started
    digest = write_receipt("e8_readout.json", body)
    print(
        json.dumps(
            {
                "status": body["status"],
                "verdict": body["preregistered_reading"]["verdict"],
                "equal_fold_ensemble": body["equal_fold"]["probe_latent_decode_ensemble"],
                "delta_vs_raw_head": body["equal_fold"]["delta"],
                "receipt_sha256": digest,
                "elapsed_seconds": round(body["total_elapsed_seconds"], 1),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
