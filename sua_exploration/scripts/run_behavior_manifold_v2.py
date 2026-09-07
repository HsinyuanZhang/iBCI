#!/usr/bin/env python3
"""Run the behavior_manifold_v2 experiment set (E1--E5) on the frozen M1 route.

Everything is closed-form or frozen-weight: the only GPU work is the
deterministic rebuild of the twelve frozen calibration-aware seed manifolds and
their nested inner refits, each verified by state SHA and prediction SHA
against the published receipts.  No target backward pass, no target optimizer
step, no formal or held-out surface is opened, and the frozen artifacts are
never modified.
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

from sua_exploration.behavior_manifold_v2.e1_oracle import experiment_e1
from sua_exploration.behavior_manifold_v2.e2_budget import experiment_e2
from sua_exploration.behavior_manifold_v2.e3_spectrum import experiment_e3
from sua_exploration.behavior_manifold_v2.e4_attribution import experiment_e4
from sua_exploration.behavior_manifold_v2.e5_estimators import experiment_e5
from sua_exploration.behavior_manifold_v2.frozen import rebuild_frozen_deployment
from sua_exploration.behavior_manifold_v2.protocol import write_receipt
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions


def _print_e1(e1: dict) -> None:
    matrix = e1["ceiling_matrix"]
    print("=" * 78, flush=True)
    print("E1 ORACLE-LATENT CEILING MATRIX (equal-session R2, strict post-M10 query)", flush=True)
    print("=" * 78, flush=True)
    header = f"{'family':<14}{'M10 deployable':>16}{'oracle full-session':>22}"
    print(header, flush=True)
    for family in ("DirectRidge", "q8_route"):
        row = matrix[family]
        print(
            f"{family:<14}{row['m10_deployable']['equal_session_mean_r2']:>16.9f}"
            f"{row['oracle_full_session']['equal_session_mean_r2']:>22.9f}",
            flush=True,
        )
    print(f"per-session oracle q8: {matrix['q8_route']['oracle_full_session']['per_session']}", flush=True)
    print(f"per-session oracle DirectRidge: {matrix['DirectRidge']['oracle_full_session']['per_session']}", flush=True)
    reading = e1["preregistered_reading"]
    print(
        f"oracle-q8 = {reading['oracle_q8_equal_session_mean']:.9f}; thresholds "
        f"low<= {e1['preregistered_reading']['thresholds']['low']}, high >= {reading['thresholds']['high']}",
        flush=True,
    )
    print(f"PRE-REGISTERED READING: {reading['reading']}", flush=True)
    print("=" * 78, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.perf_counter()
    digests: dict[str, str] = {}
    sessions = load_m1_sessions()
    print(f"loaded {len(sessions)} M1 held-in-calibration sessions", flush=True)

    frozen = rebuild_frozen_deployment(sessions, device=args.device)
    summary_q8 = frozen.evidence["equal_session"]
    print(
        "frozen q8 deployment rebuilt and SHA-verified: "
        + json.dumps(summary_q8, sort_keys=True),
        flush=True,
    )
    rebuild_receipt = {
        "schema": "m1_behavior_manifold_v2_frozen_deployment_rebuild_v1",
        "status": "COMPLETE_FROZEN_REBUILD_SHA_VERIFIED",
        "frozen_evidence": frozen.evidence,
        "directridge_baseline_recomputed": frozen.direct_baseline["equal_session"],
        "elapsed_seconds": time.perf_counter() - started,
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
        },
    }
    digests["frozen_deployment_rebuild.json"] = write_receipt("frozen_deployment_rebuild.json", rebuild_receipt)

    e1 = experiment_e1(frozen, sessions=sessions)
    _print_e1(e1)
    e1["elapsed_seconds"] = time.perf_counter() - started
    digests["e1_oracle_ceiling.json"] = write_receipt("e1_oracle_ceiling.json", e1)

    e2 = experiment_e2(frozen, sessions=sessions)
    print(
        "E2 budget sweep paired gains (q8 - DirectRidge): "
        + json.dumps(e2["gain_by_budget"], sort_keys=True),
        flush=True,
    )
    e2["elapsed_seconds"] = time.perf_counter() - started
    digests["e2_budget_sweep.json"] = write_receipt("e2_budget_sweep.json", e2)

    e3 = experiment_e3(sessions=sessions)
    print(
        "E3 participation ratios: "
        + json.dumps(
            {key: row["participation_ratio_equal_session_mean"] for key, row in e3["summary"].items()},
            sort_keys=True,
        ),
        flush=True,
    )
    digests["e3_participation_ratios.json"] = write_receipt("e3_participation_ratios.json", e3)

    e4 = experiment_e4(frozen, sessions=sessions)
    print(
        "E4 attribution loading-vs-delta: "
        + json.dumps(e4["correlation_loading_vs_delta"], sort_keys=True),
        flush=True,
    )
    digests["e4_output_attribution.json"] = write_receipt("e4_output_attribution.json", e4)

    e5 = experiment_e5(
        frozen,
        sessions=sessions,
        device=args.device,
        oracle_q8_equal_session_mean=e1["ceiling_matrix"]["q8_route"]["oracle_full_session"]["equal_session_mean_r2"],
    )
    print(
        "E5 estimator variants equal-session means: "
        + json.dumps(
            {key: row["equal_session_mean_r2"] for key, row in e5["variants"].items()}, sort_keys=True
        ),
        flush=True,
    )
    e5["elapsed_seconds"] = time.perf_counter() - started
    digests["e5_estimator_upgrades.json"] = write_receipt("e5_estimator_upgrades.json", e5)

    terminal = {
        "schema": "m1_behavior_manifold_v2_terminal_v1",
        "status": "TERMINAL_COMPLETE_BEHAVIOR_MANIFOLD_V2",
        "artifacts": dict(digests),
        "decision_gate": e1["preregistered_reading"],
        "e2_gain_by_budget": e2["gain_by_budget"],
        "e3_participation_summary": {
            key: row["participation_ratio_equal_session_mean"] for key, row in e3["summary"].items()
        },
        "e4_correlation": e4["correlation_loading_vs_delta"],
        "e5_variant_means": {
            key: row["equal_session_mean_r2"] for key, row in e5["variants"].items()
        },
        "total_elapsed_seconds": time.perf_counter() - started,
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
        },
    }
    write_receipt("terminal.json", terminal)
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "decision_gate": terminal["decision_gate"],
                "elapsed_seconds": round(terminal["total_elapsed_seconds"], 1),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
