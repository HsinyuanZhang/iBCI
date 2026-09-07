#!/usr/bin/env python3
"""Run the E6 (PCA-8 projection control) and E7 (supervised mismatched control)
additions to behavior_manifold_v2 and bind them in terminal_v2.

E6 reuses the sealed three-fold source-only full-decoder forward machinery and
projects its raw predictions through a source-frozen PCA-8 basis.  E7 trains
the same 3,224-parameter encoder/decoder family with a supervised but
deployment-mismatched objective (full-session in-loop ridge) and deploys it
through the exact frozen M10 closed-form rule, three seed offsets, nested
source-only selection.  Zero target backward/optimizer steps; no formal or
held-out surface opened; no frozen artifact modified.
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

from sua_exploration.behavior_manifold_v2.e6_pca_projection import experiment_e6
from sua_exploration.behavior_manifold_v2.e7_supervised_control import (
    SupervisedMismatchedSpec,
    three_seed_supervised_ensemble,
)
from sua_exploration.behavior_manifold_v2.frozen import THREEFOLD_RECEIPT_SHA256, _sha_file
from sua_exploration.behavior_manifold_v2.protocol import write_receipt
from sua_exploration.h1_m1_priority_v1.execute import load_m1_sessions

E7_SPECS = tuple(SupervisedMismatchedSpec(8, 64, weight) for weight in (0.03, 0.1, 0.3, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = time.perf_counter()
    digests: dict[str, str] = {}
    sessions = load_m1_sessions()
    print(f"loaded {len(sessions)} M1 held-in-calibration sessions", flush=True)

    e6 = experiment_e6(sessions, device=args.device)
    print(
        "E6 PCA-8 projection: raw "
        f"{e6['equal_fold']['raw_full_decoder_mean_r2']:.6f} -> pca8 "
        f"{e6['equal_fold']['pca8_projected_mean_r2']:.6f} (delta {e6['equal_fold']['paired_delta_mean']:+.6f}); "
        f"calibration-aware reference {e6['side_by_side_with_calibration_aware']['calibration_aware_paired_delta_mean']:+.6f}",
        flush=True,
    )
    print(f"E6 reading: {e6['side_by_side_with_calibration_aware']['reading']}", flush=True)
    e6["elapsed_seconds"] = time.perf_counter() - started
    digests["e6_pca8_projection.json"] = write_receipt("e6_pca8_projection.json", e6)

    e7 = three_seed_supervised_ensemble(sessions, E7_SPECS, device=args.device)
    print(
        "E7 supervised-mismatched ensemble: "
        f"{e7['equal_session']['mean_r2']:.9f} vs DirectRidge "
        f"{e7['directridge_equal_session']['mean_r2']:.9f} "
        f"(gain {e7['paired_delta_vs_directridge']['mean']:+.6f}, "
        f"{e7['paired_delta_vs_directridge']['positive_sessions']}/4 sessions positive)",
        flush=True,
    )
    print(f"E7 per-seed means: {json.dumps(e7['per_seed_equal_session_mean'], sort_keys=True)}", flush=True)
    print(f"E7 reading: {e7['pre_registered_reading']['reading']}", flush=True)
    e7["elapsed_seconds"] = time.perf_counter() - started
    digests["e7_supervised_control.json"] = write_receipt("e7_supervised_control.json", e7)

    prior_terminal = Path(__file__).resolve().parents[1] / "results" / "behavior_manifold_v2" / "terminal.json"
    terminal = {
        "schema": "m1_behavior_manifold_v2_terminal_v2",
        "status": "TERMINAL_COMPLETE_BEHAVIOR_MANIFOLD_V2_WITH_E6_E7",
        "predecessor_terminal_sha256": _sha_file(prior_terminal),
        "threefold_receipt_sha256": THREEFOLD_RECEIPT_SHA256,
        "artifacts": dict(digests),
        "e6_reading": e6["side_by_side_with_calibration_aware"]["reading"],
        "e6_pca8_delta": e6["equal_fold"]["paired_delta_mean"],
        "e6_calibration_aware_reference_delta": e6["side_by_side_with_calibration_aware"]["calibration_aware_paired_delta_mean"],
        "e7_gain": e7["paired_delta_vs_directridge"]["mean"],
        "e7_reading": e7["pre_registered_reading"]["reading"],
        "total_elapsed_seconds": time.perf_counter() - started,
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
        },
    }
    write_receipt("terminal_v2.json", terminal)
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "e6_pca8_delta": terminal["e6_pca8_delta"],
                "e7_gain": terminal["e7_gain"],
                "elapsed_seconds": round(terminal["total_elapsed_seconds"], 1),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
