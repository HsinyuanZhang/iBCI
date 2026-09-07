#!/usr/bin/env python3
"""Print the inert plan for the M1 held-in vs held-out fold paired gap V1.

This public CLI intentionally does not import the route package, Torch, the
M1 parser, or an artifact-root helper.  A future root-reviewed in-process
caller alone may validate the frozen producer and anchor graphs, issue the
opaque capability, and run the four-session metric-only paired evaluation.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "604f4ca30a78f1ff97dd7481513d8f8f61afa66b624b405c48483bcd846457c6"


def _payload() -> dict[str, object]:
    return {
        "cell": "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1",
        "phase": "m1_heldin_heldout_gap_v1",
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_score_root_relative": "tfpd_exploration/results/m1_heldin_heldout_gap_v1",
        "producer_root_relative": (
            "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa"
            "/fold_20120924_cswg"
        ),
        "anchor_root_relative": (
            "tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_heldin_score_v1"
        ),
        "selected_checkpoint_role": "best_source_train_loss",
        "budget": "M10_native",
        "heldout_fold_sessions": ["20120924"],
        "heldin_training_sessions": ["20120926", "20120927", "20120928"],
        "score_order": ["20120924", "20120926", "20120927", "20120928"],
        "metric": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
        "last_bin_only": True,
        "target_metric_only": True,
        "anchor_reproduction_required_exactly": True,
        "preregistered_verdict": {
            "tolerance": 0.03,
            "no_headroom": "|gap| <= 0.03 and ci within [-0.03, 0.03]",
            "else": "HEADROOM_PRESENT with direction, or INDETERMINATE",
            "bootstrap": "session bootstrap, seed 42, 10,000 draws",
        },
        "formal_benchmark_verdict": False,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static paired-gap contract")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    arguments = parser.parse_args(argv)
    if arguments.execute:
        parser.error("M1 paired gap requires an in-process root-reviewed capability")
    print(json.dumps(_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
