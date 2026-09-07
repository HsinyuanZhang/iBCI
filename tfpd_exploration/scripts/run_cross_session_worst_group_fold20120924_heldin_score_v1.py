#!/usr/bin/env python3
"""Print the inert plan for CS-WG fold-20120924 held-in R² score V1.

This public CLI intentionally does not import the route package, Torch, the
M1 parser, or an artifact-root helper.  A future root-reviewed in-process
caller alone may validate the full producer, issue the opaque capability, and
run the metric-only target evaluation.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "1ff3faab95f58aa66730d4793b524be6b5370b54863300c700aa734251e44213"


def _payload() -> dict[str, object]:
    return {
        "cell": "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1",
        "phase": "fold20120924_heldin_metric_only_score_v1",
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_score_root_relative": (
            "tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_heldin_score_v1"
        ),
        "completed_full_terminal": "efd084573843e05ada6c28c050c28e8bbf01976bec443d864bc3a0a7921afdc5",
        "selected_checkpoint_role": "best_source_train_loss",
        "target_session": "20120924",
        "metric": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
        "last_bin_only": True,
        "target_metric_only": True,
        "matched_erm_reference_bound": False,
        "formal_benchmark_verdict": False,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static held-in score contract")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    arguments = parser.parse_args(argv)
    if arguments.execute:
        parser.error("CS-WG held-in score requires an in-process root-reviewed capability")
    print(json.dumps(_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
