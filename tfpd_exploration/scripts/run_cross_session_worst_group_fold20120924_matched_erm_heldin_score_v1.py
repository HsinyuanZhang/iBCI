#!/usr/bin/env python3
"""Print the inert future MATCHED_ERM held-in score contract.

The public CLI deliberately cannot supply future producer digests, issue an
opaque capability, inspect a result/data root, import Torch, or execute a
score.  A later root-reviewed in-process caller must provide all terminal
binding literals to the typed constructor.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "4b9da84fb4f5a9cf08d0bf1cb019f00ef45fbdb28f233fdcf2c02d25d9f70e2e"


def _payload() -> dict[str, object]:
    return {
        "cell": "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1",
        "phase": "fold20120924_matched_erm_heldin_metric_only_score_v1",
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_score_root_relative": (
            "tfpd_exploration/results/cross_session_worst_group_m1_fold20120924_matched_erm_heldin_score_v1"
        ),
        "future_matched_erm_full_root_relative": (
            "tfpd_exploration/results/cross_session_worst_group_m1_matched_erm_full_v1_no_swa/"
            "fold_20120924_matched_erm"
        ),
        "target_session": "20120924",
        "n_windows_required": 54849,
        "selected_checkpoint_role": "best_source_train_loss",
        "metric": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
        "last_bin_only": True,
        "target_metric_only": True,
        "producer_binding_constructor_required": True,
        "opens_future_result_or_nwb": False,
        "opens_checkpoint_tensor": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static future score contract")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    arguments = parser.parse_args(argv)
    if arguments.execute:
        parser.error("CS-WG matched-ERM held-in score requires a future in-process root-reviewed capability")
    print(json.dumps(_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
