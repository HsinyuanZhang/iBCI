#!/usr/bin/env python3
"""Static public plan for the fold-20120924 matched-ERM no-SWA full successor.

This stdlib-only command is deliberately inert.  It cannot issue a capability,
reserve a root, read source data, import Torch, or execute the training route.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "1b2c4e15c3720a3aa72546487218b1b8bc1d403c49f14a8ab47f363644f8cf3c"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_matched_erm_full_v1_no_swa_same_fold_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_full_root_relative": (
            "tfpd_exploration/results/cross_session_worst_group_m1_matched_erm_full_v1_no_swa/"
            "fold_20120924_matched_erm"
        ),
        "accepted_v6_smoke_terminal": {
            "terminal_sha256": "dab4c9cfaad593c76d14b9d300642da31d7545248d4daf22bf2691c2b8cceb01",
            "identity_sha256": "cd8c387317e27865f2b1bbc2b6594aa7d258dd10c9eb472493df0029fdda8879",
            "closure_sha256": "4aa89da216fdf401e6c129a12af28cbb45e9f858d5f4db092fb24b1d77c6947c",
            "exact_leaf_count": 16,
        },
        "fixed_same_fold": {
            "system": "MATCHED_ERM", "lambda": 0.0, "tau": 0.01,
            "outer_target_unopened": "20120924",
            "source_sessions": ["20120926", "20120927", "20120928"],
            "shared_graph_seed_sampler_optimizer_recipe": True,
        },
        "training": {
            "epochs": 20, "optimizer": "Adam", "adam_lr": 1.0e-5,
            "adam_weight_decay": 0.0, "scheduler": "None", "batch_size": 32,
            "one_concatenated_mixed_session_forward_per_step": True,
            "checkpoint_monitor": "source_train_loss_epoch_mean",
            "checkpoint_tie_break": "first_strict_minimum_epoch",
            "checkpoint_artifacts": ["best_source_train_loss", "last"],
            "swa_enabled": False, "swa_artifact_forbidden": True,
        },
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static same-fold ERM candidate")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("matched-ERM full training requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
