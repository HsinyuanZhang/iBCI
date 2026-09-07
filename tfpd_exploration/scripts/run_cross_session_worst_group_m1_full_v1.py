#!/usr/bin/env python3
"""Static public plan for the V6-bound CS-WG M1 no-SWA full successor.

This intentionally standard-library-only entrypoint is descriptive.  It does
not import the physical route, Torch, a parser, or a result-root helper.  A
future reviewed in-process caller alone may validate the completed V6 graph,
issue an opaque capability, reserve the fresh full root, and execute.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "a71a7402d43f5012028dcd237d42861b647ec327e433f23df9c7a1bb10090179"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_full_v1_no_swa_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_full_root_relative": (
            "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa/"
            "fold_20120924_cswg"
        ),
        "accepted_v6_smoke_terminal": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6",
            "terminal_sha256": "dab4c9cfaad593c76d14b9d300642da31d7545248d4daf22bf2691c2b8cceb01",
            "identity_sha256": "cd8c387317e27865f2b1bbc2b6594aa7d258dd10c9eb472493df0029fdda8879",
            "closure_sha256": "4aa89da216fdf401e6c129a12af28cbb45e9f858d5f4db092fb24b1d77c6947c",
            "exact_leaf_count": 16,
        },
        "fixed_source_fold": {
            "system": "CS_WG",
            "outer_target_unopened": "20120924",
            "source_sessions": ["20120926", "20120927", "20120928"],
            "matched_erm_same_fold_spec_bound_not_executed": True,
        },
        "training": {
            "epochs": 20,
            "optimizer": "Adam",
            "adam_lr": 1.0e-5,
            "adam_weight_decay": 0.0,
            "scheduler": "None",
            "batch_size": 32,
            "one_concatenated_mixed_session_forward_per_step": True,
            "checkpoint_monitor": "source_train_loss_epoch_mean",
            "checkpoint_tie_break": "first_strict_minimum_epoch",
            "checkpoint_artifacts": ["best_source_train_loss", "last"],
            "swa_enabled": False,
            "swa_artifact_forbidden": True,
        },
        "swa_enabled": False,
        "swa_artifact_forbidden": True,
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static no-SWA full candidate")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("CS-WG full training requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
