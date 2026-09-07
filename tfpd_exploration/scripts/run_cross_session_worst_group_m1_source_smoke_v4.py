#!/usr/bin/env python3
"""Static public plan for the CS-WG M1 V4 source-only smoke successor.

This entrypoint deliberately uses only the standard library.  It cannot import
the V4 route, source parser, NumPy, or Torch; inspect a predecessor/result
root; resolve source data; initialize CUDA; reserve a root; mint a capability;
or execute a smoke.  A reviewed in-process caller must use the private V4
lifecycle API after separate root authorization.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "c3ae0d1f5ea1f033419891f7a5217dc3cdc0011284b545d2ae5d150886b0fbe6"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_smoke_v4_common_stratum_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_v4_root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v4",
        "accepted_v3_completed_graph": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3",
            "attempt_sha256": "f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287",
            "launch_sha256": "04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c",
            "source_authority_sha256": "0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b",
            "audit_sha256": "ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9",
            "terminal_sha256": "2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230",
            "exact_leaf_count": 10,
            "held_no_follow_validation_before_capability_reservation_and_terminal": True,
        },
        "source_only_smoke": {
            "optimizer_steps": 100,
            "batch_size": 32,
            "one_concatenated_mixed_session_forward_per_step": True,
            "m1_window": 100,
            "m1_units": 64,
            "raw_outputs": 16,
            "calibration_shape_per_row": [10, 1024, 64],
            "live_parameter_count": 15007496,
            "optimizer": "Adam",
            "lr": 1e-05,
            "weight_decay": 0.0,
            "lambda": 1.0,
            "tau": 0.01,
            "tf32": False,
            "amp": False,
            "compile": False,
        },
        "route_local_digest_cache": {
            "preserves_exact_core_input_digests": True,
            "prevalidated_session_calibration_sha_reused_per_row": True,
            "b32_per_row_calibration_stack_preserved": True,
        },
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static V4 plan")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("CS-WG V4 source smoke requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
