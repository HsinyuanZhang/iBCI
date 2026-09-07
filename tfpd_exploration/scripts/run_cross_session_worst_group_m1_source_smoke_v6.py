#!/usr/bin/env python3
"""Static public plan for the CS-WG M1 V6 audit-spec smoke successor.

This entrypoint is intentionally standard-library-only.  It cannot import a
route, Torch, or a parser; inspect any predecessor/result root; open source or
target data; reserve a root; mint a capability; or execute a smoke.  Only a
root-reviewed in-process caller may use the V6 lifecycle.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "3baf485008655aa5dc9e6a24d2a8e1805b791e1c78e377b1eeb45b3eceea1a3c"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_smoke_v6_audit_spec_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_v6_root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6",
        "accepted_v3_completed_graph": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3",
            "attempt_sha256": "f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287",
            "launch_sha256": "04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c",
            "source_authority_sha256": "0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b",
            "audit_sha256": "ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9",
            "terminal_sha256": "2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230",
            "exact_leaf_count": 10,
        },
        "failed_v5_graph": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v5",
            "attempt_sha256": "ae6bf550dfff4179d51dbb62616a7fa97d2bfb33d1e7de5ab8c4917096afb9c4",
            "launch_sha256": "03a33580948f9c95dd8f022ea1947ed9f696f551974fe546c5edb03c94d3756f",
            "failure_sha256": "0cbd5c0551fa654bf7d9b2b45dfe334d7023bcfb142087fefd297ed85199ce11",
            "error_class": "SourceAuditV3Error",
            "error_sha256": "840a7a668324571626ba202fd63a3f9590252701be5b91707f2a0b11d2d3d9db",
            "exact_leaf_count": 6,
        },
        "one_repair": {
            "v3_builder_receives": "accepted_v3_audit_spec",
            "v1_runner_receives": "v4_rebound_smoke_spec",
            "direct_smoke_spec_to_v3_builder_forbidden": True,
        },
        "source_only_smoke": {
            "optimizer_steps": 100,
            "batch_size": 32,
            "one_concatenated_mixed_session_forward_per_step": True,
            "v5_derivative_numeric_gate_retained": True,
            "amp": False,
            "tf32": False,
            "compile": False,
        },
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static V6 plan")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("CS-WG V6 source smoke requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
