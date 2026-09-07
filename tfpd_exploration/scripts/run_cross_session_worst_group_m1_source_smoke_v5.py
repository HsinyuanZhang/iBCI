#!/usr/bin/env python3
"""Static public plan for the CS-WG M1 V5 source-only smoke successor.

The entrypoint is standard-library-only.  It cannot import the route, Torch,
or a native parser; open source/target data; inspect a result root; create a
receipt; issue a capability; or execute a smoke.  A root-reviewed in-process
caller must use the private V5 lifecycle after separately validating both
immutable predecessor graphs.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "65dec719e8931270ae1d9b9bdc11c550a8f68198b6457e327ac90431ae1a85d4"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_smoke_v5_granular_derivative_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_v5_root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v5",
        "accepted_v3_completed_graph": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3",
            "attempt_sha256": "f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287",
            "launch_sha256": "04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c",
            "source_authority_sha256": "0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b",
            "audit_sha256": "ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9",
            "terminal_sha256": "2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230",
            "exact_leaf_count": 10,
        },
        "failed_v4_graph": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v4",
            "attempt_sha256": "0d4179d77cd8aa66bdad70d1164682dba38b3d960fab3d524cf5f213e4601b27",
            "launch_sha256": "422dec3d014bf3fa0fed716b8634cc6e8da6076f36f0f303f657a388f7b417ad",
            "source_authority_sha256": "40336ed6ea1f6dd2b6511eee4d22f5cf76c4af29510f8c258819bd396c78b4fe",
            "failure_sha256": "1a8237db7ddf34535bebd83cc773da7efde6b5aa97517a09638b391d6eae91d3",
            "error_sha256": "bfeff083be483479e40fb82078a3b8b1e5755b0f04ec0f341cc840af892cf813",
            "exact_leaf_count": 8,
        },
        "source_only_smoke": {
            "optimizer_steps": 100,
            "batch_size": 32,
            "one_concatenated_mixed_session_forward_per_step": True,
            "derivative_gate": {
                "loss_domain": [0.0, 10.0],
                "tau": 0.01,
                "reference": "float64_stable_softmax(session_mse/tau)",
                "sum_abs_error_max": 2e-06,
                "reference_abs_error_max": 2e-05,
                "raw_minimum": -2e-05,
            },
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
    parser.add_argument("--dry-run", action="store_true", help="print the static V5 plan")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("CS-WG V5 source smoke requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
