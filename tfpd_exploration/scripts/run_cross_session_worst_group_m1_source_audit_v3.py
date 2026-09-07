#!/usr/bin/env python3
"""Static public plan for CS-WG M1 source-audit V3.

This entrypoint uses only the standard library.  It cannot import the route,
inspect V1/V2 result roots, resolve source descriptors, import Torch,
initialize CUDA, mint a V3 capability, reserve an artifact root, or execute a
source audit.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "ebc536ff0a2e1c64e0a6e33b8b6ca0f744f79b08db2b7c56d45275481ea94d14"
V2_CLOSURE_SHA256 = "ca77c59103830fcb38e6d51af49b355485b6b916a3b4402e26d4b20e21588fa0"
V2_IDENTITY_SHA256 = "e9565bce5d0f2e4e683b599f5e0e0e65a089eae4d0308afd7aa2d58170a6b581"
M1_METADATA_MANIFEST_SHA256 = "4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_audit_v3_common_stratum_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3",
        "inherited_v2_closure_sha256": V2_CLOSURE_SHA256,
        "inherited_v2_identity_sha256": V2_IDENTITY_SHA256,
        "sealed_metadata_manifest_sha256": M1_METADATA_MANIFEST_SHA256,
        "immutable_v2_failed_predecessor": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v2",
            "attempt_sha256": "163e73fc5c793b009e7ec86f4af7cc1841bec0bbf5865629c7013a3df468350b",
            "launch_sha256": "9af2bb8c400dad927b0d3dbf711d9d24242c0e4bfb55b8163a96e0a5e4ea4fcc",
            "failure_sha256": "945a7b5f843e14a6cb4008309c9df4ab510b5f60dbda4a33ba54e6dc563889cb",
            "held_no_follow_validation_required_before_capability_and_reservation": True,
        },
        "fallback": {
            "mode": "DETERMINISTIC_COMMON_STRATUM_MIN2_PRUNE_V1",
            "minimum_per_session_count": 2,
            "minimum_eligible_strata": 10,
            "pooled_source_quantile_fit_unchanged": True,
            "task_stratum_definition_unchanged": True,
            "quota_11_11_10_unchanged": True,
            "target_labels_used": False,
        },
        "current_gpu_smoke_capability_issuable": False,
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "launches": False,
        "execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static V3 plan")
    parser.add_argument("--audit", action="store_true", help="always rejected by the public CLI")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.audit or args.execute:
        parser.error("CS-WG V3 source audit requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
