#!/usr/bin/env python3
"""Static public plan for the CS-WG M1 source-audit V2 successor.

This file uses only the standard library.  It deliberately cannot import the
route, inspect the immutable V1 root, open source data, reserve a V2 root,
construct a capability, import Torch, or initialize CUDA.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "dc439c5a5692e20e5e6cb3f37ee5fcb7137a35b0cde5f5aaedc90379c3e49607"
V1_REPAIRED_CLOSURE_SHA256 = "b5d4fdf5505fd53f160d56160daa642e36e8077171f654d9c5cdff6ee8fb7a28"
M1_METADATA_MANIFEST_SHA256 = "4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_audit_v2_namespace_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v2",
        "inherited_v1_repaired_closure_sha256": V1_REPAIRED_CLOSURE_SHA256,
        "sealed_metadata_manifest_sha256": M1_METADATA_MANIFEST_SHA256,
        "immutable_v1_failed_predecessor": {
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v1",
            "attempt_sha256": "5d0cd206644d29b4ac81139d9a7cbe4aaedc404935a1029597ef1a0f3f9a1e75",
            "launch_sha256": "895fb18d342a51a91e97e5a89ef13e53d662bd556fc2b0881ca1e7be5a70a99d",
            "failure_sha256": "f5ec355bd26d4bb13f48117e122a2f2b4c5eb3dac2e18cc9b357dfa10554436b",
            "source_authority_audit_terminal_absent": True,
            "held_no_follow_validation_required_before_capability_and_reservation": True,
        },
        "reviewed_namespace": {
            "route_import": "tfpd_exploration.src.cross_session_worst_group_v1",
            "top_level_project_src_forbidden": True,
            "top_level_streaming_src_reserved_for_deferred_parser": True,
            "sys_modules_replacement": False,
        },
        "current_gpu_smoke_capability_issuable": False,
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "launches": False,
        "execution_authorized": False,
        "capability_requirement": "opaque in-process root-reviewed V2 source-audit capability",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static V2 audit plan")
    parser.add_argument("--audit", action="store_true", help="always rejected by the public CLI")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.audit or args.execute:
        parser.error("CS-WG V2 source audit requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
