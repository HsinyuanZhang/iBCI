#!/usr/bin/env python3
"""Static public plan for the CS-WG native-M1 source reader audit.

This stdlib-only command cannot import Torch, load the historical parser, open
an NWB, reserve an audit/smoke root, or manufacture the private root-reviewed
source capability.  A future authorized CPU source audit has its own immutable
audit root and never consumes the canonical GPU smoke root.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
REPAIR_WORKORDER_SHA256 = "e535af20ba42faf7fee2053d709aeb546a01d592e434f2086f7781e3e24fa129"
ACCEPTED_STAGE0_CLOSURE_SHA256 = "dd1fc152d4f7f900d6707bcea46136e1bde7cf781ca2f99dc12991296be7bb51"
ACCEPTED_SOURCE_LIFECYCLE_CLOSURE_SHA256 = "2f2078bcd89476b84c42d78abedd6b2430d6114bd6550e1ef5e938352e429bdf"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_reader_audit_v1",
        "repair_workorder_sha256": REPAIR_WORKORDER_SHA256,
        "accepted_stage0_closure_sha256": ACCEPTED_STAGE0_CLOSURE_SHA256,
        "accepted_source_lifecycle_closure_sha256": ACCEPTED_SOURCE_LIFECYCLE_CLOSURE_SHA256,
        "audit_scope": {
            "outer_target_session": "20120924",
            "source_sessions": ["20120926", "20120927", "20120928"],
            "root_relative": "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v1",
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps": 0,
            "separate_immutable_audit_root_required": True,
            "canonical_gpu_smoke_root_untouched": True,
            "lifecycle": "attempt_then_launch_then_source_authority_then_audit_terminal_or_failure",
        },
        "current_gpu_smoke_capability_issuable": False,
        "future_gpu_smoke_requires_successor_identity": True,
        "future_gpu_smoke_required_predecessor": "accepted immutable source-audit terminal plus source-authority graph",
        "source_descriptor_resolution": {
            "current_closure_bound_metadata_only_manifest_available": True,
            "metadata_manifest": {
                "relative_path": "sua_exploration/manifests/m1_b20_source_characterization_v1_manifest.json",
                "body_sha256": "4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6",
                "schema": "m1_b20_source_characterization_manifest_v1",
            },
            "route_owned_factory_requires": "route-owned DeferredHeldM1DescriptorLoader wrapped by sealed metadata-bound descriptor provider",
            "construction": "factory descriptor-reads only the sealed metadata manifest before capability; it retains a lexical source root and constructs no source descriptor until prepare_source",
            "timing": "manifest paths/SHA are descriptor-read before capability; source byte counts and held SHA verification occur only inside prepare_source after durable attempt",
            "omitted_outer_target_resolution_forbidden": True,
        },
        "execution_authorized": False,
        "opens_source_or_target": False,
        "loads_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "capability_requirement": "opaque in-process root-reviewed source-audit capability",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static source-reader audit plan")
    parser.add_argument("--audit", action="store_true", help="always rejected by the public CLI")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.audit or args.execute:
        parser.error("CS-WG source audit requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
