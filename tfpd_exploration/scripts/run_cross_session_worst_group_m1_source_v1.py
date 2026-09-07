#!/usr/bin/env python3
"""Static public entry point for CS-WG M1 Source Lifecycle V1.

This command intentionally imports only the standard library.  It cannot read
an M1 source file, construct the M1 graph, touch a checkpoint or CUDA, reserve
a result root, or manufacture the private root-reviewed execution capability.
"""
from __future__ import annotations

import argparse
import json


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_SHA256 = "ca0c3b754c602ec490e4f5b74c5bf85a93764184e6f9a489a10ec4eed892e043"
ACCEPTED_STAGE0_CLOSURE_SHA256 = "dd1fc152d4f7f900d6707bcea46136e1bde7cf781ca2f99dc12991296be7bb51"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "m1_source_lifecycle_v1",
        "workorder_sha256": WORKORDER_SHA256,
        "accepted_stage0_closure_sha256": ACCEPTED_STAGE0_CLOSURE_SHA256,
        "source_smoke": {
            "outer_target_session": "20120924",
            "source_sessions": ["20120926", "20120927", "20120928"],
            "optimizer_steps": 100,
            "total_batch_size": 32,
            "one_concatenated_forward_per_step": True,
        },
        "all_outer_targets_supported_by_route_owned_manifest": [
            "20120924", "20120926", "20120927", "20120928",
        ],
        "fixed_recipe": {
            "window": 100,
            "units": 64,
            "calibration_shape_per_row": [10, 1024, 64],
            "raw_outputs": 16,
            "epochs": 20,
            "optimizer": "Adam",
            "lr": 1.0e-5,
            "weight_decay": 0.0,
            "scheduler": None,
            "lambda": 1.0,
            "tau": 0.01,
        },
        "execution_authorized": False,
        "opens_source_or_target": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "smokes": False,
        "trains": False,
        "scores": False,
        "launches": False,
        "capability_requirement": "opaque in-process root-reviewed source execution capability",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static source-lifecycle plan")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    parser.add_argument("--smoke", action="store_true", help="always rejected by the public CLI")
    parser.add_argument("--train", action="store_true", help="always rejected by the public CLI")
    args = parser.parse_args(argv)
    if args.execute or args.smoke or args.train:
        parser.error("CS-WG source execution requires an in-process root-reviewed capability")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
