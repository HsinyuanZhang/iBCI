#!/usr/bin/env python3
"""Print the inert plan for the M1 matched T0/C1 calibration-prefix pair V1.

This public CLI intentionally does not import the route package, Torch, the
M1 parser, or an artifact-root helper.  Only an in-process root-reviewed
caller may execute the smoke, the two 20-epoch arms (serially, GPU 1), and
the Phase-3 2x2x2 table.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "604f4ca30a78f1ff97dd7481513d8f8f61afa66b624b405c48483bcd846457c6"


def _payload() -> dict[str, object]:
    return {
        "cell": "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1",
        "phase": "m1_t0c1_prefix_v1",
        "workorder_sha256": WORKORDER_SHA256,
        "prospective_roots": {
            "smoke": "tfpd_exploration/results/m1_t0c1_prefix_v1/smoke",
            "t0": "tfpd_exploration/results/m1_t0c1_prefix_v1/t0",
            "c1": "tfpd_exploration/results/m1_t0c1_prefix_v1/c1",
            "phase3": "tfpd_exploration/results/m1_t0c1_prefix_v1/phase3_table",
        },
        "arms": {"t0": "operator disabled, same runner", "c1": "prefix cycle (10, 5, 2)"},
        "budget": {"seed": 42, "epochs": 20, "steps_per_epoch": 4951, "batch": 32,
                   "adam_lr": 1e-5, "swa": False},
        "dropout_proof_bound_before_launch": True,
        "dropout_block_sha256_both_trees": "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884",
        "hard_timeout_seconds_per_arm": 43200,
        "order": ["smoke", "t0", "c1", "phase3"],
        "gpu": "CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only)",
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static pair contract")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    arguments = parser.parse_args(argv)
    if arguments.execute:
        parser.error("m1 t0c1 pair requires an in-process root-reviewed capability")
    print(json.dumps(_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
