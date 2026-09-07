#!/usr/bin/env python3
"""Static dry plan for Posterior Carrier's reviewed 48-epoch source route.

This public command intentionally imports only the route plan.  It cannot
construct a root-reviewed in-process capability, inspect smoke-v3 receipts,
open source data, initialize CUDA, reserve an output root, or launch remote
training.  The future reviewed entry point lives in ``full_train.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _load_plan() -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from posterior_carrier_v1 import plan

    return {
        "cell": plan.CELL,
        "phase": "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_FULL_TRAIN_V1",
        "status": "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW_AND_COMPLETED_SMOKE_V3",
        "handoff": {"relative_path": plan.HANDOFF_RELATIVE, "sha256": plan.HANDOFF_SHA256},
        "accepted_smoke_v3_closure_sha256": (
            "4ac9ecd38a5040f361bb9c76920db7e26426f172f27d33a712333ad5c27e4d1a"
        ),
        "required_predecessor": {
            "root_relative": (
                "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v3"
            ),
            "requires_immutable_attempt_launch_source_authority_step100_terminal_pairs": True,
            "required_step100_status": "SOURCE_SMOKE_V3_100_STEPS_COMPLETE",
            "required_terminal_status": "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE",
        },
        "cross_stage_identity_contract": {
            "completed_smoke_identity": "reconstructed_from_immutable_v3_receipt_bytes",
            "fresh_full_stage_identity": "rebuilt_from_current_stage_authority_assets",
            "only_allowed_difference": "strict_source_metadata_sha256_descriptor_binding",
            "required_difference_reason": "fresh_stage_descriptor_identities",
            "stable_exact_fields": [
                "strict27_roster", "source_data_root", "authority_asset_body_hashes",
                "normalizer_and_raw_T4_authorities", "posterior_theta_semantics",
                "accepted_v3_closure", "remote_torch_contract", "target_free_boundaries",
            ],
        },
        "fixed_training_contract": {
            "epochs": 48,
            "steps_per_epoch": 33925,
            "total_optimizer_steps": 1628400,
            "batch_size": 32,
            "seed": 42,
            "checkpoint_epochs": [44, 45, 46, 47],
            "swa": "arithmetic final-four checkpoint state mean with strict fresh Cell-D reload",
            "posterior_cache": "one M(e,j) session-static carrier view outside batch loop",
        },
        "boundaries": {
            "source_only": True,
            "target_within_external_formal_h1": "forbidden",
            "teacher_or_pretraining": "forbidden",
            "public_cli_imports_torch": False,
            "public_cli_reads_data_or_receipts": False,
            "public_cli_writes_or_launches": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="render the static no-execution route plan")
    parser.add_argument("--execute-remote-full", action="store_true", help="always fail closed in public CLI")
    parser.add_argument("--root-reviewed-capability", action="store_true", help="cannot be materialized by public CLI")
    args = parser.parse_args(argv)
    if args.execute_remote_full or args.root_reviewed_capability:
        parser.error(
            "posterior full training requires an in-process root-reviewed capability and a completed immutable smoke-v3 terminal; public CLI cannot execute"
        )
    print(json.dumps(_load_plan(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
