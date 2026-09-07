#!/usr/bin/env python3
"""Static dry plan for the fresh Posterior Carrier Phase-B smoke-v2 route.

The real route needs a root-issued in-process capability and therefore cannot
be invoked by this public CLI.  This file intentionally imports no Torch,
source adapter, data reader, CUDA, remote client, or result writer.
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
        "phase": "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V2",
        "cell": plan.CELL,
        "handoff": {"relative_path": plan.HANDOFF_RELATIVE, "sha256": plan.HANDOFF_SHA256},
        "predecessor": {
            "root_relative": "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v1",
            "attempt_body_sha256": "4d6283534017938de44450a9cd1160496e1efaa1f91728a2c0d82fb2276454c3",
            "launch_body_sha256": "bce0ac19b3517dfeb4298faedcd322efd6f10248768ce93fef782388f21c2229",
            "failure_body_sha256": "a7dfaf466642ca92a945dc545ad0ec0228a0eac180dc3e24ec4d93471977738f",
            "required_status": "SOURCE_SMOKE_FAILED_HONESTLY",
        },
        "successor_root_relative": "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v2",
        "sole_v2_data_semantics": {
            "missing_target_dir": "same selected prefix trial target_corners centre only",
            "canonical_snap": "unique 8-way snap within pi/64",
            "expected_fallback": "sub-C_ses-CO-20150313 original trial_index=33",
            "later_row_substitution": False,
        },
        "boundaries": {
            "no_torch_import": "this dry CLI only",
            "no_data_cache_checkpoint_cuda_gpu_remote_write_or_launch": True,
            "target_within_external_formal_h1_authorized": False,
            "v1_result_mutated": False,
        },
        "status": "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="render the static no-execution v2 plan")
    parser.add_argument("--execute-remote-smoke-v2", action="store_true", help="always fails without root capability")
    args = parser.parse_args(argv)
    if args.execute_remote_smoke_v2:
        parser.error("remote source smoke-v2 requires an in-process root-reviewed capability; public CLI cannot execute it")
    print(json.dumps(_load_plan(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
