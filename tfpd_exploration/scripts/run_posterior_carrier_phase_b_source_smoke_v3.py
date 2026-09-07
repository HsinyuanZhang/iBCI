#!/usr/bin/env python3
"""Static dry plan for the additive Posterior Carrier Phase-B smoke-v3 route.

This public command intentionally imports neither Torch nor the physical
backend.  Execution requires a root-issued in-process capability and is not
available through command-line flags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _plan() -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from posterior_carrier_v1 import plan

    return {
        "phase": "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V3",
        "cell": plan.CELL,
        "handoff": {"relative_path": plan.HANDOFF_RELATIVE, "sha256": plan.HANDOFF_SHA256},
        "predecessors": {
            "v1": {
                "root_relative": "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v1",
                "attempt_body_sha256": "4d6283534017938de44450a9cd1160496e1efaa1f91728a2c0d82fb2276454c3",
                "launch_body_sha256": "bce0ac19b3517dfeb4298faedcd322efd6f10248768ce93fef782388f21c2229",
                "failure_body_sha256": "a7dfaf466642ca92a945dc545ad0ec0228a0eac180dc3e24ec4d93471977738f",
            },
            "v2": {
                "root_relative": "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v2",
                "attempt_body_sha256": "e93e1a59724d06abcc8831c3b0d4f49f81f7eecb451c6e429e96f1f70d6cea24",
                "launch_body_sha256": "fadd0c94a35a416f0c4a6d527be1ee9a85368d845d64ab64a20d2028c8ba3416",
                "failure_body_sha256": "d8c5a4eac84d14187163991209b486433a9851496f7a336b9e5a3dcd7243a2af",
                "accepted_closure_sha256": "9a692c704f2c31c470e9ee465307c9bb1ea4ec2b73b0d156218e06b387abb853",
            },
        },
        "successor_root_relative": "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_source_smoke_v3",
        "precision_policy": {
            "amp": False,
            "cuda_matmul_allow_tf32": "explicitly set False before model/optimizer",
            "cudnn_allow_tf32": "explicitly set False before model/optimizer",
            "observed_pre_state": "receipt disclosure only; not an acceptance gate",
            "post_state": "both TF32 switches and AMP must be False",
            "route_local_restore": True,
        },
        "boundaries": {
            "no_torch_import": "this dry CLI only",
            "no_data_cache_checkpoint_cuda_gpu_remote_write_or_launch": True,
            "target_within_external_formal_h1_authorized": False,
        },
        "status": "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="render static v3 plan")
    parser.add_argument("--execute-remote-smoke-v3", action="store_true", help="always fails without root capability")
    args = parser.parse_args(argv)
    if args.execute_remote_smoke_v3:
        parser.error("remote source smoke-v3 requires an in-process root-reviewed capability; public CLI cannot execute it")
    print(json.dumps(_plan(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
