#!/usr/bin/env python3
"""Static dry plan for Posterior Carrier Phase-B source smoke.

This file deliberately imports neither Torch nor the Phase-B backend.  Its
public flags can describe the reviewed route but cannot manufacture the
in-process capability required to access source data, CUDA, remote staging, or
an immutable output root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _load_plan() -> object:
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from posterior_carrier_v1 import plan

    return {
        "phase": "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V1",
        "cell": plan.CELL,
        "handoff": {"relative_path": plan.HANDOFF_RELATIVE, "sha256": plan.HANDOFF_SHA256},
        "phase_b_amendment": (
            "posterior-specific strict-source float64 normalizer over deterministic "
            "M4/M10/M30 posterior means; point-T4 moments forbidden"
        ),
        "source_smoke": {"batch_size": 32, "optimizer_steps": 100, "full_training_authorized": False},
        "remote": {"host": "xinyuan@100.103.97.12", "execution": "requires in-process root-reviewed capability"},
        "boundaries": {
            "no_torch_import": "this dry CLI only",
            "no_data_cache_checkpoint_cuda_gpu_remote_write_or_launch": True,
            "target_within_external_formal_h1_authorized": False,
        },
        "status": "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="render the static no-execution plan")
    parser.add_argument("--execute-remote-smoke", action="store_true", help="intentionally fails without root capability")
    args = parser.parse_args(argv)
    if args.execute_remote_smoke:
        parser.error("remote source smoke requires an in-process root-reviewed capability; public CLI cannot execute it")
    print(json.dumps(_load_plan(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
