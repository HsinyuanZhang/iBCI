#!/usr/bin/env python3
"""Dry-plan or root-reviewed publication of two runtime-control wrappers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["PYTHONNOUSERSITE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "cebra_exploration/src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import track_b_v2_post_synthetic_runtime_control_authority as authority  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mint", action="store_true")
    parser.add_argument("--i-have-root-review-authorization", action="store_true")
    args = parser.parse_args(argv)
    if args.mint != args.i_have_root_review_authorization:
        parser.error("publication requires both --mint and --i-have-root-review-authorization")
    try:
        plan = authority.build_dry_plan()
        if not args.mint:
            sys.stdout.write(json.dumps(plan, sort_keys=True, indent=2) + "\n")
            return 0
        launch_closure = plan["implementation_closure"]
        authority.require(launch_closure == authority.implementation_closure(),
                          "publisher implementation closure drift before publication")
        authority._assert_canonical_outputs_fresh()
        payloads = {"subject_m": plan["subject_m_payload"], "rt": plan["rt_payload"]}
        bindings = authority.publish_two_pairs_transactionally(payloads)
        sys.stdout.write(json.dumps(bindings, sort_keys=True, indent=2) + "\n")
        return 0
    except (authority.RuntimeControlAuthorityError, authority.route.TrackBV2ActualCpuError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

