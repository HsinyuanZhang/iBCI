#!/usr/bin/env python3
"""Build and privately push all-source B3 / B3S-rSyn3 M1 images.

Same challenge/phase/team path as the successful M2 cached-identity
submission 581644. Default is a local plan.

    PYTHONNOUSERSITE=1 python tfpd_exploration/scripts/run_m1_b3_allsource_remote.py --arm b3
    PYTHONNOUSERSITE=1 python tfpd_exploration/scripts/run_m1_b3_allsource_remote.py --arm b3 --build
    PYTHONNOUSERSITE=1 python tfpd_exploration/scripts/run_m1_b3_allsource_remote.py --arm b3 --preflight
    PYTHONNOUSERSITE=1 python tfpd_exploration/scripts/run_m1_b3_allsource_remote.py --arm b3 --execute \\
        --confirm-image-id sha256:... --confirm-payload-sha256 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("b3", "b3s_rsyn3", "b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4"), required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if arguments.execute_gpu:
        parser.error("cannot mint a GPU capability")
    from tfpd_exploration.src.m1_b3_allsource_v1.submit_remote import (
        build_image,
        execute_push,
        plan_report,
        preflight,
    )

    if arguments.build:
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise SystemExit("build requires PYTHONNOUSERSITE=1")
        receipt = build_image(root, arguments.arm)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if arguments.preflight or arguments.execute:
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise SystemExit("remote contact requires PYTHONNOUSERSITE=1")
        candidate, _, image, phase, limits, _runtime = preflight(arguments.arm, root)
        report = {
            "mode": "execute" if arguments.execute else "read_only_preflight",
            "arm": candidate["arm"],
            "image_tag": candidate["image_tag"],
            "image_id": image.id,
            "image_size": image.attrs.get("Size"),
            "phase_id": phase["id"],
            "phase_slug": phase["slug"],
            "phase_active": phase["is_active"],
            "team_id": 41975,
            "private": True,
            "quota": limits,
            "payload_sha256": candidate["payload_sha256"],
            "method_name": candidate["method_name"],
        }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not arguments.execute:
            return 0
        completed = execute_push(
            root,
            arguments.arm,
            confirm_image_id=arguments.confirm_image_id,
            confirm_payload_sha256=arguments.confirm_payload_sha256,
        )
        print(json.dumps(
            {
                "arm": completed["arm"],
                "submission_id": completed["submission_id"],
                "image_id": completed["image_id"],
                "payload_sha256": completed["payload_sha256"],
                "uploaded_manifest_digest": completed["uploaded_manifest_digest"],
                "uuid_tag": completed["uuid_tag"],
            },
            indent=2,
            sort_keys=True,
        ))
        return 0
    print(json.dumps(plan_report(arguments.arm, root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
