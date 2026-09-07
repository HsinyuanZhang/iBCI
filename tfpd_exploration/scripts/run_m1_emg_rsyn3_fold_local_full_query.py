#!/usr/bin/env python3
"""Public full-query CLI for fold-local EMG-rSyn3. Default is dry.

Live GPU execution requires --execute --gpu-authorized --gpu-index {0,1} --arm.
--pack allows same-route arms to share a card up to FULL_QUERY_PACK_LIMIT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _dry() -> dict[str, object]:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.plan import dry_cli_payload

    payload = dry_cli_payload()
    payload["cli"] = "run_m1_emg_rsyn3_fold_local_full_query.py"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=None, choices=(0, 1))
    parser.add_argument("--arm", choices=("Z-Fix", "S-Fix", "S-Acyc"), default=None)
    parser.add_argument("--pack", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.execute_gpu:
        parser.error("full-query public CLI cannot mint a GPU capability")
    if not arguments.execute:
        print(json.dumps(_dry(), sort_keys=True, separators=(",", ":")))
        return 0
    if not arguments.gpu_authorized or arguments.gpu_index is None or arguments.arm is None:
        parser.error("--execute requires --gpu-authorized --gpu-index --arm")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("full-query execute requires PYTHONNOUSERSITE=1")
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.full_query import execute_arm

    root = Path(__file__).resolve().parents[2]
    shas, terminal, failure = execute_arm(
        root,
        arm=arguments.arm,
        gpu_index=int(arguments.gpu_index),
        gpu_authorized=True,
        pack=bool(arguments.pack),
    )
    print(json.dumps(
        {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": shas},
        sort_keys=True, separators=(",", ":"),
    ))
    return 0 if terminal and failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
