#!/usr/bin/env python3
"""Public CPU-only token-content probe CLI for fold-local EMG-rSyn3. Default is dry.

Live execution requires --execute with PYTHONNOUSERSITE=1. This cell mints no GPU capability.
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
    payload["cli"] = "run_m1_emg_rsyn3_fold_local_token_probe.py"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.execute_gpu:
        parser.error("token probe is CPU-only and mints no GPU capability")
    if not arguments.execute:
        print(json.dumps(_dry(), sort_keys=True, separators=(",", ":")))
        return 0
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("token-probe execute requires PYTHONNOUSERSITE=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.token_probe import execute

    root = Path(__file__).resolve().parents[2]
    shas, terminal, failure = execute(root)
    print(json.dumps(
        {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": shas},
        sort_keys=True, separators=(",", ":"),
    ))
    return 0 if terminal and failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
