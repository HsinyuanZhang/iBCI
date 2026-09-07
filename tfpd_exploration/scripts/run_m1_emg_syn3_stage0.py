#!/usr/bin/env python3
"""Public Stage-0 CLI. Default is dry. --execute runs CPU Stage 0 only."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _dry() -> dict[str, object]:
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1.plan import dry_cli_payload
    payload = dry_cli_payload()
    payload["cli"] = "run_m1_emg_syn3_stage0.py"
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.execute_gpu:
        parser.error("Stage-0 CLI cannot mint a GPU capability")
    if arguments.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from tfpd_exploration.src.m1_emg_syn3_fcm_v1.stage0 import publish_stage0
        root = Path(__file__).resolve().parents[2]
        shas, terminal, failure = publish_stage0(root)
        print(json.dumps(
            {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": shas},
            sort_keys=True, separators=(",", ":"),
        ))
        return 0 if terminal and failure is None else 2
    print(json.dumps(_dry(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
