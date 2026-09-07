#!/usr/bin/env python3
"""CPU-only Stage0 runner for cross-dataset functional calibration v1.

Default is dry. Live execution requires --execute with PYTHONNOUSERSITE=1.
This cell mints no GPU capability and rejects --execute-gpu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _dry() -> dict[str, object]:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1.plan import dry_cli_payload

    payload = dry_cli_payload()
    payload["cli"] = "run_cross_dataset_functional_calibration_v1.py"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bind-p-operator", action="store_true")
    parser.add_argument("--execute-preflight", action="store_true")
    parser.add_argument("--disposable-profile", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.execute_gpu:
        parser.error("this workstream is CPU-only and mints no GPU capability")
    exclusive = [
        arguments.execute,
        arguments.bind_p_operator,
        arguments.execute_preflight,
        arguments.disposable_profile,
    ]
    if sum(bool(flag) for flag in exclusive) > 1:
        parser.error("choose one of --execute / --bind-p-operator / --execute-preflight / --disposable-profile")
    if not any(exclusive):
        print(json.dumps(_dry(), sort_keys=True, separators=(",", ":")))
        return 0
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("execute requires PYTHONNOUSERSITE=1")
    root = Path(__file__).resolve().parents[2]
    if arguments.disposable_profile:
        if os.environ.get("CDF_DISPOSABLE_PROFILE") != "1":
            raise SystemExit("disposable profile requires CDF_DISPOSABLE_PROFILE=1")
        from tfpd_exploration.src.cross_dataset_functional_calibration_v1.model_adapter import (
            launch_disposable_profile,
        )

        dry = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip() == ""
        result = launch_disposable_profile(root, dry_run=dry)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if arguments.bind_p_operator:
        from tfpd_exploration.src.cross_dataset_functional_calibration_v1.execute_p_revision import (
            execute as bind_p,
        )

        hashes, terminal = bind_p(root)
        print(
            json.dumps(
                {"terminal_sha256": terminal, "bodies": hashes, "gpu_work_started": False},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if arguments.execute_preflight:
        from tfpd_exploration.src.cross_dataset_functional_calibration_v1.execute_preflight import (
            execute as run_preflight,
        )

        hashes, terminal = run_preflight(root)
        print(
            json.dumps(
                {"terminal_sha256": terminal, "bodies": hashes, "gpu_work_started": False},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1.execute import execute

    hashes, terminal, failure = execute(root)
    print(
        json.dumps(
            {"terminal_sha256": terminal, "failure_sha256": failure, "bodies": hashes},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if terminal and failure is None else 2


if __name__ == "__main__":
    sys.exit(main())
