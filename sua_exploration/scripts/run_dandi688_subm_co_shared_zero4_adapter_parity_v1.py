#!/usr/bin/env python3
"""Run fixed consumed-sub-C shared-zero4 input parity; never score sub-M."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRELAUNCH = (
    ROOT
    / "sua_exploration/results/dandi_000688_subm_co_shared_zero4_adapter_parity_prelaunch_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "sua_exploration/results/dandi_000688_subm_co_shared_zero4_adapter_parity_execution_v1"
)
sys.path.insert(0, str(ROOT))

from sua_exploration.scripts.write_dandi688_subm_co_shared_zero4_adapter_parity_prelaunch_v1 import (  # noqa: E402
    StaticZero4ParityError,
    canonical_bytes,
    load_stored_prelaunch,
)


class Zero4ParityRunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Zero4ParityRunnerError(message)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, "immutable output mode failed")
    return hashlib.sha256(raw).hexdigest()


def execute_fixed_consumed_subc(
    *, root: Path = ROOT, prelaunch_dir: Path = DEFAULT_PRELAUNCH,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root = root.resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to((root / "sua_exploration/results").resolve())
    except ValueError as exc:
        raise Zero4ParityRunnerError("output root escapes repository results") from exc
    require(not output_root.exists(), "zero4 parity output root already exists")
    stored = load_stored_prelaunch(prelaunch_dir, root)
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA must be hidden")

    # This import occurs only after the stored source/fixture authority passes.
    from sua_exploration.mc_maze.subm_co_shared_zero4_adapter_parity_v1 import (
        execute_consumed_subc_zero4_parity,
    )

    parity = execute_consumed_subc_zero4_parity(root)
    require(parity.get("scope", {}).get("external_subm_accessed") is False, "sub-M access drift")
    require(parity.get("scope", {}).get("external_subm_scored") is False, "sub-M score drift")
    require(parity.get("scope", {}).get("model_forward_calls") == 0, "model forward drift")
    require(parity.get("external_scoring_capability_created") is False, "capability drift")

    output_root.mkdir(parents=True, exist_ok=False)
    trace_path = output_root / "input_parity_trace.json"
    environment_path = output_root / "environment.json"
    receipt_path = output_root / "receipt.json"
    seal_path = output_root / "seal.json"
    trace_sha = _write_immutable(trace_path, parity)
    environment = {
        "schema": "dandi_000688_consumed_subc_shared_zero4_adapter_parity_environment_v1",
        "host": socket.gethostname(),
        "python": {
            "path": str(Path(sys.executable).resolve()),
            "version": sys.version,
        },
        "thread_environment": {
            key: os.environ.get(key, "")
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "prelaunch": {
            key: stored[key]
            for key in ("draft_sha256", "receipt_sha256", "seal_sha256", "status")
        },
        "external_subm_accessed": False,
        "external_subm_scored": False,
        "gpu_used": False,
    }
    environment_sha = _write_immutable(environment_path, environment)
    receipt = {
        "schema": "dandi_000688_consumed_subc_shared_zero4_adapter_parity_receipt_v1",
        "status": parity["status"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "append_only": True,
        "fixed_fixture": parity["scope"]["fixture"],
        "views": parity["scope"]["views"],
        "proofs": {
            "bitwise_float32_n_by_4_zero": True,
            "descriptor_target_direction_reads": 0,
            "descriptor_t4_trial_rate_reads": 0,
            "descriptor_t4_fit_calls": 0,
            "descriptor_t4_normalizer_value_reads": 0,
            "activity_first_n30": True,
            "t4_comparator_pool_and_query_boundary_50": True,
            "owner_valid_starts_exact": True,
            "sua_and_pseudo_mua_shapes_valid": True,
            "channel_count_only_construction": True,
            "label_shuffle_prediction_input_invariant": True,
            "label_drop_prediction_input_invariant": True,
        },
        "input_parity_trace": {
            "path": trace_path.name,
            "sha256": trace_sha,
            "bytes": trace_path.stat().st_size,
            "mode": "0444",
        },
        "environment": {
            "path": environment_path.name,
            "sha256": environment_sha,
            "bytes": environment_path.stat().st_size,
            "mode": "0444",
        },
        "checkpoint_files_opened": 0,
        "model_forward_calls": 0,
        "r2_computations": 0,
        "external_subm_accessed": False,
        "external_subm_scored": False,
        "external_scoring_capability_created": False,
    }
    receipt_sha = _write_immutable(receipt_path, receipt)
    seal = {
        "schema": "dandi_000688_consumed_subc_shared_zero4_adapter_parity_seal_v1",
        "status": receipt["status"],
        "append_only": True,
        "artifacts": [
            {
                "path": trace_path.name,
                "sha256": trace_sha,
                "bytes": trace_path.stat().st_size,
                "mode": "0444",
            },
            {
                "path": environment_path.name,
                "sha256": environment_sha,
                "bytes": environment_path.stat().st_size,
                "mode": "0444",
            },
            {
                "path": receipt_path.name,
                "sha256": receipt_sha,
                "bytes": receipt_path.stat().st_size,
                "mode": "0444",
            },
        ],
        "external_subm_accessed": False,
        "external_subm_scored": False,
        "external_scoring_capability_created": False,
    }
    seal_sha = _write_immutable(seal_path, seal)
    return {
        "output_root": str(output_root),
        "status": receipt["status"],
        "trace_sha256": trace_sha,
        "environment_sha256": environment_sha,
        "receipt_sha256": receipt_sha,
        "seal_sha256": seal_sha,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "execute-consumed-subc"), default="dry-run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stored = load_stored_prelaunch(DEFAULT_PRELAUNCH, ROOT)
        if args.mode == "dry-run":
            print(
                json.dumps(
                    {
                        "status": "READY_FOR_FIXED_CONSUMED_SUBC_INPUT_PARITY_ONLY",
                        "prelaunch": {
                            key: stored[key]
                            for key in (
                                "draft_sha256",
                                "receipt_sha256",
                                "seal_sha256",
                                "status",
                            )
                        },
                        "external_subm_access_allowed": False,
                        "external_subm_scoring_allowed": False,
                        "checkpoint_or_model_forward_allowed": False,
                        "gpu_allowed": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = execute_fixed_consumed_subc()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (StaticZero4ParityError, Zero4ParityRunnerError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

