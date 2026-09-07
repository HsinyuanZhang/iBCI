#!/usr/bin/env python3
"""Small CLI for the production sub-M V9 three-arm forward-only matrix.

This command deliberately has three separate operations:

``write-input-manifest``
    Read-only pinning of frozen assets/models/normalizers into one immutable
    manifest.  It does not open a model or construct an NWB session.
``forward``
    Build selected cells and seal prediction/target artifacts only.
``finalize``
    CPU-only R2 aggregation after all 270 artifacts exist.

The explicit zero4 checkpoint/closure arguments prevent a hidden dependency on
the previous remote C1 staging directory while retaining its exact provenance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime


def _parse_assignment(value: str, *, label: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{label} must be KEY=PATH")
    key, raw_path = value.split("=", 1)
    if not key or not raw_path:
        raise argparse.ArgumentTypeError(f"{label} must be KEY=PATH")
    return key, Path(raw_path).expanduser().resolve()


def _parse_zero4(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        raw_seed, path = _parse_assignment(value, label="--zero4-checkpoint")
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError("zero4 checkpoint key must be seed 42, 43, or 44") from exc
        if seed not in runtime.SEEDS or seed in parsed:
            raise ValueError("zero4 checkpoints must specify each of seeds 42,43,44 exactly once")
        parsed[seed] = path
    if set(parsed) != set(runtime.SEEDS):
        raise ValueError("zero4 checkpoints must specify each of seeds 42,43,44 exactly once")
    return parsed


def _t4_ts4_closure_paths(repo_root: Path) -> dict[tuple[str, int], Path]:
    from sua_exploration.mc_maze import subm_co_score_only as v1
    return {
        (row.arm, row.seed): (repo_root / row.run_metadata.relative_path).resolve()
        for row in v1.TERMINAL_CHECKPOINTS
    }


def _parse_zero4_closures(values: list[str]) -> dict[tuple[str, int], Path]:
    parsed: dict[tuple[str, int], Path] = {}
    for value in values:
        raw_key, path = _parse_assignment(value, label="--zero4-closure")
        try:
            arm, raw_seed = raw_key.split(":", 1)
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError("zero4 closure key must be shared_zero4:SEED") from exc
        key = (arm, seed)
        if key[0] != "shared_zero4" or key[1] not in runtime.SEEDS or key in parsed:
            raise ValueError("zero4 closures must specify shared_zero4:42/43/44 exactly once")
        parsed[key] = path
    expected = {("shared_zero4", seed) for seed in runtime.SEEDS}
    if set(parsed) != expected:
        raise ValueError("zero4 closures must specify shared_zero4:42/43/44 exactly once")
    return parsed


def _selected_keys(args: argparse.Namespace, contract: runtime.RuntimeContract) -> tuple[runtime.CellKey, ...]:
    selected = []
    for key in contract.expected_keys:
        if args.asset_id and key.asset_id != args.asset_id:
            continue
        if args.view and key.view != args.view:
            continue
        if args.arm and key.arm != args.arm:
            continue
        if args.seed is not None and key.seed != args.seed:
            continue
        selected.append(key)
    if args.max_cells is not None:
        selected = selected[:args.max_cells]
    if not selected:
        raise ValueError("filters selected zero V9 cells")
    return tuple(selected)


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("write-input-manifest", "forward", "finalize"))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--nwb-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zero4-checkpoint", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--zero4-closure", action="append", default=[], metavar="shared_zero4:SEED=PATH")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--asset-id")
    parser.add_argument("--view", choices=runtime.VIEWS)
    parser.add_argument("--arm", choices=runtime.ARMS)
    parser.add_argument("--seed", type=int, choices=runtime.SEEDS)
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    if args.mode in {"forward", "finalize"} and args.output_root is None:
        parser.error("--output-root is required for forward/finalize")
    if args.max_cells is not None and args.max_cells <= 0:
        parser.error("--max-cells must be positive")
    return args


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    nwb_root = args.nwb_root.expanduser().resolve()
    input_manifest = args.input_manifest.expanduser().resolve()
    if args.mode == "write-input-manifest":
        zero4 = _parse_zero4(args.zero4_checkpoint)
        closures = _t4_ts4_closure_paths(repo_root)
        closures.update(_parse_zero4_closures(args.zero4_closure))
        contract = runtime.make_runtime_contract(
            repo_root=repo_root,
            nwb_root=nwb_root,
            zero4_checkpoints=zero4,
            closure_metadata=closures,
            teacher_checkpoint=args.teacher_checkpoint,
        )
        runtime.verify_runtime_inputs(contract=contract, nwb_root=nwb_root)
        digest = runtime.write_input_manifest(input_manifest, contract)
        reopened, reopened_digest = runtime.load_input_manifest(input_manifest)
        if reopened != contract or reopened_digest != digest:
            raise runtime.V9RuntimeError("input manifest reopen/hash mismatch")
        _print({
            "status": "INPUT_MANIFEST_WRITTEN_AND_REOPENED",
            "path": str(input_manifest),
            "sha256": digest,
            "cohort_sessions": len(contract.cohort),
            "models": len(contract.checkpoints),
            "query_windows_per_view": sum(row.query_window_count for row in contract.cohort),
        })
        return
    contract, manifest_sha = runtime.load_input_manifest(input_manifest)
    if args.mode == "forward":
        keys = _selected_keys(args, contract)
        result = runtime.run_forward_cells(
            repo_root=repo_root,
            nwb_root=nwb_root,
            output_root=args.output_root.expanduser().resolve(),
            contract=contract,
            device=args.device,
            input_manifest_path=input_manifest,
            input_manifest_sha256=manifest_sha,
            requested_keys=keys,
        )
        _print({
            "status": "PHASE_A_FORWARD_ARTIFACTS_COMMITTED_NO_METRIC",
            "completed_cells": len(result),
            "requested_cells": len(keys),
            "output_root": str(args.output_root.expanduser().resolve()),
            "input_manifest_sha256": manifest_sha,
        })
        return
    runtime.verify_runtime_inputs(contract=contract, nwb_root=nwb_root)
    result = runtime.finalize_artifacts(output_root=args.output_root.expanduser().resolve(), contract=contract)
    _print({"status": "FULL_270_CPU_FINALIZED", **result, "input_manifest_sha256": manifest_sha})


if __name__ == "__main__":
    main()
