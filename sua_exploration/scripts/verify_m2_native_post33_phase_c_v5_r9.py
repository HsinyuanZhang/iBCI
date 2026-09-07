#!/usr/bin/env python3
"""CPU-only r9 source/capability verifier; it cannot launch a worker."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r9 import (  # noqa: E402
    validate_r9_portable_transfer_manifest,
    validate_r9_shard_manifest,
    verify_signed_authorization,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    ARMS,
    file_metadata,
    require_canonical_regular_file,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r9_program import (  # noqa: E402
    PUBLIC_KEY,
    R9_CELL_ROOT,
    R9_RECEIPT_ROOT,
    validate_public_anchor,
    validate_r6d_retirement,
    validate_r7_prelaunch_retirement,
    validate_r8_pre_hydra_retirement,
    validate_r9_program_receipt,
)


COST = (
    ROOT
    / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost/cost_supplement_r3.json"
)
PROGRAM = R9_RECEIPT_ROOT / "program/phase_c_program_v5_r9.json"
PORTABLE = R9_RECEIPT_ROOT / "manifest/portable_v5_r9.json"
SHARDS = {
    "gpu0": R9_RECEIPT_ROOT / "manifest/shard_stage_a_gpu0_v5_r9.json",
    "gpu1": R9_RECEIPT_ROOT / "manifest/shard_stage_a_gpu1_v5_r9.json",
}
AUTHS = {
    "gpu0": R9_RECEIPT_ROOT / "auth/stage_a_execution_gpu0_v5_r9.json",
    "gpu1": R9_RECEIPT_ROOT / "auth/stage_a_execution_gpu1_v5_r9.json",
}
SUMMARY = R9_RECEIPT_ROOT / "launch/stage_a_ready_not_launched.json"
PREPARE = R9_RECEIPT_ROOT / "prelaunch/r9_stage_a_prepare_receipt.json"


def _read(path: Path) -> Mapping[str, Any]:
    source = require_canonical_regular_file(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected mapping: {path}")
    return payload


def _normalized(source: Path, *, old: str, new: str) -> str:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    rendered = ast.unparse(tree)
    return rendered.replace(new, old)


def _assert_v5_worker_equivalence() -> dict[str, Any]:
    pairs = {
        "spint_evaluator": (
            ROOT / "SPINT-main/src/evaluate_post33_phase_c_v5.py",
            ROOT / "SPINT-main/src/evaluate_post33_phase_c_v5_r9.py",
        ),
        "t4_evaluator": (
            ROOT / "streaming_calibration_exp/src/evaluate_post33_phase_c_v5.py",
            ROOT / "streaming_calibration_exp/src/evaluate_post33_phase_c_v5_r9.py",
        ),
    }
    evidence: dict[str, Any] = {}
    for label, (baseline, r9) in pairs.items():
        require_canonical_regular_file(baseline, within=ROOT)
        require_canonical_regular_file(r9, within=ROOT)
        if _normalized(r9, old="m2_native_post33_authorization_v4", new="m2_native_post33_authorization_v5_r9") != _normalized(
            baseline, old="m2_native_post33_authorization_v4", new="m2_native_post33_authorization_v4"
        ):
            raise ValueError(f"r9 {label} differs from its reviewed baseline beyond auth import")
        evidence[label] = {"baseline": file_metadata(baseline), "r9": file_metadata(r9)}
    trainers = {
        "spint_trainer": (ROOT / "SPINT-main/src/train_post33_phase_c_v5_r9.py", "spint"),
        "t4_trainer": (ROOT / "streaming_calibration_exp/src/train_post33_phase_c_v5_r9.py", "t4"),
    }
    for label, (trainer, arm) in trainers.items():
        require_canonical_regular_file(trainer, within=ROOT)
        source = trainer.read_text(encoding="utf-8")
        main_guard = source.index('if __name__ == "__main__":')
        forbidden = (
            "require_phase_c_training_wrapper_pre_hydra_gate",
            "m2_native_post33_authorization_v4",
        )
        if any(token in source for token in forbidden):
            raise ValueError(f"r9 {label} retains a prohibited v4 capability route")
        capability_only = f'verify_r9_training_wrapper_capability_only(sys.argv[1:], arm="{arm}")'
        r9_gate = f'require_r9_training_wrapper_pre_hydra_gate(sys.argv[1:], arm="{arm}")'
        if (
            capability_only not in source
            or r9_gate not in source
            or source.index(capability_only, main_guard) > source.index(r9_gate, main_guard)
            or source.index(r9_gate, main_guard) > source.index("    main()", main_guard)
        ):
            raise ValueError(f"r9 {label} does not bind the r9 gate before Hydra")
        evidence[label] = {
            "r9": file_metadata(trainer),
            "forbidden_v4_capability_route_absent": True,
            "direct_r9_pre_hydra_gate_before_main": True,
            "capability_only_branch_before_production_gate": True,
        }
    return evidence


def verify() -> dict[str, Any]:
    if R9_CELL_ROOT.exists():
        raise PermissionError("r9 verification refuses a root after any cell/selector materialization")
    anchor = validate_public_anchor()
    retirement = validate_r6d_retirement()
    r7_retirement = validate_r7_prelaunch_retirement()
    r8_retirement = validate_r8_pre_hydra_retirement()
    program = validate_r9_program_receipt(PROGRAM)
    validate_r9_portable_transfer_manifest(
        PORTABLE,
        workspace_root=ROOT,
        data_root=ROOT / "SPINT-main/data/000953",
        cell_root=R9_CELL_ROOT,
    )
    nonces: list[str] = []
    coverage: set[tuple[str, int, int]] = set()
    capabilities: dict[str, Any] = {}
    for name in ("gpu0", "gpu1"):
        shard = validate_r9_shard_manifest(
            SHARDS[name], portable_manifest_path=PORTABLE, cell_root=R9_CELL_ROOT
        )
        auth = verify_signed_authorization(
            AUTHS[name],
            AUTHS[name].with_suffix(".sig"),
            phase_c_program_receipt_path=PROGRAM,
            portable_manifest_path=PORTABLE,
            shard_manifest_path=SHARDS[name],
            cost_supplement_path=COST,
            cell_root=R9_CELL_ROOT,
        )
        nonces.append(str(auth["single_use_nonce"]))
        for fold in shard["fold_allowlist"]:
            coverage.update((arm, fold, 42) for arm in ARMS)
        capabilities[name] = {
            "authorization": file_metadata(AUTHS[name]),
            "signature": file_metadata(AUTHS[name].with_suffix(".sig")),
            "shard": file_metadata(SHARDS[name]),
            "fold_allowlist": shard["fold_allowlist"],
        }
    expected = {(arm, fold, 42) for arm in ARMS for fold in range(7)}
    if set(nonces) & {""} or len(nonces) != len(set(nonces)) or coverage != expected:
        raise PermissionError("r9 Stage-A capability coverage/nonce integrity failed")
    forbidden = list((R9_RECEIPT_ROOT / "auth").glob("*stage_b*")) + list(
        (R9_RECEIPT_ROOT / "auth").glob("*opening*"))
    if forbidden:
        raise PermissionError("r9 must not contain Stage-B or score-opening authority")
    summary, prepare = _read(SUMMARY), _read(PREPARE)
    if (
        summary.get("status") != "READY_NOT_LAUNCHED_INDEPENDENT_REVIEW_REQUIRED"
        or summary.get("stage_a_total_cells") != 14
        or summary.get("cell_root_exists") is not False
        or summary.get("selector_records_materialized") is not False
        or summary.get("opening_capability_materialized") is not False
        or summary.get("stage_b_capability_materialized") is not False
        or prepare.get("stage_a_total_cells") != 14
        or prepare.get("cell_root_exists") is not False
    ):
        raise PermissionError("r9 prepared-state receipt violates no-launch boundary")
    workers = _assert_v5_worker_equivalence()
    return {
        "schema": "m2_post33_phase_c_v5_r9_cpu_verifier_result_v1",
        "status": "PASS_READY_NOT_LAUNCHED",
        "public_anchor": anchor,
        "r6d_retirement": retirement,
        "r7_prelaunch_retirement": r7_retirement,
        "r8_pre_hydra_retirement": r8_retirement,
        "program": file_metadata(PROGRAM),
        "program_source_map_entry_count": program["source_map_entry_count"],
        "v5_worker_equivalence": workers,
        "capabilities": capabilities,
        "stage_a_total_cells": 14,
        "cell_root_exists": False,
        "selector_records_materialized": False,
        "opening_capability_materialized": False,
        "stage_b_capability_materialized": False,
        "gpu_used": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
        "private_key_serialized_or_disk_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.json_out is not None:
        # This verifier is intentionally side-effect free by default; an
        # explicit output is allowed only under the fresh r9 receipt root.
        output = args.json_out.resolve()
        try:
            output.relative_to(R9_RECEIPT_ROOT.resolve())
        except ValueError as exc:
            raise PermissionError("verifier output must remain inside r9 receipt root") from exc
        if output.exists():
            raise FileExistsError("verifier output already exists")
        from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import write_json_exclusive
        write_json_exclusive(output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
