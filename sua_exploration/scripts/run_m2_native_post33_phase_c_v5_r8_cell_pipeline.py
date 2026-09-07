#!/usr/bin/env python3
"""Authorized r8 single-cell pipeline bound to the repaired v5 workers.

The pipeline imports no scorer.  It invokes the exact evaluator binary bound by
a future root-issued authorization, then consumes only its opaque commitment
and structural runtime evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PROTOCOL_ID,
    cell_paths,
    finalize_cell_score_sealed,
    require_same_root_paired_spint_teacher,
)
from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r8 import (
    execution_capability_environment,
    require_cell_execution_capability,
    validate_r8_portable_transfer_manifest,
)


ACTIVE_CHILD: subprocess.Popen[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("spint", "t4"), required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-signature", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--program-receipt", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--cost-supplement", type=Path, required=True)
    return parser.parse_args()


def _signal(signum, frame) -> None:
    del frame
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        ACTIVE_CHILD.terminate()
    raise SystemExit(128 + signum)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    global ACTIVE_CHILD
    ACTIVE_CHILD = subprocess.Popen(command, cwd=cwd, env=env, text=True)
    return_code = ACTIVE_CHILD.wait()
    ACTIVE_CHILD = None
    if return_code != 0:
        raise RuntimeError(f"child exited with {return_code}: {command[0]}")


def _training_command(args: argparse.Namespace, key: CellKey, paths: dict[str, Path]) -> tuple[list[str], Path]:
    overrides = [
        f"data.loso_fold={key.fold}",
        f"seed={key.seed}",
        f"cell_owner_token={args.owner_token}",
        f"cell_paths.cell_dir={paths['cell_dir']}",
        f"cell_paths.owner={paths['owner']}",
        f"cell_paths.hydra={paths['hydra']}",
        f"cell_paths.selector_records={paths['selector_records']}",
        f"cell_paths.checkpoints={paths['checkpoints']}",
        f"cell_paths.resolved_config={paths['resolved_config']}",
        f"cell_paths.deployment_constants={paths['deployment_constants_run']}",
        f"cell_paths.source_cost_evidence={paths['source_cost_evidence_run']}",
        f"cell_paths.cost_supplement={args.cost_supplement.resolve(strict=True)}",
    ]
    if key.arm == "spint":
        return [
            sys.executable,
            str(ROOT / "SPINT-main/src/train_post33_phase_c_v5_r8.py"),
            "experiment=m2_native_post33_confirm_v4_spint",
            *overrides,
        ], ROOT / "SPINT-main"
    paired = cell_paths(
        args.cell_root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed)
    )["completion_receipt"]
    overrides += [
        f"cell_paths.decoder_lifecycle_stages={paths['decoder_lifecycle_stages']}",
        f"cell_paths.secondary_artifact_root={paths['secondary_artifact_root']}",
        f"cell_paths.secondary_artifacts={paths['secondary_artifacts']}",
        f"model.paired_spint_completion_receipt={paired}",
    ]
    return [
        sys.executable,
        str(ROOT / "streaming_calibration_exp/src/train_post33_phase_c_v5_r8.py"),
        "experiment=m2_native_post33_confirm_v4_t4",
        *overrides,
    ], ROOT / "streaming_calibration_exp"


def main() -> None:
    args = parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal)
    validate_r8_portable_transfer_manifest(
        args.portable_manifest,
        workspace_root=args.workspace_root,
        data_root=args.data_root,
        cell_root=args.cell_root,
    )
    if args.workspace_root.resolve(strict=True) != ROOT.resolve(strict=True):
        raise PermissionError("cell pipeline workspace differs from its source workspace")
    key = CellKey(PROTOCOL_ID, args.arm, args.fold, args.seed)
    authorization = require_cell_execution_capability(
        root=args.cell_root,
        key=key,
        authorization_path=args.authorization,
        signature_path=args.authorization_signature,
        phase_c_program_receipt_path=args.program_receipt,
        portable_manifest_path=args.portable_manifest,
        shard_manifest_path=args.shard_manifest,
        cost_supplement_path=args.cost_supplement,
    )
    paths = cell_paths(args.cell_root, key)
    if paths["selector_records"].exists():
        raise PermissionError("r8 requires a fresh selector; selector records pre-exist before training")
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    if owner.get("owner_token") != args.owner_token:
        raise PermissionError("cell pipeline does not own this cell")
    if key.arm == "t4":
        paired = cell_paths(
            args.cell_root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed)
        )["completion_receipt"]
        require_same_root_paired_spint_teacher(args.cell_root, key, paired)

    env = dict(os.environ)
    env.update(
        execution_capability_environment(
            authorization_path=args.authorization,
            signature_path=args.authorization_signature,
            phase_c_program_receipt_path=args.program_receipt,
            portable_manifest_path=args.portable_manifest,
            shard_manifest_path=args.shard_manifest,
            cost_supplement_path=args.cost_supplement,
        )
    )
    env["M2_POST33_PHASE_C_CELL_DIR"] = str(paths["cell_dir"])
    command, cwd = _training_command(args, key, paths)
    _run(command, cwd=cwd, env=env)

    evaluator = authorization["evaluator"]["canonical_path"]
    evaluator_command = [
        sys.executable, evaluator,
        "--protocol", PROTOCOL_ID,
        "--phase", "PHASE_C_V4",
        "--arm", key.arm,
        "--fold", str(key.fold),
        "--seed", str(key.seed),
        "--cell-root", str(args.cell_root.resolve()),
        "--owner-token", args.owner_token,
        "--opaque-payload-out", str(paths["opaque_payload_run"]),
        "--score-commitment-out", str(paths["score_commitment_run"]),
        "--deployment-cost-evidence-out", str(paths["deployment_cost_evidence_run"]),
        "--authorization", str(args.authorization.resolve(strict=True)),
        "--authorization-signature", str(args.authorization_signature.resolve(strict=True)),
        "--program-receipt", str(args.program_receipt.resolve(strict=True)),
        "--portable-manifest", str(args.portable_manifest.resolve(strict=True)),
        "--shard-manifest", str(args.shard_manifest.resolve(strict=True)),
        "--cost-supplement", str(args.cost_supplement.resolve(strict=True)),
    ]
    if key.arm == "t4":
        evaluator_command += [
            "--decoder-evidence-out", str(paths["decoder_lifecycle_evidence_run"]),
            "--outer-runtime-evidence-out", str(paths["outer_runtime_evidence_run"]),
        ]
    _run(evaluator_command, cwd=args.workspace_root.resolve(strict=True), env=env)

    finalize_kwargs = {}
    if key.arm == "t4":
        paired = cell_paths(
            args.cell_root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed)
        )["completion_receipt"]
        finalize_kwargs = {
            "decoder_lifecycle_path": paths["decoder_lifecycle_evidence_run"],
            "paired_spint_completion_path": paired,
            "outer_runtime_evidence_path": paths["outer_runtime_evidence_run"],
        }
    finalize_cell_score_sealed(
        root=args.cell_root,
        key=key,
        owner_token=args.owner_token,
        score_commitment_path=paths["score_commitment_run"],
        opaque_payload_path=paths["opaque_payload_run"],
        global_cost_receipt_path=authorization["cost_receipt"]["canonical_path"],
        cost_supplement_path=args.cost_supplement,
        source_cost_evidence_path=paths["source_cost_evidence_run"],
        deployment_cost_evidence_path=paths["deployment_cost_evidence_run"],
        **finalize_kwargs,
    )


if __name__ == "__main__":
    main()
