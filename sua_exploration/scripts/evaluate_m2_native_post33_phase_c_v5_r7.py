#!/usr/bin/env python3
"""r7 production evaluator bound only to the full repaired v5 workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import (
    validate_endpoint_payload,
    write_payload_commitment,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey,
    PROTOCOL_ID,
    cell_paths,
    validate_selector_payload,
)
from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r7 import (
    require_cell_execution_capability,
)


WORKERS = {
    "spint": ROOT / "SPINT-main/src/evaluate_post33_phase_c_v5_r7.py",
    "t4": ROOT / "streaming_calibration_exp/src/evaluate_post33_phase_c_v5_r7.py",
}
WORKDIRS = {"spint": ROOT / "SPINT-main", "t4": ROOT / "streaming_calibration_exp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--arm", choices=("spint", "t4"), required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--opaque-payload-out", type=Path, required=True)
    parser.add_argument("--score-commitment-out", type=Path, required=True)
    parser.add_argument("--deployment-cost-evidence-out", type=Path, required=True)
    parser.add_argument("--decoder-evidence-out", type=Path)
    parser.add_argument("--outer-runtime-evidence-out", type=Path)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-signature", type=Path, required=True)
    parser.add_argument("--program-receipt", type=Path, required=True)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--cost-supplement", type=Path, required=True)
    args = parser.parse_args()
    if args.protocol != PROTOCOL_ID or args.phase != "PHASE_C_V4":
        raise ValueError("evaluator protocol/phase mismatch")
    key = CellKey(PROTOCOL_ID, args.arm, args.fold, args.seed)
    paths = cell_paths(args.cell_root, key)
    require_cell_execution_capability(
        root=args.cell_root,
        key=key,
        authorization_path=args.authorization,
        signature_path=args.authorization_signature,
        phase_c_program_receipt_path=args.program_receipt,
        portable_manifest_path=args.portable_manifest,
        shard_manifest_path=args.shard_manifest,
        cost_supplement_path=args.cost_supplement,
    )
    if args.opaque_payload_out.resolve() != paths["opaque_payload_run"]:
        raise ValueError("evaluator opaque payload output path substitution")
    if args.score_commitment_out.resolve() != paths["score_commitment_run"]:
        raise ValueError("evaluator commitment output path substitution")
    if args.deployment_cost_evidence_out.resolve() != paths["deployment_cost_evidence_run"]:
        raise ValueError("evaluator deployment-cost output path substitution")
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    if owner.get("owner_token") != args.owner_token:
        raise PermissionError("evaluator ownership mismatch")
    selector = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector, key, run_dir=paths["run"])
    checkpoint = Path(selected["checkpoint_path"]).resolve(strict=True)
    worker = [
        sys.executable, str(WORKERS[key.arm]),
        "--cell-root", str(args.cell_root.resolve()),
        "--fold", str(key.fold), "--seed", str(key.seed),
        "--owner-token", args.owner_token,
        "--authorization", str(args.authorization.resolve(strict=True)),
        "--authorization-signature", str(args.authorization_signature.resolve(strict=True)),
        "--program-receipt", str(args.program_receipt.resolve(strict=True)),
        "--portable-manifest", str(args.portable_manifest.resolve(strict=True)),
        "--shard-manifest", str(args.shard_manifest.resolve(strict=True)),
        "--cost-supplement", str(args.cost_supplement.resolve(strict=True)),
        "--opaque-payload-out", str(paths["opaque_payload_run"]),
        "--deployment-cost-evidence-out", str(paths["deployment_cost_evidence_run"]),
    ]
    if key.arm == "t4":
        if args.decoder_evidence_out is None or args.outer_runtime_evidence_out is None:
            raise ValueError("T4 evaluator requires decoder and outer runtime evidence outputs")
        if args.decoder_evidence_out.resolve() != paths["decoder_lifecycle_evidence_run"]:
            raise ValueError("T4 decoder evidence output substitution")
        if args.outer_runtime_evidence_out.resolve() != paths["outer_runtime_evidence_run"]:
            raise ValueError("T4 outer runtime output substitution")
        worker += [
            "--decoder-evidence-out", str(paths["decoder_lifecycle_evidence_run"]),
            "--outer-runtime-evidence-out", str(paths["outer_runtime_evidence_run"]),
        ]
    elif args.decoder_evidence_out is not None or args.outer_runtime_evidence_out is not None:
        raise ValueError("SPINT evaluator forbids T4-only outputs")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORKDIRS[key.arm])
    completed = subprocess.run(worker, cwd=WORKDIRS[key.arm], env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"exact endpoint evaluator worker failed with {completed.returncode}")
    payload_path = paths["opaque_payload_run"].resolve(strict=True)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    validate_endpoint_payload(
        payload, key=key, selected_checkpoint=checkpoint,
        resolved_config=paths["resolved_config"],
    )
    write_payload_commitment(
        paths["score_commitment_run"],
        key=key,
        payload_path=payload_path,
        execution_capability_evidence=paths["execution_capability_evidence_run"],
    )


if __name__ == "__main__":
    main()
