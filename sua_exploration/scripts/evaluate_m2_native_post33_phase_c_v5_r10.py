#!/usr/bin/env python3
"""Static-r10 outer evaluator that preserves the audited v5 inner evaluators."""
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

from sua_exploration.mc_maze.m2_native_post33_evaluator_v4 import (  # noqa: E402
    validate_endpoint_payload, write_payload_commitment,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    CellKey, PROTOCOL_ID, cell_paths, validate_selector_payload,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r10_static import (  # noqa: E402
    CELL_ENV, MANIFEST_ENV, STATIC_MANIFEST, require_cell_execution_capability,
)


SHIM = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r10_worker_shim.py"
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
    args = parser.parse_args()
    if args.protocol != PROTOCOL_ID or args.phase != "PHASE_C_V4":
        raise ValueError("r10 evaluator protocol/phase mismatch")
    key = CellKey(PROTOCOL_ID, args.arm, args.fold, args.seed)
    paths = cell_paths(args.cell_root, key)
    os.environ[CELL_ENV] = str(paths["cell_dir"])
    capability = require_cell_execution_capability(root=args.cell_root, key=key)
    exact = (
        args.opaque_payload_out.resolve() == paths["opaque_payload_run"],
        args.score_commitment_out.resolve() == paths["score_commitment_run"],
        args.deployment_cost_evidence_out.resolve() == paths["deployment_cost_evidence_run"],
    )
    if not all(exact):
        raise ValueError("r10 evaluator output path substitution")
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    if owner.get("owner_token") != args.owner_token:
        raise PermissionError("r10 evaluator ownership mismatch")
    selector = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector, key, run_dir=paths["run"])
    checkpoint = Path(selected["checkpoint_path"]).resolve(strict=True)
    worker = [
        sys.executable, str(SHIM), "--arm", key.arm, "--role", "evaluate",
        "--cell-root", str(args.cell_root.resolve()), "--fold", str(key.fold), "--seed", str(key.seed),
        "--owner-token", args.owner_token, "--opaque-payload-out", str(paths["opaque_payload_run"]),
        "--deployment-cost-evidence-out", str(paths["deployment_cost_evidence_run"]),
        # The audited r9 parser still requires these positional ABI flags.  The
        # shim replaces their imported verifier with r10 static state; no r9
        # authorization file is read or accepted.
        "--authorization", str(STATIC_MANIFEST.resolve()), "--authorization-signature", str(STATIC_MANIFEST.resolve()),
        "--program-receipt", str(STATIC_MANIFEST.resolve()), "--portable-manifest", str(STATIC_MANIFEST.resolve()),
        "--shard-manifest", str(STATIC_MANIFEST.resolve()), "--cost-supplement", capability["_validated_cost_supplement_path"],
    ]
    if key.arm == "t4":
        if args.decoder_evidence_out is None or args.outer_runtime_evidence_out is None:
            raise ValueError("r10 T4 evaluator needs decoder/outer runtime evidence")
        if args.decoder_evidence_out.resolve() != paths["decoder_lifecycle_evidence_run"] or args.outer_runtime_evidence_out.resolve() != paths["outer_runtime_evidence_run"]:
            raise ValueError("r10 T4 evidence output substitution")
        worker += ["--decoder-evidence-out", str(paths["decoder_lifecycle_evidence_run"]), "--outer-runtime-evidence-out", str(paths["outer_runtime_evidence_run"])]
    elif args.decoder_evidence_out is not None or args.outer_runtime_evidence_out is not None:
        raise ValueError("r10 SPINT evaluator forbids T4-only outputs")
    env = dict(os.environ)
    env[MANIFEST_ENV] = str(STATIC_MANIFEST.resolve())
    env[CELL_ENV] = str(paths["cell_dir"])
    completed = subprocess.run(worker, cwd=WORKDIRS[key.arm], env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"r10 audited inner evaluator failed: {completed.returncode}")
    payload_path = paths["opaque_payload_run"].resolve(strict=True)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    validate_endpoint_payload(payload, key=key, selected_checkpoint=checkpoint, resolved_config=paths["resolved_config"])
    os.chmod(payload_path, 0o600)
    write_payload_commitment(
        paths["score_commitment_run"], key=key, payload_path=payload_path,
        execution_capability_evidence=paths["execution_capability_evidence_run"],
    )


if __name__ == "__main__":
    main()
