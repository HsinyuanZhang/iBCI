#!/usr/bin/env python3
"""Static-r10 one-cell pipeline using the byte-identical audited r9 workers."""
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

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    CellKey, PROTOCOL_ID, cell_paths, finalize_cell_score_sealed,
    require_same_root_paired_spint_teacher,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r10_static import (  # noqa: E402
    CELL_ENV, MANIFEST_ENV, R10_CELL_ROOT, STATIC_MANIFEST,
    require_cell_execution_capability, validate_static_manifest,
)


SHIM = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r10_worker_shim.py"
EVALUATOR = ROOT / "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v5_r10.py"
ACTIVE: subprocess.Popen[str] | None = None


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("spint", "t4"), required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--owner-token", required=True)
    return parser.parse_args()


def _signal(signum: int, frame: object) -> None:
    del frame
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.terminate()
    raise SystemExit(128 + signum)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    global ACTIVE
    ACTIVE = subprocess.Popen(command, cwd=cwd, env=env, text=True)
    status = ACTIVE.wait()
    ACTIVE = None
    if status != 0:
        raise RuntimeError(f"r10 child exited with {status}: {command[0]}")


def _training_command(args: argparse.Namespace, key: CellKey, paths: dict[str, Path]) -> tuple[list[str], Path]:
    overrides = [
        f"data.loso_fold={key.fold}", f"seed={key.seed}", f"cell_owner_token={args.owner_token}",
        f"cell_paths.cell_dir={paths['cell_dir']}", f"cell_paths.owner={paths['owner']}",
        f"cell_paths.hydra={paths['hydra']}", f"cell_paths.selector_records={paths['selector_records']}",
        f"cell_paths.checkpoints={paths['checkpoints']}", f"cell_paths.resolved_config={paths['resolved_config']}",
        f"cell_paths.deployment_constants={paths['deployment_constants_run']}",
        f"cell_paths.source_cost_evidence={paths['source_cost_evidence_run']}",
        f"cell_paths.cost_supplement={validate_static_manifest()['cost_supplement']['canonical_path']}",
    ]
    command = [sys.executable, str(SHIM), "--arm", key.arm, "--role", "train"]
    if key.arm == "spint":
        return command + ["experiment=m2_native_post33_confirm_v4_spint", *overrides], ROOT / "SPINT-main"
    paired = cell_paths(args.cell_root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed))["completion_receipt"]
    overrides += [
        f"cell_paths.decoder_lifecycle_stages={paths['decoder_lifecycle_stages']}",
        f"cell_paths.secondary_artifact_root={paths['secondary_artifact_root']}",
        f"cell_paths.secondary_artifacts={paths['secondary_artifacts']}",
        f"model.paired_spint_completion_receipt={paired}",
    ]
    return command + ["experiment=m2_native_post33_confirm_v4_t4", *overrides], ROOT / "streaming_calibration_exp"


def main() -> None:
    args = _args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal)
    if args.workspace_root.resolve(strict=True) != ROOT.resolve() or args.data_root.resolve(strict=True) != (ROOT / "SPINT-main/data/000953").resolve(strict=True):
        raise PermissionError("r10 pipeline workspace/data-root substitution")
    if args.cell_root.resolve() != R10_CELL_ROOT.resolve() or os.environ.get(MANIFEST_ENV) != str(STATIC_MANIFEST.resolve()):
        raise PermissionError("r10 pipeline static root/manifest environment mismatch")
    key = CellKey(PROTOCOL_ID, args.arm, args.fold, args.seed)
    paths = cell_paths(args.cell_root, key)
    os.environ[CELL_ENV] = str(paths["cell_dir"])
    capability = require_cell_execution_capability(root=args.cell_root, key=key)
    if paths["selector_records"].exists():
        raise PermissionError("r10 selector must be fresh before training")
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    if owner.get("owner_token") != args.owner_token:
        raise PermissionError("r10 pipeline ownership mismatch")
    if key.arm == "t4":
        paired = cell_paths(args.cell_root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed))["completion_receipt"]
        require_same_root_paired_spint_teacher(args.cell_root, key, paired)

    env = dict(os.environ)
    env[MANIFEST_ENV] = str(STATIC_MANIFEST.resolve())
    env[CELL_ENV] = str(paths["cell_dir"])
    command, cwd = _training_command(args, key, paths)
    _run(command, cwd=cwd, env=env)
    evaluator = [
        sys.executable, str(EVALUATOR), "--protocol", PROTOCOL_ID, "--phase", "PHASE_C_V4",
        "--arm", key.arm, "--fold", str(key.fold), "--seed", str(key.seed),
        "--cell-root", str(args.cell_root.resolve()), "--owner-token", args.owner_token,
        "--opaque-payload-out", str(paths["opaque_payload_run"]),
        "--score-commitment-out", str(paths["score_commitment_run"]),
        "--deployment-cost-evidence-out", str(paths["deployment_cost_evidence_run"]),
    ]
    if key.arm == "t4":
        evaluator += ["--decoder-evidence-out", str(paths["decoder_lifecycle_evidence_run"]), "--outer-runtime-evidence-out", str(paths["outer_runtime_evidence_run"])]
    _run(evaluator, cwd=ROOT, env=env)
    os.chmod(paths["opaque_payload_run"], 0o600)
    kwargs = {}
    if key.arm == "t4":
        kwargs = {
            "decoder_lifecycle_path": paths["decoder_lifecycle_evidence_run"],
            "paired_spint_completion_path": cell_paths(args.cell_root, CellKey(PROTOCOL_ID, "spint", key.fold, key.seed))["completion_receipt"],
            "outer_runtime_evidence_path": paths["outer_runtime_evidence_run"],
        }
    manifest = validate_static_manifest()
    finalize_cell_score_sealed(
        root=args.cell_root, key=key, owner_token=args.owner_token,
        score_commitment_path=paths["score_commitment_run"], opaque_payload_path=paths["opaque_payload_run"],
        global_cost_receipt_path=manifest["cost_receipt"]["canonical_path"],
        cost_supplement_path=capability["_validated_cost_supplement_path"],
        source_cost_evidence_path=paths["source_cost_evidence_run"],
        deployment_cost_evidence_path=paths["deployment_cost_evidence_run"], **kwargs,
    )
    os.chmod(paths["opaque_payload"], 0o600)


if __name__ == "__main__":
    main()
