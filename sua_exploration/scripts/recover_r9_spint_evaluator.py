#!/usr/bin/env python3
"""Recover SPINT fold-0/fold-1 cells: skip training (already done), run evaluator + finalize."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    CellKey, PROTOCOL_ID, cell_paths,
    finalize_cell_score_sealed,
    validate_selector_payload,
)
from sua_exploration.mc_maze.m2_native_post33_authorization_v5_r9 import (
    execution_capability_environment,
    require_cell_execution_capability,
    validate_r9_portable_transfer_manifest,
)

CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_cells_20260805"
RECEIPT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_launch_receipts_20260805"
DATA_ROOT = ROOT / "SPINT-main/data/000953"
COST_SUPP = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost/cost_supplement_r3.json"
PY = sys.executable

def recover_fold(fold: int, gpu_id: str, shard_manifest: Path,
                 authorization: Path, signature: Path,
                 independent_review: Path) -> None:
    key = CellKey(PROTOCOL_ID, "spint", fold, 42)
    owner_token = f"hw3090-fold{fold}-seed42-spint"
    paths = cell_paths(CELL_ROOT, key)

    print(f"\n=== Recovering fold-{fold} (evaluator + finalize) ===")
    print(f"Selector checkpoint: epoch {json.loads(paths['selector_records'].read_text())['selected_epoch']}")

    # Validate selector (checkpoint already exists from training)
    selector = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector, key, run_dir=paths["run"])
    checkpoint = Path(selected["checkpoint_path"]).resolve(strict=True)
    print(f"Selected checkpoint verified: {checkpoint.name}")

    # Execution capability
    auth = require_cell_execution_capability(
        root=CELL_ROOT, key=key,
        authorization_path=authorization,
        signature_path=signature,
        phase_c_program_receipt_path=RECEIPT_ROOT / "program/phase_c_program_v5_r9.json",
        portable_manifest_path=RECEIPT_ROOT / "manifest/portable_v5_r9.json",
        shard_manifest_path=shard_manifest,
        cost_supplement_path=COST_SUPP,
    )

    env = dict(os.environ)
    env.update(execution_capability_environment(
        authorization_path=authorization,
        signature_path=signature,
        phase_c_program_receipt_path=RECEIPT_ROOT / "program/phase_c_program_v5_r9.json",
        portable_manifest_path=RECEIPT_ROOT / "manifest/portable_v5_r9.json",
        shard_manifest_path=shard_manifest,
        cost_supplement_path=COST_SUPP,
    ))
    env["M2_POST33_PHASE_C_CELL_DIR"] = str(paths["cell_dir"])
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    evaluator = auth["evaluator"]["canonical_path"]
    evaluator_cmd = [
        PY, evaluator,
        "--protocol", PROTOCOL_ID,
        "--phase", "PHASE_C_V4",
        "--arm", "spint",
        "--fold", str(fold),
        "--seed", "42",
        "--cell-root", str(CELL_ROOT.resolve()),
        "--owner-token", owner_token,
        "--opaque-payload-out", str(paths["opaque_payload_run"]),
        "--score-commitment-out", str(paths["score_commitment_run"]),
        "--deployment-cost-evidence-out", str(paths["deployment_cost_evidence_run"]),
        "--authorization", str(authorization.resolve(strict=True)),
        "--authorization-signature", str(signature.resolve(strict=True)),
        "--program-receipt", str((RECEIPT_ROOT / "program/phase_c_program_v5_r9.json").resolve(strict=True)),
        "--portable-manifest", str((RECEIPT_ROOT / "manifest/portable_v5_r9.json").resolve(strict=True)),
        "--shard-manifest", str(shard_manifest.resolve(strict=True)),
        "--cost-supplement", str(COST_SUPP.resolve(strict=True)),
    ]
    print(f"Running evaluator...")
    result = subprocess.run(evaluator_cmd, cwd=ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"evaluator failed: fold-{fold} rc={result.returncode}")
    print(f"Evaluator completed for fold-{fold}")

    print(f"Finalizing cell...")
    finalize_cell_score_sealed(
        root=CELL_ROOT,
        key=key,
        owner_token=owner_token,
        score_commitment_path=paths["score_commitment_run"],
        opaque_payload_path=paths["opaque_payload_run"],
        global_cost_receipt_path=auth["cost_receipt"]["canonical_path"],
        cost_supplement_path=COST_SUPP,
        source_cost_evidence_path=paths["source_cost_evidence_run"],
        deployment_cost_evidence_path=paths["deployment_cost_evidence_run"],
    )
    print(f"fold-{fold} SPINT cell COMPLETED")


if __name__ == "__main__":
    R = RECEIPT_ROOT
    # GPU0: fold-0
    recover_fold(
        fold=0, gpu_id="0",
        shard_manifest=R / "manifest/shard_stage_a_gpu0_v5_r9.json",
        authorization=R / "auth/stage_a_execution_gpu0_v5_r9.json",
        signature=R / "auth/stage_a_execution_gpu0_v5_r9.sig",
        independent_review=R / "review/stage_a_gpu0_independent_review.json",
    )
    # GPU1: fold-1
    recover_fold(
        fold=1, gpu_id="1",
        shard_manifest=R / "manifest/shard_stage_a_gpu1_v5_r9.json",
        authorization=R / "auth/stage_a_execution_gpu1_v5_r9.json",
        signature=R / "auth/stage_a_execution_gpu1_v5_r9.sig",
        independent_review=R / "review/stage_a_gpu1_independent_review.json",
    )
    print("\n=== Both SPINT cells recovered ===")
