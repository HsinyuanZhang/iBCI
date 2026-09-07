#!/usr/bin/env python3
"""Recover T4 fold-4/fold-5 cells (evaluator + finalize only), then run fold-6 from scratch."""
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
)

CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_cells_20260805"
RECEIPT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_launch_receipts_20260805"
COST_SUPP = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost/cost_supplement_r3.json"
PY = sys.executable


def recover_evaluator(fold: int, gpu_id: str, shard_manifest: Path,
                      authorization: Path, signature: Path) -> None:
    key = CellKey(PROTOCOL_ID, "t4", fold, 42)
    owner_token = f"hw3090-fold{fold}-seed42-t4"
    paths = cell_paths(CELL_ROOT, key)

    print(f"\n=== Recovering fold-{fold} T4 (evaluator + finalize) ===")
    selector = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    selected = validate_selector_payload(selector, key, run_dir=paths["run"])
    checkpoint = Path(selected["checkpoint_path"]).resolve(strict=True)
    print(f"Selected checkpoint: {checkpoint.name}")

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
        "--protocol", PROTOCOL_ID, "--phase", "PHASE_C_V4",
        "--arm", "t4", "--fold", str(fold), "--seed", "42",
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
        "--decoder-evidence-out", str(paths["decoder_lifecycle_evidence_run"]),
        "--outer-runtime-evidence-out", str(paths["outer_runtime_evidence_run"]),
    ]
    print(f"Running evaluator...")
    result = subprocess.run(evaluator_cmd, cwd=ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"evaluator failed: fold-{fold} rc={result.returncode}")
    print(f"Evaluator completed for fold-{fold}")

    paired = cell_paths(CELL_ROOT, CellKey(PROTOCOL_ID, "spint", fold, 42))["completion_receipt"]
    print(f"Finalizing cell...")
    finalize_cell_score_sealed(
        root=CELL_ROOT, key=key, owner_token=owner_token,
        score_commitment_path=paths["score_commitment_run"],
        opaque_payload_path=paths["opaque_payload_run"],
        global_cost_receipt_path=auth["cost_receipt"]["canonical_path"],
        cost_supplement_path=COST_SUPP,
        source_cost_evidence_path=paths["source_cost_evidence_run"],
        deployment_cost_evidence_path=paths["deployment_cost_evidence_run"],
        decoder_lifecycle_path=paths["decoder_lifecycle_evidence_run"],
        paired_spint_completion_path=paired,
        outer_runtime_evidence_path=paths["outer_runtime_evidence_run"],
    )
    print(f"fold-{fold} T4 cell COMPLETED")


if __name__ == "__main__":
    R = RECEIPT_ROOT
    recover_evaluator(4, "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
        shard_manifest=R / "manifest/shard_stage_a_gpu0_v5_r9.json",
        authorization=R / "auth/stage_a_execution_gpu0_v5_r9.json",
        signature=R / "auth/stage_a_execution_gpu0_v5_r9.sig")
    recover_evaluator(5, "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
        shard_manifest=R / "manifest/shard_stage_a_gpu1_v5_r9.json",
        authorization=R / "auth/stage_a_execution_gpu1_v5_r9.json",
        signature=R / "auth/stage_a_execution_gpu1_v5_r9.sig")
    print("\n=== fold-4/fold-5 T4 recovered ===")
