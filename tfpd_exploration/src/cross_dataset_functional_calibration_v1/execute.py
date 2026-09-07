"""Stage0 CPU orchestrator. No GPU, no hidden files, no decoder R2 for E1."""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

from . import contracts
from . import coverage_audit
from . import documents
from . import h1_population_audit
from . import inventory as inventory_mod
from . import m1_dynamic_audit
from . import plan
from . import receipts


def _assay_or_block(name: str, runner) -> dict[str, Any]:
    try:
        status = runner()
        return {
            "name": status.name,
            "status": status.status,
            "blocker": status.blocker,
            "payload": status.payload,
        }
    except Exception as error:  # noqa: BLE001 — record the precise blocker
        return {
            "name": name,
            "status": "BLOCKED",
            "blocker": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=20),
            "payload": {},
        }


def execute(repo_root: Path) -> tuple[dict[str, str], str | None, str | None]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = Path(repo_root)
    plan.verify_bound_documents(repo_root)
    stage0 = receipts.ensure_stage0_root(repo_root)
    hashes: dict[str, str] = {}
    hashes["attempt.json"] = receipts.write_json(
        stage0 / "attempt.json",
        {
            "schema": "cross_dataset_functional_calibration_attempt_v1",
            "phase": plan.PHASE,
            "workorder_sha256": plan.WORKORDER_SHA256,
            "cpu_only": True,
            "gpu_work_started": False,
            "decoder_r2_e1": False,
        },
    )
    hashes["launch.json"] = receipts.write_json(
        stage0 / "launch.json",
        {
            "schema": "cross_dataset_functional_calibration_launch_v1",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pythonnousersite": os.environ.get("PYTHONNOUSERSITE"),
            "root": plan.STAGE0_ROOT_RELATIVE,
        },
    )

    inventory_body = inventory_mod.build(repo_root)
    hashes["inventory.json"] = receipts.write_json(stage0 / "inventory.json", inventory_body)

    e1 = _assay_or_block("E1", lambda: h1_population_audit.run(repo_root))
    e2 = _assay_or_block("E2", lambda: m1_dynamic_audit.run(repo_root))
    prior = {
        "E1": e1.get("payload") or {},
        "E2": e2.get("payload") or {},
    }
    e3 = _assay_or_block("E3", lambda: coverage_audit.run(repo_root, prior=prior))
    assays = {"E1": e1, "E2": e2, "E3": e3}

    if e1.get("payload"):
        hashes["e1_population_audit.json"] = receipts.write_json(stage0 / "e1_population_audit.json", e1["payload"])
    if e2.get("payload"):
        hashes["e2_dynamic_audit.json"] = receipts.write_json(stage0 / "e2_dynamic_audit.json", e2["payload"])
    if e3.get("payload"):
        hashes["e3_coverage_audit.json"] = receipts.write_json(stage0 / "e3_coverage_audit.json", e3["payload"])
    hashes["assay_status.json"] = receipts.write_json(
        stage0 / "assay_status.json",
        {
            "E1": {"status": e1["status"], "blocker": e1.get("blocker")},
            "E2": {"status": e2["status"], "blocker": e2.get("blocker")},
            "E3": {"status": e3["status"], "blocker": e3.get("blocker")},
        },
    )

    hashes["evidence_matrix.md"] = receipts.write_text(
        stage0 / "evidence_matrix.md", documents.evidence_matrix_md(inventory_body, assays)
    )
    hashes["STAGE1_IMPLEMENTATION_SPEC.md"] = receipts.write_text(
        stage0 / "STAGE1_IMPLEMENTATION_SPEC.md", documents.stage1_spec_md(inventory_body, assays)
    )
    hashes["HANDOFF_FOR_ASTRA_REVIEW.md"] = receipts.write_text(
        stage0 / "HANDOFF_FOR_ASTRA_REVIEW.md",
        documents.handoff_md(inventory_body, assays, hashes),
    )

    ready = [name for name, row in assays.items() if row["status"] == "READY"]
    blocked = {name: row.get("blocker") for name, row in assays.items() if row["status"] != "READY"}
    terminal = {
        "schema": "cross_dataset_functional_calibration_terminal_v1",
        "status": "STAGE0_COMPLETE" if not blocked else "STAGE0_COMPLETE_WITH_BLOCKS",
        "cpu_only": True,
        "gpu_work_started": False,
        "hidden_or_evalai_opened": False,
        "decoder_r2_e1": False,
        "ready": ready,
        "blocked": blocked,
        "stage1_frozen": False,
        "bodies": hashes,
    }
    hashes["terminal.json"] = receipts.write_json(stage0 / "terminal.json", terminal)
    # Rewrite handoff now that terminal hash exists.
    hashes["HANDOFF_FOR_ASTRA_REVIEW.md"] = receipts.write_text(
        stage0 / "HANDOFF_FOR_ASTRA_REVIEW.md",
        documents.handoff_md(inventory_body, assays, hashes),
    )
    return hashes, hashes["terminal.json"], None
