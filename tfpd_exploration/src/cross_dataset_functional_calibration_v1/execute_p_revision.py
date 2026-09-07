"""Write the P-operator revision root. CPU only. Does not touch Stage0."""
from __future__ import annotations

import os
from pathlib import Path

from . import parent_audit
from . import plan
from . import receipts


def _revision_spec_md(audit: dict[str, object]) -> str:
    return "\n".join(
        [
            "# STAGE1_IMPLEMENTATION_SPEC — named revision, not GPU-frozen",
            "",
            "Authority: `REVISION_CROSS_DATASET_P_OPERATOR_V1_20260905.md`.",
            "Stage0 draft at `20260905_113700/STAGE1_IMPLEMENTATION_SPEC.md` is unchanged.",
            "",
            "## Gate status",
            "",
            "`STAGE1_NAMED_REVISION`. GPU P-FIX / P-CA remain **not eligible**.",
            "",
            "## Bound parent",
            "",
            f"- Bytes: `{audit['parent_bytes']}`",
            f"- Path: `{audit['parent_relative']}`",
            f"- Teacher: `{audit['teacher_bytes']}`",
            f"- Claim: `{audit['claim']}`",
            f"- Full ckpt schema: `{audit['full_ckpt_schema']}`",
            "",
            "## Bound f_eta / solve",
            "",
            f"- `{plan.P_F_ETA_NAME}` shape `{list(plan.P_F_ETA_SHAPE)}`",
            "- `carrier_solver.solve_ridge` = bank `/n` ridge, intercept unpenalized",
            "",
            "## Still open before GPU",
            "",
            "- source-pooled carrier normalizer rematerialization for both arms",
            "- 12-epoch store / source-selection dates",
            "- disposable 100-step profile",
            "- workorder §6 items 4–9 on a live consumer",
            "",
        ]
    )


def _handoff_md(audit: dict[str, object], hashes: dict[str, str]) -> str:
    return "\n".join(
        [
            "# HANDOFF — P operator named revision (CPU)",
            "",
            "Does not cover the M2 dual-track pack. Stage0 E1/E2/E3 receipts are unchanged.",
            "",
            "## Decisions",
            "",
            "| Stream | Stage | Decision |",
            "|---|---|---|",
            "| E | E0–E3 | COMPLETE (Stage0 root; not rerun) |",
            "| P | parent / f_eta / torch ridge | **NAMED_REVISION** |",
            "| P | GPU pilot | NOT_RUN / still blocked |",
            "",
            "## Bound parent",
            "",
            f"- S-Fix `{audit['parent_bytes']}`",
            f"- Teacher `{audit['teacher_bytes']}` left out `{audit['outer_left_out']}`",
            f"- Claim `{audit['claim']}`",
            f"- Teacher query caveat: {audit['teacher_query_caveat']}",
            "",
            "## Smallest next step",
            "",
            "Astra freeze of this revision, then rematerialize the carrier normalizer,",
            "CPU consumer tests 4–9, and one disposable 100-step profile. No 12-epoch",
            "pair and no E2 lags until that profile exists.",
            "",
            "## Hashes this turn",
            "",
            *[f"- `{name}`: `{digest}`" for name, digest in sorted(hashes.items())],
            "",
        ]
    )


def execute(repo_root: Path) -> tuple[dict[str, str], str]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = Path(repo_root)
    plan.verify_bound_documents(repo_root)
    revision_path = repo_root / plan.P_REVISION_RELATIVE
    observed = plan.sha256_bytes(revision_path.read_bytes())
    if observed != plan.P_REVISION_SHA256:
        raise plan.PlanError(f"P revision sha drift: {observed}")
    root = receipts.ensure_revision_root(repo_root)
    hashes: dict[str, str] = {}
    audit = parent_audit.audit(repo_root)
    hashes["parent_audit.json"] = receipts.write_json(root / "parent_audit.json", audit)
    hashes["STAGE1_IMPLEMENTATION_SPEC.md"] = receipts.write_text(
        root / "STAGE1_IMPLEMENTATION_SPEC.md", _revision_spec_md(audit)
    )
    terminal = {
        "schema": "cross_dataset_functional_calibration_p_revision_terminal_v1",
        "status": "STAGE1_NAMED_REVISION",
        "cpu_only": True,
        "gpu_work_started": False,
        "stage1_frozen": False,
        "gpu_eligible": False,
        "parent_bytes": audit["parent_bytes"],
        "f_eta": plan.P_F_ETA_NAME,
        "stage0_root": plan.STAGE0_ROOT_RELATIVE,
        "revision": plan.P_REVISION_RELATIVE,
        "revision_sha256": plan.P_REVISION_SHA256,
    }
    hashes["terminal.json"] = receipts.write_json(root / "terminal.json", terminal)
    hashes["HANDOFF_FOR_ASTRA_REVIEW.md"] = receipts.write_text(
        root / "HANDOFF_FOR_ASTRA_REVIEW.md", _handoff_md(audit, hashes)
    )
    hashes["HANDOFF_FOR_ASTRA_REVIEW.md"] = receipts.write_text(
        root / "HANDOFF_FOR_ASTRA_REVIEW.md", _handoff_md(audit, hashes)
    )
    return hashes, hashes["terminal.json"]
