#!/usr/bin/env python3
"""CPU-only, fail-closed preflight for C1 teacher-domain ablation.

This is a receipt writer, not a launcher.  Root's gate is that C1 GPU work
runs only after a source-only teacher compatibility receipt.  This preflight
therefore refuses READY unless that receipt exists, is source-only, and
records interface constructibility.  It still does not authorize GPU: a
trained CO-native teacher has not been bound.

Never opens NWB, sub-M, sealed test sessions, or a GPU.  Never trains.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for path in (REPO_ROOT, SUA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mc_maze import c1_teacher_domain_ablation as core
from mc_maze.gpu_contract_common import SEALED_FORMAL_TEST_SESSIONS


def _blocker(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def build_receipt(
    *,
    result_root: Path,
    compatibility_path: Path,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    if not core.CONTRACT_PATH.is_file():
        blockers.append(_blocker("MISSING_CONTRACT", str(core.CONTRACT_PATH)))
    try:
        core.validate_config()
    except Exception as exc:
        blockers.append(_blocker("CONFIG_DRIFT", str(exc)))

    for path, label in (
        (core.MANIFEST_PATH, "manifest"),
        (core.MC_MAZE_TEACHER_PATH, "mc_maze_teacher"),
        (core.CONFIG_PATH, "config"),
        (core.CONTRACT_PATH, "contract"),
    ):
        if not path.is_file():
            blockers.append(_blocker("MISSING_BINDING", f"{label}: {path}"))

    manifest_sha = None
    teacher_sha = None
    source_train: list[str] = []
    if core.MANIFEST_PATH.is_file():
        try:
            manifest = core.load_strict_manifest()
            source_train = list(manifest["train"])
            manifest_sha = core.sha256_file(core.MANIFEST_PATH)
            if manifest_sha != core.EXPECTED_MANIFEST_SHA256:
                blockers.append(_blocker("MANIFEST_SHA_DRIFT", manifest_sha))
            core.refuse_sealed_sessions(source_train, label="source-train")
            core.refuse_sealed_sessions(manifest["val"], label="validation")
        except Exception as exc:
            blockers.append(_blocker("MANIFEST_UNREADABLE", str(exc)))
    if core.MC_MAZE_TEACHER_PATH.is_file():
        teacher_sha = core.sha256_file(core.MC_MAZE_TEACHER_PATH)
        if teacher_sha != core.EXPECTED_TEACHER_SHA256:
            blockers.append(_blocker("MC_MAZE_TEACHER_SHA_DRIFT", teacher_sha))

    compatibility: dict[str, Any] | None = None
    compatibility_sha = None
    if not compatibility_path.is_file():
        blockers.append(_blocker(
            "MISSING_SOURCE_ONLY_COMPATIBILITY_RECEIPT",
            f"C1 GPU is gated on {compatibility_path}",
        ))
    else:
        try:
            compatibility = core.load_json_object(compatibility_path)
            compatibility_sha = core.sha256_file(compatibility_path)
            if compatibility.get("screen_id") != core.SCREEN_ID:
                blockers.append(_blocker("COMPATIBILITY_SCREEN_DRIFT", str(compatibility.get("screen_id"))))
            if compatibility.get("source_only") is not True:
                blockers.append(_blocker("COMPATIBILITY_NOT_SOURCE_ONLY", "source_only must be true"))
            if compatibility.get("target_or_sealed_data_opened") is not False:
                blockers.append(_blocker("COMPATIBILITY_TOUCHED_TARGET_OR_SEALED", "receipt opened forbidden data"))
            if compatibility.get("operations", {}).get("nwb_opened") is True:
                blockers.append(_blocker("COMPATIBILITY_OPENED_NWB", "source-only receipt opened NWB"))
            if compatibility.get("interface_constructible") is not True:
                blockers.append(_blocker("INTERFACE_NOT_CONSTRUCTIBLE", str(compatibility.get("status"))))
            if compatibility.get("hypothesis_status") != "plausible_contributor_not_isolated_cause":
                blockers.append(_blocker("HYPOTHESIS_OVERCLAIM", str(compatibility.get("hypothesis_status"))))
            if compatibility.get("trained_co_native_teacher_exists") is True:
                # A future trained teacher must be bound by a reviewed successor;
                # this scaffolding preflight never treats one as present.
                blockers.append(_blocker("UNEXPECTED_TRAINED_TEACHER_CLAIM", "scaffolding must not mint a trained teacher"))
        except Exception as exc:
            blockers.append(_blocker("COMPATIBILITY_UNREADABLE", str(exc)))

    result_root = result_root.expanduser().resolve()
    if result_root.exists():
        for cell in core.source_cells():
            for domain in core.DOMAINS:
                out = result_root / core.domain_receipt_name(
                    cell["teacher_domain"], cell["carrier"], cell["seed"], domain
                )
                if out.exists():
                    blockers.append(_blocker("NONEMPTY_OUTPUT_ROOT", str(out)))
        agg = result_root / "aggregate.json"
        if agg.exists():
            blockers.append(_blocker("NONEMPTY_OUTPUT_ROOT", str(agg)))

    gpu_ready = False
    if not blockers and compatibility is not None and compatibility.get("trained_co_native_teacher_exists") is True:
        gpu_ready = False  # still requires a separate root GO; never granted here
    status = (
        "CPU_PREFLIGHT_PASSED_AWAITING_TRAINED_CO_NATIVE_TEACHER_AND_ROOT_GO"
        if not blockers
        else "STOP_C1_PREFLIGHT_BLOCKERS"
    )
    return {
        "schema_version": 1,
        "screen_id": core.SCREEN_ID,
        "kind": "c1_teacher_domain_ablation_official_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "gpu_authorized": False,
        "gpu_ready": gpu_ready,
        "training_authorized": False,
        "source_only": True,
        "nwb_opened": False,
        "target_subject_m_opened": False,
        "sealed_formal_test_sessions_opened": False,
        "sealed_formal_test_session_names_only": sorted(SEALED_FORMAL_TEST_SESSIONS),
        "add_site": core.ADD_SITE,
        "sampling": core.SAMPLING,
        "hidden_space_adapter": False,
        "expected_gpu_source_cells": core.SOURCE_CELL_COUNT,
        "expected_domain_receipts": core.DOMAIN_RECEIPT_COUNT,
        "source_cells": list(core.source_cells()),
        "compatibility_receipt_path": str(compatibility_path),
        "compatibility_receipt_sha256": compatibility_sha,
        "compatibility_interface_constructible": None if compatibility is None else compatibility.get("interface_constructible"),
        "trained_co_native_teacher_exists": False,
        "contract_path": str(core.CONTRACT_PATH),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH) if core.CONTRACT_PATH.is_file() else None,
        "config_sha256": core.sha256_file(core.CONFIG_PATH) if core.CONFIG_PATH.is_file() else None,
        "manifest_sha256": manifest_sha,
        "mc_maze_teacher_sha256": teacher_sha,
        "source_train_session_names_only": source_train,
        "implementation_blockers": blockers,
        "hypothesis_status": "plausible_contributor_not_isolated_cause",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=core.RESULT_ROOT)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--compatibility-receipt", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    receipt_path = (
        args.receipt.expanduser().resolve()
        if args.receipt is not None
        else core.official_preflight_path(result_root)
    )
    compatibility_path = (
        args.compatibility_receipt.expanduser().resolve()
        if args.compatibility_receipt is not None
        else core.compatibility_receipt_path(result_root)
    )
    receipt = build_receipt(result_root=result_root, compatibility_path=compatibility_path)
    try:
        body, sidecar, digest = core.write_immutable_json(receipt_path, receipt)
    except (FileExistsError, OSError, core.C1ContractError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(
        core.pretty_json_bytes(
            {
                "status": receipt["status"],
                "receipt": str(body),
                "sidecar": str(sidecar),
                "receipt_sha256": digest,
                "gpu_authorized": False,
                "blocker_count": len(receipt["implementation_blockers"]),
            }
        ).decode("utf-8"),
        end="",
    )
    return 0 if not receipt["implementation_blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
