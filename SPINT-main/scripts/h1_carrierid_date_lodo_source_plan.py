#!/usr/bin/env python3
"""Receipt-only Phase-2 plan for H1 CarrierID five-date source LODO.

This file deliberately has no dataset loader, model, Trainer, CUDA, launcher,
checkpoint, or evaluator.  It accepts only a completed immutable Phase-1
receipt and turns it into a machine-checkable declaration of the ten future
source-training cells (H-S/H-C x five outer dates).  It does not authorize or
implement those future cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_source_preflight import PREFLIGHT_SCHEMA, PREFLIGHT_STATUS
from src.data.h1_carrierid_date_lodo_source import validate_source_bundle_manifest
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, sha256_file, write_immutable_json


PLAN_SCHEMA = "h1_carrierid_date_lodo_phase2_plan_only_v1"
PLAN_STATUS = "PASS_PHASE2_CELLS_DECLARED_NOT_IMPLEMENTED_NOT_LAUNCHED"
FOLD0_PREFLIGHT_SCHEMA = "h1_carrierid_h32_fold0_cpu_preflight_v1"


def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_immutable_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    _need(_immutable(resolved), f"receipt must be an immutable mode-0444 file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"receipt is invalid JSON: {resolved}") from error
    _need(isinstance(value, dict), "receipt must be a JSON object")
    return value


def validate_phase1_preflight(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Fail closed on fold-0 or incomplete/non-source-only input receipts."""

    if receipt.get("schema") == FOLD0_PREFLIGHT_SCHEMA:
        raise ValueError("fold-0 CarrierID preflight cannot masquerade as confirmatory date-LODO")
    _need(receipt.get("schema") == PREFLIGHT_SCHEMA, "unexpected source preflight schema")
    _need(receipt.get("status") == PREFLIGHT_STATUS, "source preflight did not pass its CPU-only phase")
    _need(tuple(receipt.get("confirmatory_dates", ())) == CONFIRMATORY_DATES,
          "source preflight must contain exactly the five confirmatory dates in canonical order")
    rows = receipt.get("date_bundles")
    _need(isinstance(rows, Mapping) and tuple(rows) == CONFIRMATORY_DATES,
          "source preflight date bundles are not canonical five-date LODO")
    scope = receipt.get("source_only_scope")
    _need(isinstance(scope, Mapping), "source preflight lacks source-only scope")
    _need(scope.get("target_recordings_opened_total") == 0 and scope.get("target_bytes_read_total") == 0,
          "source preflight records target access")
    _need(all(scope.get(field) is False for field in (
        "minival_opened_or_enumerated", "formal_heldout_opened_or_enumerated",
        "evalai_opened_or_enumerated", "cuda_constructed_or_launched",
        "trainer_constructed_or_launched", "checkpoint_created_or_loaded",
        "cce_residual_model_imported_or_constructed",
    )), "source preflight violates its source-only boundary")
    phase = receipt.get("phase_boundaries")
    _need(isinstance(phase, Mapping) and phase.get("phase2_training_wrappers_status") == "NOT_IMPLEMENTED",
          "Phase-1 receipt must not claim a training wrapper")
    _need(phase.get("gpu_launch_authorized") is False, "Phase-1 receipt must not authorize a GPU launch")
    code_closure = receipt.get("code_sha256")
    _need(isinstance(code_closure, Mapping), "source preflight lacks code closure")
    for required in (
        "scripts/h1_carrierid_date_lodo_source_preflight.py",
        "src/data/h1_carrierid_date_lodo_source.py",
        "src/data/h1_m4_cce_date_lodo.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/h1_m4_cce_contract.py",
    ):
        _need(_is_sha256(code_closure.get(required)), f"source preflight code closure is missing {required}")

    validated: dict[str, dict[str, Any]] = {}
    for date in CONFIRMATORY_DATES:
        row = rows[date]
        _need(isinstance(row, Mapping), f"{date}: malformed bundle summary")
        _need(row.get("outer_date") == date and date != "19250101", f"{date}: fold0/date mismatch")
        _need(int(row.get("source_session_count", -1)) == (10 if date == "19250108" else 11),
              f"{date}: source session count drift")
        manifest_path = Path(str(row.get("source_manifest_path", ""))).resolve()
        _need(_immutable(manifest_path), f"{date}: source manifest is not immutable")
        _need(sha256_file(manifest_path) == row.get("source_manifest_sha256"), f"{date}: source manifest SHA drift")
        manifest = _read_immutable_json(manifest_path)
        validate_source_bundle_manifest(manifest, outer_date=date)
        hs = row.get("shared_source_binding")
        _need(isinstance(hs, Mapping) and row.get("h_s_h_c_share_binding") is True,
              f"{date}: H-S/H-C source binding missing or unequal")
        _need(
            manifest["arms"]["H-S"]["shared_source_binding"] == manifest["arms"]["H-C"]["shared_source_binding"] == hs,
            f"{date}: source receipt and persisted manifest disagree about shared binding",
        )
        validated[date] = {
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": str(row["source_manifest_sha256"]),
            "shared_source_binding": dict(hs),
            "target_filenames_indexed_only": list(row["target_filenames_indexed_only"]),
        }
    return validated


def build_plan(receipt: Mapping[str, Any], *, preflight_path: Path) -> dict[str, Any]:
    date_rows = validate_phase1_preflight(receipt)
    cells = [
        {
            "cell_id": f"H1-{arm}-{date}-SLODO",
            "outer_date": date,
            "arm": arm,
            "source_binding": dict(date_rows[date]["shared_source_binding"]),
            "target_filenames_metadata_only": list(date_rows[date]["target_filenames_indexed_only"]),
            "training_wrapper_status": "NOT_IMPLEMENTED",
            "checkpoint_warm_start_forbidden": True,
            "launch_authorized": False,
        }
        for date in CONFIRMATORY_DATES
        for arm in ("H-S", "H-C")
    ]
    return {
        "schema": PLAN_SCHEMA,
        "status": PLAN_STATUS,
        "mode": "receipt_only_plan_no_data_no_target_no_model_no_gpu_no_launch",
        "phase1_preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": sha256_file(preflight_path),
            "schema": receipt["schema"],
        },
        "code_closure": {
            "phase1_preflight_code_sha256": dict(receipt["code_sha256"]),
            "phase2_plan_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "confirmatory_dates": list(CONFIRMATORY_DATES),
        "phase2_cells": cells,
        "cell_count": len(cells),
        "global_guards": {
            "target_content_opened": False,
            "cuda_constructed_or_launched": False,
            "trainer_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "target_evaluator_implemented": False,
            "model_or_training_wrapper_implemented": False,
            "fold0_schema_forbidden": FOLD0_PREFLIGHT_SCHEMA,
        },
    }


def run(*, preflight: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase-2 plan: {output}")
    receipt = _read_immutable_json(preflight)
    plan = build_plan(receipt, preflight_path=preflight)
    write_immutable_json(output, plan)
    _need(_immutable(output), "Phase-2 plan was not written immutable mode 0444")
    return {**plan, "output": str(output.resolve()), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(preflight=args.preflight, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
