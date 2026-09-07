#!/usr/bin/env python3
"""Prepare five immutable, source-only H1 CarrierID date-LODO bundles.

Phase 1 has a deliberately narrow responsibility: it reconstructs the frozen
M=4 carrier from the *non-outer-date* public held-in sessions, writes one
source RMS normalizer and one fixed schedule, then binds that exact trio to
both future H-S and H-C arms.  It is not a trainer, model preflight, GPU
launcher, target evaluator, checkpoint writer, or selection procedure.

Outer-date NWBs are filename-indexed solely to prove the partition.  Their
bytes are never opened by this program.  Phase-2 training wrappers are
intentionally not implemented here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_date_lodo_source import (
    SOURCE_MANIFEST_SCHEMA,
    prepare_date_source_bundle,
    validate_source_bundle_manifest,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    immutable_mode_0444,
    sha256_file,
    write_immutable_json,
)


PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_source_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_NOT_LAUNCHED"
FOLD0_PREFLIGHT_SCHEMA = "h1_carrierid_h32_fold0_cpu_preflight_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _code_hashes() -> dict[str, str]:
    """Record the narrow Phase-1 data/code closure, not future wrappers."""

    relative_paths = (
        "scripts/h1_carrierid_date_lodo_source_preflight.py",
        "src/data/h1_carrierid_date_lodo_source.py",
        "src/data/h1_m4_cce_date_lodo.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/h1_m4_cce_contract.py",
    )
    return {relative: sha256_file(ROOT / relative) for relative in relative_paths}


def _validate_bundle_for_receipt(bundle: Mapping[str, Any], date: str) -> dict[str, Any]:
    validate_source_bundle_manifest(bundle, outer_date=date)
    target = bundle["target_filename_index"]
    scope = bundle["source_only_scope"]
    _require(bundle["schema"] == SOURCE_MANIFEST_SCHEMA, f"{date}: source manifest schema drift")
    _require(target["target_recordings_opened"] == 0 and target["target_bytes_read"] == 0,
             f"{date}: target bytes were opened")
    _require(all(scope[field] is False for field in (
        "minival_opened_or_enumerated", "formal_heldout_opened_or_enumerated",
        "evalai_opened_or_enumerated", "cuda_constructed_or_launched",
        "trainer_constructed_or_launched", "checkpoint_created_or_loaded",
        "cce_residual_model_imported_or_constructed",
    )), f"{date}: source-only scope drift")
    path = Path(bundle["manifest_path"])
    _require(immutable_mode_0444(path), f"{date}: source bundle manifest is not immutable")
    _require(sha256_file(path) == bundle["manifest_sha256"], f"{date}: source bundle manifest SHA drift")
    return {
        "outer_date": date,
        "source_session_count": bundle["source_session_count"],
        "source_sessions": list(bundle["source_sessions"]),
        "target_filenames_indexed_only": list(target["target_filenames_indexed_only"]),
        "source_manifest_path": str(path.resolve()),
        "source_manifest_sha256": bundle["manifest_sha256"],
        "shared_source_binding": dict(bundle["arms"]["H-S"]["shared_source_binding"]),
        "h_s_h_c_share_binding": (
            bundle["arms"]["H-S"]["shared_source_binding"] == bundle["arms"]["H-C"]["shared_source_binding"]
        ),
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }


def run(
    *,
    data_dir: Path,
    raw_receipt: Path,
    eb_receipt: Path,
    cache_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Execute Phase 1 once, fail-closed on any partial prior artifact."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("date-LODO source preflight requires CUDA_VISIBLE_DEVICES to be unset")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source preflight receipt: {output}")
    if cache_root.exists():
        raise FileExistsError(f"source bundle cache root must be fresh: {cache_root}")
    _require(raw_receipt.is_file() and eb_receipt.is_file(), "raw/EB source receipts must exist")
    _require(immutable_mode_0444(raw_receipt) and immutable_mode_0444(eb_receipt),
             "raw/EB source receipts must be immutable mode 0444")

    bundles: dict[str, dict[str, Any]] = {}
    for date in CONFIRMATORY_DATES:
        bundle = prepare_date_source_bundle(
            data_dir=data_dir,
            outer_date=date,
            raw_receipt_path=raw_receipt,
            eb_receipt_path=eb_receipt,
            cache_root=cache_root,
        )
        bundles[date] = _validate_bundle_for_receipt(bundle, date)
    _require(tuple(bundles) == CONFIRMATORY_DATES, "five-date source preflight date order drift")
    _require(set(bundles) == set(CONFIRMATORY_DATES), "source preflight misses or adds an outer date")

    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "mode": "cpu_only_source_data_preparation_no_target_open_no_gpu_no_training",
        "confirmatory_dates": list(CONFIRMATORY_DATES),
        "date_bundles": bundles,
        "inputs": {
            "data_dir": str(data_dir.resolve()),
            "raw_receipt_path": str(raw_receipt.resolve()),
            "raw_receipt_sha256": sha256_file(raw_receipt),
            "eb_receipt_path": str(eb_receipt.resolve()),
            "eb_receipt_sha256": sha256_file(eb_receipt),
        },
        "phase_boundaries": {
            "phase1_complete": "shared H-S/H-C source cache, normalizer, and schedule only",
            "phase2_training_wrappers_status": "NOT_IMPLEMENTED",
            "gpu_launch_authorized": False,
            "target_evaluator_implemented": False,
            "checkpoint_schema_implemented": False,
            "fold0_preflight_schema_forbidden": FOLD0_PREFLIGHT_SCHEMA,
        },
        "source_only_scope": {
            "target_recordings_opened_total": 0,
            "target_bytes_read_total": 0,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "trainer_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "cce_residual_model_imported_or_constructed": False,
        },
        "code_sha256": _code_hashes(),
    }
    write_immutable_json(output, receipt)
    if not immutable_mode_0444(output):
        raise RuntimeError("source preflight receipt publication was not immutable")
    return {**receipt, "output": str(output.resolve()), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument(
        "--raw-receipt", type=Path,
        default=WORKSPACE / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json",
    )
    parser.add_argument(
        "--eb-receipt", type=Path,
        default=WORKSPACE / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json",
    )
    parser.add_argument(
        "--cache-root", type=Path,
        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/source_bundles_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json",
    )
    args = parser.parse_args()
    # The default above is intentionally resolved here rather than used in a
    # hidden invocation; a caller must explicitly run this CPU-only script.
    result = run(
        data_dir=args.data_dir, raw_receipt=args.raw_receipt, eb_receipt=args.eb_receipt,
        cache_root=args.cache_root, output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
