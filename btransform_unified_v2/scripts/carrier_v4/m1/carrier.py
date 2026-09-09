#!/usr/bin/env python3
"""Exact M1 carrier-v4 adapter over the sealed source3 rSyn3 common-core fit.

No source4/aligned-NMF fit is present here.  The adapter imports the sealed
26/27/28 rSyn3 scale, NNMF dictionary, and per-column normalizer into the
shared core, then deploys it through the original M10 readers for all four
training tags and all three public held-out-calibration tags.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parents[2]
WS = V2.parent
CORE = HERE.parent / "common_estimator.py"
V1 = WS / "btransform_unified_v1"
for entry in (V2 / "src", V1 / "src", V1 / "scripts", HERE.parent, WS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from common_estimator import deploy, import_frozen_source_fit, load_fit, save_fit  # noqa: E402

SCHEMA = "m1_carrier_v4_sealed_source3_exact_pack_v1"
METHOD = "sealed_source3_rsyn3_imported_commoncore_exact_m10"
TASK = "m1"
DIM, M10 = 4, 10
SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
SEALED_SOURCE_SESSIONS = ("ses-20120926", "ses-20120927", "ses-20120928")
HO_SESSIONS = ("20121004", "20121017", "20121024")
ALL_SESSIONS = SOURCE_SESSIONS + HO_SESSIONS


class CarrierError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _typed_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def _atomic_json(path: Path, body: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _sealed_fit() -> tuple[Any, dict[str, Any], Any]:
    """Import only persisted source3 fit arrays; never invoke NMF fitting."""
    from tfpd_exploration.src.m1_optimized_v2 import bank as old_bank, plan as old_plan
    loaded = old_bank.load()
    receipt = loaded["receipt"]
    _need(tuple(receipt.get("source_sessions", ())) == SEALED_SOURCE_SESSIONS, "sealed fit source roster drift")
    _need(receipt.get("target_query_values_read") is False, "sealed fit reports target query access")
    blob = np.load(old_plan.BANK_NPZ, allow_pickle=False)
    provenance = {
        "origin": "sealed_m1_optimized_v2_source3_rsyn3",
        "source_sessions": list(SEALED_SOURCE_SESSIONS),
        "sealed_npz": str(old_plan.BANK_NPZ), "sealed_npz_sha256": _sha(old_plan.BANK_NPZ),
        "sealed_receipt": str(old_plan.BANK_RECEIPT), "sealed_receipt_sha256": _sha(old_plan.BANK_RECEIPT),
        "sealed_receipt_body": receipt,
        "arrays": {name: _typed_sha(np.asarray(blob[name])) for name in ("scale", "d0", "normalizer_mean", "normalizer_scale")},
    }
    fit = import_frozen_source_fit(TASK, SEALED_SOURCE_SESSIONS, np.asarray(blob["scale"]), np.asarray(blob["d0"]),
                                   np.asarray(blob["normalizer_mean"]), np.asarray(blob["normalizer_scale"]), provenance)
    return fit, provenance, old_plan


def _source_arrays(session: str) -> tuple[np.ndarray, np.ndarray, Path, str]:
    """Old fold-local reader: full EMG for source role, M10 mask for deployment."""
    from btransform_unified_v1 import m1_projadd as mp
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data, stage0
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan
    from tfpd_exploration.src.m1_optimized_v2 import plan as old_plan
    path = parent_data.require_source_path(WS / parent_plan.SOURCE_RELATIVE[session])
    _need(parent_data.file_sha256(path) == parent_plan.SOURCE_FILE_SHA256[session], f"source body SHA drift: {session}")
    role = "target" if session == mp.M1_OUTER_SESSION else "source"
    record = fold_data.load_fold_session(path, role=role)
    behavior, rates, identifiers = stage0._mask_budget(record, old_plan.SUPPORT_TRIALS)
    _need(np.array_equal(record.emg_trial_ids[record.emg_trial_ids < M10], identifiers), f"{session}: M10 row IDs drift")
    return np.ascontiguousarray(behavior, dtype=np.float64), np.ascontiguousarray(rates, dtype=np.float64), path, role


def _ho_arrays(session: str) -> tuple[np.ndarray, np.ndarray, Path]:
    """Sanctioned public held-out calibration reader, M10 support only."""
    import m1_projadd_depth2_series as legacy
    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
    path = legacy._heldout_calib_path(session)
    record = rsyn3_bank.load_public_calib_support(path, support_trials=M10)
    _need(record.emg.shape[0] == record.rates.shape[0] and record.rates.shape[1] == 64, f"{session}: M10 geometry drift")
    return np.ascontiguousarray(record.emg, dtype=np.float64), np.ascontiguousarray(record.rates, dtype=np.float64), path


def _actual_baseline() -> dict[str, np.ndarray]:
    """Read actual old T through its public deployment entry points."""
    import m1_projadd_depth2_series as legacy
    from btransform_unified_v1 import m1_projadd as mp
    values: dict[str, np.ndarray] = {**mp.load_source_carriers()}
    outer, _meta = mp.encode_outer_carrier(); values[mp.M1_OUTER_SESSION] = outer
    for session in HO_SESSIONS:
        values[session], _meta = legacy.encode_heldout_calib_carrier(session)
    _need(set(values) == set(ALL_SESSIONS), "actual baseline roster drift")
    return {session: np.ascontiguousarray(values[session], dtype=np.float32) for session in ALL_SESSIONS}


def build(dest: Path) -> tuple[Path, Path]:
    """Persist one source3-imported fit and a byte-exact seven-tag deployment pack."""
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite {dest}")
    dest.mkdir(parents=True)
    fit, sealed_provenance, old_plan = _sealed_fit()
    fit_json, fit_npz = (Path(item).resolve() for item in save_fit(fit, dest / "fit"))
    _need(fit_json.is_file() and fit_npz.is_file(), "shared core did not persist imported fit")
    persisted = load_fit(fit_json)
    _need(tuple(persisted.source_sessions) == SEALED_SOURCE_SESSIONS, "persisted fit source roster drift")
    baseline = _actual_baseline()
    carriers: dict[str, np.ndarray] = {}; deployment: dict[str, Any] = {}
    for session in ALL_SESSIONS:
        if session in SOURCE_SESSIONS:
            behavior, rates, path, role = _source_arrays(session)
            reader = {"kind": "old_fold_reader_full_source_EMG_then_M10_mask", "role": role}
        else:
            behavior, rates, path = _ho_arrays(session)
            reader = {"kind": "old_public_heldout_calib_reader_M10_only"}
        carrier, raw, diagnostics = deploy(persisted, behavior, rates)
        value = np.ascontiguousarray(np.asarray(carrier, dtype=np.float32))
        raw_value = np.ascontiguousarray(np.asarray(raw, dtype=np.float64))
        _need(value.shape == (64, DIM) and raw_value.shape == (64, DIM) and np.isfinite(value).all() and np.isfinite(raw_value).all(), f"{session}: core deployment geometry/nonfinite drift")
        if not np.array_equal(value, baseline[session]):
            raise CarrierError(f"{session}: imported-core carrier is not byte-identical to actual baseline T")
        carriers[session] = value
        deployment[session] = {**reader, "path": str(path.resolve()), "path_sha256": _sha(path),
            "support_trials": [0, M10], "behavior_typed_sha256": _typed_sha(behavior), "rates_typed_sha256": _typed_sha(rates),
            "raw_typed_sha256": _typed_sha(raw_value), "carrier_typed_sha256": _typed_sha(value), "baseline_typed_sha256": _typed_sha(baseline[session]),
            "baseline_byte_equal": True, "diagnostics": diagnostics}
    pack = dest / "carrier_pack.npz"; _atomic_npz(pack, **carriers)
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "status": "PASSED", "task": TASK, "method": METHOD,
        "fit_source_sessions": list(SEALED_SOURCE_SESSIONS), "training_source_sessions": list(SOURCE_SESSIONS), "ho_sessions": list(HO_SESSIONS),
        "fit": str(fit_npz), "fit_sha256": _sha(fit_npz), "fit_receipt": str(fit_json), "fit_receipt_sha256": _sha(fit_json),
        "carrier_pack": str(pack), "carrier_pack_sha256": _sha(pack), "core_code": str(CORE.resolve()), "core_code_sha256": _sha(CORE),
        "adapter_code": str(Path(__file__).resolve()), "adapter_code_sha256": _sha(Path(__file__)), "sealed_source3": sealed_provenance,
        "deployment": deployment, "actual_baseline_T": {session: {"typed_sha256": _typed_sha(baseline[session]), "byte_equal": True} for session in ALL_SESSIONS},
        "carrier_arrays": {session: {"raw_sha256": _raw_sha(value), "typed_sha256": _typed_sha(value), "shape": [64, DIM], "dtype": value.dtype.str} for session, value in carriers.items()},
        "official_test_included": False, "query_labels_read": False, "target_calibration_values_read": True,
    }
    receipt_path = dest / "carrier_pack.json"; _atomic_json(receipt_path, receipt)
    return pack, receipt_path


def load_carrier_pack(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only the source3-imported exact-M1 schema and bind its core fit."""
    path = Path(path).resolve(); receipt_path = path.with_suffix(".json")
    _need(path.is_file() and receipt_path.is_file(), "carrier pack and adjacent receipt are required")
    body = json.loads(receipt_path.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == SCHEMA and body.get("status") == "PASSED", "exact M1 v4 receipt identity drift")
    _need(body.get("task") == TASK and body.get("method") == METHOD, "exact M1 v4 method/task drift")
    _need(tuple(body.get("fit_source_sessions", ())) == SEALED_SOURCE_SESSIONS and tuple(body.get("training_source_sessions", ())) == SOURCE_SESSIONS and tuple(body.get("ho_sessions", ())) == HO_SESSIONS, "M1 source roster drift")
    _need(body.get("official_test_included") is False and body.get("query_labels_read") is False and body.get("target_calibration_values_read") is True, "public calibration contract drift")
    _need(body.get("carrier_pack") == str(path) and body.get("carrier_pack_sha256") == _sha(path), "carrier pack binding drift")
    _need(body.get("core_code") == str(CORE.resolve()) and body.get("core_code_sha256") == _sha(CORE), "shared core binding drift")
    _need(body.get("adapter_code") == str(Path(__file__).resolve()) and body.get("adapter_code_sha256") == _sha(Path(__file__)), "adapter binding drift")
    fit_path, fit_receipt = Path(body.get("fit", "")), Path(body.get("fit_receipt", ""))
    _need(fit_path.is_file() and body.get("fit_sha256") == _sha(fit_path) and fit_receipt.is_file() and body.get("fit_receipt_sha256") == _sha(fit_receipt), "imported fit binding drift")
    fit = load_fit(fit_receipt); _need(tuple(fit.source_sessions) == SEALED_SOURCE_SESSIONS and fit.metadata.get("dictionary_origin") == "imported_frozen_not_refit", "loaded fit is not sealed-source3 imported")
    declared = body.get("carrier_arrays"); _need(isinstance(declared, dict) and set(declared) == set(ALL_SESSIONS), "carrier array roster drift")
    with np.load(path, allow_pickle=False) as archive:
        _need(set(archive.files) == set(ALL_SESSIONS), "carrier NPZ key drift")
        mapping = {session: np.ascontiguousarray(archive[session]) for session in ALL_SESSIONS}
    for session, value in mapping.items():
        expected = {"raw_sha256": _raw_sha(value), "typed_sha256": _typed_sha(value), "shape": [64, DIM], "dtype": value.dtype.str}
        _need(value.dtype == np.float32 and value.shape == (64, DIM) and np.isfinite(value).all(), f"{session}: invalid carrier")
        _need(declared[session] == expected, f"{session}: carrier hash drift")
    return mapping, body


def replace_bank_carrier(bank: TaskBank, carrier: np.ndarray) -> TaskBank:
    """Return an immutable bank with only exact-v4 T and metadata replaced."""
    _need(isinstance(bank, TaskBank), "replace_bank_carrier requires TaskBank")
    value = np.ascontiguousarray(carrier, dtype=np.float32)
    _need(value.shape == (bank.E0.shape[0], DIM) and np.isfinite(value).all(), "bank/v4 carrier geometry/nonfinite drift")
    meta = dict(bank.calibration_meta)
    meta.update({"carrier_sha256": array_sha256(value), "carrier_rawdigest": _raw_sha(value), "carrier_method": METHOD, "carrier_schema": SCHEMA})
    return dataclasses.replace(bank, carrier=value, calibration_meta=meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fresh exact M1 v4 carrier pack")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    pack, receipt = build(args.dest)
    print(json.dumps({"carrier_pack": str(pack), "receipt": str(receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
