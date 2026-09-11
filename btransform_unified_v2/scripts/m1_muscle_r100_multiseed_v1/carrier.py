#!/usr/bin/env python3
"""Frozen-muscle facade plus one source-only mean-rate fourth-coordinate candidate.

The baseline path delegates to the frozen R100 loader.  The candidate leaves
columns 0..2 byte-identical and replaces only column 3 with session-M10 unit
arithmetic mean rates, standardized on source4 and scaled to the baseline
source4 normalized fourth-column standard deviation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
FROZEN_PATH = SCRIPTS / "m1_muscle_r100_v1" / "carrier.py"
CANDIDATE_SCHEMA = "m1_muscle_r100_multiseed_carrier_pack_v1"
FIT_SCHEMA = "m1_muscle_r100_multiseed_mean_rate4_fit_v1"
BASELINE_VARIANT = "muscle_response16_svd4/global_rms"
CANDIDATE_VARIANT = "muscle_response_svd3_mean_rate4_matched_scale"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _typed_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def _atomic_json(path: Path, body: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _frozen() -> Any:
    """Load the frozen producer under a private name, never by ``import carrier``."""
    name = "_frozen_m1_muscle_r100_carrier"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    _need(FROZEN_PATH.is_file(), f"missing frozen producer: {FROZEN_PATH}")
    spec = importlib.util.spec_from_file_location(name, FROZEN_PATH)
    _need(spec is not None and spec.loader is not None, "cannot construct frozen producer import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_baseline_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Add a runtime-only variant label without changing the frozen JSON."""
    body = dict(receipt)
    existing = body.get("carrier_variant")
    _need(existing in (None, BASELINE_VARIANT), "frozen receipt carrier_variant drift")
    body["carrier_variant"] = BASELINE_VARIANT
    return body


def _array_metadata(carriers: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {session: {"raw_sha256": _raw_sha(value), "typed_sha256": _typed_sha(value),
                      "shape": [64, 4], "dtype": value.dtype.str}
            for session, value in carriers.items()}


def _row_descriptor(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {"shape": list(array.shape), "dtype": array.dtype.str, "typed_sha256": _typed_sha(array)}


def _m10_row_identity(support: Any, emg: np.ndarray, rates: np.ndarray, frozen: Any) -> dict[str, Any]:
    """Describe the exact aligned M10 rows already read for the candidate."""
    emask = np.asarray(support.emg_trial_ids) < frozen.M10
    rmask = np.asarray(support.rate_trial_ids) < frozen.M10
    emg_ids = np.asarray(support.emg_trial_ids)[emask]
    rate_ids = np.asarray(support.rate_trial_ids)[rmask]
    _need(np.array_equal(emg_ids, rate_ids), "M10 row identity trial IDs drift")
    emg_global = np.asarray(support.emg_global_time_index)[emask]
    rate_global = np.asarray(support.rate_global_time_index)[rmask]
    _need(emg.shape[0] == rates.shape[0] == len(emg_ids) == len(emg_global) == len(rate_global), "M10 row identity length drift")
    return {"emg_rows": _row_descriptor(emg), "rate_rows": _row_descriptor(rates),
            "emg_trial_ids": _row_descriptor(emg_ids), "rate_trial_ids": _row_descriptor(rate_ids),
            "emg_global_time_index": _row_descriptor(emg_global), "rate_global_time_index": _row_descriptor(rate_global),
            "trial_ids_exact": True}


def _validate_m10_row_identity(value: Any, frozen: Any) -> None:
    _need(isinstance(value, Mapping), "candidate M10 row identity missing")
    required = ("emg_rows", "rate_rows", "emg_trial_ids", "rate_trial_ids", "emg_global_time_index", "rate_global_time_index")
    _need(all(isinstance(value.get(name), Mapping) for name in required) and value.get("trial_ids_exact") is True,
          "candidate M10 row identity schema drift")
    for name in required:
        row = value[name]
        shape, dtype, digest = row.get("shape"), row.get("dtype"), row.get("typed_sha256")
        _need(isinstance(shape, list) and shape and isinstance(dtype, str) and isinstance(digest, str) and len(digest) == 64,
              f"candidate M10 row identity invalid: {name}")
    _need(value["emg_rows"]["shape"][1:] == [16] and value["rate_rows"]["shape"][1:] == [64],
          "candidate M10 EMG/rate geometry identity drift")
    row_count = value["emg_rows"]["shape"][0]
    _need(isinstance(row_count, int) and row_count > 0 and value["rate_rows"]["shape"][0] == row_count,
          "candidate M10 EMG/rate row count identity drift")
    for name in ("emg_trial_ids", "rate_trial_ids", "emg_global_time_index", "rate_global_time_index"):
        _need(value[name]["shape"] == [row_count], f"candidate M10 identity vector length drift: {name}")


def _source_stack(carriers: Mapping[str, np.ndarray], frozen: Any) -> np.ndarray:
    return np.concatenate([np.asarray(carriers[session], dtype=np.float64) for session in frozen.SOURCE_SESSIONS], axis=0)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _verify_parent_fit(path: Path, expected_sha: str) -> dict[str, Any]:
    """Open only the frozen fit named by the parent receipt; never refit it."""
    _need(path.is_file() and _sha(path) == expected_sha, "parent fit SHA drift")
    with np.load(path, allow_pickle=False) as archive:
        required = {"muscle_rms", "svd_components", "normalizer_mean", "normalizer_scale"}
        _need(required.issubset(archive.files), "parent fit arrays drift")
        shapes = {name: list(np.asarray(archive[name]).shape) for name in required}
        _need(shapes["muscle_rms"] == [16] and shapes["svd_components"] == [4, 16] and
              shapes["normalizer_mean"] == [4] and shapes["normalizer_scale"] == [4], "parent fit geometry drift")
    return {"path": str(path), "sha256": expected_sha, "arrays": shapes,
            "reused_svd_columns": [0, 1, 2], "refit": False}


def _candidate_fit_body(path: Path, expected_sha: str) -> dict[str, Any]:
    _need(path.is_file() and _sha(path) == expected_sha, "candidate fit SHA drift")
    body = _read_json(path.with_suffix(".json"))
    _need(body.get("schema") == FIT_SCHEMA and body.get("status") == "PASSED", "candidate fit receipt schema/status drift")
    _need(body.get("fit_npz") == str(path) and body.get("fit_npz_sha256") == expected_sha, "candidate fit receipt path/SHA drift")
    return body


def _candidate_loader(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frozen = _frozen()
    path = path.resolve(); receipt_path = path.with_suffix(".json")
    _need(path.is_file() and receipt_path.is_file(), "candidate pack and receipt are required")
    body = _read_json(receipt_path)
    _need(body.get("schema") == CANDIDATE_SCHEMA and body.get("status") == "PASSED", "candidate receipt schema/status drift")
    _need(body.get("carrier_variant") == CANDIDATE_VARIANT, "candidate variant drift")
    _need(tuple(body.get("source_sessions", ())) == tuple(frozen.SOURCE_SESSIONS), "candidate source roster drift")
    _need(tuple(body.get("ho_sessions", ())) == tuple(frozen.HO_SESSIONS), "candidate HO roster drift")
    _need(body.get("carrier_pack") == str(path) and body.get("carrier_pack_sha256") == _sha(path), "candidate pack binding drift")
    _need(body.get("facade_code") == str(Path(__file__).resolve()) and body.get("facade_code_sha256") == _sha(Path(__file__).resolve()),
          "candidate facade-code binding drift")
    _need(body.get("frozen_producer") == str(FROZEN_PATH.resolve()) and body.get("frozen_producer_sha256") == _sha(FROZEN_PATH),
          "candidate frozen-producer binding drift")
    parent_path = Path(body.get("parent_carrier_pack", "")).resolve()
    _need(parent_path.is_file() and body.get("parent_carrier_pack_sha256") == _sha(parent_path), "candidate parent pack binding drift")
    parent, parent_receipt = _baseline_loader(parent_path)
    _need(body.get("parent_fit_sha256") == parent_receipt.get("fit_sha256"), "candidate parent fit binding drift")
    expected_parent_fit_contract = _verify_parent_fit(Path(parent_receipt["fit"]).resolve(), str(parent_receipt["fit_sha256"]))
    _need(body.get("parent_fit_contract") == expected_parent_fit_contract, "candidate parent-fit contract drift")
    fit_path = Path(body.get("fit", "")).resolve()
    fit_receipt_path = Path(body.get("fit_receipt", "")).resolve()
    _need(fit_receipt_path == fit_path.with_suffix(".json") and fit_receipt_path.is_file() and
          body.get("fit_receipt_sha256") == _sha(fit_receipt_path), "candidate fit receipt SHA/path drift")
    fit_body = _candidate_fit_body(fit_path, str(body.get("fit_sha256", "")))
    _need(fit_body.get("parent_carrier_pack_sha256") == body.get("parent_carrier_pack_sha256") and
          fit_body.get("parent_fit_sha256") == body.get("parent_fit_sha256") and
          fit_body.get("facade_code_sha256") == body.get("facade_code_sha256") and
          fit_body.get("parent_fit_contract") == body.get("parent_fit_contract"), "candidate pack/fit receipt disagreement")
    _need(body.get("first_three_columns_byte_equal") is True and fit_body.get("first_three_columns_byte_equal") is True,
          "candidate first-three assertion absent")
    _need(body.get("rate_source", {}).get("statistic") == "arithmetic_mean_rate_hz" and
          body.get("rate_source", {}).get("support_trials") == [0, 10], "candidate rate-support contract drift")
    _need(body.get("source_emg_rms_support_trials") == [0, 310], "candidate source EMG support contract drift")
    identities = body.get("m10_row_identity")
    _need(isinstance(identities, Mapping) and set(identities) == set(frozen.ALL_SESSIONS), "candidate M10 identity roster drift")
    for session in frozen.ALL_SESSIONS:
        _validate_m10_row_identity(identities[session], frozen)
    _need(fit_body.get("m10_row_identity") == identities, "candidate fit/pack M10 identity disagreement")
    declared = body.get("carrier_arrays")
    _need(isinstance(declared, dict) and set(declared) == set(frozen.ALL_SESSIONS), "candidate array receipt roster drift")
    with np.load(path, allow_pickle=False) as archive:
        _need(set(archive.files) == set(frozen.ALL_SESSIONS), "candidate NPZ key roster drift")
        carriers = {session: np.ascontiguousarray(archive[session]) for session in frozen.ALL_SESSIONS}
    for session, value in carriers.items():
        _need(value.shape == (64, 4) and value.dtype == np.float32 and np.isfinite(value).all(), f"{session}: candidate geometry/nonfinite drift")
        _need(declared[session] == {"raw_sha256": _raw_sha(value), "typed_sha256": _typed_sha(value),
                                    "shape": [64, 4], "dtype": value.dtype.str}, f"{session}: candidate array binding drift")
        _need(np.array_equal(value[:, :3], parent[session][:, :3]), f"{session}: candidate first-three bytes drift")
    source = _source_stack(carriers, frozen)
    old_source = _source_stack(parent, frozen)
    target_std = float(body.get("old_fourth_normalized_source_std", float("nan")))
    _need(np.isfinite(target_std) and np.isclose(float(source[:, 3].std(dtype=np.float64)), target_std, rtol=1e-6, atol=1e-8),
          "candidate source4 fourth-column std mismatch")
    _need(np.isclose(float(old_source[:, 3].std(dtype=np.float64)), target_std, rtol=1e-12, atol=1e-12),
          "candidate target fourth std is not parent source4 std")
    return carriers, body


def _baseline_loader(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frozen = _frozen()
    carriers, receipt = frozen.load_carrier_pack(path)
    return carriers, _canonical_baseline_receipt(receipt)


def load_carrier_pack(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load frozen baseline or fully verified mean-rate candidate by receipt schema."""
    receipt_path = Path(path).resolve().with_suffix(".json")
    _need(receipt_path.is_file(), f"carrier receipt missing: {receipt_path}")
    body = _read_json(receipt_path)
    if body.get("schema") == CANDIDATE_SCHEMA:
        return _candidate_loader(Path(path))
    return _baseline_loader(Path(path))


def replace_bank_carrier(bank: Any, carrier: np.ndarray) -> Any:
    """Exact frozen bank replacement API; carrier validation remains frozen-law."""
    return _frozen().replace_bank_carrier(bank, carrier)


def _rate_means(frozen: Any) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Read only the legal M10 rate rows through the frozen reader helpers."""
    values: dict[str, np.ndarray] = {}
    identity: dict[str, dict[str, Any]] = {}
    for session in frozen.ALL_SESSIONS:
        support = frozen._load_source_support(session) if session in frozen.SOURCE_SESSIONS else frozen._load_m10(session)
        _emg, rates = frozen._m10_rows(support)
        mean = np.asarray(rates, dtype=np.float64).mean(axis=0)
        _need(mean.shape == (64,) and np.isfinite(mean).all(), f"{session}: invalid M10 arithmetic mean rate")
        values[session] = mean
        identity[session] = _m10_row_identity(support, _emg, rates, frozen)
    return values, identity


def build(baseline_pack: Path, output_dir: Path) -> tuple[Path, Path]:
    """Build the sole candidate from a verified frozen pack and legal M10 rows."""
    frozen = _frozen()
    output_dir = Path(output_dir).resolve()
    _need(not output_dir.exists(), f"refuse to overwrite output directory: {output_dir}")
    parent_path = Path(baseline_pack).resolve()
    parent, parent_receipt = _baseline_loader(parent_path)
    _need(tuple(parent_receipt.get("source_sessions", ())) == tuple(frozen.SOURCE_SESSIONS), "parent source4 contract drift")
    _need(tuple(parent_receipt.get("ho_sessions", ())) == tuple(frozen.HO_SESSIONS), "parent HO3 contract drift")
    parent_fit = Path(parent_receipt["fit"]).resolve()
    parent_fit_contract = _verify_parent_fit(parent_fit, str(parent_receipt["fit_sha256"]))
    means, m10_row_identity = _rate_means(frozen)
    source_rates = np.concatenate([means[session] for session in frozen.SOURCE_SESSIONS])
    source_mean = float(source_rates.mean(dtype=np.float64)); source_std = float(source_rates.std(dtype=np.float64))
    _need(np.isfinite(source_mean) and np.isfinite(source_std) and source_std > 1e-12, "source4 mean-rate scale is degenerate")
    old_source = _source_stack(parent, frozen)
    old_std = float(old_source[:, 3].std(dtype=np.float64))
    _need(np.isfinite(old_std) and old_std > 0.0, "parent source4 fourth-column std is degenerate")
    carriers: dict[str, np.ndarray] = {}
    for session in frozen.ALL_SESSIONS:
        value = np.ascontiguousarray(np.asarray(parent[session], dtype=np.float32).copy())
        _need(np.array_equal(value[:, :3], parent[session][:, :3]), f"{session}: parent first-three copy drift")
        fourth = (means[session] - source_mean) / source_std * old_std
        value[:, 3] = np.asarray(fourth, dtype=np.float32)
        _need(value.shape == (64, 4) and np.isfinite(value).all() and np.array_equal(value[:, :3], parent[session][:, :3]),
              f"{session}: candidate first-three/geometry drift")
        carriers[session] = value
    candidate_source_std = float(_source_stack(carriers, frozen)[:, 3].std(dtype=np.float64))
    _need(np.isclose(candidate_source_std, old_std, rtol=1e-6, atol=1e-8), "candidate source4 fourth std did not match parent")
    output_dir.mkdir(parents=True)
    fit_path = output_dir / "mean_rate4_fit.npz"
    fit_receipt = output_dir / "mean_rate4_fit.json"
    common: dict[str, Any] = {
        "schema": FIT_SCHEMA, "status": "PASSED", "carrier_variant": CANDIDATE_VARIANT,
        "parent_carrier_pack": str(parent_path), "parent_carrier_pack_sha256": _sha(parent_path),
        "parent_fit": str(parent_fit), "parent_fit_sha256": parent_receipt["fit_sha256"],
        "parent_fit_contract": parent_fit_contract,
        "frozen_producer": str(FROZEN_PATH.resolve()), "frozen_producer_sha256": _sha(FROZEN_PATH),
        "facade_code": str(Path(__file__).resolve()), "facade_code_sha256": _sha(Path(__file__).resolve()),
        "source_sessions": list(frozen.SOURCE_SESSIONS), "ho_sessions": list(frozen.HO_SESSIONS),
        "rate_source": {"statistic": "arithmetic_mean_rate_hz", "reader": "frozen_m1_muscle_r100_carrier._m10_rows",
                        "support_trials": [0, 10], "source_calibration": "pooled source4 256 unit values"},
        "source_emg_rms_support_trials": [0, 310], "first_three_columns_byte_equal": True,
        "m10_row_identity": m10_row_identity,
        "source_mean_rate_hz": source_mean, "source_mean_rate_std_hz": source_std,
        "old_fourth_normalized_source_std": old_std, "fourth_formula": "(session_m10_mean_rate_hz-source4_pooled_mean)/source4_pooled_std*old_fourth_normalized_source_std",
        "candidate_source4_fourth_std": candidate_source_std,
        "official_test_included": False, "target_query_labels_used": False,
    }
    _atomic_npz(fit_path, metadata=np.array(json.dumps(common, sort_keys=True)), source_mean_rate_hz=np.array(source_mean),
                source_mean_rate_std_hz=np.array(source_std), old_fourth_normalized_source_std=np.array(old_std))
    fit_body = {**common, "fit_npz": str(fit_path.resolve()), "fit_npz_sha256": _sha(fit_path)}
    _atomic_json(fit_receipt, fit_body)
    pack_path = output_dir / "carrier_pack.npz"
    _atomic_npz(pack_path, **carriers)
    pack_body = {**common, "schema": CANDIDATE_SCHEMA, "carrier_pack": str(pack_path.resolve()),
                 "carrier_pack_sha256": _sha(pack_path), "fit": str(fit_path.resolve()), "fit_sha256": _sha(fit_path),
                 "fit_receipt": str(fit_receipt.resolve()), "fit_receipt_sha256": _sha(fit_receipt),
                 "carrier_arrays": _array_metadata(carriers)}
    _atomic_json(pack_path.with_suffix(".json"), pack_body)
    return pack_path, pack_path.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-only mean-rate fourth-coordinate M1 muscle candidate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--baseline-pack", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        pack, receipt = build(args.baseline_pack, args.output_dir)
        print(json.dumps({"carrier_pack": str(pack), "receipt": str(receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
