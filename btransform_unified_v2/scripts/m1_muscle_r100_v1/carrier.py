#!/usr/bin/env python3
"""Official M1 R100 muscle-response carrier construction.

This module is intentionally independent of the chronological last-date
protocol.  It fits the already specified ``muscle_response16_svd4`` / global
RMS carrier from the four official held-in calibration sessions, then deploys
the immutable fit on each official held-out calibration M10 surface.  It never
opens an official test surface and has no decoder, label, or score interface.
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
V2 = HERE.parent.parent
WORKSPACE = V2.parent
for _path in (V2, V2 / "scripts", WORKSPACE, WORKSPACE / "btransform_unified_v1" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# These are pure formula helpers.  Their historical source/target roster guards
# are not called here; R100 owns the official roster below.
from carrier_profile_v2.m1_muscle_profile import (  # noqa: E402
    DIM,
    EMG_SOURCE_STOP,
    M10,
    MuscleProfileFit,
    _orient,
    _raw16,
)
from m1_carrier_refinement_v1.aligned_carrier import load_aligned_support  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402

SCHEMA = "m1_muscle_r100_carrier_pack_v1"
FIT_SCHEMA = "m1_muscle_r100_fit_v1"
METHOD = "muscle_response16_svd4/global_rms"
SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO_SESSIONS = ("20121004", "20121017", "20121024")
ALL_SESSIONS = SOURCE_SESSIONS + HO_SESSIONS
DATA = WORKSPACE / "SPINT-main" / "data" / "000941"
FORMULA = V2 / "scripts" / "carrier_profile_v2" / "m1_muscle_profile.py"
ALIGNMENT = V2 / "scripts" / "m1_carrier_refinement_v1" / "aligned_carrier.py"


class CarrierError(RuntimeError):
    """Raised when an R100 carrier provenance or geometry contract drifts."""


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
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def _atomic_json(path: Path, body: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def source_path(session: str) -> Path:
    _need(session in SOURCE_SESSIONS, f"unknown official source session {session!r}")
    return DATA / "sub-MonkeyL-held-in-calib" / f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys.nwb"


def heldout_path(session: str) -> Path:
    _need(session in HO_SESSIONS, f"unknown official held-out session {session!r}")
    return DATA / "sub-MonkeyL-held-out-calib" / f"sub-MonkeyL-held-out-calib_ses-{session}_behavior+ecephys.nwb"


def path_for(session: str) -> Path:
    return source_path(session) if session in SOURCE_SESSIONS else heldout_path(session)


def _m10_rows(support) -> tuple[np.ndarray, np.ndarray]:
    emask = np.asarray(support.emg_trial_ids) < M10
    rmask = np.asarray(support.rate_trial_ids) < M10
    emg = np.asarray(support.emg[emask], dtype=np.float64)
    rates = np.asarray(support.rates[rmask], dtype=np.float64)
    _need(emg.ndim == 2 and rates.shape == (emg.shape[0], 64), "M10 aligned EMG/rate geometry drift")
    _need(np.array_equal(np.asarray(support.emg_trial_ids)[emask], np.asarray(support.rate_trial_ids)[rmask]), "M10 trial-row alignment drift")
    return emg, rates


def _load_source_support(session: str):
    return load_aligned_support(source_path(session), emg_trial_stop=EMG_SOURCE_STOP, neural_trial_stop=M10)


def _load_m10(session: str):
    return load_aligned_support(path_for(session), emg_trial_stop=M10, neural_trial_stop=M10)


def fit_source() -> tuple[MuscleProfileFit, dict[str, dict[str, str]]]:
    """Fit uncentered pooled source-row SVD4 and a common global RMS scale."""
    supports = {session: _load_source_support(session) for session in SOURCE_SESSIONS}
    source_emg = np.concatenate(
        [np.maximum(np.asarray(value.emg)[np.asarray(value.emg_trial_ids) < EMG_SOURCE_STOP], 0.0) for value in supports.values()],
        axis=0,
    )
    rms = np.maximum(np.sqrt(np.mean(np.square(source_emg), axis=0)), 1e-8)
    provisional = MuscleProfileFit(rms, np.zeros((DIM, 16)), np.zeros(DIM), np.ones(DIM), SOURCE_SESSIONS, {})
    raw_rows: list[np.ndarray] = []
    for session in SOURCE_SESSIONS:
        emg, rates = _m10_rows(supports[session])
        raw_rows.append(_raw16(provisional, emg, rates))
    pooled = np.concatenate(raw_rows, axis=0)
    _need(pooled.shape == (64 * len(SOURCE_SESSIONS), 16) and np.isfinite(pooled).all(), "source raw16 pool drift")
    _unused_left, _singular, right = np.linalg.svd(pooled, full_matrices=False)
    components = _orient(right[:DIM])
    projected = np.concatenate([row @ components.T for row in raw_rows], axis=0)
    mean = projected.mean(axis=0)
    # Preserve the specified order of operations: floor every projected-axis
    # standard deviation before forming the shared RMS denominator.  Flooring
    # only the final aggregate differs when an otherwise negligible axis is
    # present.
    original_std = np.maximum(projected.std(axis=0), 1e-6)
    global_rms = float(np.sqrt(np.mean(np.square(original_std))))
    metadata: dict[str, Any] = {
        "schema": FIT_SCHEMA,
        "status": "PASSED",
        "method": METHOD,
        "source_sessions": list(SOURCE_SESSIONS),
        "source_emg_trials": [0, EMG_SOURCE_STOP],
        "response_support_trials": [0, M10],
        "svd": "uncentered pooled [source-session×unit,16] raw16 rows; deterministic sign",
        "normalization": "global_rms",
        "normalizer": "source projected-row mean plus common RMS over four projected column standard deviations",
        "official_test_included": False,
    }
    inputs = {session: {"path": str(source_path(session).resolve()), "sha256": _sha(source_path(session))} for session in SOURCE_SESSIONS}
    metadata["source_raw_nwb"] = inputs
    fit = MuscleProfileFit(rms, components, mean, np.full(DIM, global_rms), SOURCE_SESSIONS, metadata)
    return fit, inputs


def project_m10(fit: MuscleProfileFit, session: str) -> np.ndarray:
    """Apply a frozen source fit to only this session's visible M10 support."""
    emg, rates = _m10_rows(_load_m10(session))
    raw4 = _raw16(fit, emg, rates) @ fit.svd_components.T
    carrier = np.ascontiguousarray((raw4 - fit.normalizer_mean) / fit.normalizer_scale, dtype=np.float32)
    _need(carrier.shape == (64, DIM) and np.isfinite(carrier).all(), f"{session}: invalid carrier")
    return carrier


def _save_fit(dest: Path, fit: MuscleProfileFit, inputs: Mapping[str, Mapping[str, str]]) -> tuple[Path, Path]:
    arrays = dest / "fit.npz"
    receipt = dest / "fit.json"
    body = {**dict(fit.metadata), "fit_npz": str(arrays.resolve()), "source_raw_nwb": dict(inputs),
            "formula_code": str(FORMULA.resolve()), "formula_code_sha256": _sha(FORMULA),
            "alignment_code": str(ALIGNMENT.resolve()), "alignment_code_sha256": _sha(ALIGNMENT),
            "implementation": str(Path(__file__).resolve()), "implementation_sha256": _sha(Path(__file__))}
    _atomic_npz(arrays, metadata=np.array(json.dumps(body, sort_keys=True)), muscle_rms=fit.muscle_rms,
                svd_components=fit.svd_components, normalizer_mean=fit.normalizer_mean,
                normalizer_scale=fit.normalizer_scale)
    body["fit_npz_sha256"] = _sha(arrays)
    _atomic_json(receipt, body)
    return arrays, receipt


def build(dest: Path) -> tuple[Path, Path]:
    """Build a fresh seven-session pack; no official-test file is addressable."""
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(f"refuse to overwrite {dest}")
    dest.mkdir(parents=True)
    fit, inputs = fit_source()
    fit_path, fit_receipt = _save_fit(dest, fit, inputs)
    carriers = {session: project_m10(fit, session) for session in ALL_SESSIONS}
    pack = dest / "carrier_pack.npz"
    _atomic_npz(pack, **carriers)
    array_meta = {
        session: {"raw_sha256": _raw_sha(value), "typed_sha256": _typed_sha(value), "shape": list(value.shape), "dtype": value.dtype.str}
        for session, value in carriers.items()
    }
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASSED",
        "method": METHOD,
        "normalization": "global_rms",
        "source_sessions": list(SOURCE_SESSIONS),
        "ho_sessions": list(HO_SESSIONS),
        "carrier_pack": str(pack.resolve()),
        "carrier_pack_sha256": _sha(pack),
        "fit": str(fit_path.resolve()),
        "fit_sha256": _sha(fit_path),
        "fit_receipt": str(fit_receipt.resolve()),
        "fit_receipt_sha256": _sha(fit_receipt),
        "source_raw_nwb": inputs,
        "deployment_raw_nwb": {
            session: {"path": str(path_for(session).resolve()), "sha256": _sha(path_for(session))}
            for session in ALL_SESSIONS
        },
        "deployment_support_trials": {session: [0, M10] for session in ALL_SESSIONS},
        "carrier_arrays": array_meta,
        "formula_code": str(FORMULA.resolve()), "formula_code_sha256": _sha(FORMULA),
        "alignment_code": str(ALIGNMENT.resolve()), "alignment_code_sha256": _sha(ALIGNMENT),
        "implementation": str(Path(__file__).resolve()), "implementation_sha256": _sha(Path(__file__)),
        "official_test_included": False,
        "target_query_labels_used": False,
    }
    receipt = dest / "carrier_pack.json"
    _atomic_json(receipt, body)
    return pack, receipt


def load_carrier_pack(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and fail closed on every pack/receipt/array binding."""
    path = Path(path).resolve()
    receipt = path.with_suffix(".json")
    _need(path.is_file() and receipt.is_file(), "carrier pack and adjacent JSON receipt are required")
    body = json.loads(receipt.read_text())
    _need(isinstance(body, dict) and body.get("schema") == SCHEMA and body.get("status") == "PASSED", "carrier receipt identity")
    _need(body.get("normalization") == "global_rms" and body.get("method") == METHOD, "carrier method/normalization drift")
    _need(body.get("official_test_included") is False and body.get("target_query_labels_used") is False,
          "carrier receipt must exclude official-test and target-query labels")
    _need(tuple(body.get("source_sessions", ())) == SOURCE_SESSIONS and tuple(body.get("ho_sessions", ())) == HO_SESSIONS, "carrier roster drift")
    _need(body.get("carrier_pack") == str(path) and body.get("carrier_pack_sha256") == _sha(path), "carrier pack SHA/path drift")
    fit = Path(body.get("fit", "")); fit_receipt = Path(body.get("fit_receipt", ""))
    _need(fit.is_file() and body.get("fit_sha256") == _sha(fit), "frozen fit binding drift")
    _need(fit_receipt.is_file() and body.get("fit_receipt_sha256") == _sha(fit_receipt), "fit receipt binding drift")
    fit_body = json.loads(fit_receipt.read_text())
    _need(isinstance(fit_body, dict) and fit_body.get("schema") == FIT_SCHEMA and fit_body.get("status") == "PASSED",
          "fit receipt identity drift")
    code_bindings = (
        ("formula_code", FORMULA),
        ("alignment_code", ALIGNMENT),
        ("implementation", Path(__file__).resolve()),
    )
    for field, current in code_bindings:
        sha_field = f"{field}_sha256"
        expected_path = str(current.resolve())
        expected_sha = _sha(current)
        _need(body.get(field) == expected_path and body.get(sha_field) == expected_sha,
              f"carrier receipt {field} binding drift")
        _need(fit_body.get(field) == body.get(field) and fit_body.get(sha_field) == body.get(sha_field),
              f"pack/fit receipt {field} disagreement")
    _need(fit_body.get("official_test_included") is False,
          "fit receipt must exclude official-test data")
    declared = body.get("carrier_arrays")
    _need(isinstance(declared, dict) and set(declared) == set(ALL_SESSIONS), "carrier array receipt roster drift")
    with np.load(path, allow_pickle=False) as archive:
        _need(set(archive.files) == set(ALL_SESSIONS), "carrier NPZ key roster drift")
        carriers = {}
        for session in ALL_SESSIONS:
            raw = archive[session]
            _need(raw.dtype == np.float32, f"{session}: carrier dtype drift")
            carriers[session] = np.ascontiguousarray(raw)
    for session, value in carriers.items():
        row = declared[session]
        _need(value.shape == (64, DIM) and value.dtype == np.float32 and np.isfinite(value).all(), f"{session}: carrier geometry/finite drift")
        _need(row == {"raw_sha256": _raw_sha(value), "typed_sha256": _typed_sha(value), "shape": [64, DIM], "dtype": value.dtype.str}, f"{session}: carrier receipt hash drift")
    return carriers, body


def replace_bank_carrier(bank: TaskBank, carrier: np.ndarray) -> TaskBank:
    """Return a new immutable bank with only its functional carrier replaced."""
    _need(isinstance(bank, TaskBank), "replace_bank_carrier requires TaskBank")
    value = np.ascontiguousarray(carrier, dtype=np.float32)
    _need(value.shape == (bank.E0.shape[0], DIM) and np.isfinite(value).all(), "bank/carrier geometry or finite drift")
    meta = dict(bank.calibration_meta)
    meta.update({"carrier_sha256": array_sha256(value), "carrier_rawdigest": _raw_sha(value),
                 "carrier_method": METHOD, "carrier_schema": SCHEMA})
    return dataclasses.replace(bank, carrier=value, calibration_meta=meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fresh official M1 R100 muscle carrier pack")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    pack, receipt = build(args.dest)
    print(json.dumps({"carrier_pack": str(pack), "receipt": str(receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
