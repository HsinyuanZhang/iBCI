"""Immutable source-only calibration panels for adaptive carrier research.

This module deliberately supplies raw behavior--unit panels only.  It contains
no profile selection, target scoring, or decoder training policy.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
M1_ALLOWED = ("ses-20120924", "ses-20120926", "ses-20120927")
M1_SEALED = "ses-20120928"
H1_INNER_LAST = "1925-01-15"
H1_OUTER_LAST = "1925-01-19"
M1_DIM, H1_DIM = 16, 14


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class PanelBundle:
    dataset: str
    sessions: tuple[str, ...]
    behavior_rms: np.ndarray
    panels: Mapping[str, np.ndarray]          # session -> [panels, units, raw_dim]
    calibration_raw: Mapping[str, np.ndarray] # session -> first panel [units, raw_dim]
    trial_groups: Mapping[str, tuple[tuple[float, ...], ...]]
    source_hashes: Mapping[str, str]
    validation: Mapping[str, np.ndarray]      # held source raw panels, never selection logic


def _m1_path(session: str) -> Path:
    _need(session in M1_ALLOWED, f"M1 source session is not allowed: {session}")
    return WORKSPACE / "SPINT-main/data/000941/sub-MonkeyL-held-in-calib" / f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys.nwb"


def _m1_support(path: Path, stop: int):
    # Self-contained path setup: this module may be launched without another
    # carrier script having populated sys.path first.
    for import_path in (ROOT, ROOT / "src", WORKSPACE, WORKSPACE / "streaming_calibration_exp"):
        text = str(import_path)
        if text not in sys.path: sys.path.insert(0, text)
    from scripts.m1_carrier_refinement_v1.aligned_carrier import load_aligned_support
    value = load_aligned_support(path, emg_trial_stop=stop, neural_trial_stop=stop)
    emg, rates = np.asarray(value.emg, np.float64), np.asarray(value.rates, np.float64)
    eid, rid = np.asarray(value.emg_trial_ids, np.int64), np.asarray(value.rate_trial_ids, np.int64)
    _need(emg.ndim == 2 and emg.shape[1] == 16 and rates.ndim == 2 and rates.shape[1] == 64, "M1 support geometry")
    _need(emg.shape[0] == rates.shape[0] and np.array_equal(eid, rid), "M1 aligned native bins")
    return emg, rates, rid


def _m1_raw(emg: np.ndarray, rates: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Exactly m1_muscle_profile._raw16, inlined to avoid fitting a profile."""
    weights = np.maximum(np.asarray(emg, np.float64), 0.0) / rms[None, :]
    mean_rate = rates.mean(axis=0)
    poisson_scale = np.sqrt(np.maximum(mean_rate * 0.02, 1.0)) / 0.02
    zrate = (rates - mean_rate[None, :]) / poisson_scale[None, :]
    return ((weights.T @ zrate) / (weights.sum(axis=0)[:, None] + 10.0)).T


def _h1_data_dir(data_dir: Path | None) -> Path:
    return (Path(data_dir) if data_dir is not None else WORKSPACE / "SPINT-main/data/000954").resolve()


def _h1_source_guard(sessions: Sequence[str]) -> None:
    """Accept only the exact v2 inner or outer chronological source rosters."""
    v1 = str(WORKSPACE / "btransform_unified_v1/src")
    if v1 not in sys.path: sys.path.insert(0, v1)
    from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
    inner_dates = ("1925-01-01", "1925-01-08", "1925-01-13", "1925-01-15")
    outer_dates = (*inner_dates, "1925-01-19")
    inner = tuple(session for date in inner_dates for session in H1_SESSIONS_BY_DATE[date])
    outer = tuple(session for date in outer_dates for session in H1_SESSIONS_BY_DATE[date])
    received = tuple(sessions)
    _need(received in (inner, outer), "H1 roster must be exact inner <=01/15 or outer <=01/19 sources")


def _h1_raw(rates: np.ndarray, velocity: np.ndarray, behavior_rms: np.ndarray) -> np.ndarray:
    """Exactly h1_profiles._raw_signed_state14 with its fixed 100-ms constants."""
    rate_mean = np.mean(rates, axis=0)
    noise_rate = np.sqrt(np.maximum(rate_mean * 0.1, 1.0)) / 0.1
    z = (rates - rate_mean[None, :]) / noise_rate[None, :]
    x = velocity / behavior_rms[None, :]
    weights = np.concatenate((np.logaddexp(0.0, x), np.logaddexp(0.0, -x)), axis=1)
    return (weights.T @ z / (weights.sum(axis=0)[:, None] + 10.0)).T


def _h1_record_blocks(record, values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for value in values:
        block = record.blocks_for(value)
        rates, velocity = np.asarray(block.rates, np.float64), np.asarray(block.velocity, np.float64)
        _need(rates.ndim == 2 and velocity.ndim == 2 and velocity.shape[1] == 7 and len(rates) == len(velocity) and len(rates) > 0, "H1 block geometry")
        _need(np.isfinite(rates).all() and np.isfinite(velocity).all(), "nonfinite H1 support")
        rows.append(rates); labels.append(velocity)
    return np.concatenate(rows), np.concatenate(labels)


def build_panels(dataset: str, sessions: Sequence[str], data_dir: Path | None = None) -> tuple[PanelBundle, dict]:
    """Build disjoint panels exclusively from declared source training trials.

    M1 panels are ten-trial groups [0,10),...,[300,310).  H1 panels are
    consecutive triples from ``trial_values[:-2]``; the final two native trials
    remain validation and are never inspected here.
    """
    names = tuple(map(str, sessions))
    _need(dataset in {"m1", "h1"} and names and len(set(names)) == len(names), "dataset/nonempty unique roster required")
    panels, calibration, groups, hashes, validation = {}, {}, {}, {}, {}
    if dataset == "m1":
        _need(all(name in M1_ALLOWED for name in names), "M1 source roster includes sealed/non-source date")
        loaded = {name: _m1_support(_m1_path(name), 310) for name in names}
        source_emg = np.concatenate([np.maximum(value[0][value[2] < 310], 0.0) for value in loaded.values()])
        behavior_rms = np.maximum(np.sqrt(np.mean(source_emg ** 2, axis=0)), 1e-8)
        for name, (emg, rates, trial_ids) in loaded.items():
            raw = []
            trial_groups = []
            for lo in range(0, 310, 10):
                mask = (trial_ids >= lo) & (trial_ids < lo + 10)
                _need(bool(mask.any()), f"empty M1 panel {name}[{lo}:{lo + 10})")
                raw.append(_m1_raw(emg[mask], rates[mask], behavior_rms)); trial_groups.append(tuple(float(x) for x in range(lo, lo + 10)))
            value = np.stack(raw).astype(np.float64, copy=False)
            panels[name], calibration[name], groups[name], validation[name] = value, value[0], tuple(trial_groups), value[1:]
            hashes[name] = _sha_file(_m1_path(name))
    else:
        _h1_source_guard(names)
        sys.path.insert(0, str(ROOT / "scripts"))
        from carrier_profile_v2 import h1_profiles
        records = h1_profiles._load_records(_h1_data_dir(data_dir), names)
        train_values = {}
        all_velocity = []
        for name in names:
            values = tuple(records[name].trial_values[:-2])
            _need(len(values) >= 6, f"H1 needs at least two source training panels: {name}")
            train_values[name] = values
            all_velocity.append(_h1_record_blocks(records[name], values)[1])
        behavior_rms = np.maximum(np.sqrt(np.mean(np.square(np.concatenate(all_velocity)), axis=0)), 1e-8)
        for name in names:
            values = train_values[name]; complete = len(values) // 3 * 3
            triples = tuple(tuple(values[i:i + 3]) for i in range(0, complete, 3))
            _need(triples, f"H1 has no complete panel triple: {name}")
            raw = []
            for triple in triples:
                rates, velocity = _h1_record_blocks(records[name], triple)
                raw.append(_h1_raw(rates, velocity, behavior_rms))
            value = np.stack(raw).astype(np.float64, copy=False)
            panels[name], calibration[name], groups[name], validation[name] = value, value[0], triples, value[1:]
            hashes[name] = str(getattr(records[name], "input_sha256", ""))
            _need(len(hashes[name]) == 64, f"H1 source input hash missing: {name}")
    bundle = PanelBundle(dataset, names, np.ascontiguousarray(behavior_rms, np.float64), panels, calibration, groups, hashes, validation)
    metadata = {"schema": "carrier_adaptive_v3_source_panels_v1", "dataset": dataset, "sessions": list(names), "raw_dim": M1_DIM if dataset == "m1" else H1_DIM, "calibration_budget": 10 if dataset == "m1" else 3, "source_only": True, "target_opened": False, "training_diagnostic_panels": "Panels after the first calibration group are source-training diagnostic panels; they are not independent held-out validation and can enter later source covariance/projection fitting.", "source_hashes": dict(hashes), "trial_groups": _jsonable(groups)}
    if dataset == "h1":
        metadata["h1_loader_semantics"] = "The frozen H1 record loader materializes the record; only first-three support trials enter calibration_raw/deploy_raw, and no query labels are used for fitting this raw panel."
        metadata["discarded_training_remainder_trial_values"] = {name: list(train_values[name][len(groups[name]) * 3:]) for name in names}
        metadata["excluded_last_two_validation_trial_values"] = {name: list(records[name].trial_values[-2:]) for name in names}
    return bundle, metadata


def _h1_target_sessions() -> tuple[str, ...]:
    v1 = str(WORKSPACE / "btransform_unified_v1/src")
    if v1 not in sys.path: sys.path.insert(0, v1)
    from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
    return tuple(H1_SESSIONS_BY_DATE["1925-01-19"] + H1_SESSIONS_BY_DATE["1925-01-20"])


def deploy_raw(dataset: str, session: str, path_or_data_dir: Path, behavior_rms: np.ndarray, source_seal: str) -> np.ndarray:
    """Return an explicit target's support-only raw panel without fitting.

    The frozen H1 loader materializes a complete record.  This function passes
    only its first three support trial values to the raw estimator; it does not
    use query labels for fitting or selection.
    """
    _need(isinstance(source_seal, str) and len(source_seal) >= 16, "nonempty source seal required")
    behavior_rms = np.asarray(behavior_rms, np.float64)
    if dataset == "m1":
        _need(session in ("ses-20120927", M1_SEALED), "M1 deployment needs explicit inner 0927 or outer 0928 target")
        target_path = Path(path_or_data_dir).resolve()
        _need(session in target_path.name, "M1 target path/session identity mismatch")
        emg, rates, ids = _m1_support(target_path, 10)
        mask = (ids >= 0) & (ids < 10)
        _need(bool(mask.any()), "empty M1 target M10")
        return np.ascontiguousarray(_m1_raw(emg[mask], rates[mask], behavior_rms), np.float64)
    if dataset == "h1":
        _need(session in _h1_target_sessions(), "H1 deployment session is not an inner/outer chronological target")
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path: sys.path.insert(0, scripts)
        from carrier_profile_v2 import h1_profiles
        record = h1_profiles._load_records(_h1_data_dir(Path(path_or_data_dir)), (session,))[session]
        values = tuple(record.trial_values[:3])
        _need(len(values) == 3, "H1 target lacks first-three support")
        rates, velocity = _h1_record_blocks(record, values)
        return np.ascontiguousarray(_h1_raw(rates, velocity, behavior_rms), np.float64)
    raise ValueError("dataset must be m1 or h1")


def save_bundle(bundle: PanelBundle, metadata: Mapping, destination: Path) -> tuple[Path, Path]:
    """Write immutable NPZ plus JSON receipt; refuses any overwrite."""
    destination = Path(destination); receipt = destination.with_suffix(".json"); arrays = destination.with_suffix(".npz")
    if receipt.exists() or arrays.exists(): raise FileExistsError("panel bundle outputs must be new")
    payload = {"behavior_rms": bundle.behavior_rms}
    for session in bundle.sessions:
        payload[f"panels/{session}"] = bundle.panels[session]; payload[f"calibration/{session}"] = bundle.calibration_raw[session]; payload[f"validation/{session}"] = bundle.validation[session]
    arrays.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(arrays, **payload)
    body = {**dict(metadata), "arrays": str(arrays.resolve()), "arrays_sha256": _sha_file(arrays), "array_sha256": {key: _array_sha(value) for key, value in payload.items()}, "implementation_sha256": _sha_file(Path(__file__).resolve())}
    receipt.write_text(json.dumps(_jsonable(body), indent=2, sort_keys=True) + "\n")
    return receipt, arrays


def load_bundle(receipt_path: Path) -> tuple[PanelBundle, dict]:
    receipt = Path(receipt_path); metadata = json.loads(receipt.read_text())
    arrays = Path(metadata["arrays"])
    _need(arrays.is_file() and metadata.get("arrays_sha256") == _sha_file(arrays), "panel array binding drift")
    sessions = tuple(metadata["sessions"])
    with np.load(arrays, allow_pickle=False) as z:
        behavior_rms = z["behavior_rms"].copy(); panels = {s: z[f"panels/{s}"].copy() for s in sessions}; calibration = {s: z[f"calibration/{s}"].copy() for s in sessions}; validation = {s: z[f"validation/{s}"].copy() for s in sessions}
    for key, value in {"behavior_rms": behavior_rms, **{f"panels/{s}": panels[s] for s in sessions}, **{f"calibration/{s}": calibration[s] for s in sessions}, **{f"validation/{s}": validation[s] for s in sessions}}.items(): _need(metadata.get("array_sha256", {}).get(key) == _array_sha(value), f"array binding drift {key}")
    groups = {s: tuple(tuple(row) for row in metadata["trial_groups"][s]) for s in sessions}
    return PanelBundle(metadata["dataset"], sessions, behavior_rms, panels, calibration, groups, metadata["source_hashes"], validation), metadata
