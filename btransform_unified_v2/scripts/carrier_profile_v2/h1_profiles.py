"""Source-fitted H1 behavioral encoding profiles.

This candidate deliberately avoids the historical H1 neural-PCA/readout/back-
projection carrier.  It estimates a static row for every recorded unit directly
from 100-ms support blocks, fits the four-dimensional basis only from declared
source records, and applies that frozen basis to later support records.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPINT = ROOT.parent / "SPINT-main"
CANDIDATES = ("velocity7", "signed_state14")
PROFILE_DIM = 4
VELOCITY_DIM = 7
BLOCK_SECONDS = 0.1
POISSON_COUNT_FLOOR = 1.0
RIDGE_PER_SAMPLE = 1.0
STATE_OCCUPANCY_PSEUDO_BLOCKS = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    """Convert diagnostic provenance without silently rounding numeric arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _legacy() -> Any:
    """Import the frozen H1 loader without opening any record during import."""
    module_name = "carrier_profile_v2_h1_loader"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    for path in (ROOT / "src", SPINT, ROOT.parent / "btransform_unified_v1" / "src"):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location(
        module_name, SPINT / "src" / "data" / "h1_m4_eb_pilot.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import SPINT H1 100-ms block loader")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through sys.modules during
    # decoration.  File-spec imports must therefore register the module before
    # executing its class definitions.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


@dataclass(frozen=True)
class ProfilePlan:
    """All fitted quantities are source-only and candidate-specific."""

    candidate: str
    source_sessions: tuple[str, ...]
    source_input_sha256: tuple[str, ...]
    behavior_rms: np.ndarray             # [7], raw source velocity RMS
    raw_basis: np.ndarray                 # [7 or 14, 4], source pooled SVD basis
    profile_mean: np.ndarray              # [4], source pooled mean
    profile_scale: np.ndarray             # [4], source pooled RMS-scale
    source_raw_sha256: str
    source_profile_sha256: str

    @property
    def raw_dim(self) -> int:
        return 7 if self.candidate == "velocity7" else 14


def _check_candidate(candidate: str) -> None:
    if candidate not in CANDIDATES:
        raise ValueError(f"candidate must be one of {CANDIDATES}")


def _support_trials(record: Any, values: Sequence[float] | None) -> tuple[float, ...]:
    trials = tuple(record.trial_values[:3] if values is None else values)
    if not trials or len(set(map(float, trials))) != len(trials):
        raise ValueError("support trials must be nonempty and unique")
    return tuple(map(float, trials))


def support_blocks(record: Any, values: Sequence[float] | None = None, *, circular_shift: bool = False) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Return legal 100-ms support rates/velocities, shifting labels within trials only."""
    trials = _support_trials(record, values)
    rate_rows: list[np.ndarray] = []
    velocity_rows: list[np.ndarray] = []
    for trial_number in trials:
        block = record.blocks_for(trial_number)
        rates, velocity = np.asarray(block.rates, np.float64), np.asarray(block.velocity, np.float64)
        if rates.ndim != 2 or rates.shape[1] <= 0 or velocity.ndim != 2 or velocity.shape[1] != VELOCITY_DIM:
            raise RuntimeError(f"{record.session_name}/{trial_number}: invalid 100-ms block shapes")
        if len(rates) == 0 or len(rates) != len(velocity) or not np.isfinite(rates).all() or not np.isfinite(velocity).all():
            raise RuntimeError(f"{record.session_name}/{trial_number}: empty/nonfinite support block")
        if circular_shift:
            # A deterministic nonidentity label rotation, confined to this native trial.
            shift = 1 if len(velocity) == 1 else max(1, len(velocity) // 2)
            velocity = np.roll(velocity, shift=shift, axis=0)
        rate_rows.append(rates)
        velocity_rows.append(velocity)
    return np.concatenate(rate_rows), np.concatenate(velocity_rows), trials


def _unit_standardize(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson-noise-floor standardization of 100-ms rate blocks.

    Each unit uses its support mean count.  The standard deviation is
    ``sqrt(max(mean_count, 1 count))/0.1 s``: a Poisson variance model with a
    one-count floor, preventing silent or low-count units from unbounded gain.
    """
    if rates.ndim != 2 or len(rates) == 0:
        raise ValueError("rates must be nonempty [blocks, units]")
    rate_mean = np.mean(rates, axis=0)
    mean_count = np.maximum(rate_mean * BLOCK_SECONDS, POISSON_COUNT_FLOOR)
    noise_rate = np.sqrt(mean_count) / BLOCK_SECONDS
    standardized = (rates - rate_mean[None, :]) / noise_rate[None, :]
    return standardized, rate_mean, noise_rate


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, value)


def _raw_velocity7(rates: np.ndarray, velocity: np.ndarray, plan: ProfilePlan) -> tuple[np.ndarray, dict[str, Any]]:
    z, rate_mean, noise_rate = _unit_standardize(rates)
    x = velocity / plan.behavior_rms[None, :]
    n = len(x)
    gram = x.T @ x
    beta = np.linalg.solve(gram + np.eye(VELOCITY_DIM) * (RIDGE_PER_SAMPLE * n), x.T @ z)
    # Per-axis analytic reliability from the same ridge prior, retained in the
    # provenance so a later consumer cannot mistake these for unshrunk slopes.
    reliability = np.diag(gram) / (np.diag(gram) + RIDGE_PER_SAMPLE * n)
    raw = (beta * reliability[:, None]).T
    return raw, {"support_blocks": int(n), "rate_mean": rate_mean, "noise_rate": noise_rate,
                 "axis_reliability": reliability, "ridge_per_sample": RIDGE_PER_SAMPLE}


def _raw_signed_state14(rates: np.ndarray, velocity: np.ndarray, plan: ProfilePlan) -> tuple[np.ndarray, dict[str, Any]]:
    z, rate_mean, noise_rate = _unit_standardize(rates)
    x = velocity / plan.behavior_rms[None, :]
    # Smooth positive and negative states are deliberately not hard thresholds.
    weights = np.concatenate((_softplus(x), _softplus(-x)), axis=1)
    occupancy = np.sum(weights, axis=0)
    # Baseline is zero after unit standardization; ten pseudo 100-ms blocks
    # therefore give an explicit occupancy reliability shrink toward zero.
    raw = (weights.T @ z / (occupancy[:, None] + STATE_OCCUPANCY_PSEUDO_BLOCKS)).T
    reliability = occupancy / (occupancy + STATE_OCCUPANCY_PSEUDO_BLOCKS)
    return raw, {"support_blocks": int(len(x)), "rate_mean": rate_mean, "noise_rate": noise_rate,
                 "state_occupancy": occupancy, "state_reliability": reliability,
                 "occupancy_pseudo_blocks": STATE_OCCUPANCY_PSEUDO_BLOCKS}


def raw_profile(record: Any, plan: ProfilePlan, values: Sequence[float] | None = None, *, circular_shift: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one session's raw [units,7/14] profile using a frozen plan."""
    _check_candidate(plan.candidate)
    rates, velocity, trials = support_blocks(record, values, circular_shift=circular_shift)
    if plan.candidate == "velocity7":
        raw, diagnostics = _raw_velocity7(rates, velocity, plan)
    else:
        raw, diagnostics = _raw_signed_state14(rates, velocity, plan)
    if raw.shape != (rates.shape[1], plan.raw_dim) or not np.isfinite(raw).all():
        raise RuntimeError("raw behavioral profile is invalid")
    diagnostics.update({"session": record.session_name, "support_trials": list(trials),
                        "labels": "within_trial_circular_shift" if circular_shift else "observed"})
    return raw, diagnostics


def deploy_profile(record: Any, plan: ProfilePlan, values: Sequence[float] | None = None, *, circular_shift: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Project a support-only raw profile through frozen source basis/statistics."""
    raw, diagnostics = raw_profile(record, plan, values, circular_shift=circular_shift)
    projected = raw @ plan.raw_basis
    profile = (projected - plan.profile_mean[None, :]) / plan.profile_scale[None, :]
    if profile.shape != (raw.shape[0], PROFILE_DIM) or not np.isfinite(profile).all():
        raise RuntimeError("deployed behavioral profile must be finite [units,4]")
    diagnostics["raw_sha256"] = array_sha256(raw)
    diagnostics["profile_sha256"] = array_sha256(profile)
    return profile, raw, diagnostics


def fit_source_plan(records: Mapping[str, Any], source_sessions: Iterable[str], candidate: str) -> tuple[ProfilePlan, dict[str, dict[str, Any]]]:
    """Fit behavior scales, pooled SVD basis, and final 4-D normalization from source only."""
    _check_candidate(candidate)
    names = tuple(source_sessions)
    if not names or len(set(names)) != len(names) or set(names) - set(records):
        raise ValueError("source_sessions must be a nonempty unique subset of records")
    if any(getattr(records[name], "session_name", name) != name for name in names):
        raise RuntimeError("record/session identity mismatch")
    support_velocity = [support_blocks(records[name])[1] for name in names]
    stacked_velocity = np.concatenate(support_velocity)
    behavior_rms = np.sqrt(np.mean(np.square(stacked_velocity), axis=0))
    behavior_rms = np.maximum(behavior_rms, 1e-8)
    provisional = ProfilePlan(candidate, names, tuple(records[name].input_sha256 for name in names), behavior_rms,
                              np.empty((7 if candidate == "velocity7" else 14, PROFILE_DIM)),
                              np.zeros(PROFILE_DIM), np.ones(PROFILE_DIM), "", "")
    raw_by_session: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for name in names:
        raw, info = raw_profile(records[name], provisional)
        raw_by_session[name], diagnostics[name] = raw, info
    pooled_raw = np.concatenate([raw_by_session[name] for name in names], axis=0)
    _, singular, right = np.linalg.svd(pooled_raw, full_matrices=False)
    if right.shape[0] < PROFILE_DIM or not np.isfinite(singular).all():
        raise RuntimeError("source pooled raw profile SVD is invalid")
    basis = np.asarray(right[:PROFILE_DIM].T, np.float64)
    projected = pooled_raw @ basis
    mean = np.mean(projected, axis=0)
    scale = np.sqrt(np.mean(np.square(projected - mean[None, :]), axis=0))
    scale = np.maximum(scale, 1e-8)
    source_raw_hash = array_sha256(pooled_raw)
    source_profile = (projected - mean[None, :]) / scale[None, :]
    plan = ProfilePlan(candidate, names, tuple(records[name].input_sha256 for name in names), behavior_rms, basis,
                       mean, scale, source_raw_hash, array_sha256(source_profile))
    for name in names:
        diagnostics[name].update({"source_raw_sha256": array_sha256(raw_by_session[name]),
                                  "source_basis_singular_values": singular.tolist()})
    return plan, diagnostics


def plan_metadata(plan: ProfilePlan) -> dict[str, Any]:
    return {"schema": "h1_behavioral_profile_plan_v2", "candidate": plan.candidate,
            "raw_dim": plan.raw_dim, "profile_dim": PROFILE_DIM,
            "source_sessions": list(plan.source_sessions), "source_input_sha256": list(plan.source_input_sha256),
            "support": "earliest three native eval-valid trials per session; 100-ms blocks only",
            "behavior_scaling": "per-axis source-only raw velocity RMS",
            "unit_standardization": "support mean rate; Poisson sqrt(max(mean count,1))/0.1s noise floor",
            "velocity_ridge_per_sample": RIDGE_PER_SAMPLE,
            "state_occupancy_pseudo_100ms_blocks": STATE_OCCUPANCY_PSEUDO_BLOCKS,
            "basis": "source-only pooled per-unit raw-profile SVD; axes have no physical labels",
            "final_normalization": "source-only pooled projected-profile mean and RMS scale",
            "source_raw_sha256": plan.source_raw_sha256, "source_profile_sha256": plan.source_profile_sha256,
            "arrays": {"behavior_rms": array_sha256(plan.behavior_rms), "raw_basis": array_sha256(plan.raw_basis),
                       "profile_mean": array_sha256(plan.profile_mean), "profile_scale": array_sha256(plan.profile_scale)}}


def save_plan(path: Path, plan: ProfilePlan, diagnostics: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write a receipt JSON and companion NPZ once; refuse overwrite."""
    receipt = path.with_suffix(".json") if path.suffix != ".json" else path
    arrays = receipt.with_suffix(".npz")
    if receipt.exists() or arrays.exists():
        raise FileExistsError("profile plan outputs must be new")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, behavior_rms=plan.behavior_rms, raw_basis=plan.raw_basis,
                        profile_mean=plan.profile_mean, profile_scale=plan.profile_scale)
    body = {**plan_metadata(plan), "arrays_path": str(arrays.resolve()), "arrays_sha256": sha256_file(arrays),
            "source_diagnostics": jsonable(diagnostics), "implementation_sha256": sha256_file(Path(__file__).resolve())}
    receipt.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt, arrays


def _load_records(data_dir: Path, sessions: Sequence[str]) -> dict[str, Any]:
    legacy = _legacy()
    paths = legacy.index_heldin_calib(data_dir)
    if set(sessions) - set(paths):
        raise RuntimeError("requested H1 session is absent from held-in-calib index")
    return {name: legacy.load_record(paths[name]) for name in sessions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="SPINT H1 held-in-calib directory")
    parser.add_argument("--out", type=Path, required=True, help="new plan JSON path (NPZ is adjacent)")
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--source-sessions", nargs="+", required=True)
    args = parser.parse_args()
    records = _load_records(args.data.resolve(), tuple(args.source_sessions))
    plan, diagnostics = fit_source_plan(records, tuple(args.source_sessions), args.candidate)
    receipt, arrays = save_plan(args.out.resolve(), plan, diagnostics)
    print(json.dumps({"receipt": str(receipt), "arrays": str(arrays), "candidate": args.candidate}, sort_keys=True))


if __name__ == "__main__":
    main()
