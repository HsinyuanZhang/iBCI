"""Source-only chronological diagnostics for H1 behavioral-profile candidates.

The two folds deliberately stop at 1925-01-15 and 1925-01-19.  In particular,
this program has no code path that opens either 1925-01-20 record.  Held dates
are used only as three-trial support deployments under a frozen source plan.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

import h1_profiles as profiles


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPINT = ROOT.parent / "SPINT-main"
FOLDS = {
    "hold_1925-01-15": {
        "train_dates": ("1925-01-01", "1925-01-08", "1925-01-13"),
        "hold_date": "1925-01-15",
    },
    "hold_1925-01-19": {
        "train_dates": ("1925-01-01", "1925-01-08", "1925-01-13", "1925-01-15"),
        "hold_date": "1925-01-19",
    },
}
FORBIDDEN_DATE = "1925-01-20"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = np.asarray(left, np.float64).ravel(), np.asarray(right, np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0.0 else float(a @ b / denominator)


def _profile_stability(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    if first.shape != second.shape:
        raise RuntimeError("profile stability requires matching unit/profile shapes")
    per_unit = [_cosine(first[index], second[index]) for index in range(len(first))]
    finite = [value for value in per_unit if value is not None and np.isfinite(value)]
    return {"profile_shape": list(first.shape), "pooled_cosine": _cosine(first, second),
            "unit_cosine_mean": float(np.mean(finite)) if finite else None,
            "unit_cosine_median": float(np.median(finite)) if finite else None,
            "valid_unit_cosines": len(finite)}


def _profile_observables(profile: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    """Expose scale and energy so a collapsed normalizer cannot hide zeros."""
    values, unnormalized = np.asarray(profile, np.float64), np.asarray(raw, np.float64)
    return {"profile_column_mean": values.mean(axis=0).tolist(),
            "profile_column_std": values.std(axis=0).tolist(),
            "profile_column_rms": np.sqrt(np.mean(np.square(values), axis=0)).tolist(),
            "held_profile_abs_max": float(np.abs(values).max()),
            "held_profile_abs_p95": float(np.quantile(np.abs(values), 0.95)),
            "raw_column_rms": np.sqrt(np.mean(np.square(unnormalized), axis=0)).tolist(),
            "raw_all_column_rms": float(np.sqrt(np.mean(np.square(unnormalized))))}


def _chronological_legacy_prepare() -> Any:
    """Get fresh_plan/fit_deployment_carrier through the chronological helper."""
    path = ROOT / "scripts" / "chronological_last2_v1" / "h1_prepare.py"
    name = "carrier_profile_v2_chronological_h1_prepare"
    cached = sys.modules.get(name)
    if cached is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import chronological H1 preparation helper")
        cached = importlib.util.module_from_spec(spec)
        sys.modules[name] = cached
        try:
            spec.loader.exec_module(cached)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return cached._load_legacy_prepare()


def _shifted_labels(record: Any, values: tuple[float, ...]) -> dict[float, np.ndarray]:
    """Rotate labels within each trial, retaining its own block count/order."""
    result: dict[float, np.ndarray] = {}
    for value in values:
        velocity = np.asarray(record.blocks_for(value).velocity, np.float64)
        if len(velocity) < 2:
            raise RuntimeError(f"{record.session_name}/{value}: circular shift needs >=2 blocks")
        result[value] = np.roll(velocity, shift=max(1, len(velocity) // 2), axis=0)
    return result


def _old_hc_profile(record: Any, legacy_prepare: Any, plan: Any, rms: float, values: tuple[float, ...], *, shifted: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    override = _shifted_labels(record, values) if shifted else None
    fitted = legacy_prepare.fit_deployment_carrier(record, plan, values, labels_override=override)
    profile = np.asarray(fitted["carrier"], np.float64) / float(rms)
    raw = np.asarray(fitted["raw_carrier"], np.float64)
    if profile.shape != raw.shape or profile.shape[1] != 4 or not np.isfinite(profile).all():
        raise RuntimeError("old H-C deployment profile is invalid")
    return profile, raw, {"support_trials": list(values), "labels": "within_trial_circular_shift" if shifted else "observed",
                          "carrier_sha256": profiles.array_sha256(profile), "raw_sha256": profiles.array_sha256(raw),
                          "eb_weight_min": float(np.min(fitted["weight"])), "eb_weight_max": float(np.max(fitted["weight"]))}


def _behavior_evidence(records: Iterable[Any]) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    trial_coverage: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        _, velocity, trials = profiles.support_blocks(record)
        rows.append(velocity)
        trial_coverage[record.session_name] = []
        for trial in trials:
            _, one_velocity, _ = profiles.support_blocks(record, (trial,))
            trial_coverage[record.session_name].append({"trial": float(trial), "blocks": int(len(one_velocity)),
                                                        "axis_rms": np.sqrt(np.mean(np.square(one_velocity), axis=0)).tolist()})
    stacked = np.concatenate(rows)
    _, singular, _ = np.linalg.svd(stacked, full_matrices=False)
    rank = int(np.linalg.matrix_rank(stacked))
    return {"support_only": True, "velocity_shape": list(stacked.shape), "velocity_rank": rank,
            "velocity_singular_values": singular.tolist(), "per_trial_behavior_coverage": trial_coverage}


def _per_trial_profiles(record: Any, plan: profiles.ProfilePlan) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    values = tuple(record.trial_values[:3])
    if len(values) != 3:
        raise RuntimeError(f"{record.session_name}: exactly three support trials required")
    arrays: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    for index, trial in enumerate(values):
        deployed, raw, details = profiles.deploy_profile(record, plan, (trial,))
        arrays[f"trial_{index + 1}"] = deployed
        info[f"trial_{index + 1}"] = {"trial": float(trial), "profile_sha256": details["profile_sha256"],
                                      "raw_sha256": details["raw_sha256"], "support_blocks": details["support_blocks"]}
    pairs: dict[str, float | None] = {}
    labels = sorted(arrays)
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            pairs[f"{left}_vs_{right}"] = _cosine(arrays[left], arrays[right])
    return {"independent_trial_profiles": info, "pairwise_pooled_cosines": pairs}, arrays


def _fold_sessions(legacy: Any, spec: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_date = legacy.H1_SESSIONS_BY_DATE
    train = tuple(session for date in spec["train_dates"] for session in by_date[date])
    hold = tuple(by_date[spec["hold_date"]])
    if set(train) & set(hold) or any(legacy.session_date(name) == FORBIDDEN_DATE for name in train + hold):
        raise RuntimeError("chronological diagnostic roster leaks its forbidden future date")
    return train, hold


def _load_records(legacy: Any, data_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    paths = legacy.index_heldin_calib(data_dir)
    if set(names) - set(paths):
        raise RuntimeError("diagnostic requested absent H1 held-in record")
    return {name: legacy.load_record(paths[name]) for name in names}


def diagnose_candidate(data_dir: Path, candidate: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the two fixed temporal folds; source axes/moments never see hold data."""
    legacy = profiles._legacy()
    if not hasattr(legacy, "H1_SESSIONS_BY_DATE"):
        # The loader module itself does not define the shared date mapping.
        from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
        legacy.H1_SESSIONS_BY_DATE = H1_SESSIONS_BY_DATE
    if not hasattr(legacy, "session_date"):
        from btransform_unified_v1.h1_config import H1_SESSION_DATES
        legacy.session_date = lambda name: next(date for date, names in H1_SESSION_DATES.items() if name in names)
    old_prepare = _chronological_legacy_prepare()
    result: dict[str, Any] = {"candidate": candidate, "folds": {}}
    arrays: dict[str, np.ndarray] = {}
    for fold_name, spec in FOLDS.items():
        train_names, hold_names = _fold_sessions(legacy, spec)
        train_records = _load_records(legacy, data_dir, train_names)
        plan, source_diagnostics = profiles.fit_source_plan(train_records, train_names, candidate)
        old_plan, old_rms, old_authority = old_prepare.fresh_plan(spec["hold_date"], train_records)
        # Do not construct axes, behaviour RMS, SVD, or normalizers after this
        # point.  The held records receive only this frozen plan and M3 support.
        hold_records = _load_records(legacy, data_dir, hold_names)
        hold_rows: dict[str, Any] = {}
        for name in hold_names:
            record = hold_records[name]
            available = tuple(record.trial_values)
            if len(available) < 6:
                raise RuntimeError(f"{name}: diagnostic needs first3 and next3 native trials")
            first_profile, first_raw, first_info = profiles.deploy_profile(record, plan, available[:3])
            next_profile, next_raw, next_info = profiles.deploy_profile(record, plan, available[3:6])
            shifted_profile, shifted_raw, shifted_info = profiles.deploy_profile(
                record, plan, available[:3], circular_shift=True
            )
            first_values, next_values = available[:3], available[3:6]
            old_first, old_first_raw, old_first_info = _old_hc_profile(record, old_prepare, old_plan, old_rms, first_values)
            old_next, old_next_raw, old_next_info = _old_hc_profile(record, old_prepare, old_plan, old_rms, next_values)
            old_shifted, old_shifted_raw, old_shifted_info = _old_hc_profile(
                record, old_prepare, old_plan, old_rms, first_values, shifted=True
            )
            per_trial, independent = _per_trial_profiles(record, plan)
            prefix = f"{fold_name}/{candidate}/{name}"
            arrays[f"{prefix}/first3_profile"] = first_profile
            arrays[f"{prefix}/next3_profile"] = next_profile
            arrays[f"{prefix}/shifted_first3_profile"] = shifted_profile
            arrays[f"{prefix}/first3_raw"] = first_raw
            arrays[f"{prefix}/next3_raw"] = next_raw
            arrays[f"{prefix}/shifted_first3_raw"] = shifted_raw
            arrays[f"{prefix}/old_hc_first3_profile"] = old_first
            arrays[f"{prefix}/old_hc_next3_profile"] = old_next
            arrays[f"{prefix}/old_hc_shifted_first3_profile"] = old_shifted
            arrays[f"{prefix}/old_hc_first3_raw"] = old_first_raw
            arrays[f"{prefix}/old_hc_next3_raw"] = old_next_raw
            arrays[f"{prefix}/old_hc_shifted_first3_raw"] = old_shifted_raw
            arrays.update({f"{prefix}/{key}_profile": value for key, value in independent.items()})
            hold_rows[name] = {
                "first3_vs_next3": _profile_stability(first_profile, next_profile),
                "real_vs_within_trial_circular_shift": _profile_stability(first_profile, shifted_profile),
                "first3": first_info, "next3": next_info, "shifted_first3": shifted_info,
                "profile_observables": {"first3": _profile_observables(first_profile, first_raw),
                                        "next3": _profile_observables(next_profile, next_raw),
                                        "shifted_first3": _profile_observables(shifted_profile, shifted_raw)},
                "old_hc_same_fold": {
                    "first3_vs_next3": _profile_stability(old_first, old_next),
                    "real_vs_within_trial_circular_shift": _profile_stability(old_first, old_shifted),
                    "first3": old_first_info, "next3": old_next_info, "shifted_first3": old_shifted_info,
                    "profile_observables": {"first3": _profile_observables(old_first, old_first_raw),
                                            "next3": _profile_observables(old_next, old_next_raw),
                                            "shifted_first3": _profile_observables(old_shifted, old_shifted_raw)},
                },
                **per_trial,
            }
        evidence = _behavior_evidence(train_records.values())
        result["folds"][fold_name] = {
            "train_dates": list(spec["train_dates"]), "hold_date": spec["hold_date"],
            "source_sessions": list(train_names), "hold_sessions": list(hold_names),
            "forbidden_unread_date": FORBIDDEN_DATE,
            "source_plan": profiles.plan_metadata(plan), "source_session_diagnostics": source_diagnostics,
            "old_hc_source_plan": {"authority": old_authority, "rms": float(old_rms),
                                    "fit": "fresh_plan on this fold's source records only"},
            "source_behavior_evidence": evidence, "hold_support_only": True,
            "hold": hold_rows,
        }
    result["old_hc_comparison"] = {"status": "COMPLETED_SAME_FOLD_SOURCE_REFIT",
                                    "scope": "fresh_plan source-only refit per inner fold; held deployment uses only first3/next3 support"}
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="SPINT H1 held-in-calib directory")
    parser.add_argument("--out", type=Path, required=True, help="new diagnostic JSON receipt")
    parser.add_argument("--arrays-out", type=Path, help="new NPZ; defaults beside --out")
    parser.add_argument("--candidate", choices=profiles.CANDIDATES, action="append", required=True,
                        help="repeat for both candidates")
    args = parser.parse_args()
    out = args.out.resolve()
    arrays_out = (args.arrays_out.resolve() if args.arrays_out else out.with_suffix(".npz"))
    if out.exists() or arrays_out.exists():
        raise FileExistsError("diagnostic outputs must be new")
    candidates = tuple(dict.fromkeys(args.candidate))
    all_results: dict[str, Any] = {}
    all_arrays: dict[str, np.ndarray] = {}
    for candidate in candidates:
        result, arrays = diagnose_candidate(args.data.resolve(), candidate)
        all_results[candidate] = result
        all_arrays.update(arrays)
    arrays_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_out, **all_arrays)
    receipt = {"schema": "h1_behavioral_profile_source_only_diagnostic_v2", "status": "COMPLETED",
               "data_dir": str(args.data.resolve()), "candidates": list(candidates),
               "arrays_path": str(arrays_out), "arrays_sha256": sha256_file(arrays_out),
               "diagnostics": profiles.jsonable(all_results), "implementation_sha256": {
                   str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
                   str(HERE / "h1_profiles.py"): sha256_file(HERE / "h1_profiles.py"),
               }}
    _atomic_json(out, receipt)
    print(json.dumps({"receipt": str(out), "arrays": str(arrays_out), "candidates": list(candidates)}, sort_keys=True))


if __name__ == "__main__":
    main()
