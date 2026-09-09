#!/usr/bin/env python3
"""Run source-only rolling diagnostics for candidate M1 behavioral profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
# HERE = <btransform_unified_v2>/scripts/carrier_profile_v2.
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT.parent)]
import m1_profiles as p

SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927")
DATA = WORKSPACE / "SPINT-main/data/000941/sub-MonkeyL-held-in-calib"


def path_for(session: str) -> Path:
    if session not in SESSIONS:
        raise ValueError("0928 is sealed and cannot be opened by this diagnostic")
    return DATA / f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys.nwb"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    x, y = x - x.mean(), y - y.mean(); d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x.dot(y) / d) if d > 0 else float("nan")


def _unit_cosine(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(x, float), np.asarray(y, float)
    c = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return {"median": float(np.median(c)), "q25": float(np.quantile(c, .25)), "mean": float(c.mean())}


def _pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return {"flattened_pearson": _pearson(x, y), "unit_cosine": _unit_cosine(x, y)}


def _column_scale(x: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(x, dtype=float)
    return {"rms": np.sqrt(np.mean(np.square(x), axis=0)).astype(float).tolist(),
            "std": x.std(axis=0).astype(float).tolist(),
            "mean": x.mean(axis=0).astype(float).tolist()}


def _normalized_support(x: np.ndarray) -> dict[str, float]:
    # x already passed the source-carrier normalizer in project_profile.
    a = np.abs(np.asarray(x, dtype=float))
    return {"max_abs": float(a.max()), "p95_abs": float(np.quantile(a, .95))}


def _energy(x: np.ndarray, behavioral_dim: int) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    behavioral = x[:, :behavioral_dim]
    return {"all4_rms": float(np.sqrt(np.mean(np.square(x)))),
            f"behavioral_components{behavioral_dim}_rms": float(np.sqrt(np.mean(np.square(behavioral)))),
            f"behavioral_components{behavioral_dim}_near_zero_fraction": float(np.mean(np.abs(behavioral) <= 1e-12))}


def _comparison(fit: p.ProfileFit, session: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    first5, first5_meta = p.project_profile(fit, path_for(session), trial_lo=0, trial_hi=5)
    second5, second5_meta = p.project_profile(fit, path_for(session), trial_lo=5, trial_hi=10)
    first10, first10_meta = p.project_profile(fit, path_for(session), trial_lo=0, trial_hi=10)
    next10, next10_meta = p.project_profile(fit, path_for(session), trial_lo=10, trial_hi=20)
    shifted, shifted_meta = p.project_profile(fit, path_for(session), trial_lo=0, trial_hi=10, circular_shift=True)
    real_shift = float(np.linalg.norm(first10 - shifted) / np.sqrt(first10.size))
    split_distance = float(np.linalg.norm(first5 - second5) / np.sqrt(first5.size))
    behavioral_dim = 4 if fit.candidate == "soft_prototype4" else 3
    behavioral_key = f"behavioral_components{behavioral_dim}"
    row = {"behavioral_dim": behavioral_dim,
           "split5_5": {behavioral_key: _pair_metrics(first5[:, :behavioral_dim], second5[:, :behavioral_dim]),
                          "all4": _pair_metrics(first5, second5)},
           "first10_next10": {behavioral_key: _pair_metrics(first10[:, :behavioral_dim], next10[:, :behavioral_dim]),
                                "all4": _pair_metrics(first10, next10)},
           "source_normalized_held_support": _normalized_support(first10),
           "profile_column_scale": {"first10": _column_scale(first10), "next10": _column_scale(next10),
                                    "shifted": _column_scale(shifted)},
           "raw_profile_energy": {"first10": first10_meta["raw_profile_energy"],
                                  "next10": next10_meta["raw_profile_energy"],
                                  "shifted": shifted_meta["raw_profile_energy"]},
           "behavioral_specificity": {"real_vs_circular_shift_rms": real_shift,
                                        "real_shift_over_split_noise": real_shift / max(split_distance, 1e-12),
                                        "source_normalized_profile_energy": {"real": _energy(first10, behavioral_dim), "circular_shift_null": _energy(shifted, behavioral_dim)}},
           "profile_digests": {"first10": p.profile_digest(first10), "next10": p.profile_digest(next10), "shifted": p.profile_digest(shifted)},
           "metadata": {"first5": first5_meta, "second5": second5_meta, "first10": first10_meta, "next10": next10_meta, "shifted": shifted_meta}}
    return row, {"first10": first10, "next10": next10, "shifted": shifted}


def run(candidates: tuple[str, ...]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    folds = (("basis_0924_validate_0926", (SESSIONS[0],), SESSIONS[1]),
             ("basis_0924_0926_validate_0927", SESSIONS[:2], SESSIONS[2]))
    report: dict[str, object] = {"schema": "m1_carrier_profile_v2_source_only_diagnostic_v1",
      "sealed_outer_target": "ses-20120928", "allowed_sessions": list(SESSIONS),
      "rolling_folds": [], "candidates": list(candidates),
      "laws": {"split_reliability": "profiles independently estimated on native [0,5) and [5,10) trials",
               "temporal_stability": "native [0,10) versus [10,20), no bin crosses a trial boundary",
               "specificity_null": "behavioral score rows circularly shifted by floor(native_trial_bin_count/2) within each support trial; rates fixed; actual shifts are recorded",
               "specificity_limit": "Within-trial circular shifts retain low-frequency source-neural association when EMG/rates are smooth; this is a conservative null, not an independence proof.",
               "prototype_coverage": "effective soft occupancy and assignment entropy reported for every projection"}}
    arrays: dict[str, np.ndarray] = {}
    for cname in candidates:
        crows = []
        for fname, train, valid in folds:
            fit = p.fit_source_profile(cname, train, path_for)
            metrics, values = _comparison(fit, valid)
            arrays[f"{cname}/{fname}/first10"] = values["first10"]
            arrays[f"{cname}/{fname}/next10"] = values["next10"]
            arrays[f"{cname}/{fname}/shifted"] = values["shifted"]
            crows.append({"fold": fname, "basis_sources": list(train), "validation_session": valid,
                          "fit": fit.metadata, "metrics": metrics})
        report["rolling_folds"].append({"candidate": cname, "folds": crows})
    return report, arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="JSON destination; companion .npz is also written")
    ap.add_argument("--candidates", default=",".join(p.CANDIDATES), help="comma-separated candidate names")
    args = ap.parse_args()
    candidates = tuple(x.strip() for x in args.candidates.split(",") if x.strip())
    if not candidates or any(x not in p.CANDIDATES for x in candidates):
        raise ValueError(f"candidates must be drawn from {p.CANDIDATES}")
    report, arrays = run(candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
