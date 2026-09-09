#!/usr/bin/env python3
"""Source-only M1 muscle-recruitment profile and rolling diagnostic.

The profile is descriptive: it measures a unit's conditional response across
the 16 rectified EMG channels.  It is neither a causal muscle-effect estimate
nor an identification of independent muscle effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import argparse, hashlib, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(WORKSPACE)]

ALLOWED = ("ses-20120924", "ses-20120926", "ses-20120927")
SEALED = "ses-20120928"
M10, EMG_SOURCE_STOP, MUSCLES, DIM = 10, 310, 16, 4
BIN_SECONDS, OCCUPANCY_PSEUDOBINS = .02, 10.


class MuscleProfileError(RuntimeError): pass


@dataclass(frozen=True)
class Support:
    emg: np.ndarray; emg_trial_ids: np.ndarray
    rates: np.ndarray; rate_trial_ids: np.ndarray


@dataclass(frozen=True)
class MuscleProfileFit:
    muscle_rms: np.ndarray
    svd_components: np.ndarray
    normalizer_mean: np.ndarray
    normalizer_scale: np.ndarray
    source_sessions: tuple[str, ...]
    metadata: dict[str, object]


def _require(ok: bool, msg: str) -> None:
    if not ok: raise MuscleProfileError(msg)


def _path(session: str) -> Path:
    _require(session in ALLOWED, "only 20120924/26/27 are allowed in source diagnostics")
    return WORKSPACE / "SPINT-main/data/000941/sub-MonkeyL-held-in-calib" / f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys.nwb"


def _load(path: Path, *, emg_stop: int, neural_stop: int) -> Support:
    _require(SEALED not in str(path), "0928 is sealed for this source-only diagnostic")
    from scripts.m1_carrier_refinement_v1.aligned_carrier import load_aligned_support
    z = load_aligned_support(path, emg_trial_stop=emg_stop, neural_trial_stop=neural_stop)
    _require(z.emg.shape[1] == MUSCLES and z.rates.shape[1] == 64, "expected M1 16 EMG channels and 64 units")
    return Support(np.asarray(z.emg, dtype=np.float64), np.asarray(z.emg_trial_ids),
                   np.asarray(z.rates, dtype=np.float64), np.asarray(z.rate_trial_ids))


def _slice(s: Support, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    em = (s.emg_trial_ids >= lo) & (s.emg_trial_ids < hi)
    ra = (s.rate_trial_ids >= lo) & (s.rate_trial_ids < hi)
    _require(bool(em.any()) and bool(ra.any()), f"empty native trials [{lo},{hi})")
    e, r, ids = s.emg[em], s.rates[ra], np.asarray(s.rate_trial_ids[ra], dtype=np.int64)
    _require(e.shape[0] == r.shape[0] and np.array_equal(np.asarray(s.emg_trial_ids[em], dtype=np.int64), ids),
             "aligned EMG/rate native-bin mismatch")
    return e, r, ids


def _raw16(fit: MuscleProfileFit, emg: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Return unit×muscle descriptive conditional responses.

    For support bin t and muscle m, w_tm=max(EMG_tm,0)/source_RMS_m.
    Counts are rate*.02; each unit's standardized centered rate is
    (rate_tu-mean_u)/(sqrt(max(mean_count_u,1))/.02).  The raw response is
    sum_t w_tm * standardized_rate_tu / (sum_t w_tm + 10), for every u,m.
    """
    weights = np.maximum(np.asarray(emg, dtype=np.float64), 0.) / fit.muscle_rms[None, :]
    mean_rate = rates.mean(axis=0)
    mean_count = mean_rate * BIN_SECONDS
    poisson_scale = np.sqrt(np.maximum(mean_count, 1.0)) / BIN_SECONDS
    zrate = (rates - mean_rate[None, :]) / poisson_scale[None, :]
    return ((weights.T @ zrate) / (weights.sum(axis=0)[:, None] + OCCUPANCY_PSEUDOBINS)).T


def _orient(components: np.ndarray) -> np.ndarray:
    out = components.copy()
    for k in range(out.shape[0]):
        anchor = int(np.argmax(np.abs(out[k])))
        if out[k, anchor] < 0: out[k] *= -1
    return out


def fit_source(sessions: Sequence[str], resolver: Callable[[str], Path] = _path, *,
               normalization: str = "per_column") -> MuscleProfileFit:
    names = tuple(str(s) for s in sessions)
    _require(names and len(set(names)) == len(names) and all(s in ALLOWED for s in names), "illegal source roster")
    supports = {s: _load(resolver(s), emg_stop=EMG_SOURCE_STOP, neural_stop=M10) for s in names}
    source_emg = np.concatenate([np.maximum(v.emg[v.emg_trial_ids < EMG_SOURCE_STOP], 0.) for v in supports.values()])
    rms = np.maximum(np.sqrt(np.mean(source_emg ** 2, axis=0)), 1e-8)
    provisional = MuscleProfileFit(rms, np.zeros((DIM, MUSCLES)), np.zeros(DIM), np.ones(DIM), names, {})
    raw_rows = []
    for s in names:
        e, r, _ = _slice(supports[s], 0, M10); raw_rows.append(_raw16(provisional, e, r))
    pooled = np.concatenate(raw_rows, axis=0)
    _require(pooled.shape[1] == MUSCLES and pooled.shape[0] >= DIM, "SVD profile geometry")
    _u, _singular, vt = np.linalg.svd(pooled, full_matrices=False)
    components = _orient(vt[:DIM])
    projected = [raw @ components.T for raw in raw_rows]
    stack = np.concatenate(projected, axis=0)
    _require(normalization in ("per_column", "global_rms"), "normalization must be per_column or global_rms")
    mean = stack.mean(axis=0)
    original_std = np.maximum(stack.std(axis=0), 1e-6)
    variance = np.square(original_std)
    variance_fraction = variance / np.maximum(float(variance.sum()), 1e-12)
    global_scale = float(np.sqrt(np.mean(variance)))
    scale = original_std if normalization == "per_column" else np.full(DIM, max(global_scale, 1e-6))
    meta = {"candidate": "muscle_response16_svd4", "sources": list(names), "source_emg_trials": [0, EMG_SOURCE_STOP],
            "support_trials": M10, "aligned_neural_bins": "official_falcon_bin_units_end_at_t",
            "muscle_rms": "source EMG [0,310), rectified RMS per muscle", "rate_scale": "sqrt(max(mean_count,1))/0.02",
            "response": "sum(rectified_EMG/source_RMS * centered_poisson_scaled_rate)/(weight_sum+10)",
            "svd": "pooled source raw unit rows, deterministic component sign", "normalizer": "source pooled unit-row mean with selected scale",
            "normalization": normalization, "original_column_std": original_std.tolist(),
            "variance_fraction": variance_fraction.tolist(), "global_scale": global_scale}
    return MuscleProfileFit(rms, components, mean, scale, names, meta)


def project(fit: MuscleProfileFit, path: Path, *, lo: int = 0, hi: int = M10,
            circular_shift: bool = False) -> tuple[np.ndarray, dict[str, object]]:
    _require(0 <= lo < hi <= EMG_SOURCE_STOP, "invalid native trial interval")
    s = _load(path, emg_stop=hi, neural_stop=hi)
    e, r, ids = _slice(s, lo, hi)
    shifts: dict[str, int] = {}
    if circular_shift:
        e = e.copy()
        for trial in np.unique(ids):
            ix = np.flatnonzero(ids == trial); shift = int(len(ix) // 2)
            _require(shift >= 1, "native trial too short for half-trial null")
            e[ix] = np.roll(e[ix], shift, axis=0); shifts[str(int(trial))] = shift
    raw16 = _raw16(fit, e, r)
    raw4 = raw16 @ fit.svd_components.T
    profile = ((raw4 - fit.normalizer_mean) / fit.normalizer_scale).astype(np.float32)
    _require(profile.shape == (64, DIM) and np.isfinite(profile).all(), "invalid [64,4] muscle profile")
    raw_energy = {"raw16_column_rms": np.sqrt(np.mean(raw16 ** 2, axis=0)).tolist(),
                  "raw4_column_rms": np.sqrt(np.mean(raw4 ** 2, axis=0)).tolist(), "raw4_rms": float(np.sqrt(np.mean(raw4 ** 2)))}
    return profile, {"trial_interval": [lo, hi], "circular_shift": circular_shift, "shift_per_trial": shifts,
                     "raw_profile_energy": raw_energy}


def save_frozen_fit(fit: MuscleProfileFit, path: Path, *, source_seal: str) -> str:
    """Persist a source-selected muscle fit as an NPZ scalar JSON metadata plan."""
    _require(isinstance(source_seal, str) and len(source_seal) >= 16, "source seal must be a nontrivial string")
    meta = {**fit.metadata, "schema": "m1_muscle_response16_svd4_frozen_fit_v1", "source_seal": source_seal,
            "source_sessions": list(fit.source_sessions)}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path, metadata=np.array(json.dumps(meta,sort_keys=True)), muscle_rms=fit.muscle_rms,
                        svd_components=fit.svd_components, normalizer_mean=fit.normalizer_mean, normalizer_scale=fit.normalizer_scale)
    return _sha(path)


def load_frozen_fit(path: Path, *, source_seal: str) -> MuscleProfileFit:
    with np.load(Path(path),allow_pickle=False) as z:
        meta=json.loads(str(z["metadata"].item()))
        _require(meta.get("schema") == "m1_muscle_response16_svd4_frozen_fit_v1", "wrong muscle frozen-fit schema")
        _require(meta.get("source_seal") == source_seal, "source seal mismatch")
        return MuscleProfileFit(z["muscle_rms"].copy(),z["svd_components"].copy(),z["normalizer_mean"].copy(),z["normalizer_scale"].copy(),tuple(meta["source_sessions"]),meta)


def deploy_project_m10(fit: MuscleProfileFit, target_path: Path, *, source_seal: str) -> tuple[np.ndarray, dict[str, object]]:
    """Post-selection target-only M10 deploy path, isolated from diagnostic ``project``."""
    _require(fit.metadata.get("source_seal") == source_seal, "deploy requires matching frozen source seal")
    _require(SEALED in str(target_path), "deploy API is reserved for the explicit outer target 0928")
    from scripts.m1_carrier_refinement_v1.aligned_carrier import load_aligned_support
    z=load_aligned_support(Path(target_path),emg_trial_stop=M10,neural_trial_stop=M10)
    e,r=np.asarray(z.emg,dtype=np.float64),np.asarray(z.rates,dtype=np.float64)
    raw4=_raw16(fit,e,r) @ fit.svd_components.T
    profile=((raw4-fit.normalizer_mean)/fit.normalizer_scale).astype(np.float32)
    _require(profile.shape==(64,DIM) and np.isfinite(profile).all(),"invalid deploy muscle profile")
    return profile,{"schema":"m1_muscle_response16_svd4_sealed_target_m10_v1","source_seal":source_seal,"target_support_trials":[0,M10],"target_query_read":False,"carrier_sha256":_digest(profile)}


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def _digest(a: np.ndarray) -> str:
    a=np.ascontiguousarray(a); return hashlib.sha256(json.dumps({"shape":list(a.shape),"dtype":str(a.dtype)},sort_keys=True).encode()+b"\n"+a.tobytes()).hexdigest()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel().astype(float), b.ravel().astype(float); x -= x.mean(); y -= y.mean(); d = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x.dot(y)/d) if d else float("nan")


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    cos = (a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
    return {"flattened_pearson": _pearson(a,b), "unit_cosine_median": float(np.median(cos)), "unit_cosine_q25": float(np.quantile(cos,.25))}


def diagnose(normalization: str = "per_column") -> tuple[dict[str, object], dict[str, np.ndarray]]:
    folds = (("basis_0924_validate_0926", (ALLOWED[0],), ALLOWED[1]), ("basis_0924_0926_validate_0927", ALLOWED[:2], ALLOWED[2]))
    _require(normalization in ("per_column", "global_rms"), "invalid normalization")
    report: dict[str, object] = {"schema":"m1_muscle_response16_svd4_source_only_diagnostic_v1", "normalization":normalization, "sealed_outer_target":SEALED, "folds":[],
      "limitations":"Descriptive muscle-recruitment association only; neither causal nor independent-muscle effects. Half-trial null can retain low-frequency EMG/neural association.",
      "null":"within each native trial, rotate all 16 rectified EMG channels by floor(bin_count/2), holding rates fixed; shifts recorded"}
    arrays: dict[str,np.ndarray] = {}
    for name, sources, valid in folds:
        fit = fit_source(sources, normalization=normalization)
        a, ma = project(fit,_path(valid),lo=0,hi=5); b, mb = project(fit,_path(valid),lo=5,hi=10)
        c, mc = project(fit,_path(valid),lo=0,hi=10); d, md = project(fit,_path(valid),lo=10,hi=20); null, mn = project(fit,_path(valid),lo=0,hi=10,circular_shift=True)
        split_rms=float(np.linalg.norm(a-b)/np.sqrt(a.size)); null_rms=float(np.linalg.norm(c-null)/np.sqrt(c.size))
        report["folds"].append({"fold":name,"basis_sources":list(sources),"validation_session":valid,"fit":fit.metadata,
          "split5_5":_metrics(a,b),"first10_next10":_metrics(c,d),"source_normalized_held_support":{"max_abs":float(np.abs(c).max()),"p95_abs":float(np.quantile(np.abs(c),.95))},
          "column_scale":{"first10_rms":np.sqrt(np.mean(c**2,axis=0)).tolist(),"first10_std":c.std(axis=0).tolist()},
          "raw_profile_energy":{"first5":ma["raw_profile_energy"],"second5":mb["raw_profile_energy"],"first10":mc["raw_profile_energy"],"next10":md["raw_profile_energy"],"null":mn["raw_profile_energy"]},
          "specificity":{"real_vs_halftrial_null_rms":null_rms,"null_over_split_noise":null_rms/max(split_rms,1e-12),"shift_per_trial":mn["shift_per_trial"],"source_normalized_real_rms":float(np.sqrt(np.mean(c**2))),"source_normalized_null_rms":float(np.sqrt(np.mean(null**2)))}})
        arrays[f"{name}/first10"] = c; arrays[f"{name}/next10"] = d; arrays[f"{name}/null"] = null
    return report, arrays


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--normalization",choices=("per_column","global_rms"),default="per_column"); args=ap.parse_args()
    report, arrays=diagnose(args.normalization); args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); np.savez_compressed(args.out.with_suffix(".npz"),**arrays)


if __name__ == "__main__": main()
