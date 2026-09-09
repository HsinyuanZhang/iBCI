"""Source-only M1 behavioral-profile constructors.

This module is intentionally outside the frozen chronological runner.  It
contains no 20120928 roster entry: callers must explicitly supply sources from
``ALLOWED_SOURCE_SESSIONS``.  The old rSyn3 branch is retained as a literal
reference implementation; the two new branches use the official Falcon
end-at-t neural bins supplied by ``aligned_carrier``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence
import hashlib
import json

import numpy as np

ALLOWED_SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927")
OUTER_TARGET_FORBIDDEN = "ses-20120928"
CANDIDATES = ("old_rsyn3", "standardized_rsyn3", "soft_prototype4")
SUPPORT_TRIALS = 10
FULL_SOURCE_EMG_TRIALS = 310
RANK = 3
CARRIER_DIM = 4
NOISE_FLOOR = 1.0e-3
PROTOTYPE_K = 4
OCCUPANCY_PSEUDOCOUNT = 10.0


class ProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class Support:
    """Native movement-bin arrays, with trial IDs retained for safe slicing."""
    emg: np.ndarray
    emg_trial_ids: np.ndarray
    rates: np.ndarray
    rate_trial_ids: np.ndarray
    unit_ids: np.ndarray
    loader: str


@dataclass(frozen=True)
class ProfileFit:
    candidate: str
    basis: object
    normalizer_mean: np.ndarray
    normalizer_scale: np.ndarray
    score_mean: np.ndarray | None
    score_scale: np.ndarray | None
    centers: np.ndarray | None
    prototype_tau2: float | None
    source_sessions: tuple[str, ...]
    metadata: dict[str, object]


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ProfileError(message)


def _rectify(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    _require(x.ndim == 2 and x.size and np.isfinite(x).all(), "EMG must be finite [time,channel]")
    return np.maximum(x, 0.0)


def _check_sessions(sessions: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(s) for s in sessions)
    _require(names and len(set(names)) == len(names), "sources must be nonempty and unique")
    _require(all(s in ALLOWED_SOURCE_SESSIONS for s in names),
             "only 20120924/20120926/20120927 are legal source sessions; 20120928 is sealed")
    return names


def _support_from_legacy(path: Path, *, emg_trial_stop: int, neural_trial_stop: int) -> Support:
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as raw_data
    z = raw_data.load_support_bins(path, emg_trial_stop=emg_trial_stop, neural_trial_stop=neural_trial_stop)
    # The legacy loader has no exposed NWB unit IDs; its rate columns are its
    # canonical row order.  A numeric identity is sufficient for diagnostics.
    return Support(np.asarray(z.emg), np.asarray(z.emg_trial_ids), np.asarray(z.rates),
                   np.asarray(z.rate_trial_ids), np.arange(z.rates.shape[1]), "legacy_rsyn3")


def _support_from_aligned(path: Path, *, emg_trial_stop: int, neural_trial_stop: int) -> Support:
    from scripts.m1_carrier_refinement_v1.aligned_carrier import load_aligned_support
    z = load_aligned_support(path, emg_trial_stop=emg_trial_stop, neural_trial_stop=neural_trial_stop)
    return Support(np.asarray(z.emg), np.asarray(z.emg_trial_ids), np.asarray(z.rates),
                   np.asarray(z.rate_trial_ids), np.asarray(z.unit_ids), "official_aligned")


def load_support(path: Path, *, trial_stop: int, candidate: str,
                 neural_trial_stop: int | None = None) -> Support:
    """Load only native trials ``[0, trial_stop)``; never opens query labels."""
    _require(candidate in CANDIDATES, f"unknown candidate {candidate!r}")
    _require(OUTER_TARGET_FORBIDDEN not in str(Path(path)),
             "ses-20120928 is sealed: this source-only profile module must not open it")
    neural_stop = int(trial_stop if neural_trial_stop is None else neural_trial_stop)
    _require(1 <= neural_stop <= int(trial_stop) <= FULL_SOURCE_EMG_TRIALS, "invalid native trial prefix")
    return (_support_from_legacy if candidate == "old_rsyn3" else _support_from_aligned)(
        Path(path), emg_trial_stop=int(trial_stop), neural_trial_stop=neural_stop)


def _slice_trials(s: Support, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return EMG, rates, and native trial IDs without crossing trial boundaries."""
    em = (s.emg_trial_ids >= lo) & (s.emg_trial_ids < hi)
    ra = (s.rate_trial_ids >= lo) & (s.rate_trial_ids < hi)
    _require(bool(em.any()) and bool(ra.any()), f"empty trial range [{lo},{hi})")
    e, r = np.asarray(s.emg[em], dtype=np.float64), np.asarray(s.rates[ra], dtype=np.float64)
    tids = np.asarray(s.rate_trial_ids[ra], dtype=np.int64)
    _require(e.shape[0] == r.shape[0], "EMG/rate bins differ after native trial slicing")
    _require(np.array_equal(np.asarray(s.emg_trial_ids[em], dtype=np.int64), tids),
             "EMG/rate trial sequence differs")
    return e, r, tids


def _source_basis(supports: dict[str, Support], sessions: tuple[str, ...]):
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    emg = []
    for name in sessions:
        # Neural rows may deliberately stop at M10; the NNMF basis alone sees
        # source EMG through trial 309.
        mask = supports[name].emg_trial_ids < FULL_SOURCE_EMG_TRIALS
        emg.append(_rectify(supports[name].emg[mask]))
    return syn3.fit_source_nmf(np.concatenate(emg, axis=0))


def _scores(basis: object, emg: np.ndarray) -> np.ndarray:
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    return syn3.project_basis(_rectify(emg), basis)


def _old_raw(basis: object, emg: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Literal legacy rSyn3 law: do not alter this reference path."""
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    scores = syn3.project_basis(_rectify(emg), basis)
    weights, intercepts = syn3.fit_all_units(scores, rates)
    return syn3.carrier_from_encoding(weights, intercepts)


def _standardized_raw(fit: ProfileFit, emg: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Four-D standardized encoding profile.

    Let z be NNMF scores and \u0304z,s_z be source-M10 moments.  Let y_u be the
    support rate, \u0304y_u its support mean and s_yu=max(sd(y_u), NOISE_FLOOR).
    Ridge uses X=[1,(z-\u0304z)/s_z], y=(rate-\u0304y)/s_yu and lambda=1 on the
    three non-intercept coordinates.  Coefficients are multiplied by
    q_u=Var(X beta)/(Var(X beta)+Var(residual)/n), an analytic reliability
    shrinkage toward zero.  The fourth raw coordinate is the unscaled support
    mean rate, before the common source carrier normalizer is applied.
    """
    _require(fit.score_mean is not None and fit.score_scale is not None, "missing score moments")
    z = (_scores(fit.basis, emg) - fit.score_mean) / fit.score_scale
    n = z.shape[0]
    design = np.column_stack((np.ones(n), z))
    gram = design.T @ design / float(n)
    penalty = np.diag([0.0, 1.0, 1.0, 1.0])
    means = rates.mean(axis=0)
    scales = np.maximum(rates.std(axis=0), NOISE_FLOOR)
    y = (rates - means[None, :]) / scales[None, :]
    beta = np.linalg.solve(gram + penalty, (design.T @ y) / float(n))
    predicted = design[:, 1:] @ beta[1:, :]
    resid = y - (design @ beta)
    signal = np.var(predicted, axis=0)
    noise = np.var(resid, axis=0) / float(max(n, 1))
    q = signal / np.maximum(signal + noise, NOISE_FLOOR ** 2)
    return np.column_stack(((beta[1:, :].T * q[:, None]), means))


def _farthest_centers(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Deterministic source-only K=4 centers, avoiding a hidden optimizer seed."""
    _require(x.shape[0] >= PROTOTYPE_K, "too few source score rows for four prototypes")
    order = np.lexsort((x[:, 2], x[:, 1], x[:, 0]))
    chosen = [int(order[0])]
    while len(chosen) < PROTOTYPE_K:
        d2 = np.min(np.sum((x[:, None, :] - x[np.asarray(chosen)][None, :, :]) ** 2, axis=2), axis=1)
        chosen.append(int(np.flatnonzero(d2 == d2.max())[0]))
    centers = x[np.asarray(chosen)].copy()
    nearest = np.min(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
    return centers, float(max(np.median(nearest), NOISE_FLOOR ** 2))


def _prototype_raw(fit: ProfileFit, emg: np.ndarray, rates: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    _require(fit.score_mean is not None and fit.score_scale is not None and fit.centers is not None
             and fit.prototype_tau2 is not None, "missing prototype parameters")
    z = (_scores(fit.basis, emg) - fit.score_mean) / fit.score_scale
    d2 = np.sum((z[:, None, :] - fit.centers[None, :, :]) ** 2, axis=2)
    logw = -d2 / (2.0 * fit.prototype_tau2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)
    overall = rates.mean(axis=0)
    rate_scale = np.maximum(rates.std(axis=0), NOISE_FLOOR)
    occupancy = w.sum(axis=0)
    means = (w.T @ rates) / np.maximum(occupancy[:, None], 1.0)
    contrast = ((means - overall[None, :]) / rate_scale[None, :]).T
    contrast *= (occupancy / (occupancy + OCCUPANCY_PSEUDOCOUNT))[None, :]
    return contrast, {"min_effective_occupancy": float(occupancy.min()),
                       "max_effective_occupancy": float(occupancy.max()),
                       "mean_assignment_entropy": float((-w * np.log(np.maximum(w, 1e-12))).sum(axis=1).mean())}


def fit_source_profile(candidate: Literal["old_rsyn3", "standardized_rsyn3", "soft_prototype4"],
                       sessions: Sequence[str], path_resolver: Callable[[str], Path]) -> ProfileFit:
    """Fit a source-only profile bank.  The caller controls the permitted roster."""
    names = _check_sessions(sessions)
    supports = {s: load_support(path_resolver(s), trial_stop=FULL_SOURCE_EMG_TRIALS,
                                neural_trial_stop=SUPPORT_TRIALS, candidate=candidate)
                for s in names}
    _require(len({v.rates.shape[1] for v in supports.values()}) == 1 and next(iter(supports.values())).rates.shape[1] == 64,
             "M1 profile requires the 64-unit decoder face")
    basis = _source_basis(supports, names)
    source_rows = []
    score_rows = []
    for s in names:
        e, r, _ = _slice_trials(supports[s], 0, SUPPORT_TRIALS)
        score_rows.append(_scores(basis, e))
    score_stack = np.concatenate(score_rows, axis=0)
    score_mean, score_scale = score_stack.mean(axis=0), np.maximum(score_stack.std(axis=0), NOISE_FLOOR)
    provisional = ProfileFit(candidate, basis, np.zeros(4), np.ones(4), score_mean, score_scale,
                             None, None, names, {})
    if candidate == "soft_prototype4":
        centers, tau2 = _farthest_centers((score_stack - score_mean) / score_scale)
        provisional = ProfileFit(candidate, basis, np.zeros(4), np.ones(4), score_mean, score_scale,
                                 centers, tau2, names, {})
    for s in names:
        e, r, _ = _slice_trials(supports[s], 0, SUPPORT_TRIALS)
        if candidate == "old_rsyn3": raw = _old_raw(basis, e, r)
        elif candidate == "standardized_rsyn3": raw = _standardized_raw(provisional, e, r)
        else: raw, _ = _prototype_raw(provisional, e, r)
        source_rows.append(raw)
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    mean, scale = syn3.source_normalizer(source_rows)
    metadata = {"candidate": candidate, "sources": list(names), "support_trials": SUPPORT_TRIALS,
                "basis_emg_trials": [0, FULL_SOURCE_EMG_TRIALS], "normalizer": "all source M10 unit rows",
                "new_profile_loader": "official_falcon_bin_units_end_at_t" if candidate != "old_rsyn3" else "legacy_literal_rsyn3"}
    if candidate == "soft_prototype4": metadata.update({"K": PROTOTYPE_K, "occupancy_pseudocount": OCCUPANCY_PSEUDOCOUNT, "tau2": provisional.prototype_tau2})
    return ProfileFit(candidate, basis, np.asarray(mean), np.asarray(scale), score_mean, score_scale,
                      provisional.centers, provisional.prototype_tau2, names, metadata)


def project_profile(fit: ProfileFit, path: Path, *, trial_lo: int = 0, trial_hi: int = SUPPORT_TRIALS,
                    circular_shift: bool = False) -> tuple[np.ndarray, dict[str, object]]:
    """Project one session using only a native trial interval.

    ``circular_shift`` rotates behavioral score rows *within each native trial*
    while leaving rates fixed; it is a behavioral-specificity null and never
    moves samples over a trial boundary.
    """
    _require(0 <= trial_lo < trial_hi <= FULL_SOURCE_EMG_TRIALS, "invalid trial interval")
    s = load_support(path, trial_stop=trial_hi, candidate=fit.candidate)
    e, r, tids = _slice_trials(s, trial_lo, trial_hi)
    if circular_shift:
        score = _scores(fit.basis, e).copy()
        shifts: dict[str, int] = {}
        for trial in np.unique(tids):
            ix = np.flatnonzero(tids == trial)
            _require(ix.size >= 2, "circular-shift trial has fewer than two bins")
            # A 20-ms shift leaves smooth EMG nearly unchanged.  Rotate by half
            # the native trial instead; every score row remains in its trial.
            shift = int(ix.size // 2)
            _require(shift >= 1, "circular-shift trial has no nonzero half-trial shift")
            score[ix] = np.roll(score[ix], shift, axis=0)
            shifts[str(int(trial))] = shift
        # Reconstruct EMG-independent profile directly from shifted scores.
        if fit.candidate == "old_rsyn3":
            from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
            w, b = syn3.fit_all_units(score, r); raw = syn3.carrier_from_encoding(w, b); extra = {"shift_per_trial": shifts}
        elif fit.candidate == "standardized_rsyn3":
            raw = _standardized_from_scores(fit, score, r); extra = {"shift_per_trial": shifts}
        else:
            raw, extra = _prototype_from_scores(fit, score, r); extra["shift_per_trial"] = shifts
    else:
        if fit.candidate == "old_rsyn3": raw, extra = _old_raw(fit.basis, e, r), {}
        elif fit.candidate == "standardized_rsyn3": raw, extra = _standardized_raw(fit, e, r), {}
        else: raw, extra = _prototype_raw(fit, e, r)
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    raw = np.asarray(raw, dtype=np.float64)
    raw_energy = {"all4_rms": float(np.sqrt(np.mean(np.square(raw)))),
                  "column_rms": np.sqrt(np.mean(np.square(raw), axis=0)).astype(float).tolist(),
                  "column_std": raw.std(axis=0).astype(float).tolist(),
                  "column_mean": raw.mean(axis=0).astype(float).tolist()}
    profile = np.asarray(syn3.normalize_carriers(raw, fit.normalizer_mean, fit.normalizer_scale), dtype=np.float32)
    _require(profile.shape == (64, 4) and np.isfinite(profile).all(), "invalid normalized profile")
    return profile, {"trial_interval": [trial_lo, trial_hi], "circular_shift": circular_shift,
                     "loader": s.loader, "raw_profile_energy": raw_energy, **extra}


def _standardized_from_scores(fit: ProfileFit, scores: np.ndarray, rates: np.ndarray) -> np.ndarray:
    # Exact score-array counterpart of _standardized_raw for the null diagnostic.
    _require(fit.score_mean is not None and fit.score_scale is not None, "missing score moments")
    z = (scores - fit.score_mean) / fit.score_scale; n = z.shape[0]
    design = np.column_stack((np.ones(n), z)); gram = design.T @ design / float(n)
    means = rates.mean(axis=0); scales = np.maximum(rates.std(axis=0), NOISE_FLOOR)
    y = (rates - means) / scales
    beta = np.linalg.solve(gram + np.diag([0., 1., 1., 1.]), design.T @ y / float(n))
    signal = np.var(design[:, 1:] @ beta[1:], axis=0); noise = np.var(y - design @ beta, axis=0) / float(n)
    q = signal / np.maximum(signal + noise, NOISE_FLOOR ** 2)
    return np.column_stack((beta[1:].T * q[:, None], means))


def _prototype_from_scores(fit: ProfileFit, scores: np.ndarray, rates: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    _require(fit.score_mean is not None and fit.score_scale is not None and fit.centers is not None and fit.prototype_tau2 is not None, "missing prototypes")
    z = (scores - fit.score_mean) / fit.score_scale
    d2 = ((z[:, None] - fit.centers[None]) ** 2).sum(axis=2); lw = -d2 / (2 * fit.prototype_tau2); lw -= lw.max(axis=1, keepdims=True)
    w = np.exp(lw); w /= w.sum(axis=1, keepdims=True); occ = w.sum(axis=0)
    overall, sd = rates.mean(axis=0), np.maximum(rates.std(axis=0), NOISE_FLOOR)
    contrast = (((w.T @ rates) / np.maximum(occ[:, None], 1.) - overall) / sd).T
    contrast *= (occ / (occ + OCCUPANCY_PSEUDOCOUNT))[None]
    return contrast, {"min_effective_occupancy": float(occ.min()), "max_effective_occupancy": float(occ.max()),
                      "mean_assignment_entropy": float((-w * np.log(np.maximum(w, 1e-12))).sum(axis=1).mean())}


def profile_digest(profile: np.ndarray) -> str:
    a = np.ascontiguousarray(profile)
    return hashlib.sha256(json.dumps({"shape": list(a.shape), "dtype": str(a.dtype)}, sort_keys=True).encode() + b"\n" + a.tobytes()).hexdigest()


def save_frozen_profile_fit(fit: ProfileFit, path: Path, *, source_seal: str) -> str:
    """Persist a source-validated fit as an NPZ with scalar JSON ``metadata``.

    ``source_seal`` is supplied by the orchestrator only after it has frozen
    source diagnostics/selection.  A deploy call must present the same value.
    """
    _require(isinstance(source_seal, str) and len(source_seal) >= 16, "source seal must be a nontrivial string")
    basis = fit.basis
    _require(getattr(basis, "kind", None) == "nnmf", "only frozen NNMF profile bases are deployable")
    meta = {**fit.metadata, "schema": "m1_carrier_profile_v2_frozen_fit_v1", "source_seal": source_seal,
            "candidate": fit.candidate, "source_sessions": list(fit.source_sessions)}
    arrays = {"metadata": np.array(json.dumps(meta, sort_keys=True)), "basis_scale": np.asarray(basis.scale),
              "basis_dictionary": np.asarray(basis.dictionary), "normalizer_mean": np.asarray(fit.normalizer_mean),
              "normalizer_scale": np.asarray(fit.normalizer_scale), "score_mean": np.asarray(fit.score_mean),
              "score_scale": np.asarray(fit.score_scale)}
    if fit.centers is not None: arrays["centers"] = np.asarray(fit.centers)
    if fit.prototype_tau2 is not None: arrays["prototype_tau2"] = np.array(fit.prototype_tau2)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
    return _sha_file(path)


def load_frozen_profile_fit(path: Path, *, source_seal: str) -> ProfileFit:
    """Load a sealed source fit; refuse a missing/mismatched selection seal."""
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1.syn3 import SourceBasis
    with np.load(Path(path), allow_pickle=False) as z:
        _require("metadata" in z, "frozen fit lacks metadata")
        meta = json.loads(str(z["metadata"].item()))
        _require(meta.get("schema") == "m1_carrier_profile_v2_frozen_fit_v1", "wrong frozen fit schema")
        _require(meta.get("source_seal") == source_seal, "source seal mismatch")
        candidate = str(meta.get("candidate")); _require(candidate in CANDIDATES, "unknown frozen candidate")
        basis = SourceBasis("nnmf", z["basis_scale"].copy(), z["basis_dictionary"].copy(), np.empty((0, RANK)),
                            tuple(range(RANK)), "frozen-deploy", {}, {})
        centers = z["centers"].copy() if "centers" in z else None
        tau2 = float(z["prototype_tau2"].item()) if "prototype_tau2" in z else None
        return ProfileFit(candidate, basis, z["normalizer_mean"].copy(), z["normalizer_scale"].copy(),
                          z["score_mean"].copy(), z["score_scale"].copy(), centers, tau2,
                          tuple(meta["source_sessions"]), meta)


def deploy_project_m10(fit: ProfileFit, target_path: Path, *, source_seal: str) -> tuple[np.ndarray, dict[str, object]]:
    """Explicit post-selection target deployment: only target M10, never diagnostic code.

    This is intentionally separate from :func:`project_profile`, whose source
    diagnostics fail closed on 0928.  It may run only for a fit restored with a
    matching source seal, and it opens exactly the target's first ten native
    trials through the candidate's original loader/alignment law.
    """
    _require(fit.metadata.get("source_seal") == source_seal, "deploy requires matching frozen source seal")
    target_path = Path(target_path)
    _require(SEALED_TARGET := ("ses-20120928" in str(target_path)), "deploy API is reserved for explicit outer target 0928")
    # Do not call load_support/project_profile: those are diagnostic APIs which
    # prohibit 0928.  This is the isolated M10-only post-source-seal path.
    if fit.candidate == "old_rsyn3": s = _support_from_legacy(target_path, emg_trial_stop=SUPPORT_TRIALS, neural_trial_stop=SUPPORT_TRIALS)
    else: s = _support_from_aligned(target_path, emg_trial_stop=SUPPORT_TRIALS, neural_trial_stop=SUPPORT_TRIALS)
    e, r, _ = _slice_trials(s, 0, SUPPORT_TRIALS)
    if fit.candidate == "old_rsyn3": raw, extra = _old_raw(fit.basis, e, r), {}
    elif fit.candidate == "standardized_rsyn3": raw, extra = _standardized_raw(fit, e, r), {}
    else: raw, extra = _prototype_raw(fit, e, r)
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    profile = np.asarray(syn3.normalize_carriers(raw, fit.normalizer_mean, fit.normalizer_scale), dtype=np.float32)
    _require(profile.shape == (64, 4) and np.isfinite(profile).all(), "invalid deploy profile")
    return profile, {"schema": "m1_carrier_profile_v2_sealed_target_m10_v1", "candidate": fit.candidate,
                     "source_seal": source_seal, "target_support_trials": [0, SUPPORT_TRIALS],
                     "target_query_read": False, "carrier_sha256": profile_digest(profile), **extra}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
