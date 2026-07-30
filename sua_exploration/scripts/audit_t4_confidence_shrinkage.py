#!/usr/bin/env python3
"""Train-only audit of uncertainty-driven shrinkage for low-budget T4.

The ordinary T4 fit supplies cosine coefficients ``a,c`` and a fit-residual
variance.  This audit asks whether shrinking only ``a,c`` toward the
population-symmetric prior of zero improves rate prediction on later trials.
Every hyperparameter is selected leave-one-*training*-session-out; validation
and formal-test NWBs are never opened.

This is a mechanistic proxy audit, not a decoding result.  Its only purpose is
to decide whether a low-label T4-shrink arm deserves GPU training after the two
already requested FP32 architecture screens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .audit_t4_confidence_predictive_validity import (
        EPS,
        _direction_design_descriptors,
        _fit_t4_matrix,
        _parse_budgets,
    )
except ImportError:  # Direct ``python path/to/script.py`` execution.
    from audit_t4_confidence_predictive_validity import (
        EPS,
        _direction_design_descriptors,
        _fit_t4_matrix,
        _parse_budgets,
    )
from mc_maze.unit_side_features import (
    CANONICAL_DIRECTIONS_RAD,
    _nearest_canonical_direction_index,
    _pool_trial_rate_matrix,
    list_datamodule_rewarded_trials,
)


DEFAULT_LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
BASELINE = "none"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shrink_factor(
    signal_power: np.ndarray,
    uncertainty_power: np.ndarray,
    *,
    family: str,
    strength: float,
) -> np.ndarray:
    """Return a bounded multiplicative factor for ``a,c``."""
    signal = np.asarray(signal_power, dtype=np.float64)
    uncertainty = np.asarray(uncertainty_power, dtype=np.float64)
    if signal.shape != uncertainty.shape:
        raise ValueError("signal and uncertainty shapes differ")
    if np.any(signal < 0.0) or np.any(uncertainty < 0.0):
        raise ValueError("signal and uncertainty must be nonnegative")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be finite and nonnegative")
    scaled_uncertainty = strength * uncertainty
    if family == "wiener":
        factor = signal / (signal + scaled_uncertainty + EPS)
    elif family == "positive_part":
        factor = np.maximum(
            0.0,
            1.0 - scaled_uncertainty / (signal + EPS),
        )
    else:
        raise ValueError(f"unsupported shrinkage family: {family}")
    return np.clip(factor, 0.0, 1.0)


def _candidate_grid(lambdas: tuple[float, ...]) -> dict[str, tuple[str, float] | None]:
    candidates: dict[str, tuple[str, float] | None] = {BASELINE: None}
    for family in ("wiener", "positive_part"):
        for strength in lambdas:
            candidates[f"{family}:{strength:g}"] = (family, strength)
    return candidates


def _session_candidate_scores(
    path: Path,
    *,
    budgets: tuple[int, ...],
    reference_pool: int,
    candidates: dict[str, tuple[str, float] | None],
) -> dict[int, dict]:
    trials = list_datamodule_rewarded_trials(
        path,
        bin_size_ms=20,
        window_size=50,
        trial_result_filter="R",
    )
    if len(trials) < reference_pool:
        raise ValueError(
            f"{path.name}: requires {reference_pool} rewarded trials, "
            f"found {len(trials)}"
        )
    trials = trials[:reference_pool]
    directions = np.asarray(
        [
            _nearest_canonical_direction_index(trial["target_dir"])
            if trial.get("target_dir") is not None
            else -1
            for trial in trials
        ],
        dtype=np.int64,
    )
    rates, _ = _pool_trial_rate_matrix(path, trials)
    output: dict[int, dict] = {}
    for budget in budgets:
        t4, confidence = _fit_t4_matrix(
            rates[:, :budget],
            directions[:budget],
        )
        residual_variance = np.exp(confidence[:, 0])
        _entropy, log_se_trace = _direction_design_descriptors(
            directions[:budget],
            residual_variance,
        )
        uncertainty = np.maximum(np.exp(2.0 * log_se_trace) - EPS, 0.0)
        signal = np.square(t4[:, 0]) + np.square(t4[:, 1])

        valid_future = directions[budget:reference_pool] >= 0
        future_directions = directions[budget:reference_pool][valid_future]
        future_rates = rates[:, budget:reference_pool][:, valid_future]
        theta = np.asarray(
            [
                CANONICAL_DIRECTIONS_RAD[int(index)]
                for index in future_directions
            ],
            dtype=np.float64,
        )
        cosine = np.cos(theta)[None, :]
        sine = np.sin(theta)[None, :]
        log_mse: dict[str, np.ndarray] = {}
        factors: dict[str, np.ndarray] = {}
        for name, specification in candidates.items():
            if specification is None:
                factor = np.ones(rates.shape[0], dtype=np.float64)
            else:
                family, strength = specification
                factor = shrink_factor(
                    signal,
                    uncertainty,
                    family=family,
                    strength=strength,
                )
            predicted = (
                t4[:, 3:4]
                + factor[:, None]
                * (
                    t4[:, 0:1] * cosine
                    + t4[:, 1:2] * sine
                )
            )
            mse = np.mean(np.square(future_rates - predicted), axis=1)
            log_mse[name] = np.log(mse + EPS)
            factors[name] = factor
        output[budget] = {
            "unit_count": rates.shape[0],
            "future_trial_count": int(valid_future.sum()),
            "log_mse": log_mse,
            "factors": factors,
        }
    return output


def nested_loso_summary(
    by_session: dict[str, dict],
    candidate_names: tuple[str, ...],
) -> dict:
    """Select one shrink rule on N-1 sessions and score the held-out train session."""
    sessions = sorted(by_session)
    if len(sessions) < 3 or BASELINE not in candidate_names:
        raise ValueError("nested LOSO requires >=3 sessions and the no-shrink baseline")
    selection_counts: Counter[str] = Counter()
    per_session: dict[str, dict] = {}
    all_deltas: list[np.ndarray] = []
    for heldout in sessions:
        train_sessions = [session for session in sessions if session != heldout]
        objectives = {
            candidate: float(
                np.mean(
                    [
                        np.mean(by_session[session]["log_mse"][candidate])
                        for session in train_sessions
                    ]
                )
            )
            for candidate in candidate_names
        }
        selected = min(candidate_names, key=lambda name: (objectives[name], name))
        selection_counts[selected] += 1
        baseline = by_session[heldout]["log_mse"][BASELINE]
        observed = by_session[heldout]["log_mse"][selected]
        delta = observed - baseline
        all_deltas.append(delta)
        factor = by_session[heldout]["factors"][selected]
        per_session[heldout] = {
            "selected_candidate": selected,
            "unit_count": int(delta.size),
            "mean_log_mse_delta": float(np.mean(delta)),
            "geometric_mse_ratio": float(np.exp(np.mean(delta))),
            "unit_fraction_improved": float(np.mean(delta < 0.0)),
            "mean_shrink_factor": float(np.mean(factor)),
        }
    session_deltas = np.asarray(
        [row["mean_log_mse_delta"] for row in per_session.values()],
        dtype=np.float64,
    )
    unit_deltas = np.concatenate(all_deltas)
    improved_sessions = int(np.sum(session_deltas < 0.0))
    nonbaseline_folds = len(sessions) - selection_counts.get(BASELINE, 0)
    return {
        "session_count": len(sessions),
        "unit_count": int(unit_deltas.size),
        "selection_counts": dict(sorted(selection_counts.items())),
        "nonbaseline_selection_folds": nonbaseline_folds,
        "mean_session_log_mse_delta": float(np.mean(session_deltas)),
        "geometric_session_mse_ratio": float(np.exp(np.mean(session_deltas))),
        "median_session_log_mse_delta": float(np.median(session_deltas)),
        "sessions_improved": improved_sessions,
        "global_unit_log_mse_delta": float(np.mean(unit_deltas)),
        "global_unit_fraction_improved": float(np.mean(unit_deltas < 0.0)),
        "candidate_for_decoding_pilot": bool(
            np.mean(session_deltas) <= math.log(0.98)
            and improved_sessions >= 18
            and nonbaseline_folds >= 18
        ),
        "per_session": per_session,
    }


def _fixed_candidate_summary(
    by_session: dict[str, dict],
    candidate: str,
) -> dict:
    deltas = []
    session_deltas = []
    per_session = {}
    for session in sorted(by_session):
        delta = (
            by_session[session]["log_mse"][candidate]
            - by_session[session]["log_mse"][BASELINE]
        )
        deltas.append(delta)
        session_deltas.append(float(np.mean(delta)))
        per_session[session] = {
            "mean_log_mse_delta": float(np.mean(delta)),
            "geometric_mse_ratio": float(np.exp(np.mean(delta))),
            "unit_fraction_improved": float(np.mean(delta < 0.0)),
        }
    return {
        "candidate": candidate,
        "mean_session_log_mse_delta": float(np.mean(session_deltas)),
        "geometric_session_mse_ratio": float(np.exp(np.mean(session_deltas))),
        "sessions_improved": int(np.sum(np.asarray(session_deltas) < 0.0)),
        "global_unit_fraction_improved": float(
            np.mean(np.concatenate(deltas) < 0.0)
        ),
        "per_session": per_session,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=(10, 15, 20, 30),
    )
    parser.add_argument("--reference-pool", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(budget >= args.reference_pool for budget in args.budgets):
        raise ValueError("every budget must be smaller than reference_pool")

    manifest_path = args.manifest.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = list(manifest["session_splits"]["train"])
    if len(sessions) != 27 or len(set(sessions)) != 27:
        raise ValueError("shrinkage audit requires the strict 27-session train split")
    candidates = _candidate_grid(DEFAULT_LAMBDAS)
    by_budget: dict[int, dict[str, dict]] = {
        budget: {} for budget in args.budgets
    }
    receipts: dict[str, str] = {}
    for index, session in enumerate(sessions, start=1):
        path = (data_dir / f"{session}_behavior+ecephys.nwb").resolve()
        if path.parent != data_dir or not path.is_file():
            raise FileNotFoundError(path)
        scores = _session_candidate_scores(
            path,
            budgets=args.budgets,
            reference_pool=args.reference_pool,
            candidates=candidates,
        )
        for budget, values in scores.items():
            by_budget[budget][session] = values
        receipts[session] = str(path)
        print(f"[{index:02d}/{len(sessions)}] {session}", flush=True)

    results = {}
    for budget, by_session in by_budget.items():
        candidate_names = tuple(candidates)
        results[str(budget)] = {
            "nested_loso": nested_loso_summary(
                by_session,
                candidate_names,
            ),
            "fixed_wiener_strength_1": _fixed_candidate_summary(
                by_session,
                "wiener:1",
            ),
            "fixed_positive_part_strength_1": _fixed_candidate_summary(
                by_session,
                "positive_part:1",
            ),
        }

    result = {
        "schema_version": 1,
        "purpose": "train_only_low_budget_t4_uncertainty_shrinkage",
        "interpretation_boundary": (
            "Mechanistic future-rate proxy only; does not establish decoding gain."
        ),
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "data_dir": str(data_dir),
            "split": "train",
            "session_count": 27,
            "budgets": list(args.budgets),
            "reference_pool": args.reference_pool,
            "fit_trials": "chronological_rewarded_trials[0:M]",
            "future_score_trials": "chronological_rewarded_trials[M:50]",
            "coefficient_prior": "population-symmetric a=c=0; intercept b unchanged",
            "nested_selection": (
                "leave-one-training-session-out; equal session weight; "
                "minimize mean unit log future-rate MSE"
            ),
            "candidate_families": {
                "wiener": "s/(s+lambda*u)",
                "positive_part": "max(0,1-lambda*u/s)",
            },
            "lambda_grid": list(DEFAULT_LAMBDAS),
            "validation_session_nwb_opened": False,
            "formal_test_session_nwb_opened": False,
        },
        "budgets": results,
        "train_nwb_receipts": receipts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                budget: {
                    key: values["nested_loso"][key]
                    for key in (
                        "geometric_session_mse_ratio",
                        "sessions_improved",
                        "nonbaseline_selection_folds",
                        "candidate_for_decoding_pilot",
                    )
                }
                for budget, values in results.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
