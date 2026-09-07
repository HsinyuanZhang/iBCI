"""Pure contracts for the M1 B20 source-only characterization.

This is deliberately a *characterization* module, not a decoder feature or a
data loader.  It accepts already validated first-ten raw calibration trials and
the scorer's later-neural target separately.  It never opens NWB files,
imports a Falcon data module, allocates CUDA state, or constructs a decoder.

The experiment separates the two halves of the A2 B20 marginal baseline:

* ``R10``: sorted exposure-normalized trial log-rates, zero-padded to width 20;
* ``Q10``: pooled per-bin ``log1p(count)`` quantiles, zero-padded to width 20;
* ``B20``: their exact concatenation; and
* a simple, width-matched aggregate rate-only descriptive control.

It also defines a session-specific *unit-row attachment* null.  The null moves
an entire B20 row to another unit within the same session, so every marginal
column distribution is exactly preserved while its attachment to the future
unit target is broken.  This answers a narrower question than a decoder
experiment: whether B20 has unit-attached source-only predictive information.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from collections import Counter
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t


B20_CHARACTERIZATION_SEMANTICS_VERSION = "m1_b20_source_characterization_v1"
BIN_SECONDS = 0.020
SUPPORT_TRIALS = 10
WIDTH = 20
RIDGE = 1.0
ROW_ATTACHMENT_RANDOM_SCHEDULES = 4095
# The score matrix includes canonical identity at row zero plus every random
# schedule.  The protocol's ``b`` indexes only the 4,095 random rows.
ROW_ATTACHMENT_TOTAL_SCHEDULES = ROW_ATTACHMENT_RANDOM_SCHEDULES + 1
# Backwards-compatible public alias; it means total scored schedules, not the
# number of random permutations.
ROW_ATTACHMENT_NULL_REPLICATES = ROW_ATTACHMENT_TOTAL_SCHEDULES
ROW_ATTACHMENT_NULL_SEED_NAMESPACE = "m1-b20-unit-row-attachment-null-v1"
THINNING_SEED_NAMESPACE = "m1-b20-complementary-thinning-v1"
RELIABILITY_REPEATS = 1024
BONFERRONI_ALPHA = 1.0 / 60.0
PRACTICAL_COMPONENT_DELTA_R2 = 0.030
A2_B20_R2_BY_SESSION: dict[str, float] = {
    "ses-20120924": 0.8924489340223727,
    "ses-20120926": 0.9425713886560603,
    "ses-20120927": 0.9634109918259469,
    "ses-20120928": 0.8205992848030775,
}
_EPS = 1.0e-12
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
R10_COLUMNS: tuple[str, ...] = tuple(f"sorted_trial_log_rate_{index:02d}" for index in range(1, 11))
Q10_COLUMNS: tuple[str, ...] = tuple(f"pooled_log1p_count_quantile_{level:.2f}" for level in QUANTILE_LEVELS)


def _seed_int(namespace: str, *parts: object) -> int:
    payload = ":".join((namespace, *(str(value) for value in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _integer_trials(value: Sequence[np.ndarray], *, name: str) -> tuple[np.ndarray, ...]:
    """Validate unpadded [valid bins, unit] nonnegative integer spike counts."""
    if isinstance(value, (str, bytes, np.ndarray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of raw trial matrices")
    if len(value) != SUPPORT_TRIALS:
        raise ValueError(f"{name} is locked to exactly {SUPPORT_TRIALS} support trials")
    output: list[np.ndarray] = []
    unit_count: int | None = None
    for index, raw in enumerate(value):
        array = np.asarray(raw)
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError(f"{name}[{index}] must be nonempty [valid bins, units]")
        if not np.isfinite(array).all() or np.any(array < 0) or not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{name}[{index}] must contain finite nonnegative integer counts")
        counts = array.astype(np.int64, copy=False)
        if unit_count is None:
            unit_count = int(counts.shape[1])
        if counts.shape[1] != unit_count:
            raise ValueError(f"{name} has inconsistent unit dimensions")
        output.append(counts)
    return tuple(output)


def _finite_matrix(value: np.ndarray, *, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] <= 0 or (width is not None and array.shape[1] != width):
        raise ValueError(f"{name} has invalid matrix shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def r10_carrier(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Return API-width-20 R10 (effective rank <=10), padded in 10..19."""
    checked = _integer_trials(trials, name="r10_trials")
    log_rates = np.stack(
        [np.log((trial.sum(axis=0, dtype=np.float64) + 0.5) / (BIN_SECONDS * trial.shape[0])) for trial in checked],
        axis=0,
    )
    output = np.zeros((log_rates.shape[1], WIDTH), dtype=np.float64)
    output[:, :SUPPORT_TRIALS] = np.sort(log_rates, axis=0).T
    return output


def q10_carrier(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Return API-width-20 Q10 (effective rank <=10), padded in 0..9."""
    checked = _integer_trials(trials, name="q10_trials")
    pooled = np.log1p(np.concatenate(checked, axis=0).astype(np.float64, copy=False))
    # ``method`` is explicit to make the B20 coordinate contract independent of
    # a possible future NumPy default change.
    quantiles = np.quantile(pooled, QUANTILE_LEVELS, axis=0, method="linear")
    output = np.zeros((pooled.shape[1], WIDTH), dtype=np.float64)
    output[:, SUPPORT_TRIALS:] = quantiles.T
    return output


def b20_carrier(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Return B20 = R10 + Q10 without any labels, timestamps, or unit ID."""
    r10 = r10_carrier(trials)
    q10 = q10_carrier(trials)
    output = r10 + q10
    if output.shape[1] != WIDTH or not np.isfinite(output).all():
        raise RuntimeError("B20 carrier construction failed")
    return output


def rate_only_carrier(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Return a width-20 aggregate-rate/exposure descriptive control.

    The first three coordinates are pooled mean count/bin, its standard
    deviation, and log valid-bin exposure.  The remaining coordinates are
    exactly zero.  Unlike B20 this is intentionally low-order.
    """
    checked = _integer_trials(trials, name="rate_only_trials")
    pooled = np.concatenate(checked, axis=0).astype(np.float64, copy=False)
    output = np.zeros((pooled.shape[1], WIDTH), dtype=np.float64)
    output[:, 0] = pooled.mean(axis=0)
    output[:, 1] = pooled.std(axis=0)
    output[:, 2] = math.log(float(pooled.shape[0]))
    return output


def b20_component_receipt() -> dict[str, object]:
    return {
        "width": WIDTH,
        "support_trials": SUPPORT_TRIALS,
        "r10": {"columns": list(R10_COLUMNS), "zero_padding_columns": list(range(10, 20))},
        "q10": {
            "columns": list(Q10_COLUMNS), "quantiles": list(QUANTILE_LEVELS),
            "quantile_interpolation": "numpy_linear", "zero_padding_columns": list(range(0, 10)),
        },
        "b20": "R10 + Q10, exactly width 20",
        "component_padding": "R10/Q10 zero-padding is API compatibility only; it does not parameter-match B20 and cannot support a full-vs-component coefficient-count claim.",
        "support_labels": "none",
        "future_labels": "forbidden from every carrier and null plan",
        "trial_and_within_trial_bin_order_invariant": True,
    }


def deterministic_unit_row_attachment_permutation(*, session_name: str, replicate: int, units: int) -> np.ndarray:
    """Return the session-specific full-row attachment permutation.

    Replicate zero is the observed identity.  All later replicates are
    independent deterministic uniform permutations and retain accidental
    identities/duplicates, preserving the finite Monte-Carlo denominator.
    The caller applies ``carrier[order, :]``; all 20 coordinates move together.
    """
    if not session_name or int(replicate) < 0 or int(units) <= 0:
        raise ValueError("row attachment requires a session, nonnegative replicate, and positive units")
    if int(replicate) == 0:
        return np.arange(int(units), dtype=np.int64)
    generator = np.random.default_rng(_seed_int(ROW_ATTACHMENT_NULL_SEED_NAMESPACE, session_name, int(replicate), int(units)))
    return generator.permutation(int(units)).astype(np.int64, copy=False)


def attachment_shuffle_carrier(carrier: np.ndarray, *, session_name: str, replicate: int, family: str = "A-all") -> np.ndarray:
    """Apply one of the three precommitted B20 block row-attachment nulls.

    ``A-all`` moves the complete B20 row.  ``A-R`` and ``A-Q`` move exactly one
    complete ten-coordinate block while leaving the complementary block in its
    original unit row.  Every operation stays within a session and preserves
    each column marginal exactly.
    """
    values = _finite_matrix(carrier, name="attachment_carrier", width=WIDTH)
    order = deterministic_unit_row_attachment_permutation(
        session_name=session_name, replicate=replicate, units=values.shape[0]
    )
    if family == "A-all":
        return values[order, :].copy()
    output = values.copy()
    if family == "A-R":
        output[:, :10] = values[order, :10]
    elif family == "A-Q":
        output[:, 10:] = values[order, 10:]
    else:
        raise ValueError("attachment family must be A-all, A-R, or A-Q")
    return output


def row_attachment_schedule_receipt(sessions: Sequence[str], *, units: int, replicates: int = ROW_ATTACHMENT_NULL_REPLICATES) -> dict[str, object]:
    """Hash the complete deterministic MC plan without exposing a target."""
    names = tuple(sorted(sessions))
    if len(names) != 4 or len(set(names)) != 4 or any(not name for name in names):
        raise ValueError("B20 row-attachment schedule requires exactly four unique source sessions")
    if int(replicates) != ROW_ATTACHMENT_TOTAL_SCHEDULES:
        raise ValueError("B20 row-attachment null count is frozen to identity plus 4,095 random schedules")
    digests: list[str] = []
    identity_schedule_indices: list[int] = []
    for replicate in range(int(replicates)):
        lines: list[str] = []
        all_identity = True
        for session in names:
            order = deterministic_unit_row_attachment_permutation(session_name=session, replicate=replicate, units=units)
            all_identity = all_identity and np.array_equal(order, np.arange(units))
            lines.append(f"{replicate}:{session}:{','.join(map(str, order.tolist()))}")
        digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        digests.append(digest)
        if all_identity:
            identity_schedule_indices.append(replicate)
    counts = Counter(digests)
    duplicates = {digest: count for digest, count in sorted(counts.items()) if count > 1}
    return {
        "seed_namespace": ROW_ATTACHMENT_NULL_SEED_NAMESPACE,
        "replicates_including_observed_identity": int(replicates),
        "random_replicates_after_identity": ROW_ATTACHMENT_RANDOM_SCHEDULES,
        "complete_rows_moved_together": True,
        "within_session_only": True,
        "per_column_marginal_exactly_preserved": True,
        "identity_and_duplicate_draws_retained": True,
        "schedule_sha256": digests,
        "identity_schedule_indices": identity_schedule_indices,
        "duplicate_schedule_hash_multiplicities": duplicates,
    }


@dataclass(frozen=True)
class OuterB20Carriers:
    """Source-only carrier matrices indexed by outer left-out session."""

    public: dict[str, dict[str, dict[str, np.ndarray]]]


def build_outer_loso_carriers(support_trials: Mapping[str, Sequence[np.ndarray]]) -> OuterB20Carriers:
    """Create width-20 arms from support only; targets cannot enter this API."""
    names = tuple(sorted(support_trials))
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("B20 source characterization requires exactly four source sessions")
    validated = {name: _integer_trials(support_trials[name], name=f"support[{name}]") for name in names}
    raw_by_arm = {
        "R10": {name: r10_carrier(validated[name]) for name in names},
        "Q10": {name: q10_carrier(validated[name]) for name in names},
        "B20": {name: b20_carrier(validated[name]) for name in names},
        "rate_only": {name: rate_only_carrier(validated[name]) for name in names},
    }
    # The representation itself has no fold-dependent fitted state.  Retaining
    # this fold axis nevertheless locks the source-only LOSO readout interface
    # and makes every arm use exactly the same rows in every fold.
    public = {
        left_out: {arm: {name: values.copy() for name, values in by_session.items()} for arm, by_session in raw_by_arm.items()}
        for left_out in names
    }
    return OuterB20Carriers(public=public)


@dataclass(frozen=True)
class RidgeReadout:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    target_mean: np.ndarray


def _fit_ridge(features: np.ndarray, targets: np.ndarray, *, ridge: float = RIDGE) -> RidgeReadout:
    x = _finite_matrix(features, name="ridge_features", width=WIDTH)
    y = _finite_matrix(targets, name="ridge_targets")
    if x.shape[0] != y.shape[0] or not math.isfinite(float(ridge)) or float(ridge) <= 0:
        raise ValueError("ridge features/targets mismatch or invalid ridge")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= _EPS] = 1.0
    standardized = (x - mean) / scale
    target_mean = y.mean(axis=0)
    weights = np.linalg.solve(standardized.T @ standardized + float(ridge) * np.eye(WIDTH), standardized.T @ (y - target_mean))
    return RidgeReadout(mean=mean, scale=scale, weights=weights, target_mean=target_mean)


def _predict_ridge(readout: RidgeReadout, features: np.ndarray) -> np.ndarray:
    x = _finite_matrix(features, name="predict_features", width=WIDTH)
    return ((x - readout.mean) / readout.scale) @ readout.weights + readout.target_mean


def bounded_utility(*, train_targets: np.ndarray, test_targets: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float, float]:
    """Return raw R² and bounded U=TSS/(TSS+RSS), with source-only centering."""
    train = _finite_matrix(train_targets, name="source_train_targets")
    test = _finite_matrix(test_targets, name="source_test_targets")
    predicted = _finite_matrix(prediction, name="source_prediction")
    if train.shape[1] != test.shape[1] or predicted.shape != test.shape:
        raise ValueError("source target dimensions do not match")
    baseline = train.mean(axis=0, keepdims=True)
    tss = float(np.square(test - baseline).sum())
    rss = float(np.square(test - predicted).sum())
    if not math.isfinite(tss) or not math.isfinite(rss) or tss <= 0.0 or rss < 0.0:
        raise RuntimeError("invalid source-only RSS/TSS")
    return 1.0 - rss / tss, tss / (tss + rss), tss, rss


def outer_loso_proxy(
    carriers: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], targets: Mapping[str, np.ndarray], *, ridge: float = RIDGE
) -> dict[str, object]:
    """Score only an already-built carrier with a source-only outer-LOSO ridge."""
    sessions = tuple(sorted(targets))
    if len(sessions) != 4 or set(carriers) != set(sessions):
        raise ValueError("B20 proxy requires four matching source LOSO folds")
    target_values = {name: _finite_matrix(targets[name], name=f"target[{name}]") for name in sessions}
    first_arms: set[str] | None = None
    result: dict[str, list[dict[str, object]]] = {}
    for left_out in sessions:
        fold = carriers[left_out]
        arms = set(fold)
        if first_arms is None:
            first_arms = arms
            result = {arm: [] for arm in sorted(arms)}
        elif arms != first_arms:
            raise ValueError("B20 arms differ across source folds")
        source_sessions = tuple(name for name in sessions if name != left_out)
        for arm in sorted(arms):
            if set(fold[arm]) != set(sessions):
                raise ValueError("B20 carrier session set mismatch")
            train_x = np.concatenate([_finite_matrix(fold[arm][name], name=f"carrier[{left_out}][{arm}][{name}]", width=WIDTH) for name in source_sessions], axis=0)
            train_y = np.concatenate([target_values[name] for name in source_sessions], axis=0)
            test_x = _finite_matrix(fold[arm][left_out], name=f"carrier[{left_out}][{arm}][{left_out}]", width=WIDTH)
            test_y = target_values[left_out]
            if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
                raise ValueError("B20 carrier/target unit rows disagree")
            r2, utility, tss, rss = bounded_utility(
                train_targets=train_y, test_targets=test_y, prediction=_predict_ridge(_fit_ridge(train_x, train_y, ridge=ridge), test_x)
            )
            result[arm].append({"left_out_session": left_out, "r2": r2, "bounded_u": utility, "tss": tss, "rss": rss})
    return {"arms": result, "outer_loso": "four source sessions; target enters scorer/readout only", "ridge": float(ridge)}


def attachment_shuffled_outer_carriers(
    base: OuterB20Carriers, *, replicate: int, family: str = "A-all"
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Construct one precommitted B20 attachment-null family for one MC draw."""
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for left_out, fold in base.public.items():
        if "B20" not in fold:
            raise ValueError("B20 arm missing from base carrier")
        result[left_out] = {
            family: {
                session: attachment_shuffle_carrier(value, session_name=session, replicate=replicate, family=family)
                for session, value in fold["B20"].items()
            }
        }
    return result


def monte_carlo_upper_rank(observed: float, null_values: Sequence[float]) -> dict[str, object]:
    """Conservative upper-tail MC rank, ties against B20, over the frozen 4096 plan."""
    values = np.asarray(null_values, dtype=np.float64)
    if not math.isfinite(float(observed)) or values.shape != (ROW_ATTACHMENT_TOTAL_SCHEDULES,) or not np.isfinite(values).all():
        raise ValueError("B20 MC rank requires 4,096 values: observed identity plus exactly 4,095 finite random bounded utilities")
    if np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("bounded utility must lie in (0,1]")
    exceedances = int(np.count_nonzero(values[1:] >= float(observed)))
    p_value = float((1 + exceedances) / ROW_ATTACHMENT_TOTAL_SCHEDULES)
    failures = ROW_ATTACHMENT_RANDOM_SCHEDULES - exceedances
    upper = 1.0 if failures == 0 else float(beta_distribution.ppf(0.975, exceedances + 1, failures))
    return {
        "observed_bounded_u": float(observed), "exceedances_among_random_4095": exceedances,
        "conservative_upper_tail_p_value": p_value,
        "one_sided_97p5_binomial_upper_bound": upper,
        "random_schedule_count": ROW_ATTACHMENT_RANDOM_SCHEDULES,
        "null_count_including_observed_identity": ROW_ATTACHMENT_TOTAL_SCHEDULES,
        "ties_count_against_candidate": True,
    }


def family_distinguishable(
    *, observed_u_by_session: Mapping[str, float], null_u_by_session: Mapping[str, Sequence[float]], rank: Mapping[str, object]
) -> dict[str, object]:
    """Frozen family criterion: Bonferroni rank/MC bound plus four-session consistency."""
    sessions = tuple(sorted(observed_u_by_session))
    if len(sessions) != 4 or set(null_u_by_session) != set(sessions):
        raise ValueError("family decision requires four aligned source sessions")
    above_median: dict[str, bool] = {}
    for session in sessions:
        values = np.asarray(null_u_by_session[session], dtype=np.float64)
        if values.shape != (ROW_ATTACHMENT_RANDOM_SCHEDULES,) or not np.isfinite(values).all():
            raise ValueError("every B20 family requires 4,095 finite random per-session U values")
        observed = float(observed_u_by_session[session])
        above_median[session] = bool(math.isfinite(observed) and observed > float(np.median(values)))
    p_value = float(rank["conservative_upper_tail_p_value"])
    upper = float(rank["one_sided_97p5_binomial_upper_bound"])
    passed = bool(p_value < BONFERRONI_ALPHA and upper < BONFERRONI_ALPHA and all(above_median.values()))
    return {
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "p_value": p_value,
        "mc_upper_bound_97p5": upper,
        "canonical_U_strictly_above_random_median_by_session": above_median,
        "distinguishable": passed,
    }


def terminal_characterization_decision(
    *, valid: bool, a2_reproduction_pass: bool, all_family: Mapping[str, object] | None,
    r_family: Mapping[str, object] | None, q_family: Mapping[str, object] | None,
    d_r_by_session: Sequence[float] | None, d_q_by_session: Sequence[float] | None,
) -> dict[str, object]:
    """Implement the protocol's ordered terminal decision table exactly.

    The reviewed table gives the both-negligible R10 lower-state-hardware-prior
    outcome precedence over either one-component simplification outcome. Missing
    component output is legal only as the prescribed A-all early stop.
    """
    common = {"gpu_authorized": False, "decoder_authorized": False, "formal_held_out": False, "evalai_authorized": False}
    if not valid:
        return {"decision": "invalid_execution_stop", **common}
    if not a2_reproduction_pass:
        return {"decision": "a2_b20_reproduction_mismatch_invalid_stop", **common}
    if all_family is None or not bool(all_family.get("distinguishable")):
        return {"decision": "b20_row_attachment_not_distinguishable_stop", "component_families": "not_run_due_to_predeclared_early_stop", **common}
    if r_family is None or q_family is None or d_r_by_session is None or d_q_by_session is None:
        return {"decision": "invalid_execution_stop", "reason": "A-all passed but both complete component families were not supplied", **common}
    d_r = np.asarray(d_r_by_session, dtype=np.float64)
    d_q = np.asarray(d_q_by_session, dtype=np.float64)
    if d_r.shape != (4,) or d_q.shape != (4,) or not np.isfinite(d_r).all() or not np.isfinite(d_q).all():
        return {"decision": "invalid_execution_stop", "reason": "component practical losses must be four finite session values", **common}
    both_negligible = bool(
        not bool(r_family.get("distinguishable")) and not bool(q_family.get("distinguishable"))
        and np.all(d_r < PRACTICAL_COMPONENT_DELTA_R2) and np.all(d_q < PRACTICAL_COMPONENT_DELTA_R2)
    )
    if both_negligible:
        return {
            "decision": "b20_redundant_or_provenance_unresolved_retain_R10_lower_state_candidate_separate_independent_validation",
            "protocol_rule": "both A-R and A-Q not distinguishable; every D_R and D_Q < 0.030; R10 fixed lower-state hardware prior",
            **common,
        }
    if not bool(q_family.get("distinguishable")) and bool(np.all(d_q < PRACTICAL_COMPONENT_DELTA_R2)):
        return {
            "decision": "b20_simplify_to_R10_candidate_separate_independent_validation",
            "protocol_rule": "A-all passes; A-Q not distinguishable and every D_Q < 0.030",
            **common,
        }
    if not bool(r_family.get("distinguishable")) and bool(np.all(d_r < PRACTICAL_COMPONENT_DELTA_R2)):
        return {
            "decision": "b20_simplify_to_Q10_candidate_separate_independent_validation",
            "protocol_rule": "A-all passes; A-R not distinguishable and every D_R < 0.030",
            **common,
        }
    if bool(r_family.get("distinguishable")) and bool(q_family.get("distinguishable")) and bool(np.all(d_r >= PRACTICAL_COMPONENT_DELTA_R2)) and bool(np.all(d_q >= PRACTICAL_COMPONENT_DELTA_R2)):
        return {
            "decision": "b20_both_blocks_row_attached_source_characterization_only",
            "protocol_rule": "A-all/A-R/A-Q distinguishable; every D_R and D_Q >= 0.030",
            **common,
        }
    return {"decision": "b20_component_attribution_indeterminate_stop", **common}


def a2_b20_reproduction_receipt(proxy: Mapping[str, object], *, tolerance: float = 1.0e-10) -> dict[str, object]:
    """Compare fresh canonical B20 source-LOSO R² against the immutable A2 checkpoint."""
    arms = proxy.get("arms") if isinstance(proxy, Mapping) else None
    rows = arms.get("B20") if isinstance(arms, Mapping) else None
    if not isinstance(rows, Sequence) or len(rows) != 4:
        raise ValueError("A2 B20 reproduction needs four fresh canonical B20 rows")
    observed = {str(row.get("left_out_session")): float(row.get("r2")) for row in rows if isinstance(row, Mapping)}
    if set(observed) != set(A2_B20_R2_BY_SESSION):
        raise ValueError("A2 B20 reproduction sessions mismatch")
    absolute_error = {session: abs(observed[session] - expected) for session, expected in A2_B20_R2_BY_SESSION.items()}
    return {
        "expected_r2_by_session": A2_B20_R2_BY_SESSION,
        "observed_r2_by_session": observed,
        "absolute_error_by_session": absolute_error,
        "absolute_tolerance": float(tolerance),
        "pass": bool(all(value <= float(tolerance) for value in absolute_error.values())),
    }


def pooled_split_standardized_cosines(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    """Apply the protocol's pooled-two-split row cosine reporting rule.

    Coordinate ``j`` is centered and scaled using the pooled 128 rows of both
    splits.  A coordinate with zero/nonfinite pooled scale remains zero in
    both standardized arrays and is explicitly reported as undefined in the
    Pearson table.  Rows are defined only when both resulting standardized
    vectors have positive finite norm.
    """
    a = _finite_matrix(first, name="reliability_first", width=WIDTH)
    b = _finite_matrix(second, name="reliability_second", width=WIDTH)
    if a.shape != b.shape:
        raise ValueError("reliability halves have mismatched unit rows")
    pooled = np.concatenate((a, b), axis=0)
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    valid_coordinate = np.isfinite(scale) & (scale > _EPS)
    standardized_a = np.zeros_like(a)
    standardized_b = np.zeros_like(b)
    standardized_a[:, valid_coordinate] = (a[:, valid_coordinate] - center[valid_coordinate]) / scale[valid_coordinate]
    standardized_b[:, valid_coordinate] = (b[:, valid_coordinate] - center[valid_coordinate]) / scale[valid_coordinate]
    norm_a = np.linalg.norm(standardized_a, axis=1)
    norm_b = np.linalg.norm(standardized_b, axis=1)
    defined = np.isfinite(norm_a) & np.isfinite(norm_b) & (norm_a > _EPS) & (norm_b > _EPS)
    cosines = np.full(a.shape[0], np.nan, dtype=np.float64)
    cosines[defined] = (standardized_a[defined] * standardized_b[defined]).sum(axis=1) / (norm_a[defined] * norm_b[defined])
    pearson: list[float | None] = []
    pearson_defined: list[bool] = []
    for column in range(WIDTH):
        left = a[:, column]
        right = b[:, column]
        left_std = float(left.std())
        right_std = float(right.std())
        if not (math.isfinite(left_std) and math.isfinite(right_std) and left_std > _EPS and right_std > _EPS):
            pearson.append(None); pearson_defined.append(False)
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
            pearson.append(correlation if math.isfinite(correlation) else None)
            pearson_defined.append(math.isfinite(correlation))
    return {
        "unit_cosines": cosines,
        "defined_row_fraction": float(defined.mean()),
        "defined_row_count": int(defined.sum()),
        "row_count": int(a.shape[0]),
        "per_coordinate_pearson": pearson,
        "per_coordinate_pearson_defined": pearson_defined,
        "pooled_split_standardization": "per-coordinate mean/std over concatenated two-split 64-row matrices",
    }


def _nanmedian_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _nanquantiles_or_nan(values: Sequence[float]) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    return [float(np.quantile(finite, q)) for q in (0.025, 0.5, 0.975)] if finite.size else [float("nan")] * 3


def complementary_binomial_thinning(trials: Sequence[np.ndarray], *, session_name: str, repeat: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Complementary 1/2 spike thinning, rescaled by x2 before B20 construction."""
    checked = _integer_trials(trials, name="thinning_trials")
    if not session_name or int(repeat) < 0:
        raise ValueError("thinning needs session and nonnegative repeat")
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for index, trial in enumerate(checked):
        generator = np.random.default_rng(_seed_int(THINNING_SEED_NAMESPACE, session_name, int(repeat), index, *trial.shape))
        first = generator.binomial(trial, 0.5).astype(np.int64, copy=False)
        left.append(2 * first)
        right.append(2 * (trial - first))
    return tuple(left), tuple(right)


def odd_even_bin_split(trials: Sequence[np.ndarray]) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Diagnostic bin partition; no labels and no random sampling."""
    checked = _integer_trials(trials, name="odd_even_trials")
    if any(trial.shape[0] < 2 for trial in checked):
        raise ValueError("odd/even B20 diagnostic needs at least two valid bins in every trial")
    return tuple(trial[0::2, :].copy() for trial in checked), tuple(trial[1::2, :].copy() for trial in checked)


def b20_split_reliability(
    trials: Sequence[np.ndarray], *, session_name: str, repeats: int = RELIABILITY_REPEATS
) -> dict[str, object]:
    """Report B20/R10/Q10 split reliability; it is diagnostic, not n inflation."""
    checked = _integer_trials(trials, name="reliability_trials")
    if int(repeats) <= 0:
        raise ValueError("B20 thinning repeat count must be positive")
    arm_builders = {"R10": r10_carrier, "Q10": q10_carrier, "B20": b20_carrier}
    thinning_median: dict[str, list[float]] = {name: [] for name in arm_builders}
    thinning_defined_fraction: dict[str, list[float]] = {name: [] for name in arm_builders}
    thinning_pearson: dict[str, list[list[float | None]]] = {name: [] for name in arm_builders}
    for repeat in range(int(repeats)):
        first, second = complementary_binomial_thinning(checked, session_name=session_name, repeat=repeat)
        for name, builder in arm_builders.items():
            metric = pooled_split_standardized_cosines(builder(first), builder(second))
            cosines = np.asarray(metric["unit_cosines"], dtype=np.float64)
            thinning_median[name].append(_nanmedian_or_nan(cosines))
            thinning_defined_fraction[name].append(float(metric["defined_row_fraction"]))
            thinning_pearson[name].append(list(metric["per_coordinate_pearson"]))
    odd, even = odd_even_bin_split(checked)
    odd_even = {name: pooled_split_standardized_cosines(builder(odd), builder(even)) for name, builder in arm_builders.items()}
    return {
        "thinning": {
            "method": "complementary_binomial_p0p5_rescaled_x2", "repeats": int(repeats),
            "seed_namespace": THINNING_SEED_NAMESPACE,
            "median_row_cosine_by_repeat": thinning_median,
            "median_row_cosine_quantiles_2p5_50_97p5": {
                name: _nanquantiles_or_nan(values) for name, values in thinning_median.items()
            },
            "defined_row_fraction_by_repeat": thinning_defined_fraction,
            "minimum_defined_row_fraction": {name: float(np.nanmin(values)) for name, values in thinning_defined_fraction.items()},
            "per_coordinate_pearson_by_repeat_including_undefined": thinning_pearson,
        },
        "odd_even_bin_diagnostic": {
            "median_row_cosine": {name: _nanmedian_or_nan(np.asarray(value["unit_cosines"], dtype=np.float64)) for name, value in odd_even.items()},
            "defined_row_fraction": {name: float(value["defined_row_fraction"]) for name, value in odd_even.items()},
            "per_coordinate_pearson_including_undefined": {name: value["per_coordinate_pearson"] for name, value in odd_even.items()},
            "rescaled": False,
        },
        "independent_biological_samples_added": 0,
    }


def paired_summary(candidate: Sequence[float], control: Sequence[float]) -> dict[str, object]:
    """Four-session descriptive paired sensitivity summary; no gate is imposed here."""
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("B20 paired summary requires four finite source-session values")
    delta = a - b
    sd = float(np.std(delta, ddof=1))
    t975 = float(student_t.ppf(0.975, df=3))
    t80 = float(student_t.ppf(0.80, df=3))
    sem = sd / math.sqrt(4.0)
    return {
        "delta_by_session": delta.tolist(), "mean": float(delta.mean()), "median": float(np.median(delta)), "sd": sd,
        "paired_ci95": [float(delta.mean() - t975 * sem), float(delta.mean() + t975 * sem)],
        "mde80": float((t975 + t80) * sem), "biological_unit": "outer source session (n=4)",
    }


def characterization_contract_receipt() -> dict[str, object]:
    """Record frozen mechanics while explicitly leaving numerical gates unfrozen."""
    return {
        "semantics_version": B20_CHARACTERIZATION_SEMANTICS_VERSION,
        "carrier": b20_component_receipt(),
        "outer_loso": {"sessions": 4, "ridge": RIDGE, "width": WIDTH, "target": "future scorer-only"},
        "attachment_null": {
            "replicates": ROW_ATTACHMENT_NULL_REPLICATES, "statistic": "mean source-session bounded U=TSS/(TSS+RSS)",
            "p_value": "(1 + count(random U >= observed U))/4096; ties against candidate",
        },
        "split_reliability": {"primary": "complementary binomial thinning x2", "diagnostic": "odd/even bins"},
        "numeric_gate_status": "UNFROZEN: no data execution, result decision, GPU, decoder, held-out, or EvalAI authorization",
    }
