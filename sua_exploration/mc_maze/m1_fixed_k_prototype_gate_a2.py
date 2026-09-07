"""Pure, source-only controls for the fixed-K temporal-prototype Gate A2.

This module is intentionally data-loader free: it accepts already validated
first-ten raw trial bins and later-neural targets from a caller.  It never
opens NWB files, imports a datamodule, creates a decoder, or uses CUDA.  The
separation makes the three A2 controls auditable before any native M1 data are
opened:

* an exact, session-keyed whole-slot quotient permutation null;
* a within-trial temporal-order null with one bin permutation shared by all
  units in a trial; and
* a fixed width-20 marginal-distribution carrier with no temporal order.

The controls deliberately retain the source-only outer-LOSO dimension.  In
particular, a time null re-fits its anchors from time-permuted *outer-train*
support, rather than scoring a representation under anchors fitted to the
unpermuted data.  That distinction is necessary for the null to test temporal
order rather than an arbitrary coordinate mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from collections import Counter
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

from mc_maze.fixed_k_temporal_prototypes import (
    FIXED_K,
    TEMPORAL_RANK,
    _as_finite_matrix,
    carrier_from_support_trials,
    fit_ordered_anchors_source_only,
    fit_ridge_readout,
    predict_ridge_readout,
    r2_from_source_baseline,
)
from mc_maze.m1_fixed_k_prototype_gate_a import BIN_SECONDS, _as_integer_counts


A2_SEMANTICS_VERSION = "m1_fixed_k_temporal_prototype_gate_a2_controls_v1"
SLOT_NULL_SEED_NAMESPACE = "m1-fixed-k-gate-a2-slot-null-v1"
TIME_NULL_SEED_NAMESPACE = "m1-fixed-k-gate-a2-time-null-v1"
# The slot null is exhaustively enumerated over the quotient (S4)^4 / S4.
# A common slot permutation of every session is merely one global feature-column
# permutation, which a source-normalized ridge readout cannot distinguish.  We
# thus fix the lexicographically first source session to identity and enumerate
# the remaining three independent S4 permutations: 24**3 = 13,824 classes.
SLOT_NULL_REPLICATES = 24 ** 3
# The time-order reference has one explicit observed identity schedule plus
# 4,095 deterministic sampled schedules.  Duplicate sampled permutations and
# sampled identities are intentionally retained rather than silently replaced.
TIME_ORDER_NULL_REPLICATES = 4096
TIME_ORDER_NULL_RANDOM_SCHEDULES = 4095
RIDGE = 1.0
WIDTH = FIXED_K * (TEMPORAL_RANK + 1)
_EPS = 1.0e-12

# Column ordering is part of the experiment contract.  All quantities are
# computed from the first-ten raw support trials only.  They contain no
# timestamps/positions/labels and remain unchanged by the time-order null.
MARGINAL_CONTROL_COLUMNS: tuple[str, ...] = (
    "sorted_trial_log_rate_01",
    "sorted_trial_log_rate_02",
    "sorted_trial_log_rate_03",
    "sorted_trial_log_rate_04",
    "sorted_trial_log_rate_05",
    "sorted_trial_log_rate_06",
    "sorted_trial_log_rate_07",
    "sorted_trial_log_rate_08",
    "sorted_trial_log_rate_09",
    "sorted_trial_log_rate_10",
    "pooled_log1p_count_quantile_0p05",
    "pooled_log1p_count_quantile_0p15",
    "pooled_log1p_count_quantile_0p25",
    "pooled_log1p_count_quantile_0p35",
    "pooled_log1p_count_quantile_0p45",
    "pooled_log1p_count_quantile_0p55",
    "pooled_log1p_count_quantile_0p65",
    "pooled_log1p_count_quantile_0p75",
    "pooled_log1p_count_quantile_0p85",
    "pooled_log1p_count_quantile_0p95",
)
MARGINAL_QUANTILES: tuple[float, ...] = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
_S4_PERMUTATIONS = tuple(np.asarray(row, dtype=np.int64) for row in itertools.permutations(range(FIXED_K)))


def _validated_trials(value: Sequence[np.ndarray], *, name: str) -> tuple[np.ndarray, ...]:
    """Validate nonempty variable-length raw-count trials without padding."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, np.ndarray)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence of [bins,units] raw-count trials")
    units: int | None = None
    trials: list[np.ndarray] = []
    for index, trial in enumerate(value):
        counts = _as_integer_counts(np.asarray(trial), name=f"{name}[trial={index}]")
        if counts.ndim != 2 or counts.shape[0] <= 0:
            raise ValueError(f"{name}[trial={index}] must be nonempty [bins,units]")
        if units is None:
            units = int(counts.shape[1])
        if counts.shape[1] != units:
            raise ValueError(f"{name} has inconsistent unit count")
        trials.append(counts)
    return tuple(trials)


def _seed_int(namespace: str, *parts: object) -> int:
    """Stable seed without process/global RNG state or data dependence."""
    payload = ":".join([namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def deterministic_session_slot_permutation(*, session_name: str, replicate: int) -> np.ndarray:
    """Compatibility helper: one deterministic uniform S4 element.

    The primary A2 procedure uses :func:`exact_slot_null_plans`, not this
    sampled helper.  It remains useful for small synthetic tests and is clearly
    labelled non-primary in receipts.
    """
    if not session_name or int(replicate) < 0:
        raise ValueError("slot null requires nonempty session_name and nonnegative replicate")
    generator = np.random.default_rng(_seed_int(SLOT_NULL_SEED_NAMESPACE, session_name, int(replicate)))
    return _S4_PERMUTATIONS[int(generator.integers(len(_S4_PERMUTATIONS)))].copy()


def exact_slot_null_plans(sessions: Sequence[str]) -> tuple[dict[str, np.ndarray], ...]:
    """Enumerate all 24**3 nonredundant session-keyed whole-slot nulls.

    The first lexicographic session is the reference and its order is fixed to
    identity.  This removes only the global S4 column-relabeling symmetry; all
    relative session arrangements remain.  The public namespace deterministically
    orders the exhaustive classes but does not subsample them.
    """
    names = tuple(sorted(sessions))
    if len(names) != 4 or len(set(names)) != 4 or any(not name for name in names):
        raise ValueError("A2 exact slot null requires exactly four unique nonempty session names")
    identity = np.arange(FIXED_K, dtype=np.int64)
    raw = []
    for choices in itertools.product(_S4_PERMUTATIONS, repeat=3):
        raw.append({names[0]: identity.copy(), **{names[index + 1]: choices[index].copy() for index in range(3)}})
    # itertools.product over lexicographically ordered permutations is itself
    # lexicographic.  Keeping that order allows disjoint static CPU chunks to
    # be audited without a seed-dependent subset or ordering ambiguity.
    ordered = tuple(raw)
    if len(ordered) != SLOT_NULL_REPLICATES:
        raise RuntimeError("exact slot-null quotient cardinality changed")
    return ordered


def exact_slot_chunk_ranges(*, workers: int = 8) -> tuple[tuple[int, int], ...]:
    """Partition [0, 13,824) into nonempty disjoint lexicographic chunks."""
    if int(workers) <= 0 or int(workers) > 8:
        raise ValueError("A2 exact slot enumeration permits from one through eight CPU workers")
    starts = np.linspace(0, SLOT_NULL_REPLICATES, int(workers) + 1, dtype=np.int64)
    chunks = tuple((int(starts[index]), int(starts[index + 1])) for index in range(int(workers)))
    if chunks[0][0] != 0 or chunks[-1][1] != SLOT_NULL_REPLICATES or any(start >= end for start, end in chunks):
        raise RuntimeError("invalid exact slot chunk partition")
    return chunks


def slot_null_plan_receipt(plans: Sequence[Mapping[str, np.ndarray]]) -> dict[str, object]:
    """Compact receipt for the exact quotient enumeration."""
    canonical: list[str] = []
    sessions: set[str] = set()
    for repeat, plan in enumerate(plans):
        for session in sorted(plan):
            order = np.asarray(plan[session], dtype=np.int64)
            if order.shape != (FIXED_K,) or set(order.tolist()) != set(range(FIXED_K)):
                raise ValueError("slot null plan contains an invalid S4 permutation")
            sessions.add(session)
            canonical.append(f"{repeat}:{session}:{','.join(map(str, order.tolist()))}")
    digest = hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()
    return {
        "seed_namespace": SLOT_NULL_SEED_NAMESPACE,
        "permutation_group": "(S4)^4 / global_S4_exact_quotient",
        "enumerated_classes": len(plans),
        "sessions": sorted(sessions),
        "plan_sha256": digest,
        "whole_slot_block_permutation": True,
        "shared_across_units_within_session": True,
        "shared_for_a_session_across_outer_folds": True,
        "reference_session": min(sessions) if sessions else None,
        "reference_session_permutation": "identity",
    }


def deterministic_trial_time_permutation(*, session_name: str, replicate: int, trial_index: int, bins: int) -> np.ndarray:
    """Return a deterministic permutation of a trial's time axis.

    The returned permutation is used as ``trial[order, :]``.  It is therefore
    identical across units in that trial and preserves every unit's exact
    multiset of bins, every trial/unit count sum, and synchronous population
    bin vectors.  It destroys only chronological order within a trial.
    """
    if not session_name or int(replicate) < 0 or int(trial_index) < 0 or int(bins) <= 0:
        raise ValueError("time null needs nonempty session, nonnegative indices, and positive bins")
    if int(replicate) == 0:
        return np.arange(int(bins), dtype=np.int64)
    generator = np.random.default_rng(
        _seed_int(TIME_NULL_SEED_NAMESPACE, session_name, int(replicate), int(trial_index), int(bins))
    )
    # Do not reject/rewrite identity draws.  This conditional schedule test
    # deliberately retains the explicit identity and all duplicate/identity
    # random schedules in its finite Monte-Carlo denominator.
    return generator.permutation(int(bins)).astype(np.int64, copy=False)


def within_trial_time_order_null(
    trials: Sequence[np.ndarray], *, session_name: str, replicate: int
) -> tuple[np.ndarray, ...]:
    """Permute each trial time axis once, jointly across all units; no labels."""
    checked = _validated_trials(trials, name="time_null_trials")
    return tuple(
        trial[deterministic_trial_time_permutation(
            session_name=session_name, replicate=replicate, trial_index=index, bins=trial.shape[0]
        ), :].copy()
        for index, trial in enumerate(checked)
    )


def time_null_receipt(trials_by_session: Mapping[str, Sequence[np.ndarray]], *, replicates: int = TIME_ORDER_NULL_REPLICATES) -> dict[str, object]:
    """Return a deterministic checksum receipt for the time-null schedule."""
    if len(trials_by_session) != 4:
        raise ValueError("A2 time-null receipt requires exactly four sessions")
    if int(replicates) != TIME_ORDER_NULL_REPLICATES:
        raise ValueError("A2 time-null schedule is locked to identity plus 4,095 deterministic draws")
    canonical: list[str] = []
    schedule_hashes: list[str] = []
    identity_indices: list[int] = []
    bins_by_session: dict[str, list[int]] = {}
    for session in sorted(trials_by_session):
        trials = _validated_trials(trials_by_session[session], name=f"time_null[{session}]")
        bins_by_session[session] = [int(trial.shape[0]) for trial in trials]
        # The schedule hash needs all sessions/trials together, so this loop
        # only records dimensions.  The full schedule loop is below.
    for repeat in range(int(replicates)):
        schedule_lines: list[str] = []
        identity = True
        for session in sorted(trials_by_session):
            trials = _validated_trials(trials_by_session[session], name=f"time_null[{session}]")
            for trial_index, trial in enumerate(trials):
                order = deterministic_trial_time_permutation(
                    session_name=session, replicate=repeat, trial_index=trial_index, bins=trial.shape[0]
                )
                line = f"{repeat}:{session}:{trial_index}:{','.join(map(str, order.tolist()))}"
                schedule_lines.append(line)
                canonical.append(line)
                identity = identity and np.array_equal(order, np.arange(trial.shape[0]))
        schedule_hashes.append(hashlib.sha256("\n".join(schedule_lines).encode("utf-8")).hexdigest())
        if identity:
            identity_indices.append(repeat)
    multiplicities = Counter(schedule_hashes)
    return {
        "seed_namespace": TIME_NULL_SEED_NAMESPACE,
        "schedules_including_explicit_identity": int(replicates),
        "random_schedules_after_identity": TIME_ORDER_NULL_RANDOM_SCHEDULES,
        "within_trial_only": True,
        "same_bin_permutation_for_all_units_within_trial": True,
        "explicit_identity_schedule": 0,
        "duplicate_and_random_identity_schedules_retained": True,
        "trial_bin_counts_by_session": bins_by_session,
        "plan_sha256": hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest(),
        "schedule_sha256": schedule_hashes,
        "identity_schedule_indices": identity_indices,
        "duplicate_schedule_hash_multiplicities": {
            digest: count for digest, count in sorted(multiplicities.items()) if count > 1
        },
    }


def marginal_distribution_carrier(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Build the locked 20-D order-free raw-count distribution control.

    All dimensions are computed independently for each unit. Exposure appears
    only in the mandated per-trial rate normalization; there is no separate
    exposure coordinate.  Thus every one of the 20 coordinates is invariant to
    trial ordering and within-trial bin ordering.
    """
    checked = _validated_trials(trials, name="marginal_trials")
    if len(checked) != 10:
        raise ValueError("A2 marginal control is locked to exactly ten support trials")
    all_bins = np.concatenate(checked, axis=0).astype(np.float64, copy=False)
    trial_log_rate = np.stack([
        np.log((trial.sum(axis=0, dtype=np.float64) + 0.5) / (BIN_SECONDS * trial.shape[0])) for trial in checked
    ], axis=0)
    pooled_log_counts = np.log1p(all_bins)
    # NumPy's "linear" interpolation is named explicitly; do not inherit a
    # future library default or use a different quantile convention per arm.
    quantiles = np.quantile(pooled_log_counts, MARGINAL_QUANTILES, axis=0, method="linear")
    columns = (
        *tuple(np.sort(trial_log_rate, axis=0)[index] for index in range(10)),
        *tuple(quantiles[index] for index in range(len(MARGINAL_QUANTILES))),
    )
    carrier = np.stack(columns, axis=1)
    if carrier.shape[1] != WIDTH or carrier.shape[1] != len(MARGINAL_CONTROL_COLUMNS):
        raise RuntimeError("marginal control schema is not width 20")
    if not np.isfinite(carrier).all():
        raise RuntimeError("marginal control produced nonfinite values")
    return carrier


def whole_slot_permutation_carrier(prototype_blocks: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Permute complete P20 blocks for the exact group, including identity.

    v3's one-off negative-control helper deliberately rejects identity.  A2's
    exact group must include identity as the observed configuration, so this
    separate helper keeps v3 semantics untouched while validating the same
    complete-block transformation.
    """
    blocks = np.asarray(prototype_blocks, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64)
    if blocks.ndim != 3 or blocks.shape[1:] != (FIXED_K, TEMPORAL_RANK + 1):
        raise ValueError(f"prototype blocks must be [N,{FIXED_K},{TEMPORAL_RANK + 1}]")
    if order.shape != (FIXED_K,) or set(order.tolist()) != set(range(FIXED_K)):
        raise ValueError("A2 slot null requires a complete S4 permutation")
    return blocks[:, order, :].reshape(blocks.shape[0], WIDTH).copy()


def marginal_control_receipt() -> dict[str, object]:
    return {
        "width": WIDTH,
        "columns": list(MARGINAL_CONTROL_COLUMNS),
        "quantiles": list(MARGINAL_QUANTILES),
        "quantile_interpolation": "numpy_linear",
        "trial_log_rate": "log((sum_valid_bin_count + 0.5) / (0.020 * valid_bin_exposure))",
        "support_labels": "none",
        "dense_behavior": "forbidden",
        "temporal_order": "not represented",
        "raw_support_trial_bins_retained_after_carrier": False,
    }


@dataclass(frozen=True)
class A2BaseCarriers:
    """Fold-specific public carriers plus private blocks for the slot null."""

    public: dict[str, dict[str, dict[str, np.ndarray]]]
    prototype_blocks: dict[str, dict[str, np.ndarray]]
    anchor_receipts: dict[str, dict[str, object]]


def build_a2_base_carriers(
    support_trials: Mapping[str, Sequence[np.ndarray]], d4_by_session: Mapping[str, np.ndarray] | None = None
) -> A2BaseCarriers:
    """Build fold-specific source-only base carriers; no target argument exists.

    ``d4_by_session`` is retained solely for synthetic backwards-compatible
    helper coverage.  The A2 runner supplies ``None``: D4 exposes calibration
    labels and is not an A2 comparator or a gate input.
    """
    sessions = tuple(sorted(support_trials))
    if len(sessions) != 4 or (d4_by_session is not None and set(d4_by_session) != set(sessions)):
        raise ValueError("A2 base carrier construction requires exactly four supports and optional matching descriptive D4")
    validated = {name: _validated_trials(support_trials[name], name=f"support[{name}]") for name in sessions}
    public: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    blocks: dict[str, dict[str, np.ndarray]] = {}
    receipts: dict[str, dict[str, object]] = {}
    for left_out in sessions:
        anchors, anchor_receipt = fit_ordered_anchors_source_only(
            {name: validated[name] for name in sessions if name != left_out}, forbidden_session=left_out
        )
        arms = ["rate_only", "marginal_distribution", "prototype"]
        if d4_by_session is not None:
            arms.insert(0, "D4")
        fold = {arm: {} for arm in arms}
        fold_blocks: dict[str, np.ndarray] = {}
        for session in sessions:
            finalized = carrier_from_support_trials(anchors, validated[session])
            if d4_by_session is not None:
                # Avoid importing the D4 helper into the A2 runner path.
                d4 = _as_finite_matrix(d4_by_session[session], name=f"descriptive_d4[{session}]", width=4)
                padded = np.zeros((d4.shape[0], WIDTH), dtype=np.float64)
                padded[:, :4] = d4
                fold["D4"][session] = padded
            fold["rate_only"][session] = np.asarray(finalized["rate_only"], dtype=np.float64)
            fold["prototype"][session] = np.asarray(finalized["prototype"], dtype=np.float64)
            fold["marginal_distribution"][session] = marginal_distribution_carrier(validated[session])
            fold_blocks[session] = np.asarray(finalized["prototype_blocks"], dtype=np.float64)
        public[left_out] = fold
        blocks[left_out] = fold_blocks
        receipts[left_out] = anchor_receipt
    return A2BaseCarriers(public=public, prototype_blocks=blocks, anchor_receipts=receipts)


def _validate_outer_carriers(
    carriers_by_left_out: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], targets: Mapping[str, np.ndarray]
) -> tuple[str, ...]:
    sessions = tuple(sorted(targets))
    if len(sessions) != 4 or set(carriers_by_left_out) != set(sessions):
        raise ValueError("A2 outer proxy requires exactly four matching source sessions")
    first_arms: set[str] | None = None
    for left_out in sessions:
        fold = carriers_by_left_out[left_out]
        arms = set(fold)
        if first_arms is None:
            first_arms = arms
        if not arms or arms != first_arms:
            raise ValueError("every A2 outer fold must contain the exact same nonempty arm set")
        for arm, by_session in fold.items():
            if set(by_session) != set(sessions):
                raise ValueError(f"A2 carrier session set mismatch for {left_out}/{arm}")
            if any(_as_finite_matrix(by_session[name], name=f"{left_out}/{arm}/{name}").shape[1] != WIDTH for name in sessions):
                raise ValueError("all A2 carrier arms must be width 20")
    return sessions


def generic_outer_loso_proxy(
    carriers_by_left_out: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    targets: Mapping[str, np.ndarray],
    *, ridge: float = RIDGE,
) -> dict[str, object]:
    """Outer-source LOSO ridge proxy for arbitrary A2 width-20 arm sets."""
    if not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("ridge must be finite and positive")
    sessions = _validate_outer_carriers(carriers_by_left_out, targets)
    arms = tuple(sorted(next(iter(carriers_by_left_out.values())).keys()))
    result: dict[str, list[dict[str, object]]] = {arm: [] for arm in arms}
    for left_out in sessions:
        train_sessions = [name for name in sessions if name != left_out]
        train_y = np.concatenate([_as_finite_matrix(targets[name], name=f"target/{name}") for name in train_sessions])
        test_y = _as_finite_matrix(targets[left_out], name=f"target/{left_out}")
        for arm in arms:
            train_x = np.concatenate([
                _as_finite_matrix(carriers_by_left_out[left_out][arm][name], name=f"carrier/{left_out}/{arm}/{name}")
                for name in train_sessions
            ])
            test_x = _as_finite_matrix(carriers_by_left_out[left_out][arm][left_out], name=f"carrier/{left_out}/{arm}/{left_out}")
            if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
                raise ValueError(f"A2 target/carrier unit mismatch in {left_out}/{arm}")
            readout = fit_ridge_readout(train_x, train_y, ridge=ridge)
            prediction = predict_ridge_readout(readout, test_x)
            r2 = r2_from_source_baseline(train_y, test_y, prediction)
            rss = float(np.square(test_y - prediction).sum())
            tss = float(np.square(test_y - train_y.mean(axis=0, keepdims=True)).sum())
            if not (tss > _EPS and rss >= 0.0):
                raise RuntimeError("invalid outer-source RSS/TSS for bounded A2 statistic")
            # Unlike R2, which is unbounded below and was responsible for the
            # heavy-tail shuffle artefact in A1, U is bounded in [0, 1].
            bounded_u = tss / (tss + rss)
            result[arm].append({
                "left_out_session": left_out,
                "r2": float(r2),
                "bounded_u": float(bounded_u),
                "rss": rss,
                "tss_source_baseline": tss,
                "train_sessions": train_sessions,
            })
    return {"source_only_outer_loso": True, "ridge": float(ridge), "sessions": list(sessions), "arms": result}


def slot_null_carriers(
    base: A2BaseCarriers, *, plan_for_replicate: Mapping[str, np.ndarray]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Add one fold-consistent session-keyed whole-slot null arm to base arms."""
    sessions = tuple(sorted(base.public))
    if set(plan_for_replicate) != set(sessions):
        raise ValueError("slot null plan must give exactly one permutation per source session")
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for left_out in sessions:
        fold = {arm: dict(values) for arm, values in base.public[left_out].items()}
        fold["slot_null"] = {
            session: whole_slot_permutation_carrier(base.prototype_blocks[left_out][session], plan_for_replicate[session])
            for session in sessions
        }
        output[left_out] = fold
    return output


def _slot_null_only_carriers(
    base: A2BaseCarriers, *, plan_for_replicate: Mapping[str, np.ndarray]
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Minimal carrier dictionary for an exact null iteration (avoids refitting base arms)."""
    sessions = tuple(sorted(base.public))
    if set(plan_for_replicate) != set(sessions):
        raise ValueError("slot null plan must give exactly one permutation per source session")
    return {
        left_out: {"slot_null": {
            session: whole_slot_permutation_carrier(base.prototype_blocks[left_out][session], plan_for_replicate[session])
            for session in sessions
        }}
        for left_out in sessions
    }


def exact_slot_null_statistics(
    base: A2BaseCarriers, targets: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Score all 13,824 exact slot-null quotient configurations with bounded U."""
    plans = exact_slot_null_plans(tuple(sorted(base.public)))
    values = np.empty(len(plans), dtype=np.float64)
    per_session = np.empty((len(plans), 4), dtype=np.float64)
    sessions = tuple(sorted(base.public))
    for index, plan in enumerate(plans):
        proxy = generic_outer_loso_proxy(_slot_null_only_carriers(base, plan_for_replicate=plan), targets, ridge=RIDGE)
        values[index] = mean_arm_metric(proxy, "slot_null", metric="bounded_u")
        rows = {str(row["left_out_session"]): float(row["bounded_u"]) for row in proxy["arms"]["slot_null"]}
        per_session[index] = [rows[session] for session in sessions]
    receipt = slot_null_plan_receipt(plans)
    receipt.update({"scored_metric": "mean_four_session_bounded_U", "raw_R2_not_used_for_rank": True})
    receipt.update({"session_order": list(sessions), "all_per_fold_U_recorded": True})
    return values, per_session, receipt


def time_order_null_carriers(
    support_trials: Mapping[str, Sequence[np.ndarray]],
    *, replicate: int
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Build a re-fit temporal-order null without labels, D4, or targets."""
    sessions = tuple(sorted(support_trials))
    if len(sessions) != 4:
        raise ValueError("A2 time null requires exactly four source sessions")
    null_support = {
        session: within_trial_time_order_null(support_trials[session], session_name=session, replicate=replicate)
        for session in sessions
    }
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for left_out in sessions:
        anchors, _ = fit_ordered_anchors_source_only(
            {name: null_support[name] for name in sessions if name != left_out}, forbidden_session=left_out
        )
        output[left_out] = {
            "time_order_null": {
                session: np.asarray(carrier_from_support_trials(anchors, null_support[session])["prototype"], dtype=np.float64)
                for session in sessions
            }
        }
    return output


def time_order_null_score_matrix(
    support_trials: Mapping[str, Sequence[np.ndarray]], targets: Mapping[str, np.ndarray], *, replicates: int = TIME_ORDER_NULL_REPLICATES
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return [schedule, session] bounded-U scores for the locked time reference."""
    if int(replicates) != TIME_ORDER_NULL_REPLICATES:
        raise ValueError("A2 time-order schedule count is locked to 4,096")
    sessions = tuple(sorted(support_trials))
    matrix = np.empty((int(replicates), len(sessions)), dtype=np.float64)
    for repeat in range(int(replicates)):
        proxy = generic_outer_loso_proxy(time_order_null_carriers(support_trials, replicate=repeat), targets, ridge=RIDGE)
        rows = {str(row["left_out_session"]): float(row["bounded_u"]) for row in proxy["arms"]["time_order_null"]}
        matrix[repeat] = [rows[session] for session in sessions]
    return sessions, matrix


def time_order_null_statistics(
    support_trials: Mapping[str, Sequence[np.ndarray]], targets: Mapping[str, np.ndarray], *, replicates: int = TIME_ORDER_NULL_REPLICATES
) -> np.ndarray:
    """Return schedule H values; detailed per-session U is available separately."""
    _, matrix = time_order_null_score_matrix(support_trials, targets, replicates=replicates)
    return matrix.mean(axis=1)


def mean_arm_metric(proxy: Mapping[str, object], arm: str, *, metric: str = "bounded_u") -> float:
    """Return a four-session mean metric, refusing malformed proxy rows."""
    rows = proxy.get("arms", {}).get(arm) if isinstance(proxy.get("arms"), Mapping) else None
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError(f"proxy has no four-session arm {arm!r}")
    values = np.asarray([row.get(metric) for row in rows if isinstance(row, Mapping)], dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"proxy arm {arm!r} has malformed/nonfinite {metric}")
    return float(values.mean())


def monte_carlo_upper_rank(observed_statistic: float, null_statistics: Sequence[float]) -> dict[str, float | int]:
    """One-sided rank of an observed larger-is-better statistic against MC nulls."""
    observed = float(observed_statistic)
    nulls = np.asarray(null_statistics, dtype=np.float64).reshape(-1)
    if not math.isfinite(observed) or nulls.size == 0 or not np.isfinite(nulls).all():
        raise ValueError("observed and all Monte-Carlo null statistics must be finite")
    more_extreme_or_tied = int((nulls >= observed).sum())
    rank = 1 + more_extreme_or_tied
    return {
        "observed_statistic": observed,
        "null_replicates": int(nulls.size),
        "upper_rank": int(rank),
        "monte_carlo_p_value": float(rank / (nulls.size + 1)),
        "null_mean": float(nulls.mean()),
        "null_std": float(nulls.std(ddof=1)) if nulls.size > 1 else 0.0,
        "null_q025": float(np.quantile(nulls, 0.025)),
        "null_q50": float(np.quantile(nulls, 0.50)),
        "null_q975": float(np.quantile(nulls, 0.975)),
    }


def exact_group_upper_rank(observed_statistic: float, group_statistics: Sequence[float]) -> dict[str, float | int]:
    """Exact one-sided finite-group rank; no Monte-Carlo correction is used.

    ``group_statistics`` must contain the complete quotient enumeration,
    including the observed identity-relative configuration.  Ties are counted
    conservatively in the upper tail.  This is the primary slot-null statistic;
    :func:`monte_carlo_upper_rank` remains available only for explicitly marked
    future fallback work when an exhaustive group is genuinely impossible.
    """
    observed = float(observed_statistic)
    values = np.asarray(group_statistics, dtype=np.float64).reshape(-1)
    if not math.isfinite(observed) or values.shape != (SLOT_NULL_REPLICATES,) or not np.isfinite(values).all():
        raise ValueError("exact slot-null rank needs every one of the 24**3 finite group statistics")
    upper_rank = int((values >= observed).sum())
    return {
        "observed_statistic": observed,
        "exact_group_classes": int(values.size),
        "upper_rank": upper_rank,
        "exact_upper_tail_p_value": float(upper_rank / values.size),
        "null_mean": float(values.mean()),
        "null_std": float(values.std(ddof=1)),
        "null_q025": float(np.quantile(values, 0.025)),
        "null_q50": float(np.quantile(values, 0.50)),
        "null_q975": float(np.quantile(values, 0.975)),
        "statistic": "mean_four_session_bounded_U=TSS/(TSS+RSS)",
        "r2_is_secondary_only": True,
    }


def scheduled_time_upper_rank(observed_statistic: float, schedule_statistics: Sequence[float]) -> dict[str, float | int]:
    """Rank against the full locked time schedule, retaining identity/duplicates.

    This is a deterministic finite schedule reference, not an exhaustive
    permutation-group p-value.  The identity schedule must be present at index
    zero and is deliberately part of both the denominator and conservative tie
    count.
    """
    observed = float(observed_statistic)
    values = np.asarray(schedule_statistics, dtype=np.float64).reshape(-1)
    if not math.isfinite(observed) or values.shape != (TIME_ORDER_NULL_REPLICATES,) or not np.isfinite(values).all():
        raise ValueError("time-order rank requires the locked identity-plus-4,095 finite schedule")
    if not np.isclose(values[0], observed, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("time-order schedule index zero must be the observed identity statistic")
    exceedances = int((values[1:] >= observed).sum())
    # Exactly the protocol formula: the explicit observation supplies the
    # leading one, while identities/duplicates among schedules 1..4095 remain.
    upper_rank = 1 + exceedances
    mc_upper = 1.0 if exceedances == TIME_ORDER_NULL_RANDOM_SCHEDULES else float(
        beta_distribution.ppf(0.975, exceedances + 1, TIME_ORDER_NULL_RANDOM_SCHEDULES - exceedances)
    )
    return {
        "observed_statistic": observed,
        "schedules_including_identity": int(values.size),
        "random_schedule_exceedances": exceedances,
        "upper_rank": upper_rank,
        "scheduled_upper_tail_p_value": float(upper_rank / values.size),
        "one_sided_97p5_binomial_upper_bound": mc_upper,
        "null_mean": float(values[1:].mean()),
        "null_std": float(values[1:].std(ddof=1)),
        "null_q025": float(np.quantile(values[1:], 0.025)),
        "null_q50": float(np.quantile(values[1:], 0.50)),
        "null_q975": float(np.quantile(values[1:], 0.975)),
        "interpretation": "deterministic_time_schedule_reference_not_exhaustive_group_test",
        "duplicate_and_identity_schedules_retained": True,
    }


def paired_b20_content_summary(observed_proxy: Mapping[str, object]) -> dict[str, object]:
    """Precommitted four-session P20-minus-B20 R2 content sensitivity receipt."""
    arms = observed_proxy.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("observed proxy has no arm map")
    prototype = arms.get("prototype")
    baseline = arms.get("marginal_distribution")
    if not isinstance(prototype, list) or not isinstance(baseline, list) or len(prototype) != 4 or len(baseline) != 4:
        raise ValueError("B20 content summary requires four P20 and B20 rows")
    by_p = {str(row.get("left_out_session")): row for row in prototype if isinstance(row, Mapping)}
    by_b = {str(row.get("left_out_session")): row for row in baseline if isinstance(row, Mapping)}
    if set(by_p) != set(by_b) or len(by_p) != 4:
        raise ValueError("P20/B20 folds do not align")
    sessions = tuple(sorted(by_p))
    deltas = np.asarray([float(by_p[name]["r2"]) - float(by_b[name]["r2"]) for name in sessions], dtype=np.float64)
    if not np.isfinite(deltas).all():
        raise ValueError("P20/B20 content deltas are nonfinite")
    sd = float(deltas.std(ddof=1))
    se = sd / math.sqrt(4)
    critical95 = float(student_t.ppf(0.975, df=3))
    critical80 = float(student_t.ppf(0.80, df=3))
    mean = float(deltas.mean())
    return {
        "sessions": list(sessions),
        "delta_B_R2_by_session": {name: float(delta) for name, delta in zip(sessions, deltas)},
        "mean_delta_B_R2": mean,
        "median_delta_B_R2": float(np.median(deltas)),
        "session_sd": sd,
        "paired_ci95": [mean - critical95 * se, mean + critical95 * se],
        "mde80": (critical95 + critical80) * se,
        "all_four_positive": bool(np.all(deltas > 0.0)),
    }


def canonical_slot_identity_rank_diagnostic(
    plans: Sequence[Mapping[str, np.ndarray]], group_h: Sequence[float]
) -> dict[str, object]:
    """Return the precommitted 24-way canonical-other-identity rank for all sessions.

    For a non-gauge session, the other three assignments are canonical identity
    and its own assignment varies over S4.  For the fixed gauge session, the
    equivalent 24-way orbit is already present in the quotient: all three
    non-gauge assignments move together.  Thus no second enumeration is
    launched.  A strictly-above-median rank uses conservative ties (only values
    strictly below identity contribute to its low-to-high rank).
    """
    if len(plans) != SLOT_NULL_REPLICATES:
        raise ValueError("conditional slot ranks require the complete exact quotient")
    values = np.asarray(group_h, dtype=np.float64).reshape(-1)
    if values.shape != (SLOT_NULL_REPLICATES,) or not np.isfinite(values).all():
        raise ValueError("conditional slot ranks require complete finite H statistics")
    sessions = tuple(sorted(plans[0]))
    if len(sessions) != 4 or any(set(plan) != set(sessions) for plan in plans):
        raise ValueError("conditional slot ranks need the four consistent session keys")
    identity = np.arange(FIXED_K, dtype=np.int64)
    observed_indices = [
        index for index, plan in enumerate(plans)
        if all(np.array_equal(plan[session], identity) for session in sessions)
    ]
    if len(observed_indices) != 1:
        raise RuntimeError("exact quotient does not contain exactly one all-identity configuration")
    observed_h = float(values[observed_indices[0]])
    rows: dict[str, object] = {}
    for target_index, target in enumerate(sessions):
        if target_index == 0:
            # Re-gauge trajectory: original p_ref varies, and after quotient
            # gauge-fixing every non-reference relative assignment is p_ref^-1.
            indices = [
                index for index, plan in enumerate(plans)
                if np.array_equal(plan[sessions[1]], plan[sessions[2]])
                and np.array_equal(plan[sessions[1]], plan[sessions[3]])
            ]
            conditioning = "re_gauged_reference_orbit_other_three_equal"
        else:
            indices = [
                index for index, plan in enumerate(plans)
                if all(np.array_equal(plan[other], identity) for other in sessions if other != target)
            ]
            conditioning = "other_three_canonical_identity"
        candidate = values[np.asarray(indices, dtype=np.int64)]
        if candidate.shape != (24,):
            raise RuntimeError("conditional identity orbit did not contain exactly 24 assignments")
        rank_low_to_high = 1 + int((candidate < observed_h).sum())
        rows[target] = {
            "conditioning": conditioning,
            "candidate_assignments": 24,
            "identity_H": observed_h,
            "conditional_H_median": float(np.median(candidate)),
            "identity_rank_low_to_high_strict_ties_conservative": rank_low_to_high,
            "strictly_above_conditional_median": bool(rank_low_to_high > 12),
        }
    return {
        "scope": "canonical_other_identity_condition; gauge session uses equivalent re-gauged orbit already in 24^3 quotient",
        "all_sessions_strictly_above_conditional_median": bool(all(
            row["strictly_above_conditional_median"] for row in rows.values()
        )),
        "sessions": rows,
    }


def a2_cpu_gate(
    *, content: Mapping[str, object], slot_rank: Mapping[str, object],
    time_rank: Mapping[str, object], observed_h: float, slot_h: Sequence[float], time_h: Sequence[float],
    observed_u_by_session: Mapping[str, float], slot_u_by_session: Mapping[str, Sequence[float]],
    time_u_by_session: Mapping[str, Sequence[float]],
    validity_contract_pass: bool, repeatability_contract_pass: bool,
) -> dict[str, object]:
    """Frozen Gate A2 conjunction.  It grants no decoder/GPU action itself."""
    content_pass = bool(
        content.get("all_four_positive")
        and float(content["paired_ci95"][0]) >= 0.030
        and float(content["mde80"]) <= 0.030
    )
    slot_values = np.asarray(slot_h, dtype=np.float64)
    time_values = np.asarray(time_h, dtype=np.float64)
    slot_session_passes = {
        session: float(observed_u_by_session[session]) > float(np.median(np.asarray(values, dtype=np.float64)))
        for session, values in slot_u_by_session.items()
    }
    slot_pass = bool(
        float(slot_rank["exact_upper_tail_p_value"]) < 0.025
        and float(observed_h) > float(np.median(slot_values))
        and len(slot_session_passes) == 4
        and all(slot_session_passes.values())
    )
    time_session_passes = {
        session: float(observed_u_by_session[session]) > float(np.median(np.asarray(values, dtype=np.float64)))
        for session, values in time_u_by_session.items()
    }
    time_pass = bool(
        float(time_rank["scheduled_upper_tail_p_value"]) < 0.025
        and float(time_rank["one_sided_97p5_binomial_upper_bound"]) < 0.025
        and float(observed_h) > float(np.median(time_values[1:]))
        and sum(not value for value in time_session_passes.values()) <= 1
    )
    validity_pass = bool(validity_contract_pass and repeatability_contract_pass)
    if not validity_pass:
        decision = "invalid_execution_stop"
    elif not content_pass:
        decision = "marginal_baseline_not_beaten_stop" if not bool(content.get("all_four_positive")) else "precision_insufficient_stop"
    elif not slot_pass:
        decision = "slot_coordinate_not_distinguishable_stop"
    elif not time_pass:
        decision = "temporal_order_not_distinguishable_stop"
    else:
        decision = "gate_a2_pass_cpu_only_prepare_separate_decoder_gpu_protocol"
    return {
        "decision": decision,
        "content_B20_pass": content_pass,
        "slot_coordinate_pass": slot_pass,
        "slot_session_canonical_U_above_full_null_median": slot_session_passes,
        "time_order_pass": time_pass,
        "validity_and_repeatability_pass": validity_pass,
        "time_session_chronological_U_above_time_median": time_session_passes,
        "gpu_authorized": False,
        "decoder_authorized": False,
    }


def a2_control_contract_receipt() -> dict[str, object]:
    """Static declaration for a prelaunch receipt; no files/data are touched."""
    return {
        "semantics_version": A2_SEMANTICS_VERSION,
        "source_only": True,
        "cuda": False,
        "decoder": False,
        "formal": False,
        "held_out": False,
        "evalai": False,
        "slot_null": {
            "replicates": SLOT_NULL_REPLICATES,
            "seed_namespace": SLOT_NULL_SEED_NAMESPACE,
            "method": "exhaustive_session_keyed_(S4)^4/global_S4_whole_count_plus_rank_vector_slot_blocks",
            "identity_relative_class_included": True,
            "test": "one_sided_exact_group_upper_rank_of_four_session_mean_bounded_U",
            "raw_R2": "secondary_descriptive_only_due_to_unbounded_lower_tail",
        },
        "time_order_null": {
            "replicates": TIME_ORDER_NULL_REPLICATES,
            "seed_namespace": TIME_NULL_SEED_NAMESPACE,
            "method": "independent_within_trial_time_permutation_shared_across_units_then_refit_outer_train_anchors",
            "explicit_identity_plus_deterministic_random_schedules": True,
            "duplicate_and_identity_schedules_retained": True,
            "preserved": "every trial/unit count sum and bin multiset; synchronous population bin vectors",
        },
        "marginal_distribution": marginal_control_receipt(),
    }
