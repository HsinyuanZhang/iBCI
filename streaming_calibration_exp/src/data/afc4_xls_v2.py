"""Isolated support-only primitive for the reviewed RT AFC4 strong-LS v2 null.

``afc4_xls_v2`` remains a support-only primitive even after the separately
reviewed matched adapter connected it to the RT data path.  Keeping generation
isolated here lets that adapter use the exact algorithm audited in
``RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json`` without copying random
permutation logic into a training module.

The generator consumes only already collected calibration-support block reach
IDs, their paired support velocity labels, an explicit session namespace, and
an explicit seed.  It never accepts neural rates, coefficients, a decoder,
query labels, or a score.  It returns an index permutation: the active adapter
fits its ordinary AFC4 OLS descriptor from ``rates`` and
``velocity[permutation]``.  This module itself intentionally does not fit a
descriptor and therefore cannot independently alter model or feature state.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np


SEED = 42
MAX_RANDOM_DERANGEMENT_ATTEMPTS = 128
MAX_ABS_SESSION_REACH_DIRECTION_COSINE = 0.50
EPS = 1.0e-12
V2_AUDIT_SCHEMA = "rt_afc4_ls_null_strength_support_audit_v2"
V2_AUDIT_STATUS = "PASS_CPU_SUPPORT_ONLY_RT_AFC4_LS_NULL_STRENGTH_AUDIT_V2"


class Afc4XlsV2Error(ValueError):
    """The frozen strong-LS-v2 support-only contract was not satisfied."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Afc4XlsV2Error(message)


def _candidate_rng(*, session_name: str, seed: int, attempt: int) -> np.random.RandomState:
    digest = hashlib.sha256(
        f"rt-afc4-ls-v2-random-cross-reach:{seed}:{session_name}:attempt={attempt}".encode()
    ).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "little"))


def _repair_group_collisions(permutation: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Make a block bijection cross-reach while preserving its exact multiset."""

    result = np.asarray(permutation, dtype=np.int64).copy()
    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    count = result.size
    _need(result.shape == group.shape, "cross-reach repair shape mismatch")
    if np.max(np.bincount(np.unique(group, return_inverse=True)[1])) * 2 > count:
        raise Afc4XlsV2Error("cross-reach derangement is impossible: one reach owns more than half of active blocks")
    while True:
        bad = np.flatnonzero(group[result] == group)
        if bad.size == 0:
            return result
        changed = False
        for left_index in range(bad.size):
            left = int(bad[left_index])
            for right_index in range(left_index + 1, bad.size):
                right = int(bad[right_index])
                if group[left] != group[right]:
                    result[left], result[right] = result[right], result[left]
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        # A sole residual collision can exchange its donor with an existing
        # valid destination without changing the global label multiset.
        left = int(bad[0])
        for right in range(count):
            if right == left or group[right] == group[left]:
                continue
            if group[result[right]] != group[left]:
                result[left], result[right] = result[right], result[left]
                changed = True
                break
        if not changed:
            raise Afc4XlsV2Error("cannot repair cross-reach derangement without a same-reach label")


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    if norm <= EPS:
        return None
    return float(np.clip(np.dot(x, y) / norm, -1.0, 1.0))


def _summary(values: list[float | None]) -> dict[str, Any]:
    usable = np.asarray([float(item) for item in values if item is not None and np.isfinite(item)], dtype=np.float64)
    if usable.size == 0:
        return {"defined": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "defined": int(usable.size), "mean": float(np.mean(usable)), "median": float(np.median(usable)),
        "minimum": float(np.min(usable)), "maximum": float(np.max(usable)),
    }


def permutation_diagnostics(groups: np.ndarray, velocity: np.ndarray, permutation: np.ndarray) -> dict[str, Any]:
    """Diagnose only label-marginal preservation and reach-level pairing break."""

    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    label = np.asarray(velocity, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64).reshape(-1)
    _need(order.shape == group.shape and label.shape == (group.size, 2), "permutation diagnostic shape mismatch")
    _need(np.array_equal(np.sort(order), np.arange(order.size)), "permutation diagnostic requires a true permutation")
    assigned = label[order]
    reaches: list[dict[str, Any]] = []
    for reach in np.unique(group).tolist():
        indices = np.flatnonzero(group == reach)
        own = label[indices].mean(axis=0)
        donor = assigned[indices].mean(axis=0)
        cosine = _safe_cosine(own, donor)
        angle = None if cosine is None else float(math.atan2(own[0] * donor[1] - own[1] * donor[0], np.dot(own, donor)))
        reaches.append({
            "reach_id": int(reach), "active_blocks": int(indices.size),
            "same_reach_assigned_blocks": int(np.sum(group[order[indices]] == reach)),
            "correct_mean_velocity": own.tolist(), "assigned_mean_velocity": donor.tolist(),
            "direction_cosine": cosine, "signed_direction_angle_rad": angle,
        })
    reach_cosines = [row["direction_cosine"] for row in reaches]
    same = int(np.sum(group[order] == group))
    return {
        "active_blocks": int(group.size), "labels_changed_blocks": int(np.sum(order != np.arange(order.size))),
        "same_reach_assigned_blocks": same, "same_reach_assigned_fraction": float(same / group.size),
        "velocity_marginal": {
            "preserved_exactly_by_index_permutation": True, "permutation_is_bijective": True,
            "global_mean_delta": (assigned.mean(axis=0) - label.mean(axis=0)).tolist(),
            "global_speed_mean_delta": float(np.linalg.norm(assigned, axis=1).mean() - np.linalg.norm(label, axis=1).mean()),
            "tolerance": 0.0,
        },
        "block_length_statistics": {
            "unit": "event-qualified fixed 100-ms blocks", "total_blocks_unchanged": True,
            "per_reach_neural_block_counts_unchanged": True, "tolerance": 0.0,
        },
        "reach_direction_transfer": {
            "defined_reaches": int(sum(item is not None for item in reach_cosines)),
            "mean_cosine": _summary(reach_cosines)["mean"], "median_cosine": _summary(reach_cosines)["median"],
            "per_reach": reaches,
        },
    }


def deterministic_random_cross_reach_derangement(
    groups: np.ndarray, velocity: np.ndarray, *, session_name: str, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate the audited v2 random cross-reach support-label permutation.

    This is the sole random primitive for the future ``afc4_xls_v2`` arm.  Its
    candidate and acceptance logic depend only on ``groups``, ``velocity``,
    ``session_name`` and ``seed``.  ``velocity`` must originate entirely from
    the chronological calibration support prefix; query labels are neither an
    accepted argument nor inspected by this module.
    """

    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    labels = np.asarray(velocity, dtype=np.float64)
    _need(labels.shape == (group.size, 2) and group.size >= 3, "v2 null needs >=3 velocity-labelled blocks")
    _need(np.isfinite(labels).all(), "v2 null requires finite support velocity labels")
    _need(np.unique(group).size >= 3, "v2 null needs at least three accepted reaches")
    _need(bool(session_name), "v2 null requires a nonempty session namespace")
    for attempt in range(MAX_RANDOM_DERANGEMENT_ATTEMPTS):
        permutation = _repair_group_collisions(
            _candidate_rng(session_name=session_name, seed=seed, attempt=attempt).permutation(group.size).astype(np.int64), group
        )
        diagnostics = permutation_diagnostics(group, labels, permutation)
        transfer = diagnostics["reach_direction_transfer"]
        mean, median = transfer["mean_cosine"], transfer["median_cosine"]
        if mean is None or median is None:
            continue
        if abs(float(mean)) <= MAX_ABS_SESSION_REACH_DIRECTION_COSINE and abs(float(median)) <= MAX_ABS_SESSION_REACH_DIRECTION_COSINE:
            _need(np.array_equal(np.sort(permutation), np.arange(group.size)), "v2 null is not a bijection")
            _need(np.all(permutation != np.arange(group.size)), "v2 null left an active block unchanged")
            _need(np.all(group[permutation] != group), "v2 null retained a same-reach label")
            return permutation, {"attempt": int(attempt), "permutation_diagnostics": diagnostics}
    raise Afc4XlsV2Error(
        "no session-namespaced random cross-reach derangement met the label-only non-extreme direction gate"
    )


def permutation_sha256(permutation: np.ndarray) -> str:
    """Canonical SHA-256 used by the immutable v2 audit and future receipts."""

    order = np.asarray(permutation, dtype=np.int64).reshape(-1)
    return hashlib.sha256(order.tobytes()).hexdigest()


def audited_session_expected_sha256(audit_receipt: Mapping[str, Any], *, session_name: str) -> str:
    """Read the exact expected per-session permutation SHA from a v2 receipt."""

    _need(audit_receipt.get("schema") == V2_AUDIT_SCHEMA and audit_receipt.get("status") == V2_AUDIT_STATUS,
          "not a passing strong-LS v2 audit receipt")
    rows = audit_receipt.get("fold_rows")
    _need(isinstance(rows, list), "v2 audit receipt fold rows absent")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("session_name") == session_name]
    _need(len(matches) == 1, f"v2 audit receipt requires exactly one row for session {session_name!r}")
    null = matches[0].get("v2_random_cross_reach_null")
    _need(isinstance(null, Mapping) and isinstance(null.get("permutation_sha256"), str) and
          len(null["permutation_sha256"]) == 64, f"v2 audit receipt permutation SHA absent for {session_name!r}")
    return str(null["permutation_sha256"])


def assert_audited_session_parity(
    permutation: np.ndarray, *, audit_receipt: Mapping[str, Any], session_name: str
) -> str:
    """Fail closed unless a generated permutation exactly matches the audit row."""

    actual = permutation_sha256(permutation)
    expected = audited_session_expected_sha256(audit_receipt, session_name=session_name)
    _need(actual == expected, f"afc4_xls_v2 permutation SHA mismatch for {session_name!r}: {actual} != {expected}")
    return actual
