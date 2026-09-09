"""Arm -> {carrier transform, identity fusion} contract for the dandi688 local benchmark.

Carrier arms (design doc section 3), all operating on the frozen prepared-cache row
  carrier [N, 4] = [a_R, c_R, m_R, delta_b] with unit_mask [N] bool:

  t4  identity: the true carrier, byte-for-byte the frozen values.
  f0  carrier fully zeroed; the E0 [N, 50] activity identity provider is
      untouched (activity-only; f0 differs from t4 only through the carrier path).
  z0  carrier and E0 both zeroed (query-window only; matches M1/M2/H1 Z_NONE).
      Carrier transform is the same all-zero as f0; E0 zeroing is applied by
      the runner after this module returns.
  ts4 session-internal channel<->carrier row permutation
      (ts4_shuffle): destroys the channel<->carrier correspondence while
      leaving every per-column marginal distribution identical.

Fusion axis (orthogonal to the carrier axis, user ruling 2026-09-09):
  t4/f0/ts4       proj_add frontend (settled P: 50->16 added onto local).
  t4_concat       the true carrier again, but fused through the matched
                  concat frontend ([local16 | E0_50 | carrier4], token_in 70,
                  v2 concat_model).  Carrier transform identical to t4;
                  only the model's identity fusion differs.  The concat arm
                  never enters the preregistered gates (readings only).

vstate-688 series (user directive 2026-09-09, verbatim: "立刻开始准备
vstate-688，注意消融——z 系列 = 完全不在新日期校准（bank 复用旧日期/source），
f 系列 = 不使用任何标签校准（label-free）"), all proj_add, all bound to their
variant cache via --prepared-cache (plan.ARM_REQUIRED_CACHE_VARIANT):
  vstate           carrier transform identity on the vstate variant cache
                   (signed-velocity-state carrier; dense cursor_vel labels).
  z_vstate_srcbank same identity transform, but the runner additionally
                   swaps each exam session's bank (carrier + E0 rows) for its
                   frozen nearest-date train session's bank AFTER this
                   module returns (plan.z_srcbank_map) -- "完全不在新日期
                   校准".  The swap is cross-session, so it cannot be a pure
                   per-array transform here; the runner records the mapping
                   and the digests before/after.
  f_labelfree      zero transform over the f_labelfree variant cache whose
                   carrier is already all-zero and whose E0 was remelted
                   with a zero side (ACTIVITY-ONLY identity) -- "不使用任何
                   标签校准".  Zeroing here is belt-and-suspenders; the
                   label-free property lives in the cache bytes.

Every transform is a pure function returning a new array; the input carrier
is never mutated.  ``verify_arm`` re-derives the digest-level assertions
(column marginals equal per column for ts4, all-zero for f0, equality for
t4/t4_concat) so callers can assert the contract at load and at scoring time.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import plan, ts4_shuffle

# --- arm -> {carrier transform, identity fusion} mapping -------------------
CARRIER_ACTIONS = ("identity", "zero", "shuffle")
ARM_SPECS: dict[str, dict[str, str]] = {
    "t4": {"carrier": "identity", "fusion": "proj_add", "e0": "identity"},
    "f0": {"carrier": "zero", "fusion": "proj_add", "e0": "identity"},
    "ts4": {"carrier": "shuffle", "fusion": "proj_add", "e0": "identity"},
    "t4_concat": {"carrier": "identity", "fusion": "concat", "e0": "identity"},
    "z0": {"carrier": "zero", "fusion": "proj_add", "e0": "zero"},
    # vstate-688 series (user directive 2026-09-09): variant-cache bound,
    # proj_add fusion; the z-series bank swap is a runner-level cross-session
    # operation recorded in the receipts (see module docstring).
    "vstate": {"carrier": "identity", "fusion": "proj_add", "e0": "identity"},
    "z_vstate_srcbank": {"carrier": "identity", "fusion": "proj_add", "e0": "identity"},
    "f_labelfree": {"carrier": "zero", "fusion": "proj_add", "e0": "identity"},
}
E0_ACTIONS = ("identity", "zero")
if set(ARM_SPECS) != set(plan.ARMS) or any(
    spec["fusion"] not in plan.FUSION_MODES
    or spec["carrier"] not in CARRIER_ACTIONS
    or spec["e0"] not in E0_ACTIONS
    for spec in ARM_SPECS.values()
):  # fail-closed module invariant: the mapping must mirror plan.ARMS exactly
    raise RuntimeError("ARM_SPECS must cover plan.ARMS with legal carrier/fusion/e0 values")


def arm_fusion(arm: str) -> str:
    """Identity-fusion mode bound to ``arm`` (see plan.FUSION_MODES)."""
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {plan.ARMS}")
    return ARM_SPECS[arm]["fusion"]


def arm_e0_action(arm: str) -> str:
    """E0 action bound to ``arm``: identity (keep frozen bank) or zero."""
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {plan.ARMS}")
    return ARM_SPECS[arm]["e0"]


def _validate(carrier: np.ndarray, unit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    carrier = np.asarray(carrier)
    unit_mask = np.asarray(unit_mask).astype(bool)
    plan.require(carrier.ndim == 2, f"carrier must be 2-D, got {carrier.shape}")
    plan.require(
        carrier.shape[1] == plan.CARRIER_DIM,
        f"carrier must have {plan.CARRIER_DIM} columns, got {carrier.shape[1]}",
    )
    plan.require(
        unit_mask.shape == (carrier.shape[0],),
        f"unit_mask must be [N]={carrier.shape[0]}, got {unit_mask.shape}",
    )
    plan.require(bool(unit_mask.any()), "unit_mask must keep at least one real unit")
    return carrier, unit_mask


def apply_arm(
    arm: str,
    session_name: str,
    carrier: np.ndarray,
    unit_mask: np.ndarray,
    seed: int = plan.SEED,
) -> np.ndarray:
    """Pure carrier transform for ``arm``; returns a new float32 array.

    The fusion axis never touches the carrier bytes: t4_concat applies the
    same identity transform as t4 (the difference lives in the model)."""
    if arm not in plan.ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {plan.ARMS}")
    action = ARM_SPECS[arm]["carrier"]
    carrier, unit_mask = _validate(carrier, unit_mask)
    plan.require(
        np.issubdtype(carrier.dtype, np.floating),
        f"carrier must be floating point, got {carrier.dtype}",
    )
    if action == "identity":
        return np.array(carrier, dtype=carrier.dtype, copy=True)
    if action == "zero":
        return np.zeros_like(carrier)
    return np.ascontiguousarray(
        ts4_shuffle.shuffle_real_rows(carrier, unit_mask, session_name, seed),
        dtype=carrier.dtype,
    )


def assert_column_marginals_equal(before: np.ndarray, after: np.ndarray) -> None:
    """Per-column marginal equality: each column's value multiset is identical.

    Exact (bitwise) comparison of the sorted columns; a row permutation never
    changes values, so exactness is guaranteed for legitimate transforms and
    any value-level tampering fails."""
    before = np.asarray(before)
    after = np.asarray(after)
    plan.require(
        before.shape == after.shape,
        f"column-marginal check needs equal shapes, got {before.shape} vs {after.shape}",
    )
    for j in range(before.shape[1]):
        if not np.array_equal(np.sort(before[:, j]), np.sort(after[:, j])):
            raise RuntimeError(
                f"column marginal distribution changed in carrier column {j}"
            )


def verify_arm(
    arm: str,
    session_name: str,
    carrier: np.ndarray,
    unit_mask: np.ndarray,
    transformed: np.ndarray,
    seed: int = plan.SEED,
) -> dict[str, Any]:
    """Assert the arm contract on an already-transformed array; return a digest record."""
    if arm not in plan.ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {plan.ARMS}")
    action = ARM_SPECS[arm]["carrier"]
    carrier, unit_mask = _validate(carrier, unit_mask)
    transformed = np.asarray(transformed)
    plan.require(
        transformed.shape == carrier.shape and transformed.dtype == carrier.dtype,
        "transformed carrier must keep shape and dtype",
    )
    real = np.flatnonzero(unit_mask)
    record: dict[str, Any] = {
        "arm": arm,
        "session": session_name,
        "carrier_action": action,
        "fusion": ARM_SPECS[arm]["fusion"],
        "carrier_sha256_before": plan.array_digest(carrier),
        "carrier_sha256_after": plan.array_digest(transformed),
        "n_real_rows": int(len(real)),
    }
    if action == "identity":
        plan.require(
            np.array_equal(transformed, carrier),
            f"{arm} arm must keep the frozen carrier byte-for-byte",
        )
        record["rows_moved"] = 0
    elif action == "zero":
        plan.require(not bool(np.any(transformed)), "f0 arm must be exactly zero")
        record["rows_moved"] = int(len(real))
    else:
        # ts4: per-column marginals unchanged (whole matrix and real-row subset),
        # padded rows untouched, real rows a permutation of the original rows.
        assert_column_marginals_equal(carrier, transformed)
        assert_column_marginals_equal(carrier[real], transformed[real])
        padded = np.flatnonzero(~unit_mask)
        if len(padded):
            plan.require(
                np.array_equal(transformed[padded], carrier[padded]),
                "ts4 must leave padded rows in place",
            )
        plan.require(
            not bool(np.any(transformed[padded])) if len(padded) else True,
            "ts4 padded rows must remain zero",
        )
        order_before = np.lexsort(carrier[real].T[::-1])
        order_after = np.lexsort(transformed[real].T[::-1])
        plan.require(
            np.array_equal(carrier[real][order_before], transformed[real][order_after]),
            "ts4 real rows must be an exact row permutation of the original",
        )
        moved = int(
            np.count_nonzero(
                np.any(transformed[real] != carrier[real], axis=1)
            )
        )
        record["rows_moved"] = moved
        record["permutation_seed"] = ts4_shuffle.permutation_seed(session_name, seed)
    record["verify"] = "PASSED"
    return record


def arm_carrier_digest(
    arm: str,
    session_name: str,
    carrier: np.ndarray,
    unit_mask: np.ndarray,
    seed: int = plan.SEED,
) -> str:
    return plan.array_digest(apply_arm(arm, session_name, carrier, unit_mask, seed))


__all__ = [
    "ARM_SPECS",
    "CARRIER_ACTIONS",
    "E0_ACTIONS",
    "arm_fusion",
    "arm_e0_action",
    "apply_arm",
    "assert_column_marginals_equal",
    "verify_arm",
    "arm_carrier_digest",
]
