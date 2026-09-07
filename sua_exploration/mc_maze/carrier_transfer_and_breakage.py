"""Pure contracts for A5 transferred/aged carrier and A3 correspondence-breakage screens.

No GPU, no training, no NWB I/O in this module.  Screen runners import these
helpers and the shared DANDI forward path from ``eval_adaptation_dandi688``.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 2
PROTOCOL_ID = "carrier_transfer_and_breakage_v1"
ROW_MATCHING_RULE = "nwb_unit_index_prefix_v1"
ROW_MATCHING_RULE_VERSION = 1

# Frozen gate on the primary A5 estimand (transferred - own_truncated).
PRIMARY_TRANSFER_DELTA_GATE = 0.0  # report sign/magnitude; no pooled headline across subsets

BREAKAGE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
UNIT_SUBSET_SIZES: tuple[int, ...] = (8, 16, 32)
BREAKAGE_MODES = (
    "unit_dropout",
    "electrode_pooling",
    "unit_subset",
)

ACTIVITY_CALIBRATION_N = 30
T4_LABEL_POOL_N = 30
EVALUATION_START_TRIAL = 50
BASE_RANDOMIZATION_SEED = 20260812

SEALED_FORMAL_TEST_SESSIONS: frozenset[str] = frozenset(
    {
        "sub-C_ses-CO-20151113",
        "sub-C_ses-CO-20151116",
        "sub-C_ses-CO-20151117",
        "sub-C_ses-CO-20151119",
        "sub-C_ses-CO-20151120",
        "sub-C_ses-CO-20151201",
    }
)

VALIDATION_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)

# Frozen NWB units-table row counts for the six development validation sessions
# (``max_units_exclusive=100`` manifest, 2026-08-12 audit).  No pair has equal N.
SESSION_UNIT_COUNTS: dict[str, int] = {
    "sub-C_ses-CO-20151103": 38,
    "sub-C_ses-CO-20151104": 59,
    "sub-C_ses-CO-20151106": 60,
    "sub-C_ses-CO-20151109": 65,
    "sub-C_ses-CO-20151110": 61,
    "sub-C_ses-CO-20151112": 42,
}

_SESSION_DATE_RE = re.compile(r"ses-CO-(\d{8})")


def session_calendar_date(session_name: str) -> date:
    match = _SESSION_DATE_RE.search(session_name)
    if match is None:
        raise ValueError(f"cannot parse calendar date from session name {session_name!r}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def calendar_gap_days(donor_session: str, recipient_session: str) -> int:
    return (session_calendar_date(recipient_session) - session_calendar_date(donor_session)).days


ADMISSIBLE_TRANSFER_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (donor, recipient)
    for donor in VALIDATION_SESSIONS
    for recipient in VALIDATION_SESSIONS
    if donor != recipient
)


def _session_unit_count(session_name: str) -> int:
    try:
        return SESSION_UNIT_COUNTS[session_name]
    except KeyError as exc:
        raise ValueError(f"unknown session {session_name!r}") from exc


def transfer_pair_subset(donor_session: str, recipient_session: str) -> Literal["primary", "secondary"]:
    """Primary pairs have ``N_donor >= N_recipient`` and therefore no zero-fill."""
    if _session_unit_count(donor_session) >= _session_unit_count(recipient_session):
        return "primary"
    return "secondary"


PRIMARY_TRANSFER_PAIRS: tuple[tuple[str, str], ...] = tuple(
    pair for pair in ADMISSIBLE_TRANSFER_PAIRS if transfer_pair_subset(*pair) == "primary"
)

SECONDARY_TRANSFER_PAIRS: tuple[tuple[str, str], ...] = tuple(
    pair for pair in ADMISSIBLE_TRANSFER_PAIRS if transfer_pair_subset(*pair) == "secondary"
)

AGED_TRANSFER_PAIRS: tuple[tuple[str, str], ...] = tuple(
    pair for pair in ADMISSIBLE_TRANSFER_PAIRS if calendar_gap_days(pair[0], pair[1]) > 0
)


@dataclass(frozen=True)
class RowMatchingReceipt:
    rule: str
    rule_version: int
    donor_rows: int
    recipient_rows: int
    matched_rows: int
    donor_rows_dropped: int
    recipient_rows_zero_filled: int

    @property
    def zero_fill_fraction(self) -> float:
        if self.recipient_rows == 0:
            return 0.0
        return float(self.recipient_rows_zero_filled) / float(self.recipient_rows)


@dataclass(frozen=True)
class ZeroFillPattern:
    """Structural zero-fill contract shared by transfer and matched-truncation arms."""

    recipient_rows: int
    matched_rows: int

    @classmethod
    def from_matching(cls, matching: RowMatchingReceipt) -> ZeroFillPattern:
        return cls(recipient_rows=matching.recipient_rows, matched_rows=matching.matched_rows)

    @property
    def recipient_rows_zero_filled(self) -> int:
        return max(0, self.recipient_rows - self.matched_rows)

    @property
    def zero_fill_fraction(self) -> float:
        if self.recipient_rows == 0:
            return 0.0
        return float(self.recipient_rows_zero_filled) / float(self.recipient_rows)

    def digest(self) -> str:
        payload = f"{ROW_MATCHING_RULE}:{self.recipient_rows}:{self.matched_rows}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def zero_row_mask(self) -> np.ndarray:
        mask = np.zeros(self.recipient_rows, dtype=bool)
        if self.matched_rows < self.recipient_rows:
            mask[self.matched_rows :] = True
        return mask


@dataclass(frozen=True)
class TransferArmCarriers:
    own_full: np.ndarray
    own_truncated: np.ndarray
    transferred: np.ndarray
    zero: np.ndarray
    matching: RowMatchingReceipt
    pattern: ZeroFillPattern

    @property
    def zero_fill_fraction(self) -> float:
        return self.pattern.zero_fill_fraction


@dataclass(frozen=True)
class TransferDeltas:
    transferred_minus_own_truncated: float
    transferred_minus_own_full: float
    own_truncated_minus_own_full: float

    def as_dict(self) -> dict[str, float]:
        return {
            "transferred_minus_own_truncated": self.transferred_minus_own_truncated,
            "transferred_minus_own_full": self.transferred_minus_own_full,
            "own_truncated_minus_own_full": self.own_truncated_minus_own_full,
        }


def zero_fill_fraction(*, recipient_rows_zero_filled: int, recipient_rows: int) -> float:
    if recipient_rows <= 0:
        raise ValueError("recipient_rows must be positive")
    return float(recipient_rows_zero_filled) / float(recipient_rows)


def assert_session_allowed(session_name: str) -> None:
    if session_name in SEALED_FORMAL_TEST_SESSIONS:
        raise ValueError(
            f"refusing sealed formal-test session {session_name!r}; "
            "only development validation sessions may be opened"
        )


def assert_transfer_pair_admissible(donor_session: str, recipient_session: str) -> None:
    assert_session_allowed(donor_session)
    assert_session_allowed(recipient_session)
    pair = (donor_session, recipient_session)
    if pair not in ADMISSIBLE_TRANSFER_PAIRS:
        raise ValueError(
            f"donor/recipient pair {pair!r} is not on the pre-registered admissible list"
        )


def derived_seed(*, family: str, base_seed: int, **parts: str | int | float) -> int:
    """Stable 32-bit seed for a named intervention family."""
    payload = f"{PROTOCOL_ID}:{family}:base={base_seed}"
    for key in sorted(parts):
        payload += f":{key}={parts[key]}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def apply_row_matching_rule(
    donor_carrier: np.ndarray,
    *,
    recipient_n_units: int,
    rule: str = ROW_MATCHING_RULE,
) -> tuple[np.ndarray, RowMatchingReceipt]:
    """Map a donor ``[N_d, 4]`` carrier onto a recipient with ``N_r`` units.

    ``nwb_unit_index_prefix_v1`` copies the first ``K=min(N_d, N_r)`` rows in
    fixed NWB units-table order.  Unmatched recipient tail rows receive an
    explicit standardized-zero carrier (the z4 coordinate), never donor padding
    or silent truncation.
    """
    if rule != ROW_MATCHING_RULE:
        raise ValueError(f"unsupported row-matching rule {rule!r}")
    values = np.asarray(donor_carrier, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"donor carrier must be [units,4], got {values.shape}")
    if recipient_n_units < 0:
        raise ValueError("recipient_n_units must be non-negative")
    donor_rows = int(values.shape[0])
    matched = min(donor_rows, recipient_n_units)
    if matched == 0 and recipient_n_units > 0:
        raise ValueError("row matching refused: donor has zero rows but recipient requires units")
    out = np.zeros((recipient_n_units, 4), dtype=np.float32)
    if matched:
        out[:matched] = values[:matched]
    receipt = RowMatchingReceipt(
        rule=rule,
        rule_version=ROW_MATCHING_RULE_VERSION,
        donor_rows=donor_rows,
        recipient_rows=recipient_n_units,
        matched_rows=matched,
        donor_rows_dropped=max(0, donor_rows - matched),
        recipient_rows_zero_filled=max(0, recipient_n_units - matched),
    )
    return out, receipt


def apply_zero_fill_pattern(
    source_carrier: np.ndarray,
    pattern: ZeroFillPattern,
) -> np.ndarray:
    """Apply the protocol zero-fill tail to any source carrier once per pair."""
    values = np.asarray(source_carrier, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"source carrier must be [units,4], got {values.shape}")
    if values.shape[0] < pattern.matched_rows:
        raise ValueError(
            "source carrier shorter than matched_rows: "
            f"{values.shape[0]} < {pattern.matched_rows}"
        )
    out = np.zeros((pattern.recipient_rows, 4), dtype=np.float32)
    out[: pattern.matched_rows] = values[: pattern.matched_rows]
    return out


def assert_shared_zero_fill_pattern(
    own_truncated: np.ndarray,
    transferred: np.ndarray,
    pattern: ZeroFillPattern,
) -> None:
    """Fail closed unless both arms share byte-identical zero-fill tails."""
    if own_truncated.shape != transferred.shape:
        raise ValueError("carrier shapes disagree under shared zero-fill pattern")
    if own_truncated.shape[0] != pattern.recipient_rows:
        raise ValueError("carrier rows disagree with zero-fill pattern recipient_rows")
    if pattern.matched_rows < pattern.recipient_rows:
        tail_own = np.asarray(own_truncated[pattern.matched_rows :], dtype=np.float32)
        tail_trans = np.asarray(transferred[pattern.matched_rows :], dtype=np.float32)
        if not np.array_equal(tail_own, tail_trans):
            raise ValueError("zero-fill tail bytes differ between transfer and own_truncated")
        if not np.allclose(tail_own, 0.0):
            raise ValueError("zero-fill tail must be standardized zero")


def build_transfer_arm_carriers(
    own_carrier: np.ndarray,
    donor_carrier: np.ndarray,
    *,
    recipient_n_units: int,
    rule: str = ROW_MATCHING_RULE,
) -> TransferArmCarriers:
    """Derive transfer and matched-truncation carriers from one shared zero-fill pattern."""
    own = np.asarray(own_carrier, dtype=np.float32)
    donor = np.asarray(donor_carrier, dtype=np.float32)
    if own.shape[0] != recipient_n_units:
        raise ValueError(
            f"own_carrier rows {own.shape[0]} != recipient_n_units {recipient_n_units}"
        )
    _, matching = apply_row_matching_rule(
        donor,
        recipient_n_units=recipient_n_units,
        rule=rule,
    )
    pattern = ZeroFillPattern.from_matching(matching)
    transferred = apply_zero_fill_pattern(donor, pattern)
    own_truncated = apply_zero_fill_pattern(own, pattern)
    assert_shared_zero_fill_pattern(own_truncated, transferred, pattern)
    return TransferArmCarriers(
        own_full=own.copy(),
        own_truncated=own_truncated,
        transferred=transferred,
        zero=zero_carrier(recipient_n_units),
        matching=matching,
        pattern=pattern,
    )


def compute_transfer_deltas(scores: Mapping[str, float]) -> TransferDeltas:
    required = ("own_carrier", "own_truncated", "transferred_carrier")
    missing = [name for name in required if name not in scores]
    if missing:
        raise ValueError(f"transfer scores missing arms: {missing}")
    transferred = float(scores["transferred_carrier"])
    own_full = float(scores["own_carrier"])
    own_truncated = float(scores["own_truncated"])
    return TransferDeltas(
        transferred_minus_own_truncated=transferred - own_truncated,
        transferred_minus_own_full=transferred - own_full,
        own_truncated_minus_own_full=own_truncated - own_full,
    )


def require_matching_shapes_or_apply_rule(
    donor_carrier: np.ndarray,
    *,
    recipient_n_units: int,
    rule: str,
) -> np.ndarray:
    """Fail closed when a caller attempts a raw shape paste across sessions."""
    values = np.asarray(donor_carrier, dtype=np.float32)
    if values.shape[0] == recipient_n_units:
        return values.copy()
    if rule != ROW_MATCHING_RULE:
        raise ValueError(f"cannot paste donor carrier without rule {ROW_MATCHING_RULE!r}")
    transferred, _ = apply_row_matching_rule(
        values, recipient_n_units=recipient_n_units, rule=rule
    )
    return transferred


def unit_dropout_indices(
    n_units: int,
    *,
    dropout_p: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError(f"dropout_p must be in [0,1], got {dropout_p}")
    if n_units <= 0:
        raise ValueError("n_units must be positive")
    if dropout_p == 0.0:
        return np.arange(n_units, dtype=np.int64)
    generator = np.random.RandomState(seed)
    keep_mask = generator.rand(n_units) >= dropout_p
    if not np.any(keep_mask):
        raise ValueError(
            f"unit dropout p={dropout_p} removed every unit for n_units={n_units}; "
            "refusing degenerate forward"
        )
    return np.flatnonzero(keep_mask).astype(np.int64, copy=False)


def apply_unit_dropout_consistent(
    *,
    neural: np.ndarray,
    side_features: np.ndarray | None,
    calib_trials: np.ndarray,
    unit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Drop identical unit columns from neural activity, carrier, and calibration."""
    idx = np.asarray(unit_indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("unit_indices must be rank 1")
    n_units = neural.shape[1]
    if calib_trials.shape[-1] != n_units:
        raise ValueError(
            "calib_trials unit axis disagrees with neural unit count: "
            f"{calib_trials.shape[-1]} vs {n_units}"
        )
    if side_features is not None and side_features.shape[0] != n_units:
        raise ValueError(
            "side_features rows disagree with neural unit count: "
            f"{side_features.shape[0]} vs {n_units}"
        )
    neural_out = neural[:, idx]
    calib_out = calib_trials[..., idx]
    side_out = None if side_features is None else side_features[idx]
    return neural_out, side_out, calib_out


def _pool_trial_rates_by_electrode(
    trial_rates: np.ndarray,
    electrode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local copy of ``unit_side_features.pool_trial_rates_by_electrode`` for import isolation."""
    if trial_rates.ndim != 2:
        raise ValueError(f"Expected trial rates [units, trials], got {trial_rates.shape}")
    if electrode_ids.ndim != 1 or electrode_ids.shape[0] != trial_rates.shape[0]:
        raise ValueError("electrode_ids must contain exactly one value per unit")
    channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
    pooled = np.zeros((channel_ids.size, trial_rates.shape[1]), dtype=np.float64)
    np.add.at(pooled, inverse, trial_rates)
    return pooled, channel_ids


def apply_electrode_pooling_consistent(
    *,
    neural: np.ndarray,
    side_features: np.ndarray,
    calib_trials: np.ndarray,
    electrode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic sorted-unit to electrode aggregation for every stream."""
    ids = np.asarray(electrode_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("electrode_ids must be rank 1")
    n_units = neural.shape[1]
    if ids.shape[0] != n_units:
        raise ValueError("electrode_ids length must match neural unit count")
    if side_features.shape[0] != n_units:
        raise ValueError("side_features rows must match neural unit count")
    if calib_trials.shape[-1] != n_units:
        raise ValueError("calib_trials unit axis must match neural unit count")

    units_by_trial = np.moveaxis(neural, -1, 0)  # [units, T, B]
    t_count, b_count = units_by_trial.shape[1], units_by_trial.shape[2]
    flat = units_by_trial.reshape(n_units, t_count * b_count)
    pooled_flat, channel_ids = _pool_trial_rates_by_electrode(flat, ids)
    pooled_neural = pooled_flat.reshape(channel_ids.size, t_count, b_count)
    pooled_neural = np.moveaxis(pooled_neural, 0, -1)  # [T, B, C]

    calib_units = np.moveaxis(calib_trials, -1, 0)  # [units, calib, trial_len]
    calib_flat = calib_units.reshape(n_units, calib_units.shape[1] * calib_units.shape[2])
    pooled_calib_flat, calib_channel_ids = _pool_trial_rates_by_electrode(calib_flat, ids)
    if not np.array_equal(channel_ids, calib_channel_ids):
        raise ValueError("electrode pooling produced inconsistent channel order")
    pooled_calib = pooled_calib_flat.reshape(
        channel_ids.size, calib_units.shape[1], calib_units.shape[2]
    )
    pooled_calib = np.moveaxis(pooled_calib, 0, -1)

    # Side features: sum within electrode then divide by group size (mean of rows).
    inverse = np.searchsorted(channel_ids, ids)
    counts = np.bincount(inverse, minlength=channel_ids.size).astype(np.float32)
    summed = np.zeros((channel_ids.size, side_features.shape[1]), dtype=np.float32)
    np.add.at(summed, inverse, side_features)
    pooled_side = summed / np.maximum(counts[:, None], 1.0)
    return pooled_neural, pooled_side, pooled_calib, channel_ids


def unit_subset_indices(n_units: int, *, subset_size: int, seed: int) -> np.ndarray:
    if subset_size <= 0 or subset_size > n_units:
        raise ValueError(f"subset_size must be in 1..{n_units}, got {subset_size}")
    generator = np.random.RandomState(seed)
    return np.sort(generator.choice(n_units, size=subset_size, replace=False)).astype(np.int64)


def zero_carrier(n_units: int) -> np.ndarray:
    # z4 is mask_standardized_t4(..., arm="z4") == zeros_like.
    return np.zeros((n_units, 4), dtype=np.float32)


def breakage_interaction_statistic(
    carrier_scores: Sequence[float],
    control_scores: Sequence[float],
    breakage_levels: Sequence[float] = BREAKAGE_LEVELS,
) -> float:
    """Primary A3 statistic: change in (carrier - control) from least to most breakage."""
    if len(carrier_scores) != len(control_scores):
        raise ValueError("carrier and control score ladders must have equal length")
    if len(carrier_scores) != len(breakage_levels):
        raise ValueError("score ladders must match the pre-registered breakage ladder")
    deltas = [float(c) - float(z) for c, z in zip(carrier_scores, control_scores)]
    return float(deltas[-1] - deltas[0])


def pooled_mean(scores: Mapping[str, float]) -> float:
    if not scores:
        raise ValueError("cannot pool an empty score map")
    return float(np.mean(list(scores.values())))


@dataclass(frozen=True)
class SyntheticSessionFixture:
    session_name: str
    neural: np.ndarray
    calib_trials: np.ndarray
    carrier: np.ndarray
    electrode_ids: np.ndarray | None = None
    carrier_sensitivity: float = 0.25
    control_breakage_slope: float = 0.40
    carrier_breakage_slope: float = 0.10
    zero_fill_penalty_slope: float = 0.50

    @property
    def n_units(self) -> int:
        return int(self.neural.shape[1])


def synthetic_transfer_arm_score(
    fixture: SyntheticSessionFixture,
    carrier: np.ndarray,
    *,
    zero_fill_fraction: float,
) -> float:
    """Deterministic stand-in for transfer-arm scoring in unit tests."""
    side = np.asarray(carrier, dtype=np.float32)
    activity_term = float(np.mean(fixture.neural))
    carrier_term = float(np.mean(side))
    return (
        0.55
        + 0.05 * activity_term
        + 0.03 * float(np.mean(fixture.calib_trials))
        + fixture.carrier_sensitivity * carrier_term
        - fixture.zero_fill_penalty_slope * float(zero_fill_fraction)
    )


def synthetic_forward_score(
    fixture: SyntheticSessionFixture,
    *,
    arm: Literal["carrier", "control"],
    breakage_level: float,
    unit_indices: np.ndarray | None = None,
    transferred_carrier: np.ndarray | None = None,
) -> float:
    """Deterministic stand-in for ``eval_r2`` in unit tests (no torch)."""
    idx = (
        np.arange(fixture.n_units, dtype=np.int64)
        if unit_indices is None
        else np.asarray(unit_indices, dtype=np.int64)
    )
    neural, side, calib = apply_unit_dropout_consistent(
        neural=fixture.neural,
        side_features=(
            transferred_carrier
            if transferred_carrier is not None
            else (fixture.carrier if arm == "carrier" else zero_carrier(fixture.n_units))
        ),
        calib_trials=fixture.calib_trials,
        unit_indices=idx,
    )
    activity_term = float(np.mean(neural))
    carrier_term = 0.0 if side is None else float(np.mean(side))
    calib_term = float(np.mean(calib))
    base = 0.55 + 0.05 * activity_term + 0.03 * calib_term
    if arm == "carrier":
        base += fixture.carrier_sensitivity * carrier_term
        penalty = fixture.carrier_breakage_slope
    else:
        penalty = fixture.control_breakage_slope
    return base - penalty * float(breakage_level)


def receipt_without_timestamps(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy with volatile timestamp fields removed."""
    def _strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: _strip(item)
                for key, item in value.items()
                if key not in {"created_at", "completed_at", "started_at"}
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    return _strip(dict(payload))


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
