"""Deterministic partial carrier corruptions for the H1 fold-0 diagnostic.

The functions in this module alter only the four-trial carrier supplied to a
frozen CarrierID checkpoint.  They never alter neural activity, the identity
tensor, target values, or query-window membership.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset

from src.data.h1_m4_eb_pilot import (
    H1PilotRecord,
    PilotDataError,
    carrier_sha256,
    fit_frozen_carrier,
)


DOSE_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
REPEAT_SEEDS: tuple[int, ...] = (2026080801, 2026080802, 2026080803, 2026080804)


def _validate_dose(p: float) -> float:
    value = float(p)
    if value not in DOSE_GRID:
        raise PilotDataError(f"dose must be one of {DOSE_GRID}, got {p!r}")
    return value


def _fixed_count(p: float, total: int) -> int:
    """Map a registered nominal dose to a fixed count by round-half-up."""

    if int(total) < 2:
        raise PilotDataError("partial corruption requires at least two rows")
    return int(np.floor(_validate_dose(p) * int(total) + 0.5))


def _nested_order(total: int, *, kind: str, session_name: str, repeat_seed: int) -> np.ndarray:
    token = hashlib.sha256(
        f"h1-carrierid-dose-v1|{kind}|{session_name}|{int(repeat_seed)}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "big")).permutation(int(total))


def _cyclic_derangement(selected: np.ndarray, *, kind: str, session_name: str, repeat_seed: int) -> np.ndarray:
    values = np.asarray(selected, dtype=np.int64).reshape(-1)
    if values.size < 2:
        raise PilotDataError("nonzero corruption dose selected fewer than two rows")
    token = hashlib.sha256(
        f"h1-carrierid-dose-v1|shift|{kind}|{session_name}|{int(repeat_seed)}".encode("utf-8")
    ).digest()
    shift = 1 + int.from_bytes(token[:8], "big") % (values.size - 1)
    return np.roll(values, shift)


@dataclass(frozen=True)
class CorruptionAudit:
    kind: str
    nominal_p: float
    repeat_seed: int
    total_rows: int
    selected_rows: int
    selected_indices_sha256: str
    source_indices_sha256: str
    all_selected_assignments_deranged: bool

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "nominal_p": self.nominal_p,
            "repeat_seed": self.repeat_seed,
            "total_rows": self.total_rows,
            "selected_rows": self.selected_rows,
            "actual_selected_fraction": self.selected_rows / self.total_rows,
            "selected_indices_sha256": self.selected_indices_sha256,
            "source_indices_sha256": self.source_indices_sha256,
            "all_selected_assignments_deranged": self.all_selected_assignments_deranged,
        }


def _indices_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def partial_row_corruption(
    carrier: np.ndarray,
    *,
    session_name: str,
    p: float,
    repeat_seed: int,
) -> tuple[np.ndarray, CorruptionAudit]:
    """Permute a fixed number of carrier-to-channel attachments.

    The selected sets are nested as ``p`` increases for a given repeat.  A
    cyclic derangement preserves the selected carrier-row multiset exactly.
    """

    values = np.asarray(carrier)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) < 2:
        raise PilotDataError("row dose requires finite [N,4] carrier with N>=2")
    if not np.isfinite(values).all():
        raise PilotDataError("row dose carrier is nonfinite")
    dose = _validate_dose(p)
    count = _fixed_count(dose, len(values))
    if dose > 0.0 and count < 2:
        raise PilotDataError("registered nonzero row dose is too small")
    order = _nested_order(len(values), kind="row", session_name=session_name, repeat_seed=repeat_seed)
    selected = np.asarray(order[:count], dtype=np.int64)
    source = selected.copy() if count == 0 else _cyclic_derangement(
        selected, kind="row", session_name=session_name, repeat_seed=repeat_seed
    )
    corrupted = values.copy()
    if count:
        corrupted[selected] = values[source]
    if count and np.array_equal(corrupted, values):
        raise PilotDataError("nonzero row dose did not change carrier values")
    audit = CorruptionAudit(
        kind="row",
        nominal_p=dose,
        repeat_seed=int(repeat_seed),
        total_rows=len(values),
        selected_rows=count,
        selected_indices_sha256=_indices_sha256(selected),
        source_indices_sha256=_indices_sha256(source),
        all_selected_assignments_deranged=bool(count == 0 or np.all(selected != source)),
    )
    return np.asarray(corrupted), audit


def partial_label_overrides(
    record: H1PilotRecord,
    trial_values: Sequence[float],
    *,
    p: float,
    repeat_seed: int,
) -> tuple[dict[float, np.ndarray], CorruptionAudit]:
    """Corrupt a fixed nested subset of support-block velocity labels.

    Rates and their order remain untouched.  Only velocity rows are reassigned
    before the already frozen analytic carrier fit is rerun.
    """

    values = tuple(float(value) for value in trial_values)
    trials = tuple(record.blocks_for(value) for value in values)
    if len(trials) != 4 or any(len(trial.velocity) < 2 for trial in trials):
        raise PilotDataError("label dose requires four legal support trials")
    labels = np.concatenate([np.asarray(trial.velocity, dtype=np.float64) for trial in trials], axis=0)
    dose = _validate_dose(p)
    count = _fixed_count(dose, len(labels))
    if dose > 0.0 and count < 2:
        raise PilotDataError("registered nonzero label dose is too small")
    order = _nested_order(len(labels), kind="label", session_name=record.session_name, repeat_seed=repeat_seed)
    selected = np.asarray(order[:count], dtype=np.int64)
    source = selected.copy() if count == 0 else _cyclic_derangement(
        selected, kind="label", session_name=record.session_name, repeat_seed=repeat_seed
    )
    corrupted = labels.copy()
    if count:
        corrupted[selected] = labels[source]
    if count and np.array_equal(corrupted, labels):
        raise PilotDataError("nonzero label dose did not change velocity-label values")
    overrides: dict[float, np.ndarray] = {}
    offset = 0
    for value, trial in zip(values, trials):
        width = len(trial.velocity)
        overrides[value] = np.asarray(corrupted[offset : offset + width], dtype=np.float64)
        offset += width
    audit = CorruptionAudit(
        kind="label",
        nominal_p=dose,
        repeat_seed=int(repeat_seed),
        total_rows=len(labels),
        selected_rows=count,
        selected_indices_sha256=_indices_sha256(selected),
        source_indices_sha256=_indices_sha256(source),
        all_selected_assignments_deranged=bool(count == 0 or np.all(selected != source)),
    )
    return overrides, audit


def partial_label_carrier(
    record: H1PilotRecord,
    plan: Any,
    trial_values: Sequence[float],
    *,
    p: float,
    repeat_seed: int,
) -> tuple[np.ndarray, CorruptionAudit]:
    overrides, audit = partial_label_overrides(
        record, trial_values, p=p, repeat_seed=repeat_seed
    )
    carrier = fit_frozen_carrier(record, plan, trial_values, labels_override=overrides)["carrier"]
    return np.asarray(carrier, dtype=np.float64), audit


class CarrierOverrideTargetDataset(Dataset):
    """Read-only carrier overlay preserving a strict target dataset verbatim."""

    def __init__(self, base: Any, carriers: Mapping[str, np.ndarray], *, audit: Mapping[str, Any]):
        if set(carriers) != set(base.records):
            raise PilotDataError("carrier overlay session set differs from strict target dataset")
        self.base = base
        self.records = base.records
        self.plan = base.plan
        self.support = base.support
        self.window_indices = base.window_indices
        self.window_indices_sha256 = base.window_indices_sha256
        self.carriers = {name: np.asarray(carriers[name], dtype=np.float32) for name in base.records}
        self.audit = dict(audit)
        for name, value in self.carriers.items():
            expected = np.asarray(base.support[name].carriers["full"])
            if value.shape != expected.shape or not np.isfinite(value).all():
                raise PilotDataError(f"{name}: invalid carrier override")
        self.carrier_sha256 = {name: carrier_sha256(value) for name, value in self.carriers.items()}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        neural, target, identity, session_name, _carrier = self.base[int(index)]
        return neural, target, identity, session_name, self.carriers[session_name]
