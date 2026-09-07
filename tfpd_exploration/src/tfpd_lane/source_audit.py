"""Deterministic source-audit split + T_pre selector for the Z4 boundary pilot.

Implements the frozen preconditions of
HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md section 5 (and the Gate 0
"source-audit split and T_pre selector tests" item of section 11):

- source sessions only: every window must belong to the strict-27
  source-train roster; within-development, external-development, formal and
  organizer-held sessions are refused outright, before any byte is hashed;
- audit membership is a pure function of the frozen triple
  ``(session_id, window_start, namespace)`` under sha256 — never of data
  bytes, RNG state, dictionary order, or wall-clock — so the same roster and
  window grid always rebuild the identical audit set;
- 5% of each source session's eligible query windows, ``floor`` rounding with
  a minimum of one window, ranked ascending by digest with an exact
  ``(session_id, window_start)`` tie-break;
- the selected audit windows are removed from the pilot's gradient index set;
- ordered indices, receptive-field coverage, labels, and byte hashes are
  bound into one deterministic structure for the preflight receipt;
- ``select_t_pre`` is the sealed earliest-near-best source rule:
  ``selected_epoch = earliest e with S_e >= S_max - 0.005`` over candidate
  epochs ``0..15``, ``T_pre = selected_epoch + 1``, ``E_t4 = 48 - T_pre``.

This module is pure Python + numpy (torch is never imported); the selector
consumes the SOURCE-AUDIT column only.  Held-out columns never reach it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

NAMESPACE = "tfpd_pilot_v1"
AUDIT_FRACTION = 0.05
MIN_AUDIT_WINDOWS = 1
SELECTOR_TOLERANCE = 0.005
CANDIDATE_EPOCHS: tuple[int, ...] = tuple(range(16))
TOTAL_BUDGET_EPOCHS = 48


def window_digest(session_id: str, window_start: int, namespace: str = NAMESPACE) -> str:
    """Frozen per-window digest: sha256 of '<session_id>|<start>|<namespace>'."""
    payload = f"{session_id}|{int(window_start)}|{namespace}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_count(
    n_windows: int,
    fraction: float = AUDIT_FRACTION,
    minimum: int = MIN_AUDIT_WINDOWS,
) -> int:
    """Per-session audit size: floor(5% of eligible windows), minimum one."""
    if not isinstance(n_windows, int) or isinstance(n_windows, bool) or n_windows <= 0:
        raise ValueError(f"n_windows must be a positive integer, got {n_windows!r}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction!r}")
    return max(minimum, int(n_windows * fraction))


@dataclass(frozen=True)
class AuditPlan:
    namespace: str
    fraction: float
    roster: tuple[str, ...]
    audit_keys: tuple[tuple[str, int], ...]
    audit_positions: tuple[int, ...]
    train_positions: tuple[int, ...]
    per_session: tuple[dict, ...]
    plan_sha256: str


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_audit_plan(
    window_keys: Sequence[tuple[str, int]],
    allowed_sessions: Sequence[str],
    namespace: str = NAMESPACE,
    fraction: float = AUDIT_FRACTION,
) -> AuditPlan:
    """Build the deterministic source-audit split.

    ``window_keys`` is the ordered ``(session_id, window_start)`` grid of the
    source-train dataset (dataset order; ascending starts within a session).
    ``allowed_sessions`` is the strict-27 source-train roster in roster order;
    its order fixes the receipt order of the audit set.  Any session outside
    the roster raises before anything is selected — the split cannot silently
    admit within/external/formal windows.
    """
    roster = tuple(dict.fromkeys(str(s) for s in allowed_sessions))
    if not roster:
        raise ValueError("allowed_sessions roster is empty")
    roster_set = set(roster)

    keys = [(str(session), int(start)) for session, start in window_keys]
    for session, _ in keys:
        if session not in roster_set:
            raise ValueError(
                "source-audit split refused a non-roster session "
                f"(within/external/formal leakage guard): {session!r}"
            )

    by_session: dict[str, list[tuple[int, int]]] = {session: [] for session in roster}
    for position, (session, start) in enumerate(keys):
        by_session[session].append((position, start))

    audit_keys: list[tuple[str, int]] = []
    audit_positions: list[int] = []
    per_session: list[dict] = []
    for session in roster:
        rows = by_session[session]
        n_windows = len(rows)
        ranked = sorted(
            rows,
            key=lambda row: (window_digest(session, row[1], namespace), row[1], row[0]),
        )
        chosen = ranked[: audit_count(n_windows, fraction)]
        per_session.append(
            {
                "session": session,
                "n_windows": n_windows,
                "n_audit": len(chosen),
                "audit_fraction_of_session": len(chosen) / n_windows,
            }
        )
        for position, start in chosen:
            audit_keys.append((session, start))
            audit_positions.append(position)

    audit_set = set(audit_positions)
    train_positions = tuple(p for p in range(len(keys)) if p not in audit_set)

    plan_payload = {
        "namespace": namespace,
        "fraction": fraction,
        "roster": list(roster),
        "audit_keys": [[session, start] for session, start in audit_keys],
    }
    return AuditPlan(
        namespace=namespace,
        fraction=fraction,
        roster=roster,
        audit_keys=tuple(audit_keys),
        audit_positions=tuple(audit_positions),
        train_positions=train_positions,
        per_session=tuple(per_session),
        plan_sha256=hashlib.sha256(_canonical_json(plan_payload).encode("utf-8")).hexdigest(),
    )


def assert_no_forbidden_sessions(
    sessions: Iterable[str], forbidden: Iterable[str]
) -> None:
    """Fail closed if any forbidden (within/external/formal) session appears."""
    forbidden_map = {str(name) for name in forbidden}
    offenders = sorted({str(name) for name in sessions} & forbidden_map)
    if offenders:
        raise ValueError(f"forbidden sessions present in source-audit surface: {offenders}")


def _float32_sha256(array) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def bind_audit_windows(plan: AuditPlan, dataset, window_size: int) -> dict:
    """Bind ordered audit indices, coverage, labels, and byte hashes.

    ``dataset`` only needs ``.sessions`` — a mapping ``session -> record``
    with ``record.neural [T, N]``, ``record.behavior [T, 2]``,
    ``record.calib_trials`` and ``record.side_features`` (the exact arrays the
    pilot's gradients and scorers will read).  The binding hashes the float32
    bytes of every selected window's neural and behavior rows plus the
    session-level calibration/side tensors, so any later drift of the audited
    bytes is detectable against the immutable preflight receipt.
    """
    if len(plan.audit_positions) != len(plan.audit_keys):
        raise ValueError("plan position/key length drift")
    windows = []
    for position, (session, start) in zip(plan.audit_positions, plan.audit_keys):
        record = dataset.sessions[session]
        neural = record.neural[start : start + window_size]
        behavior = record.behavior[start : start + window_size]
        if neural.shape[0] != window_size or behavior.shape[0] != window_size:
            raise ValueError(
                f"{session}:{start} window does not cover {window_size} bins "
                f"(neural {neural.shape}, behavior {behavior.shape})"
            )
        windows.append(
            {
                "position": int(position),
                "session": session,
                "window_start": int(start),
                "window_end": int(start + window_size),
                "receptive_field_coverage_bins": [int(start), int(start + window_size)],
                "neural_sha256": _float32_sha256(neural),
                "behavior_label_sha256": _float32_sha256(behavior),
            }
        )
    sessions = []
    for session in plan.roster:
        record = dataset.sessions[session]
        sessions.append(
            {
                "session": session,
                "n_units": int(record.neural.shape[1]),
                "calib_trials_sha256": _float32_sha256(record.calib_trials),
                "side_features_sha256": _float32_sha256(record.side_features),
            }
        )
    binding = {
        "namespace": plan.namespace,
        "window_size_bins": int(window_size),
        "sessions": sessions,
        "windows": windows,
    }
    binding["binding_sha256"] = hashlib.sha256(
        _canonical_json(binding).encode("utf-8")
    ).hexdigest()
    return binding


def select_t_pre(
    scores: Mapping[int, float],
    tolerance: float = SELECTOR_TOLERANCE,
    candidate_epochs: Sequence[int] = CANDIDATE_EPOCHS,
    total_budget_epochs: int = TOTAL_BUDGET_EPOCHS,
) -> dict:
    """Sealed earliest-near-best SOURCE rule for the boundary.

    ``scores`` must carry exactly the frozen candidate endpoints (epochs
    ``0..15``) of the source-audit column — the within-dev column is never a
    legal input here.  Returns the frozen ``T_pre``/``E_t4`` decision record.
    """
    candidates = tuple(int(e) for e in candidate_epochs)
    provided = {int(e) for e in scores}
    if provided != set(candidates):
        raise ValueError(
            f"selector requires exactly the frozen candidate epochs {list(candidates)}, "
            f"got {sorted(provided)}"
        )
    values: dict[int, float] = {}
    for epoch in candidates:
        value = float(scores[epoch])
        if not math.isfinite(value):
            raise ValueError(f"non-finite source-audit score at epoch {epoch}: {value!r}")
        values[epoch] = value

    s_max = max(values.values())
    threshold = s_max - tolerance
    selected = min(e for e in candidates if values[e] >= threshold)
    t_pre = selected + 1
    return {
        "rule": (
            "selected_epoch = earliest e with S_e >= S_max - tolerance over the frozen "
            "16 source-audit endpoints; T_pre = selected_epoch + 1; E_t4 = 48 - T_pre"
        ),
        "input_column": "source_audit_only",
        "tolerance": tolerance,
        "candidate_epochs": list(candidates),
        "scores": {str(e): values[e] for e in candidates},
        "s_max": s_max,
        "threshold": threshold,
        "selected_epoch": selected,
        "t_pre": t_pre,
        "e_t4": int(total_budget_epochs) - t_pre,
        "total_budget_epochs": int(total_budget_epochs),
    }
