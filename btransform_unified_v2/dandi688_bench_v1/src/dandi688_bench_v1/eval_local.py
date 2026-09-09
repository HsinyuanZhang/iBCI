"""Val-face local evaluation: equal-session mean R^2 with test-split refusal.

Scoring protocol (same convention as the rift_v1 score receipt and the other
tasks in this series):

  - per session: R^2 = 1 - sum((target - pred)^2) / sum((target -
    colmean(target))^2), i.e. the session's columns are mean-centered and the
    residual/total sums are taken over all queries and output dimensions
    (variance-weighted within a session), then
  - across the exam sessions of the active protocol (ADDENDUM-TWO-STAGE:
    4 sessions for exp1_narrow, 6 for exp2_full): unweighted (equal-session)
    mean of the per-session R^2 values.

The evaluation surface is the protocol-passed exam face ("ext4-equivalent"
for exp1_narrow, "ext6-equivalent" for exp2_full).  Loading functions fail
closed on the frozen formal-test session names.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import plan


class TestSplitForbiddenError(RuntimeError):
    """Raised when any code path tries to touch a formal-test session."""


class PreparedCacheError(RuntimeError):
    """Raised when the prepared cache is missing, malformed, or hash-mismatched."""


def load_session(cache_dir: Path | str, session_name: str) -> dict[str, Any]:
    """Load one prepared-cache session; refuse the formal-test split by name.

    Returns the session row dict (split, neural, behavior, starts, e0,
    carrier, mask).  Read-only: the immutable cache bundle is never written.
    """
    if session_name in plan.FORMAL_TEST_SESSIONS:
        raise TestSplitForbiddenError(
            f"session {session_name} is in the frozen formal-test split; "
            f"policy: {plan.FORMAL_TEST_POLICY}"
        )
    cache = Path(cache_dir)
    meta_path = cache / "prepared_contract.json"
    if not meta_path.is_file():
        raise PreparedCacheError(f"prepared cache contract missing: {meta_path}")
    meta = json.loads(meta_path.read_text())
    info = meta.get("sessions", {}).get(session_name)
    if info is None:
        raise PreparedCacheError(
            f"session {session_name} not present in prepared cache {cache}"
        )
    path = cache / "sessions" / f"{session_name}.npz"
    if not path.is_file():
        raise PreparedCacheError(f"prepared cache missing session file: {path}")
    z = np.load(path)
    row = {
        "split": info["split"],
        "neural": z["neural"], "behavior": z["behavior"], "starts": z["starts"],
        "e0": z["e0"], "carrier": z["carrier"], "mask": z["mask"],
    }
    expected = meta.get("rows_hashes", {}).get(session_name, {})
    for key in ("neural", "behavior", "starts", "e0", "carrier", "mask"):
        if plan.array_digest(row[key]) != expected.get(key):
            raise PreparedCacheError(f"array hash mismatch for {session_name}:{key}")
    return row


def session_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-session R^2 (column-mean centered total; nan when target is constant)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    plan.require(pred.shape == target.shape and pred.ndim == 2,
                 f"pred/target must be matching 2-D arrays, got {pred.shape} vs {target.shape}")
    total = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    if total <= 0.0:
        return float("nan")
    return float(1.0 - np.square(target - pred).sum() / total)


def session_mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.square(pred - target)))


def prediction_digest(pred: np.ndarray) -> str:
    return plan.array_digest(np.asarray(pred))


def evaluate_val_face(
    session_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    sessions: tuple[str, ...] | list[str] | None = None,
    face_role: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-session (pred, target) pairs into the exam-face receipt.

    ``sessions`` is the protocol-passed exam face (ADDENDUM-TWO-STAGE): the
    4 exp1_narrow exam sessions or the 6 exp2_full val sessions.  It defaults
    to ``plan.VAL_SESSIONS`` (the exp2_full face).  Formal-test session names
    are refused both inside ``session_outputs`` and inside ``sessions``.
    Primary metric: equal_session_mean_r2."""
    if sessions is None:
        sessions = plan.VAL_SESSIONS
    sessions = tuple(sessions)
    if len(set(sessions)) != len(sessions):
        raise RuntimeError("exam session list must not contain duplicates")
    for name in sessions:
        if name in plan.FORMAL_TEST_SESSIONS:
            raise TestSplitForbiddenError(
                f"refusing to score formal-test session {name}; "
                f"policy: {plan.FORMAL_TEST_POLICY}"
            )
    for name in session_outputs:
        if name in plan.FORMAL_TEST_SESSIONS:
            raise TestSplitForbiddenError(
                f"refusing to score formal-test session {name}; "
                f"policy: {plan.FORMAL_TEST_POLICY}"
            )
    n_expected = len(sessions)
    if set(session_outputs) != set(sessions):
        raise RuntimeError(
            f"val face requires exactly {n_expected} sessions "
            f"{list(sessions)}, got {len(session_outputs)}: {sorted(session_outputs)}"
        )
    detail: dict[str, Any] = {}
    all_pred, all_target = [], []
    for name in sorted(session_outputs):
        pred, target = session_outputs[name]
        pred = np.asarray(pred, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        r2 = session_r2(pred, target)
        mse = session_mse(pred, target)
        if not (np.isfinite(mse) and (np.isnan(r2) or np.isfinite(r2))):
            raise RuntimeError(f"nonfinite validation output for session {name}")
        detail[name] = {
            "n_query": int(pred.shape[0]),
            "out_dim": int(pred.shape[1]),
            "mse": mse,
            "r2": r2,
            "prediction_sha256": prediction_digest(pred),
            "target_sha256": prediction_digest(target),
        }
        all_pred.append(pred)
        all_target.append(target)
    r2s = [detail[name]["r2"] for name in sorted(detail)]
    mses = [detail[name]["mse"] for name in sorted(detail)]
    if not all(np.isfinite(v) for v in r2s):
        raise RuntimeError("nonfinite per-session R^2 on the val face")
    pooled_pred = np.concatenate(all_pred).astype(np.float64)
    pooled_target = np.concatenate(all_target).astype(np.float64)
    pooled_r2 = session_r2(pooled_pred, pooled_target)
    return {
        "schema": plan.SCHEMA + "_val_face",
        "val_face_role": plan.VAL_FACE_ROLE if face_role is None else face_role,
        "exam_sessions": list(sessions),
        "sessions": detail,
        "equal_session_mean_r2": float(np.mean(r2s)),
        "equal_session_mean_mse": float(np.mean(mses)),
        "pooled_r2": pooled_r2,
        "formal_test_used": False,
    }


__all__ = [
    "TestSplitForbiddenError",
    "PreparedCacheError",
    "load_session",
    "session_r2",
    "session_mse",
    "prediction_digest",
    "evaluate_val_face",
]
