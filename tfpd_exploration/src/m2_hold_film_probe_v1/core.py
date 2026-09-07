"""Pure helpers and the pre-registered kill/continue law."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from . import plan


class ProbeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def summarize_sessions(values: Mapping[str, float]) -> dict[str, object]:
    require(bool(values), "empty session map")
    ordered = {key: float(values[key]) for key in sorted(values)}
    array = np.asarray(list(ordered.values()), dtype=np.float64)
    require(np.isfinite(array).all(), "session R2 nonfinite")
    return {
        "session_count": int(array.size),
        "equal_session_mean": float(array.mean()),
        "equal_session_median": float(np.median(array)),
        "per_session_r2": ordered,
    }


def paired_contrast(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, object]:
    require(set(candidate) == set(reference) and bool(candidate), "paired session set mismatch")
    differences = {key: float(candidate[key] - reference[key]) for key in sorted(candidate)}
    values = np.asarray(list(differences.values()), dtype=np.float64)
    return {
        "candidate_minus_reference_mean": float(values.mean()),
        "candidate_minus_reference_median": float(np.median(values)),
        "positive_sessions": int(np.count_nonzero(values > 0.0)),
        "session_count": int(values.size),
        "per_session_delta": differences,
    }


def _mean(summary: Mapping[str, object]) -> float:
    return float(summary["equal_session_mean"])


def decide_verdict(
    *,
    p0_external: Mapping[str, object],
    p1_external: Mapping[str, object],
    p1_shuffle_external: Mapping[str, object],
    c2_external: Mapping[str, object],
    p1_vs_p0: Mapping[str, object],
    shuffle_vs_p0: Mapping[str, object],
    c2_vs_p0: Mapping[str, object],
) -> dict[str, object]:
    p0_mean = _mean(p0_external)
    p0_parity = abs(p0_mean - plan.SEALED_M30_EXTERNAL) <= plan.P0_R2_ABS_TOLERANCE
    p1_delta = float(p1_vs_p0["candidate_minus_reference_mean"])
    shuffle_delta = float(shuffle_vs_p0["candidate_minus_reference_mean"])
    c2_delta = float(c2_vs_p0["candidate_minus_reference_mean"])
    p1_positive = int(p1_vs_p0["positive_sessions"])
    joint = False
    if not p0_parity:
        code = "FAIL_P0_PARITY"
        reason = (
            "zero-init FiLM did not reproduce sealed ridge_static_m30 "
            f"({p0_mean} vs {plan.SEALED_M30_EXTERNAL})"
        )
    elif shuffle_delta > plan.GATE_SHUFFLE_MARGIN:
        code = "KILL_SHUFFLE_CAPACITY"
        reason = (
            "P1-shuffle also rose versus T4-only; the FiLM slot is fitting "
            "unlabeled extra dimensions, not labeled hold-vs-reach structure"
        )
    elif c2_delta > plan.GATE_SHUFFLE_MARGIN:
        code = "KILL_C2_CAPACITY"
        reason = (
            "train-on-shuffle also rose versus T4-only; extra 4-D context is "
            "usable as capacity even when channel labels are destroyed"
        )
    elif p1_delta <= 0.0:
        code = "KILL_NO_SIGNAL"
        reason = (
            "trained FiLM did not beat T4-only on held-out official query; "
            "these labeled contrasts do not speak the frozen identity slot. "
            "This cell does not authorize a joint encoder+decoder retrain"
        )
    elif (
        p1_delta > plan.GATE_MEAN
        and p1_positive >= plan.GATE_POSITIVE
        and shuffle_delta <= plan.GATE_SHUFFLE_MARGIN
        and c2_delta <= plan.GATE_SHUFFLE_MARGIN
    ):
        code = "CONTINUE_LABELED_SIGNAL"
        reason = (
            "labeled hold-vs-reach contrast moved frozen-decoder identity in a "
            "shuffle-controlled way. Joint encoder+decoder retrain is warranted "
            "if the goal is to beat act30_full; this freeze-decoder probe is not "
            "itself an official champion"
        )
        joint = True
    else:
        code = "WEAK_LABELED_SIGNAL"
        reason = (
            "P1 beat P0 on mean but missed the pre-registered mean/breadth gate. "
            "Do not auto-launch joint retrain from this cell"
        )
    return {
        "code": code,
        "reason": reason,
        "joint_retrain_warranted": joint,
        "p0_matches_sealed_m30": p0_parity,
        "p0_external_mean": p0_mean,
        "p1_external_mean": _mean(p1_external),
        "p1_shuffle_external_mean": _mean(p1_shuffle_external),
        "c2_external_mean": _mean(c2_external),
        "p1_minus_p0_mean": p1_delta,
        "shuffle_minus_p0_mean": shuffle_delta,
        "c2_minus_p0_mean": c2_delta,
        "p1_positive_sessions": p1_positive,
        "gate_mean": plan.GATE_MEAN,
        "gate_positive_sessions": plan.GATE_POSITIVE,
        "official_champion_claim": False,
        "tta": False,
        "evalai_push": False,
        "side_dim_expansion": False,
    }
