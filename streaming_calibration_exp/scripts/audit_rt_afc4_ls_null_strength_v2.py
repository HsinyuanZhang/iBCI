#!/usr/bin/env python3
"""CPU-only v2 RT AFC4 label-null audit: cross-reach, random, non-transferable.

V1 established that the deployed ``afc4_ls`` within-reach cyclic null is weak:
it changes block indices while retaining almost every reach's velocity
direction.  V1 also proposed a cross-reach candidate, but its 2-opt objective
explicitly minimised direction cosine.  Its near-uniform 180-degree inversion
is potentially learnable by a source-trained decoder, so V1's candidate is
*rejected for GPU use* and remains immutable historical evidence only.

V2 makes no rate- or W-dependent choice.  It creates a session-namespaced,
deterministic random cross-reach block derangement from reach IDs and support
velocity labels only.  It keeps the exact global velocity-label multiset and
the neural block layout, but deliberately does not retain within-reach label
autocorrelation: in static OLS, exact labels/marginal and association are the
relevant null properties, not an artificial temporal sequence.

The essential new gate is leave-one-session-out transfer.  On source sessions
only, a shared 2-by-2 linear map and a shared orthogonal map are fitted from
null reach summaries to correct reach summaries; both must fail to predict the
held-out session.  Thus a common rotation or sign inversion is rejected before
any GPU run is considered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
try:  # Direct script execution has scripts/ on sys.path; tests use project root.
    from scripts import audit_rt_afc4_ls_null_strength as v1
except ModuleNotFoundError:  # pragma: no cover - direct script execution only
    import audit_rt_afc4_ls_null_strength as v1


CALIBRATION_TRIALS = v1.CALIBRATION_TRIALS
SEED = v1.SEED
RT_EXPECTED_SESSION_COUNT = v1.RT_EXPECTED_SESSION_COUNT
DEFAULT_DATA_DIR = v1.DEFAULT_DATA_DIR
DEFAULT_OUTPUT = (
    WORKSPACE
    / "sua_exploration/results/rt_afc4_ls_null_strength_audit_v2"
    / "RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json"
)
SCHEMA = "rt_afc4_ls_null_strength_support_audit_v2"
STATUS = "PASS_CPU_SUPPORT_ONLY_RT_AFC4_LS_NULL_STRENGTH_AUDIT_V2"
MAX_ABS_SESSION_REACH_DIRECTION_COSINE = 0.50
MAX_ABS_AGGREGATE_REACH_DIRECTION_COSINE = 0.35
MAX_TRANSFER_MEAN_R2 = 0.20
MAX_TRANSFER_SESSION_R2 = 0.50
MAX_RANDOM_DERANGEMENT_ATTEMPTS = 128


class V2NullError(RuntimeError):
    """The V2 support-only null does not meet a frozen integrity gate."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise V2NullError(message)


def _candidate_rng(*, session_name: str, seed: int, attempt: int) -> np.random.RandomState:
    digest = hashlib.sha256(
        f"rt-afc4-ls-v2-random-cross-reach:{seed}:{session_name}:attempt={attempt}".encode()
    ).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "little"))


def deterministic_random_cross_reach_derangement(
    groups: np.ndarray, velocity: np.ndarray, *, session_name: str, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the first label-only, non-extreme random cross-reach bijection.

    Candidate selection consumes only ``groups`` and ``velocity``.  It never
    receives neural rates, AFC4 coefficients, a decoder output, or a query
    array.  Rejection removes only candidates whose *label-only* reach-summary
    direction is still near a common +1/-1 transfer in that session; it does
    not optimise toward either direction.
    """

    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    labels = np.asarray(velocity, dtype=np.float64)
    _need(labels.shape == (group.size, 2) and group.size >= 3, "v2 null needs >=3 velocity-labelled blocks")
    _need(np.unique(group).size >= 3, "v2 null needs at least three accepted reaches")
    for attempt in range(MAX_RANDOM_DERANGEMENT_ATTEMPTS):
        permutation = v1._repair_group_collisions(_candidate_rng(
            session_name=session_name, seed=seed, attempt=attempt
        ).permutation(group.size).astype(np.int64), group)
        diagnostics = v1.permutation_diagnostics(group, labels, permutation)
        transfer = diagnostics["reach_direction_transfer"]
        mean, median = transfer["mean_cosine"], transfer["median_cosine"]
        if mean is None or median is None:
            continue
        if abs(float(mean)) <= MAX_ABS_SESSION_REACH_DIRECTION_COSINE and abs(float(median)) <= MAX_ABS_SESSION_REACH_DIRECTION_COSINE:
            _need(np.array_equal(np.sort(permutation), np.arange(group.size)), "v2 null is not a bijection")
            _need(np.all(permutation != np.arange(group.size)), "v2 null left an active block unchanged")
            _need(np.all(group[permutation] != group), "v2 null retained a same-reach label")
            return permutation, {"attempt": int(attempt), "permutation_diagnostics": diagnostics}
    raise V2NullError(
        "no session-namespaced random cross-reach derangement met the label-only non-extreme direction gate"
    )


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    usable = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not usable.size:
        return {"defined": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "defined": int(usable.size), "mean": float(usable.mean()), "median": float(np.median(usable)),
        "minimum": float(usable.min()), "maximum": float(usable.max()),
    }


def reach_summary_pairs(groups: np.ndarray, velocity: np.ndarray, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return null and correct [reach,2] means, in a deterministic reach order."""

    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    labels = np.asarray(velocity, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64).reshape(-1)
    _need(labels.shape == (group.size, 2) and order.shape == group.shape, "reach-summary shape mismatch")
    _need(np.array_equal(np.sort(order), np.arange(order.size)), "reach summaries require a bijection")
    null, correct = [], []
    for reach in np.unique(group).tolist():
        indices = np.flatnonzero(group == reach)
        correct.append(labels[indices].mean(axis=0))
        null.append(labels[order[indices]].mean(axis=0))
    x, y = np.asarray(null, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    _need(x.shape == y.shape and x.shape[0] >= 3, "fewer than three reach summaries")
    return x, y


def _uncentered_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sum(np.asarray(target, dtype=np.float64) ** 2))
    _need(denominator > v1.EPS, "held-out reach targets have zero energy")
    return float(1.0 - np.sum((prediction - target) ** 2) / denominator)


def _fit_linear_map(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    _need(inputs.ndim == targets.ndim == 2 and inputs.shape == targets.shape and inputs.shape[1] == 2,
          "linear transfer fit needs matching [reach,2] arrays")
    result, _, rank, _ = np.linalg.lstsq(inputs, targets, rcond=None)
    _need(int(rank) == 2, "source null reach summaries cannot identify a 2x2 linear map")
    return result


def _fit_orthogonal_map(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    _need(inputs.ndim == targets.ndim == 2 and inputs.shape == targets.shape and inputs.shape[1] == 2,
          "orthogonal transfer fit needs matching [reach,2] arrays")
    left, _singular, right = np.linalg.svd(inputs.T @ targets, full_matrices=False)
    return left @ right


def leave_one_session_out_transfer_audit(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Test whether one source-learned 2x2 transform decodes every session.

    For each held-out session, maps are fitted only from other sessions'
    *label-only reach summaries*.  A fixed rotation or global negation gives
    R2=1 for both map families and is therefore fail-closed by the predeclared
    gates below.
    """

    _need(len(records) >= 3, "LOSO transfer audit needs at least three sessions")
    rows: list[dict[str, Any]] = []
    for index, held in enumerate(records):
        source = [item for item_index, item in enumerate(records) if item_index != index]
        source_x = np.concatenate([np.asarray(item["null_reach_means"], dtype=np.float64) for item in source], axis=0)
        source_y = np.concatenate([np.asarray(item["correct_reach_means"], dtype=np.float64) for item in source], axis=0)
        held_x, held_y = np.asarray(held["null_reach_means"], dtype=np.float64), np.asarray(held["correct_reach_means"], dtype=np.float64)
        linear = _fit_linear_map(source_x, source_y)
        orthogonal = _fit_orthogonal_map(source_x, source_y)
        rows.append({
            "held_out_session": str(held["session_name"]), "held_out_reaches": int(held_x.shape[0]),
            "linear_map": linear.tolist(), "orthogonal_map": orthogonal.tolist(),
            "linear_uncentered_r2": _uncentered_r2(held_x @ linear, held_y),
            "orthogonal_uncentered_r2": _uncentered_r2(held_x @ orthogonal, held_y),
        })
    linear_values = [row["linear_uncentered_r2"] for row in rows]
    orthogonal_values = [row["orthogonal_uncentered_r2"] for row in rows]
    linear_summary, orthogonal_summary = _summary(linear_values), _summary(orthogonal_values)
    passed = (
        linear_summary["mean"] <= MAX_TRANSFER_MEAN_R2 and linear_summary["maximum"] <= MAX_TRANSFER_SESSION_R2 and
        orthogonal_summary["mean"] <= MAX_TRANSFER_MEAN_R2 and orthogonal_summary["maximum"] <= MAX_TRANSFER_SESSION_R2
    )
    return {
        "definition": "source-session null reach summary -> correct reach summary, leave-one-session-out shared 2x2 map",
        "metrics": "uncentered R2 against zero prediction; no intercept",
        "linear": linear_summary, "orthogonal": orthogonal_summary, "rows": rows,
        "predeclared_gate": {
            "max_mean_r2": MAX_TRANSFER_MEAN_R2, "max_single_session_r2": MAX_TRANSFER_SESSION_R2,
            "pass": bool(passed), "failure_interpretation": "a common source-learnable rotation/sign/linear transfer may remain",
        },
    }


def audit_one_support(*, fold: int, path: Path) -> dict[str, Any]:
    raw = v1.load_rt_m24_support(path)
    rates, velocity, groups = v1.collect_m24_blocks(raw)
    session = str(raw["session_name"])
    permutation, selection = deterministic_random_cross_reach_derangement(
        groups, velocity, session_name=session, seed=SEED
    )
    aligned, aligned_audit = v1.k4_from_raw_calibration(
        raw["neural"], raw["velocity"], raw["trial_change"], calibration_n_trials=CALIBRATION_TRIALS,
        segment_ids=raw["segment_ids"],
    )
    null = v1.fit_descriptor(rates, velocity[permutation])
    _need(np.array_equal(v1.fit_descriptor(rates, velocity), aligned), f"{session}: source-only aligned parity drift")
    null_means, correct_means = reach_summary_pairs(groups, velocity, permutation)
    return {
        "fold": int(fold), "session_name": session, "nwb_path": str(path),
        "support_trial_index_range": raw["support_trial_index_range"], "support_cutoff_s": raw["support_cutoff_s"],
        "support_input_sha256": raw["support_input_sha256"], "num_channels": int(aligned.shape[0]),
        "active_blocks": int(rates.shape[0]), "accepted_reaches": int(np.unique(groups).size),
        "aligned_k4_audit": aligned_audit.as_dict(),
        "v2_random_cross_reach_null": {
            "selection": {"algorithm": "session-namespaced deterministic random cross-reach derangement with label-only rejection", **selection},
            "permutation_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
            "comparison_to_correct_w_diagnostic_only": v1.descriptor_comparison(aligned, null),
        },
        # These label-only data are needed for the aggregate source-LOSO
        # transform audit, then intentionally omitted from the final receipt.
        "_transfer_record": {"session_name": session, "null_reach_means": null_means, "correct_reach_means": correct_means},
    }


def _write_immutable(path: Path, body: dict[str, Any]) -> str:
    _need(not path.exists(), f"refusing to overwrite audit receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True, default=v1._json_default) + "\n")
    path.chmod(0o444)
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, "v2 receipt chmod to 0444 failed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-support-audit", action="store_true", help="explicitly open only RT M24 support prefixes")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _need(args.execute_support_audit, "refusing to open NWB without --execute-support-audit")
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "v2 audit is CPU-only; set CUDA_VISIBLE_DEVICES='' explicitly")
    paths = v1.find_rt_sessions(args.data_dir)
    _need(len(paths) == RT_EXPECTED_SESSION_COUNT, f"expected {RT_EXPECTED_SESSION_COUNT} RT sessions, found {len(paths)}")
    rows = [audit_one_support(fold=fold, path=path) for fold, path in enumerate(paths)]
    transfer_records = [row.pop("_transfer_record") for row in rows]
    transfer = leave_one_session_out_transfer_audit(transfer_records)
    per_session_direction = [row["v2_random_cross_reach_null"]["selection"]["permutation_diagnostics"]["reach_direction_transfer"] for row in rows]
    all_reach_cosines = [
        item["direction_cosine"] for transfer_item in per_session_direction
        for item in transfer_item["per_reach"] if item["direction_cosine"] is not None
    ]
    aggregate_direction = _summary(all_reach_cosines)
    label_gate = (
        aggregate_direction["defined"] > 0 and
        abs(float(aggregate_direction["mean"])) <= MAX_ABS_AGGREGATE_REACH_DIRECTION_COSINE and
        abs(float(aggregate_direction["median"])) <= MAX_ABS_AGGREGATE_REACH_DIRECTION_COSINE
    )
    receipt = {
        "schema": SCHEMA, "status": STATUS,
        "execution_contract": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "decoder_constructed": False,
            "optimizer_constructed": False, "datamodule_constructed": False, "formal_target_query_opened": False,
            "support_trial_indices_only": [0, CALIBRATION_TRIALS],
            "velocity_and_spikes_materialized": "strictly before trial-24 stop time only",
        },
        "v1_strong_candidate_rejected_reason": "systematic_antialignment_learnable_by_source_trained_consumer",
        "v2_generator_contract": {
            "uses_only": ["event-qualified reach IDs", "support velocity labels", "session name", "frozen seed"],
            "does_not_use": ["neural rates", "AFC4 W", "AFC4 b", "decoder/model score", "query data"],
            "exactly_preserves": ["global velocity-label multiset", "active block count", "neural block membership", "per-reach neural block counts"],
            "requires": ["bijection", "all active labels changed", "no same-reach label", "session direction cosine not near +1/-1"],
            "temporal_order_statement": "Within-reach velocity autocorrelation is not preserved. For static OLS, exact label multiset and broken neural-label association are the relevant null properties; temporal order is not a fitted variable.",
        },
        "label_only_direction_gate": {
            "metric": "all reach-level correct-vs-null assigned mean-velocity cosine pooled over sessions",
            "summary": aggregate_direction, "max_absolute_aggregate_mean_and_median": MAX_ABS_AGGREGATE_REACH_DIRECTION_COSINE,
            "pass": bool(label_gate), "not_sufficient_alone": "The transfer gate below is the protection against a source-learnable common map.",
        },
        "shared_transform_transfer_audit": transfer,
        "future_gpu_routing": {
            "priority_1": "Full minus existing MB4=[0,0,||W||,b] across matched 15 folds, to attribute signed direction before a new null arm.",
            "priority_2": "Consider Full minus v2 strong-LS only if both label-only direction and shared-transfer gates pass, then only after a separately reviewed matched-arm integration/receipt.",
            "current_decision": "CONSIDER_GPU_STRONG_NULL_AFTER_MATCHED_INTEGRATION" if label_gate and transfer["predeclared_gate"]["pass"] else "DO_NOT_OPEN_GPU_STRONG_NULL",
        },
        "fold_rows": rows,
    }
    digest = _write_immutable(args.output, receipt)
    print(f"WROTE_RT_AFC4_LS_NULL_STRENGTH_V2_RECEIPT={args.output}")
    print(f"SHA256={digest}")
    print(f"LABEL_ONLY_DIRECTION_GATE={label_gate}")
    print(f"TRANSFER_GATE={transfer['predeclared_gate']['pass']}")


if __name__ == "__main__":
    main()
