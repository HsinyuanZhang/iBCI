#!/usr/bin/env python3
"""CPU-only RT AFC4 calibration-duration and descriptor-reliability audit.

This is a diagnostic companion to the frozen M24 RT AFC4 pipeline, not an
experiment runner.  It never constructs a decoder, optimizer, DataModule,
GPU tensor, formal held-out split, or training window.  The lower support
budgets (M6/M12/M18) are *diagnostic prefixes only*: production RT AFC4
remains the fixed M24 contract.  They are fitted here with the same raw 20-ms,
same-reach, +40-ms behaviour-lag OLS calculation solely to quantify descriptor
stability as calibration duration grows.

For every sorted-SUA RT session and each budget, the receipt records:

* chronological trial budget and cue-qualified reaches;
* event-qualified and velocity-active raw bins, plus usable AFC4 blocks;
* OLS rank/condition (rank-deficient fits remain explicitly undefined);
* full-prefix versus M24 descriptor agreement;
* chronological-half and odd/even-trial-half reliability of per-unit W
  direction, ||W|| and baseline b.

M24 is additionally checked bit-for-bit against the production
``k4_from_raw_calibration`` implementation.  No core loader/datamodule/config
is modified by this script.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.falcon_k4_features import (  # noqa: E402
    K4_ACTIVE_EPSILON,
    K4_BEHAVIOR_LEAD_BINS,
    K4_BLOCK_WIDTH_BINS,
    K4_RAW_BIN_MS,
    k4_from_raw_calibration,
)
from src.data.rt_k4_loader import (  # noqa: E402
    find_rt_sessions,
    load_rt_session,
    summarize_rt_trial_budget,
)


DEFAULT_BUDGETS = (6, 12, 18, 24)
EPS = 1.0e-12


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _trial_bounds(trial_change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.asarray(trial_change, dtype=bool))
    if starts.size < 1:
        raise ValueError("RT audit needs at least one chronological trial start")
    return starts, np.r_[starts[1:], len(trial_change)]


def _selected_trial_indices(budget: int, *, mode: str) -> np.ndarray:
    if mode == "prefix":
        return np.arange(budget, dtype=np.int64)
    if mode == "chronological_first_half":
        return np.arange(0, budget // 2, dtype=np.int64)
    if mode == "chronological_second_half":
        return np.arange(budget // 2, budget, dtype=np.int64)
    if mode == "even_trials":
        return np.arange(0, budget, 2, dtype=np.int64)
    if mode == "odd_trials":
        return np.arange(1, budget, 2, dtype=np.int64)
    raise ValueError(f"unknown split mode {mode!r}")


def _prefix_event_counts(
    raw: dict[str, Any], trial_indices: Sequence[int]
) -> dict[str, Any]:
    """Count event qualification without pretending invalid trials vanished."""
    indices = np.asarray(trial_indices, dtype=np.int64)
    records = list(raw["rt_segment_audit"]["trial_records"])
    starts, ends = _trial_bounds(raw["trial_change"])
    if indices.size == 0:
        return {
            "trials_selected": 0,
            "complete_cue_trials": 0,
            "trials_with_accepted_reaches": 0,
            "accepted_reach_segments": 0,
            "excluded_reach_segments": 0,
            "event_qualified_bins": 0,
            "velocity_active_event_bins": 0,
            "trial_exclusion_reasons": {},
            "segment_exclusion_reasons": {},
        }
    if indices.min() < 0 or indices.max() >= len(records):
        raise ValueError("requested reliability split exceeds session trial count")

    segment_ids = np.asarray(raw["k4_segment_id"], dtype=np.int64)
    velocity = np.asarray(raw["covariates"], dtype=np.float64)
    active = ~np.all(np.abs(velocity) < K4_ACTIVE_EPSILON, axis=1)
    trial_reasons: Counter[str] = Counter()
    segment_reasons: Counter[str] = Counter()
    complete = 0
    accepted_trials = 0
    accepted_segments = 0
    excluded_segments = 0
    event_bins = 0
    active_event_bins = 0
    for trial_index in indices.tolist():
        record = records[trial_index]
        complete += int(bool(record["complete_cue"]))
        accepted_trials += int(int(record["accepted_segments"]) > 0)
        accepted_segments += int(record["accepted_segments"])
        excluded_segments += int(record["excluded_segments"])
        if record.get("exclusion_reason"):
            trial_reasons[str(record["exclusion_reason"])] += 1
        for key, value in dict(record.get("segment_exclusion_reasons", {})).items():
            segment_reasons[str(key)] += int(value)
        left, right = int(starts[trial_index]), int(ends[trial_index])
        qualified = segment_ids[left:right] >= 0
        event_bins += int(qualified.sum())
        active_event_bins += int((qualified & active[left:right]).sum())
    return {
        "trials_selected": int(indices.size),
        "complete_cue_trials": int(complete),
        "trials_with_accepted_reaches": int(accepted_trials),
        "accepted_reach_segments": int(accepted_segments),
        "excluded_reach_segments": int(excluded_segments),
        "event_qualified_bins": int(event_bins),
        "velocity_active_event_bins": int(active_event_bins),
        "trial_exclusion_reasons": {key: int(trial_reasons[key]) for key in sorted(trial_reasons)},
        "segment_exclusion_reasons": {key: int(segment_reasons[key]) for key in sorted(segment_reasons)},
    }


def _collect_segment_constrained_blocks(
    raw: dict[str, Any], trial_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Use exactly the production AFC4 raw block and segment eligibility rule."""
    neural = np.asarray(raw["neural"], dtype=np.float64)
    velocity = np.asarray(raw["covariates"], dtype=np.float64)
    segment_ids = np.asarray(raw["k4_segment_id"], dtype=np.int64)
    starts, ends = _trial_bounds(raw["trial_change"])
    indices = np.asarray(trial_indices, dtype=np.int64)
    if indices.size == 0:
        return (
            np.empty((0, neural.shape[1]), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            {
                "candidate_trial_bounded_blocks": 0,
                "segment_qualified_blocks": 0,
                "active_blocks": 0,
            },
        )
    if indices.min() < 0 or indices.max() >= starts.size:
        raise ValueError("requested block split exceeds session trial count")

    active = ~np.all(np.abs(velocity) < K4_ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []
    behavior: list[np.ndarray] = []
    candidate_blocks = 0
    segment_qualified_blocks = 0
    for trial_index in indices.tolist():
        start, end = int(starts[trial_index]), int(ends[trial_index])
        for left in range(
            start,
            end - K4_BLOCK_WIDTH_BINS - K4_BEHAVIOR_LEAD_BINS + 1,
            K4_BLOCK_WIDTH_BINS,
        ):
            candidate_blocks += 1
            right = left + K4_BLOCK_WIDTH_BINS
            y_left = left + K4_BEHAVIOR_LEAD_BINS
            y_right = y_left + K4_BLOCK_WIDTH_BINS
            labels = segment_ids[left:y_right]
            if labels.shape[0] != K4_BLOCK_WIDTH_BINS + K4_BEHAVIOR_LEAD_BINS:
                raise RuntimeError("unexpected AFC4 neural/lagged-behaviour block span")
            if labels[0] < 0 or not np.all(labels == labels[0]):
                continue
            segment_qualified_blocks += 1
            if not (active[left:right].all() and active[y_left:y_right].all()):
                continue
            rates.append(
                neural[left:right].sum(axis=0)
                / (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0)
            )
            behavior.append(velocity[y_left:y_right].mean(axis=0))
    return (
        np.asarray(rates, dtype=np.float64),
        np.asarray(behavior, dtype=np.float64),
        {
            "candidate_trial_bounded_blocks": int(candidate_blocks),
            "segment_qualified_blocks": int(segment_qualified_blocks),
            "active_blocks": int(len(rates)),
        },
    )


def _fit_descriptor(rates: np.ndarray, behavior: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Standalone lower-budget diagnostic fit; production M24 is parity-checked separately."""
    if rates.shape[0] != behavior.shape[0]:
        raise ValueError("AFC4 diagnostic rates/behavior block count mismatch")
    if rates.shape[0] < 3:
        return None, {
            "status": "undefined",
            "reason": "fewer_than_three_active_blocks",
            "active_blocks": int(rates.shape[0]),
            "design_rank": None,
            "design_condition": None,
        }
    design = np.column_stack([np.ones(behavior.shape[0], dtype=np.float64), behavior])
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else float("inf")
    if rank != 3 or not np.isfinite(condition):
        return None, {
            "status": "undefined",
            "reason": "rank_deficient_or_nonfinite_condition",
            "active_blocks": int(rates.shape[0]),
            "design_rank": rank,
            "design_condition": None if not np.isfinite(condition) else condition,
        }
    coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, rates, rcond=None)
    if int(fitted_rank) != 3:
        return None, {
            "status": "undefined",
            "reason": "lstsq_rank_deficient",
            "active_blocks": int(rates.shape[0]),
            "design_rank": int(fitted_rank),
            "design_condition": condition,
        }
    weights = coefficients[1:].T
    descriptor = np.column_stack(
        [weights[:, 0], weights[:, 1], np.linalg.norm(weights, axis=1), coefficients[0]]
    ).astype(np.float32)
    return descriptor, {
        "status": "defined",
        "reason": None,
        "active_blocks": int(rates.shape[0]),
        "design_rank": rank,
        "design_condition": condition,
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        return {"status": "undefined", "reason": "insufficient_or_mismatched_units", "value": None}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"status": "undefined", "reason": "nonfinite_descriptor", "value": None}
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return {"status": "undefined", "reason": "zero_variance_across_units", "value": None}
    return {"status": "defined", "reason": None, "value": float(np.corrcoef(x, y)[0, 1])}


def _descriptor_reliability(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    left_fit: dict[str, Any],
    right_fit: dict[str, Any],
) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "status": "undefined",
            "reason": "left_or_right_fit_undefined",
            "left_fit": left_fit,
            "right_fit": right_fit,
            "w_direction_cosine": None,
            "w_norm_reliability": None,
            "baseline_b_reliability": None,
        }
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("descriptor reliability requires matching [N,4] inputs")
    w_left, w_right = left[:, :2].astype(np.float64), right[:, :2].astype(np.float64)
    length_left = np.linalg.norm(w_left, axis=1)
    length_right = np.linalg.norm(w_right, axis=1)
    direction_valid = (length_left > EPS) & (length_right > EPS)
    if direction_valid.any():
        cosine = np.sum(w_left[direction_valid] * w_right[direction_valid], axis=1) / (
            length_left[direction_valid] * length_right[direction_valid]
        )
        cosine = np.clip(cosine, -1.0, 1.0)
        direction = {
            "status": "defined",
            "reason": None,
            "defined_units": int(direction_valid.sum()),
            "total_units": int(left.shape[0]),
            "mean": float(np.mean(cosine)),
            "median": float(np.median(cosine)),
            "p05": float(np.quantile(cosine, 0.05)),
            "p95": float(np.quantile(cosine, 0.95)),
        }
    else:
        direction = {
            "status": "undefined",
            "reason": "all_units_have_zero_w_norm",
            "defined_units": 0,
            "total_units": int(left.shape[0]),
        }
    return {
        "status": "defined",
        "reason": None,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "w_direction_cosine": direction,
        "w_norm_reliability": _pearson(length_left, length_right),
        "baseline_b_reliability": _pearson(left[:, 3], right[:, 3]),
        "w_x_reliability": _pearson(left[:, 0], right[:, 0]),
        "w_y_reliability": _pearson(left[:, 1], right[:, 1]),
    }


def _fit_split(raw: dict[str, Any], trial_indices: Sequence[int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    rates, behavior, block_counts = _collect_segment_constrained_blocks(raw, trial_indices)
    descriptor, fit = _fit_descriptor(rates, behavior)
    return descriptor, {"event_counts": _prefix_event_counts(raw, trial_indices), "block_counts": block_counts, "fit": fit}


def _audit_budget(raw: dict[str, Any], budget: int, m24_descriptor: np.ndarray | None) -> dict[str, Any]:
    prefix_indices = _selected_trial_indices(budget, mode="prefix")
    descriptor, full = _fit_split(raw, prefix_indices)
    chronological_left, chronological_left_audit = _fit_split(
        raw, _selected_trial_indices(budget, mode="chronological_first_half")
    )
    chronological_right, chronological_right_audit = _fit_split(
        raw, _selected_trial_indices(budget, mode="chronological_second_half")
    )
    even, even_audit = _fit_split(raw, _selected_trial_indices(budget, mode="even_trials"))
    odd, odd_audit = _fit_split(raw, _selected_trial_indices(budget, mode="odd_trials"))
    prefix_vs_m24 = _descriptor_reliability(
        descriptor,
        m24_descriptor,
        left_fit=full["fit"],
        right_fit={
            "status": "defined" if m24_descriptor is not None else "undefined",
            "reason": None if m24_descriptor is not None else "m24_fit_undefined",
        },
    )
    return {
        "budget_trials": int(budget),
        "trial_index_range": [0, int(budget)],
        "event_budget_summary": summarize_rt_trial_budget(
            raw["rt_segment_audit"], budget_trials=budget
        ),
        "full_prefix": full,
        "chronological_halves": {
            "definition": "first chronological half of budget trials versus second chronological half",
            "left_trial_indices": _selected_trial_indices(budget, mode="chronological_first_half").tolist(),
            "right_trial_indices": _selected_trial_indices(budget, mode="chronological_second_half").tolist(),
            "left": chronological_left_audit,
            "right": chronological_right_audit,
            "reliability": _descriptor_reliability(
                chronological_left,
                chronological_right,
                left_fit=chronological_left_audit["fit"],
                right_fit=chronological_right_audit["fit"],
            ),
        },
        "odd_even_halves": {
            "definition": "even versus odd chronological trial index within budget; each fit keeps full within-trial reach blocks",
            "even_trial_indices": _selected_trial_indices(budget, mode="even_trials").tolist(),
            "odd_trial_indices": _selected_trial_indices(budget, mode="odd_trials").tolist(),
            "even": even_audit,
            "odd": odd_audit,
            "reliability": _descriptor_reliability(
                even,
                odd,
                left_fit=even_audit["fit"],
                right_fit=odd_audit["fit"],
            ),
        },
        "prefix_vs_m24": {
            "definition": "full chronological M-prefix descriptor versus full chronological M24 descriptor",
            "reliability": prefix_vs_m24,
        },
        # Raw descriptor retained only as a compact integrity fingerprint, not
        # as a potentially large per-unit payload in the receipt.
        "descriptor_sha256": (
            None if descriptor is None else __import__("hashlib").sha256(descriptor.tobytes()).hexdigest()
        ),
    }


def _summary_values(rows: Iterable[dict[str, Any]], path: Sequence[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if isinstance(current, (float, int)) and np.isfinite(float(current)):
            values.append(float(current))
    return values


def _aggregate_by_budget(per_session: dict[str, Any], budgets: Sequence[int]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for budget in budgets:
        rows = [per_session[name]["budgets"][str(budget)] for name in sorted(per_session)]
        entry: dict[str, Any] = {
            "sessions": len(rows),
            "full_prefix_defined": sum(row["full_prefix"]["fit"]["status"] == "defined" for row in rows),
            "chronological_reliability_defined": sum(
                row["chronological_halves"]["reliability"]["status"] == "defined" for row in rows
            ),
            "odd_even_reliability_defined": sum(
                row["odd_even_halves"]["reliability"]["status"] == "defined" for row in rows
            ),
        }
        for label, path in {
            "chronological_w_direction_cosine_median": (
                "chronological_halves", "reliability", "w_direction_cosine", "median"
            ),
            "chronological_w_norm_pearson": (
                "chronological_halves", "reliability", "w_norm_reliability", "value"
            ),
            "chronological_b_pearson": (
                "chronological_halves", "reliability", "baseline_b_reliability", "value"
            ),
            "odd_even_w_direction_cosine_median": (
                "odd_even_halves", "reliability", "w_direction_cosine", "median"
            ),
            "odd_even_w_norm_pearson": (
                "odd_even_halves", "reliability", "w_norm_reliability", "value"
            ),
            "odd_even_b_pearson": (
                "odd_even_halves", "reliability", "baseline_b_reliability", "value"
            ),
            "prefix_vs_m24_w_direction_cosine_median": (
                "prefix_vs_m24", "reliability", "w_direction_cosine", "median"
            ),
            "prefix_vs_m24_w_norm_pearson": (
                "prefix_vs_m24", "reliability", "w_norm_reliability", "value"
            ),
            "prefix_vs_m24_b_pearson": (
                "prefix_vs_m24", "reliability", "baseline_b_reliability", "value"
            ),
        }.items():
            values = _summary_values(rows, path)
            entry[label] = {
                "defined_sessions": len(values),
                "median": None if not values else float(np.median(values)),
                "mean": None if not values else float(np.mean(values)),
                "minimum": None if not values else float(np.min(values)),
                "maximum": None if not values else float(np.max(values)),
            }
        aggregate[str(budget)] = entry
    return aggregate


def _m24_core_parity(raw: dict[str, Any], diagnostic_descriptor: np.ndarray | None) -> dict[str, Any]:
    core, audit = k4_from_raw_calibration(
        raw["neural"],
        raw["covariates"],
        raw["trial_change"],
        calibration_n_trials=24,
        segment_ids=raw["k4_segment_id"],
    )
    exact = diagnostic_descriptor is not None and np.array_equal(core, diagnostic_descriptor)
    _assert(exact, f"{raw['session_name']}: diagnostic M24 descriptor drifted from production core")
    return {
        "exact_float32_descriptor_match": bool(exact),
        "production_k4_audit": audit.as_dict(),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "sua_exploration/data/dandi_000688/sub-C",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Diagnostic prefix budgets only; production remains fixed at M24.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "streaming_calibration_exp/outputs/rt_k4_reliability_audit"
        / "rt_afc4_calibration_reliability_m6_m12_m18_m24_receipt.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError(
            "This diagnostic is 0-GPU by contract. Invoke with CUDA_VISIBLE_DEVICES=''."
        )
    budgets = tuple(sorted(set(int(value) for value in args.budgets)))
    _assert(budgets == DEFAULT_BUDGETS, "audit budget grid is frozen to diagnostic M6/M12/M18/M24")
    _assert(args.data_dir.is_dir(), f"RT data directory missing: {args.data_dir}")
    paths = find_rt_sessions(args.data_dir)
    _assert(len(paths) == 15, f"expected 15 RT sessions, found {len(paths)}")

    per_session: dict[str, Any] = {}
    for path in paths:
        raw = load_rt_session(path)
        starts, _ = _trial_bounds(raw["trial_change"])
        _assert(starts.size >= max(budgets), f"{raw['session_name']}: fewer than M24 trials")
        m24_descriptor, _m24_fit = _fit_split(raw, _selected_trial_indices(24, mode="prefix"))
        budgets_payload = {
            str(budget): _audit_budget(raw, budget, m24_descriptor) for budget in budgets
        }
        per_session[str(raw["session_name"])] = {
            "nwb_path": raw["nwb_path"],
            "num_units": int(raw["neural"].shape[1]),
            "trials_total": int(starts.size),
            "full_session_event_audit": {
                key: raw["rt_segment_audit"][key]
                for key in (
                    "trials_total",
                    "complete_cue_trials",
                    "trials_with_accepted_segments",
                    "excluded_trials",
                    "trial_exclusion_reasons",
                    "declared_reach_segments",
                    "accepted_reach_segments",
                    "excluded_reach_segments",
                    "segment_exclusion_reasons",
                    "event_qualified_bins",
                )
            },
            "m24_core_parity": _m24_core_parity(raw, m24_descriptor),
            "budgets": budgets_payload,
        }
        del raw
        gc.collect()

    aggregate = _aggregate_by_budget(per_session, budgets)
    m24 = aggregate["24"]
    receipt = {
        "schema": "rt_afc4_calibration_reliability_audit_v1",
        "status": "PASS__CPU_DIAGNOSTIC_ONLY__NO_GPU_NO_MODEL_NO_FORMAL_TEST",
        "execution_contract": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "decoder_constructed": False,
            "optimizer_constructed": False,
            "datamodule_constructed": False,
            "formal_heldout_opened": False,
            "production_rt_contract_modified": False,
        },
        "production_contract": {
            "production_budget_trials": 24,
            "production_descriptor": "per-unit [w_x,w_y,||W||_2,b] from raw same-reach 100-ms/+40-ms blocks",
            "diagnostic_budget_grid": list(budgets),
            "lower_budget_interpretation": (
                "M6/M12/M18 are descriptor-duration diagnostics only.  They are not authorised model "
                "arms and do not relax the fixed M24 production data contract."
            ),
        },
        "split_reliability_definitions": {
            "chronological": "first half of chronological support trials versus second half",
            "odd_even": "even versus odd chronological trial indices; no within-trial reach split",
            "w_direction": "per-unit cosine between 2-D W vectors; session summary is across-unit median",
            "w_norm": "Pearson correlation across units of ||W||_2",
            "baseline_b": "Pearson correlation across units of b",
            "undefined_policy": "rank-deficient/non-finite/fewer-than-three-block fits are recorded as undefined, never zero-filled",
        },
        "aggregate_by_budget": aggregate,
        "m24_summary": {
            "all_sessions_full_prefix_defined": m24["full_prefix_defined"],
            "all_sessions_m24_core_parity": all(
                item["m24_core_parity"]["exact_float32_descriptor_match"]
                for item in per_session.values()
            ),
        },
        "per_session": per_session,
        "paper_language_guard": (
            "Describe these as within-session, event-qualified AFC4 descriptor split-half reliability diagnostics. "
            "Do not call them task decoding accuracy, a held-out result, or a causal efficacy result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n")
    print(f"WROTE_RT_AFC4_RELIABILITY_RECEIPT={args.output}")
    print(f"STATUS={receipt['status']}")


if __name__ == "__main__":
    main()
