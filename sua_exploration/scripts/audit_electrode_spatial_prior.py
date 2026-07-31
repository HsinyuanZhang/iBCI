#!/usr/bin/env python3
"""Train-only audit of a geometry-informed low-label T4 prior.

This script answers two questions without training a decoder:

1. Are neighboring *electrodes* more similar in their first-50-trial
   direction-tuning vectors than expected under coordinate permutations?
2. At low label budgets, does shrinking a unit's ``a,c`` coefficients toward
   the equally weighted mean of occupied neighboring electrodes predict future
   rates better than both shrink-to-zero and shuffled-neighborhood controls?

Only the strict manifest's training sessions are opened. Validation and formal
test NWBs are never opened. This is a mechanistic proxy and a Stage-0 go/no-go
gate, not a decoding result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
from pynwb import NWBHDF5IO

# Keep the direct ``python sua_exploration/scripts/...`` entrypoint usable from
# either the repository root or the ``sua_exploration`` directory.
SUA_ROOT = Path(__file__).resolve().parents[1]
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

try:
    from .audit_t4_confidence_predictive_validity import (
        EPS,
        _direction_design_descriptors,
        _fit_t4_matrix,
        _parse_budgets,
    )
    from .audit_t4_confidence_shrinkage import shrink_factor
except ImportError:  # Direct ``python path/to/script.py`` execution.
    from audit_t4_confidence_predictive_validity import (
        EPS,
        _direction_design_descriptors,
        _fit_t4_matrix,
        _parse_budgets,
    )
    from audit_t4_confidence_shrinkage import shrink_factor

from mc_maze.multisession_datamodule import electrode_ids_from_units
from mc_maze.unit_side_features import (
    CANONICAL_DIRECTIONS_RAD,
    _nearest_canonical_direction_index,
    _pool_trial_rate_matrix,
    list_datamodule_rewarded_trials,
)


Coord = tuple[int, int]
BankPin = tuple[str, int]
CMP_SOURCE_URL = (
    "https://raw.githubusercontent.com/limblab/"
    "deprecated_limblab_analysis/master/lib/Map%20Plotting/Maps/1025-0394.cmp"
)
DEFAULT_SHRINK_STRENGTH = 3.0
DEFAULT_NULL_DRAWS = 64
DEFAULT_BOOTSTRAP_DRAWS = 50_000
MATERIAL_MSE_RATIO = 0.98
MIN_IMPROVED_TRAIN_SESSIONS = 18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cmp(path: Path) -> dict[BankPin, Coord]:
    """Parse a Blackrock ``c r b e l`` CMP file."""

    mapping: dict[BankPin, Coord] = {}
    occupied: dict[Coord, BankPin] = {}
    saw_description = False
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if not saw_description:
            saw_description = True
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"{path}:{line_number}: expected c r b e l")
        try:
            col = int(fields[0])
            row = int(fields[1])
            pin = int(fields[3])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid numeric CMP field"
            ) from exc
        bank = fields[2]
        if bank not in {"A", "B", "C", "D"} or not 1 <= pin <= 32:
            raise ValueError(
                f"{path}:{line_number}: invalid bank/pin {(bank, pin)}"
            )
        bank_pin = (bank, pin)
        coord = (col, row)
        if bank_pin in mapping:
            raise ValueError(f"{path}:{line_number}: duplicate bank/pin {bank_pin}")
        if coord in occupied:
            raise ValueError(f"{path}:{line_number}: duplicate coordinate {coord}")
        mapping[bank_pin] = coord
        occupied[coord] = bank_pin
    if not saw_description or not mapping:
        raise ValueError(f"{path}: no CMP mapping rows")
    return mapping


def adjacency_count_distribution(
    coordinates: Mapping[int, Coord],
    *,
    radius: int = 1,
) -> dict[int, int]:
    if radius < 1:
        raise ValueError("radius must be positive")
    values = list(coordinates.values())
    counts = Counter(
        sum(
            1
            for other in values
            if other != coord
            and 0 < max(abs(other[0] - coord[0]), abs(other[1] - coord[1]))
            <= radius
        )
        for coord in values
    )
    return dict(sorted(counts.items()))


def _load_electrode_contract(
    nwb_path: Path,
    cmp_mapping: Mapping[BankPin, Coord],
) -> tuple[np.ndarray, dict[int, Coord], set[BankPin]]:
    """Return per-unit electrode rows and the row-to-coordinate mapping."""

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None or nwb.electrodes is None:
            raise ValueError(f"{nwb_path}: missing units/electrodes table")
        units_df = nwb.units.to_dataframe()
        electrodes_df = nwb.electrodes.to_dataframe()
    unit_electrodes = electrode_ids_from_units(units_df)
    row_to_coord: dict[int, Coord] = {}
    observed: set[BankPin] = set()
    for row_index, row in electrodes_df.iterrows():
        bank_pin = (str(row["bank"]), int(row["pin"]))
        if bank_pin not in cmp_mapping:
            raise ValueError(
                f"{nwb_path}: electrode bank/pin {bank_pin} is absent from CMP"
            )
        integer_row = int(row_index)
        row_to_coord[integer_row] = cmp_mapping[bank_pin]
        observed.add(bank_pin)
    if set(cmp_mapping) != observed:
        missing = sorted(set(cmp_mapping) - observed)
        extra = sorted(observed - set(cmp_mapping))
        raise ValueError(
            f"{nwb_path}: CMP/NWB bank-pin mismatch; missing={missing}, extra={extra}"
        )
    if unit_electrodes.size and not set(unit_electrodes).issubset(row_to_coord):
        raise ValueError(f"{nwb_path}: unit references an unknown electrode row")
    return unit_electrodes, row_to_coord, observed


def _electrode_mean_ac(
    ac: np.ndarray,
    electrode_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    values = np.asarray(ac, dtype=np.float64)
    ids = np.asarray(electrode_ids, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"ac must be [units,2], got {values.shape}")
    if ids.shape != (values.shape[0],):
        raise ValueError("electrode_ids must contain one row per unit")
    return {
        int(electrode): values[ids == electrode].mean(axis=0)
        for electrode in np.unique(ids)
    }


def neighbor_target_ac(
    ac: np.ndarray,
    electrode_ids: np.ndarray,
    electrode_coordinates: Mapping[int, Coord],
    *,
    radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an equal-electrode-weighted neighboring-electrode target.

    The unit's own electrode is excluded. If no *occupied* neighboring
    electrode exists, the target is exactly zero, so the resulting shrinkage
    arm is identical to shrink-to-zero for that unit. This keeps the Stage-0
    contrast isolated from a fallback-policy change.
    """

    if radius < 1:
        raise ValueError("radius must be positive")
    ids = np.asarray(electrode_ids, dtype=np.int64)
    means = _electrode_mean_ac(ac, ids)
    if not set(means).issubset(electrode_coordinates):
        raise ValueError("missing coordinates for occupied electrodes")
    target_by_electrode: dict[int, np.ndarray] = {}
    neighbor_count_by_electrode: dict[int, int] = {}
    for electrode in means:
        electrode_coord = electrode_coordinates[electrode]
        neighbors = [
            other_mean
            for other, other_mean in means.items()
            if other != electrode
            and 0
            < max(
                abs(electrode_coordinates[other][0] - electrode_coord[0]),
                abs(electrode_coordinates[other][1] - electrode_coord[1]),
            )
            <= radius
        ]
        neighbor_count_by_electrode[electrode] = len(neighbors)
        target_by_electrode[electrode] = (
            np.mean(neighbors, axis=0)
            if neighbors
            else np.zeros(2, dtype=np.float64)
        )
    targets = np.stack([target_by_electrode[int(value)] for value in ids])
    counts = np.asarray(
        [neighbor_count_by_electrode[int(value)] for value in ids],
        dtype=np.int64,
    )
    return targets, counts


def permute_electrode_coordinates(
    coordinates: Mapping[int, Coord],
    *,
    seed: int,
) -> dict[int, Coord]:
    electrodes = np.asarray(sorted(coordinates), dtype=np.int64)
    coord_values = [coordinates[int(value)] for value in electrodes]
    permutation = np.random.default_rng(seed).permutation(len(electrodes))
    return {
        int(electrode): coord_values[int(source)]
        for electrode, source in zip(electrodes, permutation)
    }


def shrink_ac(
    ac: np.ndarray,
    factor: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    values = np.asarray(ac, dtype=np.float64)
    weights = np.asarray(factor, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    if targets.shape != values.shape or weights.shape != (values.shape[0],):
        raise ValueError("incompatible ac/factor/target shapes")
    return weights[:, None] * values + (1.0 - weights[:, None]) * targets


def _future_log_mse(
    t4: np.ndarray,
    ac: np.ndarray,
    future_rates: np.ndarray,
    future_directions: np.ndarray,
) -> np.ndarray:
    theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[int(index)] for index in future_directions],
        dtype=np.float64,
    )
    predicted = (
        t4[:, 3:4]
        + ac[:, 0:1] * np.cos(theta)[None, :]
        + ac[:, 1:2] * np.sin(theta)[None, :]
    )
    mse = np.mean(np.square(future_rates - predicted), axis=1)
    return np.log(mse + EPS)


def _stable_session_seed(session: str, *, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{session}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _functional_spatial_coherence(
    t4: np.ndarray,
    electrode_ids: np.ndarray,
    coordinates: Mapping[int, Coord],
    *,
    null_draws: int,
    session: str,
) -> dict:
    means = _electrode_mean_ac(t4[:, :2], electrode_ids)
    electrodes = sorted(means)
    pair_rows: list[tuple[int, int, float]] = []
    for left_index, left in enumerate(electrodes):
        left_value = means[left]
        left_norm = float(np.linalg.norm(left_value))
        if left_norm <= EPS:
            continue
        for right in electrodes[left_index + 1 :]:
            right_value = means[right]
            right_norm = float(np.linalg.norm(right_value))
            if right_norm <= EPS:
                continue
            distance = max(
                abs(coordinates[left][0] - coordinates[right][0]),
                abs(coordinates[left][1] - coordinates[right][1]),
            )
            cosine = float(
                np.dot(left_value, right_value) / (left_norm * right_norm)
            )
            pair_rows.append((left, right, cosine))
    if not pair_rows:
        raise ValueError(f"{session}: no finite nonzero electrode T4 pairs")

    def curve_for(mapping: Mapping[int, Coord]) -> dict[int, list[float]]:
        curve: dict[int, list[float]] = {}
        for left, right, cosine in pair_rows:
            distance = max(
                abs(mapping[left][0] - mapping[right][0]),
                abs(mapping[left][1] - mapping[right][1]),
            )
            curve.setdefault(int(distance), []).append(cosine)
        return curve

    observed_curve = curve_for(coordinates)
    if 1 not in observed_curve:
        raise ValueError(f"{session}: no occupied neighboring-electrode pairs")
    observed_neighbor = float(np.mean(observed_curve[1]))
    null_neighbor = []
    for draw in range(null_draws):
        shuffled = permute_electrode_coordinates(
            coordinates,
            seed=_stable_session_seed(session, salt=10_000 + draw),
        )
        curve = curve_for(shuffled)
        if 1 in curve:
            null_neighbor.append(float(np.mean(curve[1])))
    if len(null_neighbor) != null_draws:
        raise ValueError(f"{session}: a spatial null draw had no neighbor pairs")
    return {
        "occupied_electrode_count": len(electrodes),
        "usable_pair_count": len(pair_rows),
        "distance_curve": {
            str(distance): {
                "pair_count": len(values),
                "mean_ac_cosine": float(np.mean(values)),
            }
            for distance, values in sorted(observed_curve.items())
        },
        "d1_mean_ac_cosine": observed_neighbor,
        "permuted_d1_mean_ac_cosine": float(np.mean(null_neighbor)),
        "d1_minus_permuted": observed_neighbor - float(np.mean(null_neighbor)),
        "permutation_one_sided_p": float(
            (1 + np.sum(np.asarray(null_neighbor) >= observed_neighbor))
            / (1 + null_draws)
        ),
    }


def _session_budget_scores(
    *,
    t4: np.ndarray,
    confidence: np.ndarray,
    directions: np.ndarray,
    rates: np.ndarray,
    budget: int,
    reference_pool: int,
    electrode_ids: np.ndarray,
    coordinates: Mapping[int, Coord],
    null_draws: int,
    session: str,
    shrink_strength: float,
) -> dict:
    residual_variance = np.exp(confidence[:, 0])
    _entropy, log_se_trace = _direction_design_descriptors(
        directions[:budget],
        residual_variance,
    )
    uncertainty = np.maximum(np.exp(2.0 * log_se_trace) - EPS, 0.0)
    signal = np.square(t4[:, 0]) + np.square(t4[:, 1])
    factor = shrink_factor(
        signal,
        uncertainty,
        family="wiener",
        strength=shrink_strength,
    )
    zero_ac = shrink_ac(
        t4[:, :2],
        factor,
        np.zeros_like(t4[:, :2]),
    )
    neighbor_target, neighbor_counts = neighbor_target_ac(
        t4[:, :2],
        electrode_ids,
        coordinates,
    )
    neighbor_ac = shrink_ac(t4[:, :2], factor, neighbor_target)

    valid_future = directions[budget:reference_pool] >= 0
    future_directions = directions[budget:reference_pool][valid_future]
    future_rates = rates[:, budget:reference_pool][:, valid_future]
    ordinary_log_mse = _future_log_mse(
        t4,
        t4[:, :2],
        future_rates,
        future_directions,
    )
    zero_log_mse = _future_log_mse(
        t4,
        zero_ac,
        future_rates,
        future_directions,
    )
    neighbor_log_mse = _future_log_mse(
        t4,
        neighbor_ac,
        future_rates,
        future_directions,
    )

    shuffled_rows = []
    for draw in range(null_draws):
        shuffled_coordinates = permute_electrode_coordinates(
            coordinates,
            seed=_stable_session_seed(session, salt=20_000 + budget * 100 + draw),
        )
        shuffled_target, _ = neighbor_target_ac(
            t4[:, :2],
            electrode_ids,
            shuffled_coordinates,
        )
        shuffled_rows.append(
            _future_log_mse(
                t4,
                shrink_ac(t4[:, :2], factor, shuffled_target),
                future_rates,
                future_directions,
            )
        )
    shuffled_log_mse = np.mean(np.stack(shuffled_rows), axis=0)
    return {
        "unit_count": int(t4.shape[0]),
        "future_trial_count": int(valid_future.sum()),
        "occupied_electrode_count": int(np.unique(electrode_ids).size),
        "unit_fraction_with_occupied_neighbor": float(
            np.mean(neighbor_counts > 0)
        ),
        "mean_occupied_neighbor_electrode_count_per_unit": float(
            np.mean(neighbor_counts)
        ),
        "mean_shrink_factor": float(np.mean(factor)),
        "log_mse": {
            "ordinary": ordinary_log_mse,
            "zero": zero_log_mse,
            "neighbor": neighbor_log_mse,
            "shuffled_neighbor": shuffled_log_mse,
        },
    }


def _bootstrap_ci(
    values: np.ndarray,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 20260731,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("bootstrap requires at least two session values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(draws, array.size))
    sampled = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def summarize_log_mse_contrast(
    by_session: Mapping[str, dict],
    *,
    treatment: str,
    control: str,
) -> dict:
    per_session = {}
    unit_deltas = []
    for session in sorted(by_session):
        delta = (
            by_session[session]["log_mse"][treatment]
            - by_session[session]["log_mse"][control]
        )
        unit_deltas.append(delta)
        per_session[session] = {
            "mean_log_mse_delta": float(np.mean(delta)),
            "geometric_mse_ratio": float(np.exp(np.mean(delta))),
            "unit_fraction_improved": float(np.mean(delta < 0.0)),
        }
    session_values = np.asarray(
        [row["mean_log_mse_delta"] for row in per_session.values()],
        dtype=np.float64,
    )
    ci = _bootstrap_ci(session_values)
    mean_delta = float(np.mean(session_values))
    improved = int(np.sum(session_values < 0.0))
    return {
        "treatment": treatment,
        "control": control,
        "session_count": int(session_values.size),
        "mean_session_log_mse_delta": mean_delta,
        "geometric_session_mse_ratio": float(np.exp(mean_delta)),
        "sessions_improved": improved,
        "session_bootstrap_95ci_log_delta": ci,
        "global_unit_fraction_improved": float(
            np.mean(np.concatenate(unit_deltas) < 0.0)
        ),
        "passes_material_train_only_gate": bool(
            mean_delta <= math.log(MATERIAL_MSE_RATIO)
            and improved >= MIN_IMPROVED_TRAIN_SESSIONS
            and ci[1] < 0.0
        ),
        "per_session": per_session,
    }


def summarize_coherence(by_session: Mapping[str, dict]) -> dict:
    session_deltas = np.asarray(
        [
            by_session[session]["d1_minus_permuted"]
            for session in sorted(by_session)
        ],
        dtype=np.float64,
    )
    ci = _bootstrap_ci(session_deltas, seed=20260732)
    return {
        "session_count": int(session_deltas.size),
        "mean_d1_minus_permuted_ac_cosine": float(np.mean(session_deltas)),
        "sessions_positive": int(np.sum(session_deltas > 0.0)),
        "session_bootstrap_95ci": ci,
        "passes_spatial_coherence_gate": bool(
            float(np.mean(session_deltas)) > 0.0
            and int(np.sum(session_deltas > 0.0))
            >= MIN_IMPROVED_TRAIN_SESSIONS
            and ci[0] > 0.0
        ),
        "per_session": dict(by_session),
    }


def audit(
    *,
    manifest_path: Path,
    data_dir: Path,
    cmp_path: Path,
    budgets: tuple[int, ...],
    reference_pool: int,
    null_draws: int,
    shrink_strength: float,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = list((manifest.get("session_splits") or {}).get("train") or [])
    if len(sessions) != 27 or len(set(sessions)) != 27:
        raise ValueError("spatial audit requires the strict 27-session train split")
    if any(budget >= reference_pool for budget in budgets):
        raise ValueError("each budget must be smaller than reference_pool")
    if null_draws < 2:
        raise ValueError("null_draws must be at least two")
    cmp_mapping = parse_cmp(cmp_path)
    if len(cmp_mapping) != 96:
        raise ValueError(f"expected 96 CMP electrodes, found {len(cmp_mapping)}")
    cmp_coordinates = {
        index: coord
        for index, coord in enumerate(cmp_mapping.values())
    }
    grid_distribution = adjacency_count_distribution(cmp_coordinates)

    by_budget: dict[int, dict[str, dict]] = {
        budget: {} for budget in budgets
    }
    coherence_rows = {}
    receipts = {}
    reference_bankpins: set[BankPin] | None = None
    for index, session in enumerate(sessions, start=1):
        nwb_path = (data_dir / f"{session}_behavior+ecephys.nwb").resolve()
        if nwb_path.parent != data_dir or not nwb_path.is_file():
            raise FileNotFoundError(nwb_path)
        trials = list_datamodule_rewarded_trials(
            nwb_path,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
        )
        if len(trials) < reference_pool:
            raise ValueError(
                f"{session}: requires {reference_pool} rewarded trials, "
                f"found {len(trials)}"
            )
        trials = trials[:reference_pool]
        directions = np.asarray(
            [
                _nearest_canonical_direction_index(trial["target_dir"])
                if trial.get("target_dir") is not None
                else -1
                for trial in trials
            ],
            dtype=np.int64,
        )
        rates, _ = _pool_trial_rate_matrix(nwb_path, trials)
        electrode_ids, row_to_coord, observed_bankpins = _load_electrode_contract(
            nwb_path,
            cmp_mapping,
        )
        if reference_bankpins is None:
            reference_bankpins = observed_bankpins
        elif observed_bankpins != reference_bankpins:
            raise ValueError(f"{session}: electrode bank/pin table drift")
        if electrode_ids.shape != (rates.shape[0],):
            raise ValueError(f"{session}: rate/unit-electrode rows differ")

        reference_t4, _reference_confidence = _fit_t4_matrix(
            rates[:, :reference_pool],
            directions[:reference_pool],
        )
        coherence_rows[session] = _functional_spatial_coherence(
            reference_t4,
            electrode_ids,
            row_to_coord,
            null_draws=null_draws,
            session=session,
        )
        for budget in budgets:
            t4, confidence = _fit_t4_matrix(
                rates[:, :budget],
                directions[:budget],
            )
            by_budget[budget][session] = _session_budget_scores(
                t4=t4,
                confidence=confidence,
                directions=directions,
                rates=rates,
                budget=budget,
                reference_pool=reference_pool,
                electrode_ids=electrode_ids,
                coordinates=row_to_coord,
                null_draws=null_draws,
                session=session,
                shrink_strength=shrink_strength,
            )
        receipts[session] = {
            "path": str(nwb_path),
            "sha256": _sha256(nwb_path),
        }
        print(f"[{index:02d}/{len(sessions)}] {session}", flush=True)

    coherence = summarize_coherence(coherence_rows)
    budget_results = {}
    passing_low_budgets = []
    for budget, rows in by_budget.items():
        neighbor_vs_zero = summarize_log_mse_contrast(
            rows,
            treatment="neighbor",
            control="zero",
        )
        neighbor_vs_shuffle = summarize_log_mse_contrast(
            rows,
            treatment="neighbor",
            control="shuffled_neighbor",
        )
        zero_vs_ordinary = summarize_log_mse_contrast(
            rows,
            treatment="zero",
            control="ordinary",
        )
        budget_pass = bool(
            budget in {10, 15}
            and neighbor_vs_zero["passes_material_train_only_gate"]
            and neighbor_vs_shuffle["passes_material_train_only_gate"]
        )
        if budget_pass:
            passing_low_budgets.append(budget)
        budget_results[str(budget)] = {
            "neighbor_vs_zero": neighbor_vs_zero,
            "neighbor_vs_shuffled_neighbor": neighbor_vs_shuffle,
            "zero_vs_ordinary": zero_vs_ordinary,
            "candidate_budget_for_stage1": budget_pass,
            "session_descriptors": {
                session: {
                    key: value
                    for key, value in row.items()
                    if key != "log_mse"
                }
                for session, row in rows.items()
            },
        }

    stage1 = bool(
        coherence["passes_spatial_coherence_gate"]
        and passing_low_budgets
    )
    return {
        "schema_version": 1,
        "purpose": "train_only_electrode_spatial_functional_prior_stage0",
        "interpretation_boundary": (
            "Mechanistic spatial-coherence and future-rate proxy only; "
            "does not establish decoding gain."
        ),
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "data_dir": str(data_dir),
            "split": "train",
            "session_count": 27,
            "validation_session_nwb_opened": False,
            "formal_test_session_nwb_opened": False,
            "cmp_path": str(cmp_path),
            "cmp_sha256": _sha256(cmp_path),
            "cmp_source_url": CMP_SOURCE_URL,
            "cmp_electrode_count": len(cmp_mapping),
            "cmp_adjacency": "Chebyshev radius 1, own electrode excluded",
            "cmp_adjacency_count_distribution": grid_distribution,
            "budgets": list(budgets),
            "reference_pool": reference_pool,
            "fit_trials": "chronological_rewarded_trials[0:M]",
            "future_score_trials": (
                "chronological_rewarded_trials[M:reference_pool]"
            ),
            "spatial_coherence_t4_support": (
                "chronological_rewarded_trials[0:reference_pool]"
            ),
            "neighbor_target": (
                "equal mean of per-electrode a,c means over occupied adjacent "
                "electrodes; own electrode excluded"
            ),
            "no_neighbor_fallback": (
                "zero target, exactly matching shrink-to-zero for that unit"
            ),
            "shrink_family": "wiener",
            "shrink_strength": shrink_strength,
            "null_coordinate_permutation_draws_per_session": null_draws,
            "stage1_material_future_mse_ratio": MATERIAL_MSE_RATIO,
            "stage1_min_improved_train_sessions": (
                MIN_IMPROVED_TRAIN_SESSIONS
            ),
            "stage1_low_budget_scope": [10, 15],
        },
        "spatial_coherence": coherence,
        "budgets": budget_results,
        "passing_low_budgets": passing_low_budgets,
        "stage1_candidate": stage1,
        "train_nwb_receipts": receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cmp", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=(10, 15, 20),
    )
    parser.add_argument("--reference-pool", type=int, default=50)
    parser.add_argument("--null-draws", type=int, default=DEFAULT_NULL_DRAWS)
    parser.add_argument(
        "--shrink-strength",
        type=float,
        default=DEFAULT_SHRINK_STRENGTH,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = audit(
        manifest_path=args.manifest.expanduser().resolve(),
        data_dir=args.data_dir.expanduser().resolve(),
        cmp_path=args.cmp.expanduser().resolve(),
        budgets=args.budgets,
        reference_pool=args.reference_pool,
        null_draws=args.null_draws,
        shrink_strength=args.shrink_strength,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "spatial_coherence_gate": result["spatial_coherence"][
                    "passes_spatial_coherence_gate"
                ],
                "passing_low_budgets": result["passing_low_budgets"],
                "stage1_candidate": result["stage1_candidate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
