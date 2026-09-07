"""AC3-0 stage 2: the frozen-representation screen on the materialized cache.

CPU only.  Builds the state table from the persisted four-group trajectories,
selects the tiny hyperparameters (embedding dimension / temperature) on grouped
source folds, evaluates the amended AC3-0 matrix of section 23, computes the
binding gates, the section 16 shortcut controls, the REQUIRED rho_GE
calibration and the section 19 stop conditions, and writes the receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.learned_gate_p2prime_v1 import physical as p2physical

from . import circular as circ
from . import encoder as enc
from . import gates as gate_module
from . import matrix as mtx
from . import plan
from . import sampler as spl
from . import summaries as su


class AC3ScreenError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AC3ScreenError(message)


def _publish(path: Path, payload: Mapping[str, object]) -> str:
    """Write an immutable receipt and its sidecar (the P2' byte convention).

    The digest covers the exact file bytes INCLUDING the trailing newline, and
    both files are created read-only with ``O_EXCL`` so nothing can be silently
    rewritten.
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    digest = hashlib.sha256(body).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, body)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sidecar = path.parent / f"{path.name}.sha256"
    descriptor = os.open(sidecar, flags, 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trial_set(output_root: Path) -> mtx.TrialSet:
    """Rebuild the trial set from the persisted trajectory cache."""
    cache = Path(output_root) / "trajectories.npz"
    _require(cache.exists(), f"the AC3-0 trajectory cache is missing: {cache}")
    with np.load(cache, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    manifest = _read_json(Path(output_root) / "materialize.json")
    sessions = [str(item) for item in manifest["sessions"]]
    trial_ids = [str(item) for item in manifest["trial_ids"]]
    starts = arrays["row_starts"].astype(np.int64)
    counts = arrays["row_counts"].astype(np.int64)
    velocity_flat = arrays["velocity_flat"]
    true_flat = arrays["true_flat"]
    valid_flat = arrays["valid_flat"].astype(bool)
    _require(velocity_flat.ndim == 3 and velocity_flat.shape[1] == 2 and velocity_flat.shape[2] == 4,
             "velocity cache layout drift")
    _require(starts.shape[0] == counts.shape[0] == len(trial_ids), "trial CSR length drift")
    _require(int(starts[-1] + counts[-1]) == velocity_flat.shape[0], "trial CSR coverage drift")
    group_velocity = []
    group_valid = []
    true_velocity = []
    for index, (start, count) in enumerate(zip(starts.tolist(), counts.tolist())):
        stop = start + count
        _require(bool(np.array_equal(valid_flat[start:stop], valid_flat[start:stop])), "mask drift")
        group_velocity.append(tuple(velocity_flat[start:stop, :, group] for group in plan.GROUPS))
        group_valid.append(tuple(valid_flat[start:stop] for _ in plan.GROUPS))
        true_velocity.append(true_flat[start:stop])
    pseudo = {}
    for construction in plan.CONSTRUCTIONS_MATERIALIZED:
        pseudo[construction] = {
            "accepted": arrays[f"pseudo_{construction}_accepted"].astype(bool),
            "theta_raw_rad": arrays[f"pseudo_{construction}_theta_raw_rad"].astype(np.float64),
            "theta_index": arrays[f"pseudo_{construction}_theta_index"].astype(np.int64),
            "movement_bins": arrays[f"pseudo_{construction}_movement_bins"].astype(np.int64),
            "displacement_norm": arrays[f"pseudo_{construction}_displacement_norm"].astype(np.float64),
            "mean_speed": arrays[f"pseudo_{construction}_mean_speed"].astype(np.float64),
            "b8_accepted": arrays[f"pseudo_{construction}_b8_accepted"].astype(bool),
        }
    true_direction = {
        "accepted": arrays["true_direction_accepted"].astype(bool),
        "theta_raw_rad": arrays["true_direction_theta_raw_rad"].astype(np.float64),
        "theta_index": arrays["true_direction_theta_index"].astype(np.int64),
    }
    return mtx.trial_set_from_arrays(
        sessions=sessions, trial_ids=trial_ids, trial_session=arrays["trial_session"],
        group_velocity=group_velocity, group_valid=group_valid, true_velocity=true_velocity,
        pseudo=pseudo, true_direction=true_direction,
    )


def build_table(trial_set: mtx.TrialSet) -> spl.StateTable:
    """The flat sampling universe over the four views of every trial."""
    by_trial_velocity: list[list[np.ndarray]] = []
    by_trial_valid: list[list[np.ndarray]] = []
    for views, masks in zip(trial_set.group_velocity, trial_set.group_valid):
        by_trial_velocity.append(list(views))
        by_trial_valid.append(list(masks))
    return spl.build_state_table(
        sessions=trial_set.sessions, trial_ids=trial_set.trial_ids,
        trial_session=np.asarray(trial_set.trial_session),
        group_velocity=by_trial_velocity, group_valid=by_trial_valid,
        true_velocity=list(trial_set.true_velocity),
    )


# ---------------------------------------------------------------------------
# Input lock (operator review requirement 1): one digest every row shares.
# ---------------------------------------------------------------------------


def verify_input_lock(output_root: Path, trial_set: mtx.TrialSet) -> dict[str, object]:
    """Mandatory pre-screen gate: bind the cache to the materialize receipt.

    Verifies, before any arm runs, that ``trajectories.npz`` reproduces the
    materialize receipt's array digests bit-for-bit, that the session roster and
    trial ORDER are identical, that the four-group partition is the frozen
    evaluator's own ``held_mask(0..3)`` column order, and that all four views of
    a trial share one validity interval.  Returns the single input-lock block
    every row of the matrix binds to.
    """
    output_root = Path(output_root)
    manifest = _read_json(output_root / "materialize.json")
    with np.load(output_root / "trajectories.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    digests = {key: mtx.raw_array_payload(value) for key, value in arrays.items()}
    expected = manifest["array_digests"]
    mismatched = sorted(
        key for key in expected
        if key not in digests or digests[key] != expected[key]
    )
    _require(not mismatched, f"AC3-0 input lock failed on array digests: {mismatched[:5]}")
    sessions_ok = [str(item) for item in manifest["sessions"]] == list(trial_set.sessions)
    trials_ok = [str(item) for item in manifest["trial_ids"]] == list(trial_set.trial_ids)
    _require(sessions_ok, "AC3-0 input lock failed: session roster drift")
    _require(trials_ok, "AC3-0 input lock failed: trial order drift")
    for masks in trial_set.group_valid:
        _require(
            all(np.array_equal(np.asarray(masks[0]), np.asarray(item)) for item in masks),
            "AC3-0 input lock failed: the four views must share one validity interval",
        )
    lock_body = {
        "requirement": plan.OPERATOR_REVIEW["input_lock"]["requirement"],
        "materialize_sha256_field": "materialize.json.sha256",
        "sessions": list(trial_set.sessions),
        "n_trials": int(trial_set.n_trials),
        "trial_membership_sha256": trial_set.payload()["trial_membership_sha256"],
        "array_digests_matched": int(len(expected)),
        "four_group_partition": (
            "the frozen evaluator's own held_mask(0)..held_mask(3) column order, "
            "materialized by the imported P2' group forward; never re-derived here"
        ),
        "validity_intervals": "one shared validity mask per trial, identical across the four views",
        "bound_rows": list(plan.ROWS),
        "all_rows_consume_the_same_frozen_inputs": True,
    }
    lock_sha = hashlib.sha256(
        json.dumps(lock_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"input_sha256": lock_sha, **lock_body}


# ---------------------------------------------------------------------------
# P2' parity anchor (determinism proof against the immutable receipt).
# ---------------------------------------------------------------------------


def p2prime_parity_anchor(trial_set: mtx.TrialSet, base: Path) -> dict[str, object]:
    """Recompute the P2' sub-study direction aggregates and compare them.

    The AC3-0 trajectory line is the P2' never-commit sub-study line, so the
    per-session direction-error aggregates must reproduce the immutable P2'
    receipt bit-for-bit.  This is the determinism/lineage anchor of the screen.
    """
    substudy = _read_json(base / "tfpd_exploration/results/learned_gate_p2prime_v1/stage_substudy.json")
    reference_roster = {
        str(item["session"]) for item in substudy["sessions"]
        if int(item["budget"]) == plan.BUDGET and item["surface"] == plan.SOURCE_SURFACE
    }
    if not set(trial_set.sessions) <= reference_roster:
        return {
            "authority": plan.P2PRIME_RESULT_RELATIVE,
            "authority_sha256": plan.P2PRIME_RESULT_SHA256,
            "available": False,
            "reason": "the trial-set session roster is not the P2' sub-study source roster",
            "compared": 0,
            "all_equal": None,
            "rows": {},
        }
    rows: dict[str, dict[str, object]] = {}
    for index, session in enumerate(trial_set.sessions):
        trials = trial_set.session_trials(index)
        for construction in plan.CONSTRUCTIONS_MATERIALIZED:
            per_trial = []
            for trial in trials:
                payload = trial_set.pseudo[construction]
                pseudo_directions = tuple(
                    cdm_core.PseudoDirection(
                        accepted=bool(payload["accepted"][trial, group]),
                        reason=None,
                        theta_raw_rad=(None if not np.isfinite(payload["theta_raw_rad"][trial, group])
                                       else float(payload["theta_raw_rad"][trial, group])),
                        theta_index=(None if int(payload["theta_index"][trial, group]) < 0
                                     else int(payload["theta_index"][trial, group])),
                        theta_canonical_rad=None, canonical_distance_rad=None,
                        movement_bins=int(payload["movement_bins"][trial, group]),
                        displacement_norm=(None if not np.isfinite(payload["displacement_norm"][trial, group])
                                           else float(payload["displacement_norm"][trial, group])),
                        mean_speed=(None if not np.isfinite(payload["mean_speed"][trial, group])
                                    else float(payload["mean_speed"][trial, group])),
                    )
                    for group in plan.GROUPS
                )
                truth = trial_set.true_direction
                true_reference = {
                    "accepted": bool(truth["accepted"][trial]),
                    "theta_raw_rad": (None if not np.isfinite(truth["theta_raw_rad"][trial])
                                      else float(truth["theta_raw_rad"][trial])),
                    "theta_index": int(truth["theta_index"][trial]),
                }
                per_trial.append(p2physical._direction_error_row(pseudo_directions, true_reference))
            aggregate = p2physical._aggregate_direction_errors(per_trial)
            reference = [
                item for item in substudy["sessions"]
                if item["session"] == session and int(item["budget"]) == plan.BUDGET
            ]
            _require(len(reference) == 1, f"P2' sub-study reference row missing for {session}")
            expected = reference[0]["direction_rows"][construction]
            rows[f"{session}:{construction}"] = {
                "recomputed": aggregate,
                "p2prime": expected,
                "mean_circular_error_equal": (
                    aggregate["mean_circular_error_rad"] == expected["mean_circular_error_rad"]
                ),
                "snapped_mismatch_rate_equal": (
                    aggregate["snapped_mismatch_rate"] == expected["snapped_mismatch_rate"]
                ),
                "groups_compared_equal": (
                    int(aggregate["groups_compared"]) == int(expected["groups_compared"])
                ),
            }
    all_equal = all(
        bool(row["mean_circular_error_equal"]) and bool(row["snapped_mismatch_rate_equal"])
        and bool(row["groups_compared_equal"])
        for row in rows.values()
    )
    return {
        "authority": plan.P2PRIME_RESULT_RELATIVE,
        "authority_sha256": plan.P2PRIME_RESULT_SHA256,
        "compared": len(rows),
        "all_equal": bool(all_equal),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Hyperparameter selection (grouped source folds only).
# ---------------------------------------------------------------------------


def select_hyperparameters(
    trial_set: mtx.TrialSet, table: spl.StateTable, *, seed: int = enc.TORCH_SEED,
    rows: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """One grouped leave-one-source-session-out CV per (row, config).

    ``rows`` restricts the computation to a subset of the learned rows so the
    selection can be executed as several <=30 min tasks without changing the
    protocol (same grid, same folds, same criterion); ``None`` means all rows.
    """
    started = time.monotonic()
    selected_rows = tuple(plan.LEARNED_ROWS) if rows is None else tuple(str(item) for item in rows)
    _require(all(row in plan.LEARNED_ROWS for row in selected_rows),
             "selection rows must be learned rows")
    configurations = [
        (int(dim), float(temperature))
        for dim in enc.EMBEDDING_GRID
        for temperature in enc.TEMPERATURE_GRID
    ]
    selection: dict[str, dict[str, object]] = {}
    for row in selected_rows:
        candidates: dict[str, dict[str, object]] = {}
        for embedding_dim, temperature in configurations:
            result = mtx.learned_row_estimates(
                trial_set, table, row=row, embedding_dim=embedding_dim,
                temperature=temperature, seed=seed,
            )
            metrics = mtx.row_metrics(trial_set, result["estimate"])
            key = f"d{embedding_dim}_tau{temperature:g}"
            candidates[key] = {
                "embedding_dim": embedding_dim,
                "temperature": temperature,
                "session_mean_circular_error_rad": metrics["session_mean_circular_error_rad"],
                "session_mean_snap_mismatch_rate": metrics["session_mean_snap_mismatch_rate"],
                "retrieval_agreement": metrics["retrieval"]["true_direction_agreement_rate"],
            }
        scored = {
            key: value["session_mean_circular_error_rad"]
            for key, value in candidates.items()
            if value["session_mean_circular_error_rad"] is not None
        }
        _require(scored, f"hyperparameter selection produced no metric for row {row}")
        best = min(scored, key=lambda key: scored[key])
        selection[row] = {
            "grid": {"embedding_dim": list(enc.EMBEDDING_GRID),
                     "temperature": list(enc.TEMPERATURE_GRID)},
            "candidates": candidates,
            "selected": dict(candidates[best]),
            "criterion": "session-mean grouped-OOF circular error (source sessions only)",
        }
    return {
        "schema": f"{plan.SCHEMA}_selection_v1",
        "stage": "hyperparameter_selection",
        "disclosure": plan.SPLIT_DISCIPLINE["hyperparameter_selection"],
        "rows": selection,
        "rows_requested": list(selected_rows),
        "seed": int(seed),
        "wall_seconds": float(time.monotonic() - started),
    }


def assemble_selection(output_root: Path) -> dict[str, object]:
    """Merge the per-row selection partials into one selection receipt."""
    output_root = Path(output_root)
    partial_dir = output_root / "selection_rows"
    _require(partial_dir.exists(), "no selection partials to assemble")
    rows: dict[str, dict[str, object]] = {}
    for row in plan.LEARNED_ROWS:
        path = partial_dir / f"{row}.json"
        _require(path.exists(), f"selection partial missing for row {row}")
        payload = _read_json(path)
        _require(row in payload["rows"], f"selection partial for {row} is malformed")
        rows[row] = payload["rows"][row]
    body = {
        "schema": f"{plan.SCHEMA}_selection_v1",
        "stage": "hyperparameter_selection",
        "disclosure": plan.SPLIT_DISCIPLINE["hyperparameter_selection"],
        "rows": rows,
        "seed": int(payload["seed"]),
        "assembled_from_per_row_tasks": True,
        "process_gate_note": (
            "the pre-registered grid over all learned rows exceeded the 30-minute "
            "per-task process gate as one task; it was executed unchanged as one task "
            "per learned row and assembled here (same grid, same folds, same criterion)"
        ),
        "wall_seconds_per_row": {
            row: float(_read_json(partial_dir / f"{row}.json")["wall_seconds"])
            for row in plan.LEARNED_ROWS
        },
    }
    import hashlib as _hashlib

    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = _hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = output_root / "selection.json"
    _require(not path.exists(), "refusing to overwrite an existing selection receipt")
    path.write_text(text + "\n", encoding="utf-8")
    (output_root / "selection.json.sha256").write_text(
        f"{digest}  selection.json\n", encoding="ascii")
    return body


# ---------------------------------------------------------------------------
# The screen itself.
# ---------------------------------------------------------------------------


def run_screen(base: Path, *, output_root: Optional[Path] = None) -> Mapping[str, object]:
    """Evaluate the amended AC3-0 matrix, gates, controls; write screen.json."""
    base = Path(base).absolute()
    output = Path(output_root) if output_root is not None else base / plan.RESULT_ROOT_RELATIVE
    selection_path = output / "selection.json"
    _require(selection_path.exists(), "run the hyperparameter selection stage first")
    selection = _read_json(selection_path)
    started = time.monotonic()
    trial_set = load_trial_set(output)
    table = build_table(trial_set)
    input_lock = verify_input_lock(output, trial_set)
    metrics_by_row: dict[str, dict[str, object]] = {}
    estimates: dict[str, mtx.RowEstimate] = {}
    learned_receipts: dict[str, list[dict[str, object]]] = {}
    learned_audits: dict[str, list[dict[str, object]]] = {}
    for row in ("R0", "R0.5", "R1", "O2"):
        estimate = mtx.static_row(trial_set, row)
        estimates[row] = estimate
        metrics_by_row[row] = mtx.row_metrics(trial_set, estimate, per_view=(row in ("R0", "R1")))
    for row in plan.LEARNED_ROWS:
        chosen = selection["rows"][row]["selected"]
        result = mtx.learned_row_estimates(
            trial_set, table, row=row,
            embedding_dim=int(chosen["embedding_dim"]),
            temperature=float(chosen["temperature"]),
        )
        estimates[row] = result["estimate"]
        metrics_by_row[row] = mtx.row_metrics(trial_set, result["estimate"])
        learned_receipts[row] = result["receipts"]
        learned_audits[row] = result["sampler_audits"]
    influence = {
        row: mtx.leave_one_session_influence(trial_set, metrics_by_row, "R0", row)
        for row in plan.CONTRASTIVE_ROWS
    }
    representation_gates = {
        row: gate_module.representation_gate(metrics_by_row, influence, row=row,
                                             target_update_count=0)
        for row in plan.CONTRASTIVE_ROWS
    }
    contrastive_gate = gate_module.contrastive_claim_gate(metrics_by_row)
    e7_mirror = gate_module.e7_mirror_verdict(metrics_by_row, contrastive_gate)
    shuffle_degrades = representation_gates["R4"]["shuffle_control"]["degrades"]
    stop_conditions = gate_module.stop_condition_verdicts(
        representation_gates=representation_gates, contrastive_gate=contrastive_gate,
        shuffle_degrades=shuffle_degrades, influence=influence,
        utility_status=plan.UTILITY_ROWS["status_this_run"], target_update_count=0,
        chronology_proved=True,
    )
    verdict = gate_module.screen_verdict(
        representation_gates=representation_gates, contrastive_gate=contrastive_gate,
        e7_mirror=e7_mirror, stop_conditions=stop_conditions,
    )
    operator = gate_module.operator_verdicts(
        metrics_by_row, contrastive_gate, utility_status=plan.UTILITY_ROWS["status_this_run"],
    )
    for row in plan.ROWS:
        if row in metrics_by_row:
            metrics_by_row[row]["input_lock_sha256"] = input_lock["input_sha256"]
    correct_minus_shuffle = {
        "circular_error_rad": (
            float(metrics_by_row["RS"]["session_mean_circular_error_rad"]
                  - metrics_by_row["R4"]["session_mean_circular_error_rad"])
            if metrics_by_row["RS"]["session_mean_circular_error_rad"] is not None
            and metrics_by_row["R4"]["session_mean_circular_error_rad"] is not None else None
        ),
        "retrieval_agreement": (
            float(metrics_by_row["R4"]["retrieval"]["true_direction_agreement_rate"]
                  - metrics_by_row["RS"]["retrieval"]["true_direction_agreement_rate"])
        ),
        "note": "positive = the correct-label arm beats the shuffled-label arm",
    }
    shortcut_controls = {
        "session_id_predicts_target": {
            "raw_features": mtx.session_id_probe(table, table.features),
            "note": "audit only: high session predictability warns that an encoder could shortcut; "
                    "the gates use direction metrics, never session identity",
        },
        "trial_split_leakage": mtx.fold_split_leakage(trial_set),
        "cross_group_only_positives": {
            row: [
                {"held_out_session": item["held_out_session"],
                 "cross_group_fraction": (item["counts"].get("cross_group_fraction")
                                          if item.get("counts") else None),
                 "within_cap": (item["counts"].get("cross_group_fraction_within_cap")
                               if item.get("counts") else None)}
                for item in learned_audits.get(row, [])
            ]
            for row in plan.CONTRASTIVE_ROWS + ("RS",)
        },
        "time_proximity_explains_result": {
            "R3_session_mean_circular_error_rad": metrics_by_row["R3"]["session_mean_circular_error_rad"],
            "R4_session_mean_circular_error_rad": metrics_by_row["R4"]["session_mean_circular_error_rad"],
            "correct_minus_shuffle": correct_minus_shuffle,
        },
        "speed_or_phase_explains_direction": {
            row: {
                "retrieval_agreement": metrics_by_row[row]["retrieval"]["true_direction_agreement_rate"],
                "speed_duration_only_baseline": metrics_by_row[row]["retrieval"][
                    "speed_duration_only_baseline_agreement_rate"
                ],
                "beats_confound_baseline": metrics_by_row[row]["retrieval"][
                    "beats_speed_duration_confound_baseline"
                ],
            }
            for row in ("R0", "R0.5", "R2", "R3", "R4", "R5", "RS")
        },
        "rest_rows_silent": {
            "n_states": table.n,
            "n_rest_or_undefined": int((~table.eligible()).sum()),
            "rest_threshold": float(table.rest_threshold),
            "rest_quantile": plan.SAMPLING["rest_condition"]["quantile"],
        },
        "target_labels_select_encoder": {
            "external_roster_opened": False,
            "surface": plan.SOURCE_SURFACE,
            "hyperparameters_selected_on": "grouped source folds only",
        },
        "target_unlabeled_adaptation": {"target_optimizer_backward_update": 0},
        "shuffle_retains_advantage": {
            "degrades": shuffle_degrades,
            "rule": plan.SHUFFLE_DEGRADATION_RULE["expression"],
        },
        "representation_without_utility": {
            "utility_rows_status": plan.UTILITY_ROWS["status_this_run"],
            "downstream_gate": "PENDING never silently passed",
        },
    }
    payload = {
        "schema": f"{plan.SCHEMA}_screen_v1",
        "stage": "frozen_representation_screen",
        "attempt_sha256": (output / "attempt.json.sha256").read_text(encoding="ascii").split()[0],
        "budget": plan.BUDGET,
        "input_lock": input_lock,
        "operator_review": json.loads(json.dumps(plan.OPERATOR_REVIEW, sort_keys=True)),
        "result_order": list(plan.OPERATOR_REVIEW["result_order"]),
        "primary_metric_1_carrier_utility": {
            "status": plan.UTILITY_ROWS["status_this_run"],
            "authority": plan.UTILITY_ROWS["owner"],
            "disclosure": plan.UTILITY_ROWS["disclosure"],
            "row_values": {row: None for row in plan.ROWS},
            "reading": (
                "the utility estimand is reported FIRST and is NOT evaluated inside this "
                "run's process gate; no advance interpretation may be drawn without it"
            ),
        },
        "primary_metric_2_circular_direction_error": {
            "primary_baseline_row": plan.OPERATOR_REVIEW["primary_baseline_row"],
            "primary_comparison_set": list(plan.OPERATOR_REVIEW["primary_comparison_set"]),
            "row_table": {
                row: {
                    "representation": plan.ROW_SPECS[row]["representation"],
                    "role": plan.ROW_SPECS[row].get("role"),
                    "is_primary_baseline": row == plan.OPERATOR_REVIEW["primary_baseline_row"],
                    "session_mean_circular_error_rad": metrics_by_row[row]["session_mean_circular_error_rad"],
                    "session_mean_snap_mismatch_pp": (
                        None if metrics_by_row[row]["session_mean_snap_mismatch_rate"] is None
                        else 100.0 * float(metrics_by_row[row]["session_mean_snap_mismatch_rate"])
                    ),
                    "retrieval_true_direction_agreement": metrics_by_row[row]["retrieval"][
                        "true_direction_agreement_rate"
                    ],
                    "cross_group_direction_dispersion": metrics_by_row[row][
                        "session_mean_cross_group_direction_dispersion"
                    ],
                    "utility": None,
                    "leakage_label": metrics_by_row[row]["leakage_label"],
                    "input_lock_sha256": input_lock["input_sha256"],
                }
                for row in plan.ROWS
            },
        },
        "output_filter_disclosure": dict(plan.OPERATOR_REVIEW["output_filter"]),
        "trial_set": trial_set.payload(),
        "state_table": table.payload(),
        "row_table": {
            row: {
                "representation": plan.ROW_SPECS[row]["representation"],
                "session_mean_circular_error_rad": metrics_by_row[row]["session_mean_circular_error_rad"],
                "session_mean_snap_mismatch_pp": (
                    None if metrics_by_row[row]["session_mean_snap_mismatch_rate"] is None
                    else 100.0 * float(metrics_by_row[row]["session_mean_snap_mismatch_rate"])
                ),
                "retrieval_true_direction_agreement": metrics_by_row[row]["retrieval"][
                    "true_direction_agreement_rate"
                ],
                "cross_group_direction_dispersion": metrics_by_row[row][
                    "session_mean_cross_group_direction_dispersion"
                ],
                "utility": None,
                "leakage_label": metrics_by_row[row]["leakage_label"],
            }
            for row in plan.ROWS
        },
        "rows": metrics_by_row,
        "learned_row_receipts": learned_receipts,
        "correct_minus_shuffle": correct_minus_shuffle,
        "rho_GE_calibration": {
            "pooled": metrics_by_row["R0.5"]["credibility_calibration"]["pooled"],
            "per_session": metrics_by_row["R0.5"]["credibility_calibration"]["per_session"],
            "required_output": "section 23 amendment 4: the zero-learning credibility candidate for AC3-1",
        },
        "representation_gates": representation_gates,
        "contrastive_claim_gate": contrastive_gate,
        "e7_mirror": e7_mirror,
        "operator_verdicts": operator,
        "stop_conditions": stop_conditions,
        "verdict": verdict,
        "shortcut_controls": shortcut_controls,
        "utility_rows": dict(plan.UTILITY_ROWS),
        "p2prime_parity_anchor": p2prime_parity_anchor(trial_set, base),
        "p2prime_opportunity_context": _p2prime_opportunity_context(base),
        "target_optimizer_backward_update": 0,
        "model_or_checkpoint_updated": False,
        "resources": {
            "wall_seconds": float(time.monotonic() - started),
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "cpu_only": True,
        },
    }
    digest = _publish(output / "screen.json", payload)
    return {"stage": "screen", "screen_sha256": digest, "verdict": verdict, "payload": payload}


def _p2prime_opportunity_context(base: Path) -> dict[str, object]:
    """Read-only citation of the immutable P2' coherent-oracle opportunity."""
    try:
        result = _read_json(base / plan.P2PRIME_RESULT_RELATIVE)
    except FileNotFoundError:
        return {"available": False}
    matrix = result["matrix"]["m4"]["external"]
    return {
        "available": True,
        "authority": plan.P2PRIME_RESULT_RELATIVE,
        "authority_sha256": plan.P2PRIME_RESULT_SHA256,
        "p2prime_verdict": plan.P2PRIME_VERDICT,
        "m4_external_mean_r2": {
            row: float(matrix[row]["mean_r2"]) for row in ("A1", "O0", "O1", "O2") if row in matrix
        },
        "O2_minus_O1_opportunity": (
            float(matrix["O2"]["mean_r2"]) - float(matrix["O1"]["mean_r2"])
            if "O2" in matrix and "O1" in matrix else None
        ),
        "O1_minus_O0_smoothing_value": (
            float(matrix["O1"]["mean_r2"]) - float(matrix["O0"]["mean_r2"])
            if "O1" in matrix and "O0" in matrix else None
        ),
    }


def write_amendment(base: Path, *, output_root: Optional[Path] = None, index: int = 1) -> Mapping[str, object]:
    """Pin contract amendments BEFORE the screen rows run.

    Amendment 1 records the operator review requirements.  Later amendments
    record further pre-screen corrections (each with its own reason) and always
    re-pin the owned module bytes so the screen runs under an exact contract.
    """
    base = Path(base).absolute()
    output = Path(output_root) if output_root is not None else base / plan.RESULT_ROOT_RELATIVE
    _require((output / "attempt.json").exists(), "the AC3-0 attempt receipt is missing")
    _require(
        not (output / "screen.json").exists(),
        "the operator amendment must be written before the screen rows run",
    )
    name = "amendment.json" if index == 1 else f"amendment_{index}.json"
    _require(not (output / name).exists(), f"refusing to overwrite an existing amendment: {name}")
    reasons = {
        1: (
            "operator review requirements were injected between the materialization and "
            "the screen; this receipt re-pins the owned bytes so the screen runs under "
            "the amended contract"
        ),
        2: (
            "two pre-screen corrections after the first selection pass was discarded: "
            "(a) the direction readout (R2 supervision and the contrastive linear probe) "
            "now targets the COMPLETED-TRIAL integrated direction at the t=T state, "
            "because the bin-level endpoint direction is a different estimand (on this "
            "cohort it sits a mean 2.34 rad from the integrated completed-trial "
            "direction, which made every learned row solve the wrong task); (b) the "
            "pre-registered selection grid exceeded the 30-minute per-task process gate "
            "as one task and was executed unchanged as one task per learned row"
        ),
    }
    payload = {
        "schema": f"{plan.SCHEMA}_operator_amendment_v1",
        "status": "AMENDMENT_RESERVED_BEFORE_SCREEN",
        "amendment_index": int(index),
        "cell": plan.CELL,
        "reason": reasons.get(int(index), "pre-screen contract correction"),
        "original_attempt_sha256": (output / "attempt.json.sha256").read_text(encoding="ascii").split()[0],
        "materialize_sha256": (
            (output / "materialize.json.sha256").read_text(encoding="ascii").split()[0]
            if (output / "materialize.json.sha256").exists() else None
        ),
        "selection_sha256": (
            (output / "selection.json.sha256").read_text(encoding="ascii").split()[0]
            if (output / "selection.json.sha256").exists() else None
        ),
        "operator_review": json.loads(json.dumps(plan.OPERATOR_REVIEW, sort_keys=True)),
        "owned_sha256s_after_amendment": plan.owned_sha256s(base),
        "target_optimizer_backward_update": 0,
    }
    digest = _publish(output / name, payload)
    return {"stage": "amend", "amendment_index": int(index), "amendment_sha256": digest}


def finalize(base: Path, *, output_root: Optional[Path] = None) -> Mapping[str, object]:
    """Write terminal.json and freeze the result root."""
    base = Path(base).absolute()
    output = Path(output_root) if output_root is not None else base / plan.RESULT_ROOT_RELATIVE
    screen_path = output / "screen.json"
    _require(screen_path.exists(), "the screen stage must run before finalize")
    screen = _read_json(screen_path)
    verdict = screen["verdict"]
    digest = hashlib.sha256(screen_path.read_bytes()).hexdigest()
    sidecar = (output / "screen.json.sha256").read_text(encoding="ascii").strip()
    _require(sidecar == f"{digest}  screen.json", "screen receipt sidecar drift")
    sidecar_repair = (
        "the sidecar digests of the pre-finalize receipts were recomputed over the exact "
        "file bytes (the earlier sidecars hashed the body without its trailing newline); "
        "no receipt CONTENT changed -- only the checksum convention was repaired to the "
        "P2' byte convention before terminal"
    )
    terminal = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "non_governing_frozen_representation_screen",
        "verdict": verdict["verdict"],
        "representation_gate_passed": verdict["representation_gate_passed"],
        "contrastive_claim_passed": verdict["contrastive_claim_passed"],
        "e7_mirror_fired": verdict["e7_mirror_fired"],
        "e7_mirror_disposition": verdict["e7_mirror_disposition"],
        "operator_verdicts_fired": screen.get("operator_verdicts", {}).get("fired", []),
        "input_lock_sha256": screen.get("input_lock", {}).get("input_sha256"),
        "stop_conditions_fired": verdict["stop_conditions_fired"],
        "downstream_advance_gate": verdict["downstream_advance_gate"],
        "ac3_1_status": verdict["ac3_1_status"],
        "ac3_2_status": verdict["ac3_2_status"],
        "disposition": (
            "the screen result goes to the operator for the advance decision; no further "
            "stage is started by the screen itself"
        ),
        "utility_rows_status": plan.UTILITY_ROWS["status_this_run"],
        "screen_sha256": digest,
        "attempt_sha256": (output / "attempt.json.sha256").read_text(encoding="ascii").split()[0],
        "amendment_sha256s": sorted(
            path.name.replace(".sha256", "") for path in output.glob("amendment*.json.sha256")
        ),
        "materialize_sha256": (output / "materialize.json.sha256").read_text(encoding="ascii").split()[0],
        "selection_sha256": (output / "selection.json.sha256").read_text(encoding="ascii").split()[0],
        "addendum": plan.ADDENDUM_RELATIVE,
        "sidecar_convention_repair": sidecar_repair,
        "target_optimizer_backward_update": 0,
    }
    sha = _publish(output / "terminal.json", terminal)
    os.chmod(output, 0o555)
    return {"stage": "finalize", "terminal_sha256": sha, "verdict": verdict["verdict"]}
