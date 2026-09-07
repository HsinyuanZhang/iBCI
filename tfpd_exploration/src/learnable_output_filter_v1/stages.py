"""Stage drivers P1-P4 of the frozen-output route (§9).

Every stage reads only cached raw streams, sealed receipts and earlier stage
receipts of this route.  No stage performs a decoder forward, trains anything,
or reads external labels for selection.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import audit, ladder, metrics, oracle, plan, receipts, selection, streams


class StageError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageError(message)


def _cache(root: Path) -> Path:
    return Path(root) / plan.CACHE_ROOT_RELATIVE


def _results(root: Path) -> Path:
    return Path(root) / "results/learnable_output_filter_v1"


def _read_stage(root: Path, name: str) -> dict[str, Any]:
    return receipts.read_receipt(_results(root) / name)


def _publish_stage(root: Path, name: str, payload: dict[str, Any]) -> str:
    return receipts.publish(_results(root) / name, payload)


# ---------------------------------------------------------------------------
# P1 — the four-cell fixed-filter factorial.
# ---------------------------------------------------------------------------


def _spec_from_payload(payload: dict[str, Any]) -> ladder.FilterSpec:
    level = payload["level"]
    if level in ("F0", "F1"):
        return ladder.FilterSpec(level=level)
    if level == "F2":
        return ladder.f2(float(payload["alpha"]))
    if level == "F3":
        return ladder.f3(payload["weights"])
    if level == "F4":
        return ladder.f4(payload["theta"], float(payload["bias"]))
    raise StageError(f"unknown filter payload level {level}")


def run_p1(root: Path, *, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = _cache(root)
    stage_a = streams._verify_sealed(plan.SEALED_STAGE_A_REL, plan.SEALED_STAGE_A_SHA256)
    probe = streams._verify_sealed(plan.SEALED_PROBE_REL, plan.SEALED_PROBE_SHA256)
    fixed = ladder.f2(float(plan.P1_FIXED_FILTER["alpha"]))
    static = streams.load_all(cache_root, "static")
    cdm = streams.load_all(cache_root, "cdm")
    probe_arm_deltas = {
        (entry["surface"], int(entry["budget"])): entry
        for entry in probe["paired_deltas"] if entry["arm"] == "causal_window_exp_a0.25"
    }
    smoke_note = (
        "SMOKE subset run — never governing" if smoke else "governing full roster"
    )
    matrix: dict[str, Any] = {}
    readings: dict[str, Any] = {}
    per_session_rows: list[dict[str, Any]] = []
    budgets = sorted({key[2] for key in static})
    _require(bool(budgets), "P1 has no cached static budgets")
    if not smoke:
        _require(tuple(budgets) == tuple(plan.BUDGETS), "P1 governing run needs all three budgets cached")
    for budget in budgets:
        matrix[f"m{budget}"] = {}
        readings[f"m{budget}"] = {}
        for surface in plan.SURFACES:
            roster = streams.cached_roster(cache_root, "static")[surface]
            stage_a_rows = {
                str(row["session"]): row
                for row in stage_a["matrix"][f"m{budget}"][surface]["A0"]["sessions"]
            }
            stage_a_rows_filtered = {
                str(row["session"]): row
                for row in stage_a["matrix"][f"m{budget}"][surface]["A1"]["sessions"]
            }
            _require(
                set(roster) == set(stage_a_rows) if not smoke else set(roster) <= set(stage_a_rows),
                f"P1 roster mismatch static vs sealed stage A at M{budget} {surface}",
            )
            ordered = [name for name in stage_a_rows if name in set(roster)]  # sealed order
            _require(bool(ordered), f"P1 empty paired roster at M{budget} {surface}")
            a0_values, a1_values, b0_values, b1_values = [], [], [], []
            for session in ordered:
                stream = static[(surface, session, budget)]
                raw_result = ladder.apply_filter(stream, ladder.F0)
                filtered_result = ladder.apply_filter(stream, fixed)
                a0 = metrics.matrix_r2(raw_result.blocks, stream)
                a1 = metrics.matrix_r2(filtered_result.blocks, stream)
                b0 = float(stage_a_rows[session]["matrix_r2"])
                b1 = float(stage_a_rows_filtered[session]["matrix_r2"])
                a0_values.append(a0)
                a1_values.append(a1)
                b0_values.append(b0)
                b1_values.append(b1)
                per_session_rows.append({
                    "budget": budget, "surface": surface, "session": session,
                    "A0_static_matrix_r2": a0,
                    "A1_static_matrix_r2": a1,
                    "B0_cdm_matrix_r2_sealed": b0,
                    "B1_cdm_matrix_r2_sealed": b1,
                    "static_raw_sha256": metrics.prediction_sha256(raw_result.blocks, stream),
                    "static_filtered_sha256": metrics.prediction_sha256(filtered_result.blocks, stream),
                    "cdm_raw_sha256_sealed": stage_a_rows[session]["prediction_sha256_raw"],
                    "cdm_filtered_sha256_sealed": stage_a_rows_filtered[session]["filtered_prediction_sha256"],
                    "reset_event_digest": stream.chronology_proof()["reset_event_digest"],
                    "filter_payload_sha256": fixed.payload_sha256(),
                    "n_state_transitions": filtered_result.n_state_transitions,
                })
            delta_static = metrics.paired_session_deltas(
                a1_values, a0_values, label="A1_minus_A0_static_trial_local_ema",
            )
            delta_cdm = metrics.paired_session_deltas(
                b1_values, b0_values, label="B1_minus_B0_sealed_stage_a",
            )
            interaction = metrics.paired_interaction(
                b1_values, b0_values, a1_values, a0_values,
                label="activity_filter_interaction",
            )
            probe_entry = probe_arm_deltas.get((surface, budget))
            readings[f"m{budget}"][surface] = {
                "A1_minus_A0_static": delta_static,
                "B1_minus_B0_cdm_sealed": delta_cdm,
                "interaction_B1B0_minus_A1A0": interaction,
                "continuity_probe_concatenated_stream_reference": {
                    "arm": "causal_window_exp_a0.25",
                    "mean_delta": float(probe_entry["mean"]) if probe_entry else None,
                    "n_positive": probe_entry["n_positive"] if probe_entry else None,
                    "note": (
                        "the sealed probe smoothed the static stream WITHOUT trial "
                        "resets (documented defect); the P1 static cells above use "
                        "TRIAL_RESET, and this row quantifies the correction"
                    ),
                } if probe_entry else None,
            }
            matrix[f"m{budget}"][surface] = {
                "A0": {"mean_r2": metrics.equal_session_mean(a0_values)},
                "A1": {"mean_r2": metrics.equal_session_mean(a1_values)},
                "B0": {"mean_r2": metrics.equal_session_mean(b0_values),
                       "source": "sealed learned_gate_p2prime_v1/stage_a.json"},
                "B1": {"mean_r2": metrics.equal_session_mean(b1_values),
                       "source": "sealed learned_gate_p2prime_v1/stage_a.json"},
            }
    payload = {
        "schema": "learnable_output_filter_v1_p1_factorial_v1",
        "stage": "p1_fixed_filter_factorial",
        "scope": smoke_note,
        "design_authority": plan.DESIGN_RELATIVE,
        "fixed_filter": plan.P1_FIXED_FILTER,
        "fixed_filter_payload_sha256": fixed.payload_sha256(),
        "cells": matrix,
        "readings": readings,
        "sessions": per_session_rows,
        "primary_readings_summary": {
            f"m{budget}_{surface}_{key}": readings[f"m{budget}"][surface][key]["mean"]
            for budget in budgets for surface in plan.SURFACES
            for key in ("A1_minus_A0_static", "B1_minus_B0_cdm_sealed",
                        "interaction_B1B0_minus_A1A0")
        },
        "interaction_definition": "(B1 - B0) - (A1 - A0); B cells are the sealed stage-A rows",
        "forwards_performed_by_this_stage": 0,
    }
    payload.update(receipts.boundary_rows(wall_seconds=time.perf_counter() - started))
    _publish_stage(root, "p1_factorial.json", payload)
    return payload


# ---------------------------------------------------------------------------
# P2 — source-learned fixed filter (F2 alpha CV, F3 simplex FIR, §10.2 gate).
# ---------------------------------------------------------------------------


def run_p2(root: Path, *, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = _cache(root)
    cdm = streams.load_all(cache_root, "cdm")
    within_roster = streams.cached_roster(cache_root, "cdm")["within"]
    budgets = sorted({key[2] for key in cdm})
    _require(bool(budgets), "P2 has no cached CDM budgets")
    if not smoke:
        _require(tuple(budgets) == tuple(plan.BUDGETS), "P2 governing run needs all three budgets cached")
    payload: dict[str, Any] = {
        "schema": "learnable_output_filter_v1_p2_fixed_filter_v1",
        "stage": "p2_source_learned_fixed_filter",
        "scope": "SMOKE subset run — never governing" if smoke else "governing",
        "design_authority": plan.DESIGN_RELATIVE,
        "fitting_surface": "cdm_activity_only",
        "source_sessions": list(within_roster),
        "source_disclosure": (
            "filter parameters are fit only on the within-6 sub-C sessions (the "
            "held-out sessions of the decoder's training subject); the external-15 "
            "surface is never read by any fitting or selection function"
        ),
        "budgets": {},
        "forwards_performed_by_this_stage": 0,
    }
    for budget in budgets:
        source_streams = [cdm[("within", session, budget)] for session in within_roster]
        f1_r2 = [
            metrics.matrix_r2(ladder.apply_filter(stream, ladder.F1).blocks, stream)
            for stream in source_streams
        ]
        f2_selection = selection.select_alpha(source_streams)
        fir = selection.cv_fir(source_streams)
        gate = selection.select_fixed_filter(
            f1_oof=metrics.equal_session_mean(f1_r2),
            f2_oof=float(f2_selection["oof_equal_session_mean_r2"]),
            fir_oof=float(fir["oof_equal_session_mean_r2"]),
            fir_payload=fir["payload"], best_f2_payload=f2_selection["payload"],
        )
        p1_filter_oof = metrics.equal_session_mean([
            metrics.matrix_r2(ladder.apply_filter(stream, ladder.f2(0.25)).blocks, stream)
            for stream in source_streams
        ])
        payload["budgets"][f"m{budget}"] = {
            "F1_k2": {
                "per_session_r2": f1_r2,
                "oof_equal_session_mean_r2": metrics.equal_session_mean(f1_r2),
            },
            "F2_alpha_grid_cv": f2_selection,
            "F3_fir_k4_cv": fir,
            "selection_gate": gate,
            "p1_filter_ema_a0p25_source_mean_r2": p1_filter_oof,
            "session_counts": {
                "n_source_sessions": len(source_streams),
                "n_windows_total": int(sum(
                    int(np.asarray(block.valid).sum()) for stream in source_streams for block in stream.blocks
                )),
                "n_trials_total": int(sum(len(stream.blocks) for stream in source_streams)),
            },
            "selected_filter": gate["selected_payload"],
        }
    dispositions = {
        f"m{budget}": payload["budgets"][f"m{budget}"]["selection_gate"]["selected_level"]
        for budget in budgets
    }
    payload["selected_by_budget"] = dispositions
    payload.update(receipts.boundary_rows(wall_seconds=time.perf_counter() - started))
    _publish_stage(root, "p2_fixed_filter.json", payload)
    return payload


def selected_fixed_spec(p2_payload: dict[str, Any], budget: int) -> ladder.FilterSpec:
    return _spec_from_payload(p2_payload["budgets"][f"m{int(budget)}"]["selected_filter"])


# ---------------------------------------------------------------------------
# P3 — adaptive oracle (§8) on the raw B0 activity-only streams.
# ---------------------------------------------------------------------------


def run_p3(root: Path, *, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = _cache(root)
    p2 = _read_stage(root, "p2_fixed_filter.json")
    cdm = streams.load_all(cache_root, "cdm")
    rosters = streams.cached_roster(cache_root, "cdm")
    budgets = sorted({key[2] for key in cdm})
    if not smoke:
        _require(tuple(budgets) == tuple(plan.BUDGETS), "P3 governing run needs all three budgets cached")
    payload: dict[str, Any] = {
        "schema": "learnable_output_filter_v1_p3_oracle_v1",
        "stage": "p3_adaptive_oracle",
        "scope": "SMOKE subset run — never governing" if smoke else "governing",
        "design_authority": plan.DESIGN_RELATIVE,
        "oracle_rows_are_leakage_diagnostics": True,
        "target_label_leakage": True,
        "gain_grid": list(plan.GAIN_GRID),
        "disposition_primary_surface": "external",
        "budgets": {},
        "forwards_performed_by_this_stage": 0,
    }
    allowed: dict[str, bool] = {}
    for budget in budgets:
        fixed = selected_fixed_spec(p2, budget)
        budget_rows: dict[str, Any] = {
            "fixed_filter": fixed.payload(),
            "fixed_filter_payload_sha256": fixed.payload_sha256(),
            "surfaces": {},
        }
        dispositions = {}
        for surface in plan.SURFACES:
            fixed_values, oracle_values, ceiling_values = [], [], []
            per_session = []
            for session in rosters[surface]:
                stream = cdm[(surface, session, budget)]
                fixed_result = ladder.apply_filter(stream, fixed)
                fixed_r2 = metrics.matrix_r2(fixed_result.blocks, stream)
                greedy = oracle.coherent_greedy_gain_oracle(stream)
                ceiling = oracle.noncoherent_switch_ceiling(stream, parent=fixed)
                greedy_r2 = metrics.matrix_r2(greedy["blocks"], stream)
                ceiling_r2 = metrics.matrix_r2(ceiling["blocks"], stream)
                fixed_values.append(fixed_r2)
                oracle_values.append(greedy_r2)
                ceiling_values.append(ceiling_r2)
                per_session.append({
                    "session": session,
                    "fixed_matrix_r2": fixed_r2,
                    "coherent_greedy_oracle_matrix_r2": greedy_r2,
                    "noncoherent_ceiling_matrix_r2": ceiling_r2,
                    "coherent_gain_histogram": oracle.gain_histogram(greedy["gains"], greedy["lengths"]),
                    "noncoherent_gain_histogram": oracle.gain_histogram(ceiling["gains"], ceiling["lengths"]),
                    "target_label_leakage": True,
                })
            budget_rows["surfaces"][surface] = {
                "sessions": per_session,
                "coherent_oracle_minus_fixed": metrics.paired_session_deltas(
                    oracle_values, fixed_values, label=f"coherent_oracle_minus_fixed_{surface}_m{budget}",
                ),
                "noncoherent_ceiling_minus_fixed": metrics.paired_session_deltas(
                    ceiling_values, fixed_values, label=f"noncoherent_ceiling_minus_fixed_{surface}_m{budget}",
                ),
                "noncoherent_ceiling_minus_coherent_oracle": metrics.paired_session_deltas(
                    ceiling_values, oracle_values,
                    label=f"noncoherent_minus_coherent_{surface}_m{budget}",
                ),
            }
            dispositions[surface] = selection.oracle_disposition(
                oracle_minus_fixed=budget_rows["surfaces"][surface]["coherent_oracle_minus_fixed"]["mean"],
                surface=surface, budget=budget,
            )
        budget_rows["dispositions"] = dispositions
        budget_rows["disposition"] = dispositions["external"]["disposition"]
        payload["budgets"][f"m{budget}"] = budget_rows
        allowed[f"m{budget}"] = budget_rows["disposition"] in ("CONDITIONAL_F4", "PROCEED_F4")
    payload["f4_allowed_by_budget"] = allowed
    payload["disposition_summary"] = {
        f"m{budget}": {
            "external": payload["budgets"][f"m{budget}"]["dispositions"]["external"]["disposition"],
            "external_value": payload["budgets"][f"m{budget}"]["dispositions"]["external"]["coherent_oracle_minus_best_fixed"],
            "within": payload["budgets"][f"m{budget}"]["dispositions"]["within"]["disposition"],
            "within_value": payload["budgets"][f"m{budget}"]["dispositions"]["within"]["coherent_oracle_minus_best_fixed"],
        } for budget in budgets
    }
    payload.update(receipts.boundary_rows(wall_seconds=time.perf_counter() - started))
    _publish_stage(root, "p3_oracle.json", payload)
    return payload


# ---------------------------------------------------------------------------
# P4 — source-fit F4 adaptive scalar gain (only where P3 allows).
# ---------------------------------------------------------------------------


def run_p4(root: Path, *, smoke: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = _cache(root)
    p2 = _read_stage(root, "p2_fixed_filter.json")
    p3 = _read_stage(root, "p3_oracle.json")
    cdm = streams.load_all(cache_root, "cdm")
    within_roster = streams.cached_roster(cache_root, "cdm")["within"]
    payload: dict[str, Any] = {
        "schema": "learnable_output_filter_v1_p4_adaptive_v1",
        "stage": "p4_source_fit_f4",
        "scope": "SMOKE subset run — never governing" if smoke else "governing",
        "design_authority": plan.DESIGN_RELATIVE,
        "feature_whitelist": list(plan.F4_FEATURES),
        "feature_scales": dict(plan.F4_FEATURE_SCALES),
        "param_count": plan.F4_PARAM_COUNT,
        "primary_readout": "source-grouped OOF realized R2 gain over the selected fixed filter (not AUC)",
        "fitting_surface": "cdm_activity_only",
        "source_sessions": list(within_roster),
        "budgets": {},
        "forwards_performed_by_this_stage": 0,
    }
    allowed = p3["f4_allowed_by_budget"]
    skipped = []
    for budget in sorted({key[2] for key in cdm}):
        if not allowed[f"m{budget}"]:
            skipped.append(f"m{budget}")
            continue
        fixed = selected_fixed_spec(p2, budget)
        source_streams = [cdm[("within", session, budget)] for session in within_roster]
        cv = selection.cv_f4(source_streams, reference=fixed)
        payload["budgets"][f"m{budget}"] = {
            "reference_filter": fixed.payload(),
            "reference_filter_payload_sha256": fixed.payload_sha256(),
            "cv": cv,
            "pooled_fit_param_sha256": hashlib.sha256(
                ladder.canonical_json(cv["payload"]).encode("utf-8")
            ).hexdigest(),
            "gain_bounds": ladder.f4_gain_bounds(),
        }
    payload["skipped_by_disposition"] = skipped
    payload.update(receipts.boundary_rows(wall_seconds=time.perf_counter() - started))
    _publish_stage(root, "p4_adaptive.json", payload)
    return payload


# ---------------------------------------------------------------------------
# P0 driver wrapper (adds the boundary rows and publishes).
# ---------------------------------------------------------------------------


def run_p0(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    body = audit.run_p0(root)
    body["wall_seconds"] = time.perf_counter() - started
    _publish_stage(root, "p0_audit.json", body)
    return body


def cache_receipts(root: Path, static_result: dict[str, Any] | None,
                   cdm_result: dict[str, Any] | None) -> None:
    """Publish the materialization receipts (attempt-before-data satisfied)."""
    if static_result is not None:
        receipts.publish(_results(root) / "static_stream_cache.json", static_result)
    if cdm_result is not None:
        receipts.publish(_results(root) / "cdm_stream_cache.json", cdm_result)
