"""SLOT-AUDIT V1 audit — pure statistics on the cache, then the terminal read.

Work order sections 4-5.  This stage performs ZERO decoder forwards: every
array comes from the digest-bound cache, every governing R2 goes through the
house float32 scorer (the same ``session_r2`` the sealed rows used), and the
trajectory-alignment arms go through the frozen probe law by import.

Produced tables:
  1. per-slot calibration (absolute-bin law): per (surface, budget) the 50
     equal-session mean per-slot rows + the per-session R2 matrices;
  2. redundant-estimate residual pairs by slot separation delta (1..49):
     pooled correlation / covariance / residual stds + mean pair counts;
  3. slot-subset ensembles of one window ({49}, {45..49}, {40..49}, all-50,
     and the per-budget within-source-selected best single slot) with paired
     per-session deltas vs {49} and best-slot-per-session stability;
  4. trajalign re-measurement K in {2,4,8,16} + the sealed reproduction
     equality check + the latency table (delay = K-1 bins at the loader's
     bin size, read from the frozen source and cited by file:line);
  5. digests per (surface, session, budget) and the pre-registered reading
     classification, decided mechanically, with the mandated caveats.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import alignment, cache_store, ensembles, plan, trajalign
from .materialize import load_sealed_probe, sealed_baseline_rows


class SlotAuditAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditAuditError(message)


def _read_receipt(path: Path) -> tuple[dict, str]:
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_name(path.name + ".sha256").read_text(encoding="ascii")
    _require(
        sidecar.strip() == f"{digest}  {path.name}",
        f"receipt sidecar drift: {path}",
    )
    return json.loads(body), digest


def _equal_session_mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


def run_audit(
    root: Path = plan.ROOT, *, budgets: Sequence[int] = plan.BUDGETS,
    surfaces: Sequence[str] = plan.SURFACES,
) -> dict[str, Any]:
    from src.tfpd_lane.matched_scorer import paired_session_stats

    root = Path(root)
    cache_root = root / plan.CACHE_ROOT_RELATIVE
    materialize, materialize_sha = _read_receipt(root / plan.RESULTS_ROOT_RELATIVE / plan.MATERIALIZE_NAME)
    _require(
        materialize.get("schema") == "slot_audit_v1_materialize_v1",
        "materialize receipt schema drift",
    )
    sealed_probe = load_sealed_probe(root)
    baseline_rows = sealed_baseline_rows(sealed_probe)
    rosters: dict[str, list[str]] = {
        surface: list(materialize["rosters"][surface]) for surface in surfaces
    }
    behavior_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def behavior_for(surface: str, session: str) -> dict[str, np.ndarray]:
        key = (surface, session)
        if key not in behavior_cache:
            behavior_cache[key] = cache_store.load_behavior(cache_root, surface, session)
        return behavior_cache[key]

    started = time.perf_counter()

    # -- sweep 1: per-slot calibration tables (the absolute-bin law) --------
    per_slot_tables: dict[tuple[str, int], dict[str, list[dict]]] = {}
    slot49_parity: list[dict[str, Any]] = []
    max_float_gap = 0.0
    for surface in surfaces:
        for budget in budgets:
            table: dict[str, list[dict]] = {}
            for session in rosters[surface]:
                pred = cache_store.load_prediction(cache_root, surface, session, budget)
                beh = behavior_for(surface, session)
                rows = alignment.per_slot_rows(
                    pred["full_predictions"], beh["behavior"], pred["starts"],
                    pred["valid"], beh["bin_valid"],
                )
                table[session] = rows
                max_float_gap = max(
                    max_float_gap,
                    max(abs(row["variance_weighted_r2"] - row[
                        "variance_weighted_r2_float64"
                    ]) for row in rows),
                )
                anchor = baseline_rows[(surface, int(budget), session)]
                delta = float(rows[plan.GOVERNING_BIN]["variance_weighted_r2"]) - float(
                    anchor["variance_weighted_r2"]
                )
                _require(
                    abs(delta) <= plan.BASELINE_R2_TOLERANCE,
                    f"{surface} M{budget} {session}: slot-49 per-slot R2 drift "
                    f"{delta:.3e} vs the sealed baseline row",
                )
                slot49_parity.append({
                    "surface": surface, "session": session, "budget": int(budget),
                    "slot49_per_slot_r2": float(
                        rows[plan.GOVERNING_BIN]["variance_weighted_r2"]
                    ),
                    "sealed_baseline_r2": float(anchor["variance_weighted_r2"]),
                    "delta_r2": delta,
                })
            per_slot_tables[(surface, int(budget))] = table

    # the ONLY source selection (guidance section 5.2): within-6, per budget
    source_slot_by_budget: dict[int, int] = {
        int(budget): ensembles.best_source_slot(per_slot_tables[("within", int(budget))])
        for budget in budgets
        if ("within", int(budget)) in per_slot_tables
    }

    # -- sweep 2: delta curves, parity groups, subsets, trajalign ------------
    cells: dict[str, dict[str, Any]] = {}
    for surface in surfaces:
        for budget in budgets:
            sessions = rosters[surface]
            per_slot_by_session = per_slot_tables[(surface, int(budget))]
            mean_rows = alignment.equal_session_mean_per_slot(per_slot_by_session)
            session_delta_sums = []
            parity_sums = []
            subset_rows: dict[str, list[dict[str, Any]]] = {
                name: [] for name in plan.FIXED_SUBSETS
            }
            subset_rows[plan.SOURCE_SELECTED_SUBSET] = []
            trajalign_rows: dict[str, list[dict[str, Any]]] = {
                f"K{k}": [] for k in plan.K_GRID
            }
            for session in sessions:
                pred = cache_store.load_prediction(cache_root, surface, session, budget)
                beh = behavior_for(surface, session)
                session_delta_sums.append(alignment.session_delta_sums(
                    pred["full_predictions"], beh["behavior"], pred["starts"],
                    pred["valid"], beh["bin_valid"],
                ))
                parity_sums.append(alignment.session_parity_sums(
                    pred["full_predictions"], beh["behavior"], pred["starts"],
                    pred["valid"], beh["bin_valid"],
                ))
                for name, slots in plan.FIXED_SUBSETS.items():
                    subset_rows[name].append(ensembles.subset_row(
                        pred["full_predictions"], pred["targets"], pred["valid"],
                        slots, name=name, session=session,
                    ))
                if int(budget) in source_slot_by_budget:
                    subset_rows[plan.SOURCE_SELECTED_SUBSET].append(
                        ensembles.subset_row(
                            pred["full_predictions"], pred["targets"],
                            pred["valid"],
                            (source_slot_by_budget[int(budget)],),
                            name=plan.SOURCE_SELECTED_SUBSET, session=session,
                            source_selected=True,
                        )
                    )
                for k in plan.K_GRID:
                    trajalign_rows[f"K{k}"].append({
                        "session": session,
                        **trajalign.trajalign_arm_r2(
                            pred["full_predictions"], pred["targets"],
                            pred["starts"], pred["valid"], k,
                        ),
                    })

            # subset aggregation + paired deltas vs {slot49}
            reference = subset_rows["slot49"]
            base_r2 = {
                row["session"]: float(row["variance_weighted_r2"]) for row in reference
            }
            _require(
                [row["session"] for row in reference] == sessions,
                "subset roster order drift",
            )
            subset_summary: dict[str, Any] = {}
            for name, rows in subset_rows.items():
                if not rows:
                    continue
                _require(
                    [row["session"] for row in rows] == sessions,
                    f"subset roster drift: {name}",
                )
                deltas = [
                    float(row["variance_weighted_r2"]) - base_r2[row["session"]]
                    for row in rows
                ]
                subset_summary[name] = {
                    "slots": rows[0]["slots"],
                    "source_selected": rows[0]["source_selected"],
                    "equal_session_mean_r2": _equal_session_mean([
                        float(row["variance_weighted_r2"]) for row in rows
                    ]),
                    "positive_sessions_vs_slot49": int(
                        sum(1 for value in deltas if value > 0.0)
                    ),
                    "paired_delta_vs_slot49": paired_session_stats(
                        deltas, seed=42, n_boot=2000,
                    ),
                }
            # slot49 subset row must be the sealed baseline, bit-anchored
            slot49_delta = max(
                abs(float(row["variance_weighted_r2"]) - float(
                    baseline_rows[(surface, int(budget), row["session"])]["variance_weighted_r2"]
                )) for row in reference
            )
            _require(
                slot49_delta <= plan.BASELINE_R2_TOLERANCE,
                f"{surface} M{budget}: subset {{49}} drift {slot49_delta:.3e} vs sealed rows",
            )

            # trajalign aggregation + paired gains vs the baseline subset
            trajalign_cells: dict[str, Any] = {}
            for k in plan.K_GRID:
                key = f"K{k}"
                rows = trajalign_rows[key]
                deltas = [
                    float(row["variance_weighted_r2"]) - base_r2[row["session"]]
                    for row in rows
                ]
                trajalign_cells[key] = {
                    "sessions": rows,
                    "equal_session_mean_r2": _equal_session_mean([
                        float(row["variance_weighted_r2"]) for row in rows
                    ]),
                    "baseline_equal_session_mean_r2": subset_summary["slot49"][
                        "equal_session_mean_r2"
                    ],
                    "gain_vs_baseline": _equal_session_mean(deltas),
                    "paired_gain_vs_baseline": paired_session_stats(
                        deltas, seed=42, n_boot=2000,
                    ),
                }

            mean_slot_r2 = [row["mean_variance_weighted_r2"] for row in mean_rows]
            cells[f"{surface}_M{budget}"] = {
                "surface": surface,
                "budget": int(budget),
                "n_sessions": len(sessions),
                "sessions": sessions,
                "per_slot_mean": mean_rows,
                "per_slot_mean_r2": mean_slot_r2,
                "per_slot_per_session_r2": {
                    session: [
                        float(row["variance_weighted_r2"])
                        for row in per_slot_by_session[session]
                    ]
                    for session in sessions
                },
                "worst_slot": int(np.argmin(mean_slot_r2)),
                "worst_slot_mean_r2": float(np.min(mean_slot_r2)),
                "best_slot": int(np.argmax(mean_slot_r2)),
                "best_slot_mean_r2": float(np.max(mean_slot_r2)),
                "slot49_mean_r2": mean_slot_r2[plan.GOVERNING_BIN],
                "delta_curve": alignment.delta_curve(
                    alignment.pool_delta_sums(session_delta_sums)
                ),
                "disjoint_window_parity_groups": alignment.pool_parity_sums(parity_sums),
                "subset_ensembles": subset_summary,
                "trajalign": trajalign_cells,
                "best_slot_per_session": ensembles.best_slot_per_session(
                    per_slot_by_session
                ),
            }

    # -- cross-cell reads ----------------------------------------------------
    reading = ensembles.classify_reading({
        key: {
            "per_slot_mean_r2": cell["per_slot_mean_r2"],
            "subset_mean_r2": {
                name: body["equal_session_mean_r2"]
                for name, body in cell["subset_ensembles"].items()
            },
            "subset_positive_sessions": {
                name: body["positive_sessions_vs_slot49"]
                for name, body in cell["subset_ensembles"].items()
            },
            "n_sessions": cell["n_sessions"],
        }
        for key, cell in cells.items()
    })

    recomputed_trajalign = {
        f"{surface}_M{budget}": cells[f"{surface}_M{budget}"]["trajalign"]
        for surface in surfaces for budget in budgets
    }
    reproduction = trajalign.compare_to_sealed(
        recomputed_trajalign, sealed_probe, surfaces=surfaces, budgets=budgets,
    )

    latency = {
        "bin_size_ms": plan.BIN_SIZE_MS,
        "provenance": plan.BIN_SIZE_PROVENANCE,
        "phrase_ban": plan.ZERO_LAG_BAN,
        "rows": [
            {
                "K": int(k),
                "delay_bins": int(k) - 1,
                "delay_ms": float(int(k) - 1) * plan.BIN_SIZE_MS,
            }
            for k in plan.K_GRID
        ],
    }

    manifest = cache_store.load_manifest(cache_root)
    digest_rows = [
        {
            "surface": entry["surface"], "session": entry["session"],
            "budget": int(entry["budget"]),
            "full_prediction_sha256": entry["full_prediction_bytes_sha256"],
            "last_bin_prediction_sha256": entry["last_bin_prediction_bytes_sha256"],
            "probe_baseline_prediction_sha256": baseline_rows[
                (entry["surface"], int(entry["budget"]), entry["session"])
            ]["prediction_sha256"],
        }
        for entry in manifest["entries"].values()
        if entry["kind"] == "predictions"
    ]
    for row in digest_rows:
        _require(
            row["full_prediction_sha256"] == row["probe_baseline_prediction_sha256"],
            f"full prediction digest drift vs sealed probe: "
            f"{row['surface']} M{row['budget']} {row['session']}",
        )

    payload = {
        "schema": "slot_audit_v1_terminal_v1",
        "status": "TERMINAL",
        "materialize_receipt": {
            "rel": str((root / plan.RESULTS_ROOT_RELATIVE / plan.MATERIALIZE_NAME).relative_to(root)),
            "sha256": materialize_sha,
        },
        "sealed_anchor": {
            "rel": plan.SEALED_PROBE_REL, "sha256": plan.SEALED_PROBE_SHA256,
        },
        "cache_manifest": cache_store.manifest_digest(cache_root),
        "target_source": materialize["target_source"],
        "bin_size": materialize["bin_size"],
        "slot49_baseline_parity_from_cache": {
            "tolerance": plan.BASELINE_R2_TOLERANCE,
            "rows": slot49_parity,
            "max_abs_delta_r2": max(abs(row["delta_r2"]) for row in slot49_parity),
        },
        "float32_vs_float64_r2_crosscheck_max_abs": max_float_gap,
        "source_selected_slot_by_budget": {
            str(budget): int(slot) for budget, slot in source_slot_by_budget.items()
        },
        "source_selection_rule": (
            "per budget, argmax over slots of the equal-session mean per-slot "
            "R2 on the within-6 surface only, applied unchanged to external; "
            "deterministic tie-break: lowest slot id; the only selection in "
            "this audit"
        ),
        "cells": cells,
        "trajalign_sealed_reproduction": reproduction,
        "latency": latency,
        "digests": digest_rows,
        "reading": reading,
        "caveats": {
            "common_mode": plan.COMMON_MODE_CAVEAT,
            "no_training": plan.NO_TRAINING_CAVEAT,
        },
        "model_or_checkpoint_updated": False,
        "target_optimizer_backward_update": 0,
        "boundaries": {
            "cpu_only": True,
            "audit_decoder_forwards": 0,
            "audit_statistics": (
                "numpy on the sha-verified cache; every governing R2 through "
                "the house float32 scorer (tfpd_lane.matched_scorer.session_r2)"
            ),
            "trajalign_law": (
                "imported from src/continuity_probe_v1 (smooth_trajectory_"
                "aligned + _trajalign_gather), never reimplemented"
            ),
            "zero_target_optimizer_steps": True,
            "zero_target_backward_calls": True,
            "zero_target_update_calls": True,
            "training_authorized": False,
            "deployment_change_authorized": False,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    return payload


# ---------------------------------------------------------------------------
# headline summary (printed by the runner)
# ---------------------------------------------------------------------------


def headline(payload: Mapping[str, Any]) -> str:
    lines = []
    for key, cell in payload["cells"].items():
        lines.append(
            f"[audit] {key}: per-slot R2 worst slot {cell['worst_slot']} "
            f"({cell['worst_slot_mean_r2']:+.4f}) best slot {cell['best_slot']} "
            f"({cell['best_slot_mean_r2']:+.4f}) slot49 {cell['slot49_mean_r2']:+.4f}"
        )
        low = cell["delta_curve"][0]
        high = cell["delta_curve"][-1]
        mid = cell["delta_curve"][len(cell["delta_curve"]) // 2]
        lines.append(
            f"[audit] {key}: delta-corr d1 {low['correlation']:+.4f} "
            f"d{mid['delta']} {mid['correlation']:+.4f} "
            f"d{high['delta']} {high['correlation']:+.4f}"
        )
        for name in ("slot49", "last5_s45_49", "last10_s40_49", "all50",
                     plan.SOURCE_SELECTED_SUBSET):
            if name not in cell["subset_ensembles"]:
                continue
            body = cell["subset_ensembles"][name]
            lines.append(
                f"[audit] {key}: subset {name:<26} mean "
                f"{body['equal_session_mean_r2']:+.6f} "
                f"pos {body['positive_sessions_vs_slot49']}/{cell['n_sessions']}"
            )
        for k in plan.K_GRID:
            body = cell["trajalign"][f"K{k}"]
            lines.append(
                f"[audit] {key}: trajalign K{k:<2} gain "
                f"{body['gain_vs_baseline']:+.6f}"
            )
    reproduction = payload["trajalign_sealed_reproduction"]
    lines.append(
        f"[audit] trajalign sealed reproduction: max |dR2| session "
        f"{reproduction['max_abs_session_delta_r2']:.2e} mean "
        f"{reproduction['max_abs_mean_delta_r2']:.2e} "
        f"(tol {reproduction['tolerance']:.0e}, {reproduction['arms_checked']} arms)"
    )
    lines.append(f"[audit] reading: {payload['reading']['classification']}")
    return "\n".join(lines)
