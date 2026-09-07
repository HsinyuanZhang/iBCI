"""Stage-O directions: the full-grid TRUE completed-trial direction table.

Work order: the TRUE direction must be recomputed per session/budget from
``session.behavior`` rows through the same frozen law the P2' oracle rows use:

* ``filters.true_physical_velocity_one_trial`` restores the sealed z-scored
  behavior rows to physical velocity units (padding rows zeroed, counted);
* ``core.pseudo_direction_from_velocity`` integrates the completed trial and
  applies the frozen displacement / speed / canonical-snap gates under the
  session carrier's own ``CDMDConfig``.

The table is published once, digest-pinned per (surface, session, budget), and
the replay stage re-derives every direction and asserts bit-equality with it
(double entry).  A non-governing cross-reference against the sealed P2'
sub-study true-direction acceptance counts is recorded where the sub-study ran
(M4/M10, both surfaces).  This stage parses the held assets on one bound GPU
but never runs a model forward.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.learned_gate_p2prime_v1 import filters as p2filters
from src.learned_gate_p2prime_v1 import physical as p2physical

from . import plan


class StageODirectionsError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageODirectionsError(message)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_sidecar(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sidecar = Path(path).with_name(Path(path).name + ".sha256")
    _require(sidecar.exists(), f"missing sidecar: {sidecar}")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{digest}  {Path(path).name}",
        f"sidecar drift for {Path(path).name}",
    )
    return digest


def _true_direction_row(
    *,
    session: Any,
    trial: Any,
    config: Any,
    behavior_mean: Any,
    behavior_std: Any,
) -> dict[str, Any]:
    restored, padded = p2filters.true_physical_velocity_one_trial(
        np.asarray(session.behavior[trial.endpoint_bins]),
        behavior_mean=behavior_mean, behavior_std=behavior_std,
    )
    from src.causal_dual_memory_cell_d_v1 import core as cdm_core

    direction = cdm_core.pseudo_direction_from_velocity(
        restored, trial.velocity_validity.valid_mask, config=config,
    )
    return {
        "trial_id": trial.trial_id,
        "chronology_position": int(trial.chronology_position),
        "n_windows": int(np.asarray(trial.endpoint_bins).size),
        "accepted": bool(direction.accepted),
        "reason": None if direction.reason is None else direction.reason.value,
        "theta_raw_rad": direction.theta_raw_rad,
        "theta_index": None if direction.theta_index is None else int(direction.theta_index),
        "theta_canonical_rad": direction.theta_canonical_rad,
        "canonical_distance_rad": direction.canonical_distance_rad,
        "movement_bins": int(direction.movement_bins),
        "displacement_norm": direction.displacement_norm,
        "mean_speed": direction.mean_speed,
        "padded_true_rows": int(np.sum(padded)),
    }


def _cell_digest(rows: list[Mapping[str, Any]], binding: Mapping[str, Any]) -> str:
    return direction_cell_digest(rows, binding)


def direction_cell_digest(rows: list[Mapping[str, Any]], binding: Mapping[str, Any]) -> str:
    """The pinned digest of one (surface, session, budget) direction cell."""
    return hashlib.sha256(plan_json_bytes({
        "rows": [dict(row) for row in rows],
        "binding": dict(binding),
    })).hexdigest()


def plan_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _substudy_cross_reference(base: Path, table: Mapping[str, Any]) -> dict[str, Any]:
    """Non-governing: compare true-direction acceptance with the sealed sub-study."""
    path = Path(base).absolute() / "tfpd_exploration/results/learned_gate_p2prime_v1/stage_substudy.json"
    if not path.exists():
        return {"available": False}
    substudy = _read_json(path)
    comparisons: list[dict[str, Any]] = []
    mismatches = 0
    for record in substudy["sessions"]:
        budget = int(record["budget"])
        surface = str(record["surface"])
        session = str(record["session"])
        key = f"{surface}:{session}:m{budget}"
        mine = table["cells"].get(key)
        if mine is None:
            continue
        sealed_accepted = int(record["direction_rows"]["raw"]["true_accepted_trials"])
        my_accepted = int(mine["accepted_trials"])
        equal = sealed_accepted == my_accepted
        mismatches += 0 if equal else 1
        comparisons.append({
            "cell": key,
            "sealed_true_accepted_trials": sealed_accepted,
            "stage_o_accepted_trials": my_accepted,
            "equal": bool(equal),
        })
    return {
        "available": True,
        "source": "tfpd_exploration/results/learned_gate_p2prime_v1/stage_substudy.json (read-only)",
        "n_compared": len(comparisons),
        "n_mismatch": mismatches,
        "all_equal": mismatches == 0,
        "comparisons": comparisons,
        "non_governing": True,
    }


def build_direction_table(base: Path, *, gpu_index: int, attempt_sha256: str) -> dict[str, Any]:
    """Parse the full grid once and publish the digest-pinned direction table."""
    base = Path(base).absolute()
    environment = p2physical.validate_environment(gpu_index=gpu_index)
    started = time.monotonic()
    runtime, meta, identity = p2physical._runtime_for(base, gpu_index=gpu_index)
    try:
        runtime.prepare(identity=identity)
        fixed = v1score.derive_fixed_evaluation_authority(base)
        runtime.materialize_inputs(identity=identity, authority=fixed)
        state = runtime._require_state()
        from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan

        cells: dict[str, Any] = {}
        sessions_payload: dict[str, Any] = {}
        padded_total = 0
        for surface in plan.SURFACES:
            for key in p2physical._ordered_session_keys(runtime, surface):
                session = state.sessions[key]
                query_ids_by_budget = {
                    budget: list(session.query_trial_ids[budget]) for budget in plan.BUDGETS
                }
                _require(
                    all(query_ids_by_budget[b] == query_ids_by_budget[4] for b in plan.BUDGETS),
                    f"query chronology differs across budgets for {session.session}",
                )
                session_entry: dict[str, Any] = {
                    "surface": surface,
                    "session": session.session,
                    "n_query_trials": len(query_ids_by_budget[4]),
                    "budgets": {},
                }
                rows_by_budget: dict[int, list[Mapping[str, Any]]] = {}
                for budget in plan.BUDGETS:
                    memory, initial = runtime._initial_memory(session=session, budget=budget)
                    config = memory.state.carrier.config
                    rows = [
                        _true_direction_row(
                            session=session,
                            trial=session.trials_by_id[trial_id],
                            config=config,
                            behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN,
                            behavior_std=v1plan.SEALED_BEHAVIOR_STD,
                        )
                        for trial_id in query_ids_by_budget[budget]
                    ]
                    rows_by_budget[int(budget)] = rows
                    binding = {
                        "budget": int(budget),
                        "support_trial_ids": list(session.support_trial_ids[budget]),
                        "initial_carrier_sha256": str(initial["initial_carrier_sha256"]),
                        "initial_activity_sha256": str(initial["initial_activity_sha256"]),
                        "group_assignment_sha256": str(initial["group_assignment_sha256"]),
                        "config_payload_sha256": hashlib.sha256(
                            plan_json_bytes(config.payload())
                        ).hexdigest(),
                    }
                    digest = _cell_digest(rows, binding)
                    accepted = sum(1 for row in rows if row["accepted"])
                    cells[f"{surface}:{session.session}:m{budget}"] = {
                        "budget": int(budget),
                        "surface": surface,
                        "session": session.session,
                        "accepted_trials": accepted,
                        "rejected_trials": len(rows) - accepted,
                        "rejection_reasons": _reason_counts(
                            row["reason"] for row in rows if not row["accepted"]
                        ),
                        "cell_digest": digest,
                        "binding": binding,
                        "rows": rows,
                    }
                    session_entry["budgets"][f"m{budget}"] = {
                        "cell_digest": digest,
                        "accepted_trials": accepted,
                    }
                    padded_total += sum(int(row["padded_true_rows"]) for row in rows)
                    runtime._bump_trial_counters()
                sessions_payload[session.session] = session_entry
                # The direction facts are budget-independent (the frozen gates do
                # not vary with the support budget); only the binding differs.
                _require(
                    all(rows_by_budget[int(budget)] == rows_by_budget[4] for budget in plan.BUDGETS),
                    f"direction rows differ across budgets for {session.session}",
                )
        # No model forward runs in this stage, so the runtime's forward-based
        # resource counter is unavailable; record the stage-local facts instead.
        resources = {
            "note": "no model forwards in the directions stage; parse only",
            "completed_trials_counter": int(state.completed_trials),
            "within_assets_opened": bool(runtime._within_opened),
            "external_assets_opened": bool(runtime._external_opened),
        }
    finally:
        runtime.close()
    table = {
        "schema": f"{plan.SCHEMA}_directions_v1",
        "stage": "true_direction_table_full_grid",
        "status": "DIRECTIONS_COMPLETE",
        "attempt_sha256": attempt_sha256,
        "identity_sha256": meta["identity_sha256"],
        "environment": environment,
        "gpu_index": int(gpu_index),
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "direction_law": (
            "filters.true_physical_velocity_one_trial + core.pseudo_direction_from_velocity "
            "under the session carrier config (the frozen P2' oracle law)"
        ),
        "sessions": sessions_payload,
        "cells": cells,
        "padded_true_rows_total": padded_total,
        "resources": resources,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "model_or_checkpoint_updated": False,
        "target_optimizer_backward_update": 0,
    }
    table["substudy_cross_reference"] = _substudy_cross_reference(base, table)
    return table


def _reason_counts(reasons: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in reasons:
        if reason is None:
            continue
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts
