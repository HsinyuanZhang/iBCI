"""CAL-AUG section 6: frozen deployment-recipe scoring + the section 6.7 gates.

Both arms are scored ONCE each on the frozen M4/M10/M30 within-6 +
external-15 inputs through the sealed deployment recipe, reusing the
``learnable_output_filter_v1`` static materialization law as the template:

* the Z1 ``HonestOracleRuntime`` (CPU-only, reviewed provenance/asset readers);
* ``p4_stream_stats.materialize_session`` for every within/external session —
  D-opt-first-30 selection at M4, chronological at M10/M30, ridge-T4
  lambda=0.1 per budget (``inputs.selected_by_budget`` / ``side_by_budget``);
* ``p4_stream_stats._decode_static(runtime, inputs, activity, side)`` for the
  forward (one static identity, chunk-32 decode);
* governed last-bin house R2 via the sealed ``matched_scorer.session_r2``,
  equal-session means, paired per-session deltas, positive counts, fixed-seed
  session bootstrap CIs (seed 42), full prediction digests;
* the runtime's sealed Cell-D model is SWAPPED for the arm's sealed SWA
  (strict load), with the model state digest compared before/after the run;
* zero target optimizer/backward/update counts.

The ONLY difference between the two scored arms is the loaded checkpoint:
C1's trained weights versus T0's.  The prefix cycle NEVER runs here —
deployment consumes exactly the frozen recipe's selected calibration trials.

Gates (work order section 6 / guidance section 6.7), boundary rule identical
to ``plan.GATE_SPEC``: exact ``>=`` on the float64 equal-session mean, with
the 1e-12 program epsilon recorded as a disclosed boundary band only.
"""

from __future__ import annotations

import hashlib
import sys
import time
from typing import Mapping, Optional

from . import plan
from .mechanism import equal_session_mean, margin_verdict
from .receipts import sha256_file


class DeploymentError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeploymentError(message)


# ---------------------------------------------------------------------------
# pure gate arithmetic (float64; no torch needed)
# ---------------------------------------------------------------------------


def _safety_verdict(mean: float, margin: float) -> dict:
    return margin_verdict(mean, margin)


def lower_continuation_gate(
    *,
    external_m4: Mapping[str, float],
    external_m10: Mapping[str, float],
    external_m30_mean: float,
    within_means_by_budget: Mapping[str, float],
) -> dict:
    """Guidance section 6.7 lower continuation gate.

    ``external_m4``/``external_m10`` carry ``mean`` (equal-session delta),
    ``n_positive`` and ``bootstrap_lb``; the other-low-budget clause demands
    the non-lead budget's mean delta >= 0; external M30 >= -0.02; within at
    EVERY reported budget >= -0.02.
    """
    for label, summary in (("external_m4", external_m4), ("external_m10", external_m10)):
        for field in ("mean", "n_positive", "bootstrap_lb"):
            _require(field in summary, f"lower gate input drift: {label}.{field}")
    lead_m4 = bool(
        float(external_m4["mean"]) >= plan.LOWER_CONTINUATION_DELTA_R2
        and int(external_m4["n_positive"]) >= plan.EXTERNAL_POSITIVE_REQUIRED
        and float(external_m4["bootstrap_lb"]) >= 0.0
    )
    lead_m10 = bool(
        float(external_m10["mean"]) >= plan.LOWER_CONTINUATION_DELTA_R2
        and int(external_m10["n_positive"]) >= plan.EXTERNAL_POSITIVE_REQUIRED
        and float(external_m10["bootstrap_lb"]) >= 0.0
    )
    any_lead = bool(lead_m4 or lead_m10)
    if lead_m4 and lead_m10:
        other_clause = {"other_budget": "both_lead", "mean": None, "passed": True}
    elif lead_m4:
        other_mean = float(external_m10["mean"])
        other_clause = {
            "other_budget": "external_m10",
            "mean": other_mean,
            "passed": bool(other_mean >= 0.0),
            "margin_verdict": margin_verdict(other_mean, 0.0),
        }
    elif lead_m10:
        other_mean = float(external_m4["mean"])
        other_clause = {
            "other_budget": "external_m4",
            "mean": other_mean,
            "passed": bool(other_mean >= 0.0),
            "margin_verdict": margin_verdict(other_mean, 0.0),
        }
    else:
        other_clause = {"other_budget": None, "mean": None, "passed": False}
    external_m30 = _safety_verdict(external_m30_mean, plan.EXTERNAL_M30_SAFETY_MARGIN_R2)
    _require(
        set(within_means_by_budget) == {str(b) for b in plan.BUDGETS},
        "within safety must be evaluated at every reported budget",
    )
    within_verdicts = {
        budget: _safety_verdict(mean, plan.WITHIN_SAFETY_MARGIN_R2)
        for budget, mean in within_means_by_budget.items()
    }
    within_ok = all(entry["meets_margin"] for entry in within_verdicts.values())
    passed = bool(any_lead and other_clause["passed"] and external_m30["meets_margin"] and within_ok)
    return {
        "expression": plan.GATE_SPEC["lower_continuation"]["expression"],
        "authority": plan.GATE_SPEC["lower_continuation"]["authority"],
        "lead_m4": lead_m4,
        "lead_m10": lead_m10,
        "any_lead_budget": any_lead,
        "other_low_budget_clause": other_clause,
        "external_m30_safety": external_m30,
        "within_safety_by_budget": within_verdicts,
        "passed": passed,
        "disposition": plan.LOWER_CONTINUATION_PASSED if passed else plan.LOWER_CONTINUATION_FAILED,
    }


def primary_claim_gate(
    *,
    external_m4: Mapping[str, float],
    external_m10: Mapping[str, float],
    external_m30_mean: float,
    within_means_by_budget: Mapping[str, float],
) -> dict:
    """The primary performance claim: +0.03 lead with all safety conditions."""
    lead_m4 = bool(
        float(external_m4["mean"]) >= plan.PRIMARY_DELTA_R2
        and int(external_m4["n_positive"]) >= plan.EXTERNAL_POSITIVE_REQUIRED
    )
    lead_m10 = bool(
        float(external_m10["mean"]) >= plan.PRIMARY_DELTA_R2
        and int(external_m10["n_positive"]) >= plan.EXTERNAL_POSITIVE_REQUIRED
    )
    any_lead = bool(lead_m4 or lead_m10)
    if lead_m4 and lead_m10:
        other_clause = {"other_budget": "both_lead", "mean": None, "passed": True}
    elif lead_m4:
        other_mean = float(external_m10["mean"])
        other_clause = {"other_budget": "external_m10", "mean": other_mean, "passed": bool(other_mean >= 0.0)}
    elif lead_m10:
        other_mean = float(external_m4["mean"])
        other_clause = {"other_budget": "external_m4", "mean": other_mean, "passed": bool(other_mean >= 0.0)}
    else:
        other_clause = {"other_budget": None, "mean": None, "passed": False}
    lower = lower_continuation_gate(
        external_m4=external_m4,
        external_m10=external_m10,
        external_m30_mean=external_m30_mean,
        within_means_by_budget=within_means_by_budget,
    )
    passed = bool(any_lead and other_clause["passed"] and lower["passed"])
    return {
        "expression": plan.GATE_SPEC["primary_claim"]["expression"],
        "lead_m4": lead_m4,
        "lead_m10": lead_m10,
        "any_lead_budget": any_lead,
        "other_low_budget_clause": other_clause,
        "safety_conditions": {
            "external_m30_safety": lower["external_m30_safety"],
            "within_safety_by_budget": lower["within_safety_by_budget"],
        },
        "lower_gate_passed": lower["passed"],
        "passed": passed,
        "disposition": plan.PRIMARY_CLAIM_PASSED if passed else plan.PRIMARY_CLAIM_FAILED,
    }


def combined_disposition(*, mechanism_gate: Mapping, lower_gate: Mapping, primary_gate: Mapping) -> dict:
    """The section 6.7 stop rules (never average budgets to hide an M30 loss)."""
    mechanism_passed = bool(mechanism_gate.get("passed", False))
    lower_passed = bool(lower_gate.get("passed", False))
    primary_passed = bool(primary_gate.get("passed", False))
    if primary_passed:
        status = plan.PRIMARY_CLAIM_PASSED
    elif mechanism_passed and not lower_passed:
        status = plan.MECHANISM_POSITIVE_DEPLOYMENT_INCONCLUSIVE
    elif mechanism_passed and lower_passed:
        status = plan.LOWER_CONTINUATION_PASSED
    else:
        status = plan.LOWER_CONTINUATION_FAILED
    return {
        "mechanism_gate_passed": mechanism_passed,
        "lower_continuation_gate_passed": lower_passed,
        "primary_claim_gate_passed": primary_passed,
        "status": status,
        "stop_rule": (
            "MECHANISM_POSITIVE__DEPLOYMENT_INCONCLUSIVE stops C1 expansion: the "
            "mechanism is not called null and not called a deployable improvement"
            if status == plan.MECHANISM_POSITIVE_DEPLOYMENT_INCONCLUSIVE
            else plan.GATE_SPEC["never_average_budgets"]
        ),
    }


# ---------------------------------------------------------------------------
# arm runtime through the sealed Z1/P4 harness (CPU-only)
# ---------------------------------------------------------------------------


DEPLOYMENT_SURFACES = ("within", "external")
DEPLOYMENT_BUDGETS = tuple(plan.BUDGETS)


def _swap_runtime_model(runtime, pop_robust, arm_common, swa_path):
    """Strict-load the arm SWA into a fresh Cell-D graph and swap it in."""
    import torch

    payload = torch.load(swa_path, map_location="cpu", weights_only=False)
    model = pop_robust.build_population_robustness_model(seed=plan.SEED, cell="D")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    runtime._model = model
    return {
        "arm_state_sha256": arm_common.state_sha256(model),
        "artifact_sha256": sha256_file(swa_path),
        "strict_load": True,
        "eval_mode": not model.training,
    }


def score_arm_on_surfaces(runtime, *, surfaces=DEPLOYMENT_SURFACES, budgets=DEPLOYMENT_BUDGETS) -> dict:
    """Score one arm on the frozen deployment recipe over within-6/external-15."""
    import torch

    from src.calibration_gap_v1 import p4_stream_stats as p4

    matched_scorer = sys.modules["tfpd_lane_matched_scorer"]
    session_r2 = matched_scorer.session_r2
    pop_robust = sys.modules["tfpd_lane_pop_robust"]
    state_before = runtime.state_digest()
    rows: list[dict] = []
    for surface in surfaces:
        roster = runtime.external_roster if surface == "external" else runtime.within_roster
        for session_name in roster:
            inputs = p4.materialize_session(runtime, surface, session_name)
            for budget in budgets:
                selected = inputs.selected_by_budget[budget]
                activity = inputs.calib[list(selected)]
                side = inputs.side_by_budget[budget]
                t0 = time.perf_counter()
                with pop_robust.dynamic_dropout_recorder() as recorder:
                    prediction, _identities = p4._decode_static(
                        runtime, inputs, activity, side
                    )
                if recorder["uniform_calls"] != 0 or recorder["dropout_calls"]:
                    raise DeploymentError(
                        "deployment eval dropout became active "
                        f"({surface} {session_name} M{budget})"
                    )
                wall_s = time.perf_counter() - t0
                last = prediction[:, 49, :].contiguous()
                target = torch.from_numpy(inputs.last_targets.copy())
                valid = torch.from_numpy(inputs.last_valid_mask.copy())
                n_valid = int(valid.sum().item())
                if n_valid <= 0:
                    raise DeploymentError(
                        f"no valid last-bin rows: {surface} {session_name} M{budget}"
                    )
                r2 = session_r2(last[valid], target[valid])
                rows.append(
                    {
                        "arm": None,
                        "surface": surface,
                        "session": session_name,
                        "budget": int(budget),
                        "r2": float(r2),
                        "n_windows": int(inputs.n_windows),
                        "n_valid_last_bin": n_valid,
                        "all_rows_valid": bool(valid.all().item()),
                        "prediction_sha256": hashlib.sha256(
                            prediction.detach().contiguous().numpy().tobytes()
                        ).hexdigest(),
                        "target_sha256": inputs.target_sha256,
                        "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
                        "normalized_side_sha256": inputs.side_sha_by_budget[budget],
                        "raw_t4_sha256": inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"],
                        "calibration_prefix_sha256": calibration_prefix_digest(
                            runtime, inputs, budget
                        ),
                        "wall_seconds": wall_s,
                        "target_optimizer_steps": 0,
                        "target_backward_calls": 0,
                        "target_update_calls": 0,
                    }
                )
    state_after = runtime.state_digest()
    _require(state_before == state_after, "deployment changed the arm model state")
    return {"rows": rows, "state_sha256": state_before, "state_unchanged": True}


def calibration_prefix_digest(runtime, inputs, budget) -> str:
    """Digest of the frozen recipe's selected calibration rows for one cell."""
    import torch

    selected = inputs.selected_by_budget[budget]
    activity = inputs.calib[list(selected)]
    from . import schedule

    return schedule.visible_slice_digest(activity)


def paired_arm_summary(t0_rows: list, c1_rows: list) -> dict:
    """Per surface/budget paired deltas C1 - T0 with bootstrap CIs (seed 42)."""
    matched_scorer = sys.modules["tfpd_lane_matched_scorer"]

    def cells(rows):
        table: dict[tuple[str, int], list[dict]] = {}
        for row in rows:
            _require(row["arm"] is not None, "row missing arm label")
            table.setdefault((row["surface"], int(row["budget"])), []).append(row)
        return table

    t0_cells, c1_cells = cells(t0_rows), cells(c1_rows)
    _require(set(t0_cells) == set(c1_cells), "T0/C1 scoring cell topology drift")
    summaries = {}
    for key in sorted(t0_cells, key=lambda k: (k[0], k[1])):
        surface, budget = key
        t0_rows_cell, c1_rows_cell = t0_cells[key], c1_cells[key]
        _require(
            [r["session"] for r in t0_rows_cell] == [r["session"] for r in c1_rows_cell],
            f"paired roster drift at {surface} M{budget}",
        )
        deltas = [
            float(c["r2"]) - float(t["r2"])
            for t, c in zip(t0_rows_cell, c1_rows_cell, strict=True)
        ]
        stats = matched_scorer.paired_session_stats(deltas, seed=plan.BOOTSTRAP_SEED)
        summaries[f"{surface}:m{budget}"] = {
            "surface": surface,
            "budget": int(budget),
            "n_sessions": len(deltas),
            "t0_equal_session_mean_r2": equal_session_mean([r["r2"] for r in t0_rows_cell]),
            "c1_equal_session_mean_r2": equal_session_mean([r["r2"] for r in c1_rows_cell]),
            "equal_session_mean_delta": equal_session_mean([r["r2"] for r in c1_rows_cell])
            - equal_session_mean([r["r2"] for r in t0_rows_cell]),
            "mean_of_paired_deltas": stats["mean"],
            "positive_sessions": stats["n_positive"],
            "bootstrap_lb": stats["bootstrap_95_interval"][0],
            "bootstrap_95_interval": stats["bootstrap_95_interval"],
            "per_session": [
                {"session": t["session"], "t0_r2": float(t["r2"]), "c1_r2": float(c["r2"]),
                 "delta_r2": float(c["r2"]) - float(t["r2"])}
                for t, c in zip(t0_rows_cell, c1_rows_cell, strict=True)
            ],
        }
    return summaries


def gate_inputs_from_summary(summaries: Mapping[str, Mapping]) -> dict:
    """Extract the gate-facing external/within rows from the paired summary."""
    def cell(surface: str, budget: int) -> dict:
        entry = summaries[f"{surface}:m{budget}"]
        return {
            "mean": float(entry["equal_session_mean_delta"]),
            "n_positive": int(entry["positive_sessions"]),
            "bootstrap_lb": float(entry["bootstrap_lb"]),
            "n_sessions": int(entry["n_sessions"]),
        }

    within_means = {
        str(budget): float(summaries[f"within:m{budget}"]["equal_session_mean_delta"])
        for budget in plan.BUDGETS
    }
    return {
        "external_m4": cell("external", 4),
        "external_m10": cell("external", 10),
        "external_m30_mean": float(summaries["external:m30"]["equal_session_mean_delta"]),
        "within_means_by_budget": within_means,
    }


def evaluate_deployment(
    *,
    repo_root,
    t0_swa,
    c1_swa,
    mechanism_gate: Optional[Mapping] = None,
) -> dict:
    """Score both arms once each and evaluate the section 6.7 gates."""
    from src.calibration_gap_v1 import z1_oracle_cells as z1

    arm_common = sys.modules["tfpd_lane_arm_common"]
    pop_robust = sys.modules["tfpd_lane_pop_robust"]
    started = time.perf_counter()
    arm_results = {}
    arm_bindings = {}
    for arm, swa in (("t0", t0_swa), ("c1", c1_swa)):
        runtime = z1.HonestOracleRuntime(root=repo_root)
        sealed_state_sha = runtime.sealed_state_sha256
        swap = _swap_runtime_model(runtime, pop_robust, arm_common, swa)
        swap["sealed_state_sha256_at_build"] = sealed_state_sha
        arm_bindings[arm] = swap
        scored = score_arm_on_surfaces(runtime)
        for row in scored["rows"]:
            row["arm"] = arm
        arm_results[arm] = scored
        runtime.close()
    summaries = paired_arm_summary(arm_results["t0"]["rows"], arm_results["c1"]["rows"])
    gate_inputs = gate_inputs_from_summary(summaries)
    lower = lower_continuation_gate(
        external_m4=gate_inputs["external_m4"],
        external_m10=gate_inputs["external_m10"],
        external_m30_mean=gate_inputs["external_m30_mean"],
        within_means_by_budget=gate_inputs["within_means_by_budget"],
    )
    primary = primary_claim_gate(
        external_m4=gate_inputs["external_m4"],
        external_m10=gate_inputs["external_m10"],
        external_m30_mean=gate_inputs["external_m30_mean"],
        within_means_by_budget=gate_inputs["within_means_by_budget"],
    )
    mechanism = dict(mechanism_gate) if mechanism_gate is not None else None
    combined = combined_disposition(
        mechanism_gate=mechanism or {"passed": False, "source": "mechanism gate not supplied"},
        lower_gate=lower,
        primary_gate=primary,
    )
    return {
        "schema": plan.SCHEMA + "_deployment",
        "arms": {
            arm: {
                "bindings": arm_bindings[arm],
                "state_sha256": arm_results[arm]["state_sha256"],
                "state_unchanged": arm_results[arm]["state_unchanged"],
                "rows": arm_results[arm]["rows"],
            }
            for arm in ("t0", "c1")
        },
        "paired_summary": summaries,
        "gate_inputs": gate_inputs,
        "lower_continuation_gate": lower,
        "primary_claim_gate": primary,
        "mechanism_gate_input": mechanism,
        "combined_disposition": combined,
        "wall_seconds": time.perf_counter() - started,
        "target_optimizer_backward_update": 0,
    }
