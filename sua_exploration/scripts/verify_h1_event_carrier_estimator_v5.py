#!/usr/bin/env python3
"""Independent arithmetic/scope verifier for an immutable H1 V5 estimator receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "h1_event_carrier_estimator_v5_source_screen_v2"
PROTOCOL = "h1_event_carrier_likelihood_nuisance_prior_source_screen_20260812_v2"
INVALIDATED_V5R1_SHA256 = "a50ff3706c3b2bbc922d83bae09fc6b8cb4dee6d8eae74989a4a183af29eeb86"
CANDIDATES = (
    "hse5_pca_delta_q4", "poisson_exposure_irls", "within_trial_contrast_ridge", "eb_channel_shrinkage",
)
DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: Any, right: Any, tolerance: float = 1.0e-12) -> None:
    need(left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance),
         f"V5 numeric mismatch: {left!r} vs {right!r}")


def summary(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    values = np.asarray([float(value) for _name, value in rows], dtype=np.float64)
    need(values.size == 13 and np.isfinite(values).all(), "V5 aggregate must contain 13 finite sessions")
    removed = int(np.argmax(np.abs(values)))
    return {
        "defined_sessions": 13,
        "mean": float(values.mean()), "median": float(np.median(values)),
        "positive": int(np.sum(values > 0)), "zero": int(np.sum(values == 0)), "negative": int(np.sum(values < 0)),
        "leave_largest_absolute_out_mean": float(np.delete(values, removed).mean()),
        "removed_session": rows[removed][0],
    }


def verify_summary(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        need(observed.get(key) == expected.get(key), f"V5 summary drift at {key}")
    for key in ("mean", "median", "leave_largest_absolute_out_mean"):
        close(observed.get(key), expected.get(key))


def positive(value: Mapping[str, Any]) -> bool:
    return bool(value["defined_sessions"] == 13 and value["mean"] > 0 and value["median"] > 0
                and value["positive"] >= 10 and value["leave_largest_absolute_out_mean"] > 0)


def verify_poisson_fail_closed(rows: Mapping[str, Mapping[str, Any]], aggregate: Mapping[str, Any]) -> None:
    """Ensure an incomplete IRLS fit cannot be silently scored as a negative arm."""

    undefined = [name for name, row in rows.items() if row.get("status") == "undefined_nonconverged_poisson_fit"]
    need(bool(undefined) and all(row.get("status") in {"defined", "undefined_nonconverged_poisson_fit"}
                                 for row in rows.values()), "V5 Poisson row-status drift")
    for name in undefined:
        for forbidden in ("median_r2_correct", "median_delta_label_shuffle", "median_delta_intercept", "median_delta_row_shuffle"):
            need(forbidden not in rows[name], f"V5 nonconverged Poisson row was scored: {name}/{forbidden}")
    need(aggregate.get("status") == "undefined_nonconverged_poisson_fits", "V5 Poisson aggregate did not fail closed")
    need(aggregate["defined_session_count"] == len(rows) - len(undefined)
         and aggregate["undefined_session_count"] == len(undefined)
         and aggregate["undefined_sessions"] == undefined,
         "V5 Poisson undefined-session accounting drift")
    convergence = aggregate["poisson_convergence"]
    need(convergence["fail_closed"] is True and "no Poisson forward R2" in convergence["interpretation"],
         "V5 Poisson fail-closed statement missing")
    for key, fit_key in (("correct_fit", "fit"), ("label_shuffle_fit", "label_fit")):
        flags = [bool(row[fit_key]["converged"]) for row in rows.values()]
        expected = convergence[key]
        need(expected["evaluated_sessions"] == len(rows)
             and expected["converged_sessions"] == int(sum(flags))
             and expected["nonconverged_sessions"] == int(len(rows) - sum(flags)),
             f"V5 Poisson {key} convergence accounting drift")
        maximum = max(float(row[fit_key]["maximum_final_beta_update"]) for row in rows.values())
        close(expected["maximum_final_beta_update"], maximum)


def poisson_convergence_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Derive and validate the mandatory IRLS convergence report from raw rows."""

    output: dict[str, Any] = {}
    for key, fit_key in (("correct_fit", "fit"), ("label_shuffle_fit", "label_fit")):
        flags = [bool(row[fit_key]["converged"]) for row in rows.values()]
        iterations = [int(row[fit_key]["iterations"]) for row in rows.values()]
        update = [float(row[fit_key]["maximum_final_beta_update"]) for row in rows.values()]
        output[key] = {
            "evaluated_sessions": len(rows),
            "converged_sessions": int(sum(flags)),
            "nonconverged_sessions": int(len(rows) - sum(flags)),
            "maximum_iterations": int(max(iterations)),
            "maximum_final_beta_update": float(max(update)),
        }
    return output


def verify(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    need(resolved.is_file() and stat.S_IMODE(resolved.stat().st_mode) == 0o444, "V5 receipt must be immutable mode 0444")
    receipt_sha = file_sha(resolved)
    body = json.loads(resolved.read_text(encoding="utf-8"))
    need(body.get("schema") == SCHEMA and body.get("protocol") == PROTOCOL, "V5 schema/protocol mismatch")
    constants = body["frozen_constants"]
    need(constants["rank"] == 4 and constants["carrier_dim"] == 5, "V5 width drift")
    need(tuple(constants["support_budgets"]) == (3, 4), "V5 budget drift")
    need(tuple(constants["candidates_in_fixed_order"]) == CANDIDATES, "V5 candidate matrix drift")
    need(body["candidate_matrix_predeclared_before_data_run"] is True, "V5 predeclaration missing")
    supersedes = body["supersedes"]
    old = Path(supersedes["invalidated_receipt_path"]).resolve()
    need(supersedes["invalidated_receipt_sha256"] == INVALIDATED_V5R1_SHA256
         and supersedes["old_receipt_preserved_unmodified"] is True
         and supersedes["same_predeclared_candidate_matrix"] is True,
         "V5r2 supersession manifest drift")
    need(old.is_file() and stat.S_IMODE(old.stat().st_mode) == 0o444 and file_sha(old) == INVALIDATED_V5R1_SHA256,
         "V5r2 invalidated V5r1 receipt was not preserved exactly")
    scope = body["scope"]
    expected_scope = {
        "native_position_endpoints_per_event": 2, "native_event_timestamps_per_event": 2,
        "dense_velocity_opened": False, "within_event_position_trajectory_opened": False,
        "target_session_optimizer_steps": 0, "target_session_backward_steps": 0,
        "public_held_in_calibration_nwbs_opened": 13, "minival_nwbs_opened": 0,
        "held_out_nwbs_opened": 0, "formal_test_labels_opened": 0,
        "decoder_constructed": False, "trainer_constructed": False, "cuda_used": False,
    }
    for key, value in expected_scope.items():
        need(scope.get(key) == value, f"V5 scope drift at {key}")
    binding = body["implementation_binding"]
    for prefix in ("module", "runner", "shared_design", "event_parser"):
        bound = Path(binding[f"{prefix}_path"]).resolve()
        need(file_sha(bound) == binding[f"{prefix}_sha256"], f"V5 {prefix} SHA drift")
    reproduction = body["baseline_reproduction"]
    need(reproduction["passed"] is True and reproduction["comparisons"] == 78,
         "V5 exact H-SE5 reproduction missing")
    need(float(reproduction["maximum_absolute_difference"]) <= float(reproduction["absolute_tolerance"]),
         "V5 H-SE5 reproduction exceeds tolerance")
    need(file_sha(Path(reproduction["reference_path"])) == reproduction["reference_sha256"], "V5 H-SE5 reference SHA drift")
    sources = body["source_binding"]
    names = tuple(sources["sessions"])
    need(len(names) == len(set(names)) == 13 and tuple(sources["dates"]) == DATES, "V5 source allowlist/date drift")
    for row in sources["files"]:
        need(file_sha(Path(row["path"])) == row["sha256"], f"V5 source SHA drift: {row['session']}")

    passing: list[str] = []
    poisson_reports: dict[str, Any] = {}
    for budget in (3, 4):
        item = body["budgets"][f"M{budget}"]
        candidates = item["candidates"]
        need(set(candidates) == set(CANDIDATES), f"V5 M{budget} candidate set drift")
        need(set(item["endpoint_maps"]) == set(DATES) and set(item["source_eb_priors"]) == set(DATES),
             f"V5 M{budget} map/prior set drift")
        baseline = candidates["hse5_pca_delta_q4"]["sessions"]
        need(tuple(baseline) == names, f"V5 M{budget} session order drift")
        for candidate_name in CANDIDATES:
            candidate = candidates[candidate_name]
            spec = candidate["spec"]
            need(spec["carrier_dim"] == 5 and spec["target_session_optimizer_steps"] == 0
                 and spec["target_session_backward_steps"] == 0 and spec["reads_only_endpoint_positions_event_timestamps_and_spike_times"] is True,
                 "V5 candidate scope/width drift")
            rows = candidate["sessions"]
            need(tuple(rows) == names, f"V5 M{budget}/{candidate_name} session order drift")
            aggregate = candidate["aggregate"]
            gate = candidate["gate"]
            if aggregate.get("status") == "undefined_nonconverged_poisson_fits":
                need(candidate_name == "poisson_exposure_irls", "V5 only Poisson may be undefined")
                verify_poisson_fail_closed(rows, aggregate)
                need(gate["passed"] is False and gate["material_gain_vs_hse5"] is False
                     and gate.get("undefined_reason") == "undefined_nonconverged_poisson_fits",
                     f"V5 M{budget} Poisson undefined gate drift")
                poisson_reports[f"M{budget}"] = {
                    **poisson_convergence_summary(rows),
                    "receipt_aggregate_status": aggregate["status"],
                    "scoring_permitted": False,
                }
                continue
            fields = {
                "correct_r2": [(name, float(rows[name]["median_r2_correct"])) for name in names],
                "correct_minus_hse5": [(name, float(rows[name]["median_r2_correct"] - baseline[name]["median_r2_correct"])) for name in names],
                "correct_minus_label_shuffle": [(name, float(rows[name]["median_delta_label_shuffle"])) for name in names],
                "correct_minus_intercept": [(name, float(rows[name]["median_delta_intercept"])) for name in names],
                "correct_minus_row_shuffle": [(name, float(rows[name]["median_delta_row_shuffle"])) for name in names],
            }
            for key, values in fields.items():
                verify_summary(aggregate[key], summary(values))
            material = bool(positive(aggregate["correct_minus_hse5"])
                            and aggregate["correct_minus_hse5"]["mean"] >= 0.02
                            and aggregate["correct_minus_hse5"]["median"] >= 0.01)
            expected_gate = bool(candidate_name != "hse5_pca_delta_q4" and material
                                 and positive(aggregate["correct_minus_label_shuffle"])
                                 and positive(aggregate["correct_minus_intercept"])
                                 and positive(aggregate["correct_minus_row_shuffle"]))
            need(gate["material_gain_vs_hse5"] is material and gate["passed"] is expected_gate,
                 f"V5 M{budget}/{candidate_name} gate arithmetic drift")
            if candidate_name == "poisson_exposure_irls":
                report = poisson_convergence_summary(rows)
                # If a fixed-iteration fit is not converged, it must have
                # entered the branch above and been marked undefined; a scored
                # Poisson result is therefore legal only at 13/13 in both arms.
                need(report["correct_fit"]["nonconverged_sessions"] == 0
                     and report["label_shuffle_fit"]["nonconverged_sessions"] == 0,
                     f"V5 M{budget} scored a nonconverged Poisson fit")
                poisson_reports[f"M{budget}"] = {
                    **report,
                    "receipt_aggregate_status": "defined",
                    "scoring_permitted": True,
                }
        for candidate_name in CANDIDATES[1:]:
            if candidates[candidate_name]["gate"]["passed"]:
                passing.append(f"M{budget}:{candidate_name}")
    both = [name for name in CANDIDATES[1:] if all(
        body["budgets"][f"M{budget}"]["candidates"][name]["gate"]["passed"] for budget in (3, 4)
    )]
    need(body["passing_candidates"] == both and (body["selected_candidate"] is None) == (not both),
         "V5 cross-budget selection drift")
    expected_status = "PASS_CPU_ESTIMATOR_CANDIDATE_SELECTED" if both else "STOP_CPU_ESTIMATOR_CANDIDATES_NOT_MATERIAL"
    need(body["status"] == expected_status and body["gpu_authorized_by_this_screen"] is False, "V5 terminal status drift")
    return {"status": "PASS", "receipt": str(resolved), "receipt_sha256": receipt_sha,
            "terminal_status": body["status"], "cross_budget_passing_candidates": both,
            "single_budget_passes": passing,
            "superseded_v5r1_receipt": str(old),
            "poisson_irls_convergence": poisson_reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
