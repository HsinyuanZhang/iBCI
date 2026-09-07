#!/usr/bin/env python3
"""Independent arithmetic, source-scope, and gate verifier for C2F5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "h1_calibration_future_correction_c2f5_source_screen_v1"
PROTOCOL = "h1_c2f5_shared_residual_operator_source_lodo_20260812_v1"
PREDECLARATION_SCHEMA = "h1_c2f5_predeclaration_v1"
DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")
GRID = (0.01, 0.1, 1.0, 10.0)
MATERIAL_MEAN = 0.02
MATERIAL_MEDIAN = 0.01
MIN_POSITIVE = 10


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
    need(left is not None and right is not None and math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0),
         f"C2F5 numeric mismatch: {left!r} vs {right!r}")


def summary(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    values = np.asarray([float(value) for _name, value in rows], dtype=np.float64)
    need(values.size == 13 and np.isfinite(values).all(), "C2F5 aggregate must have exactly 13 finite recordings")
    removed = int(np.argmax(np.abs(values)))
    return {
        "defined_sessions": 13,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive": int(np.sum(values > 0)),
        "zero": int(np.sum(values == 0)),
        "negative": int(np.sum(values < 0)),
        "leave_largest_absolute_out_mean": float(np.delete(values, removed).mean()),
        "removed_session": rows[removed][0],
    }


def verify_summary(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        need(observed.get(key) == expected.get(key), f"C2F5 summary drift at {key}")
    for key in ("mean", "median", "leave_largest_absolute_out_mean"):
        close(observed.get(key), expected.get(key))


def positive(value: Mapping[str, Any]) -> bool:
    return bool(
        value["defined_sessions"] == 13 and value["mean"] > 0 and value["median"] > 0
        and value["positive"] >= MIN_POSITIVE and value["leave_largest_absolute_out_mean"] > 0
    )


def expected_lambda(selection_rows: Sequence[Mapping[str, Any]]) -> float:
    need(tuple(float(row["ridge_lambda"]) for row in selection_rows) == GRID,
         "C2F5 source selection grid/order drift")
    best = max(float(row["mean_score"]) for row in selection_rows)
    return min(float(row["ridge_lambda"]) for row in selection_rows if float(row["mean_score"]) >= best - 1.0e-12)


def verify(path: Path) -> dict[str, Any]:
    receipt = path.resolve()
    need(receipt.is_file() and stat.S_IMODE(receipt.stat().st_mode) == 0o444,
         "C2F5 receipt must exist and be immutable mode 0444")
    body = json.loads(receipt.read_text(encoding="utf-8"))
    need(body.get("schema") == SCHEMA and body.get("protocol") == PROTOCOL, "C2F5 schema/protocol mismatch")
    pre = body["predeclaration"]
    pre_path = Path(pre["path"]).resolve()
    need(pre_path.is_file() and stat.S_IMODE(pre_path.stat().st_mode) == 0o444 and file_sha(pre_path) == pre["sha256"],
         "C2F5 immutable predeclaration SHA/mode drift")
    pre_body = json.loads(pre_path.read_text(encoding="utf-8"))
    need(pre_body.get("schema") == PREDECLARATION_SCHEMA and pre_body.get("protocol") == PROTOCOL
         and pre_body.get("written_before_source_nwb_open") is True, "C2F5 predeclaration contract drift")

    constants = body["frozen_constants"]
    need(constants["rank"] == 4 and constants["carrier_dim"] == 5 and constants["quality_dim"] == 7,
         "C2F5 carrier/quality dimensions drift")
    need(tuple(constants["support_budgets"]) == (3, 4) and tuple(constants["operator_ridge_grid"]) == GRID,
         "C2F5 budget/grid drift")
    need(constants["channel_index_feature"] is False and constants["target_session_optimizer_steps"] == 0
         and constants["target_session_backward_steps"] == 0, "C2F5 deployment contract drift")
    scope = body["scope"]
    expected_scope = {
        "native_position_endpoints_per_event": 2,
        "native_event_timestamps_per_event": 2,
        "dense_velocity_opened": False,
        "within_event_position_trajectory_opened": False,
        "target_session_later_labels_enter_operator_fit": False,
        "target_session_optimizer_steps": 0,
        "target_session_backward_steps": 0,
        "public_held_in_calibration_nwbs_opened": 13,
        "minival_nwbs_opened": 0,
        "held_out_nwbs_opened": 0,
        "formal_test_labels_opened": 0,
        "decoder_constructed": False,
        "trainer_constructed": False,
        "cuda_used": False,
        "target_session_later_labels_used_only_for_outer_score": True,
    }
    for key, value in expected_scope.items():
        need(scope.get(key) == value, f"C2F5 scope drift at {key}")
    for prefix in ("module", "runner", "event_parser", "hse5_v2"):
        binding = body["implementation_binding"]
        bound = Path(binding[f"{prefix}_path"]).resolve()
        need(bound.is_file() and file_sha(bound) == binding[f"{prefix}_sha256"], f"C2F5 {prefix} binding drift")

    sources = body["source_binding"]
    names = tuple(sources["sessions"])
    need(len(names) == len(set(names)) == 13 and tuple(sources["dates"]) == DATES,
         "C2F5 source allowlist/date drift")
    for item in sources["files"]:
        need(file_sha(Path(item["path"])) == item["sha256"], f"C2F5 source input SHA drift: {item['session']}")
    reproduced = body["baseline_reproduction"]
    reference = Path(reproduced["reference_path"]).resolve()
    need(reference.is_file() and file_sha(reference) == reproduced["reference_sha256"], "C2F5 H-SE5 reference drift")

    budget_passes: list[bool] = []
    for budget in (3, 4):
        item = body["budgets"][f"M{budget}"]
        replication = item["raw_hse5_v2_reproduction"]
        need(replication["passed"] is True and replication["comparisons"] == 39
             and float(replication["maximum_absolute_difference"]) <= float(replication["absolute_tolerance"]),
             f"C2F5 M{budget} raw H-SE5 reproduction drift")
        rows = item["sessions"]
        need(tuple(rows) == names, f"C2F5 M{budget} session order/coverage drift")
        for name in names:
            row = rows[name]
            need(row["status"] == "defined" and row["budget"] == budget and row["support_events"] >= 8
                 and row["later_events"] >= 4 and row["defined_channels"] > 0, f"C2F5 M{budget}/{name} event coverage drift")
            need(row["target_session_later_labels_enter_operator_fit"] is False
                 and row["row_shuffle"]["fixed_points"] == 0, f"C2F5 M{budget}/{name} control/scope drift")
            for field in (
                "median_r2_hse5", "median_r2_c2f5", "median_r2_label_shuffled_c2f5",
                "median_r2_row_shuffled_c2f5", "median_r2_quality_only_c2f5", "median_r2_intercept_only",
                "delta_c2f5_minus_hse5", "delta_c2f5_minus_label_shuffled",
                "delta_c2f5_minus_row_shuffled", "delta_c2f5_minus_quality_only", "delta_c2f5_minus_intercept",
            ):
                need(math.isfinite(float(row[field])), f"C2F5 M{budget}/{name} nonfinite {field}")
            close(row["delta_c2f5_minus_hse5"], row["median_r2_c2f5"] - row["median_r2_hse5"])
            close(row["delta_c2f5_minus_label_shuffled"], row["median_r2_c2f5"] - row["median_r2_label_shuffled_c2f5"])
            close(row["delta_c2f5_minus_row_shuffled"], row["median_r2_c2f5"] - row["median_r2_row_shuffled_c2f5"])
            close(row["delta_c2f5_minus_quality_only"], row["median_r2_c2f5"] - row["median_r2_quality_only_c2f5"])
            close(row["delta_c2f5_minus_intercept"], row["median_r2_c2f5"] - row["median_r2_intercept_only"])

        outer = item["outer_date_details"]
        need(set(outer) == set(DATES), f"C2F5 M{budget} outer-date coverage drift")
        for date in DATES:
            detail = outer[date]
            selection = detail["operator_lambda_selection"]
            selected = expected_lambda(selection["source_date_lodo_rows"])
            close(selection["selected_lambda"], selected, tolerance=0.0)
            for operator_name, expected_input_dim, expected_kind in (
                ("primary_operator", 12, "carrier_plus_quality"),
                ("quality_only_operator", 7, "quality_only_no_carrier"),
            ):
                operator = detail[operator_name]
                need(operator["outer_date"] == date and operator["budget"] == budget
                     and operator["input_dim"] == expected_input_dim and operator["input_kind"] == expected_kind
                     and operator["has_channel_index_feature"] is False
                     and operator["target_session_later_labels_enter_operator_fit"] is False,
                     f"C2F5 M{budget}/{date}/{operator_name} operator contract drift")
                need(all(date not in source for source in operator["source_sessions"]),
                     f"C2F5 M{budget}/{date}/{operator_name} outer date leaked")
            teacher = detail["teacher_headroom_and_learnability_diagnostic"]
            need(teacher["role"] == "diagnostic_only_not_a_deployment_or_outer_score"
                 and teacher["source_future_teacher_used_only_for_operator_training"] is True,
                 f"C2F5 M{budget}/{date} teacher diagnostic role drift")
            for lambda_row in selection["source_date_lodo_rows"]:
                need(lambda_row["outer_date"] == date
                     and tuple(lambda_row["inner_source_dates"]) == tuple(item_date for item_date in DATES if item_date != date),
                     f"C2F5 M{budget}/{date} inner LODO date coverage")
                for selection_row in lambda_row["rows"]:
                    need(selection_row["outer_date"] == date
                         and date not in selection_row["operator_source_dates"]
                         and selection_row["inner_validation_date"] not in selection_row["operator_source_dates"],
                         f"C2F5 M{budget}/{date} inner LODO leakage")

        summaries = {
            "correct_minus_hse5": summary([(name, float(rows[name]["delta_c2f5_minus_hse5"])) for name in names]),
            "correct_minus_label_shuffled": summary([(name, float(rows[name]["delta_c2f5_minus_label_shuffled"])) for name in names]),
            "correct_minus_row_shuffled": summary([(name, float(rows[name]["delta_c2f5_minus_row_shuffled"])) for name in names]),
            "correct_minus_quality_only": summary([(name, float(rows[name]["delta_c2f5_minus_quality_only"])) for name in names]),
            "correct_minus_intercept": summary([(name, float(rows[name]["delta_c2f5_minus_intercept"])) for name in names]),
        }
        gate = item["gate"]
        for key, expected in summaries.items():
            verify_summary(gate[key], expected)
        material = bool(positive(summaries["correct_minus_hse5"])
                        and summaries["correct_minus_hse5"]["mean"] >= MATERIAL_MEAN
                        and summaries["correct_minus_hse5"]["median"] >= MATERIAL_MEDIAN)
        expected_controls = {key: positive(summaries[key]) for key in summaries if key != "correct_minus_hse5"}
        need(gate["material_gain_vs_hse5"] is material and gate["controls"] == expected_controls
             and gate["passed"] is bool(material and all(expected_controls.values())),
             f"C2F5 M{budget} gate arithmetic drift")
        budget_passes.append(bool(gate["passed"]))

    expected_terminal = "PASS_CPU_C2F5_MATERIAL" if all(budget_passes) else "STOP_CPU_C2F5_NOT_MATERIAL"
    need(body["passing_candidate"] is all(budget_passes) and body["status"] == expected_terminal
         and body["gpu_authorized_by_this_screen"] is False, "C2F5 terminal status drift")
    return {
        "status": "PASS",
        "receipt": str(receipt),
        "receipt_sha256": file_sha(receipt),
        "terminal_status": body["status"],
        "M3_gate": body["budgets"]["M3"]["gate"]["passed"],
        "M4_gate": body["budgets"]["M4"]["gate"]["passed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
