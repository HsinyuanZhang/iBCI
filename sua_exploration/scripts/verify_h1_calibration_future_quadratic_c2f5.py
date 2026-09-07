#!/usr/bin/env python3
"""Independent scope, polynomial-design, reproduction, and gate verifier for QC2F5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "h1_calibration_future_quadratic_c2f5_source_screen_v1"
PROTOCOL = "h1_qc2f5_fixed_degree2_residual_operator_source_lodo_20260812_v1"
DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")
GRID = (0.01, 0.1, 1.0, 10.0)
FIELDS = ("delta_qc2f5_minus_hse5", "delta_qc2f5_minus_linear_c2f5", "delta_qc2f5_minus_label_shuffled",
          "delta_qc2f5_minus_source_teacher_shuffled", "delta_qc2f5_minus_quality_only", "delta_qc2f5_minus_intercept")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: Any, right: Any, tolerance: float = 1e-12) -> None:
    need(math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0), f"QC2F5 numeric mismatch {left!r} vs {right!r}")


def summary(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    values = np.asarray([value for _name, value in rows], dtype=np.float64)
    need(values.size == 13 and np.isfinite(values).all(), "QC2F5 needs 13 finite session deltas")
    removed = int(np.argmax(np.abs(values)))
    return {"defined_sessions": 13, "mean": float(values.mean()), "median": float(np.median(values)),
            "positive": int(np.sum(values > 0)), "zero": int(np.sum(values == 0)), "negative": int(np.sum(values < 0)),
            "leave_largest_absolute_out_mean": float(np.delete(values, removed).mean()), "removed_session": rows[removed][0]}


def positive(item: Mapping[str, Any]) -> bool:
    return bool(item["defined_sessions"] == 13 and item["mean"] > 0 and item["median"] > 0 and item["positive"] >= 10 and item["leave_largest_absolute_out_mean"] > 0)


def verify_summary(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        need(observed[key] == expected[key], f"QC2F5 aggregate drift {key}")
    for key in ("mean", "median", "leave_largest_absolute_out_mean"):
        close(observed[key], expected[key])


def feature_width(dim: int) -> int:
    return dim + dim + dim * (dim - 1) // 2


def expected_lambda(rows: Sequence[Mapping[str, Any]]) -> float:
    need(tuple(float(row["ridge_lambda"]) for row in rows) == GRID, "QC2F5 grid/order drift")
    best = max(float(row["mean_score"]) for row in rows)
    # Predeclared strongest-ridge tie rule.
    return max(float(row["ridge_lambda"]) for row in rows if float(row["mean_score"]) >= best - 1e-12)


def verify(path: Path) -> dict[str, Any]:
    receipt = path.resolve()
    need(receipt.is_file() and stat.S_IMODE(receipt.stat().st_mode) == 0o444, "QC2F5 receipt must be 0444")
    body = json.loads(receipt.read_text(encoding="utf-8"))
    need(body.get("schema") == SCHEMA and body.get("protocol") == PROTOCOL, "QC2F5 schema/protocol drift")
    pre = body["predeclaration"]
    pre_path = Path(pre["path"]).resolve()
    need(pre_path.is_file() and stat.S_IMODE(pre_path.stat().st_mode) == 0o444 and sha(pre_path) == pre["sha256"], "QC2F5 predeclaration binding drift")
    pre_body = json.loads(pre_path.read_text(encoding="utf-8"))
    need(pre_body.get("schema") == "h1_qc2f5_predeclaration_v1" and pre_body.get("written_before_source_nwb_open") is True, "QC2F5 predeclaration contract drift")
    constants = body["frozen_constants"]
    need(constants["rank"] == 4 and constants["carrier_dim"] == 5 and constants["quality_dim"] == 7 and tuple(constants["support_budgets"]) == (3, 4) and tuple(constants["operator_ridge_grid"]) == GRID, "QC2F5 frozen dimensions/grid drift")
    need(constants["channel_index_feature"] is False and constants["target_session_optimizer_steps"] == 0 and constants["target_session_backward_steps"] == 0, "QC2F5 target contract drift")
    expected_scope = {"dense_velocity_opened": False, "within_event_position_trajectory_opened": False, "minival_nwbs_opened": 0, "held_out_nwbs_opened": 0, "formal_test_labels_opened": 0, "decoder_constructed": False, "trainer_constructed": False, "cuda_used": False, "target_session_later_labels_enter_operator_fit": False, "target_session_later_labels_used_only_for_outer_score": True}
    for key, value in expected_scope.items():
        need(body["scope"].get(key) == value, f"QC2F5 scope drift {key}")
    for prefix in ("module", "runner", "event_parser", "hse5_v2"):
        bind = body["implementation_binding"]
        item = Path(bind[f"{prefix}_path"]).resolve()
        need(item.is_file() and sha(item) == bind[f"{prefix}_sha256"], f"QC2F5 source hash drift {prefix}")
    names = tuple(body["source_binding"]["sessions"])
    need(len(names) == len(set(names)) == 13 and tuple(body["source_binding"]["dates"]) == DATES, "QC2F5 source cohort drift")
    for item in body["source_binding"]["files"]:
        need(sha(Path(item["path"])) == item["sha256"], f"QC2F5 input hash drift {item['session']}")
    for reference_key in ("baseline_reproduction", "linear_c2f5_reference"):
        reference = body[reference_key]
        if "reference_path" in reference:
            item, digest = Path(reference["reference_path"]), reference["reference_sha256"]
        else:
            item, digest = Path(reference["path"]), reference["sha256"]
        need(item.is_file() and sha(item) == digest, f"QC2F5 reference hash drift {reference_key}")
    linear_repro = body["linear_c2f5_exact_reproduction"]
    need(linear_repro["passed"] is True and linear_repro["comparisons"] == 128 and linear_repro["maximum_absolute_difference"] <= 1e-10, "QC2F5 exact linear C2F5 reproduction drift")

    budget_passes: list[bool] = []
    for budget in (3, 4):
        item = body["budgets"][f"M{budget}"]
        raw = item["raw_hse5_v2_reproduction"]
        need(raw["passed"] is True and raw["comparisons"] == 39 and raw["maximum_absolute_difference"] <= 1e-10, f"QC2F5 M{budget} H-SE5 reproduction drift")
        rows = item["sessions"]
        need(tuple(rows) == names, f"QC2F5 M{budget} session coverage/order drift")
        for name in names:
            row = rows[name]
            need(row["status"] == "defined" and row["target_session_later_labels_enter_operator_fit"] is False and row["row_shuffle"]["fixed_points"] == 0, f"QC2F5 M{budget}/{name} row scope/control drift")
            for field in FIELDS:
                need(math.isfinite(float(row[field])), f"QC2F5 M{budget}/{name} nonfinite {field}")
            for left, right in (("delta_qc2f5_minus_hse5", "median_r2_hse5"), ("delta_qc2f5_minus_linear_c2f5", "median_r2_linear_c2f5"), ("delta_qc2f5_minus_label_shuffled", "median_r2_label_shuffled_qc2f5"), ("delta_qc2f5_minus_source_teacher_shuffled", "median_r2_source_teacher_shuffled_qc2f5"), ("delta_qc2f5_minus_quality_only", "median_r2_quality_only_qc2f5"), ("delta_qc2f5_minus_intercept", "median_r2_intercept_only")):
                close(row[left], row["median_r2_qc2f5"] - row[right])
        for date in DATES:
            detail = item["outer_date_details"][date]
            selection = detail["operator_lambda_selection"]
            close(selection["selected_lambda"], expected_lambda(selection["source_date_lodo_rows"]), 0.0)
            for choice in selection["source_date_lodo_rows"]:
                need(choice["outer_date"] == date and tuple(choice["inner_source_dates"]) == tuple(value for value in DATES if value != date), "QC2F5 inner LODO date drift")
                for row in choice["rows"]:
                    need(date not in row["operator_source_dates"] and row["inner_validation_date"] not in row["operator_source_dates"], "QC2F5 inner source leakage")
            for key, dim, kind in (("primary_operator", 12, "carrier_plus_quality"), ("quality_only_operator", 7, "quality_only_no_carrier"), ("source_teacher_shuffle_operator", 12, "carrier_plus_quality_source_teacher_pairing_shuffled")):
                operator = detail[key]
                need(operator["input_kind"] == kind and operator["raw_input_dim"] == dim and operator["polynomial_feature_dim"] == feature_width(dim) and operator["design_width_with_intercept"] == feature_width(dim) + 1 and operator["has_channel_index_feature"] is False, f"QC2F5 F2 feature width/order drift {budget}/{date}/{key}")
                names_order = operator["polynomial_feature_order"]
                need(len(names_order) == feature_width(dim) and names_order[:dim] == [f"linear:{value}" for value in operator["raw_feature_order"]], f"QC2F5 polynomial order drift {budget}/{date}/{key}")
            shuffled = detail["source_teacher_pairing_shuffle"]
            need(shuffled["fixed_points"] == 0 and shuffled["teacher_marginal_sha256_before"] == shuffled["teacher_marginal_sha256_after"], "QC2F5 teacher shuffle did not preserve marginal/destroy pairing")
        expected = {"correct_minus_hse5": summary([(name, rows[name]["delta_qc2f5_minus_hse5"]) for name in names]),
                    "correct_minus_linear_c2f5": summary([(name, rows[name]["delta_qc2f5_minus_linear_c2f5"]) for name in names]),
                    "correct_minus_label_shuffled": summary([(name, rows[name]["delta_qc2f5_minus_label_shuffled"]) for name in names]),
                    "correct_minus_source_teacher_shuffled": summary([(name, rows[name]["delta_qc2f5_minus_source_teacher_shuffled"]) for name in names]),
                    "correct_minus_quality_only": summary([(name, rows[name]["delta_qc2f5_minus_quality_only"]) for name in names]),
                    "correct_minus_intercept": summary([(name, rows[name]["delta_qc2f5_minus_intercept"]) for name in names])}
        gate = item["gate"]
        for key, value in expected.items():
            verify_summary(gate[key], value)
        material = positive(expected["correct_minus_hse5"]) and expected["correct_minus_hse5"]["mean"] >= .02 and expected["correct_minus_hse5"]["median"] >= .01
        controls = {key: positive(value) for key, value in expected.items() if key != "correct_minus_hse5"}
        need(gate["material_gain_vs_hse5"] is material and gate["controls"] == controls and gate["passed"] is bool(material and all(controls.values())), f"QC2F5 M{budget} gate recompute drift")
        budget_passes.append(bool(gate["passed"]))
    expected_status = "PASS_CPU_QC2F5_MATERIAL" if all(budget_passes) else "STOP_CPU_QC2F5_NOT_MATERIAL"
    need(body["status"] == expected_status and body["passing_candidate"] is all(budget_passes) and body["gpu_authorized_by_this_screen"] is False, "QC2F5 terminal status drift")
    return {"status": "PASS", "receipt": str(receipt), "receipt_sha256": sha(receipt), "terminal_status": body["status"], "M3_gate": budget_passes[0], "M4_gate": budget_passes[1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(verify(parser.parse_args().receipt), indent=2, sort_keys=True))
