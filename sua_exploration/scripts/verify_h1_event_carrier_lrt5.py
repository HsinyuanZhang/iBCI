#!/usr/bin/env python3
"""Independent structural/arithmetic verifier for an immutable H1 LRT5 receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "h1_event_carrier_lrt5_source_screen_v2"
PROTOCOL = "h1_event_carrier_low_rank_channel_tuning_source_screen_20260812_v2"
PREVIOUS_RECEIPT_SHA256 = "f19e333dc1a57830593d380778f5bef10cf41a5bb1ecd7717c21f386ae25bc3a"
DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")
RANK_GRID = (1, 2, 4, 8)
LAMBDA_GRID = (0.1, 1.0, 3.0, 10.0)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> None:
    _need(left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance),
          f"LRT5 numeric mismatch {left!r} vs {right!r}")


def _summary(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    values = np.asarray([float(value) for _name, value in rows], dtype=np.float64)
    _need(values.size == 13 and np.isfinite(values).all(), "LRT5 summary needs 13 finite sessions")
    removed = int(np.argmax(np.abs(values)))
    return {
        "defined_sessions": 13, "mean": float(values.mean()), "median": float(np.median(values)),
        "positive": int(np.sum(values > 0)), "zero": int(np.sum(values == 0)), "negative": int(np.sum(values < 0)),
        "leave_largest_absolute_out_mean": float(np.delete(values, removed).mean()),
        "removed_session": rows[removed][0],
    }


def _verify_summary(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        _need(observed.get(key) == expected.get(key), f"LRT5 summary drift at {key}")
    for key in ("mean", "median", "leave_largest_absolute_out_mean"):
        _close(observed.get(key), expected.get(key))


def _positive(value: Mapping[str, Any]) -> bool:
    return bool(value["defined_sessions"] == 13 and value["mean"] > 0 and value["median"] > 0
                and value["positive"] >= 10 and value["leave_largest_absolute_out_mean"] > 0)


def _verify_selection(outer_date: str, selection: Mapping[str, Any], session_names: Sequence[str]) -> None:
    _need(selection["outer_date"] == outer_date and selection["budget"] in (3, 4), "LRT5 selection date/budget drift")
    _need(selection["selected_rank"] in RANK_GRID and float(selection["selected_ridge_lambda"]) in LAMBDA_GRID,
          "LRT5 selected hyperparameter outside literal grid")
    grid = selection["grid"]
    _need(len(grid) == 16, "LRT5 grid must be literal 4x4")
    selected_rows: list[Mapping[str, Any]] = []
    for rank in RANK_GRID:
        for lam in LAMBDA_GRID:
            key = f"r{rank}_lambda{lam:g}"
            row = grid.get(key)
            _need(row is not None and row["rank"] == rank and float(row["ridge_lambda"]) == lam,
                  f"LRT5 candidate missing/drift {key}")
            summary = row["inner_source_date_lodo_median_r2"]
            _need(summary["defined_sessions"] == len(session_names) - sum(name.startswith(f"ses-{outer_date}") for name in session_names),
                  "LRT5 inner selection score count drift")
            _need(len(row["inner_folds"]) == 5, "LRT5 inner source-date fold count drift")
            for inner_date, fold in row["inner_folds"].items():
                _need(inner_date != outer_date, "LRT5 outer date entered inner fold")
                train = tuple(fold["train_source_sessions"])
                valid = tuple(fold["validation_source_sessions"])
                _need(train and valid and all(not name.startswith(f"ses-{outer_date}") for name in train + valid),
                      "LRT5 outer date leaked into selection")
                _need(all(name.startswith(f"ses-{inner_date}") for name in valid),
                      "LRT5 inner validation date drift")
                _need(set(train).isdisjoint(valid), "LRT5 inner source train/validation overlap")
                _need(set(train + valid) == {name for name in session_names if not name.startswith(f"ses-{outer_date}")},
                      "LRT5 inner source session cover drift")
            selected_rows.append(row)
    expected = min(selected_rows, key=lambda row: (
        -float(row["inner_source_date_lodo_median_r2"]["mean"]),
        -float(row["inner_source_date_lodo_median_r2"]["median"]),
        int(row["rank"]), float(row["ridge_lambda"]),
    ))
    _need(int(selection["selected_rank"]) == int(expected["rank"])
          and float(selection["selected_ridge_lambda"]) == float(expected["ridge_lambda"]),
          "LRT5 source-only hyperparameter selection drift")


def verify(path: Path) -> dict[str, Any]:
    receipt = path.resolve()
    _need(receipt.is_file() and stat.S_IMODE(receipt.stat().st_mode) == 0o444, "LRT5 receipt must be immutable mode 0444")
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    _need(sidecar.is_file() and stat.S_IMODE(sidecar.stat().st_mode) == 0o444, "LRT5 SHA sidecar must be immutable")
    digest = _sha(receipt)
    _need(sidecar.read_text(encoding="ascii").strip() == f"{digest}  {receipt.name}", "LRT5 SHA sidecar mismatch")
    body = json.loads(receipt.read_text(encoding="utf-8"))
    _need(body.get("schema") == SCHEMA and body.get("protocol") == PROTOCOL, "LRT5 schema/protocol mismatch")
    _need(body.get("candidate_matrix_predeclared_before_data_run") is True, "LRT5 predeclaration missing")
    supersedes = body["supersedes"]
    previous = Path(supersedes["previous_receipt_path"]).resolve()
    _need(supersedes["previous_receipt_sha256"] == PREVIOUS_RECEIPT_SHA256
          and supersedes["previous_receipt_preserved_immutable"] is True
          and supersedes["candidate_grid_unchanged"] is True
          and supersedes["correct_lrt5_arm_unchanged"] is True,
          "LRT5 r1 supersession contract drift")
    _need(previous.is_file() and stat.S_IMODE(previous.stat().st_mode) == 0o444
          and _sha(previous) == PREVIOUS_RECEIPT_SHA256,
          "LRT5 r1 immutable receipt was not preserved exactly")
    _need("mean(Y)-mean(Z)@W" in supersedes["correction"], "LRT5 mean-alignment correction missing")
    constants = body["frozen_constants"]
    _need(constants["carrier_dim"] == 5 and tuple(constants["rank_grid"]) == RANK_GRID
          and tuple(float(value) for value in constants["target_ridge_grid"]) == LAMBDA_GRID,
          "LRT5 fixed candidate grid/width drift")
    _need(tuple(constants["support_budgets"]) == (3, 4) and constants["source_slope_ridge_lambda"] == 3.0,
          "LRT5 budget/source ridge drift")
    _need("never added" in constants["correct_slope_form"], "LRT5 zero-mean source carrier contract missing")

    scope = body["scope"]
    expected_scope = {
        "public_held_in_calibration_nwbs_opened": 13, "minival_nwbs_opened": 0,
        "held_out_nwbs_opened": 0, "formal_test_labels_opened": 0, "dense_velocity_opened": False,
        "target_session_optimizer_steps": 0, "target_session_backward_steps": 0,
        "decoder_constructed": False, "trainer_constructed": False, "cuda_used": False,
        "native_position_endpoints_per_event": 2, "native_event_timestamps_per_event": 2,
        "within_event_position_trajectory_opened": False,
    }
    for key, expected in expected_scope.items():
        _need(scope.get(key) == expected, f"LRT5 scope drift at {key}")
    binding = body["implementation_binding"]
    for prefix in ("module", "runner", "event_parser", "hse5"):
        bound = Path(binding[f"{prefix}_path"]).resolve()
        _need(_sha(bound) == binding[f"{prefix}_sha256"], f"LRT5 source binding drift: {prefix}")
    source = body["source_binding"]
    names = tuple(source["sessions"])
    _need(len(names) == len(set(names)) == 13 and tuple(source["dates"]) == DATES, "LRT5 source allowlist/date drift")
    for row in source["files"]:
        _need(_sha(Path(row["path"])) == row["sha256"], f"LRT5 input SHA drift: {row['session']}")
    reproduction = body["baseline_reproduction"]
    _need(reproduction["passed"] is True and reproduction["comparisons"] == 78
          and float(reproduction["maximum_absolute_difference"]) <= float(reproduction["absolute_tolerance"]),
          "LRT5 H-SE5 exact reproduction drift")
    _need(_sha(Path(reproduction["reference_path"])) == reproduction["reference_sha256"],
          "LRT5 H-SE5 reference SHA drift")

    passed_budgets: list[str] = []
    report: dict[str, Any] = {}
    for budget in (3, 4):
        item = body["budgets"][f"M{budget}"]
        _need(item["budget_trials"] == budget, "LRT5 budget label drift")
        _need(set(item["selection_by_outer_date"]) == set(DATES)
              and set(item["endpoint_map_by_outer_date"]) == set(DATES)
              and set(item["subspace_by_outer_date"]) == set(DATES), "LRT5 outer-date manifest drift")
        for outer_date in DATES:
            _verify_selection(outer_date, item["selection_by_outer_date"][outer_date], names)
            endpoint = item["endpoint_map_by_outer_date"][outer_date]
            subspace = item["subspace_by_outer_date"][outer_date]
            expected_source = tuple(name for name in names if not name.startswith(f"ses-{outer_date}"))
            _need(tuple(endpoint["source_sessions"]) == expected_source
                  and tuple(subspace["source_sessions"]) == expected_source,
                  "LRT5 outer source map/subspace leak")
            _need(endpoint["retained_variance"] > 0 and subspace["rank"] in RANK_GRID
                  and subspace["deployable_correct_carrier_uses_source_slope_mean"] is False,
                  "LRT5 endpoint/subspace manifest drift")
        rows = item["lrt5_sessions"]
        baseline = item["hse5_baseline_sessions"]
        _need(tuple(rows) == names and tuple(baseline) == names, "LRT5 session order drift")
        for name in names:
            row = rows[name]
            _need(row["date"] == name[4:].split("T", 1)[0] and row["carrier_dim"] == 5
                  and row["subspace_rank"] in RANK_GRID and row["target_ridge_lambda"] in LAMBDA_GRID,
                  "LRT5 row dimension/grid drift")
            _need(row["centered_design_rank"] == 4 and row["outer_future_used_only_for_scoring"] is True,
                  "LRT5 target fitting/outer scoring drift")
            _need(row["correct_fit"]["target_pairing_required"] is True
                  and row["correct_fit"]["target_optimizer_steps"] == 0
                  and row["correct_fit"]["target_backward_steps"] == 0,
                  "LRT5 correct pairing/closed-form manifest drift")
            _need(row["label_shuffle"]["fixed_points"] == 0 and row["u_row_shuffle"]["fixed_points"] == 0,
                  "LRT5 control permutation has fixed point")
        aggregate = item["aggregate"]
        fields = {
            "correct_r2": [(name, rows[name]["median_r2_correct"]) for name in names],
            "correct_minus_hse5": [(name, rows[name]["median_r2_correct"] - baseline[name]["median_r2_correct"]) for name in names],
            "correct_minus_label_shuffle": [(name, rows[name]["median_delta_label_shuffle"]) for name in names],
            "correct_minus_u_row_shuffle": [(name, rows[name]["median_delta_u_row_shuffle"]) for name in names],
            "correct_minus_intercept": [(name, rows[name]["median_delta_intercept"]) for name in names],
            "correct_minus_source_prior": [(name, rows[name]["median_delta_source_prior"]) for name in names],
        }
        for field, values in fields.items():
            _verify_summary(aggregate[field], _summary(values))
        delta = aggregate["correct_minus_hse5"]
        material = bool(_positive(delta) and delta["mean"] >= 0.02 and delta["median"] >= 0.01)
        controls = {field: _positive(aggregate[field]) for field in (
            "correct_minus_label_shuffle", "correct_minus_u_row_shuffle", "correct_minus_intercept", "correct_minus_source_prior",
        )}
        expected_gate = bool(material and all(controls.values()))
        gate = item["gate"]
        _need(gate["material_gain_vs_hse5"] is material and gate["passed"] is expected_gate,
              "LRT5 gate arithmetic drift")
        for key, expected in controls.items():
            _need(gate[key] is expected, f"LRT5 gate control drift at {key}")
        if expected_gate:
            passed_budgets.append(f"M{budget}")
        report[f"M{budget}"] = {
            "gate": gate, "correct_minus_hse5": aggregate["correct_minus_hse5"],
            "correct_minus_label_shuffle": aggregate["correct_minus_label_shuffle"],
            "correct_minus_u_row_shuffle": aggregate["correct_minus_u_row_shuffle"],
            "correct_minus_source_prior": aggregate["correct_minus_source_prior"],
        }
    expected_status = "PASS_CPU_LRT5_MATERIAL" if len(passed_budgets) == 2 else "STOP_CPU_LRT5_NOT_MATERIAL"
    _need(body["status"] == expected_status and body["gpu_authorized_by_this_screen"] is False,
          "LRT5 terminal status/GPU discipline drift")
    return {"status": "PASS", "receipt": str(receipt), "receipt_sha256": digest,
            "terminal_status": body["status"], "passing_budgets": passed_budgets, "budget_report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
