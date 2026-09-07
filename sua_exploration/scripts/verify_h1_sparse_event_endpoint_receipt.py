#!/usr/bin/env python3
"""Independently verify a frozen H1 sparse-event endpoint source receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_SCHEMA = "h1_sparse_event_endpoint_source_audit_v1"
EXPECTED_PROTOCOL = "h1_sparse_event_endpoint_carrier_20260811_v1"
EXPECTED_SESSIONS = (
    "ses-19250101T111740", "ses-19250101T112404", "ses-19250108T110520",
    "ses-19250108T111022", "ses-19250108T111455", "ses-19250113T120811",
    "ses-19250113T121303", "ses-19250115T110633", "ses-19250115T111328",
    "ses-19250119T113543", "ses-19250119T114045", "ses-19250120T115044",
    "ses-19250120T115537",
)
EXPECTED_DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")


class ReceiptVerificationError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptVerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(rows: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [(name, float(row["forward"][key])) for name, row in rows.items()]
    _need(len(values) == 13, "paired summary does not contain 13 sessions")
    numbers = [value for _, value in values]
    ordered = sorted(numbers)
    mean = sum(numbers) / len(numbers)
    median = ordered[len(ordered) // 2]
    remove = max(range(len(numbers)), key=lambda index: abs(numbers[index]))
    kept = numbers[:remove] + numbers[remove + 1 :]
    return {
        "defined_sessions": 13,
        "mean": mean,
        "median": median,
        "positive": sum(value > 0 for value in numbers),
        "zero": sum(value == 0 for value in numbers),
        "negative": sum(value < 0 for value in numbers),
        "leave_largest_absolute_out_mean": sum(kept) / len(kept),
        "removed_session": values[remove][0],
    }


def _close(first: float, second: float, tolerance: float = 1.0e-12) -> bool:
    return abs(float(first) - float(second)) <= tolerance


def _verify_budget(name: str, body: Mapping[str, Any]) -> bool:
    budget = int(name[1:])
    _need(budget in (3, 4) and body["budget_trials"] == budget, f"{name}: budget drift")
    bases = body["basis_by_outer_date"]
    _need(tuple(bases) == EXPECTED_DATES, f"{name}: outer-date basis roster drift")
    rows = body["sessions"]
    _need(tuple(rows) == EXPECTED_SESSIONS, f"{name}: session roster drift")
    for session, row in rows.items():
        _need(row["session"] == session and row["budget"] == budget, f"{name}/{session}: binding drift")
        _need(row["support_events"] >= 8 and row["design_rank"] == 4, f"{name}/{session}: carrier undefined")
        _need(row["forward"]["status"] == "defined", f"{name}/{session}: forward transfer undefined")
        _need(row["position_source"]["dense_velocity_series_opened"] is False,
              f"{name}/{session}: dense-velocity access claimed")
        accounting = row["label_accounting"]
        _need(accounting["acquisition_endpoint_position_scalars"] == 14 * row["support_events"],
              f"{name}/{session}: endpoint scalar accounting drift")
        _need(accounting["derived_displacement_scalars"] == 7 * row["support_events"],
              f"{name}/{session}: displacement scalar accounting drift")
        _need(accounting["projected_model_input_scalars"] == 3 * row["support_events"],
              f"{name}/{session}: projected scalar accounting drift")
        _need(row["forward"]["shuffled_fit"]["shuffle"]["fixed_points"] == 0,
              f"{name}/{session}: endpoint shuffle has fixed points")
        _need(row["row_shuffle"]["fixed_points"] == 0, f"{name}/{session}: row shuffle has fixed points")

    shuffle = _summary(rows, "median_delta_shuffle")
    intercept = _summary(rows, "median_delta_intercept")
    stored_shuffle = body["aggregate"]["correct_minus_shuffle"]
    stored_intercept = body["aggregate"]["correct_minus_intercept"]
    for key in ("mean", "median", "leave_largest_absolute_out_mean"):
        _need(_close(shuffle[key], stored_shuffle[key]), f"{name}: shuffle {key} drift")
        _need(_close(intercept[key], stored_intercept[key]), f"{name}: intercept {key} drift")
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        _need(shuffle[key] == stored_shuffle[key], f"{name}: shuffle {key} mismatch")
        _need(intercept[key] == stored_intercept[key], f"{name}: intercept {key} mismatch")

    retained = [float(bases[date]["retained_variance"]) for date in EXPECTED_DATES]
    sorted_retained = sorted(retained)
    retained_median = (sorted_retained[2] + sorted_retained[3]) / 2.0
    defined_gate = all(
        row["support_events"] >= 8 and row["design_rank"] == 4 and row["forward"]["status"] == "defined"
        for row in rows.values()
    )
    basis_gate = min(retained) >= 0.50 and retained_median >= 0.65
    shuffle_gate = (
        shuffle["mean"] > 0 and shuffle["median"] > 0 and shuffle["positive"] >= 8
        and shuffle["leave_largest_absolute_out_mean"] > 0
    )
    intercept_gate = intercept["median"] > 0 and intercept["positive"] >= 7
    passed = defined_gate and basis_gate and shuffle_gate and intercept_gate
    gate = body["gpu_entrance_gate"]
    _need(gate["defined_carriers_all_13"] == defined_gate, f"{name}: defined gate drift")
    _need(gate["basis_retained_variance_gate"] == basis_gate, f"{name}: basis gate drift")
    _need(gate["correct_minus_shuffle_gate"] == shuffle_gate, f"{name}: shuffle gate drift")
    _need(gate["correct_minus_intercept_gate"] == intercept_gate, f"{name}: intercept gate drift")
    _need(gate["passed"] == passed, f"{name}: aggregate gate drift")
    return passed


def verify(path: Path) -> dict[str, Any]:
    receipt_path = path.resolve()
    _need(receipt_path.is_file(), f"missing receipt {receipt_path}")
    body = json.loads(receipt_path.read_text(encoding="utf-8"))
    _need(body["schema"] == EXPECTED_SCHEMA and body["protocol"] == EXPECTED_PROTOCOL,
          "receipt schema/protocol mismatch")
    scope = body["scope"]
    _need(scope == {
        "public_held_in_calibration_nwbs_opened": 13,
        "minival_nwbs_opened": 0,
        "held_out_nwbs_opened": 0,
        "formal_test_labels_opened": 0,
        "dense_velocity_series_opened": False,
        "decoder_constructed": False,
        "trainer_constructed": False,
        "cuda_used": False,
    }, "scope boundary drift")
    source = body["source_binding"]
    _need(tuple(source["sessions"]) == EXPECTED_SESSIONS and tuple(source["dates"]) == EXPECTED_DATES,
          "source roster drift")
    _need(len(source["files"]) == 13, "source file count drift")
    for row in source["files"]:
        file_path = Path(row["path"])
        _need(file_path.is_file() and sha256_file(file_path) == row["sha256"], f"source SHA drift: {file_path}")
    binding = body["implementation_binding"]
    for prefix in ("protocol", "implementation", "runner"):
        bound_path = Path(binding[f"{prefix}_path"])
        _need(bound_path.is_file() and sha256_file(bound_path) == binding[f"{prefix}_sha256"],
              f"{prefix} binding SHA drift")
    passed = {name: _verify_budget(name, body["budgets"][name]) for name in ("M3", "M4")}
    passing = [budget for budget in (3, 4) if passed[f"M{budget}"]]
    selected = 3 if 3 in passing else (4 if 4 in passing else None)
    _need(body["passing_budgets"] == passing and body["selected_gpu_budget"] == selected,
          "selected GPU budget drift")
    expected_status = (
        "PASS_CPU_HSE4_M3_GPU_ENTRANCE_READY" if selected == 3
        else "PASS_CPU_HSE4_M4_DEVELOPMENT_GPU_ENTRANCE_READY" if selected == 4
        else "STOP_CPU_HSE4_GPU_ENTRANCE_GATE_FAILED"
    )
    _need(body["status"] == expected_status, "receipt status drift")
    sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    _need(sidecar.is_file(), "receipt SHA sidecar missing")
    claimed = sidecar.read_text(encoding="ascii").split()[0]
    digest = sha256_file(receipt_path)
    _need(claimed == digest, "receipt sidecar SHA mismatch")
    return {"status": "PASS", "receipt": str(receipt_path), "sha256": digest, "selected_gpu_budget": selected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
