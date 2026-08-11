#!/usr/bin/env python3
"""Receipt-only five-date aggregate and frozen mechanism gate for H1-CI64.

The program reads only the five canonical CI evaluation receipts.  It reports
all date/session rows and applies the predeclared CI64-vs-CI32/C0/LS gate.  It
does not launch H64; a pass only makes that separately named escalation
eligible.  H-LS and EST4 are neither selected nor rejected here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")
ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")
EVALUATION_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_evaluation_v1"
AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_ci_fivedate_terminal_aggregate_v1"
AGGREGATE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_FIVEDATE_AGGREGATED_WITH_FROZEN_GATE"
EVALUATION_DIR = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_ci" / "terminal_evaluations"
DEFAULT_OUTPUT = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_ci" / "H1_CARRIERID_DATE_LODO_CI_FIVEDATE_TERMINAL_AGGREGATE_v1.json"


class CiFiveDateAggregateError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CiFiveDateAggregateError(message)


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(date: str, directory: Path) -> Path:
    return directory / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"


def _read(date: str, directory: Path) -> tuple[Path, dict[str, Any]]:
    path = _path(date, directory).resolve()
    _need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444,
          f"{date}: immutable canonical CI evaluation receipt missing")
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == EVALUATION_SCHEMA
          and body.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_EVALUATED"
          and body.get("outer_date") == date,
          f"{date}: CI evaluation schema/status/date drift")
    _need(body.get("one_shot", {}).get("canonical_output_path") == str(path)
          and body.get("one_shot", {}).get("same_date_prior_terminal_evaluation_receipts") == 0,
          f"{date}: CI evaluation is not canonical one-shot evidence")
    _need(body.get("deployment_updates") == {
        "optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True,
    }, f"{date}: CI evaluation records deployment updates")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("formal_heldout_opened") is False
          and scope.get("minival_opened") is False and scope.get("evalai_opened") is False,
          f"{date}: CI evaluation scope exceeds development held-source-date endpoint")
    target, metrics = body.get("target"), body.get("metrics")
    _need(isinstance(target, Mapping) and isinstance(metrics, Mapping) and set(metrics) == set(ARMS),
          f"{date}: CI target/metrics arm contract drift")
    shared_hash = target.get("shared_query_window_indices_sha256")
    sessions = tuple(target.get("sessions", ()))
    _need(sessions, f"{date}: CI target sessions missing")
    for arm in ARMS:
        row = metrics[arm]
        _need(isinstance(row, Mapping) and row.get("query_window_indices_sha256") == shared_hash
              and row.get("state_immutable") is True
              and tuple(row.get("per_session", {})) == sessions,
              f"{date}: CI {arm} metric/window/state contract drift")
        _finite(row.get("pooled_r2"), f"{date}/{arm}/pooled_r2")
        for session in sessions:
            _finite(row["per_session"][session].get("r2"), f"{date}/{arm}/{session}/r2")
    return path, body


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "positive_date_count": int(np.count_nonzero(values > 0.0)),
        "negative_date_count": int(np.count_nonzero(values < 0.0)),
        "zero_date_count": int(np.count_nonzero(values == 0.0)),
        "per_date": {date: float(value) for date, value in zip(DATES, values, strict=True)},
    }


def aggregate(*, evaluation_dir: Path = EVALUATION_DIR, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from src.h1_m4_cce_contract import write_immutable_json

    rows: dict[str, Any] = {}
    arm_values = {arm: [] for arm in ARMS}
    recording_rows: list[dict[str, Any]] = []
    for date in DATES:
        path, body = _read(date, Path(evaluation_dir))
        metrics, sessions = body["metrics"], tuple(body["target"]["sessions"])
        pooled = {arm: float(metrics[arm]["pooled_r2"]) for arm in ARMS}
        for arm in ARMS:
            arm_values[arm].append(pooled[arm])
        session_rows: dict[str, Any] = {}
        for session in sessions:
            values = {arm: float(metrics[arm]["per_session"][session]["r2"]) for arm in ARMS}
            session_rows[session] = values
            recording_rows.append({"outer_date": date, "session": session, "r2": values})
        rows[date] = {"receipt": {"path": str(path), "sha256": _sha(path)},
                      "pooled_r2": pooled, "sessions": session_rows}
    arrays = {arm: np.asarray(values, dtype=np.float64) for arm, values in arm_values.items()}
    deltas = {
        "ci64_full_minus_ci32_full": arrays["CI64-FULL"] - arrays["CI32-FULL"],
        "ci64_full_minus_ci64_c0": arrays["CI64-FULL"] - arrays["CI64-C0"],
        "ci64_full_minus_ci64_ls": arrays["CI64-FULL"] - arrays["CI64-LS"],
        "ci64_full_minus_ci64_rs": arrays["CI64-FULL"] - arrays["CI64-RS"],
    }
    summaries = {name: _summary(values) for name, values in deltas.items()}
    required_names = (
        "ci64_full_minus_ci32_full", "ci64_full_minus_ci64_c0", "ci64_full_minus_ci64_ls",
    )
    criteria = {
        name: summaries[name]["mean"] > 0.0 and summaries[name]["positive_date_count"] >= 4
        for name in required_names
    }
    passed = all(criteria.values())
    body = {
        "schema": AGGREGATE_SCHEMA, "status": AGGREGATE_STATUS,
        "inference_unit": "outer_date", "required_outer_dates": list(DATES),
        "all_five_dates_reported": True, "arms": list(ARMS),
        "per_date": rows, "all_recordings": recording_rows,
        "arm_pooled_r2": {arm: _summary(values) for arm, values in arrays.items()},
        "paired_deltas": summaries,
        "frozen_gate": {
            "criteria": criteria, "all_required": True, "passed": passed,
            "decision": "H1_H64_SLODO_ELIGIBLE_NOT_AUTHORIZED" if passed else "STOP_CONSUMER_WIDTH_ROUTE_NO_H64",
            "rules": [
                "mean_date(CI64-FULL - CI32-FULL) > 0 and at least 4/5 dates positive",
                "mean_date(CI64-FULL - CI64-C0) > 0 and at least 4/5 dates positive",
                "mean_date(CI64-FULL - CI64-LS) > 0 and at least 4/5 dates positive",
            ],
            "ci64_full_minus_ci64_rs_reported_but_not_hard_gate": True,
            "H64_launch_authorized": False,
        },
        "route_boundary": {
            "H_LS_automatic_route_selection": "FORBIDDEN",
            "EST4_selected_or_rejected_by_this_receipt": False,
            "checkpoint_date_seed_or_width_selection": "FORBIDDEN",
        },
        "scope": {"receipt_only": True, "nwb_opened": 0, "checkpoint_opened": 0,
                  "trainer_constructed": False, "cuda_constructed": False,
                  "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": AGGREGATE_STATUS, "receipt_path": str(written),
            "receipt_sha256": digest, "decision": body["frozen_gate"]["decision"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(aggregate(evaluation_dir=args.evaluation_dir, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
