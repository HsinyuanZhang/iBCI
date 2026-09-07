#!/usr/bin/env python3
"""Receipt-only five-date aggregate and frozen gate for H1-EST4.

The program opens no recording, checkpoint, model, Trainer, or CUDA context.
It consumes exactly the five canonical six-arm evaluation receipts, reports
all date/session rows, and applies only the predeclared EST4 gates.  It never
selects CI64 and never consumes H-LS as an automatic route selector.
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
ARMS = ("B-C", "B-LS", "L-C", "L-C0", "L-LS", "L-RS")
EVALUATION_SCHEMA = "h1_carrierid_date_lodo_est4_six_arm_terminal_evaluation_v1"
AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_est4_fivedate_terminal_aggregate_v1"
AGGREGATE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_EST4_FIVEDATE_AGGREGATED_WITH_FROZEN_GATE"
EVALUATION_DIR = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_est4" / "terminal_evaluations"
DEFAULT_OUTPUT = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_est4" / "H1_CARRIERID_DATE_LODO_EST4_FIVEDATE_TERMINAL_AGGREGATE_v1.json"


class Est4FiveDateAggregateError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4FiveDateAggregateError(message)


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
    return directory / f"H1_CARRIERID_DATE_LODO_EST4_{date}_SIX_ARM_TERMINAL_EVALUATION_v1.json"


def _read(date: str, directory: Path) -> tuple[Path, dict[str, Any]]:
    path = _path(date, directory).resolve()
    _need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444,
          f"{date}: immutable canonical EST4 evaluation receipt missing")
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == EVALUATION_SCHEMA
          and body.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_EST4_{date}_SIX_ARM_EVALUATED"
          and body.get("outer_date") == date,
          f"{date}: EST4 evaluation schema/status/date drift")
    _need(body.get("one_shot", {}).get("canonical_output_path") == str(path)
          and body.get("one_shot", {}).get("same_date_prior_terminal_evaluation_receipts") == 0,
          f"{date}: EST4 evaluation is not canonical one-shot evidence")
    _need(body.get("deployment_updates") == {
        "optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True,
    }, f"{date}: EST4 evaluation records deployment updates")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("formal_heldout_opened") is False
          and scope.get("minival_opened") is False and scope.get("evalai_opened") is False,
          f"{date}: EST4 evaluation scope exceeds development held-source-date endpoint")
    target, metrics = body.get("target"), body.get("metrics")
    _need(isinstance(target, Mapping) and isinstance(metrics, Mapping) and set(metrics) == set(ARMS),
          f"{date}: EST4 target/metrics arm contract drift")
    shared_hash = target.get("shared_query_window_indices_sha256")
    sessions = tuple(target.get("sessions", ()))
    _need(sessions, f"{date}: EST4 target sessions missing")
    for arm in ARMS:
        row = metrics[arm]
        _need(isinstance(row, Mapping) and row.get("query_window_indices_sha256") == shared_hash
              and row.get("state_immutable") is True
              and tuple(row.get("per_session", {})) == sessions,
              f"{date}: EST4 {arm} metric/window/state contract drift")
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
        "l_c_minus_b_c": arrays["L-C"] - arrays["B-C"],
        "l_c_minus_l_ls": arrays["L-C"] - arrays["L-LS"],
        "b_c_minus_b_ls": arrays["B-C"] - arrays["B-LS"],
        "l_c_minus_l_c0": arrays["L-C"] - arrays["L-C0"],
        "l_c_minus_l_rs": arrays["L-C"] - arrays["L-RS"],
    }
    summaries = {name: _summary(values) for name, values in deltas.items()}
    criteria = {
        "learned_beats_fresh_baseline": (
            summaries["l_c_minus_b_c"]["mean"] > 0.0
            and summaries["l_c_minus_b_c"]["positive_date_count"] >= 4
        ),
        "learned_behavioral_contrast_exceeds_ordinary": (
            summaries["l_c_minus_l_ls"]["mean"] > summaries["b_c_minus_b_ls"]["mean"]
        ),
        "learned_correct_beats_zero_carrier": summaries["l_c_minus_l_c0"]["mean"] > 0.0,
        "learned_correct_beats_row_shuffle": summaries["l_c_minus_l_rs"]["mean"] > 0.0,
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
            "decision": "EST4_WHOLE_PIPELINE_MECHANISM_PASS" if passed else "STOP_ESTIMATOR_ROUTE",
            "rules": [
                "mean_date(L-C - B-C) > 0 and at least 4/5 dates positive",
                "mean_date(L-C - L-LS) > mean_date(B-C - B-LS)",
                "mean_date(L-C - L-C0) > 0",
                "mean_date(L-C - L-RS) > 0",
            ],
        },
        "route_boundary": {
            "H_LS_automatic_route_selection": "FORBIDDEN",
            "CI64_selected_or_rejected_by_this_receipt": False,
            "parameter_mechanism_claim": "BLOCKED_BY_EST4_GAUGE; whole-pipeline prediction only",
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
