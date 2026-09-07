#!/usr/bin/env python3
"""Receipt-only five-date aggregate for the H-C minus H-LS endpoint."""
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
EVALUATION_SCHEMA = "h1_carrierid_date_lodo_hls_terminal_evaluation_v1"
AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_hls_fivedate_terminal_aggregate_v1"
AGGREGATE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_FIVEDATE_HC_MINUS_HLS_AGGREGATED"
EVALUATION_DIR = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls/terminal_evaluations"
DEFAULT_OUTPUT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls/H1_CARRIERID_DATE_LODO_HLS_FIVEDATE_TERMINAL_AGGREGATE_v1.json"


class HlsFiveDateAggregateError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsFiveDateAggregateError(message)


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _path(date: str, directory: Path) -> Path:
    return directory / f"H1_CARRIERID_DATE_LODO_HLS_{date}_HC_HLS_TERMINAL_EVALUATION_v1.json"


def _read(date: str, directory: Path) -> tuple[Path, dict[str, Any]]:
    path = _path(date, directory).resolve()
    _need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444,
          f"{date}: immutable canonical H-LS evaluation receipt missing")
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == EVALUATION_SCHEMA
          and body.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_HLS_{date}_HC_HLS_EVALUATED"
          and body.get("outer_date") == date,
          f"{date}: H-LS evaluation receipt schema/status drift")
    _need(body.get("one_shot", {}).get("canonical_output_path") == str(path)
          and body.get("one_shot", {}).get("same_date_prior_hls_terminal_evaluation_receipts") == 0,
          f"{date}: H-LS evaluation is not canonical one-shot evidence")
    _need(body.get("deployment_updates") == {
        "optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True,
    }, f"{date}: H-LS evaluation records deployment updates")
    target, metrics = body.get("target"), body.get("metrics")
    _need(isinstance(target, Mapping) and isinstance(metrics, Mapping)
          and target.get("same_support_identity_and_query_windows") is True,
          f"{date}: paired target-window contract missing")
    hc, hls = metrics.get("h_c"), metrics.get("h_ls")
    _need(isinstance(hc, Mapping) and isinstance(hls, Mapping)
          and hc.get("query_window_indices_sha256") == hls.get("query_window_indices_sha256")
          == target.get("shared_query_window_indices_sha256")
          and hc.get("state_immutable") is True and hls.get("state_immutable") is True,
          f"{date}: paired metric/window/state contract drift")
    delta = _finite(hc.get("pooled_r2"), f"{date}/H-C R2") - _finite(hls.get("pooled_r2"), f"{date}/H-LS R2")
    _need(math.isclose(delta, _finite(metrics.get("h_c_minus_h_ls"), f"{date}/delta"),
                       rel_tol=0.0, abs_tol=1e-12), f"{date}: pooled delta drift")
    sessions = tuple(target.get("sessions", ()))
    _need(sessions and tuple(hc.get("per_session", {})) == tuple(hls.get("per_session", {})) == sessions,
          f"{date}: recording rows/order drift")
    return path, body


def aggregate(*, evaluation_dir: Path = EVALUATION_DIR, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import sha256_file, write_immutable_json

    rows: dict[str, Any] = {}
    date_deltas: list[float] = []
    recording_rows: list[dict[str, Any]] = []
    for date in DATES:
        path, body = _read(date, Path(evaluation_dir))
        metrics, sessions = body["metrics"], tuple(body["target"]["sessions"])
        delta = float(metrics["h_c_minus_h_ls"])
        date_deltas.append(delta)
        recordings = {}
        for session in sessions:
            hc = float(metrics["h_c"]["per_session"][session]["r2"])
            hls = float(metrics["h_ls"]["per_session"][session]["r2"])
            recordings[session] = {"h_c_r2": hc, "h_ls_r2": hls, "h_c_minus_h_ls": hc - hls}
            recording_rows.append({"outer_date": date, "session": session, "h_c_minus_h_ls": hc - hls})
        rows[date] = {
            "receipt": {"path": str(path), "sha256": sha256_file(path)},
            "h_c_pooled_r2": float(metrics["h_c"]["pooled_r2"]),
            "h_ls_pooled_r2": float(metrics["h_ls"]["pooled_r2"]),
            "h_c_minus_h_ls": delta, "recordings": recordings,
        }
    values = np.asarray(date_deltas, dtype=np.float64)
    rng = np.random.default_rng(20260808)
    bootstrap = values[rng.integers(0, len(values), size=(100_000, len(values)))].mean(axis=1)
    body = {
        "schema": AGGREGATE_SCHEMA, "status": AGGREGATE_STATUS,
        "inference_unit": "outer_date", "recordings_nested_and_reported": True,
        "required_outer_dates": list(DATES), "all_five_dates_reported": True,
        "per_date": rows, "all_recordings": recording_rows,
        "summary": {
            "n_outer_dates": len(values), "n_recordings": len(recording_rows),
            "h_c_minus_h_ls_mean": float(values.mean()),
            "h_c_minus_h_ls_median": float(np.median(values)),
            "h_c_minus_h_ls_paired_standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
            "bootstrap": {"seed": 20260808, "replicates": 100_000,
                          "percentile_95_interval": [float(np.quantile(bootstrap, 0.025)),
                                                     float(np.quantile(bootstrap, 0.975))]},
            "positive_date_count": int(np.count_nonzero(values > 0.0)),
            "negative_date_count": int(np.count_nonzero(values < 0.0)),
            "zero_date_count": int(np.count_nonzero(values == 0.0)),
        },
        "interpretation_limit": (
            "This complete-grid control tests necessity of correct temporal velocity-label pairing under the "
            "audited strong misalignment; mixed/null evidence limits that mechanism claim and does not erase H-C versus H-S."
        ),
        "scope": {"receipt_only": True, "nwb_opened": 0, "checkpoint_opened": 0,
                  "trainer_constructed": False, "cuda_constructed": False},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": AGGREGATE_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(aggregate(evaluation_dir=args.evaluation_dir, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
