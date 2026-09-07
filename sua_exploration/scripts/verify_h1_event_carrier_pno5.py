#!/usr/bin/env python3
"""Independent PNO5 receipt verifier: scope, hashes, H-SE5 parity and metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from sua_exploration.mc_maze import h1_event_carrier_pno5 as pno5  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1  # noqa: E402


def _need(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def _close(left: Any, right: Any, tolerance: float = 1.0e-12) -> None:
    _need(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance), f"PNO5 metric mismatch {left!r} {right!r}")


def _summary(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    values = np.asarray([value for _name, value in rows], dtype=np.float64); _need(values.shape == (13,) and np.isfinite(values).all(), "PNO5 13 finite rows")
    removed = int(np.argmax(np.abs(values)))
    return {"defined_sessions": 13, "mean": float(values.mean()), "median": float(np.median(values)), "positive": int((values > 0).sum()),
            "zero": int((values == 0).sum()), "negative": int((values < 0).sum()), "leave_largest_absolute_out_mean": float(np.delete(values, removed).mean()), "removed_session": rows[removed][0]}


def _verify_summary(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"): _need(observed.get(key) == expected[key], f"PNO5 summary {key}")
    for key in ("mean", "median", "leave_largest_absolute_out_mean"): _close(observed.get(key), expected[key])


def _positive(item: Mapping[str, Any]) -> bool:
    return bool(item["defined_sessions"] == 13 and item["mean"] > 0 and item["median"] > 0 and item["positive"] >= 10 and item["leave_largest_absolute_out_mean"] > 0)


def verify(receipt_path: Path, *, data_root: Path | None = None, recompute: bool = True) -> dict[str, Any]:
    receipt = receipt_path.resolve(); sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    _need(receipt.is_file() and stat.S_IMODE(receipt.stat().st_mode) == 0o444, "PNO5 immutable receipt required")
    _need(sidecar.is_file() and stat.S_IMODE(sidecar.stat().st_mode) == 0o444, "PNO5 immutable SHA sidecar required")
    digest = _sha(receipt); _need(sidecar.read_text(encoding="ascii").strip() == f"{digest}  {receipt.name}", "PNO5 sidecar mismatch")
    body = json.loads(receipt.read_text(encoding="utf-8")); _need(body.get("schema") == pno5.SCHEMA and body.get("protocol") == pno5.PROTOCOL, "PNO5 schema")
    _need(body["candidate_matrix_predeclared_before_data_run"] is True and body["predeclaration"] == dict(pno5.PREDECLARATION), "PNO5 immutable predeclaration")
    predecl = body["predeclaration_binding"]; predecl_path = Path(predecl["path"]).resolve(); predecl_sidecar = Path(predecl["sidecar_path"]).resolve()
    _need(predecl["published_before_nwb_open"] is True and predecl_path.is_file() and predecl_sidecar.is_file()
          and stat.S_IMODE(predecl_path.stat().st_mode) == 0o444 and stat.S_IMODE(predecl_sidecar.stat().st_mode) == 0o444
          and _sha(predecl_path) == predecl["sha256"] and predecl_sidecar.read_text(encoding="ascii").strip() == f"{predecl['sha256']}  {predecl_path.name}", "PNO5 predeclaration artifact")
    predecl_body = json.loads(predecl_path.read_text(encoding="utf-8"))
    _need(predecl_body.get("schema") == "h1_event_carrier_pno5_predeclaration_v1" and predecl_body.get("protocol") == pno5.PROTOCOL
          and predecl_body.get("data_opened_before_declaration") is False and predecl_body.get("predeclaration") == dict(pno5.PREDECLARATION), "PNO5 predeclaration contents")
    constants = body["frozen_constants"]; _need(constants["rank"] == 4 and constants["carrier_dim"] == 5 and tuple(constants["carrier_order"]) == ("w1", "w2", "w3", "w4", "b") and tuple(constants["support_budgets"]) == (3, 4) and constants["model_grid"] == [], "PNO5 5D/grid drift")
    scope = body["scope"]
    expected_scope = {"public_held_in_calibration_nwbs_opened": 13, "minival_nwbs_opened": 0, "held_out_nwbs_opened": 0, "formal_test_labels_opened": 0, "dense_velocity_opened": False, "target_session_optimizer_steps": 0, "target_session_backward_steps": 0, "cuda_used": False, "query_population_mean_used": False, "query_labels_used": False, "query_future_fit_used": False}
    for key, value in expected_scope.items(): _need(scope.get(key) == value, f"PNO5 scope {key}")
    binding = body["implementation_binding"]
    for prefix in ("module", "runner", "event_parser", "hse5"): _need(_sha(Path(binding[f"{prefix}_path"])) == binding[f"{prefix}_sha256"], f"PNO5 bound source drift {prefix}")
    source = body["source_binding"]; names = tuple(source["sessions"]); _need(names == v1.H1_HELDIN_SESSIONS and tuple(source["dates"]) == v1.H1_DATES, "PNO5 session/date count")
    forbidden = tuple(v1.FORBIDDEN_PATH_TOKENS)
    for row in source["files"]:
        _need(not any(word in str(row["path"]).lower() for word in forbidden) and _sha(Path(row["path"])) == row["sha256"], f"PNO5 bad input {row['session']}")
    reproduction = body["baseline_reproduction"]; _need(reproduction["passed"] is True and reproduction["comparisons"] == 78 and float(reproduction["maximum_absolute_difference"]) <= float(reproduction["absolute_tolerance"]), "PNO5 HSE5 exact reproduction")
    reports: dict[str, Any] = {}
    for budget in (3, 4):
        item = body["budgets"][f"M{budget}"]; _need(item["budget_trials"] == budget and set(item["endpoint_map_by_outer_date"]) == set(v1.H1_DATES), "PNO5 dates")
        for date, manifest in item["endpoint_map_by_outer_date"].items():
            expected = tuple(name for name in names if v1.session_date(name) != date)
            _need(tuple(manifest["source_sessions"]) == expected and date not in {v1.session_date(name) for name in manifest["source_sessions"]}, "PNO5 source/target separation")
        rows, baseline = item["pno5_sessions"], item["hse5_baseline_sessions"]; _need(tuple(rows) == names and tuple(baseline) == names, "PNO5 rows")
        fields = {"correct_r2": "median_r2_correct", "correct_minus_hse5": None, "correct_minus_label_shuffle": "median_delta_label_shuffle", "correct_minus_nuisance_row_shuffle": "median_delta_nuisance_row_shuffle", "correct_minus_carrier_attachment_shuffle": "median_delta_carrier_attachment_shuffle", "correct_minus_intercept": "median_delta_intercept", "correct_minus_nuisance_only": "median_delta_nuisance_only"}
        for name, row in rows.items():
            _need(row["carrier_dim"] == 5 and row["design_rank"] == 5 and row["outer_future_used_only_for_raw_log_rate_scoring"] is True and row["query_population_mean_used"] is False and row["query_labels_used"] is False and row["query_nuisance_estimated"] is False, "PNO5 query/carrier contract")
            _need(row["label_shuffle"]["fixed_points"] == 0 and row["nuisance_row_shuffle"]["fixed_points"] == 0 and row["carrier_attachment_shuffle"]["fixed_points"] == 0, "PNO5 shuffle contract")
            _need(row["nuisance"]["support_only"] is True and row["nuisance"]["uses_query_statistics"] is False and row["nuisance_only"]["endpoint_pairing_used"] is False, "PNO5 nuisance scope")
        for output, source_field in fields.items():
            values = [(name, rows[name]["median_r2_correct"] - baseline[name]["median_r2_correct"] if source_field is None else rows[name][source_field]) for name in names]
            _verify_summary(item["aggregate"][output], _summary(values))
        aggregate = item["aggregate"]; material = bool(_positive(aggregate["correct_minus_hse5"]) and aggregate["correct_minus_hse5"]["mean"] >= .02 and aggregate["correct_minus_hse5"]["median"] >= .01)
        required = ("correct_minus_label_shuffle", "correct_minus_nuisance_only", "correct_minus_intercept", "correct_minus_nuisance_row_shuffle", "correct_minus_carrier_attachment_shuffle")
        expected_gate = bool(material and all(_positive(aggregate[key]) for key in required)); _need(item["gate"]["passed"] is expected_gate and item["gate"]["material_gain_vs_hse5"] is material, "PNO5 gate")
        reports[f"M{budget}"] = {"gate": item["gate"], "gain": aggregate["correct_minus_hse5"]}
    expected_status = "PASS_CPU_PNO5_MATERIAL" if all(body["budgets"][f"M{b}"]["gate"]["passed"] for b in (3, 4)) else "STOP_CPU_PNO5_NOT_MATERIAL"
    _need(body["status"] == expected_status and body["gpu_authorized_by_this_screen"] is False, "PNO5 terminal status")
    recomputed = False
    if recompute:
        _need(data_root is not None, "PNO5 recompute requires data root")
        paths = v1.index_heldin_calib(data_root.resolve()); sessions = {name: v1.load_event_session(paths[name]) for name in names}; fresh = pno5.run_screen(sessions)
        for budget in (3, 4):
            for name in names: _close(fresh["budgets"][f"M{budget}"]["pno5_sessions"][name]["median_r2_correct"], body["budgets"][f"M{budget}"]["pno5_sessions"][name]["median_r2_correct"])
        recomputed = True
    return {"status": "PASS", "receipt": str(receipt), "receipt_sha256": digest, "terminal_status": body["status"], "metrics_recomputed": recomputed, "budgets": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("receipt", type=Path); parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954"); parser.add_argument("--no-recompute", action="store_true")
    args = parser.parse_args(); print(json.dumps(verify(args.receipt, data_root=args.data_root, recompute=not args.no_recompute), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
