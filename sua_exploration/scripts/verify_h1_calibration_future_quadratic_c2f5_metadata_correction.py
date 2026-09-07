#!/usr/bin/env python3
"""Verify the additive, no-data metadata correction for immutable QC2F5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_calibration_future_quadratic_c2f5 as qc2f5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


WRONG = "smallest_lambda_within_1e-12_of_best_mean_source_date_LODO_score"
AUTHORITATIVE = "strongest_lambda_within_1e-12_of_best_mean_source_date_LODO_score"
GRID = (0.01, 0.1, 1.0, 10.0)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strongest(rows: list[Mapping[str, Any]]) -> float:
    need(tuple(float(row["ridge_lambda"]) for row in rows) == GRID, "QC2F5 correction grid/order drift")
    best = max(float(row["mean_score"]) for row in rows)
    return max(float(row["ridge_lambda"]) for row in rows if float(row["mean_score"]) >= best - 1e-12)


def canonical_5d_row_hash(values: np.ndarray) -> str:
    """Hash a row multiset by lexsorting full 5D rows, never individual columns."""

    rows = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1, 5))
    need(np.isfinite(rows).all(), "QC2F5 synthetic row probe nonfinite")
    order = np.lexsort(tuple(rows[:, index] for index in range(4, -1, -1)))
    return hashlib.sha256(np.ascontiguousarray(rows[order]).tobytes()).hexdigest()


def assert_complete_5d_teacher_permutation() -> dict[str, Any]:
    """No-data probe of the bound implementation's complete-row shuffle contract."""

    sessions = 3
    total = sessions * v1.EXPECTED_NEURONS
    support = np.zeros((sessions, v1.EXPECTED_NEURONS, 5), dtype=np.float64)
    quality = np.zeros((sessions, v1.EXPECTED_NEURONS, qc2f5.QUALITY_DIM), dtype=np.float64)
    # Each complete teacher row is unique, so both lexsort multiset equality
    # and changed row attachment are exact rather than probabilistic.
    teachers = np.arange(total * 5, dtype=np.float64).reshape(sessions, v1.EXPECTED_NEURONS, 5)
    examples = qc2f5.SourceExamples(tuple(f"ses-19250108T00{index:04d}" for index in range(sessions)), "19250101", 3,
                                     support, quality, teachers, tuple({"synthetic": index} for index in range(sessions)))
    shuffled, manifest = qc2f5.source_teacher_pairing_shuffle(examples)
    before = canonical_5d_row_hash(examples.future_teachers)
    after = canonical_5d_row_hash(shuffled.future_teachers)
    need(before == after and manifest["fixed_points"] == 0, "QC2F5 complete 5D teacher row multiset/permutation drift")
    need(not np.array_equal(examples.future_teachers, shuffled.future_teachers), "QC2F5 teacher pairing was not destroyed")
    return {"method": "canonical_lexsort_complete_5D_row_multiset_hash_synthetic_api_probe",
            "before_row_multiset_sha256": before, "after_row_multiset_sha256": after,
            "fixed_points": manifest["fixed_points"], "complete_row_attachment_changed": True,
            "no_source_data_opened": True}


def verify(correction_path: Path) -> dict[str, Any]:
    correction_path = correction_path.resolve()
    need(correction_path.is_file() and stat.S_IMODE(correction_path.stat().st_mode) == 0o444,
         "QC2F5 metadata correction must be immutable mode 0444")
    body = json.loads(correction_path.read_text(encoding="utf-8"))
    need(body["schema"] == "h1_qc2f5_metadata_correction_v1" and body["additive_only"] is True,
         "QC2F5 correction schema/additive contract drift")
    bindings = body["immutable_bindings"]
    for key in ("receipt", "predeclaration", "module"):
        item = Path(bindings[key]["path"]).resolve()
        need(item.is_file() and sha(item) == bindings[key]["sha256"], f"QC2F5 correction bound {key} hash drift")
    receipt = json.loads(Path(bindings["receipt"]["path"]).read_text(encoding="utf-8"))
    affected = body["affected_fields"]
    need(len(affected) == 12, "QC2F5 correction must enumerate exactly 12 stale field paths")
    for item in affected:
        budget, date = item["budget"], item["outer_date"]
        selection = receipt["budgets"][budget]["outer_date_details"][date]["operator_lambda_selection"]
        need(item["field_path"] == f"budgets.{budget}.outer_date_details.{date}.operator_lambda_selection.tie_break",
             "QC2F5 correction field path drift")
        need(selection["tie_break"] == item["stored_wrong_string"] == WRONG and item["authoritative_rule"] == AUTHORITATIVE,
             "QC2F5 correction wrong/authoritative metadata mismatch")
        recomputed = strongest(selection["source_date_lodo_rows"])
        need(math.isclose(float(selection["selected_lambda"]), recomputed, rel_tol=0.0, abs_tol=0.0),
             f"QC2F5 stored lambda not strongest-rule-consistent {budget}/{date}")
        need(math.isclose(float(item["stored_selected_lambda"]), recomputed, rel_tol=0.0, abs_tol=0.0),
             f"QC2F5 correction lambda record mismatch {budget}/{date}")
    invariants = body["unchanged_evidence"]
    need(invariants["status"] == receipt["status"] and invariants["passing_candidate"] == receipt["passing_candidate"],
         "QC2F5 correction claims changed terminal metrics")
    for budget in ("M3", "M4"):
        gate = receipt["budgets"][budget]["gate"]
        observed = invariants[budget]
        need(observed["gate_passed"] == gate["passed"], f"QC2F5 correction claims changed gate {budget}")
        for key in ("mean", "median"):
            need(math.isclose(float(observed[f"correct_minus_hse5_{key}"]), float(gate["correct_minus_hse5"][key]), rel_tol=0.0, abs_tol=0.0),
                 f"QC2F5 correction claims changed metric {budget}/{key}")
        for date, value in observed["selected_lambdas_by_outer_date"].items():
            need(math.isclose(float(value), float(receipt["budgets"][budget]["outer_date_details"][date]["operator_lambda_selection"]["selected_lambda"]), rel_tol=0.0, abs_tol=0.0),
                 f"QC2F5 correction claims changed selected lambda {budget}/{date}")
    teacher = assert_complete_5d_teacher_permutation()
    need(body["source_teacher_complete_5d_row_permutation_confirmation"]["method"] == teacher["method"],
         "QC2F5 correction 5D-row verification method drift")
    need(body["no_data_gpu_retraining_or_reevaluation"] == {"source_nwb_opened": 0, "gpu_used": False, "training": False, "outer_score_reevaluation": False},
         "QC2F5 correction no-work scope drift")
    return {"status": "PASS", "correction": str(correction_path), "correction_sha256": sha(correction_path),
            "affected_fields": len(affected), "teacher_complete_5d_row_permutation": teacher}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correction", type=Path)
    print(json.dumps(verify(parser.parse_args().correction), indent=2, sort_keys=True))
