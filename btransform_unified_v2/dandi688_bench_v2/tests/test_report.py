import json

import numpy as np
import pytest

from dandi688_bench_v2.common import sha256
from dandi688_bench_v2.final_access import FINAL_SCORE_CELLS, REQUIRED_CELLS, SEAL_SCHEMA, SEAL_STATUS, SUPPLEMENTAL_FULL_CELLS
from dandi688_bench_v2.report import build_report


ROSTER = ["final_a", "final_b", "final_c", "final_d", "final_e", "final_f"]


def _metrics(value):
    sessions = [{"session_id": name, "r2": value} for name in ROSTER]
    return {"n_sessions": 6, "mean_r2": value, "sessions": sessions}


def _receipt(tmp_path, *, missing=None):
    seal_path = tmp_path / "selection_seal.json"
    seal_path.write_text(json.dumps({"schema": SEAL_SCHEMA, "status": SEAL_STATUS,
                                     "cell_selections": {cell: {} for cell in REQUIRED_CELLS},
                                     "supplemental_full_selections": {cell: {} for cell in SUPPLEMENTAL_FULL_CELLS}}))
    results = {}
    seed_values = {"full_sua": 1.0, "full_pmua": 0.5, "full_sua_s43": 2.0,
                   "full_pmua_s43": 1.0, "full_sua_s44": 3.0, "full_pmua_s44": 1.5}
    for index, cell in enumerate(FINAL_SCORE_CELLS):
        if cell == missing:
            continue
        if cell == "aligned_fa_wf_pmua":
            results[cell] = {"status": "UNAVAILABLE", "reason": "rank gate"}
            continue
        value = seed_values.get(cell, float(index) / 10.0)
        prediction = tmp_path / f"{cell}.npz"
        np.savez_compressed(prediction, **{name: np.array([value]) for name in ROSTER})
        results[cell] = {"status": "SCORED", "result": {"metrics": _metrics(value)},
                         "predictions": {"path": str(prediction), "sha256": sha256(prediction),
                                         "sessions": {name: {} for name in ROSTER}}}
    receipt_path = tmp_path / "final_score_receipt.json"
    receipt_path.write_text(json.dumps({"status": "FINAL_SCORED", "seal_path": str(seal_path),
                                        "seal_sha256": sha256(seal_path), "roster": ROSTER, "results": results}))
    return receipt_path


def test_report_writes_main_and_three_seed_summaries(tmp_path):
    outputs = build_report(_receipt(tmp_path), tmp_path / "report")
    summary = json.loads(outputs["full_seed_summary_json"].read_text())
    assert summary["full"]["sua"]["three_seed_mean_r2"] == 2.0
    assert summary["full"]["sua"]["three_seed_sample_sd_r2"] == 1.0
    assert summary["matched_sua_minus_pmua"]["three_seed_mean_r2_difference"] == 1.0
    assert summary["matched_sua_minus_pmua"]["three_seed_sample_sd_r2_difference"] == 0.5
    main = outputs["main_results"].read_text()
    assert "aligned_fa_wf_pmua,UNAVAILABLE,NA" in main
    assert "Full−ACT is not a pure causal label effect" in outputs["report"].read_text()


def test_report_rejects_missing_full_seed_cell(tmp_path):
    with pytest.raises(ValueError, match="exactly FINAL_SCORE_CELLS"):
        build_report(_receipt(tmp_path, missing="full_pmua_s44"), tmp_path / "report")


def test_report_preserves_final_time_fa_per_session_reasons(tmp_path):
    receipt_path = _receipt(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["results"]["aligned_fa_wf_pmua"] = {
        "status": "UNAVAILABLE",
        "result": {"selected": True, "status": "UNAVAILABLE", "metrics": None,
                   "per_session": {name: {"eligible": False, "reason": f"rank gate {name}"} for name in ROSTER}},
        "predictions": None,
    }
    receipt_path.write_text(json.dumps(receipt))
    outputs = build_report(receipt_path, tmp_path / "report")
    rendered = outputs["report"].read_text()
    assert "final_a: UNAVAILABLE (rank gate final_a)" in rendered
    assert "final_f: UNAVAILABLE (rank gate final_f)" in outputs["main_results"].read_text()


def test_report_preserves_mixed_fa_statuses_and_partial_predictions(tmp_path):
    receipt_path = _receipt(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    scored, unavailable = ROSTER[:3], ROSTER[3:]
    prediction = tmp_path / "aligned_fa_partial_predictions.npz"
    np.savez_compressed(prediction, **{name: np.array([1.]) for name in scored})
    receipt["results"]["aligned_fa_wf_pmua"] = {
        "status": "UNAVAILABLE",
        "result": {"selected": True, "status": "UNAVAILABLE", "metrics": None,
                   "per_session": {**{name: {"status": "SCORED", "prediction_sha256": f"sha-{name}"} for name in scored},
                                   **{name: {"status": "UNAVAILABLE", "reason": f"rank gate {name}"} for name in unavailable}}},
        "predictions": {"path": str(prediction), "sha256": sha256(prediction),
                        "sessions": {name: {} for name in scored}},
    }
    receipt_path.write_text(json.dumps(receipt))
    outputs = build_report(receipt_path, tmp_path / "report")
    rendered = outputs["report"].read_text()
    assert "final_a: SCORED (prediction retained)" in rendered
    assert "final_d: UNAVAILABLE (rank gate final_d)" in rendered
    assert "aligned_fa_wf_pmua,UNAVAILABLE,NA" in outputs["main_results"].read_text()


def test_report_rejects_scored_cell_without_prediction_npz(tmp_path):
    receipt_path = _receipt(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["results"]["full_sua"]["predictions"] = None
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="lacks a prediction NPZ receipt"):
        build_report(receipt_path, tmp_path / "report")
