from __future__ import annotations

import json
from pathlib import Path

from sua_exploration.scripts import verify_h1_sparse_event_endpoint_extended_cpu_v2 as closure


def test_v2_closure_includes_true_estimator_v5r2_and_correct_hse_v2_names() -> None:
    artifact = Path(__file__).resolve().parents[1] / "results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_EXTENDED_CPU_CLOSURE_v2.json"
    observed = closure.verify_closure(artifact)
    body = json.loads(artifact.read_text(encoding="utf-8"))
    assert observed["status"] == "PASS"
    assert body["route_final_statuses"]["HSE5_V2_SOURCE_GATE"] == "PASS_CPU_HSE5_M3_GPU_READY"
    assert body["route_final_statuses"]["ESTIMATOR_V5R2"] == "STOP_CPU_ESTIMATOR_CANDIDATES_NOT_MATERIAL"
    assert "estimator_v5r2_authoritative" in body["audited_artifacts"]
    assert "hse5_v2_source_audit" in body["audited_artifacts"]
    assert "hse5_v5r" not in json.dumps(body)
