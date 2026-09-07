from __future__ import annotations

from pathlib import Path

from sua_exploration.scripts import verify_h1_sparse_event_endpoint_extended_cpu as closure


def test_extended_cpu_closure_is_immutable_and_preserves_historic_gpu_preflight() -> None:
    artifact = Path(__file__).resolve().parents[1] / "results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_EXTENDED_CPU_CLOSURE_v1.json"
    observed = closure.verify_closure(artifact)
    assert observed["status"] == "PASS"
    assert observed["new_gpu_arms_authorized"] == 0
