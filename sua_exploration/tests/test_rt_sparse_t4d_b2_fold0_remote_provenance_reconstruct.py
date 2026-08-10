"""Local contract tests for the fold-0 remote provenance receipt builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/rt_sparse_t4d_b2_fold0_remote_provenance_reconstruct.py"


def _module():
    spec = importlib.util.spec_from_file_location("fold0_reconstruct_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_query_program_is_cpu_data_sampler_only():
    module = _module()
    program = module._query_program("/preserved/source", t4d_closure="/stage/cell.json")
    assert "FalconDataset" in program
    assert "SessionBatchSampler" in program
    assert "evaluated_digests" in program
    assert "torch" not in program
    assert "Trainer" not in program
    assert "load_from_checkpoint" not in program
    assert "backward" not in program


def test_reconstruction_program_replays_all_eligible_not_legacy_audit_digests():
    module = _module()
    program = module._query_program("/preserved/source")
    # Older source trees may lack the digest fields in query_window_audit.
    # The receipt must recompute them from the actual dataset values.
    assert "all_indices=list(range(len(ds)))" in program
    assert "all_eligible_digests" in program
    assert "audit[k]" not in program
