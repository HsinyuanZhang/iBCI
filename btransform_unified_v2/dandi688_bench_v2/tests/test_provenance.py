"""A model-only migration must preserve old CPU results without hiding code drift."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dandi688_bench_v2.common import PACKAGE, source_hashes
from dandi688_bench_v2.provenance import baseline_common_compatibility, verify_execution_hashes


def _baseline() -> dict:
    return json.loads((PACKAGE / "results/baselines_dev_formal/selection.json").read_text())


def test_real_completed_cpu_results_survive_only_the_documented_b3s_metadata_change():
    receipt = _baseline()
    binding = verify_execution_hashes(receipt, kind="baseline")
    assert len(binding["explicit_metadata_migrations"]) == 1
    assert next(iter(binding["explicit_metadata_migrations"].values()))["all_other_python_ast_unchanged"]


def test_cpu_feature_source_drift_is_rejected():
    receipt = _baseline()
    receipt["code"]["btransform_unified_v2/dandi688_bench_v2/baselines.py"] = "0" * 64
    with pytest.raises(ValueError, match="baselines.py"):
        verify_execution_hashes(receipt, kind="baseline")


def test_metadata_exception_cannot_cover_a_changed_metric_function(tmp_path: Path):
    receipt = _baseline()
    key = "btransform_unified_v2/dandi688_bench_v2/common.py"
    changed = tmp_path / "changed_common.py"
    original = (PACKAGE / "common.py").read_text()
    assert 'float(np.mean([s["r2"] for s in scores]))' in original
    changed.write_text(original.replace('float(np.mean([s["r2"] for s in scores]))', '123.0'))
    assert baseline_common_compatibility(receipt["code"][key], current_path=changed) is None


def test_neural_receipt_cannot_omit_shared_execution_dependencies():
    receipt = {"code_hashes": source_hashes()}
    verify_execution_hashes(receipt, kind="neural")
    receipt["code_hashes"].pop("btransform_unified_v1/src/btransform_unified_v1/schedule.py")
    with pytest.raises(ValueError, match="schedule.py"):
        verify_execution_hashes(receipt, kind="neural")
