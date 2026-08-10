"""No-launch contracts for B2 exact-query forward-only re-evaluation preparation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_t4d_b2_forward_reeval_preflight.py"


def module():
    spec = importlib.util.spec_from_file_location("rt_b2_reeval_preflight_test", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _manifest(mod):
    stage2 = mod._load_module(mod.STAGE2, "stage2_fixture")
    result = stage2._synthetic_manifest()
    for name, path in {
        "outer_evaluator": mod.EVALUATOR,
        "datamodule": ROOT / "streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py",
        "falcon_dataset": ROOT / "streaming_calibration_exp/src/data/falcon_datamodule.py",
    }.items():
        result["surfaces"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_print_only_preflight_prepares_no_gpu_commands_and_flags_missing_fold0(tmp_path):
    mod = module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(mod)), encoding="utf-8")
    result = mod.prepare(manifest, output_root=tmp_path / "new_outer")
    assert result["status"] == "PASS_PRINT_ONLY_REEVAL_PREPARED_NOT_LAUNCHED"
    assert result["non_interference"] == {"nwb_opened": False, "torch_imported": False, "gpu_opened": False, "trainer_started": False, "artifact_written": False}
    assert result["cpu_practicality"]["unavailable_checkpoint_folds"] == [0]
    assert result["cpu_practicality"]["available_checkpoint_folds"] == list(range(1, 15))
    plan1 = result["plans"][1]
    assert plan1["available"] is True
    assert "--device" in plan1["cpu_command"] and plan1["cpu_command"][-1] == "cpu"
    assert plan1["minimal_gpu_substitution"][-1] == "cuda"
    assert not (tmp_path / "new_outer").exists()


def test_source_sha_drift_fails_before_any_command(tmp_path):
    mod = module()
    manifest = _manifest(mod)
    manifest["surfaces"]["outer_evaluator"]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(mod.PreflightError, match="SHA drift"):
        mod.prepare(path)
