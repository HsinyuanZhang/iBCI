"""Score-blind regressions for the static sub-C scorer-adapter parity design.

These tests intentionally inspect only source text and JSON metadata.  They do
not import Torch/PyNWB, open an NWB/checkpoint/normalizer, run a forward pass,
or calculate a metric.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch.py"


def _module():
    spec = importlib.util.spec_from_file_location("subm_parity_prelaunch_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_design_pins_one_consumed_dev_fixture_and_no_external_execution() -> None:
    writer = _module()
    audit = writer.static_authority(ROOT)
    assert len(audit["source_hashes"]) == 9
    assert audit["c1_static_metadata"]["manifest"]["sha256"] == "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb"
    draft = writer.build_draft(ROOT)
    assert draft["status"] == "NOT_AUTHORIZED_FOR_PARITY_EXECUTION"
    assert draft["fixed_fixture"]["terminal_checkpoint"]["seed"] == 44
    assert draft["fixed_fixture"]["consumed_development_session"]["session"] == "sub-C_ses-CO-20151103"
    assert draft["fixed_fixture"]["chronology"]["rewarded_trial_support"] == "trials[0:50]"
    assert draft["comparison_thresholds"]["prediction_values"].endswith("max_rel=0")
    assert draft["device_policy"]["parity_execution_device"] == "cpu"
    assert draft["operations_by_this_draft"] == {
        "checkpoint_files_opened": 0,
        "normalizer_files_opened": 0,
        "nwb_files_opened": 0,
        "subm_nwb_paths_constructed": False,
        "subm_nwb_files_accessed": False,
        "model_forward_calls": 0,
        "prediction_calls": 0,
        "torchmetrics_calls": 0,
        "gpu_used": False,
        "normalizer_fitting_calls": 0,
        "optimizer_or_backward_calls": 0,
    }


def test_prelaunch_bundle_is_immutable_and_cannot_be_reused(tmp_path: Path) -> None:
    writer = _module()
    output = tmp_path / "parity-prelaunch"
    result = writer.write_prelaunch(output, ROOT)
    assert result["status"] == "STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED"
    draft = json.loads((output / "parity_protocol_draft.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    seal = json.loads((output / "seal.json").read_text(encoding="utf-8"))
    assert draft["parity_execution_receipt_schema"] == "dandi_000688_subc_scorer_adapter_parity_receipt_v1"
    assert receipt["status"] == "STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED"
    assert seal["parity_execution_still_not_authorized"] is True
    for path in (output / "parity_protocol_draft.json", output / "receipt.json", output / "seal.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(writer.StaticParityError, match="already exists"):
        writer.write_prelaunch(output, ROOT)


def test_writer_source_has_no_runtime_model_or_data_import() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(name == "pynwb" or name.startswith("pynwb.") for name in imported)
    assert "load_frozen_model" not in calls
    assert "load_session_with_trials" not in calls
