"""Static-only tests for append-only independently-loaded data-adapter parity v3.

The tests import only the metadata writer. They never import the runtime helper,
Torch, PyNWB, score runners, or data/model owners and do not open a checkpoint,
normalizer NPZ, NWB, or execute a forward/R2.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[2]
WRITER_SOURCE = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v3.py"
HELPER_SOURCE = ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v3.py"
RUNNER_SOURCE = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v3.py"


def _writer():
    spec = importlib.util.spec_from_file_location("subm_parity_v3_writer_test", WRITER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_authority_preserves_v2_as_insufficient_non_authorizing() -> None:
    writer = _writer()
    audit = writer.static_authority(ROOT)
    assert (
        audit["final_score_only_v2_sources"][
            "sua_exploration/mc_maze/subm_co_score_only_v2.py"
        ]["sha256"]
        == "213e5495b4fa8776967ea07bf57743db18619b74aea593bac1d17d5944308e4d"
    )
    assert (
        audit["final_score_only_v2_prelaunch"]["receipt.json"]["sha256"]
        == "44c51dd5aa399636138f18a43ab6dbf44a2d2fe065ace3600a61e044e25f0868"
    )
    disposition = audit["parity_v2_disposition"]
    assert disposition["status"] == "INSUFFICIENT_DATA_ADAPTER_PARITY_NON_AUTHORIZING"
    assert (
        disposition["preserved_artifacts"]["receipt.json"]["sha256"]
        == "33876977d1a6ee2aed5b0f46d51651a15d76c24060f9f32fce9fc79875a68602"
    )
    assert (
        audit["score_only_v1_owner_sources"][
            "sua_exploration/mc_maze/multisession_datamodule.py"
        ]["sha256"]
        == "674fb4c235ba8f9393a6d1614f1f6f4260177ed9751e88acb4c05c4396d81e2d"
    )


def test_v3_prelaunch_is_write_once_immutable_and_has_zero_runtime_operations(
    tmp_path: Path,
) -> None:
    writer = _writer()
    output = tmp_path / "parity-v3"
    result = writer.write_prelaunch(output, ROOT)
    assert result["status"] == (
        "STATIC_PROTOCOL_AUDITED_V3_DATA_ADAPTER_PARITY_EXECUTION_NOT_AUTHORIZED"
    )
    stored = writer.verify_stored_prelaunch(output, ROOT)
    assert stored["status"] == result["status"]
    draft = json.loads((output / "parity_protocol_draft.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    protocol = draft["data_adapter_protocol"]
    assert protocol["reference_batch_reused_by_adapter"] is False
    assert protocol["calibration_gate"] == "LOADER_CALIBRATION_50_REBUILT_TO_C1_FIRST_N30"
    assert protocol["t4_pool_size"] == 50
    assert receipt["operations"] == {
        "checkpoint_files_opened": 0,
        "normalizer_files_opened": 0,
        "nwb_files_opened": 0,
        "runtime_helper_imported": False,
        "model_forward_calls": 0,
        "torchmetrics_calls": 0,
        "gpu_used": False,
        "subm_nwb_paths_constructed": False,
        "subm_nwb_files_accessed": False,
    }
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(writer.StaticParityV3Error, match="already exists"):
        writer.write_prelaunch(output, ROOT)


def test_helper_has_two_independent_owner_pipelines_then_one_observer_and_one_forward() -> None:
    helper = HELPER_SOURCE.read_text(encoding="utf-8")
    for required in (
        "def prepare_c1_reference_fixture(",
        "def prepare_score_only_adapter_fixture(",
        "def _forward_and_observe(",
        "def compare_prediction_target_exact(",
        "c1.load_session_with_trials(",
        "c1.attach_side_features(",
        "c1.make_subset_dataset(",
        "owners[\"load_dandi688_session\"](",
        "owners[\"list_datamodule_rewarded_trials\"](",
        "owners[\"load_unit_side_features\"](",
        "owners[\"MCMazeSessionDataset\"](",
        "calibration_n_trials=SUPPORT_TRIALS",
        "exclude_calibration_trials_from_windows=True",
        "rebuild_record, indices, IDENTITY_TRIALS",
        "LOADER_CALIBRATION_50_REBUILT_TO_C1_FIRST_N30",
        "for scope in (\"full\", \"first16\")",
    ):
        assert required in helper
    assert "Protocol" not in helper
    assert helper.count("def _forward_and_observe(") == 1
    assert "reference_batch_reused" not in helper


def test_writer_ast_has_no_runtime_data_or_model_import_and_runner_hard_fences_execute() -> None:
    tree = ast.parse(WRITER_SOURCE.read_text(encoding="utf-8"), filename=str(WRITER_SOURCE))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(name == "pynwb" or name.startswith("pynwb.") for name in imported)
    assert not any("subm_co_scorer_adapter_parity_v3" in name for name in imported)
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert 'choices=("dry-run", "execute")' in runner
    assert "Do not import the helper on this path" in runner
    assert "PARITY_EXECUTION_NOT_AUTHORIZED" in runner

