"""Static-only tests for the append-only scorer-adapter parity-v2 chain.

They deliberately do not import the runtime helper, Torch, PyNWB, a model, or
the score runners.  No checkpoint, normalizer NPZ, NWB, forward, or R² is
executed by this test file.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[2]
WRITER_SOURCE = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_scorer_adapter_parity_prelaunch_v2.py"
HELPER_SOURCE = ROOT / "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v2.py"
RUNNER_SOURCE = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_scorer_adapter_parity_v2.py"


def _writer():
    spec = importlib.util.spec_from_file_location("subm_parity_v2_writer_test", WRITER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_authority_binds_final_score_only_v2_and_marks_v1_stale() -> None:
    writer = _writer()
    audit = writer.static_authority(ROOT)
    assert audit["final_score_only_v2_sources"]["sua_exploration/mc_maze/subm_co_score_only_v2.py"]["sha256"] == "213e5495b4fa8776967ea07bf57743db18619b74aea593bac1d17d5944308e4d"
    assert audit["final_score_only_v2_prelaunch"]["receipt.json"]["sha256"] == "44c51dd5aa399636138f18a43ab6dbf44a2d2fe065ace3600a61e044e25f0868"
    disposition = audit["parity_v1_disposition"]
    assert disposition["status"] == "STALE_NON_AUTHORIZING_SUPERSEDED"
    assert disposition["old_core_sha256"] != disposition["final_core_sha256"]


def test_v2_prelaunch_is_write_once_immutable_and_stays_non_authorizing(tmp_path: Path) -> None:
    writer = _writer()
    output = tmp_path / "parity-v2"
    result = writer.write_prelaunch(output, ROOT)
    assert result["status"] == "STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED"
    stored = writer.verify_stored_prelaunch(output, ROOT)
    assert stored["status"] == result["status"]
    draft = json.loads((output / "parity_protocol_draft.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert draft["fixed_fixture"]["checkpoint"]["seed"] == 44
    assert draft["fixed_fixture"]["support_query_batch"]["rows"] == "0:16"
    assert receipt["operations"] == {
        "checkpoint_files_opened": 0,
        "normalizer_files_opened": 0,
        "nwb_files_opened": 0,
        "model_forward_calls": 0,
        "torchmetrics_calls": 0,
        "gpu_used": False,
        "subm_nwb_paths_constructed": False,
        "subm_nwb_files_accessed": False,
    }
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(writer.StaticParityV2Error, match="already exists"):
        writer.write_prelaunch(output, ROOT)


def test_helper_and_runner_sources_have_one_shared_trace_contract_and_hard_execute_fence() -> None:
    helper = HELPER_SOURCE.read_text(encoding="utf-8")
    for required in (
        "def prepare_fixed_c1_batch(",
        "def capture_c1_reference_trace(",
        "def capture_future_adapter_trace(",
        "def assert_trace_parity(",
        "def execute_parity_once_not_authorized(",
        "c1.load_session_with_trials(",
        "c1.build_calib_trials_for_indices(",
        "c1.make_subset_dataset(",
        "c1.attach_side_features(",
        "c1._unpack_loader_batch(",
        "c1.decode_last_behavior(",
    ):
        assert required in helper
    assert helper.count("def _capture(") == 1
    assert helper.count("def _reference_raw_output(") == 1
    runner = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert 'choices=("dry-run", "execute")' in runner
    assert "Do not import the helper on this path" in runner
    assert "PARITY_EXECUTION_NOT_AUTHORIZED" in runner


def test_prelaunch_writer_ast_has_no_runtime_model_or_data_import() -> None:
    tree = ast.parse(WRITER_SOURCE.read_text(encoding="utf-8"), filename=str(WRITER_SOURCE))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any(name == "pynwb" or name.startswith("pynwb.") for name in imported)
