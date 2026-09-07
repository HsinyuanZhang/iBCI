"""Static/unit contracts for the additive date-2 H-SE5 terminal tools.

These tests intentionally require neither H1 NWB data nor checkpoints.  The
full source/target integrity check belongs to the date-2 preflight and the
terminal receipt verifier.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EVALUATOR = ROOT / "scripts/h1_hse5_lodo_date2_terminal_evaluate.py"
VERIFIER = WORKSPACE / "sua_exploration/scripts/verify_h1_hse5_lodo_date2_terminal_receipt.py"
RECOMPUTE = WORKSPACE / "sua_exploration/scripts/recompute_h1_hse5_lodo_date2_r2.py"
QUERY_SHA = "b0cd153750cb484af1237b7af1861600aca69a9b201244c22d140b42b1da7f6e"


def _strings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_date2_tools_compile_and_bind_exact_query_contract() -> None:
    for path in (EVALUATOR, VERIFIER, RECOMPUTE):
        py_compile.compile(str(path), doraise=True)
        assert QUERY_SHA in _strings(path)


def test_terminal_evaluator_has_required_matched_controls_and_gate() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "single_immutable_date2_source_snapshot" in source
    assert "build_dated_sparse_target_dataset" in source
    assert "full_same_checkpoint_interventions" in source
    assert "independently_trained_zero5" in source
    assert "all_three_recording_full_minus_independently_trained_zero5_positive" in source
    assert "target_optimizer_steps\": 0" in source
    assert "target_backward_steps\": 0" in source


def test_structural_verifier_does_not_import_terminal_evaluator() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "h1_hse5_lodo_date2_terminal_evaluate" not in imported_modules
    assert "full_minus_same_checkpoint_endpoint_label_shuffle_pooled" in source


def test_independent_r2_and_batch29_contract() -> None:
    module = _module(RECOMPUTE, "hse5_date2_recompute_test")
    import numpy as np

    truth = np.asarray([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert module.independent_r2(truth, truth) == 1.0
    assert abs(module.independent_r2(truth, np.zeros_like(truth)) - (-21.75)) < 1e-12
    assert "args.batch_size == 29" in RECOMPUTE.read_text(encoding="utf-8")
