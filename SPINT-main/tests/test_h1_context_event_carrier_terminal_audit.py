from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = WORKSPACE / "sua_exploration/scripts"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRUCTURAL = SCRIPTS / "verify_h1_context_event_carrier_terminal.py"
RECOMPUTE = SCRIPTS / "recompute_h1_context_event_carrier_r2.py"


def test_context_terminal_audits_never_import_the_terminal_evaluator() -> None:
    forbidden = "h1_context_event_carrier_evaluate"
    for path in (STRUCTURAL, RECOMPUTE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert all(forbidden not in item for item in imported), path


def test_context_terminal_audits_are_terminal_gated_and_fail_closed_when_absent(tmp_path: Path) -> None:
    structural = _module("context_structural_audit", STRUCTURAL)
    recompute = _module("context_r2_audit", RECOMPUTE)
    absent = tmp_path / "not-created-terminal.json"
    with pytest.raises(ValueError, match="missing regular terminal receipt"):
        structural.verify(absent)
    args = type("Args", (), {"receipt": absent, "data_dir": tmp_path, "device": "cpu", "batch_size": 29,
                               "tolerance": 2.0e-6})()
    with pytest.raises(ValueError, match="missing terminal receipt"):
        recompute.run(args)


def test_context_terminal_audits_reject_an_immutable_synthetic_nonterminal_receipt(tmp_path: Path) -> None:
    structural = _module("context_structural_synthetic", STRUCTURAL)
    recompute = _module("context_r2_synthetic", RECOMPUTE)
    terminal = tmp_path / "synthetic-terminal.json"
    terminal.write_text("{}\n", encoding="utf-8")
    terminal.chmod(0o444)
    with pytest.raises(ValueError, match="terminal identity/schema mismatch"):
        structural.verify(terminal)
    args = type("Args", (), {"receipt": terminal, "data_dir": tmp_path, "device": "cpu", "batch_size": 29,
                               "tolerance": 2.0e-6})()
    with pytest.raises(ValueError, match="terminal receipt schema mismatch"):
        recompute.run(args)


def test_context_recompute_r2_is_float64_and_requires_exact_shapes() -> None:
    recompute = _module("context_r2_float64", RECOMPUTE)
    truth = np.array([[1.0, 2.0], [3.0, 7.0], [8.0, -2.0]], dtype=np.float32)
    prediction = np.array([[1.5, 1.0], [2.5, 8.0], [9.0, -3.0]], dtype=np.float32)
    expected = 1.0 - float(np.square(truth.astype(np.float64) - prediction.astype(np.float64)).sum(dtype=np.float64)) / float(
        np.square(truth.astype(np.float64) - truth.astype(np.float64).mean(axis=0, keepdims=True)).sum(dtype=np.float64)
    )
    assert recompute.r2(truth, prediction) == pytest.approx(expected, abs=0.0)
    with pytest.raises(ValueError, match="R2 shape mismatch"):
        recompute.r2(truth[:, :1], prediction)


def test_context_terminal_audit_contract_has_v3_and_8965_bindings() -> None:
    structural = _module("context_structural_constants", STRUCTURAL)
    recompute = _module("context_recompute_constants", RECOMPUTE)
    assert structural.SCHEMA == recompute.SCHEMA
    assert structural.QUERY_SHA == recompute.QUERY_SHA
    assert sum(structural.TARGET_SAMPLES.values()) == 8965
    assert recompute.TARGET_SAMPLES == structural.TARGET_SAMPLES
    assert structural.INTERVENTIONS == ("full", "zero", "row", "label", "tag", "midpoint")
    assert structural.REQUIRED_CLAUSES == (
        "common_independent_zero", "same_checkpoint_zero", "row_shuffle",
        "endpoint_label_shuffle", "tag_shuffle", "label_two_of_two", "tag_two_of_two",
    )
