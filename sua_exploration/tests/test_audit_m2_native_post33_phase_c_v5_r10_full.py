"""Small CPU-only tests for the post-completion r10 receiver audit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_m2_native_post33_phase_c_v5_r10_full.py"


def _module():
    spec = importlib.util.spec_from_file_location("r10_full_audit_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(module):
    rows = []
    for seed in module.SEEDS:
        for fold, session in module.FOLDS.items():
            spint = 0.10 + 0.001 * fold
            t4 = spint + 0.05
            rows.append({
                "seed": seed, "fold": fold, "outer_session": session,
                "spint_r2": spint, "t4_r2": t4, "delta": t4 - spint,
            })
    return rows


def test_missing_aggregate_fails_closed_before_static_or_cell_read(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _module()
    root = tmp_path / "r10-cells"
    root.mkdir()
    monkeypatch.setattr(module, "R10_CELL_ROOT", root)
    assert module.main(["--cell-root", str(root)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "FAIL_CLOSED"
    assert "opened full aggregate missing" in payload["reason"]


def test_row_gate_checker_recomputes_and_rejects_bad_delta() -> None:
    module = _module()
    rows = _rows(module)
    stage_rows = [{key: value for key, value in row.items() if key != "seed"} for row in rows[:7]]
    summary = module.validate_rows_and_gates(rows, stage_rows=stage_rows)
    assert summary["all_six_pass"] is True
    assert summary["gates"]["mean_delta"]["pass"] is True
    rows[-1] = {**rows[-1], "delta": 0.0}
    with pytest.raises(module.AuditError, match="delta"):
        module.validate_rows_and_gates(rows, stage_rows=stage_rows)
