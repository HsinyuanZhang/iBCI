"""Hermetic contracts for the receipt-first RT Stage-2 matrix supervisor."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_endpoint_stage2_matrix.py"


def module():
    spec = importlib.util.spec_from_file_location("rt_stage2_matrix", SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture()
def synthetic_matrix(tmp_path: Path):
    """Build all 45 closure receipts without NWBs, checkpoints, or ignored results."""

    m = module()
    root = tmp_path / "stage2"
    root.mkdir()
    manifest = {"matrix": {"cells": m.cells()}}
    manifest_path = root / "STAGE2_MATRIX_MANIFEST_v1.json"
    _write(manifest_path, manifest)
    manifest_sha = _sha(manifest_path)

    query_ids = {
        fold: {
            "ordered_window_start_sha256": hashlib.sha256(f"window-{fold}".encode()).hexdigest(),
            "ordered_target_covariate_evalmask_sha256": hashlib.sha256(f"mask-{fold}".encode()).hexdigest(),
            "ordered_query_identity_sha256": hashlib.sha256(f"query-{fold}".encode()).hexdigest(),
        }
        for fold in range(15)
    }
    for cell in m.cells():
        key = cell["output_key"]
        artifact = root / "artifacts" / key
        config = artifact / "config.yaml"
        checkpoint = artifact / "best.ckpt"
        split = artifact / "split_manifest.json"
        selection = artifact / "selection.json"
        outer = artifact / "outer.json"
        _write(config, "synthetic config")
        _write(checkpoint, b"synthetic checkpoint")
        target = f"ses-RT-{cell['fold']:02d}"
        _write(split, {
            "requested_side_feature_group": cell["arm"],
            "outer_loso_fold": cell["fold"],
            "target_session": target,
        })
        selection_body = {
            "schema": "rt_clean_nested_loso_selection_receipt_v1",
            "status": "PASS_FIT_INNER_SELECTION_ONLY",
            "arm": cell["arm"],
            "outer_loso_fold": cell["fold"],
            "seed": 42,
            "best_model_path": str(checkpoint),
            "best_model_sha256": _sha(checkpoint),
            "config_path": str(config),
            "config_sha256": _sha(config),
            "split_manifest_path": str(split),
            "split_manifest_sha256": _sha(split),
        }
        _write(selection, selection_body)
        outer_body = {
            "arm": cell["arm"],
            "outer_loso_fold": cell["fold"],
            "seed": 42,
            "outer_target_session": target,
            "target_backpropagation": False,
            "model_state_three_point_unchanged": True,
            "matched_query_window_identity": {target: query_ids[cell["fold"]]},
            "checkpoint_sha256": _sha(checkpoint),
            "config_path": str(config),
            "selection_receipt_path": str(selection),
            "fit_split_manifest": str(split),
            # T4d is deliberately above both controls in this synthetic
            # fixture, making aggregate deltas and sign accounting observable.
            "r2_variance_weighted": {
                "rt_sparse_endpoint_t4d": 0.50,
                "afc4_vel": 0.35,
                "zero4": 0.20,
            }[cell["arm"]],
        }
        _write(outer, outer_body)
        closure = {
            "schema": "rt_sparse_endpoint_stage2_cell_closure_v2",
            "matrix_manifest_sha256": manifest_sha,
            "cell": cell,
            "selection_receipt_sha256": _sha(selection),
            "config_sha256": _sha(config),
            "checkpoint_sha256": _sha(checkpoint),
            "split_manifest_sha256": _sha(split),
            "outer_receipt_sha256": _sha(outer),
            "artifact_paths": {
                "selection_receipt": str(selection),
                "config": str(config),
                "checkpoint": str(checkpoint),
                "split_manifest": str(split),
                "outer_receipt": str(outer),
            },
            "outer_receipt": outer_body,
        }
        _write(root / "cells" / f"{key}.json", closure)
    return m, root, manifest_path


def test_fixed_45_cell_fresh_matrix_and_unique_outputs():
    m = module()
    cells = m.cells()
    assert len(cells) == 45
    assert {(x["fold"], x["arm"]) for x in cells} == {
        (fold, arm) for fold in range(15) for arm in m.ARMS
    }
    assert len({x["run_id"] for x in cells}) == 45
    assert len({x["output_key"] for x in cells}) == 45
    assert all(x["seed"] == 42 and x["fresh_fit"] and x["exactly_once_outer_eval"] for x in cells)


def test_summary_is_ordered_robust_and_primary_gate_is_frozen():
    result = module().summarize({"b": 0.04, "a": 0.02, "c": -0.01})
    assert [x["session"] for x in result["ordered"]] == ["a", "b", "c"]
    assert result["positive"] == 2 and result["negative"] == 1
    assert result["primary_gate_pass"] is True
    assert result["removed_session"] == "b" and result["mean_ge_003"] is False


def test_synthetic_45_cell_aggregate_and_sign_accounting(synthetic_matrix):
    m, root, manifest_path = synthetic_matrix
    aggregate = m.aggregate(json.loads(manifest_path.read_text()), root)
    assert aggregate["status"] == "PASS_MATRIX_TERMINAL"
    assert aggregate["cells"] == 45
    delta = aggregate["t4d_minus_zero4"]
    assert delta["mean"] == pytest.approx(0.30)
    assert delta["median"] == pytest.approx(0.30)
    assert delta["positive"] == 15 and delta["negative"] == 0
    assert aggregate["t4d_minus_full"]["mean"] == pytest.approx(0.15)


def test_resume_accounting_distinguishes_closed_and_pending_without_launch(synthetic_matrix):
    m, root, manifest_path = synthetic_matrix
    state = m.audit_resume_state(json.loads(manifest_path.read_text()), root, manifest_path)
    assert state["cells_total"] == 45
    assert state["closed_cells"] == 45
    assert state["pending_cells"] == 0
    assert state["all_closed"] is True

    pending_key = m.cells()[-1]["output_key"]
    (root / "cells" / f"{pending_key}.json").unlink()
    resumed = m.audit_resume_state(json.loads(manifest_path.read_text()), root, manifest_path)
    assert resumed["closed_cells"] == 44
    assert resumed["pending_cells"] == 1
    assert resumed["pending_output_keys"] == [pending_key]
    assert resumed["all_closed"] is False


def test_resume_accounting_fails_closed_on_tampered_existing_cell(synthetic_matrix):
    m, root, manifest_path = synthetic_matrix
    key = m.cells()[0]["output_key"]
    closure_path = root / "cells" / f"{key}.json"
    body = json.loads(closure_path.read_text())
    body["outer_receipt_sha256"] = "0" * 64
    closure_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact path/SHA drift: outer_receipt"):
        m.audit_resume_state(json.loads(manifest_path.read_text()), root, manifest_path)


def test_cell_closure_rejects_missing_state_or_identity_proof():
    m = module()
    cell = m.cells()[0]
    outer = {
        "arm": cell["arm"],
        "outer_loso_fold": cell["fold"],
        "seed": 42,
        "outer_target_session": "ses-RT-synthetic",
        "target_backpropagation": False,
        "model_state_three_point_unchanged": True,
        "matched_query_window_identity": {
            "ses-RT-synthetic": {
                "ordered_window_start_sha256": "a" * 64,
                "ordered_target_covariate_evalmask_sha256": "b" * 64,
                "ordered_query_identity_sha256": "c" * 64,
            }
        },
    }
    good = {
        "selection_receipt_sha256": "a" * 64,
        "config_sha256": "a" * 64,
        "checkpoint_sha256": "a" * 64,
        "split_manifest_sha256": "a" * 64,
        "outer_receipt_sha256": "a" * 64,
        "outer_receipt": outer,
    }
    with pytest.raises(ValueError, match="schema|artifact"):
        m.validate_cell(cell, good)
    outer["model_state_three_point_unchanged"] = False
    with pytest.raises(ValueError):
        m.validate_cell(cell, good)


def test_supervisor_source_never_hardcodes_historical_full_score():
    source = SCRIPT.read_text()
    assert "0.44195" not in source
    assert "experiment=rt_clean_nested_loso_m24" not in source
    assert "--run-id" in source and "if not outer.exists()" in source


def test_readiness_paths_are_portable_root_relative_and_reject_escape():
    m = module()
    relative = m.root_relative(m.RUNNER)
    assert m.resolve_readiness_path(relative) == m.RUNNER.resolve()
    with pytest.raises(ValueError):
        m.resolve_readiness_path(str(m.RUNNER.resolve()))
    with pytest.raises(ValueError):
        m.resolve_readiness_path("../outside")
