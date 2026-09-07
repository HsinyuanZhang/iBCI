"""Static contracts for the not-yet-integrated RT strong-LS-v2 plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PLAN = WORKSPACE / "sua_exploration/results/rt_afc4_xls_v2_integration_draft_v1/RT_AFC4_XLS_V2_MATCHED_PLAN_DRAFT_v1.json"
BLUEPRINT = WORKSPACE / "sua_exploration/docs/RT_AFC4_XLS_V2_INTEGRATION_BLUEPRINT_20260808.md"


def test_draft_binds_the_exact_passing_v2_audit_and_remains_unlaunched() -> None:
    body = json.loads(PLAN.read_text(encoding="utf-8"))
    assert body["schema"] == "rt_afc4_xls_v2_matched_plan_draft_v1"
    assert body["status"] == "DRAFT_NOT_LAUNCHED_NO_ACTIVE_INTEGRATION"
    audit = WORKSPACE / body["v2_audit_binding"]["path"]
    assert hashlib.sha256(audit.read_bytes()).hexdigest() == body["v2_audit_binding"]["sha256"]
    audit_rows = json.loads(audit.read_text(encoding="utf-8"))["fold_rows"]
    assert body["v2_audit_binding"]["per_session_expected_permutation_sha256"] == {
        row["session_name"]: row["v2_random_cross_reach_null"]["permutation_sha256"] for row in audit_rows
    }
    assert body["v2_audit_binding"]["seed"] == 42
    assert body["arm"]["side_dim"] == 4
    assert "source-or-target shared 2x2 inverse" in body["arm"]["forbidden_compensation"]


def test_draft_freezes_matched_15_fold_no_target_bp_protocol_after_mb4() -> None:
    body = json.loads(PLAN.read_text(encoding="utf-8"))
    protocol = body["frozen_protocol"]
    assert protocol["folds"] == list(range(15)) and protocol["seed"] == 42
    assert protocol["support_trial_index_range"] == [0, 24]
    assert protocol["query_start_trial"] == 24 and protocol["window_size_bins"] == 50
    assert protocol["checkpoint_selection"] == "inner validation only: val_heldin/r2_mean"
    assert "no target backpropagation" in protocol["outer_evaluation"]
    assert "run all 15 folds" in protocol["stop_rule"]
    prerequisites = body["execution_prerequisites"]["required_before_any_xls_v2_gpu_launch"]
    assert any("MB4" in item and "15/15" in item for item in prerequisites)
    roots = body["new_roots_only"]
    assert len(set(roots.values())) == 3


def test_blueprint_names_all_deferred_integration_surfaces_and_no_inverse_escape_hatch() -> None:
    text = BLUEPRINT.read_text(encoding="utf-8")
    for required in (
        "normalizer", "split manifest", "outer evaluator", "Hydra", "supervisor + finalizer",
        "共同 `2×2` inverse", "`R-C − XLSv2`", "side_dim=4", "one-shot",
    ):
        assert required in text
    assert "不修改" in text and "不启动" in text
