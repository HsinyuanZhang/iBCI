"""Focused score-free regressions for the sub-M CO score-only runner.

All numerical values below are hard-coded synthetic metric fixtures.  These
tests never load a C1 checkpoint, instantiate a model, call ``R2Score``, open
an NWB, use a GPU, or perform an optimizer/backward operation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze import subm_co_score_only as score_only  # noqa: E402


def test_metadata_only_dry_run_freezes_n15_without_scoring() -> None:
    plan = score_only.build_dry_run_plan(ROOT)
    assert plan["status"] == "NOT_AUTHORIZED_FOR_SCORING"
    assert plan["frozen_N"] == 15
    assert len(plan["frozen_asset_ids"]) == 15
    assert plan["candidate_arms"] == ["shared_t4", "shared_ts4"]
    assert plan["seeds"] == [42, 43, 44]
    counters = plan["audit_counters"]
    assert counters["checkpoint_hash_checks"] == 0
    assert counters["checkpoint_load_calls"] == 0
    assert counters["model_forward_calls"] == 0
    assert counters["r2_update_calls"] == 0
    assert counters["r2_compute_calls"] == 0
    assert counters["gpu_attempts_blocked"] == 0
    assert plan["static_audit"]["forbidden_imports"] == []
    assert plan["static_audit"]["forbidden_calls"] == []


def test_prelaunch_draft_and_receipt_are_explicitly_non_authorizing_and_append_only(tmp_path: Path) -> None:
    output = tmp_path / "prelaunch"
    result = score_only.write_prelaunch_artifacts(output, ROOT)
    assert result["status"] == "NOT_AUTHORIZED_FOR_SCORING"
    draft = json.loads((output / "prelaunch_authorization_draft.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert draft["status"] == "NOT_AUTHORIZED_FOR_SCORING"
    assert draft["cohort_policy"]["N"] == 15
    assert draft["cohort_policy"]["cohort_rediscovery_or_reselection"] == "FORBIDDEN"
    assert draft["authorization_required_before_scoring"]["this_draft_is_an_authorization"] is False
    assert receipt["status"] == "NOT_AUTHORIZED_FOR_SCORING"
    assert receipt["dry_run_audit"]["checkpoint_files_opened"] == 0
    assert receipt["dry_run_audit"]["model_forward_calls"] == 0
    assert receipt["dry_run_audit"]["r2_computations"] == 0
    with pytest.raises(score_only.ScoreOnlyContractError, match="already exists"):
        score_only.write_prelaunch_artifacts(output, ROOT)


def test_prelaunch_receipt_cannot_be_used_as_future_authorization(tmp_path: Path) -> None:
    draft, receipt = score_only.build_prelaunch_receipt(ROOT)
    authorization = tmp_path / "not_authorization.json"
    authorization.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(score_only.ScoreOnlyContractError, match="status"):
        score_only.require_future_authorization(
            authorization_path=authorization,
            output_root=tmp_path / "fresh-output",
            permitted_action="seal_session_predictions_metrics",
            root=ROOT,
            output_root_must_be_new=True,
            counters=score_only.RuntimeAuditCounters(),
        )
    assert draft["status"] == "NOT_AUTHORIZED_FOR_SCORING"


def test_hard_coded_metric_fixture_reproduces_all_n15_endpoint_gates() -> None:
    # This does endpoint arithmetic over precomputed fixture scalar values only;
    # it deliberately does not calculate an R2 from predictions or targets.
    t4 = np.full((3, 15), 0.20, dtype=np.float64)
    ts4 = np.full((3, 15), 0.15, dtype=np.float64)
    summary = score_only.summarize_endpoint_view(t4, ts4)
    assert summary["N"] == 15
    assert summary["grand_paired_mean_delta"] == pytest.approx(0.05)
    assert summary["required_positive_session_cross_seed_means"] == 12
    assert summary["positive_session_cross_seed_means"] == 15
    assert summary["hierarchical_bootstrap"]["replicates"] == 100_000
    assert summary["hierarchical_bootstrap"]["rng"] == "numpy.random.Generator(numpy.random.PCG64(68820260805))"
    assert summary["hierarchical_bootstrap"]["quantile_method"] == "linear"
    assert summary["view_pass"] is True
    assert all(summary["gates"].values())


def test_hard_coded_metric_fixture_fails_exact_plus_003_gate_without_reselection() -> None:
    # Still no R2 computation: these are fixed synthetic scalar metric inputs.
    t4 = np.full((3, 15), 0.20, dtype=np.float64)
    ts4 = np.full((3, 15), 0.175, dtype=np.float64)
    summary = score_only.summarize_endpoint_view(t4, ts4)
    assert summary["grand_paired_mean_delta"] == pytest.approx(0.025)
    assert summary["gates"]["grand_paired_mean_at_least_0_03"] is False
    assert summary["gates"]["all_three_seed_mean_deltas_strictly_positive"] is True
    assert summary["view_pass"] is False


def test_sealed_synthetic_fixture_requires_complete_fixed_matrix_and_does_not_aggregate(tmp_path: Path) -> None:
    # NPZ arrays are tiny fixed test fixtures, not model outputs.  The test
    # exercises the write-once seal/index shape and then reads metrics only.
    contract = score_only.validate_authority_chain(ROOT)
    authorization = tmp_path / "fixture_authorization.json"
    authorization.write_text('{"fixture_only": true}\n', encoding="utf-8")
    output = tmp_path / "sealed-output"
    writer = score_only.SealedOutputWriter(output, contract, authorization)
    counters = score_only.RuntimeAuditCounters()
    for session in contract.frozen_sessions:
        for view in score_only.VIEWS:
            for arm in score_only.ARMS:
                score = 0.20 if arm == "shared_t4" else 0.15
                for seed in score_only.SEEDS:
                    terminal = contract.checkpoint_by_key()[(arm, seed)]
                    permutation = (
                        None
                        if arm == "shared_t4"
                        else {
                            "fixture_only": True,
                            "seed": seed,
                            "identity_is_retained_not_excluded": True,
                        }
                    )
                    writer.write_session_result(
                        session=session,
                        view=view,
                        arm=arm,
                        seed=seed,
                        checkpoint_sha256=terminal.checkpoint.sha256,
                        normalizer=contract.normalizers[view],
                        r2_value=score,
                        predictions=np.zeros((1, 2), dtype=np.float32),
                        targets=np.zeros((1, 2), dtype=np.float32),
                        query_window_count=1,
                        ts4_permutation=permutation,
                        audit_counters=counters,
                    )
    seal = writer.finalize(counters)
    assert seal["status"] == "SEALED_PER_SESSION_PREDICTIONS_AND_METRICS"
    assert counters.sealed_session_metric_writes == 180
    assert not (output / "aggregate").exists()
    grids, evidence = score_only._read_sealed_metric_grid(output, contract)
    assert evidence["nonfinite_cells"] == []
    assert grids["sua"]["shared_t4"].shape == (3, 15)
    assert np.allclose(grids["pseudo_mua"]["shared_t4"], 0.20)
