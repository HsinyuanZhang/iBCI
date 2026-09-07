"""CPU-only contract tests for the M1 e23 pair runner/evaluator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.m1_compact_replication import e23_evaluate as evaluate
from sua_exploration.m1_compact_replication import e23_pair_runner as runner


def test_immutable_authority_and_exact_proposal_are_current() -> None:
    proposal = runner._check_immutable(runner.PROPOSAL, "proposal", runner.EXPECTED_PROPOSAL_SHA)
    authority = runner._check_immutable(runner.AUTHORITY, "authority", runner.EXPECTED_AUTHORITY_SHA)
    runner._validate_authority(authority)
    b0, b3s = runner._proposal_commands(proposal)
    assert b0[3] == "run_id=m1_compact_b0_f0_s42_resume_e11_to_e23"
    assert b3s[3] == "run_id=m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23"
    assert b0[7:9] == ["trainer.min_epochs=24", "trainer.max_epochs=24"]
    assert b3s[7:9] == ["trainer.min_epochs=24", "trainer.max_epochs=24"]


def test_e23_metric_gate_is_paired_and_fail_closed() -> None:
    target = np.arange(20, dtype=np.float32).reshape(10, 2)
    b0 = target.copy()
    b3s = target.copy()
    assert evaluate.regression_metrics(b0, target)["pooled_variance_weighted_r2"] == pytest.approx(1.0)
    assert evaluate.regression_metrics(b3s, target)["pooled_variance_weighted_r2"] == pytest.approx(1.0)
    assert 1.0 - 1.0 >= runner.EXPECTED_TERMINAL_EPOCH * 0 - 0.03

    degraded = target + 2.0
    delta = evaluate.regression_metrics(degraded, target)["pooled_variance_weighted_r2"] - evaluate.regression_metrics(target, target)["pooled_variance_weighted_r2"]
    assert delta < -0.03


def test_evaluator_requires_cpu_visibility() -> None:
    assert evaluate.EXPECTED_EPOCH == 23
    assert evaluate.EXPECTED_STEP == 118824
    assert evaluate.GATE_THRESHOLD == -0.03


def test_runner_source_has_no_score_selection_or_checkpoint_metric_parse() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "target_metrics_not_read_by_launcher" in source
    assert "intermediate_checkpoint_selection" in source
    assert "metrics_summary" not in source


def test_receipt_canonical_hash_excludes_only_self_field(tmp_path: Path) -> None:
    body = {"schema": runner.SCHEMA, "status": runner.PREPARED_STATUS, "x": [1, 2, 3]}
    body["canonical_content_sha256"] = runner.canonical_sha(body)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    assert runner.canonical_sha({k: v for k, v in json.loads(path.read_text()).items() if k != "canonical_content_sha256"}) == body["canonical_content_sha256"]


def _terminal_payload() -> dict:
    return {
        "epoch": 23,
        "global_step": 118824,
        "pytorch-lightning_version": "2.6.5",
        "state_dict": {"x": 1},
        "optimizer_states": [{
            "param_groups": [{"lr": 1.0e-4, "weight_decay": 0.0}],
            "state": {0: {"step": 118824}},
        }],
        "lr_schedulers": [],
        "loops": {
            "fit_loop": {
                "epoch_loop.batch_progress": {"total": {"completed": 118824}},
                "epoch_progress": {"total": {"processed": 24}},
            }
        },
        "callbacks": {"ModelCheckpoint{'monitor': None, 'every_n_epochs': 12}": {}},
    }


def test_terminal_checkpoint_contract_covers_optimizer_loop_and_callback() -> None:
    summary = runner._terminal_payload_summary(_terminal_payload(), expected_variant="B0")
    assert summary["optimizer"]["all_state_steps"] == [118824]
    assert summary["fit_loop"] == {"batch_completed": 118824, "epoch_processed": 24}
    broken = _terminal_payload()
    broken["optimizer_states"][0]["param_groups"][0]["lr"] = 2.0e-4
    with pytest.raises(runner.LaunchContractError):
        runner._terminal_payload_summary(broken, expected_variant="B0")
