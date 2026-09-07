"""Focused CPU-only tests for staged M1 fold-1 preparation/evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.m1_compact_replication import fold1_evaluate as evaluate
from sua_exploration.m1_compact_replication import fold1_runner as runner
from sua_exploration.m1_compact_replication import fold1_v2_contract as v2


def test_fold1_commands_are_exactly_bound_to_proposal() -> None:
    proposal = runner.read(runner.PROPOSAL, "proposal")
    b0, b3s = runner._commands(proposal)
    assert b0[2] == "experiment=m1_version_b_hs_continuation"
    assert b3s[2] == "experiment=m1_version_b_c0"
    assert b0[3] == "run_id=m1_compact_b0_f1_s42_fresh_e11"
    assert b3s[3] == "run_id=m1_compact_b3s_zero4_f1_s42_fresh_e11"
    assert "data.source_session_names=[ses-20120924,ses-20120927,ses-20120928]" in b0
    assert "trainer.limit_val_batches=0" in b0 and "trainer.num_sanity_val_steps=0" in b0
    assert b0[-1] == "ckpt_path=null" and b3s[-1] == "ckpt_path=null"


def test_prior_artifact_guard_matches_exact_hydra_prefix(tmp_path: Path) -> None:
    (tmp_path / "m1_compact_b0_f1_s42_fresh_e11_f1_s42_20260810_010000").mkdir()
    assert runner.prior_artifact_exists(tmp_path, "m1_compact_b0_f1_s42_fresh_e11")
    other = tmp_path / "m1_compact_b0_f1_s42_fresh_e11x_f1_s42_20260810_010000"
    other.mkdir()
    assert runner.prior_artifact_exists(tmp_path, "m1_compact_b0_f1_s42_fresh_e11")
    assert not runner.prior_artifact_exists(tmp_path, "m1_compact_b3s_zero4_f1_s42_fresh_e11")


def test_gate_is_noninferiority_and_negative_delta_stops() -> None:
    target = np.arange(40, dtype=np.float32).reshape(20, 2)
    good = target.copy()
    bad = target + 2.0
    delta = evaluate.regression(bad, target)["pooled_variance_weighted_r2"] - evaluate.regression(good, target)["pooled_variance_weighted_r2"]
    assert delta < -0.03
    assert evaluate.THRESHOLD == -0.03


def test_staged_receipt_is_immutable_not_launched() -> None:
    path = runner.DEFAULT_RECEIPT
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["status"] == runner.STATUS_DRY
    assert body["execution"]["launched"] is False
    assert os.stat(path).st_mode & 0o777 == 0o444
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == runner.sha(path)
    assert body["canonical_content_sha256"] == runner.canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"})


def test_execute_requires_explicit_root_token(tmp_path: Path) -> None:
    with pytest.raises(runner.Fold1ContractError):
        runner.execute(runner.DEFAULT_RECEIPT, tmp_path / "state.json", None)


def test_evaluator_is_forward_only_by_contract() -> None:
    source = Path(evaluate.__file__).read_text(encoding="utf-8")
    assert "trainer_constructed" in source
    assert "target_backward_steps" in source
    assert "zero4_prediction_bit_exact" in source
    assert "intermediate_target_metric_read" in source


def test_v2_commands_bind_source_only_fit_and_no_metric() -> None:
    proposal = json.loads(v2.V2_PROPOSAL.read_text(encoding="utf-8"))
    v2._validate_v2_proposal(proposal)
    for cell in proposal["cells"]:
        argv = cell["argv"]
        assert argv[0] == "/home/xinyuan/miniconda3/envs/spint/bin/python"
        assert argv[1].endswith("streaming_calibration_exp/src/train.py")
        assert "data._target_=src.data.m1_version_b_source_loso_datamodule.M1VersionBSourceOnlyFitDataModule" in argv
        assert "test=false" in argv and "test=true" not in argv
        assert "optimized_metric=null" in argv


def test_v2_receipt_excludes_target_data_and_binds_equivalence() -> None:
    receipt = v2._read_immutable(v2.V2_RECEIPT, "v2 receipt", v2.RECEIPT_SCHEMA, v2.RECEIPT_STATUS)
    inventory = receipt["inventory"]
    assert inventory == v2.source_only_inventory()
    assert inventory["source_data_file_count"] == 3
    assert "data_files_sha256" not in inventory
    assert inventory["target_data_hashed_by_prelaunch"] is False
    assert "streaming_calibration_exp/src/data/m1_version_b_source_loso_datamodule.py" in inventory["code_files_sha256"]
    binding = v2.validate_equivalence_binding(v2.V2_RECEIPT)
    assert binding["equivalence"]["train_dataset_equal"]
    assert binding["equivalence"]["sampler_equal"]
    assert binding["equivalence"]["zero4_side_exact"]
    assert binding["equivalence"]["target_nwb_opened"] is False


def test_source_fit_equivalence_receipt_is_positive_and_target_guarded() -> None:
    path = v2.EQUIVALENCE_RECEIPT
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["status"] == "PASS_M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENT"
    assert body["train_dataset"]["equal"] is True
    assert body["sampler"]["equal"] is True
    assert body["source_only_guard"]["target_nwb_opened"] is False
    assert body["source_only_guard"]["target_path_attribute_absent"] is True
