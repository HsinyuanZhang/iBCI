"""Focused CPU-only contracts for the staged M1 fold-2 v2 pair."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sua_exploration.m1_compact_replication import fold1_runner as fold1_runner
from sua_exploration.m1_compact_replication import fold2_evaluate as evaluator
from sua_exploration.m1_compact_replication import fold2_runner_v2 as runner
from sua_exploration.m1_compact_replication import fold2_v2_contract as contract


def test_fold2_proposal_and_receipt_are_exact_and_waiting() -> None:
    proposal = json.loads(contract.V2_PROPOSAL.read_text(encoding="utf-8"))
    contract._validate_proposal(proposal)
    assert proposal["status"] == contract.V2_STATUS
    for cell in proposal["cells"]:
        assert cell["argv"][0] == contract.PYTHON
        assert cell["argv"][1].endswith("streaming_calibration_exp/src/train.py")
        assert "test=false" in cell["argv"] and "test=true" not in cell["argv"]
        assert "optimized_metric=null" in cell["argv"]
        assert f"data._target_={contract.SOURCE_ONLY_TARGET}" in cell["argv"]
    receipt = json.loads(contract.V2_RECEIPT.read_text(encoding="utf-8"))
    runner._validate_staged(receipt, contract.V2_RECEIPT)
    assert receipt["status"] == contract.RECEIPT_STATUS
    assert receipt["fold1_gate"] == {
        "required": True,
        "available": False,
        "path": str((contract.ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_GATE_v2.json").resolve()),
        "launch_authorized": False,
    }
    assert os.stat(contract.V2_RECEIPT).st_mode & 0o777 == 0o444
    assert receipt["canonical_content_sha256"] == fold1_runner.canonical(
        {k: v for k, v in receipt.items() if k != "canonical_content_sha256"}
    )
    assert hashlib.sha256(contract.V2_RECEIPT.read_bytes()).hexdigest() == fold1_runner.sha(contract.V2_RECEIPT)


def test_missing_fold1_gate_fails_closed_without_training(tmp_path: Path) -> None:
    missing = tmp_path / "M1_COMPACT_B3S_F1_S42_GATE_v2.json"
    with pytest.raises(runner.Fold2ContractError, match="gate missing"):
        runner.validate_fold1_gate_file(missing)
    staged = json.loads(contract.V2_RECEIPT.read_text(encoding="utf-8"))
    staged["fold1_gate"]["path"] = str(missing)
    with pytest.raises(runner.Fold2ContractError, match="frozen fold-1 v2 gate"):
        runner.validate_fold1_gate(staged)


def test_fold1_runner_cannot_accept_fold2_waiting_receipt() -> None:
    staged = fold1_runner.read(contract.V2_RECEIPT, "fold2 receipt")
    with pytest.raises(fold1_runner.Fold1ContractError):
        # This is a read-only guard: no execute path is called.
        from sua_exploration.m1_compact_replication import fold1_runner_v2

        fold1_runner_v2._validate_staged(staged, contract.V2_RECEIPT)


def test_fold2_evaluator_is_cpu_forward_only_and_target_gated() -> None:
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "trainer_constructed" in source
    assert "optimizer_constructed" in source
    assert "target_opened_here_after_terminal_pair" in source
    assert "zero4_prediction_bit_exact" in source
    assert "fold1_gate" in source
    assert "target_backward_steps" in source


def test_fold2_runner_binds_gate_to_fold1_receipt() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "fold1_receipt = fold1_contract.V2_RECEIPT.resolve()" in source
    assert '"fold-1 gate receipt binding drift"' in source


def test_fold2_has_no_execution_state_or_prior_artifact() -> None:
    state = contract.ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_EXECUTION_v2.json"
    assert not state.exists()
    output_root = contract.ROOT / "outputs/streaming_calibration"
    assert not fold1_runner.prior_artifact_exists(output_root, "m1_compact_b0_f2_s42_fresh_e11")
    assert not fold1_runner.prior_artifact_exists(output_root, "m1_compact_b3s_zero4_f2_s42_fresh_e11")
