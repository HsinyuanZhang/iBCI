"""CPU-only contract checks for the staged, not-launched fold-2 v2 plan."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sua_exploration.m1_compact_replication import fold2_v2_contract as c
from sua_exploration.m1_compact_replication import fold2_sampler_audit as audit


def test_fold2_proposal_exact_source_only_commands() -> None:
    proposal = json.loads(c.V2_PROPOSAL.read_text(encoding="utf-8"))
    c._validate_proposal(proposal)
    for cell in proposal["cells"]:
        argv = cell["argv"]
        assert "test=false" in argv and "test=true" not in argv
        assert "optimized_metric=null" in argv
        assert f"data._target_={c.SOURCE_ONLY_TARGET}" in argv
        assert "data.loso_fold=2" in argv
        assert "data.source_session_names=[ses-20120924,ses-20120926,ses-20120928]" in argv


def test_fold2_receipt_waits_for_fold1_and_excludes_target_data() -> None:
    body = c.build_receipt()
    assert body["status"] == c.RECEIPT_STATUS
    assert body["fold1_gate"]["required"] is True
    assert body["fold1_gate"]["available"] is False
    assert body["fold1_gate"]["launch_authorized"] is False
    assert body["inventory"]["source_data_file_count"] == 3
    assert body["inventory"]["target_data_hashed_by_prelaunch"] is False
    assert os.stat(c.V2_RECEIPT).st_mode & 0o777 == 0o444
    assert body["canonical_content_sha256"] == c.base.canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"})
    assert hashlib.sha256(c.V2_RECEIPT.read_bytes()).hexdigest() == c.base.sha(c.V2_RECEIPT)


def test_fold2_source_sampler_derives_its_own_terminal_step() -> None:
    counts = {"ses-20120924": 1714, "ses-20120926": 1702, "ses-20120928": 1711}
    value = audit.derive_sampler_cardinality(
        counts, batch_size=32, scored_windows=164064, dataset_windows=164108
    )
    assert value["source_batches_per_epoch"] == 5127
    assert value["expected_global_step"] == 61524
    assert value["dropped_windows_per_epoch"] == 44
    assert "59412" not in value["derivation"]


def test_fold2_sampler_rejects_fold0_session_set() -> None:
    with pytest.raises(RuntimeError):
        audit.derive_sampler_cardinality(
            {"ses-20120926": 1702, "ses-20120927": 1538, "ses-20120928": 1711},
            batch_size=32,
            scored_windows=158432,
            dataset_windows=158487,
        )
