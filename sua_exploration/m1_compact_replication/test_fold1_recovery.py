"""CPU contracts for the append-only fold-1 B0 recovery path."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sua_exploration.m1_compact_replication import fold1_recovery as recovery


def _equivalence(counts: dict[str, int], windows: int) -> dict:
    return {
        "sampler": {
            "equal": True,
            "source_batch_counts": counts,
            "legacy_batch_counts": counts,
            "source_windows": windows,
            "legacy_windows": windows,
            "source_sha256": "source",
            "legacy_sha256": "legacy",
        }
    }


def test_fold1_steps_are_derived_from_bound_source_sampler() -> None:
    counts = {"ses-20120924": 1714, "ses-20120927": 1538, "ses-20120928": 1711}
    contract = recovery.derive_sampler_contract(_equivalence(counts, 158816), batch_size=32, epochs=12)
    assert contract["source_batches_per_epoch"] == 4963
    assert contract["source_scored_windows_per_epoch"] == 158816
    assert contract["expected_global_step"] == 59556
    assert "fold-0" not in contract["derivation"]


def test_fold0_literal_is_not_reused_for_fold1() -> None:
    counts = {"ses-20120926": 1702, "ses-20120927": 1538, "ses-20120928": 1711}
    # The helper deliberately rejects a session set that is not the bound
    # fold-1 source set, rather than silently accepting the old 59412 value.
    with pytest.raises(Exception):
        recovery.derive_sampler_contract(_equivalence(counts, 158432), batch_size=32, epochs=12)


def test_recovery_is_append_only_and_b3s_only() -> None:
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    assert "retrain_b0" in source
    assert "launch_only" in source
    assert 'staged["commands"]["b3s_zero4"]' in source
    assert 'staged["commands"]["b0"]' not in source


def test_recovered_state_schema_is_separate_from_failed_v2() -> None:
    assert recovery.STATE_SCHEMA != "m1_compact_b3s_f1_s42_execution_v2"
    assert recovery.RECOVERY_STATUS.endswith("PENDING_B3S")

