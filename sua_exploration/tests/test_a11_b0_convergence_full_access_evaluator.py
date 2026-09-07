"""Focused unit tests for the A11 B0 exact-source CPU evaluator.

These deliberately use synthetic paths/payloads only.  They do not load a
checkpoint, create an NWB reader, access the six sealed formal-test sessions,
or invoke a GPU.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import a11_b0_convergence_full_access_evaluator as a11  # noqa: E402


def test_fixed_scope_is_exactly_three_b0_seeds_and_twelve_checkpoints():
    assert a11.SEEDS == (42, 43, 44)
    assert a11.LIGHTNING_EPOCHS == tuple(range(12))
    assert a11.PROTOCOL_EPOCHS == tuple(range(5, 13))
    assert a11.VARIANT == "B0"
    assert a11.CALIBRATION_N == 30
    assert a11.POOL_SIZE == 30


def test_behavior_normalizer_cache_key_matches_historical_canonical_payload(tmp_path):
    train_paths = []
    for index in range(2):
        path = tmp_path / f"train_{index}.nwb"
        path.write_bytes(bytes([index]) * (index + 3))
        train_paths.append(path)

    payload = {
        "cache_format_version": 1,
        "kind": "behavior_stats",
        "bin_size_ms": 20,
        "train_sources": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in train_paths
        ],
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert a11.behavior_stats_cache_key(train_paths) == expected
    assert a11.normalizer_cache_path(tmp_path / "cache", train_paths).name == f"{expected[:20]}.npz"


def test_query_contract_hash_changes_if_the_query_boundary_changes():
    original = a11.query_contract()
    mutated = dict(original)
    mutated["excluded_prefix_pool_trials"] = 31

    assert original["calibration_selection_mode"] == "first"
    assert original["query_trial_rule"] == "usable_rewarded_trials[30:]"
    assert a11.sha256_json(original) != a11.sha256_json(mutated)


def test_full_plan_has_every_seed_and_every_epoch_but_smoke_cannot_make_science_claims():
    full = a11._full_plan()
    assert full["science_claim_allowed"] is True
    assert full["epochs_by_seed"] == {"42": list(range(12)), "43": list(range(12)), "44": list(range(12))}

    smoke = a11._smoke_plan()
    assert smoke["science_claim_allowed"] is False
    assert smoke["epochs_by_seed"] == {"42": [0], "43": [], "44": []}


def test_immutable_receipt_refuses_overwrite_and_writes_matching_sidecar(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    payload = {"program_id": "synthetic", "status": "completed", "answer": 42}

    receipt, sidecar, digest = a11.write_immutable_json(receipt_path, payload)

    assert receipt == receipt_path
    assert sidecar == Path(f"{receipt_path}.sha256")
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == digest
    assert sidecar.read_text(encoding="ascii") == f"{digest}  receipt.json\n"
    with pytest.raises(a11.AuthorityError, match="will not be overwritten"):
        a11.write_immutable_json(receipt_path, {"replacement": True})


def _complete_driver_payload() -> dict:
    per_seed = {}
    for seed in a11.SEEDS:
        per_epoch = {}
        for epoch in a11.LIGHTNING_EPOCHS:
            value = seed / 1000.0 + epoch / 100.0
            per_epoch[str(epoch)] = {
                "mean_r2": value,
                "per_session_r2": {"synthetic_session": value},
                "model_state_sha256_before": "same",
                "model_state_sha256_after": "same",
            }
        per_seed[str(seed)] = {"per_epoch": per_epoch}
    return {"per_seed": per_seed}


def test_full_summary_uses_exactly_human_epochs_five_through_twelve():
    summary = a11.summarize_full_curve(_complete_driver_payload())

    assert summary["epoch_5_to_12_average_supported"] is True
    assert summary["extension_candidate_diagnostic"]["candidate_flag"] is True
    assert summary["extension_candidate_diagnostic"]["authorizes_training_or_continuation"] is False
    # Seed 42 scores lightning epochs 4..11, i.e. human epochs 5..12.
    expected = sum(42 / 1000.0 + epoch / 100.0 for epoch in range(4, 12)) / 8
    assert summary["per_seed"]["42"]["epoch_5_to_12_mean_validation_r2"] == pytest.approx(expected)
    assert summary["per_seed"]["42"]["best_human_epoch"] == 12


def test_extension_candidate_is_false_without_three_positive_material_deltas():
    payload = _complete_driver_payload()
    # Flatten seed 44 so its late-minus-early delta is exactly zero; a positive
    # aggregate trend in the other seeds must not pass the all-seed gate.
    for epoch in a11.LIGHTNING_EPOCHS:
        value = 0.044
        payload["per_seed"]["44"]["per_epoch"][str(epoch)]["mean_r2"] = value
        payload["per_seed"]["44"]["per_epoch"][str(epoch)]["per_session_r2"] = {
            "synthetic_session": value
        }
    summary = a11.summarize_full_curve(payload)
    assert summary["extension_candidate_diagnostic"]["candidate_flag"] is False
    assert summary["extension_candidate_diagnostic"]["per_seed_late_minus_early"]["44"] == pytest.approx(0.0)


def test_full_summary_fails_closed_on_missing_checkpoint_epoch():
    payload = _complete_driver_payload()
    del payload["per_seed"]["44"]["per_epoch"]["11"]

    with pytest.raises(a11.AuthorityError, match="full curve requires epochs"):
        a11.summarize_full_curve(payload)


def test_compatibility_driver_contains_cpu_and_validation_scope_guards():
    source = a11.COMPATIBILITY_DRIVER_SOURCE

    assert 'torch.device("cpu")' in source
    assert 'torch.set_grad_enabled(False)' in source
    assert "A11 data-scope violation" in source
    assert "normalizer_cache_path" in source
    assert "model_state_sha256_before" in source
    assert "load_frozen_model" in source
