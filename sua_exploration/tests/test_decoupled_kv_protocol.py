"""Artifact-level contracts for the decoupled K/V screen."""
from __future__ import annotations

import json

import pytest

from scripts.aggregate_sua_decoupled_kv import ARMS, KEY_MODES, aggregate


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_cell(tmp_path, arm: str, score: float, *, cache_bytes: int | None = None):
    metadata_path = tmp_path / f"{arm}_metadata.json"
    decoupled = arm != "coupled_t4"
    decoder = {
        "mode": "decoupled" if decoupled else "coupled",
        "key_mode": KEY_MODES[arm],
        "fixed_slot_count": 0,
        "shared_decoder_base_sha256": "b" * 64,
        "decoder_cost_comparison_receipt_reference_n64": {
            "coupled": {"persistent_state_width": 50},
            **(
                {
                    "decoupled": {
                        "online_mac_reduction_fraction_vs_coupled": 0.80,
                        "persistent_state_nonincreasing_vs_coupled": True,
                    }
                }
                if decoupled
                else {}
            ),
        },
    }
    if decoupled:
        decoder.update({
            "key_width": 32,
            "value_width": 32,
            "num_layers": 1,
            "num_heads": 2,
            "encoder_side_input": "aligned_real_t4",
            "online_cost_receipt_reference_n64": {
                "online_macs_per_frame": {"no_unit_quadratic_term": True}
            },
            "persistent_cache_receipt_reference_n64": (
                {
                    "applicable": False,
                    "cache_bytes": 0,
                }
                if arm == "kv_x_only"
                else {
                    "cache_bytes": 8192 if cache_bytes is None else cache_bytes,
                    "state_nonincreasing_vs_identity": True,
                }
            ),
            "new_projection_init_sha256": "a" * 64,
        })
    metadata = {
        "status": "completed",
        "variant": "B3S",
        "seed": 42,
        "teacher_sha256": "c" * 64,
        "encoder_warmstart_path": None,
        "held_out_test_evaluated": False,
        "side_features": {
            "group": "t4",
            "feature_version": 1,
            "pool_size": 50,
            "normalization_sha256": "d" * 64,
        },
        "training": {
            "calibration_n_trials": 30,
            "max_epochs": 12,
            "no_early_stopping": True,
        },
        "decoder_architecture": decoder,
    }
    _write_json(metadata_path, metadata)
    sessions = [f"session_{index}" for index in range(6)]
    result = {
        "protocol": {
            "calibration_n": 30,
            "pool_size": 50,
            "epoch_window": list(range(5, 13)),
        },
        "run_metadata_path": str(metadata_path),
        "per_epoch": {
            str(epoch): {
                "per_session_r2": {session: score for session in sessions}
            }
            for epoch in range(5, 13)
        },
    }
    _write_json(tmp_path / f"{arm}_m50_s42.json", result)


def test_seed42_aggregate_accepts_strict_matrix_but_not_formal_claim(tmp_path):
    scores = {
        "coupled_t4": 0.40,
        "kv_e_t4": 0.46,
        "kv_e_ts4": 0.42,
        "kv_e_only": 0.45,
        "kv_x_only": 0.39,
    }
    for arm in ARMS:
        _make_cell(tmp_path, arm, scores[arm])
    result = aggregate(tmp_path, (42,))
    assert result["stage0_descriptive_mechanism_pass"] is True
    assert result["stage0_candidate_pass"] == {
        "kv_e_t4": True,
        "kv_e_only": True,
    }
    assert result["formal_effectiveness_eligible"] is False
    assert result["formal_effectiveness_pass"] is False
    assert result["selected_effective_candidate"] is None


def test_aggregate_fails_closed_on_static_cache_receipt_drift(tmp_path):
    for arm in ARMS:
        _make_cell(
            tmp_path,
            arm,
            0.4,
            cache_bytes=9000 if arm == "kv_e_t4" else None,
        )
    with pytest.raises(ValueError, match="n64 cache bytes"):
        aggregate(tmp_path, (42,))
