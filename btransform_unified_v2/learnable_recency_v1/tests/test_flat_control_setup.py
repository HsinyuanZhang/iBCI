"""Prepared-only contracts for the FULL learned-recency flat control."""
from __future__ import annotations

import sys
from copy import deepcopy

import pytest

from learnable_recency_v1.flat_control import (
    TASKS,
    build_manifest,
    flat_config,
    verify_manifest,
    verify_zero_temporal,
)


@pytest.mark.parametrize("task", TASKS)
def test_flat_config_is_the_fixed_all_zero_d4_control(task: str) -> None:
    config = flat_config(task)
    assert config.tier == "fixed"
    assert config.half_life_seconds == (None,) * 8
    assert config.ladder == "default"
    assert config.per_layer is True
    assert config.layers == 4
    proof = verify_zero_temporal(task)
    assert proof["shape"] == [8]
    assert proof["zero_count"] == 8


@pytest.mark.parametrize("task", TASKS)
def test_prepared_manifest_is_reference_bound_and_parser_compatible(task: str) -> None:
    manifest = build_manifest(task, python_executable=sys.executable, device="cpu")
    assert manifest["status"] == "PREPARED_NOT_TRAINED"
    assert manifest["training_started"] is False
    assert manifest["arm"] == "FULL_FLAT"
    assert manifest["effective_bias"] == "allzero"
    assert manifest["flat_config"]["half_life_seconds"] == [None] * 8
    assert manifest["seed"] == 42 and manifest["proj_dim"] == 16
    assert manifest["batch"] == 32 and manifest["total_updates"] == manifest["epochs"] * manifest["updates_per_epoch"]
    assert len(manifest["reference"]["run_meta_sha256"]) == 64
    assert len(manifest["reference"]["selection_sha256"]) == 64
    assert any(path.endswith("btransform_unified_v2/temporal.py") for path in manifest["source_hashes"])
    assert any(path.endswith("learnable_recency_v1/flat_control.py") for path in manifest["source_hashes"])
    assert manifest["results_dir"].endswith(f"{task}_projadd_flat_p16_s42" if task != "h1" else "h1_flat_p16_s42")
    if task == "m2":
        assert manifest["selection_dir"].endswith("selection_m2_projadd_flat_p16_s42_ext6")
        assert "--half-lives" not in manifest["score_argv"]
    else:
        assert manifest["selection_dir"] is None
        assert "--half-lives" in manifest["score_argv"]
    verify_manifest(manifest)


def test_manifest_rejects_train_tier_seed_and_half_life_drift() -> None:
    manifest = build_manifest("m1", python_executable=sys.executable, device="cpu")
    for index, replacement in ((3, "learned_slope"), (12, "43"), (5, "0.08")):
        broken = deepcopy(manifest)
        broken["train_argv"][index] = replacement
        with pytest.raises(ValueError, match="train argv drift"):
            verify_manifest(broken)


def test_manifest_rejects_formal_epoch_and_smoke_drift() -> None:
    manifest = build_manifest("m1", python_executable=sys.executable, device="cpu")
    for option, value in (("--epochs", "1"), ("--max-updates-smoke", "1")):
        broken = deepcopy(manifest)
        broken["train_argv"].extend((option, value))
        with pytest.raises(ValueError, match="argv drift"):
            verify_manifest(broken)
