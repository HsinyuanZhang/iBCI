"""P1–P5 gates for the repaired P factory. Must fail if wired to old build_p_pair."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tfpd_exploration.src.two_mainlines_long_v1.calibration import constants as C
from tfpd_exploration.src.two_mainlines_long_v1.calibration import factory
from tfpd_exploration.src.two_mainlines_long_v1.calibration import parent_parity
from tfpd_exploration.src.two_mainlines_long_v1.calibration.factory import FORBIDDEN_OLD_FACTORY, build_fresh_arm


ROOT = Path(__file__).resolve().parents[2]


def test_factory_source_rejects_old_pair_path() -> None:
    source = inspect.getsource(build_fresh_arm)
    assert "build_p_pair(" not in source
    assert "global _PAIR" not in source
    assert "_PAIR =" not in source
    assert FORBIDDEN_OLD_FACTORY.endswith("build_p_pair")
    assert inspect.getmodule(build_fresh_arm).__name__.endswith("calibration.factory")


def test_importing_old_build_p_pair_is_the_forbidden_path() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    assert model_adapter.build_p_pair.__module__.endswith("model_adapter")
    assert "global _PAIR" in inspect.getsource(model_adapter)
    assert inspect.getsource(build_fresh_arm).count("load_sfix_student_fresh") >= 1


def test_old_pair_is_global_singleton(monkeypatch) -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    class Dummy:
        pass

    first = Dummy()
    model_adapter._PAIR = None

    def fake_build(repo_root, frozen):
        global_pair = model_adapter._PAIR
        if global_pair is not None:
            return global_pair
        model_adapter._PAIR = first
        return first

    monkeypatch.setattr(model_adapter, "build_p_pair", fake_build)
    a = model_adapter.build_p_pair(ROOT, None)
    b = model_adapter.build_p_pair(ROOT, None)
    assert a is b
    first.marker = 7
    assert b.marker == 7
    model_adapter._PAIR = None


def test_old_resume_is_dict_copy_not_reload() -> None:
    source = inspect.getsource(
        __import__(
            "tfpd_exploration.src.cross_dataset_functional_calibration_v1.model_adapter",
            fromlist=["PConsumerPair"],
        ).PConsumerPair.new_stage_resume_roundtrip
    )
    assert "roundtrip_ok" in source and "True" in source
    assert "load_state_dict" not in source
    assert "torch.load" not in source


def test_old_parent_normalized_is_new_raw_plus_old_mean() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer as sealed_normalizer

    frozen = sealed_normalizer.materialize(ROOT)
    assert parent_parity.old_path_parent_is_new_raw_plus_old_mean(frozen)
    report = parent_parity.independent_parent_parity(ROOT, frozen)
    assert report["used_new_raw_plus_old_mean_as_parent"] is True
    assert report["d0_digest_matches_sealed"] is False
    assert report["target_m10_digest_matches_sealed"] is False
    assert report["passed"] is False
    assert report["red_stop"] is True


def test_p4_does_not_loosen_when_d0_drifts() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer as sealed_normalizer

    frozen = sealed_normalizer.materialize(ROOT)
    report = parent_parity.independent_parent_parity(ROOT, frozen)
    assert report["sealed_d0_digest"] == C.SEALED_D0_DIGEST
    assert report["sealed_target_m10_digest"] == C.SEALED_TARGET_M10_DIGEST
    assert report["estimator_definition_unchanged"] is True
    if not report["d0_digest_matches_sealed"]:
        assert report["passed"] is False


def test_factory_cannot_be_aliased_to_old_pair_without_failing() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    if build_fresh_arm is model_adapter.build_p_pair:
        pytest.fail("fresh factory was replaced by old build_p_pair")
    source = inspect.getsource(build_fresh_arm)
    assert "model_adapter.build_p_pair" not in source
    assert "global _PAIR" not in source


def test_old_p2_loader_calls_eval_and_does_not_seed() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    load_src = inspect.getsource(model_adapter._load_sfix_student)
    loop_src = inspect.getsource(model_adapter._execute_disposable_profile_live)
    spec_src = inspect.getsource(model_adapter.disposable_profile_spec)
    assert "lit.eval()" in load_src
    assert "student.train(" not in loop_src
    assert "manual_seed" not in loop_src
    assert "seed" in spec_src


def test_old_p5_assertions_are_weaker_than_names() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    grad_src = inspect.getsource(model_adapter.PConsumerPair.source_query_loss_dictionary_grad)
    split_src = inspect.getsource(model_adapter.PConsumerPair.query_history_disjointness)
    target_src = inspect.getsource(model_adapter.PConsumerPair.target_fit_no_backward)
    assert "torch.zeros_like(pred)" in grad_src
    assert "del query_labels" in inspect.getsource(model_adapter.PConsumerPair.estimate_support_carrier)
    assert "query_labels=forbidden" in split_src
    assert "window_indices" not in split_src
    assert "learned_state" not in target_src
    assert "before_sha" not in target_src


def test_live_p1_fresh_objects_and_old_path_fails() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer as sealed_normalizer

    frozen = sealed_normalizer.materialize(ROOT)
    model_adapter._PAIR = None
    old_a = model_adapter.build_p_pair(ROOT, frozen)
    old_b = model_adapter.build_p_pair(ROOT, frozen)
    assert old_a is old_b
    fix = build_fresh_arm(ROOT, frozen, "P-FIX", device="cpu", seed=C.FORMAL_SEED)
    ca = build_fresh_arm(ROOT, frozen, "P-CA", device="cpu", seed=C.FORMAL_SEED)
    assert fix.student is not ca.student
    assert fix.basis is not ca.basis
    assert fix.optimizer is not ca.optimizer
    assert fix.consumer_sha256() == ca.consumer_sha256()
    before = ca.consumer_sha256()
    with torch.no_grad():
        fix.student.carrier_projection_weight.add_(0.25)
    assert ca.consumer_sha256() == before
    model_adapter._PAIR = None
