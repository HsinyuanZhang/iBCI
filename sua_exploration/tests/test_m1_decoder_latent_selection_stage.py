"""Synthetic CPU contracts for the sealed M1 DLA selection-stage implementation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration" / "scripts" / "m1_decoder_latent_selection_stage.py"


def module():
    spec = importlib.util.spec_from_file_location("m1_dla_selection", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def features(m, name: str, offset: float = 0.0):
    sums = np.arange(1, 41, dtype=float).reshape(10, 4) + offset
    labels = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
    full = m.support_unit_features(sums, np.arange(1, 11, dtype=float), labels, session_name=name, seed=42)
    shuffled = m.support_unit_features(sums, np.arange(1, 11, dtype=float), labels, session_name=name, seed=42, shuffle_labels=True)
    return {"full": full, "rate_only": full, "label_shuffle": shuffled, "rate_residualized_condition_only": full}


def source_session(m, name: str, offset: float = 0.0):
    e0 = np.zeros((4, 3))
    delta = np.column_stack((np.arange(4) + offset, np.arange(4) * 0.5 + offset, np.full(4, offset)))
    neural = np.arange(24, dtype=float).reshape(6, 4)
    behavior = neural[:, :2] * 0.1
    return m.SourceSelectionSession(name, e0, delta, features(m, name, offset), neural, behavior)


def decoder(neural, identity):
    # Deliberately identity-dependent synthetic frozen decoder.
    return neural[:, :2] * 0.1 + identity.mean(axis=0)[:2]


def test_frozen_feature_layout_masks_and_deterministic_shuffle():
    m = module()
    labels = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
    sums, exposure = np.ones((10, 3)), np.ones(10)
    full = m.support_unit_features(sums, exposure, labels, session_name="a", seed=3)
    shuffle_a = m.deterministic_support_label_shuffle(labels, session_name="a", seed=3)
    shuffle_b = m.deterministic_support_label_shuffle(labels, session_name="a", seed=3)
    assert full.rate.shape == (3, 2) and full.conditioned.shape == (3, 8)
    assert np.all(full.conditioned[:, 4:] == 1.0)
    assert np.array_equal(shuffle_a, shuffle_b) and sorted(shuffle_a) == sorted(labels)
    missing = m.support_unit_features(sums, exposure, np.ones(10, dtype=int), session_name="b", seed=3)
    assert np.all(missing.conditioned[:, 5:] == 0.0)


def test_train_only_transforms_and_svd_ridge_map_are_shape_correct():
    m = module()
    train = [features(m, "a"), features(m, "b", 5.0)]
    transform = m.fit_feature_transform("rate_residualized_condition_only", [item["full"] for item in train])
    assert transform.transform(train[0]["full"]).shape == (4, 8)
    fitted = m.fit_reduced_rank_map("full", [item["full"] for item in train], [np.ones((4, 3)), np.arange(12.0).reshape(4, 3)], rank=2, ridge_lambda=0.01)
    assert fitted.basis_b.shape == (3, 2)
    assert fitted.predict_delta(features(m, "c")["full"]).shape == (4, 3)
    with pytest.raises(ValueError, match="frozen rank"):
        m.fit_reduced_rank_map("full", [item["full"] for item in train], [np.ones((4, 3)), np.ones((4, 3))], rank=4, ridge_lambda=0.01)


def test_inner_loso_excludes_outer_leftout_and_locked_gate_has_no_delta_argument():
    m = module()
    sources = [source_session(m, "a"), source_session(m, "b", 1.0), source_session(m, "c", 2.0)]
    candidates = m.inner_loso_candidates("full", sources, decoder=decoder, outer_left_out_name="outer")
    locked = m.inner_loso_select("full", sources, decoder=decoder, outer_left_out_name="outer")
    assert locked.rank in m.RANK_GRID and locked.ridge_lambda in m.LAMBDA_GRID
    assert locked.mean_delta_r2 == max(item.mean_delta_r2 for item in candidates)
    with pytest.raises(ValueError, match="excludes outer-left-out"):
        m.inner_loso_select("full", sources, decoder=decoder, outer_left_out_name="a")
    outer_neural = np.arange(24.0).reshape(6, 4)
    outer = m.OuterSelectionSession("outer", np.zeros((4, 3)), features(m, "outer", 3.0), outer_neural, outer_neural[:, :2] * 0.1)
    result = m.run_locked_outer_selection_gate("full", sources, outer, locked, decoder=decoder)
    assert result.rank == locked.rank
    assert not hasattr(outer, "delta_star")


def test_selection_window_and_report_barrier_are_fail_closed():
    m = module()
    with pytest.raises(ValueError, match=r"\[10,210\)"):
        m.OuterSelectionSession("bad", np.zeros((4, 3)), features(m, "bad"), np.ones((3, 4)), np.ones((3, 2)), query_window=(210, 300))
    assert m.SEALED_REPORT_START == 210
    assert m.feature_state_accounting("full", 2) == {"support_feature_values": 640, "residual_coefficients": 128, "shared_basis_values": 200, "ridge_weight_values": 22,
                                                       "feature_summary_mac_proxy": 3840, "ridge_coefficient_map_mac": 1408, "low_rank_ABt_map_mac": 12800}


def test_paired_bootstrap_recomputes_aggregate_r2_on_contiguous_trial_blocks():
    m = module()
    target = np.column_stack((np.arange(24.0), np.arange(24.0) * 2.0))
    f0 = target + 1.0
    dla = target + 0.25
    result = m.paired_contiguous_trial_block_bootstrap(f0, dla, target, block_trials=4, replicates=20, seed=7)
    assert result["block_trials"] == 4 and result["replicates"] == 20
    assert result["paired_delta_r2_mean"] > 0.0
    assert result["two_sided_mde_r2"] >= 0.0
    assert len(result["bootstrap_deltas_r2"]) == 20


def test_prelaunch_writer_declares_all_execution_actions_out_of_scope():
    source = (ROOT / "sua_exploration/scripts/write_m1_decoder_latent_selection_prelaunch.py").read_text()
    for forbidden_action in ("sealed_report_reading", "GPU_launch", "heldout_access", "EvalAI_access_or_submission"):
        assert forbidden_action in source


def test_real_runner_is_default_preflight_and_rejects_report_or_heldout_requests():
    runner = ROOT / "sua_exploration/scripts/m1_decoder_latent_selection_runner.py"
    spec = importlib.util.spec_from_file_location("m1_dla_runner", runner)
    assert spec and spec.loader
    r = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r
    spec.loader.exec_module(r)
    with pytest.raises(ValueError, match="exactly held-in query trials"):
        r.assert_selection_only_request(start_trial=210, end_trial=300, heldout=False, evalai=False)
    with pytest.raises(ValueError, match="forbids held-out"):
        r.assert_selection_only_request(start_trial=10, end_trial=210, heldout=True, evalai=False)
    assert r.SELECTION == (10, 210) and r.REPORT_START == 210
    # Exercise the runner's own dynamic import path.  A missing sys.modules
    # registration fails here under Python 3.10 when dataclasses resolve
    # postponed type annotations.
    loaded_api = r._selection_api()
    assert loaded_api.SUPPORT_END == 10 and loaded_api.SELECTION_END == 210
    source = runner.read_text(encoding="utf-8")
    assert 'side_feature_group="d4"' in source
    outer_branch = source[source.index("if name == OUTER_LEFT_OUT"):source.index("else:", source.index("if name == OUTER_LEFT_OUT"))]
    assert "_teacher_delta_for_source_only" not in outer_branch


def test_runner_query_windows_bypass_side_feature_getitem():
    """The DLA scorer may load object IDs, but must never materialize D4 side features."""
    runner = ROOT / "sua_exploration/scripts/m1_decoder_latent_selection_runner.py"
    spec = importlib.util.spec_from_file_location("m1_dla_runner_raw_windows", runner)
    assert spec and spec.loader
    r = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r
    spec.loader.exec_module(r)

    class DatasetThatRejectsGetitem:
        window_size = 2
        window_indices = [("ses-a", 20)]
        trial_start_indices = {"ses-a": np.arange(211, dtype=np.int64) * 2}
        neural_data = {"ses-a": np.arange(66, dtype=np.float32).reshape(22, 3)}
        covariate_data = {"ses-a": np.arange(44, dtype=np.float32).reshape(22, 2)}

        def __getitem__(self, index):
            raise AssertionError("query reader must not invoke the D4 side-feature path")

    neural, behavior, trial_ids = r._query_windows(DatasetThatRejectsGetitem(), "ses-a")
    assert neural.shape == (1, 2, 3)
    assert np.array_equal(behavior, np.asarray([[42.0, 43.0]], dtype=np.float32))
    assert trial_ids.tolist() == [10]
