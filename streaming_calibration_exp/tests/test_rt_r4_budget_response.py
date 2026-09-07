"""Focused fail-closed tests for RT R4 common-q24 preparation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_rt_r4_budget_response as prepare
import src.data.rt_nested_loso_datamodule as nested
from src.data.falcon_k4_features import k4_from_raw_calibration
from src.data.rt_r4_budget_response_datamodule import (
    RtR4BudgetResponseError,
    RtR4BudgetResponseNestedLossoDataModule,
    fit_r4_prefix_descriptor,
)


SESSION_NAMES = (
    "ses-RT-20131009", "ses-RT-20131010", "ses-RT-20131011",
    "ses-RT-20131028", "ses-RT-20131029", "ses-RT-20131209",
    "ses-RT-20131210", "ses-RT-20131212", "ses-RT-20131213",
    "ses-RT-20131217", "ses-RT-20131218", "ses-RT-20150316",
    "ses-RT-20150317", "ses-RT-20150318", "ses-RT-20150320",
)


def _path(name: str) -> Path:
    return Path(f"sub-C_{name}_behavior+ecephys.nwb")


def _session(name: str, units: int = 4) -> dict:
    trials, bins = 32, 20
    total = trials * bins
    rng = np.random.RandomState(sum(name.encode("utf-8")))
    trial_change = np.zeros(total, dtype=bool)
    velocity = np.zeros((total, 2), dtype=np.float32)
    segment = np.empty(total, dtype=np.int64)
    for trial in range(trials):
        left = trial * bins
        trial_change[left] = True
        velocity[left:left + bins] = np.asarray(
            [0.35 + 0.02 * trial, -0.25 + 0.07 * (trial % 5)], dtype=np.float32,
        )
        segment[left:left + bins] = trial
    neural = rng.poisson(
        0.15 + 0.03 * velocity[:, :1] + 0.02 * (velocity[:, 1:2] + 0.3),
        size=(total, units),
    ).astype(np.float32)
    return {
        "session_name": name,
        "neural": neural,
        "covariates": velocity,
        "trial_change": trial_change,
        "eval_mask": np.ones(total, dtype=bool),
        "k4_segment_id": segment,
        "rt_segment_audit": {
            "complete_cue_trials": trials,
            "accepted_reach_segments": trials,
            "event_qualified_bins": total,
            "trial_records": [
                {"trial_index": trial, "complete_cue": True,
                 "accepted_segments": 1, "declared_segments": 1,
                 "excluded_segments": 0, "exclusion_reason": None,
                 "segment_exclusion_reasons": {}}
                for trial in range(trials)
            ],
        },
        "rt_velocity_audit": {"loader_standardization": "none"},
    }


def _patch_source_loader(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr(nested, "find_rt_sessions", lambda _data_dir: [_path(n) for n in SESSION_NAMES])

    def load(path: Path) -> dict:
        name = nested.session_name_from_path(path)
        calls.append(name)
        return _session(name)

    monkeypatch.setattr(nested, "load_rt_session", load)


def _module(monkeypatch: pytest.MonkeyPatch, *, budget: int, arm: str = "afc4_vel"):
    calls: list[str] = []
    _patch_source_loader(monkeypatch, calls)
    dm = RtR4BudgetResponseNestedLossoDataModule(
        data_dir="/synthetic/rt",
        batch_size=2,
        window_size=50,
        calibration_n_trials=24,
        side_feature_calibration_n_trials=budget,
        query_start_trial=24,
        rt_r4_common_query_start=True,
        allow_conditional_m18=False,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=100,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        validation_protocol="nested_loso",
        loso_fold=0,
        outer_loso_fold=0,
        side_feature_group=arm,
        side_feature_shuffle_seed=42,
        session_window_budget=4,
        expected_session_count=15,
        num_workers=0,
        pin_memory=False,
        sampler_seed=42,
    )
    dm.setup("fit")
    return dm, calls


def test_r4_keeps_activity_m24_and_query_q24_but_fits_carrier_from_m6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dm, loaded = _module(monkeypatch, budget=6)
    assert dm.split.outer_target_session == SESSION_NAMES[0]
    assert dm.split.outer_target_session not in loaded
    assert len(set(loaded)) == 14
    sample = dm.train_dataset[0]
    assert sample[2].shape[0] == 24  # neural/activity calibration tensor
    assert sample[4].shape[1] == 4
    assert all(audit["calibration_trials"] == 6
               for audit in dm.train_dataset.k4_audits.values())
    manifest = dm.get_split_manifest()
    r4 = manifest["rt_r4_budget_response"]
    assert r4["activity_calibration_trials"] == 24
    assert r4["carrier_calibration_trials"] == 6
    assert r4["common_query_start_trial"] == 24
    assert r4["unused_for_carrier_and_query_trial_range"] == [6, 24]
    assert r4["only_carrier_fit_prefix_varies"] is True
    assert manifest["nested_selection"]["outer_target_loaded_during_fit"] is False
    audits = [*manifest["source_query_window_audit"].values(),
              *manifest["inner_validation_query_window_audit"].values()]
    assert len(audits) == 14
    assert all(row["query_start_trial"] == 24 and row["full_window_disjoint"]
               and row["eligible_windows"] > 0 for row in audits)


def test_r4_full_and_mb4_share_raw_prefix_and_only_mask_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full, _ = _module(monkeypatch, budget=12, arm="afc4_vel")
    mb4, _ = _module(monkeypatch, budget=12, arm="afc4_mb4")
    name = full.split.inner_train_sessions[0]
    np.testing.assert_array_equal(
        full.train_dataset.k4_raw_features[name], mb4.train_dataset.k4_raw_features[name]
    )
    full_side = full.train_dataset[0][4]
    mb4_side = mb4.train_dataset[0][4]
    np.testing.assert_array_equal(mb4_side[:, :2], np.zeros_like(mb4_side[:, :2]))
    np.testing.assert_allclose(full_side[:, 2:], mb4_side[:, 2:], rtol=0.0, atol=0.0)


def test_r4_prefix_helper_matches_production_at_m24() -> None:
    raw_session = _session(SESSION_NAMES[0])
    raw = {
        "neural": raw_session["neural"],
        "covariates": raw_session["covariates"],
        "trial_change": raw_session["trial_change"],
        "segment_ids": raw_session["k4_segment_id"],
    }
    diagnostic, audit = fit_r4_prefix_descriptor(raw, carrier_budget_trials=24)
    production, production_audit = k4_from_raw_calibration(
        raw["neural"], raw["covariates"], raw["trial_change"],
        calibration_n_trials=24, segment_ids=raw["segment_ids"],
    )
    np.testing.assert_array_equal(diagnostic, production)
    assert audit["design_rank"] == production_audit.design_rank == 3
    assert audit["active_blocks"] == production_audit.active_blocks


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"calibration_n_trials": 12}, "activity calibration must remain M24"),
        ({"query_start_trial": 12}, "common evaluation start must remain trial24"),
        ({"side_feature_group": "zero4"}, "permits only the matched Full/MB4"),
        ({"side_feature_calibration_n_trials": 18}, "conditional M18 requires"),
    ],
)
def test_r4_rejects_budget_or_arm_drift(changes: dict, match: str) -> None:
    kwargs = {
        "data_dir": "/synthetic/rt", "calibration_n_trials": 24,
        "side_feature_calibration_n_trials": 6, "query_start_trial": 24,
        "rt_r4_common_query_start": True, "allow_conditional_m18": False,
        "random_calibration": False, "smooth_calibration": False,
        "loso_fold": 0, "outer_loso_fold": 0, "side_feature_group": "afc4_vel",
    }
    kwargs.update(changes)
    with pytest.raises(RtR4BudgetResponseError, match=match):
        RtR4BudgetResponseNestedLossoDataModule(**kwargs)


def test_prepare_plan_is_source_only_and_freezes_three_fold_two_stage_pilot(tmp_path: Path) -> None:
    plan = prepare.build_plan(output=tmp_path / "prepare.json")
    assert plan["fixed_protocol"]["activity_neural_calibration_trials"] == 24
    assert plan["fixed_protocol"]["carrier_fit_budgets_primary"] == [6, 12]
    assert plan["fixed_protocol"]["common_query_start_trial"] == 24
    assert plan["two_stage_execution"]["pilot_folds_fixed_before_results"] == [0, 7, 14]
    assert plan["two_stage_execution"]["pilot_cells"] == 12
    assert len(plan["two_stage_execution"]["pilot_commands_not_executed"]) == 12
    assert plan["two_stage_execution"]["per_budget_expansion_gate"] == {
        "positive_pilot_fold_deltas_required": "3/3",
        "mean_full_minus_mb4_at_least_r2": 0.03,
        "paired_by": ["budget", "fold", "seed", "query_start_trial"],
    }
    assert plan["two_stage_execution"]["expansion_folds_if_and_only_if_that_budget_passes"] == [
        1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13,
    ]
    assert plan["two_stage_execution"]["m18_policy"]["automatic_launch"] is False
    assert plan["scope"] == {
        "nwb_files_opened": 0,
        "outer_target_payloads_opened": 0,
        "outer_scores_computed": 0,
        "trainer_constructed": False,
        "optimizer_constructed": False,
        "cuda_queried": False,
        "gpu_processes_started": 0,
        "pilot_commands_executed": 0,
        "gpu_authorized": False,
        "outer_evaluation_authorized": False,
    }
    assert plan["existing_evidence"]["q24_constructibility"][
        "all_source_and_inner_validation_sessions_have_q24_windows"
    ] is True
    path, digest = prepare.write_immutable(tmp_path / "prepare.json", plan)
    assert path.stat().st_mode & 0o777 == 0o444
    body = json.loads(path.read_text())
    assert body["status"] == prepare.STATUS
    assert len(digest) == 64
