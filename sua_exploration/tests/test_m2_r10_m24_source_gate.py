"""Synthetic/no-NWB tests for the M2 R10 M24 source gate."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))

from mc_maze.m2_r10_m24_source_gate import (  # noqa: E402
    CHANNELS,
    RANDOM_SCHEDULES,
    UINT32_MAX,
    future_autocorr,
    l10,
    mc,
    null_features,
    paired,
    r10,
    schedule_receipt,
    score_loso,
    split_reliability,
    streaming_receipt,
)

SCRIPT = SUA / "scripts/audit_m2_r10_m24_source_gate.py"


def load_runner(name: str = "m2_r10_runner"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def support_trials(offset: int, channels: int = 16) -> tuple[np.ndarray, ...]:
    return tuple(
        (
            (
                np.arange((20 + trial_index % 5) * channels).reshape(
                    20 + trial_index % 5, channels
                )
                + offset
                + 3 * trial_index
            )
            % 11
        ).astype(np.int64)
        for trial_index in range(24)
    )


def target_record(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    target = np.asarray(values, dtype=np.float64).copy()
    target[~mask] = np.nan
    return {"target": target, "mask": np.asarray(mask, dtype=bool)}


def test_r10_l10_exact_shapes_reference_and_order_invariance():
    trials = support_trials(1)
    r10_value = r10(trials)
    l10_value = l10(trials)
    assert r10_value.shape == (16, 10)
    assert l10_value.shape == (16, 10)
    rates = np.stack(
        [np.log((trial.sum(axis=0) + 0.5) / (0.020 * trial.shape[0])) for trial in trials]
    )
    reference = np.quantile(
        rates,
        (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95),
        axis=0,
        method="linear",
    ).T
    assert np.allclose(r10_value, reference)
    reordered = tuple(trial[::-1] for trial in reversed(trials))
    assert np.allclose(r10_value, r10(reordered))
    assert np.allclose(l10_value, l10(reordered))
    with pytest.raises(ValueError, match="exactly 24"):
        r10(trials[:23])


def test_raw_count_and_uint32_overflow_fail_closed():
    trials = list(support_trials(1, channels=2))
    trials[0] = np.asarray([[UINT32_MAX, 0], [1, 0]], dtype=np.uint64)
    with pytest.raises(OverflowError, match="uint32"):
        r10(tuple(trials))
    trials = list(support_trials(1, channels=2))
    trials[0] = trials[0].astype(float)
    trials[0][0, 0] = 0.5
    with pytest.raises(ValueError, match="integer-count"):
        l10(tuple(trials))


def test_future_autocorr_exact_numeric_and_whole_row_mask():
    alternating = np.where(np.arange(200) % 2 == 0, 1, 3)
    trial = np.stack((alternating, np.ones(200, dtype=int)), axis=1)
    result = future_autocorr((trial, trial.copy()))
    assert result["target"].shape == (2, 8)
    assert result["pair_exposure"][0, 0] == 398
    assert result["pair_exposure"][0, -1] == 144
    assert result["target"][0, 0] == pytest.approx(-1.0)
    assert result["target"][0, 1] == pytest.approx(1.0)
    assert result["mask"][0].all()
    assert not result["mask"][1].any()
    short = future_autocorr((trial[:100], trial[:100]))
    assert not short["mask"].any()


def test_positive_seven_session_loso_fixture():
    rng = np.random.default_rng(7)
    weight = rng.normal(size=(10, 8))
    features = {}
    targets = {}
    for session_index in range(7):
        feature = rng.normal(size=(CHANNELS, 10))
        target = feature @ weight + 0.001 * rng.normal(size=(CHANNELS, 8))
        name = f"s{session_index}"
        features[name] = feature
        targets[name] = target_record(target)
    scores = score_loso(features, targets)
    assert len(scores) == 7
    assert min(float(row["r2"]) for row in scores) > 0.99
    assert {row["left_out_session"] for row in scores} == set(features)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("feature_shape", "shape-finiteness"),
        ("target_shape", "shape-finiteness"),
        ("numeric_mask", "boolean"),
        ("partial_row", "whole-row"),
        ("too_few_defined", "below 0.90"),
    ],
)
def test_loso_shape_mask_and_defined_fraction_guards(mutation: str, match: str):
    features = {f"s{i}": np.ones((CHANNELS, 10)) for i in range(7)}
    targets = {f"s{i}": target_record(np.ones((CHANNELS, 8))) for i in range(7)}
    if mutation == "feature_shape":
        features["s0"] = np.ones((95, 10))
    elif mutation == "target_shape":
        targets["s0"] = target_record(np.ones((CHANNELS, 7)))
    elif mutation == "numeric_mask":
        targets["s0"]["mask"] = np.ones((CHANNELS, 8), dtype=np.int64)
    elif mutation == "partial_row":
        targets["s0"]["target"][0, 0] = np.nan
        targets["s0"]["mask"][0, 0] = False
    elif mutation == "too_few_defined":
        targets["s0"]["target"][:10] = np.nan
        targets["s0"]["mask"][:10] = False
    with pytest.raises(ValueError, match=match):
        score_loso(features, targets)


def test_paired_operational_gate_and_invalid_inputs():
    result = paired(np.full(7, 0.5), np.full(7, 0.4))
    assert result["mean"] == pytest.approx(0.1)
    assert result["operational_mde80"] == pytest.approx(0.0)
    assert result["operational_interval_95"][0] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="seven"):
        paired(np.ones(6), np.zeros(6))
    bad = np.ones(7)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        paired(bad, np.zeros(7))


def test_full_permutation_schedule_is_lossless_deterministic_and_allows_fixed_points():
    sessions = tuple(f"s{i}" for i in range(7))
    first = schedule_receipt(sessions, 4)
    second = schedule_receipt(sessions, 4)
    assert first["random_schedules"] == RANDOM_SCHEDULES
    assert first["all_vectors_complete_permutations"] is True
    assert first["full_plan_sha256"] == second["full_plan_sha256"]
    assert len(first["complete_permutation_vectors_by_schedule"]) == RANDOM_SCHEDULES
    for schedule in first["complete_permutation_vectors_by_schedule"]:
        assert set(schedule) == set(sessions)
        assert all(set(vector) == set(range(4)) for vector in schedule.values())
    assert sum(first["per_session_identity_draw_counts"].values()) > 0
    all_fixed_counts = [
        count
        for schedule in first["fixed_point_count_by_schedule_session"].values()
        for count in schedule.values()
    ]
    assert any(count > 0 for count in all_fixed_counts)
    assert null_features({"s0": np.arange(40).reshape(4, 10)}, 1)["s0"].shape == (4, 10)


def test_mc_exact_arithmetic_ties_and_invalid_values():
    null = np.full(RANDOM_SCHEDULES, 0.2)
    result = mc(0.9, null)
    assert result["p_attach"] == pytest.approx(1 / 4096)
    assert result["mc_upper_97p5"] < 0.025
    tied = mc(0.2, null)
    assert tied["exceedances"] == RANDOM_SCHEDULES
    for invalid in (null[:-1], np.r_[null[:-1], np.nan], np.r_[null[:-1], 0.0]):
        with pytest.raises(ValueError, match="4095"):
            mc(0.9, invalid)
    with pytest.raises(ValueError, match="observed"):
        mc(0.0, null)


def test_thinning_and_streaming_state_contracts():
    result = split_reliability(support_trials(3), "s0", repeats=4)
    assert result["repeats"] == 4
    assert len(result["median_standardized_row_cosine"]) == 4
    assert np.isfinite(result["per_coordinate_pearson"]).all()
    state = streaming_receipt(CHANNELS)
    assert state["accumulator_bytes"] == 10040
    assert state["peak_fp32_bytes"] == 13880
    assert state["raw_bin_retained"] is False


def make_guard_files(runner, tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    expected = runner._expected_hashes()
    formal = runner._formal_schedule_contract()
    prelaunch = tmp_path / "prelaunch.json"
    prelaunch.write_text(
        json.dumps(
            {
                "schema_version": "m2_r10_m24_source_prelaunch_v5",
                "status": "pending_root_review_no_nwb_opened",
                "protocol": {"sha256": expected["protocol_sha256"]},
                "manifest": {"sha256": expected["manifest_sha256"]},
                "prior_input_receipt": {"sha256": expected["prior_input_receipt_sha256"]},
                "code": {
                    "runner": expected["runner_sha256"],
                    "pure": expected["pure_sha256"],
                    "production_falcon_dataset": expected["falcon_dataset_sha256"],
                },
                "hard_exclusions": {
                    key: False
                    for key in ("nwb_opened", "cuda", "held_out", "decoder", "gpu", "evalai")
                },
                "formal_null": formal,
            }
        )
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "authorization": runner.REVIEW,
                "prelaunch_receipt_sha256": runner.sha(prelaunch),
                "formal_schedule_plan_sha256": formal["full_plan_sha256"],
                **expected,
            }
        )
    )
    return prelaunch, review


def test_review_prelaunch_and_current_hash_guards(tmp_path: Path, monkeypatch):
    runner = load_runner("m2_r10_guard_runner")
    formal = {
        "seed_namespace": runner.SEED,
        "distribution": "independent uniform full S_96 per session",
        "random_schedules": runner.RANDOM_SCHEDULES,
        "fixed_points_identity_draws_and_duplicates_retained": True,
        "full_plan_sha256": "1" * 64,
    }
    monkeypatch.setattr(runner, "_formal_schedule_contract", lambda: formal)
    prelaunch, review = make_guard_files(runner, tmp_path)
    assert runner.check(review, prelaunch)["runner_sha256"] == runner.sha(SCRIPT)
    review_value = json.loads(review.read_text())
    review_value["pure_sha256"] = "0" * 64
    review.write_text(json.dumps(review_value))
    with pytest.raises(PermissionError, match="pure_sha256"):
        runner.check(review, prelaunch)
    prelaunch, review = make_guard_files(runner, tmp_path / "second")
    prelaunch_value = json.loads(prelaunch.read_text())
    prelaunch_value["code"]["runner"] = "f" * 64
    prelaunch.write_text(json.dumps(prelaunch_value))
    review_value = json.loads(review.read_text())
    review_value["prelaunch_receipt_sha256"] = runner.sha(prelaunch)
    review.write_text(json.dumps(review_value))
    with pytest.raises(PermissionError, match="stale prelaunch runner"):
        runner.check(review, prelaunch)

    prelaunch, review = make_guard_files(runner, tmp_path / "third")
    prelaunch_value = json.loads(prelaunch.read_text())
    prelaunch_value["formal_null"]["random_schedules"] = 4094
    prelaunch.write_text(json.dumps(prelaunch_value))
    review_value = json.loads(review.read_text())
    review_value["prelaunch_receipt_sha256"] = runner.sha(prelaunch)
    review.write_text(json.dumps(review_value))
    with pytest.raises(PermissionError, match="formal-null"):
        runner.check(review, prelaunch)


def fake_loader_dependencies(runner, tmp_path: Path, *, bad: str | None = None):
    rows = []
    for index in range(7):
        relative = f"data/session_{index}.nwb"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}".encode())
        rows.append(
            {
                "session": f"s{index}",
                "relative_path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    def fake_load_nwb(path, task):
        del path, task
        neural = np.ones((10, CHANNELS), dtype=np.uint8)
        covariates = np.full((10, 2), 17.0)
        changes = np.zeros(10, dtype=bool)
        changes[0] = True
        mask = np.ones(10, dtype=bool)
        return neural, covariates, changes, mask

    class FakeDataset:
        saw_zero_covariates = False

        def __init__(self, *, sessions_dict, calib_sessions_dict, **kwargs):
            del kwargs
            assert sessions_dict is calib_sessions_dict
            FakeDataset.saw_zero_covariates = all(
                np.count_nonzero(value["covariates"]) == 0 for value in sessions_dict.values()
            )
            assert all(
                np.issubdtype(value["neural"].dtype, np.signedinteger)
                and value["neural"].dtype.itemsize >= 4
                for value in sessions_dict.values()
            )
            self.calib_trialized_neural = {}
            self.calib_trial_lengths = {}
            self.calib_trial_spike_sums = {}
            for session in sessions_dict:
                padded = np.full((25, 5, CHANNELS), -1.0)
                padded[:, :3] = 1.0
                lengths = np.full(25, 3, dtype=np.int64)
                sums = np.full((25, CHANNELS), 3.0)
                if bad == "padding" and session == "s0":
                    padded[0, 4, 0] = 0.0
                if bad == "sum" and session == "s0":
                    sums[0, 0] = 4.0
                if bad == "truncation" and session == "s0":
                    lengths[0] = 1024
                self.calib_trialized_neural[session] = padded
                self.calib_trial_lengths[session] = lengths
                self.calib_trial_spike_sums[session] = sums

    return {"sessions": rows}, fake_load_nwb, FakeDataset


def test_mocked_production_loader_padding_sum_and_covariate_discard(tmp_path: Path, monkeypatch):
    runner = load_runner("m2_r10_loader_runner")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    specification, load_nwb, dataset = fake_loader_dependencies(runner, tmp_path)
    raw, receipts = runner._load_after_review(
        specification,
        load_nwb_fn=load_nwb,
        dataset_cls=dataset,
        falcon_task="m2",
    )
    assert dataset.saw_zero_covariates is True
    assert len(raw) == 7 and all(len(trials) == 25 for trials in raw.values())
    assert all(value["all_padding_tails_exact_minus_one"] for value in receipts.values())
    assert all(value["all_prefix_sums_match_production"] for value in receipts.values())
    assert all(value["signed_neural_dtype_before_falcon_dataset"] == "int64" for value in receipts.values())


def test_geometry_only_trial_mask_ignores_behavior_side_masks():
    runner = load_runner("m2_r10_geometry_runner")
    timestamps = np.arange(20, dtype=float) * 0.02
    starts = np.asarray([0.04, 0.20])
    stops = np.asarray([0.12, 0.30])
    trial_change, geometry, receipt = runner._geometry_trial_mask(timestamps, starts, stops)
    arbitrary_eval_mask_a = np.arange(20) % 2 == 0
    arbitrary_eval_mask_b = ~arbitrary_eval_mask_a
    assert not np.array_equal(arbitrary_eval_mask_a, arbitrary_eval_mask_b)
    assert np.flatnonzero(trial_change).tolist() == [2, 10]
    assert np.flatnonzero(geometry).tolist() == [2, 3, 4, 5, 10, 11, 12, 13, 14]
    assert receipt["geometry_mask_true_bins"] == 9
    source = inspect.getsource(runner._load_m2_neural_geometry)
    assert "acquisition[\"eval_mask\"]" not in source
    assert "acquisition['eval_mask']" not in source
    assert "to_dataframe" not in source
    assert "get_timeseries(label).data" not in source
    assert 'trials["start_time"]' in source and 'trials["stop_time"]' in source


@pytest.mark.parametrize("bad,match", [("padding", "padding"), ("sum", "prefix sum"), ("truncation", "truncation")])
def test_mocked_production_loader_fail_closed(
    bad: str, match: str, tmp_path: Path, monkeypatch
):
    runner = load_runner(f"m2_r10_bad_loader_{bad}")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    specification, load_nwb, dataset = fake_loader_dependencies(runner, tmp_path, bad=bad)
    with pytest.raises(ValueError, match=match):
        runner._load_after_review(
            specification,
            load_nwb_fn=load_nwb,
            dataset_cls=dataset,
            falcon_task="m2",
        )


def test_prelaunch_static_receipt_does_not_open_nwb(tmp_path: Path, monkeypatch):
    runner = load_runner("m2_r10_prelaunch_runner")
    monkeypatch.setattr(
        runner,
        "bench",
        lambda: {
            "within_24h": True,
            "cuda_visible_devices": "",
            "formal_schedule_plan_sha256": "a" * 64,
            "formal_schedule_complete": True,
            "projection_seconds": 1.0,
        },
    )
    record = runner.receipt()
    assert record["hard_exclusions"]["nwb_opened"] is False
    assert record["formal_null"]["distribution"] == "independent uniform full S_96 per session"
    path = runner.write_prelaunch(tmp_path / "prelaunch")
    assert path.is_file()
    assert path.with_name("prelaunch_receipt.sha256").is_file()
    source = SCRIPT.read_text()
    assert "rglob(" not in source
    assert 'CUDA_VISIBLE_DEVICES"] = ""' in source


def test_guard_failure_cannot_call_loader_and_writes_terminal(tmp_path: Path):
    runner = load_runner("m2_r10_guard_terminal_runner")
    prelaunch = tmp_path / "pre.json"
    review = tmp_path / "review.json"
    prelaunch.write_text("{}")
    review.write_text("{}")
    calls = 0

    def forbidden_loader(_):
        nonlocal calls
        calls += 1
        raise AssertionError("loader must not be called")

    path = runner.execute_with_terminal(
        tmp_path / "guard_failure", prelaunch, review, loader=forbidden_loader
    )
    result = json.loads(path.read_text())
    assert calls == 0
    assert result["decision"] == "r10_invalid_execution_stop"
    assert result["contracts"]["source_loader_called"] is False
    assert result["error"]["stage"] == "pre_source_guard"


def test_deadline_and_invalid_execution_write_terminal_receipts(tmp_path: Path, monkeypatch):
    runner = load_runner("m2_r10_terminal_runner")
    monkeypatch.setattr(runner, "check", lambda review, prelaunch: {})
    prelaunch = tmp_path / "pre.json"
    review = tmp_path / "review.json"
    prelaunch.write_text("{}")
    review.write_text("{}")

    names = [row["session"] for row in runner.manifest()["sessions"]]

    def legal_loader(_):
        raw = {
            name: tuple(np.ones((2, CHANNELS), dtype=np.int64) for _ in range(25))
            for name in names
        }
        receipts = {name: {"behavior_covariates_discarded_before_dataset": True} for name in names}
        return raw, receipts

    deadline_output = tmp_path / "deadline"
    deadline_path = runner.execute_with_terminal(
        deadline_output,
        prelaunch,
        review,
        deadline_seconds=0.0,
        loader=legal_loader,
    )
    deadline_result = json.loads(deadline_path.read_text())
    assert deadline_result["decision"] == "r10_null_budget_infeasible_stop"
    assert deadline_result["contracts"]["valid_execution"] is False
    assert set(deadline_result["binding"]) == {
        "protocol",
        "manifest",
        "prior_input_receipt",
        "falcon_dataset",
        "runner",
        "pure",
        "prelaunch",
        "root_review",
    }

    def invalid_loader(_):
        raise ValueError("synthetic raw failure")

    invalid_path = runner.execute_with_terminal(
        tmp_path / "invalid",
        prelaunch,
        review,
        loader=invalid_loader,
    )
    invalid_result = json.loads(invalid_path.read_text())
    assert invalid_result["decision"] == "r10_invalid_execution_stop"
    assert invalid_result["error"]["message"] == "synthetic raw failure"
