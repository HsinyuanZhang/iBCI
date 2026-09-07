"""No-NWB contracts for the actual-but-guarded Step-3 source CPU Gate-A runner."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.m1_fixed_k_prototype_gate_a import (  # noqa: E402
    M1RawSession,
    category_mean_log1p_hz,
    disjoint_binomial_partition,
    extract_valid_raw_trials,
    future_neural_oracle_target,
    paired_session_summary,
    strict_gate,
    validate_obj_coverage,
)

SCRIPT = ROOT / "sua_exploration" / "scripts" / "audit_m1_fixed_k_prototypes_gate_a.py"
MANIFEST = ROOT / "sua_exploration" / "manifests" / "m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json"


def _module():
    spec = importlib.util.spec_from_file_location("m1_fixed_k_prototype_gate_a_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _padded_trials() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.full((2, 5, 64), -1.0)
    raw[0, :2] = 1.0
    raw[1, :3] = 2.0
    lengths = np.asarray([2, 3])
    sums = np.stack([raw[0, :2].sum(axis=0), raw[1, :3].sum(axis=0)])
    return raw, lengths, sums


def test_raw_valid_prefix_extraction_rejects_padding_and_interpolation_and_preserves_sums():
    padded, lengths, sums = _padded_trials()
    trials = extract_valid_raw_trials(padded, lengths, sums)
    assert [trial.shape for trial in trials] == [(2, 64), (3, 64)]
    assert all(np.issubdtype(trial.dtype, np.integer) for trial in trials)
    bad_tail = padded.copy(); bad_tail[0, 2, 0] = 0.0
    with pytest.raises(ValueError, match="tail"):
        extract_valid_raw_trials(bad_tail, lengths, sums)
    bad_interpolated = padded.copy(); bad_interpolated[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="non-integer"):
        extract_valid_raw_trials(bad_interpolated, lengths, sums)
    bad_sum = sums.copy(); bad_sum[0, 0] += 1.0
    with pytest.raises(ValueError, match="disagrees"):
        extract_valid_raw_trials(padded, lengths, bad_sum)


def test_manifest_locks_exact_paths_hashes_counts_and_excludes_non_source_scopes():
    module = _module()
    manifest = module.load_manifest(MANIFEST)
    assert [row["session"] for row in manifest["sessions"]] == [
        "ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"
    ]
    assert [row["support_obj_id_counts_0_10"] for row in manifest["sessions"]] == [
        [3, 2, 3, 2], [2, 3, 3, 2], [3, 3, 2, 2], [3, 2, 2, 3]
    ]
    assert [row["future_obj_id_counts_210_end"] for row in manifest["sessions"]] == [
        [70, 46, 41, 47], [64, 46, 38, 51], [59, 50, 26, 31], [72, 45, 37, 48]
    ]
    assert all("held-in-calib" in row["relative_path"] and len(row["sha256"]) == 64 for row in manifest["sessions"])


def test_future_target_is_log1p_and_future_obj_ids_are_scorer_only_boundary():
    support = tuple(np.ones((2, 64), dtype=np.int64) for _ in range(10))
    labels = np.asarray([1, 2, 3, 4] * 3, dtype=np.int64)[:10]
    future_counts = np.vstack([np.full(64, level, dtype=np.int64) for level in (1, 2, 3, 4)])
    session = M1RawSession(
        session="synthetic", support_trials=support, future_trial_counts=future_counts,
        future_trial_bins=np.ones(4, dtype=np.int64), support_obj_ids=labels,
        future_obj_ids=np.asarray([1, 2, 3, 4]), source_path="source", source_sha256="0" * 64,
    )
    target = future_neural_oracle_target(session)
    expected = np.log1p(np.asarray([50.0, 100.0, 150.0, 200.0]))
    assert target.shape == (64, 4)
    assert np.allclose(target[0], expected)
    assert validate_obj_coverage(session.future_obj_ids, name="future", expected_counts=[1, 1, 1, 1]).tolist() == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="exactly obj_id"):
        validate_obj_coverage(np.asarray([1, 1, 2, 2]), name="future")


def test_disjoint_binomial_resampling_is_exactly_conservative_within_each_trial_bin():
    trials = (np.asarray([[1, 2], [3, 0]], dtype=np.int64), np.asarray([[4, 5]], dtype=np.int64))
    left, right = disjoint_binomial_partition(trials, rng=np.random.default_rng(7))
    for original, first, second in zip(trials, left, right):
        assert np.array_equal(first + second, original)
        assert np.all(first >= 0) and np.all(second >= 0)


def test_strict_gate_requires_both_controls_mde_ci_and_contracts():
    positive = [2.0, 2.0, 2.0, 2.0]
    passed = strict_gate(positive, positive, repeatability_contract_pass=True, oracle_contract_pass=True, state_contract_pass=True)
    assert passed["decision"] == "pass_for_separate_decoder_review"
    failed = strict_gate(positive, [-1.0, -1.0, -1.0, -1.0], repeatability_contract_pass=True, oracle_contract_pass=True, state_contract_pass=True)
    assert failed["decision"] == "stop_cpu_gate_not_met"
    summary = paired_session_summary(positive)
    assert summary["n_sessions"] == 4 and summary["ci95"][0] == pytest.approx(2.0)


def test_prelaunch_is_no_nwb_pending_review_and_fresh_output(tmp_path):
    module = _module()
    receipt = module.build_prelaunch_receipt()
    assert receipt["status"] == "pending_root_review_no_nwb_opened"
    assert receipt["hard_exclusions"]["formal_paths_resolved"] is False
    assert receipt["execution_guard"]["fresh_output_required"] is True
    path = module.write_prelaunch(tmp_path / "prelaunch")
    loaded = json.loads(path.read_text())
    assert loaded["endpoint"]["target_range"] == [210, "end"]
    assert (path.parent / "prelaunch_receipt.sha256").is_file()
    with pytest.raises(FileExistsError):
        module.write_prelaunch(path.parent)


def test_exact_loader_has_no_nonmanifest_fallback_path(monkeypatch, tmp_path):
    module = _module()
    allowed = tmp_path / "sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb"
    rejected = tmp_path / "sub-MonkeyL-held-in-minival_ses-20120924_behavior+ecephys.nwb"
    allowed.touch(); rejected.touch()
    calls: list[Path] = []

    def guarded_load_nwb(path, task):
        actual = Path(path)
        calls.append(actual)
        if actual != allowed:
            raise AssertionError(f"non-manifest load_nwb path attempted: {actual}")
        return (
            np.zeros((10, 64), dtype=np.int64), np.zeros((10, 1), dtype=np.float64),
            np.ones(10, dtype=bool), np.ones(10, dtype=bool),
        )

    monkeypatch.setattr(module, "load_nwb", guarded_load_nwb)
    labels = lambda path, task: np.asarray([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=np.int64)
    item = module._exact_source_session_dict(allowed, session="ses-20120924", label_loader=labels)
    assert item["neural"].shape == (10, 64) and calls == [allowed]
    with pytest.raises(AssertionError, match="non-manifest"):
        module._exact_source_session_dict(rejected, session="minival", label_loader=labels)


def test_exact_source_object_builder_and_static_runner_guard_have_no_discovery_paths(tmp_path):
    module = _module()
    names = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
    paths = {name: tmp_path / f"{name}.nwb" for name in names}
    calls: list[tuple[str, Path]] = []

    def builder(path, *, session):
        calls.append((session, Path(path)))
        assert session in paths and path == paths[session]
        return {"source_session": session}

    objects = module._source_objects_from_exact_paths(paths, builder=builder)
    assert list(objects) == list(names)
    assert calls == [(name, paths[name]) for name in names]
    source = SCRIPT.read_text()
    assert "FalconDataModule" not in source
    assert ".setup(" not in source
    assert "rglob(" not in source


def test_repeatability_thresholds_and_half_rate_rescaling_are_nonvacuous(monkeypatch):
    module = _module()
    anchors = np.zeros((4, 4), dtype=np.float64)
    trials = tuple(np.ones((1, 2), dtype=np.int64) for _ in range(10))
    captured: list[tuple[np.ndarray, ...]] = []

    def fake_carrier(_anchors, fed_trials):
        captured.append(tuple(np.asarray(trial).copy() for trial in fed_trials))
        return {"prototype_blocks": np.ones((2, 4, 5), dtype=np.float64)}

    monkeypatch.setattr(module, "carrier_from_support_trials", fake_carrier)
    count_cos, value_cos, _, _ = module._cosine_repeatability(anchors, trials, seed=4)
    assert count_cos == pytest.approx(1.0) and value_cos == pytest.approx(1.0)
    assert len(captured) == 2
    assert all(np.all((trial == 0) | (trial == 2)) for group in captured for trial in group)

    session = M1RawSession("s0", trials, np.ones((4, 2), dtype=np.int64), np.ones(4, dtype=np.int64),
                           np.asarray([1, 2, 3, 4, 1, 2, 3, 4, 1, 2]), np.asarray([1, 2, 3, 4]), "p", "0" * 64)
    fold = {"s0": {"anchor_receipt": {"anchor_values": anchors.tolist()}}}
    monkeypatch.setattr(module, "REPEATABILITY_RESAMPLES", 3)
    monkeypatch.setattr(module, "_cosine_repeatability", lambda *args, **kwargs: (0.49, 0.8, 2, 2))
    _, failed_count = module.repeatability_receipt({"s0": session}, fold)
    assert failed_count is False
    monkeypatch.setattr(module, "_cosine_repeatability", lambda *args, **kwargs: (0.8, 0.8, 2, 1))
    _, failed_defined = module.repeatability_receipt({"s0": session}, fold)
    assert failed_defined is False
    monkeypatch.setattr(module, "_cosine_repeatability", lambda *args, **kwargs: (0.8, 0.8, 2, 2))
    rows, passed = module.repeatability_receipt({"s0": session}, fold)
    assert passed is True and rows["s0"]["minimum_value_defined_fraction"] == pytest.approx(1.0)
