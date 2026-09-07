from __future__ import annotations

import numpy as np
import torch

from mc_maze.cebra_pseudosession import SizeMatchedPseudoSessionDataset
from mc_maze.multisession_datamodule import (
    Dandi688MultiSessionDataset,
    SessionBatchSampler,
    SessionRecord,
)


def _record(name: str, n_units: int, offset: float) -> SessionRecord:
    time = 28
    neural = np.stack([
        np.arange(time, dtype=np.float32) + offset + 1000.0 * unit
        for unit in range(n_units)
    ], axis=1)
    behavior = np.stack([
        np.linspace(-1.0, 1.0, time, dtype=np.float32) + offset * 1e-3,
        np.linspace(1.0, -1.0, time, dtype=np.float32) - offset * 1e-3,
    ], axis=1)
    calib = np.stack([neural[:5] + 10_000.0 * trial for trial in range(3)], axis=0)
    side = np.stack([
        np.asarray([offset, unit, offset + unit, 1.0], dtype=np.float32)
        for unit in range(n_units)
    ])
    return SessionRecord(
        name=name,
        neural=neural,
        behavior=behavior,
        calib_trials=calib,
        valid_starts=np.asarray([0, 3, 6, 9, 12, 15, 18], dtype=np.int64),
        side_features=side,
    )


def _base() -> Dandi688MultiSessionDataset:
    sessions = {
        "s0": _record("s0", 8, 10.0),
        "s1": _record("s1", 9, 20.0),
        "s2": _record("s2", 10, 30.0),
        "s3": _record("s3", 11, 40.0),
    }
    return Dandi688MultiSessionDataset(
        sessions, window_size=4, random_calibration=False, calibration_n_trials=3
    )


def test_probability_zero_is_exact_base_parity() -> None:
    base = _base()
    wrapped = SizeMatchedPseudoSessionDataset(base, seed=42, mix_probability=0.0, contributor_count=3)
    for index in (0, 5, len(base) - 1):
        left, right = base[index], wrapped[index]
        assert left[3] == right[3]
        for a, b in zip(left[:3] + left[4:], right[:3] + right[4:]):
            assert torch.equal(a, b)


def test_mixed_plan_is_size_matched_distinct_and_deterministic() -> None:
    base = _base()
    first = SizeMatchedPseudoSessionDataset(base, seed=43, mix_probability=1.0, contributor_count=3)
    second = SizeMatchedPseudoSessionDataset(base, seed=43, mix_probability=1.0, contributor_count=3)
    for index in range(len(base)):
        one, two = first.plan_for_index(index), second.plan_for_index(index)
        anchor_n = base.sessions[one.anchor_session].neural.shape[-1]
        assert one.mixed and one.total_units == anchor_n
        assert len({item.session for item in one.contributors}) == 3
        assert max(item.unit_indices.size for item in one.contributors) - min(
            item.unit_indices.size for item in one.contributors
        ) <= 1
        assert [(x.session, x.start, x.unit_indices.tolist()) for x in one.contributors] == [
            (x.session, x.start, x.unit_indices.tolist()) for x in two.contributors
        ]
        assert np.array_equal(one.final_permutation, two.final_permutation)


def test_activity_calibration_and_side_rows_use_one_unit_mapping() -> None:
    base = _base()
    wrapped = SizeMatchedPseudoSessionDataset(base, seed=44, mix_probability=1.0, contributor_count=3)
    index = 4
    neural, behavior, calibration, session, side = wrapped[index]
    plan = wrapped.plan_for_index(index)
    assert session == plan.anchor_session
    anchor_start = plan.anchor_start
    assert torch.equal(behavior, torch.from_numpy(base.sessions[session].behavior[anchor_start:anchor_start + 4]))
    # Rebuild the expected arrays independently from the plan.  This catches a
    # unit permutation applied to only one of neural/calibration/side.
    expected_neural = []
    expected_calibration = []
    expected_side = []
    for item in plan.contributors:
        record = base.sessions[item.session]
        expected_neural.append(record.neural[item.start:item.start + 4, item.unit_indices])
        expected_calibration.append(record.calib_trials[..., item.unit_indices])
        expected_side.append(record.side_features[item.unit_indices])
    permutation = plan.final_permutation
    assert torch.equal(neural, torch.from_numpy(np.concatenate(expected_neural, axis=-1)[:, permutation]))
    assert torch.equal(calibration, torch.from_numpy(np.concatenate(expected_calibration, axis=-1)[..., permutation]))
    assert torch.equal(side, torch.from_numpy(np.concatenate(expected_side, axis=0)[permutation]))


def test_nearest_donor_endpoint_and_manifest() -> None:
    base = _base()
    wrapped = SizeMatchedPseudoSessionDataset(base, seed=45, mix_probability=0.5, contributor_count=3)
    for index in range(len(base)):
        plan = wrapped.plan_for_index(index)
        if not plan.mixed:
            continue
        anchor = base.sessions[plan.anchor_session]
        target = anchor.behavior[plan.anchor_start + 3]
        for donor in plan.contributors[1:]:
            record = base.sessions[donor.session]
            endpoints = record.behavior[record.valid_starts + 3]
            expected = int(record.valid_starts[np.argmin(np.linalg.norm(endpoints - target, axis=1))])
            assert donor.start == expected
    manifest = wrapped.audit_manifest()
    assert manifest["examples_audited"] == len(base)
    assert 0 < manifest["mixed_examples"] < len(base)
    assert manifest["rejected_candidate_examples"] == 0
    assert manifest["accepted_endpoint_residual"]["count"] == 2 * manifest["mixed_examples"]
    assert len(manifest["schedule_sha256"]) == 64


def test_bad_behavior_match_falls_back_to_exact_anchor() -> None:
    base = _base()
    wrapped = SizeMatchedPseudoSessionDataset(
        base,
        seed=47,
        mix_probability=1.0,
        contributor_count=3,
        max_behavior_residual=1e-8,
    )
    plan = wrapped.plan_for_index(0)
    assert plan.mix_candidate and not plan.mixed
    original, fallback = base[0], wrapped[0]
    assert original[3] == fallback[3]
    for left, right in zip(original[:3] + original[4:], fallback[:3] + fallback[4:]):
        assert torch.equal(left, right)
    manifest = wrapped.audit_manifest()
    assert manifest["mix_candidate_examples"] == len(base)
    assert manifest["mixed_examples"] == 0
    assert manifest["rejected_candidate_examples"] == len(base)


def test_ordinary_session_batch_sampler_still_collates_fixed_n() -> None:
    base = _base()
    wrapped = SizeMatchedPseudoSessionDataset(base, seed=46, mix_probability=1.0, contributor_count=3)
    sampler = SessionBatchSampler(wrapped, batch_size=3, shuffle=False, seed=46)
    for batch_indices in sampler:
        rows = [wrapped[index] for index in batch_indices]
        assert len({row[3] for row in rows}) == 1
        assert len({row[0].shape[-1] for row in rows}) == 1
        assert len({row[2].shape[-1] for row in rows}) == 1
        assert len({row[4].shape[0] for row in rows}) == 1
