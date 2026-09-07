"""Sampler revision: session-pure shuffled batches, shared X/y permutation."""

from __future__ import annotations

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import contracts, jobs, plan, sampler, training


def _fake_banks() -> dict[str, contracts.SessionBank]:
    banks = {}
    for i, session in enumerate(plan.HELDIN_SESSIONS):
        n = 40 + 3 * i
        x = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
        x = np.broadcast_to(x, (n, plan.WINDOW, plan.CHANNELS)).copy()
        y = np.stack([np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32) + 100], axis=1)
        starts = np.arange(1000, 1000 + n, dtype=np.int64)
        e0 = torch.zeros((plan.CHANNELS, plan.IDENTITY_DIM), dtype=torch.float32)
        t = torch.zeros((plan.CHANNELS, plan.T4_DIM), dtype=torch.float32)
        u = torch.zeros((plan.SUPPORT_HORIZON, plan.CHANNELS, plan.HIDDEN_DIM), dtype=torch.float32)
        banks[session] = contracts.SessionBank(
            session_id=session,
            support_trial_ids=tuple(range(plan.SUPPORT_HORIZON)),
            raw_trial_ids=tuple(range(plan.SUPPORT_HORIZON)),
            X_store=x,
            target_store=y,
            eligible_starts=starts,
            E0=e0,
            T=t,
            unit_mask=torch.ones(plan.CHANNELS, dtype=torch.bool),
            provenance={"kind": "sampler_test"},
            frozen_u=u,
        )
    return banks


def test_manifest_covers_every_window_once_per_epoch() -> None:
    banks = _fake_banks()
    lengths = sampler.session_lengths(banks)
    manifest = sampler.build_shuffled_manifest(lengths, batch_size=8, epochs=2, sampler_seed=7)
    for epoch in (1, 2):
        seen = {session: [] for session in plan.HELDIN_SESSIONS}
        for spec in manifest["batches"][str(epoch)]:
            assert spec["session"] in plan.HELDIN_SESSIONS
            assert spec["indices"], "empty batch"
            seen[spec["session"]].extend(spec["indices"])
        for session, idx in seen.items():
            assert sorted(idx) == list(range(lengths[session]))


def test_window_and_target_share_permutation() -> None:
    banks = _fake_banks()
    session = plan.HELDIN_SESSIONS[0]
    bank = banks[session]
    batch = sampler.batch_from_indices(bank, [5, 1, 9], device="cpu", target_space=plan.SCORING_TARGET_SPACE)
    assert batch.session_id == session
    assert batch.window_ids == (1005, 1001, 1009)
    x0 = batch.X[:, 0, 0].cpu().numpy()
    y0 = batch.last_target[:, 0].cpu().numpy()
    np.testing.assert_array_equal(x0, np.array([5, 1, 9], dtype=np.float32))
    np.testing.assert_array_equal(y0, np.array([5, 1, 9], dtype=np.float32))


def test_global_batch_order_is_not_calendar() -> None:
    banks = _fake_banks()
    manifest = sampler.build_shuffled_manifest(sampler.session_lengths(banks), batch_size=8, epochs=1, sampler_seed=7)
    seq = sampler.manifest_session_sequence(manifest, 1)
    calendar = []
    for session in plan.HELDIN_SESSIONS:
        n = (sampler.session_lengths(banks)[session] + 7) // 8
        calendar.extend([session] * n)
    assert seq != calendar
    assert set(seq) == set(plan.HELDIN_SESSIONS)


def test_manifest_is_replayable() -> None:
    banks = _fake_banks()
    lengths = sampler.session_lengths(banks)
    a = sampler.build_shuffled_manifest(lengths, batch_size=8, epochs=2, sampler_seed=7)
    b = sampler.build_shuffled_manifest(lengths, batch_size=8, epochs=2, sampler_seed=7)
    assert a["digest"] == b["digest"]
    assert a["batches"]["1"] == b["batches"]["1"]
    ids_a = [batch.window_ids for batch in sampler.iter_manifest_batches(banks, a, 1, device="cpu")]
    ids_b = [batch.window_ids for batch in sampler.iter_manifest_batches(banks, b, 1, device="cpu")]
    assert ids_a == ids_b
    ids_train = [
        batch.window_ids for batch in training.epoch_batches_shuffled(banks, a, 1, device="cpu")
    ]
    assert ids_train == ids_a


def test_cosine_tail_is_identity_until_epoch13() -> None:
    assert training.cosine_tail_factor(12) == 1.0
    assert training.cosine_tail_factor(13) == 1.0
    assert abs(training.cosine_tail_factor(24) - 0.1) < 1e-12
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    lr12 = training.apply_b_lr(opt, 3e-4, global_step=4000, warmup_steps=10, epoch=12, max_epochs=12)
    assert abs(lr12 - 3e-4) < 1e-12
    lr24 = training.apply_b_lr(opt, 3e-4, global_step=4000, warmup_steps=10, epoch=24, max_epochs=24)
    assert abs(lr24 - 3e-5) < 1e-12


def test_timeline_store_indexes_by_start_ids() -> None:
    session = plan.HELDIN_SESSIONS[0]
    t_len = 300
    x = np.arange(t_len, dtype=np.float32).reshape(t_len, 1)
    x = np.broadcast_to(x, (t_len, plan.CHANNELS)).copy()
    starts = np.array([10, 40, 80, 120], dtype=np.int64)
    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    bank = contracts.SessionBank(
        session_id=session,
        support_trial_ids=tuple(range(plan.SUPPORT_HORIZON)),
        raw_trial_ids=tuple(range(plan.SUPPORT_HORIZON)),
        X_store=x,
        target_store=y,
        eligible_starts=starts,
        E0=torch.zeros((plan.CHANNELS, plan.IDENTITY_DIM)),
        T=torch.zeros((plan.CHANNELS, plan.T4_DIM)),
        unit_mask=torch.ones(plan.CHANNELS, dtype=torch.bool),
        provenance={"kind": "timeline_test"},
        frozen_u=torch.zeros((plan.SUPPORT_HORIZON, plan.CHANNELS, plan.HIDDEN_DIM)),
    )
    batch = sampler.batch_from_indices(bank, [2, 0], device="cpu", target_space=plan.SCORING_TARGET_SPACE)
    assert batch.window_ids == (80, 10)
    np.testing.assert_array_equal(batch.X[:, 0, 0].cpu().numpy(), np.array([80, 10], dtype=np.float32))
    np.testing.assert_array_equal(batch.last_target[:, 0].cpu().numpy(), np.array([5, 1], dtype=np.float32))


def test_visible_gpu_maps_lease_uuid(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert jobs.visible_gpu_index() == 0
    assert jobs.visible_gpu_uuid() == plan.GPU0_UUID
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert jobs.visible_gpu_uuid() == plan.GPU1_UUID
