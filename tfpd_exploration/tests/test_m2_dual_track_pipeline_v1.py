"""Pipeline contracts for m2_dual_track_v1. Dry-run: no NWB load in pytest.

Real NWB ingestion lives in ``data.stage0_data()`` and writes receipts under
``tfpd_exploration/results/m2_dual_track_v1/20260905_101500/``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from tfpd_exploration.src.m2_dual_track_v1 import champion, contracts, data, evaluation, plan, training


def test_source_allowlist_is_heldin_only() -> None:
    paths = data.source_nwb_allowlist()
    assert len(paths) == 14
    kinds = {data.classify_nwb_role(path) for path in paths}
    assert kinds == {"source_calib", "source_minival"}
    assert all("held-in" in path.name for path in paths)
    assert not any("held-out" in path.name for path in paths)
    assert not any(data.classify_nwb_role(path) == "hidden_or_test" for path in paths)
    sessions = {data.parse_session_name(path) for path in paths}
    assert sessions == set(plan.HELDIN_SESSIONS)


def test_ext4_allowlist_excludes_nov24_and_hidden() -> None:
    paths = data.ext4_nwb_allowlist()
    assert len(paths) == 4
    assert {data.parse_session_name(path) for path in paths} == set(plan.EXT4_SESSIONS)
    assert all(data.classify_nwb_role(path) == "ext4" for path in paths)
    forbidden = data.forbidden_nwb_paths()
    assert any("2020-11-24" in path.name for path in forbidden)
    assert not any(data.parse_session_name(path) in plan.EXCLUDED_EXTERNAL_SESSIONS for path in paths)


def test_file_access_log_train_opens_only_source(tmp_path: Path) -> None:
    log = data.FileAccessLog(role="source")
    source = data.source_nwb_allowlist()[0]
    log.record(source)
    with pytest.raises(plan.DualTrackError, match="non-source"):
        log.record(data.ext4_nwb_allowlist()[0])
    hidden = tmp_path / "sub-X-held-out-test_ses-2020-12-01-Run1_behavior+ecephys.nwb"
    hidden.write_bytes(b"")
    with pytest.raises(plan.DualTrackError, match="hidden/test"):
        log.record(hidden)


def test_file_access_log_evaluator_opens_only_ext4(tmp_path: Path) -> None:
    log = data.FileAccessLog(role="ext4")
    log.record(data.ext4_nwb_allowlist()[0])
    with pytest.raises(plan.DualTrackError, match="non-ext4"):
        log.record(data.source_nwb_allowlist()[0])
    nov24 = next(path for path in data.forbidden_nwb_paths() if "2020-11-24" in path.name)
    with pytest.raises(plan.DualTrackError, match="non-ext4"):
        log.record(nov24)
    hidden = tmp_path / "hidden_evalai_test.nwb"
    hidden.write_bytes(b"")
    with pytest.raises(plan.DualTrackError, match="hidden/test"):
        log.record(hidden)


def test_disjointness_uses_trial_ids_not_array_offsets() -> None:
    pad = data.QUERY_PAD_BINS
    # Synthetic padded query timeline: trials at raw bins 0, 40, 80, ... 33*40.
    raw_starts = np.arange(40, dtype=np.int64) * 40
    padded_starts = raw_starts + pad
    boundary = data.query_support_boundary_padded(padded_starts, 33)
    assert boundary == int(padded_starts[33])
    assert data.raw_bin_from_padded(boundary) == int(raw_starts[33])
    # Windows whose last bin is "in trial" — include some that overlap support.
    candidate = np.arange(0, int(padded_starts[39]), 7, dtype=np.int64)
    legal = data.filter_disjoint_window_starts(candidate, boundary)
    assert legal.size > 0
    assert np.all(legal >= boundary)
    # Full 50-bin span starts after support trial 33.
    assert np.all(legal + 0 >= padded_starts[33])
    with pytest.raises(plan.DualTrackError, match="never compare"):
        data.refuse_calib_vs_query_offset_compare(12, 49)


def test_mapping_receipt_is_auditable() -> None:
    class _DS:
        trial_start_indices = {"ses-x": np.array([49, 89, 129] + [169 + 40 * i for i in range(40)])}

    starts = np.array([169 + 40 * 30, 169 + 40 * 31], dtype=np.int64)
    receipt = data._mapping_receipt(
        session="ses-x",
        surface="ext4",
        dataset=_DS(),
        eligible=starts,
        apply_disjoint=True,
    )
    assert receipt["query_pad_bins"] == 49
    assert receipt["do_not_compare_calib_offset_to_query_offset"] is True
    assert receipt["support_trial_ids"] == list(range(33))
    assert receipt["support_boundary_padded_bin"] == int(_DS.trial_start_indices["ses-x"][33])


def test_output_space_divide_only_at_score_time() -> None:
    bank = contracts.make_stub_bank(session_id="ses-2020-10-30-Run1", num_units=8, k=4)

    class _Stub:
        name = "stub"
        training_target_space = plan.TRAINING_TARGET_SPACE

        def forward_last(self, X, bank, unit_mask=None):
            # decoder_raw = native * 5
            return torch.full((X.shape[0], 2), 10.0)

        def trainable_parameters(self):
            return {}

        def eval(self):
            return self

    raw = _Stub().forward_last(torch.zeros(3, 50, 8), bank)
    assert torch.equal(raw, torch.full((3, 2), 10.0))
    native = evaluation.decoder_raw_to_native(raw)
    assert np.allclose(native, 2.0)
    assert champion.OUTPUT_SPACE_CONTRACT["divide_at"] == "evaluator_only"
    assert plan.BEHAVIOR_SCALE == 5.0
    src = inspect.getsource(champion.FrozenChampion.forward_last)
    assert "BEHAVIOR_SCALE" not in src
    assert "decode_with_identity" in src


def test_evaluator_r2_after_divide(monkeypatch: pytest.MonkeyPatch) -> None:
    bank = contracts.make_stub_bank(session_id="ses-2020-10-30-Run1", num_units=4, k=3)
    native = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    object.__setattr__(bank, "target_store", native)
    object.__setattr__(bank, "eligible_starts", np.arange(4, dtype=np.int64))
    object.__setattr__(bank, "X_store", np.zeros((60, 4), dtype=np.float32))

    class _Cand:
        name = "perfect"
        training_target_space = plan.TRAINING_TARGET_SPACE

        def __init__(self) -> None:
            self.cursor = 0

        def forward_last(self, X, bank, unit_mask=None):
            sl = native[self.cursor : self.cursor + X.shape[0]] * plan.BEHAVIOR_SCALE
            self.cursor += int(X.shape[0])
            return torch.from_numpy(np.ascontiguousarray(sl, dtype=np.float32))

        def eval(self):
            return self

    cand = _Cand()

    def _banks(self, scoring_manifest):
        return {bank.session_id: bank}

    monkeypatch.setattr(evaluation.DualTrackEvaluator, "_banks", _banks)
    report = evaluation.DualTrackEvaluator(device="cpu", batch_size=2)(
        cand, {"surface": "ext4", "sessions": [bank.session_id]}
    )
    assert report["per_session"][bank.session_id]["divided_by_behavior_scale"] is True
    assert report["per_session"][bank.session_id]["r2"] == pytest.approx(1.0)
    assert report["subtracted_historical_six_session"] is False


def test_native_e0_uses_push_trial_not_mean_gemm() -> None:
    src = inspect.getsource(champion.native_e0_and_u)
    assert "push_trial" in src
    assert "finalize_identity" in src
    assert "torch.mean" not in src
    from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder

    torch.manual_seed(0)
    enc = HoldContrastFiLMEarlyPoolEncoder(
        trial_length=100, window_size=50, hidden_dim=64, side_dim=8, film_rank=8
    )
    enc.eval()
    trials = torch.randn(3, 100, 96)
    t4 = torch.randn(96, 4)
    side = champion.empty_contrast_side(t4)
    assert torch.equal(side[:, 4:], torch.zeros(96, 4))
    e0, u = champion.native_e0_and_u(enc, trials, side)
    assert tuple(e0.shape) == (96, 50)
    assert tuple(u.shape) == (3, 96, 64)
    # Sequential push/finalize matches the helper (not a batched mean GEMM).
    state = enc.reset_stream(1, 96, torch.device("cpu"), torch.float32)
    state["side_features"] = side.unsqueeze(0)
    with torch.no_grad():
        for trial in trials:
            state = enc.push_trial(state, trial)
        ref = enc.finalize_identity(state).squeeze(0)
    assert torch.equal(e0, ref.cpu())


def test_empty_means_zero_contrast_input() -> None:
    assert champion.OUTPUT_SPACE_CONTRACT["forward_last"] == "decoder_raw"
    t4 = np.ones((96, 4), dtype=np.float32)
    side = champion.empty_contrast_side(t4)
    assert side.shape == (96, 8)
    assert torch.equal(side[:, :4], torch.ones(96, 4))
    assert torch.equal(side[:, 4:], torch.zeros(96, 4))


def test_trainer_not_ready_and_does_not_drop_tail() -> None:
    with pytest.raises(plan.DualTrackError, match="NOT_READY"):
        training.train_arm("A-QMEM", seed=42, device="cpu", max_epochs=12)
    scores = {1: 0.1, 2: 0.2, 8: 0.20000000001, 12: 0.15}
    assert training.select_checkpoint(scores) == 2
    near = {1: 0.4, 5: 0.4 + 0.5e-10, 12: 0.3}
    assert training.select_checkpoint(near) == 1
    bank = contracts.make_stub_bank()
    batches = list(data.iter_session_batches(bank, batch_size=3, drop_last=False))
    assert sum(batch.X.shape[0] for batch in batches) == int(bank.eligible_starts.size)
    assert batches[-1].X.shape[0] == int(bank.eligible_starts.size) % 3 or batches[-1].X.shape[0] == 3
    with pytest.raises(plan.DualTrackError, match="forbids dropping"):
        list(data.iter_session_batches(bank, batch_size=3, drop_last=True))


def test_adamw_excludes_bias_norm_and_warmup() -> None:
    layer = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    groups = training.adamw_param_groups(layer.named_parameters(), weight_decay=1e-2)
    decay = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    no_decay = {id(p) for g in groups if g["weight_decay"] == 0 for p in g["params"]}
    assert id(layer[0].weight) in decay
    assert id(layer[0].bias) in no_decay
    assert id(layer[1].weight) in no_decay
    assert training.warmup_factor(0, 10) == pytest.approx(0.1)
    assert training.warmup_factor(9, 10) == pytest.approx(1.0)
    assert "rebuild_optimizer" in training.formal_epoch1_reset_checklist()


def test_stage0_data_hook_exists_and_is_source_only() -> None:
    assert callable(data.stage0_data)
    src = inspect.getsource(data.construct_source_datamodule)
    assert 'setup("fit")' in src
    assert "include_heldout_in_fit" in src
    champ_src = inspect.getsource(champion.load_frozen_champion)
    assert "export_t4_payload" not in champ_src
    assert "load_frozen_model_and_data()" not in champ_src
    base_src = inspect.getsource(champion.load_base_lightning_module)
    assert 'model.setup("fit")' in base_src
    assert "export_t4_payload" not in base_src
    assert "FalconDataModule" not in base_src


def test_score_reference_writes_gpu_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan, "active_run_root", lambda: tmp_path)
    monkeypatch.setattr(data, "cache_root", lambda: tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    evaluation.score_reference(device="cpu")
    payload = (tmp_path / "stage0" / "ref_clean.json").read_text(encoding="utf-8")
    assert "PREPARED_NOT_SCORED" in payload
    assert "CUDA_VISIBLE_DEVICES" in (tmp_path / "stage0" / "run_ref_gpu0.sh").read_text(encoding="utf-8")
    assert "0.360449" not in payload or "subtracted" in payload
