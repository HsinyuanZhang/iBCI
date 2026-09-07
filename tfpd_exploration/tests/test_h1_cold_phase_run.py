"""CPU-only unit contracts for the hash-authorized H1 cold phase worker."""
from __future__ import annotations

import copy
import random
import numpy as np
import pytest
import torch

from tfpd_exploration.src.h1_family_v1 import cold_phase_run as phase
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA


class TinyDecoder(torch.nn.Module):
    """Real parameters, AdamW and EMA; inexpensive phase-compatible forward."""
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(7, 7, bias=False); self.forward_calls = 0; self.last_bins = []
    def forward_last(self, x, bank, dropout_keep=None):
        self.forward_calls += 1; self.last_bins.append(x[:, -1].detach().clone())
        return self.linear(x[:, -1, :7])


def ready(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01); ema = DecoderEMA(model, decay=.9995)
    model.linear.weight.square().sum().backward(); opt.step(); ema.update_after_step(model)
    return opt, ema


def payload(monkeypatch):
    monkeypatch.setattr(phase, "UPDATES", 731)
    model = TinyDecoder(); opt, ema = ready(model); ema.n_updates = 731
    identities = {"sampler_sha256": "a", "keep_sha256": "b", "prefix_sha256": "c"}
    return model, opt, ema, identities, phase.checkpoint(torch, model, opt, ema, epoch=1, identities=identities)


def test_real_ema_adamw_disk_checkpoint_restore_rng_and_digest(monkeypatch, tmp_path):
    fake_cuda_rng, restored_cuda_rng = [torch.tensor([17], dtype=torch.uint8)], []
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: fake_cuda_rng)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda value: restored_cuda_rng.append(value))
    model, opt, ema, identities, saved = payload(monkeypatch)
    path = tmp_path / "epoch01.pt"; torch.save(saved, path); loaded = torch.load(path, map_location="cpu", weights_only=False)
    expected_rng, expected_np, expected_py = saved["rng_cpu"].clone(), saved["rng_numpy"], saved["rng_python"]
    with torch.no_grad():
        for value in model.parameters(): value.add_(3)
        for value in ema.shadow.values(): value.add_(2)
    torch.manual_seed(999); np.random.seed(999); random.seed(999)
    phase.restore(torch, loaded, model, opt, ema, 1, identities)
    assert phase.state_digest(model.state_dict()) == saved["raw_digest"]
    assert phase.state_digest(ema.shadow) == saved["ema_digest"]
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert np.random.get_state()[1].tolist() == expected_np[1].tolist() and random.getstate() == expected_py
    assert saved["rng_cuda"] == fake_cuda_rng and restored_cuda_rng == [fake_cuda_rng]


@pytest.mark.parametrize("change", [
    lambda p: p.__setitem__("epoch", 2),
    lambda p: p.__setitem__("identities", {"wrong": True}),
    lambda p: p["ema"].__setitem__("n_updates", 1),
    lambda p: p["raw"].__setitem__(next(iter(p["raw"])), torch.zeros_like(next(iter(p["raw"].values())))),
])
def test_restore_rejects_epoch_identity_ema_count_and_state_corruption(monkeypatch, change):
    model, opt, ema, identities, saved = payload(monkeypatch); corrupt = copy.deepcopy(saved); change(corrupt)
    with pytest.raises(RuntimeError, match="strict checkpoint"):
        phase.restore(torch, corrupt, model, opt, ema, 1, identities)


def test_checked_score_real_ema_restores_raw_even_callback_error():
    model = TinyDecoder(); _opt, ema = ready(model)
    with torch.no_grad(): next(model.parameters()).add_(.7)
    raw = phase.state_digest(model.state_dict())
    seen = phase.checked_score(model, ema, lambda candidate: phase.state_digest(candidate.state_dict()))
    assert seen == phase.state_digest(ema.shadow) and phase.state_digest(model.state_dict()) == raw
    def explode(candidate):
        with torch.no_grad(): next(candidate.parameters()).add_(99)
        raise ValueError("callback failure")
    with pytest.raises(ValueError, match="callback"):
        phase.checked_score(model, ema, explode)
    assert phase.state_digest(model.state_dict()) == raw


def test_train_update_ragged_8_plus_2_one_step_weighted_loss_and_current_bin():
    torch.manual_seed(7); cells = {name: TinyDecoder() for name in phase.CELLS}
    with torch.no_grad():
        for model in cells.values(): model.linear.weight.zero_()
    opts = {name: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01) for name, model in cells.items()}
    emas = {name: DecoderEMA(model, decay=.9995) for name, model in cells.items()}
    x = torch.randn(10, 700, 176); x[:, -1] = 13; target = torch.ones(10, 7); keep = torch.ones(10, 176, dtype=torch.bool)
    row = phase.train_update(torch, cells, opts, emas, x, target, object(), keep, epoch=1, batch_id=4)
    assert row["rows"] == 10 and 0 <= row["cold_rows"] <= 10
    assert row["loss"] == {"CONTROL": pytest.approx(1.), "PREFIX": pytest.approx(1.)}
    for name, model in cells.items():
        assert model.forward_calls == 2 and emas[name].n_updates == 1
        assert all(torch.equal(last, torch.full_like(last, 13)) for last in model.last_bins)


def test_validate_source_batch_requires_finite_fp32_genuine_query_membership():
    neural = np.arange(710 * 176, dtype=np.float32).reshape(710, 176); velocity = np.arange(710 * 7, dtype=np.float32).reshape(710, 7)
    row = {"neural": neural, "velocity": velocity, "query_starts": np.array([0, 10], np.int64)}
    x, target = phase.validate_source_batch(row, np.array([0, 10], np.int64))
    assert x.shape == (2, 700, 176) and np.array_equal(target[1], velocity[709] * 20)
    for starts in (np.array([1], np.int64), np.array([0, 0], np.int64), np.array([10], np.int32)):
        with pytest.raises(RuntimeError, match="membership"):
            phase.validate_source_batch(row, starts)
    with pytest.raises(RuntimeError, match="membership"):
        phase.validate_source_batch({**row, "neural": neural.astype(np.float64)}, np.array([0], np.int64))


def test_preflight_go_nonoverwrite_and_authorization_sha_precede_bindings(monkeypatch, tmp_path):
    formal, output, authorization = tmp_path / "formal", tmp_path / "output", tmp_path / "authorization.json"; authorization.write_text("{}")
    monkeypatch.delenv("H1_COLD_PHASE_GO", raising=False)
    monkeypatch.setattr(phase, "collect_bindings", lambda *a, **k: pytest.fail("bindings before gate"))
    with pytest.raises(RuntimeError, match="GO"):
        phase.preflight(formal, output, authorization, authorization_sha256="0" * 64, arm="flat", physical_gpu=0, threads=1)
    output.mkdir()
    with pytest.raises(FileExistsError):
        phase.preflight(formal, output, authorization, authorization_sha256="0" * 64, arm="flat", physical_gpu=0, threads=1)
    output.rmdir(); monkeypatch.setenv("H1_COLD_PHASE_GO", "1"); monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="authorization SHA"):
        phase.preflight(formal, output, authorization, authorization_sha256="0" * 64, arm="flat", physical_gpu=0, threads=1)


def test_atomic_npz_roundtrip_and_overwrite_guard(tmp_path):
    path = tmp_path / "arrays.npz"; arrays = {"x": np.arange(6, dtype=np.float64).reshape(2, 3), "y": np.array([1, 2], np.int64)}
    phase.atomic_npz(arrays, path)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for name, value in arrays.items(): assert np.array_equal(loaded[name], value)
    with pytest.raises(FileExistsError): phase.atomic_npz(arrays, path)
