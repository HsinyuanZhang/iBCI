"""CPU contracts for the prospective QueryAge formal-prefix trainer."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_train as f


class _EMA:
    def __init__(self, model):
        self.n_updates = f.EPOCH_UPDATES
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
    def checkpoint_state(self): return {"decay": f.EMA, "n_updates": self.n_updates, "shadow": copy.deepcopy(self.shadow)}
    def load_checkpoint_state(self, state):
        self.n_updates = state["n_updates"]; self.shadow = copy.deepcopy(state["shadow"])


def test_prefix_precedes_keep_and_preserves_current_bin():
    x = torch.arange(2 * 700 * 176, dtype=torch.float32).reshape(2, 700, 176)
    prefix, lengths, keep = f.prefix_and_keep(x, epoch=1, batch_index=0, bank_mask=torch.ones(176, dtype=torch.bool), device=torch.device("cpu"))
    assert prefix.shape == x.shape and keep.shape == (2, 176)
    torch.testing.assert_close(prefix[:, -1], x[:, -1], atol=0, rtol=0)
    assert lengths.dtype == torch.int64 and bool(((lengths == 700) | ((lengths >= 1) & (lengths < 700))).all())
    again, again_lengths, again_keep = f.prefix_and_keep(x, epoch=1, batch_index=0, bank_mask=torch.ones(176, dtype=torch.bool), device=torch.device("cpu"))
    torch.testing.assert_close(prefix, again, atol=0, rtol=0); torch.testing.assert_close(lengths, again_lengths, atol=0, rtol=0); torch.testing.assert_close(keep, again_keep, atol=0, rtol=0)
    assert not bool((keep & ~torch.ones(176, dtype=torch.bool)).any())


@pytest.mark.parametrize("n", (1, 8, 32))
@pytest.mark.parametrize("epoch,batch_index", ((1, 0), (1, 17), (12, 730)))
def test_prefix_lengths_is_the_exact_cold_history_draw_without_materializing_windows(n, epoch, batch_index):
    """The lightweight identity path must retain the real prefix RNG law."""
    x = torch.zeros((n, 700, 176), dtype=torch.float32)
    _, actual, _ = f.prefix_and_keep(
        x, epoch=epoch, batch_index=batch_index,
        bank_mask=torch.ones(176, dtype=torch.bool), device=torch.device("cpu"),
    )
    torch.testing.assert_close(f.prefix_lengths(n, epoch=epoch, batch_index=batch_index), actual, atol=0, rtol=0)


def test_warmup_and_sampler_contracts_are_exact():
    assert f.warmup_lr(epoch=1, global_step=1) == pytest.approx(f.LR / f.EPOCH_UPDATES)
    assert f.warmup_lr(epoch=1, global_step=f.EPOCH_UPDATES) == pytest.approx(f.LR)
    assert f.warmup_lr(epoch=2, global_step=f.EPOCH_UPDATES) == f.LR
    with pytest.raises(ValueError): f.warmup_lr(epoch=1, global_step=0)
    with pytest.raises(RuntimeError): f.sampler_identity_digest([])


def test_checkpoint_disk_roundtrip_is_recursive_and_rng_strict(tmp_path: Path):
    torch.manual_seed(8); np.random.seed(8)
    model = torch.nn.Linear(3, 2); opt = torch.optim.AdamW(model.parameters(), lr=f.LR); ema = _EMA(model)
    identities = {"sampler_sha256": "a" * 64, "keep_sha256": "b" * 64, "prefix_sha256": "c" * 64}
    bindings, shared = {"fixture": "formal"}, "d" * 64
    payload = f.checkpoint_payload(model=model, optimizer=opt, ema=ema, epoch=1, identities=identities, arm="flat", shared_init_sha256=shared, bindings=bindings)
    path = tmp_path / "epoch001.pt"; f.atomic_torch_save(payload, path); loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert f.same(payload, loaded)
    with torch.no_grad(): model.weight.add_(1)
    f.strict_restore(payload=loaded, model=model, optimizer=opt, ema=ema, epoch=1, identities=identities, arm="flat", shared_init_sha256=shared, bindings=bindings)
    assert f.state_digest(model.state_dict()) == payload["raw_state_sha256"]
    draw_a = (torch.rand(3), np.random.rand(3)); f.strict_restore(payload=loaded, model=model, optimizer=opt, ema=ema, epoch=1, identities=identities, arm="flat", shared_init_sha256=shared, bindings=bindings); draw_b = (torch.rand(3), np.random.rand(3))
    torch.testing.assert_close(draw_a[0], draw_b[0], atol=0, rtol=0); np.testing.assert_array_equal(draw_a[1], draw_b[1])
    bad = copy.deepcopy(loaded); bad["identities"]["prefix_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="identity"): f.strict_restore(payload=bad, model=model, optimizer=opt, ema=ema, epoch=1, identities=identities, arm="flat", shared_init_sha256=shared, bindings=bindings)


def test_authority_binding_is_absolute_and_binds_planned_smoke_launcher(tmp_path: Path, monkeypatch):
    launcher, smoke = (tmp_path / "launcher.py").resolve(), (tmp_path / "smoke.py").resolve()
    launcher.write_text("# planned launcher\n"); smoke.write_text("# planned smoke\n")
    monkeypatch.setattr(f, "FORMAL_LAUNCHER", launcher); monkeypatch.setattr(f, "FORMAL_SMOKE", smoke)
    cap, checkpoint = (tmp_path / "capacity.json").resolve(), (tmp_path / "capacity.pt").resolve(); cap.write_text("{}\n"); checkpoint.write_bytes(b"checkpoint")
    out = (tmp_path / "out").resolve(); binding = f.collect_bindings(out, capacity_receipt=cap, capacity_checkpoint=checkpoint)
    assert binding["capacity_state_used_for_warmstart"] is False and binding["code_closure"]["launcher"] == f.sha(launcher)
    f.validate_authority_payload({"schema": "h1_queryage_formal_prefix_root_authorization_v1", "mode": "formal", "status": "ROOT_REVIEW_GO", "bindings": binding}, output=out, bindings=binding)
    with pytest.raises(RuntimeError): f.collect_bindings(Path("relative"), capacity_receipt=cap, capacity_checkpoint=checkpoint)


class _ToyDecoder(nn.Module):
    """Small real differentiable decoder with the production forward-last shape."""
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(176, 7)

    def forward_last(self, x, bank, dropout_keep=None):
        if dropout_keep is None:
            dropout_keep = torch.ones((len(x), 176), dtype=torch.bool, device=x.device)
        assert dropout_keep.shape == (len(x), 176)
        return self.readout(x[:, -1] * dropout_keep.to(dtype=x.dtype))


def test_cpu_worker_freeze_finalizer_lifecycle_uses_real_updates_checkpoints_and_scorers(tmp_path: Path, monkeypatch):
    """Exercise the formal worker lifecycle without a source cache or GPU.

    Only immutable input/cache and model-factory boundaries are test doubles.
    The worker body, AdamW updates, DecoderEMA, checkpoint round trip, guarded
    native scorers, selection freeze, finalizer reload, and both exports run.
    """
    from tfpd_exploration.src.h1_optimized_v2 import cache as cache_module
    from tfpd_exploration.src.h1_optimized_v2 import paired_train as groups_module
    from tfpd_exploration.src.h1_optimized_v4 import paired_train as sampler_module
    from tfpd_exploration.src.h1_family_v1 import model as family_model
    from tfpd_exploration.src.h1_queryage_family_v1 import model as queryage_model
    from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_score as score

    # This is a test-only recipe reduction, while every lifecycle operation is
    # still the production implementation.
    monkeypatch.setattr(f, "EPOCHS", 1)
    monkeypatch.setattr(f, "EPOCH_UPDATES", 2)
    monkeypatch.setattr(f, "MICRO", 2)
    monkeypatch.setattr(f, "EFFECTIVE", 4)
    monkeypatch.setattr(score, "SELECTION_BINS", 26)
    monkeypatch.setattr(score, "COMPLETE_BINS", 26)
    monkeypatch.setattr(cache_module, "validate_authority", lambda cache, authority: None)
    monkeypatch.setattr(groups_module, "groups", lambda model: list(model.parameters()))

    def make_pair(*, seed):
        torch.manual_seed(seed)
        flat, route = _ToyDecoder(), _ToyDecoder()
        route.load_state_dict(flat.state_dict())
        return flat, route

    monkeypatch.setattr(queryage_model, "make_queryage_localbalanced_pair", make_pair)
    monkeypatch.setattr(family_model, "initialization_receipt", lambda flat, route: {"shared_parameter_max_abs_diff": 0.0, "route_only_parameter_tensors": 1})
    monkeypatch.setattr(family_model, "zero_gate_parity", lambda flat, route, x, bank: {"max_abs_diff": 0.0})
    monkeypatch.setattr(family_model, "route_gate_gradient_l1", lambda route: 1.0)

    starts = np.arange(4, dtype=np.int64)
    ordered = [("s00", starts), ("s00", starts)]
    monkeypatch.setattr(sampler_module, "batches", lambda cache, epoch: ordered)

    def collate(row, batch_starts, device):
        base = torch.arange(len(batch_starts) * 700 * 176, dtype=torch.float32).reshape(len(batch_starts), 700, 176) / 1e6
        return base.to(device), (base[:, -1, :7] / 20.0).to(device)

    monkeypatch.setattr(sampler_module, "collate", collate)
    identities = {"1": {"sampler_sha256": "a" * 64, "keep_sha256": "b" * 64, "prefix_sha256": "c" * 64}}
    monkeypatch.setattr(f, "_all_epoch_identities", lambda cache: identities)
    monkeypatch.setattr(f, "identity_digest", lambda ordered, cache, epoch: identities[str(epoch)])

    bank = {"E0": torch.zeros((176, 700)), "T": torch.zeros((176, 4)), "unit_mask": torch.ones(176, dtype=torch.bool)}
    neural = np.zeros((701, 176), dtype=np.float32)
    neural[:, :7] = np.arange(701, dtype=np.float32)[:, None] / 100.0
    velocity = neural[:, :7].copy() / 20.0
    minival = {
        f"s{i:02d}": {"neural": neural.copy(), "velocity": velocity.copy(), "query_starts": np.array([0, 1], dtype=np.int64),
                       "eval_mask": np.array([False] * 699 + [True, True], dtype=np.bool_), "bank": bank}
        for i in range(13)
    }
    cache = {"train": {f"s{i:02d}": {"bank": bank} for i in range(13)}, "minival": minival}

    # Construct the same external root-authorization and immutable binding
    # format used by the launcher; the source loader is the sole cache seam.
    protocol, source, capacity, checkpoint = (tmp_path / "protocol.md"), (tmp_path / "source_cache.pt"), (tmp_path / "capacity.json"), (tmp_path / "capacity.pt")
    for path, content in ((protocol, b"protocol\n"), (source, b"cache\n"), (capacity, b"{}\n"), (checkpoint, b"capacity\n")):
        path.write_bytes(content)
    cache_authority = tmp_path / "source_cache_authority.json"; cache_authority.write_text("{}\n")
    monkeypatch.setattr(f, "PROTOCOL_DOC", protocol)
    monkeypatch.setattr(f, "CACHE", source)
    monkeypatch.setattr(f, "CACHE_AUTHORITY", cache_authority)
    output = (tmp_path / "formal").resolve()
    bindings = f.collect_bindings(output, capacity_receipt=capacity.resolve(), capacity_checkpoint=checkpoint.resolve())
    authorization = (tmp_path / "root_go.json").resolve()
    authorization.write_text(__import__("json").dumps({"schema": "h1_queryage_formal_prefix_root_authorization_v1", "mode": "formal", "status": "ROOT_REVIEW_GO", "bindings": bindings}, sort_keys=True))
    output.mkdir(); (output / "barrier").mkdir(); (output / "workers").mkdir(); (output / "checkpoints").mkdir(); (output / "exports").mkdir()
    (output / "input_authority.json").write_text(__import__("json").dumps({"authorization_path": str(authorization), "authorization_sha256": f.sha(authorization), "bindings": bindings}, sort_keys=True))
    start = output / "barrier" / "START"; start.write_text("released\n")
    loader = lambda: (cache, {})

    for arm in f.ARMS:
        f.worker_run(arm=arm, output=output, physical_gpu=f.ARM_DEVICE[arm], start_marker=start, allow_cpu_fixture=True, source_loader=loader)

    from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_launcher as launcher
    monkeypatch.setattr(launcher, "SELECTION_BINS", 26)
    freeze = launcher.freeze_selection(output, {arm: __import__("json").loads((output / "barrier" / f"{arm}.ready.json").read_text()) for arm in f.ARMS})
    assert freeze["schema"] == "h1_queryage_formal_prefix_selection_freeze_v1"
    for arm in f.ARMS:
        f.finalizer_run(arm=arm, output=output, physical_gpu=f.ARM_DEVICE[arm], allow_cpu_fixture=True, source_loader=loader)
        final = __import__("json").loads((output / "workers" / f"{arm}_final.json").read_text())
        assert final["status"] == "COMPLETE_POST_FREEZE"
        for label in ("selected", "epoch12"):
            report = final["reports"][label]
            assert report["selection_reproduced"]["n_bins"] == 26
            assert report["complete"]["n_bins"] == 26
            assert Path(report["complete_archive"]).is_file()
            assert Path(report["plain_ema_path"]).is_file()
