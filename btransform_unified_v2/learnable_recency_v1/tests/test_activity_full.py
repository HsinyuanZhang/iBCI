"""Contract tests for the B3S two-stage FULL arm (activity trunk + real carrier T)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


runner = _load("activity_full_train")


def _m2_models(trunk_state=None, trunk_side="none"):
    from learnable_recency_v1.activity_model import ActivityLearnableRiftDecoder
    from learnable_recency_v1.config import dataset_config

    cfg = dataset_config("m2", tier="learned_slope", ladder="default")
    torch.manual_seed(42)
    act_model = ActivityLearnableRiftDecoder("m2", cfg, context_bins=50, seed=42, support_bins=100, identity_hidden=64)
    torch.manual_seed(42)
    full_model = runner.B3SFullRiftDecoder("m2", cfg, context_bins=50, seed=42, support_bins=100, identity_hidden=64,
                                           trunk_state=trunk_state, trunk_side=trunk_side)
    return act_model, full_model


def _m2_inputs(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 50, 96, generator=g)
    activity = torch.rand(33, 100, 96, generator=g)
    keep = torch.ones(96, dtype=torch.bool)
    valid = torch.ones(4, 50, dtype=torch.bool)
    return x, activity, keep, valid


# ---------------------------------------------------------------------------
# Parser / stage gates.
# ---------------------------------------------------------------------------


def test_parser_defaults_and_locked_recipe():
    p = runner.build_parser()
    a = p.parse_args(["--task", "m2", "--dest", "/tmp/b3s-run"])
    assert a.stage == 1 and a.device == "cpu" and a.seed == 42 and a.proj_dim == 16
    assert a.freeze_trunk_from is None and a.freeze_trunk_view == "ema"
    assert a.trunk_side == "none"
    assert a.ladder == "default" and runner.TASK_EPOCHS == {"m1": 24, "m2": 24, "h1": 32}
    assert runner.SCHEMA == "b3s_full_train_v1" and runner.CKPT_SCHEMA == "b3s_full_epoch_checkpoint_v1"


def test_stage_gates_reject_bad_combinations_before_touching_data():
    p = runner.build_parser()
    base = ["--task", "m2", "--dest", "/tmp/b3s-run", "--epochs", "24"]
    with pytest.raises(ValueError, match="seed 42"):
        runner.run(p.parse_args([*base, "--seed", "43", "--stage", "1"]))
    with pytest.raises(ValueError, match="--freeze-trunk-from"):
        runner.run(p.parse_args([*base, "--stage", "2"]))
    with pytest.raises(ValueError, match="epochs"):
        runner.run(p.parse_args(["--task", "m2", "--dest", "/tmp/b3s-run", "--epochs", "3"]))
    with pytest.raises(ValueError, match="P16"):
        runner.run(p.parse_args([*base, "--proj-dim", "32"]))


def test_stage2_accepts_nondefault_seed_gate():
    # Seed 43 is a stage-2-only freedom; the gate must reject it only at stage 1
    # (the stage-2 seed-43 run then fails later on the missing trunk source).
    p = runner.build_parser()
    args = p.parse_args(["--task", "m2", "--dest", "/tmp/b3s-run", "--epochs", "24", "--seed", "43", "--stage", "2"])
    with pytest.raises(ValueError, match="--freeze-trunk-from"):
        runner.run(args)


# ---------------------------------------------------------------------------
# Model contracts: zero-carrier parity, injection, frozen trunk, E0 identity.
# ---------------------------------------------------------------------------


def test_init_and_zero_carrier_forward_match_act_v2_bitexact():
    act_model, full_model = _m2_models()
    shared = set(dict(act_model.named_parameters())) & set(dict(full_model.named_parameters()))
    assert shared and all(torch.equal(dict(act_model.named_parameters())[n], dict(full_model.named_parameters())[n]) for n in shared)
    x, activity, keep, valid = _m2_inputs()
    act_model.eval(); full_model.eval()
    with torch.no_grad():
        left = act_model(x, activity, dropout_keep=keep, input_valid_mask=valid)
        right = full_model(x, activity, dropout_keep=keep, input_valid_mask=valid, carrier=None)
        right_zero = full_model(x, activity, dropout_keep=keep, input_valid_mask=valid,
                                carrier=torch.zeros(96, 4))
    assert torch.equal(left, right) and torch.equal(left, right_zero)
    identity = act_model.calibrate(activity)
    with torch.no_grad():
        assert torch.equal(act_model(x, identity=identity, dropout_keep=keep),
                           full_model(x, identity=identity, dropout_keep=keep, carrier=None))


def test_real_carrier_changes_output_and_is_validated():
    _act, full_model = _m2_models()
    full_model.eval()
    x, activity, keep, valid = _m2_inputs()
    carrier = torch.randn(96, 4, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        base = full_model(x, activity, dropout_keep=keep, input_valid_mask=valid, carrier=None)
        injected = full_model(x, activity, dropout_keep=keep, input_valid_mask=valid, carrier=carrier)
    assert not torch.equal(base, injected)
    with pytest.raises(ValueError, match="carrier"):
        full_model(x, activity, dropout_keep=keep, carrier=torch.zeros(95, 4))


def _random_trunk(seed: int = 3, *, concat: bool = False) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    post_entrance = 68 if concat else 64
    return {
        "pre_pool.0.weight": torch.randn(64, 100, generator=g),
        "pre_pool.0.bias": torch.randn(64, generator=g),
        "post_pool.0.weight": torch.randn(64, post_entrance, generator=g),
        "post_pool.0.bias": torch.randn(64, generator=g),
        "post_pool.2.weight": torch.randn(64, 64, generator=g),
        "post_pool.2.bias": torch.randn(64, generator=g),
        "post_pool.4.weight": torch.randn(50, 64, generator=g),
        "post_pool.4.bias": torch.randn(50, generator=g),
    }


def test_trunk_state_determines_e0_and_loading_is_strict():
    trunk = _random_trunk()
    _a, first = _m2_models(trunk)
    _b, second = _m2_models(trunk)
    _c, other = _m2_models(_random_trunk(seed=4))
    _, activity, _, _ = _m2_inputs()
    first.eval(); second.eval(); other.eval()
    with torch.no_grad():
        e1, e2, e3 = first.encode_activity(activity), second.encode_activity(activity), other.encode_activity(activity)
    assert torch.equal(e1, e2) and not torch.equal(e1, e3)
    with pytest.raises(RuntimeError):
        _m2_models({"pre_pool.0.weight": torch.randn(64, 100)})


def test_stage2_trunk_frozen_bitexact_and_outside_optimizer_and_ema():
    _act, full_model = _m2_models(_random_trunk())
    full_model.train()
    assert full_model.identity_encoder.training is False  # eval-only trunk under train()
    trunk_before = {n: p.detach().clone() for n, p in full_model.identity_encoder.named_parameters()}
    from btransform_unified_v1.ema import DecoderEMA
    from learnable_recency_v1.config import dataset_config
    from learnable_recency_v1.wrap import split_optimizer_parameters

    cfg = dataset_config("m2", tier="learned_slope", ladder="default")
    ema = DecoderEMA(full_model, decay=0.9995)
    groups = split_optimizer_parameters(full_model, peak_lr=3e-4, weight_decay=0.01, lr_multiplier=1.0)
    assert not any(n.startswith("identity_encoder.") for n in ema.shadow)
    opt_ids = {id(p) for group in groups for p in group["params"]}
    assert not any(id(p) in opt_ids for p in full_model.identity_encoder.parameters())
    opt = torch.optim.AdamW(groups, lr=3e-4)
    x, activity, keep, valid = _m2_inputs()
    carrier = torch.randn(96, 4, generator=torch.Generator().manual_seed(12))
    decoder_before = {n: p.detach().clone() for n, p in full_model.named_parameters()
                      if not n.startswith("identity_encoder.")}
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(full_model(x, activity, dropout_keep=keep, input_valid_mask=valid,
                                                 carrier=carrier), torch.zeros(4, 2))
        loss.backward()
        assert all(p.grad is None for p in full_model.identity_encoder.parameters())
        nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
        opt.step()
        ema.update_after_step(full_model)
    assert all(torch.equal(p.detach(), trunk_before[n]) for n, p in full_model.identity_encoder.named_parameters())
    assert full_model.trunk_parameter_sha256() == runner.B3SFullRiftDecoder(
        "m2", cfg, context_bins=50, seed=42, support_bins=100, identity_hidden=64,
        trunk_state=_random_trunk()).trunk_parameter_sha256()
    assert any(not torch.equal(p.detach(), decoder_before[n]) for n, p in full_model.named_parameters()
               if not n.startswith("identity_encoder."))
    assert ema.n_updates == 2


# ---------------------------------------------------------------------------
# Trunk checkpoint resolution.
# ---------------------------------------------------------------------------


def _fake_checkpoint(trunk: dict[str, torch.Tensor], *, schema="b3s_full_epoch_checkpoint_v1", epoch=1, trunk_side="none"):
    raw = {f"identity_encoder.{k}": v for k, v in trunk.items()}
    raw["zero_carrier"] = torch.zeros(96, 4)
    ema_shadow = {k: v + 0.25 for k, v in raw.items() if k.startswith("identity_encoder.")}
    return {"schema": schema, "stage": "1_pretain", "task": "m2", "epoch": epoch, "global_step": epoch * 10,
            "smoke": False, "trunk_side": trunk_side, "run_meta": {"trunk_side": trunk_side},
            "raw_state_dict": raw, "ema": {"decay": 0.9995, "n_updates": epoch * 10, "shadow": ema_shadow}}


def test_extract_trunk_view_selection_and_schema_gate():
    trunk = _random_trunk()
    state = _fake_checkpoint(trunk)
    raw_view = runner.extract_trunk_state(state, "raw")
    ema_view = runner.extract_trunk_state(state, "ema")
    assert set(raw_view) == set(trunk) and all(torch.equal(raw_view[k], trunk[k]) for k in trunk)
    assert all(torch.equal(ema_view[k], trunk[k] + 0.25) for k in trunk)
    with pytest.raises(RuntimeError, match="not an activity-joint checkpoint"):
        runner.extract_trunk_state({"schema": "something_else"}, "ema")


def test_resolve_trunk_from_checkpoint_selection_and_epoch(tmp_path):
    trunk = _random_trunk()
    run_dir = tmp_path / "stage1_run"
    run_dir.mkdir()
    torch.save(_fake_checkpoint(trunk, epoch=3), run_dir / "epoch_003.pt")
    # Direct checkpoint path.
    got, prov = runner.resolve_trunk(run_dir / "epoch_003.pt", view="raw")
    assert all(torch.equal(got[k], trunk[k]) for k in trunk)
    assert prov["epoch"] == 3 and prov["checkpoint_sha256"] == runner._sha(run_dir / "epoch_003.pt")
    # Selection dest carrying score_receipt.json.
    sel = tmp_path / "selection"
    sel.mkdir()
    (sel / "score_receipt.json").write_text(json.dumps(
        {"run_dir": str(run_dir), "selection": {"epoch": 3}, "status": "COMPLETED"}))
    got2, prov2 = runner.resolve_trunk(sel, view="ema")
    assert prov2["selection_receipt_used"] and prov2["epoch"] == 3
    assert all(torch.equal(got2[k], trunk[k] + 0.25) for k in trunk)
    # Bare run dir requires an explicit epoch.
    with pytest.raises(ValueError, match="freeze-trunk-epoch"):
        runner.resolve_trunk(run_dir, view="raw")
    got3, prov3 = runner.resolve_trunk(run_dir, view="raw", epoch=3)
    assert prov3["epoch"] == 3
    # The ACT v2 schema is loadable by the same generic resolver.
    act_state = _fake_checkpoint(trunk, schema="activity_joint_early_pool_epoch_checkpoint_v2")
    act_dir = tmp_path / "act_run"
    act_dir.mkdir()
    torch.save(act_state, act_dir / "epoch_020.pt")
    got4, _ = runner.resolve_trunk(act_dir / "epoch_020.pt", view="ema")
    assert all(torch.equal(got4[k], trunk[k] + 0.25) for k in trunk)


# ---------------------------------------------------------------------------
# Score surface carrier plumbing.
# ---------------------------------------------------------------------------


def test_score_surface_passes_per_session_carrier_and_calibrates_once(monkeypatch):
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.seen_carriers = []

        def eval(self):
            return self

        def calibrate(self, a, tm=None):
            assert not torch.is_grad_enabled()
            self.calls += 1
            return torch.zeros(2, 3)

        def forward(self, x, identity=None, input_valid_mask=None, carrier=None):
            self.seen_carriers.append(None if carrier is None else carrier.clone())
            return torch.zeros(x.shape[0], 2)

    model = Fake()
    item = {"activity": torch.zeros(1, 4, 2).numpy(), "starts": np.arange(3), "X": torch.zeros(8, 2).numpy(),
            "Y": np.asarray([[1., 2.], [2., 3.], [3., 4.]], np.float32), "pad": 0, "support_provenance": {}}
    monkeypatch.setattr(runner.activity_data, "windows",
                        lambda item, ids, context, device, scale: (torch.zeros(len(ids), context, 2), torch.zeros(len(ids), 2),
                                                                   torch.ones(len(ids), context, dtype=torch.bool)))
    carriers = {"s": torch.full((2, 4), 5.0)}
    out = runner.score_surface(model, {"s": item}, context=2, device=torch.device("cpu"), behavior_scale=1.0,
                               max_batches=1, carriers=carriers)
    assert model.calls == 1 and out["n_windows"] == 3 and out["partial"]
    assert len(model.seen_carriers) == 1 and torch.equal(model.seen_carriers[0], torch.full((2, 4), 5.0))


# ---------------------------------------------------------------------------
# Concat route (carrier joins the trunk at the post_pool entrance).
# ---------------------------------------------------------------------------


def _concat_trunk(input_bins=100, output_dim=50, hidden=64, seed=42):
    torch.manual_seed(seed)
    return runner.ConcatSideTrunk(input_bins, output_dim, hidden_dim=hidden, seed=seed)


def test_concat_trunk_geometry_rng_and_carrier_join():
    from learnable_recency_v1.activity_model import ActivityIdentityEncoder

    torch.manual_seed(0)
    bare = ActivityIdentityEncoder(100, 50, hidden_dim=64, seed=42)
    trunk = _concat_trunk()
    again = _concat_trunk()
    # pre_pool keeps the bare trunk's RNG domain byte-for-byte; post_pool is
    # rebuilt widened (hidden+4 entrance) under the concat offset.
    assert torch.equal(trunk.pre_pool[0].weight, bare.pre_pool[0].weight)
    assert torch.equal(trunk.pre_pool[0].bias, bare.pre_pool[0].bias)
    assert trunk.post_pool[0].weight.shape == (64, 68) and bare.post_pool[0].weight.shape == (64, 64)
    assert torch.equal(trunk.post_pool[0].weight, again.post_pool[0].weight)
    # H1 geometry: hidden 32 -> 36-dim post_pool entrance.
    h1 = runner.ConcatSideTrunk(1024, 700, hidden_dim=32, seed=42)
    assert h1.post_pool[0].weight.shape == (32, 36) and h1.pre_pool[0].weight.shape == (32, 1024)
    # Forward: [33,100,96] activity + [96,4] carrier -> [96,50] E0.
    g = torch.Generator().manual_seed(7)
    activity = torch.rand(33, 100, 96, generator=g)
    carrier = torch.randn(96, 4, generator=g)
    with torch.no_grad():
        fused = trunk(activity, None, carrier)
        zero = trunk(activity, None, None)  # carrier=None is the literal zero side
        other = trunk(activity, None, carrier * 0.5)
    assert fused.shape == (96, 50) and not torch.equal(fused, zero) and not torch.equal(fused, other)
    with pytest.raises(ValueError, match="concat carrier"):
        trunk(activity, None, torch.zeros(95, 4))
    # The carrier never receives gradients through the trunk.
    live = torch.randn(96, 4, requires_grad=True)
    trunk(activity, None, live).sum().backward()
    assert live.grad is None


def test_concat_model_e0_and_token_channel_both_use_carrier():
    _act, model = _m2_models(trunk_side="concat")
    x, activity, keep, valid = _m2_inputs()
    carrier = torch.randn(96, 4, generator=torch.Generator().manual_seed(13))
    model.eval()
    with torch.no_grad():
        e0_zero = model.calibrate(activity)
        e0_fused = model.calibrate(activity, carrier=carrier)
    assert e0_zero.shape == e0_fused.shape == (96, 50) and not torch.equal(e0_zero, e0_fused)
    with torch.no_grad():
        out_none = model(x, activity, dropout_keep=keep, input_valid_mask=valid, carrier=None)
        out_fused = model(x, activity, dropout_keep=keep, input_valid_mask=valid, carrier=carrier)
    assert not torch.equal(out_none, out_fused)
    model.train()
    with pytest.raises(RuntimeError, match="eval-only"):
        model.calibrate(activity, carrier=carrier)
    # The bare route refuses a carrier inside E0 (token channel only there).
    _bare_act, bare = _m2_models()
    with pytest.raises(ValueError, match="bare trunk"):
        bare.encode_activity(activity, None, carrier)


def test_concat_stage1_joint_training_updates_trunk_and_keeps_carrier_frozen():
    _act, model = _m2_models(trunk_side="concat")
    model.train()
    trunk_before = {n: p.detach().clone() for n, p in model.identity_encoder.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x, activity, keep, valid = _m2_inputs()
    carrier_np = np.asarray(torch.randn(96, 4, generator=torch.Generator().manual_seed(14)))
    carrier = torch.as_tensor(carrier_np)
    assert not carrier.requires_grad
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x, activity, dropout_keep=keep, input_valid_mask=valid, carrier=carrier),
                                      torch.zeros(4, 2))
        loss.backward()
        trunk_grads = [n for n, p in model.identity_encoder.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
        assert not trunk_grads  # joint pretrain: every trunk tensor trains
        opt.step()
    assert any(not torch.equal(p.detach(), trunk_before[n]) for n, p in model.identity_encoder.named_parameters())
    assert np.array_equal(carrier.numpy(), carrier_np)  # T frozen input, bit-identical after training


def test_concat_stage2_side_binding_and_formal_gate(tmp_path):
    # Generic resolver: side must match the requesting run.
    run_dir = tmp_path / "concat_stage1"
    run_dir.mkdir()
    concat_trunk = _random_trunk(concat=True)
    torch.save(_fake_checkpoint(concat_trunk, trunk_side="concat"), run_dir / "epoch_001.pt")
    with pytest.raises(RuntimeError, match="side mismatch"):
        runner.resolve_trunk(run_dir / "epoch_001.pt", view="raw", expected_side="none")
    got, prov = runner.resolve_trunk(run_dir / "epoch_001.pt", view="raw", expected_side="concat")
    assert prov["trunk_side"] == "concat" and all(torch.equal(got[k], concat_trunk[k]) for k in concat_trunk)
    torch.save(_fake_checkpoint(_random_trunk(), trunk_side="none"), run_dir / "epoch_002.pt")
    with pytest.raises(RuntimeError, match="side mismatch"):
        runner.resolve_trunk(run_dir / "epoch_002.pt", view="raw", expected_side="concat")
    # Loading a concat trunk into a bare model (or vice versa) fails strictly.
    with pytest.raises(RuntimeError):
        _m2_models(trunk_state=concat_trunk)
    _act, concat_model = _m2_models(trunk_state=concat_trunk, trunk_side="concat")
    assert concat_model.frozen_trunk and all(not p.requires_grad for p in concat_model.identity_encoder.parameters())
    # Formal gate: a stage-1 concat selection only feeds a concat stage-2 run.
    sel = tmp_path / "concat_selection"
    sel.mkdir()
    ckpt = run_dir / "epoch_001.pt"
    (sel / "score_receipt.json").write_text(json.dumps({
        "schema": "b3s_full_score_v1", "status": "COMPLETED", "stage": "1_pretain", "task": "m2",
        "trunk_side": "concat", "view": "EMA", "partial": False, "run_dir": str(run_dir),
        "selection": {"epoch": 1},
        "ema_by_epoch": {"1": {"checkpoint_sha256": runner._sha(ckpt)}},
    }))
    (run_dir / "run_meta.json").write_text(json.dumps({
        "schema": runner.SCHEMA, "task": "m2", "stage": "1_pretain", "status": "FORMAL", "trunk_side": "concat", "epochs": 1,
    }))
    (run_dir / "train_receipt.json").write_text(json.dumps({"status": "COMPLETED"}))
    with pytest.raises(RuntimeError, match="side"):
        runner.resolve_formal_trunk(sel, task="m2", side="none")
    formal, fprov = runner.resolve_formal_trunk(sel, task="m2", side="concat")
    assert fprov["trunk_side"] == "concat" and fprov["view"] == "ema"
    assert all(torch.equal(formal[k], concat_trunk[k] + 0.25) for k in concat_trunk)





def test_m2_source_carriers_are_dual_track_T_and_ext6_has_six():
    carriers, binding = runner.load_m2_carriers()
    assert sorted(carriers) == sorted(binding["per_session"]) and len(carriers) == 7
    for session, value in carriers.items():
        assert value.shape == (96, 4) and value.dtype == np.float32 and np.isfinite(value).all()
    from btransform_unified_v1 import adapters

    probe = next(iter(sorted(carriers)))
    reference = np.ascontiguousarray(adapters.build_m2_bank("source_train", probe, budget=33).carrier, np.float32)
    assert np.array_equal(carriers[probe], reference)
    ext6_root = runner.resolve_m2_ext6_root()
    ext6, ext6_binding = runner.load_m2_ext6_carriers(ext6_root)
    assert len(ext6) == 6 and ext6_binding["root"] == str(ext6_root)
    for value in ext6.values():
        assert value.shape == (96, 4)


def test_m1_carrier_pack_binding_is_fail_closed():
    carriers, binding = runner.load_m1_carriers(runner.M1_CARRIER_PACK_DEFAULT)
    assert sorted(carriers) == ["20121004", "20121017", "20121024", "ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"]
    with np.load(runner.M1_CARRIER_PACK_DEFAULT) as archive:
        for session, value in carriers.items():
            assert value.shape == (64, 4) and np.array_equal(value, archive[session])
    assert binding["fit_sha256"] and len(binding["fit_sha256"]) == 64


def test_h1_m3_carriers_align_with_activity_support_and_ho_payload():
    activity = _load("activity_data")
    support = {s: item["support_provenance"]["trial_ids"] for s, item in activity.load_h1_data()["train"].items()}
    carriers, binding = runner.load_h1_carriers(runner.H1_BANKS_DEFAULT, support)
    assert sorted(binding["per_session"]) == sorted(support)
    for session, value in carriers.items():
        assert value.shape == (176, 4) and np.isfinite(value).all()
    # Deterministic recompute: identical bytes on a second load.
    again, _ = runner.load_h1_carriers(runner.H1_BANKS_DEFAULT, support)
    assert all(np.array_equal(carriers[k], again[k]) for k in carriers)
    # HO payload carriers are the 27-tag bank rows for the falcon groups.
    assert len([k for k in binding["per_session_ho"] if k.startswith("ses-")]) == 14
    assert not np.array_equal(carriers["ses-19250101T111740"], carriers["ses-19250101T112404"])


def test_formal_trunk_rejects_non_b3s_stage1_sources(tmp_path=None) -> None:
    """P1 fix: formal stage 2 binds only completed same-task stage-1 selections."""
    import json
    from pathlib import Path
    import activity_full_train as train

    pkg = Path(__file__).resolve().parents[1]
    # (a) the superseded ACT v2 selection dir must be rejected
    act_sel = pkg / "results/selection_m2_activity_only_v2_early_pool_s42_ext6"
    if act_sel.is_dir():
        try:
            train.resolve_formal_trunk(act_sel, task="m2")
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("ACT v2 selection accepted as formal trunk source")
    # (b) bare checkpoints / run dirs are rejected outright
    ckpt = pkg / "results/m2_activity_only_v2_early_pool_s42/epoch_020.pt"
    try:
        train.resolve_formal_trunk(ckpt, task="m2")
    except ValueError:
        pass
    else:
        raise AssertionError("bare checkpoint accepted as formal trunk source")
    # (c) a receipt with the wrong task / stage / partial scan is rejected
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "score_receipt.json").write_text(json.dumps({
            "schema": "b3s_full_score_v1", "status": "COMPLETED", "stage": "1_pretain",
            "task": "h1", "view": "EMA", "partial": True, "run_dir": str(d),
            "selection": {"epoch": 1}, "ema_by_epoch": {},
        }))
        try:
            train.resolve_formal_trunk(d, task="m2")
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("mismatched-task partial-scan receipt accepted")
