"""Phase 0 CPU smoke for btransform_unified_v1 (B-transformer series, NOT SPINT).

Covers the workorder P0 gate: geometry instantiation / causality self-check /
static-fold parity / EMA numerics / schedule / R^2 dual report / sealed
receipts / synthetic convergence / dropout semantics / scale bridge /
alignment-table template. CPU only; no training beyond a 60-step synthetic
smoke; no official or EvalAI surface; no historical result root touched.
"""

from __future__ import annotations

import dataclasses
import os
import stat
import typing
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from btransform_unified_v1 import adapters, bank as bank_mod, plan, receipts
from btransform_unified_v1.bank import TaskBank, array_sha256, make_synthetic_bank
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import (
    BTransformerUnifiedDecoder,
    unit_dropout_seed,
    whole_unit_dropout,
)
from btransform_unified_v1.r2 import equal_session_mean, session_mean_report, variance_weighted_r2
from btransform_unified_v1.scale_bridge import assert_scale_bridge
from btransform_unified_v1.schedule import warmup_cosine_lr

# Blueprint invariant: the m2 build of the unified skeleton has exactly the
# S1-SMALL-COS decoder parameter count (3,543,010) — the split first token
# layer keeps the same element budget as Linear(70 -> 256).
M2_PARAM_COUNT = 3_543_010

# ---------------------------------------------------------------------------
# plan / bank contracts
# ---------------------------------------------------------------------------


def test_plan_schema_geometry_and_constants() -> None:
    assert plan.SCHEMA == "btransform_unified_v1"
    assert plan.TASK_GEOMETRY["m2"] == {
        "window": 50, "prefix": 50, "units": 96, "e0_dim": 50,
        "carrier_dim": 4, "out_dim": 2, "target_scale": 5.0,
    }
    assert plan.TASK_GEOMETRY["m1"] == {
        "window": 100, "prefix": 100, "units": 64, "e0_dim": 100,
        "carrier_dim": 4, "out_dim": 16, "target_scale": 1.0,
    }
    h1 = plan.TASK_GEOMETRY["h1"]
    # code item P: the H1 prefix carries the frozen PENDING sentinel (workorder
    # §7 latency-budget gate), never a bare integer like 350.
    assert (h1["window"], h1["units"], h1["carrier_dim"], h1["out_dim"]) == (700, 176, 4, 7)
    assert h1["prefix"] == plan.H1_PREFIX_PENDING == "H1_PREFIX_PENDING"
    assert h1["e0_dim"] == "PENDING_SEC6_CONTROL"
    assert h1["target_scale"] == "PENDING"
    assert plan.LR_PEAK == pytest.approx(3e-4)
    assert plan.LR_MIN_FACTOR == 0.1
    assert plan.EPOCHS == 24 and plan.WARMUP_EPOCHS == 1
    # code item Q: update-caliber constants (epochs are never the unit).
    assert plan.UPDATES_PER_EPOCH == {"m2": 3165, "m1": None, "h1": 731}
    assert plan.EMA_HORIZON_UPDATES == 2000
    assert plan.EMA_DECAY == 0.9995
    assert plan.UNIT_DROPOUT == pytest.approx(0.10)
    assert plan.BATCH_SIZE == 32 and plan.WEIGHT_DECAY == pytest.approx(0.01)
    assert plan.GRAD_CLIP == 1.0 and plan.SEED == 42
    assert len(plan.GPU_UUIDS) == 2
    assert plan.REPO_ROOT.name == "btransform_unified_v1"
    assert plan.ALIGNMENT_TABLE_FIELDS == [
        "system", "consumer", "calibration_object", "scoring_surface",
        "scale", "single_difference_vs_historical_best",
    ]


def test_require_raises_contract_error() -> None:
    plan.require(True, "never")
    with pytest.raises(plan.BTransformerUnifiedError):
        plan.require(False, "contract broken")


def test_synthetic_bank_contract() -> None:
    bank = make_synthetic_bank("m2", n_windows=8, seed=3)
    geo = plan.TASK_GEOMETRY["m2"]
    assert bank.E0.shape == (geo["units"], geo["e0_dim"])
    assert bank.carrier.shape == (geo["units"], 4)
    assert bank.unit_mask.dtype == np.bool_ and bool(bank.unit_mask.any())
    assert bank.X_store.shape == (8, geo["window"] + geo["prefix"], geo["units"])
    assert bank.target_store.shape == (8, geo["out_dim"])
    assert bank.window_ids.shape == (8,)
    for key in ("shape", "trial_count", "estimator", "array_sha256"):
        assert key in bank.calibration_meta, key
    assert bank.calibration_meta["shape"] == bank.E0.shape
    assert bank.calibration_meta["array_sha256"] == array_sha256(bank.E0)
    assert np.isfinite(bank.E0).all() and np.isfinite(bank.carrier).all()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bank.session_id = "mutated"  # type: ignore[misc]


def test_bank_post_init_rejects_bad_shapes_and_missing_meta() -> None:
    good = make_synthetic_bank("m1", n_windows=2, seed=1)
    with pytest.raises(plan.BTransformerUnifiedError):
        # E0 with wrong trailing dim
        TaskBank(
            session_id=good.session_id, E0=good.E0[:, :3], carrier=good.carrier,
            unit_mask=good.unit_mask, X_store=good.X_store, target_store=good.target_store,
            window_ids=good.window_ids, calibration_meta=dict(good.calibration_meta),
        )
    with pytest.raises(plan.BTransformerUnifiedError):
        # carrier not [N, 4]
        TaskBank(
            session_id=good.session_id, E0=good.E0, carrier=good.carrier[:, :2],
            unit_mask=good.unit_mask, X_store=good.X_store, target_store=good.target_store,
            window_ids=good.window_ids, calibration_meta=dict(good.calibration_meta),
        )
    with pytest.raises(plan.BTransformerUnifiedError):
        # missing NOTE P0-2 keys
        meta = {k: v for k, v in good.calibration_meta.items() if k != "estimator"}
        TaskBank(
            session_id=good.session_id, E0=good.E0, carrier=good.carrier,
            unit_mask=good.unit_mask, X_store=good.X_store, target_store=good.target_store,
            window_ids=good.window_ids, calibration_meta=meta,
        )
    with pytest.raises(plan.BTransformerUnifiedError):
        # NaN in E0 violates the P0-5 observation contract
        e0_nan = good.E0.copy(); e0_nan[0, 0] = np.nan
        TaskBank(
            session_id=good.session_id, E0=e0_nan, carrier=good.carrier,
            unit_mask=good.unit_mask, X_store=good.X_store, target_store=good.target_store,
            window_ids=good.window_ids, calibration_meta=dict(good.calibration_meta),
        )


def test_bank_requires_budget_key_and_synthetic_defaults() -> None:
    # code item S: budget (CAL calibration-trial count) is a REQUIRED
    # calibration_meta key — CAL-1 stores one bank per (session, M), so the
    # budget must travel with the bank.
    assert "budget" in bank_mod.CALIBRATION_META_REQUIRED_KEYS
    good = make_synthetic_bank("m2", n_windows=2, seed=2)
    with pytest.raises(plan.BTransformerUnifiedError):
        meta = {k: v for k, v in good.calibration_meta.items() if k != "budget"}
        TaskBank(
            session_id=good.session_id, E0=good.E0, carrier=good.carrier,
            unit_mask=good.unit_mask, X_store=good.X_store, target_store=good.target_store,
            window_ids=good.window_ids, calibration_meta=meta,
        )
    with pytest.raises(plan.BTransformerUnifiedError):
        # budget present but not a positive int
        bad = dict(good.calibration_meta, budget=0)
        TaskBank(
            session_id=good.session_id, E0=good.E0, carrier=good.carrier,
            unit_mask=good.unit_mask, X_store=good.X_store, target_store=good.target_store,
            window_ids=good.window_ids, calibration_meta=bad,
        )
    # synthetic defaults: m2 M33 (S1 parity), m1 M10, h1 deploy M3; explicit
    # budget mimics a CAL-1 multi-budget bank
    assert make_synthetic_bank("m2", n_windows=2, seed=0).calibration_meta["budget"] == 33
    assert make_synthetic_bank("m1", n_windows=2, seed=0).calibration_meta["budget"] == 10
    h1_bank = make_synthetic_bank("h1", n_windows=2, seed=0, e0_dim=700, prefix=700)
    assert h1_bank.calibration_meta["budget"] == 3
    assert make_synthetic_bank(
        "h1", n_windows=2, seed=0, e0_dim=700, prefix=700, budget=7
    ).calibration_meta["budget"] == 7  # CAL-1 prefix-cycle member {7,5,4,3}


# ---------------------------------------------------------------------------
# (a) geometry instantiation + forward shapes (h1 = temporary PENDING geometry)
# ---------------------------------------------------------------------------


def test_m2_and_m1_forward_shapes() -> None:
    for task, out_dim in (("m2", 2), ("m1", 16)):
        bank = make_synthetic_bank(task, n_windows=4, seed=11)
        model = BTransformerUnifiedDecoder(task, seed=42).eval()
        x = torch.from_numpy(bank.X_store[:3])
        with torch.no_grad():
            y = model(x, bank)
            scores = model.forward_scores(x, bank)
        assert y.shape == (3, out_dim)
        assert scores.shape == (3, x.size(1), out_dim)
        assert torch.isfinite(y).all()
    model_m2 = BTransformerUnifiedDecoder("m2", seed=42).eval()
    assert sum(p.numel() for p in model_m2.trainable_parameters().values()) == M2_PARAM_COUNT


def test_h1_temporary_geometry_shape_only_pending() -> None:
    # PENDING (NOTE §6 control not closed; workorder §7 latency gate): both
    # e0_dim=700 and override_prefix=700 (a 700+700 window) are TEMPORARY dev
    # values for shape smoke only — they back no alignment claim, and
    # building h1 by name must refuse on the sentinels.
    with pytest.raises(plan.BTransformerUnifiedError):
        BTransformerUnifiedDecoder("h1", seed=42)
    geo = dict(plan.TASK_GEOMETRY["h1"])
    geo["e0_dim"] = 700
    # code item P: the prefix sentinel refuses without an explicit override.
    with pytest.raises(plan.BTransformerUnifiedError, match="H1_PREFIX_PENDING"):
        BTransformerUnifiedDecoder(geo, seed=42)
    bank = make_synthetic_bank("h1", n_windows=2, seed=6, e0_dim=700, prefix=700)
    model = BTransformerUnifiedDecoder(geo, seed=42, override_prefix=700).eval()
    x = torch.from_numpy(bank.X_store[:1])
    with torch.no_grad():
        y = model(x, bank)
    assert y.shape == (1, 7)
    assert model.e0_dim == 700 and model.l_in == 700 + 700
    assert model.pending_prefix == plan.H1_PREFIX_PENDING
    # a settled integer prefix is frozen: no override allowed (m2 sanity)
    with pytest.raises(plan.BTransformerUnifiedError):
        BTransformerUnifiedDecoder("m2", seed=42, override_prefix=10)


# ---------------------------------------------------------------------------
# (L) init source = initialize_decoder | fallback, param count invariant
# ---------------------------------------------------------------------------


def test_init_source_initialize_decoder_and_param_count() -> None:
    model = BTransformerUnifiedDecoder("m2", seed=42)
    # initialize_decoder re-initializes by module name; the structure (and
    # therefore the blueprint parameter count) is untouched.
    assert sum(p.numel() for p in model.trainable_parameters().values()) == M2_PARAM_COUNT
    meta = model.init_meta
    assert meta["source"] in ("initialize_decoder", "fallback")
    assert isinstance(meta["fallback"], bool)
    assert meta["fallback"] == (meta["source"] == "fallback")
    assert meta["seed"] == 42
    if not meta["fallback"]:
        assert meta["origin"] == "tfpd_exploration.src.m2_dual_track_v1.decoders"
    # determinism: same seed -> bitwise identical init on a fresh build
    model2 = BTransformerUnifiedDecoder("m2", seed=42)
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert n1 == n2 and torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# (M) generator-reproducible whole-unit dropout
# ---------------------------------------------------------------------------


def test_dropout_generator_reproducible_bitwise() -> None:
    bank = make_synthetic_bank("m2", n_windows=4, seed=8)
    model = BTransformerUnifiedDecoder("m2", seed=42).train()
    x = torch.from_numpy(bank.X_store)
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        a = model(x, bank, dropout_generator=g1)
        b = model(x, bank, dropout_generator=g2)
    assert torch.equal(a, b)  # same seed -> bitwise identical dropout draw
    g3 = torch.Generator().manual_seed(1235)
    with torch.no_grad():
        c = model(x, bank, dropout_generator=g3)
    assert not torch.equal(a, c)  # different seed -> different mask (N=96)
    # unit_dropout_seed mirrors m2_b_small_stability_v1: payload-string sha
    assert unit_dropout_seed(42, 0, 5) == unit_dropout_seed(42, 0, 5)
    assert unit_dropout_seed(42, 0, 5) != unit_dropout_seed(42, 0, 6)
    assert unit_dropout_seed(42, 0, 5) != unit_dropout_seed(43, 0, 5)
    assert 0 <= unit_dropout_seed(42, 3, 17) < 2**63
    # forward_scores carries the same passthrough (per-bin readout)
    g4 = torch.Generator().manual_seed(unit_dropout_seed(42, 0, 0))
    g5 = torch.Generator().manual_seed(unit_dropout_seed(42, 0, 0))
    with torch.no_grad():
        s1 = model.forward_scores(x, bank, dropout_generator=g4)
        s2 = model.forward_scores(x, bank, dropout_generator=g5)
    assert torch.equal(s1, s2)


# ---------------------------------------------------------------------------
# (N) input length pinned to l_in in forward
# ---------------------------------------------------------------------------


def test_forward_pins_input_length() -> None:
    bank = make_synthetic_bank("m2", n_windows=4, seed=5)
    model = BTransformerUnifiedDecoder("m2", seed=42).eval()
    assert model.l_in == plan.TASK_GEOMETRY["m2"]["window"] + plan.TASK_GEOMETRY["m2"]["prefix"]
    # shorter than l_in -> refuse (PE counted from the window head)
    short = torch.from_numpy(bank.X_store[:2, :-1, :])
    with pytest.raises(plan.BTransformerUnifiedError):
        model(short, bank)
    # longer than l_in -> refuse as well
    pad = torch.zeros(2, model.l_in + 1, model.units)
    with pytest.raises(plan.BTransformerUnifiedError):
        model(pad, bank)
    # full length still works; causal_check keeps working at pinned length
    x = torch.from_numpy(bank.X_store[:2])
    with torch.no_grad():
        assert model(x, bank).shape == (2, model.out_dim)
    assert model.causal_check(x, bank)["passed"] is True


# ---------------------------------------------------------------------------
# (b) causal_check | (c) static-fold parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["m2", "m1"])
def test_causal_check_tolerance(task: str) -> None:
    # Route B: the default conv path is the F.conv1d primitive (S1 path),
    # which couples output positions at the ~1e-7 float level (Phase 0
    # record), so the causality gate is max|delta| <= 1e-5, not bit equality.
    bank = make_synthetic_bank(task, n_windows=4, seed=5)
    model = BTransformerUnifiedDecoder(task, seed=42).eval()
    x = torch.from_numpy(bank.X_store[:2])
    report = model.causal_check(x, bank)  # scrambles the trailing 5 bins
    assert report["passed"] is True
    assert report["scrambled_bins"] == 5
    assert report["tolerance"] == pytest.approx(1e-5)
    assert report["tail_within_tolerance"] and report["last_causal_bin_within_tolerance"]
    assert report["tail_max_abs_delta"] <= 1e-5
    assert report["last_causal_bin_max_abs_delta"] <= 1e-5
    assert report["perturbation_effective"]


def test_shared_causal_conv_primitive_vs_local_reference() -> None:
    # Route B: SharedCausalConv.forward IS the S1 pad+F.conv1d path; the
    # per-tap accumulation survives as forward_local_reference. The two
    # evaluate the identical linear causal operator — parity caliber 1e-5
    # (conv-primitive cross-position roundoff ~1e-7, Phase 0 record).
    from btransform_unified_v1.model import CAUSAL_CHECK_TOLERANCE, SharedCausalConv

    conv = SharedCausalConv()
    x = torch.randn(3, 17, 9)
    with torch.no_grad():
        y_prim = conv(x)
        y_ref = conv.forward_local_reference(x)
    assert y_prim.shape == y_ref.shape == (3, 17, 9, 16)
    delta = (y_prim - y_ref).abs().max().item()
    assert delta <= CAUSAL_CHECK_TOLERANCE, f"conv primitive vs local reference: {delta:.3e}"


def test_unit_dropout_seed_s1_domain_bitwise() -> None:
    # Route B: the dropout seed payload domain IS S1's —
    # "m2_small_unit_dropout|{seed}|{epoch}|{batch_id}" (sha256, first 8
    # bytes little-endian % 2**63). When the S1 module is importable the
    # imported function itself must match bit for bit.
    import hashlib

    s1 = pytest.importorskip("tfpd_exploration.src.m2_b_small_stability_v1.training")
    for seed, epoch, batch_id in ((42, 1, 0), (42, 1, 1), (42, 24, 3164), (7, 3, 17)):
        payload = f"m2_small_unit_dropout|{seed}|{epoch}|{batch_id}".encode()
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)
        assert unit_dropout_seed(seed, epoch, batch_id) == expected
        assert unit_dropout_seed(seed, epoch, batch_id) == s1.unit_dropout_seed(seed, epoch, batch_id)
    # a different domain string must NOT reproduce (guards against regressions
    # to the P1a btransform_unified_v1_unit_dropout payload)
    wrong = int.from_bytes(
        hashlib.sha256(b"btransform_unified_v1_unit_dropout|42|1|0").digest()[:8], "little"
    ) % (2**63)
    assert unit_dropout_seed(42, 1, 0) != wrong


def test_static_fold_parity_within_1e6() -> None:
    # The base forward keeps S1's single token GEMM; the folded path
    # (SPD-A1) evaluates the first token layer as split GEMMs over column
    # views of that ONE weight. Parity caliber is therefore the workorder
    # P3 gate (FP32 max|delta| <= 1e-6), not bitwise.
    bank = make_synthetic_bank("m2", n_windows=4, seed=9)
    model = BTransformerUnifiedDecoder("m2", seed=42).eval()
    x = torch.from_numpy(bank.X_store)
    static = model.bank_static_term(bank)
    assert static.shape == (96, 256)
    with torch.no_grad():
        y_ref = model(x, bank)
        y_folded = model.forward_static_folded(x, bank, static)
    delta = (y_ref - y_folded).abs().max().item()
    assert delta <= 1e-6, f"static-fold parity exceeded 1e-6: {delta:.3e}"


def test_s1_state_dict_key_parity_and_bitwise_init() -> None:
    # P1a prerequisite: the state-dict KEY SET (and shapes) of the m2 build
    # must equal SmallTransformerDecoder(seed=42) exactly, because
    # initialize_decoder consumes its RNG streams in sorted-name order —
    # identical (name, shape) sequences reproduce the S1 init bit for bit.
    pytest.importorskip("tfpd_exploration.src.m2_b_small_stability_v1.decoder")
    from tfpd_exploration.src.m2_b_small_stability_v1.decoder import SmallTransformerDecoder

    geometry = {"task": "m2", "window": 50, "prefix": 0, "units": 96,
                "e0_dim": 50, "carrier_dim": 4, "out_dim": 2}
    model = BTransformerUnifiedDecoder(geometry, seed=42)
    s1 = SmallTransformerDecoder(seed=42)
    own, ref = model.state_dict(), s1.state_dict()
    assert sorted(own) == sorted(ref) and len(ref) == 77
    for key in ref:
        assert own[key].shape == ref[key].shape, key
        assert torch.equal(own[key], ref[key]), f"init value drift at {key}"
    assert sum(p.numel() for p in model.parameters()) == M2_PARAM_COUNT
    report = model.assert_s1_state_dict_parity()
    assert report["keys_match"] and report["value_parity"] == "bitwise"


# ---------------------------------------------------------------------------
# (d) EMA | (e) schedule
# ---------------------------------------------------------------------------


def test_ema_decay_numerics_and_roundtrip() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema = DecoderEMA(lin)  # decay 0.9995
    assert ema.decay == pytest.approx(0.9995)
    ema.update_after_step(lin)  # first update copies RAW
    assert torch.equal(ema.shadow["weight"], torch.ones_like(lin.weight))
    with torch.no_grad():
        lin.weight.fill_(2.0)
    ema.update_after_step(lin)  # shadow = 0.9995*1 + 0.0005*2 (same op order)
    expected = torch.full_like(lin.weight, 1.0).mul_(0.9995).add_(
        torch.full_like(lin.weight, 2.0), alpha=1.0 - 0.9995
    )
    assert torch.equal(ema.shadow["weight"], expected)
    assert ema.n_updates == 2
    # state_dict / load_state_dict roundtrip on a fresh EMA
    state = ema.state_dict()
    lin2 = nn.Linear(3, 2, bias=False)
    ema2 = DecoderEMA(lin2)
    ema2.load_state_dict(state)
    assert ema2.n_updates == 2
    assert torch.equal(ema2.shadow["weight"], ema.shadow["weight"])
    # apply_to copies the shadow into RAW
    ema2.apply_to(lin2)
    assert torch.equal(lin2.weight, ema.shadow["weight"])


def test_warmup_cosine_schedule_points() -> None:
    peak = 3e-4
    warm, total = 10, 240
    assert warmup_cosine_lr(0, total, warm) == pytest.approx(0.0)
    assert warmup_cosine_lr(warm, total, warm, peak=peak) == peak  # step == warmup -> peak
    last = warmup_cosine_lr(total, total, warm, peak=peak)
    assert abs(last - 0.3e-4) < 1e-15  # min_factor 0.1 -> 3e-5
    lrs = [warmup_cosine_lr(s, total, warm, peak=peak) for s in range(warm, total + 1)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))  # cosine never rises
    # warmup disabled: pure cosine starting at peak
    assert warmup_cosine_lr(0, total, 0, peak=peak) == peak


def test_recipe_updates_update_caliber() -> None:
    # code item Q: the recipe transfers by UPDATE count, not epoch count.
    m2 = plan.recipe_updates("m2")
    assert m2 == {"warmup_updates": 3165, "total_updates": 75960, "ema_horizon_updates": 2000}
    h1 = plan.recipe_updates("h1")
    assert h1 == {"warmup_updates": 731, "total_updates": 731 * 24, "ema_horizon_updates": 2000}
    assert plan.recipe_updates("m2", epochs=12)["total_updates"] == 3165 * 12
    # m1's update caliber is PENDING until Phase 2a -> refuse, do not guess
    with pytest.raises(NotImplementedError):
        plan.recipe_updates("m1")
    with pytest.raises(plan.BTransformerUnifiedError):
        plan.recipe_updates("nope")


# ---------------------------------------------------------------------------
# (f) R^2 hand example + dual report
# ---------------------------------------------------------------------------


def test_r2_hand_example_and_dual_report() -> None:
    target = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 3.0])
    # SStot = 5.0, SSres = 1.0 -> R^2 = 0.8 (hand-computed)
    assert variance_weighted_r2(target, pred) == pytest.approx(0.8, abs=1e-12)
    sessions = np.array(["a", "a", "b", "b"])
    # session a: perfect -> 1.0 ; session b: SStot = 0.5, SSres = 1.0 -> -1.0
    assert equal_session_mean(target, pred, sessions) == pytest.approx(0.0, abs=1e-12)
    report = session_mean_report(target, pred, sessions)
    assert report["pooled_r2"] == pytest.approx(0.8, abs=1e-12)
    assert report["session_mean_r2"] == pytest.approx(0.0, abs=1e-12)
    assert report["per_session_r2"] == {"a": pytest.approx(1.0), "b": pytest.approx(-1.0)}
    assert report["n_points"] == 4 and report["n_sessions"] == 2
    # both surfaces present and explicitly non-interchangeable (NOTE P1-11)
    assert "pooled_r2" in report and "session_mean_r2" in report
    assert "never subtract" in report["contract"]
    with pytest.raises(plan.BTransformerUnifiedError):
        variance_weighted_r2(np.ones(4), np.ones(4))  # constant target undefined


# ---------------------------------------------------------------------------
# (g) sealed receipts
# ---------------------------------------------------------------------------


def test_seal_read_roundtrip_and_tamper_detection(tmp_path) -> None:
    path = tmp_path / "results" / "smoke_receipt.json"
    payload = {"schema": plan.SCHEMA, "task": "m2", "numbers": [1, 2, 3], "nested": {"b": 2, "a": 1}}
    digest = receipts.seal_json(path, payload)
    assert len(digest) == 64
    assert path.is_file()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o444  # read-only seal
    sidecar = receipts.sidecar_path(path)
    assert sidecar.is_file() and digest in sidecar.read_text()
    assert receipts.read_sealed(path) == payload
    # re-seal over an existing 0444 file is an atomic replace
    assert receipts.seal_json(path, payload) == digest
    assert receipts.read_sealed(path) == payload
    # tamper with the sealed file -> refuse
    os.chmod(path, 0o644)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"task": "m2"', '"task": "m1"'), encoding="utf-8")
    os.chmod(path, 0o444)
    with pytest.raises(plan.BTransformerUnifiedError):
        receipts.read_sealed(path)
    # tamper with the sidecar -> refuse
    sidecar.write_text("0" * 64 + f"  {path.name}\n", encoding="utf-8")
    with pytest.raises(plan.BTransformerUnifiedError):
        receipts.read_sealed(path)


# ---------------------------------------------------------------------------
# (h) synthetic convergence (60 steps, small lr, loss drop >= 30%)
# ---------------------------------------------------------------------------


def test_synthetic_convergence_sixty_steps() -> None:
    torch.manual_seed(0)
    bank = make_synthetic_bank("m2", n_windows=64, seed=7)
    model = BTransformerUnifiedDecoder("m2", seed=42).train()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters().values(), lr=plan.LR_PEAK, weight_decay=plan.WEIGHT_DECAY
    )
    x = torch.from_numpy(bank.X_store)
    y = torch.from_numpy(bank.target_store)
    sampler = torch.Generator().manual_seed(123)
    losses: list[float] = []
    for _ in range(60):
        idx = torch.randint(0, x.size(0), (16,), generator=sampler)
        loss = torch.nn.functional.mse_loss(model(x[idx], bank), y[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters().values(), plan.GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss))
    first = sum(losses[:3]) / 3.0
    last = sum(losses[-3:]) / 3.0
    assert last <= 0.70 * first, f"loss drop < 30%: first={first:.5f} last={last:.5f}"


# ---------------------------------------------------------------------------
# (i) whole-unit dropout only in training mode
# ---------------------------------------------------------------------------


def test_unit_dropout_only_in_training_mode() -> None:
    bank = make_synthetic_bank("m2", n_windows=8, seed=4)
    model = BTransformerUnifiedDecoder("m2", seed=42)
    x = torch.from_numpy(bank.X_store[:4])
    # eval: deterministic, no units dropped
    model.eval()
    with torch.no_grad():
        a, b = model(x, bank), model(x, bank)
    assert torch.equal(a, b)
    with torch.no_grad():
        explicit = model(x, bank, unit_mask=torch.from_numpy(bank.unit_mask))
    assert torch.equal(a, explicit)
    # train with p=0.10: stochastic across calls (identical-mask probability ~2e-9 at N=96)
    model.train()
    with torch.no_grad():
        c, d = model(x, bank), model(x, bank)
    assert not torch.equal(c, d)
    # train with p=0: deterministic again -> dropout is the only stochasticity
    model.unit_dropout_p = 0.0
    with torch.no_grad():
        e, f = model(x, bank), model(x, bank)
    assert torch.equal(e, f)
    # mask semantics: p=1.0 drops everything then restores the lowest
    # originally-valid index (exactly one kept unit per window)
    keep = whole_unit_dropout(torch.ones(200, 96, dtype=torch.bool), p=1.0)
    assert bool(keep[:, 0].all()) and int(keep.sum()) == keep.size(0)
    partial = whole_unit_dropout(torch.ones(50, 96, dtype=torch.bool), p=0.10)
    assert partial.dtype == torch.bool and bool(partial.any(dim=1).all())
    dropped_fraction = 1.0 - partial.float().mean().item()
    assert 0.02 < dropped_fraction < 0.20


# ---------------------------------------------------------------------------
# (j) scale bridge | (k) alignment table
# ---------------------------------------------------------------------------


def test_scale_bridge_m2_passes_h1_ratio_acceptance() -> None:
    rng = np.random.default_rng(0)
    native = rng.standard_normal((64, 2))
    raw = 5.0 * rng.standard_normal((64, 2))  # decoder_raw scale
    bridged = assert_scale_bridge("m2", raw, native)
    assert bridged.shape == raw.shape
    assert np.allclose(bridged, raw / 5.0)
    # m1: divisor = 1, identity bridge
    raw_m1 = rng.standard_normal((64, 16))
    out_m1 = assert_scale_bridge("m1", raw_m1, rng.standard_normal((64, 16)))
    assert np.array_equal(out_m1, raw_m1)
    # near-constant m2 output trips the P0-3 guard
    with pytest.raises(plan.BTransformerUnifiedError):
        assert_scale_bridge("m2", np.zeros((64, 2)), native)
    # h1: x20 RATIO bridge implemented (2026-09-06, MATRIX §0 / NOTE P0-3):
    # MSE(raw,20y) == 400 * MSE(raw/20,y) at relative tolerance 1e-9 — the
    # former PENDING raise is retired.
    raw_h1 = rng.standard_normal((32, 7))
    native_h1 = rng.standard_normal((32, 7))
    bridged_h1 = assert_scale_bridge("h1", raw_h1, native_h1)
    assert bridged_h1.shape == raw_h1.shape
    assert np.allclose(bridged_h1, raw_h1 / 20.0)
    with pytest.raises(plan.BTransformerUnifiedError):
        assert_scale_bridge("h1", np.zeros((32, 7)), native_h1)  # std guard


def test_alignment_table_six_rows_all_fields() -> None:
    for task in ("m2", "m1", "h1"):
        table = plan.alignment_table(task)
        assert list(table.keys()) == plan.ALIGNMENT_TABLE_FIELDS
        assert len(table) == 6
        assert all(isinstance(v, str) and v.strip() for v in table.values())
        assert "NOT SPINT" in table["system"]  # identity declaration frozen
        assert "SPINT" in table["consumer"]
    assert "PENDING" in plan.alignment_table("h1")["scale"]
    assert "§6" in plan.alignment_table("h1")["calibration_object"]


# ---------------------------------------------------------------------------
# adapter stubs carry their phase gates
# ---------------------------------------------------------------------------


def test_adapter_builders_phase_gates() -> None:
    # code item R: budget is a REQUIRED keyword on every builder. m2 is live
    # from Phase 1a (frozen dual_track cache -> TaskBank, CAL-2 M33); m1/h1
    # keep their NotImplementedError phase gates.
    for builder, gate, budget in (
        (adapters.build_m1_bank, "Phase 2a", 10),
        (adapters.build_h1_bank, "Phase 2b", 3),
    ):
        with pytest.raises(NotImplementedError, match=gate):
            builder(budget=budget)
        assert typing.get_type_hints(builder)["return"] is TaskBank
        assert typing.get_type_hints(builder)["budget"] is int
        assert bank_mod.TaskBank is TaskBank
    # m2 builder contract: CAL-2 budget must be 33, cache-backed.
    assert "budget" in typing.get_type_hints(adapters.build_m2_bank)
    assert typing.get_type_hints(adapters.build_m2_bank)["return"] is TaskBank
    with pytest.raises(plan.BTransformerUnifiedError):
        adapters.build_m2_bank("source_minival", "ses-2020-10-19-Run1", budget=10)
    with pytest.raises(plan.BTransformerUnifiedError):
        adapters.build_m2_bank("not-a-surface", "ses-2020-10-19-Run1")
    cache = Path(
        plan.REPO_ROOT.parent
        / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
    )
    if not cache.is_dir():  # environments without the frozen cache
        pytest.skip("frozen m2 dual_track cache unavailable")
    bank = adapters.build_m2_bank("source_minival", "ses-2020-10-19-Run1")
    assert isinstance(bank, TaskBank)
    assert bank.E0.shape == (96, 50) and bank.carrier.shape == (96, 4)
    assert bank.unit_mask.shape == (96,) and bool(bank.unit_mask.all())
    assert bank.X_store.ndim == 3 and bank.X_store.shape[1:] == (50, 96)
    assert bank.X_store.shape[0] == bank.target_store.shape[0] == bank.window_ids.shape[0]
    assert bank.calibration_meta["budget"] == 33
    assert bank.calibration_meta["trial_count"] == 33
    assert "array_sha256" in bank.calibration_meta
    assert bank.calibration_meta["array_sha256"] == array_sha256(bank.E0)


def test_cpu_only_environment() -> None:
    # Phase 0 hard constraint: no CUDA initialization anywhere in this suite.
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
    assert not torch.cuda.is_initialized()
