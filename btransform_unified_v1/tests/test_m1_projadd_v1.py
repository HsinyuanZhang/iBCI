"""M1 proj_add series tests (ADDENDUM-UNIFIED-ADD M1 leg, 2026-09-06).

CPU only (conftest masks CUDA): geometry/interface contract for both series
cells, P-fold (SPD-A1) parity, the DEFERRED-locked bank contract with a stub
_family_bank identity provider (budget=10), the divisor=1 scale identity, the
script stage enumeration, and a hand-computed R^2. No real NWB file, no
checkpoint load, no GPU, no training. NOT SPINT.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from btransform_unified_v1 import m1_projadd as mp
from btransform_unified_v1 import plan
from btransform_unified_v1.bank import array_sha256
from btransform_unified_v1.identity_variant import (
    BTransformerUnifiedDecoderIdentity,
    token_e0_dim,
)
from btransform_unified_v1.model import (
    CONV_CHANNELS,
    SET_DIM,
    BTransformerUnifiedDecoder,
)
from btransform_unified_v1.r2 import session_mean_report, variance_weighted_r2

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "m1_projadd_series.py"


def _bank_for(proj_dim: int, seed: int = 5, n_windows: int = 2) -> object:
    """Synthetic [64,100]-E0 bank satisfying the M1 bank contract."""
    rng = np.random.default_rng(seed)
    e0 = rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32)
    carrier = rng.standard_normal((mp.M1_UNITS, mp.M1_CARRIER_DIM)).astype(np.float32)
    x_store = rng.standard_normal((n_windows, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32)
    return mp.make_m1_bank(
        mp.M1_OUTER_SESSION,
        e0,
        carrier,
        X_store=x_store,
        target_store=np.zeros((n_windows, mp.M1_OUT_DIM), dtype=np.float32),
        window_ids=np.arange(n_windows, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def models() -> dict[int, BTransformerUnifiedDecoderIdentity]:
    return {p: mp.build_m1_projadd_model(p, seed=42).eval() for p in mp.SERIES_PROJ_DIMS}


# ---------------------------------------------------------------------------
# (1) M1 geometry: both proj_dims instantiate, forward shapes, causal_check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("proj_dim", mp.SERIES_PROJ_DIMS)
def test_m1_geometry_contract(models, proj_dim: int) -> None:
    model = models[proj_dim]
    assert model.task == "m1"
    assert model.window == mp.M1_WINDOW and model.prefix == mp.M1_PREFIX
    assert model.l_in == 100 and model.base_e0_dim == 100
    assert model.units == 64 and model.out_dim == 16
    assert model.identity_mode == "proj_add" and model.init_meta["matrix_letter"] == "f"
    assert model.token_in == proj_dim + 4  # [local-block(R) | carrier4]
    assert model.proj_out_dim == proj_dim and model.proj_groups == proj_dim // CONV_CHANNELS
    assert tuple(model.frontend.e0_proj.weight.shape) == (proj_dim, 100)
    assert model.frontend.e0_proj.bias is None
    assert model.init_meta["proj_dim_override"] == proj_dim
    assert model.init_meta["param_count"] == mp.m1_projadd_param_count(proj_dim)


@pytest.mark.parametrize("proj_dim", mp.SERIES_PROJ_DIMS)
def test_m1_forward_shapes_and_causal_check(models, proj_dim: int) -> None:
    model = models[proj_dim]
    bank = _bank_for(proj_dim, seed=proj_dim)
    x = torch.from_numpy(np.ascontiguousarray(bank.X_store))
    with torch.no_grad():
        y = model(x, bank)
        scores = model.forward_scores(x, bank)
    assert y.shape == (x.size(0), mp.M1_OUT_DIM)
    assert scores.shape == (x.size(0), mp.M1_WINDOW, mp.M1_OUT_DIM)
    assert torch.isfinite(y).all() and torch.isfinite(scores).all()
    report = model.causal_check(x, bank)
    assert report["passed"] is True
    assert report["length"] == mp.M1_WINDOW


def test_m1_geometry_does_not_touch_frozen_plan_table() -> None:
    # The DEFERRED-locked explicit mapping (prefix=0) never edits TASK_GEOMETRY.
    assert plan.TASK_GEOMETRY["m1"]["prefix"] == 100  # frozen table untouched
    geometry = mp.m1_projadd_geometry(16)
    assert geometry["prefix"] == 0 and geometry["window"] == 100
    assert geometry is not plan.TASK_GEOMETRY["m1"]
    with pytest.raises(plan.BTransformerUnifiedError):
        mp.m1_projadd_geometry(24)  # not a series cell


# ---------------------------------------------------------------------------
# (2) P fold (SPD-A1): parity + hand-checked grouped static term
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("proj_dim", mp.SERIES_PROJ_DIMS)
def test_m1_fold_parity(models, proj_dim: int) -> None:
    model = models[proj_dim]
    bank = _bank_for(proj_dim, seed=11)
    x = torch.from_numpy(np.ascontiguousarray(bank.X_store))
    static = model.bank_static_term(bank)
    assert static.shape == (mp.M1_UNITS, SET_DIM)
    delta = mp.assert_fold_parity(model, bank, x, static, tolerance=1e-6)
    assert delta <= 1e-6


@pytest.mark.parametrize("proj_dim", mp.SERIES_PROJ_DIMS)
def test_static_term_matches_manual_group_fold(models, proj_dim: int) -> None:
    """static_term == sum_g P_g @ W_g^T + carrier @ W_carrier^T + b, by hand."""
    model = models[proj_dim]
    rng = np.random.default_rng(23)
    e0 = rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32)
    carrier = rng.standard_normal((mp.M1_UNITS, mp.M1_CARRIER_DIM)).astype(np.float32)
    static = model.static_term(torch.from_numpy(e0), torch.from_numpy(carrier))
    # manual computation
    with torch.no_grad():
        proj = model.frontend.e0_proj(torch.from_numpy(e0))  # [N, R]
        weight = model.frontend.first_token_weight()  # [SET_DIM, R+4]
        groups = proj.reshape(mp.M1_UNITS, model.proj_groups, CONV_CHANNELS)
        w_groups = weight[:, :proj_dim].reshape(SET_DIM, model.proj_groups, CONV_CHANNELS)
        manual = torch.einsum("ngc,ogc->no", groups, w_groups)
        manual = manual + torch.from_numpy(carrier) @ weight[:, proj_dim:].t()  # [N,4]@[4,256]
        manual = manual + model.frontend.first_token_bias()
    assert torch.allclose(static, manual, atol=1e-5), float((static - manual).abs().max())


@pytest.mark.parametrize("proj_dim", mp.SERIES_PROJ_DIMS)
def test_p_zero_degenerates_to_e0_free_pathway(models, proj_dim: int) -> None:
    """P = 0 -> forward is bitwise independent of the bank E0 (zero semantics)."""
    model = models[proj_dim]
    rng = np.random.default_rng(31)
    x = torch.from_numpy(rng.standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32))
    bank_a = _bank_for(proj_dim, seed=41)
    e0_alt = rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32)
    bank_b = mp.make_m1_bank(
        mp.M1_OUTER_SESSION,
        e0_alt,
        bank_a.carrier,
        X_store=bank_a.X_store,
        target_store=bank_a.target_store,
        window_ids=bank_a.window_ids,
    )
    with torch.no_grad():
        model.frontend.e0_proj.weight.zero_()
        assert torch.equal(model.forward_scores(x, bank_a), model.forward_scores(x, bank_b))


def test_default_proj_dim_stays_settled_p16_build() -> None:
    """proj_dim=None must reproduce the settled P=16 interface (M2/H1 cells)."""
    geometry = mp.m1_projadd_geometry(16)
    default = BTransformerUnifiedDecoderIdentity(geometry, seed=42, identity_mode="proj_add").eval()
    p16 = mp.build_m1_projadd_model(16, seed=42).eval()
    assert default.proj_out_dim == 16 == p16.proj_out_dim
    assert default.token_in == 20
    assert sorted(default.state_dict()) == sorted(p16.state_dict())
    for key, tensor in p16.state_dict().items():
        assert torch.equal(default.state_dict()[key], tensor), key
    # token_e0_dim width arithmetic: P16 -> 0 token-side, P32 -> 16 token-side
    assert token_e0_dim(100, "proj_add") == 0
    assert token_e0_dim(100, "proj_add", proj_dim=16) == 0
    assert token_e0_dim(100, "proj_add", proj_dim=32) == 16
    with pytest.raises(plan.BTransformerUnifiedError):
        token_e0_dim(100, "proj_add", proj_dim=8)  # would truncate the local block


# ---------------------------------------------------------------------------
# (3) Bank contract: the _family_bank mechanism with a stub identity provider
# ---------------------------------------------------------------------------


class _StubEarlyPoolEncoder:
    """Duck-typed EarlyPoolEncoder: reset_stream/push_trial/finalize_identity."""

    def __init__(self, out_shape=(mp.M1_UNITS, mp.M1_E0_DIM), seed: int = 0) -> None:
        self.out_shape = out_shape
        self.seed = seed
        self.pushed = 0

    def reset_stream(self, batch: int, n_units: int, device, dtype):
        assert batch == 1 and n_units == mp.M1_UNITS
        return {"trials": []}

    def push_trial(self, stream, trial: torch.Tensor) -> None:
        assert trial.dim() == 3 and trial.shape[0] == 1  # [1, 1024, 64]
        stream["trials"].append(trial)
        self.pushed += 1

    def finalize_identity(self, stream) -> torch.Tensor:
        assert len(stream["trials"]) == mp.M10_BUDGET
        generator = torch.Generator().manual_seed(self.seed)
        flat = torch.rand(self.out_shape, generator=generator)
        return flat.unsqueeze(0)  # [1, N, d_e] — b3_identity squeezes batch


def test_b3_identity_squeezes_stream_batch_and_checks_shape() -> None:
    encoder = _StubEarlyPoolEncoder(seed=3)
    calib = np.random.default_rng(1).standard_normal((mp.M10_BUDGET, 1024, mp.M1_UNITS)).astype(np.float32)
    identity = mp.b3_identity(encoder, calib)
    assert encoder.pushed == mp.M10_BUDGET
    assert identity.shape == (mp.M1_UNITS, mp.M1_E0_DIM)
    assert identity.dtype == np.float32
    # wrong identity shape must be refused
    bad = _StubEarlyPoolEncoder(out_shape=(mp.M1_UNITS, 50))
    with pytest.raises(RuntimeError, match="identity shape drift"):
        mp.b3_identity(bad, calib)


def test_build_banks_contract_with_stub_provider() -> None:
    """Full bank contract via the DEFERRED _family_bank mechanism (stubbed)."""
    rng = np.random.default_rng(2)
    calib = {
        name: rng.standard_normal((mp.M10_BUDGET, 1024, mp.M1_UNITS)).astype(np.float32)
        for name in mp.M1_SESSIONS
    }
    carriers = {
        name: rng.standard_normal((mp.M1_UNITS, mp.M1_CARRIER_DIM)).astype(np.float32)
        for name in mp.M1_SESSIONS
    }
    banks, report = mp.build_banks(
        calib, carriers, identity_provider=lambda _calib: np.full((mp.M1_UNITS, mp.M1_E0_DIM), 0.5, np.float32)
    )
    assert sorted(banks) == sorted(mp.M1_SESSIONS)
    for name, bank in banks.items():
        assert isinstance(bank, mp.TaskBank)
        assert bank.session_id == name
        assert bank.E0.shape == (mp.M1_UNITS, mp.M1_E0_DIM)
        assert bank.carrier.shape == (mp.M1_UNITS, mp.M1_CARRIER_DIM)
        assert bool(bank.unit_mask.all())
        meta = bank.calibration_meta
        for key in ("shape", "trial_count", "estimator", "array_sha256", "budget"):
            assert key in meta, key
        assert meta["budget"] == mp.M10_BUDGET == 10  # CAL-2 M10 (code item S)
        assert meta["trial_count"] == mp.M10_BUDGET
        assert meta["shape"] == (mp.M1_UNITS, mp.M1_E0_DIM)
        assert meta["array_sha256"] == array_sha256(bank.E0)
        assert meta["carrier_sha256"] == array_sha256(bank.carrier)
        # outer session carrier must declare the sealed-basis encode path
        expected_carrier_source = (
            "sealed-basis-encode" if name == mp.M1_OUTER_SESSION else "rSyn3-refit-v1.source-only.npz normalized"
        )
        assert meta["carrier_source"] == expected_carrier_source
        assert report[name]["budget"] == mp.M10_BUDGET
    # window stores stay empty (fold-local FalconDataset serves windows)
    for bank in banks.values():
        assert bank.X_store.shape == (0, mp.M1_WINDOW, mp.M1_UNITS)
    # the frozen model consumes every bank
    model = mp.build_m1_projadd_model(16, seed=42).eval()
    for name, bank in banks.items():
        x = torch.zeros(1, mp.M1_WINDOW, mp.M1_UNITS)
        with torch.no_grad():
            assert model(x, bank).shape == (1, mp.M1_OUT_DIM)


def test_build_banks_refuses_incomplete_inputs() -> None:
    rng = np.random.default_rng(4)
    good_calib = {
        name: rng.standard_normal((mp.M10_BUDGET, 1024, mp.M1_UNITS)).astype(np.float32)
        for name in mp.M1_SESSIONS
    }
    carriers = {
        name: rng.standard_normal((mp.M1_UNITS, mp.M1_CARRIER_DIM)).astype(np.float32)
        for name in mp.M1_SESSIONS
    }
    with pytest.raises(plan.BTransformerUnifiedError, match="calib sessions"):
        mp.build_banks(
            {k: v for k, v in list(good_calib.items())[:3]},
            carriers,
            identity_provider=lambda c: np.zeros((mp.M1_UNITS, mp.M1_E0_DIM), np.float32),
        )
    with pytest.raises(RuntimeError, match="fewer than 10 calib trials"):
        short = dict(good_calib)
        short[mp.M1_OUTER_SESSION] = good_calib[mp.M1_OUTER_SESSION][:5]
        mp.build_banks(short, carriers, identity_provider=lambda c: np.zeros((64, 100), np.float32))


# ---------------------------------------------------------------------------
# (4) Scale = identity: divisor 1
# ---------------------------------------------------------------------------


def test_divisor_identity_contract(models) -> None:
    for model in models.values():
        report = mp.assert_divisor_identity(model)
        assert report["divisor"] == 1.0
        assert report["identity_map_bitexact"] is True
    assert float(models[16].geometry["target_scale"]) == 1.0
    # numeric: MSE(raw, y) == MSE(raw/1, y) exactly
    y = torch.tensor([[-1.5, 0.25, 2.0]])
    yhat = torch.tensor([[-1.0, 0.0, 1.0]])
    assert float(((y / 1.0 - yhat) ** 2).mean()) == float(((y - yhat) ** 2).mean())


# ---------------------------------------------------------------------------
# (5) Script: stage enumeration + series wiring
# ---------------------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location("m1_projadd_series_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_stage_enumeration_and_series_wiring() -> None:
    script = _load_script()
    assert script.STAGES == ("preflight", "probe", "train", "score", "all")
    assert script.TRAIN_ENV_FLAG == "BTRANSFORM_M1_PROJADD_TRAIN"
    assert script.GPU1_UUID == "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
    assert script.SEED == 42 and script.EPOCHS == 24 and script.BATCH_SIZE == 32
    # LR axis: default 1e-4 (H1 lesson recorded), NOT the TRN-1 3e-4
    assert script.mp.DEFAULT_PEAK_LR == 1e-4
    assert "3e-4" in script.mp.PEAK_LR_NOTE
    # the script's PICKS travel the LR note + acceptance law into receipts
    joined = json.dumps(script.PICKS)
    assert "LR-NOTE" in joined and "peak" in joined.replace("peak_lr", "peak")
    assert "0.839" in joined and "M10" in joined


def test_series_cells_manifest() -> None:
    cells = mp.series_cells()
    assert [c["cell_id"] for c in cells] == ["M1-PROJADD-P16", "M1-PROJADD-P32"]
    assert [c["proj_dim"] for c in cells] == [16, 32]
    for cell, expected in zip(cells, (3533616, 3539312)):
        assert cell["expected_param_count"] == expected
        assert cell["peak_lr"] == 1e-4 and cell["seed"] == 42 and cell["epochs"] == 24
        assert cell["identity_mode"] == "proj_add" and cell["matrix_letter"] == "f"


def test_param_arithmetic_matches_instantiation(models) -> None:
    concat = BTransformerUnifiedDecoder(mp.m1_projadd_geometry(16), seed=42)
    n_concat = sum(p.numel() for p in concat.parameters())
    assert n_concat == mp.M1_CONCAT_PARAM_COUNT
    for proj_dim, model in models.items():
        n = sum(p.numel() for p in model.parameters())
        expected = n_concat - SET_DIM * (mp.M1_TOKEN_IN_CONCAT - (proj_dim + 4)) + proj_dim * 100
        assert n == expected == mp.m1_projadd_param_count(proj_dim)


# ---------------------------------------------------------------------------
# (6) R^2 hand example + acceptance constants
# ---------------------------------------------------------------------------


def test_variance_weighted_r2_hand_example() -> None:
    # target [1,2,3], pred [1,2,2]: SSres = 1, SStot = 2 -> R^2 = 0.5
    target = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 2.0])
    assert variance_weighted_r2(target, pred) == pytest.approx(0.5)
    # perfect prediction -> exactly 1; mean prediction -> exactly 0
    assert variance_weighted_r2(target, target) == pytest.approx(1.0)
    assert variance_weighted_r2(target, np.full(3, 2.0)) == pytest.approx(0.0)
    # session-mean surface: per-session R^2 0.5 (a) and 1.0 (b) -> mean 0.75;
    # pooled = 1 - SSres/SStot = 1 - 1/2.8 (mean-pooled target 1.8)
    t = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
    p = np.array([1.0, 2.0, 2.0, 1.0, 2.0])
    s = np.array(["a", "a", "a", "b", "b"])
    report = session_mean_report(t, p, s)
    assert report["pooled_r2"] == pytest.approx(1.0 - 1.0 / 2.8)
    assert report["session_mean_r2"] == pytest.approx(0.75)
    assert report["per_session_r2"]["a"] == pytest.approx(0.5)
    assert report["per_session_r2"]["b"] == pytest.approx(1.0)


def test_acceptance_constants_and_readonly_references() -> None:
    assert mp.ORIGINAL_MINIVAL_POOLED == pytest.approx(0.809289)
    assert mp.OFFICIAL_ORIGINAL == pytest.approx(0.649)
    assert mp.ACCEPT_MARGIN == pytest.approx(0.03)
    assert mp.ACCEPT_THRESHOLD_MINIVAL == pytest.approx(0.809289 + 0.03)
    assert mp.SOURCE_MINIVAL_WINDOWS == 31252
    assert mp.LOSO_OUTER_WINDOWS == 26496
    assert mp.M1_SESSIONS == (
        "ses-20120924",
        "ses-20120926",
        "ses-20120927",
        "ses-20120928",
    )
    assert mp.M1_OUTER_SESSION == "ses-20120924"
    # DEFERRED receipt + family_flat references exist on disk (read-only)
    assert mp.DEFERRED_RECEIPT_PATH.is_file()
    assert mp.FAMILY_FLAT_PATH.is_file()
    deferred = json.loads(mp.DEFERRED_RECEIPT_PATH.read_text(encoding="utf-8"))
    locked = " ".join(deferred["work_preserved_for_reuse"]["mechanism_decisions_locked"])
    assert "_family_bank" in locked and "compute_identity(side_features=None)" in locked
    assert deferred["work_preserved_for_reuse"]["assets_verified_on_disk"]["sfix_e11"][
        "sha256_verified_ok"
    ]
    assert "leakage" in mp.ORIGINAL_LOSO_LEAKAGE_NOTE


def test_cpu_only_suite() -> None:
    assert not torch.cuda.is_initialized()
