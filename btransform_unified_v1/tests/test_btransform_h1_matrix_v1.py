"""H1 L x identity matrix skeleton tests (MATRIX_H1_L_IDENTITY_V1_20260906).

CPU only; synthetic banks for shape/semantics, one SHA-gated real-data test
for the build_h1_bank materializer path. Five-arm script and every historical
result root are strictly read-only. NOT SPINT.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from btransform_unified_v1 import adapters, h1_config, matrix_cells, plan
from btransform_unified_v1.bank import array_sha256, make_synthetic_bank
from btransform_unified_v1.identity_variant import (
    BTransformerUnifiedDecoderIdentity,
    apply_add_tail,
)
from btransform_unified_v1.matrix_cells import (
    MATRIX_CELLS,
    MatrixCell,
    baseline_cell,
    cell_by_id,
    cell_geometry,
    control_cells,
    primary_cells,
    validate_matrix_cells,
)
from btransform_unified_v1.model import BTransformerUnifiedDecoder
from btransform_unified_v1.scale_bridge import (
    H1_BRIDGE,
    H1_SCALE_RATIO,
    H1_SCALE_RATIO_REL_TOL,
    assert_scale_bridge,
)

TOKEN_LOCAL = 16
TOKEN_CARRIER = 4


def _matrix_model(length: int, mode: str, seed: int = 42):
    geometry = h1_config.h1_matrix_geometry(length, mode)
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=seed, override_prefix=0, override_window=length,
        identity_mode=mode,
    ).eval()
    return model, geometry


def _matrix_bank(mode: str, seed: int = 5, n_windows: int = 2):
    return make_synthetic_bank(
        "h1", n_windows=n_windows, seed=seed,
        e0_dim=h1_config.matrix_e0_dim(mode), prefix=0,
    )


def _tail_slice(store: np.ndarray, length: int) -> torch.Tensor:
    """L-bin tail of the full-window synthetic store, as a tensor."""
    return torch.from_numpy(np.ascontiguousarray(store[:, -length:, :]))


# ---------------------------------------------------------------------------
# (1) five identity modes x L in {250, 350}: instantiation + forward shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", h1_config.MATRIX_L)
@pytest.mark.parametrize("mode", h1_config.IDENTITY_MODES)
def test_identity_modes_forward_shapes(length: int, mode: str) -> None:
    model, geometry = _matrix_model(length, mode)
    bank = _matrix_bank(mode)
    x = _tail_slice(bank.X_store, length)
    expected_token_e0 = {
        "concat": h1_config.FULL_E0_DIM,
        "joined": h1_config.JOINED_DIM,
        "add_tail": 0,
        "zero": h1_config.FULL_E0_DIM,
        "permute": h1_config.FULL_E0_DIM,
        "proj_add": 0,
    }[mode]
    assert model.identity_mode == mode
    assert model.window == length and model.l_in == length  # prefix 0
    assert model.window_override == length
    assert model.token_e0_width == expected_token_e0
    assert model.token_in == TOKEN_LOCAL + expected_token_e0 + TOKEN_CARRIER
    assert geometry["e0_dim"] == h1_config.matrix_e0_dim(mode)
    with torch.no_grad():
        y = model(x, bank)
        scores = model.forward_scores(x, bank)
    assert y.shape == (x.size(0), h1_config.OUT_DIM)
    assert scores.shape == (x.size(0), length, h1_config.OUT_DIM)
    assert torch.isfinite(y).all() and torch.isfinite(scores).all()
    # init_meta records the mode-dependent geometry (params vary with mode)
    meta = model.init_meta
    assert meta["identity_mode"] == mode
    assert meta["token_e0_width"] == expected_token_e0
    assert meta["token_in"] == model.token_in
    assert isinstance(meta["param_count"], int) and meta["param_count"] > 0
    # causality self-certification keeps working in every mode (add_tail
    # modifies the input bins before the causal conv, so scrambling the tail
    # still cannot leak into earlier readouts)
    report = model.causal_check(x, bank)
    assert report["passed"] is True
    assert report["length"] == length


def test_identity_variant_concat_backcompat_with_parent() -> None:
    # default mode "concat" keeps the parent build bit-identical (same
    # state-dict keys, shapes, and init values) — model.py stays byte-stable.
    parent = BTransformerUnifiedDecoder("m2", seed=42)
    child = BTransformerUnifiedDecoderIdentity("m2", seed=42)
    assert child.identity_mode == h1_config.IDENTITY_DEFAULT == "concat"
    own, ref = child.state_dict(), parent.state_dict()
    assert sorted(own) == sorted(ref)
    for key in ref:
        assert own[key].shape == ref[key].shape, key
        assert torch.equal(own[key], ref[key]), f"init value drift at {key}"


# ---------------------------------------------------------------------------
# (2) add_tail alignment: exactly the LAST L bins of E0 land on X's tail
# ---------------------------------------------------------------------------


def test_add_tail_alignment_known_e0() -> None:
    length = 250
    n_units = 5
    e0 = np.arange(n_units * h1_config.FULL_E0_DIM, dtype=np.float32).reshape(
        n_units, h1_config.FULL_E0_DIM
    )
    x = torch.randn(2, length, n_units)
    out = apply_add_tail(x, torch.from_numpy(e0))
    expected_tail = torch.from_numpy(e0[:, -length:].copy()).t()  # [L, N]
    assert torch.equal(out, x + expected_tail.unsqueeze(0))
    # min(L, 700) = L: a full 700 window consumes the whole E0 time axis
    x_full = torch.randn(1, h1_config.FULL_WINDOW, n_units)
    out_full = apply_add_tail(x_full, torch.from_numpy(e0))
    assert torch.equal(out_full, x_full + torch.from_numpy(e0).t().unsqueeze(0))
    # the head of E0 (outside the last L columns) is never touched
    e0_head_junk = e0.copy()
    e0_head_junk[:, : h1_config.FULL_E0_DIM - length] = -1e6
    out_junk = apply_add_tail(x, torch.from_numpy(e0_head_junk))
    assert torch.equal(out, out_junk)


def test_add_tail_model_level_equivalence() -> None:
    length = 250
    model, _ = _matrix_model(length, "add_tail")
    bank = _matrix_bank("add_tail")
    e0 = torch.from_numpy(bank.E0.copy())
    zeros = torch.zeros_like(e0)
    x = _tail_slice(bank.X_store, length)
    from btransform_unified_v1.bank import TaskBank

    def bank_with(e0_arr: np.ndarray) -> TaskBank:
        meta = dict(bank.calibration_meta)
        meta["array_sha256"] = array_sha256(e0_arr)
        return TaskBank(
            session_id=bank.session_id, E0=e0_arr, carrier=bank.carrier,
            unit_mask=bank.unit_mask, X_store=bank.X_store,
            target_store=bank.target_store, window_ids=bank.window_ids,
            calibration_meta=meta,
        )

    with torch.no_grad():
        y1 = model(x, bank_with(e0.numpy()))  # tail add happens inside forward
        manual = x + e0[:, -length:].t().unsqueeze(0)
        y2 = model(manual, bank_with(zeros.numpy()))
    assert torch.equal(y1, y2)  # identical inputs enter the causal conv


# ---------------------------------------------------------------------------
# (3) zero mode == concat mode fed an all-zero E0 (bitwise)
# ---------------------------------------------------------------------------


def test_zero_mode_equals_concat_with_zero_e0() -> None:
    length = 250
    bank_random = _matrix_bank("concat", seed=11)
    zero_e0 = np.zeros_like(bank_random.E0)
    meta = dict(bank_random.calibration_meta, array_sha256=array_sha256(zero_e0))
    bank_zero = type(bank_random)(
        session_id="zero-e0", E0=zero_e0, carrier=bank_random.carrier,
        unit_mask=bank_random.unit_mask, X_store=bank_random.X_store,
        target_store=bank_random.target_store, window_ids=bank_random.window_ids,
        calibration_meta=meta,
    )
    concat, _ = _matrix_model(length, "concat")
    zero, _ = _matrix_model(length, "zero")
    x = _tail_slice(bank_random.X_store, length)
    with torch.no_grad():
        y_concat = concat(x, bank_zero)      # concat fed all-zero E0
        y_zero = zero(x, bank_random)        # zero mode fed the random E0
    assert torch.equal(y_concat, y_zero)


# ---------------------------------------------------------------------------
# (4) permute != concat (numerically different on a random E0)
# ---------------------------------------------------------------------------


def test_permute_mode_differs_from_concat() -> None:
    length = 350
    bank = _matrix_bank("permute", seed=13)
    concat, _ = _matrix_model(length, "concat")
    permute, _ = _matrix_model(length, "permute")
    assert concat.token_in == permute.token_in  # same parameterization width
    assert concat.init_meta["param_count"] == permute.init_meta["param_count"]
    x = _tail_slice(bank.X_store, length)
    with torch.no_grad():
        y_concat = concat(x, bank)
        y_permute = permute(x, bank)
    assert not torch.equal(y_concat, y_permute)
    assert float((y_concat - y_permute).abs().max()) > 0.0


# ---------------------------------------------------------------------------
# (5) L override mechanism (override_window; prefix sentinel still enforced)
# ---------------------------------------------------------------------------


def test_window_override_mechanism_and_refusals() -> None:
    geometry = h1_config.h1_matrix_geometry(250, "concat")
    # prefix sentinel still refuses without an explicit override (item P)
    with pytest.raises(plan.BTransformerUnifiedError, match="H1_PREFIX_PENDING"):
        BTransformerUnifiedDecoderIdentity(geometry, seed=42, override_window=250)
    for length in h1_config.MATRIX_L:
        model = BTransformerUnifiedDecoderIdentity(
            geometry, seed=42, override_prefix=0, override_window=length
        )
        assert model.window == length and model.l_in == length
        assert model.window_override == length
        assert model.geometry["window"] == length
    # refusals: extension, no-op, and non-H1 settled windows stay frozen
    with pytest.raises(plan.BTransformerUnifiedError, match="SHORTEN"):
        BTransformerUnifiedDecoderIdentity(
            geometry, seed=42, override_prefix=0, override_window=800
        )
    with pytest.raises(plan.BTransformerUnifiedError, match="SHORTEN"):
        BTransformerUnifiedDecoderIdentity(
            geometry, seed=42, override_prefix=0, override_window=700
        )
    with pytest.raises(plan.BTransformerUnifiedError, match="frozen"):
        BTransformerUnifiedDecoderIdentity("m2", seed=42, override_window=40)
    with pytest.raises(plan.BTransformerUnifiedError):
        BTransformerUnifiedDecoderIdentity(
            geometry, seed=42, override_prefix=0, override_window=True  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# (6) scale_bridge H1 ratio acceptance: numeric example
# ---------------------------------------------------------------------------


def test_scale_bridge_h1_ratio_numeric_example() -> None:
    rng = np.random.default_rng(7)
    native = rng.standard_normal((128, 7)).astype(np.float32) * 0.8
    raw = (20.0 * native) + rng.standard_normal((128, 7)) * 0.5  # trained on 20y
    assert H1_BRIDGE == pytest.approx(20.0)
    assert H1_SCALE_RATIO == pytest.approx(400.0)
    assert H1_SCALE_RATIO_REL_TOL == 1e-9
    bridged = assert_scale_bridge("h1", raw, native)
    assert bridged.shape == raw.shape
    # float64 bridging vs float32 division agree to float32 roundoff
    np.testing.assert_allclose(bridged, raw / 20.0, rtol=1e-6, atol=0)
    # the ratio identity holds at float64 well inside the 1e-9 relative gate
    # (recomputing in float32 accumulates ~3e-9 roundoff — the module checks
    # in float64, exactly like this recheck)
    raw64 = raw.astype(np.float64)
    native64 = native.astype(np.float64)
    mse_raw = float(np.mean((raw64 - 20.0 * native64) ** 2))
    mse_bridge = float(np.mean((raw64 / 20.0 - native64) ** 2))
    assert abs(mse_raw - 400.0 * mse_bridge) <= 1e-9 * max(1.0, mse_raw)
    # near-constant output trips the P0-3 std guard
    with pytest.raises(plan.BTransformerUnifiedError, match="near-constant"):
        assert_scale_bridge("h1", np.zeros((128, 7)), native)


# ---------------------------------------------------------------------------
# (7) CAL1 budget rotation: deterministic (session, epoch) -> budget
# ---------------------------------------------------------------------------


def test_cal1_budget_rotation_deterministic_and_covering() -> None:
    assert h1_config.CAL1_BUDGETS == (7, 5, 4, 3)
    assert h1_config.DEPLOY_BUDGET == 3
    for session in ("ses-19250101T111740", "ses-19250120T115537"):
        for epoch in range(24):
            budget = h1_config.cal1_budget(session, epoch)
            assert budget in h1_config.CAL1_BUDGETS
            assert budget == h1_config.cal1_budget(session, epoch)  # deterministic
        # any 4 consecutive epochs visit every budget exactly once
        for start in (0, 5, 20):
            window = [h1_config.cal1_budget(session, e) for e in range(start, start + 4)]
            assert sorted(window) == sorted(h1_config.CAL1_BUDGETS)
    # different sessions may sit at different phases (rotation is per-session)
    phases = {
        h1_config.cal1_budget(s, 0) for s in h1_config.H1_ALL_SESSIONS
    }
    assert phases <= set(h1_config.CAL1_BUDGETS)


# ---------------------------------------------------------------------------
# (8) LODO date split: holdout >= 2 sessions, train/exam disjoint
# ---------------------------------------------------------------------------


def test_lodo_split_invariants() -> None:
    split = h1_config.lodo_split()
    assert h1_config.LODO_HOLDOUT_DATE == "1925-01-20"
    assert split["holdout_date"] == "1925-01-20"
    assert split["holdout_sessions"] == [
        "ses-19250120T115044", "ses-19250120T115537",
    ]
    assert split["n_holdout_sessions"] >= 2
    train, holdout = set(split["train_sessions"]), set(split["holdout_sessions"])
    assert train.isdisjoint(holdout)
    assert train | holdout == set(h1_config.H1_ALL_SESSIONS)
    assert len(h1_config.H1_ALL_SESSIONS) == 13
    assert len(h1_config.H1_SESSION_DATES) == 6
    # every session appears under exactly one date
    seen = [s for sessions in h1_config.H1_SESSIONS_BY_DATE.values() for s in sessions]
    assert sorted(seen) == sorted(h1_config.H1_ALL_SESSIONS)
    assert h1_config.LODO_HOLDOUT_DATE == h1_config.H1_SESSION_DATES[-1]  # last date
    # cross-check the frozen routing (tfpd_exploration quick_product config)
    tfpd_config = pytest.importorskip(
        "tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config"
    )
    assert tuple(tfpd_config.HELDIN_SESSIONS) == h1_config.H1_ALL_SESSIONS


# ---------------------------------------------------------------------------
# matrix cell registry
# ---------------------------------------------------------------------------


def test_matrix_cells_registry() -> None:
    report = validate_matrix_cells()
    assert report["n_cells"] == 9
    expected = {
        "M-A250": (250, "joined", False),
        "M-A350": (350, "joined", False),
        "M-B250": (250, "concat", False),
        "M-B350": (350, "concat", False),
        "M-C250": (250, "add_tail", True),
        "M-F250": (250, "proj_add", False),
        "M-F700": (700, "proj_add", False),
        "M-D250": (250, "zero", False),
        "M-E250": (250, "permute", False),
    }
    assert len(MATRIX_CELLS) == 9
    for cell in MATRIX_CELLS:
        assert isinstance(cell, MatrixCell)
        length, mode, conditional = expected[cell.cell_id]
        assert (cell.L, cell.identity_mode, cell.conditional) == (length, mode, conditional)
        assert cell.est_cost_hours > 0 and cell.note
    assert [c.cell_id for c in primary_cells()] == ["M-A250", "M-A350", "M-B250", "M-B350"]
    assert {c.cell_id for c in control_cells()} == {"M-D250", "M-E250"}
    assert cell_by_id("M-C250").conditional is True
    assert cell_by_id("M-F250").identity_mode == "proj_add"  # user 2026-09-06 proposal
    assert "截断披露" in cell_by_id("M-C250").note  # 250/700 truncation disclosure
    with pytest.raises(plan.BTransformerUnifiedError):
        cell_by_id("M-A150")  # withdrawn by the 2026-09-06 user ruling
    # geometry per cell keeps the sentinels (explicit opt-in at instantiation)
    geo = cell_geometry(cell_by_id("M-A250"))
    assert geo["e0_dim"] == h1_config.JOINED_DIM
    assert geo["prefix"] == plan.H1_PREFIX_PENDING and geo["window"] == 700
    assert cell_geometry(cell_by_id("M-B350"))["e0_dim"] == 700
    assert cell_geometry(cell_by_id("M-F250"))["e0_dim"] == 700


def test_mf700_baseline_cell_registered_and_full_window_geometry() -> None:
    """Blocking item C (REVIEW 2026-09-06): M-F700 is a formal registry cell.

    L=700 is the ORIGINAL settled H1 window, not a matrix shortening — the
    registry exemption is the matrix doc §2 revision (formalized 2026-09-06);
    the model builds with the settled window itself (override_window stays
    None: it is a shorten-only control and must keep refusing 700).
    """
    cell = baseline_cell()
    assert cell.cell_id == "M-F700" == matrix_cells.BASELINE_CELL_ID
    assert cell.L == h1_config.FULL_WINDOW == 700
    assert cell.identity_mode == "proj_add"
    assert cell.conditional is False
    geometry = cell_geometry(cell)
    assert geometry["window"] == 700 and geometry["e0_dim"] == h1_config.FULL_E0_DIM
    assert geometry["prefix"] == plan.H1_PREFIX_PENDING  # sentinel kept
    # settled-window build: NO override_window (only ever a shortening control)
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=42, override_prefix=0, identity_mode="proj_add"
    ).eval()
    assert model.window == 700 and model.l_in == 700
    assert model.window_override is None
    assert model.identity_mode == "proj_add"
    assert model.init_meta["token_in"] == 20
    assert model.init_meta["proj_out_dim"] == h1_config.PROJ_ADD_OUT_DIM == 16
    assert model.init_meta["proj_in_dim"] == h1_config.FULL_E0_DIM
    # proj_add is L-independent in the frontend: the controlled pair M-F700 vs
    # M-F250 shares parameterization width AND total parameter count (the PE
    # is a non-persistent sinusoidal buffer, not a parameter) — only l_in and
    # compute differ, so "only variable L" holds at the parameter level too
    f250 = cell_by_id("M-F250")
    m250 = BTransformerUnifiedDecoderIdentity(
        cell_geometry(f250), seed=42, override_prefix=0, override_window=f250.L,
        identity_mode="proj_add",
    )
    assert m250.init_meta["param_count"] == model.init_meta["param_count"]
    assert m250.l_in == 250 and model.l_in == 700
    # the shorten-only override control is unchanged by the exemption
    with pytest.raises(plan.BTransformerUnifiedError, match="SHORTEN"):
        BTransformerUnifiedDecoderIdentity(
            geometry, seed=42, override_prefix=0, override_window=700
        )
    assert validate_matrix_cells()["n_baseline_full_window"] == 1


# ---------------------------------------------------------------------------
# proj_add (M-F250): fold parity + zero-projection degeneration
# ---------------------------------------------------------------------------


def test_proj_add_fold_parity_and_projection_meta() -> None:
    model, _ = _matrix_model(250, "proj_add")
    assert model.frontend.e0_proj is not None
    assert model.frontend.e0_proj.bias is None  # bias choice frozen, recorded
    assert model.init_meta["proj_bias"] is False
    assert model.init_meta["proj_in_dim"] == h1_config.FULL_E0_DIM
    assert model.init_meta["proj_out_dim"] == h1_config.PROJ_ADD_OUT_DIM == 16
    bank = _matrix_bank("proj_add")
    x = _tail_slice(bank.X_store, 250)
    static = model.bank_static_term(bank)
    assert static.shape == (h1_config.UNITS, 256)
    with torch.no_grad():
        y_ref = model(x, bank)
        y_folded = model.forward_static_folded(x, bank, static)
    # SPD-A1 parity caliber (workorder P3 gate): FP32 max|delta| <= 1e-6.
    # P(E0) is session-static, so the rank-16 term folds exactly like the
    # concat-mode static term (split-GEMM float neighbor of the fused path).
    delta = (y_ref - y_folded).abs().max().item()
    assert delta <= 1e-6, f"proj_add static-fold parity exceeded 1e-6: {delta:.3e}"


def test_proj_add_zero_projection_degenerates_to_no_e0() -> None:
    length = 250
    proj_model, _ = _matrix_model(length, "proj_add")
    # (a) P zeroed -> outputs are bitwise independent of the bank E0 (the
    # alt bank keeps X/carrier/mask IDENTICAL and swaps only E0)
    with torch.no_grad():
        proj_model.frontend.e0_proj.weight.zero_()
    bank_a = _matrix_bank("proj_add", seed=21)
    alt_e0 = np.random.default_rng(99).standard_normal(bank_a.E0.shape).astype(np.float32)
    from btransform_unified_v1.bank import TaskBank

    bank_alt = TaskBank(
        session_id="alt-e0",
        E0=alt_e0,
        carrier=bank_a.carrier,
        unit_mask=bank_a.unit_mask,
        X_store=bank_a.X_store,
        target_store=bank_a.target_store,
        window_ids=bank_a.window_ids,
        calibration_meta=dict(bank_a.calibration_meta, array_sha256=array_sha256(alt_e0)),
    )
    assert not np.array_equal(bank_a.E0, alt_e0)
    x = _tail_slice(bank_a.X_store, length)
    with torch.no_grad():
        y_a = proj_model(x, bank_a)
        y_b = proj_model(x, bank_alt)
    assert torch.equal(y_a, y_b)

    # (b) under IDENTICAL shared weights, proj_add with P=0 equals the no-E0
    # local+carrier pathway of add_tail fed a zero E0 (x + 0 leaves X intact)
    tail_model, _ = _matrix_model(length, "add_tail")
    report = proj_model.load_state_dict(tail_model.state_dict(), strict=False)
    assert report.unexpected_keys == []
    assert report.missing_keys == ["frontend.e0_proj.weight"]
    with torch.no_grad():
        proj_model.frontend.e0_proj.weight.zero_()
    zero_e0 = np.zeros_like(bank_a.E0)
    bank_tail = TaskBank(
        session_id="zero-e0-tail",
        E0=zero_e0,
        carrier=bank_a.carrier,
        unit_mask=bank_a.unit_mask,
        X_store=bank_a.X_store,
        target_store=bank_a.target_store,
        window_ids=bank_a.window_ids,
        calibration_meta=dict(bank_a.calibration_meta, array_sha256=array_sha256(zero_e0)),
    )
    with torch.no_grad():
        y_proj = proj_model(x, bank_a)      # tokens = [local + P(E0)=0 | carrier]
        y_tail = tail_model(x, bank_tail)   # tokens = [local | carrier], X + 0
    assert torch.equal(y_proj, y_tail)


# ---------------------------------------------------------------------------
# real-data build_h1_bank (SHA-gated; skips when frozen sources are absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_h1_sources_available() -> None:
    paths = [
        h1_config.SOURCE_CACHE_PATH,
        h1_config.C2_M3_PAYLOAD_PATH,
        h1_config.C2_CKPT_PATH,
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"frozen H1 sources unavailable: {missing}")


def test_build_h1_bank_phase_gate_and_budget_stub() -> None:
    # the no-args call keeps the Phase 2b gate (surface/session required)
    with pytest.raises(NotImplementedError, match="Phase 2b"):
        adapters.build_h1_bank(budget=3)


def test_build_h1_bank_real_data(real_h1_sources_available) -> None:
    session = "ses-19250120T115537"  # LODO holdout date session
    bank = adapters.build_h1_bank(
        "minival", session, budget=3, window=250, identity_mode="concat",
        limit_windows=8,
    )
    assert bank.E0.shape == (176, 700) and bank.carrier.shape == (176, 4)
    assert bank.X_store.shape == (8, 250, 176)  # L=250, end-anchored
    assert bank.target_store.shape == (8, 7)
    assert bank.window_ids.dtype == np.int64 and len(bank.window_ids) == 8
    meta = bank.calibration_meta
    assert meta["budget"] == 3 and meta["trial_count"] == 3
    assert meta["identity_mode"] == "concat"
    assert meta["window"] == 250
    assert meta["lodo_holdout"] is True and meta["lodo_holdout_date"] == "1925-01-20"
    assert meta["array_sha256"] == array_sha256(bank.E0)
    # BOTH calibration objects travel with the bank (fused 700 + true joined 36)
    joined = meta["joined36"]
    assert isinstance(joined, np.ndarray) and joined.shape == (176, 36)
    assert meta["joined36_sha256"] == array_sha256(joined.astype(np.float32))
    assert meta["ckpt_sha256"].startswith("ce46267e")
    # native targets: velocity columns stay unscaled (train loop applies x20)
    cache = adapters._h1_source_cache()  # same memoized, SHA-verified object
    row = cache["minival"][session]
    ends = np.flatnonzero(np.asarray(row["eval_mask"], dtype=np.bool_))[:8]
    np.testing.assert_array_equal(bank.window_ids, ends)
    np.testing.assert_array_equal(
        bank.target_store, np.asarray(row["velocity"], dtype=np.float32)[ends]
    )
    # provenance loop: E0 reproduces the cache bank the five-arm consumed
    cached_e0 = row["bank"]["E0"]
    cached_e0 = cached_e0.cpu().numpy() if hasattr(cached_e0, "cpu") else np.asarray(cached_e0)
    np.testing.assert_array_equal(bank.E0, cached_e0)

    # joined mode carries the TRUE 36-d object in the E0 slot
    bank_j = adapters.build_h1_bank(
        "minival", session, budget=3, window=250, identity_mode="joined",
        limit_windows=4,
    )
    assert bank_j.E0.shape == (176, 36)
    np.testing.assert_array_equal(bank_j.E0, joined)

    # train surface: (session, end) coordinates frozen at query_starts + 699
    bank_t = adapters.build_h1_bank(
        "train", session, budget=3, window=250, identity_mode="concat",
        limit_windows=4,
    )
    train_row = cache["train"][session]
    starts = np.asarray(train_row["query_starts"], dtype=np.int64)
    expected_ends = starts + 699
    mask = np.asarray(train_row["eval_mask"], dtype=np.bool_)
    expected_ends = expected_ends[(expected_ends >= 0) & (expected_ends < mask.shape[0])]
    expected_ends = expected_ends[mask[expected_ends]][:4]
    np.testing.assert_array_equal(bank_t.window_ids, expected_ends)

    # CAL-1 budgets above the M3 payload are an explicit stub (no fabrication)
    with pytest.raises(NotImplementedError, match="not materialized"):
        adapters.build_h1_bank("minival", session, budget=7, window=250)

    # bad surface / window / budget refusals
    with pytest.raises(plan.BTransformerUnifiedError):
        adapters.build_h1_bank("hidden", session, budget=3)
    with pytest.raises(plan.BTransformerUnifiedError):
        adapters.build_h1_bank("minival", session, budget=3, window=150)  # withdrawn L
    with pytest.raises(plan.BTransformerUnifiedError):
        adapters.build_h1_bank("minival", session, budget=6)  # CAL-2 forbidden


# ---------------------------------------------------------------------------
# M2 + proj_add (ADDENDUM-M2-PROJADD 2026-09-06): m2 geometry on the identity
# class — forward shapes, geometry bookkeeping, and the P-zero degeneration
# ---------------------------------------------------------------------------


def _m2_projadd_model(seed: int = 42):
    geometry = {  # P1A geometry mapping (prefix=0, S1 parity), task m2
        "task": "m2",
        "window": 50,
        "prefix": 0,
        "units": 96,
        "e0_dim": 50,
        "carrier_dim": 4,
        "out_dim": 2,
    }
    model = BTransformerUnifiedDecoderIdentity(geometry, seed=seed, identity_mode="proj_add").eval()
    return model, geometry


def _m2_projadd_bank(n_windows: int = 3, seed: int = 7) -> "TaskBank":
    """Synthetic m2-prefix0 bank (P1a mapping: [S, L_in=50, 96], E0 [96,50])."""
    from btransform_unified_v1.bank import TaskBank

    rng = np.random.default_rng(seed)
    units, e0_dim, l_in = 96, 50, 50
    e0 = rng.standard_normal((units, e0_dim)).astype(np.float32)
    carrier = rng.standard_normal((units, 4)).astype(np.float32)
    keep = rng.random(units) >= 0.1
    keep[0] = True
    x_store = rng.standard_normal((n_windows, l_in, units)).astype(np.float32)
    target = rng.standard_normal((n_windows, 2)).astype(np.float32)
    return TaskBank(
        session_id=f"synthetic-m2-p0-{seed}",
        E0=e0,
        carrier=carrier,
        unit_mask=keep,
        X_store=x_store,
        target_store=target,
        window_ids=np.arange(n_windows, dtype=np.int64),
        calibration_meta={
            "shape": (units, e0_dim),
            "trial_count": 33,
            "estimator": "synthetic_standard_normal",
            "array_sha256": array_sha256(e0),
            "budget": 33,
            "synthetic": True,
        },
    )


def test_m2_projadd_forward_shapes_and_geometry() -> None:
    model, geometry = _m2_projadd_model()
    bank = _m2_projadd_bank(3, seed=7)
    assert model.identity_mode == "proj_add"
    assert model.task == "m2"
    assert model.window == 50 and model.prefix == 0 and model.l_in == 50
    assert model.window_override is None  # settled m2 window, no L override
    assert model.base_e0_dim == 50  # bank-side width keeps the m2 E0 [96,50]
    assert model.token_e0_width == 0
    assert model.token_in == TOKEN_LOCAL + 0 + TOKEN_CARRIER == 20
    assert model.frontend.e0_proj is not None
    assert model.frontend.e0_proj.weight.shape == (h1_config.PROJ_ADD_OUT_DIM, 50)
    assert model.frontend.e0_proj.bias is None
    meta = model.init_meta
    assert meta["identity_mode"] == "proj_add"
    assert meta["matrix_letter"] == "f"
    assert meta["token_in"] == 20 and meta["token_e0_width"] == 0
    assert meta["proj_in_dim"] == 50 and meta["proj_out_dim"] == 16
    assert meta["proj_bias"] is False
    # parameter arithmetic vs the concat m2 build: token_mlp.0 narrows
    # 70 -> 20 inputs and the rank-16 projection P adds 16*50 weights
    concat = BTransformerUnifiedDecoder(geometry, seed=42)  # same mapping, concat
    n_concat = sum(p.numel() for p in concat.parameters())
    n_proj = sum(p.numel() for p in model.parameters())
    assert n_proj == n_concat - 256 * (70 - 20) + 16 * 50
    x = torch.from_numpy(np.ascontiguousarray(bank.X_store))
    assert x.shape == (3, 50, 96)
    with torch.no_grad():
        y = model(x, bank)
        scores = model.forward_scores(x, bank)
    assert y.shape == (3, 2)
    assert scores.shape == (3, 50, 2)
    assert torch.isfinite(y).all() and torch.isfinite(scores).all()
    report = model.causal_check(x, bank)
    assert report["passed"] is True and report["length"] == 50
    # bank E0 keeps the [96, 50] m2 contract in proj_add mode (the projection
    # is applied inside the frontend, not by shrinking the bank)
    e0, carrier, mask = model._bank_arrays(bank)
    assert e0.shape == (96, 50) and carrier.shape == (96, 4) and mask.shape == (96,)


def test_m2_projadd_zero_projection_degenerates_to_no_e0() -> None:
    from btransform_unified_v1.bank import TaskBank

    model, _ = _m2_projadd_model()
    bank = _m2_projadd_bank(2, seed=21)
    with torch.no_grad():
        model.frontend.e0_proj.weight.zero_()
    alt_e0 = np.random.default_rng(99).standard_normal(bank.E0.shape).astype(np.float32)
    bank_alt = TaskBank(
        session_id="alt-e0",
        E0=alt_e0,
        carrier=bank.carrier,
        unit_mask=bank.unit_mask,
        X_store=bank.X_store,
        target_store=bank.target_store,
        window_ids=bank.window_ids,
        calibration_meta=dict(bank.calibration_meta, array_sha256=array_sha256(alt_e0)),
    )
    assert not np.array_equal(bank.E0, alt_e0)
    x = torch.from_numpy(np.ascontiguousarray(bank.X_store))
    with torch.no_grad():
        y_a = model(x, bank)
        y_b = model(x, bank_alt)
    # P = 0 => tokens = [local16 + 0 | carrier4]: outputs are BITWISE
    # independent of the bank E0 (zero-mode semantics at the 20-wide build)
    assert torch.equal(y_a, y_b)


def test_m2_projadd_p32_width_axis_and_fusion() -> None:
    """proj_dim=32 (M2 P32 widening retry, user directive 2026-09-06).

    The bracket generalizes by broadcast: each 16-d group of P(E0) is added
    onto the local conv channels -> [local+P_1 ; local+P_2], bracket 32,
    token_in 36. Default proj_dim=None must stay the P16 build (token_in 20).
    """
    geometry = {
        "task": "m2",
        "window": 50,
        "prefix": 0,
        "units": 96,
        "e0_dim": 50,
        "carrier_dim": 4,
        "out_dim": 2,
    }
    model = BTransformerUnifiedDecoderIdentity(
        geometry, seed=42, identity_mode="proj_add", proj_dim=32
    ).eval()
    bank = _m2_projadd_bank(3, seed=7)
    assert model.token_in == 36  # bracket 32 + carrier 4
    assert model.proj_out_dim == 32 and model.proj_groups == 2
    assert model.frontend.e0_proj.weight.shape == (32, 50)
    assert model.frontend.e0_proj.bias is None
    meta = model.init_meta
    assert meta["proj_out_dim"] == 32 and meta["proj_groups"] == 2
    assert meta["proj_dim_override"] == 32 and meta["fused_local_width"] == 32
    # parameter arithmetic: token_mlp.0 narrows 70 -> 36 inputs, P adds 32*50
    concat = BTransformerUnifiedDecoder(geometry, seed=42)
    n_concat = sum(p.numel() for p in concat.parameters())
    n_p32 = sum(p.numel() for p in model.parameters())
    assert n_p32 == n_concat - 256 * (70 - 36) + 32 * 50
    x = torch.from_numpy(np.ascontiguousarray(bank.X_store))
    with torch.no_grad():
        y = model(x, bank)
        scores = model.forward_scores(x, bank)
    assert y.shape == (3, 2) and scores.shape == (3, 50, 2)
    assert torch.isfinite(y).all() and torch.isfinite(scores).all()
    # SPD-A1 folded fast path parity at the multi-group bracket (1e-6 caliber)
    with torch.no_grad():
        folded = model.forward_static_folded(x, bank, model.bank_static_term(bank))
        delta = float((scores[:, -1, :] - folded).abs().max())
    assert delta <= 1e-6, f"P32 static-fold parity exceeded 1e-6: {delta:.3e}"
    # P = 0 -> bitwise independence of the bank E0 (zero-mode semantics)
    with torch.no_grad():
        model.frontend.e0_proj.weight.zero_()
    alt_e0 = np.random.default_rng(99).standard_normal(bank.E0.shape).astype(np.float32)
    from btransform_unified_v1.bank import TaskBank

    bank_alt = TaskBank(
        session_id="alt-e0-p32",
        E0=alt_e0,
        carrier=bank.carrier,
        unit_mask=bank.unit_mask,
        X_store=bank.X_store,
        target_store=bank.target_store,
        window_ids=bank.window_ids,
        calibration_meta=dict(bank.calibration_meta, array_sha256=array_sha256(alt_e0)),
    )
    with torch.no_grad():
        assert torch.equal(model(x, bank), model(x, bank_alt))
    # width axis refuses non-multiples of the 16 local channels
    with pytest.raises(Exception):
        BTransformerUnifiedDecoderIdentity(geometry, seed=42, identity_mode="proj_add", proj_dim=24)
    # default stays the P16 build
    default_model = BTransformerUnifiedDecoderIdentity(geometry, seed=42, identity_mode="proj_add")
    assert default_model.token_in == 20 and default_model.proj_out_dim == 16
    assert default_model.init_meta["proj_dim_override"] is None


def test_cpu_only_matrix_suite() -> None:
    assert not torch.cuda.is_initialized()
