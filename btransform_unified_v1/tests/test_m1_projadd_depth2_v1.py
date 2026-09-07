"""M1 proj_add depth-2 front-row cell tests (ADDENDUM-DEPTH2-PROMOTED, 2026-09-07).

CPU only (conftest masks CUDA). Workorder
``WORKORDER_M1_PROJADD_RUNTIME_QUALITY_V1_20260907.md`` §7 priority plan
promoted to a formal cell by user directive 2026-09-07: proj_add P16 +
CausalPE temporal depth 4->2, everything else frozen. These tests pin:

  - the depth-2 geometry (2 blocks, init_meta contract) and the exact
    parameter arithmetic (P16 count minus 2 x per-block count);
  - the public default stays the 4-layer build (bit-identical init walk,
    unchanged state-dict keys) — no silent global depth change;
  - forward/causal semantics survive the depth cut;
  - the new series script wiring (GPU pin, faces, gate, cells).

No real NWB file, no checkpoint load, no GPU, no training. NOT SPINT.
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
from btransform_unified_v1.identity_variant import (
    BTransformerUnifiedDecoderIdentity,
    CausalTransformerStackDepth,
)
from btransform_unified_v1.model import N_LAYERS, CausalTransformerBlock

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "m1_projadd_depth2_series.py"

DEPTH2_LAYERS = 2


def _per_block_param_count() -> int:
    """Exact parameter count of one CausalTransformerBlock (256-wide, FFN 512)."""
    block = CausalTransformerBlock()
    return int(sum(p.numel() for p in block.parameters()))


def _expected_params(temporal_layers: int) -> int:
    """P16 parameter arithmetic at a given temporal depth."""
    return int(mp.m1_projadd_param_count(16) - (N_LAYERS - temporal_layers) * _per_block_param_count())


def _build(temporal_layers: int | None, seed: int = 42) -> BTransformerUnifiedDecoderIdentity:
    model = BTransformerUnifiedDecoderIdentity(
        mp.m1_projadd_geometry(16),
        seed=seed,
        identity_mode="proj_add",
        proj_dim=16,
        temporal_layers=temporal_layers,
    )
    return model


def _bank(seed: int = 5) -> object:
    rng = np.random.default_rng(seed)
    return mp.make_m1_bank(
        mp.M1_OUTER_SESSION,
        rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32),
        rng.standard_normal((mp.M1_UNITS, mp.M1_CARRIER_DIM)).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# (1) depth-2 geometry + parameter arithmetic
# ---------------------------------------------------------------------------


def test_depth2_geometry_contract() -> None:
    model = _build(DEPTH2_LAYERS).eval()
    # everything except the temporal block count is the frozen P16 contract
    assert model.task == "m1"
    assert model.window == mp.M1_WINDOW == 100 and model.prefix == mp.M1_PREFIX == 0
    assert model.l_in == 100 and model.base_e0_dim == 100
    assert model.units == 64 and model.out_dim == 16
    assert model.identity_mode == "proj_add" and model.init_meta["matrix_letter"] == "f"
    assert model.token_in == 20 and model.proj_out_dim == 16
    assert tuple(model.frontend.e0_proj.weight.shape) == (16, 100)
    # the depth axis itself
    assert len(model.temporal.blocks) == DEPTH2_LAYERS
    assert isinstance(model.temporal, CausalTransformerStackDepth)
    assert model.temporal.n_layers == DEPTH2_LAYERS
    assert model.init_meta["temporal_layers"] == DEPTH2_LAYERS
    # PE buffer survives the depth cut at the full window length
    assert tuple(model.temporal.pe.shape) == (mp.M1_WINDOW, model.temporal.pe.size(1))


def test_depth2_param_count_arithmetic() -> None:
    model = _build(DEPTH2_LAYERS)
    n = sum(p.numel() for p in model.parameters())
    assert _per_block_param_count() == 527104  # frozen per-block count (256/8h/FFN512)
    assert n == _expected_params(DEPTH2_LAYERS) == 2479408
    assert n == mp.m1_projadd_param_count(16) - 2 * 527104
    assert model.init_meta["param_count"] == n


@pytest.mark.parametrize("depth", [1, 3])
def test_other_depths_are_wellformed(depth: int) -> None:
    # any positive int is a legal depth-axis build (parameterized axis); the
    # CELL only uses 2, and non-int / non-positive depths are refused.
    model = _build(depth)
    assert len(model.temporal.blocks) == depth
    assert sum(p.numel() for p in model.parameters()) == _expected_params(depth)
    with pytest.raises(plan.BTransformerUnifiedError):
        _build(0)
    with pytest.raises(plan.BTransformerUnifiedError):
        _build("two")  # type-strict: strings are not a depth


def test_depth_stack_refuses_non_positive() -> None:
    from btransform_unified_v1.model import CausalTransformerStack

    with pytest.raises(plan.BTransformerUnifiedError):
        CausalTransformerStackDepth(mp.M1_WINDOW, 0)
    ok = CausalTransformerStackDepth(mp.M1_WINDOW, 2)
    assert len(ok.blocks) == 2
    # default parent build is untouched by this class existing
    assert len(CausalTransformerStack(mp.M1_WINDOW).blocks) == N_LAYERS == 4


# ---------------------------------------------------------------------------
# (2) public default stays 4 layers, bit-identical
# ---------------------------------------------------------------------------


def test_default_depth_stays_four_bit_identical() -> None:
    """temporal_layers=None must reproduce the settled P16 build bit for bit."""
    default = _build(None)
    ref = mp.build_m1_projadd_model(16, seed=42)
    assert default.temporal_layers == N_LAYERS == 4
    assert not isinstance(default.temporal, CausalTransformerStackDepth)
    assert len(default.temporal.blocks) == 4
    assert default.init_meta["temporal_layers"] == 4
    assert sorted(default.state_dict()) == sorted(ref.state_dict())
    for key, tensor in ref.state_dict().items():
        assert torch.equal(default.state_dict()[key], tensor), key
    assert sum(p.numel() for p in default.parameters()) == mp.m1_projadd_param_count(16) == 3533616


def test_depth4_explicit_equals_default() -> None:
    explicit = _build(4)
    default = _build(None)
    assert explicit.temporal_layers == default.temporal_layers == 4
    for key, tensor in default.state_dict().items():
        assert torch.equal(explicit.state_dict()[key], tensor), key


def test_depth2_forward_and_causal_check() -> None:
    model = _build(DEPTH2_LAYERS).eval()
    bank = _bank(seed=7)
    x = torch.from_numpy(
        np.random.default_rng(9).standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32)
    )
    with torch.no_grad():
        y = model(x, bank)
        scores = model.forward_scores(x, bank)
    assert y.shape == (2, mp.M1_OUT_DIM) and scores.shape == (2, mp.M1_WINDOW, mp.M1_OUT_DIM)
    assert torch.isfinite(y).all() and torch.isfinite(scores).all()
    report = model.causal_check(x, bank)
    assert report["passed"] is True and report["length"] == mp.M1_WINDOW


def test_depth2_p_zero_degeneration_control() -> None:
    """P=0 -> depth-2 forward bitwise independent of the bank E0 (letter f)."""
    model = _build(DEPTH2_LAYERS).eval()
    rng = np.random.default_rng(11)
    x = torch.from_numpy(rng.standard_normal((2, mp.M1_WINDOW, mp.M1_UNITS)).astype(np.float32))
    bank_a = _bank(seed=13)
    bank_b = mp.make_m1_bank(
        mp.M1_OUTER_SESSION,
        rng.standard_normal((mp.M1_UNITS, mp.M1_E0_DIM)).astype(np.float32),
        bank_a.carrier,
    )
    with torch.no_grad():
        model.frontend.e0_proj.weight.zero_()
        assert torch.equal(model.forward_scores(x, bank_a), model.forward_scores(x, bank_b))


# ---------------------------------------------------------------------------
# (3) script wiring: cells, faces, gate, GPU pin
# ---------------------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location("m1_projadd_depth2_series_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_cells_faces_gate_and_gpu_pin() -> None:
    script = _load_script()
    # arbitration target switched GPU1 -> GPU0 per user directive 2026-09-07
    assert script.CUDA_PIN == "0"
    assert script.GPU_UUIDS == {
        "0": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
        "1": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
    }
    assert script.GPU_UUIDS[script.CUDA_PIN] == "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
    assert script.SEED == 42 and script.EPOCHS == 24 and script.BATCH_SIZE == 32
    assert script.PEAK_LR == 1e-4
    assert script.DEPTH_CELLS == (2, 4)  # depth-2 + matched depth-4 baseline pair
    # judgment face + preregistered gate
    assert script.LOSO_WINDOWS == 26496 and script.MINIVAL_WINDOWS == 31252
    assert script.GATE_DELTA_R2 == -0.01  # preregistered §7 折中门
    assert script.GATE_DELTA_R2_ADJUSTED == -0.03  # user-adjusted threshold 2026-09-07
    assert script.TRAIN_SESSIONS == mp.M1_SOURCE_SESSIONS  # 26/27/28 only
    assert mp.M1_OUTER_SESSION not in script.TRAIN_SESSIONS
    # picks travel the promotion note + gate into receipts
    joined = json.dumps(script.PICKS)
    assert "depth 4->2" in joined and "2026-09-07" in joined and "endpoint24" in joined.lower()


def test_depth2_param_expectations_table() -> None:
    script = _load_script()
    table = script.EXPECTED_PARAMS_BY_DEPTH
    assert table[4] == 3533616 == mp.m1_projadd_param_count(16)
    assert table[2] == 2479408 == _expected_params(2)
    assert table[2] == table[4] - 2 * 527104


def test_cpu_only_suite() -> None:
    assert not torch.cuda.is_initialized()
