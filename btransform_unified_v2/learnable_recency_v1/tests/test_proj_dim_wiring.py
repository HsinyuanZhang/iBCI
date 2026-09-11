"""--proj-dim wiring for the M2/M1/H1 proj_add learnable arms (CPU only).

P16 defaults must keep every historical name/dest/behavior; a P32 run rebuilds
the run's own decoders and pairing references at P32 while every frozen
reference binding stays P16.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from learnable_recency_v1.config import dataset_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m2_projadd_learnable_score as score_mod  # noqa: E402
import m2_projadd_learnable_train as m2_mod  # noqa: E402

RUNNER_PATH = SCRIPTS / "m1_projadd_learnable_train.py"
_spec = importlib.util.spec_from_file_location("_m1_projadd_learnable_train_projdim", RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
m1_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1_mod)

DEVICE = torch.device("cpu")
WORKSPACE = SCRIPTS.parents[2]
SPINT_MAIN = WORKSPACE / "SPINT-main"


def _named_sha(model: torch.nn.Module, names: list[str]) -> str:
    named = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _same_shape_shared(left: torch.nn.Module, right: torch.nn.Module) -> list[str]:
    left_named, right_named = dict(left.named_parameters()), dict(right.named_parameters())
    return sorted(name for name, param in left_named.items() if name in right_named and right_named[name].shape == param.shape)


def _import_h1():
    """Import h1_learnable_train with SPINT-main's top-level ``src`` package winning.

    The M2 import chain binds the top-level ``src`` module to
    streaming_calibration_exp/src, while the H1 signed-state chain imports
    ``src.data.h1_m4_eb_pilot`` from SPINT-main/src. Pop the cached binding so
    the H1 chain re-resolves it (the full-suite collection order does the same
    implicitly by importing the H1 ablation tests first).
    """
    pytest.importorskip("sklearn")
    sys.modules.pop("src", None)
    if str(SPINT_MAIN) not in sys.path:
        sys.path.insert(0, str(SPINT_MAIN))
    import h1_learnable_train as h1_mod

    return h1_mod


def test_parser_defaults_proj_dim_16_everywhere() -> None:
    assert m2_mod.build_parser().parse_args([]).proj_dim == 16 == m2_mod.PROJ_DIM
    assert m2_mod.build_parser().parse_args(["--proj-dim", "32"]).proj_dim == 32
    assert score_mod.PROJ_DIM == 16  # frozen reference binding stays P16
    h1_mod = _import_h1()
    assert h1_mod.build_parser().parse_args([]).proj_dim == 16 == h1_mod.PROJ_DIM
    assert h1_mod.build_parser().parse_args(["--proj-dim", "32"]).proj_dim == 32
    assert m1_mod.build_parser().parse_args([]).proj_dim == 16 == m1_mod.PROJ_DIM
    assert m1_mod.build_parser().parse_args(["--proj-dim", "32"]).proj_dim == 32


def test_dest_naming_unchanged_at_p16_and_p32_variant() -> None:
    args16 = m2_mod.build_parser().parse_args([])
    assert m2_mod.default_dest(args16) == m2_mod.RESULTS / "m2_projadd_learned_slope_s42"
    seed43 = m2_mod.build_parser().parse_args(["--seed", "43", "--ladder", "default"])
    assert m2_mod.default_dest(seed43) == m2_mod.RESULTS / "m2_projadd_learned_slope_default_s43"
    args32 = m2_mod.build_parser().parse_args(["--proj-dim", "32"])
    assert m2_mod.default_dest(args32) == m2_mod.RESULTS / "m2_projadd_learned_slope_scaled_p32_s42"
    smoke32 = m2_mod.build_parser().parse_args(["--proj-dim", "32", "--max-updates-smoke", "1"])
    assert m2_mod.default_dest(smoke32) == m2_mod.RESULTS / "smoke" / "m2_projadd_learned_slope_scaled_p32_s42_v2"

    h1_mod = _import_h1()
    assert h1_mod.default_dest(args16, dataset_config("h1")).name == "h1_learned_slope_s42"
    assert h1_mod.default_dest(args32, dataset_config("h1")).name == "h1_learned_slope_scaled_p32_s42"

    assert m1_mod.default_dest(args16) == m1_mod.RESULTS / "m1_projadd_learned_slope_s42"
    smoke16 = m1_mod.build_parser().parse_args(["--max-updates-smoke", "1"])
    assert m1_mod.default_dest(smoke16) == m1_mod.RESULTS / "smoke" / "m1_projadd_learned_slope_s42_v1"
    assert m1_mod.default_dest(args32) == m1_mod.RESULTS / "m1_projadd_learned_slope_scaled_p32_s42"


def test_m2_p32_decoder_pairing_and_sha_split() -> None:
    cfg = dataset_config("m2", ladder="default")
    model32 = m2_mod.learnable_decoder(DEVICE, cfg, 42, 32)
    assert model32.proj_dim == 32
    init32 = m2_mod.assert_paired_initialization(model32, DEVICE, cfg, 42, 32)
    assert init32["named_parameters_byte_equal"] is True
    assert init32["init_forward_max_abs"] <= 1e-6
    assert "P32" in init32["reference_factory"] and "seed=42" in init32["reference_factory"]
    assert init32["trainable_new_parameter_count"] == 24  # proj_dim never touches new scalars
    model16 = m2_mod.learnable_decoder(DEVICE, cfg, 42, 16)
    shared = _same_shape_shared(model16, model32)
    assert shared, "p16/p32 builds must share same-shape parameter names"
    assert _named_sha(model16, shared) != _named_sha(model32, shared)


def test_h1_stock_decoder_matches_ht_decoder_bit_for_bit() -> None:
    h1_mod = _import_h1()
    local = h1_mod.stock_decoder("recency", DEVICE, "dense", 42)
    frozen = h1_mod.ht._decoder("recency", DEVICE, 300, "dense")
    assert h1_mod.ht._sha_state(local) == h1_mod.ht._sha_state(frozen)
    local_state, frozen_state = local.state_dict(), frozen.state_dict()
    assert set(local_state) == set(frozen_state)
    for key, value in local_state.items():
        assert torch.equal(value, frozen_state[key]), f"byte drift at {key}"


def test_h1_p32_decoder_pairing_and_sha_split() -> None:
    h1_mod = _import_h1()
    cfg = dataset_config("h1", ladder="default")
    reference_sha = h1_mod.ht._sha_state(h1_mod.stock_decoder("recency", DEVICE, "dense", 42))
    model32 = h1_mod.learnable_decoder(DEVICE, cfg, "dense", 42, 32)
    assert model32.proj_dim == 32
    init32 = h1_mod.assert_paired(model32, DEVICE, {"reference_initialization_sha256": reference_sha}, cfg, 42, 32)
    assert init32["init_forward_max_abs"] <= 1e-6
    model16 = h1_mod.learnable_decoder(DEVICE, cfg, "dense", 42, 16)
    init16 = h1_mod.assert_paired(model16, DEVICE, {"reference_initialization_sha256": reference_sha}, cfg, 42, 16)
    assert init16["shared_initialization_sha256"] == reference_sha
    with pytest.raises(RuntimeError):
        h1_mod.assert_paired(model16, DEVICE, {"reference_initialization_sha256": "0" * 64}, cfg, 42, 16)
    shared = _same_shape_shared(model16, model32)
    assert shared, "p16/p32 builds must share same-shape parameter names"
    assert _named_sha(model16, shared) != _named_sha(model32, shared)


def test_m1_p32_decoder_pairing_and_sha_split() -> None:
    cfg = dataset_config("m1", tier="learned_slope")
    model32 = m1_mod.learnable_decoder(DEVICE, cfg, 32)
    assert model32.proj_dim == 32
    assert m1_mod.learnable_decoder(DEVICE, cfg).proj_dim == 16  # default unchanged
    # P32 cannot match the P16 reference SHA; pairing stays structural (byte-equal
    # against the same-seed P32 stock build) and the frozen SHA check is skipped.
    init32 = m1_mod.assert_paired(model32, DEVICE, {"initialization_sha256": "0" * 64}, cfg, 32)
    assert init32["init_forward_max_abs"] <= 1e-6
    model16 = m1_mod.learnable_decoder(DEVICE, cfg, 16)
    with pytest.raises(RuntimeError):
        m1_mod.assert_paired(model16, DEVICE, {"initialization_sha256": "0" * 64}, cfg, 16)
    shared = _same_shape_shared(model16, model32)
    assert shared, "p16/p32 builds must share same-shape parameter names"
    assert _named_sha(model16, shared) != _named_sha(model32, shared)


def test_score_side_reads_proj_dim_from_run_meta() -> None:
    cfg = dataset_config("m2")
    assert score_mod.rebuild_model({}, cfg, DEVICE).proj_dim == 16
    assert score_mod.rebuild_model({"proj_dim": 32}, cfg, DEVICE).proj_dim == 32
    binding = {"run": str(score_mod.REFERENCE)}
    p32_meta = {"identity_interface": "proj_add", "proj_dim": 32, "paired_recency_reference": binding}
    assert score_mod.assert_projadd_reference(p32_meta) == 32
    p16_meta = {"identity_interface": "proj_add", "paired_recency_reference": binding}
    assert score_mod.assert_projadd_reference(p16_meta) == 16
    with pytest.raises(RuntimeError):
        score_mod.assert_projadd_reference({**p32_meta, "identity_interface": "concat"})
    with pytest.raises(RuntimeError):
        score_mod.assert_projadd_reference({**p32_meta, "proj_dim": 0})
    assert m1_mod.rebuild_model({}, DEVICE).proj_dim == 16
    assert m1_mod.rebuild_model({"proj_dim": 32}, DEVICE).proj_dim == 32
    h1_mod = _import_h1()
    assert h1_mod.rebuild_model({}, DEVICE).proj_dim == 16
    assert h1_mod.rebuild_model({"proj_dim": 32}, DEVICE).proj_dim == 32


def test_proj_dim_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, module, bad in (
        ("m2_projadd_learnable_train.py", m2_mod, "0"),
        ("m1_projadd_learnable_train.py", m1_mod, "-4"),
    ):
        monkeypatch.setattr(sys, "argv", [name, "--proj-dim", bad])
        with pytest.raises(SystemExit) as exc:
            module.main()
        assert exc.value.code == 2
    h1_mod = _import_h1()
    monkeypatch.setattr(sys, "argv", ["h1_learnable_train.py", "--proj-dim", "0"])
    with pytest.raises(SystemExit) as exc:
        h1_mod.main()
    assert exc.value.code == 2
