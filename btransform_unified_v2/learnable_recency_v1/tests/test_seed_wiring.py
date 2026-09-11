"""--seed wiring for the M2 proj_add and H1 learnable trainers.

Default runs must stay seed-42 bit-for-bit; ``--seed 43`` must rebuild the model
under 43 while every reference binding still validates against the frozen
seed-42 reference runs.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pytest
import torch

from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args, dataset_config
from learnable_recency_v1.wrap import assert_shared_byte_equal

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m2_projadd_learnable_train as m2_mod  # noqa: E402

DEVICE = torch.device("cpu")


def _shared_digest(model: torch.nn.Module, names: list[str]) -> str:
    named = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _shared_differ(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_named, right_named = dict(left.named_parameters()), dict(right.named_parameters())
    shared = sorted(set(left_named) & set(right_named))
    assert shared
    return any(not torch.equal(left_named[name], right_named[name]) for name in shared)


def test_m2_parser_defaults_seed_42_and_config_unchanged() -> None:
    args = m2_mod.build_parser().parse_args([])
    assert args.seed == m2_mod.SEED == 42
    plain = argparse.ArgumentParser()
    add_learnable_flags(plain)
    plain_args = plain.parse_args([])
    assert {key: value for key, value in vars(args).items() if key in vars(plain_args)} == vars(plain_args)
    assert config_from_args(args, "m2") == dataset_config("m2")
    args43 = m2_mod.build_parser().parse_args(["--seed", "43"])
    assert args43.seed == 43
    assert config_from_args(args43, "m2") == config_from_args(args, "m2")


def test_m2_default_dest_keeps_seed42_name_and_carries_other_seeds() -> None:
    args42 = m2_mod.build_parser().parse_args([])
    assert m2_mod.default_dest(args42) == m2_mod.RESULTS / "m2_projadd_learned_slope_s42"
    smoke42 = m2_mod.build_parser().parse_args(["--max-updates-smoke", "1"])
    assert m2_mod.default_dest(smoke42) == m2_mod.RESULTS / "smoke" / "m2_projadd_learned_slope_s42_v2"
    args43 = m2_mod.build_parser().parse_args(["--seed", "43", "--ladder", "default"])
    assert m2_mod.default_dest(args43) == m2_mod.RESULTS / "m2_projadd_learned_slope_default_s43"
    smoke43 = m2_mod.build_parser().parse_args(["--seed", "43", "--ladder", "default", "--max-updates-smoke", "1"])
    assert m2_mod.default_dest(smoke43) == m2_mod.RESULTS / "smoke" / "m2_projadd_learned_slope_default_s43_v2"


def test_m2_seed_43_build_differs_from_42_and_pairs_to_same_seed_stock() -> None:
    cfg = dataset_config("m2", ladder="default")
    model43 = m2_mod.learnable_decoder(DEVICE, cfg, 43)
    stock43 = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=43, proj_dim=16)
    shared = assert_shared_byte_equal(model43, stock43)
    assert shared
    stock42 = RiftDecoder("m2", context_bins=50, bias_mode="recency", seed=42, proj_dim=16)
    assert _shared_differ(model43, stock42)
    init43 = m2_mod.assert_paired_initialization(model43, DEVICE, cfg, 43)
    assert init43["named_parameters_byte_equal"] is True
    assert init43["init_forward_max_abs"] <= 1e-6
    assert "seed=43" in init43["reference_factory"]
    model42 = m2_mod.learnable_decoder(DEVICE, cfg, 42)
    init42 = m2_mod.assert_paired_initialization(model42, DEVICE, cfg, 42)
    assert init42["shared_parameter_names"] == init43["shared_parameter_names"]
    assert _shared_digest(model43, shared) != _shared_digest(model42, shared)


def test_h1_seed_wiring() -> None:
    pytest.importorskip("sklearn")
    import h1_learnable_train as h1_mod

    args = h1_mod.build_parser().parse_args([])
    assert args.seed == h1_mod.SEED == 42
    assert config_from_args(args, "h1") == dataset_config("h1")
    assert h1_mod.default_dest(args, config_from_args(args, "h1")).name == "h1_learned_slope_s42"
    args43 = h1_mod.build_parser().parse_args(["--seed", "43", "--ladder", "default"])
    assert h1_mod.default_dest(args43, config_from_args(args43, "h1")).name == "h1_learned_slope_default_s43"

    cfg = dataset_config("h1", ladder="default")
    reference_sha = h1_mod.ht._sha_state(h1_mod.stock_decoder("recency", DEVICE, "dense", 42))
    model43 = h1_mod.learnable_decoder(DEVICE, cfg, "dense", 43)
    assert_shared_byte_equal(model43, h1_mod.stock_decoder("recency", DEVICE, "dense", 43))
    assert _shared_differ(model43, h1_mod.stock_decoder("recency", DEVICE, "dense", 42))
    init43 = h1_mod.assert_paired(model43, DEVICE, {"reference_initialization_sha256": reference_sha}, cfg, 43)
    model42 = h1_mod.learnable_decoder(DEVICE, cfg, "dense", 42)
    init42 = h1_mod.assert_paired(model42, DEVICE, {"reference_initialization_sha256": reference_sha}, cfg, 42)
    assert init43["shared_parameter_names"] == init42["shared_parameter_names"]
    assert init43["shared_initialization_sha256"] != init42["shared_initialization_sha256"]
    assert init43["init_forward_max_abs"] <= 1e-6
    assert init42["shared_initialization_sha256"] == reference_sha


def test_seed_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["m2_projadd_learnable_train.py", "--seed", "0"])
    with pytest.raises(SystemExit) as exc:
        m2_mod.main()
    assert exc.value.code == 2
    pytest.importorskip("sklearn")
    import h1_learnable_train as h1_mod

    monkeypatch.setattr(sys, "argv", ["h1_learnable_train.py", "--seed", "-1"])
    with pytest.raises(SystemExit) as exc:
        h1_mod.main()
    assert exc.value.code == 2
