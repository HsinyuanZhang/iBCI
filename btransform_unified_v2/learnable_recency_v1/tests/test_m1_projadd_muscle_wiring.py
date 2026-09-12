"""Wiring tests for the M1 muscle proj_add learnable-recency runner (CPU only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from btransform_unified_v1.bank import array_sha256
from learnable_recency_v1.config import dataset_config
from learnable_recency_v1.p8 import PaddedProj8
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from learnable_recency_v1.wrap import (
    assert_shared_byte_equal,
    new_parameter_names,
    trainable_new_parameter_count,
)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "m1_projadd_muscle_train.py"
_spec = importlib.util.spec_from_file_location("_m1_projadd_muscle_train", RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

DEVICE = torch.device("cpu")


def test_carrier_pack_binding_pins_the_official_pack() -> None:
    carriers, binding = runner._carrier_binding(runner.m1_flat.DEFAULT_PACK)
    assert set(carriers) == set(runner.SOURCE) | set(runner.HO)
    for session, value in carriers.items():
        assert value.shape == (64, 4) and value.dtype == np.float32
        assert bool(np.isfinite(value).all())
    assert binding["carrier_variant"] == "muscle_response16_svd4/global_rms"
    assert binding["identity_interface"] == "proj_add"
    assert binding["carrier_pack_npz_sha256"] == runner._sha_file(runner.m1_flat.DEFAULT_PACK)
    assert binding["carrier_pack_receipt_sha256"] == runner._sha_file(
        runner.m1_flat.DEFAULT_PACK.with_suffix(".json")
    )
    assert binding["carrier_pack_receipt_body"]["method"] == "muscle_response16_svd4/global_rms"
    assert binding["carrier_pack_receipt_body"]["official_test_included"] is False
    assert len(binding["fit_sha256"]) == 64
    assert binding["projadd_muscle_runner_py_sha256"] == runner._sha_file(RUNNER_PATH)


def test_reference_anchor_records_582205_as_provenance_only() -> None:
    class _Args:
        carrier_pack = runner.m1_flat.DEFAULT_PACK
        paired_reference = runner.m1_flat.DEFAULT_REFERENCE

    carriers, binding, paired, anchor = runner.muscle_reference(_Args())
    assert binding["carrier_pack_npz_sha256"] == anchor["carrier_pack_npz_sha256"]
    assert anchor["official_submission_id"] == 582205
    assert anchor["official_test_split_m1_held_out_r2_mean"] == 0.6259910151098528
    assert anchor["provenance_only"] is True
    assert anchor["participates_in_contract_validation"] is False
    # The muscle formal run pins the data contracts; its own carrier pack must
    # be exactly this pack.
    assert paired["fit_sha256"] == binding["fit_sha256"]
    assert paired["mainline"] == runner.MAINLINE_NOTE
    assert runner._paired_summary(paired) == anchor["paired_reference_summary"]


def _src_modules() -> dict:
    return {name: module for name, module in sys.modules.items() if name == "src" or name.startswith("src.")}


@pytest.fixture(scope="module")
def muscle_banks():
    """Fullsession face + legacy banks with the official carrier replaced.

    Under the full pytest suite the top-level ``src`` package can already be
    bound to ``SPINT-main/src`` by the H1 runner chains, which breaks the
    legacy LOSO imports (they need ``streaming_calibration_exp/src``).  Swap the
    ``src`` bindings for the build and restore the previous state afterwards.
    """
    saved = _src_modules()
    saved_path = list(sys.path)
    for name in list(saved):
        sys.modules.pop(name, None)
    streaming_root = str(runner.WS / "streaming_calibration_exp")
    if streaming_root in sys.path:
        sys.path.remove(streaming_root)
    sys.path.insert(0, streaming_root)
    try:
        carriers, _binding = runner._carrier_binding(runner.m1_flat.DEFAULT_PACK)
        dataset, _sampler = runner.frozen.legacy.build_fullsession_face()
        legacy_banks, _report = runner.frozen.legacy.build_fullsession_banks(dataset)
        banks = runner.frozen._replace_carriers(legacy_banks, carriers, runner.SOURCE)
        yield carriers, legacy_banks, banks
    finally:
        for name in _src_modules():
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


def test_replace_carriers_matches_pack_bitwise(muscle_banks) -> None:
    carriers, _legacy_banks, banks = muscle_banks
    for session in runner.SOURCE:
        expected = np.ascontiguousarray(carriers[session], dtype=np.float32)
        assert np.array_equal(banks[session].carrier, expected)
        assert banks[session].calibration_meta["carrier_sha256"] == array_sha256(expected)
        assert banks[session].calibration_meta["carrier_method"] == "muscle_response16_svd4/global_rms"


def test_replace_carriers_keeps_legacy_b3_identity_e0(muscle_banks) -> None:
    _carriers, legacy_banks, banks = muscle_banks
    for session in runner.SOURCE:
        assert banks[session].E0.shape == legacy_banks[session].E0.shape == (64, 100)
        assert np.array_equal(banks[session].E0, legacy_banks[session].E0)
        assert banks[session].calibration_meta["array_sha256"] == legacy_banks[session].calibration_meta["array_sha256"]
        assert banks[session].calibration_meta["budget"] == legacy_banks[session].calibration_meta["budget"] == 10
        assert np.array_equal(banks[session].unit_mask, legacy_banks[session].unit_mask)


def test_learnable_decoder_p16_wiring() -> None:
    cfg = dataset_config("m1", tier="learned_slope")
    model = runner.learnable_decoder(DEVICE, cfg)
    assert isinstance(model.temporal, LearnableRecencyTemporal)
    assert model.temporal._attention_backend == "local"
    assert model.context_bins == 100 and model.proj_dim == 16
    assert tuple(model.temporal_config.windows) == (25, 25, 25, 24)
    assert int(model.temporal_config.width) == 256
    assert new_parameter_names(model) == ["temporal.slope_log"]
    assert trainable_new_parameter_count(model) == 24  # 4 layers x 6 active heads
    fixed = runner.learnable_decoder(DEVICE, dataset_config("m1", tier="fixed"))
    assert not isinstance(fixed.temporal, LearnableRecencyTemporal)
    assert new_parameter_names(fixed) == []
    assert trainable_new_parameter_count(fixed) == 0


def test_learnable_decoder_p8_truncation() -> None:
    cfg = dataset_config("m1", tier="learned_slope")
    model = runner.learnable_decoder(DEVICE, cfg, proj_dim=8)
    assert model.proj_dim == 8
    proj = model.frontend.e0_proj
    assert isinstance(proj, PaddedProj8)
    assert proj.linear.bias is None and proj.linear.out_features == 8 and proj.linear.in_features == 100
    tokens = torch.zeros(1, 100)
    out = proj(tokens)
    assert out.shape[-1] == 16 and int(torch.count_nonzero(out[..., 8:])) == 0
    # The truncated rows are the first 8 rows of the same-seed P16 stock build.
    stock16 = runner.stock_decoder(DEVICE, cfg.layers, 16)
    assert torch.equal(proj.linear.weight, stock16.frontend.e0_proj.weight[:8])


def test_assert_paired_against_local_stock_decoder() -> None:
    cfg = dataset_config("m1", tier="learned_slope")
    model = runner.learnable_decoder(DEVICE, cfg)
    stock = runner.stock_decoder(DEVICE, cfg.layers)
    shared = assert_shared_byte_equal(model, stock)
    assert shared and "temporal.blocks.0.qkv.weight" in shared
    assert set(new_parameter_names(model)).isdisjoint(shared)
    assert runner.shared_init_sha(model, shared) == runner.shared_init_sha(stock, shared)
    report = runner.assert_paired(model, DEVICE, cfg)
    assert report["new_parameter_names"] == ["temporal.slope_log"]
    assert report["trainable_new_parameter_count"] == 24
    assert report["init_forward_max_abs"] <= 1e-6
    fixed_report = runner.assert_paired(
        runner.learnable_decoder(DEVICE, dataset_config("m1", tier="fixed")),
        DEVICE,
        dataset_config("m1", tier="fixed"),
    )
    assert fixed_report["new_parameter_names"] == []
    assert fixed_report["trainable_new_parameter_count"] == 0
    assert fixed_report["init_forward_max_abs"] <= 1e-6


def test_train_refuses_nonempty_destination(tmp_path) -> None:
    (tmp_path / "occupied").write_text("stale", encoding="utf-8")
    args = runner.build_parser().parse_args(
        ["--tier", "learned_slope", "--stage", "train", "--device", "cpu", "--dest", str(tmp_path)]
    )
    with pytest.raises(FileExistsError):
        runner.run_train(args)


def test_default_dest_convention() -> None:
    formal = runner.build_parser().parse_args(["--tier", "fixed"])
    assert runner.default_dest(formal) == runner.RESULTS / "m1_projadd_muscle_fixed_p16_s42"
    smoke = runner.build_parser().parse_args(["--tier", "learned_slope", "--max-updates-smoke", "1"])
    assert runner.default_dest(smoke) == runner.RESULTS / "smoke" / "m1_projadd_muscle_learned_slope_p16_s42_v1"


def test_source_hashes_bind_frozen_muscle_pipeline_and_this_runner() -> None:
    hashes = runner._source_hashes()
    assert str(RUNNER_PATH.resolve()) in hashes
    assert str(runner.M1_FLAT.resolve()) in hashes
    frozen_hashes = runner.frozen._source_hashes()
    assert all(hashes[path] == sha for path, sha in frozen_hashes.items())
    assert len(hashes) == len(frozen_hashes) + 2
