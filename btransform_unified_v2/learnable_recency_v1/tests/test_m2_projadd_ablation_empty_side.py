"""activity_only_empty_side ablation tests: zero-side E0, determinism, no label reads.

The empty-side arm recomputes E0 with the frozen champion encoder on
side = zeros(N, 8) so labels (MOVE-T4 from calibration target angles) cannot
reach the identity path.  These tests pin: the encoder is the one that built
the frozen cache, the zero-side E0 differs from the cached E0, the digest is
deterministic, the transform zeroes the carrier without mutating the input
bank, and the compute path never opens ``T.npy``/angles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from btransform_unified_v1.bank import array_sha256, make_synthetic_bank

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m2_projadd_ablation_train as train_mod  # noqa: E402
import m2_projadd_ablation_score as score_mod  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[2].parent
DUAL_CACHE = WORKSPACE / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
SESSION = "ses-2020-10-19-Run1"
ACTIVITY = DUAL_CACHE / "source_train" / SESSION / "calib_activity.npy"
CACHED_E0 = DUAL_CACHE / "source_train" / SESSION / "e0_u.pt"
ENCODER_FILES = (
    WORKSPACE / "tfpd_exploration/results/m2_hold_film_probe_v1/film_states.pt",
    WORKSPACE / "tfpd_exploration/results/m2_movement_t4_empty_epoch_pick_v1/selected_head.pt",
)
REAL_ENCODER = pytest.mark.skipif(
    not ACTIVITY.is_file() or not all(path.is_file() for path in ENCODER_FILES),
    reason="frozen dual-track cache / champion film+head artifacts not present",
)


def _bank(seed: int = 3):
    return make_synthetic_bank("m2", n_windows=4, seed=seed)


def test_sealed_identity_tuples_unchanged_and_empty_side_arm_additive() -> None:
    # The historical seals stay byte-stable; the label-free arm is additive.
    assert train_mod.IDENTITIES == ("full", "activity_only", "norm_only")
    assert train_mod.ABLATION_IDENTITIES == ("activity_only", "norm_only")
    assert train_mod.EMPTY_SIDE_IDENTITY == "activity_only_empty_side"
    assert train_mod.IDENTITY_CHOICES == ("full", "activity_only", "norm_only", "activity_only_empty_side")
    assert train_mod.ABLATION_ARMS == ("activity_only", "norm_only", "activity_only_empty_side")
    assert train_mod.SIDE_DIM == 8
    assert train_mod.EMPTY_SIDE_SEMANTICS == "zeros(N,8): no MOVE-T4, no contrast"
    parser = train_mod.build_parser()
    args = parser.parse_args(["--identity", "activity_only_empty_side"])
    assert args.identity == "activity_only_empty_side"
    train_mod.ensure_ablation_identity("activity_only_empty_side")
    with pytest.raises(ValueError):
        train_mod.ensure_ablation_identity("full")
    with pytest.raises(ValueError):
        train_mod.ensure_ablation_identity("made_up")


@REAL_ENCODER
def test_encoder_is_the_one_that_built_the_cache() -> None:
    """The film-only construction reproduces the cached E0 from [MOVE-T4, 0].

    This test is encoder-equivalence evidence ONLY — the production empty-side
    path never reads ``T.npy``.  Here the cached side is rebuilt from the
    cached MOVE-T4 to prove the encoder weights equal stage0's
    ``load_frozen_champion().student.id_encoder``.
    """
    from tfpd_exploration.src.m2_dual_track_v1 import champion as old_champion
    from tfpd_exploration.src.m2_dual_track_v1 import plan as old_dual_plan

    encoder, record = train_mod.frozen_empty_side_encoder()
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.training is False
    assert record["head_state_sha256"] == old_dual_plan.SELECTED_HEAD_STATE_SHA256
    activity = np.load(ACTIVITY, mmap_mode="r")
    assert tuple(activity.shape) == train_mod.SUPPORT_SHAPE == (33, 100, 96)
    trials = torch.from_numpy(np.array(activity, dtype=np.float32, copy=True))
    move_t4 = np.load(DUAL_CACHE / "source_train" / SESSION / "T.npy")
    e0_repro, _ = old_champion.native_e0_and_u(encoder, trials, old_champion.empty_contrast_side(move_t4))
    cached = torch.load(CACHED_E0, map_location="cpu", weights_only=False)["E0"]
    assert torch.equal(e0_repro, cached)


@REAL_ENCODER
def test_zero_side_e0_differs_from_cache_and_is_deterministic() -> None:
    encoder, _ = train_mod.frozen_empty_side_encoder()
    e0_first = train_mod.empty_side_e0(encoder, ACTIVITY)
    e0_second = train_mod.empty_side_e0(encoder, ACTIVITY)
    assert e0_first.shape == (96, 50)
    assert e0_first.dtype == np.float32
    # same input -> identical bits and digest
    np.testing.assert_array_equal(e0_first, e0_second)
    assert train_mod.float64_sha256(e0_first) == train_mod.float64_sha256(e0_second)
    cached = torch.load(CACHED_E0, map_location="cpu", weights_only=False)["E0"].numpy()
    assert not np.array_equal(e0_first, cached)
    assert array_sha256(e0_first) != array_sha256(cached)
    record_sha = train_mod.float64_sha256(e0_first)
    assert record_sha != train_mod.float64_sha256(cached)


@REAL_ENCODER
def test_empty_side_e0_reads_only_calib_activity_never_t_or_angles(tmp_path, monkeypatch) -> None:
    """A poisoned np.load proves the identity path never opens T.npy/angles."""
    encoder, _ = train_mod.frozen_empty_side_encoder()
    np.save(tmp_path / "calib_activity.npy", np.asarray(np.load(ACTIVITY, mmap_mode="r")))
    np.save(tmp_path / "T.npy", np.full((96, 4), 7.0, dtype=np.float32))  # decoy label carrier
    np.save(tmp_path / "angles.npy", np.full((33,), 1.5, dtype=np.float32))  # decoy label angles
    opened: list[str] = []
    real_load = np.load

    def poisoned_load(file, *args, **kwargs):
        name = Path(str(file)).name
        opened.append(name)
        if name in {"T.npy", "angles.npy"} or "angle" in name.lower():
            raise AssertionError(f"empty-side identity path must not read {name}")
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", poisoned_load)
    e0 = train_mod.empty_side_e0(encoder, tmp_path / "calib_activity.npy")
    monkeypatch.undo()
    assert e0.shape == (96, 50)
    assert opened == ["calib_activity.npy"]


@REAL_ENCODER
def test_transform_bank_empty_side_uses_recomputed_e0_and_zeroes_carrier() -> None:
    encoder, _ = train_mod.frozen_empty_side_encoder()
    e0 = train_mod.empty_side_e0(encoder, ACTIVITY)
    bank = _bank()
    assert bank.E0.shape == e0.shape == (96, 50)
    e0_before, carrier_before = bank.E0.copy(), bank.carrier.copy()
    meta_before = dict(bank.calibration_meta)
    transformed = train_mod.transform_bank(bank, train_mod.EMPTY_SIDE_IDENTITY, e0=e0)
    assert transformed is not bank
    assert np.all(transformed.carrier == 0.0)
    assert transformed.carrier.shape == bank.carrier.shape == (96, 4)
    np.testing.assert_array_equal(transformed.E0, e0)
    assert transformed.E0 is not bank.E0
    assert transformed.E0 is not e0
    # original bank untouched
    np.testing.assert_array_equal(bank.E0, e0_before)
    np.testing.assert_array_equal(bank.carrier, carrier_before)
    assert bank.calibration_meta == meta_before
    record = transformed.calibration_meta["identity_ablation"]
    assert record["identity"] == "activity_only_empty_side"
    assert record["input_e0_sha256"] == bank.calibration_meta["array_sha256"]
    assert record["empty_side"]["side"] == "zeros(N,8): no MOVE-T4, no contrast"
    assert record["empty_side"]["e0_sha256"] == array_sha256(e0)
    assert record["empty_side"]["e0_sha256_differs_from_cached"] is (
        array_sha256(e0) != bank.calibration_meta["array_sha256"]
    )


def test_transform_bank_empty_side_fail_closed() -> None:
    bank = _bank()
    with pytest.raises(ValueError, match="requires the recomputed zero-side e0"):
        train_mod.transform_bank(bank, train_mod.EMPTY_SIDE_IDENTITY)
    with pytest.raises(RuntimeError, match="shape/finiteness drift"):
        train_mod.transform_bank(bank, train_mod.EMPTY_SIDE_IDENTITY, e0=np.zeros((96, 49), dtype=np.float32))
    with pytest.raises(RuntimeError, match="shape/finiteness drift"):
        train_mod.transform_bank(bank, train_mod.EMPTY_SIDE_IDENTITY, e0=np.full((96, 50), np.nan, dtype=np.float32))


def test_identity_ablation_meta_definition_mentions_zero_side() -> None:
    banks = {"source_train": {"s": _bank()}}
    meta = train_mod.identity_ablation_meta(train_mod.EMPTY_SIDE_IDENTITY, banks, empty_side_block={"side": train_mod.EMPTY_SIDE_SEMANTICS})
    assert "zeros(N,8)" in meta["definition"]
    assert meta["empty_side"]["side"] == train_mod.EMPTY_SIDE_SEMANTICS
    legacy = train_mod.identity_ablation_meta("activity_only", banks)
    assert legacy["definition"].startswith("activity_only: E0 unchanged")


@REAL_ENCODER
def test_score_side_empty_side_block_validation_fails_closed() -> None:
    _, encoder_record = train_mod.frozen_empty_side_encoder()
    good = {
        "identity_ablation": {
            "identity": train_mod.EMPTY_SIDE_IDENTITY,
            "empty_side": {
                "side": train_mod.EMPTY_SIDE_SEMANTICS,
                "side_dim": 8,
                "encoder": encoder_record,
            },
        }
    }
    block = score_mod.empty_side_block_from_meta(good)
    assert block["side_dim"] == 8
    # tampered side semantics -> refuse
    bad_side = {"identity_ablation": {"empty_side": {**good["identity_ablation"]["empty_side"], "side": "[MOVE-T4, 0]"}}}
    with pytest.raises(RuntimeError, match="side semantics drift"):
        score_mod.empty_side_block_from_meta(bad_side)
    # missing encoder identity -> refuse
    bad_encoder = {"identity_ablation": {"empty_side": {"side": train_mod.EMPTY_SIDE_SEMANTICS, "side_dim": 8, "encoder": {}}}}
    with pytest.raises(RuntimeError, match="encoder identity missing"):
        score_mod.empty_side_block_from_meta(bad_encoder)
    # a training run sealed with a DIFFERENT encoder must be rejected before
    # any bank/support file is touched (banks deliberately empty)
    tampered = {
        "identity_ablation": {
            "identity": train_mod.EMPTY_SIDE_IDENTITY,
            "empty_side": {
                "side": train_mod.EMPTY_SIDE_SEMANTICS,
                "side_dim": 8,
                "encoder": {**encoder_record, "head_state_sha256": "0" * 64},
            },
        }
    }
    with pytest.raises(RuntimeError, match="encoder identity differs"):
        score_mod.transform_query_banks({}, train_mod.EMPTY_SIDE_IDENTITY, tampered, score_mod.EXT6_RAW_M33)
