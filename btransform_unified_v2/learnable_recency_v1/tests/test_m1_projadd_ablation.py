"""M1 muscle proj_add activity_only ablation unit tests: carrier zeroing, E0
bitwise unchanged, no bank mutation, HO material same transform, required
--identity, muscle carrier pack/reference bindings."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from btransform_unified_v1.bank import array_sha256, make_synthetic_bank

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import m1_projadd_ablation_train as train_mod  # noqa: E402


def _bank(seed: int = 5):
    return make_synthetic_bank("m1", n_windows=4, seed=seed)


def test_script_constants_and_schemas() -> None:
    assert train_mod.IDENTITY == "activity_only"
    assert train_mod.ABLATION_IDENTITIES == ("activity_only",)
    assert train_mod.CARRIER_SHAPE == (64, 4)
    assert train_mod.CARRIER_VARIANT == "muscle_response16_svd4/global_rms"
    assert train_mod.PROJ_DIM == 16
    assert train_mod.SEED == 42 and train_mod.CONTEXT == 100 and train_mod.EPOCHS == 24
    assert train_mod.TRAIN_SCHEMA == "m1_rift_projadd_ablation_train_v1"
    assert train_mod.CHECKPOINT_SCHEMA == "m1_rift_projadd_ablation_epoch_checkpoint_v1"
    assert train_mod.PROGRESS_SCHEMA == "m1_rift_projadd_ablation_score_progress_v1"
    assert train_mod.SCORE_SCHEMA == "m1_rift_projadd_ablation_ho_calib_epoch_scan_v1"
    assert train_mod.DEFAULT_PACK.name == "carrier_pack.npz"
    assert train_mod.DEFAULT_PACK.parent.name == "carrier_official4"
    assert train_mod.DEFAULT_REFERENCE.name == "formal_s42_gpu1"
    assert train_mod.REFERENCE_ARM == "D_JOINT"


def test_identity_flag_is_required() -> None:
    # missing --identity must be an argparse error
    with pytest.raises(SystemExit):
        train_mod.build_parser().parse_args([])
    # non-activity_only identities are not choices for this runner
    with pytest.raises(SystemExit):
        train_mod.build_parser().parse_args(["--identity", "full"])
    args = train_mod.build_parser().parse_args(["--identity", "activity_only"])
    assert args.identity == "activity_only"


def test_frozen_recipe_accepts_defaults_and_rejects_drift() -> None:
    args = train_mod.build_parser().parse_args(["--identity", "activity_only"])
    assert args.tier == "learned_slope" and args.ladder == "default" and args.layers == 4
    train_mod.ensure_frozen_recipe(args)
    for extra in (["--tier", "cable"], ["--ladder", "scaled"], ["--layers", "3"],
                  ["--lr-multiplier", "2.0"], ["--cable-nw"], ["--learn-flat-heads"], ["--no-per-layer"],
                  ["--half-lives", "0.1,0.2"]):
        with pytest.raises(ValueError):
            train_mod.ensure_frozen_recipe(train_mod.build_parser().parse_args(["--identity", "activity_only", *extra]))


def test_transform_bank_zeroes_carrier_and_keeps_e0_bitwise() -> None:
    bank = _bank()
    e0_before = bank.E0.copy()
    carrier_before = bank.carrier.copy()
    meta_before = dict(bank.calibration_meta)
    assert np.any(carrier_before != 0.0)  # synthetic carrier is nonzero
    transformed = train_mod.transform_bank(bank, "activity_only")
    assert transformed is not bank
    assert transformed.carrier.shape == (64, 4)
    assert transformed.carrier.dtype == np.float32
    assert np.all(transformed.carrier == 0.0)
    assert np.array_equal(transformed.E0, e0_before)  # bitwise E0 preservation
    assert transformed.E0 is not bank.E0
    assert transformed.carrier is not bank.carrier
    # original bank is untouched (arrays and meta)
    assert np.array_equal(bank.E0, e0_before)
    assert np.array_equal(bank.carrier, carrier_before)
    assert "identity_ablation" not in bank.calibration_meta
    assert bank.calibration_meta == meta_before
    record = transformed.calibration_meta["identity_ablation"]
    assert record["identity"] == "activity_only"
    assert record["e0_bitwise_unchanged"] is True
    assert record["input_e0_sha256"] == array_sha256(e0_before)
    assert record["transformed_e0_sha256"] == array_sha256(e0_before)
    assert record["input_carrier_sha256"] == array_sha256(carrier_before)
    assert record["transformed_carrier_sha256"] != record["input_carrier_sha256"]
    # new bank meta describes its own (zeroed) carrier
    assert transformed.calibration_meta["carrier_sha256"] == record["transformed_carrier_sha256"]
    assert transformed.unit_mask.shape == bank.unit_mask.shape
    assert np.array_equal(transformed.unit_mask, bank.unit_mask)


def test_transform_bank_rejects_other_identities_and_bad_carrier_shape() -> None:
    bank = _bank()
    for bad in ("full", "norm_only", "activity_only_empty_side", ""):
        with pytest.raises(ValueError):
            train_mod.transform_bank(bank, bad)
    wrong_shape = make_synthetic_bank("m2", n_windows=2, seed=1)  # (96, 4) carrier
    with pytest.raises(RuntimeError):
        train_mod.transform_bank(wrong_shape, "activity_only")


def test_transform_banks_record_and_originals_not_rewritten() -> None:
    banks = {f"ses-{i}": _bank(seed=i) for i in range(3)}
    digests = {s: (array_sha256(b.E0), array_sha256(b.carrier)) for s, b in banks.items()}
    out, rows = train_mod.transform_banks(banks, "activity_only")
    assert set(out) == set(banks) == set(rows)
    for session, bank in banks.items():
        assert (array_sha256(bank.E0), array_sha256(bank.carrier)) == digests[session]
        assert out[session].E0 is not bank.E0 and out[session].carrier is not bank.carrier
        assert np.all(out[session].carrier == 0.0)
        row = rows[session]
        assert row["carrier_all_zero"] is True
        assert row["e0_sha256_unchanged"] is True
        assert row["identity_arrays_not_shared"] is True
        assert row["input_e0_sha256"] == digests[session][0]
        assert row["input_carrier_sha256"] == digests[session][1]
        assert row["transformed_e0_sha256"] == row["input_e0_sha256"]
        assert row["transformed_carrier_sha256"] != row["input_carrier_sha256"]
    block = train_mod.identity_ablation_block("activity_only", rows)
    assert block["identity"] == "activity_only"
    assert block["carrier_shape"] == [64, 4]
    assert block["carrier_variant"] == "muscle_response16_svd4/global_rms"
    assert block["e0_sha256_unchanged"] is True
    assert block["carrier_zeroed"] is True
    assert block["identity_arrays_not_shared"] is True
    assert set(block["source_sessions"]) == set(banks)
    assert "heldout_sessions" not in block


def _material(bank):
    return {
        "20121004": {
            "dataset": object(),
            "bank": bank,
            "calib10": np.zeros((10, 100, 64), dtype=np.float32),
            "starts": (97, 150),
            "body_sha256": "b" * 64,
            "carrier_sha256": array_sha256(bank.carrier),
            "e0_sha256": array_sha256(bank.E0),
            "starts_sha256": "s" * 64,
            "target_sha256": "t" * 64,
            "neural_sha256": "n" * 64,
            "covariate_sha256": "c" * 64,
            "calib10_sha256": "a" * 64,
            "window_count": 2,
            "carrier_meta": {"legacy": {"sealed": True}, "replacement": "muscle_pack_T_only"},
        }
    }


def test_transform_material_applies_the_same_transform() -> None:
    bank = _bank(seed=7)
    material = _material(bank)
    out, rows = train_mod.transform_material(material, "activity_only")
    item = out["20121004"]
    assert item is not material["20121004"]
    # same bank-level transform: fresh zeroed carrier, bitwise-identical E0
    assert item["bank"] is not bank
    assert np.all(item["bank"].carrier == 0.0)
    assert item["bank"].carrier.shape == (64, 4)
    assert np.array_equal(item["bank"].E0, bank.E0)
    assert item["bank"].E0 is not bank.E0 and item["bank"].carrier is not bank.carrier
    # digest fields follow the transformed bank
    assert item["e0_sha256"] == array_sha256(bank.E0)
    assert item["carrier_sha256"] == array_sha256(item["bank"].carrier)
    assert item["carrier_sha256"] != array_sha256(bank.carrier)
    # everything else carried over untouched (muscle material field set);
    # dataset/calib10 are read-only objects shared by identity with the input row
    for key in ("body_sha256", "starts_sha256", "target_sha256",
                "neural_sha256", "covariate_sha256", "calib10_sha256", "window_count", "carrier_meta"):
        assert item[key] == material["20121004"][key]
    assert item["dataset"] is material["20121004"]["dataset"]
    assert item["calib10"] is material["20121004"]["calib10"]
    assert item["starts"] == material["20121004"]["starts"]
    row = rows["20121004"]
    assert row["e0_sha256_unchanged"] is True
    assert row["carrier_all_zero"] is True
    assert row["identity_arrays_not_shared"] is True
    assert row["input_e0_sha256"] == array_sha256(bank.E0)
    # the frozen material bank itself is untouched
    assert np.any(bank.carrier != 0.0)
    assert "identity_ablation" not in bank.calibration_meta
    with pytest.raises(ValueError):
        train_mod.transform_material(material, "norm_only")


def test_identity_ablation_block_with_heldout_rows() -> None:
    source_rows = {"ses-a": {"e0_sha256_unchanged": True, "carrier_all_zero": True, "identity_arrays_not_shared": True}}
    ho_rows = {"20121004": {"e0_sha256_unchanged": True, "carrier_all_zero": True, "identity_arrays_not_shared": True}}
    block = train_mod.identity_ablation_block("activity_only", source_rows, ho_rows)
    assert block["heldout_sessions"] == ho_rows
    assert block["heldout_e0_sha256_unchanged"] is True
    assert block["heldout_carrier_zeroed"] is True
    assert "activity_only" in block["definition"] and "label-free" in block["definition"]


def test_default_dest_names() -> None:
    formal = train_mod.build_parser().parse_args(["--identity", "activity_only"])
    assert train_mod.default_dest(formal) == train_mod.RESULTS / "m1_projadd_activity_only_s42"
    smoke = train_mod.build_parser().parse_args(["--identity", "activity_only", "--max-updates-smoke", "1"])
    assert train_mod.default_dest(smoke) == train_mod.RESULTS / "smoke" / "m1_projadd_activity_only_s42"


@pytest.mark.skipif(not train_mod.DEFAULT_PACK.is_file(), reason="frozen official muscle carrier pack not present")
def test_carrier_binding_extends_frozen_receipt() -> None:
    carriers, binding = train_mod.carrier_binding(train_mod.DEFAULT_PACK)
    assert set(carriers) == set(train_mod.SOURCE) | set(train_mod.HO)
    for value in carriers.values():
        assert value.shape == (64, 4) and value.dtype == np.float32
    for key in ("carrier_pack_npz_sha256", "carrier_pack_receipt_sha256", "carrier_pack_receipt_body",
                "fit_sha256", "carrier_py_sha256", "train_py_sha256"):
        assert key in binding
    assert binding["carrier_variant"] == train_mod.CARRIER_VARIANT
    assert binding["identity_ablation"] == "activity_only"
    assert len(binding["binding_sha256"]) == 64
