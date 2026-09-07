from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.falcon_m1_source_only_decoder_datamodule import (  # noqa: E402
    FOLD0_SOURCES, FOLD1_SOURCES, FOLD2_SOURCES, M1_SOURCE_ONLY_FOLDS, M1SourceOnlyDecoderFold0DataModule,
)


def test_fold0_source_list_is_exact_and_target_free() -> None:
    assert FOLD0_SOURCES == ("ses-20120926", "ses-20120927", "ses-20120928")
    instance = object.__new__(M1SourceOnlyDecoderFold0DataModule)
    instance.outer_fold = 0
    instance.outer_left_out = "ses-20120924"
    instance.source_session_names = FOLD0_SOURCES
    manifest = instance.get_split_manifest()
    assert manifest["outer_left_out"] == "ses-20120924"
    assert manifest["train_sessions"] == list(FOLD0_SOURCES)
    assert manifest["validation_sessions"] == []
    assert manifest["minival_opened"] is False
    assert manifest["heldout_opened"] is False


def test_rejects_any_noncanonical_fold0_source_list() -> None:
    with pytest.raises(ValueError, match="exactly"):
        M1SourceOnlyDecoderFold0DataModule(
            task="m1", data_dir=".", source_session_names=list(reversed(FOLD0_SOURCES)),
            random_calibration=False, include_heldout_in_fit=False,
            include_heldout_in_test=False, side_feature_group="none",
        )


def test_fold1_policy_is_exact_and_keeps_its_target_out_of_sources() -> None:
    assert M1_SOURCE_ONLY_FOLDS[1] == ("ses-20120926", FOLD1_SOURCES)
    instance = M1SourceOnlyDecoderFold0DataModule(
        task="m1", data_dir=".", loso_fold=1, source_session_names=list(FOLD1_SOURCES),
        random_calibration=False, include_heldout_in_fit=False,
        include_heldout_in_test=False, side_feature_group="none",
    )
    assert instance.outer_fold == 1
    assert instance.outer_left_out not in instance.source_session_names


def test_rejects_unknown_outer_fold_fail_closed() -> None:
    with pytest.raises(ValueError, match="no approved"):
        M1SourceOnlyDecoderFold0DataModule(
            task="m1", data_dir=".", loso_fold=3, source_session_names=list(FOLD0_SOURCES),
            random_calibration=False, include_heldout_in_fit=False,
            include_heldout_in_test=False, side_feature_group="none",
        )


def test_fold2_policy_is_exact_and_keeps_its_target_out_of_sources() -> None:
    assert M1_SOURCE_ONLY_FOLDS[2] == ("ses-20120927", FOLD2_SOURCES)
    instance = M1SourceOnlyDecoderFold0DataModule(
        task="m1", data_dir=".", loso_fold=2, source_session_names=list(FOLD2_SOURCES),
        random_calibration=False, include_heldout_in_fit=False,
        include_heldout_in_test=False, side_feature_group="none",
    )
    assert instance.outer_fold == 2
    assert instance.outer_left_out not in instance.source_session_names


def test_source_only_manifest_writer_emits_a_hash_receipt(tmp_path: Path) -> None:
    instance = object.__new__(M1SourceOnlyDecoderFold0DataModule)
    instance.get_split_manifest = lambda: {"source_only": True, "outer_left_out": "ses-20120924"}
    output = instance.write_source_only_manifest(tmp_path)
    assert output.name == "source_only_decoder_manifest.json"
    assert output.is_file()
    receipt = output.with_suffix(".sha256")
    assert receipt.is_file()
    assert output.name in receipt.read_text(encoding="utf-8")
