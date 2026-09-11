"""Contract tests for the raw, no-calibration M1 static data plane."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


PACKAGE = Path(__file__).resolve().parents[1]
MODULE = PACKAGE / "scripts" / "m1_static_data.py"
FORMAL_ROOT = PACKAGE / "results" / "m1_projadd_learned_slope_default_s42"
SPEC = importlib.util.spec_from_file_location("m1_static_data_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def test_padded_windows_and_coordinate_validity_are_explicit() -> None:
    raw = np.arange(5 * 64, dtype=np.float32).reshape(5, 64)
    windows = data.padded_windows(raw, np.asarray([0, 2], dtype=np.int64))
    assert windows.shape == (2, 100, 64)
    assert np.count_nonzero(windows[0, :99]) == 0
    np.testing.assert_array_equal(windows[0, 99], raw[0])
    np.testing.assert_array_equal(windows[1, 97:], raw[:3])
    item = {
        "X": np.pad(raw, ((99, 0), (0, 0))),
        "Y": raw[:, :16],
        "starts": np.asarray([0, 2]),
    }
    x, y, valid = data.batch_tensors(item, np.asarray([0, 1]), "cpu")
    assert x.shape == (2, 100, 64) and y.shape == (2, 16)
    assert valid.dtype == torch.bool
    assert valid.sum(dim=1).tolist() == [1, 3]


def test_target_rows_follow_exact_query_indices() -> None:
    raw = np.zeros((4, 64), dtype=np.uint8)
    target = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    mask = np.asarray([False, True, False, True])
    item = data._load_item("fake", Path(__file__), lambda _path: (raw, target, mask, mask))
    np.testing.assert_array_equal(item["starts"], np.asarray([1, 3]))
    np.testing.assert_array_equal(item["Y"], target[[1, 3]])


def test_module_has_no_calibration_or_identity_dependencies() -> None:
    source = MODULE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "falcondataset",
        "falcondatamodule",
        "default_identity_provider",
        "rsyn3",
        "make_pick_bank",
    ):
        assert forbidden not in source


def test_source_contract_matches_latest_formal_query_and_sampler_receipt() -> None:
    material = data.load_source()
    contract = material["contract"]
    receipt = json.loads((FORMAL_ROOT / "run_meta.json").read_text(encoding="utf-8"))["source_contract"]
    assert contract["total_windows"] == 213336
    assert contract["updates_per_epoch"] == 6665
    assert contract["used_windows_per_epoch"] == 213280
    assert contract["dropped_tail_windows"] == 56
    assert contract["dropped_tail_windows_by_session"] == {
        "ses-20120924": 1,
        "ses-20120926": 12,
        "ses-20120927": 12,
        "ses-20120928": 31,
    }
    assert contract["sampler_batch_sha256"] == receipt["sampler_batch_sha256"]
    for session, count in data.EXPECTED_SOURCE_WINDOWS.items():
        assert contract["per_session"][session]["window_count"] == count
        reference = receipt["query_hashes"][session]
        assert contract["per_session"][session]["window_count"] == reference["window_count"]
        assert contract["per_session"][session]["starts_sha256"] == reference["window_starts_sha256"]
        assert contract["per_session"][session]["eval_mask_sha256"] == reference["eval_mask_sha256"]


def test_heldout_contract_matches_latest_formal_score_receipt() -> None:
    material = data.load_heldout()
    receipt = json.loads(
        (FORMAL_ROOT / "score_receipt.json").read_text(encoding="utf-8")
    )["ho_contract"]
    assert material["contract"]["total_windows"] == 3881
    for session, expected in data.EXPECTED_HELDOUT.items():
        actual = material["contract"]["per_session"][session]
        for key, value in expected.items():
            assert actual[key] == value
            assert actual[key] == receipt["per_session"][session][key]
