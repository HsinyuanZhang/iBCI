"""Contract tests for source-only separately-trained H1 CarrierID RS/LS."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import h1_carrierid_shuffle as shuffle
from src.data.h1_carrierid_shuffle import H1CarrierIdShuffleDataModule


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB = ROOT.parent / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"


def _dm(tmp_path: Path, arm: str) -> H1CarrierIdShuffleDataModule:
    return H1CarrierIdShuffleDataModule(task="h1", data_dir=str(ROOT / "data/000954"),
        raw_receipt_path=str(RAW), eb_receipt_path=str(EB), cache_dir=str(tmp_path / arm),
        carrier_intervention=arm)


def test_label_refits_are_precomputed_once_not_repeated_by_getitem(tmp_path, monkeypatch):
    calls = 0
    original = shuffle.label_rotation_carrier

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(shuffle, "label_rotation_carrier", counted)
    dm = _dm(tmp_path, "ls"); dm.setup("fit")
    assert calls == 116
    request = (0, int(dm.train_batch_sampler.schedule[0, 0]))
    for _ in range(8):
        assert dm.train_dataset[request][4].shape[1] == 4
    assert calls == 116


def test_all_entry_controls_are_deterministic_and_nonidentity_without_target_access(tmp_path, monkeypatch):
    import src.data.h1_m4_eb_pilot as pilot

    def forbidden_target(*args, **kwargs):
        raise AssertionError("target loader must never be called by source-only RS/LS setup")

    monkeypatch.setattr(pilot, "load_target_records", forbidden_target)
    rs_a, ls_a = _dm(tmp_path / "a", "rs"), _dm(tmp_path / "a", "ls")
    rs_a.setup("fit"); ls_a.setup("fit")
    rs_b, ls_b = _dm(tmp_path / "b", "rs"), _dm(tmp_path / "b", "ls")
    rs_b.setup("fit"); ls_b.setup("fit")
    for first, second in ((rs_a, rs_b), (ls_a, ls_b)):
        ds = first.train_dataset
        assert ds.effective_source_carriers_count == 116
        assert ds.effective_source_carriers_shape == [116, 176, 4]
        assert ds.effective_source_carriers_nonidentity_all is True
        assert ds.effective_source_carriers_sha256 == second.train_dataset.effective_source_carriers_sha256
        assert first.pilot_manifest()["target_nwb_opened_during_training_setup"] is False
        assert first.pilot_manifest()["minival_or_heldout_enumerated"] is False
    assert rs_a.pilot_manifest()["carrier_cache_sha256"] == ls_a.pilot_manifest()["carrier_cache_sha256"]
    assert rs_a.pilot_manifest()["effective_source_carriers_sha256"] != ls_a.pilot_manifest()["effective_source_carriers_sha256"]
    request = (0, int(rs_a.train_batch_sampler.schedule[0, 0]))
    assert not np.array_equal(rs_a.train_dataset[request][4], ls_a.train_dataset[request][4])


def test_no_target_loader_symbol_is_imported_by_shuffle_module():
    text = (ROOT / "src/data/h1_carrierid_shuffle.py").read_text(encoding="utf-8")
    assert "load_target_records" not in text
    assert "H1M4EBStrictTargetDataset" not in text
