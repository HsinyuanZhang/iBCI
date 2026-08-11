"""Fail-closed contracts for the clean RT nested-LOSO repair."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

import src.data.rt_nested_loso_datamodule as nested_data
import src.rt_clean_nested_loso_eval as outer_eval
from src.data.rt_nested_loso_datamodule import (
    RtNestedLossoDataModule,
    build_outer_target_dataset,
    nested_loso_partition,
)
from src.rt_clean_nested_loso_eval import _r2_variance_weighted
from scripts.run_rt_clean_nested_loso import _train_command
from src.callbacks.rt_nested_selection_receipt import RtNestedSelectionReceipt
from lightning.pytorch.callbacks import ModelCheckpoint


SESSION_NAMES = [
    "ses-RT-20131009",
    "ses-RT-20131010",
    "ses-RT-20131011",
    "ses-RT-20131028",
    "ses-RT-20131029",
    "ses-RT-20131209",
    "ses-RT-20131210",
    "ses-RT-20131212",
    "ses-RT-20131213",
    "ses-RT-20131217",
    "ses-RT-20131218",
    "ses-RT-20150316",
    "ses-RT-20150317",
    "ses-RT-20150318",
    "ses-RT-20150320",
]


def _path_for(name: str) -> Path:
    return Path(f"sub-C_{name}_behavior+ecephys.nwb")


def _fake_rt_session(name: str, *, units: int = 3) -> dict:
    """Small deterministic RT payload with an M24 post-support query."""

    trials = 32
    bins_per_trial = 20
    total = trials * bins_per_trial
    rng = np.random.RandomState(abs(hash(name)) % (2**32))
    neural = rng.poisson(0.2, size=(total, units)).astype(np.float32)
    covariates = np.zeros((total, 2), dtype=np.float32)
    trial_change = np.zeros(total, dtype=bool)
    for trial in range(trials):
        left = trial * bins_per_trial
        trial_change[left] = True
        # Nonzero query behavior keeps every post-M24 window eligible.
        covariates[left : left + bins_per_trial] = np.asarray(
            [0.4 + trial / 100.0, -0.2 + (trial % 5) / 20.0], dtype=np.float32
        )
    trial_records = [
        {
            "trial_index": trial,
            "complete_cue": True,
            "accepted_segments": 1,
            "declared_segments": 1,
            "excluded_segments": 0,
            "exclusion_reason": None,
            "segment_exclusion_reasons": {},
        }
        for trial in range(trials)
    ]
    return {
        "session_name": name,
        "neural": neural,
        "covariates": covariates,
        "trial_change": trial_change,
        "eval_mask": np.ones(total, dtype=bool),
        "k4_segment_id": np.repeat(np.arange(trials, dtype=np.int64), bins_per_trial),
        "rt_segment_audit": {
            "complete_cue_trials": trials,
            "accepted_reach_segments": trials,
            "event_qualified_bins": total,
            "trial_records": trial_records,
        },
        "rt_velocity_audit": {"loader_standardization": "none"},
    }


def _patch_fake_loader(monkeypatch: pytest.MonkeyPatch, *, calls: list[str]):
    paths = [_path_for(name) for name in SESSION_NAMES]

    def fake_find(_data_dir):
        return paths

    def fake_load(path):
        name = nested_data.session_name_from_path(path)
        calls.append(name)
        return _fake_rt_session(name)

    monkeypatch.setattr(nested_data, "find_rt_sessions", fake_find)
    monkeypatch.setattr(nested_data, "load_rt_session", fake_load)


def test_nested_partition_is_sorted_cyclic_next_source_and_disjoint():
    for fold, outer in enumerate(sorted(SESSION_NAMES)):
        split = nested_loso_partition(SESSION_NAMES, fold, expected_session_count=15)
        assert split.outer_target_session == outer
        assert split.inner_validation_session == sorted(SESSION_NAMES)[(fold + 1) % 15]
        assert len(split.outer_source_sessions) == 14
        assert len(split.inner_train_sessions) == 13
        assert set(split.inner_train_sessions).isdisjoint(
            {split.outer_target_session, split.inner_validation_session}
        )
        assert set(split.inner_train_sessions) | {split.inner_validation_session} == set(
            split.outer_source_sessions
        )


def test_nested_partition_rejects_alias_or_cardinality_errors():
    with pytest.raises(ValueError, match="expects 15"):
        nested_loso_partition(SESSION_NAMES[:-1], 0, expected_session_count=15)
    with pytest.raises(ValueError, match="outer_loso_fold"):
        nested_loso_partition(SESSION_NAMES, 15, expected_session_count=15)


def test_fit_never_opens_outer_target_and_test_loader_is_fail_closed(monkeypatch):
    loaded: list[str] = []
    _patch_fake_loader(monkeypatch, calls=loaded)
    dm = RtNestedLossoDataModule(
        data_dir="/synthetic/rt",
        batch_size=2,
        session_window_budget=4,
        expected_session_count=15,
        outer_loso_fold=0,
        loso_fold=0,
        side_feature_group="zero4",
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("fit")
    split = dm.split
    assert split.outer_target_session == SESSION_NAMES[0]
    assert split.inner_validation_session == SESSION_NAMES[1]
    assert len(dm.train_session_names) == 13
    assert set(loaded) == set(SESSION_NAMES[1:])
    assert split.outer_target_session not in loaded
    assert dm.outer_target_loaded is False
    assert dm.outer_target_query_labels_read is False
    manifest = dm.get_split_manifest()
    assert manifest["protocol"]["split"] == "development_clean_nested_outer_LOSO"
    assert "joint RT-source retraining" in manifest["protocol"]["decoder_training"]
    assert "afc4_b4" in manifest["protocol"]["arms"]
    assert "afc4_w4" in manifest["protocol"]["arms"]
    assert "afc4_vel minus afc4_b4" in manifest["gates"]["content"]
    assert "afc4_vel minus zero4" in manifest["gates"]["system"]
    assert manifest["nested_selection"]["outer_target_loaded_during_fit"] is False
    assert manifest["target_query_window_audit"] is None
    assert manifest["nested_selection"]["inner_validation_only_for_checkpoint_selection"] is True
    assert manifest["nested_selection"]["checkpoint_metric_scope"] == "inner_validation_session_only"
    assert manifest["carrier_transform_fit_sessions"] == []
    with pytest.raises(RuntimeError, match="no fit-time test loader"):
        dm.test_dataloader()


def test_b2_no_carrier_mode_has_the_same_clean_split_and_a_four_item_batch(monkeypatch):
    """Stage-R B2 must reuse the nested split without manufacturing Zero4 rows."""

    loaded: list[str] = []
    _patch_fake_loader(monkeypatch, calls=loaded)
    dm = RtNestedLossoDataModule(
        data_dir="/synthetic/rt",
        batch_size=2,
        session_window_budget=4,
        expected_session_count=15,
        outer_loso_fold=2,
        loso_fold=2,
        side_feature_group="none",
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("fit")
    assert dm.split.outer_target_session == SESSION_NAMES[2]
    assert dm.split.outer_target_session not in loaded
    assert len(dm.train_dataset[0]) == 4
    manifest = dm.get_split_manifest()
    assert manifest["requested_side_feature_group"] == "none"
    assert manifest["arm"]["canonical_arm"] == "none"
    assert manifest["source_only_normalizer"] is None
    assert manifest["carrier_transform_fit_sessions"] == []


def test_k4_normalizer_is_fit_from_exactly_13_inner_train_sessions(monkeypatch):
    loaded: list[str] = []
    _patch_fake_loader(monkeypatch, calls=loaded)
    fit_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def fake_fit(features, session_names):
        fit_calls.append((tuple(features), tuple(session_names)))
        return np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)

    monkeypatch.setattr(nested_data, "fit_train_k4_stats", fake_fit)
    dm = RtNestedLossoDataModule(
        data_dir="/synthetic/rt",
        batch_size=2,
        session_window_budget=4,
        expected_session_count=15,
        outer_loso_fold=3,
        loso_fold=3,
        side_feature_group="afc4_vel",
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("fit")
    assert len(fit_calls) == 1
    feature_names, fit_names = fit_calls[0]
    assert feature_names == fit_names == tuple(dm.split.inner_train_sessions)
    assert len(fit_names) == 13
    assert dm.split.outer_target_session not in fit_names
    assert dm.split.inner_validation_session not in fit_names
    normalizer = dm.native_k4_normalization
    assert normalizer is not None
    assert normalizer["fit_scope"] == "inner_train_sessions_only"
    assert normalizer["fit_sessions"] == list(dm.split.inner_train_sessions)
    assert normalizer["excluded_outer_target_session"] == dm.split.outer_target_session
    assert set(loaded) == set(dm.split.outer_source_sessions)


def test_outer_target_builder_is_the_only_explicit_target_open(monkeypatch):
    loaded: list[str] = []
    _patch_fake_loader(monkeypatch, calls=loaded)
    target_dataset, split, target_path = build_outer_target_dataset(
        data_dir="/synthetic/rt",
        outer_loso_fold=0,
        side_feature_group="zero4",
        side_feature_shuffle_seed=42,
        calibration_n_trials=24,
        query_start_trial=24,
        window_size=50,
        max_trial_length=100,
        expected_session_count=15,
    )
    assert split.outer_target_session == SESSION_NAMES[0]
    assert target_path.name == _path_for(SESSION_NAMES[0]).name
    assert loaded == [SESSION_NAMES[0]]
    assert len(target_dataset) > 0


def test_outer_r2_is_finite_and_fail_closed_for_zero_variance():
    target = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    assert _r2_variance_weighted(target, target) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero or non-finite variance"):
        _r2_variance_weighted(np.zeros_like(target), np.ones_like(target))


def test_runner_requires_explicit_arm_and_binds_each_arm_identity():
    arms = ("afc4_vel", "zero4", "afc4_rs", "afc4_ls", "afc4_b4", "afc4_w4")
    for arm in arms:
        command = _train_command(
            argparse.Namespace(fold=0, arm=arm, seed=42, accelerator="cpu", devices=1)
        )
        assert f"data.side_feature_group={arm}" in command
        assert f"run_id=rt_clean_nested_loso_m24_{arm}" in command
        assert "test=false" in command
    with pytest.raises(ValueError, match="Unsupported RT clean arm"):
        _train_command(argparse.Namespace(fold=0, arm="implicit", seed=42, accelerator="cpu", devices=1))
    with pytest.raises(ValueError, match="explicit side_feature_group"):
        RtNestedLossoDataModule(data_dir="/synthetic/rt", outer_loso_fold=0)


def _write_selection_fixture(tmp_path: Path, *, valid: bool = True):
    run_dir = tmp_path / "fit-run"
    checkpoint = run_dir / "checkpoints" / "best_ckpt" / "epoch_003.ckpt"
    config = run_dir / ".hydra" / "config.yaml"
    split_manifest = run_dir / "split_manifest.json"
    receipt = run_dir / "rt_nested_selection_receipt.json"
    checkpoint.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    config.write_text(
        "run_id: rt_clean_nested_loso_m24_zero4\n"
        "seed: 42\n"
        "data:\n"
        "  _target_: src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule\n"
        "  data_dir: /synthetic/rt\n"
        "  batch_size: 2\n"
        "  side_feature_group: zero4\n"
        "  side_feature_shuffle_seed: 42\n"
        "  outer_loso_fold: 0\n"
        "  expected_session_count: 15\n"
        "  calibration_n_trials: 24\n"
        "  query_start_trial: 24\n"
        "  window_size: 50\n"
        "  max_trial_length: 100\n"
        "  interpolate_trials: true\n"
        "  interpolate_trials_kind: cubic\n"
        "  pad_value: -1.0\n"
        "model:\n"
        "  _target_: fake.TinyModel\n",
        encoding="utf-8",
    )
    sessions = sorted(SESSION_NAMES)
    split = nested_loso_partition(sessions, 0, expected_session_count=15)
    manifest = {
        "validation_protocol": "nested_loso",
        "requested_side_feature_group": "zero4",
        "outer_loso_fold": 0,
        "loso_fold": 0,
        "session_names": sessions,
        "all_sessions": sessions,
        "source_sessions": list(split.outer_source_sessions),
        "inner_train_sessions": list(split.inner_train_sessions),
        "inner_validation_session": split.inner_validation_session,
        "target_session": split.outer_target_session,
        "source_only_normalizer": None,
        "nested_selection": {
            "clean": True,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True,
            "checkpoint_metric": "val_heldin/r2_mean",
            "checkpoint_metric_scope": "inner_validation_session_only",
        },
    }
    split_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "epoch": 3,
            "global_step": 7,
            "state_dict": {"weight": torch.tensor([1.0])},
            "callbacks": {
                "ModelCheckpoint": {
                    "monitor": "val_heldin/r2_mean",
                    "best_model_path": str(checkpoint.resolve()),
                    "best_model_score": 0.75,
                }
            },
        },
        checkpoint,
    )
    def sha(path: Path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    selection = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "selection_receipt_path": str(receipt.resolve()),
        "run_id": "rt_clean_nested_loso_m24_zero4",
        "run_dir": str(run_dir.resolve()),
        "arm": "zero4",
        "outer_loso_fold": 0,
        "seed": 42,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
        "selected_metric_value": 0.75,
        "selected_epoch": 3,
        "selected_global_step": 7,
        "best_model_path": str(checkpoint.resolve()),
        "best_model_sha256": sha(checkpoint),
        "config_path": str(config.resolve()),
        "config_sha256": sha(config),
        "split_manifest_path": str(split_manifest.resolve()),
        "split_manifest_sha256": sha(split_manifest),
        "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False,
    }
    if not valid:
        selection["best_model_sha256"] = "bad"
    receipt.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config, checkpoint, split_manifest, receipt, manifest, split


def test_outer_evaluator_rejects_bad_selection_before_target_open(monkeypatch, tmp_path):
    config, checkpoint, split_manifest, receipt, manifest, _ = _write_selection_fixture(tmp_path, valid=False)
    called = []
    monkeypatch.setattr(
        outer_eval,
        "build_outer_target_dataset",
        lambda **kwargs: called.append(kwargs) or (_ for _ in ()).throw(AssertionError("target opened")),
    )
    with pytest.raises(ValueError, match="checkpoint bytes"):
        outer_eval.evaluate_outer_target(
            config_path=config,
            checkpoint_path=checkpoint,
            split_manifest_path=split_manifest,
            selection_receipt_path=receipt,
            output_path=tmp_path / "out.json",
        )
    assert called == []


def test_outer_evaluator_rejects_output_reuse_or_data_override_before_target_open(monkeypatch, tmp_path):
    config, checkpoint, split_manifest, receipt, manifest, _ = _write_selection_fixture(tmp_path)
    called = []
    monkeypatch.setattr(
        outer_eval,
        "build_outer_target_dataset",
        lambda **kwargs: called.append(kwargs) or (_ for _ in ()).throw(AssertionError("target opened")),
    )
    existing = tmp_path / "already.json"
    existing.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite outer-evaluation"):
        outer_eval.evaluate_outer_target(
            config_path=config,
            checkpoint_path=checkpoint,
            split_manifest_path=split_manifest,
            selection_receipt_path=receipt,
            output_path=existing,
        )
    with pytest.raises(ValueError, match="data-dir override is disabled"):
        outer_eval.evaluate_outer_target(
            config_path=config,
            checkpoint_path=checkpoint,
            split_manifest_path=split_manifest,
            selection_receipt_path=receipt,
            output_path=tmp_path / "new.json",
            data_dir="/unbound/override",
        )
    assert called == []


def test_outer_evaluator_rejects_unbound_split_manifest_path(monkeypatch, tmp_path):
    config, checkpoint, split_manifest, receipt, manifest, _ = _write_selection_fixture(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["split_manifest_path"] = str((tmp_path / "different_manifest.json").resolve())
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(
        outer_eval,
        "build_outer_target_dataset",
        lambda **kwargs: called.append(kwargs) or (_ for _ in ()).throw(AssertionError("target opened")),
    )
    with pytest.raises(ValueError, match="split manifest path"):
        outer_eval.evaluate_outer_target(
            config_path=config,
            checkpoint_path=checkpoint,
            split_manifest_path=split_manifest,
            selection_receipt_path=receipt,
            output_path=tmp_path / "new.json",
        )
    assert called == []


def test_outer_evaluator_one_shot_state_identity_and_receipt(monkeypatch, tmp_path):
    config, checkpoint, split_manifest, receipt, manifest, split = _write_selection_fixture(tmp_path)

    class TinyDataset:
        def __init__(self):
            self.window_indices = [
                (split.outer_target_session, 0),
                (split.outer_target_session, 1),
                (split.outer_target_session, 2),
                (split.outer_target_session, 3),
            ]
            self.query_window_audit = {"mock": True, "matched_to_clean_rt_contract": True}
        def __len__(self):
            return len(self.window_indices)
        def __getitem__(self, index):
            target = torch.tensor([[float(index), float(index + 1)]], dtype=torch.float32)
            return (
                torch.zeros((1, 2), dtype=torch.float32),
                target,
                torch.zeros((24, 1, 2), dtype=torch.float32),
                split.outer_target_session,
                torch.zeros((2, 4), dtype=torch.float32),
            )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))
        def setup(self, stage):
            return None
        def model_step(self, batch):
            target = batch[1][:, -1:, :]
            return {"behavior_pred": target * 0.5, "behavior_target": target}

    monkeypatch.setattr(outer_eval.hydra.utils, "instantiate", lambda cfg: TinyModel())
    monkeypatch.setattr(
        outer_eval,
        "build_outer_target_dataset",
        lambda **kwargs: (TinyDataset(), split, checkpoint.parent.parent.parent / "target.nwb"),
    )
    result = outer_eval.evaluate_outer_target(
        config_path=config,
        checkpoint_path=checkpoint,
        split_manifest_path=split_manifest,
        selection_receipt_path=receipt,
        output_path=tmp_path / "outer.json",
    )
    assert result["status"] == "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP"
    assert result["arm"] == "zero4"
    assert result["seed"] == 42
    assert result["target_support_calibration_velocity_used"] is False
    assert result["target_query_labels_used_for_scoring_only"] is True
    assert result["target_query_labels_used_for_checkpoint_selection"] is False
    assert result["model_state_unchanged"] is True
    assert result["model_state_three_point_unchanged"] is True
    assert result["model_state_sha256_before"] == result["model_state_sha256_after_target_carrier"] == result["model_state_sha256_after"]
    assert result["matched_query_window_identity"]["mock"] is True
    assert (tmp_path / "outer.json").is_file()


def test_fit_selection_callback_writes_exact_checkpoint_receipt(tmp_path):
    config, checkpoint, split_manifest, receipt_path, manifest, split = _write_selection_fixture(tmp_path)
    # The fixture's prewritten receipt is removed so the callback itself owns
    # the write path in this test.
    receipt_path.unlink()
    split_manifest.unlink()
    selected = ModelCheckpoint(monitor="val_heldin/r2_mean", mode="max", every_n_epochs=1)
    selected.best_model_path = str(checkpoint.resolve())
    selected.best_model_score = torch.tensor(0.75)

    class FakeDataModule:
        def get_split_manifest(self):
            return manifest

    class FakeTrainer:
        callbacks = [selected]
        datamodule = FakeDataModule()
        is_global_zero = True

    callback = RtNestedSelectionReceipt(
        output_path=str(receipt_path),
        split_manifest_path=str(split_manifest),
        config_path=str(config),
        run_id="rt_clean_nested_loso_m24_zero4",
        arm="zero4",
        outer_loso_fold=0,
        seed=42,
    )
    callback.on_fit_end(FakeTrainer(), None)
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["status"] == "PASS_FIT_INNER_SELECTION_ONLY"
    assert written["best_model_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert written["selected_epoch"] == 3
    assert written["selected_global_step"] == 7
