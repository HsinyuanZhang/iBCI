"""Regression tests for the isolated M1 Version-B source scope.

These tests use synthetic in-memory records and fake NWB filenames.  They do
not open any experiment data and specifically guard against accidentally
calling the broad legacy ``FalconDataModule.setup`` path.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = REPO_ROOT / "streaming_calibration_exp"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from src.data import m1_version_b_source_loso_datamodule as version_b  # noqa: E402
from src.data.falcon_datamodule import FalconDataModule  # noqa: E402


EXPECTED_FILES = {
    "ses-20120924": "sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb",
    "ses-20120926": "sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb",
    "ses-20120927": "sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys.nwb",
    "ses-20120928": "sub-MonkeyL-held-in-calib_ses-20120928_behavior+ecephys.nwb",
}


def make_fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "000941"
    source = root / "sub-MonkeyL-held-in-calib"
    source.mkdir(parents=True)
    for filename in EXPECTED_FILES.values():
        (source / filename).write_bytes(b"synthetic placeholder; never opened as NWB")
    # A trap for accidental recursive discovery.  The Version-B path must not
    # enumerate or read this sibling directory.
    sibling = root / "sub-MonkeyL-held-in-minival"
    sibling.mkdir()
    (sibling / "forbidden.nwb").write_bytes(b"must remain untouched")
    return root


def fake_record(path: Path) -> dict[str, np.ndarray]:
    del path
    neurons = 3
    covariates = 2
    trials = 220
    bins_per_trial = 8
    total = trials * bins_per_trial
    neural = np.arange(total * neurons, dtype=np.float32).reshape(total, neurons) % 7
    cov = np.ones((total, covariates), dtype=np.float32)
    trial_change = np.zeros(total, dtype=bool)
    trial_change[::bins_per_trial] = True
    eval_mask = np.ones(total, dtype=bool)
    return {
        "neural": neural,
        "covariates": cov,
        "trial_change": trial_change,
        "eval_mask": eval_mask,
        "covariates_mean": np.zeros(covariates, dtype=np.float32),
        "covariates_std": np.ones(covariates, dtype=np.float32),
    }


def make_datamodule(root: Path, *, arm: str) -> version_b.M1VersionBSourceLOSODataModule:
    return version_b.M1VersionBSourceLOSODataModule(
        task="m1",
        data_dir=str(root),
        source_session_names=["ses-20120926", "ses-20120927", "ses-20120928"],
        heldin_session_names=["ses-20120926", "ses-20120927", "ses-20120928"],
        batch_size=8,
        window_size=4,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=16,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        calib_n_active_segments=1,
        interpolate_trials=False,
        interpolate_trials_kind="linear",
        pad_value=-1.0,
        validation_protocol="loso",
        loso_fold=0,
        rotation_id=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=10,
        heldin_query_end_trial=210,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=42,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        side_feature_shuffle_seed=42,
        afc4_arm=arm,
    )


class FakeSourcePlan:
    def __init__(self, source_paths, *, shuffle_seed):
        self.source_session_names = tuple(sorted(source_paths))
        self.target_fit_calls = {name: 1 for name in self.source_session_names}
        self.target_added = []
        self.shuffle_seed = shuffle_seed

    def add_target(self, path):
        self.target_added.append(Path(path))
        self.target_fit_calls["ses-20120924"] = 0
        return "ses-20120924"

    def receipt(self, *, arm):
        return {
            "arm": arm,
            "target_fit_calls": dict(self.target_fit_calls),
            "b4_mask": "post_normalization_coordinates_0_to_2_zero" if arm == "b4" else None,
            "rs4": "complete_normalized_row_permutation" if arm == "rs4" else None,
            "ls4": "M10 EMG-score rows deranged" if arm == "ls4" else None,
        }

    def normalized(self, session_name, *, arm):
        values = np.tile(np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), (3, 1))
        if arm == "b4":
            values[:, :3] = 0.0
        elif arm == "rs4":
            values = values[::-1].copy()
        elif arm == "ls4":
            values = -values
        return values


def test_source_paths_are_exact_and_nonrecursive(tmp_path):
    root = make_fake_root(tmp_path)
    dm = make_datamodule(root, arm="none")
    paths = dm._source_paths()
    assert tuple(paths) == (
        "ses-20120924",
        "ses-20120926",
        "ses-20120927",
        "ses-20120928",
    )
    assert all(path.parent.name == "sub-MonkeyL-held-in-calib" for path in paths.values())
    assert not any("minival" in str(path) for path in paths.values())


def test_extra_source_file_fails_closed(tmp_path):
    root = make_fake_root(tmp_path)
    extra = root / "sub-MonkeyL-held-in-calib" / "unexpected.nwb"
    extra.write_bytes(b"unexpected")
    dm = make_datamodule(root, arm="none")
    with pytest.raises((RuntimeError, ValueError)):
        dm._source_paths()


def test_setup_does_not_call_legacy_setup_and_zero4_does_not_add_target(tmp_path, monkeypatch):
    root = make_fake_root(tmp_path)
    dm = make_datamodule(root, arm="zero4")
    monkeypatch.setattr(
        FalconDataModule,
        "setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy setup called")),
    )
    monkeypatch.setattr(version_b, "SourceFrozenEMGAFC4Plan", FakeSourcePlan)
    dm.prepare_session_data = lambda path, task, **kwargs: fake_record(Path(path))
    dm.setup("fit")
    assert dm.val_heldout_dataset is None
    assert dm.val_heldin_session_names == ["ses-20120924"]
    assert dm.carrier_plan is not None
    assert dm.carrier_plan.target_added == []
    batch = dm.val_heldin_dataset[0]
    assert len(batch) == 5
    assert batch[-1].shape == (3, 4)
    assert np.array_equal(batch[-1], np.zeros((3, 4), dtype=np.float32))
    manifest = dm.get_split_manifest()
    assert manifest["minival_files_opened"] is False
    assert manifest["heldout_files_opened"] is False
    assert manifest["target_backpropagation"] is False
    assert len(manifest["train_sampler_sha256"]) == 64
    assert len(manifest["query_sampler_sha256"]) == 64
    assert manifest["query_scored_windows"] == manifest["query_batch_count"] * dm.batch_size_per_device


def test_manifest_supports_remote_sampler_without_session_batch_counts(tmp_path, monkeypatch):
    root = make_fake_root(tmp_path)
    dm = make_datamodule(root, arm="zero4")
    monkeypatch.setattr(version_b, "SourceFrozenEMGAFC4Plan", FakeSourcePlan)
    dm.prepare_session_data = lambda path, task, **kwargs: fake_record(Path(path))
    dm.setup("fit")
    expected = dict(dm.train_batch_sampler.original_session_batch_counts)
    del dm.train_batch_sampler.session_batch_counts

    manifest = dm.get_split_manifest()

    assert manifest["train_batch_counts"] == expected


@pytest.mark.parametrize("arm", ["full", "b4", "rs4", "ls4"])
def test_control_arms_add_target_and_reach_dataset_with_exact_arm(tmp_path, monkeypatch, arm):
    root = make_fake_root(tmp_path)
    dm = make_datamodule(root, arm=arm)
    monkeypatch.setattr(version_b, "SourceFrozenEMGAFC4Plan", FakeSourcePlan)
    dm.prepare_session_data = lambda path, task, **kwargs: fake_record(Path(path))
    dm.setup("fit")

    assert dm.carrier_plan is not None
    assert [path.name for path in dm.carrier_plan.target_added] == [EXPECTED_FILES["ses-20120924"]]
    assert dm.train_dataset.carrier_arm == arm
    assert dm.val_heldin_dataset.carrier_arm == arm
    side = dm.val_heldin_dataset[0][-1]
    expected_first = {"full": 1.0, "b4": 0.0, "rs4": 1.0, "ls4": -1.0}[arm]
    assert side[0, 0] == expected_first
    manifest = dm.get_split_manifest()
    assert manifest["carrier_arm"] == arm
    assert manifest["carrier_plan_receipt"]["arm"] == arm
    if arm == "b4":
        assert manifest["carrier_plan_receipt"]["b4_mask"] is not None
    if arm == "rs4":
        assert manifest["carrier_plan_receipt"]["rs4"] is not None
    if arm == "ls4":
        assert manifest["carrier_plan_receipt"]["ls4"] is not None


@pytest.mark.parametrize("arm", ["full", "b4", "rs4", "ls4"])
def test_target_m10_carrier_is_frozen_before_query_record_materialization(
    tmp_path, monkeypatch, arm,
):
    root = make_fake_root(tmp_path)
    dm = make_datamodule(root, arm=arm)
    events = []

    class OrderedPlan(FakeSourcePlan):
        def add_target(self, path):
            events.append("target_m10_carrier_frozen")
            return super().add_target(path)

    monkeypatch.setattr(version_b, "SourceFrozenEMGAFC4Plan", OrderedPlan)

    def prepare(path, task, **kwargs):
        if "_ses-20120924_" in Path(path).name:
            events.append("target_query_record_materialized")
        return fake_record(Path(path))

    dm.prepare_session_data = prepare
    dm.setup("fit")
    assert events == ["target_m10_carrier_frozen", "target_query_record_materialized"]


def test_constructor_rejects_non_m1_or_wrong_fold(tmp_path):
    root = make_fake_root(tmp_path)
    with pytest.raises(ValueError, match="native M1"):
        version_b.M1VersionBSourceLOSODataModule(
            task="m2",
            data_dir=str(root),
            source_session_names=["ses-20120926", "ses-20120927", "ses-20120928"],
            loso_fold=0,
        )
    with pytest.raises(ValueError, match="fold-1"):
        version_b.M1VersionBSourceLOSODataModule(
            task="m1",
            data_dir=str(root),
            source_session_names=["ses-20120926", "ses-20120927", "ses-20120928"],
            loso_fold=1,
        )


def test_all_arms_use_single_terminal_query_policy():
    from hydra import compose, initialize_config_dir

    config_dir = (STREAMING_ROOT / "configs").resolve()
    for experiment in (
        "m1_version_b_hs_continuation",
        "m1_version_b_c0",
        "m1_version_b_c",
        "m1_version_b_b4",
        "m1_version_b_rs4",
        "m1_version_b_ls4",
    ):
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    f"experiment={experiment}",
                    f"paths.root_dir={REPO_ROOT}",
                    "hydra.job.chdir=false",
                ],
            )
        assert str(cfg.optimized_metric) == "test_heldin/r2_mean"
        assert int(cfg.trainer.limit_val_batches) == 0
        assert int(cfg.trainer.num_sanity_val_steps) == 0
        fixed = cfg.callbacks.fixed_last_checkpoint
        assert fixed.monitor is None
        assert bool(fixed.save_last) is False
        assert int(fixed.every_n_epochs) == 12
        assert int(fixed.save_top_k) == -1
        assert cfg.callbacks.early_stopping is None


def test_control_configs_are_resolved_training_matched_and_not_authorizations():
    from sua_exploration.scripts import m1_version_b_controls_preflight as preflight

    bound = preflight.bind_frozen_precommit()
    assert bound["mechanism_controls_in_fixed_order"] == ["B-B4", "B-RS4", "B-LS4"]
    receipt = preflight.build_receipt(live_data=False)
    assert receipt["status"] == "PASS_LOCAL_CONTROL_IMPLEMENTATION_DRY_RUN_NOT_EXECUTION_AUTHORIZATION"
    assert receipt["execution_authorized"] is False
    assert receipt["gpu_started"] is False
    assert receipt["remote_connected_or_synchronized"] is False
    assert receipt["main_gate_result_read"] is False
    arms = receipt["config_dry_run"]["arms"]
    assert arms["m1_version_b_b4"]["arm"] == "b4"
    assert arms["m1_version_b_rs4"]["arm"] == "rs4"
    assert arms["m1_version_b_ls4"]["arm"] == "ls4"
    assert len({row["matched_training_hash"] for row in arms.values()}) == 1
