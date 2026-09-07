from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from src.data.falcon_datamodule import FalconDataModule


SESSIONS = [
    "ses-2020-10-19-Run1", "ses-2020-10-19-Run2", "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2", "ses-2020-10-27-Run1", "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
]
HELDOUT_SESSIONS = [
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
]


def _tree(root: Path) -> set[Path]:
    paths = set()
    for directory, token in (("sub-MonkeyN-held-in-calib", "held-in-calib"), ("sub-MonkeyN-held-in-minival", "held-in-minival")):
        folder = root / directory
        folder.mkdir(parents=True)
        for session in SESSIONS:
            path = folder / f"sub-MonkeyN-{token}_{session}_behavior+ecephys.nwb"
            path.write_bytes(b"synthetic")
            paths.add(path.resolve())
    return paths


def _manifest(root: Path) -> Path:
    rows = []
    for directory, token, role in (
        ("sub-MonkeyN-held-in-calib", "held-in-calib", "heldin_calib"),
        ("sub-MonkeyN-held-in-minival", "held-in-minival", "heldin_minival"),
    ):
        for session in SESSIONS:
            path = (root / directory / f"sub-MonkeyN-{token}_{session}_behavior+ecephys.nwb").resolve()
            rows.append({
                "role": role,
                "session": session,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    payload = {
        "schema_version": 1,
        "protocol": "m2_clean_teacher_v1",
        "task": "m2",
        "calibration_n_trials": 24,
        "expected_heldin_sessions": SESSIONS,
        "forbidden_heldout_sessions": HELDOUT_SESSIONS,
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "files": rows,
    }
    output = root / "clean_teacher_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _heldout_tree(root: Path) -> None:
    folder = root / "sub-MonkeyN-held-out-calib"
    folder.mkdir(parents=True)
    for session in HELDOUT_SESSIONS:
        (folder / f"sub-MonkeyN-held-out-calib_{session}_behavior+ecephys.nwb").write_bytes(b"heldout")


def _fake_load(calls: list[Path]):
    def load(path, task):
        calls.append(Path(path).resolve())
        neural = np.zeros((48, 3), dtype=np.float32)
        covariates = np.zeros((48, 2), dtype=np.float32)
        trial_change = np.zeros(48, dtype=bool)
        trial_change[::2] = True
        return neural, covariates, trial_change, np.ones(48, dtype=bool)
    return load


def _module(root: Path, **extra) -> FalconDataModule:
    manifest_path = str(extra.pop("clean_teacher_manifest_path")) if "clean_teacher_manifest_path" in extra else str(_manifest(root))
    kwargs = dict(
        task="m2", data_dir=str(root), heldin_session_names=[], expected_heldin_sessions=SESSIONS,
        clean_teacher=True, include_heldout_in_fit=False, include_heldout_in_test=False,
        clean_teacher_manifest_path=manifest_path,
        batch_size=2, window_size=2, calibration_n_trials=24, random_calibration=False,
        smooth_calibration=False, max_trial_length=2, use_intertrials=True,
        interpolate_trials=False, num_workers=0,
    )
    kwargs.update(extra)
    return FalconDataModule(**kwargs)


def test_clean_teacher_setup_loads_exactly_fourteen_heldin_files(monkeypatch, tmp_path):
    expected = _tree(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr("src.data.falcon_datamodule.load_nwb", _fake_load(calls))
    module = _module(tmp_path)
    module.setup("fit")
    assert set(calls) == expected
    assert len(calls) == 14
    assert module.val_calib_heldout_sessions is None
    assert module.val_heldout_dataset is None
    assert module.val_heldout_batch_sampler is None
    assert not isinstance(module.val_dataloader(), list)


def test_clean_teacher_never_opens_synthetic_heldout_files(monkeypatch, tmp_path):
    expected = _tree(tmp_path)
    _heldout_tree(tmp_path)
    calls: list[Path] = []
    fake = _fake_load(calls)

    def deny_heldout(path, task):
        if "held-out" in str(path):
            raise AssertionError(f"heldout load attempted: {path}")
        return fake(path, task)

    monkeypatch.setattr("src.data.falcon_datamodule.load_nwb", deny_heldout)
    _module(tmp_path).setup("fit")
    assert set(calls) == expected
    assert all("held-out" not in str(path) for path in calls)


def test_clean_teacher_manifest_detects_runtime_input_drift_before_load(monkeypatch, tmp_path):
    _tree(tmp_path)
    manifest = _manifest(tmp_path)
    target = tmp_path / "sub-MonkeyN-held-in-calib" / f"sub-MonkeyN-held-in-calib_{SESSIONS[0]}_behavior+ecephys.nwb"
    target.write_bytes(b"changed-after-manifest")
    calls: list[Path] = []
    monkeypatch.setattr("src.data.falcon_datamodule.load_nwb", _fake_load(calls))
    with pytest.raises(ValueError, match="runtime input manifest"):
        _module(tmp_path, clean_teacher_manifest_path=str(manifest)).setup("fit")
    assert calls == []


def test_checkpoint_callback_embeds_verified_runtime_manifest_and_rejects_external_drift(monkeypatch, tmp_path):
    _tree(tmp_path)
    manifest = _manifest(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr("src.data.falcon_datamodule.load_nwb", _fake_load(calls))
    module = _module(tmp_path, clean_teacher_manifest_path=str(manifest))
    module.setup("fit")
    from src.callbacks.clean_teacher_provenance import CleanTeacherProvenanceCallback

    callback = CleanTeacherProvenanceCallback(str(manifest))
    trainer = SimpleNamespace(datamodule=module, callback_metrics={"val_heldin/r2_mean": 0.42})
    checkpoint = {"epoch": 7}
    callback.on_save_checkpoint(trainer, None, checkpoint)
    provenance = checkpoint["clean_teacher_provenance_v1"]
    assert provenance["runtime_manifest_equals_external"] is True
    assert len(provenance["runtime_manifest"]["files"]) == 14
    manifest.write_text("{}\n")
    with pytest.raises(ValueError, match="manifest drifted"):
        callback.on_save_checkpoint(trainer, None, {"epoch": 8})


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink"])
def test_clean_teacher_rejects_missing_extra_or_symlink_escape_before_load(monkeypatch, tmp_path, mutation):
    _tree(tmp_path)
    manifest = _manifest(tmp_path)
    if mutation == "missing":
        (tmp_path / "sub-MonkeyN-held-in-calib" / f"sub-MonkeyN-held-in-calib_{SESSIONS[0]}_behavior+ecephys.nwb").unlink()
    elif mutation == "extra":
        (tmp_path / "sub-MonkeyN-held-in-calib" / "sub-MonkeyN-held-in-calib_ses-extra_behavior+ecephys.nwb").write_bytes(b"extra")
    else:
        path = tmp_path / "sub-MonkeyN-held-in-calib" / f"sub-MonkeyN-held-in-calib_{SESSIONS[0]}_behavior+ecephys.nwb"
        path.unlink()
        outside = tmp_path / "outside.nwb"
        outside.write_bytes(b"outside")
        path.symlink_to(outside)
    calls: list[Path] = []
    monkeypatch.setattr("src.data.falcon_datamodule.load_nwb", _fake_load(calls))
    with pytest.raises(ValueError, match="clean teacher"):
        _module(tmp_path, clean_teacher_manifest_path=str(manifest)).setup("fit")
    assert calls == []


@pytest.mark.parametrize("kwargs,stage", [
    ({"include_heldout_in_fit": True}, "fit"),
    ({"include_heldout_in_test": True}, "fit"),
    ({}, "test"),
])
def test_clean_teacher_rejects_heldout_flags_or_test_before_any_load(monkeypatch, tmp_path, kwargs, stage):
    _tree(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr("src.data.falcon_datamodule.load_nwb", _fake_load(calls))
    with pytest.raises(ValueError, match="clean teacher"):
        _module(tmp_path, **kwargs).setup(stage)
    assert calls == []


def test_clean_teacher_callback_contract_rejects_heldout_selector():
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from src.train import assert_clean_teacher_selection_contract

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train.yaml", overrides=["experiment=m2_clean_teacher_m24"])
    assert cfg.callbacks.best_checkpoint.monitor == "val_heldin/r2_mean"
    assert cfg.callbacks.early_stopping.monitor == "val_heldin/r2_mean"
    callbacks = [
        ModelCheckpoint(dirpath="/tmp/clean_teacher_best", monitor="val_heldin/r2_mean", mode="max"),
        EarlyStopping(monitor="val_heldin/r2_mean", mode="max"),
    ]
    assert_clean_teacher_selection_contract(cfg, callbacks)
    bad = ModelCheckpoint(monitor="val_heldout/r2_mean", mode="max")
    with pytest.raises(ValueError, match="selector"):
        assert_clean_teacher_selection_contract(cfg, [*callbacks, bad])
