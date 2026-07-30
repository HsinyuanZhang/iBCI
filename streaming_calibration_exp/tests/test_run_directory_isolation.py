"""M1: unique run directory per (variant, fold, seed).

Regression tests for sua_exploration/docs/CURRENT_RESULTS.md section H.4 -- two seeds of
the same MUA variant/fold once resolved to the same Hydra run directory (named only from
a second-resolution ``${now:...}`` timestamp) and silently commingled checkpoints and
tfevents from two PIDs. See configs/hydra/default.yaml and
src.metrics.run_artifacts.assert_run_dir_is_fresh.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.metrics.run_artifacts import RunDirectoryCollisionError, assert_run_dir_is_fresh


def _resolve_run_dir(overrides: list[str]) -> str:
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        return str(cfg.hydra.run.dir)


def test_two_seeds_resolve_to_different_run_dirs():
    base_overrides = [
        "experiment=b15p_m2_loso_internal",
        "data.loso_fold=1",
        "run_id=attention_arch_screen_v3_b15p_m2",
    ]
    dir_seed42 = _resolve_run_dir(base_overrides + ["seed=42"])
    dir_seed43 = _resolve_run_dir(base_overrides + ["seed=43"])
    assert dir_seed42 != dir_seed43
    # Both must carry their own seed, not just differ by timestamp noise.
    assert "_s42" in dir_seed42
    assert "_s43" in dir_seed43


def test_two_folds_same_seed_resolve_to_different_run_dirs():
    base_overrides = [
        "experiment=b15p_m2_loso_internal",
        "seed=42",
        "run_id=attention_arch_screen_v3_b15p_m2",
    ]
    dir_fold1 = _resolve_run_dir(base_overrides + ["data.loso_fold=1"])
    dir_fold2 = _resolve_run_dir(base_overrides + ["data.loso_fold=2"])
    assert dir_fold1 != dir_fold2
    assert "_f1_" in dir_fold1
    assert "_f2_" in dir_fold2


def test_identical_overrides_launched_twice_still_differ():
    # Regression case for the actual bug: two processes launched with the exact same
    # CLI overrides (e.g. a retry, or two GPUs racing) must not collide even though
    # run_id/fold/seed are identical -- the microsecond timestamp must break the tie.
    overrides = [
        "experiment=b15p_m2_loso_internal",
        "data.loso_fold=1",
        "seed=42",
        "run_id=attention_arch_screen_v3_b15p_m2",
    ]
    first = _resolve_run_dir(overrides)
    second = _resolve_run_dir(overrides)
    assert first != second


def test_ad_hoc_run_without_experiment_run_id_or_fold_does_not_crash():
    # No experiment= override means run_id stays at its train.yaml default (null) and
    # data.loso_fold stays at its data/*.yaml default (null). Both interpolate into the
    # directory name as the literal text "None" rather than raising an OmegaConf
    # interpolation error -- oc.select is what makes the loso_fold half of this safe for
    # non-LOSO data configs.
    run_dir = _resolve_run_dir(["seed=42"])
    assert run_dir
    assert "rid-None_fNone_s42" in run_dir


def test_non_loso_experiment_without_fold_does_not_crash():
    # Most experiment configs are non-LOSO and only set run_id; data.loso_fold still
    # defaults to null for these and must not raise.
    run_dir = _resolve_run_dir(["experiment=b3_d64"])
    assert run_dir
    assert "_fNone_s42" in run_dir
    assert "rid-b3_d64_screen" in run_dir


def test_assert_run_dir_is_fresh_allows_nonexistent_directory(tmp_path):
    assert_run_dir_is_fresh(tmp_path / "does_not_exist_yet")


def test_assert_run_dir_is_fresh_allows_hydra_bookkeeping_files(tmp_path):
    # Hydra itself writes job logs / .hydra/*.yaml / hparams.yaml into a freshly
    # resolved run directory before user code ever runs; those must not trip the check.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train.log").write_text("log\n")
    (run_dir / "config_tree.log").write_text("tree\n")
    (run_dir / "hparams.yaml").write_text("seed: 42\n")
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text("seed: 42\n")
    assert_run_dir_is_fresh(run_dir)


def test_assert_run_dir_is_fresh_raises_on_existing_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints" / "best_ckpt"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch_005.ckpt").write_bytes(b"not a real checkpoint")
    with pytest.raises(RunDirectoryCollisionError, match="checkpoint"):
        assert_run_dir_is_fresh(run_dir)


def test_assert_run_dir_is_fresh_raises_on_existing_tfevents(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "events.out.tfevents.1784934706.hw3090.268035.0").write_bytes(b"")
    with pytest.raises(RunDirectoryCollisionError):
        assert_run_dir_is_fresh(run_dir)


def test_assert_run_dir_is_fresh_reports_the_two_pid_collision_shape(tmp_path):
    # Mirrors the exact evidence from logs/train/runs/2026-07-25-07-11-45/: two PIDs'
    # tfevents streams plus Lightning "-v1" dedup-suffixed checkpoints in one directory.
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints" / "best_ckpt"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "events.out.tfevents.1784934706.hw3090.268035.0").write_bytes(b"")
    (run_dir / "events.out.tfevents.1784934706.hw3090.268037.0").write_bytes(b"")
    (ckpt_dir / "epoch_017.ckpt").write_bytes(b"")
    (ckpt_dir / "epoch_017-v1.ckpt").write_bytes(b"")
    with pytest.raises(RunDirectoryCollisionError):
        assert_run_dir_is_fresh(run_dir)
