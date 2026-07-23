"""Tests for held-out test gating in FalconDataModule."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir

from src.data.falcon_datamodule import FalconDataModule


@pytest.mark.parametrize("include_heldout_in_test", [False, True])
def test_test_dataloader_respects_include_heldout_in_test(include_heldout_in_test: bool):
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=b2_d128_anchor",
                "data.loso_fold=0",
                f"data.include_heldout_in_test={str(include_heldout_in_test).lower()}",
            ],
        )
    dm = FalconDataModule(**{k: v for k, v in cfg.data.items() if k != "_target_"})

    trainer = MagicMock()
    trainer.world_size = 1
    dm.trainer = trainer
    dm.setup(stage="test")
    loaders = dm.test_dataloader()
    if include_heldout_in_test:
        assert isinstance(loaders, list)
        assert len(loaders) == 2
        assert dm.val_heldout_dataset is not None
    else:
        assert not isinstance(loaders, list)
        assert dm.val_heldout_dataset is None
