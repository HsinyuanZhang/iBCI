"""Static contracts for the isolated remote D-Q4e staging preflight."""
from __future__ import annotations

from pathlib import Path

from scripts import h1_carrierid_distribution_exposure_remote_stage_preflight as stage


def test_q4e_command_is_fresh_fixed_epoch_and_has_absolute_readonly_overrides(tmp_path):
    command = stage.q4e_command(
        python=Path("/usr/bin/python3"), data_root=Path("/readonly/data"),
        raw_receipt=Path("/readonly/raw.json"), eb_receipt=Path("/readonly/eb.json"),
        cache_dir=Path("/stage/cache"), output_root=tmp_path / "outputs",
    )
    text = " ".join(command)
    assert "experiment=h1_carrierid_distribution_exposure_q4" in text
    assert "data.raw_receipt_path=/readonly/raw.json" in text
    assert "data.eb_receipt_path=/readonly/eb.json" in text
    assert "trainer.max_epochs=50" in text and "trainer.min_epochs=50" in text
    assert "model.optimizer.lr=5e-5" in text and "seed=42" in text
    assert "test=false" in text and "ckpt_path=null" in text


def test_remote_stage_preflight_is_nonlaunch_and_no_data_loader():
    source = Path(stage.__file__).read_text(encoding="utf-8")
    assert "DataModule(" not in source
    assert "load_source_records" not in source
    assert "load_target_records" not in source
    assert '"gpu_queued": False' in source
    assert '"launch_authorized": False' in source
