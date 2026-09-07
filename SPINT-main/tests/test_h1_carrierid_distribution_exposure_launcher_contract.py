"""No-I/O execution contracts for the H1 D-S4e/D-Q4e plan-only launcher."""
from __future__ import annotations

from pathlib import Path

from scripts import h1_carrierid_distribution_exposure_launcher as launcher


def test_commands_have_exact_exposure_arms_and_fixed_terminal_schedule(tmp_path):
    values = launcher.commands(
        output_root=tmp_path / "fresh", source_cache_dir=tmp_path / "cache", python=Path("/usr/bin/python3")
    )
    assert set(values) == {"s4e", "q4e"}
    for arm, command in values.items():
        text = " ".join(command)
        assert f"experiment=h1_carrierid_distribution_exposure_{'s4' if arm == 's4e' else 'q4'}" in text
        assert "trainer.max_epochs=50" in text and "trainer.min_epochs=50" in text
        assert "model.optimizer.lr=5e-5" in text and "seed=42" in text
        assert "test=false" in text and "ckpt_path=null" in text


def test_checker_contract_requires_fixed_epoch_pair_and_no_target_before_audit():
    values = launcher.terminal_checker_requirements()
    assert values["checkpoint_paths"]["d_s4e"].endswith("epoch_049.ckpt")
    assert values["checkpoint_paths"]["d_q4e"].endswith("epoch_049.ckpt")
    assert values["each_checkpoint"]["metadata_schema"] == launcher.TERMINAL_SCHEMA
    assert values["each_checkpoint"]["checkpoint_epoch_zero_based"] == 49
    assert any("target/minival" in item for item in values["forbidden"])


def test_launcher_does_not_offer_execution_or_subprocess_path():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source and "from subprocess" not in source
    assert '"--execute"' not in source
    assert "commands_not_executed" in source
