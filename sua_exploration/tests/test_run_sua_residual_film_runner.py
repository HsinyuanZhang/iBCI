from __future__ import annotations

import subprocess
from pathlib import Path


def test_residual_runner_is_m50_only_and_dry_run_capable():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sua_residual_film_one_cell.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert 'M_T4=50' in text
    assert 'ARM must be residual_film, residual_shuffle, or residual_nofilm' in text
    assert 'SEED must be one of 42,43,44' in text
    assert '--freeze_decoder' in text and '--freeze_encoder_base' in text
    assert 'required existing reference is missing' in text
    assert 'Refusing to reuse run directory' in text
    assert 'Refusing to overwrite result' in text
    assert 'Refusing to overwrite log' in text
    assert text.count('[[ ! -e "$RESULT" ]]') == 2
