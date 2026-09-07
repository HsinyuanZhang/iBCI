from __future__ import annotations

import subprocess
import os
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_h1_baseline_staged.sh"
PINNED_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"


def test_staged_runner_has_fail_closed_shell_and_interpreter_contract():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")
    assert f"PYTHON_BIN=${{PYTHON_BIN:-{PINNED_PYTHON}}}" in source
    assert '[[ ! -x "$PYTHON_BIN" ]]' in source
    assert '"$PYTHON_BIN" -c \'import hydra, lightning, torch\'' in source
    assert source.count('"$PYTHON_BIN" src/train.py') == 2
    assert " python src/train.py" not in source
    assert '[[ -e "$RUN_ROOT" ]]' in source
    assert "prevents stale logs/checkpoints" in source


def test_staged_runner_refuses_existing_run_root_before_any_launch(tmp_path):
    marker = tmp_path / "rt_sealed.marker"
    marker.write_text("PASS\n", encoding="utf-8")
    run_root = tmp_path / "already-used"
    run_root.mkdir()
    env = os.environ.copy()
    # Existing-root rejection must happen before interpreter probing or any
    # CUDA/trainer process is started, so an intentionally invalid override is
    # safe in this test.
    env["PYTHON_BIN"] = "/definitely/not/an/interpreter"
    result = subprocess.run(
        ["bash", str(RUNNER), str(marker), str(run_root)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 5
    assert "already exists" in result.stderr

