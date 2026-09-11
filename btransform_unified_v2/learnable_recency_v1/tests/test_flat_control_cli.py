"""The flat-control preparer records commands but never starts an arm."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_flat_control.py"
WORKSPACE = ROOT.parents[1]


def test_prepare_cli_writes_all_untrained_manifests_without_result_targets(tmp_path):
    dest = tmp_path / "setup"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "all",
            "--dest",
            str(dest),
            "--python",
            "python with spaces",
            "--device",
            "cpu",
        ],
        cwd=WORKSPACE,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "PREPARED_NOT_TRAINED"
    assert receipt["training_started"] is False

    combined = json.loads((dest / "manifest.json").read_text())
    assert combined["status"] == "PREPARED_NOT_TRAINED"
    assert [item["task"] for item in combined["tasks"]] == ["m1", "m2", "h1"]
    for item in combined["tasks"]:
        assert item["training_started"] is False
        control = item["control"]
        assert control["effective_bias"] == "allzero"
        assert control["flat_config"]["half_life_seconds"] == [None] * 8
        assert control["reference"]["run_meta_sha256"]
        assert control["reference"]["selection_sha256"]
        assert control["train_argv"][0] == "python with spaces"
        assert (
            control["train_argv"][control["train_argv"].index("--device") + 1] == "cpu"
        )
        assert (dest / f"{item['task']}.json").is_file()
        assert not Path(control["results_dir"]).exists()
        if control["selection_dir"] is not None:
            assert not Path(control["selection_dir"]).exists()

    shell = (dest / "run_commands.sh").read_text()
    assert "PREPARED_NOT_TRAINED" in shell
    assert "'python with spaces'" in shell


def test_preparer_rejects_non_prepare_destination_content(tmp_path):
    sys.path.insert(0, str(SCRIPT.parent))
    import prepare_flat_control as prepare

    destination = tmp_path / "setup"
    destination.mkdir()
    (destination / "someone_else.txt").write_text("do not overwrite\n")
    with pytest.raises(FileExistsError, match="non-prepare destination"):
        prepare._check_destination(destination)
