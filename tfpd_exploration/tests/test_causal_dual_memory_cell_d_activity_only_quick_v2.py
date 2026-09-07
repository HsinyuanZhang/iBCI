"""No-data checks for the environment-bound activity-only successor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.causal_dual_memory_cell_d_activity_only_quick_v2 import physical, plan


ROOT = Path(__file__).resolve().parents[2]


def _copy_failed_v1(tmp_path: Path) -> Path:
    source = ROOT / plan.V1_FAILED_ROOT_RELATIVE
    target_root = tmp_path / plan.V1_FAILED_ROOT_RELATIVE
    target_root.mkdir(parents=True)
    for item in source.iterdir():
        shutil.copyfile(item, target_root / item.name)
        os.chmod(target_root / item.name, 0o444)
    os.chmod(target_root, 0o555)
    return target_root


def test_actual_failed_v1_graph_is_exact() -> None:
    binding = physical.validate_failed_v1(ROOT)
    assert binding["body_sha256s"] == plan.V1_FAILED_BODY_SHA256S
    assert binding["input_authority_opened"] is False
    assert binding["cuda_initialized"] is False


def test_failed_v1_graph_rejects_body_and_topology_drift(tmp_path: Path) -> None:
    target = _copy_failed_v1(tmp_path)
    os.chmod(target, 0o755)
    os.chmod(target / "failure.json", 0o644)
    (target / "failure.json").write_text("{}\n", encoding="utf-8")
    os.chmod(target / "failure.json", 0o444)
    os.chmod(target, 0o555)
    with pytest.raises(physical.ActivityOnlyQuickV2Error):
        physical.validate_failed_v1(tmp_path)

    os.chmod(target, 0o755)
    for item in target.iterdir():
        os.chmod(item, 0o644)
    shutil.rmtree(target)
    target = _copy_failed_v1(tmp_path)
    os.chmod(target, 0o755)
    (target / "extra").write_text("x", encoding="utf-8")
    os.chmod(target / "extra", 0o444)
    os.chmod(target, 0o555)
    with pytest.raises(physical.ActivityOnlyQuickV2Error, match="topology"):
        physical.validate_failed_v1(tmp_path)


def test_environment_requires_exact_data_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {**plan.EXACT_DATA_ROOT_ENV, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    assert physical.validate_environment(gpu_index=1) == expected
    monkeypatch.setenv("SUBC_DATA_ROOT", expected["SUBC_DATA_ROOT"] + "/drift")
    with pytest.raises(physical.ActivityOnlyQuickV2Error, match="SUBC_DATA_ROOT"):
        physical.validate_environment(gpu_index=1)


def test_dry_cli_is_inert_and_torch_free() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_activity_only_quick_v2.py"
    command = "import runpy,sys; p=sys.argv[1]; sys.argv=[p]; runpy.run_path(p,run_name='__main__'); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", command, str(script)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["no_data_access"] is True
    assert payload["v1_failed_body_sha256s"] == plan.V1_FAILED_BODY_SHA256S
