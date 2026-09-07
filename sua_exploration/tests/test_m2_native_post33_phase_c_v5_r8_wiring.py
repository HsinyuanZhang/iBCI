"""CPU-only regression tests for the fresh r8 prelaunch wiring."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r8_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase_c_v5_r8_matrix_test", MATRIX)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r8_dry_run_uses_the_r8_portable_field() -> None:
    module = _module()
    payload = module.dry_run_payload(
        {
            "host_id": "host",
            "gpu_id": "GPU-uuid",
            "absolute_cell_root": "/tmp/r8-cells",
            "fold_allowlist": [0, 2],
            "seed_allowlist": [42],
            "portable_manifest_sha256": "a" * 64,
        }
    )
    assert payload["portable_manifest_sha256"] == "a" * 64
    assert payload["pair_count"] == 2
    assert payload["execution_started"] is False


def test_r8_matrix_has_no_stale_v4_shard_field() -> None:
    source = MATRIX.read_text(encoding="utf-8")
    assert "portable_transfer_manifest_sha256" not in source
    assert "portable_manifest_sha256" in source
