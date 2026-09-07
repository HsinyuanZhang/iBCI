"""CPU contracts for the independent H-SE5 19250108 Full/Zero5 pair."""
from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
PREFLIGHT = SPINT / "scripts/h1_hse5_lodo_date2_preflight.py"
SNAPSHOT_BUILDER = SPINT / "scripts/build_h1_sparse_event_source_snapshot_dated.py"
LAUNCH = SPINT / "scripts/h1_hse5_lodo_date2_launch.sh"
DATA = SPINT / "data/000954"


@pytest.mark.skipif(not DATA.is_dir(), reason="public H1 held-in data unavailable")
def test_date2_preflight_proves_fold0_fidelity_and_strict_date2_contract(tmp_path: Path) -> None:
    output, snapshot, snapshot_receipt, cache_dir = (
        tmp_path / "preflight.json", tmp_path / "source.npz", tmp_path / "source.json", tmp_path / "cache")
    done = subprocess.run([sys.executable, str(PREFLIGHT), "--data-dir", str(DATA), "--output", str(output),
                           "--snapshot", str(snapshot), "--snapshot-receipt", str(snapshot_receipt),
                           "--cache-dir", str(cache_dir), "--temp-mode"],
                          cwd=SPINT, text=True, capture_output=True, check=True)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "PASS_HSE5_LODO_DATE2_PREFLIGHT", done.stdout + done.stderr
    assert receipt["fold0_fidelity"]["pass"] is True
    assert receipt["date2"]["pass"] is True
    assert receipt["date2"]["target"]["query_window_indices_sha256"] == "b0cd153750cb484af1237b7af1861600aca69a9b201244c22d140b42b1da7f6e"
    assert receipt["date2"]["basis"]["basis_sha256"] == "d07975b99004b2a5f61c403d86b3151057e4609e80d344b8e7657eebfcea155b"
    binding = receipt["source_snapshot_binding"]
    assert binding["source_manifest_sha256"] == receipt["date2"]["source_manifest_sha256"]
    assert binding["basis_sha256"] == receipt["date2"]["basis"]["basis_sha256"]
    assert receipt["snapshot_publication"]["snapshot"] == str(snapshot.resolve())
    assert snapshot.is_file() and snapshot_receipt.is_file()


def test_date2_launch_dry_run_has_two_isolated_sparse_arms() -> None:
    done = subprocess.run(["bash", str(LAUNCH), "--dry-run"], cwd=SPINT, text=True, capture_output=True, check=True)
    text = done.stdout + done.stderr
    assert "h1_hse5_lodo_full_19250108" in text
    assert "h1_hse5_lodo_zero5_19250108" in text
    assert "CUDA_VISIBLE_DEVICES=0" in text and "CUDA_VISIBLE_DEVICES=1" in text
    assert "--dry-run: no tmux sessions started." in text


def test_standalone_builder_refuses_unbound_or_official_builds() -> None:
    unbound = subprocess.run([sys.executable, str(SNAPSHOT_BUILDER), "--temp-mode"], cwd=SPINT, text=True, capture_output=True)
    assert unbound.returncode != 0
    assert "--expected-manifest-sha256" in (unbound.stdout + unbound.stderr)
    official = subprocess.run([sys.executable, str(SNAPSHOT_BUILDER)], cwd=SPINT, text=True, capture_output=True)
    assert official.returncode != 0
    assert "published only by h1_hse5_lodo_date2_preflight.py" in (official.stdout + official.stderr)


def test_preflight_snapshot_binding_rejects_any_material_mismatch() -> None:
    spec = importlib.util.spec_from_file_location("hse5_preflight_test", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = types.SimpleNamespace(
        manifest_sha256="m" * 64,
        basis=types.SimpleNamespace(outer_date="19250108", basis_sha256="b" * 64),
        cache=types.SimpleNamespace(manifest={"cache_sha256": "c" * 64}),
        normalizer=types.SimpleNamespace(normalizer_sha256="n" * 64),
        source_window_indices_sha256="w" * 64,
        batch_order_sha256="o" * 64,
        schedule_sha256="s" * 64,
        receipt_sha256="r" * 64,
        snapshot_sha256="p" * 64,
    )
    binding = module._binding_from_snapshot(snapshot)
    receipt = {"status": "PASS_HSE5_LODO_DATE2_PREFLIGHT", "outer_date": "19250108", "source_snapshot_binding": dict(binding)}
    module.validate_preflight_snapshot_binding(receipt, snapshot)
    receipt["source_snapshot_binding"]["normalizer_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        module.validate_preflight_snapshot_binding(receipt, snapshot)
