"""CPU integration tests for A1 immutable evidence and execution surfaces."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
PY = "/home/xinyuan/miniconda3/envs/spint/bin/python"
PREFLIGHT = SUA / "scripts/a1_hidden_space_carrier_preflight.py"
RUNNER = SUA / "scripts/run_a1_hidden_space_carrier_one_cell.sh"
AGGREGATOR = SUA / "scripts/aggregate_a1_hidden_space_carrier.py"
if str(SUA) not in sys.path:
    sys.path.insert(0, str(SUA))

from a1_hidden_carrier.a2_anchors import verify_sealed_a2_reuse
from a1_hidden_carrier.artifacts import (
    load_verified_immutable_json,
    sidecar_path,
    write_immutable_json,
)
from a1_hidden_carrier.contract import PILOT_SEED, SCREEN_ID, SESSIONS, synthetic_fresh_score_receipt
from a1_hidden_carrier.cpu_proofs import run_cpu_proofs


def _run(cmd, *, env=None):
    merged = {**os.environ, **(env or {})}
    merged["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, env=merged, cwd=SUA)


def test_production_cpu_proofs() -> None:
    receipt = run_cpu_proofs()
    assert receipt["all_passed"] is True
    assert receipt["gpu_used"] is False
    assert receipt["nwb_opened"] is False
    assert receipt["proofs"]["production_optimizer_covers_p_and_all_trainables_once"] is True
    assert receipt["proofs"]["production_strict_checkpoint_roundtrip"] is True
    assert receipt["proofs"]["nonzero_t4_reaches_p"] is True


def test_immutable_writer_and_reader(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    body, sidecar, digest = write_immutable_json(path, {"a": 1})
    assert stat.S_IMODE(body.stat().st_mode) == 0o444
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    payload, observed = load_verified_immutable_json(path)
    assert payload == {"a": 1} and observed == digest
    with pytest.raises(FileExistsError):
        write_immutable_json(path, {"a": 2})


def test_immutable_reader_rejects_bad_mode_and_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    body, sidecar, _ = write_immutable_json(path, {"a": 1})
    body.chmod(0o644)
    with pytest.raises(ValueError, match="0444"):
        load_verified_immutable_json(path)
    body.chmod(0o444); sidecar.chmod(0o644)
    with pytest.raises(ValueError, match="0444"):
        load_verified_immutable_json(path)


def test_a2_anchor_full_checkpoint_audit() -> None:
    result = verify_sealed_a2_reuse(seed=42, verify_checkpoint_bytes=True)
    assert result["status"] == "SEALED_A2_W_REUSE_VERIFIED"
    for arm in ("source_z4", "source_t4"):
        cell = result["sealed_w_cells"][arm]
        assert len(cell["checkpoint_payload_audits"]) == 8
        assert cell["optimizer_coverage"] == {
            "decoder_tensor_count": 31,
            "identity_encoder_tensor_count": 8,
            "total_unique_parameter_count": 39,
            "verified_every_epoch_5_through_12": True,
        }


def test_runner_minimal_dry_run_and_launch_denial() -> None:
    dry = _run(["bash", str(RUNNER), "--dry-run"], env={"CELL": "H/T4", "SEED": "42", "GPU": "0"})
    assert dry.returncode == 0, dry.stderr
    assert "FRESH_TRAINING_FAMILIES=H/T4" in dry.stdout
    assert "STRUCTURAL_ALIAS=H/Z4->W/Z4" in dry.stdout
    assert "NO_COMMAND_EXECUTED=true" in dry.stdout
    denied = _run(["bash", str(RUNNER), "--launch"], env={"CELL": "H/T4", "SEED": "42", "GPU": "0"})
    assert denied.returncode == 3


def test_scorer_dry_run_opens_nothing(tmp_path: Path) -> None:
    proc = _run([
        PY, "-m", "a1_hidden_carrier.scorer", "--run-dir", str(tmp_path / "missing"),
        "--official-preflight", str(tmp_path / "missing_preflight.json"),
        "--launch-receipt", str(tmp_path / "missing_launch.json"),
        "--out", str(tmp_path / "score.json"),
    ])
    assert proc.returncode == 0
    assert "DRY_RUN_NO_CHECKPOINT_OR_NWB_OPEN" in proc.stdout


def test_preflight_mints_immutable_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "preflight.json"
    proc = _run([PY, str(PREFLIGHT), "--receipt", str(receipt), "--result-root", str(tmp_path / "empty")])
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload, _ = load_verified_immutable_json(receipt)
    assert payload["status"] == "CPU_PREFLIGHT_PASSED_AWAITING_ROOT_GO"
    assert payload["fresh_gpu_training_cells_in_pilot"] == 1
    assert payload["fresh_training_families"] == ["H/T4"]
    assert payload["a2_reuse_evidence"]["status"] == "SEALED_A2_W_REUSE_VERIFIED"


def test_aggregator_with_synthetic_fresh_score(tmp_path: Path) -> None:
    # Build the preflight payload in-process so its implementation bindings are current.
    from scripts.a1_hidden_space_carrier_preflight import build_receipt
    preflight_payload = build_receipt(
        result_root=tmp_path / "unused", verify_checkpoint_bytes=False,
        run_model_proofs=True,
    )
    assert not preflight_payload["implementation_blockers"]
    preflight = tmp_path / "preflight.json"
    write_immutable_json(preflight, preflight_payload)
    w_t4 = preflight_payload["a2_reuse_evidence"]["sealed_w_cells"]["source_t4"]["per_session_mean_r2"]
    fresh_payload = synthetic_fresh_score_receipt(delta=0.0)
    fresh_payload["aligned"]["per_session_mean_r2"] = {
        session: float(w_t4[session]) + 0.04 for session in SESSIONS
    }
    fresh = tmp_path / "fresh.json"
    write_immutable_json(fresh, fresh_payload)
    out = tmp_path / "aggregate.json"
    proc = _run([PY, str(AGGREGATOR), "--preflight", str(preflight), "--fresh-score", str(fresh), "--out", str(out)])
    assert proc.returncode == 0, proc.stderr + proc.stdout
    result, _ = load_verified_immutable_json(out)
    assert result["passes"] is True
    assert result["overall"]["mean_primary_interaction"] == pytest.approx(0.04)
    assert all(row["H/Z4"] == row["W/Z4"] for row in result["per_session"])
