"""Focused synthetic/no-NWB contracts for the EST4 source projection audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h1_carrierid_date_lodo_est4_projection_identifiability_audit.py"
SPEC = importlib.util.spec_from_file_location("est4_projection_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_immutable(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _bundle(root: Path, date: str, index: int) -> None:
    directory = root / date
    directory.mkdir(parents=True)
    rng = np.random.default_rng(100 + index)
    pcs = np.linalg.qr(rng.normal(size=(AUDIT.N, AUDIT.Q)))[0][:, :AUDIT.Q].T.astype(np.float64)
    U = np.linalg.qr(rng.normal(size=(AUDIT.K, AUDIT.D)))[0][:, :AUDIT.D].astype(np.float64)
    arrays = {
        "mean": rng.normal(size=AUDIT.N).astype(np.float64),
        "scale": (0.5 + rng.random(AUDIT.N)).astype(np.float64),
        "pcs": pcs,
        "q": np.asarray(AUDIT.Q, dtype=np.int64),
        "lambda": np.asarray(100.0, dtype=np.float64),
        "U": U,
        "mu": rng.normal(size=AUDIT.D).astype(np.float64),
        "tau2": np.asarray(1e-10, dtype=np.float64),
    }
    array_path = directory / "frozen_m4_plan.npz"
    np.savez(array_path, **arrays)
    array_path.chmod(0o444)
    manifest_path = directory / "frozen_m4_plan.manifest.json"
    _write_immutable(manifest_path, {
        "schema": AUDIT.PLAN_SCHEMA, "outer_date": date, "raw_plan_sha256": _sha(array_path),
        "array_sha256": {name: _array_sha(arrays[name]) for name in ("mean", "scale", "pcs", "U", "mu")},
    })
    shared_path = directory / "shared_source_manifest.json"
    _write_immutable(shared_path, {
        "schema": AUDIT.BUNDLE_SCHEMA, "status": AUDIT.BUNDLE_STATUS, "outer_date": date,
        "source_only_scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                              "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False},
        "frozen_plan": {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": _sha(manifest_path)},
        "normalizer": {"denominator": 1.0e-4},
        "source_session_count": 2,
        "source_sessions": [f"source-{date}-a", f"source-{date}-b"],
        "source_files": [{"role": "source_heldin_calib"}, {"role": "source_heldin_calib"}],
    })


def _bundles(tmp_path: Path) -> Path:
    root = tmp_path / "source_bundles"
    for index, date in enumerate(AUDIT.DATES):
        _bundle(root, date, index)
    return root


def test_gauge_aware_metrics_ignore_basis_rotation_and_positive_scale() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(AUDIT.Q, AUDIT.N))
    rotation, _ = np.linalg.qr(rng.normal(size=(AUDIT.Q, AUDIT.Q)))
    changed = 3.7 * rotation @ matrix
    principal = AUDIT._principal_angle_summary(matrix, changed, rank=AUDIT.Q, label="synthetic pcs")
    residual = AUDIT._scale_rotation_procrustes(matrix, changed, label="synthetic pcs")
    assert principal["max_principal_angle_degrees"] == pytest.approx(0.0, abs=2e-6)
    assert residual["orthogonal_scale_aligned_relative_residual"] == pytest.approx(0.0, abs=2e-7)
    assert AUDIT._matrix_summary(matrix, name="a", expected_rank=AUDIT.Q)["condition_number"] == pytest.approx(
        AUDIT._matrix_summary(changed, name="b", expected_rank=AUDIT.Q)["condition_number"]
    )


def test_cpu_source_bundle_audit_writes_immutable_blocked_consumer_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = _bundles(tmp_path)
    output = tmp_path / "audit.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = AUDIT.run(bundle_root=source_root, component_path=AUDIT.DEFAULT_COMPONENT,
                       consumer_path=AUDIT.DEFAULT_CONSUMER, output=output)
    assert result["status"] == AUDIT.AUDIT_STATUS
    assert output.stat().st_mode & 0o777 == 0o444
    body = json.loads(output.read_text())
    assert body["source_scope"]["nwb_opened"] == 0
    assert body["source_scope"]["outer_target_recordings_opened"] == 0
    assert body["fail_closed_judgment"]["est4_consumer_attribution"] == "BLOCKED_FAIL_CLOSED"
    assert body["fail_closed_judgment"]["whole_pipeline_predictive_comparison"].startswith("NOT_YET_MEASURED")
    assert tuple(body["source_bundles"]) == AUDIT.DATES
    assert len(body["cross_date_gauge_aware_subspace_metrics"]["pcs_row_space"]) == 10
    assert body["per_date_initialization_metrics"][AUDIT.DATES[0]]["pcs"]["rank"] == AUDIT.Q
    assert "raw pcs/U/mu/lambda/tau2" in " ".join(body["fail_closed_judgment"]["existing_source_receipts_not_sufficient_for"])


def test_target_scope_drift_and_gpu_visibility_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = _bundles(tmp_path)
    shared = source_root / AUDIT.DATES[0] / "shared_source_manifest.json"
    shared.chmod(0o644)
    body = json.loads(shared.read_text())
    body["source_only_scope"]["target_bytes_read"] = 1
    _write_immutable(shared, body)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(AUDIT.Est4ProjectionAuditError, match="scope"):
        AUDIT.run(bundle_root=source_root, component_path=AUDIT.DEFAULT_COMPONENT,
                  consumer_path=AUDIT.DEFAULT_CONSUMER, output=tmp_path / "bad-scope.json")

    source_root = _bundles(tmp_path / "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(AUDIT.Est4ProjectionAuditError, match="CUDA_VISIBLE_DEVICES"):
        AUDIT.run(bundle_root=source_root, component_path=AUDIT.DEFAULT_COMPONENT,
                  consumer_path=AUDIT.DEFAULT_CONSUMER, output=tmp_path / "gpu.json")


def test_cli_requires_explicit_source_bundle_flag_without_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["h1_carrierid_date_lodo_est4_projection_identifiability_audit.py"])
    with pytest.raises(SystemExit, match="refusing implicit"):
        AUDIT.main()
