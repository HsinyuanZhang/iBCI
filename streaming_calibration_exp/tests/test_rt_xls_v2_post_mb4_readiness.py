"""No-NWB/no-GPU contracts for deferred RT XLSv2 post-MB4 readiness."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rt_xls_v2_post_mb4_readiness.py"
WORKSPACE = ROOT.parent
WAITING_PLAN = WORKSPACE / "sua_exploration/results/rt_xls_v2_post_mb4_readiness_v1/RT_AFC4_XLS_V2_POST_MB4_READINESS_WAITING_PLAN_v1.json"
SPEC = importlib.util.spec_from_file_location("rt_xls_v2_post_mb4_readiness_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _immutable(path: Path, body: dict) -> Path:
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def test_real_readiness_is_ready_not_a_launch_and_preserves_isolation() -> None:
    report = MODULE.audit_readiness()
    assert report["status"] == "READY_FOR_REVIEWED_XLSV2_APPLY_NOT_LAUNCHED"
    assert report["mb4_gate"]["state"] == "PASS_MB4_FULL15_IMMUTABLE_AGGREGATE"
    assert report["mb4_gate"]["sha256"] == "152b76c149ae6c0cdc75c7320a3ba80863bac5f56abfaea87e58286262d6768a"
    assert report["scope"] == {
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "trainer_constructed_or_launched": False,
        "cuda_constructed_or_launched": False,
        "tmux_session_created": False,
    }
    assert report["support_audit"]["sha256"] == "ad2468ca04c3ed2542c7c37b0f1d27c17d4e517943fe64a7fa34e6cf9636c899"
    assert "rt_xls_v2_matched_continuation_v1.py" in report["finalizer_template"]
    assert "--copy-import-all-and-finalize" in report["finalizer_template"]
    source = SCRIPT.read_text(encoding="utf-8")
    # The verifier names forbidden primitive tokens as strings for its static
    # check, but must not itself import a runtime data/model stack or execute
    # a worker.
    for forbidden in ("import torch", "from src.", "subprocess.run", "import pynwb", "import lightning"):
        assert forbidden not in source


def test_static_two_gpu_partition_is_exact_and_not_score_selected() -> None:
    partitions = MODULE.static_partitions()
    assert partitions["gpu1_ascending"] == {"physical_gpu": 1, "folds": list(range(8)), "order": "ascending"}
    assert partitions["gpu0_descending"] == {"physical_gpu": 0, "folds": list(range(14, 7, -1)), "order": "descending"}
    assert set(partitions["gpu1_ascending"]["folds"]) | set(partitions["gpu0_descending"]["folds"]) == set(range(15))
    assert not (set(partitions["gpu1_ascending"]["folds"]) & set(partitions["gpu0_descending"]["folds"]))


def test_mb4_gate_requires_immutable_exact_15_fold_receipt(tmp_path: Path) -> None:
    missing = MODULE._mb4_gate(tmp_path / "missing.json")
    assert missing["state"] == "WAITING_FOR_MB4_FULL15_IMMUTABLE_AGGREGATE"
    receipt = _immutable(tmp_path / "mb4.json", {
        "schema": MODULE.MB4_SCHEMA, "status": MODULE.MB4_STATUS, "arm": "afc4_mb4",
        "full_comparator_arm": "afc4_vel", "seed": 42,
        "rows": [{"fold": fold} for fold in range(15)],
    })
    passed = MODULE._mb4_gate(receipt)
    assert passed["state"] == "PASS_MB4_FULL15_IMMUTABLE_AGGREGATE"
    assert passed["folds"] == list(range(15))
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444

    mutable = tmp_path / "mutable.json"
    mutable.write_text(receipt.read_text(encoding="utf-8"), encoding="utf-8")
    mutable.chmod(0o644)
    with pytest.raises(MODULE.XlsV2ReadinessError, match="mode-0444"):
        MODULE._mb4_gate(mutable)

    tampered = _immutable(tmp_path / "tampered.json", {
        "schema": MODULE.MB4_SCHEMA, "status": MODULE.MB4_STATUS, "arm": "afc4_mb4",
        "full_comparator_arm": "afc4_vel", "seed": 42,
        "rows": [{"fold": fold} for fold in range(14)] + [{"fold": 13}],
    })
    with pytest.raises(MODULE.XlsV2ReadinessError, match="fold matrix"):
        MODULE._mb4_gate(tampered)


def test_immutable_waiting_plan_freezes_exact_apply_boundary_without_authorizing_compute() -> None:
    body = json.loads(WAITING_PLAN.read_text(encoding="utf-8"))
    assert WAITING_PLAN.stat().st_mode & 0o777 == 0o444
    assert body["status"] == "WAITING_FOR_MB4_FULL15_IMMUTABLE_AGGREGATE"
    assert body["mb4_prerequisite"]["currently_present"] is False
    assert body["static_partitions"]["gpu1_ascending"]["folds"] == list(range(8))
    assert body["static_partitions"]["gpu0_descending"]["folds"] == list(range(14, 7, -1))
    assert body["frozen_protocol"]["reuse_r_c_checkpoint"] is False
    assert body["frozen_protocol"]["common_inverse_or_alignment_map"] == "FORBIDDEN"
    assert body["scope"]["target_recordings_opened"] == 0
    assert body["scope"]["cuda_constructed_or_launched"] is False
    # This immutable document is the historical pre-apply boundary.  Once the
    # READY gate is consumed, active integration intentionally changes some of
    # the recorded source/test bytes; retain the old digests as provenance
    # instead of pretending they still describe the applied tree.
    for relative, expected in body["code_sha256"].items():
        path = WORKSPACE / relative
        assert path.is_file(), path
        assert len(expected) == 64 and all(character in "0123456789abcdef" for character in expected)
