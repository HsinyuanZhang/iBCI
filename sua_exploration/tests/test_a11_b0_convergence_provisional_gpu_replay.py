"""Focused unit tests for the additive A11 provisional GPU replay.

These tests exercise parity, scope, and immutable-receipt contracts with
synthetic payloads only.  They do not initialize CUDA, open NWB files, load a
checkpoint, or modify the running CPU evaluator/receipts.
"""
from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import a11_b0_convergence_provisional_gpu_replay as gpu  # noqa: E402


def _cpu_payload() -> dict:
    sessions = {
        "s1": 0.1,
        "s2": 0.2,
        "s3": 0.3,
        "s4": 0.4,
        "s5": 0.5,
        "s6": 0.6,
    }
    query = {
        name: {
            "name": name,
            "query_trial_count": 1,
            "query_trial_sha256": f"trial-{name}",
            "scored_window_count": 2,
            "scored_window_start_sha256": f"window-{name}",
            "source_unit_count": 3,
            "support_trial_count": 1,
            "support_trial_sha256": f"support-{name}",
        }
        for name in sessions
    }
    mean = sum(sessions.values()) / len(sessions)
    epoch = {
        "mean_r2": mean,
        "per_session_r2": sessions,
        "model_state_sha256_before": "state",
        "model_state_sha256_after": "state",
    }
    return {
        "cpu_forward_result": {
            "device": "cpu",
            "torch_grad_enabled": False,
            "normalizer_arrays_sha256": "normalizer",
            "query_windows": query,
            "opened_nwb_paths": [f"p{i}" for i in range(6)],
            "opened_nwb_path_count": 6,
            "formal_test_nwb_access": False,
            "per_seed": {
                "42": {"per_epoch": {"0": epoch}},
                "43": {"per_epoch": {}},
                "44": {"per_epoch": {}},
            },
        }
    }


def _gpu_from_cpu(cpu: dict) -> dict:
    forward = cpu["cpu_forward_result"]
    epoch = forward["per_seed"]["42"]["per_epoch"]["0"]
    query = json.loads(json.dumps(forward["query_windows"]))
    return {
        "device": "cuda",
        "torch_grad_enabled": False,
        "normalizer_arrays_sha256": forward["normalizer_arrays_sha256"],
        "query_windows": query,
        "opened_nwb_paths": list(forward["opened_nwb_paths"]),
        "opened_nwb_path_count": 6,
        "formal_test_nwb_access": False,
        "per_seed": {
            "42": {
                "per_epoch": {
                    "0": {
                        "mean_r2": epoch["mean_r2"],
                        "per_session_r2": dict(epoch["per_session_r2"]),
                        "model_state_sha256_before": "same",
                        "model_state_sha256_after": "same",
                    }
                }
            }
        },
    }


def test_gpu_driver_is_cuda_only_and_keeps_exact_source_guards():
    source = gpu._gpu_driver_source()

    assert 'torch.device("cuda:0")' in source
    assert 'torch.device("cpu")' not in source
    assert 'torch.set_grad_enabled(False)' in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "A11 GPU data-scope violation" in source
    assert "formal_test_nwb_access" in source
    assert "checkpoint_sha256_by_seed" in source
    assert "model_state_sha256_before" in source
    assert "python_no_user_site" in source
    assert "torch_cuda_version" in source
    assert "torch.no_grad()" in source


def test_parity_requires_mean_and_all_six_session_values_with_exact_query_hashes():
    cpu = _cpu_payload()
    gpu_payload = _gpu_from_cpu(cpu)

    result = gpu.compare_gpu_parity(cpu, gpu_payload)

    assert result["passed"] is True
    assert result["max_abs_r2_difference"] == pytest.approx(0.0)
    assert result["query_window_hashes_identical"] is True
    assert result["validation_session_count"] == 6

    gpu_payload["per_seed"]["42"]["per_epoch"]["0"]["per_session_r2"]["s6"] += 2e-5
    with pytest.raises(gpu.GPUReplayError, match="parity failed"):
        gpu.compare_gpu_parity(cpu, gpu_payload)


def test_parity_fails_closed_on_query_window_hash_drift():
    cpu = _cpu_payload()
    gpu_payload = _gpu_from_cpu(cpu)
    gpu_payload["query_windows"]["s3"]["scored_window_start_sha256"] = "drift"

    with pytest.raises(gpu.GPUReplayError, match="query/window hashes"):
        gpu.compare_gpu_parity(cpu, gpu_payload)


def test_parity_tolerance_cannot_be_weakened():
    with pytest.raises(gpu.GPUReplayError, match="<= 1e-05"):
        gpu.compare_gpu_parity(_cpu_payload(), _gpu_from_cpu(_cpu_payload()), tolerance=1.1e-5)


def test_provisional_receipt_is_o_excl_read_only_and_sidecar_bound(tmp_path):
    path = tmp_path / "provisional.json"
    payload = {
        "program_id": gpu.PROGRAM_ID,
        "receipt_kind": gpu.RECEIPT_KIND,
        "status": "completed",
    }

    receipt, sidecar, digest = gpu.write_provisional_receipt(path, payload)

    assert receipt == path
    assert sidecar == Path(f"{path}.sha256")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert sidecar.read_text(encoding="ascii") == f"{digest}  provisional.json\n"
    assert stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
    assert stat.S_IMODE(sidecar.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
    with pytest.raises(gpu.GPUReplayError, match="will not be overwritten"):
        gpu.write_provisional_receipt(path, {"replacement": True})


def test_gpu_failure_receipt_carries_explicit_provisional_kind(tmp_path):
    receipt, sidecar, _ = gpu._failure_receipt(
        Path("/tmp/a11-test-repo"), tmp_path, gpu.PARITY_MODE, RuntimeError("CUDA unavailable")
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))

    assert sidecar.is_file()
    assert data["receipt_kind"] == "PROVISIONAL_GPU_REPLAY"
    assert data["status"] == "blocked"
    assert data["formal_test_nwb_access"] is False
    assert data["training_nwb_access"] is False
