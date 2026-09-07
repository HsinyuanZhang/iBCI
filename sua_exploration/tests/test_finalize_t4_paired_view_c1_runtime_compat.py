from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = ROOT / "sua_exploration/scripts/finalize_t4_paired_view_c1_runtime_compat.py"


def _load_wrapper():
    name = "c1_runtime_compat_test"
    spec = importlib.util.spec_from_file_location(name, WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WRAPPER = _load_wrapper()


def _full_runtime(
    *,
    hostname: str = "fixture-host",
    torch_uuid_a: str = "A1111111-1111-1111-1111-111111111111",
    smi_uuid_a: str = "GPU-a1111111-1111-1111-1111-111111111111",
) -> dict:
    return {
        "hostname": hostname,
        "pytorch": "fixture-torch",
        "pytorch_cuda": "fixture-cuda",
        "cuda_available": True,
        "torch_gpus": [
            {
                "logical_index": 0,
                "name": "fixture GPU A",
                "uuid": torch_uuid_a,
                "total_memory_bytes": 1000,
            },
            {
                "logical_index": 1,
                "name": "fixture GPU B",
                "uuid": "b2222222-2222-2222-2222-222222222222",
                "total_memory_bytes": 2000,
            },
        ],
        "nvidia_smi": [
            f"0, fixture GPU A, {smi_uuid_a}, fixture-driver, 10 MiB",
            "1, fixture GPU B, GPU-B2222222-2222-2222-2222-222222222222, fixture-driver, 20 MiB",
        ],
    }


def _scoped_cost(
    *,
    hostname: str = "fixture-host",
    visible: str = "1",
    name: str = "fixture GPU B",
    memory: int = 2000,
) -> dict:
    return {
        "hostname": hostname,
        "pytorch": "fixture-torch",
        "pytorch_cuda": "fixture-cuda",
        "cuda_available": True,
        "cuda_visible_devices": visible,
        "torch_visible_devices": [
            {
                "logical_index": 0,
                "name": name,
                "total_memory_bytes": memory,
            }
        ],
        "nvidia_smi_gpus": [
            "0, fixture GPU A, GPU-a1111111-1111-1111-1111-111111111111, fixture-driver",
            "1, fixture GPU B, GPU-b2222222-2222-2222-2222-222222222222, fixture-driver",
        ],
    }


def _adapter():
    frozen = WRAPPER.load_frozen_finalizer()
    return frozen, WRAPPER.RuntimeCompatibilityAdapter(
        frozen.normalize_runtime_environment,
        frozen.need,
    )


def test_prefix_and_case_only_uuid_difference_is_accepted() -> None:
    _frozen, adapter = _adapter()
    normalized = adapter.normalize(_full_runtime(), "fixture/s42 start")
    assert {row["uuid"] for row in normalized["torch_gpus"]} == {
        "a1111111-1111-1111-1111-111111111111",
        "b2222222-2222-2222-2222-222222222222",
    }
    assert {row["uuid"] for row in normalized["nvidia_smi"]} == {
        "a1111111-1111-1111-1111-111111111111",
        "b2222222-2222-2222-2222-222222222222",
    }


def test_genuinely_different_uuid_is_rejected() -> None:
    frozen, adapter = _adapter()
    runtime = _full_runtime(smi_uuid_a="GPU-c3333333-3333-3333-3333-333333333333")
    with pytest.raises(frozen.FinalizationError, match="UUID mismatch"):
        adapter.normalize(runtime, "fixture/s42 start")


def test_legacy_scoped_cost_maps_visible_device_to_start_runtime() -> None:
    _frozen, adapter = _adapter()
    start = adapter.normalize(_full_runtime(), "shared_t4/s42 start")
    cost = adapter.normalize(_scoped_cost(), "shared_t4/s42 cost")
    assert cost == start


def test_legacy_scoped_cost_rejects_wrong_visible_gpu() -> None:
    frozen, adapter = _adapter()
    adapter.normalize(_full_runtime(), "shared_t4/s42 start")
    with pytest.raises(frozen.FinalizationError, match="visible GPU name mismatch"):
        adapter.normalize(
            _scoped_cost(name="fixture GPU A", memory=1000),
            "shared_t4/s42 cost",
        )


def test_legacy_scoped_cost_rejects_memory_mismatch() -> None:
    frozen, adapter = _adapter()
    adapter.normalize(_full_runtime(), "shared_t4/s42 start")
    with pytest.raises(frozen.FinalizationError, match="visible GPU memory mismatch"):
        adapter.normalize(_scoped_cost(memory=2001), "shared_t4/s42 cost")


def test_legacy_scoped_cost_rejects_host_mismatch() -> None:
    frozen, adapter = _adapter()
    adapter.normalize(_full_runtime(), "shared_t4/s42 start")
    with pytest.raises(frozen.FinalizationError, match="hostname mismatch"):
        adapter.normalize(
            _scoped_cost(hostname="different-host"),
            "shared_t4/s42 cost",
        )


def test_wrapper_pins_the_frozen_finalizer_hash() -> None:
    assert WRAPPER.sha256_file(WRAPPER.FROZEN_FINALIZER_PATH) == WRAPPER.FROZEN_FINALIZER_SHA256


def test_install_replaces_only_the_runtime_normalizer() -> None:
    frozen = WRAPPER.load_frozen_finalizer()
    unchanged = {
        name: getattr(frozen, name)
        for name in (
            "validate_cell_evidence",
            "transfer_file_index",
            "validate_data_stamps",
            "finalize_evidence",
            "main",
        )
    }
    original_normalize = frozen.normalize_runtime_environment
    adapter = WRAPPER.install_runtime_compat(frozen)
    assert frozen.normalize_runtime_environment == adapter.normalize
    assert frozen.normalize_runtime_environment != original_normalize
    for name, function in unchanged.items():
        assert getattr(frozen, name) is function
