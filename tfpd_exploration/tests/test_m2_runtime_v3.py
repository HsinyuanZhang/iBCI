"""Parity and contract tests for the isolated M2 runtime v3."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder, RuntimeV3Error, _reference_module


ROOT = Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions")
PAYLOADS = {
    "small": ROOT / "evalai_m2_small_trf_e_opt_v1/artifacts/m2_small_trf_s1_ema_e19.pkl",
    "large": ROOT / "evalai_m2_large_trf_pick_v1/artifacts/m2_large_trf_s2_raw_e20.pkl",
}
TAGS = ["Run1_20201019", "Run2_20201019", "Run1_20201020", "Run2_20201020", "Run1_20201027", "Run2_20201027", "Run1_20201028"]


def _reference_engine(payload: Path, tags: list[str]):
    ref = _reference_module()
    value = ref.load_payload(payload)
    model = ref.build_decoder(value["kind"]).eval()
    model.load_state_dict({name: torch.as_tensor(x, dtype=torch.float32) for name, x in value["state_dict"].items()}, strict=True)
    banks = [ref.SessionBank(E0=torch.as_tensor(value["bank_by_dataset_tag"][tag]["E0"], dtype=torch.float32), T=torch.as_tensor(value["bank_by_dataset_tag"][tag]["T"], dtype=torch.float32), unit_mask=torch.as_tensor(value["bank_by_dataset_tag"][tag]["unit_mask"], dtype=torch.bool)) for tag in tags]
    engine = ref._ExactEEngine(model, banks)
    engine.rebuild(torch.zeros((len(tags), 50, 96), dtype=torch.float32))
    return engine


@pytest.mark.parametrize("kind", ["small", "large"])
def test_final_q_projection_matches_frozen_exact_e_startup_and_steady_state(kind: str):
    torch.set_num_threads(1)
    tags = TAGS[:1]
    got = RuntimeV3Decoder(PAYLOADS[kind], batch_size=1)
    got.reset(tags)
    values = np.random.default_rng(55).poisson(0.35, size=(51, 1, 96)).astype(np.float32)
    raw = np.zeros((1, 50, 96), dtype=np.float32)
    # Explicitly cover zero/startup 1/3/4/5 and the W-1/W/W+1 boundary.
    checkpoints = {0, 1, 3, 4, 5, 49, 50}
    for step, row in enumerate(values):
        raw = np.concatenate((raw[:, 1:], row[:, None]), axis=1)
        with torch.inference_mode():
            expected = got.model.forward_last(
                torch.from_numpy(raw), got._engine.bank, got._engine.bank.unit_mask
            ).numpy() / 5.0
        actual = got.predict(row)
        if step in checkpoints:
            np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
        else:
            assert float(np.max(np.abs(actual - expected))) <= 1.0e-5


def test_b7_heterobanks_reset_mask_roster_dtype_and_static_state_contracts():
    torch.set_num_threads(1)
    runtime = RuntimeV3Decoder(PAYLOADS["small"], batch_size=7)
    runtime.reset(TAGS)
    row = np.zeros((7, 96), dtype=np.float32)
    assert runtime.predict(row).shape == (7, 2)
    state = runtime.state_bytes()
    assert state.dynamic_owned_tensor_bytes == 7 * 50 * (96 + 256) * 4
    assert state.host_observation_bytes == 7 * 96 * 4
    assert state.static_model_tensor_bytes > 0 and state.static_active_bank_tensor_bytes > 0
    runtime.reset(TAGS[:1])
    assert runtime.predict(np.zeros((1, 96), dtype=np.float32)).shape == (7, 2)
    with pytest.raises(RuntimeV3Error, match="float32"):
        runtime.predict(np.zeros((1, 96), dtype=np.float64))
    with pytest.raises(RuntimeV3Error, match="C-contiguous"):
        runtime.predict(np.zeros((1, 192), dtype=np.float32)[:, ::2])
    with pytest.raises(RuntimeV3Error, match="unknown frozen"):
        runtime.reset(["not-a-frozen-bank"])


def test_mutation_versions_refresh_cache_or_fail_closed():
    torch.set_num_threads(1)
    runtime = RuntimeV3Decoder(PAYLOADS["small"], batch_size=1)
    runtime.reset(TAGS[:1])
    row = np.random.default_rng(4).random((1, 96), dtype=np.float32)
    first = runtime.predict(row)
    # A bank/order-sensitive in-place mutation must not retain cached frontend
    # values; output changes after the automatic refresh.
    with torch.no_grad():
        runtime._engine.bank.E0.copy_(runtime._engine.bank.E0.flip(1))
    changed = runtime.predict(row)
    assert not np.allclose(first, changed)
    # A weight mutation likewise invalidates and recomputes static projections.
    with torch.no_grad():
        runtime.model.frontend.token_mlp[0].weight.add_(1.0e-4)
    assert np.isfinite(runtime.predict(row)).all()
    # Unsupported dtype migration is fail-closed rather than reusing CPU fp32 state.
    runtime.model.to(dtype=torch.float64)
    with pytest.raises(RuntimeV3Error, match="CPU float32"):
        runtime.predict(row)
    # Inference-mode tensors have no usable version counter and are rejected.
    other = RuntimeV3Decoder(PAYLOADS["small"], batch_size=1)
    other.reset(TAGS[:1])
    with torch.inference_mode():
        other._engine.static_token = torch.zeros_like(other._engine.static_token)
    with pytest.raises(RuntimeV3Error, match="inference_mode"):
        other.predict(row)


def test_b7_session_row_permutation_and_mask_mutation_do_not_reuse_stale_cache():
    torch.set_num_threads(1)
    rows = np.random.default_rng(9).poisson(0.3, size=(2, 7, 96)).astype(np.float32)
    runtime = RuntimeV3Decoder(PAYLOADS["small"], batch_size=7)
    runtime.reset(TAGS)
    runtime.predict(rows[0])
    permutation = torch.arange(6, -1, -1)
    with torch.no_grad():
        for value in (runtime._engine.bank.E0, runtime._engine.bank.T, runtime._engine.bank.unit_mask):
            value.copy_(value[permutation])
    actual = runtime.predict(rows[1])
    fresh = RuntimeV3Decoder(PAYLOADS["small"], batch_size=7)
    fresh.reset(list(reversed(TAGS)))
    fresh.predict(rows[0])
    expected = fresh.predict(rows[1])
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    old_signature = runtime._engine._signature
    with torch.no_grad():
        runtime._engine.bank.unit_mask[0, 0] = False
    assert np.isfinite(runtime.predict(rows[1])).all()
    assert runtime._engine._signature != old_signature


@pytest.mark.parametrize("kind", ["small", "large"])
def test_train_mode_is_rejected_before_stream_state_changes_and_eval_restores(kind: str):
    torch.set_num_threads(1)
    row = np.random.default_rng(14).random((1, 96), dtype=np.float32)
    runtime = RuntimeV3Decoder(PAYLOADS[kind], batch_size=1)
    runtime.model.train()
    with pytest.raises(RuntimeV3Error, match="model.eval"):
        runtime.reset(TAGS[:1])
    assert runtime._engine is None
    runtime.model.eval(); runtime.reset(TAGS[:1])
    expected = runtime.predict(row.copy())
    raw_before = runtime._engine.raw.clone()
    runtime.model.train()
    with pytest.raises(RuntimeV3Error, match="model.eval"):
        runtime.predict(row)
    torch.testing.assert_close(runtime._engine.raw, raw_before)
    runtime.model.eval()
    restored = RuntimeV3Decoder(PAYLOADS[kind], batch_size=1)
    restored.reset(TAGS[:1])
    np.testing.assert_allclose(expected, restored.predict(row.copy()), atol=1.0e-5, rtol=1.0e-5)
