#!/usr/bin/env python3
"""CPU-only fail-closed feasibility receipt for M2 M24 B3TS+aligned-T4."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
import hydra

ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(SCE))
from src.models.components.streaming_encoders import TemporalBasisSideFeatureEarlyPoolEncoder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stream(enc, calib, lengths, side):
    state = enc.reset_stream(calib.shape[0], calib.shape[-1], calib.device, calib.dtype)
    state["side_features"] = side
    for m in range(calib.shape[1]):
        state = enc.start_trial(state, lengths[:, m])
        for t in range(int(lengths[:, m].max().item())):
            state = enc.push_sample(state, calib[:, m, t], t)
        state = enc.end_trial(state)
    return enc.finalize_identity(state), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "sua_exploration/results/m2_m24_b3ts_t4_v1/feasibility.json")
    args = parser.parse_args()
    config_dir = SCE / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train.yaml", overrides=["experiment=b3ts_t4_m2_m24_loso_internal", "seed=42", f"paths.root_dir={SCE}"])
    expected = {"variant": "B3TS", "side_dim": 4, "M": 24, "window": 50, "trial": 100,
                "heldout_fit": False, "heldout_test": False, "random": False, "fold": 1, "side": "t4"}
    observed = {"variant": cfg.model.variant, "side_dim": cfg.model.side_dim, "M": cfg.data.calibration_n_trials,
                "window": cfg.data.window_size, "trial": cfg.data.max_trial_length,
                "heldout_fit": cfg.data.include_heldout_in_fit, "heldout_test": cfg.data.include_heldout_in_test,
                "random": cfg.data.random_calibration, "fold": cfg.data.loso_fold, "side": cfg.data.side_feature_group}
    if observed != expected:
        raise ValueError(f"B3TS M24 config contract mismatch: {observed}")
    torch.manual_seed(20260731)
    encoder = TemporalBasisSideFeatureEarlyPoolEncoder(100, 50, hidden_dim=64, side_dim=4).eval()
    calib = torch.randn(2, 3, 100, 5)
    lengths = torch.tensor([[100, 73, 41], [88, 52, 0]])
    for b in range(2):
        for m in range(3):
            calib[b, m, int(lengths[b, m]):] = 1_000.0
    side = torch.randn(2, 5, 4)
    with torch.no_grad():
        batch = encoder.forward_batch(calib, trial_lengths=lengths, side_features=side)
        streamed, stream_state = run_stream(encoder, calib, lengths, side)
    max_error = float((batch - streamed).abs().max())
    if max_error > 2.0e-5:
        raise ValueError(f"B3TS dense/bin-stream mismatch: {max_error}")
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    sample = next(iter(dm.train_dataloader()))
    _, _, native_calib, _, native_t4 = sample
    if native_calib.shape[1:] != (24, 100, native_calib.shape[-1]) or native_t4.shape[-1] != 4:
        raise ValueError("M2 native T4 batch does not match [B,24,100,N] + [B,N,4]")
    with torch.no_grad():
        cached = encoder.forward_batch(native_calib[:1], side_features=native_t4[:1])
        state = encoder.reset_stream(1, native_calib.shape[-1], native_calib.device, native_calib.dtype)
        state["side_features"] = native_t4[:1]
        for m in range(24):
            state = encoder.push_trial(state, native_calib[:1, m])
        finalized = encoder.finalize_identity(state)
    cache_error = float((cached - finalized).abs().max())
    if cache_error != 0.0:
        raise ValueError(f"B3TS T4 cache/finalize mismatch: {cache_error}")
    profile = encoder.cost_profile(num_neurons=96, trial_length=100, num_trials=24)
    if profile.trial_buffer_bytes != 96 * 12 * 4 or profile.support_state_bytes != 96 * 64 * 4:
        raise ValueError("B3TS cost/state receipt mismatch")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "purpose": "CPU_feasibility_for_M2_M24_B3TS_aligned_T4_only",
               "formal_heldout_evaluated": False, "hidden_evalai_evaluated": False,
               "config": observed, "config_path": str((config_dir / "experiment/b3ts_t4_m2_m24_loso_internal.yaml").resolve()),
               "config_sha256": sha256(config_dir / "experiment/b3ts_t4_m2_m24_loso_internal.yaml"),
               "dense_bin": {"variable_lengths": lengths.tolist(), "max_abs_error": max_error, "passed": True},
               "native_t4_cache_finalize": {"calib_shape": list(native_calib.shape), "side_shape": list(native_t4.shape), "max_abs_error": cache_error, "passed": True},
               "encoder_cost": profile.__dict__, "state_contract": {"basis_count": 12, "trial_buffer_bytes": profile.trial_buffer_bytes,
               "support_state_bytes": profile.support_state_bytes, "no_full_trial_persistent_state": True},
               "label_budget": "same 24 chronological calibration trials; trial-level target-direction T4 labels only; no K4/continuous velocity labels"}
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
