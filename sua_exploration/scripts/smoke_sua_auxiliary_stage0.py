#!/usr/bin/env python3
"""One-train/fixed-validation CPU smoke for Stage-0 component wiring.

This is explicitly a tensor/gradient contract, not an R² experiment.  It reads
the already-built Stage-0 cache only, uses no behavior labels, and reports no
task-performance claim.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mc_maze.sua_auxiliary_stage0 import ZeroInitLowRankFiLM  # noqa: E402
from scripts.audit_sua_auxiliary_stage0 import _feature_cache_path  # noqa: E402
from mc_maze.multisession_datamodule import discover_nwb_files, chronological_session_split, session_name_from_path  # noqa: E402


def _digest(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _inputs(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(cache_path, allow_pickle=False) as cached:
        rates = cached["rates"].astype(np.float32)
        residual = cached["t4_relative_residual"].astype(np.float32)
        exposure = cached["spike_exposure"].astype(np.float32)
        condition = float(cached["design_condition"])
        rank = float(cached["design_rank"] == 3)
    # Eight activity bins are sufficient to validate B,N,D broadcasting.  The
    # context is source separation only; SNR/template stability are excluded.
    activity = torch.from_numpy(np.log1p(rates[:, :8])).unsqueeze(0)
    context = np.column_stack((residual, np.full(rates.shape[0], np.log1p(condition), np.float32), np.full(rates.shape[0], rank, np.float32), np.log1p(exposure))).astype(np.float32)
    return activity, torch.from_numpy(context).unsqueeze(0)


def main() -> None:
    cache_dir = ROOT / "cache"
    data_dir = ROOT / "data/dandi_000688/sub-C"
    all_files = discover_nwb_files(data_dir, "CO", max_units_exclusive=100)
    train, validation, _ = chronological_session_split(all_files, (27, 6, 6), max_units_exclusive=100)
    train_file, validation_file = train[0], validation[0]  # frozen, chronological first/first contract
    train_cache = _feature_cache_path(cache_dir, train_file, pool_size=50)
    validation_cache = _feature_cache_path(cache_dir, validation_file, pool_size=50)
    if not train_cache.is_file() or not validation_cache.is_file():
        raise FileNotFoundError("Stage-0 caches are required; run audit_sua_auxiliary_stage0.py first")
    torch.manual_seed(42)
    train_activity, train_context = _inputs(train_cache)
    val_activity, val_context = _inputs(validation_cache)
    model = ZeroInitLowRankFiLM(activity_dim=8, confidence_dim=4, rank=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    with torch.no_grad():
        initial_exact_baseline = bool(torch.equal(model(train_activity, train_context), train_activity))
        val_loss_before = float(model(val_activity, val_context).square().mean())
    train_loss = model(train_activity, train_context).square().mean()
    optimizer.zero_grad(); train_loss.backward(); optimizer.step()
    with torch.no_grad():
        val_loss_after = float(model(val_activity, val_context).square().mean())
    output = {
        "schema_version": 1,
        "purpose": "CPU tensor/gradient smoke only; not a behavior-label or R2 experiment",
        "train_session": session_name_from_path(train_file),
        "fixed_validation_session": session_name_from_path(validation_file),
        "source_scope": "Stage-0 caches only; no formal-test cache or NWB contents opened",
        "context_columns": ["t4_relative_residual", "log1p_design_condition", "t4_rank_valid", "log1p_spike_exposure"],
        "excluded_from_context": ["snr", "waveform_residual_cv", "waveform_template_drift"],
        "initial_exact_baseline": initial_exact_baseline,
        "train_loss_after_one_step": float(train_loss),
        "fixed_validation_loss_before": val_loss_before,
        "fixed_validation_loss_after": val_loss_after,
        "train_cache_sha256": _digest(train_cache),
        "validation_cache_sha256": _digest(validation_cache),
        "formal_test_evaluated": False,
    }
    output_path = ROOT / "results/sua_auxiliary_stage0/component_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
