#!/usr/bin/env python3
"""Audit teacher-attention spectra before resizing decoupled K/V projections.

This is a checkpoint-only diagnostic.  It opens no neural dataset, validation
session, or formal held-out session.  The checkpoint is a trusted local
Lightning artifact, so loading it necessarily uses ``weights_only=False``:
the saved hyperparameters contain the original ``SpintModel`` object.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch


DEFAULT_RANKS = (16, 32, 48, 50, 64, 96, 128, 256)
DEFAULT_CANDIDATES = (
    (32, 32),
    (48, 32),
    (48, 48),
    (48, 64),
    (50, 64),
    (32, 64),
    (50, 96),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def energy_receipt(matrix: torch.Tensor, ranks: Iterable[int]) -> dict:
    singular_values = torch.linalg.svdvals(matrix.float())
    energy = singular_values.square()
    total = energy.sum()
    return {
        "shape": list(matrix.shape),
        "stable_rank": float(total / singular_values[0].square()),
        "top_singular_value": float(singular_values[0]),
        "cumulative_energy_fraction": {
            str(rank): float(energy[:rank].sum() / total)
            for rank in ranks
            if rank <= min(matrix.shape)
        },
    }


def decoder_cost_receipt(
    *,
    batch_size: int,
    num_units: int,
    num_covariates: int,
    window_size: int,
    model_dim: int,
    feedforward_dim: int,
    key_dim: int,
    value_dim: int,
) -> dict:
    batch = batch_size
    units = num_units
    covariates = num_covariates
    window = window_size
    model = model_dim
    feedforward = feedforward_dim

    query_readin = batch * covariates * (
        window * model + model * model
    )
    ffn = batch * covariates * 2 * model * feedforward
    output_readout = batch * covariates * model * window
    source_readin = batch * units * (
        window * model + model * model
    )
    coupled_qkv = batch * (
        covariates * model * model + 2 * units * model * model
    )
    coupled_attention_output = batch * covariates * model * model
    coupled_scores = batch * covariates * units * model
    coupled_total = (
        source_readin
        + query_readin
        + coupled_qkv
        + coupled_attention_output
        + 2 * coupled_scores
        + ffn
        + output_readout
    )

    decoupled_total = (
        query_readin
        + batch * covariates * model * key_dim
        + batch * units * window * value_dim
        + batch * covariates * units * key_dim
        + batch * covariates * units * value_dim
        + batch * covariates * value_dim * model
        + ffn
        + output_readout
    )
    return {
        "key_dim": key_dim,
        "value_dim": value_dim,
        "coupled_macs_per_frame": coupled_total,
        "decoupled_macs_per_frame": decoupled_total,
        "online_mac_reduction_fraction_vs_coupled": (
            1.0 - decoupled_total / coupled_total
        ),
        "persistent_static_key_width": key_dim,
        "persistent_state_reduction_fraction_vs_E50": (
            1.0 - key_dim / window
        ),
        "persistent_state_nonincreasing_vs_E50": key_dim <= window,
    }


def audit(
    checkpoint: Path,
    *,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    candidates: tuple[tuple[int, int], ...] = DEFAULT_CANDIDATES,
    reference_num_units: int = 64,
) -> dict:
    # The local checkpoint is produced by this repository and is explicitly
    # supplied by the caller. See the module docstring for the load boundary.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing a Lightning state_dict")

    prefix = "net.transformer.layers.0.cross_attn."
    in_projection = state.get(prefix + "in_proj_weight")
    out_projection = state.get(prefix + "out_proj.weight")
    rep = state.get("net.rep")
    ffn_weight = state.get("net.transformer.layers.0.ffn.0.weight")
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (in_projection, out_projection, rep, ffn_weight)
    ):
        raise ValueError("checkpoint does not contain the expected SPINT tensors")
    if in_projection.ndim != 2 or in_projection.shape[0] != 3 * in_projection.shape[1]:
        raise ValueError("unexpected attention in-projection shape")
    assert isinstance(in_projection, torch.Tensor)
    assert isinstance(out_projection, torch.Tensor)
    assert isinstance(rep, torch.Tensor)
    assert isinstance(ffn_weight, torch.Tensor)

    query, key, value = in_projection.chunk(3, dim=0)
    model_dim = int(query.shape[0])
    num_covariates = int(rep.shape[1])
    window_size = int(rep.shape[2])
    feedforward_dim = int(ffn_weight.shape[0])
    matrices = {
        "Wq": query,
        "Wk": key,
        "Wv": value,
        "Wo": out_projection,
        "Wo_at_Wv": out_projection @ value,
    }
    return {
        "schema_version": 1,
        "purpose": "checkpoint_only_decoupled_kv_rank_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "data_access_receipt": {
            "neural_datasets_opened": 0,
            "train_sessions_opened": 0,
            "validation_sessions_opened": 0,
            "formal_sessions_opened": 0,
        },
        "teacher_shape": {
            "model_dim": model_dim,
            "num_covariates": num_covariates,
            "window_size": window_size,
            "feedforward_dim": feedforward_dim,
        },
        "spectral_energy": {
            name: energy_receipt(matrix, ranks)
            for name, matrix in matrices.items()
        },
        "configured_cost_reference_n64": [
            decoder_cost_receipt(
                batch_size=1,
                num_units=reference_num_units,
                num_covariates=num_covariates,
                window_size=window_size,
                model_dim=model_dim,
                feedforward_dim=feedforward_dim,
                key_dim=key_dim,
                value_dim=value_dim,
            )
            for key_dim, value_dim in candidates
        ],
        "interpretation_boundary": (
            "Singular-value energy is a checkpoint-only architecture diagnostic, "
            "not a decoding-R2 result and not proof that a particular low-rank "
            "factorization will optimize successfully."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-num-units", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.reference_num_units <= 0:
        raise ValueError("--reference-num-units must be positive")
    result = audit(
        args.checkpoint,
        reference_num_units=args.reference_num_units,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
