#!/usr/bin/env python3
"""Checkpoint-only audit for the teacher-readin/SVD decoupled K/V v2.

No neural dataset, cache, validation session, or formal held-out session is
opened.  The explicitly supplied local Lightning checkpoint is trusted and is
loaded with ``weights_only=False`` because its hyperparameters contain the
repository's ``SpintModel`` object.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from src.models.components.decoupled_kv_v2 import (
    TeacherSVDDecoupledCrossAttention,
)

from scripts.audit_decoupled_teacher_low_rank import sha256_file


def audit(
    checkpoint: Path,
    *,
    key_dim: int = 48,
    value_dim: int = 64,
    direct_feature_dim: int = 4,
    reference_num_units: int = 64,
) -> dict:
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    hyperparameters = payload.get("hyper_parameters") or {}
    teacher = hyperparameters.get("net")
    if teacher is None:
        raise ValueError("checkpoint hyperparameters do not contain net")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing a Lightning state_dict")
    teacher_state = {
        name.removeprefix("net."): tensor
        for name, tensor in state.items()
        if name.startswith("net.")
    }
    if not teacher_state:
        raise ValueError("checkpoint has no net.* tensors")
    teacher.load_state_dict(teacher_state, strict=True)
    teacher.eval()
    if teacher.num_layers != 1:
        raise ValueError("v2 audit requires a one-layer teacher")

    legacy = teacher.transformer.layers[0]
    module = TeacherSVDDecoupledCrossAttention(
        d_model=teacher.model_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        direct_feature_dim=direct_feature_dim,
        dim_feedforward=legacy.ffn[0].out_features,
        dropout=teacher.tf_drop_rate,
    )
    initialization = module.initialize_from_teacher(
        teacher_attn=legacy.cross_attn,
        teacher_norm1=legacy.norm1,
        teacher_norm2=legacy.norm2,
        teacher_ffn=legacy.ffn,
    )
    static_cost = module.cost_receipt(
        batch_size=1,
        num_units=reference_num_units,
        num_queries=teacher.num_covariates,
        window_size=teacher.window_size,
        dynamic_activity_key=False,
    )
    dynamic_cost = module.cost_receipt(
        batch_size=1,
        num_units=reference_num_units,
        num_queries=teacher.num_covariates,
        window_size=teacher.window_size,
        dynamic_activity_key=True,
    )
    return {
        "schema_version": 1,
        "purpose": "checkpoint_only_teacher_readin_decoupled_v2_svd_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "data_access_receipt": {
            "neural_datasets_opened": 0,
            "train_caches_opened": 0,
            "train_sessions_opened": 0,
            "validation_sessions_opened": 0,
            "formal_sessions_opened": 0,
        },
        "teacher_shape": {
            "model_dim": teacher.model_dim,
            "num_covariates": teacher.num_covariates,
            "window_size": teacher.window_size,
            "feedforward_dim": legacy.ffn[0].out_features,
            "head_count": legacy.cross_attn.num_heads,
            "head_dim": teacher.model_dim // legacy.cross_attn.num_heads,
            "num_layers": teacher.num_layers,
        },
        "svd_provenance": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
            "factor_hash_is_environment_specific": True,
        },
        "initialization_receipt": initialization,
        "cost_reference": {
            "reference_num_units": reference_num_units,
            "static_e_t4": static_cost,
            "dynamic_x_only": dynamic_cost,
        },
        "interpretation_boundary": (
            "This audit measures checkpoint weight-factorization and configured "
            "cost properties only. It is not a decoding-R2 result, does not "
            "preserve the teacher's head-wise softmaxes, and opens no data."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-dim", type=int, default=48)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--direct-feature-dim", type=int, default=4)
    parser.add_argument("--reference-num-units", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for name in (
        "key_dim",
        "value_dim",
        "direct_feature_dim",
        "reference_num_units",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    result = audit(
        args.checkpoint,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        direct_feature_dim=args.direct_feature_dim,
        reference_num_units=args.reference_num_units,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint_sha256": result["checkpoint_sha256"],
        "teacher_shape": result["teacher_shape"],
        "initialization_receipt": result["initialization_receipt"],
        "cost_reference": result["cost_reference"],
        "data_access_receipt": result["data_access_receipt"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
