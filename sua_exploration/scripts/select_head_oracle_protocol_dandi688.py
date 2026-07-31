"""Validation-only fixed protocol for the exact-head K/V oracle.

Only train and validation paths from the frozen manifest are resolved.  Formal
test entries remain receipt names and are never opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_adaptation_dandi688 import (  # noqa: E402
    attach_side_features,
    load_session_with_trials,
    load_side_feature_stats_for_run_metadata,
)
from head_oracle_runtime import (  # noqa: E402
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    checkpoint_oracle_config,
    load_frozen_oracle_model,
)
from mc_maze.multisession_datamodule import (  # noqa: E402
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    nwb_unit_count,
    session_name_from_path,
)
from select_teacher_readin_decoupled_kv_v2_protocol_dandi688 import (  # noqa: E402
    evaluate_session_configs,
    sha256_file,
    validate_t4_side_receipt,
)


def _validate_oracle_run_metadata(
    metadata: dict,
    *,
    checkpoint: dict,
    manifest_path: Path,
) -> None:
    expected = {
        "runner_family": "teacher_head_preserving_kv_oracle",
        "status": "completed",
        "held_out_test_evaluated": False,
        "variant": "B3S",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(
                f"oracle metadata {name}={metadata.get(name)!r}, "
                f"expected {value!r}"
            )
    side = metadata.get("side_features") or {}
    if (
        side.get("group") != "t4"
        or side.get("pool_size") != 50
        or side.get("side_dim") != 4
        or side.get("permutation_seed") is not None
    ):
        raise ValueError(
            "oracle validation requires aligned real T4@50"
        )
    if str(manifest_path) != metadata.get("train_val_manifest"):
        raise ValueError(
            "strict manifest path differs from oracle metadata"
        )
    if sha256_file(manifest_path) != metadata.get(
        "train_val_manifest_sha256"
    ):
        raise ValueError(
            "strict manifest SHA256 differs from oracle metadata"
        )

    decoder = metadata.get("decoder_architecture") or {}
    config = checkpoint_oracle_config(checkpoint)
    receipt = checkpoint["teacher_head_oracle_receipt"]
    if receipt.get("teacher_checkpoint_sha256") != metadata.get(
        "teacher_sha256"
    ):
        raise ValueError(
            "oracle checkpoint teacher SHA differs from run metadata"
        )
    required_decoder = {
        "architecture_family": (
            "teacher_head_preserving_decoupled_kv_oracle"
        ),
        "active_decoder_mode": (
            "teacher_head_preserving_decoupled_oracle"
        ),
        "base_decoder_mode_argument": "coupled",
        "key_mode": config["oracle_key_mode"],
        "key_width": 512,
        "value_width": 512,
        "attention_heads": 64,
        "head_dim": 8,
        "key_permutation_seed": config[
            "oracle_key_permutation_seed"
        ],
        "direct_t4_branch": "none",
        "encoder_side_input": "aligned_real_t4",
        "headwise_softmax_preserved": True,
        "low_rank_factorization_used": False,
        "head_averaging_used": False,
        "legacy_decoder_transformer_active": False,
        "legacy_decoder_transformer_trainable": False,
    }
    for name, value in required_decoder.items():
        if decoder.get(name) != value:
            raise ValueError(
                f"oracle decoder {name}={decoder.get(name)!r}, "
                f"expected {value!r}"
            )
    mode = config["oracle_key_mode"]
    permutation_seed = config["oracle_key_permutation_seed"]
    if mode == "e_ts4":
        if permutation_seed != metadata.get("seed"):
            raise ValueError(
                "oracle e_ts4 permutation seed must equal run seed"
            )
        if decoder.get("decoder_ts4_control") != (
            "fixed_E_row_permutation_only"
        ):
            raise ValueError(
                "oracle e_ts4 must permute decoder-K E rows only"
            )
    else:
        if permutation_seed is not None:
            raise ValueError(
                "oracle e_t4 must not have a permutation seed"
            )
        if decoder.get("decoder_ts4_control") != "none":
            raise ValueError("oracle e_t4 TS4 receipt is inconsistent")


def evaluate_fixed_oracle_protocol_over_validation_sessions(
    *,
    ckpt_path: Path,
    teacher_ckpt: Path,
    variant: str,
    data_dir: Path,
    task: str,
    split_counts: tuple[int, int, int],
    max_units_exclusive: int,
    cache_dir: Path | None,
    pool_size: int,
    selection_mode: str,
    calibration_n: int,
    signal_view: str,
    train_val_manifest: Path,
    device: torch.device | None = None,
) -> dict:
    """Score one oracle checkpoint on the six validation sessions only."""
    if (
        selection_mode != "first"
        or calibration_n != 30
        or pool_size != 50
    ):
        raise ValueError(
            "oracle validation fixes first / n=30 / pool=50"
        )
    if (
        variant != "B3S"
        or task != "CO"
        or split_counts != (27, 6, 6)
        or max_units_exclusive != 100
        or signal_view != "sua"
    ):
        raise ValueError(
            "oracle evaluator requires B3S/SUA/CO/27-6-6/units<100"
        )
    train_files, val_files, test_names = (
        load_frozen_train_val_manifest(train_val_manifest, data_dir)
    )
    if not val_files:
        raise ValueError(
            "strict manifest contains no validation sessions"
        )

    run_dir = ckpt_path.parent.parent
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        str(ckpt_path), map_location="cpu", weights_only=False
    )
    _validate_oracle_run_metadata(
        metadata,
        checkpoint=checkpoint,
        manifest_path=train_val_manifest,
    )

    mean, std = fit_behavior_stats(
        train_files, 20, cache_dir=cache_dir
    )
    side_feature_config = load_side_feature_stats_for_run_metadata(
        metadata, train_files, cache_dir
    )
    if side_feature_config is None:
        raise ValueError(
            "oracle T4 normalization/configuration is missing"
        )
    (
        side_feature_group,
        waveform_feature_group,
        side_pool_size,
        permutation_seed,
        side_mean,
        side_std,
    ) = side_feature_config
    validate_t4_side_receipt(
        metadata,
        side_feature_group=side_feature_group,
        side_pool_size=side_pool_size,
        permutation_seed=permutation_seed,
        side_mean=side_mean,
        side_std=side_std,
    )
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_frozen_oracle_model(
        ckpt_path, teacher_ckpt, variant, device
    )

    configs = [(selection_mode, calibration_n)]
    config_name = (
        f"gradient_free_calibrated_{selection_mode}_n{calibration_n}"
    )
    per_session_r2: dict[str, float] = {}
    selections: dict[str, dict] = {}
    with torch.no_grad():
        for path in val_files:
            rec = load_session_with_trials(
                path,
                20,
                WINDOW_SIZE,
                pool_size,
                TRIAL_LENGTH,
                PAD_VALUE,
                mean,
                std,
                cache_dir=cache_dir,
                signal_view=signal_view,
            )
            rec = attach_side_features(
                rec,
                path,
                side_feature_group=side_feature_group,
                waveform_feature_group=waveform_feature_group,
                pool_size=side_pool_size,
                permutation_seed=permutation_seed,
                mean=side_mean,
                std=side_std,
                cache_dir=cache_dir,
            )
            session_r2, session_selections = evaluate_session_configs(
                rec, configs, pool_size, model, device
            )
            per_session_r2[rec["name"]] = session_r2[config_name]
            selections[rec["name"]] = session_selections[
                config_name
            ]
    return {
        "per_session_r2": per_session_r2,
        "mean_r2": (
            sum(per_session_r2.values()) / len(per_session_r2)
        ),
        "trial_selections": selections,
        "session_splits": {
            "train": [
                session_name_from_path(path) for path in train_files
            ],
            "val": [
                session_name_from_path(path) for path in val_files
            ],
            "test": test_names,
        },
        "session_unit_counts": {
            session_name_from_path(path): nwb_unit_count(path)
            for path in train_files + val_files
        },
        "formal_test_paths_resolved": False,
        "formal_test_files_opened": 0,
    }
