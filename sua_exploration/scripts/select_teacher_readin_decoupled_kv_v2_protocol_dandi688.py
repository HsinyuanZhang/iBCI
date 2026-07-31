"""Validation-only fixed protocol for teacher-readin decoupled K/V v2.

Only train and validation paths from the strict manifest are resolved.  Formal
test entries remain names and are never opened, counted or evaluated.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import torch

_SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SUA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dandi688_gradient_free_protocol import (  # noqa: E402
    canonical_direction_key,
    select_calibration_trial_indices,
)
from decoupled_kv_v2_runtime import (  # noqa: E402
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    checkpoint_v2_config,
    load_frozen_v2_model,
)
from eval_adaptation_dandi688 import (  # noqa: E402
    attach_side_features,
    build_calib_trials_for_indices,
    eval_r2,
    load_session_with_trials,
    load_side_feature_stats_for_run_metadata,
    make_subset_dataset,
)
from mc_maze.multisession_datamodule import (  # noqa: E402
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    nwb_unit_count,
    session_name_from_path,
)
from mc_maze.unit_side_features import (  # noqa: E402
    feature_semantics_version,
    side_feature_stats_sha256,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_session_configs(
    rec: dict,
    configs: list[tuple[str, int]],
    pool_size: int,
    model,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Evaluate one loaded validation session under fixed calibration configs."""
    if len(rec["trials"]) <= pool_size:
        raise ValueError(
            f"{rec['name']}: no evaluation trial after pool_size={pool_size}"
        )
    eval_trials = rec["trials"][pool_size:]
    session_r2: dict[str, float] = {}
    session_selections: dict[str, dict] = {}
    for mode, calibration_n in configs:
        indices = select_calibration_trial_indices(
            rec["trials"], calibration_n, pool_size, mode
        )
        rec["calib_trials"] = build_calib_trials_for_indices(
            rec, indices, calibration_n
        )
        eval_ds = make_subset_dataset(rec, eval_trials, rec["name"])
        if not len(eval_ds):
            raise ValueError(
                f"{rec['name']}: common evaluation trials have no windows"
            )
        config_name = (
            f"gradient_free_calibrated_{mode}_n{calibration_n}"
        )
        session_r2[config_name] = eval_r2(model, eval_ds, device)
        session_selections[config_name] = {
            "usable_trial_list_indices": indices,
            "original_trial_indices": [
                rec["trials"][index]["trial_index"] for index in indices
            ],
            "direction_keys": [
                repr(canonical_direction_key(rec["trials"][index]))
                for index in indices
            ],
        }
    return session_r2, session_selections


def _validate_v2_run_metadata(
    metadata: dict,
    *,
    checkpoint: dict,
    manifest_path: Path,
) -> None:
    if metadata.get("runner_family") != "teacher_readin_decoupled_kv_v2":
        raise ValueError("run metadata is not from the isolated v2 runner")
    if metadata.get("status") != "completed":
        raise ValueError("v2 run metadata must be completed before scoring")
    if metadata.get("held_out_test_evaluated") is not False:
        raise ValueError("v2 validation refuses formal-evaluated metadata")
    if metadata.get("variant") != "B3S":
        raise ValueError("v2 validation requires B3S")
    if (metadata.get("side_features") or {}).get("group") != "t4":
        raise ValueError("v2 validation requires real T4 side features")
    if metadata.get("split_counts") != [27, 6, 6]:
        raise ValueError("v2 validation requires strict 27/6/6 split")
    if metadata.get("max_units_exclusive") != 100:
        raise ValueError("v2 validation requires units<100")
    if str(manifest_path) != metadata.get("train_val_manifest"):
        raise ValueError("strict manifest path differs from v2 metadata")
    if sha256_file(manifest_path) != metadata.get(
        "train_val_manifest_sha256"
    ):
        raise ValueError("strict manifest SHA256 differs from v2 run metadata")
    decoder = metadata.get("decoder_architecture") or {}
    config = checkpoint_v2_config(checkpoint)
    checkpoint_receipt = checkpoint[
        "teacher_readin_decoupled_v2_receipt"
    ]
    if checkpoint_receipt.get("teacher_checkpoint_sha256") != metadata.get(
        "teacher_sha256"
    ):
        raise ValueError(
            "checkpoint teacher SHA256 differs from v2 run metadata"
        )
    expected = {
        "key_mode": config["v2_key_mode"],
        "key_width": config["v2_key_dim"],
        "value_width": config["v2_value_dim"],
        "key_permutation_seed": config["v2_key_permutation_seed"],
    }
    if decoder.get("architecture_family") != (
        "teacher_readin_decoupled_kv_v2"
    ):
        raise ValueError("v2 decoder architecture family is missing")
    for name, value in expected.items():
        if decoder.get(name) != value:
            raise ValueError(
                f"v2 metadata {name}={decoder.get(name)!r} does not "
                f"match checkpoint {value!r}"
            )
    key_mode = config["v2_key_mode"]
    permutation_seed = config["v2_key_permutation_seed"]
    run_seed = metadata.get("seed")
    if key_mode == "e_ts4":
        if permutation_seed != run_seed:
            raise ValueError(
                "e_ts4 direct-key permutation seed must equal run seed"
            )
    elif permutation_seed is not None:
        raise ValueError("non-e_ts4 v2 checkpoint must not have a key seed")
    side = metadata.get("side_features") or {}
    if side.get("permutation_seed") is not None:
        raise ValueError("v2 encoder-side T4 must remain aligned")
    if decoder.get("encoder_side_input") != "aligned_real_t4":
        raise ValueError("v2 encoder_side_input must be aligned_real_t4")
    if decoder.get("direct_t4_branch") != (
        "additive_4_to_48_zero_initialized"
    ):
        raise ValueError("v2 direct T4 branch receipt is inconsistent")


def validate_t4_side_receipt(
    metadata: dict,
    *,
    side_feature_group: str,
    side_pool_size: int,
    permutation_seed,
    side_mean,
    side_std,
) -> None:
    """Bind reconstructed train-only T4 statistics to the training receipt."""
    if (
        side_feature_group != "t4"
        or side_pool_size != 50
        or permutation_seed is not None
    ):
        raise ValueError("v2 scoring requires aligned T4 fitted from pool 50")
    side_receipt = metadata.get("side_features") or {}
    if feature_semantics_version("t4") != side_receipt.get(
        "feature_version"
    ):
        raise ValueError("T4 feature semantics drifted from training receipt")
    if side_feature_stats_sha256(
        side_mean, side_std
    ) != side_receipt.get("normalization_sha256"):
        raise ValueError(
            "T4 normalization SHA256 drifted from training receipt"
        )


def evaluate_fixed_v2_protocol_over_validation_sessions(
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
    """Score one v2 checkpoint on six validation sessions only."""
    if (
        selection_mode != "first"
        or calibration_n != 30
        or pool_size != 50
    ):
        raise ValueError(
            "v2 matched validation protocol fixes first / n=30 / pool=50"
        )
    if (
        variant != "B3S"
        or task != "CO"
        or split_counts != (27, 6, 6)
        or max_units_exclusive != 100
        or signal_view != "sua"
    ):
        raise ValueError("v2 evaluator requires B3S/SUA/CO/27-6-6/units<100")
    train_files, val_files, test_names = load_frozen_train_val_manifest(
        train_val_manifest, data_dir
    )
    if not val_files:
        raise ValueError("strict manifest contains no validation sessions")

    run_dir = ckpt_path.parent.parent
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        str(ckpt_path), map_location="cpu", weights_only=False
    )
    _validate_v2_run_metadata(
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
        raise ValueError("v2 T4 normalization/configuration is missing")
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
    model = load_frozen_v2_model(
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
            selections[rec["name"]] = session_selections[config_name]
    return {
        "per_session_r2": per_session_r2,
        "mean_r2": sum(per_session_r2.values()) / len(per_session_r2),
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
