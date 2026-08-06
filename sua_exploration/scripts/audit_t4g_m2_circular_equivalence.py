#!/usr/bin/env python3
"""CPU-only native-M2 receipt proving circular AFC4 reduces exactly to T4."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(SCE))

from src.data.afc4_features import (  # noqa: E402
    circular_afc4_from_trial_sums,
    deterministic_afc4_row_permutation,
    fit_train_circular_afc4_stats,
)
from src.data.falcon_t4_features import (  # noqa: E402
    deterministic_row_permutation,
    fit_train_t4_stats,
    t4_from_trial_sums,
)
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder  # noqa: E402


DEFAULT_CHECKPOINTS = {
    24: SCE
    / "outputs/streaming_calibration/m2_m24_disjoint_source_v1_t4_m2_f1_s42_20260731_220722/checkpoints/best.ckpt",
    33: SCE
    / "outputs/streaming_calibration/m2_m33_disjoint_replay_correction_v1_t4_m2_final1_f1_s42_20260801_120203/checkpoints/best.ckpt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder(checkpoint_path: Path) -> tuple[SideFeatureEarlyPoolEncoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    encoder = SideFeatureEarlyPoolEncoder(
        trial_length=int(hparams["trial_length"]),
        window_size=int(hparams["window_size"]),
        hidden_dim=int(hparams["hidden_dim"]),
        side_dim=int(hparams["side_dim"]),
        electrode_embed_dim=int(hparams.get("electrode_embed_dim", 0)),
        num_electrodes=int(hparams.get("num_electrodes", 0)),
    ).eval()
    prefix = "student.id_encoder."
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    encoder.load_state_dict(state, strict=True)
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "variant": hparams["variant"],
        "trial_length": int(hparams["trial_length"]),
        "window_size": int(hparams["window_size"]),
        "hidden_dim": int(hparams["hidden_dim"]),
        "side_dim": int(hparams["side_dim"]),
        "datamodule": {
            key: checkpoint.get("datamodule_hyper_parameters", {}).get(key)
            for key in ("calibration_n_trials", "loso_fold", "side_feature_group", "query_start_trial")
        },
    }
    return encoder, metadata


def build_datamodule(budget: int):
    experiment = "b3s_t4_m2_m24_loso_internal" if budget == 24 else "b3s_t4_m2_loso_internal"
    overrides = [
        f"experiment={experiment}",
        "data.loso_fold=1",
        f"data.calibration_n_trials={budget}",
        "data.random_calibration=false",
        "data.include_heldout_in_fit=false",
        "data.include_heldout_in_test=false",
        "seed=42",
        f"paths.root_dir={SCE}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(SCE / "configs")):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    if dm.val_heldout_dataset is not None:
        raise RuntimeError("Circular-equivalence audit must not construct a held-out dataset")
    observed = {
        "task": str(cfg.data.task),
        "calibration_n_trials": int(cfg.data.calibration_n_trials),
        "validation_protocol": str(cfg.data.validation_protocol),
        "loso_fold": int(cfg.data.loso_fold),
        "side_feature_group": str(cfg.data.side_feature_group),
        "random_calibration": bool(cfg.data.random_calibration),
        "include_heldout_in_fit": bool(cfg.data.include_heldout_in_fit),
        "include_heldout_in_test": bool(cfg.data.include_heldout_in_test),
        "train_sessions": list(dm.train_session_names),
        "validation_sessions": list(dm.val_heldin_session_names),
    }
    return dm, observed, overrides


def audit_budget(budget: int, checkpoint_path: Path) -> dict:
    dm, config, overrides = build_datamodule(budget)
    encoder, checkpoint_metadata = load_encoder(checkpoint_path)
    train = dm.train_dataset
    train_sessions = list(dm.train_session_names)
    sums, lengths, angles = train.native_t4_statistics_inputs(train_sessions)

    legacy_mean, legacy_std = fit_train_t4_stats(
        sums, lengths, angles, train_sessions, budget
    )
    afc_mean, afc_std = fit_train_circular_afc4_stats(
        sums, lengths, angles, train_sessions, budget
    )
    dm_norm = dm.native_t4_normalization
    normalization = {
        "legacy_vs_afc_mean_max_abs": float(np.max(np.abs(legacy_mean - afc_mean))),
        "legacy_vs_afc_std_max_abs": float(np.max(np.abs(legacy_std - afc_std))),
        "datamodule_vs_afc_mean_max_abs": float(np.max(np.abs(dm_norm["mean"] - afc_mean))),
        "datamodule_vs_afc_std_max_abs": float(np.max(np.abs(dm_norm["std"] - afc_std))),
        "mean": afc_mean.tolist(),
        "std": afc_std.tolist(),
    }

    session_rows = []
    maxima = {
        "raw_descriptor_max_abs": 0.0,
        "normalized_descriptor_max_abs": 0.0,
        "row_shuffled_descriptor_max_abs": 0.0,
        "encoder_aligned_max_abs": 0.0,
        "encoder_row_shuffled_max_abs": 0.0,
    }
    session_datasets = [
        *((name, train, "source_train") for name in train_sessions),
        *((name, dm.val_heldin_dataset, "source_validation") for name in dm.val_heldin_session_names),
    ]
    with torch.no_grad():
        for session_name, dataset, split in session_datasets:
            session_sums = dataset.calib_trial_spike_sums[session_name][:budget]
            session_lengths = dataset.calib_trial_lengths[session_name][:budget]
            session_angles = dataset.calib_trial_target_angles[session_name][:budget]
            legacy_raw = t4_from_trial_sums(
                session_sums, session_lengths, session_angles, source=f"{session_name}[0:{budget}]"
            )
            afc_raw = circular_afc4_from_trial_sums(
                session_sums,
                session_lengths,
                session_angles,
                source=f"{session_name}[0:{budget}]",
                ridge=0.0,
            )
            legacy_norm = ((legacy_raw - legacy_mean) / legacy_std).astype(np.float32)
            afc_norm = ((afc_raw - afc_mean) / afc_std).astype(np.float32)
            legacy_perm = deterministic_row_permutation(
                legacy_norm.shape[0], session_name=session_name, seed=42
            )
            afc_perm = deterministic_afc4_row_permutation(
                afc_norm.shape[0], session_name=session_name, seed=42
            )
            legacy_shuffled = legacy_norm[legacy_perm]
            afc_shuffled = afc_norm[afc_perm]

            calibration = np.asarray(
                dataset.calib_trialized_neural_features[session_name][:budget], dtype=np.float32
            )
            calibration_tensor = torch.from_numpy(calibration).unsqueeze(0)
            legacy_output = encoder.forward_batch(
                calibration_tensor, side_features=torch.from_numpy(legacy_norm).unsqueeze(0)
            )
            afc_output = encoder.forward_batch(
                calibration_tensor, side_features=torch.from_numpy(afc_norm).unsqueeze(0)
            )
            legacy_shuffle_output = encoder.forward_batch(
                calibration_tensor, side_features=torch.from_numpy(legacy_shuffled).unsqueeze(0)
            )
            afc_shuffle_output = encoder.forward_batch(
                calibration_tensor, side_features=torch.from_numpy(afc_shuffled).unsqueeze(0)
            )

            usable = np.isfinite(session_angles)
            design = np.stack(
                [
                    np.ones(int(usable.sum()), dtype=np.float64),
                    np.cos(session_angles[usable]),
                    np.sin(session_angles[usable]),
                ],
                axis=1,
            )
            errors = {
                "raw_descriptor_max_abs": float(np.max(np.abs(legacy_raw - afc_raw))),
                "normalized_descriptor_max_abs": float(np.max(np.abs(legacy_norm - afc_norm))),
                "row_shuffled_descriptor_max_abs": float(
                    np.max(np.abs(legacy_shuffled - afc_shuffled))
                ),
                "encoder_aligned_max_abs": float(torch.max(torch.abs(legacy_output - afc_output))),
                "encoder_row_shuffled_max_abs": float(
                    torch.max(torch.abs(legacy_shuffle_output - afc_shuffle_output))
                ),
            }
            for key, value in errors.items():
                maxima[key] = max(maxima[key], value)
            session_rows.append(
                {
                    "session": session_name,
                    "scope": split,
                    "num_channels": int(session_sums.shape[1]),
                    "support_trials": budget,
                    "valid_directional_trials": int(usable.sum()),
                    "unlabeled_centre_or_rest_trials": int((~usable).sum()),
                    "unique_directions_1e6": int(np.unique(np.round(session_angles[usable], 6)).size),
                    "design_rank": int(np.linalg.matrix_rank(design)),
                    "design_condition": float(np.linalg.cond(design)),
                    "total_exposure_bins": int(np.asarray(session_lengths).sum()),
                    "errors": errors,
                }
            )

    all_errors = [*maxima.values(), *[normalization[key] for key in normalization if key.endswith("max_abs")]]
    passed = all(value == 0.0 for value in all_errors)
    if not passed:
        raise RuntimeError(f"M={budget} circular AFC4 reduction is not exact: {maxima}, {normalization}")
    return {
        "budget": budget,
        "passed": passed,
        "config": config,
        "hydra_overrides": overrides,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            **checkpoint_metadata,
        },
        "normalization": normalization,
        "max_errors": maxima,
        "sessions": session_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "sua_exploration/results/t4g_m2_circular_equivalence_v1/audit.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing receipt: {args.out}")

    torch.set_num_threads(1)
    budgets = [audit_budget(budget, DEFAULT_CHECKPOINTS[budget]) for budget in (24, 33)]
    payload = {
        "schema_version": 1,
        "programme": "T4G_generalized_analytic_functional_carrier",
        "purpose": "native_M2_circular_AFC4_exact_reduction_to_T4",
        "created_local_date": "2026-08-06",
        "execution": "CPU_only",
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "query_labels_or_covariates_used": False,
        "target_backpropagation_used": False,
        "estimator": {
            "basis": ["cos(target_angle)", "sin(target_angle)"],
            "ridge_lambda": 0.0,
            "valid_mask": "finite target angle; M2 centre/rest remains unlabeled",
            "descriptor": ["w_cos", "w_sin", "l2_norm_w", "baseline_rate"],
            "normalizer": "source-train sessions only, first chronological M trials",
        },
        "tolerances": {
            "descriptor_max_abs": 0.0,
            "normalizer_max_abs": 0.0,
            "encoder_output_max_abs": 0.0,
        },
        "source_files": {
            "legacy_t4": {
                "path": str((SCE / "src/data/falcon_t4_features.py").resolve()),
                "sha256": sha256(SCE / "src/data/falcon_t4_features.py"),
            },
            "afc4": {
                "path": str((SCE / "src/data/afc4_features.py").resolve()),
                "sha256": sha256(SCE / "src/data/afc4_features.py"),
            },
            "encoder": {
                "path": str((SCE / "src/models/components/streaming_encoders.py").resolve()),
                "sha256": sha256(SCE / "src/models/components/streaming_encoders.py"),
            },
        },
        "budgets": budgets,
        "overall_passed": all(row["passed"] for row in budgets),
        "interpretation": (
            "AFC4 with the circular two-coordinate task basis and lambda=0 is an exact "
            "implementation-level specialisation of native T4 at both audited budgets. "
            "This is a correctness result, not evidence of additional decoding accuracy."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
