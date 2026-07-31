#!/usr/bin/env python3
"""Train-cache-only audit of joint ``LayerNorm([E,T4])`` in decoupled keys.

The audit reconstructs identity ``E`` from the selected ordinary T4 anchor on
the strict manifest's 27 training sessions.  It requires existing train caches
and refuses to recompute missing features or sessions from NWB.  Validation and
formal paths are recorded from metadata but never resolved or opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from mc_maze.multisession_datamodule import (
    _session_cache_path,
    fit_behavior_stats,
    session_name_from_path,
)
from mc_maze.unit_side_features import (
    _side_feature_cache_path,
    fit_side_feature_stats,
    load_unit_side_features,
    side_feature_stats_sha256,
)
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def normalization_metrics(
    identity: torch.Tensor,
    t4: torch.Tensor,
) -> dict[str, np.ndarray | float]:
    """Measure initial joint-LN T4 energy and the E-only counterfactual drift."""
    if identity.ndim != 2 or t4.ndim != 2:
        raise ValueError("identity and t4 must both have shape [N,D]")
    if identity.shape[0] != t4.shape[0] or t4.shape[1] != 4:
        raise ValueError("identity and four-dimensional t4 must share N")
    key_input = torch.cat([identity, t4], dim=-1)
    zero_input = torch.cat([identity, torch.zeros_like(t4)], dim=-1)
    width = key_input.shape[-1]
    joint = functional.layer_norm(key_input, (width,))
    zero = functional.layer_norm(zero_input, (width,))
    eps = torch.finfo(joint.dtype).eps
    t4_energy_fraction = (
        joint[:, -4:].square().sum(dim=-1)
        / joint.square().sum(dim=-1).clamp_min(eps)
    )
    identity_counterfactual_relative_l2 = (
        (joint[:, :-4] - zero[:, :-4]).norm(dim=-1)
        / joint[:, :-4].norm(dim=-1).clamp_min(eps)
    )
    full_counterfactual_relative_l2 = (
        (joint - zero).norm(dim=-1)
        / joint.norm(dim=-1).clamp_min(eps)
    )
    return {
        "t4_energy_fraction_by_unit": (
            t4_energy_fraction.detach().cpu().numpy().astype(np.float64)
        ),
        "identity_counterfactual_relative_l2_by_unit": (
            identity_counterfactual_relative_l2.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "full_counterfactual_relative_l2_by_unit": (
            full_counterfactual_relative_l2.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        ),
        "global_t4_energy_fraction": float(
            joint[:, -4:].square().sum() / joint.square().sum().clamp_min(eps)
        ),
        "identity_raw_rms": float(identity.square().mean().sqrt()),
        "t4_raw_rms": float(t4.square().mean().sqrt()),
    }


def _build_selected_anchor(
    *,
    checkpoint: Path,
    teacher_checkpoint: Path,
) -> StreamingCalibrationLitModule:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("anchor checkpoint is missing state_dict")
    module = StreamingCalibrationLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path=str(teacher_checkpoint),
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        pad_value=-1.0,
        freeze_decoder=False,
        loss_mode="task_only",
        identity_mode="calibrated",
        side_dim=4,
        compile=False,
    )
    module.setup("fit")
    module.load_state_dict(state, strict=True)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def audit(
    *,
    run_metadata_path: Path,
    checkpoint: Path,
) -> dict:
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("variant") != "B3S":
        raise ValueError("normalization audit requires an ordinary B3S/T4 anchor")
    side = metadata.get("side_features") or {}
    if (
        side.get("group") != "t4"
        or side.get("pool_size") != 50
        or side.get("side_dim") != 4
    ):
        raise ValueError("normalization audit requires T4@50 with side_dim=4")
    training = metadata.get("training") or {}
    if training.get("calibration_n_trials") != 30:
        raise ValueError("normalization audit requires M_activity=30")
    train_files = [
        Path(value) for value in (metadata.get("session_files") or {}).get("train", [])
    ]
    validation_files = (
        (metadata.get("session_files") or {}).get("val", [])
    )
    if len(train_files) != 27 or len(validation_files) != 6:
        raise ValueError("normalization audit requires the strict 27/6 split")
    cache_dir = Path(metadata["cache_dir"])
    teacher_checkpoint = Path(metadata["teacher_checkpoint"])
    for required in (checkpoint, teacher_checkpoint, cache_dir):
        if not required.exists():
            raise FileNotFoundError(required)

    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, 20, cache_dir=cache_dir
    )
    side_mean, side_std = fit_side_feature_stats(
        train_files,
        feature_group="t4",
        pool_size=50,
        cache_dir=cache_dir,
        bin_size_ms=20,
        window_size=50,
        trial_result_filter="R",
        signal_view="sua",
    )
    observed_stats_sha = side_feature_stats_sha256(side_mean, side_std)
    if observed_stats_sha != side.get("normalization_sha256"):
        raise ValueError("train-only T4 normalization SHA differs from run metadata")

    module = _build_selected_anchor(
        checkpoint=checkpoint,
        teacher_checkpoint=teacher_checkpoint,
    )
    assert module.student is not None
    all_t4_energy: list[np.ndarray] = []
    all_identity_drift: list[np.ndarray] = []
    all_full_drift: list[np.ndarray] = []
    per_session: dict[str, dict] = {}

    with torch.no_grad():
        for nwb_path in train_files:
            session = session_name_from_path(nwb_path)
            session_cache = _session_cache_path(
                cache_dir,
                nwb_path,
                bin_size_ms=20,
                window_size=50,
                calibration_n_trials=30,
                max_trial_length=100,
                pad_value=-1.0,
                interpolate_trials=True,
                behavior_mean=behavior_mean,
                behavior_std=behavior_std,
                trial_result_filter="R",
                exclude_calibration_trials_from_windows=False,
                signal_view="sua",
            )
            side_cache = _side_feature_cache_path(
                cache_dir,
                nwb_path,
                feature_group="t4",
                pool_size=50,
                bin_size_ms=20,
                window_size=50,
                trial_result_filter="R",
                signal_view="sua",
            )
            # Fail closed: this diagnostic may consume existing training caches
            # but is not authorized to open/recompute from any NWB.
            if not session_cache.is_file() or not side_cache.is_file():
                raise FileNotFoundError(
                    f"{session}: required train cache is missing; refusing NWB fallback"
                )
            with np.load(session_cache, allow_pickle=False) as cached:
                calibration = cached["calib_trials"].astype(
                    np.float32, copy=False
                )
                signal_view = (
                    str(cached["signal_view"].item())
                    if "signal_view" in cached.files
                    else "sua"
                )
            if calibration.shape[0:2] != (30, 100) or signal_view != "sua":
                raise ValueError(f"{session}: unexpected cached calibration contract")
            normalized_t4, _ = load_unit_side_features(
                nwb_path,
                feature_group="t4",
                pool_size=50,
                mean=side_mean,
                std=side_std,
                cache_dir=cache_dir,
                bin_size_ms=20,
                window_size=50,
                trial_result_filter="R",
                signal_view="sua",
            )
            if normalized_t4.shape[0] != calibration.shape[2]:
                raise ValueError(f"{session}: T4/calibration unit count differs")
            identity = module.student.compute_identity(
                torch.from_numpy(calibration).unsqueeze(0),
                side_features=torch.from_numpy(normalized_t4).unsqueeze(0),
            )[0]
            metrics = normalization_metrics(
                identity,
                torch.from_numpy(normalized_t4),
            )
            t4_energy = metrics.pop("t4_energy_fraction_by_unit")
            identity_drift = metrics.pop(
                "identity_counterfactual_relative_l2_by_unit"
            )
            full_drift = metrics.pop(
                "full_counterfactual_relative_l2_by_unit"
            )
            assert isinstance(t4_energy, np.ndarray)
            assert isinstance(identity_drift, np.ndarray)
            assert isinstance(full_drift, np.ndarray)
            all_t4_energy.append(t4_energy)
            all_identity_drift.append(identity_drift)
            all_full_drift.append(full_drift)
            per_session[session] = {
                "unit_count": int(identity.shape[0]),
                **metrics,
                "t4_energy_fraction": _quantiles(t4_energy),
                "identity_counterfactual_relative_l2": _quantiles(
                    identity_drift
                ),
                "full_counterfactual_relative_l2": _quantiles(full_drift),
            }

    t4_energy = np.concatenate(all_t4_energy)
    identity_drift = np.concatenate(all_identity_drift)
    full_drift = np.concatenate(all_full_drift)
    return {
        "schema_version": 1,
        "purpose": "train_cache_only_decoupled_joint_key_normalization_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_anchor_checkpoint": str(checkpoint.resolve()),
        "selected_anchor_sha256": sha256_file(checkpoint),
        "run_metadata": str(run_metadata_path.resolve()),
        "run_metadata_sha256": sha256_file(run_metadata_path),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "side_feature_normalization_sha256": observed_stats_sha,
        "data_access_receipt": {
            "train_cache_sessions_opened": len(train_files),
            "train_nwb_sessions_opened": 0,
            "validation_cache_sessions_opened": 0,
            "validation_nwb_sessions_opened": 0,
            "formal_sessions_opened": 0,
        },
        "aggregate": {
            "session_count": len(train_files),
            "unit_count": int(t4_energy.size),
            "t4_coordinate_fraction": 4 / 54,
            "t4_energy_fraction_after_joint_layernorm": _quantiles(
                t4_energy
            ),
            "identity_counterfactual_relative_l2_when_t4_zeroed": _quantiles(
                identity_drift
            ),
            "full_key_counterfactual_relative_l2_when_t4_zeroed": _quantiles(
                full_drift
            ),
        },
        "per_session": per_session,
        "interpretation_boundary": (
            "Uses the selected T4 anchor identities and initial affine-free joint "
            "LayerNorm geometry. It diagnoses control entanglement/capacity risk, "
            "not decoding R2 and not the learned final key_norm affine parameters."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        run_metadata_path=args.run_metadata,
        checkpoint=args.checkpoint,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
