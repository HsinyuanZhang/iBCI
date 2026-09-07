#!/usr/bin/env python3
"""CPU-only feasibility audit for a frozen M1 decoder-identity alignment oracle.

This is intentionally not an oracle fit. It derives only the canonical frozen
teacher identity target from first-ten calibration tensors and checks that the
target and prospective controls are well-defined without query labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
DATA = ROOT / "SPINT-main" / "data" / "000941" / "sub-MonkeyL-held-in-calib"
TEACHER = ROOT / "SPINT-main" / "logs" / "train" / "runs" / "2026-07-21-19-11-01" / "checkpoints" / "best_ckpt" / "epoch_019.ckpt"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "M1_DECODER_LATENT_ALIGNMENT_ORACLE_PROTOCOL.md"
D4_HELPER = SCE / "src" / "data" / "falcon_d4_features.py"
SPINT_SOURCE = SCE / "src" / "models" / "components" / "spint.py"
STREAMING_MODULE_SOURCE = SCE / "src" / "models" / "streaming_calibration_module.py"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v1"
SUPPORT = 10
RANK_GRID = (1, 2, 3)
LAMBDA_GRID = (1.0e-4, 1.0e-2, 1.0, 100.0)
LEVELS = (1, 2, 3, 4)
CONTROL_SPECS = (
    "identity_no_alignment",
    "rate_only",
    "condition_label_shuffle",
    "orthogonal_only",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


def validate_rank_lambda_grid(ranks: Sequence[int], lambdas: Sequence[float]) -> None:
    if tuple(ranks) != RANK_GRID:
        raise ValueError(f"decoder-latent oracle rank grid is frozen to {RANK_GRID}, got {tuple(ranks)}")
    if not lambdas or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in lambdas):
        raise ValueError("decoder-latent oracle lambda grid must be finite and strictly positive")


def deterministic_label_shuffle(labels: np.ndarray, *, session_name: str, seed: int) -> np.ndarray:
    """Shuffle exactly the M=10 support labels; preserve their multiset and change order."""
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.shape != (SUPPORT,) or tuple(sorted(set(values.tolist()))) != LEVELS:
        raise ValueError("condition-label shuffle requires all four labels in exactly M=10 support")
    permutation = np.random.RandomState(stable_seed("m1-latent-label-shuffle", session_name, seed)).permutation(SUPPORT)
    if np.array_equal(permutation, np.arange(SUPPORT)):
        permutation = np.roll(permutation, 1)
    return values[permutation]


def support_rate_features(sums: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Label-free per-unit rate/exposure baseline features from exactly first ten trials."""
    x = np.asarray(sums, dtype=np.float64)
    exposure = np.asarray(lengths, dtype=np.float64).reshape(-1)
    if x.shape[0] != SUPPORT or exposure.shape != (SUPPORT,) or x.ndim != 2:
        raise ValueError("rate-only features require [10,N] sums and [10] lengths")
    if np.any(exposure <= 0.0) or not np.all(np.isfinite(x)):
        raise ValueError("invalid support rates")
    rates = x / exposure[:, None]
    return np.column_stack([
        np.log1p(np.maximum(rates.mean(axis=0), 0.0)),
        np.log1p(np.maximum(rates.std(axis=0, ddof=1), 0.0)),
        np.full(rates.shape[1], np.log(exposure.mean())),
    ])


def support_condition_features(sums: np.ndarray, lengths: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Four M=10 condition means, exposure-corrected per trial, no query input."""
    x = np.asarray(sums, dtype=np.float64)
    exposure = np.asarray(lengths, dtype=np.float64).reshape(-1)
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.shape[0] != SUPPORT or exposure.shape != (SUPPORT,) or values.shape != (SUPPORT,):
        raise ValueError("condition features require aligned chronological M=10 support")
    if tuple(sorted(set(values.tolist()))) != LEVELS:
        raise ValueError(f"condition features require levels {LEVELS}")
    rates = x / exposure[:, None]
    return np.stack([rates[values == level].mean(axis=0) for level in LEVELS], axis=1)


@dataclass(frozen=True)
class Orthogonalizer:
    rate_mean: np.ndarray
    rate_std: np.ndarray
    label_mean: np.ndarray
    weights: np.ndarray


def fit_orthogonalizer(rate_train: np.ndarray, label_train: np.ndarray, ridge: float = 1.0e-6) -> Orthogonalizer:
    """Fit projection label~rate on outer-train rows only; no session/query leakage."""
    x = np.asarray(rate_train, dtype=np.float64)
    y = np.asarray(label_train, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("orthogonalizer requires aligned rank-2 train matrices")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std <= 1.0e-10] = 1.0
    xz = (x - mean) / std
    y_mean = y.mean(axis=0)
    weights = np.linalg.solve(xz.T @ xz + ridge * np.eye(xz.shape[1]), xz.T @ (y - y_mean))
    return Orthogonalizer(mean, std, y_mean, weights)


def orthogonal_only(orthogonalizer: Orthogonalizer, rate: np.ndarray, label: np.ndarray) -> np.ndarray:
    x = np.asarray(rate, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    return y - (orthogonalizer.label_mean + ((x - orthogonalizer.rate_mean) / orthogonalizer.rate_std) @ orthogonalizer.weights)


def low_rank_decoder_residual(residual: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact rank-r factorization in the fixed decoder identity `[N,W]` coordinate."""
    if int(rank) not in RANK_GRID:
        raise ValueError(f"rank must be one of {RANK_GRID}")
    matrix = np.asarray(residual, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("decoder residual must be a finite [N,W] matrix")
    left, values, right_t = np.linalg.svd(matrix, full_matrices=False)
    if rank > min(matrix.shape):
        raise ValueError("rank exceeds residual dimensions")
    a = left[:, :rank] * values[:rank]
    b = right_t[:rank, :].T
    return a, b, a @ b.T


def nested_train_only_choice(records: Iterable[dict[str, Any]], train_sessions: Sequence[str]) -> dict[str, Any]:
    """Select only records whose declared source folds are strictly outer-train sessions."""
    allowed = set(train_sessions)
    valid = []
    for record in records:
        if int(record["rank"]) not in RANK_GRID or float(record["lambda"]) <= 0.0:
            raise ValueError("nested selection record violates frozen rank/lambda grid")
        folds = set(record["inner_validation_sessions"])
        if not folds or not folds <= allowed:
            raise ValueError("nested selection received a non-train source session")
        score = float(record["mean_identity_mse"])
        if math.isfinite(score):
            valid.append(record)
    if not valid:
        raise RuntimeError("no finite nested train-only candidate")
    return min(valid, key=lambda item: (float(item["mean_identity_mse"]), int(item["rank"]), float(item["lambda"])))


def canonical_identity_target(calib: torch.Tensor, fc_id_in: nn.Module, fc_id_out: nn.Module) -> torch.Tensor:
    """Exact frozen teacher identity path; deliberately takes no query or label argument."""
    if calib.ndim != 4 or calib.shape[1] != SUPPORT:
        raise ValueError(f"canonical identity requires [B,{SUPPORT},T,N], got {tuple(calib.shape)}")
    with torch.no_grad():
        trials = calib.permute(0, 1, 3, 2)
        phi = fc_id_in(trials)
        target = fc_id_out(phi.mean(dim=1))
    if target.ndim != 3 or target.shape[1] != calib.shape[-1] or not torch.isfinite(target).all():
        raise ValueError("canonical teacher identity target has invalid shape or non-finite values")
    return target


def _sequential_from_state(state: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    weights = []
    for key, tensor in state.items():
        if key.startswith(prefix) and key.endswith(".weight"):
            index = int(key[len(prefix):].split(".")[0])
            weights.append((index, tensor))
    if not weights:
        raise ValueError(f"checkpoint lacks {prefix} weights")
    modules: list[nn.Module] = []
    for order, (index, weight) in enumerate(sorted(weights)):
        bias_key = f"{prefix}{index}.bias"
        if bias_key not in state:
            raise ValueError(f"checkpoint lacks {bias_key}")
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(state[bias_key])
        modules.append(linear)
        if order < len(weights) - 1:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules).eval()


def load_frozen_identity_path(checkpoint: Path) -> tuple[nn.Sequential, nn.Sequential, dict[str, Any]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen M1 teacher checkpoint missing: {checkpoint}")
    # This is a local checkpoint explicitly frozen by the M1 pilot provenance.
    # Its Lightning metadata references local ``src`` classes, so expose only the
    # checked-in streaming source before CPU deserialization; no trainer is built.
    if str(SCE) not in sys.path:
        sys.path.insert(0, str(SCE))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("frozen teacher does not contain a state_dict")
    fc_id_in = _sequential_from_state(state, "net.fc_id_in.")
    fc_id_out = _sequential_from_state(state, "net.fc_id_out.")
    receipt = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "sha256": sha256(checkpoint),
        "fc_id_in_input_width": int(next(module for module in fc_id_in if isinstance(module, nn.Linear)).in_features),
        "fc_id_in_output_width": int([module for module in fc_id_in if isinstance(module, nn.Linear)][-1].out_features),
        "identity_width_W": int([module for module in fc_id_out if isinstance(module, nn.Linear)][-1].out_features),
        "fc_id_in_linear_count": int(sum(isinstance(module, nn.Linear) for module in fc_id_in)),
        "fc_id_out_linear_count": int(sum(isinstance(module, nn.Linear) for module in fc_id_out)),
    }
    return fc_id_in, fc_id_out, receipt


def _datamodule() -> Any:
    sys.path.insert(0, str(SCE))
    from src.data.falcon_datamodule import FalconDataModule

    dm = FalconDataModule(
        task="m1", data_dir=str(DATA.parent) + "/", heldin_session_names=[""], batch_size=2,
        window_size=100, calibration_n_trials=SUPPORT, random_calibration=False,
        smooth_calibration=False, max_trial_length=1024, standardize_covariates=False,
        use_intertrials=True, use_calib_intertrials=False, trial_feature_type="raw",
        interpolate_trials=True, interpolate_trials_kind="cubic", pad_value=-1.0,
        validation_protocol="loso", loso_fold=0, include_heldout_in_fit=False,
        include_heldout_in_test=False, query_start_trial=0, heldin_query_start_trial=0,
        num_workers=0, pin_memory=False, side_feature_group="none",
    )
    dm.trainer = SimpleNamespace(world_size=1)
    dm.setup("fit")
    return dm


def _session_path(session_name: str) -> Path:
    matches = sorted(DATA.glob(f"*{session_name.replace('ses-', 'ses-')}*.nwb"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one held-in calibration NWB for {session_name}, got {matches}")
    return matches[0]


def feasibility() -> dict[str, Any]:
    validate_rank_lambda_grid(RANK_GRID, LAMBDA_GRID)
    torch.set_num_threads(1)
    fc_id_in, fc_id_out, teacher = load_frozen_identity_path(TEACHER)
    dm = _datamodule()
    from src.data.falcon_d4_features import calibration_obj_id_labels, d4_from_trial_sums

    sessions: dict[str, dict[str, Any]] = {}
    for dataset in (dm.train_dataset, dm.val_heldin_dataset):
        for name, all_calib in dataset.calib_trialized_neural.items():
            if name in sessions:
                continue
            path = _session_path(name)
            support = np.asarray(all_calib[:SUPPORT], dtype=np.float32)
            sums = np.asarray(dataset.calib_trial_spike_sums[name][:SUPPORT], dtype=np.float64)
            lengths = np.asarray(dataset.calib_trial_lengths[name][:SUPPORT], dtype=np.float64)
            labels = calibration_obj_id_labels(path, "m1")[:SUPPORT]
            d4 = d4_from_trial_sums(sums, lengths, labels, source=f"{name}[0:10]")
            rate = support_rate_features(sums, lengths)
            condition = support_condition_features(sums, lengths, labels)
            target = canonical_identity_target(torch.from_numpy(support).unsqueeze(0), fc_id_in, fc_id_out)
            changed = np.asarray(all_calib, dtype=np.float32).copy()
            changed[SUPPORT:] += 12345.0
            stable = canonical_identity_target(torch.from_numpy(changed[:SUPPORT]).unsqueeze(0), fc_id_in, fc_id_out)
            sessions[name] = {
                "nwb_path": str(path.relative_to(ROOT)), "nwb_sha256": sha256(path),
                "support_tensor_shape": list(support.shape), "support_label_levels": sorted(set(labels.tolist())),
                "d4_shape": list(d4.shape), "rate_feature_shape": list(rate.shape),
                "condition_feature_shape": list(condition.shape), "canonical_target_shape": list(target.shape),
                "canonical_target_finite": bool(torch.isfinite(target).all()),
                "post_support_mutation_invariant": bool(torch.equal(target, stable)),
            }
    if len(sessions) != 4:
        raise RuntimeError(f"expected exactly four held-in source sessions, got {sorted(sessions)}")
    defined = all(
        record["support_label_levels"] == list(LEVELS)
        and record["canonical_target_shape"][1] == 64
        and record["canonical_target_shape"][2] == teacher["identity_width_W"]
        and record["canonical_target_finite"] and record["post_support_mutation_invariant"]
        for record in sessions.values()
    )
    return {
        "schema_version": "m1_decoder_latent_alignment_oracle_feasibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"cpu_only": True, "inference_only": True, "formal_oracle_fit_executed": False,
                  "GPU_launch": False, "EvalAI_access": False, "hidden_data_access": False,
                  "source_split": "M1 held-in-calib only", "support": [0, SUPPORT]},
        "input_hashes": {
            "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256(PROTOCOL)},
            "teacher_checkpoint": teacher,
            "spint_source": {"path": str(SPINT_SOURCE.relative_to(ROOT)), "sha256": sha256(SPINT_SOURCE)},
            "streaming_module_source": {"path": str(STREAMING_MODULE_SOURCE.relative_to(ROOT)), "sha256": sha256(STREAMING_MODULE_SOURCE)},
            "d4_helper": {"path": str(D4_HELPER.relative_to(ROOT)), "sha256": sha256(D4_HELPER)},
        },
        "canonical_target": {
            "formula": "E_star=fc_id_out(mean_m(fc_id_in(C[:,m,:,:].permute_to_unit_time)))",
            "decoder_attachment": "exact additive identity input before frozen decoder.fc_in(x + E_star)",
            "query_label_dependency": "none; target function has only calibration tensor and frozen fc_id modules",
            "sessions": dict(sorted(sessions.items())),
            "defined": defined,
        },
        "frozen_future_oracle_contract": {
            "rank_grid": list(RANK_GRID), "lambda_grid": list(LAMBDA_GRID),
            "selection": "nested LOSO using outer-train source sessions only", "controls": list(CONTROL_SPECS),
            "not_executed": "No rank/lambda fit or behavioral/future-rate scoring was run by this feasibility audit.",
        },
        "distinction_from_E4": "E4 fitted ridge against later [210,end) category-rate targets. This audit defines E_star solely from frozen teacher fc_id path and support [0,10) calibration tensors.",
        "decision": "canonical_target_defined_cpu_feasibility_pass" if defined else "fail_closed_canonical_target_undefined",
        "limitations": [
            "Canonical target existence does not demonstrate that any support-only learned map generalizes across sessions.",
            "This is neither decoder R2 nor a behavioral, future-rate, report-window, held-out, or EvalAI result.",
            "A formal low-rank oracle remains blocked pending root review and must use nested train-only rank/lambda selection and all declared controls.",
        ],
    }


def run(out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing feasibility output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = strict_json(feasibility())
        written = out / "feasibility.json"
        with written.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        (out / "feasibility.sha256").write_text(f"{sha256(written)}  feasibility.json\n", encoding="utf-8")
        return written
    except Exception:
        for child in out.iterdir():
            child.unlink()
        out.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    print(run(parser.parse_args().out))


if __name__ == "__main__":
    main()
