#!/usr/bin/env python3
"""CPU-only v2 addendum: bind a non-circular F0/B3 base to the teacher target.

This program intentionally performs no adapter fit, R2 scoring, selection/report
evaluation, hidden-data access, GPU work, or EvalAI action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
DATA = ROOT / "SPINT-main" / "data" / "000941" / "sub-MonkeyL-held-in-calib"
TEACHER = ROOT / "SPINT-main" / "logs" / "train" / "runs" / "2026-07-21-19-11-01" / "checkpoints" / "best_ckpt" / "epoch_019.ckpt"
F0 = ROOT / "streaming_calibration_exp" / "outputs" / "streaming_calibration" / "m1_clean_selection_v1_f0_m1_f1_s42_20260801_192017" / "checkpoints" / "best.ckpt"
TEACHER_SHA256 = "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"
F0_SHA256 = "1ec318f81cfaa9f6eb5e998a2b34135e2bde9941c48fb47bc46f47632f0d6cd8"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "M1_DECODER_LATENT_ALIGNMENT_ORACLE_V2_PROTOCOL.md"
V1_AUDIT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v1" / "feasibility.json"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2"
SUPPORT, N_UNITS, WINDOW, TRIAL_LENGTH, HIDDEN = 10, 64, 100, 1024, 64


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
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _payload(path: Path) -> dict[str, Any]:
    if str(SCE) not in sys.path:
        sys.path.insert(0, str(SCE))
    # Local, named artifacts only; map_location and CUDA visibility make this CPU-only.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not isinstance(value.get("state_dict"), dict):
        raise ValueError(f"checkpoint does not have a Lightning state_dict: {path}")
    return value


def _prefixed_state(state: dict[str, Any], prefix: str) -> dict[str, torch.Tensor]:
    result = {key[len(prefix):]: value for key, value in state.items()
              if isinstance(key, str) and key.startswith(prefix)}
    if not result or not all(isinstance(value, torch.Tensor) for value in result.values()):
        raise ValueError(f"missing or non-tensor state under {prefix}")
    return result


def compare_decoder_states(teacher_state: dict[str, torch.Tensor], f0_decoder_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Strict tensor-wise equality; a state-dict hash alone would be insufficient."""
    teacher_keys, f0_keys = set(teacher_state), set(f0_decoder_state)
    shared = sorted(teacher_keys & f0_keys)
    mismatches = [
        key for key in shared
        if teacher_state[key].dtype != f0_decoder_state[key].dtype
        or tuple(teacher_state[key].shape) != tuple(f0_decoder_state[key].shape)
        or not torch.equal(teacher_state[key], f0_decoder_state[key])
    ]
    return {
        "teacher_tensor_count": len(teacher_state), "f0_decoder_tensor_count": len(f0_decoder_state),
        "missing_from_f0": sorted(teacher_keys - f0_keys), "extra_in_f0": sorted(f0_keys - teacher_keys),
        "unequal_tensor_keys": mismatches,
        "bit_exact": teacher_keys == f0_keys and not mismatches,
    }


def _sequential_from_state(state: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    weights = []
    for key, tensor in state.items():
        if key.startswith(prefix) and key.endswith(".weight"):
            weights.append((int(key[len(prefix):].split(".")[0]), tensor))
    if not weights:
        raise ValueError(f"checkpoint lacks {prefix} weights")
    modules: list[nn.Module] = []
    for index, weight in sorted(weights):
        bias = state.get(f"{prefix}{index}.bias")
        if not isinstance(bias, torch.Tensor):
            raise ValueError(f"checkpoint lacks {prefix}{index}.bias")
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        modules.extend([layer, nn.ReLU()])
    modules.pop()  # teacher's saved affine stack has ReLU only between linear layers
    return nn.Sequential(*modules).eval()


def teacher_identity(calib: torch.Tensor, fc_id_in: nn.Module, fc_id_out: nn.Module) -> torch.Tensor:
    if tuple(calib.shape[1:]) != (SUPPORT, TRIAL_LENGTH, N_UNITS):
        raise ValueError(f"expected support [B,{SUPPORT},{TRIAL_LENGTH},{N_UNITS}], got {tuple(calib.shape)}")
    with torch.no_grad():
        result = fc_id_out(fc_id_in(calib.permute(0, 1, 3, 2)).mean(dim=1))
    if tuple(result.shape[1:]) != (N_UNITS, WINDOW) or not torch.isfinite(result).all():
        raise ValueError("teacher identity has invalid shape or non-finite values")
    return result


def load_f0_b3_encoder(f0_payload: dict[str, Any]) -> nn.Module:
    if f0_payload.get("hyper_parameters", {}).get("variant") != "B3":
        raise ValueError("F0 base checkpoint is not the locked B3 variant")
    if str(SCE) not in sys.path:
        sys.path.insert(0, str(SCE))
    from src.models.components.streaming_encoders import EarlyPoolEncoder

    encoder = EarlyPoolEncoder(TRIAL_LENGTH, WINDOW, HIDDEN).eval()
    state = _prefixed_state(f0_payload["state_dict"], "student.id_encoder.")
    encoder.load_state_dict(state, strict=True)
    return encoder


def _datamodule() -> Any:
    if str(SCE) not in sys.path:
        sys.path.insert(0, str(SCE))
    from src.data.falcon_datamodule import FalconDataModule

    dm = FalconDataModule(
        task="m1", data_dir=str(DATA.parent) + "/", heldin_session_names=[""], batch_size=2,
        window_size=WINDOW, calibration_n_trials=SUPPORT, random_calibration=False,
        smooth_calibration=False, max_trial_length=TRIAL_LENGTH, standardize_covariates=False,
        use_intertrials=True, use_calib_intertrials=False, trial_feature_type="raw",
        interpolate_trials=True, interpolate_trials_kind="cubic", pad_value=-1.0,
        validation_protocol="loso", loso_fold=0, include_heldout_in_fit=False,
        include_heldout_in_test=False, query_start_trial=0, heldin_query_start_trial=0,
        num_workers=0, pin_memory=False, side_feature_group="none",
    )
    dm.trainer = SimpleNamespace(world_size=1)
    dm.setup("fit")
    return dm


def addendum() -> dict[str, Any]:
    torch.set_num_threads(1)
    if sha256(TEACHER) != TEACHER_SHA256 or sha256(F0) != F0_SHA256:
        raise ValueError("a frozen checkpoint hash differs from the v2 protocol")
    teacher_payload, f0_payload = _payload(TEACHER), _payload(F0)
    teacher_state = _prefixed_state(teacher_payload["state_dict"], "net.")
    f0_decoder_state = _prefixed_state(f0_payload["state_dict"], "student.decoder.")
    decoder_relation = compare_decoder_states(teacher_state, f0_decoder_state)
    if not decoder_relation["bit_exact"]:
        raise ValueError("F0 query decoder is not bit-exact to the frozen teacher decoder")
    fc_id_in, fc_id_out = _sequential_from_state(teacher_state, "fc_id_in."), _sequential_from_state(teacher_state, "fc_id_out.")
    b3 = load_f0_b3_encoder(f0_payload)
    dm, sessions = _datamodule(), {}
    for dataset in (dm.train_dataset, dm.val_heldin_dataset):
        for name, all_calib in dataset.calib_trialized_neural.items():
            if name in sessions:
                continue
            support = torch.from_numpy(np.asarray(all_calib[:SUPPORT], dtype=np.float32)).unsqueeze(0)
            altered = np.asarray(all_calib, dtype=np.float32).copy()
            altered[SUPPORT:] += 12345.0
            altered_support = torch.from_numpy(altered[:SUPPORT]).unsqueeze(0)
            e_teacher = teacher_identity(support, fc_id_in, fc_id_out)
            e0 = b3.forward_batch(support)
            e_teacher_changed = teacher_identity(altered_support, fc_id_in, fc_id_out)
            e0_changed = b3.forward_batch(altered_support)
            delta = e_teacher - e0
            sessions[name] = {
                "support_tensor_shape": list(support.shape), "E_teacher_shape": list(e_teacher.shape),
                "E0_f0_b3_shape": list(e0.shape), "Delta_star_shape": list(delta.shape),
                "E_teacher_finite": bool(torch.isfinite(e_teacher).all()), "E0_finite": bool(torch.isfinite(e0).all()),
                "Delta_star_finite": bool(torch.isfinite(delta).all()),
                "Delta_star_nonzero": bool(torch.count_nonzero(delta).item() > 0),
                "E_teacher_post_support_mutation_invariant": bool(torch.equal(e_teacher, e_teacher_changed)),
                "E0_post_support_mutation_invariant": bool(torch.equal(e0, e0_changed)),
            }
    if len(sessions) != 4:
        raise RuntimeError(f"expected four M1 held-in source sessions, got {sorted(sessions)}")
    defined = decoder_relation["bit_exact"] and all(
        item["support_tensor_shape"] == [1, SUPPORT, TRIAL_LENGTH, N_UNITS]
        and item["E_teacher_shape"] == [1, N_UNITS, WINDOW]
        and item["E0_f0_b3_shape"] == [1, N_UNITS, WINDOW]
        and item["Delta_star_shape"] == [1, N_UNITS, WINDOW]
        and item["E_teacher_finite"] and item["E0_finite"] and item["Delta_star_finite"]
        and item["Delta_star_nonzero"]
        and item["E_teacher_post_support_mutation_invariant"] and item["E0_post_support_mutation_invariant"]
        for item in sessions.values()
    )
    return {
        "schema_version": "m1_decoder_latent_alignment_oracle_base_addendum_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"cpu_only": True, "inference_only": True, "adapter_fit_executed": False,
                  "R2_scoring_executed": False, "selection_or_report_opened": False, "GPU_launch": False,
                  "heldout_access": False, "EvalAI_access": False, "support": [0, SUPPORT]},
        "inputs": {
            "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256(PROTOCOL)},
            "v1_target_feasibility": {"path": str(V1_AUDIT.relative_to(ROOT)), "sha256": sha256(V1_AUDIT)},
            "teacher_checkpoint": {"path": str(TEACHER.relative_to(ROOT)), "sha256": TEACHER_SHA256},
            "f0_b3_checkpoint": {"path": str(F0.relative_to(ROOT)), "sha256": F0_SHA256,
                                 "variant": "B3", "trial_length": TRIAL_LENGTH, "hidden_dim": HIDDEN,
                                 "window_size": WINDOW, "side_dim": 0, "identity_mode": "calibrated"},
        },
        "locked_query_decoder": {"source": "F0 checkpoint student.decoder", "relation_to_teacher_net": decoder_relation},
        "noncircular_identity_contract": {
            "E_teacher": "teacher.fc_id_out(mean_m(teacher.fc_id_in(C_m)))",
            "E0": "F0/B3 student.id_encoder.forward_batch(C)",
            "Delta_star": "E_teacher - E0", "sessions": dict(sorted(sessions.items())), "defined": defined,
        },
        "decision": "pass_non_circular_base_and_bit_exact_decoder" if defined else "fail_closed_base_or_decoder_undefined",
        "not_evidence_of": ["adapter quality", "behavioral R2", "cross-session generalization", "held-out performance", "deployment value"],
    }


def run(out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing addendum output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = strict_json(addendum())
        written = out / "identity_base_addendum.json"
        written.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (out / "identity_base_addendum.sha256").write_text(f"{sha256(written)}  identity_base_addendum.json\n", encoding="utf-8")
        return written
    except Exception:
        for child in out.iterdir():
            child.unlink()
        out.rmdir()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    print(run(parser.parse_args().out))
