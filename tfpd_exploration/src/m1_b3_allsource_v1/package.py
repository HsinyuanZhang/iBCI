"""Offline cached-identity export for all-source B3 / B3S-rSyn3 students.

Reads only the seven public calibration NWBs. Native T4, if present on an
abandoned control arm, keeps the train-fit normalizer. EMG-rSyn3 carriers are
taken from the train-fit bank and never have their NMF dictionary refit on
later-day calib files.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any, Mapping

import numpy as np

from . import plan


class PackageError(RuntimeError):
    """Fail closed for cached-identity export."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


SESSION_PATTERN = re.compile(r"(ses-\d{8})")
SOURCE_CALIB_DIR = "sub-MonkeyL-held-in-calib"
LATER_CALIB_DIR = "sub-MonkeyL-held-out-calib"
SCHEMA_VERSION = "m1_b3_allsource_cached_identity_v1"
PAYLOAD_LEAF = "decoder.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neural_loader_side_group(group: str) -> str:
    """FalconDataset does not know rSyn3; the neural windows stay side=none."""
    if group == "rsyn3":
        return "none"
    return group


def rsyn3_carrier_for_session(
    session_name: str,
    path: Path,
    bank: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, list[int]]:
    """Source-frozen NMF encode. Chrono-10 reuses the train bank on source days."""
    from . import carrier_k
    from . import rsyn3_bank as bank_module

    _require(bank.get("later_day_in_fit") is False, "later-day entered the NMF fit")
    method = spec.get("carrier_method")
    budget = spec.get("carrier_k")
    pool = int(spec.get("encoder_neural_trials") or plan.CALIBRATION_N_TRIALS)
    if method is None:
        if session_name in plan.SOURCE_SESSION_NAMES:
            _require(session_name in bank["normalized"], f"source carrier missing for {session_name}")
            carrier = np.asarray(bank["normalized"][session_name], dtype=np.float32)
        else:
            later = tuple(f"ses-{item}" for item in plan.LATER_CALIB_SESSIONS)
            _require(session_name in later, f"unexpected rSyn3 session {session_name}")
            carrier = bank_module.encode_public_session(path, bank)
        selected = list(range(int(bank.get("support_trials", plan.CALIBRATION_N_TRIALS))))
        _require(carrier.ndim == 2 and carrier.shape[1] == 4, f"rSyn3 carrier shape {carrier.shape}")
        return np.ascontiguousarray(carrier, dtype=np.float32), selected

    _require(method == plan.TOP4_CARRIER_METHOD, f"unsupported carrier method {method}")
    _require(int(budget) == plan.TOP4_CARRIER_K, f"unsupported carrier k {budget}")
    _require(pool == plan.CALIBRATION_N_TRIALS, "top4 encoder neural is not M10")
    from src.data.falcon_t4_features import calibration_target_angles

    record = bank_module.load_public_calib_support(path, support_trials=pool)
    basis = bank_module._basis_from_bank(bank)
    synergy = carrier_k.trial_mean_synergy(record, basis, pool_trials=pool)
    angles = calibration_target_angles(path, "m1")
    selected_np = carrier_k.select_indices(
        str(method), int(budget), angles=angles, synergy_means=synergy, pool_trials=pool,
    )
    carrier = bank_module.encode_record_selected(
        record, bank, selected_np, pool_trials=pool,
    )
    _require(carrier.ndim == 2 and carrier.shape[1] == 4, f"rSyn3 carrier shape {carrier.shape}")
    return np.ascontiguousarray(carrier, dtype=np.float32), [int(item) for item in selected_np.tolist()]


def _load_rsyn3_bank(hydra_dir: Path) -> dict[str, Any]:
    from . import rsyn3_bank as bank_module

    train_manifest_path = hydra_dir / "all_source_train_manifest.json"
    _require(train_manifest_path.is_file(), "train rSyn3 manifest missing")
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    payload = train_manifest.get("rsyn3_bank")
    _require(isinstance(payload, dict), "rsyn3_bank missing from train manifest")
    return bank_module.bank_from_manifest(payload)


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _session_from_path(path: Path) -> str:
    match = SESSION_PATTERN.search(path.name)
    _require(match is not None, f"cannot derive M1 session from {path}")
    return match.group(1)


def calibration_file_map(data_dir: Path) -> OrderedDict[str, Path]:
    """Resolve exactly seven public calibration NWBs without a recursive search."""
    source_dir = data_dir / SOURCE_CALIB_DIR
    later_dir = data_dir / LATER_CALIB_DIR
    _require(source_dir.is_dir() and later_dir.is_dir(), f"public calib dirs missing under {data_dir}")
    source = sorted(source_dir.glob("*.nwb"))
    later = sorted(later_dir.glob("*.nwb"))
    expected_source = tuple(f"ses-{item}" for item in plan.SOURCE_SESSIONS)
    expected_later = tuple(f"ses-{item}" for item in plan.LATER_CALIB_SESSIONS)
    _require(len(source) == len(expected_source), "source calib coverage changed")
    _require(len(later) == len(expected_later), "later-day calib coverage changed")
    mapping: OrderedDict[str, Path] = OrderedDict()
    for expected, paths in ((expected_source, source), (expected_later, later)):
        for path in paths:
            session = _session_from_path(path)
            _require(session in expected, f"unexpected calibration session {session}: {path}")
            _require(session not in mapping, f"duplicate calibration session {session}")
            text = str(path.resolve()).lower()
            _require("minival" not in text, f"refusing query-shaped calib path {path}")
            mapping[session] = path.resolve()
    _require(tuple(mapping) == expected_source + expected_later, "unexpected calibration ordering")
    return mapping


def _ensure_streaming_paths(repo_root: Path) -> None:
    streaming = str(Path(repo_root) / "streaming_calibration_exp")
    spint_main = str(Path(repo_root) / "SPINT-main")
    root = str(Path(repo_root))
    # Both trees export a top-level ``src`` package. Streaming must win.
    for entry in (spint_main, root, streaming):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def _load_train_hydra(repo_root: Path, arm: str) -> dict[str, Any]:
    train_arm = plan.source_train_arm(arm)
    train_root = Path(repo_root) / plan.train_root_relative(train_arm)
    hydra_path = train_root / "hydra_run.json"
    terminal_path = train_root / "terminal.json"
    _require(hydra_path.is_file() and terminal_path.is_file(), f"train receipts missing under {train_root}")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    _require(terminal.get("status") == "COMPLETE", f"train arm {train_arm} is not COMPLETE")
    payload = json.loads(hydra_path.read_text(encoding="utf-8"))
    _require(payload.get("arm") == train_arm, "hydra_run arm mismatch")
    _require(payload.get("teacher_sha256") == plan.TEACHER_SHA256, "teacher sha drift in hydra_run")
    return payload


def _manual_decode(decoder: Any, neural: Any, identity: Any) -> Any:
    source = neural.permute(0, 2, 1) + identity
    source = decoder.fc_in(source)
    query = decoder.fc_in(decoder.rep).to(source)
    transformed, _ = decoder.transformer(query.repeat(source.shape[0], 1, 1), source)
    return decoder.fc_out(transformed).permute(0, 2, 1)


def export_payload(repo_root: Path, arm: str) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _ensure_streaming_paths(repo_root)
    spec = plan.ARM_SPECS[arm]
    hydra_run = _load_train_hydra(repo_root, arm)
    ckpt = Path(hydra_run["student_checkpoint"])
    _require(ckpt.is_file(), f"student checkpoint missing: {ckpt}")
    _require(sha256_file(ckpt) == hydra_run["student_checkpoint_sha256"], "student checkpoint drift")
    teacher = Path(repo_root) / plan.TEACHER_CHECKPOINT_RELATIVE
    _require(sha256_file(teacher) == plan.TEACHER_SHA256, "teacher checkpoint drift")
    hydra_dir = Path(hydra_run["hydra_output_dir"])
    resolved = hydra_dir / ".hydra" / "config.yaml"
    _require(resolved.is_file(), f"resolved hydra config missing: {resolved}")
    data_dir = Path(repo_root) / plan.DATA_DIR_RELATIVE
    mapping = calibration_file_map(data_dir)

    import torch
    from falcon_challenge.config import FalconConfig, FalconTask
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from src.data.falcon_datamodule import FalconDataset

    config = OmegaConf.load(resolved)
    _require(str(config.data.task).lower() == "m1", "resolved config is not native M1")
    _require(str(config.data.validation_protocol).lower() == "all_source", "resolved config is not all-source")
    _require(int(config.data.calibration_n_trials) == plan.CALIBRATION_N_TRIALS, "M10 drift")
    _require(bool(config.data.random_calibration) is False, "random calibration must stay off")
    _require(int(config.trainer.max_epochs) == plan.EPOCHS, "epoch budget drift")
    _require(str(config.model.loss_mode) == "task_only", "loss_mode drift")
    _require(bool(config.model.freeze_decoder) is spec["freeze_decoder"], "freeze_decoder drift")
    _require(str(config.model.variant) == spec["variant"], "variant drift")
    _require(str(config.data.side_feature_group).lower() == spec["side_feature_group"], "side group drift")
    if spec["side_dim"]:
        _require(int(config.model.side_dim) == spec["side_dim"], "side_dim drift")
    config.model.teacher_ckpt_path = str(teacher.resolve())
    config.data.data_dir = str(data_dir.resolve())
    config.data.num_workers = 0
    config.data.pin_memory = False

    model = instantiate(config.model)
    model.setup("fit")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    _require(isinstance(state, dict) and isinstance(state.get("state_dict"), dict), "not a Lightning ckpt")
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    student = model.student
    _require(student is not None and student.decoder_mode == "coupled", "student is not coupled SPINT")

    datamodule = instantiate(config.data)
    group = spec["side_feature_group"]
    neural_group = neural_loader_side_group(group)
    include_trial_targets = group == "t4"
    rsyn3_fitted = None
    sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
    covariates_mean = covariates_std = None
    task = FalconConfig(task=FalconTask.m1).task
    for index, (name, path) in enumerate(mapping.items()):
        record = datamodule.prepare_session_data(
            path,
            task,
            standardize_covariates=bool(config.data.standardize_covariates),
            covariates_mean=covariates_mean,
            covariates_std=covariates_std,
            use_intertrials=bool(config.data.use_intertrials),
            include_trial_targets=include_trial_targets,
        )
        if index == 0:
            covariates_mean, covariates_std = record["covariates_mean"], record["covariates_std"]
        sessions[name] = record

    side_mean = side_std = None
    if group == "t4":
        train_manifest_path = hydra_dir / "all_source_train_manifest.json"
        _require(train_manifest_path.is_file(), "train T4 manifest missing")
        train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        encoded = train_manifest.get("native_t4_normalization")
        _require(isinstance(encoded, dict), "native T4 normalization missing from train manifest")
        _require(encoded.get("feature_group") == "t4", "T4 normalization arm mismatch")
        _require(
            list(encoded.get("train_sessions", [])) == list(plan.SOURCE_SESSION_NAMES),
            "T4 normalizer was not fit on the four source sessions",
        )
        side_mean = np.asarray(encoded["mean"], dtype=np.float32)
        side_std = np.asarray(encoded["std"], dtype=np.float32)
    elif group == "rsyn3":
        rsyn3_fitted = _load_rsyn3_bank(hydra_dir)
        _require(
            list(rsyn3_fitted.get("source_sessions", [])) == list(plan.SOURCE_SESSION_NAMES),
            "rSyn3 bank was not fit on the four source sessions",
        )

    dataset = FalconDataset(
        sessions_dict=sessions,
        calib_sessions_dict=sessions,
        window_size=int(config.data.window_size),
        split="train",
        calibration_n_trials=plan.CALIBRATION_N_TRIALS,
        random_calibration=False,
        smooth_calibration=bool(config.data.smooth_calibration),
        max_trial_length=int(config.data.max_trial_length),
        use_calib_intertrials=bool(config.data.use_calib_intertrials),
        trial_feature_type=str(config.data.trial_feature_type),
        remove_still_times=bool(config.data.remove_still_times),
        remove_calib_still_times=bool(config.data.remove_calib_still_times),
        use_calib_active_segments=bool(config.data.use_calib_active_segments),
        calib_n_active_segments=int(config.data.calib_n_active_segments),
        interpolate_trials=bool(config.data.interpolate_trials),
        interpolate_trials_kind=str(config.data.interpolate_trials_kind),
        pad_value=float(config.data.pad_value),
        side_feature_group=neural_group,
        side_feature_mean=side_mean,
        side_feature_std=side_std,
        query_start_trial=0,
    )

    task_config = FalconConfig(task=FalconTask.m1)
    identity_by_tag: dict[str, np.ndarray] = {}
    session_records: dict[str, Any] = {}
    max_direct_vs_cached = 0.0
    max_direct_vs_decoder = 0.0
    with torch.inference_mode():
        for session_name, path in mapping.items():
            calibration_np = np.asarray(
                dataset.calib_trialized_neural_features[session_name][: plan.CALIBRATION_N_TRIALS],
                dtype=np.float32,
            )
            _require(
                calibration_np.shape == (plan.CALIBRATION_N_TRIALS, plan.TRIAL_BINS, plan.CHANNELS),
                f"unexpected M10 calibration shape for {session_name}: {calibration_np.shape}",
            )
            side = None
            side_np = None
            selected_ids: list[int] | None = None
            if group == "t4":
                side_np = np.asarray(
                    dataset._native_t4_side_features(session_name, 0, plan.CALIBRATION_N_TRIALS),
                    dtype=np.float32,
                )
                _require(side_np.shape == (plan.CHANNELS, 4), f"unexpected T4 side shape for {session_name}")
                side = torch.from_numpy(side_np).unsqueeze(0)
            elif group == "rsyn3":
                _require(rsyn3_fitted is not None, "rSyn3 bank was not loaded")
                side_np, selected_ids = rsyn3_carrier_for_session(
                    session_name, path, rsyn3_fitted, spec=spec,
                )
                _require(
                    side_np.shape == (calibration_np.shape[-1], 4),
                    f"unexpected rSyn3 side shape for {session_name}: {side_np.shape}",
                )
                side = torch.from_numpy(side_np).unsqueeze(0)
            calibration = torch.from_numpy(calibration_np).unsqueeze(0)
            neural_np = np.asarray(dataset.neural_data[session_name][100:200], dtype=np.float32)
            _require(neural_np.shape == (plan.WINDOW_SIZE, plan.CHANNELS), "audit window shape")
            neural = torch.from_numpy(neural_np).unsqueeze(0)
            identity = student.compute_identity(calibration, side_features=side)
            direct, _ = student(neural, calib_trials=calibration, side_features=side)
            cached, _ = student(neural, identity=identity)
            decoder_only = _manual_decode(student.decoder, neural, identity)
            delta_cached = float((direct - cached).abs().max().item())
            delta_decoder = float((direct - decoder_only).abs().max().item())
            max_direct_vs_cached = max(max_direct_vs_cached, delta_cached)
            max_direct_vs_decoder = max(max_direct_vs_decoder, delta_decoder)
            _require(
                delta_cached == 0.0 and delta_decoder == 0.0,
                f"cached deployment is not exact for {session_name}: "
                f"cached={delta_cached}, decoder={delta_decoder}",
            )
            identity_np = np.ascontiguousarray(identity.squeeze(0).numpy(), dtype=np.float32)
            _require(
                identity_np.shape == (plan.CHANNELS, plan.IDENTITY_DIM) and np.isfinite(identity_np).all(),
                f"invalid identity for {session_name}",
            )
            tag = task_config.hash_dataset(path.stem)
            identity_by_tag[tag] = identity_np
            record = {
                "dataset_tag": tag,
                "file": str(path),
                "file_sha256": sha256_file(path),
                "calibration_sha256": sha256_array(calibration_np),
                "identity_sha256": sha256_array(identity_np),
                "identity_shape": list(identity_np.shape),
                "direct_vs_cached_identity_max_abs": delta_cached,
                "direct_vs_decoder_only_max_abs": delta_decoder,
                "future_query_values_read": False,
            }
            if side_np is not None:
                key = "rsyn3_side_sha256" if group == "rsyn3" else "t4_side_sha256"
                record[key] = sha256_array(side_np)
            if selected_ids is not None:
                record["carrier_selected_trial_ids"] = selected_ids
                record["carrier_method"] = spec.get("carrier_method") or "chronological"
                record["carrier_k"] = int(spec.get("carrier_k") or plan.CALIBRATION_N_TRIALS)
            session_records[session_name] = record
    _require(len(identity_by_tag) == 7, f"expected seven public calib tags, got {len(identity_by_tag)}")

    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": task_config.task,
        "arm": arm,
        "decoder": decoder,
        "identity_by_dataset_tag": identity_by_tag,
        "dataset_tags": sorted(identity_by_tag),
        "window_size": int(decoder.window_size),
        "behavior_scaling_factor": 1.0,
        "smooth_observations": False,
        "metadata": {
            "checkpoint_sha256": hydra_run["student_checkpoint_sha256"],
            "teacher_checkpoint_sha256": plan.TEACHER_SHA256,
            "resolved_config_sha256": sha256_file(resolved),
            "side_feature_group": group,
            "freeze_decoder": spec["freeze_decoder"],
            "source_train_arm": plan.source_train_arm(arm),
            "calibration_selection": (
                f"encoder_chrono_m10_carrier_{spec['carrier_method']}_k{spec['carrier_k']}"
                if spec.get("carrier_method")
                else "chronological_first_10"
            ),
            "carrier_method": spec.get("carrier_method"),
            "carrier_k": spec.get("carrier_k"),
            "encoder_neural_trials": int(spec.get("encoder_neural_trials") or plan.CALIBRATION_N_TRIALS),
            "calibration_n_trials": plan.CALIBRATION_N_TRIALS,
            "t4_normalizer_refit_on_later_days": False,
            "rsyn3_nmf_refit_on_later_days": False,
            "target_query_values_read": False,
            "online_state": "cached_E[N,100]",
            "online_backward_pass": False,
            "session_records": session_records,
            "max_direct_vs_cached_identity_abs": max_direct_vs_cached,
            "max_direct_vs_decoder_only_abs": max_direct_vs_decoder,
        },
    }
    return {
        "payload": payload,
        "receipt": {
            "schema_version": "m1_b3_allsource_export_receipt_v1",
            "arm": arm,
            "checkpoint_sha256": hydra_run["student_checkpoint_sha256"],
            "teacher_checkpoint_sha256": plan.TEACHER_SHA256,
            "source_train_arm": plan.source_train_arm(arm),
            "calibration_n_trials": plan.CALIBRATION_N_TRIALS,
            "encoder_neural_trials": int(spec.get("encoder_neural_trials") or plan.CALIBRATION_N_TRIALS),
            "carrier_method": spec.get("carrier_method"),
            "carrier_k": spec.get("carrier_k"),
            "source_sessions": list(plan.SOURCE_SESSIONS),
            "later_calib_sessions": list(plan.LATER_CALIB_SESSIONS),
            "session_count": len(identity_by_tag),
            "dataset_tags": sorted(identity_by_tag),
            "max_direct_vs_cached_identity_abs": max_direct_vs_cached,
            "max_direct_vs_decoder_only_abs": max_direct_vs_decoder,
            "target_query_values_read": False,
            "t4_normalizer_refit_on_later_days": False,
            "rsyn3_nmf_refit_on_later_days": False,
            "formal_benchmark_verdict": False,
            "session_records": session_records,
        },
    }


def publish_package(
    repo_root: Path,
    *,
    arm: str,
    artifact: Any,
    progress: dict[str, object],
) -> dict[str, str]:
    exported = export_payload(Path(repo_root), arm)
    body = pickle.dumps(exported["payload"], protocol=pickle.HIGHEST_PROTOCOL)
    payload_sha = hashlib.sha256(body).hexdigest()
    exported["receipt"]["payload_sha256"] = payload_sha
    exported["receipt"]["payload_bytes"] = len(body)
    exported["receipt"]["payload_leaf"] = PAYLOAD_LEAF
    progress["payload_bytes"] = len(body)
    shas = {
        "decoder.pt": artifact.publish_bytes(PAYLOAD_LEAF, body),
        "export_receipt.json": artifact.publish_json("export_receipt.json", exported["receipt"]),
    }
    _require(shas["decoder.pt"] == payload_sha, "payload sha256 mismatch")
    return shas
