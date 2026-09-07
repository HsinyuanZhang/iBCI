"""Score P1 dim ablations, retrain mean-rate contrast, prepare cached-identity payload."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_hold_film_probe_v1.physical import (
    _atomic_bytes,
    _atomic_json,
    install_film,
    score_arm,
    train_film,
)

from . import core, plan


def _mask(name: str) -> np.ndarray:
    return np.asarray(plan.MASKS[name], dtype=np.float32)


def _ext(arm: dict[str, object]) -> dict[str, object]:
    return arm["summaries"]["external_official_query"]


def _load_p1_state(repo_root: Path) -> dict[str, torch.Tensor]:
    path = plan.probe_root(repo_root) / "film_states.pt"
    core.require(path.is_file(), f"sealed P1 film_states missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    core.require("p1" in payload and "p0" in payload, "film_states.pt missing p0/p1")
    return payload


def _export_cached_identities(
    *,
    repo_root: Path,
    student: Any,
    datasets: dict[str, Any],
    device: torch.device,
    candidate: str,
    contrast_mask: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        EXPECTED_SESSION_COUNT,
        M2_DATA_DIR,
        calibration_file_map,
        manual_decode,
        sha256_array,
        sha256_file,
    )
    from falcon_challenge.config import FalconConfig, FalconTask

    task_config = FalconConfig(task=FalconTask.m2)
    session_to_tag = calibration_file_map(M2_DATA_DIR, task_config)
    identity_by_tag: dict[str, np.ndarray] = {}
    session_records: dict[str, object] = {}
    max_direct_vs_identity = 0.0
    max_direct_vs_decoder = 0.0
    from tfpd_exploration.src.m2_hold_film_probe_v1.physical import session_side

    student.eval()
    for dataset in datasets.values():
        for session_name in sorted(dataset.calib_trialized_neural_features):
            if session_name not in session_to_tag:
                raise core.ProbeError(f"No FALCON dataset tag for {session_name}")
            dataset_tag = session_to_tag[session_name]
            if dataset_tag in identity_by_tag:
                raise core.ProbeError(f"Duplicate FALCON dataset tag {dataset_tag}")
            activity, side, evidence = session_side(
                dataset,
                session_name,
                shuffle=False,
                contrast_mask=contrast_mask,
                horizon=plan.M33_HORIZON,
                t4_mode="native",
            )
            support = torch.from_numpy(activity).unsqueeze(0).to(device)
            side_tensor = torch.from_numpy(side).unsqueeze(0).to(device)
            neural = torch.from_numpy(
                np.asarray(dataset.neural_data[session_name][50:100], dtype=np.float32)
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                identity = student.compute_identity(support, side_features=side_tensor)
                direct, _ = student(neural, calib_trials=support, side_features=side_tensor)
                cached, _ = student(neural, identity=identity)
                decoder_only = manual_decode(student.decoder, neural, identity)
            direct_vs_identity = float((direct - cached).abs().max().item())
            direct_vs_decoder = float((direct - decoder_only).abs().max().item())
            max_direct_vs_identity = max(max_direct_vs_identity, direct_vs_identity)
            max_direct_vs_decoder = max(max_direct_vs_decoder, direct_vs_decoder)
            core.require(direct_vs_identity == 0.0, f"cached identity drift {session_name}")
            core.require(direct_vs_decoder == 0.0, f"decoder-only drift {session_name}")
            identity_np = np.ascontiguousarray(
                identity.squeeze(0).detach().cpu().numpy(), dtype=np.float32
            )
            identity_by_tag[dataset_tag] = identity_np
            session_records[session_name] = {
                "dataset_tag": dataset_tag,
                "identity_sha256": sha256_array(identity_np),
                "carrier": evidence,
            }
    core.require(len(identity_by_tag) == EXPECTED_SESSION_COUNT, "session coverage drift")
    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    payload = {
        "schema_version": "e8_t4_m2_cached_identity_v1",
        "task": task_config.task,
        "decoder": decoder,
        "identity_by_dataset_tag": identity_by_tag,
        "window_size": int(decoder.window_size),
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            **metadata,
            "candidate": candidate,
            "calibration_selection": "native_t4_first_33_plus_hold_reach_contrast",
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
            "film_online": False,
            "evalai_push": False,
        },
    }
    output = plan.result_root(repo_root) / "decoder.pkl"
    buffer = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_bytes(output, buffer)
    receipt = {
        "schema": plan.SCHEMA + "_payload",
        "candidate": candidate,
        "payload_bytes": output.stat().st_size,
        "payload_sha256": sha256_file(output),
        "session_count": len(identity_by_tag),
        "max_direct_vs_cached_identity_abs": max_direct_vs_identity,
        "max_direct_vs_decoder_only_abs": max_direct_vs_decoder,
        "evalai_push": False,
        "session_records": session_records,
    }
    _atomic_json(plan.result_root(repo_root) / "payload.receipt.json", receipt)
    decoder.to(device)
    student.decoder.to(device)
    return receipt


def execute(
    repo_root: Path,
    *,
    gpu_index: int = plan.DEFAULT_GPU_INDEX,
    batch_size: int = 1024,
) -> dict[str, object]:
    if isinstance(gpu_index, bool) or gpu_index < 0:
        raise core.ProbeError("gpu_index must be a non-negative integer")
    expected_cvd = str(gpu_index)
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected_cvd, "CUDA_VISIBLE_DEVICES mismatch")
    core.require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    root = plan.result_root(repo_root)
    core.require(not root.exists(), f"result root already exists: {root}")
    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    states = _load_p1_state(repo_root)
    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    core.require(metadata["checkpoint_sha256"] == probe_plan.CHECKPOINT_SHA256, "checkpoint drift")
    student = model.student.to(device)
    install_film(student, device)
    student.id_encoder.load_state_dict(states["p1"], strict=True)
    student.id_encoder.to(device)
    student.id_encoder.freeze_base_path()

    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }

    def progress(message: str) -> None:
        print(message, flush=True)

    interp: dict[str, dict[str, object]] = {}
    for name, mask in plan.MASKS.items():
        progress(f"scoring sealed-P1 mask {name}")
        interp[name] = score_arm(
            student=student,
            datasets=datasets,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            arm=f"p1_mask_{name}",
            contrast_mask=_mask(name),
        )
    p1_full_mean = float(_ext(interp["p1_full"])["equal_session_mean"])
    core.require(
        abs(p1_full_mean - plan.SEALED_P1_EXTERNAL) <= plan.P1_FULL_ABS_TOLERANCE,
        f"P1 full rescore drift {p1_full_mean} vs {plan.SEALED_P1_EXTERNAL}",
    )

    progress("scoring native M33 P0 (zero FiLM)")
    student.id_encoder.load_state_dict(states["p0"], strict=True)
    student.id_encoder.to(device)
    p0_m33 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="p0_native_m33",
        horizon=plan.M33_HORIZON,
        t4_mode="native",
        contrast_mask=np.zeros(4, dtype=np.float32),
    )
    progress("scoring sealed-P1 transfer on native M33")
    student.id_encoder.load_state_dict(states["p1"], strict=True)
    student.id_encoder.to(device)
    p1_m33 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="p1_native_m33_transfer",
        horizon=plan.M33_HORIZON,
        t4_mode="native",
        contrast_mask=_mask("p1_full"),
    )
    p1_m33_shuffle = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        arm="p1_native_m33_shuffle",
        horizon=plan.M33_HORIZON,
        t4_mode="native",
        contrast_mask=_mask("p1_full"),
    )

    progress("retraining mean-rate contrast at first-30 ridge")
    student.id_encoder.load_state_dict(states["p0"], strict=True)
    student.id_encoder.to(device)
    student.id_encoder.freeze_base_path()
    means_m30_train = train_film(
        student=student,
        train_dataset=datasets["within_post30"],
        device=device,
        shuffle=False,
        contrast_mask=_mask("means"),
    )
    means_m30 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="means_m30_retrain",
        contrast_mask=_mask("means"),
    )
    means_m30_shuffle = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        arm="means_m30_shuffle",
        contrast_mask=_mask("means"),
    )
    means_m30_state = {
        key: value.detach().cpu().clone() for key, value in student.id_encoder.state_dict().items()
    }

    progress("retraining mean-rate contrast at native first-33")
    student.id_encoder.load_state_dict(states["p0"], strict=True)
    student.id_encoder.to(device)
    student.id_encoder.freeze_base_path()
    means_m33_train = train_film(
        student=student,
        train_dataset=datasets["within_post30"],
        device=device,
        shuffle=False,
        contrast_mask=_mask("means"),
        horizon=plan.M33_HORIZON,
        t4_mode="native",
    )
    means_m33 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="means_m33_retrain",
        contrast_mask=_mask("means"),
        horizon=plan.M33_HORIZON,
        t4_mode="native",
    )
    means_m33_shuffle = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        arm="means_m33_shuffle",
        contrast_mask=_mask("means"),
        horizon=plan.M33_HORIZON,
        t4_mode="native",
    )
    means_m33_state = {
        key: value.detach().cpu().clone() for key, value in student.id_encoder.state_dict().items()
    }

    sealed_score = json.loads((plan.probe_root(repo_root) / "score.json").read_text(encoding="utf-8"))
    p0_m30_sessions = sealed_score["arms"]["p0_zero_film_t4"]["summaries"]["external_official_query"][
        "per_session_r2"
    ]

    interp_table = {}
    for name, arm in interp.items():
        mean = float(_ext(arm)["equal_session_mean"])
        interp_table[name] = {
            "external_mean": mean,
            "minus_p0": mean - plan.SEALED_P0_EXTERNAL,
            "keeps_80pct_of_p1_gain": core.keeps_gain(mean),
            "per_session_r2": _ext(arm)["per_session_r2"],
        }

    p1_m33_vs_p0 = core.paired_contrast(
        p1_m33["session_maps"]["external_official_query"],
        p0_m33["session_maps"]["external_official_query"],
    )
    p1_m33_shuffle_vs_p0 = core.paired_contrast(
        p1_m33_shuffle["session_maps"]["external_official_query"],
        p0_m33["session_maps"]["external_official_query"],
    )
    means_m33_vs_p0 = core.paired_contrast(
        means_m33["session_maps"]["external_official_query"],
        p0_m33["session_maps"]["external_official_query"],
    )
    means_m33_shuffle_vs_p0 = core.paired_contrast(
        means_m33_shuffle["session_maps"]["external_official_query"],
        p0_m33["session_maps"]["external_official_query"],
    )
    means_m30_vs_p0 = core.paired_contrast(
        means_m30["session_maps"]["external_official_query"],
        p0_m30_sessions,
    )
    means_m30_shuffle_vs_p0 = core.paired_contrast(
        means_m30_shuffle["session_maps"]["external_official_query"],
        p0_m30_sessions,
    )

    submit = core.choose_submit_candidate(
        p0_m33=_ext(p0_m33),
        p1_m33=_ext(p1_m33),
        means_m33=_ext(means_m33),
        p1_m33_shuffle=_ext(p1_m33_shuffle),
        means_m33_shuffle=_ext(means_m33_shuffle),
        p1_m33_vs_p0=p1_m33_vs_p0,
        means_m33_vs_p0=means_m33_vs_p0,
        means_shuffle_vs_p0=means_m33_shuffle_vs_p0,
        p1_shuffle_vs_p0=p1_m33_shuffle_vs_p0,
    )

    payload_receipt = None
    root.mkdir(parents=True, exist_ok=False)
    if submit["export_licensed"]:
        progress(f"exporting cached identities for {submit['candidate']}")
        if submit["candidate"] == "means_m33_retrain":
            student.id_encoder.load_state_dict(means_m33_state, strict=True)
            export_mask = _mask("means")
        else:
            student.id_encoder.load_state_dict(states["p1"], strict=True)
            export_mask = _mask("p1_full")
        student.id_encoder.to(device)
        payload_receipt = _export_cached_identities(
            repo_root=repo_root,
            student=student,
            datasets=datasets,
            device=device,
            candidate=str(submit["candidate"]),
            contrast_mask=export_mask,
            metadata=metadata,
        )

    result = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "workorder": plan.WORKORDER_RELATIVE,
        "elapsed_seconds": float(time.monotonic() - started),
        "gpu": {
            "cuda_visible_devices": expected_cvd,
            "name": torch.cuda.get_device_name(0),
        },
        "cost": core.cost_card(),
        "interpretability": interp_table,
        "m30_means_retrain": {
            "summaries": means_m30["summaries"],
            "vs_p0": means_m30_vs_p0,
            "shuffle_vs_p0": means_m30_shuffle_vs_p0,
            "training": means_m30_train,
        },
        "m33": {
            "p0": {"summaries": p0_m33["summaries"]},
            "p1_transfer": {"summaries": p1_m33["summaries"], "vs_p0": p1_m33_vs_p0},
            "p1_shuffle": {"summaries": p1_m33_shuffle["summaries"], "vs_p0": p1_m33_shuffle_vs_p0},
            "means_retrain": {
                "summaries": means_m33["summaries"],
                "vs_p0": means_m33_vs_p0,
                "shuffle_vs_p0": means_m33_shuffle_vs_p0,
                "training": means_m33_train,
            },
        },
        "submit": submit,
        "payload": payload_receipt,
        "evalai_push": False,
    }
    _atomic_json(root / "score.json", result)
    _atomic_json(root / "terminal.json", {"schema": plan.SCHEMA, "status": "TERMINAL", "submit": submit, "cost": result["cost"]})
    buffer = __import__("io").BytesIO()
    torch.save({"means_m30": means_m30_state, "means_m33": means_m33_state}, buffer)
    _atomic_bytes(root / "film_states.pt", buffer.getvalue())
    return result
