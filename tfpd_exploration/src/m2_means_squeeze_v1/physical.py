"""Train the native-M33 FiLM catalog, pick the EvalAI-max arm, export cached identities."""

from __future__ import annotations

import json
import os
import pickle
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
    session_side,
    train_film,
)
from tfpd_exploration.src.m2_p1_contrast_opt_v1 import plan as opt_plan

from . import core, plan


def _mask(name: str) -> np.ndarray:
    return np.asarray(plan.MASKS[name], dtype=np.float32)


def _ext(arm: dict[str, object]) -> dict[str, object]:
    return arm["summaries"]["external_official_query"]


def _clone_state(student: Any) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in student.id_encoder.state_dict().items()}


def _restore_b3s(student: Any, base_encoder: Any) -> None:
    student.id_encoder = base_encoder


def _warm_start(student: Any, probe_p0: dict[str, torch.Tensor], film_input: str) -> None:
    if film_input != "t4_plus_contrast":
        return
    student.id_encoder.load_state_dict(probe_p0, strict=True)
    student.id_encoder.to(next(student.decoder.parameters()).device)
    student.id_encoder.freeze_base_path()


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
    from falcon_challenge.config import FalconConfig, FalconTask

    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        EXPECTED_SESSION_COUNT,
        M2_DATA_DIR,
        calibration_file_map,
        manual_decode,
        sha256_array,
        sha256_file,
    )

    task_config = FalconConfig(task=FalconTask.m2)
    session_to_tag = calibration_file_map(M2_DATA_DIR, task_config)
    identity_by_tag: dict[str, np.ndarray] = {}
    session_records: dict[str, object] = {}
    max_direct_vs_identity = 0.0
    max_direct_vs_decoder = 0.0
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
                t4_mode=plan.T4_MODE,
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
            "evalai_push": True,
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
        "evalai_push": True,
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

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    core.require(metadata["checkpoint_sha256"] == plan.CHECKPOINT_SHA256, "checkpoint drift")
    core.require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "normalizer drift")
    student = model.student.to(device)
    base_b3s = student.id_encoder
    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    core.require(datasets["within_post30"] is not None, "within dataset missing")
    core.require(datasets["external_official_query"] is not None, "external dataset missing")

    def progress(message: str) -> None:
        print(message, flush=True)

    sealed_path = opt_plan.result_root(repo_root) / "film_states.pt"
    probe_states_path = Path(repo_root) / probe_plan.RESULT_ROOT_RELATIVE / "film_states.pt"
    core.require(sealed_path.is_file(), f"sealed means weights missing: {sealed_path}")
    core.require(probe_states_path.is_file(), f"probe film_states missing: {probe_states_path}")
    sealed_payload = torch.load(sealed_path, map_location="cpu", weights_only=False)
    probe_payload = torch.load(probe_states_path, map_location="cpu", weights_only=False)
    core.require("means_m33" in sealed_payload, "film_states.pt missing means_m33")
    core.require("p0" in probe_payload, "probe film_states.pt missing p0")
    probe_p0 = {key: value.detach().cpu().clone() for key, value in probe_payload["p0"].items()}

    _restore_b3s(student, base_b3s)
    install_film(student, device, film_input="t4_plus_contrast", seed=probe_plan.SEED)
    _warm_start(student, probe_p0, "t4_plus_contrast")
    progress("scoring P0 native M33 T4-only")
    p0 = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="p0_native_m33",
        horizon=plan.M33_HORIZON,
        t4_mode=plan.T4_MODE,
        contrast_mask=np.zeros(4, dtype=np.float32),
    )
    p0_mean = float(_ext(p0)["equal_session_mean"])
    core.require(
        abs(p0_mean - plan.P0_M33_EXTERNAL) <= 1.0e-6,
        f"P0 M33 drift {p0_mean} vs {plan.P0_M33_EXTERNAL}",
    )

    student.id_encoder.load_state_dict(sealed_payload["means_m33"], strict=True)
    student.id_encoder.to(device)
    student.id_encoder.freeze_base_path()
    progress("scoring sealed means_m33")
    sealed_score = score_arm(
        student=student,
        datasets=datasets,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        arm="sealed_means_m33",
        horizon=plan.M33_HORIZON,
        t4_mode=plan.T4_MODE,
        contrast_mask=_mask("means"),
    )
    sealed_arm = {
        "name": "sealed_means_m33",
        "summaries": sealed_score["summaries"],
        "vs_p0": core.paired_contrast(
            sealed_score["session_maps"]["external_official_query"],
            p0["session_maps"]["external_official_query"],
        ),
        "state": {key: value.detach().cpu().clone() for key, value in sealed_payload["means_m33"].items()},
        "config": plan.config_by_name("sealed_means_m33"),
    }

    trained_arms: list[dict[str, object]] = []
    states: dict[str, dict[str, torch.Tensor]] = {"sealed_means_m33": sealed_arm["state"]}
    for row in plan.CONFIGS:
        name = str(row["name"])
        progress(f"training {name}")
        _restore_b3s(student, base_b3s)
        install_film(
            student,
            device,
            film_input=str(row["film_input"]),
            seed=int(row["seed"]),
        )
        _warm_start(student, probe_p0, str(row["film_input"]))
        training = train_film(
            student=student,
            train_dataset=datasets["within_post30"],
            device=device,
            shuffle=False,
            contrast_mask=_mask(str(row["mask"])),
            horizon=int(row["horizon"]),
            t4_mode=str(row["t4_mode"]),
            learning_rate=float(row["learning_rate"]),
            epochs=int(row["epochs"]),
            seed=int(row["seed"]),
            windows_per_session=int(row["windows_per_session"]),
        )
        labeled = score_arm(
            student=student,
            datasets=datasets,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            arm=name,
            horizon=int(row["horizon"]),
            t4_mode=str(row["t4_mode"]),
            contrast_mask=_mask(str(row["mask"])),
        )
        shuffled = score_arm(
            student=student,
            datasets=datasets,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            arm=f"{name}_shuffle",
            horizon=int(row["horizon"]),
            t4_mode=str(row["t4_mode"]),
            contrast_mask=_mask(str(row["mask"])),
        )
        state = _clone_state(student)
        states[name] = state
        arm = {
            "name": name,
            "config": row,
            "summaries": labeled["summaries"],
            "shuffle_summaries": shuffled["summaries"],
            "vs_p0": core.paired_contrast(
                labeled["session_maps"]["external_official_query"],
                p0["session_maps"]["external_official_query"],
            ),
            "shuffle_vs_p0": core.paired_contrast(
                shuffled["session_maps"]["external_official_query"],
                p0["session_maps"]["external_official_query"],
            ),
            "training": training,
            "film_head_sha256": labeled["film_head_sha256"],
        }
        trained_arms.append(arm)
        progress(
            f"{name} external={float(_ext(labeled)['equal_session_mean']):.6f} "
            f"median={float(_ext(labeled)['equal_session_median']):.6f} "
            f"shuffle={float(shuffled['summaries']['external_official_query']['equal_session_mean']):.6f}"
        )

    shuffle_means = {
        str(arm["name"]): float(arm["shuffle_summaries"]["external_official_query"]["equal_session_mean"])
        for arm in trained_arms
    }
    submit = core.choose_submit_candidate(
        p0_external=_ext(p0),
        arms=trained_arms + [sealed_arm],
        shuffle_means=shuffle_means,
        sealed_means_m33=sealed_arm,
    )
    core.require(submit["export_licensed"] and submit["candidate"], "no export-licensed candidate")

    winner_name = str(submit["candidate"])
    winner_cfg = plan.config_by_name(winner_name)
    _restore_b3s(student, base_b3s)
    install_film(
        student,
        device,
        film_input=str(winner_cfg["film_input"]),
        seed=int(winner_cfg["seed"]),
    )
    student.id_encoder.load_state_dict(states[winner_name], strict=True)
    student.id_encoder.to(device)
    student.id_encoder.freeze_base_path()

    root.mkdir(parents=True, exist_ok=False)
    progress(f"exporting cached identities for {winner_name}")
    payload_receipt = _export_cached_identities(
        repo_root=repo_root,
        student=student,
        datasets=datasets,
        device=device,
        candidate=winner_name,
        contrast_mask=_mask(str(winner_cfg["mask"])),
        metadata=metadata,
    )

    table = {
        str(arm["name"]): {
            "external_mean": float(arm["summaries"]["external_official_query"]["equal_session_mean"]),
            "external_median": float(arm["summaries"]["external_official_query"]["equal_session_median"]),
            "within_mean": float(arm["summaries"]["within_post30"]["equal_session_mean"]),
            "vs_p0": arm["vs_p0"],
            "shuffle_external_mean": None
            if "shuffle_summaries" not in arm
            else float(arm["shuffle_summaries"]["external_official_query"]["equal_session_mean"]),
            "passes_floor": core.passes_floor(arm, _ext(p0)),
            "config": arm.get("config"),
        }
        for arm in trained_arms + [sealed_arm]
    }
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
        "p0": {"summaries": p0["summaries"]},
        "arms": table,
        "submit": submit,
        "payload": payload_receipt,
        "evalai_push": False,
        "evalai_push_next": True,
    }
    _atomic_json(root / "score.json", result)
    _atomic_json(
        root / "terminal.json",
        {
            "schema": plan.SCHEMA,
            "status": "TERMINAL",
            "submit": submit,
            "payload_sha256": None if payload_receipt is None else payload_receipt["payload_sha256"],
        },
    )
    buffer = __import__("io").BytesIO()
    torch.save(states, buffer)
    _atomic_bytes(root / "film_states.pt", buffer.getvalue())
    return result
