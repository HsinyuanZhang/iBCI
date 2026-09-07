#!/usr/bin/env python3
"""Export the seed-42 profile-free MOVE-T4 conditioned adapter for EvalAI."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_hold_film_probe_v1.physical import (
    install_film,
    score_arm,
    session_side,
)
from tfpd_exploration.src.m2_movement_t4_ablation_v1.physical import (
    HORIZON,
    START_BIN,
    STOP_BIN,
    _fit_source_normalizer,
    _override,
)


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "tfpd_exploration/results/m2_movement_t4_film_ablation_v1"
OUTPUT = Path(__file__).resolve().parent / "artifacts/t4_m2_seed42_movement_t4_empty_identity.pkl"
EXPECTED_LOCAL = 0.35351138886628103


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES=0 required")
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    from falcon_challenge.config import FalconConfig, FalconTask
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        EXPECTED_SESSION_COUNT,
        M2_DATA_DIR,
        calibration_file_map,
        load_frozen_model_and_data,
        manual_decode,
        sha256_array,
    )

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    student = model.student.cuda().eval()
    base_encoder = student.id_encoder
    train_base = data_module.train_dataset
    external_base = data_module.val_heldout_dataset
    mean, std, normalizer = _fit_source_normalizer(train_base)
    train, _ = _override(train_base, mean, std)
    external, _ = _override(external_base, mean, std)
    datasets = {"within_post30": train, "external_official_query": external}

    canonical_payload = torch.load(
        ROOT / probe_plan.RESULT_ROOT_RELATIVE / "film_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    if "p0" not in canonical_payload:
        raise RuntimeError("canonical p0 state missing")
    head_payload = torch.load(
        RESULT / "seed42/film_heads.pt", map_location="cpu", weights_only=False
    )
    if "EMPTY" not in head_payload:
        raise RuntimeError("seed42 EMPTY head missing")
    student.id_encoder = base_encoder
    install_film(student, torch.device("cuda:0"), film_input="t4_plus_contrast", seed=42)
    student.id_encoder.load_state_dict(canonical_payload["p0"], strict=True)
    full_state = student.id_encoder.state_dict()
    for key, value in head_payload["EMPTY"].items():
        if key not in full_state:
            raise RuntimeError(f"unexpected EMPTY head key {key}")
        full_state[key] = value
    student.id_encoder.load_state_dict(full_state, strict=True)
    student.id_encoder.freeze_base_path()
    student.eval()

    zero = np.zeros(4, dtype=np.float32)
    score = score_arm(
        student=student,
        datasets=datasets,
        device=torch.device("cuda:0"),
        batch_size=1024,
        shuffle=False,
        arm="MOVE-EMPTY-S42-EXPORT",
        horizon=HORIZON,
        t4_mode="native",
        contrast_mask=zero,
    )
    observed = float(score["summaries"]["external_official_query"]["equal_session_mean"])
    if abs(observed - EXPECTED_LOCAL) > 1.0e-7:
        raise RuntimeError(f"seed42 EMPTY score drift: {observed} != {EXPECTED_LOCAL}")

    config = FalconConfig(task=FalconTask.m2)
    tag_map = calibration_file_map(M2_DATA_DIR, config)
    identities: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    max_parity = 0.0
    for dataset in datasets.values():
        for session in sorted(dataset.calib_trialized_neural_features):
            activity, side, evidence = session_side(
                dataset,
                session,
                shuffle=False,
                contrast_mask=zero,
                horizon=HORIZON,
                t4_mode="native",
            )
            support = torch.from_numpy(activity).unsqueeze(0).cuda()
            side_tensor = torch.from_numpy(side).unsqueeze(0).cuda()
            neural = torch.from_numpy(
                np.asarray(dataset.neural_data[session][50:100], dtype=np.float32)
            ).unsqueeze(0).cuda()
            with torch.inference_mode():
                identity = student.compute_identity(support, side_features=side_tensor)
                cached, _ = student(neural, identity=identity)
                direct = manual_decode(student.decoder, neural, identity)
            parity = float((cached - direct).abs().max().item())
            max_parity = max(max_parity, parity)
            if parity != 0.0:
                raise RuntimeError(f"cached decoder parity drift for {session}: {parity}")
            identity_np = np.ascontiguousarray(identity.squeeze(0).cpu().numpy(), dtype=np.float32)
            tag = tag_map[session]
            identities[tag] = identity_np
            records[session] = {
                "dataset_tag": tag,
                "identity_sha256": sha256_array(identity_np),
                "activity_and_side": evidence,
            }
    if len(identities) != EXPECTED_SESSION_COUNT:
        raise RuntimeError(f"expected {EXPECTED_SESSION_COUNT} identities, got {len(identities)}")

    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    payload = {
        "schema_version": "e8_t4_m2_cached_identity_v1",
        "task": config.task,
        "decoder": decoder,
        "identity_by_dataset_tag": identities,
        "window_size": int(decoder.window_size),
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            "candidate": "movement_t4_m33_empty_adapter_seed42",
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "movement_window_bins": [START_BIN, STOP_BIN],
            "movement_window_ms": [100, 600],
            "profile": "positive_zero_4d",
            "trainable_adapter_parameters": 1224,
            "adapter_training": "12 epochs on seven held-in sessions; base identity and decoder frozen",
            "local_external_equal_session_mean": observed,
            "normalizer": {"mean": mean.tolist(), "std": std.tolist()},
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {
        "schema": "m2_movement_t4_empty_evalai_payload_v1",
        "payload_sha256": sha256(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "session_count": len(identities),
        "max_cached_decoder_parity": max_parity,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "local_external_equal_session_mean": observed,
        "normalizer": normalizer,
        "session_records": records,
        "evalai_push": False,
    }
    (OUTPUT.parent / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: receipt[key]
        for key in ("payload_sha256", "bytes", "session_count", "max_cached_decoder_parity", "local_external_equal_session_mean")
    }, sort_keys=True))


if __name__ == "__main__":
    main()
