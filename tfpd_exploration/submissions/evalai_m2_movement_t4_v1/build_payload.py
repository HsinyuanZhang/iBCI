#!/usr/bin/env python3
"""Export cached M2 identities using the frozen movement-window T4 carrier."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_movement_t4_ablation_v1.physical import (
    HORIZON,
    START_BIN,
    STOP_BIN,
    _fit_source_normalizer,
    _override,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "artifacts/t4_m2_seed42_movement_t4_identity.pkl"


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
    from tfpd_exploration.src.m2_hold_film_probe_v1.physical import session_side

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    student = model.student.cuda().eval()
    train = data_module.train_dataset
    external = data_module.val_heldout_dataset
    mean, std, normalizer = _fit_source_normalizer(train)
    move_train, _ = _override(train, mean, std)
    move_external, _ = _override(external, mean, std)
    config = FalconConfig(task=FalconTask.m2)
    tag_map = calibration_file_map(M2_DATA_DIR, config)
    identities: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    max_parity = 0.0
    for dataset in (move_train, move_external):
        for session in sorted(dataset.calib_trialized_neural_features):
            activity, _, evidence = session_side(
                dataset, session, shuffle=False, contrast_mask=np.zeros(4, dtype=np.float32),
                horizon=HORIZON, t4_mode="native",
            )
            side = dataset._native_t4_side_features(session, 0, HORIZON)
            support = torch.from_numpy(activity).unsqueeze(0).cuda()
            side_t = torch.from_numpy(np.asarray(side, dtype=np.float32)).unsqueeze(0).cuda()
            neural = torch.from_numpy(
                np.asarray(dataset.neural_data[session][50:100], dtype=np.float32)
            ).unsqueeze(0).cuda()
            with torch.inference_mode():
                identity = student.compute_identity(support, side_features=side_t)
                cached, _ = student(neural, identity=identity)
                direct = manual_decode(student.decoder, neural, identity)
            parity = float((cached - direct).abs().max().item())
            max_parity = max(max_parity, parity)
            if parity != 0.0:
                raise RuntimeError(f"cached decoder parity drift for {session}: {parity}")
            value = np.ascontiguousarray(identity.squeeze(0).cpu().numpy(), dtype=np.float32)
            tag = tag_map[session]
            identities[tag] = value
            records[session] = {
                "dataset_tag": tag,
                "identity_sha256": sha256_array(value),
                "activity": evidence,
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
            "candidate": "movement_t4_bins5_30_m33",
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "old_normalization_sha256": metadata["normalization_sha256"],
            "movement_window_bins": [START_BIN, STOP_BIN],
            "movement_window_ms": [100, 600],
            "normalizer": {"mean": mean.tolist(), "std": std.tolist()},
            "deployment_inputs": ["raw calibration neural counts", "trial boundaries", "sparse target direction"],
            "dense_behavior_used_at_deployment": False,
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {
        "schema": "m2_movement_t4_evalai_payload_v1",
        "payload_sha256": sha256(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "session_count": len(identities),
        "max_cached_decoder_parity": max_parity,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "normalizer": normalizer,
        "session_records": records,
    }
    (OUTPUT.parent / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: receipt[k] for k in ("payload_sha256", "bytes", "session_count", "max_cached_decoder_parity")}, sort_keys=True))


if __name__ == "__main__":
    main()
