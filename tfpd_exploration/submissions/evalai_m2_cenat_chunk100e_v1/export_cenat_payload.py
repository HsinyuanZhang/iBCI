#!/usr/bin/env python3
"""Export the Ce-NAT chunk100e TTA payload (build-time calibration only).

Seals (per the audit): training.json sidecar verify -> checkpoint body SHA
verify -> checkpoint bytes copied into artifacts 0444+sidecar -> payload with
frozen decoder + id encoder + per-tag seed pool/side -> receipt binding
training receipt SHA -> checkpoint SHA -> V4 score SHA -> payload SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TRAINING_RECEIPT = ROOT / "tfpd_exploration/results/m2_ajpf_c_v2/training.json"
V4_SCORE = ROOT / "tfpd_exploration/results/m2_ajpf_c_v4/score.json"
CHECKPOINT = ROOT / "tfpd_exploration/results/m2_ajpf_c_v2/checkpoints/Ce-NAT_epoch12.pt"
CHECKPOINT_SHA = "c5672a2bfc94f55729d64e5eeb2f6917162c345713e7102dec4ad371cd147ea2"
STUDENT_STATE_SHA = "f1d89a60967513dcc5613c129a67e16644d10a1832aaf0c750029cfe2a19f15f"

PAYLOAD_NAME = "t4_m2_seed42_cenat_chunk100e_tta.pkl"
PAYLOAD_SCHEMA = "e8_t4_m2_cenat_chunk100e_tta_v1"
PAYLOAD_ARM = "cenat_chunk100e_tta"
CHANNELS, WINDOW_SIZE, SEED_ROWS = 96, 50, 30


class ExportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def seal_0444(source: Path, target: Path) -> str:
    digest = sha256_file(source)
    require(digest == CHECKPOINT_SHA, f"checkpoint body drift: {digest}")
    shutil.copyfile(source, target)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"schema": "m2_cenat_chunk100e_export_dry_v1",
                          "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE",
                          "arm": PAYLOAD_ARM}, sort_keys=True))
        return
    import sys

    apfg_package = ROOT / "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"
    for path in (ROOT, ROOT / "streaming_calibration_exp", Path(__file__).resolve().parent,
                 apfg_package):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401
    import torch

    # 1) sealed training receipt (sidecar-verified)
    receipt_sha = sha256_file(TRAINING_RECEIPT)
    sidecar = (TRAINING_RECEIPT.with_name(TRAINING_RECEIPT.name + ".sha256")).read_text("ascii").split()[0]
    require(sidecar == receipt_sha, "training receipt sidecar drift")
    training = json.loads(TRAINING_RECEIPT.read_text(encoding="utf-8"))
    entry = training["laws"]["chunk100e"]["checkpoints"]["Ce-NAT"]
    require(entry["sha256"] == CHECKPOINT_SHA, "training receipt checkpoint literal drift")

    # 2) sealed V4 score
    v4_sha = sha256_file(V4_SCORE)
    v4_sidecar = (V4_SCORE.with_name(V4_SCORE.name + ".sha256")).read_text("ascii").split()[0]
    require(v4_sidecar == v4_sha, "V4 score sidecar drift")

    # 3) seal the checkpoint bytes
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sealed_target = ARTIFACTS / "Ce-NAT_epoch12.sealed.pt"
    if sealed_target.exists():
        sealed_sha = sha256_file(sealed_target)
        require(sealed_sha == CHECKPOINT_SHA, "existing sealed checkpoint drift")
        require((sealed_target.with_name(sealed_target.name + ".sha256")).read_text("ascii").split()[0]
                == sealed_sha, "sealed sidecar drift")
    else:
        sealed_sha = seal_0444(CHECKPOINT, sealed_target)

    # 4) strict load
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        EXPECTED_CHECKPOINT_SHA256, load_frozen_model_and_data)
    model, data_module, task_config, metadata = load_frozen_model_and_data()
    require(metadata["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "base checkpoint drift")
    state = torch.load(io.BytesIO(CHECKPOINT.read_bytes()), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    student = model.student
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.physical import _student_state_sha
    got = _student_state_sha(model)
    require(got == STUDENT_STATE_SHA, f"strict-load student state drift: {got}")

    decoder = student.decoder.cpu().eval()
    id_encoder = student.id_encoder.cpu().eval()
    for module in (decoder, id_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    # 5) per-tag build-time calibration (public calib only)
    from laws import sealed_fit_ridge_side, select_dopt4_support, select_first30_activity_pool
    from sua_exploration.evalai_t4_m2.export_t4_payload import calibration_file_map

    session_to_tag = calibration_file_map(ROOT / "SPINT-main/data/000953", task_config)
    seed_by_tag, side_by_tag = {}, {}
    records = {}
    for dataset in (data_module.train_dataset, data_module.val_heldout_dataset):
        require(dataset is not None, "both session sets required")
        for session in sorted(dataset.calib_trialized_neural_features):
            require(session in session_to_tag, f"{session}: no official tag")
            angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
            selected = select_dopt4_support(angles)
            calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
            seed_pool = select_first30_activity_pool(calibration)
            side, _evidence = sealed_fit_ridge_side(dataset, session, selected)
            require(seed_pool.shape == (SEED_ROWS, 100, CHANNELS) and np.isfinite(seed_pool).all(),
                    f"{session}: seed drift")
            require(side.shape == (CHANNELS, 4) and np.isfinite(side).all(), f"{session}: side drift")
            tag = session_to_tag[session]
            require(tag not in seed_by_tag, "duplicate tag")
            seed_by_tag[tag] = np.ascontiguousarray(seed_pool, dtype=np.float32)
            side_by_tag[tag] = np.ascontiguousarray(side, dtype=np.float32)
            records[session] = {"dataset_tag": tag, "selected_indices": [int(v) for v in selected],
                                "seed_pool_sha256": sha256_array(seed_pool),
                                "side_sha256": sha256_array(side)}
    require(len(seed_by_tag) == 13, "session coverage drift")

    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": task_config.task,
        "decoder": decoder,
        "id_encoder_state": id_encoder.state_dict(),
        "seed_pool_by_dataset_tag": seed_by_tag,
        "side_by_dataset_tag": side_by_tag,
        "window_size": WINDOW_SIZE,
        "behavior_scaling_factor": 5.0,
        "metadata": {
            "arm": PAYLOAD_ARM,
            "test_time_adaptive": True,
            "label_budget": 4,
            "calibration_disclosure": (
                "T4 few-shot calibration uses the public calibration labels "
                "(D-opt4 support, ridge carrier); after the hidden query stream "
                "begins, the continual activity-memory update is fully label-free."),
            "online_state": "seed30_pool + chunk100e energy-gated commits (median rule), capacity 30, prefix-4 protected",
            "online_backward_pass": False,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "student_state_sha256": STUDENT_STATE_SHA,
            "base_checkpoint_sha256": metadata["checkpoint_sha256"],
            "normalization_sha256": metadata["normalization_sha256"],
            "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
            "session_count": len(seed_by_tag),
            "seed": 42,
        },
    }
    output = ARTIFACTS / PAYLOAD_NAME
    require(not output.exists(), f"refusing to overwrite {output}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    receipt = {
        "schema_version": "m2_cenat_chunk100e_payload_receipt_v1",
        "status": "EXPORTED_NOT_SUBMITTED",
        "arm": PAYLOAD_ARM,
        "chain": {
            "training_receipt_sha256": receipt_sha,
            "checkpoint_body_sha256": sealed_sha,
            "v4_score_sha256": v4_sha,
            "payload_sha256": sha256_file(output),
        },
        "payload_path": str(output),
        "payload_bytes": output.stat().st_size,
        "sealed_checkpoint": str(sealed_target),
        "session_count": len(seed_by_tag),
        "test_time_adaptive": True,
        "session_records": records,
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "arm", "session_count", "chain")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
