#!/usr/bin/env python3
"""Export official-compatible M2 cached identities for the D-opt-4 activity-30 arm.

Calibration phase (offline, build time, per session):

1. greedy forward D-optimal selection of k=4 trials from the finite-angle
   candidates inside the first 30 labelled calibration trials
   (frozen law of the sealed m2_t4_activity_budget_screen_v1);
2. ridge lambda=0.1 T4 fit on the selected four trials' spike-sum rates;
3. B3S activity pool = the full label-free first-30 calibration block
   (activity needs no target labels and every calibration-block trial belongs
   to the official calibration phase, so the pool is contract-legal);
4. frozen-checkpoint ``compute_identity`` -> one cached ``E[96,50]``.

This is the sealed screen's ``ridge_activity30_m4`` cell, deployment-shaped.
Deployment phase (EvalAI runtime) only serves the cached identities; see
``act30_dopt4_decoder.py``.  This exporter is deliberately read-only outside
its own ``artifacts`` directory and refuses to overwrite an existing payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
for _path in (ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from laws import (  # noqa: E402
    ACTIVITY_BUDGET,
    ACTIVITY_HORIZON,
    RIDGE_NORMALIZED_LAMBDA,
    SUPPORT_BUDGET,
    select_dopt4_support,
    select_first30_activity_pool,
    verify_against_sealed_sources,
)

CHANNELS = 96
WINDOW_SIZE = 50
TRIAL_LENGTH = 100

# Sealed anchors from tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json
# (schema m2_t4_activity_budget_screen_v1, checkpoint 25d7bc72...): the
# ridge_activity30_m4 rows this deployment reproduces.
SEALED_SCREEN_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_SCREEN_SHA256 = None  # bound at export time from the sidecar on disk
SEALED_CELL = "ridge_activity30_m4"
SEALED_EXTERNAL_ACTIVITY30_M4_MEAN = 0.2909923623168425
SEALED_WITHIN_ACTIVITY30_M4_MEAN = 0.6772200181830803

# Previously deployed, officially scored image of the same law
# (sua_exploration/evalai_t4_m2_activity_budget, m4 arm).  Recorded as a
# read-only continuity cross-check (informational; the bitwise gates below
# bind to the sealed screen rows, not to this receipt).
PRIOR_DEPLOYMENT_RECEIPT_RELATIVE = (
    "sua_exploration/evalai_t4_m2_activity_budget/artifacts/"
    "t4_m2_seed42_ridge_m4_activity30_identity.receipt.json"
)


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


def array_sha256_screen_law(value: np.ndarray) -> str:
    """The sealed screen's array hash (tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py:54-64)."""
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_sealed_screen_rows() -> dict[str, dict[str, Any]]:
    path = ROOT / SEALED_SCREEN_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for row in payload["rows"]:
        if row["cell"] != SEALED_CELL:
            continue
        rows[(row["surface"], row["session"])] = row
    require(len(rows) == 13, "sealed screen activity30_m4 row coverage drift")
    return rows


def load_prior_deployment_records() -> dict[str, dict[str, Any]]:
    path = ROOT / PRIOR_DEPLOYMENT_RECEIPT_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["session_records"]
    require(len(records) == 13, "prior deployment receipt coverage drift")
    return records


def build_static_identity_inputs(
    dataset: Any, session: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return (selected, activity, side, evidence) for one session.

    ``selected`` is the D-opt-4 support; ``activity`` is the label-free
    first-30 calibration block (the static B3S pool);
    ``side`` is the normalized ridge T4 carrier from the sealed law.
    """
    from laws import sealed_fit_ridge_side

    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
    selected = select_dopt4_support(angles)

    calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
    require(
        calibration.ndim == 3 and calibration.shape[0] >= ACTIVITY_HORIZON,
        f"{session}: calibration activity shape drift",
    )
    activity = select_first30_activity_pool(calibration)
    require(
        activity.shape == (ACTIVITY_BUDGET, TRIAL_LENGTH, CHANNELS)
        and np.isfinite(activity).all(),
        f"{session}: activity-30 pool shape/nonfinite drift",
    )
    require(
        np.array_equal(activity, calibration[:ACTIVITY_HORIZON]),
        f"{session}: activity pool is not exactly the first-30 calibration block",
    )
    verification = verify_against_sealed_sources(angles, calibration, selected)

    side, evidence = sealed_fit_ridge_side(dataset, session, selected)
    require(side.shape == (CHANNELS, 4) and np.isfinite(side).all(), f"{session}: side drift")
    return selected, activity, side, {
        **evidence,
        "selection": "doptimal_noncentre_first30",
        "selection_law_verification": verification,
        "label_budget": SUPPORT_BUDGET,
        "activity_budget": ACTIVITY_BUDGET,
        "activity_pool": "first30_calibration_block_label_free",
        "activity_sha256": sha256_array(activity),
        "activity_sha256_screen_law": array_sha256_screen_law(activity),
    }


def default_output() -> Path:
    return ARTIFACT_ROOT / "t4_m2_seed42_dopt4_act30_identity.pkl"


def export_payload(output: Path) -> dict[str, object]:
    require(not output.exists(), f"refusing to overwrite {output}")
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_NORMALIZATION_SHA256,
        EXPECTED_TEACHER_SHA256,
        calibration_file_map,
        load_frozen_model_and_data,
        manual_decode,
    )

    model, data_module, task_config, metadata = load_frozen_model_and_data()
    require(metadata["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "T4 checkpoint drift")
    require(metadata["teacher_checkpoint_sha256"] == EXPECTED_TEACHER_SHA256, "teacher drift")
    require(metadata["normalization_sha256"] == EXPECTED_NORMALIZATION_SHA256, "normalizer drift")
    student = model.student.cpu().eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    session_to_tag = calibration_file_map(ROOT / "SPINT-main/data/000953", task_config)

    sealed_rows = load_sealed_screen_rows()
    prior_records = load_prior_deployment_records()
    screen_path = ROOT / SEALED_SCREEN_RELATIVE
    screen_sidecar = screen_path.with_name(screen_path.name + ".sha256")
    screen_sha = sha256_file(screen_path)
    require(
        screen_sidecar.read_text(encoding="ascii").split()[0] == screen_sha,
        "sealed screen sha sidecar drift",
    )

    identities: dict[str, np.ndarray] = {}
    records: dict[str, object] = {}
    max_direct = 0.0
    max_decoder = 0.0
    prior_identity_matches = 0
    surface_of_dataset = {"train": "within_post30", "heldout": "external_official_query"}
    import torch

    for dataset_role, dataset in (
        ("train", data_module.train_dataset),
        ("heldout", data_module.val_heldout_dataset),
    ):
        require(dataset is not None, "official export requires both M2 session sets")
        for session in sorted(dataset.calib_trialized_neural_features):
            require(session in session_to_tag, f"{session}: missing official dataset tag")
            selected, activity, side, evidence = build_static_identity_inputs(dataset, session)
            sealed = sealed_rows[(surface_of_dataset[dataset_role], session)]
            # Bitwise anchors against the sealed screen activity30_m4 run.
            require(
                evidence["selected_indices"] == sealed["side_evidence"]["selected_indices"],
                f"{session}: D-opt selection disagrees with the sealed screen",
            )
            require(
                evidence["activity_sha256_screen_law"] == sealed["activity_sha256"],
                f"{session}: activity-30 pool bytes disagree with the sealed screen",
            )
            require(
                evidence["normalized_t4_sha256"] == sealed["side_evidence"]["normalized_t4_sha256"],
                f"{session}: normalized ridge T4 bytes disagree with the sealed screen",
            )

            support = torch.from_numpy(activity).unsqueeze(0)
            side_tensor = torch.from_numpy(side).unsqueeze(0)
            with torch.inference_mode():
                identity = student.compute_identity(support, side_features=side_tensor)
                neural = torch.from_numpy(
                    np.asarray(dataset.neural_data[session][50:100], dtype=np.float32)
                ).unsqueeze(0)
                direct, _ = student(neural, calib_trials=support, side_features=side_tensor)
                cached, _ = student(neural, identity=identity)
                decoder_only = manual_decode(student.decoder, neural, identity)
            direct_error = float((direct - cached).abs().max().item())
            decoder_error = float((direct - decoder_only).abs().max().item())
            require(
                direct_error == 0.0 and decoder_error == 0.0,
                f"{session}: cached identity is not bit-exact",
            )
            max_direct = max(max_direct, direct_error)
            max_decoder = max(max_decoder, decoder_error)
            identity_np = np.ascontiguousarray(identity.squeeze(0).numpy(), dtype=np.float32)
            require(
                identity_np.shape == (CHANNELS, WINDOW_SIZE) and np.isfinite(identity_np).all(),
                f"{session}: identity drift",
            )
            prior_identity_sha256 = prior_records[session]["identity_sha256"]
            prior_identity_matches += int(
                sha256_array(identity_np) == prior_identity_sha256
            )
            tag = session_to_tag[session]
            require(tag not in identities, "duplicate official dataset tag")
            identities[tag] = identity_np
            records[session] = {
                "dataset_tag": tag,
                "surface": surface_of_dataset[dataset_role],
                "selected_indices": [int(v) for v in selected],
                "label_budget": SUPPORT_BUDGET,
                "activity_budget": ACTIVITY_BUDGET,
                "activity_shape": list(activity.shape),
                "activity_sha256": evidence["activity_sha256"],
                "side_sha256": sha256_array(side),
                "identity_sha256": sha256_array(identity_np),
                "prior_deployed_identity_sha256": prior_identity_sha256,
                "prior_deployed_identity_sha256_match": bool(
                    sha256_array(identity_np) == prior_identity_sha256
                ),
                "side_evidence": evidence,
                "sealed_screen_activity30_m4_r2": float(sealed["r2"]),
                "direct_vs_cached_max_abs": direct_error,
                "direct_vs_decoder_max_abs": decoder_error,
            }
    require(
        len(identities) == 13 and set(records) == set(session_to_tag),
        "official session coverage drift",
    )

    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    payload = {
        "schema_version": "e8_t4_m2_cached_identity_v1",
        "task": task_config.task,
        "decoder": decoder,
        "identity_by_dataset_tag": identities,
        "window_size": WINDOW_SIZE,
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
            "normalization_sha256": metadata["normalization_sha256"],
            "screen_id": "m2_spint_t4_mainline_fp32_v1",
            "sealed_screen_result": SEALED_SCREEN_RELATIVE,
            "sealed_screen_sha256": screen_sha,
            "arm": "dopt4_static_act30",
            "seed": 42,
            "label_budget": SUPPORT_BUDGET,
            "activity_budget": ACTIVITY_BUDGET,
            "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
            "calibration_selection": "greedy_forward_d_optimal_first30_k4",
            "calibration_activity_pool": "first30_calibration_block_label_free",
            "calibration_uses_target_labels": True,
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
            "session_count": len(identities),
            "max_direct_vs_cached_identity_abs": max_direct,
            "max_direct_vs_decoder_only_abs": max_decoder,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
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
        "schema_version": "m2_dopt4_act30_official_payload_receipt_v1",
        "status": "EXPORTED_NOT_SUBMITTED",
        "arm": "dopt4_static_act30",
        "label_budget": SUPPORT_BUDGET,
        "activity_budget": ACTIVITY_BUDGET,
        "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
        "payload_path": str(output),
        "payload_bytes": output.stat().st_size,
        "payload_sha256": sha256_file(output),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "sealed_screen_result": SEALED_SCREEN_RELATIVE,
        "sealed_screen_sha256": screen_sha,
        "prior_deployment_receipt": PRIOR_DEPLOYMENT_RECEIPT_RELATIVE,
        "prior_deployed_identity_sha256_matches": prior_identity_matches,
        "sealed_external_activity30_m4_equal_session_mean": SEALED_EXTERNAL_ACTIVITY30_M4_MEAN,
        "sealed_within_activity30_m4_equal_session_mean": SEALED_WITHIN_ACTIVITY30_M4_MEAN,
        "session_count": len(identities),
        "max_direct_vs_cached_identity_abs": max_direct,
        "max_direct_vs_decoder_only_abs": max_decoder,
        "session_records": records,
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or default_output()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_dopt4_act30_official_payload_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE_NO_SUBMISSION",
            "arm": "dopt4_static_act30",
            "label_budget": SUPPORT_BUDGET,
            "activity_budget": ACTIVITY_BUDGET,
            "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
            "output": str(output),
        }, sort_keys=True))
        return
    print(json.dumps(export_payload(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
