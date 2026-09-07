#!/usr/bin/env python3
"""Export the static-pool APFG M2 candidate as decoder + cached identities.

Calibration phase (offline, build time, per session):

1. greedy forward D-optimal selection of k=4 trials from the finite-angle
   candidates inside the first 30 labelled calibration trials, ridge
   lambda=0.1 T4 fit on the selected four trials, and the full label-free
   first-30 calibration block as the B3S activity pool -- byte-identical to
   the already officially scored ``dopt4_static_act30`` image;
2. native cached identity ``E_native[96,50]`` from the frozen checkpoint;
3. the APFG adapter (``tfpd_exploration/src/m2_anchored_postfusion_gate_v1/
   adapter.py``) is installed on the *same* in-memory frozen encoder and the
   identity is recomputed twice:
   - ``E_zero``  with alpha at exact IEEE ``+0.0``  -- must equal ``E_native``
     bitwise (the operational no-op sentinel);
   - ``E_learned`` with the frozen V2 alpha ``-0.20759029686450958`` -- the
     deployed identity;
4. per session, a 50-bin probe window must decode bitwise-identically through
   the adapter path (``calib_trials``) and the cached-identity path
   (``identity``) and the manual decoder path.

Deployment serves only ``E_learned``; ``E_native``/``E_zero`` ship in a
separate validation artifact for the local replay sentinel.  No online state,
no optimizer, no target access.  This exporter is read-only outside its own
``artifacts`` directory and refuses to overwrite an existing payload.
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
    CHANNELS,
    FROZEN_ALPHA,
    RIDGE_NORMALIZED_LAMBDA,
    SUPPORT_BUDGET,
    TRIAL_LENGTH,
    V2_RESULT_ROOT_RELATIVE,
    V2_SCORE_SHA256,
    V2_TERMINAL_SHA256,
    select_dopt4_support,
    select_first30_activity_pool,
    verify_against_sealed_sources,
    verify_v2_alpha_provenance,
)

WINDOW_SIZE = 50

PAYLOAD_SCHEMA = "e8_t4_m2_apfg_static_cached_identity_v1"
PAYLOAD_ARM = "apfg_static_act30_dopt4"
VALIDATION_SCHEMA = "e8_t4_m2_apfg_static_validation_v1"

# Sealed anchors from tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json
# (the ridge_activity30_m4 rows this deployment's native arm reproduces) and
# the prior officially scored deployment of the same law.
SEALED_SCREEN_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_CELL = "ridge_activity30_m4"
SEALED_EXTERNAL_ACTIVITY30_M4_MEAN = 0.2909923623168425
SEALED_WITHIN_ACTIVITY30_M4_MEAN = 0.6772200181830803
PRIOR_DEPLOYMENT_RECEIPT_RELATIVE = (
    "sua_exploration/evalai_t4_m2_activity_budget/artifacts/"
    "t4_m2_seed42_ridge_m4_activity30_identity.receipt.json"
)

DEFAULT_PAYLOAD_NAME = "t4_m2_seed42_dopt4_act30_apfg_identity.pkl"
DEFAULT_VALIDATION_NAME = "t4_m2_seed42_dopt4_act30_apfg_validation.pkl"


class ExportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def require_output_absent(output: Path) -> None:
    if Path(output).exists():
        raise ExportError(f"refusing to overwrite {output}")


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


def load_sealed_screen_rows() -> dict[tuple[str, str], dict[str, Any]]:
    path = ROOT / SEALED_SCREEN_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        if row["cell"] != SEALED_CELL:
            continue
        rows[(row["surface"], row["session"])] = row
    require(len(rows) == 13, "sealed screen activity30_m4 row coverage drift")
    return rows


def load_prior_deployment_records() -> dict[str, Any]:
    path = ROOT / PRIOR_DEPLOYMENT_RECEIPT_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["session_records"]
    require(len(records) == 13, "prior deployment receipt coverage drift")
    return records


def build_static_identity_inputs(
    dataset: Any, session: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return (selected, activity, side, evidence) for one session."""
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
    return ARTIFACT_ROOT / DEFAULT_PAYLOAD_NAME


def default_validation_output() -> Path:
    return ARTIFACT_ROOT / DEFAULT_VALIDATION_NAME


def export_payload(output: Path, validation_output: Path | None = None) -> dict[str, object]:
    require_output_absent(output)
    validation_output = Path(validation_output or default_validation_output())
    require_output_absent(validation_output)

    alpha_evidence = verify_v2_alpha_provenance(ROOT)

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
    student = model.student
    require(student is not None and student.decoder_mode == "coupled", "coupled student drift")
    require(bool(student._decoder_frozen), "frozen student decoder contract drift")
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

    import torch
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.adapter import (
        AnchoredPostFusionGate,
        exact_positive_zero,
        freeze_and_install,
    )

    # Capture the untouched native encoder, then install the APFG adapter on
    # the student exactly once, before any session loop.  The native identity
    # is computed through the captured encoder (the exact pre-install
    # ``compute_identity`` call); the ZERO/LEARNED identities route through
    # the adapter-bearing ``student.compute_identity``.
    native_encoder = student.id_encoder
    require(
        getattr(native_encoder, "variant", None) == "B3S",
        "exporter-route encoder is not the native B3S id encoder",
    )
    adapter = freeze_and_install(student)
    require(isinstance(student.id_encoder, AnchoredPostFusionGate), "APFG install drift")

    identities: dict[str, np.ndarray] = {}
    native_map: dict[str, np.ndarray] = {}
    zero_map: dict[str, np.ndarray] = {}
    records: dict[str, object] = {}
    max_direct = 0.0
    max_decoder = 0.0
    max_learned_native_abs = 0.0
    zero_equals_native_sessions = 0
    prior_identity_matches = 0
    surface_of_dataset = {"train": "within_post30", "heldout": "external_official_query"}

    for dataset_role, dataset in (
        ("train", data_module.train_dataset),
        ("heldout", data_module.val_heldout_dataset),
    ):
        require(dataset is not None, "official export requires both M2 session sets")
        for session in sorted(dataset.calib_trialized_neural_features):
            require(session in session_to_tag, f"{session}: missing official dataset tag")
            selected, activity, side, evidence = build_static_identity_inputs(dataset, session)
            sealed = sealed_rows[(surface_of_dataset[dataset_role], session)]
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
                identity_native = native_encoder.forward_batch(
                    support, side_features=side_tensor
                )
            native_np = np.ascontiguousarray(identity_native.squeeze(0).numpy(), dtype=np.float32)
            require(
                native_np.shape == (CHANNELS, WINDOW_SIZE) and np.isfinite(native_np).all(),
                f"{session}: native identity drift",
            )
            prior_identity_sha256 = prior_records[session]["identity_sha256"]
            prior_identity_matches += int(sha256_array(native_np) == prior_identity_sha256)

            # ZERO sentinel: exact +0.0 must be an operational no-op through
            # the adapter-bearing student.compute_identity route.
            with torch.no_grad():
                adapter.alpha.fill_(0.0)
            require(exact_positive_zero(adapter.alpha), f"{session}: zero alpha not IEEE +0.0")
            with torch.inference_mode():
                identity_zero = student.compute_identity(support, side_features=side_tensor)
            zero_np = np.ascontiguousarray(identity_zero.squeeze(0).numpy(), dtype=np.float32)
            zero_is_native = sha256_array(zero_np) == sha256_array(native_np)
            require(
                zero_is_native,
                f"{session}: APFG-ZERO identity is not the bitwise native anchor",
            )
            zero_equals_native_sessions += int(zero_is_native)

            with torch.no_grad():
                adapter.alpha.fill_(FROZEN_ALPHA)
            require(
                float(adapter.alpha.item()) == FROZEN_ALPHA,
                f"{session}: frozen alpha literal drift",
            )
            with torch.inference_mode():
                identity_learned = student.compute_identity(support, side_features=side_tensor)
                neural = torch.from_numpy(
                    np.asarray(dataset.neural_data[session][50:100], dtype=np.float32)
                ).unsqueeze(0)
                direct, _ = student(neural, calib_trials=support, side_features=side_tensor)
                cached, _ = student(neural, identity=identity_learned)
                decoder_only = manual_decode(student.decoder, neural, identity_learned)
            learned_np = np.ascontiguousarray(identity_learned.squeeze(0).numpy(), dtype=np.float32)
            require(
                learned_np.shape == (CHANNELS, WINDOW_SIZE) and np.isfinite(learned_np).all(),
                f"{session}: learned identity drift",
            )
            direct_error = float((direct - cached).abs().max().item())
            decoder_error = float((direct - decoder_only).abs().max().item())
            require(
                direct_error == 0.0 and decoder_error == 0.0,
                f"{session}: cached learned identity is not bit-exact",
            )
            max_direct = max(max_direct, direct_error)
            max_decoder = max(max_decoder, decoder_error)
            learned_native_abs = float(np.abs(learned_np - native_np).max())
            require(learned_native_abs > 0.0, f"{session}: learned gate had no effect")
            max_learned_native_abs = max(max_learned_native_abs, learned_native_abs)

            tag = session_to_tag[session]
            require(tag not in identities, "duplicate official dataset tag")
            identities[tag] = learned_np
            native_map[session] = native_np
            zero_map[session] = zero_np
            records[session] = {
                "dataset_tag": tag,
                "surface": surface_of_dataset[dataset_role],
                "selected_indices": [int(v) for v in selected],
                "label_budget": SUPPORT_BUDGET,
                "activity_budget": ACTIVITY_BUDGET,
                "activity_shape": list(activity.shape),
                "activity_sha256": evidence["activity_sha256"],
                "side_sha256": sha256_array(side),
                "native_identity_sha256": sha256_array(native_np),
                "zero_identity_sha256": sha256_array(zero_np),
                "learned_identity_sha256": sha256_array(learned_np),
                "prior_deployed_identity_sha256": prior_identity_sha256,
                "prior_deployed_identity_sha256_match": bool(
                    sha256_array(native_np) == prior_identity_sha256
                ),
                "side_evidence": evidence,
                "sealed_screen_activity30_m4_r2": float(sealed["r2"]),
                "direct_vs_cached_max_abs": direct_error,
                "direct_vs_decoder_max_abs": decoder_error,
                "learned_minus_native_identity_max_abs": learned_native_abs,
                "learned_minus_native_identity_l2": float(
                    np.sqrt(np.square(learned_np - native_np).sum())
                ),
            }
    require(
        len(identities) == 13 and set(records) == set(session_to_tag),
        "official session coverage drift",
    )

    decoder = student.decoder.cpu().eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "task": task_config.task,
        "decoder": decoder,
        "identity_by_dataset_tag": identities,
        "window_size": WINDOW_SIZE,
        "behavior_scaling_factor": 5.0,
        "smooth_observations": False,
        "metadata": {
            "arm": PAYLOAD_ARM,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
            "normalization_sha256": metadata["normalization_sha256"],
            "screen_id": "m2_spint_t4_mainline_fp32_v1",
            "sealed_screen_result": SEALED_SCREEN_RELATIVE,
            "sealed_screen_sha256": screen_sha,
            "apfg_adapter": "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py",
            "frozen_alpha": FROZEN_ALPHA,
            "alpha_provenance": {
                "v2_result_root": V2_RESULT_ROOT_RELATIVE,
                "v2_terminal_sha256": V2_TERMINAL_SHA256,
                "v2_score_sha256": V2_SCORE_SHA256,
                "terminal_sidecar_verified": True,
            },
            "seed": 42,
            "label_budget": SUPPORT_BUDGET,
            "activity_budget": ACTIVITY_BUDGET,
            "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
            "calibration_selection": "greedy_forward_d_optimal_first30_k4",
            "calibration_activity_pool": "first30_calibration_block_label_free",
            "calibration_uses_target_labels": True,
            "gate": "static_offline_anchored_postfusion_scalar_alpha",
            "online_state": "cached_E[N,50]",
            "online_backward_pass": False,
            "session_count": len(identities),
            "zero_equals_native_sessions": zero_equals_native_sessions,
            "max_direct_vs_cached_identity_abs": max_direct,
            "max_direct_vs_decoder_only_abs": max_decoder,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle_dump(payload, output)

    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "task": task_config.task,
        "window_size": WINDOW_SIZE,
        "behavior_scaling_factor": 5.0,
        "native_identity_by_session": native_map,
        "zero_identity_by_session": zero_map,
        "learned_identity_by_session": {
            session: identities[str(record["dataset_tag"])]
            for session, record in records.items()
        },
        "session_to_dataset_tag": {
            session: str(record["dataset_tag"]) for session, record in records.items()
        },
        "metadata": dict(payload["metadata"]),
    }
    _atomic_pickle_dump(validation, validation_output)

    receipt = {
        "schema_version": "m2_apfg_static_official_payload_receipt_v1",
        "status": "EXPORTED_NOT_SUBMITTED",
        "arm": PAYLOAD_ARM,
        "frozen_alpha": FROZEN_ALPHA,
        "label_budget": SUPPORT_BUDGET,
        "activity_budget": ACTIVITY_BUDGET,
        "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
        "payload_path": str(output),
        "payload_bytes": output.stat().st_size,
        "payload_sha256": sha256_file(output),
        "validation_path": str(validation_output),
        "validation_sha256": sha256_file(validation_output),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "alpha_provenance": {
            "v2_result_root": V2_RESULT_ROOT_RELATIVE,
            "v2_terminal_sha256": alpha_evidence["terminal_sha256"],
            "v2_score_sha256": alpha_evidence["score_sha256"],
            "learned_refit_alpha": alpha_evidence["learned_refit_alpha"],
        },
        "sealed_screen_result": SEALED_SCREEN_RELATIVE,
        "sealed_screen_sha256": screen_sha,
        "prior_deployment_receipt": PRIOR_DEPLOYMENT_RECEIPT_RELATIVE,
        "prior_deployed_identity_sha256_matches": prior_identity_matches,
        "sealed_external_activity30_m4_equal_session_mean": SEALED_EXTERNAL_ACTIVITY30_M4_MEAN,
        "sealed_within_activity30_m4_equal_session_mean": SEALED_WITHIN_ACTIVITY30_M4_MEAN,
        "session_count": len(identities),
        "zero_equals_native_sessions": zero_equals_native_sessions,
        "max_direct_vs_cached_identity_abs": max_direct,
        "max_direct_vs_decoder_only_abs": max_decoder,
        "max_learned_minus_native_identity_abs": max_learned_native_abs,
        "session_records": records,
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _atomic_pickle_dump(payload: dict[str, object], output: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or default_output()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_apfg_static_official_payload_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE_NO_SUBMISSION",
            "arm": PAYLOAD_ARM,
            "frozen_alpha": FROZEN_ALPHA,
            "label_budget": SUPPORT_BUDGET,
            "activity_budget": ACTIVITY_BUDGET,
            "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
            "output": str(output),
        }, sort_keys=True))
        return
    print(json.dumps(export_payload(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
