"""Deny-by-default verification of the sealed A2 W/Z4 and W/T4 anchors.

This module treats the historical immutable A2 receipts and checkpoint bytes as
the authority.  It deliberately distinguishes that historical binding from the
additive A1 implementation: A1 files are not part of the sealed A2 tree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    A1ArtifactError,
    canonical_json_sha256,
    load_verified_immutable_json,
    require,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
A2_RESULTS = SUA / "results/a2_matched_subject_shift_v2"

A2_TERMINAL_SHA256 = "5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc"
A2_PREFLIGHT_SHA256 = "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd"
A2_CONTRACT_SHA256 = "f8d2f1f9e2420423584f18e667df27bc209c5ae5d1730232d703886244ea5cb2"
A2_CONFIG_SHA256 = "68aa9b599b5d30c690af8639ece6f3d2e7d63e47c5ef3f7fedcd853cae6ed73c"
A2_IMPLEMENTATION_SHA256 = "7e46115fc3bafa1010d454c80215d6cf99987973464b0720997366d92b2887b3"
TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
T4_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"

TEACHER = SUA / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
MANIFEST = SUA / "configs/subc_co_27_6_strict_train_val_manifest.json"
DEV_SESSIONS = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)
FORMAL_SESSIONS = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)

WITHIN_RECEIPT_SHA256: Mapping[str, Mapping[int, str]] = {
    "source_z4": {
        42: "0e10261a59d3c111b4bf4766c5511d62735fe73910b744a0e60eafce71cc83d5",
        43: "05fa7b154abeba402caa7e48546692a6ad95c6094686874ce47365218393c383",
        44: "d3d59f5a5f9f91b80dcd98bd3916e95f2d885c350cd9f89b482f0529e070af0d",
    },
    "source_t4": {
        42: "588c4123b895878e03b6b4a18d829a8fc9bcc9f1771fc943d9cc06e72c106955",
        43: "0ad9117aeec4c7771ff93f68d727150cc3cfaebb800cd2c7654a2f941badc227",
        44: "9df1022e23dff0868b33c40e795c203e99b0c51629a354b1a4ae56c639ce307c",
    },
}
EPOCH_BUNDLE_SHA256: Mapping[str, Mapping[int, str]] = {
    "source_z4": {
        42: "c1d5482640eea00b649810b0107527cdcfb2c1d0b5c0cbbb886887a21eb27a45",
        43: "dfe39ea79db1c4a856e4932d18ddfd26542688f67950fc1ec53ee1cf399e6085",
        44: "25ffc59a551f83d5b7c146a9a3f8df98488550beeda45780e70e563ec3229c01",
    },
    "source_t4": {
        42: "16daf346541c7594ec9bb097ddae369899dc59f16bfe4f5bfd2e993565409260",
        43: "dbcc34ada3aa616a933d55840f7f5c31687f8fe02c993dae798fd6a3f0711e65",
        44: "200c39adac1e7d3a37865b833ce15c582b6ec9bb50ee8d9a2d11e9191562bab2",
    },
}


def _resolved(path_value: object) -> Path:
    require(isinstance(path_value, str), "bound path must be a string")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SUA / path
    return path.resolve()


def _require_exact(container: Mapping[str, Any], key: str, expected: object, label: str) -> None:
    require(type(container.get(key)) is type(expected), f"{label}.{key} type drift")
    require(container.get(key) == expected, f"{label}.{key} drift")


def _verify_checkpoint_payload(path: Path, *, arm: str) -> dict[str, object]:
    # Local import keeps --skip-checkpoint-bytes metadata-only tests Torch-free.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, Mapping), f"checkpoint payload malformed: {path}")
    hyper = payload.get("hyper_parameters")
    state = payload.get("state_dict")
    optimizers = payload.get("optimizer_states")
    require(isinstance(hyper, Mapping), f"checkpoint hyperparameters missing: {path}")
    require(isinstance(state, Mapping), f"checkpoint state_dict missing: {path}")
    require(isinstance(optimizers, list) and len(optimizers) == 1, f"optimizer state drift: {path}")
    expected_hyper = {
        "variant": "B3S",
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "loss_mode": "task_only",
        "identity_mode": "calibrated",
        "side_dim": 4,
        "decoder_mode": "coupled",
    }
    for key, expected in expected_hyper.items():
        _require_exact(hyper, key, expected, str(path))
    decoder_keys = [key for key in state if str(key).startswith("student.decoder.")]
    encoder_keys = [key for key in state if str(key).startswith("student.id_encoder.")]
    require(len(decoder_keys) == 31, f"expected 31 student decoder tensors: {path}")
    require(len(encoder_keys) == 8, f"expected 8 student identity tensors: {path}")
    optimizer = optimizers[0]
    require(isinstance(optimizer, Mapping), f"optimizer payload malformed: {path}")
    groups = optimizer.get("param_groups")
    require(isinstance(groups, list) and len(groups) >= 1, f"optimizer groups missing: {path}")
    ids: list[int] = []
    for group in groups:
        require(isinstance(group, Mapping) and isinstance(group.get("params"), list), f"optimizer group malformed: {path}")
        ids.extend(int(value) for value in group["params"])
    require(len(ids) == 39 and len(set(ids)) == 39, f"optimizer must cover exactly 31 decoder + 8 identity tensors: {path}")
    return {
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "decoder_tensor_count": 31,
        "identity_encoder_tensor_count": 8,
        "optimizer_unique_parameter_count": 39,
        "source_arm": arm,
    }


def _verify_one_arm(arm: str, seed: int, *, verify_checkpoint_bytes: bool) -> dict[str, object]:
    short = arm.removeprefix("source_")
    receipt_path = A2_RESULTS / f"within_subject_{arm}_s{seed}.json"
    receipt, receipt_sha = load_verified_immutable_json(
        receipt_path,
        label=f"sealed A2 {arm} s{seed} within receipt",
        expected_sha256=WITHIN_RECEIPT_SHA256[arm][seed],
    )
    expected_fields = {
        "schema_version": 3,
        "screen_id": "a2_matched_subject_shift_v2",
        "source_arm": arm,
        "arm": short,
        "seed": seed,
        "domain": "within_subject",
        "variant": "B3S",
        "official_preflight_sha256": A2_PREFLIGHT_SHA256,
        "implementation_bindings_sha256": A2_IMPLEMENTATION_SHA256,
        "contract_sha256": A2_CONTRACT_SHA256,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
        "backward_gradients": False,
        "decoder_weight_updates": False,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
        "target_domain_normalizer_refit_performed": False,
    }
    for key, expected in expected_fields.items():
        _require_exact(receipt, key, expected, f"{arm}/s{seed}")
    require(tuple(receipt.get("domain_sessions") or ()) == DEV_SESSIONS, f"{arm}/s{seed} development roster drift")
    per_session = receipt.get("per_session_mean_r2")
    require(isinstance(per_session, Mapping) and tuple(per_session) == DEV_SESSIONS, f"{arm}/s{seed} score roster/order drift")
    require(all(isinstance(per_session[s], (int, float)) for s in DEV_SESSIONS), f"{arm}/s{seed} score malformed")

    protocol = receipt.get("protocol")
    require(isinstance(protocol, Mapping), f"{arm}/s{seed} protocol missing")
    expected_protocol = {
        "activity_calibration_n": 30,
        "pool_size": 30,
        "evaluation_start_trial_index": 30,
        "selection_mode": "first",
        "total_epochs": 12,
        "epoch_window": list(range(5, 13)),
        "loss_mode": "task_only",
        "identity_mode": "calibrated",
        "signal_view": "sua",
        "epoch_score_rule": "unweighted mean session R2 over exactly source epochs 5..12",
    }
    require(dict(protocol) == expected_protocol, f"{arm}/s{seed} exact M30/epoch protocol drift")
    query = receipt.get("query_policy")
    require(isinstance(query, Mapping), f"{arm}/s{seed} query policy missing")
    for key, expected in {
        "activity_calibration_trials": 30,
        "side_feature_label_pool_trials": 30,
        "evaluation_start_trial_index": 30,
        "selection_mode": "first",
        "backward_gradients": False,
        "decoder_weight_updates": False,
        "target_velocity_labels_used_for_weight_updates": False,
        "target_session_carrier_fit_performed": True,
        "target_direction_labels_used_for_carrier": True,
    }.items():
        _require_exact(query, key, expected, f"{arm}/s{seed}.query")
    normalizer = receipt.get("normalizer_authority")
    require(isinstance(normalizer, Mapping), f"{arm}/s{seed} normalizer authority missing")
    for key in ("side_normalizer_value_sha256", "source_training_side_normalizer_value_sha256"):
        _require_exact(normalizer, key, T4_NORMALIZER_SHA256, f"{arm}/s{seed}.normalizer")
    _require_exact(normalizer, "target_domain_normalizer_refit_performed", False, f"{arm}/s{seed}.normalizer")
    _require_exact(normalizer, "target_domain_normalizer_refit_forbidden", True, f"{arm}/s{seed}.normalizer")

    metadata_path = _resolved(receipt.get("source_run_metadata_path"))
    metadata_sha = receipt.get("source_run_metadata_sha256")
    require(isinstance(metadata_sha, str) and sha256_file(metadata_path) == metadata_sha, f"{arm}/s{seed} source metadata SHA drift")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("status") == "completed", f"{arm}/s{seed} source run incomplete")
    training = metadata.get("training") or {}
    for key, expected in {
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "loss_mode": "task_only",
        "calibration_n_trials": 30,
        "calibration_selection": "chronological_first_n",
        "max_epochs": 12,
        "no_early_stopping": True,
    }.items():
        _require_exact(training, key, expected, f"{arm}/s{seed}.metadata.training")
    require(tuple((metadata.get("session_splits") or {}).get("val") or ()) == DEV_SESSIONS, f"{arm}/s{seed} metadata val roster drift")
    require(tuple((metadata.get("session_splits") or {}).get("test") or ()) == FORMAL_SESSIONS, f"{arm}/s{seed} formal exclusion roster drift")
    require(metadata.get("held_out_test_evaluated") is False, f"{arm}/s{seed} formal test was evaluated")
    require(_resolved(metadata.get("teacher_checkpoint")) == TEACHER.resolve(), f"{arm}/s{seed} teacher path drift")
    require(_resolved(metadata.get("train_val_manifest")) == MANIFEST.resolve(), f"{arm}/s{seed} manifest path drift")
    require(metadata.get("teacher_sha256") == TEACHER_SHA256, f"{arm}/s{seed} teacher SHA drift")
    require(metadata.get("train_val_manifest_sha256") == MANIFEST_SHA256, f"{arm}/s{seed} manifest SHA drift")

    bundle = receipt.get("source_checkpoint_sha256_bundle")
    require(isinstance(bundle, Mapping) and tuple(sorted(int(k) for k in bundle)) == tuple(range(5, 13)), f"{arm}/s{seed} epoch bundle roster drift")
    require(receipt.get("source_checkpoint_sha256_bundle_sha256") == EPOCH_BUNDLE_SHA256[arm][seed], f"{arm}/s{seed} bundle SHA drift")
    require(canonical_json_sha256(bundle) == EPOCH_BUNDLE_SHA256[arm][seed], f"{arm}/s{seed} bundle canonical digest drift")
    checkpoint_audits: dict[str, object] = {}
    run_dir = metadata_path.parent
    for epoch in range(5, 13):
        checkpoint = run_dir / "epoch_ckpts" / f"epoch_{epoch - 1:03d}.ckpt"
        expected_hash = bundle[str(epoch)]
        require(checkpoint.is_file() and sha256_file(checkpoint) == expected_hash, f"{arm}/s{seed} epoch {epoch} checkpoint SHA drift")
        if verify_checkpoint_bytes:
            checkpoint_audits[str(epoch)] = _verify_checkpoint_payload(checkpoint, arm=arm)

    return {
        "logical_cell": "W/Z4" if arm == "source_z4" else "W/T4",
        "source_arm": arm,
        "seed": seed,
        "within_receipt_path": str(receipt_path.resolve()),
        "within_receipt_sha256": receipt_sha,
        "source_run_metadata_path": str(metadata_path),
        "source_run_metadata_sha256": metadata_sha,
        "source_checkpoint_sha256_bundle": dict(bundle),
        "source_checkpoint_sha256_bundle_sha256": EPOCH_BUNDLE_SHA256[arm][seed],
        "per_session_mean_r2": {session: float(per_session[session]) for session in DEV_SESSIONS},
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "optimizer_coverage": {
            "decoder_tensor_count": 31,
            "identity_encoder_tensor_count": 8,
            "total_unique_parameter_count": 39,
            "verified_every_epoch_5_through_12": verify_checkpoint_bytes,
        },
        "checkpoint_payload_audits": checkpoint_audits,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
    }


def verify_sealed_a2_reuse(
    *, seed: int = 42, verify_checkpoint_bytes: bool = True
) -> dict[str, object]:
    """Verify all historical bytes needed for one A1 seed's W-cell reuse."""
    require(seed in (42, 43, 44), "A1 reuse seed must be 42, 43, or 44")
    terminal, terminal_sha = load_verified_immutable_json(
        A2_RESULTS / "terminal_aggregate.json",
        label="sealed A2 terminal aggregate",
        expected_sha256=A2_TERMINAL_SHA256,
    )
    preflight, preflight_sha = load_verified_immutable_json(
        A2_RESULTS / "official_cpu_preflight.json",
        label="sealed A2 official preflight",
        expected_sha256=A2_PREFLIGHT_SHA256,
    )
    for payload, label in ((terminal, "terminal"), (preflight, "preflight")):
        _require_exact(payload, "screen_id", "a2_matched_subject_shift_v2", label)
        _require_exact(payload, "contract_sha256", A2_CONTRACT_SHA256, label)
        _require_exact(payload, "implementation_bindings_sha256", A2_IMPLEMENTATION_SHA256, label)
        _require_exact(payload, "formal_subc_test_nwb_opened", False, label)
    _require_exact(preflight, "config_sha256", A2_CONFIG_SHA256, "preflight")
    _require_exact(preflight, "teacher_sha256", TEACHER_SHA256, "preflight")
    _require_exact(preflight, "manifest_sha256", MANIFEST_SHA256, "preflight")
    _require_exact(preflight, "gpu_used", False, "preflight")
    _require_exact(preflight, "training_started", False, "preflight")
    require(sha256_file(TEACHER) == TEACHER_SHA256, "teacher checkpoint byte drift")
    require(sha256_file(MANIFEST) == MANIFEST_SHA256, "strict manifest byte drift")

    arms = {
        arm: _verify_one_arm(arm, seed, verify_checkpoint_bytes=verify_checkpoint_bytes)
        for arm in ("source_z4", "source_t4")
    }
    z4, t4 = arms["source_z4"], arms["source_t4"]
    require(z4["per_session_mean_r2"].keys() == t4["per_session_mean_r2"].keys(), "W score roster mismatch")
    return {
        "schema_version": 2,
        "status": "SEALED_A2_W_REUSE_VERIFIED",
        "seed": seed,
        "terminal_aggregate_path": str((A2_RESULTS / "terminal_aggregate.json").resolve()),
        "terminal_aggregate_sha256": terminal_sha,
        "official_preflight_path": str((A2_RESULTS / "official_cpu_preflight.json").resolve()),
        "official_preflight_sha256": preflight_sha,
        "a2_contract_sha256": A2_CONTRACT_SHA256,
        "a2_config_sha256": A2_CONFIG_SHA256,
        "a2_implementation_bindings_sha256": A2_IMPLEMENTATION_SHA256,
        "teacher_path": str(TEACHER.resolve()),
        "teacher_sha256": TEACHER_SHA256,
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": MANIFEST_SHA256,
        "t4_normalizer_sha256": T4_NORMALIZER_SHA256,
        "development_sessions": list(DEV_SESSIONS),
        "sealed_formal_sessions_names_only": list(FORMAL_SESSIONS),
        "sealed_w_cells": arms,
        "h_z4_structural_alias": {
            "logical_cell": "H/Z4",
            "structural_alias_of": "W/Z4",
            "w_z4_within_receipt_sha256": z4["within_receipt_sha256"],
            "separate_training_run": False,
            "separate_scoring_run": False,
            "carrier_port": "exact_zero",
        },
        "target_backward_gradients": False,
        "target_weight_updates": False,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
    }


__all__ = [
    "A1ArtifactError",
    "DEV_SESSIONS",
    "FORMAL_SESSIONS",
    "verify_sealed_a2_reuse",
]
