#!/usr/bin/env python3
"""Offline, local-only construction of the frozen AOF-S payload.

This command is intentionally not a submission mechanism.  It writes only a
previously absent payload under this package's ``artifacts`` directory after
validating AOF-M's immutable source OOF authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
ARTIFACTS = HERE / "artifacts" / "local_build_v1"
for item in (REPO_ROOT, HERE):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
_SPINT_MAIN = REPO_ROOT / "SPINT-main"
if _SPINT_MAIN.is_dir() and str(_SPINT_MAIN) not in sys.path:
    sys.path.append(str(_SPINT_MAIN))

try:  # package execution through the route-owned lifecycle
    from .laws import (  # noqa: E402
        AOFM_CLOSURE_SHA256,
        BEHAVIOR_SCALING_FACTOR,
        FROZEN_BETA,
        sha256_array,
        validate_aofm_authority,
        validate_identity_map,
    )
except ImportError:  # copied flat into the local Docker context
    from laws import (  # type: ignore[no-redef]  # noqa: E402
        AOFM_CLOSURE_SHA256,
        BEHAVIOR_SCALING_FACTOR,
        FROZEN_BETA,
        sha256_array,
        validate_aofm_authority,
        validate_identity_map,
    )

PAYLOAD_NAME = "t4_m2_seed42_dopt4_act30_aofs_identity.pkl"
RECEIPT_NAME = "t4_m2_seed42_dopt4_act30_aofs_identity.receipt.json"
GOVERNING_PAYLOAD_RELATIVE = "sua_exploration/evalai_t4_m2_activity_budget/artifacts/t4_m2_seed42_ridge_m4_activity30_identity.pkl"
GOVERNING_PAYLOAD_SHA256 = "c51b71167d81490927fee8a552785ee27ddff7e90085ed3ff4b18b857afbac40"
GOVERNING_RECEIPT_RELATIVE = "sua_exploration/evalai_t4_m2_activity_budget/artifacts/t4_m2_seed42_ridge_m4_activity30_identity.receipt.json"
GOVERNING_RECEIPT_SHA256 = "a3c304106ce56105c3bde59b3237604e95388608d740800da4ff2da0fa7186db"
BRIDGE_E4_PAYLOAD_SHA256 = "e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261"
BRIDGE_E4_PAYLOAD_RELATIVE = "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/t4_m2_seed42_dopt4_act30_identity.pkl"
BRIDGE_E4_RECEIPT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/t4_m2_seed42_dopt4_act30_identity.receipt.json"
BRIDGE_E4_RECEIPT_SHA256 = "6f90230f9f8f330edeec970ea0108defde24cd43ca3195fea264083cac6fa583"
CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"


class ExportError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


@dataclass(frozen=True)
class ReceiptCodec:
    """Immutable selection-index placement contract for a build profile."""

    name: str
    selected_path: tuple[str, ...]
    selected_sha_path: tuple[str, ...]

    def selected(self, record: dict) -> list[int]:
        value: object = record
        for key in self.selected_path:
            _need(isinstance(value, dict) and key in value, f"{self.name}: selected-index path missing: {key}")
            value = value[key]
        _need(isinstance(value, list) and all(isinstance(index, int) for index in value),
              f"{self.name}: selected-index encoding drift")
        return value

    def selected_sha256(self, record: dict) -> str:
        value: object = record
        for key in self.selected_sha_path:
            _need(isinstance(value, dict) and key in value, f"{self.name}: selected-index SHA path missing: {key}")
            value = value[key]
        _need(isinstance(value, str) and len(value) == 64, f"{self.name}: selected-index SHA encoding drift")
        return value


# Kept as the literal historical V1 expectation so its immutable failure
# remains reproducible.  The additive V2 profile opts into the actual c51
# receipt layout below; V1 never silently changes interpretation.
V1_TOP_LEVEL_RECEIPT_CODEC = ReceiptCodec(
    name="v1_top_level_selected_indices",
    selected_path=("selected_indices",),
    selected_sha_path=("selected_indices_sha256",),
)
V2_SIDE_EVIDENCE_RECEIPT_CODEC = ReceiptCodec(
    name="v2_side_evidence_selected_indices",
    selected_path=("side_evidence", "selected_indices"),
    selected_sha_path=("side_evidence", "selected_indices_sha256"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _governing_payload() -> tuple[dict, dict]:
    """Read exact 581361 c51 decoder/native bytes; e4 is bridge-only."""
    # The checkpoint serializes streaming's historical top-level ``src``.
    # Pin that namespace before unpickling; qualified TFPD imports must never
    # occupy it first.
    from tfpd_exploration.src.pit_m2_v1.trainer import ensure_streaming_paths
    ensure_streaming_paths(REPO_ROOT)
    from sua_exploration.evalai_t4_m2.t4_spint_decoder import CPUUnpickler

    receipt_path = REPO_ROOT / GOVERNING_RECEIPT_RELATIVE
    payload_path = REPO_ROOT / GOVERNING_PAYLOAD_RELATIVE
    bridge_receipt_path = REPO_ROOT / BRIDGE_E4_RECEIPT_RELATIVE
    bridge_payload_path = REPO_ROOT / BRIDGE_E4_PAYLOAD_RELATIVE
    _need(_sha256_file(receipt_path) == GOVERNING_RECEIPT_SHA256, "581361 c51 receipt drift")
    _need(_sha256_file(payload_path) == GOVERNING_PAYLOAD_SHA256, "581361 c51 payload drift")
    _need(_sha256_file(bridge_receipt_path) == BRIDGE_E4_RECEIPT_SHA256, "e4 bridge receipt drift")
    _need(_sha256_file(bridge_payload_path) == BRIDGE_E4_PAYLOAD_SHA256, "e4 bridge payload body drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    bridge_receipt = json.loads(bridge_receipt_path.read_text(encoding="utf-8"))
    _need(receipt.get("payload_sha256") == GOVERNING_PAYLOAD_SHA256 and receipt.get("session_count") == 13,
          "581361 c51 receipt link drift")
    _need(bridge_receipt.get("payload_sha256") == BRIDGE_E4_PAYLOAD_SHA256 and bridge_receipt.get("session_count") == 13,
          "e4 bridge receipt link drift")
    with payload_path.open("rb") as handle:
        payload = CPUUnpickler(handle).load()
    _need(payload.get("schema_version") == "e8_t4_m2_cached_identity_v1"
          and payload.get("behavior_scaling_factor") == BEHAVIOR_SCALING_FACTOR,
          "581361 c51 payload schema/scale drift")
    validate_identity_map(payload.get("identity_by_dataset_tag"), expected_tags=13)
    return payload, receipt


def _bridge_e4_payload() -> dict:
    """Read the later local reconstruction only as numerical bridge evidence."""
    from tfpd_exploration.submissions.evalai_m2_act30_dopt4_v1.act30_dopt4_decoder import CPUUnpickler

    path = REPO_ROOT / BRIDGE_E4_PAYLOAD_RELATIVE
    _need(_sha256_file(path) == BRIDGE_E4_PAYLOAD_SHA256, "e4 bridge payload body drift")
    with path.open("rb") as handle:
        payload = CPUUnpickler(handle).load()
    _need(payload.get("schema_version") == "e8_t4_m2_cached_identity_v1"
          and payload.get("behavior_scaling_factor") == BEHAVIOR_SCALING_FACTOR,
          "e4 bridge payload schema/scale drift")
    validate_identity_map(payload.get("identity_by_dataset_tag"), expected_tags=13)
    return payload


def _post_identity(encoder, activity, side):
    """Exact ordered sequential post-pool mean used by the frozen AOF seam."""
    import torch

    trials = torch.as_tensor(np.asarray(activity), dtype=torch.float32).unsqueeze(0)
    side_tensor = torch.as_tensor(np.asarray(side), dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        values = encoder.pre_pool(trials.permute(0, 1, 3, 2))
        expanded = side_tensor.unsqueeze(1).expand(-1, int(values.shape[1]), -1, -1)
        values = encoder.post_pool(torch.cat((values, expanded), dim=-1))
        total = values[:, 0]
        for index in range(1, int(values.shape[1])):
            total = total + values[:, index]
        return total.div(int(values.shape[1])).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _module_state_sha256(module) -> str:
    """Strict CPU state proof before using a loader only as a post-identity encoder."""
    import torch

    digest = hashlib.sha256()
    with torch.inference_mode():
        for name, value in sorted(module.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_payload(output: Path, *, receipt_codec: ReceiptCodec = V1_TOP_LEVEL_RECEIPT_CODEC) -> dict:
    """Build once on CPU.  This is local package construction, never submission."""
    _need(not output.exists(), f"refusing to overwrite payload: {output}")
    authority = validate_aofm_authority(REPO_ROOT)
    governing_payload, governing_receipt = _governing_payload()
    bridge_e4_payload = _bridge_e4_payload()
    from sua_exploration.evalai_t4_m2.export_t4_payload import calibration_file_map, load_frozen_model_and_data
    from tfpd_exploration.submissions.evalai_m2_act30_dopt4_v1.export_act30_dopt4_payload import build_static_identity_inputs

    model, data_module, task_config, metadata = load_frozen_model_and_data()
    student = model.student.cpu().eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    session_to_tag = calibration_file_map(REPO_ROOT / "SPINT-main/data/000953", task_config)
    native = governing_payload["identity_by_dataset_tag"]
    governing_decoder = governing_payload["decoder"].cpu().eval()
    for parameter in governing_decoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    _need(
        metadata.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and governing_payload.get("metadata", {}).get("checkpoint_sha256") == metadata.get("checkpoint_sha256")
        and _module_state_sha256(student.decoder) == _module_state_sha256(governing_decoder),
        "fresh post-identity loader decoder is not the exact c51 governing decoder state",
    )
    post: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    for dataset in (data_module.train_dataset, data_module.val_heldout_dataset):
        _need(dataset is not None, "AOF-S requires exactly the official M2 13-session source stack")
        for session in sorted(dataset.calib_trialized_neural_features):
            record = governing_receipt["session_records"].get(session)
            _need(isinstance(record, dict), f"581361 c51 receipt lacks {session}")
            tag = session_to_tag.get(session)
            _need(tag == record.get("dataset_tag"), f"581361 dataset-tag drift: {session}")
            selected, activity, side, evidence = build_static_identity_inputs(dataset, session)
            official_native = np.ascontiguousarray(native[tag], dtype=np.float32)
            bridge_native = np.ascontiguousarray(bridge_e4_payload["identity_by_dataset_tag"][tag], dtype=np.float32)
            side_evidence = record.get("side_evidence")
            _need(isinstance(side_evidence, dict), f"official side evidence absent: {session}")
            official_selected = receipt_codec.selected(record)
            official_selected_sha256 = receipt_codec.selected_sha256(record)
            _need(
                sha256_array(official_native) == record.get("identity_sha256")
                and evidence["activity_sha256"] == record.get("activity_sha256")
                and sha256_array(np.asarray(side, dtype=np.float32)) == record.get("side_sha256")
                and evidence["selected_indices"] == official_selected
                and evidence["selected_indices"] == side_evidence.get("selected_indices")
                and evidence["raw_t4_sha256"] == side_evidence.get("raw_t4_sha256")
                and evidence["normalized_t4_sha256"] == side_evidence.get("normalized_t4_sha256")
                and evidence["selected_indices_sha256"] == official_selected_sha256
                and evidence["selected_indices_sha256"] == side_evidence.get("selected_indices_sha256"),
                f"581361 activity/support/raw-T4/side/native identity drift: {session}",
            )
            bridge_maxabs = float(np.max(np.abs(official_native - bridge_native)))
            _need(np.isfinite(bridge_maxabs) and bridge_maxabs <= 2.0e-6,
                  f"c51-to-e4 native identity bridge exceeds declared limit: {session}")
            post_identity = np.ascontiguousarray(_post_identity(student.id_encoder, activity, side), dtype=np.float32)
            _need(post_identity.shape == (96, 50) and np.isfinite(post_identity).all(), f"post identity drift: {session}")
            post[tag] = post_identity
            records[session] = {
                "dataset_tag": tag,
                "native_identity_sha256": sha256_array(official_native),
                "bridge_e4_identity_sha256": sha256_array(bridge_native),
                "c51_to_e4_identity_maxabs": bridge_maxabs,
                "c51_to_e4_identity_maxabs_limit": 2.0e-6,
                "post_identity_sha256": sha256_array(post_identity),
                "activity_sha256": evidence["activity_sha256"],
                "side_sha256": sha256_array(np.asarray(side, dtype=np.float32)),
                "raw_t4_sha256": evidence["raw_t4_sha256"],
                "normalized_t4_sha256": evidence["normalized_t4_sha256"],
                "selected_indices": evidence["selected_indices"],
                "selected_indices_sha256": evidence["selected_indices_sha256"],
                "side_evidence": side_evidence,
            }
    _need(len(records) == 13 and set(native) == set(post), "AOF-S 13-tag payload coverage drift")
    payload = {
        "schema_version": "e8_t4_m2_aofs_static_decoded_output_v1",
        "task": task_config.task,
        # Governing deployment decoder is the literal object from c51/581361.
        # ``student`` above is used only for the proven-identical post-pool
        # encoder and never serialized into this payload.
        "decoder": governing_decoder,
        "native_identity_by_dataset_tag": native,
        "post_identity_by_dataset_tag": post,
        "window_size": 50,
        "behavior_scaling_factor": BEHAVIOR_SCALING_FACTOR,
        "smooth_observations": False,
        "metadata": {
            "arm": "aofs_static_act30_dopt4",
            "frozen_beta": FROZEN_BETA,
            "label_budget": 4,
            "activity_budget": 30,
            "online_backward_pass": False,
            "online_state": "two_cached_E[N,50]_maps",
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "aofm_terminal_sha256": authority["terminal_sha256"],
            "aofm_closure_sha256": AOFM_CLOSURE_SHA256,
            "governing_581361_payload_sha256": GOVERNING_PAYLOAD_SHA256,
            "bridge_only_e4_payload_sha256": BRIDGE_E4_PAYLOAD_SHA256,
            "governing_decoder_state_sha256": _module_state_sha256(governing_decoder),
            "post_encoder_decoder_state_sha256": _module_state_sha256(student.decoder),
            "session_records": records,
        },
    }
    _need(not output.parent.exists(), f"refusing to reuse package artifact root: {output.parent}")
    output.parent.mkdir(parents=True, exist_ok=False)
    with output.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    output.chmod(0o444)
    digest = _sha256_file(output)
    receipt = {
        "schema_version": "m2_aofs_static_payload_receipt_v1",
        "status": "LOCAL_PAYLOAD_BUILT_NOT_SUBMITTED",
        "payload_path": str(output),
        "payload_sha256": digest,
        "frozen_beta": FROZEN_BETA,
        "aofm_authority": authority,
        "governing_581361_native_payload_sha256": GOVERNING_PAYLOAD_SHA256,
        "bridge_only_e4_payload_sha256": BRIDGE_E4_PAYLOAD_SHA256,
        "session_records": records,
        "network_submission": False,
        "hidden_evalai_target_opened": False,
        "optimizer_constructed": False,
        "parameter_updates": 0,
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only AOF-S payload builder; no network submission")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / PAYLOAD_NAME)
    parser.add_argument("--execute", action="store_true", help="deprecated: production build is driver-only")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "INERT", "action": "requires --execute for local payload build", "network": False}, sort_keys=True))
        return
    raise SystemExit(
        "standalone --execute is intentionally disabled; use the route-owned "
        "execute_production_local lifecycle after independent closure review"
    )


if __name__ == "__main__":
    main()
