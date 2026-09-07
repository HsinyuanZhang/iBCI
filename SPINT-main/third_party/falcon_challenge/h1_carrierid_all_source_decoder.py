"""Isolated FALCON runtime for the H1 all-source CarrierID candidate.

The original :class:`SpintDecoder` remains untouched.  This subclass reuses
its payload loading, streaming buffers, filtering, and ``observe`` semantics,
then adds the per-dataset carrier that ``H1CarrierIdSpint`` requires.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from third_party.falcon_challenge.spint_decoder import CPU_Unpickler, SpintDecoder


H1_ALL_SOURCE_PAYLOAD_SCHEMA = "h1_carrierid_all_public_source_falcon_payload_v1"
H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5 = "h1_carrierid_all_public_source_falcon_payload_v5"


class H1CarrierIdPayloadError(ValueError):
    """The packaged per-dataset CarrierID runtime payload is malformed."""


def validate_carrier_payload(payload: dict, *, expected_task=None) -> dict[str, np.ndarray]:
    payload_schema = payload.get("carrier_payload_schema")
    if payload_schema not in (H1_ALL_SOURCE_PAYLOAD_SCHEMA, H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5):
        raise H1CarrierIdPayloadError("H1 all-source carrier payload schema mismatch")
    if expected_task is not None and payload.get("task") != expected_task:
        raise H1CarrierIdPayloadError("payload task does not match the requested FALCON task")
    features = payload.get("calib_trial_features")
    carriers = payload.get("calib_carriers")
    if not isinstance(features, dict) or not isinstance(carriers, dict) or set(features) != set(carriers):
        raise H1CarrierIdPayloadError("calibration feature/carrier dataset keys differ")
    if not carriers:
        raise H1CarrierIdPayloadError("payload contains no dataset carriers")
    receipts = payload.get("calibration_receipts")
    receipts_by_tag = {
        str(row.get("dataset_tag")): row
        for row in receipts
        if isinstance(row, dict) and row.get("dataset_tag") is not None
    } if isinstance(receipts, list) else {}
    if payload_schema == H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5 and set(receipts_by_tag) != set(features):
        raise H1CarrierIdPayloadError("v5 payload calibration receipt/dataset keys differ")
    normalized: dict[str, np.ndarray] = {}
    for dataset_tag, value in carriers.items():
        carrier = np.asarray(value, dtype=np.float32)
        identity = np.asarray(features[dataset_tag])
        if carrier.ndim != 2 or carrier.shape[1] != 4 or not np.isfinite(carrier).all():
            raise H1CarrierIdPayloadError(f"{dataset_tag}: carrier must be finite [N,4], got {carrier.shape}")
        if identity.ndim != 3 or identity.shape[0] not in (3, 4) or identity.shape[1] != 1024:
            raise H1CarrierIdPayloadError(
                f"{dataset_tag}: identity must be [M,1024,N] with M in {{3,4}}, got {identity.shape}"
            )
        if carrier.shape[0] != identity.shape[2]:
            raise H1CarrierIdPayloadError(f"{dataset_tag}: carrier/identity channel count mismatch")
        if payload_schema == H1_ALL_SOURCE_PAYLOAD_SCHEMA_V5:
            receipt = receipts_by_tag.get(str(dataset_tag))
            if not isinstance(receipt, dict) or receipt.get("support_m") != int(identity.shape[0]):
                raise H1CarrierIdPayloadError(f"{dataset_tag}: v5 receipt support_m/identity mismatch")
            support_values = receipt.get("support_trial_numbers")
            if not isinstance(support_values, list) or len(support_values) != int(identity.shape[0]):
                raise H1CarrierIdPayloadError(f"{dataset_tag}: v5 support trial list/identity mismatch")
            if len({float(value) for value in support_values}) != len(support_values):
                raise H1CarrierIdPayloadError(f"{dataset_tag}: v5 support trial list contains padding/duplicates")
        elif identity.shape[0] != 4:
            raise H1CarrierIdPayloadError(f"{dataset_tag}: legacy payload identity must retain M=4")
        normalized[str(dataset_tag)] = carrier
    metadata = payload.get("deployment_contract")
    if not isinstance(metadata, dict):
        raise H1CarrierIdPayloadError("payload lacks deployment contract")
    if metadata.get("target_optimizer_steps") != 0 or metadata.get("target_backward_steps") != 0:
        raise H1CarrierIdPayloadError("payload permits target-session optimization/backpropagation")
    if metadata.get("formal_test_labels_packaged") != 0 or metadata.get("query_labels_read") != 0:
        raise H1CarrierIdPayloadError("payload records forbidden query/formal labels")
    return normalized


class H1CarrierIdAllSourceDecoder(SpintDecoder):
    """SPINT streaming runtime with one frozen normalized carrier per dataset."""

    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        super().__init__(task_config=task_config, model_path=model_path, batch_size=batch_size)
        with open(model_path, "rb") as handle:
            payload = CPU_Unpickler(handle).load()
        self.calib_carriers = validate_carrier_payload(payload, expected_task=task_config.task)
        self.carrier_asset_manifest_sha256 = str(payload.get("carrier_asset_manifest_sha256", ""))
        self.carrier_transform_sha256 = str(payload.get("carrier_transform_sha256", ""))
        self.carrier_normalizer_sha256 = str(payload.get("carrier_normalizer_sha256", ""))
        if not all(len(value) == 64 for value in (
            self.carrier_asset_manifest_sha256,
            self.carrier_transform_sha256,
            self.carrier_normalizer_sha256,
        )):
            raise H1CarrierIdPayloadError("payload CarrierID provenance hashes are malformed")

    def reset(self, dataset_tags: List[Path] = [""]):
        super().reset(dataset_tags)
        hashed = [self._task_config.hash_dataset(dataset.stem) for dataset in dataset_tags]
        missing = [tag for tag in hashed if tag not in self.calib_carriers]
        if missing:
            raise H1CarrierIdPayloadError(f"dataset carriers missing for {missing}")
        self.local_carriers = [torch.tensor(self.calib_carriers[tag], dtype=torch.float32) for tag in hashed]

    def predict(self, neural_observations: np.ndarray):
        """Forward-only streaming prediction; no optimizer or backward object exists."""

        self.local_clf = self.local_clf.to(self.device)
        self.observe(neural_observations)
        decoder_in = torch.tensor(
            self.observation_buffer.copy().transpose(1, 0, 2), dtype=torch.float32, device=self.device
        )
        if len(self.local_carriers) != len(self.local_calib_trial_features):
            raise H1CarrierIdPayloadError("runtime carrier/identity batch cardinality mismatch")
        outputs = []
        with torch.inference_mode():
            for index, (identity, carrier) in enumerate(zip(self.local_calib_trial_features, self.local_carriers)):
                output = self.local_clf(
                    decoder_in[index : index + 1],
                    calib_trialized_neural_features=identity.unsqueeze(0).to(decoder_in),
                    carrier=carrier.unsqueeze(0).to(decoder_in),
                )
                outputs.append(output)
        prediction = torch.cat(outputs, dim=0)
        return prediction[:, -1, :].cpu().numpy() / self.behavior_scaling_factor
