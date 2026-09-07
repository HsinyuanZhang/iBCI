#!/usr/bin/env python3
"""Terminal held-in evaluator for the label-free H-U arm.

The H-U adapter is intentionally independent of the historical behaviour-
supervised H-C carrier reconstruction.  This evaluator binds the immutable
fold-0 target *lineage* (NWB files, support segmentation, fifth-trial boundary,
strict query windows and sample counts), then builds the H-U carrier only from
support neural counts and the source-only H-U normalizer.

Do not run until a source checkpoint exists.  Formal/organizer test data are
out of scope; this script accepts only the two public development fold-0
recordings under ``data/000954/sub-HumanPitt-held-in-calib``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
from lightning import Trainer
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset

from src.data.h1_carrierid_hu import (
    H1CarrierIdHuDataModule,
    HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
)
from src.data.h1_carrierid_hu_features import hu_from_record
from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_TARGET,
    WINDOW,
    H1PilotRecord,
    PilotDataError,
    _window_manifest_hash,
    carrier_sha256,
    interpolate_identity,
    load_target_records,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    canonical_sha256,
    sha256_file,
)
from scripts.h1_carrierid_evaluate import _evaluate


RAW = ROOT.parent / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB = ROOT.parent / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
PREREG = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_PREREGISTRATION_v2.json"
ADDENDUM_V3 = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_EXECUTION_ADDENDUM_v3.json"
ADDENDUM_V4 = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_EXECUTION_ADDENDUM_v4.json"
ADDENDUM_V5 = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_EXECUTION_ADDENDUM_v5.json"
ADDENDUM_V6 = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_EXECUTION_ADDENDUM_v6.json"
ADDENDUM_V7 = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_EXECUTION_ADDENDUM_v7.json"
ADDENDUM_V8 = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_EXECUTION_ADDENDUM_v8.json"
ADDENDUM_V9 = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_EXECUTION_ADDENDUM_v9.json"
GATE = ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/H1_CARRIERID_H32_FOLD0_TERMINAL_GATE.json"
SOURCE_AUTHORITY = ROOT / "pilot_artifacts/h1_carrierid_hu/source_authority_v1"
FRESH_SOURCE_PREFLIGHT = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_FRESH_SOURCE_PREFLIGHT_v1.json"
TMUX_ENV_PREFLIGHT = ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_TMUX_ENV_PREFLIGHT_v1.json"

PREREG_SHA = "6a94ee2ac96885d7462d14e47b8ea454dcf6414f07ddf8f29a43660ea602dd8c"
ADDENDUM_V3_SHA = "2876461eb0ee73e18437df5f22d7c289bfe7e720c5074760e6dde6b422cd1095"
ADDENDUM_V4_SHA = "5659ac674c6afda72bff220517dac0525b0509096bba1438bc2a2956ee54ea95"
ADDENDUM_V5_SHA = "6a3ac9e83aacaab7c410f7cc5db438a3df07859566fa391af18d54f3976dd48e"
ADDENDUM_V6_SHA = "30f0109280df182abb507e9a2a1f725196faba9cb8434e67a3e19502e0317684"
ADDENDUM_V7_SHA = "73f0b9dc77c33587112a97eabc4f8cddecf4f6f2232fd703ea1db657b9d73cea"
ADDENDUM_V8_SHA = "38de46d526db081bfa76700006846d095fea341ed1886ebce5f683c50ab45f45"
GATE_SHA = "6783513e868e77dc222b43737a239f9b14330544eb03ce3cfe9d976de992a9a4"
QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
HC_CACHE_SHA = "88261cc03532b605da1790e8669760d4d47e2f87d2db1428060445541638b0af"
CURRENT_M4_PRODUCER_SHA = "8f459b7a06c1b18865fc602b10e46876b4bb1dbf7ecba0fdf8e8f654737c86cb"
HISTORICAL_M4_PRODUCER_SHA = "71de3630d12cfaaa36e059acfb4c23ae293ba38ce785fbad1771fe21e86907b2"
EVALUATOR_SHA_BEFORE_REPAIR = "25b21ed4269f8938fa521ce6e83f69c6624056ca511100981b89141f068f7e2c"
EXPECTED_TARGET_FILE_SHA = {
    "ses-19250101T111740": "f8af652a31228f08b26ff8a7ecfd2a6bd05a345e368da8c98be649229f80adb2",
    "ses-19250101T112404": "b946c4cf49f00c2ea8f0051481765ed090294e3e1e5a035cc5716ad93aa052b2",
}
EXPECTED_SESSION_SAMPLES = {
    "ses-19250101T111740": 6735,
    "ses-19250101T112404": 2230,
}
EXPECTED_TOTAL_SAMPLES = 8965
CHECKPOINT_SCHEMA = "h1_carrierid_h32_hu_terminal_checkpoint_v1"
MATCHED_INITIAL_STATE_SHA = "8208b6ebec793e424e1b95cd1cbefdd9c2d3dceea9ee9d01c598a007446e90eb"
HU_VERSION = "HU-v1"
HU_FEATURE_NAMES = ("mean_rate", "fano", "log_isi_cv", "autocorr_tau_s")
FRESH_SOURCE_PREFLIGHT_SHA = "2ac532a2983c5806ab5892b958a993b2c98f0dc24f90f6e348aa406ed7f96642"
TMUX_ENV_PREFLIGHT_SHA = "3c05c37f74003b11654b047762f2fe0a0d7935e7f24b2a1fbf73872eaeef4e18"
# Authoritative launch/evaluator order: training seeds before source setup;
# evaluation consumes the checkpoint bytes with torch.load before source
# setup.  Both yield this same source provenance.  The old no-checkpoint CPU
# preflight did not model either order and is retained only as diagnostic
# evidence; notably the effective float32 carrier bytes were identical.
HU_AUDITOR_ACTUAL_RAW_SHA = "7e62fc22446c6674574fecdaa6fdc255c7a621458aa184de01c659e6de4af996"
HU_AUDITOR_ACTUAL_NORMALIZER_SHA = "850c75da4d429a2c4f924b8409af34e765c3668185dfe57587da997627be9d7f"
HU_AUTHORITATIVE_EFFECTIVE_SHA = "e8c1330334aae58f34df0acbd465190690e939bf39444870938cf94b6ff6d488"
HU_AUDITOR_ACTUAL_MANIFEST_SHA = "30e33f2e9527010bba60c8c8f1025d9b9e0ab1fd309ce4717b7b79e14116e153"
HU_LOCAL_EXACT_ACTUAL_RAW_SHA = "358684dd9dc4fae7e2e4d6e5dfb48c0f3859357523a5eb843b6c1f8e79da4c27"
HU_LOCAL_EXACT_ACTUAL_NORMALIZER_SHA = "bd44a393a695bb3a813d1cd7679fdcffecc9d77bcf00df765a018fbc218ee27b"
HU_LOCAL_EXACT_ACTUAL_MANIFEST_SHA = "3e30e31560633034b507a5ce912598af56f7f512c7bfad800deeb91409a5d8d8"
HU_NONAUTHORITATIVE_NO_CHECKPOINT_RAW_SHA = "3c6f333aef4212bc4ab72ace3076fcbbff3c8cdf0975b845df1cbdfd000adf37"
HU_NONAUTHORITATIVE_NO_CHECKPOINT_NORMALIZER_SHA = "5f7c55653d9e277f9358fb8737d3b57fce36bbc70842c35db6cb3eacebd676c8"


@dataclass(frozen=True)
class HuLineageSupport:
    session_name: str
    trial_values: tuple[float, float, float, float]
    fifth_trial: float
    query_first_bin: int
    identity: np.ndarray
    support_sha256: str


@dataclass(frozen=True)
class HuTargetLineage:
    support: Mapping[str, HuLineageSupport]
    window_indices: tuple[tuple[str, int], ...]
    window_indices_sha256: str
    session_samples: Mapping[str, int]


class H1CarrierIdHuStrictTargetDataset(Dataset):
    """Strict fold-0 target view with one neural-only H-U carrier per session."""

    def __init__(
        self,
        records: Mapping[str, H1PilotRecord],
        lineage: HuTargetLineage,
        hu_normalizer: Any,
    ) -> None:
        self.records = {name: records[name] for name in H1_M4_FOLD0_TARGET}
        self.lineage = lineage
        self.window_indices = list(lineage.window_indices)
        self.window_indices_sha256 = lineage.window_indices_sha256
        self.carriers: dict[str, np.ndarray] = {}
        self.carrier_audits: dict[str, dict[str, Any]] = {}
        for name in H1_M4_FOLD0_TARGET:
            record = self.records[name]
            support = lineage.support[name]
            # This is the only H-U target carrier construction call.  The
            # builder signature has no velocity/behaviour argument.
            raw, audit = hu_from_record(record, support.trial_values)
            if audit.used_velocity or audit.used_behaviour_labels:
                raise NormalizedV2ContractError("H-U target adapter read behaviour")
            normalized = np.asarray(hu_normalizer.normalize(raw), dtype=np.float32)
            if normalized.shape != (record.num_neurons, 4) or not np.isfinite(normalized).all():
                raise NormalizedV2ContractError("H-U target carrier shape/finite drift")
            self.carriers[name] = normalized
            audit_body = audit.as_dict()
            audit_body.update(
                {
                    "raw_carrier_sha256": carrier_sha256(np.asarray(raw)),
                    "normalized_carrier_sha256": carrier_sha256(normalized),
                    "source_only_normalizer_sha256": str(hu_normalizer.normalizer_sha256),
                }
            )
            self.carrier_audits[name] = audit_body

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int):
        session, start = self.window_indices[int(index)]
        record = self.records[session]
        support = self.lineage.support[session]
        end = start + WINDOW
        if start < support.query_first_bin or not record.eval_mask[end - 1]:
            raise PilotDataError("H-U strict query crossed support boundary or invalid output")
        return (
            record.neural[start:end],
            record.velocity[start:end],
            support.identity,
            session,
            self.carriers[session],
        )


def _support_sha256(record: H1PilotRecord, values: tuple[float, ...], identity: np.ndarray) -> str:
    """Reproduce the frozen gate's non-carrier support-lineage digest.

    The historical digest includes support outcomes because it identifies the
    complete already-frozen target support slice.  Those outcomes are never
    passed to ``hu_from_record`` and never participate in the H-U carrier.
    """

    digest = hashlib.sha256()
    digest.update(np.asarray(values, np.float64).tobytes())
    digest.update(identity.tobytes())
    for value in values:
        trial = record.blocks_for(value)
        digest.update(trial.rates.tobytes())
        digest.update(trial.velocity.tobytes())
        digest.update(trial.block_indices.tobytes())
    return digest.hexdigest()


def build_target_lineage(records: Mapping[str, H1PilotRecord]) -> HuTargetLineage:
    """Build support/query lineage without constructing any H-C carrier."""

    if set(records) != set(H1_M4_FOLD0_TARGET) or len(records) != 2:
        raise NormalizedV2ContractError("H-U requires exactly the two fold-0 target sessions")
    support: dict[str, HuLineageSupport] = {}
    windows: list[tuple[str, int]] = []
    samples: dict[str, int] = {}
    for name in H1_M4_FOLD0_TARGET:
        record = records[name]
        if record.session_name != name:
            raise NormalizedV2ContractError(f"H-U target record/session drift at {name}")
        values = tuple(float(value) for value in record.trial_values[:4])
        if len(values) != 4 or len(record.trial_values) < 5:
            raise NormalizedV2ContractError(f"{name}: H-U strict support requires four plus query trials")
        fifth = float(record.trial_values[4])
        fifth_bins = np.flatnonzero(
            record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth)
        )
        if fifth_bins.size == 0:
            raise NormalizedV2ContractError(f"{name}: fifth trial has no eval-valid boundary")
        boundary = int(fifth_bins[0])
        identity = interpolate_identity(record, values)
        support[name] = HuLineageSupport(
            session_name=name,
            trial_values=values,  # type: ignore[arg-type]
            fifth_trial=fifth,
            query_first_bin=boundary,
            identity=identity,
            support_sha256=_support_sha256(record, values, identity),
        )
        before = len(windows)
        for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
            if record.eval_mask[start + WINDOW - 1]:
                windows.append((name, start))
        samples[name] = len(windows) - before
    if not windows:
        raise NormalizedV2ContractError("H-U strict target query is empty")
    frozen_windows = tuple(windows)
    return HuTargetLineage(
        support=support,
        window_indices=frozen_windows,
        window_indices_sha256=_window_manifest_hash(frozen_windows),
        session_samples=samples,
    )


def validate_target_lineage(
    records: Mapping[str, H1PilotRecord],
    lineage: HuTargetLineage,
    gate_target: Mapping[str, Any],
    *,
    expected_file_sha: Mapping[str, str] = EXPECTED_TARGET_FILE_SHA,
    expected_query_sha: str = QUERY_SHA,
    expected_session_samples: Mapping[str, int] = EXPECTED_SESSION_SAMPLES,
    expected_total_samples: int = EXPECTED_TOTAL_SAMPLES,
) -> dict[str, Any]:
    """Fail closed on every target lineage field H-U scientifically needs."""

    expected_names = tuple(H1_M4_FOLD0_TARGET)
    if tuple(gate_target.get("sessions", ())) != expected_names:
        raise NormalizedV2ContractError("gate receipt target session names drifted")
    if set(records) != set(expected_names) or set(lineage.support) != set(expected_names):
        raise NormalizedV2ContractError("target record/support session set drifted")
    if gate_target.get("files") != dict(expected_file_sha):
        raise NormalizedV2ContractError("gate receipt target NWB SHA binding drifted")
    frozen_support = gate_target.get("support_and_carrier_hashes")
    if not isinstance(frozen_support, Mapping) or set(frozen_support) != set(expected_names):
        raise NormalizedV2ContractError("gate receipt target support binding is malformed")
    bound: dict[str, Any] = {}
    for name in expected_names:
        record = records[name]
        value = lineage.support[name]
        gate_value = frozen_support[name]
        if record.input_sha256 != expected_file_sha[name]:
            raise NormalizedV2ContractError(f"{name}: target NWB input SHA drifted")
        observed = {
            "trial_values": list(value.trial_values),
            "fifth_trial": value.fifth_trial,
            "query_first_bin": value.query_first_bin,
            "support_sha256": value.support_sha256,
        }
        expected = {key: gate_value[key] for key in observed}
        if observed != expected:
            raise NormalizedV2ContractError(f"{name}: target support/boundary lineage drifted")
        # Deliberately do not read gate_value['carrier_sha256'] here.
        bound[name] = observed
    gate_query = gate_target.get("strict_query_window_indices_sha256")
    if gate_query != expected_query_sha or lineage.window_indices_sha256 != expected_query_sha:
        raise NormalizedV2ContractError("H-U strict target query SHA drifted")
    if dict(lineage.session_samples) != dict(expected_session_samples):
        raise NormalizedV2ContractError("H-U per-session query sample counts drifted")
    if len(lineage.window_indices) != expected_total_samples:
        raise NormalizedV2ContractError("H-U total query sample count drifted")
    return {
        "sessions": list(expected_names),
        "files": dict(expected_file_sha),
        "support": bound,
        "query_window_indices_sha256": lineage.window_indices_sha256,
        "samples": len(lineage.window_indices),
        "session_samples": dict(lineage.session_samples),
        "historical_carrier_sha256_fields_consulted": False,
    }


@dataclass(frozen=True)
class ArtifactSnapshot:
    path: Path
    raw: bytes
    sha256: str

    def json_object(self, label: str) -> dict[str, Any]:
        try:
            value = json.loads(self.raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NormalizedV2ContractError(f"{label} is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise NormalizedV2ContractError(f"{label} is not a JSON object")
        return value


def _snapshot(
    path: Path,
    label: str,
    *,
    expected_sha: str | None = None,
    require_0444: bool = False,
) -> ArtifactSnapshot:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if require_0444 and stat.S_IMODE(resolved.stat().st_mode) != 0o444:
        raise NormalizedV2ContractError(f"{label} must have exact mode 0444")
    raw = resolved.read_bytes()  # decisive artifact is consumed exactly once
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise NormalizedV2ContractError(f"{label} SHA-256 drift")
    return ArtifactSnapshot(resolved, raw, digest)


def _snapshot_immutable_json_pair(path: Path, label: str, *, expected_sha: str) -> tuple[ArtifactSnapshot, dict[str, Any]]:
    snapshot = _snapshot(path, label, expected_sha=expected_sha, require_0444=True)
    sidecar = _snapshot(Path(f"{path}.sha256"), f"{label} sidecar", require_0444=True)
    if sidecar.raw != f"{snapshot.sha256}  {path.name}\n".encode("ascii"):
        raise NormalizedV2ContractError(f"{label} SHA sidecar mismatch")
    return snapshot, snapshot.json_object(label)


def _runtime_closure_paths() -> dict[str, Path]:
    """Exact bounded runtime closure whose hashes v4 freezes and rechecks."""

    return {
        "scripts/h1_carrierid_hu_terminal_evaluate.py": Path(__file__).resolve(),
        "scripts/h1_carrierid_hu_launcher.py": ROOT / "scripts/h1_carrierid_hu_launcher.py",
        "scripts/h1_carrierid_evaluate.py": ROOT / "scripts/h1_carrierid_evaluate.py",
        "src/h1_m4_eb_normalized_v2_contract.py": ROOT / "src/h1_m4_eb_normalized_v2_contract.py",
        "src/data/h1_m4_eb_pilot.py": ROOT / "src/data/h1_m4_eb_pilot.py",
        "src/data/h1_m4_eb_fold0_datamodule.py": ROOT / "src/data/h1_m4_eb_fold0_datamodule.py",
        "src/data/h1_m4_eb_normalized_v2.py": ROOT / "src/data/h1_m4_eb_normalized_v2.py",
        "src/data/h1_carrierid_hu.py": ROOT / "src/data/h1_carrierid_hu.py",
        "src/data/h1_carrierid_hu_features.py": ROOT / "src/data/h1_carrierid_hu_features.py",
        "src/models/h1_carrierid_module.py": ROOT / "src/models/h1_carrierid_module.py",
        "src/models/h1_carrierid_hu_module.py": ROOT / "src/models/h1_carrierid_hu_module.py",
        "src/models/components/h1_carrierid_spint.py": ROOT / "src/models/components/h1_carrierid_spint.py",
        "configs/experiment/h1_carrierid_hu.yaml": ROOT / "configs/experiment/h1_carrierid_hu.yaml",
        "configs/data/falcon_h1_carrierid_hu.yaml": ROOT / "configs/data/falcon_h1_carrierid_hu.yaml",
        "configs/model/falcon_h1_carrierid_hu.yaml": ROOT / "configs/model/falcon_h1_carrierid_hu.yaml",
        "pilot_artifacts/h1_carrierid_hu/source_authority_v1/H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json": SOURCE_AUTHORITY / "H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json",
        "pilot_artifacts/h1_carrierid_hu/source_authority_v1/H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json.sha256": SOURCE_AUTHORITY / "H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json.sha256",
        "pilot_artifacts/h1_carrierid_hu/source_authority_v1/fold0_frozen_eb_plan.npz": SOURCE_AUTHORITY / "fold0_frozen_eb_plan.npz",
        "pilot_artifacts/h1_carrierid_hu/source_authority_v1/fold0_frozen_eb_plan.manifest.json": SOURCE_AUTHORITY / "fold0_frozen_eb_plan.manifest.json",
        "pilot_artifacts/h1_carrierid_hu/source_authority_v1/fold0_all_source_m4_carriers.npz": SOURCE_AUTHORITY / "fold0_all_source_m4_carriers.npz",
        "pilot_artifacts/h1_carrierid_hu/source_authority_v1/fold0_all_source_m4_carriers.manifest.json": SOURCE_AUTHORITY / "fold0_all_source_m4_carriers.manifest.json",
        "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_FRESH_SOURCE_PREFLIGHT_v1.json": FRESH_SOURCE_PREFLIGHT,
        "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_FRESH_SOURCE_PREFLIGHT_v1.json.sha256": Path(f"{FRESH_SOURCE_PREFLIGHT}.sha256"),
        "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_TMUX_ENV_PREFLIGHT_v1.json": TMUX_ENV_PREFLIGHT,
        "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_TMUX_ENV_PREFLIGHT_v1.json.sha256": Path(f"{TMUX_ENV_PREFLIGHT}.sha256"),
    }


def _code_sha256() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in _runtime_closure_paths().items()}


def _formal_scope_v4() -> dict[str, Any]:
    return {
        "development_fold0_only": True,
        "fold_date": "19250101",
        "public_target_sessions": list(H1_M4_FOLD0_TARGET),
        "formal_or_organizer_test_forbidden": True,
        "gpu_launched_while_writing_addendum": False,
    }


def _scientific_read_rule_v4() -> dict[str, Any]:
    return {
        "authoritative_preregistration_schema": "h1_carrierid_h32_hu_preregistration_v2",
        "authoritative_preregistration_sha256": PREREG_SHA,
        "primary_comparators": ["H-C", "H-C0"],
        "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
        "third_escape_reading_forbidden": True,
    }


def _execution_contract_v4() -> dict[str, Any]:
    return {
        "ordering": [
            "output_pair_freshness_before_all_decisive_artifacts",
            "trusted_external_v4_sha_plus_0444_sidecar",
            "exact_live_runtime_closure_hash_map",
            "read_once_prereg_v3_gate_config_checkpoint",
            "checkpoint_config_and_source_provenance_before_target_open",
            "public_development_fold0_target_lineage",
            "label_free_HU_carrier_then_CPU_metric_evaluation",
        ],
        "resolved_config": {
            "seed": 42,
            "train": True,
            "test": False,
            "ckpt_path": None,
            "pilot_arm": "hu",
            "fold_date": "19250101",
            "optimizer": "torch.optim.Adam",
            "lr": 5.0e-5,
            "precision": "32-true",
            "epochs": 50,
            "data_target": "src.data.h1_carrierid_hu.H1CarrierIdHuDataModule",
            "model_target": "src.models.h1_carrierid_hu_module.H1CarrierIdHuLitModule",
            "net_target": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
            "batch_size": 32,
            "window_size": 700,
            "calibration_n_trials": 4,
            "max_trial_length": 1024,
            "model_dim": 1024,
            "num_covariates": 7,
            "carrier_dim": 4,
            "carrier_hidden_dim": 32,
            "carrier_trial_length": 1024,
        },
        "checkpoint": {
            "epoch_zero_based": 49,
            "epochs_completed": 50,
            "schema": CHECKPOINT_SCHEMA,
            "arm": "hu",
            "seed_42_binding": "resolved_config_sha256",
            "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        },
        "source_provenance": {
            "hc_source_cache_sha256": HC_CACHE_SHA,
            "hu_version": HU_VERSION,
            "hu_feature_names": list(HU_FEATURE_NAMES),
            "effective_source_carriers_shape": [116, 176, 4],
            "effective_source_carriers_count": 116,
        },
        "target_and_metric_wording": {
            "carrier_construction_label_free": True,
            "carrier_builder_used_velocity": False,
            "carrier_builder_used_behaviour_labels": False,
            "evaluator_reads_support_behaviour_for_frozen_lineage_digest": True,
            "evaluator_reads_query_behaviour_for_metric": True,
        },
        "output_transaction": {
            "preexisting_body_or_sidecar_fails_before_checkpoint_or_target": True,
            "temporary_files_mode": "0444",
            "publish": "hard_link_O_EXCL_semantics",
            "rollback_new_links_on_pair_failure": True,
            "file_and_directory_fsync": True,
        },
    }


def _execution_contract_v5() -> dict[str, Any]:
    contract = json.loads(json.dumps(_execution_contract_v4()))
    contract["ordering"].insert(
        5, "exact_immutable_HC_HC0_source_authority_before_source_setup"
    )
    contract["resolved_config"].update(
        {
            "source_authority_directory": "source_authority_v1",
            "source_authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
        }
    )
    contract["source_provenance"].update(
        {
            "authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
            "fresh_source_preflight_sha256": FRESH_SOURCE_PREFLIGHT_SHA,
            "authority_transform_sha256": "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a",
            "fresh_numeric_rebuild_forbidden_as_authority": True,
        }
    )
    return contract


def _validate_addendum_v4(
    path: Path,
    *,
    trusted_external_sha256: str,
) -> tuple[ArtifactSnapshot, dict[str, Any]]:
    if len(trusted_external_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in trusted_external_sha256):
        raise NormalizedV2ContractError("H-U v4 requires a lowercase 64-hex trusted external SHA anchor")
    snapshot, body = _snapshot_immutable_json_pair(
        path, "H-U v4 execution addendum", expected_sha=trusted_external_sha256
    )
    expected_top = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v4",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "formal_scope": _formal_scope_v4(),
        "scientific_read_rule": _scientific_read_rule_v4(),
        "execution_contract": _execution_contract_v4(),
    }
    for key, expected in expected_top.items():
        if body.get(key) != expected:
            raise NormalizedV2ContractError(f"H-U v4 execution addendum exact contract drift at {key}")
    bindings = body.get("bindings")
    if not isinstance(bindings, Mapping):
        raise NormalizedV2ContractError("H-U v4 bindings missing")
    expected_bindings = {
        "preregistration_v2_sha256": PREREG_SHA,
        "execution_addendum_v3_sha256": ADDENDUM_V3_SHA,
        "terminal_gate_sha256": GATE_SHA,
        "current_m4_producer_sha256": CURRENT_M4_PRODUCER_SHA,
        "historical_m4_producer_sha256": HISTORICAL_M4_PRODUCER_SHA,
        "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
    }
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise NormalizedV2ContractError(f"H-U v4 binding drift at {key}")
    live = _code_sha256()
    if bindings.get("runtime_closure_sha256") != live:
        raise NormalizedV2ContractError("H-U v4 complete runtime closure hash map drift")
    if set(bindings["runtime_closure_sha256"]) != set(_runtime_closure_paths()):
        raise NormalizedV2ContractError("H-U v4 runtime closure key set drift")
    return snapshot, body


def _validate_addendum_v5(
    path: Path,
    *,
    trusted_external_sha256: str,
) -> tuple[ArtifactSnapshot, dict[str, Any]]:
    if len(trusted_external_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in trusted_external_sha256):
        raise NormalizedV2ContractError("H-U v5 requires a lowercase 64-hex trusted external SHA anchor")
    snapshot, body = _snapshot_immutable_json_pair(
        path, "H-U v5 execution addendum", expected_sha=trusted_external_sha256
    )
    expected_top = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v5",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "formal_scope": _formal_scope_v4(),
        "scientific_read_rule": _scientific_read_rule_v4(),
        "execution_contract": _execution_contract_v5(),
    }
    for key, expected in expected_top.items():
        if body.get(key) != expected:
            raise NormalizedV2ContractError(f"H-U v5 execution addendum exact contract drift at {key}")
    bindings = body.get("bindings")
    if not isinstance(bindings, Mapping):
        raise NormalizedV2ContractError("H-U v5 bindings missing")
    expected_bindings = {
        "preregistration_v2_sha256": PREREG_SHA,
        "execution_addendum_v4_sha256": ADDENDUM_V4_SHA,
        "terminal_gate_sha256": GATE_SHA,
        "source_authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
        "fresh_source_preflight_sha256": FRESH_SOURCE_PREFLIGHT_SHA,
        "source_authority_cache_sha256": HC_CACHE_SHA,
        "source_authority_transform_sha256": "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a",
        "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
    }
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise NormalizedV2ContractError(f"H-U v5 binding drift at {key}")
    live = _code_sha256()
    if bindings.get("runtime_closure_sha256") != live:
        raise NormalizedV2ContractError("H-U v5 complete runtime closure hash map drift")
    if set(bindings["runtime_closure_sha256"]) != set(_runtime_closure_paths()):
        raise NormalizedV2ContractError("H-U v5 runtime closure key set drift")
    return snapshot, body


def _validate_addendum_v6(
    path: Path,
    *,
    trusted_external_sha256: str,
) -> tuple[ArtifactSnapshot, dict[str, Any]]:
    if len(trusted_external_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in trusted_external_sha256):
        raise NormalizedV2ContractError("H-U v6 requires a lowercase 64-hex trusted external SHA anchor")
    snapshot, body = _snapshot_immutable_json_pair(
        path, "H-U v6 execution addendum", expected_sha=trusted_external_sha256
    )
    expected_top = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v6",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "formal_scope": _formal_scope_v4(),
        "scientific_read_rule": _scientific_read_rule_v4(),
        "execution_contract": _execution_contract_v5(),
    }
    for key, expected in expected_top.items():
        if body.get(key) != expected:
            raise NormalizedV2ContractError(f"H-U v6 execution addendum exact contract drift at {key}")
    bindings = body.get("bindings")
    expected_bindings = {
        "preregistration_v2_sha256": PREREG_SHA,
        "execution_addendum_v5_sha256": ADDENDUM_V5_SHA,
        "terminal_gate_sha256": GATE_SHA,
        "source_authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
        "source_authority_cache_sha256": HC_CACHE_SHA,
        "source_authority_transform_sha256": "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a",
        "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
    }
    if not isinstance(bindings, Mapping):
        raise NormalizedV2ContractError("H-U v6 bindings missing")
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise NormalizedV2ContractError(f"H-U v6 binding drift at {key}")
    if bindings.get("runtime_closure_sha256") != _code_sha256():
        raise NormalizedV2ContractError("H-U v6 complete runtime closure hash map drift")
    launch = body.get("launch_order_reproducibility")
    if launch != {
        "training_seed_before_source_setup": True,
        "evaluator_checkpoint_load_before_source_setup": True,
        "authoritative_hu_raw_sha256": HU_AUDITOR_ACTUAL_RAW_SHA,
        "authoritative_hu_normalizer_sha256": HU_AUDITOR_ACTUAL_NORMALIZER_SHA,
        "authoritative_effective_source_carriers_sha256": HU_AUTHORITATIVE_EFFECTIVE_SHA,
        "authoritative_source_manifest_sha256": HU_AUDITOR_ACTUAL_MANIFEST_SHA,
        "no_checkpoint_preflight_raw_sha256": HU_NONAUTHORITATIVE_NO_CHECKPOINT_RAW_SHA,
        "no_checkpoint_preflight_normalizer_sha256": HU_NONAUTHORITATIVE_NO_CHECKPOINT_NORMALIZER_SHA,
        "no_checkpoint_preflight_is_authoritative": False,
        "train_and_evaluator_actual_paths_match": True,
        "hash_contract_relaxed": False,
    }:
        raise NormalizedV2ContractError("H-U v6 launch-order reproducibility contract drift")
    return snapshot, body


def _validate_addendum_v7(
    path: Path,
    *,
    trusted_external_sha256: str,
) -> tuple[ArtifactSnapshot, dict[str, Any]]:
    snapshot, body = _snapshot_immutable_json_pair(
        path, "H-U v7 execution addendum", expected_sha=trusted_external_sha256
    )
    expected = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v7",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "formal_scope": _formal_scope_v4(),
        "scientific_read_rule": _scientific_read_rule_v4(),
        "execution_contract": _execution_contract_v5(),
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise NormalizedV2ContractError(f"H-U v7 exact contract drift at {key}")
    bindings = body.get("bindings", {})
    if (
        bindings.get("execution_addendum_v6_sha256") != ADDENDUM_V6_SHA
        or bindings.get("preregistration_v2_sha256") != PREREG_SHA
        or bindings.get("runtime_closure_sha256") != _code_sha256()
    ):
        raise NormalizedV2ContractError("H-U v7 binding/runtime closure drift")
    rule = body.get("source_hash_equality_rule", {})
    if (
        rule.get("checkpoint_metadata_must_equal_evaluator_reconstruction") is not True
        or rule.get("hardcode_float64_raw_or_normalizer_hash") is not False
        or rule.get("effective_float32_carrier_sha256") != HU_AUTHORITATIVE_EFFECTIVE_SHA
        or rule.get("hash_contract_relaxed") is not False
    ):
        raise NormalizedV2ContractError("H-U v7 source hash equality rule drift")
    return snapshot, body


def _validate_addendum_v8(
    path: Path,
    *,
    trusted_external_sha256: str,
) -> tuple[ArtifactSnapshot, dict[str, Any]]:
    snapshot, body = _snapshot_immutable_json_pair(
        path, "H-U v8 execution addendum", expected_sha=trusted_external_sha256
    )
    expected = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v8",
        "status": "SEALED_AFTER_FAILED_ENVIRONMENT_LAUNCH_BEFORE_ANY_HU_TRAINING",
        "supersedes_only": "launcher_and_evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "formal_scope": _formal_scope_v4(),
        "scientific_read_rule": _scientific_read_rule_v4(),
        "execution_contract": _execution_contract_v5(),
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise NormalizedV2ContractError(f"H-U v8 exact contract drift at {key}")
    bindings = body.get("bindings", {})
    if (
        bindings.get("execution_addendum_v7_sha256") != ADDENDUM_V7_SHA
        or bindings.get("tmux_environment_preflight_sha256") != TMUX_ENV_PREFLIGHT_SHA
        or bindings.get("preregistration_v2_sha256") != PREREG_SHA
        or bindings.get("runtime_closure_sha256") != _code_sha256()
    ):
        raise NormalizedV2ContractError("H-U v8 binding/runtime closure drift")
    environment = body.get("tmux_child_environment_contract", {})
    if environment != {
        "command_prefix": ["env", "PYTHONNOUSERSITE=1", "PYTHONPATH=", "CUDA_VISIBLE_DEVICES=<gpu_id>"],
        "python_bin": "/home/xinyuan/miniconda3/envs/spint/bin/python",
        "expected_torch_prefix": "/home/xinyuan/miniconda3/envs/spint",
        "package_identity_preflight_required": True,
        "launch_receipt_records_effective_environment_and_package_identity": True,
    }:
        raise NormalizedV2ContractError("H-U v8 tmux environment contract drift")
    failed = body.get("failed_launch_v1", {})
    if (
        failed.get("output_root")
        != str(ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v1")
        or failed.get("preserved_immutable_history") is not True
        or failed.get("training_started") is not False
        or failed.get("rerun_performed_during_v8_repair") is not False
    ):
        raise NormalizedV2ContractError("H-U v8 failed-launch preservation contract drift")
    return snapshot, body


def _validate_addendum_v9(
    path: Path,
    *,
    trusted_external_sha256: str,
) -> tuple[ArtifactSnapshot, dict[str, Any]]:
    snapshot, body = _snapshot_immutable_json_pair(
        path, "H-U v9 execution addendum", expected_sha=trusted_external_sha256
    )
    expected = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v9",
        "status": "SEALED_AFTER_TERMINAL_TRAINING_BEFORE_SUCCESSFUL_TARGET_EVALUATION",
        "supersedes_only": "evaluator_source_reconstruction_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "formal_scope": _formal_scope_v4(),
        "scientific_read_rule": _scientific_read_rule_v4(),
        "execution_contract": _execution_contract_v5(),
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise NormalizedV2ContractError(f"H-U v9 exact contract drift at {key}")
    bindings = body.get("bindings", {})
    if (
        bindings.get("execution_addendum_v8_sha256") != ADDENDUM_V8_SHA
        or bindings.get("terminal_checkpoint_sha256")
        != "36833d4b1260bbbc3f69c4840f8f1bd330e8a6c2a3a1cb05c3a1ec63a74cb599"
        or bindings.get("resolved_config_sha256")
        != "b113a982395d642569e2854932769ecdedf3734d54602a9cebb9da3750707c5b"
        or bindings.get("runtime_closure_sha256") != _code_sha256()
    ):
        raise NormalizedV2ContractError("H-U v9 binding/runtime closure drift")
    repair = body.get("evaluator_source_reconstruction_repair", {})
    if (
        repair.get("checkpoint_metadata_equals_live_reconstruction_required") is not True
        or repair.get("equality_relaxed") is not False
        or repair.get("target_opened_during_repair_preflight") is not False
        or repair.get("trainer_fit_called") is not False
    ):
        raise NormalizedV2ContractError("H-U v9 source reconstruction repair drift")
    return snapshot, body


def _validate_checkpoint(ckpt: Mapping[str, Any]) -> Mapping[str, Any]:
    if "state_dict" not in ckpt or int(ckpt.get("epoch", -1)) != 49:
        raise NormalizedV2ContractError("H-U evaluator requires a real fixed epoch-49 checkpoint")
    if int(ckpt.get("global_step", 0)) <= 0:
        raise NormalizedV2ContractError("H-U checkpoint lacks positive training global_step")
    meta = ckpt.get("h1_carrierid")
    if not isinstance(meta, Mapping):
        raise NormalizedV2ContractError("H-U checkpoint lacks h1_carrierid metadata")
    required = {
        "schema": CHECKPOINT_SCHEMA,
        "arm": "hu",
        "fold_date": "19250101",
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_selection",
        "initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
        "carrier_hidden_dim": 32,
        "carrier_dim": 4,
        "carrier_trial_length": 1024,
        "carrier_mode": "source_hu_label_free_descriptor",
        "carrier_intervention": "hu",
        "hu_version": HU_VERSION,
        "hu_feature_names": list(HU_FEATURE_NAMES),
        "effective_source_carriers_shape": [116, 176, 4],
        "effective_source_carriers_count": 116,
        "label_free": True,
        "used_principal_components": False,
        "deployment_target_optimizer_steps": 0,
        "deployment_target_backward_steps": 0,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise NormalizedV2ContractError(f"H-U terminal checkpoint metadata drift at {key}")
    return meta


def _validate_resolved_config(config: Any) -> None:
    expected = {
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
        "pilot.arm": "hu",
        "pilot.fold_date": "19250101",
        "pilot.zero_carrier": False,
        "pilot.calibration_n_trials": 4,
        "pilot.batch_size": 32,
        "pilot.fixed_terminal_epochs": 50,
        "pilot.no_checkpoint_selection": True,
        "trainer.max_epochs": 50,
        "trainer.min_epochs": 50,
        "trainer.precision": "32-true",
        "trainer.limit_val_batches": 0,
        "trainer.num_sanity_val_steps": 0,
        "data._target_": "src.data.h1_carrierid_hu.H1CarrierIdHuDataModule",
        "data.task": "h1",
        "data.batch_size": 32,
        "data.window_size": 700,
        "data.calibration_n_trials": 4,
        "data.max_trial_length": 1024,
        "data.random_calibration": True,
        "data.smooth_calibration": False,
        "data.interpolate_trials": True,
        "data.interpolate_trials_kind": "cubic",
        "data.seed": 42,
        "data.fixed_epochs": 50,
        "data.normalizer_floor": 1.0e-12,
        "data.source_authority_dir": str(SOURCE_AUTHORITY),
        "data.source_authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
        "model._target_": "src.models.h1_carrierid_hu_module.H1CarrierIdHuLitModule",
        "model.task": "h1",
        "model.pilot_arm": "hu",
        "model.fold_date": "19250101",
        "model.optimizer._target_": "torch.optim.Adam",
        "model.optimizer._partial_": True,
        "model.optimizer.lr": 5.0e-5,
        "model.optimizer.weight_decay": 0.0,
        "model.net._target_": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        "model.net.carrier_hidden_dim": 32,
        "model.net.carrier_dim": 4,
        "model.net.carrier_trial_length": 1024,
        "model.net.zero_carrier": False,
        "model.net.model_dim": 1024,
        "model.net.num_covariates": 7,
        "model.net.window_size": 700,
        "model.net.num_heads": 64,
        "model.net.num_layers": 1,
        "model.net.num_id_layers": 3,
    }
    for key, value in expected.items():
        if OmegaConf.select(config, key) != value:
            raise NormalizedV2ContractError(f"resolved H-U config drift at {key}")


def _validate_gate_matched_initial_state(gate: Mapping[str, Any]) -> None:
    checkpoints = gate.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise NormalizedV2ContractError("terminal gate lacks checkpoint bindings")
    for arm in ("h_c_full", "h_c0_separate_literal_zero"):
        value = checkpoints.get(arm)
        if not isinstance(value, Mapping) or value.get("metadata", {}).get("initial_state_sha256") != MATCHED_INITIAL_STATE_SHA:
            raise NormalizedV2ContractError(f"terminal gate {arm} matched initial-state drift")
    pair = checkpoints.get("h_c_pair_binding")
    if not isinstance(pair, Mapping) or pair.get("initial_state_sha256") != MATCHED_INITIAL_STATE_SHA:
        raise NormalizedV2ContractError("terminal gate H-C/H-C0 pair initial-state drift")


def _validate_source_checkpoint_binding(meta: Mapping[str, Any], dm: H1CarrierIdHuDataModule) -> dict[str, Any]:
    manifest = dm.pilot_manifest()
    authority = manifest.get("source_authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("receipt_sha256") != HU_SOURCE_AUTHORITY_RECEIPT_SHA256
        or authority.get("carrier_cache_sha256") != HC_CACHE_SHA
        or authority.get("transform_sha256")
        != "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a"
        or authority.get("target_nwb_opened") is not False
    ):
        raise NormalizedV2ContractError("H-U reconstructed source authority provenance drift")
    expected = {
        "source_manifest_sha256": dm.pilot_manifest_sha256,
        "source_cache_sha256": HC_CACHE_SHA,
        "normalized_cache_sha256": manifest["normalized_cache_sha256"],
        "normalizer_sha256": manifest["hu_normalizer_sha256"],
        "hu_raw_sha256": manifest["hu_raw_sha256"],
        "effective_source_carriers_sha256": manifest["effective_source_carriers_sha256"],
        "effective_source_carriers_shape": [116, 176, 4],
        "effective_source_carriers_count": 116,
        "hu_version": HU_VERSION,
        "hu_feature_names": list(HU_FEATURE_NAMES),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise NormalizedV2ContractError(f"H-U checkpoint/source provenance drift at {key}")
    if manifest.get("effective_source_carriers_sha256") != HU_AUTHORITATIVE_EFFECTIVE_SHA:
        raise NormalizedV2ContractError("H-U evaluator effective carrier drift")
    if manifest.get("carrier_cache_sha256") != HC_CACHE_SHA:
        raise NormalizedV2ContractError("H-U reconstructed source cache SHA drift")
    if dm.hu_normalizer.normalizer_sha256 != manifest["hu_normalizer_sha256"]:
        raise NormalizedV2ContractError("H-U reconstructed source normalizer SHA drift")
    if not manifest.get("effective_source_carriers_nonidentity_all"):
        raise NormalizedV2ContractError("H-U effective source carrier nonidentity audit failed")
    return {
        "source_manifest_sha256": dm.pilot_manifest_sha256,
        "source_cache_sha256": HC_CACHE_SHA,
        "normalized_cache_sha256": manifest["normalized_cache_sha256"],
        "hu_normalizer_sha256": manifest["hu_normalizer_sha256"],
        "hu_s_src": manifest["hu_s_src"],
        "hu_raw_sha256": manifest["hu_raw_sha256"],
        "effective_source_carriers_sha256": manifest["effective_source_carriers_sha256"],
        "effective_source_carriers_shape": manifest["effective_source_carriers_shape"],
        "effective_source_carriers_count": manifest["effective_source_carriers_count"],
        "hu_version": manifest["hu_version"],
        "hu_feature_names": manifest["hu_feature_names"],
        "source_authority_receipt_sha256": authority["receipt_sha256"],
        "source_authority_transform_sha256": authority["transform_sha256"],
    }


def _reconstruct_source_in_training_order(
    config: Any,
    *,
    cache_dir: Path,
) -> tuple[H1CarrierIdHuDataModule, dict[str, Any]]:
    """Replay the source-affecting `src/train.py` setup order exactly on CPU.

    This is an explicit execution closure, not an accepted-hash workaround:
    seed first, instantiate DataModule, instantiate model, instantiate Trainer,
    then call DataModule.setup.  The model and CPU Trainer perform no forward,
    backward, optimizer, or data-loader step.  They exist solely because their
    construction/import order is part of the numerical runtime that preceded
    the terminal checkpoint's source manifest.
    """

    L.seed_everything(int(config.seed), workers=True)
    dm = hydra.utils.instantiate(config.data, cache_dir=str(cache_dir.resolve()))
    model = hydra.utils.instantiate(config.model)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=int(config.trainer.max_epochs),
        logger=False,
        enable_checkpointing=False,
    )
    dm.trainer = trainer
    dm.setup("fit")
    evidence = {
        "order": [
            "checkpoint_snapshot_and_torch_load",
            "lightning.seed_everything(seed=42,workers=True)",
            "hydra.instantiate(config.data)",
            "hydra.instantiate(config.model)",
            "Trainer(accelerator=cpu,devices=1,precision=32-true)",
            "datamodule.setup(fit)",
        ],
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "trainer_class": f"{type(trainer).__module__}.{type(trainer).__qualname__}",
        "trainer_accelerator": type(trainer.accelerator).__name__,
        "trainer_fit_called": False,
        "model_forward_or_backward_called": False,
        "target_opened": False,
    }
    return dm, evidence


def pretarget_source_only_validate(
    *, checkpoint: Path, config_path: Path, cache_dir: Path,
) -> dict[str, Any]:
    """Validate the actual terminal checkpoint through source setup, then stop."""

    config_snapshot = _snapshot(config_path, "resolved H-U config")
    config = OmegaConf.create(config_snapshot.raw.decode("utf-8"))
    _validate_resolved_config(config)
    checkpoint_snapshot = _snapshot(checkpoint, "H-U terminal checkpoint")
    if checkpoint_snapshot.sha256 != "36833d4b1260bbbc3f69c4840f8f1bd330e8a6c2a3a1cb05c3a1ec63a74cb599":
        raise NormalizedV2ContractError("H-U pretarget checkpoint differs from terminal checkpoint binding")
    if config_snapshot.sha256 != "b113a982395d642569e2854932769ecdedf3734d54602a9cebb9da3750707c5b":
        raise NormalizedV2ContractError("H-U pretarget config differs from resolved config binding")
    ckpt = torch.load(io.BytesIO(checkpoint_snapshot.raw), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise NormalizedV2ContractError("H-U checkpoint is not a mapping")
    meta = _validate_checkpoint(ckpt)
    if meta.get("config_sha256") != config_snapshot.sha256:
        raise NormalizedV2ContractError("resolved H-U config SHA does not bind checkpoint")
    dm, order = _reconstruct_source_in_training_order(config, cache_dir=cache_dir)
    binding = _validate_source_checkpoint_binding(meta, dm)
    return {
        "schema": "h1_carrierid_hu_pretarget_source_validation_v1",
        "status": "PASS_SOURCE_ONLY_STOP_BEFORE_TARGET",
        "checkpoint_sha256": checkpoint_snapshot.sha256,
        "config_sha256": config_snapshot.sha256,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_global_step": int(ckpt["global_step"]),
        "source_provenance": binding,
        "source_reconstruction_order": order,
        "target_opened": False,
        "formal_or_organizer_opened": False,
        "trainer_fit_called": False,
    }


def _assert_output_pair_fresh(path: Path) -> tuple[Path, Path]:
    output = path.resolve()
    sidecar = Path(f"{output}.sha256")
    conflicts = [str(value) for value in (output, sidecar) if value.exists()]
    if conflicts:
        raise FileExistsError(f"refusing H-U output pair conflict before checkpoint/target access: {conflicts}")
    return output, sidecar


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_temp_0444(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.fchmod(fd, 0o444)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def write_immutable_json_o_excl(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    """Transactionally publish an immutable JSON/SHA pair without partials."""

    output, sidecar = _assert_output_pair_fresh(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    side_bytes = f"{digest}  {output.name}\n".encode("ascii")
    token = f"{os.getpid()}.{secrets.token_hex(12)}"
    temp_output = output.parent / f".{output.name}.tmp.{token}"
    temp_sidecar = output.parent / f".{sidecar.name}.tmp.{token}"
    published_output = False
    published_sidecar = False
    try:
        _write_temp_0444(temp_output, encoded)
        _write_temp_0444(temp_sidecar, side_bytes)
        os.link(temp_output, output)  # fails with EEXIST; never replaces
        published_output = True
        try:
            os.link(temp_sidecar, sidecar)
            published_sidecar = True
        except BaseException:
            output.unlink()
            published_output = False
            raise
        _fsync_directory(output.parent)
    except BaseException:
        if published_sidecar and sidecar.exists():
            sidecar.unlink()
        if published_output and output.exists():
            output.unlink()
        _fsync_directory(output.parent)
        raise
    finally:
        for temporary in (temp_output, temp_sidecar):
            if temporary.exists():
                temporary.unlink()
        _fsync_directory(output.parent)
    return output, digest


def write_execution_addendum_v4(path: Path = ADDENDUM_V4) -> tuple[Path, str]:
    """Seal the v4 execution-only successor and print its external SHA anchor."""

    code = _code_sha256()
    if code["src/data/h1_m4_eb_pilot.py"] != CURRENT_M4_PRODUCER_SHA:
        raise NormalizedV2ContractError("current M4 producer changed before H-U v4 sealing")
    prereg = _snapshot(PREREG, "H-U v2 preregistration", expected_sha=PREREG_SHA, require_0444=True)
    v3, _ = _snapshot_immutable_json_pair(ADDENDUM_V3, "H-U v3 execution addendum", expected_sha=ADDENDUM_V3_SHA)
    gate = _snapshot(GATE, "H-C fold-0 terminal gate", expected_sha=GATE_SHA).json_object("terminal gate")
    body = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v4",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "formal_scope": _formal_scope_v4(),
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "scientific_read_rule": _scientific_read_rule_v4(),
        "bindings": {
            "preregistration_v2_sha256": PREREG_SHA,
            "execution_addendum_v3_sha256": v3.sha256,
            "terminal_gate_sha256": GATE_SHA,
            "current_m4_producer_sha256": CURRENT_M4_PRODUCER_SHA,
            "historical_m4_producer_sha256": HISTORICAL_M4_PRODUCER_SHA,
            "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
            "runtime_closure_sha256": code,
        },
        "execution_contract": _execution_contract_v4(),
        "historical_carrier_reconstruction_check": {
            "status": "skipped_by_design_for_label_free_HU",
            "not_claimed_as_passed": True,
            "historical_gate_carrier_sha256_fields_forbidden": True,
        },
        "h_u_scientific_inputs": {
            "target_files": gate["target"]["files"],
            "query_window_indices_sha256": QUERY_SHA,
            "samples": EXPECTED_TOTAL_SAMPLES,
            "session_samples": EXPECTED_SESSION_SAMPLES,
        },
        "trusted_external_anchor_instruction": "Pass this file's printed lowercase SHA-256 via --execution-addendum-sha256; the sidecar alone is not the trust anchor.",
    }
    return write_immutable_json_o_excl(path, body)


def write_execution_addendum_v5(path: Path = ADDENDUM_V5) -> tuple[Path, str]:
    """Seal the source-authority execution repair as the v5 successor."""

    code = _code_sha256()
    v4, _ = _snapshot_immutable_json_pair(
        ADDENDUM_V4, "H-U v4 execution addendum", expected_sha=ADDENDUM_V4_SHA
    )
    authority, authority_body = _snapshot_immutable_json_pair(
        SOURCE_AUTHORITY / "H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json",
        "H-U source authority",
        expected_sha=HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
    )
    preflight, preflight_body = _snapshot_immutable_json_pair(
        FRESH_SOURCE_PREFLIGHT,
        "H-U fresh source preflight",
        expected_sha=FRESH_SOURCE_PREFLIGHT_SHA,
    )
    if preflight_body.get("status") != "PASS_EXACT_LAUNCHER_DATAMODULE_FRESH_SOURCE_CACHE":
        raise NormalizedV2ContractError("H-U fresh source preflight status drift before v5 sealing")
    if authority_body.get("bindings", {}).get("h_c_h_c0_source_cache_sha256") != HC_CACHE_SHA:
        raise NormalizedV2ContractError("H-U source authority cache binding drift before v5 sealing")
    body = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v5",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "formal_scope": _formal_scope_v4(),
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "scientific_read_rule": _scientific_read_rule_v4(),
        "bindings": {
            "preregistration_v2_sha256": PREREG_SHA,
            "execution_addendum_v4_sha256": v4.sha256,
            "terminal_gate_sha256": GATE_SHA,
            "source_authority_receipt_sha256": authority.sha256,
            "fresh_source_preflight_sha256": preflight.sha256,
            "source_authority_cache_sha256": HC_CACHE_SHA,
            "source_authority_transform_sha256": "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a",
            "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
            "runtime_closure_sha256": code,
        },
        "execution_contract": _execution_contract_v5(),
        "fresh_cache_blocker_repair": {
            "blocked_fresh_transform_sha256": "5ef0991434cb3a8c911df1f07a28ede8e4c0096472e92a8d79333900d6feb72c",
            "blocked_fresh_carrier_cache_sha256": "85e0b0655036674d0b84103c83051a50cae830548be40af7bbe2078a2ab331a2",
            "authoritative_transform_sha256": "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a",
            "authoritative_carrier_cache_sha256": HC_CACHE_SHA,
            "root_cause": "roundoff-scale LAPACK/SVD backend drift changed byte hashes while source partition, mean, scale and scientific algorithm stayed fixed",
            "hash_contract_relaxed": False,
            "fresh_85e_cache_adopted": False,
            "repair": "consume exact immutable H-C/H-C0 source authority for matched H-U",
        },
        "trusted_external_anchor_instruction": "Pass this v5 file's printed lowercase SHA-256 via --execution-addendum-sha256; v4 remains immutable history.",
    }
    return write_immutable_json_o_excl(path, body)


def write_execution_addendum_v6(path: Path = ADDENDUM_V6) -> tuple[Path, str]:
    """Seal the read-once authority and launch-order correction."""

    code = _code_sha256()
    v5, _ = _snapshot_immutable_json_pair(
        ADDENDUM_V5, "H-U v5 execution addendum", expected_sha=ADDENDUM_V5_SHA
    )
    authority, authority_body = _snapshot_immutable_json_pair(
        SOURCE_AUTHORITY / "H1_CARRIERID_HU_SOURCE_AUTHORITY_v1.json",
        "H-U source authority",
        expected_sha=HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
    )
    if authority_body.get("bindings", {}).get("h_c_h_c0_source_cache_sha256") != HC_CACHE_SHA:
        raise NormalizedV2ContractError("H-U source authority cache binding drift before v6 sealing")
    body = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v6",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "formal_scope": _formal_scope_v4(),
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "scientific_read_rule": _scientific_read_rule_v4(),
        "bindings": {
            "preregistration_v2_sha256": PREREG_SHA,
            "execution_addendum_v5_sha256": v5.sha256,
            "terminal_gate_sha256": GATE_SHA,
            "source_authority_receipt_sha256": authority.sha256,
            "source_authority_cache_sha256": HC_CACHE_SHA,
            "source_authority_transform_sha256": "92c932d059d93aaffae3d676fe247b46971232027fe906cb4b28add93ddee66a",
            "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
            "runtime_closure_sha256": code,
        },
        "execution_contract": _execution_contract_v5(),
        "launch_order_reproducibility": {
            "training_seed_before_source_setup": True,
            "evaluator_checkpoint_load_before_source_setup": True,
            "authoritative_hu_raw_sha256": HU_AUDITOR_ACTUAL_RAW_SHA,
            "authoritative_hu_normalizer_sha256": HU_AUDITOR_ACTUAL_NORMALIZER_SHA,
            "authoritative_effective_source_carriers_sha256": HU_AUTHORITATIVE_EFFECTIVE_SHA,
            "authoritative_source_manifest_sha256": HU_AUDITOR_ACTUAL_MANIFEST_SHA,
            "no_checkpoint_preflight_raw_sha256": HU_NONAUTHORITATIVE_NO_CHECKPOINT_RAW_SHA,
            "no_checkpoint_preflight_normalizer_sha256": HU_NONAUTHORITATIVE_NO_CHECKPOINT_NORMALIZER_SHA,
            "no_checkpoint_preflight_is_authoritative": False,
            "train_and_evaluator_actual_paths_match": True,
            "hash_contract_relaxed": False,
        },
        "source_authority_loader": {
            "receipt_and_sidecar_read_once_via_file_descriptor": True,
            "all_manifest_and_npz_files_read_once_via_file_descriptor": True,
            "mode_and_regular_file_checked_with_fstat": True,
            "parsed_json_and_npz_from_verified_snapshot_bytes": True,
            "pathname_hash_then_reopen_forbidden": True,
        },
        "trusted_external_anchor_instruction": "Pass this v6 file's printed lowercase SHA-256 via --execution-addendum-sha256; v2-v5 remain immutable history.",
    }
    return write_immutable_json_o_excl(path, body)


def write_execution_addendum_v7(path: Path = ADDENDUM_V7) -> tuple[Path, str]:
    """Seal the corrected paired-runtime equality rule after v6 evidence."""

    v6, _ = _snapshot_immutable_json_pair(
        ADDENDUM_V6, "H-U v6 execution addendum", expected_sha=ADDENDUM_V6_SHA
    )
    body = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v7",
        "status": "SEALED_BEFORE_ANY_HU_GPU_EXECUTION",
        "formal_scope": _formal_scope_v4(),
        "supersedes_only": "evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "scientific_read_rule": _scientific_read_rule_v4(),
        "bindings": {
            "preregistration_v2_sha256": PREREG_SHA,
            "execution_addendum_v6_sha256": v6.sha256,
            "terminal_gate_sha256": GATE_SHA,
            "source_authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
            "source_authority_cache_sha256": HC_CACHE_SHA,
            "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
            "runtime_closure_sha256": _code_sha256(),
        },
        "execution_contract": _execution_contract_v5(),
        "source_hash_equality_rule": {
            "checkpoint_metadata_must_equal_evaluator_reconstruction": True,
            "compared_fields": [
                "source_manifest_sha256", "normalizer_sha256", "hu_raw_sha256",
                "effective_source_carriers_sha256", "effective_source_carriers_shape",
                "effective_source_carriers_count", "hu_version", "hu_feature_names",
            ],
            "hardcode_float64_raw_or_normalizer_hash": False,
            "reason": "float64 reduction low bits vary by numerical process/backend state; training and evaluator actual orders match within a run and are compared exactly",
            "effective_float32_carrier_sha256": HU_AUTHORITATIVE_EFFECTIVE_SHA,
            "hash_contract_relaxed": False,
        },
        "observed_launch_order_evidence": {
            "independent_auditor_actual": {
                "raw_sha256": HU_AUDITOR_ACTUAL_RAW_SHA,
                "normalizer_sha256": HU_AUDITOR_ACTUAL_NORMALIZER_SHA,
                "manifest_sha256": HU_AUDITOR_ACTUAL_MANIFEST_SHA,
            },
            "local_fresh_exact_training_and_evaluator_actual": {
                "raw_sha256": HU_LOCAL_EXACT_ACTUAL_RAW_SHA,
                "normalizer_sha256": HU_LOCAL_EXACT_ACTUAL_NORMALIZER_SHA,
                "manifest_sha256": HU_LOCAL_EXACT_ACTUAL_MANIFEST_SHA,
            },
            "effective_float32_sha256_both_environments": HU_AUTHORITATIVE_EFFECTIVE_SHA,
            "train_evaluator_match_in_each_actual_environment": True,
            "no_checkpoint_preflight_is_authoritative": False,
        },
        "source_authority_loader": {
            "all_decisive_files_read_once_and_parsed_from_verified_bytes": True,
            "pathname_hash_then_reopen_forbidden": True,
        },
        "trusted_external_anchor_instruction": "Pass this v7 file's printed lowercase SHA-256 via --execution-addendum-sha256; v2-v6 remain immutable history.",
    }
    return write_immutable_json_o_excl(path, body)


def write_execution_addendum_v8(path: Path = ADDENDUM_V8) -> tuple[Path, str]:
    """Seal the tmux child environment isolation repair."""

    v7, _ = _snapshot_immutable_json_pair(
        ADDENDUM_V7, "H-U v7 execution addendum", expected_sha=ADDENDUM_V7_SHA
    )
    preflight, preflight_body = _snapshot_immutable_json_pair(
        TMUX_ENV_PREFLIGHT,
        "H-U tmux environment preflight",
        expected_sha=TMUX_ENV_PREFLIGHT_SHA,
    )
    if preflight_body.get("status") != "PASS_ENV_ISOLATED_TMUX_CHILD_NO_TRAINING_OR_DATA":
        raise NormalizedV2ContractError("H-U tmux environment preflight status drift")
    failed = preflight_body.get("failed_v1_launch_preserved", {})
    if failed.get("path") != str(ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v1"):
        raise NormalizedV2ContractError("H-U failed-v1 preflight binding drift")
    body = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v8",
        "status": "SEALED_AFTER_FAILED_ENVIRONMENT_LAUNCH_BEFORE_ANY_HU_TRAINING",
        "formal_scope": _formal_scope_v4(),
        "supersedes_only": "launcher_and_evaluator_execution_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "scientific_read_rule": _scientific_read_rule_v4(),
        "bindings": {
            "preregistration_v2_sha256": PREREG_SHA,
            "execution_addendum_v7_sha256": v7.sha256,
            "terminal_gate_sha256": GATE_SHA,
            "source_authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
            "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
            "tmux_environment_preflight_sha256": preflight.sha256,
            "runtime_closure_sha256": _code_sha256(),
        },
        "execution_contract": _execution_contract_v5(),
        "tmux_child_environment_contract": {
            "command_prefix": ["env", "PYTHONNOUSERSITE=1", "PYTHONPATH=", "CUDA_VISIBLE_DEVICES=<gpu_id>"],
            "python_bin": "/home/xinyuan/miniconda3/envs/spint/bin/python",
            "expected_torch_prefix": "/home/xinyuan/miniconda3/envs/spint",
            "package_identity_preflight_required": True,
            "launch_receipt_records_effective_environment_and_package_identity": True,
        },
        "failed_launch_v1": {
            "output_root": str(ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v1"),
            "preserved_immutable_history": True,
            "failure": "tmux child enabled user site and imported ~/.local torch 2.12.0+cu130; CUDA driver rejected before training",
            "training_started": False,
            "rerun_performed_during_v8_repair": False,
            "artifact_file_bindings": failed.get("files"),
        },
        "recommended_fresh_output_root": str(
            ROOT / "pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v2_envfix"
        ),
        "repair_scope": {
            "gpu_launched": False,
            "training_or_data_opened": False,
            "target_or_formal_opened": False,
            "v2_science_changed": False,
        },
        "trusted_external_anchor_instruction": "Pass this v8 file's printed lowercase SHA-256 via --execution-addendum-sha256; v2-v7 and failed launch v1 remain history.",
    }
    return write_immutable_json_o_excl(path, body)


def write_execution_addendum_v9(path: Path = ADDENDUM_V9) -> tuple[Path, str]:
    """Seal the byte-exact training-order evaluator reconstruction repair."""

    v8, _ = _snapshot_immutable_json_pair(
        ADDENDUM_V8, "H-U v8 execution addendum", expected_sha=ADDENDUM_V8_SHA
    )
    body = {
        "schema": "h1_carrierid_h32_hu_execution_addendum_v9",
        "status": "SEALED_AFTER_TERMINAL_TRAINING_BEFORE_SUCCESSFUL_TARGET_EVALUATION",
        "formal_scope": _formal_scope_v4(),
        "supersedes_only": "evaluator_source_reconstruction_procedure",
        "does_not_supersede": "v2_scientific_read_rule",
        "scientific_read_rule_unchanged": True,
        "scientific_read_rule": _scientific_read_rule_v4(),
        "bindings": {
            "preregistration_v2_sha256": PREREG_SHA,
            "execution_addendum_v8_sha256": v8.sha256,
            "terminal_gate_sha256": GATE_SHA,
            "source_authority_receipt_sha256": HU_SOURCE_AUTHORITY_RECEIPT_SHA256,
            "matched_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
            "terminal_checkpoint_sha256": "36833d4b1260bbbc3f69c4840f8f1bd330e8a6c2a3a1cb05c3a1ec63a74cb599",
            "resolved_config_sha256": "b113a982395d642569e2854932769ecdedf3734d54602a9cebb9da3750707c5b",
            "runtime_closure_sha256": _code_sha256(),
        },
        "execution_contract": _execution_contract_v5(),
        "terminal_training": {
            "epoch_zero_based": 49,
            "global_step": 180500,
            "checkpoint_preserved": True,
            "retraining_performed": False,
            "source_manifest_sha256": "30e33f2e9527010bba60c8c8f1025d9b9e0ab1fd309ce4717b7b79e14116e153",
            "hu_raw_sha256": "7e62fc22446c6674574fecdaa6fdc255c7a621458aa184de01c659e6de4af996",
            "hu_normalizer_sha256": "850c75da4d429a2c4f924b8409af34e765c3668185dfe57587da997627be9d7f",
            "effective_source_carriers_sha256": HU_AUTHORITATIVE_EFFECTIVE_SHA,
        },
        "failed_evaluation": {
            "stopped_before_target": True,
            "output_receipt_published": False,
            "preserved_as_failure_evidence": True,
        },
        "evaluator_source_reconstruction_repair": {
            "order": [
                "snapshot and torch.load actual checkpoint bytes",
                "lightning.seed_everything(seed=42,workers=True)",
                "hydra.instantiate resolved config.data",
                "hydra.instantiate resolved config.model",
                "construct CPU Trainer with fixed precision/epochs",
                "datamodule.setup(fit)",
                "strict checkpoint metadata versus live source validation",
            ],
            "checkpoint_metadata_equals_live_reconstruction_required": True,
            "equality_relaxed": False,
            "accepted_hash_fallback_added": False,
            "trainer_fit_called": False,
            "model_forward_backward_or_optimizer_called": False,
            "target_opened_during_repair_preflight": False,
            "formal_or_organizer_opened": False,
        },
        "scientific_changes": {
            "checkpoint": False,
            "model": False,
            "data": False,
            "training": False,
            "config": False,
            "query_or_comparator": False,
        },
        "trusted_external_anchor_instruction": "Pass this v9 file's printed lowercase SHA-256 via --execution-addendum-sha256; v2-v8 and failed pre-target evaluation remain history.",
    }
    return write_immutable_json_o_excl(path, body)


def evaluate(
    *,
    checkpoint: Path,
    config_path: Path,
    cache_dir: Path,
    output: Path,
    execution_addendum_sha256: str,
) -> dict[str, Any]:
    # First operation: an output or sidecar conflict must fail before any
    # addendum/config/checkpoint read, source setup, or target access.
    _assert_output_pair_fresh(output)

    # The SHA argument is the external trust anchor.  The adjacent sidecar is
    # checked too, but is not treated as self-authenticating.
    addendum_snapshot, addendum = _validate_addendum_v9(
        ADDENDUM_V9, trusted_external_sha256=execution_addendum_sha256
    )
    code_sha = dict(addendum["bindings"]["runtime_closure_sha256"])

    # Read every decisive small artifact exactly once and retain the consumed
    # bytes/SHA for the terminal receipt.
    prereg_snapshot = _snapshot(PREREG, "H-U v2 preregistration", expected_sha=PREREG_SHA, require_0444=True)
    prereg = prereg_snapshot.json_object("H-U v2 preregistration")
    if prereg.get("schema") != "h1_carrierid_h32_hu_preregistration_v2" or prereg.get("status") != "PRE_REGISTERED_BEFORE_ANY_GPU_RUN":
        raise NormalizedV2ContractError("H-U v2 preregistration schema/status drift")
    v3_snapshot, v3 = _snapshot_immutable_json_pair(
        ADDENDUM_V3, "H-U v3 execution addendum", expected_sha=ADDENDUM_V3_SHA
    )
    if v3.get("supersedes_only") != "evaluator_execution_procedure" or v3.get("does_not_supersede") != "v2_scientific_read_rule":
        raise NormalizedV2ContractError("H-U v3 predecessor scope drift")
    v4_snapshot, v4 = _snapshot_immutable_json_pair(
        ADDENDUM_V4, "H-U v4 execution addendum", expected_sha=ADDENDUM_V4_SHA
    )
    if v4.get("supersedes_only") != "evaluator_execution_procedure" or v4.get("does_not_supersede") != "v2_scientific_read_rule":
        raise NormalizedV2ContractError("H-U v4 predecessor scope drift")
    gate_snapshot = _snapshot(GATE, "H-C fold-0 terminal gate", expected_sha=GATE_SHA)
    gate = gate_snapshot.json_object("H-C fold-0 terminal gate")
    _validate_gate_matched_initial_state(gate)

    config_snapshot = _snapshot(config_path, "resolved H-U config")
    try:
        config = OmegaConf.create(config_snapshot.raw.decode("utf-8"))
    except (UnicodeDecodeError, Exception) as exc:
        # OmegaConf raises several concrete parser/interpolation exception
        # classes; normalize all of them into the execution contract error.
        raise NormalizedV2ContractError("resolved H-U config parse failed") from exc
    _validate_resolved_config(config)

    checkpoint_snapshot = _snapshot(checkpoint, "H-U terminal checkpoint")
    if checkpoint_snapshot.sha256 != addendum["bindings"]["terminal_checkpoint_sha256"]:
        raise NormalizedV2ContractError("H-U evaluator checkpoint differs from v9 terminal checkpoint binding")
    if config_snapshot.sha256 != addendum["bindings"]["resolved_config_sha256"]:
        raise NormalizedV2ContractError("H-U evaluator config differs from v9 resolved config binding")
    ckpt = torch.load(io.BytesIO(checkpoint_snapshot.raw), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping):
        raise NormalizedV2ContractError("H-U checkpoint is not a mapping")
    meta = _validate_checkpoint(ckpt)
    if meta.get("config_sha256") != config_snapshot.sha256:
        raise NormalizedV2ContractError("resolved H-U config SHA does not bind checkpoint")

    # Source-only reconstruction and all checkpoint provenance bindings remain
    # before the first call to load_target_records.
    dm, source_reconstruction_order = _reconstruct_source_in_training_order(
        config, cache_dir=cache_dir
    )
    source_binding = _validate_source_checkpoint_binding(meta, dm)

    target = load_target_records(ROOT / "data/000954")
    lineage = build_target_lineage(target)
    lineage_binding = validate_target_lineage(target, lineage, gate["target"])
    dataset = H1CarrierIdHuStrictTargetDataset(target, lineage, dm.hu_normalizer)
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    metrics = _evaluate(model, dataset, device, "H-U")
    if metrics["samples"] != EXPECTED_TOTAL_SAMPLES or metrics["session_samples"] != EXPECTED_SESSION_SAMPLES:
        raise NormalizedV2ContractError("H-U evaluator metrics sample accounting drift")
    body = {
        "schema": "h1_carrierid_h32_hu_terminal_eval_v4",
        "status": "PASS_HU_LABEL_FREE_TERMINAL_EVALUATION",
        "arm": "hu",
        "fold_date": "19250101",
        "formal_scope": "development_fold0_only",
        "formal_or_organizer_test_opened": False,
        "checkpoint": {"path": str(checkpoint_snapshot.path), "sha256": checkpoint_snapshot.sha256, "metadata": dict(meta)},
        "config": {"path": str(config_snapshot.path), "sha256": config_snapshot.sha256},
        "immutable_bindings": {
            "preregistration_v2": {"path": str(prereg_snapshot.path), "sha256": prereg_snapshot.sha256, "schema": prereg["schema"]},
            "execution_addendum_v3_predecessor": {"path": str(v3_snapshot.path), "sha256": v3_snapshot.sha256, "schema": v3["schema"]},
            "execution_addendum_v4_predecessor": {"path": str(v4_snapshot.path), "sha256": v4_snapshot.sha256, "schema": v4["schema"]},
            "execution_addendum_v9": {
                "path": str(addendum_snapshot.path),
                "sha256": addendum_snapshot.sha256,
                "trusted_external_sha256": execution_addendum_sha256,
                "schema": addendum["schema"],
            },
            "terminal_gate": {"path": str(gate_snapshot.path), "sha256": gate_snapshot.sha256},
            "runtime_closure_sha256_exact": code_sha,
            "matched_h_c_h_c0_initial_state_sha256": MATCHED_INITIAL_STATE_SHA,
        },
        "source_provenance": source_binding,
        "source_reconstruction_order": source_reconstruction_order,
        "target_lineage": lineage_binding,
        "hu_target_carrier": {
            "construction": "support_neural_plus_source_only_HU_normalizer",
            "session_audits": dataset.carrier_audits,
            "label_free": True,
            "used_velocity": False,
            "used_behaviour_labels": False,
            "used_principal_components": False,
        },
        "evaluator_behaviour_reads": {
            "support_behaviour_read_for_frozen_noncarrier_lineage_digest": True,
            "query_behaviour_read_for_terminal_r2_metric": True,
            "behaviour_passed_to_HU_carrier_builder": False,
        },
        "historical_carrier_reconstruction_check": {
            "status": "skipped_by_design_for_label_free_HU",
            "reason": "Historical full/raw/row/label behaviour-carrier hashes are not H-U inputs; current producer SHA differs from their historical producer binding.",
            "historical_producer_sha256": HISTORICAL_M4_PRODUCER_SHA,
            "current_producer_sha256": CURRENT_M4_PRODUCER_SHA,
            "not_claimed_as_passed": True,
        },
        "metrics": metrics,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "model_state_immutable_during_evaluation": bool(metrics["state_immutable"]),
        "samples_expected": EXPECTED_TOTAL_SAMPLES,
        "session_samples_expected": EXPECTED_SESSION_SAMPLES,
    }
    write_immutable_json_o_excl(output, body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-addendum-sha256")
    parser.add_argument("--write-execution-addendum-v5", action="store_true")
    parser.add_argument("--write-execution-addendum-v6", action="store_true")
    parser.add_argument("--write-execution-addendum-v7", action="store_true")
    parser.add_argument("--write-execution-addendum-v8", action="store_true")
    parser.add_argument("--write-execution-addendum-v9", action="store_true")
    parser.add_argument("--pretarget-source-only", action="store_true")
    parser.add_argument("--i-have-authorization", action="store_true")
    args = parser.parse_args()
    if args.write_execution_addendum_v5:
        path, digest = write_execution_addendum_v5()
        print(json.dumps({
            "path": str(path),
            "trusted_external_sha256": digest,
            "required_evaluation_argument": f"--execution-addendum-sha256 {digest}",
        }, indent=2, sort_keys=True))
        return
    if args.write_execution_addendum_v6:
        path, digest = write_execution_addendum_v6()
        print(json.dumps({
            "path": str(path),
            "trusted_external_sha256": digest,
            "required_evaluation_argument": f"--execution-addendum-sha256 {digest}",
        }, indent=2, sort_keys=True))
        return
    if args.write_execution_addendum_v7:
        path, digest = write_execution_addendum_v7()
        print(json.dumps({
            "path": str(path),
            "trusted_external_sha256": digest,
            "required_evaluation_argument": f"--execution-addendum-sha256 {digest}",
        }, indent=2, sort_keys=True))
        return
    if args.write_execution_addendum_v8:
        path, digest = write_execution_addendum_v8()
        print(json.dumps({
            "path": str(path),
            "trusted_external_sha256": digest,
            "required_evaluation_argument": f"--execution-addendum-sha256 {digest}",
        }, indent=2, sort_keys=True))
        return
    if args.write_execution_addendum_v9:
        path, digest = write_execution_addendum_v9()
        print(json.dumps({
            "path": str(path),
            "trusted_external_sha256": digest,
            "required_evaluation_argument": f"--execution-addendum-sha256 {digest}",
        }, indent=2, sort_keys=True))
        return
    if args.pretarget_source_only:
        if any(value is None for value in (args.checkpoint, args.config_path, args.cache_dir)):
            parser.error("--pretarget-source-only requires --checkpoint, --config-path and --cache-dir")
        print(json.dumps(
            pretarget_source_only_validate(
                checkpoint=args.checkpoint,
                config_path=args.config_path,
                cache_dir=args.cache_dir,
            ),
            indent=2,
            sort_keys=True,
        ))
        return
    if not args.i_have_authorization:
        raise SystemExit(
            "H-U terminal evaluator opens fold-0 target recordings.  Pass "
            "--i-have-authorization only after the source checkpoint exists.  "
            "This script was landed ready-to-run and must not be executed in the implementation pass."
        )
    if any(value is None for value in (
        args.checkpoint,
        args.config_path,
        args.cache_dir,
        args.output,
        args.execution_addendum_sha256,
    )):
        parser.error(
            "evaluation requires --checkpoint, --config-path, --cache-dir, --output, "
            "and externally trusted --execution-addendum-sha256"
        )
    print(
        json.dumps(
            evaluate(
                checkpoint=args.checkpoint,
                config_path=args.config_path,
                cache_dir=args.cache_dir,
                output=args.output,
                execution_addendum_sha256=args.execution_addendum_sha256,
            ),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
