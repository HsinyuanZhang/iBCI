"""Isolated h=32 H-LS source view for the H1 five-date date-LODO control.

The base Phase-2 loader remains the sole owner of source NWB access, source
windows, the chronological M=4 schedule, and the source-fitted RMS
normalizer.  This module changes only the four-dimensional carrier: velocity
rows are deterministically rotated *within each support trial* and the frozen
analytic carrier is refitted.  It has no target loader.
"""
from __future__ import annotations

import json
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np

from src.data.h1_carrierid_date_lodo_phase2 import (
    CarrierIdDateLodoPhase2Error,
    H1CarrierIdDateLodoSchedule,
    H1CarrierIdDateLodoSourceDataModule,
    H1CarrierIdDateLodoSourceDataset,
    Phase2SourceBinding,
    _immutable,
    _need,
)
from src.data.h1_m4_eb_pilot import FrozenEBPlan, array_sha256, label_rotation_carrier
from src.h1_m4_cce_contract import canonical_sha256, sha256_file


HLS_SOURCE_BINDING_SCHEMA = "h1_carrierid_date_lodo_hls_source_binding_v1"
HLS_SOURCE_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_hls_source_preflight_v1"
HLS_SOURCE_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_SOURCE_ONLY_NOT_LAUNCHED"
ROOT = Path(__file__).resolve().parents[2]


def _read_immutable_hls_preflight(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and not candidate.is_symlink()
          and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"H-LS requires immutable source preflight: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CarrierIdDateLodoPhase2Error(f"invalid H-LS source preflight: {candidate}") from error
    _need(isinstance(body, dict) and body.get("schema") == HLS_SOURCE_PREFLIGHT_SCHEMA
          and body.get("status") == HLS_SOURCE_PREFLIGHT_STATUS,
          "H-LS source preflight schema/status drift")
    scope, controls = body.get("scope"), body.get("source_controls")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
          and scope.get("target_bytes_read") == 0,
          "H-LS source preflight records target access")
    _need(isinstance(controls, Mapping)
          and controls.get("carrier_intervention") == "temporal_velocity_label_rotation"
          and controls.get("same_h_c_source_windows") is True
          and controls.get("same_h_c_source_schedule") is True
          and controls.get("same_h_c_normalizer") is True
          and controls.get("seed") == 42 and controls.get("epochs") == 50
          and controls.get("fixed_terminal_epoch_zero_based") == 49,
          "H-LS source preflight controls drift")
    return candidate, body, sha256_file(candidate)


def load_bound_frozen_plan(binding: Phase2SourceBinding) -> FrozenEBPlan:
    """Recover the exact Phase-1 frozen estimator without opening a target."""

    source_manifest_path = binding.source_manifest_path
    _need(_immutable(source_manifest_path), "H-LS source manifest must remain immutable")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    frozen = source_manifest.get("frozen_plan")
    _need(isinstance(frozen, Mapping), "H-LS source manifest lacks frozen plan")
    manifest_path = Path(str(frozen.get("manifest_path", ""))).resolve()
    arrays_path = source_manifest_path.parent / "frozen_m4_plan.npz"
    _need(manifest_path.parent == source_manifest_path.parent and _immutable(manifest_path)
          and _immutable(arrays_path), "H-LS frozen-plan artifacts escape the immutable source bundle")
    _need(sha256_file(manifest_path) == frozen.get("manifest_sha256"),
          "H-LS frozen-plan manifest SHA drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected = {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"}
        _need(set(archive.files) == expected, "H-LS frozen-plan array members drift")
        mean = np.asarray(archive["mean"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
        pcs = np.asarray(archive["pcs"], dtype=np.float64)
        U = np.asarray(archive["U"], dtype=np.float64)
        mu = np.asarray(archive["mu"], dtype=np.float64)
        q = int(np.asarray(archive["q"]).item())
        ridge_lambda = float(np.asarray(archive["lambda"]).item())
        tau2 = float(np.asarray(archive["tau2"]).item())
    hashes = manifest.get("array_sha256")
    _need(isinstance(hashes, Mapping), "H-LS frozen-plan array hashes missing")
    for name, value in {"mean": mean, "scale": scale, "pcs": pcs, "U": U, "mu": mu}.items():
        _need(hashes.get(name) == array_sha256(value), f"H-LS frozen-plan {name} SHA drift")
    _need(manifest.get("outer_date") == binding.outer_date
          and tuple(manifest.get("source_sessions", ())) == binding.source_sessions,
          "H-LS frozen plan date/source partition drift")
    input_hashes = tuple(binding.records[name].input_sha256 for name in binding.source_sessions)
    _need(tuple(manifest.get("source_input_sha256", ())) == input_hashes,
          "H-LS frozen plan source-file binding drift")
    return FrozenEBPlan(
        outer_date=binding.outer_date, source_sessions=binding.source_sessions,
        source_input_sha256=input_hashes, mean=mean, scale=scale, pcs=pcs, q=q,
        ridge_lambda=ridge_lambda, U=U, mu=mu, tau2=tau2,
        raw_plan_sha256=str(manifest["raw_plan_sha256"]),
        raw_receipt_sha256=str(manifest["raw_receipt_sha256"]),
        eb_receipt_sha256=str(manifest["eb_receipt_sha256"]),
        transform_sha256=str(manifest["transform_sha256"]),
    )


class H1CarrierIdDateLodoHlsSourceDataset(H1CarrierIdDateLodoSourceDataset):
    """Exact Phase-2 source samples with only the fitted carrier misaligned."""

    def __init__(self, binding: Phase2SourceBinding) -> None:
        super().__init__(binding)
        plan = load_bound_frozen_plan(binding)
        effective: dict[tuple[str, int], np.ndarray] = {}
        full_rows: list[np.ndarray] = []
        hls_rows: list[np.ndarray] = []
        for entry in self.cache.entries:
            key = (str(entry.session_name), int(entry.start_index))
            changed = label_rotation_carrier(self.records[key[0]], plan, entry.trial_values)
            normalized = self.normalizer.normalize(np.asarray(changed, dtype=np.float64))
            full = self.normalizer.normalize(np.asarray(entry.carrier, dtype=np.float64))
            _need(normalized.shape == entry.carrier.shape and np.isfinite(normalized).all(),
                  "H-LS effective source carrier is malformed")
            _need(not np.array_equal(normalized, full),
                  "H-LS temporal label rotation collapsed to the full carrier")
            effective[key] = normalized.astype(np.float32)
            full_rows.append(full)
            hls_rows.append(normalized)
        self._effective = effective
        full_stack, hls_stack = np.stack(full_rows), np.stack(hls_rows)
        self.effective_source_carriers_sha256 = array_sha256(hls_stack)
        self.effective_source_carriers_shape = list(hls_stack.shape)
        self.effective_source_carriers_count = len(effective)
        self.full_source_carriers_sha256 = array_sha256(full_stack)
        self.effective_source_carriers_nonidentity_all = not np.array_equal(full_stack, hls_stack)

    def __getitem__(self, request: tuple[int, int]):
        neural, target, identity, session, _full = super().__getitem__(request)
        calibration_start = int(request[1])
        try:
            carrier = self._effective[(session, calibration_start)]
        except KeyError as error:
            raise CarrierIdDateLodoPhase2Error("H-LS source request misses its rotated carrier") from error
        return neural, target, identity, session, carrier


def hls_source_manifest(binding: Phase2SourceBinding,
                        dataset: H1CarrierIdDateLodoHlsSourceDataset) -> dict[str, Any]:
    """Canonical transform binding, intentionally independent of its receipt."""

    base = binding.manifest()
    body = {
        **base,
        "schema": HLS_SOURCE_BINDING_SCHEMA,
        "carrier_intervention": "temporal_velocity_label_rotation",
        "carrier_intervention_scope": "carrier_refit_only; neural/target/identity/windows/schedule unchanged",
        "label_rotation_definition": "deterministic nonzero within-each-support-trial velocity-row rotation, replicate 0",
        "same_h_c_source_windows": True,
        "same_h_c_source_schedule": True,
        "same_h_c_normalizer": True,
        "phase2_base_source_binding_sha256": canonical_sha256(base),
        "effective_source_carriers_sha256": dataset.effective_source_carriers_sha256,
        "effective_source_carriers_shape": dataset.effective_source_carriers_shape,
        "effective_source_carriers_count": dataset.effective_source_carriers_count,
        "effective_source_carriers_nonidentity_all": dataset.effective_source_carriers_nonidentity_all,
        "full_source_carriers_sha256": dataset.full_source_carriers_sha256,
    }
    body["hls_source_binding_sha256"] = canonical_sha256(body)
    return body


class H1CarrierIdDateLodoHlsSourceDataModule(H1CarrierIdDateLodoSourceDataModule):
    """Fit-only H-LS DataModule with a receipt gate before source NWB access."""

    def __init__(self, *, hls_source_preflight_path: str, **kwargs: Any) -> None:
        if not str(hls_source_preflight_path):
            raise CarrierIdDateLodoPhase2Error("H-LS DataModule requires its immutable source preflight")
        super().__init__(**kwargs)
        self.hls_source_preflight_path = Path(hls_source_preflight_path).resolve()
        self._hls_preflight: dict[str, Any] | None = None
        self._hls_preflight_sha256: str | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._setup_done:
            return
        _path, receipt, receipt_sha = _read_immutable_hls_preflight(self.hls_source_preflight_path)
        _need(receipt.get("outer_date") == str(self.hparams.outer_date),
              "H-LS source preflight outer-date drift")
        closure = receipt.get("code_sha256")
        expected_closure = {
            "data": ROOT / "src/data/h1_carrierid_date_lodo_hls.py",
            "model": ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py",
            "component": ROOT / "src/models/components/h1_carrierid_spint.py",
            "experiment": ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_phase2.yaml",
            "data_config": ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls.yaml",
            "model_config": ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml",
            "terminal_callback": ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml",
        }
        _need(isinstance(closure, Mapping)
              and all(closure.get(name) == sha256_file(path) for name, path in expected_closure.items()),
              "H-LS implementation/config changed after source preflight")
        # The parent performs the only source-recording opens in this module.
        super().setup(stage)
        original_dataset, original_schedule = self.train_dataset, self.train_batch_sampler
        dataset = H1CarrierIdDateLodoHlsSourceDataset(self.binding)
        schedule = H1CarrierIdDateLodoSchedule(dataset, self.binding)
        _need(dataset.window_indices == original_dataset.window_indices,
              "H-LS intervention changed source windows")
        _need(np.array_equal(schedule.binding.calibration_schedule,
                             original_schedule.binding.calibration_schedule),
              "H-LS intervention changed the M=4 schedule")
        _need(schedule.binding.batch_order_sha256 == original_schedule.binding.batch_order_sha256,
              "H-LS intervention changed batch order")
        actual = hls_source_manifest(self.binding, dataset)
        _need(receipt.get("source_binding_sha256") == canonical_sha256(actual)
              and receipt.get("source_binding") == actual,
              "H-LS runtime source binding differs from its immutable preflight")
        self.train_dataset, self.train_batch_sampler = dataset, schedule
        self._hls_preflight, self._hls_preflight_sha256 = receipt, receipt_sha

    @property
    def hls_source_preflight_sha256(self) -> str:
        if self._hls_preflight_sha256 is None:
            raise RuntimeError("H-LS source preflight has not been validated")
        return self._hls_preflight_sha256

    def phase2_source_manifest(self) -> dict[str, Any]:
        if not self._setup_done or not isinstance(self.train_dataset, H1CarrierIdDateLodoHlsSourceDataset):
            raise RuntimeError("H-LS source binding is not set up")
        return hls_source_manifest(self.binding, self.train_dataset)
