"""Source-only H1-EST4 data route.

EST4 reuses the immutable H-S/H-C date-LODO source partition, source windows,
M=4 schedule and ordinary-carrier RMS normalizer.  It adds only padded 100-ms
calibration rate/kinematic blocks needed by the differentiable ridge layer.
There is no target loader, validation loader, target evaluator, or checkpoint
restore path in this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np

from src.data.h1_carrierid_date_lodo_ci import (
    FIVE_DATE_AGGREGATE_SCHEMA,
    FIVE_DATE_AGGREGATE_STATUS,
    SOURCE_DATE_SCREEN_COMPLETE,
    _load_bound_frozen_plan,
    _read_immutable_json,
)
from src.data.h1_carrierid_date_lodo_phase2 import (
    CarrierIdDateLodoPhase2Error,
    H1CarrierIdDateLodoSchedule,
    H1CarrierIdDateLodoSourceDataModule,
    H1CarrierIdDateLodoSourceDataset,
    Phase2SourceBinding,
    _need,
)
from src.data.h1_m4_eb_pilot import EXPECTED_NEURONS, SUPPORT_TRIALS, VELOCITY_DIM, complete_row_shuffle
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256


EST4_ARMS = ("B-C", "B-LS", "L-C", "L-C0", "L-LS", "L-RS")
EST4_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_est4_cpu_preflight_v1"
EST4_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_EST4_SOURCE_ONLY_NOT_LAUNCHED"
# This narrow pair is deliberately a separate protocol from the historical
# six-arm attribution design.  It exists only to make one fresh, matched
# whole-pipeline B-C/L-C source-training screen possible after the existing
# five-date H-S/H-C aggregate is present.  Accepting it below must not relax
# any condition of the six-arm branch.
EST4_PAIR_ARMS = ("B-C", "L-C")
EST4_PAIR_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_est4_pair_cpu_preflight_v1"
EST4_PAIR_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_EST4_PAIR_SOURCE_ONLY_NOT_LAUNCHED"
EST4_MAX_BLOCKS_PER_TRIAL = 256
ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_mode(arm: str) -> tuple[str, str, bool]:
    normalized = str(arm).upper()
    if normalized not in EST4_ARMS:
        raise CarrierIdDateLodoPhase2Error(f"EST4 arm must be one of {EST4_ARMS}")
    if normalized == "B-C":
        return "baseline", "full", False
    if normalized == "B-LS":
        return "baseline", "ls", False
    if normalized == "L-C":
        return "learned", "full", False
    if normalized == "L-C0":
        return "learned", "full", False
    if normalized == "L-LS":
        return "learned", "label_rotate", False
    return "learned", "full", True


def _label_rotate(blocks: np.ndarray, *, session_name: str, trial_value: float) -> np.ndarray:
    """Match the existing nonzero deterministic per-trial label rotation law."""

    values = np.asarray(blocks, dtype=np.float32)
    _need(values.ndim == 2 and values.shape[1] == VELOCITY_DIM and values.shape[0] >= 2,
          "EST4 L-LS requires at least two valid 100-ms label blocks per support trial")
    token = hashlib.sha256(f"20260807|{session_name}|{trial_value}|0".encode()).digest()
    shift = 1 + int.from_bytes(token[:8], "big") % (values.shape[0] - 1)
    return np.roll(values, shift, axis=0)


class H1CarrierIdDateLodoEst4SourceDataset(H1CarrierIdDateLodoSourceDataset):
    """Shared H-S/H-C source samples plus analytic M=4 ridge inputs."""

    def __init__(self, binding: Phase2SourceBinding, *, est4_arm: str) -> None:
        super().__init__(binding)
        self.est4_arm = str(est4_arm).upper()
        self.estimator_mode, self.baseline_intervention, self.row_shuffle_output = _arm_mode(self.est4_arm)
        self._full_carriers: dict[tuple[str, int], np.ndarray] = {}
        self._baseline_carriers: dict[tuple[str, int], np.ndarray] = {}
        self._permutations: dict[str, np.ndarray] = {}
        frozen_plan = _load_bound_frozen_plan(binding)
        for entry in self.cache.entries:
            key = (str(entry.session_name), int(entry.start_index))
            raw = np.asarray(entry.carrier, dtype=np.float64)
            full = self.normalizer.normalize(raw).astype(np.float32)
            if self.baseline_intervention == "ls":
                # This route intentionally reuses the established frozen
                # ordinary-estimator LS construction for B-LS.
                from src.data.h1_m4_eb_pilot import label_rotation_carrier
                changed = label_rotation_carrier(self.records[key[0]], frozen_plan, entry.trial_values)
                baseline = self.normalizer.normalize(changed).astype(np.float32)
                _need(not np.array_equal(baseline, full), "EST4 B-LS carrier collapsed to B-C")
            else:
                baseline = full
            self._full_carriers[key], self._baseline_carriers[key] = full, baseline
        for session in binding.source_sessions:
            ids = np.repeat(np.arange(EXPECTED_NEURONS, dtype=np.int64).reshape(-1, 1), 4, axis=1)
            self._permutations[session] = complete_row_shuffle(ids, session, outer_date=binding.outer_date)[:, 0]

    def _ridge_inputs(self, *, session: str, calibration_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.records[session]
        values = record.trial_values[calibration_start : calibration_start + SUPPORT_TRIALS]
        _need(len(values) == SUPPORT_TRIALS, "EST4 schedule does not identify exactly four support trials")
        rates = np.zeros((SUPPORT_TRIALS, EST4_MAX_BLOCKS_PER_TRIAL, EXPECTED_NEURONS), dtype=np.float32)
        labels = np.zeros((SUPPORT_TRIALS, EST4_MAX_BLOCKS_PER_TRIAL, VELOCITY_DIM), dtype=np.float32)
        mask = np.zeros((SUPPORT_TRIALS, EST4_MAX_BLOCKS_PER_TRIAL), dtype=bool)
        for trial_index, value in enumerate(values):
            trial = record.blocks_for(value)
            blocks = int(trial.rates.shape[0])
            _need(2 <= blocks <= EST4_MAX_BLOCKS_PER_TRIAL, "EST4 support block count exceeds frozen padding bound")
            rates[trial_index, :blocks] = np.asarray(trial.rates, dtype=np.float32)
            velocity = np.asarray(trial.velocity, dtype=np.float32)
            if self.est4_arm == "L-LS":
                velocity = _label_rotate(velocity, session_name=session, trial_value=float(value))
            labels[trial_index, :blocks] = velocity
            mask[trial_index, :blocks] = True
        return rates, labels, mask

    def __getitem__(self, request: tuple[int, int]):
        neural, target, identity, session, _ordinary = super().__getitem__(request)
        _index, calibration_start = (int(request[0]), int(request[1]))
        key = (session, calibration_start)
        rates, labels, mask = self._ridge_inputs(session=session, calibration_start=calibration_start)
        return (
            neural, target, identity, session,
            self._baseline_carriers[key], rates, labels, mask, self._permutations[session],
        )


class H1CarrierIdDateLodoEst4DataModule(H1CarrierIdDateLodoSourceDataModule):
    """Fail-closed EST4 source route; receipt validation precedes source NWB I/O."""

    def __init__(
        self, *, est4_arm: str, est4_preflight_path: str, five_date_aggregate_path: str,
        frozen_plan_path: str, **kwargs: Any,
    ) -> None:
        if str(est4_arm).upper() not in EST4_ARMS:
            raise CarrierIdDateLodoPhase2Error("EST4 DataModule needs one frozen EST4 arm")
        if not all(str(value) for value in (est4_preflight_path, five_date_aggregate_path, frozen_plan_path)):
            raise CarrierIdDateLodoPhase2Error("EST4 requires preflight, 5/5 aggregate and frozen-plan paths")
        super().__init__(**kwargs)
        self.est4_arm = str(est4_arm).upper()
        self.est4_preflight_path = Path(est4_preflight_path).resolve()
        self.five_date_aggregate_path = Path(five_date_aggregate_path).resolve()
        self.frozen_plan_path = Path(frozen_plan_path).resolve()
        self._est4_preflight: dict[str, Any] | None = None
        self._est4_preflight_sha256: str | None = None
        self._five_date_aggregate_sha256: str | None = None

    def _validate_preopen_receipts(self) -> None:
        aggregate_path, aggregate, aggregate_sha = _read_immutable_json(
            self.five_date_aggregate_path, schema=FIVE_DATE_AGGREGATE_SCHEMA, status=FIVE_DATE_AGGREGATE_STATUS,
        )
        _need(tuple(aggregate.get("required_outer_dates", ())) == tuple(CONFIRMATORY_DATES)
              and aggregate.get("all_five_date_receipts_present_and_validated") is True,
              "EST4 requires the completed canonical 5/5 source-date aggregate")
        route = aggregate.get("route_prerequisite")
        _need(isinstance(route, Mapping) and route.get("status") == SOURCE_DATE_SCREEN_COMPLETE
              and route.get("automatic_route_selection") == "FORBIDDEN",
              "EST4 five-date aggregate must be a non-selecting source/date screen")
        # Do not silently downgrade the original six-arm study.  The only
        # alternative accepted here is the explicitly named B-C/L-C protocol,
        # whose immutable preflight has independently checked the same source
        # roster, schedule, normalizer, seed, and common consumer backbone.
        try:
            preflight_path, preflight, preflight_sha = _read_immutable_json(
                self.est4_preflight_path, schema=EST4_PREFLIGHT_SCHEMA, status=EST4_PREFLIGHT_STATUS,
            )
            expected_arms = EST4_ARMS
            expected_preflight_script = ROOT / "scripts/h1_carrierid_date_lodo_est4_preflight.py"
        except CarrierIdDateLodoPhase2Error:
            preflight_path, preflight, preflight_sha = _read_immutable_json(
                self.est4_preflight_path, schema=EST4_PAIR_PREFLIGHT_SCHEMA, status=EST4_PAIR_PREFLIGHT_STATUS,
            )
            expected_arms = EST4_PAIR_ARMS
            expected_preflight_script = ROOT / "scripts/h1_carrierid_date_lodo_est4_pair_preflight.py"
        _need(preflight.get("outer_date") == str(self.hparams.outer_date), "EST4 preflight outer-date drift")
        controls = preflight.get("source_controls")
        _need(isinstance(controls, Mapping) and controls.get("all_arms") == list(expected_arms)
              and controls.get("same_hs_hc_source_partition") is True
              and controls.get("same_source_windows") is True and controls.get("same_source_schedule") is True
              and controls.get("same_source_normalizer") is True and controls.get("fixed_terminal_epoch_zero_based") == 49,
              "EST4 preflight does not bind matched H-S/H-C source protocol")
        model = preflight.get("fresh_models", {}).get(self.est4_arm)
        _need(isinstance(model, Mapping) and model.get("arm") == self.est4_arm and model.get("fresh_seed") == 42,
              "EST4 preflight does not bind requested fresh arm")
        frozen = preflight.get("frozen_estimator_initialization")
        _need(isinstance(frozen, Mapping) and frozen.get("path") == str(self.frozen_plan_path)
              and frozen.get("sha256") == _sha256_file(self.frozen_plan_path),
              "EST4 frozen-plan path/hash differs from immutable preflight")
        config_path = ROOT / "configs/experiment" / f"h1_carrierid_date_lodo_est4_{self.est4_arm.lower().replace('-', '_')}.yaml"
        configured = preflight.get("configuration", {}).get(self.est4_arm, {})
        _need(config_path.is_file() and configured.get("sha256") == _sha256_file(config_path),
              "EST4 configuration template hash drift")
        code = preflight.get("code_sha256")
        code_paths = {
            "data": ROOT / "src/data/h1_carrierid_date_lodo_est4.py",
            "model": ROOT / "src/models/h1_carrierid_date_lodo_est4_module.py",
            "component": ROOT / "src/models/components/h1_carrierid_est4_spint.py",
            "preflight": expected_preflight_script,
        }
        _need(isinstance(code, Mapping) and all(code.get(name) == _sha256_file(path) for name, path in code_paths.items()),
              "EST4 code closure drift")
        self._est4_preflight, self._est4_preflight_sha256, self._five_date_aggregate_sha256 = preflight, preflight_sha, aggregate_sha
        self._validated_receipt_paths = {"est4_preflight": preflight_path, "five_date_aggregate": aggregate_path}

    def setup(self, stage: str | None = None) -> None:
        if self._setup_done:
            return
        self._validate_preopen_receipts()
        super().setup(stage)
        actual = self.binding.manifest()
        # The Phase-1 manifest is the canonical evidence that H-S and H-C
        # shared this exact source schedule/normalizer before EST4 exists.
        _need(self._est4_preflight is not None and self._est4_preflight_sha256 is not None and self._five_date_aggregate_sha256 is not None,
              "EST4 receipt gate was not captured")
        _need(self._est4_preflight.get("source_binding") == actual
              and self._est4_preflight.get("source_binding_sha256") == canonical_sha256(actual),
              "EST4 actual source binding differs from immutable preflight")
        dataset = H1CarrierIdDateLodoEst4SourceDataset(self.binding, est4_arm=self.est4_arm)
        sampler = H1CarrierIdDateLodoSchedule(dataset, self.binding)
        _need(dataset.window_indices == self.train_dataset.window_indices,
              "EST4 changed H-S/H-C source windows")
        _need(np.array_equal(sampler.binding.calibration_schedule, self.train_batch_sampler.binding.calibration_schedule),
              "EST4 changed H-S/H-C source calibration schedule")
        self._base_source_binding_sha256 = canonical_sha256(actual)
        self.train_dataset, self.train_batch_sampler = dataset, sampler

    def phase2_source_manifest(self) -> dict[str, Any]:
        base = super().phase2_source_manifest()
        body = {
            **base,
            "schema": "h1_carrierid_date_lodo_est4_source_binding_v1",
            "est4_arm": self.est4_arm,
            "phase2_base_source_binding_sha256": self._base_source_binding_sha256,
            "est4_preflight_path": str(self.est4_preflight_path),
            "est4_preflight_sha256": self.est4_preflight_sha256,
            "five_date_aggregate_path": str(self.five_date_aggregate_path),
            "five_date_aggregate_sha256": self.five_date_aggregate_sha256,
            "frozen_plan_path": str(self.frozen_plan_path), "frozen_plan_sha256": _sha256_file(self.frozen_plan_path),
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "deployment_target_optimizer_steps": 0, "deployment_target_backward_steps": 0,
        }
        body["est4_source_binding_sha256"] = canonical_sha256(body)
        return body

    @property
    def est4_preflight_sha256(self) -> str:
        if self._est4_preflight_sha256 is None:
            raise RuntimeError("EST4 preflight has not been validated")
        return self._est4_preflight_sha256

    @property
    def five_date_aggregate_sha256(self) -> str:
        if self._five_date_aggregate_sha256 is None:
            raise RuntimeError("EST4 5/5 aggregate has not been validated")
        return self._five_date_aggregate_sha256
