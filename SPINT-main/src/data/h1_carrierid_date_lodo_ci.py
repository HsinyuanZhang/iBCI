"""Source-only data controls for fresh H1 CarrierID CI32/CI64 training.

The shared date-LODO source cache, normalizer, window order, and M=4 schedule
are inherited unchanged from ``h1_carrierid_date_lodo_phase2``.  This module
only replaces the carrier returned to the source-training batch for LS/RS.
It intentionally has no target loader or evaluator.
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
from src.data.h1_m4_eb_pilot import (
    FrozenEBPlan,
    PilotDataError,
    array_sha256,
    complete_row_shuffle,
    label_rotation_carrier,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256


CI_CARRIER_INTERVENTIONS = ("full", "c0", "ls", "rs")
CI_ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")
CI_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_ci_cpu_preflight_v1"
CI_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_SOURCE_ONLY_NOT_LAUNCHED"
FIVE_DATE_AGGREGATE_SCHEMA = "h1_carrierid_date_lodo_five_date_heldout_aggregate_v1"
FIVE_DATE_AGGREGATE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_FIVE_DATE_SOURCE_DATE_SCREEN_COMPLETE_NO_ROUTE_SELECTED"
SOURCE_DATE_SCREEN_COMPLETE = "source/date screen complete"
ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_immutable_json(path: str | Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"CI requires immutable mode-0444 receipt: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CarrierIdDateLodoPhase2Error(f"CI receipt is invalid JSON: {candidate}") from error
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"CI receipt schema/status drift: {candidate}")
    return candidate, body, _sha256_file(candidate)


def _load_bound_frozen_plan(binding: Phase2SourceBinding) -> FrozenEBPlan:
    """Recover the exact persisted per-date plan for LS refits.

    This reads only the immutable source bundle already bound by Phase 1; it
    never recreates a plan from an outer-date target or changes normalisation.
    """

    manifest_path = binding.source_manifest_path
    _need(_immutable(manifest_path), "CI source manifest must remain immutable")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = source_manifest.get("frozen_plan")
    _need(isinstance(frozen, Mapping), "CI source manifest lacks frozen plan")
    plan_manifest_path = Path(str(frozen.get("manifest_path", ""))).resolve()
    _need(plan_manifest_path.parent == manifest_path.parent and _immutable(plan_manifest_path),
          "CI frozen-plan manifest escapes immutable date bundle")
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    _need(plan_manifest.get("outer_date") == binding.outer_date, "CI frozen plan outer-date drift")
    plan_array_path = manifest_path.parent / "frozen_m4_plan.npz"
    _need(_immutable(plan_array_path), "CI frozen-plan array must remain immutable")
    with np.load(plan_array_path, allow_pickle=False) as arrays:
        expected = {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"}
        _need(set(arrays.files) == expected, "CI frozen-plan array members drift")
        mean = np.asarray(arrays["mean"], dtype=np.float64)
        scale = np.asarray(arrays["scale"], dtype=np.float64)
        pcs = np.asarray(arrays["pcs"], dtype=np.float64)
        U = np.asarray(arrays["U"], dtype=np.float64)
        mu = np.asarray(arrays["mu"], dtype=np.float64)
        q = int(np.asarray(arrays["q"]).item())
        ridge_lambda = float(np.asarray(arrays["lambda"]).item())
        tau2 = float(np.asarray(arrays["tau2"]).item())
    claimed_hashes = plan_manifest.get("array_sha256")
    _need(isinstance(claimed_hashes, Mapping), "CI frozen-plan hashes missing")
    for name, value in {"mean": mean, "scale": scale, "pcs": pcs, "U": U, "mu": mu}.items():
        _need(claimed_hashes.get(name) == array_sha256(value), f"CI frozen-plan {name} hash drift")
    _need(tuple(plan_manifest.get("source_sessions", ())) == binding.source_sessions,
          "CI frozen-plan source sessions differ from binding")
    input_hashes = tuple(binding.records[name].input_sha256 for name in binding.source_sessions)
    _need(tuple(plan_manifest.get("source_input_sha256", ())) == input_hashes,
          "CI frozen-plan source file hashes differ from binding")
    return FrozenEBPlan(
        outer_date=binding.outer_date,
        source_sessions=binding.source_sessions,
        source_input_sha256=input_hashes,
        mean=mean, scale=scale, pcs=pcs, q=q, ridge_lambda=ridge_lambda,
        U=U, mu=mu, tau2=tau2,
        raw_plan_sha256=str(plan_manifest["raw_plan_sha256"]),
        raw_receipt_sha256=str(plan_manifest["raw_receipt_sha256"]),
        eb_receipt_sha256=str(plan_manifest["eb_receipt_sha256"]),
        transform_sha256=str(plan_manifest["transform_sha256"]),
    )


class H1CarrierIdDateLodoCiSourceDataset(H1CarrierIdDateLodoSourceDataset):
    """Shared source dataset with a fixed Full/C0/LS/RS carrier intervention."""

    def __init__(self, binding: Phase2SourceBinding, *, carrier_intervention: str) -> None:
        if carrier_intervention not in CI_CARRIER_INTERVENTIONS:
            raise CarrierIdDateLodoPhase2Error("CI carrier_intervention must be full, c0, ls, or rs")
        super().__init__(binding)
        self.carrier_intervention = carrier_intervention
        plan = _load_bound_frozen_plan(binding) if carrier_intervention == "ls" else None
        effective: dict[tuple[str, int], np.ndarray] = {}
        full_rows: list[np.ndarray] = []
        effective_rows: list[np.ndarray] = []
        for entry in self.cache.entries:
            key = (str(entry.session_name), int(entry.start_index))
            raw = np.asarray(entry.carrier, dtype=np.float64)
            if carrier_intervention == "ls":
                assert plan is not None
                changed = label_rotation_carrier(self.records[key[0]], plan, entry.trial_values)
            elif carrier_intervention == "rs":
                changed = complete_row_shuffle(raw, key[0], outer_date=binding.outer_date)
            else:
                changed = raw
            normalized = self.normalizer.normalize(np.asarray(changed, dtype=np.float64))
            full = self.normalizer.normalize(raw)
            if normalized.shape != raw.shape or not np.isfinite(normalized).all():
                raise CarrierIdDateLodoPhase2Error("CI effective source carrier is malformed")
            if carrier_intervention in {"ls", "rs"} and np.array_equal(normalized, full):
                raise CarrierIdDateLodoPhase2Error("CI LS/RS intervention collapsed to full carrier")
            effective[key] = normalized.astype(np.float32)
            full_rows.append(full)
            effective_rows.append(normalized)
        full_stack, effective_stack = np.stack(full_rows), np.stack(effective_rows)
        self._effective = effective
        self.effective_source_carriers_sha256 = array_sha256(effective_stack)
        self.effective_source_carriers_shape = list(effective_stack.shape)
        self.effective_source_carriers_count = len(effective)
        self.effective_source_carriers_nonidentity_all = not np.array_equal(full_stack, effective_stack)
        if carrier_intervention in {"ls", "rs"} and not self.effective_source_carriers_nonidentity_all:
            raise CarrierIdDateLodoPhase2Error("CI control has no source carrier intervention")

    def __getitem__(self, request: tuple[int, int]):
        neural, target, identity, session, _normal_full = super().__getitem__(request)
        _index, calibration_start = (int(request[0]), int(request[1]))
        try:
            effective = self._effective[(session, calibration_start)]
        except KeyError as error:
            raise CarrierIdDateLodoPhase2Error("CI source request misses effective carrier") from error
        return neural, target, identity, session, effective


class H1CarrierIdDateLodoCiDataModule(H1CarrierIdDateLodoSourceDataModule):
    """Source-only CI32/CI64 DataModule; its sampler is the shared Phase-1 one."""

    def __init__(
        self, *, carrier_intervention: str, ci_arm: str, ci_preflight_path: str,
        five_date_aggregate_path: str, **kwargs: Any,
    ) -> None:
        if str(carrier_intervention) not in CI_CARRIER_INTERVENTIONS:
            raise CarrierIdDateLodoPhase2Error("CI carrier_intervention must be full, c0, ls, or rs")
        if str(ci_arm).upper() not in CI_ARMS:
            raise CarrierIdDateLodoPhase2Error("CI source DataModule requires one declared CI arm")
        expected_intervention = str(ci_arm).split("-", 1)[1].lower()
        if expected_intervention != str(carrier_intervention):
            raise CarrierIdDateLodoPhase2Error("CI arm/intervention mismatch")
        if not str(ci_preflight_path) or not str(five_date_aggregate_path):
            raise CarrierIdDateLodoPhase2Error("CI DataModule requires immutable preflight and five-date aggregate paths")
        super().__init__(**kwargs)
        self.carrier_intervention = str(carrier_intervention)
        self.ci_arm = str(ci_arm).upper()
        self.ci_preflight_path = Path(ci_preflight_path).resolve()
        self.five_date_aggregate_path = Path(five_date_aggregate_path).resolve()
        self._ci_preflight: dict[str, Any] | None = None
        self._ci_preflight_sha256: str | None = None
        self._five_date_aggregate_sha256: str | None = None

    def _validate_preopen_receipts(self) -> None:
        """Verify every receipt/config/code gate before the parent can open an NWB."""

        aggregate_path, aggregate, aggregate_sha = _read_immutable_json(
            self.five_date_aggregate_path, schema=FIVE_DATE_AGGREGATE_SCHEMA, status=FIVE_DATE_AGGREGATE_STATUS,
        )
        _need(tuple(aggregate.get("required_outer_dates", ())) == tuple(CONFIRMATORY_DATES),
              "CI five-date aggregate date order drift")
        _need(aggregate.get("all_five_date_receipts_present_and_validated") is True,
              "CI five-date aggregate is incomplete")
        route = aggregate.get("route_prerequisite")
        _need(isinstance(route, Mapping) and route.get("status") == SOURCE_DATE_SCREEN_COMPLETE
              and route.get("automatic_route_selection") == "FORBIDDEN",
              "CI five-date aggregate has no source/date-screen-complete non-selector gate")
        preflight_path, preflight, preflight_sha = _read_immutable_json(
            self.ci_preflight_path, schema=CI_PREFLIGHT_SCHEMA, status=CI_PREFLIGHT_STATUS,
        )
        _need(preflight.get("outer_date") == str(self.hparams.outer_date), "CI preflight outer date drift")
        source_controls = preflight.get("source_controls")
        _need(isinstance(source_controls, Mapping) and source_controls.get("all_arms") == list(CI_ARMS)
              and source_controls.get("same_source_windows") is True
              and source_controls.get("same_source_schedule") is True
              and source_controls.get("same_source_normalizer") is True
              and source_controls.get("same_fresh_seed") == 42
              and source_controls.get("fixed_terminal_epoch_zero_based") == 49,
              "CI preflight does not bind the common source-control policy")
        model_row = preflight.get("fresh_models", {}).get(self.ci_arm)
        _need(isinstance(model_row, Mapping) and model_row.get("carrier_intervention") == self.carrier_intervention
              and model_row.get("fresh_seed") == 42 and model_row.get("carrier_columns_literal_zero_at_init") is True,
              "CI preflight does not bind this arm's initialization/control")
        configurations = preflight.get("configuration")
        _need(isinstance(configurations, Mapping) and isinstance(configurations.get(self.ci_arm), Mapping),
              "CI preflight lacks this arm's template configuration hash")
        declared_config = configurations[self.ci_arm]
        config_path = ROOT / "configs" / "experiment" / f"h1_carrierid_date_lodo_{self.ci_arm.lower().replace('-', '_')}.yaml"
        _need(config_path.is_file() and Path(str(declared_config.get("path", ""))).name == config_path.name
              and declared_config.get("sha256") == _sha256_file(config_path),
              "CI preflight config template hash drift")
        code_paths = {
            "data": ROOT / "src/data/h1_carrierid_date_lodo_ci.py",
            "model": ROOT / "src/models/h1_carrierid_date_lodo_ci_module.py",
            "component": ROOT / "src/models/components/h1_carrierid_ci_spint.py",
            "preflight": ROOT / "scripts/h1_carrierid_date_lodo_ci_preflight.py",
        }
        code = preflight.get("code_sha256")
        _need(isinstance(code, Mapping) and all(code.get(name) == _sha256_file(path) for name, path in code_paths.items()),
              "CI preflight code closure drift")
        self._ci_preflight, self._ci_preflight_sha256 = preflight, preflight_sha
        self._five_date_aggregate_sha256 = aggregate_sha
        self._validated_receipt_paths = {"ci_preflight": preflight_path, "five_date_aggregate": aggregate_path}

    def setup(self, stage: str | None = None) -> None:
        if self._setup_done:
            return
        # This must precede ``super().setup``: that method invokes the only
        # source record loader in this route.
        self._validate_preopen_receipts()
        super().setup(stage)
        _need(self._ci_preflight is not None and self._ci_preflight_sha256 is not None
              and self._five_date_aggregate_sha256 is not None, "CI receipt gate was not captured")
        actual_source = self.binding.manifest()
        _need(self._ci_preflight.get("source_binding_sha256") == canonical_sha256(actual_source)
              and self._ci_preflight.get("source_binding") == actual_source,
              "CI actual source binding differs from its preflight receipt")
        original = self.train_dataset
        self._base_source_binding_sha256 = canonical_sha256(actual_source)
        ci_dataset = H1CarrierIdDateLodoCiSourceDataset(self.binding, carrier_intervention=self.carrier_intervention)
        ci_sampler = H1CarrierIdDateLodoSchedule(ci_dataset, self.binding)
        _need(ci_dataset.window_indices == original.window_indices, "CI intervention changed source window index")
        _need(np.array_equal(ci_sampler.binding.calibration_schedule, self.train_batch_sampler.binding.calibration_schedule),
              "CI intervention changed fixed source calibration schedule")
        _need(ci_sampler.binding.batch_order_sha256 == self.train_batch_sampler.binding.batch_order_sha256,
              "CI intervention changed source batch order")
        self.train_dataset, self.train_batch_sampler = ci_dataset, ci_sampler

    def phase2_source_manifest(self) -> dict[str, Any]:
        base = super().phase2_source_manifest()
        dataset = self.train_dataset
        body = {
            **base,
            "schema": "h1_carrierid_date_lodo_ci_source_binding_v1",
            "carrier_intervention": self.carrier_intervention,
            "carrier_intervention_scope": "source_training_batch_only; C0 is literal-zero at model boundary",
            "effective_source_carriers_sha256": dataset.effective_source_carriers_sha256,
            "effective_source_carriers_shape": dataset.effective_source_carriers_shape,
            "effective_source_carriers_count": dataset.effective_source_carriers_count,
            "effective_source_carriers_nonidentity_all": dataset.effective_source_carriers_nonidentity_all,
            "shared_phase1_normalizer_sha256": self.binding.normalizer.normalizer_sha256,
            "shared_phase1_schedule_sha256": self.binding.calibration_schedule_sha256,
            "ci_arm": self.ci_arm,
            "phase2_base_source_binding_sha256": self._base_source_binding_sha256,
            "ci_preflight_path": str(self.ci_preflight_path),
            "ci_preflight_sha256": self.ci_preflight_sha256,
            "five_date_aggregate_path": str(self.five_date_aggregate_path),
            "five_date_aggregate_sha256": self.five_date_aggregate_sha256,
            "five_date_source_date_screen_complete": True,
            "five_date_automatic_route_selection_forbidden": True,
        }
        body["ci_source_binding_sha256"] = canonical_sha256(body)
        return body

    @property
    def ci_preflight_sha256(self) -> str:
        if self._ci_preflight_sha256 is None:
            raise RuntimeError("CI source receipt has not been validated")
        return self._ci_preflight_sha256

    @property
    def five_date_aggregate_sha256(self) -> str:
        if self._five_date_aggregate_sha256 is None:
            raise RuntimeError("CI five-date aggregate has not been validated")
        return self._five_date_aggregate_sha256
