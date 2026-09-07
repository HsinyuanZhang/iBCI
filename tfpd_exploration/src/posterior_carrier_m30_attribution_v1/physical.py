"""Deferred physical composition for the M30 attribution quick screen.

No function in this module is reached by the public CLI.  The physical route
is deliberately a narrow subclass composition over V3: it reuses V3's
descriptor-safe no-cache parser, strict model loading, Torch-only 5070Ti
attestation, and manual last-bin metric, while adding only three genuinely
new carrier/consumer forwards.  It never monkeypatches the frozen V3 or base
modules.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import attribution as contract


class PhysicalAttributionError(RuntimeError):
    """Fail closed during a future reviewed attribution evaluation."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise PhysicalAttributionError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_v3() -> tuple[Any, Any]:
    """Load V3 only on the deferred physical path, never from dry CLI code."""
    try:
        v3_contract = importlib.import_module("src.posterior_carrier_quick_screen_v1.quick_screen")
        v3_physical = importlib.import_module("src.posterior_carrier_quick_screen_v1.physical")
    except Exception as error:
        raise PhysicalAttributionError("closure-bound V3 quick-screen substrate cannot be imported") from error
    return v3_contract, v3_physical


def _raw_sha(runtime: Mapping[str, Any], value: Any) -> str:
    try:
        return runtime["core"].tensor_digest(value)
    except Exception as error:
        raise PhysicalAttributionError("cannot bind carrier tensor digest") from error


def _literal_digest(label: str) -> str:
    return _digest(label.encode("utf-8"))


def _load_v3_result(
    *, root: Path, v3_contract: Any, v3_physical: Any,
) -> tuple[Any, dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, str]]:
    """Descriptor-validate the completed V3 chain before target resolution.

    The V3 stage identity is intentionally rebuilt against the *immutable V3
    stage*, never against this fresh attribution stage.  This preserves the
    actual V3 closure without pretending its stage-bound identity is local.
    """
    v3_stage = Path(contract.V3_STAGE_ROOT)
    result = v3_stage / contract.V3_RESULT_RELATIVE
    try:
        closure = v3_contract.implementation_closure(v3_stage)
        identity = v3_contract.QuickScreenIdentity(closure=closure)
        v3_contract.validate_identity(identity)
    except Exception as error:
        raise PhysicalAttributionError("cannot rebuild V3 immutable stage identity") from error

    # V3's base physical module owns the exact held O_NOFOLLOW directory
    # implementation.  Use it rather than reopening names through pathlib.
    try:
        base, physical = v3_physical.QuickScreenPhysicalBackend(root=root, engineering_device=v3_physical.RemoteEngineeringDevice())._load_base()
        del base
        with physical.ImmutableDirectory.open(result) as held:
            expected_names = {
                "attempt.json", "attempt.json.sha256", "input_authority.json", "input_authority.json.sha256",
                "score.json", "score.json.sha256", "terminal.json", "terminal.json.sha256",
            }
            if set(held.names()) != expected_names:
                raise PhysicalAttributionError("V3 result topology must be exactly four immutable receipt pairs")
            attempt_item = held.read_pair("attempt.json", expected_sha256=contract.V3_ATTEMPT_SHA256)
            input_item = held.read_pair("input_authority.json", expected_sha256=contract.V3_INPUT_AUTHORITY_SHA256)
            score_item = held.read_pair("score.json", expected_sha256=contract.V3_SCORE_SHA256)
            terminal_item = held.read_pair("terminal.json", expected_sha256=contract.V3_TERMINAL_SHA256)
            attempt, inputs, score, terminal = (
                attempt_item.json_object(), input_item.json_object(), score_item.json_object(), terminal_item.json_object(),
            )
            held.reverify()
        v3_contract.validate_attempt_payload(attempt, identity=identity)
        checked_inputs = v3_contract.validate_input_authority_payload(inputs, identity=identity)
        checked_score = v3_contract.validate_score_payload(score, identity=identity, input_payload=checked_inputs)
        v3_contract.validate_terminal_payload(terminal, identity=identity, score_sha256=contract.V3_SCORE_SHA256)
    except PhysicalAttributionError:
        raise
    except Exception as error:
        raise PhysicalAttributionError("V3 predecessor body/sidecar/schema chain drift") from error

    matrix = checked_score.get("matrix")
    if not isinstance(matrix, list):
        raise PhysicalAttributionError("V3 score matrix is unavailable")
    reused: dict[str, str] = {}
    for surface in contract.SURFACES:
        expected = {
            contract.SEALED_OLS_POINT: (v3_contract.SEALED_POINT_MODE, 30),
            contract.POSTERIOR_MEAN_PRECISION: (v3_contract.POSTERIOR_MODE, 30),
        }
        for system, pair in expected.items():
            rows = [row for row in matrix if isinstance(row, Mapping) and row.get("cell", {}).get("surface") == surface
                    and row.get("cell", {}).get("mode") == pair[0] and row.get("cell", {}).get("budget") == pair[1]]
            if len(rows) != 1:
                raise PhysicalAttributionError("V3 M30 reusable-cell topology drift")
            reused[f"{surface}:{system}"] = _digest(_json(dict(rows[0])))
    if reused != contract.V3_REUSED_M30_CELL_SHA256S:
        raise PhysicalAttributionError("V3 M30 reusable-cell literal body digest drift")
    validation = {
        "schema": "posterior_carrier_m30_attribution_v3_replay_validation_v1",
        "v3_predecessor": contract.V3Evidence().payload(),
        "v3_identity_closure_sha256": closure.payload()["closure_sha256"],
        "v3_input_replayed_exactly": True,
        "v3_score_terminal_chain_valid": True,
        "reused_v3_m30_cell_sha256s": dict(contract.V3_REUSED_M30_CELL_SHA256S),
    }
    return identity, checked_inputs, checked_score, terminal, validation, dict(contract.V3_REUSED_M30_CELL_SHA256S)


def build_capturing_v3_backend_class(*, v3_physical: Any, physical: Any) -> type[Any]:
    """Add only raw OLS capture before the frozen normalizer is applied.

    The inherited implementation still computes the exact no-cache M4/M10/M30
    point carrier and its raw-axis proof.  We retain a detached M30 copy only
    so system C can feed the *same* raw OLS estimate through the posterior
    normalizer.  Algebraically inverting ``point_side`` is prohibited.
    """
    parent = v3_physical.build_torch_only_engineering_backend_class(
        physical=physical, device=v3_physical.RemoteEngineeringDevice(),
    )

    class CapturingV3Backend(parent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._attribution_raw_ols_m30: dict[str, Any] = {}
            self._attribution_raw_axis_proofs: dict[str, Mapping[str, object]] = {}

        def _raw_t4_by_budget(self, *, runtime: Mapping[str, Any], snapshot: Any, session: str, record: Any) -> Any:
            raw, proofs = super()._raw_t4_by_budget(
                runtime=runtime, snapshot=snapshot, session=session, record=record,
            )
            np = runtime["np"]
            value = np.ascontiguousarray(raw[30], dtype=np.float32).copy()
            proof = dict(proofs[30])
            if proof.get("raw_t4_sha256") != physical._array_digest(value):
                raise physical.PhysicalScoreError("attribution captured raw OLS M30/proof digest drift")
            self._attribution_raw_ols_m30[session] = value
            self._attribution_raw_axis_proofs[session] = proof
            return raw, proofs

    CapturingV3Backend.__name__ = "CapturingV3M30AttributionBackend"
    return CapturingV3Backend


class PhysicalM30AttributionBackend:
    """Future reviewed physical backend; construction itself is inert.

    `prepare` must be called only after this route's durable attempt exists.
    It validates the completed V3 result before creating either model or an
    evaluation path.  `resolve_inputs` performs exactly one inherited V3
    selected-3+3 materialization pass.  B/C/D are then newly forwarded; A/E
    are exact V3 replays after this equivalence check.
    """

    _BATCH_SIZE = 32

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root).absolute()
        self._v3_contract: Any | None = None
        self._v3_physical: Any | None = None
        self._quick: Any | None = None
        self._v3_identity: Any | None = None
        self._identity: contract.AttributionIdentity | None = None
        self._v3_input_payload: Mapping[str, object] | None = None
        self._v3_score_payload: Mapping[str, object] | None = None
        self._v3_validation: Mapping[str, object] | None = None
        self._reused_sha: Mapping[str, str] | None = None
        self._input_records: dict[tuple[str, str], Any] = {}
        self._closed = False

    def _require_prepared(self) -> tuple[Any, Any, Any, Any, contract.AttributionIdentity]:
        if (
            self._closed or self._v3_contract is None or self._v3_physical is None or self._quick is None
            or self._v3_identity is None or self._identity is None or self._v3_input_payload is None
            or self._v3_score_payload is None or self._v3_validation is None or self._reused_sha is None
        ):
            raise PhysicalAttributionError("M30 attribution physical backend is not prepared")
        return self._v3_contract, self._v3_physical, self._quick, self._v3_identity, self._identity

    def prepare(self, *, identity: contract.AttributionIdentity, flags: contract.RuntimeFlags) -> None:
        if self._closed or self._identity is not None:
            raise PhysicalAttributionError("M30 attribution physical backend may prepare exactly once")
        contract.validate_identity(identity)
        v3_contract, v3_physical = _load_v3()
        v3_identity, v3_inputs, v3_score, _v3_terminal, validation, reused = _load_v3_result(
            root=self._root, v3_contract=v3_contract, v3_physical=v3_physical,
        )

        # Compose V3's reviewed prepare sequence but choose the sole additive
        # subclass that captures raw OLS M30 before its normalizer.  No frozen
        # class/module/global is replaced.
        quick = v3_physical.QuickScreenPhysicalBackend(
            root=self._root, engineering_device=v3_physical.RemoteEngineeringDevice(),
        )
        try:
            base_identity, original_preflight, _authorization = quick._reload_base_evidence()
            base, physical = quick._load_base()
            preflight = v3_physical._validated_upstream_preflight_copy(
                base=base, identity=base_identity, original_preflight=original_preflight,
            )
            backend_class = build_capturing_v3_backend_class(v3_physical=v3_physical, physical=physical)
            backend = backend_class(root=self._root, preflight=preflight, contract=base)
            backend.prepare(identity=base_identity, flags=flags)
            runtime = backend._load_runtime()
            if tuple(runtime["torch"].cuda.get_device_capability(0)) != (12, 0):
                raise PhysicalAttributionError("remote device is not the reviewed 5070Ti capability")
        except PhysicalAttributionError:
            raise
        except Exception as error:
            raise PhysicalAttributionError("attribution V3-derived physical prepare failed") from error

        # Populate V3's own private state exactly as its selected-input helper
        # expects; retain this route's identity separately.
        quick._identity = v3_identity
        quick._base_identity = base_identity
        quick._base_preflight = preflight
        quick._upstream_formal_preflight_device = v3_contract.upstream_formal_preflight_device_payload(
            preflight["device_contract"],
        )
        quick._backend = backend
        self._v3_contract, self._v3_physical, self._quick = v3_contract, v3_physical, quick
        self._v3_identity, self._identity = v3_identity, identity
        self._v3_input_payload, self._v3_score_payload = v3_inputs, v3_score
        self._v3_validation, self._reused_sha = validation, reused
        flags.remote_initialized = True

    def resolve_inputs(self, *, identity: contract.AttributionIdentity, flags: contract.RuntimeFlags) -> contract.InputReplay:
        v3_contract, _v3_physical, quick, v3_identity, prepared_identity = self._require_prepared()
        if identity is not prepared_identity:
            raise PhysicalAttributionError("attribution identity object drift before input materialization")
        try:
            v3_authority = quick.resolve_inputs(identity=v3_identity, flags=flags)
            payload = v3_authority.payload(identity=v3_identity)
            checked = v3_contract.validate_input_authority_payload(payload, identity=v3_identity)
        except Exception as error:
            raise PhysicalAttributionError("V3 selected input materialization failed") from error
        if checked != self._v3_input_payload:
            raise PhysicalAttributionError("new selected input materialization differs from sealed V3 input authority")
        records = tuple(
            dict(row) for row in checked["records"]
        )
        replay = contract.InputReplay(records=records)
        payload = replay.payload(identity=prepared_identity)
        for row in records:
            self._input_records[(row["surface"], row["session"])] = quick._input_records[(row["surface"], row["session"])]
        return replay

    @staticmethod
    def _model_system(system: str) -> str:
        return "sealed_cell_d" if system in {contract.SEALED_OLS_POINT, contract.SEALED_POSTERIOR_MEAN} else "posterior"

    def _runtime(self) -> Mapping[str, Any]:
        _v3_contract, _v3_physical, quick, _v3_identity, _identity = self._require_prepared()
        backend = quick._backend
        return backend._load_runtime()

    def _session(self, cell: contract.AttributionCell, session: str) -> Any:
        value = self._input_records.get((cell.surface, session))
        if value is None:
            raise PhysicalAttributionError("attribution session is unavailable from the one shared V3 input pass")
        return value

    def _carrier_for_new_system(self, *, cell: contract.AttributionCell, session: Any) -> tuple[str, Any, Any]:
        """Return consumer, exact carrier, and raw pre-normalization T4 tensor."""
        runtime = self._runtime()
        torch, core = runtime["torch"], runtime["core"]
        _v3_contract, _v3_physical, quick, _v3_identity, _identity = self._require_prepared()
        backend = quick._backend
        if cell.system == contract.SEALED_POSTERIOR_MEAN:
            raw = session.posterior_raw[30].raw_t4.detach().to(
                device=backend._ordinary_point_normalizer()[0].device,
                dtype=backend._ordinary_point_normalizer()[0].dtype,
            ).clone()
            mean, std, _authority = backend._ordinary_point_normalizer()
            carrier = ((raw - mean) / std).detach().clone()
            if not bool(torch.isfinite(carrier).all().item()):
                raise PhysicalAttributionError("posterior mean under sealed normalizer is nonfinite")
            return "sealed_cell_d", carrier, raw

        raw_ols = backend._attribution_raw_ols_m30.get(session.session)
        axis = backend._attribution_raw_axis_proofs.get(session.session)
        if raw_ols is None or not isinstance(axis, Mapping):
            raise PhysicalAttributionError("exact raw OLS M30 cache is unavailable; normalized inversion is forbidden")
        np = runtime["np"]
        physical = quick._physical
        raw_ols_np = np.ascontiguousarray(raw_ols, dtype=np.float32)
        if axis.get("raw_t4_sha256") != physical._array_digest(raw_ols_np):
            raise PhysicalAttributionError("captured raw OLS M30 fails its raw-axis proof")
        view = session.posterior_view[30]
        raw = torch.as_tensor(raw_ols_np, device=view.normalized_t4.device, dtype=view.normalized_t4.dtype).detach().clone()
        if cell.system == contract.POSTERIOR_OLS_UNIFORM:
            normalized = backend._posterior_normalizer_view.normalize_raw(raw).detach().clone()
            zero_mask = raw.eq(0).all(dim=1)
            beta = torch.stack((raw[:, 0], raw[:, 1], raw[:, 3]), dim=1)
            carrier = core.PosteriorCarrierView(
                raw_beta=beta, raw_t4=raw, normalized_t4=normalized,
                credibility=torch.ones(raw.shape[0], device=raw.device, dtype=raw.dtype),
                zero_spike_mask=zero_mask, sampled=False, session_id=None, epoch=None,
                posterior_sha256=_literal_digest(f"ols-point-m30-as-posterior-consumer:{session.session}:{axis['body_sha256']}"),
                normalizer_authority_sha256=view.normalizer_authority_sha256,
            )
            return "posterior", carrier, raw
        if cell.system == contract.POSTERIOR_MEAN_UNIFORM:
            carrier = core.PosteriorCarrierView(
                raw_beta=view.raw_beta.detach().clone(), raw_t4=view.raw_t4.detach().clone(),
                normalized_t4=view.normalized_t4.detach().clone(),
                credibility=torch.ones_like(view.credibility), zero_spike_mask=view.zero_spike_mask.detach().clone(),
                sampled=False, session_id=None, epoch=None, posterior_sha256=view.posterior_sha256,
                normalizer_authority_sha256=view.normalizer_authority_sha256,
            )
            return "posterior", carrier, carrier.raw_t4
        raise PhysicalAttributionError("requested system is not a new attribution forward")

    def _forward(self, *, system: str, carrier: Any, neural: Any, calibration: Any) -> Any:
        """Mirror V3's eval/no-dropout forward with a supplied exact carrier."""
        runtime = self._runtime()
        torch, pop_robust = runtime["torch"], runtime["pop_robust"]
        _v3_contract, _v3_physical, quick, _v3_identity, _identity = self._require_prepared()
        backend = quick._backend
        model = backend._models.get(system)
        if model is None or model.training or not backend._all_gradients_none(model):
            raise PhysicalAttributionError("attribution model eval/gradient state drift")
        base = model if system == "sealed_cell_d" else model.cell_d
        calls = {"post_pool": 0}

        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            calls["post_pool"] += 1

        handle = base.id_encoder.post_pool.register_forward_hook(hook)
        try:
            with pop_robust.dynamic_dropout_recorder() as recorder:
                with torch.no_grad():
                    if torch.is_grad_enabled():
                        raise PhysicalAttributionError("attribution forward unexpectedly enables gradients")
                    if system == "sealed_cell_d":
                        side = carrier.unsqueeze(0).expand(neural.shape[0], -1, -1).detach().clone()
                        output, identity = model(neural, calib_trials=calibration, side_features=side)
                    else:
                        output, identity = model(neural, calib_trials_m30=calibration, carrier=carrier)
        finally:
            handle.remove()
        if (
            calls["post_pool"] <= 0 or not torch.is_tensor(identity) or not bool(torch.isfinite(identity).all().item())
            or recorder["uniform_calls"] != 0 or recorder["dropout_calls"] != []
            or not torch.is_tensor(output) or tuple(output.shape) != (neural.shape[0], 50, 2)
            or not bool(torch.isfinite(output).all().item())
        ):
            raise PhysicalAttributionError("attribution B3S/eval-no-dropout/output invariant drift")
        return output

    def _repeat_probe(self, *, cell: contract.AttributionCell, session: Any) -> bool:
        runtime = self._runtime()
        quick = self._require_prepared()[2]
        starts = tuple(int(item) for item in session.starts[:self._BATCH_SIZE])
        neural, _behavior, calibration = quick._backend._session_batch(session, starts)
        system, carrier, _raw = self._carrier_for_new_system(cell=cell, session=session)
        first = self._forward(system=system, carrier=carrier, neural=neural, calibration=calibration)
        second = self._forward(system=system, carrier=carrier, neural=neural, calibration=calibration)
        if not runtime["torch"].equal(first, second):
            raise PhysicalAttributionError("attribution repeated fixed-batch forward is not bitwise equal")
        return True

    def _carrier_digests(self, *, cell: contract.AttributionCell, session: Any) -> tuple[str, str, str]:
        runtime = self._runtime()
        if cell.system == contract.SEALED_OLS_POINT:
            raw = self._require_prepared()[2]._backend._attribution_raw_ols_m30[session.session]
            raw_tensor = runtime["torch"].as_tensor(raw, device=runtime["device"], dtype=runtime["torch"].float32)
            return _raw_sha(runtime, raw_tensor), _raw_sha(runtime, session.point_side[30]), _literal_digest("no-credibility-sealed")
        if cell.system == contract.POSTERIOR_MEAN_PRECISION:
            carrier = session.posterior_view[30]
            return _raw_sha(runtime, carrier.raw_t4), _raw_sha(runtime, carrier.normalized_t4), _raw_sha(runtime, carrier.credibility)
        _system, carrier, raw = self._carrier_for_new_system(cell=cell, session=session)
        raw_digest = _raw_sha(runtime, raw)
        if cell.system == contract.SEALED_POSTERIOR_MEAN:
            return raw_digest, _raw_sha(runtime, carrier), _literal_digest("no-credibility-sealed")
        return raw_digest, _raw_sha(runtime, carrier.normalized_t4), _raw_sha(runtime, carrier.credibility)

    def _reused_cell(self, *, cell: contract.AttributionCell, input_payload: Mapping[str, object]) -> contract.CellEvidence:
        _v3_contract, _v3_physical, quick, _v3_identity, identity = self._require_prepared()
        if cell.system not in contract.V3_REUSED_SYSTEMS:
            raise PhysicalAttributionError("only A/E may reuse validated V3 result rows")
        v3_mode = "sealed_cell_d_ols_point" if cell.system == contract.SEALED_OLS_POINT else "posterior_full_swa_aligned"
        rows = [row for row in self._v3_score_payload["matrix"] if row["cell"] == {
            "surface": cell.surface, "mode": v3_mode, "budget": 30,
        }]
        if len(rows) != 1:
            raise PhysicalAttributionError("V3 reused M30 score row disappeared")
        source = rows[0]
        records = {row["session"]: row for row in input_payload["records"] if row["surface"] == cell.surface}
        sessions: list[contract.SessionScore] = []
        raw: dict[str, str] = {}; normalized: dict[str, str] = {}; credibility: dict[str, str] = {}
        for source_row in source["sessions"]:
            session = self._session(cell, source_row["session"])
            raw_value, norm_value, cred_value = self._carrier_digests(cell=cell, session=session)
            raw[session.session], normalized[session.session], credibility[session.session] = raw_value, norm_value, cred_value
            sessions.append(contract.SessionScore(
                session=session.session, n_windows=source_row["n_windows"], r2=source_row["r2"],
                prediction_sha256=source_row["prediction_sha256"], input_record_sha256=_digest(_json(records[session.session])),
            ))
        system = self._model_system(cell.system)
        runtime = self._runtime(); model = quick._backend._models[system]
        state = runtime["arm_common"].state_sha256(model)
        if state != quick._backend._state_at_load[system] or model.training or not quick._backend._all_gradients_none(model):
            raise PhysicalAttributionError("V3-reused model is not unchanged/eval/no-gradient")
        return contract.CellEvidence(
            cell=cell,
            model_swa_sha256=("626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
                              if system == "sealed_cell_d" else "def27d8edffc0c6ed17dee80292dd1c1b628ba1d454dc214278caee814cf40fa"),
            sessions=tuple(sessions), input_replay_sha256=_digest(_json(input_payload)),
            model_state_before_sha256=state, model_state_after_sha256=state,
            carrier_raw_sha256s=raw, carrier_normalized_sha256s=normalized, credibility_sha256s=credibility,
            raw_before_normalization_verified=True, eval_mode=True, dropout_disabled=True, gradients_none=True,
            finite_outputs=True, repeated_fixed_batch_bitwise_equal=True, b3s_m30_recomputed=True,
            no_target_sampling=True, evidence_origin="V3_REUSED_VALIDATED",
            v3_reused_cell_sha256=self._reused_sha[f"{cell.surface}:{cell.system}"],
        )

    def _new_forward_cell(self, *, cell: contract.AttributionCell, input_payload: Mapping[str, object]) -> contract.CellEvidence:
        if cell.system not in contract.NEW_FORWARD_SYSTEMS:
            raise PhysicalAttributionError("unexpected attribution new-forward system")
        _v3_contract, v3_physical, quick, _v3_identity, _identity = self._require_prepared()
        backend = quick._backend; runtime = self._runtime(); torch, np = runtime["torch"], runtime["np"]
        system = self._model_system(cell.system); model = backend._models[system]
        if model.training or not backend._all_gradients_none(model):
            raise PhysicalAttributionError("new attribution forward begins with model not eval/no-gradient")
        state_before = runtime["arm_common"].state_sha256(model)
        if state_before != backend._state_at_load[system]:
            raise PhysicalAttributionError("new attribution forward model state drift before score")
        roster = contract.SELECTED_ROSTERS[cell.surface]
        records = {row["session"]: row for row in input_payload["records"] if row["surface"] == cell.surface}
        if tuple(records) != roster:
            raise PhysicalAttributionError("new attribution forward input roster/order drift")
        repeated = self._repeat_probe(cell=cell, session=self._session(cell, roster[0]))
        output_rows: list[contract.SessionScore] = []
        raw: dict[str, str] = {}; normalized: dict[str, str] = {}; credibility: dict[str, str] = {}
        for session_name in roster:
            session = self._session(cell, session_name)
            carrier_system, carrier, raw_value = self._carrier_for_new_system(cell=cell, session=session)
            if carrier_system != system:
                raise PhysicalAttributionError("new attribution carrier/consumer binding drift")
            raw[session_name] = _raw_sha(runtime, raw_value)
            if system == "sealed_cell_d":
                normalized[session_name] = _raw_sha(runtime, carrier)
                credibility[session_name] = _literal_digest("no-credibility-sealed")
            else:
                normalized[session_name] = _raw_sha(runtime, carrier.normalized_t4)
                credibility[session_name] = _raw_sha(runtime, carrier.credibility)
            predictions: list[Any] = []; targets: list[Any] = []; digest = hashlib.sha256()
            starts = tuple(int(item) for item in session.starts)
            for offset in range(0, len(starts), self._BATCH_SIZE):
                chunk = starts[offset:offset + self._BATCH_SIZE]
                neural, behavior, calibration = backend._session_batch(session, chunk)
                output = self._forward(system=system, carrier=carrier, neural=neural, calibration=calibration)
                valid = (behavior[:, -1, :] != -1.0).all(dim=-1)
                if not bool(valid.all().item()):
                    raise PhysicalAttributionError("new attribution score has invalid last-bin target")
                prediction = output[:, -1, :].detach().cpu().contiguous()
                target = behavior[:, -1, :].detach().cpu().contiguous()
                digest.update(prediction.numpy().tobytes())
                predictions.append(prediction); targets.append(target)
            prediction = torch.cat(predictions); target = torch.cat(targets)
            target_array = np.ascontiguousarray(target.numpy(), dtype=np.float32)
            mask_array = np.ascontiguousarray(np.all(target_array != -1.0, axis=1), dtype=np.uint8)
            record = records[session_name]
            if (
                target_array.shape != session.last_targets.shape or not np.array_equal(target_array, session.last_targets)
                or not np.array_equal(mask_array, session.last_valid_mask)
                or _digest(target_array.tobytes()) != record["last_bin_target_sha256"]
                or _digest(mask_array.tobytes()) != record["last_bin_valid_mask_sha256"]
                or int(mask_array.sum()) != record["last_bin_valid_count"]
            ):
                raise PhysicalAttributionError("new attribution governing target/mask/count binding drift")
            r2 = v3_physical.manual_variance_weighted_two_coordinate_r2(
                predictions=prediction, targets=target, torch=torch,
            )
            output_rows.append(contract.SessionScore(
                session=session_name, n_windows=len(starts), r2=float(r2), prediction_sha256=digest.hexdigest(),
                input_record_sha256=_digest(_json(record)),
            ))
        state_after = runtime["arm_common"].state_sha256(model)
        if state_after != state_before or model.training or not backend._all_gradients_none(model):
            raise PhysicalAttributionError("new attribution forward changed model state")
        return contract.CellEvidence(
            cell=cell,
            model_swa_sha256=("626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
                              if system == "sealed_cell_d" else "def27d8edffc0c6ed17dee80292dd1c1b628ba1d454dc214278caee814cf40fa"),
            sessions=tuple(output_rows), input_replay_sha256=_digest(_json(input_payload)),
            model_state_before_sha256=state_before, model_state_after_sha256=state_after,
            carrier_raw_sha256s=raw, carrier_normalized_sha256s=normalized, credibility_sha256s=credibility,
            raw_before_normalization_verified=True, eval_mode=True, dropout_disabled=True, gradients_none=True,
            finite_outputs=True, repeated_fixed_batch_bitwise_equal=repeated, b3s_m30_recomputed=True,
            no_target_sampling=True, evidence_origin="NEW_FORWARD",
        )

    def score_cell(self, *, cell: contract.AttributionCell, input_payload: Mapping[str, object],
                   flags: contract.RuntimeFlags) -> contract.CellEvidence:
        del flags
        contract.validate_input_replay_payload(input_payload, identity=self._require_prepared()[4])
        if cell.system in contract.V3_REUSED_SYSTEMS:
            return self._reused_cell(cell=cell, input_payload=input_payload)
        return self._new_forward_cell(cell=cell, input_payload=input_payload)

    def reverify_after_forwards(self, *, identity: contract.AttributionIdentity,
                                flags: contract.RuntimeFlags) -> contract.ImplementationClosure:
        _v3_contract, _v3_physical, quick, _v3_identity, prepared_identity = self._require_prepared()
        if identity is not prepared_identity:
            raise PhysicalAttributionError("attribution final identity object drift")
        backend = quick._backend; runtime = backend._load_runtime()
        for held in backend._held_assets:
            held.reverify()
        for held in backend._held_roots:
            held.reverify()
        for name, model in backend._models.items():
            if (
                runtime["arm_common"].state_sha256(model) != backend._state_at_load.get(name)
                or model.training or not backend._all_gradients_none(model)
            ):
                raise PhysicalAttributionError(f"attribution model state/eval drift after forwards: {name}")
        # Re-read all V3 durable evidence after forwards.  It must remain the
        # same completed chain before a successor receipt is licensed.
        v3_contract, v3_physical = _load_v3()
        _v3_identity, inputs, score, _terminal, validation, reused = _load_v3_result(
            root=self._root, v3_contract=v3_contract, v3_physical=v3_physical,
        )
        if inputs != self._v3_input_payload or score != self._v3_score_payload or validation != self._v3_validation or reused != self._reused_sha:
            raise PhysicalAttributionError("V3 predecessor changed during attribution forwards")
        final = contract.implementation_closure(self._root)
        if final.payload() != identity.closure.payload():
            raise PhysicalAttributionError("attribution closure changed during forwards")
        contract._validate_runtime_flags(flags, complete=True)
        return final

    def device_attestation(self) -> Mapping[str, object]:
        _v3_contract, _v3_physical, quick, _v3_identity, _identity = self._require_prepared()
        try:
            return contract.validate_v3_torch_only_device_payload(quick.device_attestation())
        except contract.AttributionError as error:
            raise PhysicalAttributionError("attribution physical device attestation drift") from error

    def v3_validation(self) -> Mapping[str, object]:
        self._require_prepared()
        if self._v3_validation is None:
            raise PhysicalAttributionError("V3 validation payload missing")
        return dict(self._v3_validation)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._quick is not None:
            self._quick.close()
        self._input_records.clear()
