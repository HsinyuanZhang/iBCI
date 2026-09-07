"""Deferred physical 3+3 PIRG performance-screen adapter.

This is deliberately a *composition* over the completed V3 quick-screen
substrate, not a second parser, model loader, or target-data route.  At a
future reviewed execution boundary it uses V3's descriptor-safe selected
3-within/3-external materialization, held assets, ordinary OLS point carriers,
M30 calibration tensors, and Torch-only 5070Ti device seam.  The only new
forward is the frozen Cell-D identity residual multiplied by the trained
one-scalar PIRG gate.  No module global is modified and no posterior mean,
posterior normalizer, posterior sample, or attention-logit bias can enter the
PIRG forward surface.

Nothing here is reached by the public dry CLI.  Importing this module is
inert with respect to data, checkpoints, CUDA, remote hosts, result roots, and
receipt publication.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import plan
from . import score as contract
from .core import PosteriorIdentityResidualGate
from . import train


class PhysicalPIRGScoreError(contract.PIRGScoreError):
    """Fail closed in the reviewed physical PIRG score route."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise PhysicalPIRGScoreError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_attribution() -> tuple[Any, Any]:
    """Deferred import of the already reviewed V3 private composition seam."""
    try:
        attribution_contract = importlib.import_module("src.posterior_carrier_m30_attribution_v1.attribution")
        attribution_physical = importlib.import_module("src.posterior_carrier_m30_attribution_v1.physical")
    except Exception as error:
        raise PhysicalPIRGScoreError("closure-bound V3 composition seam cannot be imported") from error
    return attribution_contract, attribution_physical


def _lazy_safe_alpha_payload(*, torch: Any, body: bytes, expected_sha256: str) -> Mapping[str, Any]:
    """Load the one-scalar final artifact with ``weights_only=True`` only."""
    if _digest(body) != expected_sha256:
        raise PhysicalPIRGScoreError("PIRG final-alpha body SHA drift")
    try:
        payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PhysicalPIRGScoreError("PIRG final-alpha weights-only load failed") from error
    if not isinstance(payload, Mapping):
        raise PhysicalPIRGScoreError("PIRG final-alpha payload root drift")
    alpha = payload.get("alpha")
    if (
        payload.get("schema") != "posterior_identity_residual_gate_final_alpha_v1"
        or payload.get("base_swa_sha256") != plan.SEALED_CELL_D_SWA_SHA256
        or not torch.is_tensor(alpha)
        or tuple(alpha.shape) != ()
        or alpha.dtype != torch.float32
        or not bool(torch.isfinite(alpha).all().item())
    ):
        raise PhysicalPIRGScoreError("PIRG final-alpha schema/scalar boundary drift")
    return payload


class PIRGForwardSeam:
    """Small local Cell-D/PIRG forward seam over one V3 materialized session.

    ``session.point_side[budget]`` is the sealed ordinary normalized OLS
    carrier.  The only posterior-derived field read is
    ``session.posterior_view[budget].credibility``.  In particular, this class
    has no API taking a posterior mean or normalized posterior carrier.
    """

    def __init__(self, *, sealed_cell_d: Any, pirg: PosteriorIdentityResidualGate,
                 torch: Any, pop_robust: Any) -> None:
        self.sealed_cell_d = sealed_cell_d
        self.pirg = pirg
        self.torch = torch
        self.pop_robust = pop_robust

    def _require_eval_no_grad(self, model: Any, *, label: str) -> None:
        if model.training:
            raise PhysicalPIRGScoreError(f"{label} unexpectedly remains in training mode")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise PhysicalPIRGScoreError(f"{label} gradient state is not empty before score forward")

    @staticmethod
    def _side_for_batch(*, carrier: Any, neural: Any) -> Any:
        if getattr(carrier, "ndim", None) != 2 or tuple(carrier.shape[-1:]) != (4,):
            raise PhysicalPIRGScoreError("PIRG score ordinary point carrier must be [N,4]")
        if int(carrier.shape[0]) != int(neural.shape[-1]):
            raise PhysicalPIRGScoreError("PIRG score ordinary point carrier/unit axis drift")
        return carrier.unsqueeze(0).expand(neural.shape[0], -1, -1).detach().clone()

    def _run_sealed(self, *, neural: Any, calibration: Any, carrier: Any) -> tuple[Any, Any]:
        self._require_eval_no_grad(self.sealed_cell_d, label="sealed Cell-D")
        calls = {"post_pool": 0}

        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            calls["post_pool"] += 1

        handle = self.sealed_cell_d.id_encoder.post_pool.register_forward_hook(hook)
        try:
            with self.pop_robust.dynamic_dropout_recorder() as recorder:
                with self.torch.no_grad():
                    if self.torch.is_grad_enabled():
                        raise PhysicalPIRGScoreError("sealed Cell-D score forward enabled gradients")
                    output, identity = self.sealed_cell_d(
                        neural, calib_trials=calibration,
                        side_features=self._side_for_batch(carrier=carrier, neural=neural),
                    )
        finally:
            handle.remove()
        if (
            calls["post_pool"] <= 0 or recorder["uniform_calls"] != 0 or recorder["dropout_calls"] != []
            or not self.torch.is_tensor(identity) or not bool(self.torch.isfinite(identity).all().item())
            or not self.torch.is_tensor(output) or tuple(output.shape) != (neural.shape[0], 50, 2)
            or not bool(self.torch.isfinite(output).all().item())
        ):
            raise PhysicalPIRGScoreError("sealed Cell-D B3S/eval/no-dropout/output invariant drift")
        return output, identity

    def _run_pirg(self, *, neural: Any, calibration: Any, carrier: Any, credibility: Any) -> tuple[Any, Any, Any, Any]:
        self._require_eval_no_grad(self.pirg, label="PIRG")
        calls = {"post_pool": 0}

        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            calls["post_pool"] += 1

        handle = self.pirg.cell_d.id_encoder.post_pool.register_forward_hook(hook)
        try:
            with self.pop_robust.dynamic_dropout_recorder() as recorder:
                with self.torch.no_grad():
                    if self.torch.is_grad_enabled():
                        raise PhysicalPIRGScoreError("PIRG score forward enabled gradients")
                    output, identity, gate, z = self.pirg(
                        neural, calib_trials_m30=calibration,
                        ordinary_ols_side_features=self._side_for_batch(carrier=carrier, neural=neural),
                        directional_credibility=credibility,
                    )
        finally:
            handle.remove()
        if (
            calls["post_pool"] <= 0 or recorder["uniform_calls"] != 0 or recorder["dropout_calls"] != []
            or not self.torch.is_tensor(identity) or not self.torch.is_tensor(gate) or not self.torch.is_tensor(z)
            or not bool(self.torch.isfinite(identity).all().item())
            or not bool(self.torch.isfinite(gate).all().item()) or not bool(self.torch.isfinite(z).all().item())
            or not self.torch.is_tensor(output) or tuple(output.shape) != (neural.shape[0], 50, 2)
            or not bool(self.torch.isfinite(output).all().item())
        ):
            raise PhysicalPIRGScoreError("PIRG B3S/identity-gate/eval/no-dropout/output invariant drift")
        return output, identity, gate, z

    def forward_pair(self, *, session: Any, budget: int, neural: Any, calibration: Any) -> tuple[Any, Any, Any]:
        """Forward Cell-D and PIRG using the same exact point-side input.

        Accessing ``credibility`` is deliberately separated from obtaining the
        point carrier.  No posterior mean/normalizer/sampled tensor is read.
        """
        if budget not in plan.BUDGETS:
            raise PhysicalPIRGScoreError("PIRG score budget drift")
        try:
            carrier = session.point_side[budget]
            credibility = session.posterior_view[budget].credibility
        except (AttributeError, KeyError) as error:
            raise PhysicalPIRGScoreError("V3 physical session lacks point-side or credibility control") from error
        if (
            getattr(credibility, "ndim", None) != 1
            or int(credibility.shape[0]) != int(neural.shape[-1])
            or not bool(self.torch.isfinite(credibility).all().item())
        ):
            raise PhysicalPIRGScoreError("PIRG score credibility/unit axis drift")
        sealed, _sealed_identity = self._run_sealed(neural=neural, calibration=calibration, carrier=carrier)
        pirg, _pirg_identity, gate, _z = self._run_pirg(
            neural=neural, calibration=calibration, carrier=carrier, credibility=credibility,
        )
        return sealed, pirg, gate


class PhysicalPIRGQuickScoreBackend:
    """Future reviewed 3+3 physical scorer, inert until lifecycle ``prepare``.

    The base V3 quick backend remains responsible for descriptor-safe parser,
    held input assets, model/device attestation, and exact target materializer.
    This adapter holds a separate strict Cell-D copy plus final alpha only.
    """

    _BATCH_SIZE = 32

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root).absolute()
        self._attr_contract: Any | None = None
        self._attr_physical: Any | None = None
        self._v3_stage_root: Path | None = None
        self._v3_contract: Any | None = None
        self._v3_physical: Any | None = None
        self._v3_identity: Any | None = None
        self._v3_inputs: Mapping[str, object] | None = None
        self._v3_score: Mapping[str, object] | None = None
        self._quick: Any | None = None
        self._v3_flags: Any | None = None
        self._identity: contract.PIRGScoreIdentity | None = None
        self._input_sessions: dict[tuple[str, str], Any] = {}
        self._input_payload: Mapping[str, object] | None = None
        self._pirg: PosteriorIdentityResidualGate | None = None
        self._seam: PIRGForwardSeam | None = None
        self._pirg_state_at_load: str | None = None
        self._source_terminal: Mapping[str, object] | None = None
        self._source_authority: Mapping[str, object] | None = None
        self._closed = False

    def _require_prepared(self) -> tuple[Any, Any, Any, Any, contract.PIRGScoreIdentity, PIRGForwardSeam]:
        if (
            self._closed or self._quick is None or self._v3_contract is None or self._v3_physical is None
            or self._v3_stage_root is None
            or self._v3_identity is None or self._identity is None or self._pirg is None or self._seam is None
            or self._source_terminal is None or self._source_authority is None
        ):
            raise PhysicalPIRGScoreError("PIRG quick physical backend is not prepared")
        return self._v3_contract, self._v3_physical, self._quick, self._v3_identity, self._identity, self._seam

    @staticmethod
    def _wrapper_state_digest(*, arm_common: Any, core: Any, wrapper: PosteriorIdentityResidualGate) -> str:
        base = arm_common.state_sha256(wrapper.cell_d)
        alpha = core.tensor_digest(wrapper.alpha.detach())
        return _digest(_json({"base_cell_d_state_sha256": base, "alpha_tensor_sha256": alpha}))

    def _load_completed_source_artifact(self, *, torch: Any, physical: Any,
                                        identity: contract.PIRGScoreIdentity) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, Any]]:
        """Descriptor-validate the completed three-epoch PIRG source chain."""
        directory = self._root / train.RESULT_ROOT_RELATIVE
        names = {
            "attempt.json", "attempt.json.sha256", "source_authority.json", "source_authority.json.sha256",
            "epoch-00.json", "epoch-00.json.sha256", "epoch-01.json", "epoch-01.json.sha256",
            "epoch-02.json", "epoch-02.json.sha256", "final_alpha.pt", "final_alpha.pt.sha256",
            "terminal.json", "terminal.json.sha256",
        }
        try:
            with physical.ImmutableDirectory.open(directory) as held:
                if set(held.names()) != names:
                    raise PhysicalPIRGScoreError("PIRG source result topology must be terminal-only exact pairs")
                attempt_item = held.read_pair("attempt.json")
                attempt = attempt_item.json_object()
                authority_item = held.read_pair("source_authority.json", expected_sha256=identity.source_authority_sha256)
                authority = authority_item.json_object()
                epoch_items = [held.read_pair(f"epoch-{epoch:02d}.json") for epoch in range(plan.EPOCHS)]
                epochs = [item.json_object() for item in epoch_items]
                alpha_item = held.read_pair("final_alpha.pt", expected_sha256=identity.final_alpha_sha256)
                terminal_item = held.read_pair("terminal.json", expected_sha256=identity.source_terminal_sha256)
                terminal = terminal_item.json_object()
                held.reverify()
        except PhysicalPIRGScoreError:
            raise
        except Exception as error:
            raise PhysicalPIRGScoreError("PIRG source result descriptor/sidecar read failed") from error

        try:
            checked_attempt_identity = train.validate_identity(attempt["identity"])
            if attempt != train._attempt_payload(train.PIRGIdentity(
                train.ImplementationClosure(dict(checked_attempt_identity["closure"]["sha256_by_path"])),
            )):
                raise PhysicalPIRGScoreError("PIRG source attempt schema/boundary drift")
            if attempt.get("identity") != terminal.get("identity") or authority.get("identity") != terminal.get("identity"):
                raise PhysicalPIRGScoreError("PIRG source identity chain drift")
            if terminal.get("schema") != "posterior_identity_residual_gate_terminal_v1" or terminal.get("status") != "PIRG_SOURCE_TRAINING_COMPLETE":
                raise PhysicalPIRGScoreError("PIRG source terminal schema/status drift")
            if terminal.get("source_authority_sha256") != identity.source_authority_sha256:
                raise PhysicalPIRGScoreError("PIRG source terminal/source-authority binding drift")
            if terminal.get("attempt_sha256") != attempt_item.sha256:
                raise PhysicalPIRGScoreError("PIRG source terminal/attempt binding drift")
            if terminal.get("final_alpha_sha256") != identity.final_alpha_sha256:
                raise PhysicalPIRGScoreError("PIRG source terminal/final-alpha binding drift")
            if terminal.get("optimizer_steps") != plan.EPOCHS * train.SOURCE_STEPS_PER_EPOCH:
                raise PhysicalPIRGScoreError("PIRG source terminal optimizer boundary drift")
            if terminal.get("launch_closure") != terminal.get("final_closure"):
                raise PhysicalPIRGScoreError("PIRG source launch/final closure drift")
            if terminal.get("launch_closure") != terminal["identity"].get("closure"):
                raise PhysicalPIRGScoreError("PIRG source terminal/identity closure binding drift")
            if train.implementation_closure(self._root).payload() != terminal["identity"].get("closure"):
                raise PhysicalPIRGScoreError("PIRG current score stage does not reproduce source training closure")
            if terminal.get("target_opened") is not False or terminal.get("within_opened") is not False or terminal.get("external_opened") is not False:
                raise PhysicalPIRGScoreError("PIRG source terminal crossed an evaluation boundary")
            train._validate_source_authority(authority, identity=train.PIRGIdentity(
                train.ImplementationClosure(dict(terminal["identity"]["closure"]["sha256_by_path"])),
            ))
            if len(epochs) != plan.EPOCHS or any(epoch.get("epoch") != index for index, epoch in enumerate(epochs)):
                raise PhysicalPIRGScoreError("PIRG source epoch receipt topology drift")
            if terminal.get("epochs") != [
                {"epoch": epoch, "sha256": epoch_items[epoch].sha256}
                for epoch in range(plan.EPOCHS)
            ]:
                raise PhysicalPIRGScoreError("PIRG source terminal/epoch receipt binding drift")
            if any(epoch.get("optimizer_steps") != train.SOURCE_STEPS_PER_EPOCH for epoch in epochs):
                raise PhysicalPIRGScoreError("PIRG source epoch step boundary drift")
            for index, epoch in enumerate(epochs):
                if not isinstance(epoch, Mapping) or set(epoch) != {
                    "epoch", "budget_schedule", "alpha_before", "alpha_after", "loss_first", "loss_last",
                    "loss_min", "loss_max", "optimizer_steps", "only_alpha_gradient", "alpha_gradient_nonzero",
                    "finite_model", "finite_optimizer", "gate_stats_by_budget", "throughput_steps_per_second",
                    "cache", "cumulative_optimizer_steps",
                }:
                    raise PhysicalPIRGScoreError("PIRG source epoch receipt schema drift")
                summary = train.EpochSummary(
                    epoch=epoch["epoch"], budget_schedule=epoch["budget_schedule"],
                    alpha_before=epoch["alpha_before"], alpha_after=epoch["alpha_after"],
                    loss_first=epoch["loss_first"], loss_last=epoch["loss_last"], loss_min=epoch["loss_min"],
                    loss_max=epoch["loss_max"], optimizer_steps=epoch["optimizer_steps"],
                    only_alpha_gradient=epoch["only_alpha_gradient"], alpha_gradient_nonzero=epoch["alpha_gradient_nonzero"],
                    finite_model=epoch["finite_model"], finite_optimizer=epoch["finite_optimizer"],
                    gate_stats_by_budget=epoch["gate_stats_by_budget"],
                    throughput_steps_per_second=epoch["throughput_steps_per_second"], cache=epoch["cache"],
                ).payload()
                if summary != {key: value for key, value in epoch.items() if key != "cumulative_optimizer_steps"}:
                    raise PhysicalPIRGScoreError("PIRG source epoch receipt semantic drift")
                if epoch["cumulative_optimizer_steps"] != (index + 1) * train.SOURCE_STEPS_PER_EPOCH:
                    raise PhysicalPIRGScoreError("PIRG source epoch cumulative-step binding drift")
        except PhysicalPIRGScoreError:
            raise
        except Exception as error:
            raise PhysicalPIRGScoreError("PIRG source receipt semantic validation failed") from error

        alpha_payload = _lazy_safe_alpha_payload(torch=torch, body=alpha_item.body, expected_sha256=identity.final_alpha_sha256)
        manifest = terminal.get("final_alpha_manifest")
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "schema", "artifact_sha256", "alpha", "base_swa_sha256", "base_state_sha256", "cpu_reload_equal",
        } or manifest.get("schema") != "posterior_identity_residual_gate_final_alpha_manifest_v1" or manifest.get("artifact_sha256") != identity.final_alpha_sha256:
            raise PhysicalPIRGScoreError("PIRG source final-alpha manifest drift")
        if (
            alpha_payload.get("base_swa_sha256") != plan.SEALED_CELL_D_SWA_SHA256
            or alpha_payload.get("base_state_sha256") != manifest.get("base_state_sha256")
            or manifest.get("base_swa_sha256") != plan.SEALED_CELL_D_SWA_SHA256
            or manifest.get("cpu_reload_equal") is not True
            or not isinstance(manifest.get("alpha"), (int, float))
            or float(alpha_payload["alpha"].item()) != float(manifest["alpha"])
        ):
            raise PhysicalPIRGScoreError("PIRG source final-alpha/base-state binding drift")
        return terminal, authority, alpha_payload

    def prepare(self, *, identity: contract.PIRGScoreIdentity) -> None:
        if self._closed or self._identity is not None:
            raise PhysicalPIRGScoreError("PIRG quick physical backend may prepare exactly once")
        contract.validate_score_identity(identity)
        live = contract.score_implementation_closure(self._root).payload()
        if live != identity.closure.payload():
            raise PhysicalPIRGScoreError("PIRG quick score closure drift before V3/data/model prepare")
        attr_contract, attr_physical = _load_attribution()
        try:
            v3_contract, v3_physical = attr_physical._load_v3()
            # The frozen V3 substrate remains at its own completed stage.
            # PIRG's fresh stage holds only its source alpha/result bytes; it
            # must not reconstruct V3's formal preflight or re-stage the six
            # evaluation NWBs under a new pathname.
            v3_stage_root = Path(attr_contract.V3_STAGE_ROOT)
            if str(v3_stage_root) != contract.V3_STAGE_ROOT:
                raise PhysicalPIRGScoreError("PIRG score V3 stage-root provenance drift")
            v3_identity, v3_inputs, v3_score, _v3_terminal, _validation, _reused = attr_physical._load_v3_result(
                root=v3_stage_root, v3_contract=v3_contract, v3_physical=v3_physical,
            )
            quick = v3_physical.QuickScreenPhysicalBackend(
                root=v3_stage_root, engineering_device=v3_physical.RemoteEngineeringDevice(),
            )
            flags = v3_contract.RuntimeFlags()
            quick.prepare(identity=v3_identity, flags=flags)
            backend = quick._backend
            runtime = backend._load_runtime()
            terminal, authority, alpha_payload = self._load_completed_source_artifact(
                torch=runtime["torch"], physical=quick._physical, identity=identity,
            )
            sealed = backend._models.get("sealed_cell_d")
            if sealed is None or backend._sealed is None:
                raise PhysicalPIRGScoreError("V3 sealed Cell-D model/material is unavailable")
            sealed_payload = backend._safe_weights_only_load(backend._sealed.swa_body, label="sealed Cell-D SWA")
            sealed_state = sealed_payload.get("state_dict", sealed_payload.get("state"))
            if not isinstance(sealed_state, Mapping):
                raise PhysicalPIRGScoreError("V3 sealed Cell-D state payload drift")
            # The frozen base helper strict-loads fresh Cell-D topology on the
            # exact V3 device.  The second copy makes baseline/PIRG state
            # comparisons independent and leaves the V3 sealed model intact.
            pirg_base = backend._fresh_base_cell_d(sealed_state, label="PIRG sealed Cell-D base", place_on_device=True)
            wrapper = PosteriorIdentityResidualGate(pirg_base)
            wrapper.to(runtime["device"])
            with runtime["torch"].no_grad():
                wrapper.alpha.copy_(alpha_payload["alpha"].to(device=runtime["device"], dtype=runtime["torch"].float32))
            wrapper.eval()
            if alpha_payload.get("preservation") != wrapper.preservation().payload():
                raise PhysicalPIRGScoreError("PIRG final-alpha preservation topology drift")
            if wrapper.training or any(parameter.grad is not None for parameter in wrapper.parameters()):
                raise PhysicalPIRGScoreError("PIRG model eval/gradient initialization drift")
            if runtime["arm_common"].state_sha256(wrapper.cell_d) != backend._state_at_load.get("sealed_cell_d"):
                raise PhysicalPIRGScoreError("PIRG strict Cell-D base state differs from V3 sealed baseline")
            if alpha_payload.get("base_state_sha256") != backend._state_at_load.get("sealed_cell_d"):
                raise PhysicalPIRGScoreError("PIRG alpha artifact binds a different sealed Cell-D base state")
            seam = PIRGForwardSeam(sealed_cell_d=sealed, pirg=wrapper, torch=runtime["torch"], pop_robust=runtime["pop_robust"])
        except PhysicalPIRGScoreError:
            raise
        except Exception as error:
            raise PhysicalPIRGScoreError("PIRG V3-composed physical prepare failed") from error

        self._attr_contract, self._attr_physical, self._v3_stage_root = attr_contract, attr_physical, v3_stage_root
        self._v3_contract, self._v3_physical = v3_contract, v3_physical
        self._v3_identity, self._v3_inputs, self._v3_score = v3_identity, v3_inputs, v3_score
        self._quick, self._v3_flags, self._identity = quick, flags, identity
        self._pirg, self._seam = wrapper, seam
        self._source_terminal, self._source_authority = terminal, authority
        self._pirg_state_at_load = self._wrapper_state_digest(
            arm_common=runtime["arm_common"], core=runtime["core"], wrapper=wrapper,
        )

    def resolve_inputs(self, *, identity: contract.PIRGScoreIdentity) -> contract.InputAuthority:
        v3_contract, _v3_physical, quick, v3_identity, prepared_identity, _seam = self._require_prepared()
        if identity is not prepared_identity:
            raise PhysicalPIRGScoreError("PIRG score identity object drift before input materialization")
        try:
            authority = quick.resolve_inputs(identity=v3_identity, flags=self._v3_flags)
            v3_payload = authority.payload(identity=v3_identity)
            checked = v3_contract.validate_input_authority_payload(v3_payload, identity=v3_identity)
        except Exception as error:
            raise PhysicalPIRGScoreError("PIRG selected V3 input materialization failed") from error
        if checked != self._v3_inputs:
            raise PhysicalPIRGScoreError("PIRG selected materialization differs from immutable V3 input authority")
        backend = quick._backend
        runtime = backend._load_runtime()
        records: dict[str, list[dict[str, object]]] = {surface: [] for surface in contract.SURFACES}
        for source_row in checked["records"]:
            surface, session_name = source_row["surface"], source_row["session"]
            if surface not in contract.SURFACES or session_name not in contract.FIXED_SESSIONS[surface]:
                raise PhysicalPIRGScoreError("PIRG V3 selected input surface/session drift")
            session = quick._input_records.get((surface, session_name))
            if session is None:
                raise PhysicalPIRGScoreError("PIRG V3 physical session cache is missing")
            base_record = session.input_record.payload()
            point = {str(budget): runtime["core"].tensor_digest(session.point_side[budget]) for budget in plan.BUDGETS}
            credibility = {str(budget): runtime["core"].tensor_digest(session.posterior_view[budget].credibility) for budget in plan.BUDGETS}
            prefixes = base_record.get("matched_prefix_row_ids_sha256s")
            if not isinstance(prefixes, Mapping) or set(prefixes) != {"30", "10", "4"}:
                raise PhysicalPIRGScoreError("PIRG V3 source prefix M4/M10/M30 authority is incomplete")
            records[surface].append({
                "session": session_name,
                "n_windows": int(base_record["n_windows"]),
                "input_sha256": str(source_row["materialization_sha256"]),
                "last_bin_target_sha256": str(base_record["last_bin_target_sha256"]),
                "last_bin_mask_sha256": str(base_record["last_bin_valid_mask_sha256"]),
                "point_side_sha256s": point,
                "directional_credibility_sha256s": credibility,
                "prefix_row_ids_sha256s": {str(key): str(prefixes[str(key)]) for key in ("30", "10", "4")},
            })
            self._input_sessions[(surface, session_name)] = session
        result = contract.InputAuthority(records).payload(identity=identity)
        self._input_payload = result
        return contract.InputAuthority(records)

    def _session(self, *, surface: str, session_name: str) -> Any:
        session = self._input_sessions.get((surface, session_name))
        if session is None:
            raise PhysicalPIRGScoreError("PIRG score physical session unavailable after shared V3 materialization")
        return session

    @staticmethod
    def _validate_budget_control_binding(*, session: Any, record: Mapping[str, object], budget: int, core: Any) -> None:
        """Bind the exact OLS and credibility controls before every cell.

        A V3 materialized record retains all M4/M10/M30 tensors even though
        V3 itself scored only M4/M30.  This check prevents a new M10 result
        from silently using another prefix or a carrier recomputed after the
        shared input authority was written.
        """
        key = str(budget)
        try:
            point = core.tensor_digest(session.point_side[budget])
            credibility = core.tensor_digest(session.posterior_view[budget].credibility)
            source_prefixes = session.input_record.payload()["matched_prefix_row_ids_sha256s"]
            expected_point = record["point_side_sha256s"][key]
            expected_credibility = record["directional_credibility_sha256s"][key]
            expected_prefix = record["prefix_row_ids_sha256s"][key]
        except (AttributeError, KeyError, TypeError) as error:
            raise PhysicalPIRGScoreError("PIRG score M4/M10/M30 control provenance is incomplete") from error
        if point != expected_point or credibility != expected_credibility or source_prefixes.get(key) != expected_prefix:
            raise PhysicalPIRGScoreError("PIRG score point/credibility/prefix control binding drift")

    def _repeat_probe(self, *, session: Any, budget: int) -> bool:
        _v3_contract, _v3_physical, quick, _v3_identity, _identity, seam = self._require_prepared()
        starts = tuple(int(item) for item in session.starts[:self._BATCH_SIZE])
        neural, _behavior, calibration = quick._backend._session_batch(session, starts)
        first_sealed, first_pirg, _first_gate = seam.forward_pair(
            session=session, budget=budget, neural=neural, calibration=calibration,
        )
        second_sealed, second_pirg, _second_gate = seam.forward_pair(
            session=session, budget=budget, neural=neural, calibration=calibration,
        )
        if not quick._backend._load_runtime()["torch"].equal(first_sealed, second_sealed):
            raise PhysicalPIRGScoreError("PIRG repeated sealed Cell-D forward is not bitwise equal")
        if not quick._backend._load_runtime()["torch"].equal(first_pirg, second_pirg):
            raise PhysicalPIRGScoreError("PIRG repeated gated forward is not bitwise equal")
        return True

    def score_cell(self, *, cell: contract.ScoreCell, input_payload: Mapping[str, object]) -> contract.CellEvidence:
        _v3_contract, _v3_physical, quick, _v3_identity, identity, seam = self._require_prepared()
        contract.validate_input_authority(input_payload, identity=identity)
        if self._input_payload != input_payload:
            raise PhysicalPIRGScoreError("PIRG score cell received a nonshared input authority")
        runtime = quick._backend._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        if cell.mode == contract.BASELINE_MODE:
            model = quick._backend._models["sealed_cell_d"]
            state_before = runtime["arm_common"].state_sha256(model)
            expected_state = quick._backend._state_at_load.get("sealed_cell_d")
            model_artifact = plan.SEALED_CELL_D_SWA_SHA256
        elif cell.mode == contract.PIRG_MODE:
            model = self._pirg
            state_before = self._wrapper_state_digest(arm_common=runtime["arm_common"], core=runtime["core"], wrapper=model)
            expected_state = self._pirg_state_at_load
            model_artifact = identity.final_alpha_sha256
        else:
            raise PhysicalPIRGScoreError("PIRG score mode drift")
        if model is None or state_before != expected_state or model.training or any(parameter.grad is not None for parameter in model.parameters()):
            raise PhysicalPIRGScoreError("PIRG score model state/eval/gradient invariant drift")
        records = {row["session"]: row for row in input_payload["surfaces"][cell.surface]}
        if tuple(records) != contract.FIXED_SESSIONS[cell.surface]:
            raise PhysicalPIRGScoreError("PIRG score selected session order drift")
        repeated = self._repeat_probe(session=self._session(surface=cell.surface, session_name=contract.FIXED_SESSIONS[cell.surface][0]), budget=cell.budget)
        rows: list[contract.SessionScore] = []
        for session_name in contract.FIXED_SESSIONS[cell.surface]:
            session = self._session(surface=cell.surface, session_name=session_name)
            self._validate_budget_control_binding(
                session=session, record=records[session_name], budget=cell.budget, core=runtime["core"],
            )
            prediction_chunks: list[Any] = []
            target_chunks: list[Any] = []
            digest = hashlib.sha256()
            for offset in range(0, len(session.starts), self._BATCH_SIZE):
                starts = tuple(int(item) for item in session.starts[offset:offset + self._BATCH_SIZE])
                neural, behavior, calibration = quick._backend._session_batch(session, starts)
                sealed, pirg, _gate = seam.forward_pair(
                    session=session, budget=cell.budget, neural=neural, calibration=calibration,
                )
                output = sealed if cell.mode == contract.BASELINE_MODE else pirg
                valid = (behavior[:, -1, :] != -1.0).all(dim=-1)
                if not bool(valid.all().item()):
                    raise PhysicalPIRGScoreError("PIRG score contains an invalid last-bin target")
                prediction = output[:, -1, :].detach().cpu().contiguous()
                target = behavior[:, -1, :].detach().cpu().contiguous()
                digest.update(prediction.numpy().tobytes())
                prediction_chunks.append(prediction)
                target_chunks.append(target)
            if not prediction_chunks:
                raise PhysicalPIRGScoreError("PIRG score session emitted no windows")
            prediction = torch.cat(prediction_chunks)
            target = torch.cat(target_chunks)
            target_array = np.ascontiguousarray(target.numpy(), dtype=np.float32)
            mask_array = np.ascontiguousarray(np.all(target_array != -1.0, axis=1), dtype=np.uint8)
            record = records[session_name]
            if (
                target_array.shape != session.last_targets.shape or not np.array_equal(target_array, session.last_targets)
                or not np.array_equal(mask_array, session.last_valid_mask)
                or _digest(target_array.tobytes()) != record["last_bin_target_sha256"]
                or _digest(mask_array.tobytes()) != record["last_bin_mask_sha256"]
                or int(mask_array.sum()) != int(record["n_windows"])
            ):
                raise PhysicalPIRGScoreError("PIRG score target/mask/window binding drift")
            r2 = self._v3_physical.manual_variance_weighted_two_coordinate_r2(
                predictions=prediction, targets=target, torch=torch,
            )
            rows.append(contract.SessionScore(
                session=session_name, n_windows=len(session.starts), r2=float(r2),
                prediction_sha256=digest.hexdigest(), input_sha256=_digest(_json(record)),
            ))
        if cell.mode == contract.BASELINE_MODE:
            state_after = runtime["arm_common"].state_sha256(model)
        else:
            state_after = self._wrapper_state_digest(arm_common=runtime["arm_common"], core=runtime["core"], wrapper=model)
        if state_after != state_before or model.training or any(parameter.grad is not None for parameter in model.parameters()):
            raise PhysicalPIRGScoreError("PIRG score forward mutated model state")
        return contract.CellEvidence(
            cell=cell, model_state_before_sha256=state_before, model_state_after_sha256=state_after,
            model_artifact_sha256=model_artifact, base_cell_d_swa_sha256=plan.SEALED_CELL_D_SWA_SHA256,
            rows=tuple(rows), eval_mode=True, dropout_disabled=True,
            gradients_none=True, repeated_fixed_batch_bitwise_equal=repeated, finite_outputs=True,
        )

    def final_reverify(self, *, identity: contract.PIRGScoreIdentity) -> contract.ScoreClosure:
        _v3_contract, _v3_physical, quick, _v3_identity, prepared_identity, _seam = self._require_prepared()
        if identity is not prepared_identity:
            raise PhysicalPIRGScoreError("PIRG score final identity object drift")
        runtime = quick._backend._load_runtime()
        try:
            for held in quick._backend._held_assets:
                held.reverify()
            for held in quick._backend._held_roots:
                held.reverify()
            for name, model in quick._backend._models.items():
                if runtime["arm_common"].state_sha256(model) != quick._backend._state_at_load.get(name) or model.training or not quick._backend._all_gradients_none(model):
                    raise PhysicalPIRGScoreError(f"PIRG V3 model state/eval drift after score: {name}")
            if self._wrapper_state_digest(arm_common=runtime["arm_common"], core=runtime["core"], wrapper=self._pirg) != self._pirg_state_at_load:
                raise PhysicalPIRGScoreError("PIRG gated model state drift after score")
            # Re-read V3 results and the source artifact chain before score
            # publication.  This makes predecessor/selected-input swaps fail
            # before a scientific receipt is minted.
            _identity, inputs, v3_score, _terminal, _validation, _reused = self._attr_physical._load_v3_result(
                root=self._v3_stage_root, v3_contract=self._v3_contract, v3_physical=self._v3_physical,
            )
            if inputs != self._v3_inputs or v3_score != self._v3_score:
                raise PhysicalPIRGScoreError("PIRG V3 predecessor changed during score forwards")
            self._load_completed_source_artifact(torch=runtime["torch"], physical=quick._physical, identity=identity)
        except PhysicalPIRGScoreError:
            raise
        except Exception as error:
            raise PhysicalPIRGScoreError("PIRG final held/predecessor revalidation failed") from error
        final = contract.score_implementation_closure(self._root).payload()
        if final != identity.closure.payload():
            raise PhysicalPIRGScoreError("PIRG score launch/final closure drift")
        return contract.ScoreClosure(final)

    def progress(self) -> Mapping[str, object]:
        flags = self._v3_flags
        if flags is None:
            return {"within_opened": False, "external_opened": False, "cuda_initialized": False,
                    "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0}
        return {
            "within_opened": bool(flags.within_opened), "external_opened": bool(flags.external_opened),
            "cuda_initialized": bool(flags.remote_initialized),
            "target_optimizer_steps": int(flags.target_optimizer_steps),
            "target_backward_calls": int(flags.target_backward_calls),
            "target_update_calls": int(flags.target_update_calls),
            "formal_opened": bool(flags.formal_opened), "h1_opened": bool(flags.h1_opened),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._quick is not None:
            self._quick.close()
        self._input_sessions.clear()
        self._pirg = None
        self._seam = None
