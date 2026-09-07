"""Deferred source-only physical backend for PIRG.

Nothing in this module runs at import time.  The constructor is inert; only a
future reviewed lifecycle can call :meth:`PhysicalPIRGBackend.prepare` after
it has written an immutable attempt.  The implementation composes the audited
strict-27 posterior source adapter for *credibility lookup only* and rebuilds
the sealed Cell-D SWA as the frozen base graph.  It does not use a posterior
mean, posterior normalizer, posterior sample, or attention bias.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import plan
from .core import CredibilityGateCache, PIRGError, PosteriorIdentityResidualGate
from .train import (
    ALPHA_LR,
    SOURCE_STEPS_PER_EPOCH,
    EpochSummary,
    ImplementationClosure,
    PIRGIdentity,
    PIRGTrainError,
    build_schedule_for_count,
    implementation_closure,
)


class PIRGPhysicalError(PIRGTrainError):
    """A source/CUDA/model boundary failed after an immutable attempt."""


def _load_exact_module(*, name: str, path: Path) -> Any:
    """Load one closure-bound helper without executing ``tfpd_lane.__init__``."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PIRGPhysicalError(f"PIRG cannot load exact runtime helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Runtime:
    torch: Any
    adapter: Any
    wrapper: Any
    optimizer: Any
    loader: Any
    device: Any
    arm_common: Any
    base_state_sha256: str
    cache: CredibilityGateCache
    source_authority_metadata_sha256: str
    remote_device: Mapping[str, object]
    tf32_enforcement: Any
    source_opened: bool
    cuda_initialized: bool
    current_epoch: int = -1
    iterator: Any | None = None
    optimizer_steps: int = 0


def _finite_state(*, torch: Any, model: Any, optimizer: Any) -> tuple[bool, bool]:
    """One boundary-only finite scan that leaves Cell-D lazy tensors inert."""
    from torch.nn.parameter import UninitializedParameter

    finite_model = True
    for parameter in model.parameters():
        if isinstance(parameter, UninitializedParameter):
            continue
        finite_model = finite_model and bool(torch.isfinite(parameter).all().item())
    finite_optimizer = True
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point():
                finite_optimizer = finite_optimizer and bool(torch.isfinite(value).all().item())
    return finite_model, finite_optimizer


def _strict_sealed_state(*, torch: Any, body: bytes) -> Mapping[str, Any]:
    """Load the sealed Cell-D SWA without globally widening safe globals."""
    from torch.nn.parameter import UninitializedParameter
    from torch.torch_version import TorchVersion

    try:
        with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
            payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PIRGPhysicalError("sealed Cell-D SWA weights-only load failed") from error
    if not isinstance(payload, Mapping):
        raise PIRGPhysicalError("sealed Cell-D SWA payload root drift")
    state = payload.get("state_dict", payload.get("state"))
    manifest = payload.get("swa_manifest")
    if (
        not isinstance(state, Mapping) or not isinstance(manifest, Mapping)
        or manifest.get("uninitialized_lazy_tensor_count") != 2
        or manifest.get("optimizer_state_included") is not False
    ):
        raise PIRGPhysicalError("sealed Cell-D SWA schema/lazy topology drift")
    return state


def _dense_valid_bin_mse(*, prediction: Any, behavior: Any, torch: Any) -> Any:
    valid = (behavior != -1.0).all(dim=-1)
    if not bool(valid.any().item()):
        raise PIRGPhysicalError("PIRG source batch has no valid dense-loss bin")
    return (((prediction - behavior).square().sum(dim=-1) * valid).sum()
            / (valid.sum() * behavior.shape[-1]))


class PhysicalPIRGBackend:
    """A narrow composition over the reviewed posterior source adapter.

    The object has no public construction path from the dry CLI.  ``source_data``
    must be the existing typed source-root capability produced by the audited
    Phase-B route; it is deliberately opaque here rather than re-created from
    a pathname.
    """

    def __init__(self, *, root: Path, source_data: object, num_workers: int = 4,
                 nvml_status: str = "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH") -> None:
        self.root = Path(root).absolute()
        self.source_data = source_data
        self.num_workers = num_workers
        self.nvml_status = nvml_status
        self._runtime: _Runtime | None = None
        self._closed = False

    def _load_modules(self) -> Mapping[str, Any]:
        """Deferred imports only; callers must already have an attempt receipt."""
        import sys

        for extra in (self.root / "tfpd_exploration", self.root / "sua_exploration", self.root / "streaming_calibration_exp"):
            rendered = str(extra)
            if rendered not in sys.path:
                sys.path.insert(0, rendered)
        torch = importlib.import_module("torch")
        phase_b = importlib.import_module("src.posterior_carrier_v1.phase_b")
        phase_b_v3 = importlib.import_module("src.posterior_carrier_v1.phase_b_v3")
        source_adapter_v2 = importlib.import_module("src.posterior_carrier_v1.source_adapter_v2")
        matched = importlib.import_module("src.posterior_carrier_v1.matched_score_physical")
        # These helpers are loaded through their exact closure paths; importing
        # ``src.tfpd_lane`` would execute unrelated exploratory mechanisms.
        pop_robust = _load_exact_module(
            name="_pirg_pop_robust", path=self.root / "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        )
        arm_common = _load_exact_module(
            name="_pirg_arm_common", path=self.root / "tfpd_exploration/src/tfpd_lane/arm_common.py",
        )
        return {
            "torch": torch, "phase_b": phase_b, "phase_b_v3": phase_b_v3,
            "source_adapter_v2": source_adapter_v2, "matched": matched,
            "pop_robust": pop_robust, "arm_common": arm_common,
        }

    @staticmethod
    def _only_alpha_optimizer(*, wrapper: PosteriorIdentityResidualGate, optimizer: Any) -> None:
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if parameters != [wrapper.alpha] or any(parameter.requires_grad for name, parameter in wrapper.named_parameters() if name != "alpha"):
            raise PIRGPhysicalError("PIRG optimizer/trainable-parameter topology drift")

    def prepare(self, *, identity: PIRGIdentity) -> None:
        if self._closed or self._runtime is not None:
            raise PIRGPhysicalError("PIRG physical backend prepare lifecycle drift")
        if self.num_workers != 4:
            raise PIRGPhysicalError("PIRG holds the audited source worker count at four")
        if implementation_closure(self.root).payload() != identity.closure.payload():
            raise PIRGPhysicalError("PIRG live closure drift before source/CUDA prepare")
        modules = self._load_modules()
        torch, phase_b, phase_b_v3 = modules["torch"], modules["phase_b"], modules["phase_b_v3"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise PIRGPhysicalError("PIRG requires reviewed CVD=0 / PCI_BUS_ID before device preparation")
        progress = {"source_opened": False, "cuda_initialized": False}

        def mark_source_opened() -> None:
            progress["source_opened"] = True

        enforcement = None
        try:
            # This descriptor-safe call is deliberately before TF32 policy,
            # source-adapter construction, and every CUDA query.  The fresh
            # stage must contain all three exact Cell-D body+sidecar pairs;
            # the matched loader retains ownership of their semantic checks.
            material = modules["matched"].load_sealed_cell_d_material(self.root)
            if material.swa_sha256 != plan.SEALED_CELL_D_SWA_SHA256:
                raise PIRGPhysicalError("PIRG sealed Cell-D SWA authority drift")
            # V3's route-local policy is reused before source/model/optimizer
            # construction.  This is a device-execution convention, not a
            # new PIRG scientific factor.
            enforcement = phase_b_v3.TF32Enforcement.enforce(torch)
            adapter = modules["source_adapter_v2"].build_physical_source_adapter_v2(
                self.root, source_data=self.source_data, num_workers=self.num_workers,
                on_source_opened=mark_source_opened,
            )
            if len(adapter.roster) != 27 or len(set(adapter.roster)) != 27:
                raise PIRGPhysicalError("PIRG strict-27 source roster drift")
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise PIRGPhysicalError("PIRG requires exactly one reviewed visible CUDA device")
            progress["cuda_initialized"] = True
            device = torch.device("cuda:0")
            attestation = phase_b.attest_remote_torch_only(torch, nvml_status=self.nvml_status)
            # Match the reviewed Cell-D source-run ordering: source inventory
            # is already fixed, then seed the model/dropout stream immediately
            # before building the exact graph.  The credibility cache itself
            # uses no global RNG and is built only after this step.
            import random
            import numpy as np

            random.seed(plan.SEED)
            np.random.seed(plan.SEED)
            torch.manual_seed(plan.SEED)
            state = _strict_sealed_state(torch=torch, body=material.swa_body)
            base = modules["pop_robust"].build_population_robustness_model(seed=plan.SEED, cell="D")
            if set(state) != set(base.state_dict()):
                raise PIRGPhysicalError("PIRG sealed Cell-D state-key topology drift")
            base.load_state_dict(state, strict=True)
            wrapper = PosteriorIdentityResidualGate(base)
            preservation = wrapper.preservation()
            if (
                preservation.base_live_parameters != 3_510_842
                or preservation.wrapper_live_parameters != 3_510_843
                or preservation.new_trainable_parameter_names != ("alpha",)
            ):
                raise PIRGPhysicalError("PIRG sealed Cell-D/one-scalar preservation drift")
            wrapper.to(device)
            optimizer = torch.optim.Adam([wrapper.alpha], lr=ALPHA_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
            self._only_alpha_optimizer(wrapper=wrapper, optimizer=optimizer)
            from torch.utils.data import DataLoader

            loader = DataLoader(adapter.dataset, batch_sampler=adapter.sampler, num_workers=self.num_workers, pin_memory=True)
            if len(adapter.sampler) != SOURCE_STEPS_PER_EPOCH:
                raise PIRGPhysicalError("PIRG strict source B32 batch count drift")
            # All 27×3 credibility controls are copied and centred before any
            # iterator exists.  The B32 optimizer core can only retrieve them.
            cache = CredibilityGateCache(roster=adapter.roster, posterior_bank=adapter.bank)
            for epoch in range(plan.EPOCHS):
                cache.prewarm_epoch(epoch=epoch, device=device)
            if cache.observer().controls_built != 27 * plan.EPOCHS:
                raise PIRGPhysicalError("PIRG gate-cache topology drift")
            torch.cuda.reset_peak_memory_stats(0)
            self._runtime = _Runtime(
                torch=torch, adapter=adapter, wrapper=wrapper, optimizer=optimizer, loader=loader, device=device,
                arm_common=modules["arm_common"], base_state_sha256=modules["arm_common"].state_sha256(wrapper.cell_d),
                cache=cache,
                source_authority_metadata_sha256=hashlib.sha256(
                    json_canonical(adapter.source_authority_metadata)
                ).hexdigest(),
                remote_device=dict(attestation.payload), tf32_enforcement=enforcement,
                source_opened=bool(progress["source_opened"]), cuda_initialized=bool(progress["cuda_initialized"]),
            )
        except BaseException:
            if enforcement is not None:
                enforcement.restore(torch)
            raise

    def _require_runtime(self) -> _Runtime:
        if self._runtime is None or self._closed:
            raise PIRGPhysicalError("PIRG physical runtime is unavailable")
        return self._runtime

    def source_authority(self, *, identity: PIRGIdentity) -> Mapping[str, object]:
        runtime = self._require_runtime()
        controls = []
        for epoch, row in enumerate(build_schedule_for_count(len(runtime.adapter.roster))):
            for index, (session, budget) in enumerate(zip(runtime.adapter.roster, row, strict=True)):
                control = runtime.cache.control_for_receipt_audit(session=session, epoch=epoch, device=runtime.device)
                controls.append({"session": session, "session_index": index, "epoch": epoch, "budget": budget,
                                 "posterior_sha256": control.posterior_sha256})
        observer = runtime.cache.observer()
        if observer.optimizer_batch_requests != 0:
            raise PIRGPhysicalError("PIRG source authority consumed an optimizer-batch cache access")
        return {
            "schema": "posterior_identity_residual_gate_source_authority_v1",
            "identity": identity.payload(), "source_only": True, "roster": list(runtime.adapter.roster),
            "budget_schedule": [list(row) for row in build_schedule_for_count(len(runtime.adapter.roster))],
            "posterior_credibility_only": True, "ordinary_ols_t4_held": True,
            "posterior_mean_used": False, "posterior_normalizer_used": False,
            "posterior_sampling_used": False, "posterior_attention_bias_used": False,
            "batch_loop_inverse_calls": 0,
            "gate_cache": observer.payload(),
            "control_digest_rows": controls,
            "base_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
            "model_dropout_rng_seed": plan.SEED,
            "source_adapter_metadata_sha256": runtime.source_authority_metadata_sha256,
            "remote_torch_authority": dict(runtime.remote_device),
            "tf32_enforcement": runtime.tf32_enforcement.payload(),
            "source_opened": True, "target_opened": False, "within_opened": False,
            "external_opened": False, "formal_opened": False,
        }

    def run_epoch(self, *, epoch: int, identity: PIRGIdentity) -> EpochSummary:
        del identity
        runtime = self._require_runtime()
        if epoch != runtime.current_epoch + 1 or not 0 <= epoch < plan.EPOCHS:
            raise PIRGPhysicalError("PIRG epoch ordering drift")
        runtime.iterator = iter(runtime.loader)
        runtime.current_epoch = epoch
        schedule = build_schedule_for_count(len(runtime.adapter.roster))[epoch]
        expected_budget = dict(zip(runtime.adapter.roster, schedule, strict=True))
        alpha_before = float(runtime.wrapper.alpha.detach().cpu().item())
        loss_values: list[float] = []
        gradient_nonzero = False
        started = time.perf_counter()
        runtime.wrapper.train(True)
        for local_step in range(SOURCE_STEPS_PER_EPOCH):
            try:
                batch = next(runtime.iterator)
            except StopIteration as error:
                raise PIRGPhysicalError("PIRG source sampler exhausted before exact epoch boundary") from error
            neural, behavior, calib, sessions, side = batch[:5]
            names = tuple(sessions)
            if len(names) != plan.BATCH_SIZE or len(set(names)) != 1 or names[0] not in expected_budget:
                raise PIRGPhysicalError("PIRG source B32/session-homogeneity drift")
            session = names[0]
            control = runtime.cache.control_for_optimizer_batch(session=session, epoch=epoch, device=runtime.device)
            if control.budget != expected_budget[session]:
                raise PIRGPhysicalError("PIRG session×epoch credibility budget drift")
            neural = neural.to(runtime.device, non_blocking=True)
            behavior = behavior.to(runtime.device, non_blocking=True)
            calib = calib.to(runtime.device, non_blocking=True)
            side = side.to(runtime.device, non_blocking=True)
            runtime.optimizer.zero_grad(set_to_none=True)
            prediction, _identity, _gate, _z = runtime.wrapper(
                neural, calib_trials_m30=calib, ordinary_ols_side_features=side,
                directional_credibility=control.credibility,
            )
            loss = _dense_valid_bin_mse(prediction=prediction, behavior=behavior, torch=runtime.torch)
            loss.backward()
            if runtime.wrapper.alpha.grad is not None:
                gradient_nonzero = gradient_nonzero or bool(runtime.wrapper.alpha.grad.detach().ne(0).any().item())
            self._only_alpha_optimizer(wrapper=runtime.wrapper, optimizer=runtime.optimizer)
            runtime.optimizer.step()
            runtime.optimizer_steps += 1
            scalar = float(loss.detach().cpu().item())
            if not math_isfinite(scalar):
                raise PIRGPhysicalError("PIRG source loss became nonfinite")
            loss_values.append(scalar)
        finite_model, finite_optimizer = _finite_state(torch=runtime.torch, model=runtime.wrapper, optimizer=runtime.optimizer)
        if not finite_model or not finite_optimizer or not gradient_nonzero:
            raise PIRGPhysicalError("PIRG epoch boundary only-alpha/finite/nonzero-gradient proof failed")
        if runtime.arm_common.state_sha256(runtime.wrapper.cell_d) != runtime.base_state_sha256:
            raise PIRGPhysicalError("PIRG modified a frozen Cell-D parameter")
        gate_stats: dict[str, dict[str, float]] = {}
        for budget in plan.BUDGETS:
            values = []
            for index, session in enumerate(runtime.adapter.roster):
                if schedule[index] == budget:
                    gate = runtime.cache.control_for_receipt_audit(session=session, epoch=epoch, device=runtime.device).gate(runtime.wrapper.alpha)
                    values.extend(float(item) for item in gate.detach().cpu().reshape(-1).tolist())
            if not values:
                raise PIRGPhysicalError("PIRG gate summary omitted a scheduled budget")
            gate_stats[str(budget)] = {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
        observer = runtime.cache.observer().payload()
        return EpochSummary(
            epoch=epoch, budget_schedule=schedule, alpha_before=alpha_before,
            alpha_after=float(runtime.wrapper.alpha.detach().cpu().item()), loss_first=loss_values[0],
            loss_last=loss_values[-1], loss_min=min(loss_values), loss_max=max(loss_values),
            optimizer_steps=SOURCE_STEPS_PER_EPOCH, only_alpha_gradient=True,
            alpha_gradient_nonzero=gradient_nonzero, finite_model=finite_model, finite_optimizer=finite_optimizer,
            gate_stats_by_budget=gate_stats,
            throughput_steps_per_second=SOURCE_STEPS_PER_EPOCH / max(time.perf_counter() - started, 1e-12),
            cache=observer,
        )

    def final_artifact(self, *, identity: PIRGIdentity) -> tuple[bytes, Mapping[str, object]]:
        del identity
        runtime = self._require_runtime()
        runtime.wrapper.eval()
        if runtime.wrapper.training:
            raise PIRGPhysicalError("PIRG final artifact must be saved in eval mode")
        payload = {
            "schema": "posterior_identity_residual_gate_final_alpha_v1",
            "alpha": runtime.wrapper.alpha.detach().cpu().clone(),
            "base_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
            "base_state_sha256": runtime.base_state_sha256,
            "preservation": runtime.wrapper.preservation().payload(),
        }
        buffer = io.BytesIO()
        runtime.torch.save(payload, buffer)
        body = buffer.getvalue()
        # CPU reload is part of the final artifact proof; it never touches an
        # evaluation surface or CUDA state.
        reloaded = runtime.torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
        if not isinstance(reloaded, Mapping) or not runtime.torch.equal(reloaded.get("alpha"), payload["alpha"]):
            raise PIRGPhysicalError("PIRG final alpha CPU reload/digest proof failed")
        return body, {
            "schema": "posterior_identity_residual_gate_final_alpha_manifest_v1",
            "artifact_sha256": hashlib.sha256(body).hexdigest(),
            "alpha": float(payload["alpha"].item()),
            "base_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
            "base_state_sha256": runtime.base_state_sha256,
            "cpu_reload_equal": True,
        }

    def final_reverify(self, *, identity: PIRGIdentity) -> ImplementationClosure:
        runtime = self._require_runtime()
        if runtime.optimizer_steps != plan.EPOCHS * SOURCE_STEPS_PER_EPOCH:
            raise PIRGPhysicalError("PIRG final optimizer-step count drift")
        if runtime.arm_common.state_sha256(runtime.wrapper.cell_d) != runtime.base_state_sha256:
            raise PIRGPhysicalError("PIRG final frozen Cell-D state drift")
        observer = runtime.cache.observer()
        if (
            observer.controls_built != 27 * plan.EPOCHS
            or observer.optimizer_batch_inverse_calls != 0
            or observer.posterior_mean_view_builds != 0
            or observer.posterior_sampling_view_builds != 0
            or observer.posterior_normalizer_view_builds != 0
            or observer.optimizer_batch_requests != runtime.optimizer_steps
        ):
            raise PIRGPhysicalError("PIRG final posterior-cache/batch-loop proof drift")
        closure = implementation_closure(self.root)
        if closure.payload() != identity.closure.payload():
            raise PIRGPhysicalError("PIRG final live closure drift")
        return closure

    def progress(self) -> Mapping[str, object]:
        if self._runtime is None:
            return {"source_opened": False, "cuda_initialized": False}
        return {
            "source_opened": self._runtime.source_opened,
            "cuda_initialized": self._runtime.cuda_initialized,
            "optimizer_steps_completed": self._runtime.optimizer_steps,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runtime is not None:
            self._runtime.tf32_enforcement.restore(self._runtime.torch)
            self._runtime = None


def json_canonical(value: object) -> bytes:
    """Local canonical source-metadata digest without importing a broad helper."""
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def math_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
