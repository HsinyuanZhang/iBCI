"""Deferred remote physical backend for the Posterior Carrier quick screen.

The implementation composes the frozen Posterior matched-score substrate
instead of modifying or monkeypatching it.  The base substrate remains
responsible for strict Cell-D/Posterior SWA loading, no-cache target parsing,
M30 B3S calibration, ordinary-point and posterior carrier construction,
last-bin metric evaluation, held-FD input checks, and model-state invariants.

This wrapper changes only the *screen scope*: it uses the exact full authority
to derive six fixed rows, materializes just the frozen 3+3 subset, and runs
the four non-governing cells.  It is deliberately inaccessible from the
public CLI and is never imported by the dry contract.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import quick_screen as contract


class PhysicalQuickScreenError(RuntimeError):
    """Fail closed during the future reviewed remote engineering screen."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise PhysicalQuickScreenError(message)


# The remote TorchMetrics 1.9 implementation and the local frozen 1.5.1
# helper agree on the variance-weighted two-output value at float32 precision.
# A float64 re-expression is intentionally not required to be bit-identical
# (the independently checked difference on a 257x2 fixture is ~1.5e-8).
_MANUAL_R2_PARITY_TOLERANCE = 1e-7
_FROZEN_BASE_FP64_FINAL4_MISMATCH = "full SWA is not exact arithmetic final-four state mean"


def manual_variance_weighted_two_coordinate_r2(*, predictions: Any, targets: Any, torch: Any) -> float:
    """Route-local R² compatible with the formal two-coordinate metric label.

    This function exists only because the frozen shared helper instantiates
    ``R2Score(num_outputs=...)``, an API rejected by TorchMetrics 1.9.  It
    keeps the normal float32 accumulation of the scored tensors, rejects
    degenerate/nonfinite inputs, and implements the same variance-weighted
    aggregation: ``sum_c SST_c * R2_c / sum_c SST_c``.  It never fits a target
    statistic beyond the per-session metric mean.
    """
    try:
        if (
            getattr(predictions, "ndim", None) != 2
            or getattr(targets, "ndim", None) != 2
            or tuple(predictions.shape) != tuple(targets.shape)
            or int(predictions.shape[-1]) != 2
            or int(predictions.shape[0]) < 2
        ):
            raise PhysicalQuickScreenError("manual variance-weighted R2 requires nondegenerate [n,2] tensors")
        if not bool(torch.isfinite(predictions).all()) or not bool(torch.isfinite(targets).all()):
            raise PhysicalQuickScreenError("manual variance-weighted R2 input is nonfinite")
        target_mean = targets.mean(dim=0)
        residual = ((targets - predictions) ** 2).sum(dim=0)
        total = ((targets - target_mean) ** 2).sum(dim=0)
        if not bool((total > 0).all()) or not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(total).all()):
            raise PhysicalQuickScreenError("manual variance-weighted R2 target variance is degenerate")
        value = ((1.0 - residual / total) * (total / total.sum())).sum()
        if not bool(torch.isfinite(value)):
            raise PhysicalQuickScreenError("manual variance-weighted R2 result is nonfinite")
        return float(value.detach().cpu())
    except PhysicalQuickScreenError:
        raise
    except Exception as error:
        raise PhysicalQuickScreenError("manual variance-weighted R2 evaluation failed") from error


def _load_exact_module(root: Path, relative: str, module_name: str) -> Any:
    """Load one closure-bound source file without importing a package tree."""
    path = Path(root).absolute() / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PhysicalQuickScreenError(f"cannot load closure-bound module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_immutable_pair(directory: Path, name: str, expected_sha256: str, physical: Any) -> dict[str, object]:
    """Read one body+sidecar through the frozen descriptor-safe helper."""
    try:
        with physical.ImmutableDirectory.open(directory) as held:
            item = held.read_pair(name, expected_sha256=expected_sha256)
            held.reverify()
    except physical.PhysicalScoreError as error:
        raise PhysicalQuickScreenError(f"immutable evidence read failed: {error}") from error
    return item.json_object()


def _validate_v1_attempt_semantics(value: Mapping[str, object], *, evidence: Mapping[str, object]) -> None:
    """Validate the known V1 pre-input attempt shape after SHA verification."""
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "boundaries",
        "evaluation_assets_resolved", "remote_initialized", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls",
    }
    attempt_evidence = evidence["attempt"]
    if (
        not isinstance(value, Mapping) or set(value) != expected
        or value.get("schema") != attempt_evidence["schema"]
        or value.get("status") != attempt_evidence["status"]
        or value.get("classification") != contract.CLASSIFICATION
        or value.get("cell") != contract.CELL
        or value.get("phase") != "POSTERIOR_CARRIER_QUICK_SCREEN_V1"
        or not isinstance(value.get("identity"), Mapping) or not isinstance(value.get("boundaries"), Mapping)
        or value.get("evaluation_assets_resolved") is not False
        or value.get("remote_initialized") is not False
        or value.get("target_optimizer_steps") != 0
        or value.get("target_backward_calls") != 0
        or value.get("target_update_calls") != 0
    ):
        raise PhysicalQuickScreenError("V1 failed predecessor attempt semantic drift")


def _validate_v1_failure_semantics(
    value: Mapping[str, object], *, attempt: Mapping[str, object], evidence: Mapping[str, object],
) -> None:
    """Validate the known prepare failure without rebuilding V1 identity bytes."""
    expected = {
        "schema", "classification", "cell", "phase", "identity", "stage", "attempt_sha256",
        "input_authority_sha256", "flags", "terminal_published", "error_class", "error_sha256", "traceback_sha256",
    }
    failure_evidence = evidence["failure"]
    if (
        not isinstance(value, Mapping) or set(value) != expected
        or value.get("schema") != failure_evidence["schema"]
        or value.get("classification") != contract.CLASSIFICATION
        or value.get("cell") != contract.CELL
        or value.get("phase") != "POSTERIOR_CARRIER_QUICK_SCREEN_V1"
        or value.get("identity") != attempt.get("identity")
        or value.get("stage") != failure_evidence["stage"]
        or value.get("attempt_sha256") != evidence["attempt"]["sha256"]
        or value.get("input_authority_sha256") is not failure_evidence["input_authority_sha256"]
        or value.get("terminal_published") is not failure_evidence["terminal_published"]
        or value.get("error_class") != failure_evidence["error_class"]
        or value.get("error_sha256") != failure_evidence["error_sha256"]
        or not isinstance(value.get("traceback_sha256"), str)
        or not isinstance(value.get("flags"), Mapping)
    ):
        raise PhysicalQuickScreenError("V1 failed predecessor failure semantic drift")
    flags = value["flags"]
    expected_flag_keys = {
        "within_opened", "external_opened", "remote_initialized", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "normalizer_refit", "target_sampling",
        "h1_opened", "formal_opened", "forward_cells",
    }
    if (
        set(flags) != expected_flag_keys
        or flags.get("within_opened") is not False or flags.get("external_opened") is not False
        or flags.get("remote_initialized") is not False or flags.get("normalizer_refit") is not False
        or flags.get("target_sampling") is not False or flags.get("h1_opened") is not False
        or flags.get("formal_opened") is not False or flags.get("forward_cells") != []
        or flags.get("target_optimizer_steps") != evidence["target_optimizer_steps"]
        or flags.get("target_backward_calls") != evidence["target_backward_calls"]
        or flags.get("target_update_calls") != evidence["target_update_calls"]
    ):
        raise PhysicalQuickScreenError("V1 failed predecessor update/input boundary drift")


def _validate_v2_attempt_semantics(value: Mapping[str, object], *, evidence: Mapping[str, object]) -> None:
    """Validate the known V2 pre-input attempt shape after SHA verification."""
    expected = {
        "schema", "status", "classification", "cell", "phase", "identity", "boundaries",
        "evaluation_assets_resolved", "remote_initialized", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls",
    }
    attempt_evidence = evidence["attempt"]
    if (
        not isinstance(value, Mapping) or set(value) != expected
        or value.get("schema") != attempt_evidence["schema"]
        or value.get("status") != attempt_evidence["status"]
        or value.get("classification") != contract.CLASSIFICATION
        or value.get("cell") != contract.CELL
        or value.get("phase") != "POSTERIOR_CARRIER_QUICK_SCREEN_V2"
        or not isinstance(value.get("identity"), Mapping) or not isinstance(value.get("boundaries"), Mapping)
        or value.get("evaluation_assets_resolved") is not False
        or value.get("remote_initialized") is not False
        or value.get("target_optimizer_steps") != 0
        or value.get("target_backward_calls") != 0
        or value.get("target_update_calls") != 0
    ):
        raise PhysicalQuickScreenError("V2 failed predecessor attempt semantic drift")


def _validate_v2_failure_semantics(
    value: Mapping[str, object], *, attempt: Mapping[str, object], evidence: Mapping[str, object],
) -> None:
    """Validate the known V2 prepare failure without rebuilding V2 identity bytes."""
    expected = {
        "schema", "classification", "cell", "phase", "identity", "stage", "attempt_sha256",
        "input_authority_sha256", "flags", "terminal_published", "error_class", "error_sha256", "traceback_sha256",
    }
    failure_evidence = evidence["failure"]
    if (
        not isinstance(value, Mapping) or set(value) != expected
        or value.get("schema") != failure_evidence["schema"]
        or value.get("classification") != contract.CLASSIFICATION
        or value.get("cell") != contract.CELL
        or value.get("phase") != "POSTERIOR_CARRIER_QUICK_SCREEN_V2"
        or value.get("identity") != attempt.get("identity")
        or value.get("stage") != failure_evidence["stage"]
        or value.get("attempt_sha256") != evidence["attempt"]["sha256"]
        or value.get("input_authority_sha256") is not failure_evidence["input_authority_sha256"]
        or value.get("terminal_published") is not failure_evidence["terminal_published"]
        or value.get("error_class") != failure_evidence["error_class"]
        or value.get("error_sha256") != failure_evidence["error_sha256"]
        or not isinstance(value.get("traceback_sha256"), str)
        or not isinstance(value.get("flags"), Mapping)
    ):
        raise PhysicalQuickScreenError("V2 failed predecessor failure semantic drift")
    flags = value["flags"]
    expected_flag_keys = {
        "within_opened", "external_opened", "remote_initialized", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "normalizer_refit", "target_sampling",
        "h1_opened", "formal_opened", "forward_cells",
    }
    if (
        set(flags) != expected_flag_keys
        or flags.get("within_opened") is not False or flags.get("external_opened") is not False
        or flags.get("remote_initialized") is not False or flags.get("normalizer_refit") is not False
        or flags.get("target_sampling") is not False or flags.get("h1_opened") is not False
        or flags.get("formal_opened") is not False or flags.get("forward_cells") != []
        or flags.get("target_optimizer_steps") != evidence["target_optimizer_steps"]
        or flags.get("target_backward_calls") != evidence["target_backward_calls"]
        or flags.get("target_update_calls") != evidence["target_update_calls"]
    ):
        raise PhysicalQuickScreenError("V2 failed predecessor update/input boundary drift")


def _validate_failed_predecessor_directory_with_expected(
    *, physical: Any, directory: Path, evidence: Mapping[str, object],
) -> dict[str, object]:
    """Testable held-FD reader for exactly one literal failed predecessor."""
    selected = Path(directory).absolute()
    try:
        with physical.ImmutableDirectory.open(selected) as held:
            expected_names = {
                evidence["attempt"]["name"], f"{evidence['attempt']['name']}.sha256",
                evidence["failure"]["name"], f"{evidence['failure']['name']}.sha256",
            }
            if set(held.names()) != expected_names:
                raise PhysicalQuickScreenError("failed predecessor receipt topology drift")
            attempt_item = held.read_pair(
                evidence["attempt"]["name"], expected_sha256=evidence["attempt"]["sha256"],
            )
            failure_item = held.read_pair(
                evidence["failure"]["name"], expected_sha256=evidence["failure"]["sha256"],
            )
            attempt = attempt_item.json_object()
            failure = failure_item.json_object()
            held.reverify()
    except PhysicalQuickScreenError:
        raise
    except physical.PhysicalScoreError as error:
        raise PhysicalQuickScreenError(f"failed predecessor descriptor validation failed: {error}") from error
    schema = evidence.get("attempt", {}).get("schema") if isinstance(evidence.get("attempt"), Mapping) else None
    if schema == "posterior_carrier_quick_screen_attempt_v1":
        _validate_v1_attempt_semantics(attempt, evidence=evidence)
        _validate_v1_failure_semantics(failure, attempt=attempt, evidence=evidence)
    elif schema == "posterior_carrier_quick_screen_attempt_v2":
        _validate_v2_attempt_semantics(attempt, evidence=evidence)
        _validate_v2_failure_semantics(failure, attempt=attempt, evidence=evidence)
    else:
        raise PhysicalQuickScreenError("failed predecessor attempt schema is not a reviewed V1/V2 schema")
    return dict(evidence)


def validate_v1_failed_predecessor_directory(
    *, physical: Any, directory: Path | None = None,
) -> dict[str, object]:
    """Descriptor-validate the immutable V1 failure before V3 reservation.

    ``directory`` is injectable only for no-data focused tests. Production
    calls leave it unset, which selects the literal V1 remote result path from
    the V3 identity. It is not a public CLI or caller configuration field.
    """
    evidence = contract.validate_v1_failed_predecessor_evidence(
        contract.V1FailedPredecessorEvidence(),
    )
    selected = Path(contract.V1_REMOTE_RESULT_DIRECTORY) if directory is None else Path(directory).absolute()
    return _validate_failed_predecessor_directory_with_expected(
        physical=physical,
        directory=selected,
        evidence=evidence,
    )


def validate_v2_failed_predecessor_directory(
    *, physical: Any, directory: Path | None = None,
) -> dict[str, object]:
    """Descriptor-validate immutable V2 failure before V3 reservation."""
    evidence = contract.validate_v2_failed_predecessor_evidence(
        contract.V2FailedPredecessorEvidence(),
    )
    selected = Path(contract.V2_REMOTE_RESULT_DIRECTORY) if directory is None else Path(directory).absolute()
    return _validate_failed_predecessor_directory_with_expected(
        physical=physical,
        directory=selected,
        evidence=evidence,
    )


def _base_identity_from_payload(base: Any, value: Mapping[str, object]) -> Any:
    """Rebuild the frozen 6/15 base score identity without a local stage guess."""
    if not isinstance(value, Mapping):
        raise PhysicalQuickScreenError("base authority identity must be a mapping")
    try:
        full_value = value["full_training"]
        sealed_value = value["sealed_cell_d"]
        closure_value = value["closure"]
        rosters = value["rosters"]
    except KeyError as error:
        raise PhysicalQuickScreenError("base authority identity schema is incomplete") from error
    if not all(isinstance(item, Mapping) for item in (full_value, sealed_value, closure_value, rosters)):
        raise PhysicalQuickScreenError("base authority identity nested schema drift")
    try:
        full = base.FullTrainingEvidence(
            terminal_sha256=full_value["terminal_sha256"], swa_sha256=full_value["swa_sha256"],
            swa_state_sha256=full_value["swa_state_sha256"], source_authority_sha256=full_value["source_authority_sha256"],
            checkpoint_sha256s=full_value["checkpoint_sha256s"], terminal_status=full_value["terminal_status"],
            full_closure_sha256=full_value["full_closure_sha256"],
        )
        sealed = base.SealedCellDEvidence(
            terminal_sha256=sealed_value["terminal_sha256"], swa_sha256=sealed_value["swa_sha256"],
            m30_ols_point_last_bin_table_sha256=sealed_value["m30_ols_point_last_bin_table_sha256"],
            sealed_point_replay_authority_sha256=sealed_value["sealed_point_replay_authority_sha256"],
        )
        closure = base.ImplementationClosure(sha256_by_path=closure_value["sha256_by_path"])
        identity = base.ScoreIdentity(
            full_training=full, sealed_cell_d=sealed, closure=closure,
            within_roster=tuple(rosters[contract.WITHIN]), external_roster=tuple(rosters[contract.EXTERNAL]),
        )
        rebuilt = base.validate_score_identity(identity)
    except (KeyError, TypeError, base.ScoreError) as error:
        raise PhysicalQuickScreenError(f"base score identity reconstruction failed: {error}") from error
    if rebuilt != dict(value):
        raise PhysicalQuickScreenError("base score identity does not reproduce immutable authority bytes")
    return identity


@dataclass(frozen=True)
class RemoteEngineeringDevice:
    """Exact Torch-only 5070 Ti authority, explicitly not an NVML authority."""

    engineering_only: bool = True
    current_local_authority: bool = False

    def payload(self) -> dict[str, object]:
        if self.engineering_only is not True or self.current_local_authority is not False:
            raise PhysicalQuickScreenError("remote engineering device classification drift")
        try:
            return contract.validate_remote_engineering_device_payload(
                contract.remote_engineering_device_payload(),
            )
        except contract.QuickScreenError as error:
            raise PhysicalQuickScreenError(f"remote engineering device literal drift: {error}") from error


def _torch_only_runtime_attestation(
    *,
    physical: Any,
    expected: Mapping[str, object],
    torch: Any,
    torchmetrics: Any,
) -> dict[str, object]:
    """Attest the reviewed remote through Torch alone—never through NVML.

    The 5070 Ti has a documented driver/library mismatch for NVML.  Calling
    ``nvidia-smi`` would turn that known absent capability into a pre-NWB
    launch failure.  UUID, BDF, and nominal NVML memory therefore remain
    explicit ``null`` values in the durable engineering receipt rather than
    guessed values or local-device fallbacks.
    """
    try:
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or torch.cuda.current_device() != 0
        ):
            raise physical.PhysicalScoreError(
                "Torch-only engineering device authority drift: requires one visible cuda:0",
            )
        properties = torch.cuda.get_device_properties(0)
        actual = {
            "schema": contract.REMOTE_ENGINEERING_DEVICE_SCHEMA,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "visible_device_count": int(torch.cuda.device_count()),
            "logical_device": "cuda:0",
            "uuid": None,
            "bdf": None,
            "name": str(properties.name),
            "nvidia_smi_memory_total_mib": None,
            "nvml_status": "UNAVAILABLE_DRIVER_LIBRARY_MISMATCH",
            "nvidia_smi_called": False,
            "torch_total_memory_bytes": int(properties.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cudnn_version": int(torch.backends.cudnn.version()),
            "torchmetrics_version": str(torchmetrics.__version__),
            "compute_capability": [int(item) for item in torch.cuda.get_device_capability(0)],
            "engineering_only": True,
            "current_local_authority": False,
            "route_role": "remote_5070ti_non_governing_quick_screen",
            "variance_weighted_r2_parity_gate": "PASS__SYNTHETIC_MANUAL_REFERENCE",
        }
    except physical.PhysicalScoreError:
        raise
    except Exception as error:
        raise physical.PhysicalScoreError("Torch-only engineering device attestation is unavailable") from error
    if actual != dict(expected):
        raise physical.PhysicalScoreError("Torch-only engineering device authority drift")
    _assert_variance_weighted_r2_parity(torch=torch, torchmetrics=torchmetrics, physical=physical)
    return actual


def _assert_variance_weighted_r2_parity(*, torch: Any, torchmetrics: Any, physical: Any) -> None:
    """Lock the house R² definition across the remote TorchMetrics upgrade.

    The quick screen remains non-governing, but it still reports the same
    variance-weighted last-bin metric family.  This deterministic synthetic
    gate compares TorchMetrics against its direct two-coordinate definition
    before any model prediction is scored.
    """
    try:
        from torchmetrics.regression import R2Score

        targets = torch.tensor(
            [[-2.0, 1.0], [-0.5, 3.0], [1.0, -1.0], [3.5, 4.0]],
            dtype=torch.float32,
            device="cpu",
        )
        predictions = torch.tensor(
            [[-1.5, 1.5], [-0.25, 2.5], [0.25, -0.5], [4.0, 3.5]],
            dtype=torch.float32,
            device="cpu",
        )
        # TorchMetrics 1.9 removed ``num_outputs``; the shape is inferred from
        # these fixed two-coordinate tensors.  Do not restore the legacy
        # parameter here or via a compatibility wrapper.
        metric = R2Score(multioutput="variance_weighted")
        metric.update(predictions, targets)
        observed = metric.compute()
        manual = manual_variance_weighted_two_coordinate_r2(
            predictions=predictions,
            targets=targets,
            torch=torch,
        )
        if not bool(torch.isfinite(observed).all()) or not math.isfinite(float(manual)):
            raise physical.PhysicalScoreError("Torch-only variance-weighted R2 parity produced nonfinite values")
        if abs(float(observed.detach().cpu()) - float(manual)) > _MANUAL_R2_PARITY_TOLERANCE:
            raise physical.PhysicalScoreError("Torch-only variance-weighted R2 parity gate failed")
    except physical.PhysicalScoreError:
        raise
    except Exception as error:
        raise physical.PhysicalScoreError("Torch-only variance-weighted R2 parity gate is unavailable") from error


def authoritative_full_train_fp32_final4_state(
    *, checkpoints: Mapping[int, Mapping[str, Any]], torch: Any, physical: Any,
) -> dict[str, Any]:
    """Reproduce the completed full trainer's exact final-four state rule.

    This is intentionally not a numerically improved mean.  The sealed full
    route constructed its SWA in native tensor dtype via ``zeros_like``, four
    ordered in-place adds, and ``div(4)``.  Replacing that producer rule with
    FP64 accumulation changes low bits in a real stored state, so V3 requires
    this exact construction and then requires bitwise equality.
    """
    from torch.nn.parameter import UninitializedParameter

    try:
        states = [checkpoints[epoch]["model_state"] for epoch in (44, 45, 46, 47)]
    except (KeyError, TypeError) as error:
        raise physical.PhysicalScoreError("authoritative FP32 SWA checkpoint-state map drift") from error
    if not all(isinstance(state, Mapping) for state in states):
        raise physical.PhysicalScoreError("authoritative FP32 SWA checkpoint states are not mappings")
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise physical.PhysicalScoreError("authoritative FP32 SWA checkpoint state-key topology drift")
    averaged: dict[str, Any] = {}
    for key in keys:
        values = [state[key] for state in states]
        if all(isinstance(item, UninitializedParameter) for item in values):
            # Preserve the two intentionally dead lazy entries without
            # materializing or dropping their topology, exactly as full_train.
            averaged[key] = UninitializedParameter()
            continue
        if any(isinstance(item, UninitializedParameter) for item in values):
            raise physical.PhysicalScoreError("authoritative FP32 SWA lazy topology differs across checkpoints")
        if not all(
            torch.is_tensor(item)
            and tuple(item.shape) == tuple(values[0].shape)
            and item.dtype == values[0].dtype
            for item in values
        ):
            raise physical.PhysicalScoreError("authoritative FP32 SWA tensor type/shape drift")
        if values[0].is_floating_point():
            total = torch.zeros_like(values[0])
            for item in values:
                total.add_(item)
            averaged[key] = total.div(len(values))
        elif not all(torch.equal(values[0], item) for item in values[1:]):
            raise physical.PhysicalScoreError("authoritative FP32 SWA nonfloating state buffer drift")
        else:
            averaged[key] = values[0].detach().clone()
    return averaged


def build_torch_only_engineering_backend_class(*, physical: Any, device: RemoteEngineeringDevice) -> type[Any]:
    """Return the narrow V3 subclass for attestation and exact full-SWA math.

    Everything that touches models, immutable artifacts, parsing, carrier
    construction, batching, forward execution, scoring, and sealed Cell-D
    validation remain inherited verbatim from ``PhysicalPosteriorMatchedBackend``.
    No method is patched on the frozen base class or module.  The two
    route-local overrides are the reviewed Torch-only device attestation and
    the Posterior full-SWA accumulator rule frozen by the completed trainer.
    """
    expected = device.payload()
    parent = physical.PhysicalPosteriorMatchedBackend

    class TorchOnly5070TiEngineeringBackend(parent):
        _quick_screen_engineering_device = expected

        def _runtime_device_attestation(self, *, torch: Any, torchmetrics: Any) -> dict[str, object]:
            # ``self._preflight`` remains the descriptor-validated original
            # formal preflight and is deliberately not rewritten into a
            # fictional NVML-shaped remote contract.  This is the sole
            # override: all source/model/parser/forward methods are inherited.
            if not isinstance(self._preflight.get("device_contract"), Mapping):
                raise physical.PhysicalScoreError("upstream formal preflight device binding is unavailable")
            return _torch_only_runtime_attestation(
                physical=physical,
                expected=self._quick_screen_engineering_device,
                torch=torch,
                torchmetrics=torchmetrics,
            )

        def _validate_swa_against_checkpoints(
            self,
            *,
            swa_body: bytes,
            checkpoints: Mapping[int, Mapping[str, Any]],
            terminal: Mapping[str, object],
            expected_swa_state_sha256: str,
        ) -> Mapping[str, Any]:
            """Accept only the known FP64-vs-producer-FP32 compatibility gap.

            Calling the frozen validator first keeps all of its exact SWA
            schema, launch/source/closure binding, checkpoint state, and
            persisted proof checks.  We catch *only* its known final
            arithmetic comparison.  The V3 path then reconstructs with the
            authoritative full-training FP32 algorithm, requires bitwise
            stored-state equality, and repeats strict fresh-load/digest proof.
            """
            if expected_swa_state_sha256 != contract.POSTERIOR_FULL_SWA_STATE_SHA256:
                raise physical.PhysicalScoreError("V3 reviewed Posterior SWA state-digest authority drift")
            try:
                # Any result or error other than the one known construction
                # mismatch remains the frozen base outcome.
                super()._validate_swa_against_checkpoints(
                    swa_body=swa_body,
                    checkpoints=checkpoints,
                    terminal=terminal,
                    expected_swa_state_sha256=expected_swa_state_sha256,
                )
            except physical.PhysicalScoreError as error:
                if str(error) != _FROZEN_BASE_FP64_FINAL4_MISMATCH:
                    raise
            else:
                raise physical.PhysicalScoreError(
                    "V3 requires the reviewed frozen FP64 final-four compatibility mismatch",
                )

            runtime = self._load_runtime()
            torch = runtime["torch"]
            payload = self._safe_weights_only_load(swa_body, label="full final-four SWA V3 authoritative FP32")
            state = payload.get("state") if isinstance(payload, Mapping) else None
            if (
                not isinstance(state, Mapping)
                or payload.get("state_sha256") != expected_swa_state_sha256
            ):
                raise physical.PhysicalScoreError("V3 authoritative FP32 SWA schema/state binding drift")
            expected = authoritative_full_train_fp32_final4_state(
                checkpoints=checkpoints,
                torch=torch,
                physical=physical,
            )
            if not self._state_equal(torch, state, expected):
                raise physical.PhysicalScoreError(
                    "full SWA does not equal authoritative full-train FP32 final-four state mean",
                )
            fresh = self._fresh_posterior_wrapper(
                state,
                label="full SWA V3 authoritative FP32",
                place_on_device=False,
            )
            actual_state = runtime["arm_common"].state_sha256(fresh)
            if actual_state != expected_swa_state_sha256:
                raise physical.PhysicalScoreError("V3 authoritative FP32 SWA strict-loaded state digest drift")
            self._quick_screen_posterior_final4_compatibility = contract.validate_posterior_final4_compatibility_payload(
                contract.posterior_final4_compatibility_payload(),
            )
            return payload

    TorchOnly5070TiEngineeringBackend.__name__ = "TorchOnly5070TiEngineeringBackend"
    return TorchOnly5070TiEngineeringBackend


def _validated_upstream_preflight_copy(
    *,
    base: Any,
    identity: Any,
    original_preflight: Mapping[str, object],
) -> dict[str, object]:
    """Preserve formal-preflight provenance without rebuilding it on remote.

    The original formal preflight is a durable local authority for the input
    roster, model graph, normalizers, and imported full-result mirror.  It is
    validated before this copy is made and remains the actual preflight passed
    to the inherited backend.  The narrow subclass separately attests the
    current 5070 Ti through Torch; no remote NVML-shaped device mapping is
    invented or substituted into upstream evidence.
    """
    try:
        checked = base.validate_target_free_preflight(original_preflight, identity=identity)
        upstream = json.loads(json.dumps(checked, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as error:
        raise PhysicalQuickScreenError("upstream formal preflight JSON reconstruction failed") from error
    except base.ScoreError as error:
        raise PhysicalQuickScreenError(f"upstream formal preflight revalidation failed: {error}") from error
    if upstream.get("identity") != base.validate_score_identity(identity):
        raise PhysicalQuickScreenError("upstream formal preflight identity drift")
    if not isinstance(upstream.get("device_contract"), Mapping):
        raise PhysicalQuickScreenError("upstream formal preflight device contract is absent")
    return upstream


class QuickScreenPhysicalBackend:
    """Thin, non-monkeypatching selected-roster composition over frozen code."""

    _BATCH_SIZE = 32

    def __init__(self, *, root: Path, engineering_device: RemoteEngineeringDevice) -> None:
        self._root = Path(root).absolute()
        self._engineering_device = engineering_device
        self._base: Any | None = None
        self._physical: Any | None = None
        self._base_identity: Any | None = None
        self._base_preflight: Mapping[str, object] | None = None
        self._upstream_formal_preflight_device: Mapping[str, object] | None = None
        self._backend: Any | None = None
        self._identity: contract.QuickScreenIdentity | None = None
        self._input_records: dict[tuple[str, str], Any] = {}
        self._closed = False

    def _load_base(self) -> tuple[Any, Any]:
        if self._base is None:
            self._base = _load_exact_module(
                self._root, "tfpd_exploration/src/posterior_carrier_v1/matched_score.py",
                "_posterior_carrier_quick_screen_base_contract",
            )
            self._physical = self._base._physical_module()
        return self._base, self._physical

    def _reload_base_evidence(self) -> tuple[Any, Mapping[str, object], Mapping[str, object]]:
        """Reload old authority pairs at both prepare and final revalidation."""
        base, physical = self._load_base()
        directory = self._root / contract.BASE_SCORE_AUTHORITY_ROOT
        preflight = _read_immutable_pair(
            directory, "official_preflight.json", contract.BASE_OFFICIAL_PREFLIGHT_SHA256, physical,
        )
        authorization = _read_immutable_pair(
            directory, "root_authorization.json", contract.BASE_ROOT_AUTHORIZATION_SHA256, physical,
        )
        try:
            identity = _base_identity_from_payload(base, preflight["identity"])
            checked = base.validate_target_free_preflight(preflight, identity=identity)
            base.validate_root_authorization(
                authorization, official_preflight_sha256=contract.BASE_OFFICIAL_PREFLIGHT_SHA256,
                preflight=checked, identity=identity,
            )
        except (KeyError, base.ScoreError) as error:
            raise PhysicalQuickScreenError(f"base authority validation failed: {error}") from error
        predecessor = contract.BasePredecessorEvidence().payload()
        if (
            checked["full_import_provenance"].get("provenance_sha256") != predecessor["posterior_full"]["import_provenance_sha256"]
            or checked["identity"]["full_training"]["terminal_sha256"] != predecessor["posterior_full"]["terminal_sha256"]
            or checked["identity"]["full_training"]["swa_sha256"] != predecessor["posterior_full"]["swa_sha256"]
            or checked["identity"]["full_training"]["swa_state_sha256"] != predecessor["posterior_full"]["swa_state_sha256"]
            or checked["identity"]["sealed_cell_d"]["terminal_sha256"] != predecessor["sealed_cell_d"]["terminal_sha256"]
            or checked["identity"]["sealed_cell_d"]["swa_sha256"] != predecessor["sealed_cell_d"]["swa_sha256"]
        ):
            raise PhysicalQuickScreenError("base authority does not bind reviewed quick-screen predecessors")
        return identity, checked, authorization

    def _expected_selected_rows(self, preflight: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
        assets = preflight.get("input_assets")
        if not isinstance(assets, Mapping):
            raise PhysicalQuickScreenError("base preflight input-assets table absent")
        try:
            return contract.select_from_full_authority_rows(assets)
        except contract.QuickScreenError as error:
            raise PhysicalQuickScreenError(str(error)) from error

    def prepare(self, *, identity: contract.QuickScreenIdentity, flags: contract.RuntimeFlags) -> None:
        if self._closed:
            raise PhysicalQuickScreenError("closed quick physical backend cannot prepare")
        if self._identity is not None:
            raise PhysicalQuickScreenError("quick physical backend may prepare exactly once")
        contract.validate_identity(identity)
        base_identity, original_preflight, _authorization = self._reload_base_evidence()
        base, _physical = self._load_base()
        try:
            preflight = _validated_upstream_preflight_copy(
                base=base,
                identity=base_identity,
                original_preflight=original_preflight,
            )
            backend_class = build_torch_only_engineering_backend_class(
                physical=self._physical,
                device=self._engineering_device,
            )
            backend = backend_class(root=self._root, preflight=preflight, contract=base)
            backend.prepare(identity=base_identity, flags=flags)
            runtime = backend._load_runtime()
            if tuple(runtime["torch"].cuda.get_device_capability(0)) != (12, 0):
                raise PhysicalQuickScreenError("remote physical capability is not the reviewed 5070 Ti [12,0]")
        except (base.ScoreError, self._physical.PhysicalScoreError) as error:
            raise PhysicalQuickScreenError(f"quick physical pre-input prepare failed: {error}") from error
        self._identity = identity
        self._base_identity = base_identity
        self._base_preflight = preflight
        self._upstream_formal_preflight_device = contract.upstream_formal_preflight_device_payload(
            preflight["device_contract"],
        )
        self._backend = backend
        flags.remote_initialized = True

    def _require_prepared(self) -> tuple[Any, Any, Any, contract.QuickScreenIdentity, Mapping[str, object]]:
        if (
            self._closed or self._backend is None or self._base is None or self._base_identity is None
            or self._identity is None or self._base_preflight is None or self._upstream_formal_preflight_device is None
        ):
            raise PhysicalQuickScreenError("quick physical backend is not prepared")
        return self._base, self._physical, self._backend, self._identity, self._base_preflight

    def resolve_inputs(self, *, identity: contract.QuickScreenIdentity, flags: contract.RuntimeFlags) -> contract.InputAuthority:
        base, _physical, backend, prepared_identity, preflight = self._require_prepared()
        if identity is not prepared_identity:
            raise PhysicalQuickScreenError("quick resolve identity object drift")
        # The attempt has already been written by the core.  Only now do we
        # bind the stage's subject roots and invoke the no-cache parser.
        expected_subc = self._root / contract.REMOTE_EVALUATION_ROOT_RELATIVE / "sub-C"
        expected_subm = self._root / contract.REMOTE_EVALUATION_ROOT_RELATIVE / "sub-M"
        for variable, expected in (("SUBC_DATA_ROOT", expected_subc), ("SUBM_DATA_ROOT", expected_subm)):
            prior = os.environ.get(variable)
            if prior is not None and Path(prior).absolute() != expected:
                raise PhysicalQuickScreenError(f"{variable} environment substitution before quick input access")
            os.environ[variable] = str(expected)
        selected = self._expected_selected_rows(preflight)
        records: list[contract.SessionInput] = []
        for surface in contract.SURFACES:
            for row in selected[surface]:
                try:
                    parsed = backend._parse_one(asset=row, flags=flags)
                except self._physical.PhysicalScoreError as error:
                    raise PhysicalQuickScreenError(f"quick selected input parse failed: {error}") from error
                if parsed.session in backend._sessions[surface]:
                    raise PhysicalQuickScreenError("quick selected input duplicated a session")
                backend._sessions[surface][parsed.session] = parsed
                base_payload = parsed.input_record.payload()
                data = {
                    "neural": base_payload["neural_sha256"], "calibration": base_payload["calibration_m30_sha256"],
                    "target": base_payload["last_bin_target_sha256"], "mask": base_payload["last_bin_valid_mask_sha256"],
                    "n_windows": base_payload["n_windows"], "prefix": {
                        "30": base_payload["matched_prefix_row_ids_sha256s"]["30"],
                        "4": base_payload["matched_prefix_row_ids_sha256s"]["4"],
                    },
                    "point": {"30": base_payload["normalized_ols_point_carrier_sha256s"]["30"], "4": base_payload["normalized_ols_point_carrier_sha256s"]["4"]},
                    "posterior": {"30": base_payload["normalized_posterior_carrier_sha256s"]["30"], "4": base_payload["normalized_posterior_carrier_sha256s"]["4"]},
                }
                item = contract.SessionInput(
                    surface=surface, session=parsed.session, n_windows=base_payload["n_windows"],
                    neural_sha256=base_payload["neural_sha256"], calibration_m30_sha256=base_payload["calibration_m30_sha256"],
                    last_bin_target_sha256=base_payload["last_bin_target_sha256"],
                    last_bin_valid_mask_sha256=base_payload["last_bin_valid_mask_sha256"],
                    last_bin_valid_count=base_payload["last_bin_valid_count"],
                    prefix_row_ids_sha256s=data["prefix"], point_carrier_sha256s=data["point"],
                    posterior_carrier_sha256s=data["posterior"], materialization_sha256=contract._digest(contract._json(data)),
                )
                records.append(item)
                self._input_records[(surface, parsed.session)] = parsed
        expected = tuple(
            (surface, item.session)
            for surface in contract.SURFACES
            for item in contract.selected_assets_from_payload(identity.selected_assets)[surface]
        )
        if tuple((item.surface, item.session) for item in records) != expected:
            raise PhysicalQuickScreenError("quick physical selected roster/order drift")
        return contract.InputAuthority(records=tuple(records))

    @staticmethod
    def _base_cell(base: Any, cell: contract.ScoreCell) -> Any:
        if cell.mode == contract.SEALED_POINT_MODE:
            return base.ScoreCell(cell.surface, "sealed_cell_d_ols_point", cell.budget, "sealed_point_system", False)
        if cell.mode == contract.POSTERIOR_MODE:
            return base.ScoreCell(cell.surface, "posterior_mean_precision", cell.budget, "posterior_system", False)
        raise PhysicalQuickScreenError("quick physical score-cell mode drift")

    def _score_selected_cell(
        self, *, cell: contract.ScoreCell, input_payload: Mapping[str, object], flags: contract.RuntimeFlags,
    ) -> contract.CellEvidence:
        base, _physical, backend, identity, _preflight = self._require_prepared()
        base_cell = self._base_cell(base, cell)
        runtime = backend._load_runtime()
        torch, np = runtime["torch"], runtime["np"]
        system = "sealed_cell_d" if cell.mode == contract.SEALED_POINT_MODE else "posterior"
        model = backend._models.get(system)
        if model is None or model.training or not backend._all_gradients_none(model):
            raise PhysicalQuickScreenError("quick physical model eval/gradient invariant drift")
        state_before = runtime["arm_common"].state_sha256(model)
        if state_before != backend._state_at_load.get(system):
            raise PhysicalQuickScreenError("quick physical model state drifted before selected cell")
        selected_roster = tuple(
            item.session for item in contract.selected_assets_from_payload(identity.selected_assets)[cell.surface]
        )
        records = {
            item["session"]: item
            for item in input_payload["records"] if item["surface"] == cell.surface
        }
        if tuple(records) != selected_roster:
            raise PhysicalQuickScreenError("quick physical input authority roster differs from selected roster")
        first = self._input_records.get((cell.surface, selected_roster[0]))
        if first is None:
            raise PhysicalQuickScreenError("quick physical first selected session is unavailable")
        try:
            repeated = backend._repeat_probe(cell=base_cell, session=first)
        except self._physical.PhysicalScoreError as error:
            raise PhysicalQuickScreenError(f"quick repeated forward probe failed: {error}") from error
        rows: list[contract.SessionScore] = []
        for session_name in selected_roster:
            session = self._input_records.get((cell.surface, session_name))
            if session is None:
                raise PhysicalQuickScreenError("quick physical selected session disappeared")
            predictions: list[Any] = []
            targets: list[Any] = []
            digest = hashlib.sha256()
            starts = tuple(int(item) for item in session.starts)
            for offset in range(0, len(starts), self._BATCH_SIZE):
                chunk = starts[offset:offset + self._BATCH_SIZE]
                neural, behavior, calibration = backend._session_batch(session, chunk)
                output = backend._forward(cell=base_cell, session=session, neural=neural, calibration=calibration)
                valid = (behavior[:, -1, :] != -1.0).all(dim=-1)
                if not bool(valid.all().item()):
                    raise PhysicalQuickScreenError("quick score has invalid last-bin target")
                prediction = output[:, -1, :].detach().cpu().contiguous()
                target = behavior[:, -1, :].detach().cpu().contiguous()
                digest.update(prediction.numpy().tobytes())
                predictions.append(prediction)
                targets.append(target)
            if not predictions:
                raise PhysicalQuickScreenError("quick selected session emitted no prediction")
            prediction = torch.cat(predictions)
            target = torch.cat(targets)
            target_array = np.ascontiguousarray(target.numpy(), dtype=np.float32)
            mask_array = np.ascontiguousarray(np.all(target_array != -1.0, axis=1), dtype=np.uint8)
            input_record = records[session_name]
            if (
                target_array.shape != session.last_targets.shape or not np.array_equal(target_array, session.last_targets)
                or not np.array_equal(mask_array, session.last_valid_mask)
                or contract._digest(target_array.tobytes()) != input_record["last_bin_target_sha256"]
                or contract._digest(mask_array.tobytes()) != input_record["last_bin_valid_mask_sha256"]
                or int(mask_array.sum()) != input_record["last_bin_valid_count"]
            ):
                raise PhysicalQuickScreenError("quick selected governing target/mask/count binding drift")
            r2 = manual_variance_weighted_two_coordinate_r2(
                predictions=prediction,
                targets=target,
                torch=torch,
            )
            rows.append(contract.SessionScore(
                session=session_name, n_windows=len(starts), r2=float(r2), prediction_sha256=digest.hexdigest(),
                input_record_sha256=contract._digest(contract._json(input_record)),
            ))
        state_after = runtime["arm_common"].state_sha256(model)
        if state_after != state_before or model.training or not backend._all_gradients_none(model):
            raise PhysicalQuickScreenError("quick physical selected forward changed model state")
        return contract.CellEvidence(
            cell=cell,
            model_system="sealed_cell_d_checkpoint" if system == "sealed_cell_d" else "posterior_carrier_full_swa",
            model_swa_sha256=(contract.SEALED_CELL_D_SWA_SHA256 if system == "sealed_cell_d" else contract.POSTERIOR_FULL_SWA_SHA256),
            sessions=tuple(rows), input_authority_sha256=contract._digest(contract._json(input_payload)),
            model_state_before_sha256=state_before, model_state_after_sha256=state_after,
            eval_mode=True, dropout_disabled=True, gradients_none=True, finite_outputs=True,
            repeated_fixed_batch_bitwise_equal=repeated, b3s_m30_recomputed=True, no_target_sampling=True,
        )

    def score_cell(self, *, cell: contract.ScoreCell, input_payload: Mapping[str, object],
                   flags: contract.RuntimeFlags) -> contract.CellEvidence:
        return self._score_selected_cell(cell=cell, input_payload=input_payload, flags=flags)

    def reverify_after_forwards(self, *, identity: contract.QuickScreenIdentity,
                                flags: contract.RuntimeFlags) -> contract.ImplementationClosure:
        base, _physical, backend, prepared_identity, _preflight = self._require_prepared()
        if identity is not prepared_identity:
            raise PhysicalQuickScreenError("quick final identity object drift")
        # The inherited ``reverify_after_forwards`` additionally insists on
        # the full 16-cell matrix, which is correctly false for this 3+3
        # engineering route.  Reuse its held-FD/model-state checks directly,
        # while validating this route's separate 8-cell matrix below.
        try:
            runtime = backend._load_runtime()
            for held in backend._held_assets:
                held.reverify()
            for held in backend._held_roots:
                held.reverify()
            for name, model in backend._models.items():
                if (
                    runtime["arm_common"].state_sha256(model) != backend._state_at_load.get(name)
                    or model.training or not backend._all_gradients_none(model)
                ):
                    raise PhysicalQuickScreenError(f"quick physical model state/eval drift after forwards: {name}")
            base_final = base.implementation_closure(self._root).payload()
        except self._physical.PhysicalScoreError as error:
            raise PhysicalQuickScreenError(f"quick physical post-forward reverify failed: {error}") from error
        # Re-read the upstream authority after all forwards as well, so the
        # existing completed posterior/Cell-D provenance cannot be swapped in
        # between model preparation and receipt publication.
        _identity, original_preflight, _authorization = self._reload_base_evidence()
        if base_final != original_preflight["identity"]["closure"]:
            raise PhysicalQuickScreenError("base closure changed during selected quick forwards")
        if contract.upstream_formal_preflight_device_payload(original_preflight["device_contract"]) != self._upstream_formal_preflight_device:
            raise PhysicalQuickScreenError("upstream formal preflight device changed during selected forwards")
        quick_final = contract.implementation_closure(self._root)
        if quick_final.payload() != identity.closure.payload():
            raise PhysicalQuickScreenError("quick closure changed during selected forwards")
        contract._validate_runtime_flags(flags, require_all_cells=True)
        return quick_final

    def device_attestation(self) -> Mapping[str, object]:
        _base, _physical, backend, _identity, _preflight = self._require_prepared()
        runtime = backend._load_runtime()
        observed = dict(runtime["device_attestation"])
        expected = self._engineering_device.payload()
        if observed != expected:
            raise PhysicalQuickScreenError("remote engineering device changed after prepare")
        return observed

    def upstream_formal_preflight_device(self) -> Mapping[str, object]:
        self._require_prepared()
        if self._upstream_formal_preflight_device is None:
            raise PhysicalQuickScreenError("upstream formal preflight device is unavailable")
        try:
            return contract.validate_upstream_formal_preflight_device_payload(
                self._upstream_formal_preflight_device,
            )
        except contract.QuickScreenError as error:
            raise PhysicalQuickScreenError(f"upstream formal preflight device receipt drift: {error}") from error

    def posterior_final4_compatibility(self) -> Mapping[str, object]:
        """Return the exact V3 proof produced while strict-loading full SWA."""
        _base, _physical, backend, _identity, _preflight = self._require_prepared()
        value = getattr(backend, "_quick_screen_posterior_final4_compatibility", None)
        try:
            return contract.validate_posterior_final4_compatibility_payload(value)
        except contract.QuickScreenError as error:
            raise PhysicalQuickScreenError(f"authoritative Posterior final-four compatibility proof is unavailable: {error}") from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            self._backend.close()
        self._input_records.clear()


def execute_reviewed_remote(
    *, root: Path, identity: contract.QuickScreenIdentity, engineering_device: RemoteEngineeringDevice,
    execution_capability: object,
) -> dict[str, object]:
    """Private remote in-process launch seam, never called by the public CLI."""
    backend = QuickScreenPhysicalBackend(root=Path(root), engineering_device=engineering_device)
    # This is intentionally before both acceptance of the V3 capability and
    # V3 result-root reservation.  V1/V2 are read-only failure evidence, not
    # evaluation assets, and no target/model/CUDA route is entered here.
    _base, predecessor_physical = backend._load_base()
    validate_v1_failed_predecessor_directory(physical=predecessor_physical)
    validate_v2_failed_predecessor_directory(physical=predecessor_physical)
    contract._require_capability(execution_capability, identity=identity)
    artifact = contract.reserve_result_artifact(Path(root))

    def final_authorization_reverify() -> None:
        # ``reverify_after_forwards`` does the concrete descriptor reads.  This
        # callback exists to preserve the lifecycle's separate final boundary.
        if contract.implementation_closure(Path(root)).payload() != identity.closure.payload():
            raise PhysicalQuickScreenError("quick root closure changed before success publication")

    return contract.run_quick_screen_lifecycle(
        artifact=artifact, identity=identity, execution_capability=execution_capability,
        backend=backend, final_authorization_reverify=final_authorization_reverify,
    )
