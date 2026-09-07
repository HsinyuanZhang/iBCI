"""Static, no-Torch contract for posterior-marginalized Cell-D.

This file is deliberately usable by a zero-argument public CLI.  It does not
import the Cell-D model, posterior implementation, Torch, a dataset, or an
artifact helper.  Physical source/CUDA work remains behind a later
root-reviewed capability.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CELL = "POSTERIOR_MARGINALIZED_CELL_D_SEED42"
SCHEMA = "posterior_marginalized_cell_d_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTERIOR_MARGINALIZED_CELL_D_20260823.md"
WORKORDER_SHA256 = "897271e3ed4e510358a1d6a10c1529851397b65a63a8fc78449c174072a9fdfc"

SEED = 42
EPOCHS = 48
BATCH_SIZE = 32
STEPS_PER_EPOCH = 33_925
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
CHECKPOINT_EPOCHS = (44, 45, 46, 47)
BUDGETS = (4, 10, 30)
STRICT_SOURCE_SESSION_COUNT = 27
# Source-only calibration safety literal.  It is an audit/reporting boundary,
# never a clipping rule or caller-tuned threshold.
RAW_SAMPLE_ABS_SAFETY_BOUND = 1_000.0

# These two roots are prospective names only.  No public path in this route
# creates, reserves, or writes either one before a separate review.
SOURCE_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_seed42_source_smoke_v1"
FULL_TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_seed42_full_train_v1"

# PMC is allowed to use either locally compatible RTX-3090, selected by a
# root reviewer at launch time.  A profile is never inferred from an ordinal:
# the exact CVD selector, physical UUID/BDF, nominal NVML capacity, Torch
# allocator-visible byte capacity, and runtime stack travel together in the
# durable identity.  The two memory authorities are deliberately distinct.
COMPATIBLE_DEVICE_PROFILES: Mapping[str, Mapping[str, object]] = {
    "gpu0": {
        "cuda_visible_devices": "0",
        "cuda_device_order": "PCI_BUS_ID",
        "logical_device": "cuda:0",
        "uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
        "bdf": "00000000:01:00.0",
        "name": "NVIDIA GeForce RTX 3090",
        "nvidia_smi_memory_total_mib": 24_576,
        "torch_total_memory_bytes": 25_435_111_424,
        "torch_version": "2.5.1.post303",
        "torch_cuda_version": "11.8",
        "cudnn_version": 90_300,
    },
    "gpu1": {
        "cuda_visible_devices": "1",
        "cuda_device_order": "PCI_BUS_ID",
        "logical_device": "cuda:0",
        "uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
        "bdf": "00000000:03:00.0",
        "name": "NVIDIA GeForce RTX 3090",
        "nvidia_smi_memory_total_mib": 24_576,
        "torch_total_memory_bytes": 25_438_126_080,
        "torch_version": "2.5.1.post303",
        "torch_cuda_version": "11.8",
        "cudnn_version": 90_300,
    },
}


def validate_compatible_device_profile(value: Mapping[str, object]) -> dict[str, object]:
    """Return one exact reviewed local-device profile or fail closed.

    This has no CUDA dependency.  It lets identity/capability construction
    bind a root-selected idle device without changing the scientific route or
    treating a CUDA ordinal as a durable authority.
    """
    observed = dict(value) if isinstance(value, Mapping) else None
    if observed is None:
        raise PMCPlanError("PMC device profile must be a mapping")
    for profile in COMPATIBLE_DEVICE_PROFILES.values():
        if observed == dict(profile):
            return dict(profile)
    raise PMCPlanError("PMC device profile is not an exact compatible local authority")

SEALED_CELL_D_INITIALIZED_PARAMETERS = 3_510_842
SEALED_CELL_D_LAZY_KEYS = (
    "decoder.fc_id_in.0.bias",
    "decoder.fc_id_in.0.weight",
)
SEALED_OLS_T4_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
# These literal float32 moments are the sealed Cell-D ordinary point-T4
# normalizer.  Keeping the values here (rather than accepting a semantic SHA
# supplied by a caller) prevents a posterior-specific or numerically drifted
# normalizer from being relabelled as the Cell-D one.  The canonical semantic
# digest uses the exact ``side_feature_stats_sha256`` JSON domain:
# ``{"mean": [...], "std": [...]}``, sorted compact UTF-8 JSON.
SEALED_OLS_T4_MEAN_FLOAT32 = (
    0.04627712443470955,
    0.4544036388397217,
    1.3432163000106812,
    10.150517463684082,
)
SEALED_OLS_T4_STD_FLOAT32 = (
    1.126278281211853,
    1.284820556640625,
    1.2352101802825928,
    9.115250587463379,
)
SEALED_BEHAVIOR_NORMALIZER_SHA256 = "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
CANONICAL_INITIAL_STATE_SHA256 = "b0a340fe4d09eac1f2b498658d39a8a87e87f52304cad2539753ae9e040fcbd4"
CANONICAL_INITIAL_STATE_STATE_SHA256 = "65bacb85447df40ea5e03cffed03b1763d1964bfba50cbbe3d21d1614c07f2a3"

# Source direction is never reconstructed ad hoc by PMC.  Its later source
# authority must carry the approved Phase-B-v2 same-prefix recovery evidence:
# one audited corners-derived fallback on an already selected prefix row, and
# no later-row substitution.  These are literal protocol bindings, not a
# proposal to reimplement the recovery algorithm here.
THETA_RECOVERY_SCHEMA = "posterior_carrier_same_prefix_theta_recovery_v2"
THETA_RECOVERY_SEMANTICS = (
    "chronological_labelled_rewarded_trial_target_direction_radians__"
    "same_prefix_target_corners_xyxy_center_unique_canonical_8way_snap_v2"
)
THETA_FALLBACK_TOPOLOGY_SHA256 = "949431bc77bb484cd2a84e2469d2131bccfb3207655166b6921f41bad8e737c8"

# The Phase-1 logical treatment candidate was independently reviewed with
# this exact closure.  Phase-2 may add a physical/lifecycle layer, but it may
# not quietly revise the Phase-1 cache/sampling seam while claiming to extend
# it.  Every Phase-2 identity therefore carries this literal predecessor.
PHASE1_ACCEPTED_CLOSURE_SHA256 = "8a6a6fada108becb99d7651ed2f4f8c2251ef57cc4755bc47f046a868829f731"

# The PMC source adapter is a consumer of this already terminal immutable
# Posterior full-import authority.  It is not a data input and it is never
# accepted from a caller mapping: the deferred loader opens this one 0444
# body/sidecar pair through a held directory descriptor, rehydrates the
# nested Phase-B-v2 identity, and runs its semantic validator.
PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_RELATIVE = (
    "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_import_mirror_v1/"
    "source_authority.json"
)
PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256 = "26d85c5f92f353752786f75dff6c959e3475e8a9eb0920095691a14ff8c95241"

# This is an explicit non-glob closure.  It includes only route bytes and the
# actual implementation surfaces used by the later composition seam; data,
# checkpoint, authority, and result artifacts are intentionally not code
# closure paths.
IMPLEMENTATION_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/core.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/lifecycle.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/source_audit.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/physical.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/runner.py",
    "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_seed42.py",
    "tfpd_exploration/tests/test_posterior_marginalized_cell_d_v1.py",
    "tfpd_exploration/src/cell_d_equal_session_v1.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py",
    "tfpd_exploration/src/posterior_carrier_v1/full_train.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/tfpd_lane/__init__.py",
    "tfpd_exploration/src/tfpd_lane/mech_diag.py",
    "tfpd_exploration/src/tfpd_lane/pregate.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)


class PMCPlanError(RuntimeError):
    """Raised when the static PMC contract or closure drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PMCPlanError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def sealed_ols_t4_normalizer_payload() -> dict[str, list[float]]:
    """Return the exact frozen Cell-D OLS semantic payload.

    This is deliberately independent of Torch/NumPy so the dry plan can bind
    the numeric authority without importing a tensor library.  Runtime code
    additionally requires actual tensor dtype ``torch.float32`` before it
    compares values against these literals.
    """
    values = {
        "mean": list(SEALED_OLS_T4_MEAN_FLOAT32),
        "std": list(SEALED_OLS_T4_STD_FLOAT32),
    }
    if any(not math.isfinite(value) for row in values.values() for value in row):
        raise PMCPlanError("sealed OLS moment literal is non-finite")
    if any(value <= 0.0 for value in values["std"]):
        raise PMCPlanError("sealed OLS standard-deviation literal is nonpositive")
    return values


def sealed_ols_t4_normalizer_sha256(value: Mapping[str, object] | None = None) -> str:
    """Recompute the exact frozen OLS semantic SHA from numeric moments."""
    payload = sealed_ols_t4_normalizer_payload() if value is None else dict(value)
    if set(payload) != {"mean", "std"}:
        raise PMCPlanError("sealed OLS normalizer semantic schema drift")
    for name, expected in (("mean", SEALED_OLS_T4_MEAN_FLOAT32), ("std", SEALED_OLS_T4_STD_FLOAT32)):
        observed = payload[name]
        if (not isinstance(observed, (tuple, list)) or len(observed) != 4
                or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in observed)
                or tuple(float(item) for item in observed) != expected):
            raise PMCPlanError(f"sealed OLS {name} numeric literal drift")
    return _sha256(_canonical_json({"mean": list(payload["mean"]), "std": list(payload["std"])}))


if sealed_ols_t4_normalizer_sha256() != SEALED_OLS_T4_NORMALIZER_SHA256:
    raise RuntimeError("sealed OLS normalizer literals/semantic SHA drift")


@dataclass(frozen=True)
class PMCTrainingSpec:
    """The entire held Cell-D training budget, with the one side-tensor edit."""

    cell: str = CELL
    seed: int = SEED
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    steps_per_epoch: int = STEPS_PER_EPOCH
    checkpoint_epochs: tuple[int, ...] = CHECKPOINT_EPOCHS
    budgets: tuple[int, ...] = BUDGETS

    def __post_init__(self) -> None:
        _require(self.cell == CELL and self.seed == SEED, "PMC cell/seed drift")
        _require(self.epochs == EPOCHS and self.batch_size == BATCH_SIZE and self.steps_per_epoch == STEPS_PER_EPOCH,
                 "PMC fixed training budget drift")
        _require(self.checkpoint_epochs == CHECKPOINT_EPOCHS and self.budgets == BUDGETS,
                 "PMC checkpoint/budget topology drift")

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    def payload(self) -> dict[str, object]:
        return {
            "cell": self.cell,
            "seed": self.seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "budget_rotation": {
                "budgets": list(self.budgets),
                "formula": "budgets[(epoch + session_index) % 3]",
                "per_session_budget_count_over_48_epochs": 16,
            },
            "held_cell_d": {
                "canonical_initial_state_sha256": CANONICAL_INITIAL_STATE_SHA256,
                "canonical_initial_state_state_sha256": CANONICAL_INITIAL_STATE_STATE_SHA256,
                "initialized_trainable_parameters": SEALED_CELL_D_INITIALIZED_PARAMETERS,
                "uninitialized_lazy_keys": list(SEALED_CELL_D_LAZY_KEYS),
                "b3s_calibration_activity_prefix": "M30",
                "attention_heads": 2,
                "dynamic_whole_unit_dropout": "p~U(0,1), complete fused token, held",
                "optimizer": "Adam(lr=1e-4, weight_decay=0.0), held warmup/cosine schedule",
                "loss": "dense valid-bin supervised MSE",
            },
            "sole_intervention": {
                "training_side": "cached_sample_from_closed_form_posterior_then_sealed_ordinary_OLS_normalizer",
                "inference_side": "deterministic_ordinary_OLS_point_T4",
                "posterior_mean_at_inference": False,
                "posterior_normalizer": False,
                "posterior_credibility_or_attention_bias": False,
                "new_learned_parameters": 0,
            },
            "held_normalizers": {
                "ordinary_ols_t4_semantic_sha256": SEALED_OLS_T4_NORMALIZER_SHA256,
                "ordinary_ols_t4_float32_mean": list(SEALED_OLS_T4_MEAN_FLOAT32),
                "ordinary_ols_t4_float32_std": list(SEALED_OLS_T4_STD_FLOAT32),
                "behavior_semantic_sha256": SEALED_BEHAVIOR_NORMALIZER_SHA256,
            },
            "source_calibration_safety": {
                "raw_sample_abs_hard_bound": RAW_SAMPLE_ABS_SAFETY_BOUND,
                "clipping_or_posthoc_threshold_selection": False,
            },
        }


PUBLIC_SPEC = PMCTrainingSpec()


def budget_for(epoch: int, session_index: int) -> int:
    """The frozen M4/M10/M30 logical-epoch rotation, with no RNG."""
    if type(epoch) is not int or epoch < 0 or type(session_index) is not int or session_index < 0:
        raise PMCPlanError("PMC epoch/session index must be nonnegative exact ints")
    return BUDGETS[(epoch + session_index) % len(BUDGETS)]


def build_budget_schedule(*, session_count: int = STRICT_SOURCE_SESSION_COUNT) -> tuple[tuple[int, ...], ...]:
    if type(session_count) is not int or session_count < 1:
        raise PMCPlanError("PMC session count must be a positive exact int")
    schedule = tuple(
        tuple(budget_for(epoch, session_index) for session_index in range(session_count))
        for epoch in range(EPOCHS)
    )
    validate_budget_schedule(schedule, session_count=session_count)
    return schedule


def validate_budget_schedule(schedule: Sequence[Sequence[int]], *, session_count: int) -> None:
    if type(session_count) is not int or session_count < 1 or len(schedule) != EPOCHS:
        raise PMCPlanError("PMC schedule must contain exactly 48 epochs")
    for epoch, row in enumerate(schedule):
        if not isinstance(row, (tuple, list)) or len(row) != session_count:
            raise PMCPlanError("PMC schedule session axis drift")
        expected = tuple(budget_for(epoch, session_index) for session_index in range(session_count))
        if tuple(row) != expected:
            raise PMCPlanError("PMC schedule is not the exact frozen rotation")
    for session_index in range(session_count):
        assigned = [row[session_index] for row in schedule]
        if any(assigned.count(budget) != 16 for budget in BUDGETS):
            raise PMCPlanError("PMC session does not receive M4/M10/M30 exactly 16 times")


def budget_schedule_sha256(schedule: Sequence[Sequence[int]]) -> str:
    validate_budget_schedule(schedule, session_count=len(schedule[0]) if schedule else 0)
    return _sha256(_canonical_json([list(row) for row in schedule]))


def assert_fresh_prospective_roots(root: Path, *, spec_kind: str | None = None) -> dict[str, bool]:
    """Read-only freshness gate; it never creates an output directory.

    The smoke and full lifecycles have independent canonical roots.  A
    completed smoke must not syntactically prevent the later full route from
    reserving its *other* fresh root.  Callers that have not chosen a spec
    (for example the dry plan) still receive the conservative all-root check.
    """
    root = Path(root).absolute()
    selected = {
        None: (SOURCE_SMOKE_ROOT_RELATIVE, FULL_TRAIN_ROOT_RELATIVE),
        "source_smoke": (SOURCE_SMOKE_ROOT_RELATIVE,),
        "full_train": (FULL_TRAIN_ROOT_RELATIVE,),
    }.get(spec_kind)
    if selected is None:
        raise PMCPlanError("PMC prospective root spec kind drift")
    result: dict[str, bool] = {}
    for relative in selected:
        candidate = root / relative
        if candidate.exists() or candidate.is_symlink():
            raise PMCPlanError(f"PMC prospective result root is not fresh: {relative}")
        result[relative] = True
    return result


def implementation_closure(root: Path) -> dict[str, object]:
    """Descriptor-free, data-free hash of the explicit code closure.

    This review-stage helper deliberately reads only source/doc/test files.
    It does not discover an artifact root, open a dataset, or import Torch.
    """
    root = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_CLOSURE:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise PMCPlanError(f"PMC closure path is missing or aliased: {relative}")
        hashes[relative] = _sha256(candidate.read_bytes())
    if hashes.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise PMCPlanError("PMC work-order SHA drift")
    payload = {
        "paths": list(IMPLEMENTATION_CLOSURE),
        "sha256_by_path": hashes,
        "predecessor_phase1_closure_sha256": PHASE1_ACCEPTED_CLOSURE_SHA256,
        "phase_b_v2_import_source_authority": {
            "relative_path": PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_RELATIVE,
            "body_sha256": PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256,
        },
    }
    return {**payload, "closure_sha256": _sha256(_canonical_json(payload))}


def validate_implementation_closure(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "paths", "sha256_by_path", "predecessor_phase1_closure_sha256",
        "phase_b_v2_import_source_authority", "closure_sha256",
    }:
        raise PMCPlanError("PMC closure schema drift")
    paths, hashes = value.get("paths"), value.get("sha256_by_path")
    if paths != list(IMPLEMENTATION_CLOSURE) or not isinstance(hashes, Mapping) or set(hashes) != set(IMPLEMENTATION_CLOSURE):
        raise PMCPlanError("PMC closure path topology drift")
    if any(not _is_sha256(hashes[path]) for path in IMPLEMENTATION_CLOSURE):
        raise PMCPlanError("PMC closure leaf digest drift")
    if hashes[WORKORDER_RELATIVE] != WORKORDER_SHA256:
        raise PMCPlanError("PMC closure work-order binding drift")
    if value.get("predecessor_phase1_closure_sha256") != PHASE1_ACCEPTED_CLOSURE_SHA256:
        raise PMCPlanError("PMC Phase-1 accepted closure binding drift")
    expected_import = {
        "relative_path": PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_RELATIVE,
        "body_sha256": PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256,
    }
    if value.get("phase_b_v2_import_source_authority") != expected_import:
        raise PMCPlanError("PMC Phase-B-v2 imported source-authority binding drift")
    payload = {
        "paths": list(paths),
        "sha256_by_path": dict(hashes),
        "predecessor_phase1_closure_sha256": PHASE1_ACCEPTED_CLOSURE_SHA256,
        "phase_b_v2_import_source_authority": expected_import,
    }
    digest = _sha256(_canonical_json(payload))
    if value.get("closure_sha256") != digest:
        raise PMCPlanError("PMC closure aggregate digest drift")
    return {**payload, "closure_sha256": digest}


def dry_plan() -> dict[str, Any]:
    """Inspection-only plan used by the public static CLI."""
    schedule = build_budget_schedule()
    return {
        "schema": SCHEMA,
        "cell": CELL,
        "status": "DRY_NO_DATA_NO_NWB_NO_CHECKPOINT_NO_CUDA_NO_GPU_NO_WRITE_NO_LAUNCH",
        "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
        "training_spec": PUBLIC_SPEC.payload(),
        "phase1_accepted_closure_sha256": PHASE1_ACCEPTED_CLOSURE_SHA256,
        "schedule_sha256": budget_schedule_sha256(schedule),
        "source_only": True,
        "inference": {
            "carrier": "deterministic ordinary OLS point T4",
            "posterior_sample_or_covariance_or_credibility_exposed": False,
        },
        "prospective_result_roots": {
            "source_smoke": SOURCE_SMOKE_ROOT_RELATIVE,
            "full_train": FULL_TRAIN_ROOT_RELATIVE,
            "created_by_dry_cli": False,
        },
        "physical_execution": "requires a later root-reviewed capability; not implemented by this CLI",
    }
