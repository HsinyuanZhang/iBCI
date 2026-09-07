"""CPU-only mathematical oracle for the A1 hidden-space carrier adapter.

This module is intentionally *not* a production model, trainer, scorer, runner,
or receipt writer.  It uses synthetic CPU tensors only and exists to make the
following narrow facts executable before a production integration is considered:

* the new hidden carrier path is ``P(T4)`` with a no-bias, directly-zeroed
  ``[512, 4]`` projection;
* a zero carrier port is bit-identical to the activity-only W path, including
  task-only source-training updates of all shared parameters;
* the 2x2 reporting matrix retains ``H/Z4`` as a structural alias of ``W/Z4``;
* reuse of the sealed A2 W cells is denied unless every frozen binding matches.

The reference deliberately has no filesystem or dataset access.  In particular,
it does not load NWB files, A2 checkpoints, receipt files, or production code,
and it does not mint a receipt.  ``torch.cuda`` is not used: all tensor and
parameter inputs are required to be CPU ``float32`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# These dimensions are deliberately fixed by the A1 contract.  Permitting a
# convenient smaller synthetic width would weaken the very shape contract that
# the CPU gate is meant to protect.
HIDDEN_DIM = 512
SIDE_DIM = 4
DEFAULT_PREDICTION_DIM = 3
PILOT_SEED = 42
PILOT_SESSION_COUNT = 6
PILOT_MEAN_DELTA_FLOOR = 0.03


class A1HiddenAdapterContractError(ValueError):
    """Raised when an A1 reference or frozen-contract invariant is violated."""


@dataclass(frozen=True)
class HiddenCarrierProjectionSpec:
    """Declarative contract for the only new A1 model family component."""

    hidden_dim: int = HIDDEN_DIM
    side_dim: int = SIDE_DIM
    has_bias: bool = False
    initialization: str = "direct_zero_no_rng"
    attach_point: str = "after_fc_in_before_student_decoder"


HIDDEN_T4_PROJECTION_SPEC = HiddenCarrierProjectionSpec()

# The teacher is intentionally declarative here: this reference owns only the
# trainable student-side algebra and never constructs, loads, or mutates a
# teacher module.  A production receipt must bind the real frozen teacher under
# the A2 reuse evidence instead.
SOURCE_TRAINING_SEMANTICS: Mapping[str, object] = {
    "teacher_frozen": True,
    "loss_mode": "task_only",
    "student_joint_trainables": (
        "activity_identity.weight",
        "fc_in.weight",
        "fc_in.bias",
        "decoder.weight",
        "decoder.bias",
        "carrier_projection.weight",
    ),
    "target_backward_gradients": False,
    "target_weight_updates": False,
}


def validate_hidden_carrier_projection_spec(spec: HiddenCarrierProjectionSpec) -> None:
    """Reject a projection specification that is not the exact A1 P contract."""

    if spec.hidden_dim != HIDDEN_DIM or spec.side_dim != SIDE_DIM:
        raise A1HiddenAdapterContractError(
            "A1 P must have exact weight shape [512, 4]; "
            f"received [{spec.hidden_dim}, {spec.side_dim}]"
        )
    if spec.has_bias:
        raise A1HiddenAdapterContractError("A1 P forbids a bias term")
    if spec.initialization != "direct_zero_no_rng":
        raise A1HiddenAdapterContractError(
            "A1 P must use direct_zero_no_rng initialization, not a sampled initializer"
        )
    if spec.attach_point != "after_fc_in_before_student_decoder":
        raise A1HiddenAdapterContractError(
            "A1 P must attach after fc_in and before the student decoder"
        )


@dataclass(frozen=True)
class A1MatrixCell:
    """One logical cell in the minimal A1 W/H x Z4/T4 matrix.

    ``H/Z4`` is intentionally a logical reporting cell rather than a new run:
    it aliases the exact W/Z4 cell.  This keeps the interaction algebra explicit
    without creating a duplicate control family.
    """

    key: str
    regime: str
    arm: str
    provenance: str
    structural_alias_of: str | None = None
    is_new_family: bool = False


A1_MATRIX_CELLS: tuple[A1MatrixCell, ...] = (
    A1MatrixCell(
        key="W/Z4",
        regime="W",
        arm="Z4",
        provenance="sealed_a2_reuse",
    ),
    A1MatrixCell(
        key="W/T4",
        regime="W",
        arm="T4",
        provenance="sealed_a2_reuse",
    ),
    A1MatrixCell(
        key="H/Z4",
        regime="H",
        arm="Z4",
        provenance="structural_alias",
        structural_alias_of="W/Z4",
    ),
    A1MatrixCell(
        key="H/T4",
        regime="H",
        arm="T4",
        provenance="new_hidden_adapter",
        is_new_family=True,
    ),
)


def matrix_cells_by_key(
    cells: Sequence[A1MatrixCell] = A1_MATRIX_CELLS,
) -> dict[str, A1MatrixCell]:
    """Return a unique-key view of a matrix definition."""

    by_key = {cell.key: cell for cell in cells}
    if len(by_key) != len(cells):
        raise A1HiddenAdapterContractError("A1 matrix contains duplicate logical cell keys")
    return by_key


def validate_minimal_a1_matrix(cells: Sequence[A1MatrixCell] = A1_MATRIX_CELLS) -> None:
    """Enforce the 2x2 topology and the single-new-family attribution design."""

    by_key = matrix_cells_by_key(cells)
    expected_keys = {"W/Z4", "W/T4", "H/Z4", "H/T4"}
    if set(by_key) != expected_keys:
        raise A1HiddenAdapterContractError(
            "A1 must contain exactly W/Z4, W/T4, H/Z4, H/T4; "
            f"received {sorted(by_key)}"
        )

    for key in ("W/Z4", "W/T4"):
        cell = by_key[key]
        if cell.provenance != "sealed_a2_reuse" or cell.is_new_family:
            raise A1HiddenAdapterContractError(
                f"{key} must be a non-new sealed A2 reuse cell"
            )

    h_z4 = by_key["H/Z4"]
    if (
        h_z4.provenance != "structural_alias"
        or h_z4.structural_alias_of != "W/Z4"
        or h_z4.is_new_family
    ):
        raise A1HiddenAdapterContractError(
            "H/Z4 must be an exact structural alias of W/Z4, not a separate H family"
        )

    h_t4 = by_key["H/T4"]
    if h_t4.provenance != "new_hidden_adapter" or not h_t4.is_new_family:
        raise A1HiddenAdapterContractError(
            "H/T4 must be the only new A1 hidden-adapter family"
        )

    if sum(cell.is_new_family for cell in by_key.values()) != 1:
        raise A1HiddenAdapterContractError("A1 permits exactly one new model family: H/T4")


# Exact A2 reuse bindings, transcribed from the immutable A2 terminal aggregate,
# official preflight, and source/domain receipts.  This is deliberately a
# data-free, literal contract.  A future production integration must obtain the
# values from its own bound receipt and pass them to validate_a2_reuse_evidence;
# this reference does not open any artifact itself.
A2_TERMINAL_AGGREGATE_SHA256 = (
    "5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc"
)
A2_OFFICIAL_PREFLIGHT_SHA256 = (
    "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd"
)
A2_CONTRACT_SHA256 = "f8d2f1f9e2420423584f18e667df27bc209c5ae5d1730232d703886244ea5cb2"
A2_CONFIG_SHA256 = "68aa9b599b5d30c690af8639ece6f3d2e7d63e47c5ef3f7fedcd853cae6ed73c"
A2_IMPLEMENTATION_BINDINGS_SHA256 = (
    "7e46115fc3bafa1010d454c80215d6cf99987973464b0720997366d92b2887b3"
)
A2_TEACHER_CHECKPOINT = (
    "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
A2_TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
A2_TRAIN_VAL_MANIFEST = "configs/subc_co_27_6_strict_train_val_manifest.json"
A2_TRAIN_VAL_MANIFEST_SHA256 = (
    "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
)
A2_T4_NORMALIZER_SHA256 = (
    "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
)
A2_EPOCH_WINDOW = tuple(range(5, 13))

# The source checkpoint *bundle* hashes bind the exact epoch 5--12 assets for
# each reusable A2 W cell.  A real integration must independently validate the
# underlying eight checkpoint hashes against its sealed A2 receipt; this
# reference deliberately stores only this small ledger and never reads those
# artifacts itself.
A2_SEALED_EPOCH_BUNDLE_SHA256: Mapping[str, Mapping[int, str]] = {
    "source_z4": {
        42: "c1d5482640eea00b649810b0107527cdcfb2c1d0b5c0cbbb886887a21eb27a45",
        43: "dfe39ea79db1c4a856e4932d18ddfd26542688f67950fc1ec53ee1cf399e6085",
        44: "25ffc59a551f83d5b7c146a9a3f8df98488550beeda45780e70e563ec3229c01",
    },
    "source_t4": {
        42: "16daf346541c7594ec9bb097ddae369899dc59f16bfe4f5bfd2e993565409260",
        43: "dbcc34ada3aa616a933d55840f7f5c31687f8fe02c993dae798fd6a3f0711e65",
        44: "200c39adac1e7d3a37865b833ce15c582b6ec9bb50ee8d9a2d11e9191562bab2",
    },
}

A2_SOURCE_TRAIN_SESSIONS: tuple[str, ...] = (
    "sub-C_ses-CO-20131003",
    "sub-C_ses-CO-20131022",
    "sub-C_ses-CO-20131023",
    "sub-C_ses-CO-20131031",
    "sub-C_ses-CO-20131101",
    "sub-C_ses-CO-20131203",
    "sub-C_ses-CO-20131204",
    "sub-C_ses-CO-20131219",
    "sub-C_ses-CO-20131220",
    "sub-C_ses-CO-20150309",
    "sub-C_ses-CO-20150311",
    "sub-C_ses-CO-20150312",
    "sub-C_ses-CO-20150313",
    "sub-C_ses-CO-20150319",
    "sub-C_ses-CO-20150629",
    "sub-C_ses-CO-20150630",
    "sub-C_ses-CO-20150701",
    "sub-C_ses-CO-20150703",
    "sub-C_ses-CO-20150706",
    "sub-C_ses-CO-20150707",
    "sub-C_ses-CO-20150708",
    "sub-C_ses-CO-20150709",
    "sub-C_ses-CO-20150710",
    "sub-C_ses-CO-20150713",
    "sub-C_ses-CO-20150714",
    "sub-C_ses-CO-20150715",
    "sub-C_ses-CO-20150716",
)


def expected_a2_reuse_evidence(*, reuse_seed: int = PILOT_SEED) -> dict[str, object]:
    """Return a JSON-shaped copy of every frozen A2 binding required for reuse.

    The returned object is purposely a fresh dictionary/list tree so callers and
    tests cannot mutate module-level contract state.  Extra fields in an actual
    receipt are allowed, but every field below must be present with this exact
    type and value.
    """

    if type(reuse_seed) is not int or reuse_seed not in A2_SEALED_EPOCH_BUNDLE_SHA256["source_z4"]:
        raise A1HiddenAdapterContractError(
            "A2 reuse seed must be exactly one of 42, 43, 44"
        )
    return {
        "reuse_seed": reuse_seed,
        "terminal_aggregate_sha256": A2_TERMINAL_AGGREGATE_SHA256,
        "official_preflight_sha256": A2_OFFICIAL_PREFLIGHT_SHA256,
        "a2_contract_sha256": A2_CONTRACT_SHA256,
        "a2_config_sha256": A2_CONFIG_SHA256,
        "a2_implementation_bindings_sha256": A2_IMPLEMENTATION_BINDINGS_SHA256,
        "teacher": {
            "checkpoint": A2_TEACHER_CHECKPOINT,
            "sha256": A2_TEACHER_SHA256,
            "frozen_during_source_training": True,
        },
        "train_val_manifest": {
            "path": A2_TRAIN_VAL_MANIFEST,
            "sha256": A2_TRAIN_VAL_MANIFEST_SHA256,
        },
        "backbone": {
            "variant": "B3S",
            "side_dim": SIDE_DIM,
            "identity_mode": "calibrated",
            "decoder_mode": "coupled",
        },
        "sealed_w_cells": {
            "W/Z4": {
                "source_arm": "source_z4",
                "source_seed": reuse_seed,
                "side_feature_group": "z4",
                "variant": "B3S",
                "side_dim": SIDE_DIM,
                "source_checkpoint_sha256_bundle_sha256": A2_SEALED_EPOCH_BUNDLE_SHA256[
                    "source_z4"
                ][reuse_seed],
            },
            "W/T4": {
                "source_arm": "source_t4",
                "source_seed": reuse_seed,
                "side_feature_group": "t4",
                "variant": "B3S",
                "side_dim": SIDE_DIM,
                "source_checkpoint_sha256_bundle_sha256": A2_SEALED_EPOCH_BUNDLE_SHA256[
                    "source_t4"
                ][reuse_seed],
            },
        },
        "t4_normalizer": {
            "authority": "strict_subc_source_train_27_only",
            "value_sha256": A2_T4_NORMALIZER_SHA256,
            "target_domain_refit_forbidden": True,
            "target_domain_refit_performed": False,
        },
        "source_train_roster": list(A2_SOURCE_TRAIN_SESSIONS),
        "m30_and_chronology": {
            "activity_calibration_trials": 30,
            "side_feature_label_pool_trials": 30,
            "evaluation_start_trial_index": 30,
            "calibration_selection": "chronological_first_n",
            "selection_mode": "first",
            "trial_result_filter": "R",
        },
        "query_policy": {
            "activity_calibration_trials": 30,
            "backward_gradients": False,
            "bin_size_ms": 20,
            "decoder_weight_updates": False,
            "evaluation_start_trial_index": 30,
            "query_rule": (
                "usable rewarded trials[30:] only; each 50-bin query window is "
                "contained in a trial strictly after chronological rewarded trial 30"
            ),
            "selection_mode": "first",
            "side_feature_label_pool_trials": 30,
            "target_direction_labels_used_for_carrier": True,
            "target_session_carrier_fit_performed": True,
            "target_velocity_labels_used_for_weight_updates": False,
            "trial_length_bins": 100,
            "trial_result_filter": "R",
            "window_size_bins": 50,
        },
        "epoch_selection": {
            "total_epochs": 12,
            "epoch_window": list(A2_EPOCH_WINDOW),
            "no_early_stopping": True,
            "epoch_score_rule": "unweighted mean session R2 over exactly source epochs 5..12",
        },
        "source_training": {"loss_mode": "task_only"},
        "data_isolation": {
            "formal_subc_test_nwb_opened": False,
            "no_test_files_evaluated": True,
        },
        "target_session_updates": {
            "backward_gradients": False,
            "decoder_weight_updates": False,
            "target_velocity_labels_used_for_weight_updates": False,
            "target_session_carrier_fit_performed": True,
            "target_direction_labels_used_for_carrier": True,
        },
    }


_MISSING = object()


def _first_exact_difference(expected: object, actual: object, path: str) -> str | None:
    """Return the first strict JSON-shaped mismatch, otherwise ``None``.

    Bool/int type equality is deliberately not accepted (``False != 0`` here),
    because a receipt's safety flags must not be silently coerced.
    """

    if type(expected) is not type(actual):
        return (
            f"{path}: expected type {type(expected).__name__}, "
            f"received {type(actual).__name__}"
        )
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        for key, expected_value in expected.items():
            actual_value = actual.get(key, _MISSING)
            child_path = f"{path}.{key}"
            if actual_value is _MISSING:
                return f"{child_path}: required field is missing"
            mismatch = _first_exact_difference(expected_value, actual_value, child_path)
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(expected) != len(actual):
            return f"{path}: expected length {len(expected)}, received {len(actual)}"
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            mismatch = _first_exact_difference(expected_value, actual_value, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, received {actual!r}"
    return None


def validate_a2_reuse_evidence(evidence: Mapping[str, object]) -> None:
    """Deny sealed-A2 reuse if any frozen binding differs.

    This is a validator only: it neither reads the immutable A2 artifacts nor
    creates a new receipt.  A production integration must supply values already
    captured by its own immutable bound receipt.
    """

    if not isinstance(evidence, Mapping):
        raise A1HiddenAdapterContractError("A2 reuse evidence must be a mapping")
    reuse_seed = evidence.get("reuse_seed", _MISSING)
    if type(reuse_seed) is not int:
        raise A1HiddenAdapterContractError(
            "A2 reuse denied: a2_reuse.reuse_seed must be an integer"
        )
    try:
        expected = expected_a2_reuse_evidence(reuse_seed=reuse_seed)
    except A1HiddenAdapterContractError as error:
        raise A1HiddenAdapterContractError(f"A2 reuse denied: {error}") from error
    mismatch = _first_exact_difference(expected, evidence, "a2_reuse")
    if mismatch is not None:
        raise A1HiddenAdapterContractError(f"A2 reuse denied: {mismatch}")


def a1_development_receipt_schema() -> dict[str, object]:
    """Describe, but never write, the future non-official A1 development receipt.

    The schema is intentionally descriptive rather than a writer API.  See the
    companion contract for the required O_EXCL/fsync/0444/sidecar lifecycle.
    """

    return {
        "schema_version": 1,
        "receipt_kind": "a1_hidden_space_carrier_development_pilot",
        "non_authorizing_status": "DEVELOPMENT_ROUTING_ONLY_NO_GPU_AUTHORITY",
        "required_top_level_fields": [
            "schema_version",
            "receipt_kind",
            "status",
            "a2_reuse",
            "matrix",
            "source_training_semantics",
            "target_session_policy",
            "pilot_routing",
            "immutable_integrity",
        ],
        "a2_reuse_validator": "validate_a2_reuse_evidence",
        "matrix_required_cells": ["W/Z4", "W/T4", "H/Z4", "H/T4"],
        "integrity_fields": {
            "body_write": "O_CREAT|O_EXCL; fsync; chmod 0444",
            "sidecar": "<body>.sha256 written separately with O_CREAT|O_EXCL; fsync; chmod 0444",
            "acceptance": "body and sidecar must both exist, be read-only, and match exactly",
        },
        "prohibited": [
            "official_receipt_minting",
            "formal_test_data_access",
            "target_backward_or_weight_update",
        ],
    }


def _require_cpu_float32(name: str, value: Tensor, *, ndim: int | None = None) -> None:
    if not isinstance(value, Tensor):
        raise A1HiddenAdapterContractError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise A1HiddenAdapterContractError(f"{name} must stay on CPU, got {value.device}")
    if value.dtype != torch.float32:
        raise A1HiddenAdapterContractError(
            f"{name} must be float32 for the CPU bit-parity oracle, got {value.dtype}"
        )
    if ndim is not None and value.ndim != ndim:
        raise A1HiddenAdapterContractError(
            f"{name} must have rank {ndim}, got shape {tuple(value.shape)}"
        )


def _require_cpu_module(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.device.type != "cpu":
            raise A1HiddenAdapterContractError(
                f"reference parameter {name} must stay on CPU, got {parameter.device}"
            )
        if parameter.dtype != torch.float32:
            raise A1HiddenAdapterContractError(
                f"reference parameter {name} must be float32, got {parameter.dtype}"
            )


def _validate_side_inputs(x: Tensor, z4: Tensor, t4: Tensor | None = None) -> None:
    _require_cpu_float32("x", x, ndim=3)
    _require_cpu_float32("z4", z4, ndim=3)
    if x.shape[:2] != z4.shape[:2] or x.shape[-1] != HIDDEN_DIM or z4.shape[-1] != SIDE_DIM:
        raise A1HiddenAdapterContractError(
            "expected x=[B,N,512] and z4=[B,N,4] with matching B,N; "
            f"got x={tuple(x.shape)}, z4={tuple(z4.shape)}"
        )
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise A1HiddenAdapterContractError("A1 reference requires B >= 1 and N >= 1")
    if t4 is not None:
        _require_cpu_float32("t4", t4, ndim=3)
        if t4.shape != z4.shape:
            raise A1HiddenAdapterContractError(
                "T4 carrier port must have the same [B,N,4] shape as Z4; "
                f"got t4={tuple(t4.shape)}, z4={tuple(z4.shape)}"
            )


class ZeroInitNoBiasCarrierProjection(nn.Module):
    """The exact ``P: R^4 -> R^512`` A1 hidden carrier projection.

    ``nn.Linear`` is intentionally not used here because its default constructor
    initializes a weight (and normally a bias) by sampling from the global RNG.
    Creating the zero parameter directly makes the no-RNG rule inspectable.
    """

    def __init__(self) -> None:
        super().__init__()
        validate_hidden_carrier_projection_spec(HIDDEN_T4_PROJECTION_SPEC)
        self.weight = nn.Parameter(torch.zeros((HIDDEN_DIM, SIDE_DIM), dtype=torch.float32))
        # A concrete None attribute makes the no-bias contract cheap to inspect
        # without registering a second parameter.
        self.bias: None = None

    def forward(self, t4: Tensor) -> Tensor:
        _require_cpu_float32("t4", t4, ndim=3)
        if t4.shape[-1] != SIDE_DIM:
            raise A1HiddenAdapterContractError(
                f"P expects T4 width {SIDE_DIM}, got shape {tuple(t4.shape)}"
            )
        _require_cpu_module(self)
        return F.linear(t4, self.weight, bias=None)


class WAddReference(nn.Module):
    """Synthetic W-add oracle: ``fc_in(x + E_A(Z4))`` then student decoding.

    ``activity_identity`` is a minimal trainable stand-in for the *already
    matched* A2 B3S Z4/activity identity path ``E_A``.  It is not a substitute
    for A2 feature construction or normalisation; those remain frozen receipt
    bindings and are never recomputed here.
    """

    def __init__(self, *, prediction_dim: int = DEFAULT_PREDICTION_DIM) -> None:
        super().__init__()
        if prediction_dim < 1:
            raise A1HiddenAdapterContractError("prediction_dim must be positive")
        self.prediction_dim = int(prediction_dim)
        # E_A is trainable in source training and maps the matched four-column
        # activity identity representation into the fc_in hidden width.
        self.activity_identity = nn.Linear(SIDE_DIM, HIDDEN_DIM, bias=False)
        self.fc_in = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True)
        self.decoder = nn.Linear(HIDDEN_DIM, self.prediction_dim, bias=True)

    def activity_embedding(self, z4: Tensor) -> Tensor:
        _require_cpu_float32("z4", z4, ndim=3)
        if z4.shape[-1] != SIDE_DIM or z4.shape[0] < 1 or z4.shape[1] < 1:
            raise A1HiddenAdapterContractError(
                "E_A expects Z4 shape [B,N,4] with B,N >= 1; "
                f"got {tuple(z4.shape)}"
            )
        _require_cpu_module(self)
        return self.activity_identity(z4)

    def hidden_from_activity_embedding(self, x: Tensor, e_a: Tensor) -> Tensor:
        _require_cpu_float32("x", x, ndim=3)
        _require_cpu_float32("E_A", e_a, ndim=3)
        if x.shape != e_a.shape or x.shape[-1] != HIDDEN_DIM:
            raise A1HiddenAdapterContractError(
                "fc_in requires x and E_A to have identical [B,N,512] shapes; "
                f"got x={tuple(x.shape)}, E_A={tuple(e_a.shape)}"
            )
        _require_cpu_module(self)
        return self.fc_in(x + e_a)

    def hidden(self, x: Tensor, z4: Tensor) -> Tensor:
        _validate_side_inputs(x, z4)
        return self.hidden_from_activity_embedding(x, self.activity_embedding(z4))

    def forward(self, x: Tensor, z4: Tensor) -> Tensor:
        return self.decoder(self.hidden(x, z4))


class HAddReference(WAddReference):
    """Synthetic H-add oracle: ``fc_in(x + E_A(Z4)) + P(T4)`` then decoding."""

    def __init__(self, *, prediction_dim: int = DEFAULT_PREDICTION_DIM) -> None:
        super().__init__(prediction_dim=prediction_dim)
        # This must be the final construction action.  It creates a direct zero
        # tensor only, so H construction advances RNG exactly as W construction.
        self.carrier_projection = ZeroInitNoBiasCarrierProjection()

    def hidden(self, x: Tensor, z4: Tensor, t4: Tensor) -> Tensor:  # type: ignore[override]
        _validate_side_inputs(x, z4, t4)
        base_hidden = self.hidden_from_activity_embedding(x, self.activity_embedding(z4))
        # Do not special-case Z4 here: retaining the zero-valued P computation is
        # what proves P.grad is an exact zero tensor under task-only training.
        return base_hidden + self.carrier_projection(t4)

    def forward(self, x: Tensor, z4: Tensor, t4: Tensor) -> Tensor:  # type: ignore[override]
        return self.decoder(self.hidden(x, z4, t4))


@dataclass(frozen=True)
class ConstructorRNGAudit:
    """A non-mutating report for the W/H construction-stream comparison."""

    w_rng_state_after: Tensor
    h_rng_state_after: Tensor

    @property
    def passes(self) -> bool:
        return torch.equal(self.w_rng_state_after, self.h_rng_state_after)


def paired_references(
    *, seed: int = 20260813, prediction_dim: int = DEFAULT_PREDICTION_DIM
) -> tuple[WAddReference, HAddReference, ConstructorRNGAudit]:
    """Build independently initialized W/H references with the same RNG stream.

    The caller's global CPU RNG state is restored before return.  Thus the
    returned models can be used in a test without turning model construction
    itself into an unrelated source of random-number drift.
    """

    if not isinstance(seed, int):
        raise A1HiddenAdapterContractError("seed must be an integer")
    rng_before = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        w_model = WAddReference(prediction_dim=prediction_dim)
        w_after = torch.random.get_rng_state()

        torch.manual_seed(seed)
        h_model = HAddReference(prediction_dim=prediction_dim)
        h_after = torch.random.get_rng_state()
    finally:
        torch.random.set_rng_state(rng_before)

    audit = ConstructorRNGAudit(
        w_rng_state_after=w_after.clone(), h_rng_state_after=h_after.clone()
    )
    if not audit.passes:
        raise A1HiddenAdapterContractError(
            "H construction consumed RNG beyond W construction; P must be directly zero initialized"
        )
    assert_shared_parameters_bit_equal(w_model, h_model)
    return w_model, h_model, audit


def shared_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Names that must remain bit-identical between W and H/Z4 reference paths."""

    return tuple(
        name
        for name, _ in model.named_parameters()
        if not name.startswith("carrier_projection.")
    )


def _parameters_by_name(model: nn.Module, *, include_adapter: bool) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if include_adapter or not name.startswith("carrier_projection.")
    }


def assert_shared_parameters_bit_equal(w_model: WAddReference, h_model: HAddReference) -> None:
    """Require exact shared parameter values, not merely numerical closeness."""

    w_params = _parameters_by_name(w_model, include_adapter=False)
    h_params = _parameters_by_name(h_model, include_adapter=False)
    if tuple(w_params) != tuple(h_params):
        raise A1HiddenAdapterContractError(
            f"W/H shared parameter names differ: {tuple(w_params)} != {tuple(h_params)}"
        )
    for name, w_parameter in w_params.items():
        h_parameter = h_params[name]
        if not torch.equal(w_parameter.detach(), h_parameter.detach()):
            raise A1HiddenAdapterContractError(f"shared parameter drift at {name}")


def assert_shared_gradients_bit_equal(w_model: WAddReference, h_model: HAddReference) -> None:
    """Require task-only shared gradients to be bit-identical between W and H/Z4."""

    w_params = _parameters_by_name(w_model, include_adapter=False)
    h_params = _parameters_by_name(h_model, include_adapter=False)
    if tuple(w_params) != tuple(h_params):
        raise A1HiddenAdapterContractError("cannot compare gradients with different shared parameters")
    for name, w_parameter in w_params.items():
        h_parameter = h_params[name]
        if (w_parameter.grad is None) != (h_parameter.grad is None):
            raise A1HiddenAdapterContractError(f"shared gradient presence differs at {name}")
        if w_parameter.grad is not None and not torch.equal(w_parameter.grad, h_parameter.grad):
            raise A1HiddenAdapterContractError(f"shared gradient drift at {name}")


def assert_projection_grad_exact_zero(model: HAddReference) -> None:
    """Require a materialized, exactly-zero P gradient for an H/Z4 alias batch."""

    gradient = model.carrier_projection.weight.grad
    if gradient is None:
        raise A1HiddenAdapterContractError("P.grad is absent; the zero-carrier path was bypassed")
    if not torch.equal(gradient, torch.zeros_like(gradient)):
        raise A1HiddenAdapterContractError("P.grad must be exact zero for the Z4 alias path")


def assert_z4_carrier_port_exact_zero(carrier_port: Tensor) -> None:
    """Reject a purported H/Z4 alias that sends nonzero values into P."""

    _require_cpu_float32("H/Z4 carrier port", carrier_port, ndim=3)
    if carrier_port.shape[-1] != SIDE_DIM:
        raise A1HiddenAdapterContractError(
            f"H/Z4 carrier port must end in width {SIDE_DIM}, got {tuple(carrier_port.shape)}"
        )
    if torch.count_nonzero(carrier_port).item() != 0:
        raise A1HiddenAdapterContractError(
            "H/Z4 must be a structural alias of W/Z4: its P carrier port must be exact zero"
        )


def assert_z4_alias_bit_equal(
    w_model: WAddReference,
    h_model: HAddReference,
    *,
    x: Tensor,
    z4: Tensor,
    carrier_port: Tensor,
) -> None:
    """Verify forward parity for the explicit H/Z4 structural-alias control."""

    assert_z4_carrier_port_exact_zero(carrier_port)
    w_hidden = w_model.hidden(x, z4)
    h_hidden = h_model.hidden(x, z4, carrier_port)
    if not torch.equal(w_hidden, h_hidden):
        raise A1HiddenAdapterContractError("H/Z4 hidden output drifted from W/Z4")
    w_prediction = w_model.decoder(w_hidden)
    h_prediction = h_model.decoder(h_hidden)
    if not torch.equal(w_prediction, h_prediction):
        raise A1HiddenAdapterContractError("H/Z4 prediction drifted from W/Z4")


def _clone_optimizer_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_optimizer_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_value(child) for child in value)
    if isinstance(value, list):
        return [_clone_optimizer_value(child) for child in value]
    return value


def optimizer_state_by_parameter_name(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    include_adapter: bool = False,
) -> dict[str, dict[str, Any]]:
    """Create a stable, parameter-name keyed optimizer-state snapshot."""

    parameters = _parameters_by_name(model, include_adapter=include_adapter)
    return {
        name: _clone_optimizer_value(dict(optimizer.state.get(parameter, {})))
        for name, parameter in parameters.items()
    }


def _assert_nested_exact(left: Any, right: Any, path: str) -> None:
    if type(left) is not type(right):
        raise A1HiddenAdapterContractError(
            f"optimizer state type drift at {path}: {type(left).__name__} != {type(right).__name__}"
        )
    if isinstance(left, Tensor):
        if not torch.equal(left, right):
            raise A1HiddenAdapterContractError(f"optimizer tensor state drift at {path}")
        return
    if isinstance(left, Mapping):
        if tuple(left) != tuple(right):
            raise A1HiddenAdapterContractError(f"optimizer state keys differ at {path}")
        for key, value in left.items():
            _assert_nested_exact(value, right[key], f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise A1HiddenAdapterContractError(f"optimizer state length drift at {path}")
        for index, (value, other) in enumerate(zip(left, right)):
            _assert_nested_exact(value, other, f"{path}[{index}]")
        return
    if left != right:
        raise A1HiddenAdapterContractError(f"optimizer scalar state drift at {path}")


def assert_shared_optimizer_state_bit_equal(
    w_model: WAddReference,
    w_optimizer: torch.optim.Optimizer,
    h_model: HAddReference,
    h_optimizer: torch.optim.Optimizer,
) -> None:
    """Compare all shared Adam/optimizer state by parameter name and exactly."""

    assert_optimizer_state_bit_equal(
        w_model,
        w_optimizer,
        h_model,
        h_optimizer,
        include_adapter=False,
    )


def assert_optimizer_state_bit_equal(
    left_model: nn.Module,
    left_optimizer: torch.optim.Optimizer,
    right_model: nn.Module,
    right_optimizer: torch.optim.Optimizer,
    *,
    include_adapter: bool,
) -> None:
    """Compare named optimizer state exactly, optionally including P itself."""

    left_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in left_optimizer.param_groups
    ]
    right_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in right_optimizer.param_groups
    ]
    _assert_nested_exact(left_groups, right_groups, "optimizer.param_groups")
    _assert_nested_exact(
        optimizer_state_by_parameter_name(
            left_model, left_optimizer, include_adapter=include_adapter
        ),
        optimizer_state_by_parameter_name(
            right_model, right_optimizer, include_adapter=include_adapter
        ),
        "optimizer.named_state",
    )


@dataclass(frozen=True)
class SyntheticBatch:
    """A deterministic CPU-only synthetic batch for reference tests."""

    x: Tensor
    z4: Tensor
    t4: Tensor
    target: Tensor


def synthetic_batch(
    *,
    batch_size: int,
    num_units: int,
    prediction_dim: int = DEFAULT_PREDICTION_DIM,
    seed: int = 20260813,
    zero_t4: bool = False,
) -> SyntheticBatch:
    """Create an isolated-generator batch without consuming global construction RNG."""

    if batch_size < 1 or num_units < 1 or prediction_dim < 1:
        raise A1HiddenAdapterContractError("batch_size, num_units, and prediction_dim must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((batch_size, num_units, HIDDEN_DIM), generator=generator, dtype=torch.float32)
    z4 = torch.randn((batch_size, num_units, SIDE_DIM), generator=generator, dtype=torch.float32)
    t4 = (
        torch.zeros((batch_size, num_units, SIDE_DIM), dtype=torch.float32)
        if zero_t4
        else torch.randn((batch_size, num_units, SIDE_DIM), generator=generator, dtype=torch.float32)
    )
    target = torch.randn(
        (batch_size, num_units, prediction_dim), generator=generator, dtype=torch.float32
    )
    return SyntheticBatch(x=x, z4=z4, t4=t4, target=target)


def task_only_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """The reference's sole source objective; no teacher/distillation term exists."""

    _require_cpu_float32("prediction", prediction, ndim=3)
    _require_cpu_float32("target", target, ndim=3)
    if prediction.shape != target.shape:
        raise A1HiddenAdapterContractError(
            f"task_only target shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    return F.mse_loss(prediction, target)


def source_trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Expose the reference analogue of joint student-decoder + E_A + P training."""

    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def validate_h_source_training_parameter_contract(model: HAddReference) -> None:
    """Require exactly the joint A1 H/T4 source-trainable student parameter set.

    The frozen teacher is deliberately not a member of this synthetic model.
    Its freeze requirement is carried by ``SOURCE_TRAINING_SEMANTICS`` and the
    immutable A2 teacher binding, rather than by a fake teacher tensor.
    """

    if not isinstance(model, HAddReference):
        raise A1HiddenAdapterContractError("A1 source parameter audit requires HAddReference")
    observed = source_trainable_parameter_names(model)
    expected = tuple(SOURCE_TRAINING_SEMANTICS["student_joint_trainables"])
    if observed != expected:
        raise A1HiddenAdapterContractError(
            "A1 source trainables must be joint student decoder + E_A + P; "
            f"expected {expected}, received {observed}"
        )


def score_only_primary_interaction(cell_scores: Mapping[str, float]) -> float:
    """Compute the required A1 interaction from four already-produced scores.

    This is deliberately a scalar-only function.  It cannot load data, choose a
    checkpoint, evaluate a session, or perform any target update.  The H/Z4
    structural-alias equality is enforced before the interaction is returned.
    """

    expected = {"W/Z4", "W/T4", "H/Z4", "H/T4"}
    if set(cell_scores) != expected:
        raise A1HiddenAdapterContractError(
            f"A1 scorer contract requires exactly {sorted(expected)}, got {sorted(cell_scores)}"
        )
    values: dict[str, float] = {}
    for key, value in cell_scores.items():
        numeric = float(value)
        if not isfinite(numeric):
            raise A1HiddenAdapterContractError(f"A1 score for {key} must be finite")
        values[key] = numeric
    if values["H/Z4"] != values["W/Z4"]:
        raise A1HiddenAdapterContractError(
            "H/Z4 must equal the W/Z4 structural alias score exactly"
        )
    return (values["H/T4"] - values["H/Z4"]) - (
        values["W/T4"] - values["W/Z4"]
    )


@dataclass(frozen=True)
class PilotRoutingDecision:
    """Development-only Stage-P routing result; it is never a GPU authorization."""

    seed: int
    per_session_delta: tuple[float, ...]
    mean_delta: float
    median_delta: float
    positive_session_count: int
    passes_mean_floor: bool
    passes_median_positive: bool
    passes_positive_session_count: bool
    passes_all: bool
    authorizes_gpu: bool = False


def frozen_stage_p_routing_decision(
    *,
    w_t4_scores: Sequence[float],
    h_t4_scores: Sequence[float],
    seed: int = PILOT_SEED,
) -> PilotRoutingDecision:
    """Apply the frozen six-session, seed-42 A1 development routing gate.

    It intentionally evaluates the requested ``H/T4 - W/T4`` pilot contrast.
    The formal reporting statistic remains ``score_only_primary_interaction``;
    under the exact H/Z4 alias they are algebraically equal, but the interaction
    must still be written and stored as the primary estimand.
    """

    if seed != PILOT_SEED:
        raise A1HiddenAdapterContractError(
            f"Stage-P is frozen to development seed {PILOT_SEED}, got {seed}"
        )
    if len(w_t4_scores) != PILOT_SESSION_COUNT or len(h_t4_scores) != PILOT_SESSION_COUNT:
        raise A1HiddenAdapterContractError(
            f"Stage-P requires exactly {PILOT_SESSION_COUNT} paired development sessions"
        )
    deltas = tuple(float(h_score) - float(w_score) for w_score, h_score in zip(w_t4_scores, h_t4_scores))
    if not all(isfinite(delta) for delta in deltas):
        raise A1HiddenAdapterContractError("Stage-P scores must produce finite paired deltas")
    ordered = sorted(deltas)
    median = (ordered[2] + ordered[3]) / 2.0
    mean = sum(deltas) / PILOT_SESSION_COUNT
    positive_count = sum(delta > 0.0 for delta in deltas)
    passes_mean = mean >= PILOT_MEAN_DELTA_FLOOR
    passes_median = median > 0.0
    passes_count = positive_count >= 4
    return PilotRoutingDecision(
        seed=seed,
        per_session_delta=deltas,
        mean_delta=mean,
        median_delta=median,
        positive_session_count=positive_count,
        passes_mean_floor=passes_mean,
        passes_median_positive=passes_median,
        passes_positive_session_count=passes_count,
        passes_all=passes_mean and passes_median and passes_count,
    )
