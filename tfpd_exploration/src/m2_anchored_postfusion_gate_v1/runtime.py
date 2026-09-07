"""Post-attempt APFG runtime contracts, deliberately uninvoked in Stage-0.

This module names the only legal execution sequence so a later production
executor cannot accidentally install APFG before strict loading the selected
POOLED student or optimize inherited parameters.  It does not construct PIT,
read a checkpoint, materialize source data, or touch CUDA at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from . import plan


class RuntimeContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceTrainingContract:
    checkpoint_sha256: str
    t4_pool_size: int
    source_fit_sessions: tuple[str, ...]
    source_validation_sessions: tuple[str, ...]
    pool_cycle: tuple[int, ...]
    batch_controller: str
    selection_operator: str
    epochs: int
    adam_lr: float
    alpha_only: bool
    inherited_eval_no_dropout: bool


@dataclass(frozen=True)
class FrozenIdentityCacheKey:
    """Exact cache key for alpha-independent frozen branch identities.

    The live producer may cache detached ``(h_pre, h_post)`` only under this
    full key and only after a no-grad frozen B3S pass.  It must combine these
    cached values with alpha outside ``no_grad``.  Including query trial and
    pool law prevents two otherwise similar FIFO snapshots from crossing a
    causal endpoint boundary.
    """
    session_id: str
    pool_state_sha256: str
    ordered_activity_sha256: tuple[str, ...]
    selected_support4_carrier_hz_sha256: str
    normalized_side_sha256: str
    normalizer_sha256: str
    query_trial_index: int
    requested_pool_size: int

    def validate(self) -> None:
        if not self.session_id or self.requested_pool_size not in plan.POOL_CYCLE or self.query_trial_index < 30:
            raise RuntimeContractError("APFG frozen-identity cache geometry drift")
        digests = (*self.ordered_activity_sha256, self.selected_support4_carrier_hz_sha256,
                   self.normalized_side_sha256, self.normalizer_sha256, self.pool_state_sha256)
        if not digests or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
            raise RuntimeContractError("APFG frozen-identity cache digest drift")


def build_source_training_contract(source_sessions: tuple[str, ...]) -> SourceTrainingContract:
    from .selection import lexical_source_split
    split = lexical_source_split(source_sessions)
    return SourceTrainingContract(
        checkpoint_sha256=plan.SELECTED_T4_POOLED_CHECKPOINT_SHA256,
        t4_pool_size=30, source_fit_sessions=split["fit"], source_validation_sessions=split["validation"],
        pool_cycle=plan.POOL_CYCLE, batch_controller=plan.SOURCE_BATCH_CONTROLLER,
        selection_operator=plan.SOURCE_SELECTION_OPERATOR, epochs=plan.EPOCHS, adam_lr=plan.ADAM_LR, alpha_only=True,
        inherited_eval_no_dropout=True,
    )


def validate_post_attempt_runtime_evidence(evidence: Mapping[str, Any]) -> None:
    """Validate only receipt facts; physical operations remain route-owned later."""
    required = {
        "attempt_published_before_checkpoint_data_cuda", "pit_materializations", "strict_load_before_adapter",
        "inherited_trainable_parameter_names", "alpha_trainable_parameter_names", "inherited_eval",
        "dropout_calls", "optimizer_parameter_names", "source_split", "selection_epochs", "refit_alpha_positive_zero",
        "training_loss", "behavior_scaling_factor", "predict_scaled_behavior", "teacher_forward_calls",
        "seed_evidence",
    }
    if set(evidence) != required:
        raise RuntimeContractError("APFG runtime evidence key topology drift")
    if evidence["attempt_published_before_checkpoint_data_cuda"] is not True or evidence["pit_materializations"] != 1:
        raise RuntimeContractError("APFG attempt/materialization law drift")
    if evidence["strict_load_before_adapter"] is not True or evidence["inherited_trainable_parameter_names"] != []:
        raise RuntimeContractError("APFG strict-load/freeze law drift")
    if evidence["alpha_trainable_parameter_names"] != ["id_encoder.alpha"]:
        raise RuntimeContractError("APFG alpha topology drift")
    if evidence["inherited_eval"] is not True or evidence["dropout_calls"] != 0:
        raise RuntimeContractError("APFG inherited eval/no-dropout law drift")
    if evidence["optimizer_parameter_names"] != ["id_encoder.alpha"]:
        raise RuntimeContractError("APFG optimizer topology drift")
    split = evidence["source_split"]
    if (not isinstance(split, Mapping) or set(split) != {"fit", "validation"}
            or len(tuple(split["fit"])) != plan.FIT_SESSION_COUNT
            or len(tuple(split["validation"])) != plan.VALIDATION_SESSION_COUNT):
        raise RuntimeContractError("APFG lexical 5/2 source split evidence drift")
    if tuple(evidence["selection_epochs"]) != tuple(range(1, plan.EPOCHS + 1)):
        raise RuntimeContractError("APFG selection horizon drift")
    if evidence["refit_alpha_positive_zero"] is not True:
        raise RuntimeContractError("APFG refit must restart alpha at +0.0")
    if (evidence["training_loss"] != plan.TRAINING_LOSS
            or float(evidence["behavior_scaling_factor"]) != plan.BEHAVIOR_SCALING_FACTOR
            or evidence["predict_scaled_behavior"] is not True
            or evidence["teacher_forward_calls"] != plan.TEACHER_FORWARD_CALLS):
        raise RuntimeContractError("APFG task-only scaled last-bin loss law drift")
    seed = evidence["seed_evidence"]
    seed_required = {
        "seed", "python_random_seeded", "numpy_random_seeded", "torch_manual_seeded",
        "torch_cuda_manual_seed_all", "cudnn_deterministic", "cudnn_benchmark",
        "deterministic_algorithms_forced",
    }
    if not isinstance(seed, Mapping) or set(seed) != seed_required:
        raise RuntimeContractError("APFG seed evidence key topology drift")
    if (seed["seed"] != plan.SEED
            or any(seed[name] is not True for name in ("python_random_seeded", "numpy_random_seeded",
                                                       "torch_manual_seeded", "torch_cuda_manual_seed_all"))
            or seed["cudnn_deterministic"] is not True
            or seed["cudnn_benchmark"] is not False
            or seed["deterministic_algorithms_forced"] is not False):
        raise RuntimeContractError("APFG post-attempt seed/determinism evidence drift")


def establish_post_attempt_seed_evidence(*, torch: Any, random_module: Any | None = None,
                                         numpy_module: Any | None = None) -> dict[str, object]:
    """Set the route's fixed seed after attempt publication and before PIT.

    CUDA is already launch-attested at this point.  We deliberately do not
    call ``use_deterministic_algorithms``: this route's fixed cuBLAS workspace
    and deterministic cuDNN flags provide the documented contract without
    imposing a global unsupported-op policy on the inherited model.
    Optional modules exist solely for a no-CUDA call-order test; production
    uses the standard libraries.
    """
    if random_module is None:
        import random as random_module
    if numpy_module is None:
        import numpy as numpy_module
    random_module.seed(plan.SEED)
    numpy_module.random.seed(plan.SEED)
    torch.manual_seed(plan.SEED)
    torch.cuda.manual_seed_all(plan.SEED)
    try:
        cudnn = torch.backends.cudnn
    except AttributeError as error:
        raise RuntimeContractError("APFG cuDNN deterministic controls are unavailable") from error
    cudnn.deterministic = True
    cudnn.benchmark = False
    evidence = {
        "seed": plan.SEED,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_manual_seeded": True,
        "torch_cuda_manual_seed_all": True,
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "deterministic_algorithms_forced": False,
    }
    if evidence["cudnn_deterministic"] is not True or evidence["cudnn_benchmark"] is not False:
        raise RuntimeContractError("APFG cuDNN deterministic control assignment failed")
    return evidence


def task_only_scaled_last_bin_mse(*, torch: Any, student: Any, neural: Any, identity: Any, target: Any) -> Any:
    """The sole authorized alpha objective; teacher is never called."""
    prediction = student.decode_with_identity(neural, identity)
    if tuple(prediction.shape) != tuple(target.shape):
        raise RuntimeContractError("APFG decoder/target tensor topology drift")
    scaled = prediction / float(plan.BEHAVIOR_SCALING_FACTOR)
    return torch.mean((scaled[:, -1:, :] - target[:, -1:, :]) ** 2)


__all__ = ("RuntimeContractError", "SourceTrainingContract", "FrozenIdentityCacheKey", "build_source_training_contract",
           "validate_post_attempt_runtime_evidence", "establish_post_attempt_seed_evidence", "task_only_scaled_last_bin_mse")
