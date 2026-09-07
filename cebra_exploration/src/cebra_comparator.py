"""CEBRA adaptation comparator skeleton (CPU only).

Part A audits constructibility without fitting CEBRA.  Part B implements the
arms in TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md.  ``adapt=True`` after a
multi-session source fit does **not** land the target in the source latent
(coordinator F8); the primary arms are joint multi-session fits.

Do not modify sealed modules under ``sua_exploration/mc_maze/`` or
``SPINT-main/src/``.  Vendored CEBRA may be edited only for
``freeze_sessions`` / ``init_from`` (see ``third_party/CEBRA_PROVENANCE.txt``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_DATE = "20260813"
PROTOCOL_PATH = "cebra_exploration/docs/TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md"
SEED_MATERIAL = "cebra-comparator-20260813"
DEFAULT_DEVICE = "cpu"

MODEL_ARCHITECTURE = "offset1-model"
OUTPUT_DIMENSION = 8
NUM_HIDDEN_UNITS = 32
LEARNING_RATE = 3.0e-4
SOURCE_MAX_ITERATIONS = 10000
MAX_ADAPT_ITERATIONS = 500
RECOMMENDED_BATCH_SIZE = 512
MIN_ADAPT_SAMPLES = 16
D_GRID = (3, 8, 16)
NORMALIZED_LAMBDA_FIXED = 1.0
KNN_NEIGHBORS = 3
KNN_METRIC = "cosine"

SUBM_CALIBRATION_TRIALS = 50
SUBM_BUDGETS = (15, 30, 50)
SUBM_EXPECTED_SESSIONS = 15
RT_CALIBRATION_TRIALS = 24
RT_EXPECTED_FOLDS = 15
M2_CALIBRATION_TRIALS = 24
M2_CHANNELS = 96
H1_SUPPORT_TRIALS = 4
H1_CHANNELS = 176
H1_VELOCITY_DIM = 7
H1_EXPECTED_QUERY_SHA256 = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"

VIEWS = ("sua", "pseudo_mua")
PRIMARY_DATASETS = ("subject_m", "rt")
BOUND_DATASETS = ("subject_m", "rt", "falcon_h1", "falcon_m2")
PRIMARY_ARM = "cebra_joint_behavior"
ARMS = (
    "cebra_joint_behavior",
    "cebra_joint_time",
    "cebra_frozen_source_adapt",
    "cebra_adapt_unaligned",
    "cebra_no_adapt",
    "cebra_joint_time_query_unlabelled",
)
NEGATIVE_CONTROL_ARM = "cebra_adapt_unaligned"
PRIMARY_DECODER = "linear_ridge"
SECONDARY_DECODER = "knn"
INTEGRITY_ATOL = 1.0e-5

# Direction of bias, named so a handicapped arm cannot be presented as CEBRA's best
# (HANDOFF_COMPARATORS_20260812.md §10; reporting rule 5).
ARM_BIAS = {
    "cebra_joint_behavior": (
        "Primary. Favours CEBRA on accuracy: the target calib prefix participates in the "
        "joint fit, so source encoders may move to meet the target. Favours us honestly on "
        "cost: adaptation retrains every session encoder, not 20% of one."
    ),
    "cebra_joint_time": (
        "NOT BUILT. sklearn multi-session CEBRA requires a shared auxiliary "
        "(RuntimeError if y is omitted). Time is not a cross-session coordinate. "
        "Manufacturing np.arange would assume clocks correspond and would cheat "
        "a time-parameterised positive control. Verdict: "
        "CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY."
    ),
    "cebra_frozen_source_adapt": (
        "Fairest analogue of us: source encoders frozen, only the target encoder trains, "
        "cross-session positives kept. Slightly against CEBRA vs joint (source cannot move). "
        "Still requires a target-session backward pass, unlike a closed-form solve."
    ),
    "cebra_adapt_unaligned": (
        "DECLARED NEGATIVE CONTROL. Official adapt=True without cross-session sampling. "
        "Structurally unaligned (F8). Never present as CEBRA's best. Must fail the "
        "positive-control gate."
    ),
    "cebra_no_adapt": (
        "Zero-shot floor. Undefined when N differs. Isolates what any target-session "
        "update buys. Not a handicapped CEBRA."
    ),
    "cebra_joint_time_query_unlabelled": (
        "NOT BUILT. Unlabelled query activity cannot enter a sklearn joint fit: "
        "every sample needs an auxiliary for cross-session sampling. Assigning "
        "query behavior would be a label leak; assigning arange would invent "
        "correspondence. Declared: query activity does not participate. "
        "CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY."
    ),
}
MULTISESSION_TIME_VERDICT = "CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY"
MULTISESSION_TIME_REASON = (
    "sklearn multi-session CEBRA requires a shared auxiliary to define "
    "cross-session positives (integrations/sklearn/cebra.py:670-673). "
    "CEBRA-Time has none. Manufacturing np.arange would assume session clocks "
    "correspond and would spuriously pass a time-parameterised positive control. "
    "This arm is not built."
)
QUERY_UNLABELLED_REASON = (
    "Unlabelled target query activity cannot join a sklearn multi-session fit "
    "without an auxiliary on those samples. Assigning query behavior is a label "
    "leak; assigning time indices invents correspondence. Query activity is "
    "declared out of the fit. Same structural block as joint Time."
)
QUERY_ACTIVITY_IN_PRIMARY = False
TARGET_QUERY_LABELS_IN_FIT = False

POSITIVE_CONTROL_N_CHANNELS = (24, 31, 37)
POSITIVE_CONTROL_N_SAMPLES = 240
POSITIVE_CONTROL_LATENT_DIM = 2
POSITIVE_CONTROL_NOISE = 0.05
POSITIVE_CONTROL_ITERATIONS = 250
POSITIVE_CONTROL_ADAPT_ITERATIONS = 250
POSITIVE_CONTROL_OUTPUT_DIMENSION = 3
POSITIVE_CONTROL_MIN_TARGET_R2 = 0.70
POSITIVE_CONTROL_UNALIGNED_MAX_TARGET_R2 = 0.20
DEFAULT_RT_DATA_DIR = "sua_exploration/data/dandi_000688/sub-C"

SUBM_COHORT_SESSIONS: tuple[str, ...] = (
    "sub-M_ses-CO-20140307",
    "sub-M_ses-CO-20140626",
    "sub-M_ses-CO-20140627",
    "sub-M_ses-CO-20141203",
    "sub-M_ses-CO-20150511",
    "sub-M_ses-CO-20150512",
    "sub-M_ses-CO-20150610",
    "sub-M_ses-CO-20150611",
    "sub-M_ses-CO-20150612",
    "sub-M_ses-CO-20150615",
    "sub-M_ses-CO-20150616",
    "sub-M_ses-CO-20150617",
    "sub-M_ses-CO-20150623",
    "sub-M_ses-CO-20150625",
    "sub-M_ses-CO-20150626",
)

M2_HELD_OUT_DIR_NAME = "sub-MonkeyN-held-out-calib"

SEALED_REFERENCES = {
    "subject_m": {
        "carrier_sua": 0.3568,
        "carrier_pseudo_mua": 0.3061,
        "dense_ridge_sua": 0.4179,
        "dense_ridge_pseudo_mua": 0.4102,
    },
    "rt": {
        "t4d": 0.448176,
        "ridge": 0.200202,
        "zero4": 0.179272,
    },
    "falcon_h1": {
        "ridge_sealed": 0.25823473332303337,
        "carrier": 0.5000,
    },
    "falcon_m2": {
        "ridge": 0.1139,
        "carrier": 0.2268,
    },
}

# Channel-count ranges already measured by the FA alignment Part A (2026-08-13).
# Used as structural priors when a real NWB audit has not been run.
STRUCTURAL_PRIORS = {
    "subject_m_sua": {
        "n_channels_range": (25, 92),
        "channel_counts_vary": True,
        "continuous_velocity": True,
        "discrete_direction_nondegenerate": True,
    },
    "subject_m_pseudo_mua": {
        "n_channels_range": (19, 66),
        "channel_counts_vary": True,
        "continuous_velocity": True,
        "discrete_direction_nondegenerate": True,
    },
    "rt": {
        "n_channels_range": (57, 88),
        "channel_counts_vary": True,
        "continuous_velocity": True,
        "discrete_direction_nondegenerate": False,
        "native_direction_unique_values": 1,
    },
    "falcon_h1": {
        "n_channels_range": (176, 176),
        "channel_counts_vary": False,
        "continuous_velocity": True,
        "discrete_direction_nondegenerate": False,
        "velocity_dim": 7,
    },
    "falcon_m2": {
        "n_channels_range": (96, 96),
        "channel_counts_vary": False,
        "continuous_velocity": True,
        "discrete_direction_nondegenerate": True,
        "m24_support_rows_range": (1451, 1698),
    },
}

IMPLEMENTATION_BINDING = {
    "protocol": PROTOCOL_PATH,
    "cebra_comparator": "cebra_exploration/src/cebra_comparator.py",
    "runner": "cebra_exploration/scripts/run_cebra_comparator.py",
    "vendored_cebra_sklearn": "cebra_exploration/third_party/cebra/cebra/integrations/sklearn/cebra.py",
    "vendored_provenance": "cebra_exploration/third_party/CEBRA_PROVENANCE.txt",
    "linear_readout": "sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py",
    "h1_discovery": "sua_exploration/mc_maze/h1_sparse_event_endpoint.py",
    "m2_allowlist": "sua_exploration/mc_maze/native_m2_m24_ridge_w50.py",
    "rt_query_identity": "sua_exploration/mc_maze/rt_classical_comparators.py",
    "rt_discovery": "streaming_calibration_exp/src/data/rt_k4_loader.py:find_rt_sessions + rt_classical_comparators.session_name_from_nwb_path",
}


class CebraComparatorError(RuntimeError):
    """Raised when a CEBRA-comparator contract is violated."""


class LabelLeakError(CebraComparatorError):
    """Raised when a target query label would enter a CEBRA fit."""


class PositiveControlError(CebraComparatorError):
    """Raised when an arm cannot recover a shared synthetic latent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CebraComparatorError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def seed_from_material(material: str = SEED_MATERIAL) -> int:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def rng_from_material(material: str = SEED_MATERIAL) -> np.random.Generator:
    return np.random.default_rng(seed_from_material(material))


def seed_computational_rng(material: str) -> int:
    """Seed NumPy and Torch so the positive-control gate is reproducible on CPU."""
    seed = int(seed_from_material(material) % (2**32))
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    return seed


def require_cpu_device(device: str = DEFAULT_DEVICE) -> str:
    require(str(device) == "cpu", f"this comparator refuses GPU work; got device={device!r}")
    return "cpu"


def cuda_visible_devices_is_blank() -> bool:
    return os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""


def first_layer_parameter_count(n_neurons: int, n_hidden: int = NUM_HIDDEN_UNITS) -> int:
    require(n_neurons > 0 and n_hidden > 0, "first-layer parameter count requires positive N and H")
    return int(n_neurons) * int(n_hidden) + int(n_hidden)


def samples_per_input_layer_parameter(
    n_samples: int,
    n_neurons: int,
    n_hidden: int = NUM_HIDDEN_UNITS,
) -> float:
    params = first_layer_parameter_count(n_neurons, n_hidden)
    return float(n_samples) / float(params) if params else float("nan")


def resolve_batch_size(n_samples: int, recommended: int = RECOMMENDED_BATCH_SIZE) -> int:
    require(n_samples > 0, "batch size requires a positive sample count")
    return int(min(int(recommended), int(n_samples)))


def offset1_encoder_forward_flops(
    n_neurons: int,
    n_hidden: int,
    n_output: int,
    batch_size: int,
) -> int:
    hidden_half = max(int(n_hidden) // 2, 1)
    return int(
        2 * batch_size * n_neurons * n_hidden
        + 2 * batch_size * n_hidden * n_hidden
        + 2 * batch_size * n_hidden * hidden_half
        + 2 * batch_size * hidden_half * n_output
    )


def estimate_offset1_training_flops(
    *,
    session_n_neurons: Sequence[int],
    trainable_sessions: Sequence[bool],
    batch_size: int,
    n_iterations: int,
    n_hidden: int = NUM_HIDDEN_UNITS,
    n_output: int = OUTPUT_DIMENSION,
) -> int:
    """Honest FLOP estimate: frozen encoders forward only, trainable encoders fwd+bwd.

    Joint fit trains every session (3x forward per session per step).
    Frozen-source trains only the target encoder; source sessions still
    forward because they participate in cross-session InfoNCE.
    """
    require(len(session_n_neurons) == len(trainable_sessions), "trainable mask / N mismatch")
    per_step = 0
    for n_neurons, trainable in zip(session_n_neurons, trainable_sessions):
        forward = offset1_encoder_forward_flops(int(n_neurons), n_hidden, n_output, batch_size)
        per_step += int(forward * (3 if trainable else 1))
    return int(n_iterations) * int(per_step)


def estimate_offset1_adapt_flops(
    *,
    n_neurons: int,
    n_hidden: int,
    n_output: int,
    batch_size: int,
    n_iterations: int,
) -> int:
    """First-layer-only adapt (negative-control arm). Frozen stack still forwards."""
    forward = offset1_encoder_forward_flops(n_neurons, n_hidden, n_output, batch_size)
    first_layer = 2 * batch_size * n_neurons * n_hidden
    return int(n_iterations) * int(2 * forward + first_layer)


def r2_pooled(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth64 = np.asarray(truth, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    require(truth64.shape == estimate64.shape and truth64.ndim == 2 and truth64.shape[0] >= 3, "invalid pooled R2 arrays")
    sse = float(np.square(truth64 - estimate64).sum())
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    require(math.isfinite(tss) and tss > 0.0, "pooled R2 denominator is not positive")
    score = 1.0 - sse / tss
    require(math.isfinite(score), "pooled R2 is non-finite")
    return float(score)


def r2_variance_weighted(predictions: np.ndarray, targets: np.ndarray) -> float:
    return r2_pooled(targets, predictions)


def metric_for_dataset(dataset: str) -> str:
    if dataset == "falcon_h1":
        return "pooled_r2"
    if dataset in {"subject_m", "rt", "falcon_m2"}:
        return "variance_weighted_r2"
    raise CebraComparatorError(f"unknown dataset for metric: {dataset}")


def score_predictions(predictions: np.ndarray, targets: np.ndarray, *, dataset: str) -> float:
    name = metric_for_dataset(dataset)
    if name in {"variance_weighted_r2", "pooled_r2"}:
        return r2_pooled(targets, predictions)
    raise CebraComparatorError(f"unhandled metric: {name}")


def calibration_budget_for(dataset: str) -> int:
    if dataset == "subject_m":
        return SUBM_CALIBRATION_TRIALS
    if dataset == "rt":
        return RT_CALIBRATION_TRIALS
    if dataset == "falcon_m2":
        return M2_CALIBRATION_TRIALS
    if dataset == "falcon_h1":
        return H1_SUPPORT_TRIALS
    raise CebraComparatorError(f"unknown dataset for budget: {dataset}")


def assert_matched_budget(dataset: str, n_trials_used: int, *, requested: int | None = None) -> int:
    budget = int(requested if requested is not None else calibration_budget_for(dataset))
    require(n_trials_used > 0, "calibration prefix is empty")
    require(
        n_trials_used <= budget,
        f"refusing to give CEBRA more than the carrier budget: used {n_trials_used} > {budget}",
    )
    require(
        n_trials_used == budget,
        f"refusing to starve CEBRA below the carrier budget: used {n_trials_used} < {budget}",
    )
    return budget


@dataclass(frozen=True)
class SessionAuditRow:
    session_name: str
    n_channels: int
    n_samples: int
    n_unique_discrete_labels: int | None
    has_continuous_velocity: bool
    resolved_batch_size: int
    first_layer_params: int
    samples_per_input_layer_param: float
    calibration_too_short: bool
    underdetermined_input_layer: bool
    samples_measured: bool = True


@dataclass(frozen=True)
class AdaptationCost:
    target_parameters_updated: int
    target_parameter_count: int
    target_adapt_wall_clock_s: float
    target_adapt_iterations: int
    target_adapt_flops_estimate: int
    device: str = DEFAULT_DEVICE
    flops_are_estimate: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_parameters_updated": int(self.target_parameters_updated),
            "target_parameter_count": int(self.target_parameter_count),
            "target_adapt_wall_clock_s": float(self.target_adapt_wall_clock_s),
            "target_adapt_iterations": int(self.target_adapt_iterations),
            "target_adapt_flops_estimate": int(self.target_adapt_flops_estimate),
            "device": self.device,
            "flops_are_estimate": True,
            "carrier_comparable_cost": "closed_form_ols_plus_one_forward_pass",
        }


def zero_adaptation_cost() -> AdaptationCost:
    return AdaptationCost(
        target_parameters_updated=0,
        target_parameter_count=0,
        target_adapt_wall_clock_s=0.0,
        target_adapt_iterations=0,
        target_adapt_flops_estimate=0,
    )


def _import_cebra():
    try:
        import cebra
    except ImportError as exc:
        raise CebraComparatorError(
            "cebra is not importable; set PYTHONNOUSERSITE=1 and PYTHONPATH to "
            "cebra_exploration/third_party/cebra"
        ) from exc
    return cebra


def make_cebra_estimator(
    *,
    max_iterations: int,
    max_adapt_iterations: int = MAX_ADAPT_ITERATIONS,
    batch_size: int | None,
    output_dimension: int = OUTPUT_DIMENSION,
    device: str = DEFAULT_DEVICE,
) -> Any:
    cebra = _import_cebra()
    require_cpu_device(device)
    return cebra.CEBRA(
        model_architecture=MODEL_ARCHITECTURE,
        device=device,
        batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        output_dimension=int(output_dimension),
        num_hidden_units=NUM_HIDDEN_UNITS,
        max_iterations=int(max_iterations),
        max_adapt_iterations=int(max_adapt_iterations),
        verbose=False,
    )


def refuse_target_query_labels(labels_query: np.ndarray | None) -> None:
    if labels_query is not None:
        raise LabelLeakError(
            "target query labels must not enter the CEBRA estimator; only the "
            "matched calibration prefix may carry labels"
        )


def count_trainable_parameters(module: Any) -> tuple[int, int]:
    n_tensors = 0
    n_scalars = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            n_tensors += 1
            n_scalars += int(parameter.numel())
    return n_tensors, n_scalars


def encoder_parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def numpy_encoder_params(estimator: Any, session_id: int) -> list[np.ndarray]:
    return [
        parameter.detach().cpu().numpy().copy()
        for parameter in estimator.model_[session_id].parameters()
    ]


def encoder_params_equal(left: Sequence[np.ndarray], right: Sequence[np.ndarray]) -> bool:
    if len(left) != len(right):
        return False
    return all(np.array_equal(a, b) for a, b in zip(left, right))


def fit_source_cebra(
    neural_sessions: Sequence[np.ndarray],
    label_sessions: Sequence[np.ndarray] | None,
    *,
    max_iterations: int,
    output_dimension: int = OUTPUT_DIMENSION,
    device: str = DEFAULT_DEVICE,
) -> Any:
    require(len(neural_sessions) >= 1, "source fit requires at least one session")
    xs = [np.asarray(item, dtype=np.float64) for item in neural_sessions]
    for array in xs:
        require(array.ndim == 2 and array.shape[0] >= MIN_ADAPT_SAMPLES, "source session is too short for CEBRA")
    batch = resolve_batch_size(min(item.shape[0] for item in xs))
    estimator = make_cebra_estimator(
        max_iterations=max_iterations,
        batch_size=batch,
        output_dimension=output_dimension,
        device=device,
    )
    if label_sessions is None:
        if len(xs) == 1:
            estimator.fit(xs[0])
        else:
            estimator.fit(xs)
    else:
        ys = [np.asarray(item, dtype=np.float64) for item in label_sessions]
        require(len(ys) == len(xs), "label/session count mismatch")
        for neural, labels in zip(xs, ys):
            require(labels.ndim == 2 and labels.shape[0] == neural.shape[0], "label alignment mismatch")
        if len(xs) == 1:
            estimator.fit(xs[0], ys[0])
        else:
            estimator.fit(xs, ys)
    return estimator


def fit_joint_cebra(
    neural_sessions: Sequence[np.ndarray],
    label_sessions: Sequence[np.ndarray] | None,
    *,
    max_iterations: int,
    output_dimension: int = OUTPUT_DIMENSION,
    device: str = DEFAULT_DEVICE,
    freeze_sessions: Sequence[int] | None = None,
    init_from: Any | None = None,
) -> Any:
    require(len(neural_sessions) >= 2, "joint fit requires source plus target")
    xs = [np.asarray(item, dtype=np.float64) for item in neural_sessions]
    batch = resolve_batch_size(min(item.shape[0] for item in xs))
    estimator = make_cebra_estimator(
        max_iterations=max_iterations,
        batch_size=batch,
        output_dimension=output_dimension,
        device=device,
    )
    kwargs: dict[str, Any] = {}
    if freeze_sessions is not None:
        kwargs["freeze_sessions"] = [int(index) for index in freeze_sessions]
    if init_from is not None:
        kwargs["init_from"] = init_from
    if label_sessions is None:
        estimator.fit(xs, **kwargs)
    else:
        ys = [np.asarray(item, dtype=np.float64) for item in label_sessions]
        require(len(ys) == len(xs), "label/session count mismatch")
        estimator.fit(xs, ys, **kwargs)
    return estimator


def source_embeddings_concat(estimator: Any, source_x: Sequence[np.ndarray]) -> np.ndarray:
    pieces = [
        transform_embedding(estimator, array, session_id=index)
        for index, array in enumerate(source_x)
    ]
    return np.concatenate(pieces, axis=0)


def cost_from_estimator(
    estimator: Any,
    *,
    session_n_neurons: Sequence[int],
    trainable_sessions: Sequence[bool],
    batch_size: int,
    n_iterations: int,
    wall_clock_s: float,
    n_output: int,
    device: str = DEFAULT_DEVICE,
) -> AdaptationCost:
    n_tensors, n_scalars = count_trainable_parameters(estimator.model_)
    measured = int(getattr(estimator, "n_trainable_parameters_", n_scalars))
    flops = estimate_offset1_training_flops(
        session_n_neurons=session_n_neurons,
        trainable_sessions=trainable_sessions,
        batch_size=batch_size,
        n_iterations=n_iterations,
        n_output=n_output,
    )
    return AdaptationCost(
        target_parameters_updated=int(n_tensors),
        target_parameter_count=int(measured if measured else n_scalars),
        target_adapt_wall_clock_s=float(wall_clock_s),
        target_adapt_iterations=int(n_iterations),
        target_adapt_flops_estimate=int(flops),
        device=device,
    )


def transform_embedding(estimator: Any, X: np.ndarray, *, session_id: int | None = None) -> np.ndarray:
    array = np.asarray(X, dtype=np.float64)
    kwargs: dict[str, Any] = {}
    if session_id is not None:
        kwargs["session_id"] = int(session_id)
    embedding = estimator.transform(array, **kwargs)
    out = np.asarray(embedding, dtype=np.float64)
    require(out.ndim == 2 and out.shape[0] == array.shape[0], "embedding length mismatch")
    return out


def fit_linear_readout(embeddings: np.ndarray, labels: np.ndarray) -> Any:
    from sua_exploration.mc_maze.subm_v9_f0_pv_ridge import fit_ridge

    return fit_ridge(
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        normalized_lambda=NORMALIZED_LAMBDA_FIXED,
        device="cpu",
    )


def predict_linear_readout(embeddings: np.ndarray, readout: Any) -> np.ndarray:
    from sua_exploration.mc_maze.subm_v9_f0_pv_ridge import predict_ridge

    return np.asarray(predict_ridge(np.asarray(embeddings, dtype=np.float32), readout, device="cpu"), dtype=np.float64)


def fit_knn_readout(embeddings: np.ndarray, labels: np.ndarray) -> Any:
    from sklearn.neighbors import KNeighborsRegressor

    model = KNeighborsRegressor(n_neighbors=KNN_NEIGHBORS, metric=KNN_METRIC)
    model.fit(np.asarray(embeddings, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    return model


def predict_knn_readout(embeddings: np.ndarray, readout: Any) -> np.ndarray:
    return np.asarray(readout.predict(np.asarray(embeddings, dtype=np.float64)), dtype=np.float64)


def no_adapt_verdict(source_n_channels: Sequence[int], target_n_channels: int) -> dict[str, Any]:
    unique_source = sorted({int(value) for value in source_n_channels})
    matching = [int(value) for value in unique_source if int(value) == int(target_n_channels)]
    if not matching:
        return {
            "verdict": "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH",
            "reason": (
                "no source encoder has the target unit count; a frozen encoder cannot be "
                "applied and no unit mapping is manufactured"
            ),
            "source_n_channels": unique_source,
            "target_n_channels": int(target_n_channels),
        }
    return {
        "verdict": "CEBRA_DEFINABLE",
        "reason": (
            "target unit count matches at least one source encoder; no_adapt applies that "
            "encoder with no new input layer.  Same N is not a correspondence proof."
        ),
        "source_n_channels": unique_source,
        "target_n_channels": int(target_n_channels),
    }


def auxiliary_verdicts(row: SessionAuditRow) -> dict[str, Any]:
    behavior = {
        "verdict": "CEBRA_DEFINABLE" if row.has_continuous_velocity else "CEBRA_UNDEFINED_NO_BEHAVIOR_LABEL",
        "reason": None if row.has_continuous_velocity else "no continuous velocity auxiliary on this session",
        "auxiliary": "continuous_velocity",
    }
    if row.n_unique_discrete_labels is None:
        direction = {
            "verdict": "CEBRA_UNDEFINED_NO_BEHAVIOR_LABEL",
            "reason": "no discrete direction field",
            "auxiliary": "discrete_direction",
            "n_unique": None,
        }
    elif int(row.n_unique_discrete_labels) <= 1:
        direction = {
            "verdict": "CEBRA_UNDEFINED_DEGENERATE_DIRECTION",
            "reason": "discrete direction field has one unique value (RT native field is the type case)",
            "auxiliary": "discrete_direction",
            "n_unique": int(row.n_unique_discrete_labels),
        }
    else:
        direction = {
            "verdict": "CEBRA_DEFINABLE",
            "reason": None,
            "auxiliary": "discrete_direction",
            "n_unique": int(row.n_unique_discrete_labels),
        }
    return {"cebra_behavior_velocity": behavior, "cebra_behavior_direction": direction}


def session_row_from_counts(
    *,
    session_name: str,
    n_channels: int,
    n_samples: int,
    n_unique_discrete_labels: int | None,
    has_continuous_velocity: bool,
    samples_measured: bool = True,
) -> SessionAuditRow:
    too_short = bool(samples_measured and int(n_samples) < MIN_ADAPT_SAMPLES)
    params = first_layer_parameter_count(int(n_channels))
    spp = samples_per_input_layer_parameter(max(int(n_samples), 0), int(n_channels))
    return SessionAuditRow(
        session_name=session_name,
        n_channels=int(n_channels),
        n_samples=int(n_samples),
        n_unique_discrete_labels=None if n_unique_discrete_labels is None else int(n_unique_discrete_labels),
        has_continuous_velocity=bool(has_continuous_velocity),
        resolved_batch_size=resolve_batch_size(max(int(n_samples), 1)),
        first_layer_params=params,
        samples_per_input_layer_param=float(spp),
        calibration_too_short=bool(too_short),
        underdetermined_input_layer=bool(samples_measured and spp < 1.0),
        samples_measured=bool(samples_measured),
    )


def emit_arm_verdicts(
    *,
    dataset: str,
    rows: Sequence[SessionAuditRow],
) -> dict[str, Any]:
    require(len(rows) >= 2, "leave-one-session-out CEBRA requires at least two sessions")
    channel_counts = [row.n_channels for row in rows]
    vary = len(set(channel_counts)) > 1
    too_short = [row.session_name for row in rows if row.calibration_too_short]
    unmeasured = [row.session_name for row in rows if not row.samples_measured]
    velocity_ok = all(row.has_continuous_velocity for row in rows)
    direction_unique = [row.n_unique_discrete_labels for row in rows]
    direction_degenerate = any(value is not None and int(value) <= 1 for value in direction_unique)
    direction_absent = all(value is None for value in direction_unique)

    if unmeasured:
        adapt_sample_verdict = "CEBRA_AUDIT_UNMEASURED_CALIBRATION_LENGTH"
        adapt_sample_reason = (
            "calibration prefix length has not been measured on real NWBs; "
            "not a TOO_SHORT finding"
        )
    elif too_short:
        adapt_sample_verdict = "CEBRA_UNDEFINED_CALIBRATION_TOO_SHORT"
        adapt_sample_reason = f"prefix shorter than MIN_ADAPT_SAMPLES={MIN_ADAPT_SAMPLES} on {too_short}"
    else:
        adapt_sample_verdict = "CEBRA_DEFINABLE"
        adapt_sample_reason = None

    if not velocity_ok:
        behavior = {
            "verdict": "CEBRA_UNDEFINED_NO_BEHAVIOR_LABEL",
            "reason": "continuous velocity auxiliary is missing on at least one session",
        }
    elif adapt_sample_verdict != "CEBRA_DEFINABLE":
        behavior = {"verdict": adapt_sample_verdict, "reason": adapt_sample_reason}
    else:
        behavior = {"verdict": "CEBRA_DEFINABLE", "reason": None}

    time_arm = {"verdict": adapt_sample_verdict, "reason": adapt_sample_reason}

    no_adapt_mismatch = any(
        no_adapt_verdict(channel_counts[:index] + channel_counts[index + 1 :], row.n_channels)["verdict"]
        != "CEBRA_DEFINABLE"
        for index, row in enumerate(rows)
    )
    if no_adapt_mismatch:
        no_adapt = {
            "verdict": "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH",
            "reason": "at least one leave-one-out fold has no source encoder with the target unit count",
        }
    else:
        no_adapt = {
            "verdict": "CEBRA_DEFINABLE",
            "reason": "every fold has a source encoder with matching N; same N is not correspondence",
        }

    if direction_absent:
        direction = {
            "verdict": "CEBRA_UNDEFINED_NO_BEHAVIOR_LABEL",
            "reason": "no discrete direction field",
        }
    elif direction_degenerate:
        direction = {
            "verdict": "CEBRA_UNDEFINED_DEGENERATE_DIRECTION",
            "reason": "discrete direction field is degenerate (one unique value)",
        }
    else:
        direction = {"verdict": "CEBRA_DEFINABLE", "reason": None}

    underdetermined = [row.session_name for row in rows if row.underdetermined_input_layer]
    joint_behavior = dict(behavior)
    frozen = dict(behavior)
    unaligned = dict(time_arm) if adapt_sample_verdict != "CEBRA_DEFINABLE" else {
        "verdict": "CEBRA_DEFINABLE_NEGATIVE_CONTROL",
        "reason": (
            "runnable as a declared negative control: official adapt=True without "
            "cross-session sampling; must fail the positive-control gate"
        ),
    }
    joint_time = {
        "verdict": MULTISESSION_TIME_VERDICT,
        "reason": MULTISESSION_TIME_REASON,
    }
    query_unlabelled = {
        "verdict": MULTISESSION_TIME_VERDICT,
        "reason": QUERY_UNLABELLED_REASON,
    }
    return {
        "cebra_joint_behavior": joint_behavior,
        "cebra_joint_time": joint_time,
        "cebra_frozen_source_adapt": frozen,
        "cebra_adapt_unaligned": unaligned,
        "cebra_no_adapt": no_adapt,
        "cebra_joint_time_query_unlabelled": query_unlabelled,
        "cebra_behavior_direction": direction,
        "shared_d": {
            "constructible": True,
            "selected_d": OUTPUT_DIMENSION,
            "grid": list(D_GRID),
            "note": (
                "CEBRA d is a training hyperparameter, not a variance-explained quantity; "
                "a single d is required because it is the shared embedding size"
            ),
        },
        "underdetermined_input_layer_sessions": underdetermined,
        "channel_counts_vary": bool(vary),
        "min_n_channels": min(channel_counts),
        "max_n_channels": max(channel_counts),
        "min_n_samples": min(row.n_samples for row in rows),
        "max_n_samples": max(row.n_samples for row in rows),
        "dataset": dataset,
    }


def buildable_arms_from_verdicts(verdicts: Mapping[str, Any]) -> list[str]:
    buildable: list[str] = []
    for arm in ARMS:
        status = str(verdicts[arm]["verdict"])
        if status.startswith("CEBRA_DEFINABLE"):
            buildable.append(arm)
    return buildable


def integrity_gate(
    dataset: str,
    *,
    reproduced: Mapping[str, float] | None = None,
    view: str | None = None,
) -> dict[str, Any]:
    require(dataset in SEALED_REFERENCES, f"no sealed reference for {dataset}")
    expected = dict(SEALED_REFERENCES[dataset])
    payload = {
        "dataset": dataset,
        "view": view,
        "expected": expected,
        "atol": INTEGRITY_ATOL,
        "implementation_binding": dict(IMPLEMENTATION_BINDING),
    }
    if reproduced is None:
        payload.update(
            {
                "status": "HOOK_WIRED_NOT_EXECUTED",
                "passed": None,
                "max_deviation": None,
                "observed": None,
            }
        )
        return payload
    observed = {key: float(reproduced[key]) for key in expected if key in reproduced}
    require(observed, f"reproduced reference shares no keys with sealed {dataset} table")
    deviations = {key: abs(observed[key] - expected[key]) for key in observed}
    max_deviation = float(max(deviations.values()))
    passed = bool(max_deviation <= INTEGRITY_ATOL)
    payload.update(
        {
            "status": "PASSED" if passed else "FAILED",
            "passed": passed,
            "max_deviation": max_deviation,
            "observed": observed,
            "deviations": deviations,
        }
    )
    if not passed:
        raise CebraComparatorError(
            f"integrity gate failed on {dataset}: max deviation {max_deviation} > {INTEGRITY_ATOL}"
        )
    return payload


def build_part_a_payload(
    *,
    dataset: str,
    view: str,
    rows: Sequence[SessionAuditRow],
    query_identity: Mapping[str, Any] | None,
    structural_prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verdicts = emit_arm_verdicts(dataset=dataset, rows=rows)
    return {
        "dataset": dataset,
        "view": view,
        "protocol_date": PROTOCOL_DATE,
        "calibration_budget_trials": calibration_budget_for(dataset),
        "min_adapt_samples": MIN_ADAPT_SAMPLES,
        "recommended_batch_size": RECOMMENDED_BATCH_SIZE,
        "model_architecture": MODEL_ARCHITECTURE,
        "output_dimension": OUTPUT_DIMENSION,
        "num_hidden_units": NUM_HIDDEN_UNITS,
        "sessions": [
            {
                "session_name": row.session_name,
                "n_channels": row.n_channels,
                "n_samples": row.n_samples,
                "n_unique_discrete_labels": row.n_unique_discrete_labels,
                "has_continuous_velocity": row.has_continuous_velocity,
                "resolved_batch_size": row.resolved_batch_size,
                "first_layer_params": row.first_layer_params,
                "samples_per_input_layer_param": row.samples_per_input_layer_param,
                "calibration_too_short": row.calibration_too_short,
                "underdetermined_input_layer": row.underdetermined_input_layer,
                "samples_measured": row.samples_measured,
                "auxiliaries": auxiliary_verdicts(row),
            }
            for row in rows
        ],
        "verdicts": verdicts,
        "buildable_arms": buildable_arms_from_verdicts(verdicts),
        "query_identity": query_identity,
        "structural_prior": dict(structural_prior) if structural_prior is not None else None,
        "integrity_gate": integrity_gate(dataset, view=view, reproduced=None),
        "primary_decoder": PRIMARY_DECODER,
        "secondary_decoder": SECONDARY_DECODER,
        "scoring_arm_executed": False,
    }


def structural_prior_rows(dataset: str, view: str | None = None) -> tuple[list[SessionAuditRow], dict[str, Any]]:
    """Synthetic stand-in rows from already-measured structural facts.

    Used by dry-run and unit tests.  Not a substitute for a real NWB Part A.
    Sample counts that have not been measured are set high enough to be
    definable so the prior isolates unit-count / label degeneracy, except H1
    whose M4 length is unknown and is therefore marked unmeasured.
    """
    if dataset == "subject_m":
        key = "subject_m_pseudo_mua" if view == "pseudo_mua" else "subject_m_sua"
    else:
        key = dataset
    prior = STRUCTURAL_PRIORS[key]
    lo, hi = prior["n_channels_range"]
    n_sessions = {"subject_m": 15, "rt": 15, "falcon_h1": 13, "falcon_m2": 6}[dataset]
    if dataset == "falcon_m2":
        n_samples = int(prior["m24_support_rows_range"][0])
        measured_samples = True
    elif dataset == "falcon_h1":
        n_samples = 0
        measured_samples = False
    else:
        n_samples = 0
        measured_samples = False
    rows: list[SessionAuditRow] = []
    for index in range(n_sessions):
        if prior["channel_counts_vary"]:
            n_channels = int(lo + round((hi - lo) * index / max(n_sessions - 1, 1)))
        else:
            n_channels = int(lo)
        n_unique = 8 if prior["discrete_direction_nondegenerate"] else (
            1 if dataset == "rt" else None
        )
        if dataset == "falcon_h1":
            n_unique = None
        rows.append(
            session_row_from_counts(
                session_name=f"prior_{index:02d}",
                n_channels=n_channels,
                n_samples=n_samples,
                n_unique_discrete_labels=n_unique,
                has_continuous_velocity=bool(prior["continuous_velocity"]),
                samples_measured=measured_samples,
            )
        )
    prior_out = dict(prior)
    prior_out["sample_counts_measured"] = measured_samples
    if not measured_samples:
        prior_out["sample_count_status"] = "UNMEASURED_PLACEHOLDER" if n_samples else "UNMEASURED_ZERO"
    return rows, prior_out


def audit_from_structural_prior(dataset: str, *, view: str | None = None) -> dict[str, Any]:
    resolved_view = view if dataset == "subject_m" else dataset
    if dataset == "subject_m":
        require(view in VIEWS, "subject-M requires view sua|pseudo_mua")
        resolved_view = str(view)
    rows, prior = structural_prior_rows(dataset, view)
    return build_part_a_payload(
        dataset=dataset,
        view=str(resolved_view),
        rows=rows,
        query_identity={"status": "NOT_OPENED_DURING_STRUCTURAL_PRIOR"},
        structural_prior=prior,
    )


def discover_subject_m_sessions(data_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    data_dir = data_dir.expanduser().resolve()
    for session_name in SUBM_COHORT_SESSIONS:
        path = data_dir / f"{session_name}_behavior+ecephys.nwb"
        require(path.is_file(), f"missing subject-M NWB: {path}")
        indexed[session_name] = path
    require(len(indexed) == SUBM_EXPECTED_SESSIONS, "subject-M cohort is not 15 sessions")
    return indexed


def discover_rt_sessions(data_dir: Path) -> dict[str, Path]:
    """RT discovery through the sealed loader.  Not NLB MC_RTT / 000129/sub-Indy."""
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    data_dir = data_dir.expanduser().resolve()
    require("000129" not in str(data_dir) and "sub-Indy" not in str(data_dir), "refusing NLB MC_RTT path; RT is dandi_000688/sub-C")
    indexed: dict[str, Path] = {}
    for path in find_rt_sessions(data_dir):
        resolved = Path(path).resolve()
        session_name = rt.session_name_from_nwb_path(resolved)
        require(session_name not in indexed, f"duplicate RT session {session_name}")
        indexed[session_name] = resolved
    require(len(indexed) == RT_EXPECTED_FOLDS, f"expected {RT_EXPECTED_FOLDS} sub-C_ses-RT sessions, found {len(indexed)}")
    return indexed


def discover_h1_sessions(data_dir: Path) -> dict[str, Path]:
    """H1 file discovery.  The only legal entry point is index_heldin_calib."""
    from sua_exploration.mc_maze.h1_sparse_event_endpoint import index_heldin_calib

    return index_heldin_calib(data_dir)


def discover_m2_sessions(data_dir: Path) -> dict[str, Path]:
    """Allowlist against the sealed six-session M24 roster.  Not a raw glob.

    ``held-out-calib`` is in scope for FALCON M2 (released few-shot split).
    """
    from sua_exploration.mc_maze.native_m2_m24_ridge_w50 import EXPECTED_HELDOUT_SESSIONS

    root = data_dir.expanduser().resolve()
    directory = root / M2_HELD_OUT_DIR_NAME if (root / M2_HELD_OUT_DIR_NAME).is_dir() else root
    require(
        M2_HELD_OUT_DIR_NAME in directory.name or M2_HELD_OUT_DIR_NAME in str(directory),
        f"M2 discovery expects {M2_HELD_OUT_DIR_NAME} under {root}",
    )
    indexed: dict[str, Path] = {}
    for session in EXPECTED_HELDOUT_SESSIONS:
        path = directory / f"{M2_HELD_OUT_DIR_NAME}_{session}_behavior+ecephys.nwb"
        require(path.is_file(), f"missing M2 NWB (allowlist, not glob): {path}")
        indexed[session] = path
    require(len(indexed) == len(EXPECTED_HELDOUT_SESSIONS), "M2 allowlist is incomplete")
    return indexed


def _source_pool(xs: Sequence[np.ndarray], ys: Sequence[np.ndarray], target_index: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    source_x = [array for index, array in enumerate(xs) if index != target_index]
    source_y = [array for index, array in enumerate(ys) if index != target_index]
    require(len(source_x) >= 1, "source pool is empty")
    return source_x, source_y


def _fit_unaligned_adapt(
    source_estimator: Any,
    source_x0: np.ndarray,
    source_y0: np.ndarray | None,
    target_x: np.ndarray,
    target_y: np.ndarray | None,
    *,
    max_adapt_iterations: int,
    output_dimension: int,
    device: str,
) -> tuple[Any, AdaptationCost]:
    """Official single-session adapt=True.  Negative control: no cross-session sampling."""
    template = make_cebra_estimator(
        max_iterations=1,
        max_adapt_iterations=max_adapt_iterations,
        batch_size=resolve_batch_size(int(source_x0.shape[0])),
        output_dimension=output_dimension,
        device=device,
    )
    if source_y0 is None:
        template.fit(source_x0)
    else:
        template.fit(source_x0, source_y0)
    template.model_.load_state_dict(source_estimator.model_[0].state_dict())
    start = time.monotonic()
    if target_y is None:
        template.fit(target_x, adapt=True)
    else:
        template.fit(target_x, target_y, adapt=True)
    elapsed = time.monotonic() - start
    n_tensors, n_scalars = count_trainable_parameters(template.model_)
    batch = resolve_batch_size(int(target_x.shape[0]))
    flops = estimate_offset1_adapt_flops(
        n_neurons=int(target_x.shape[1]),
        n_hidden=NUM_HIDDEN_UNITS,
        n_output=int(output_dimension),
        batch_size=batch,
        n_iterations=int(max_adapt_iterations),
    )
    return template, AdaptationCost(
        target_parameters_updated=int(n_tensors),
        target_parameter_count=int(n_scalars if n_scalars else first_layer_parameter_count(int(target_x.shape[1]))),
        target_adapt_wall_clock_s=float(elapsed),
        target_adapt_iterations=int(max_adapt_iterations),
        target_adapt_flops_estimate=int(flops),
        device=device,
    )


def run_arm_on_synthetic_fold(
    *,
    arm: str,
    neural_sessions: Sequence[np.ndarray],
    label_sessions: Sequence[np.ndarray],
    target_index: int,
    dataset: str,
    max_source_iterations: int,
    max_adapt_iterations: int = MAX_ADAPT_ITERATIONS,
    output_dimension: int = OUTPUT_DIMENSION,
    target_query_neural: np.ndarray | None = None,
    target_query_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    require(arm in ARMS, f"unknown arm: {arm}")
    refuse_target_query_labels(target_query_labels)
    xs = [np.asarray(item, dtype=np.float64) for item in neural_sessions]
    ys = [np.asarray(item, dtype=np.float64) for item in label_sessions]
    require(0 <= target_index < len(xs), "target_index out of range")
    source_x, source_y = _source_pool(xs, ys, target_index)
    target_x = xs[target_index]
    target_y = ys[target_index]
    source_counts = [int(item.shape[1]) for item in source_x]
    target_count = int(target_x.shape[1])
    n_output = int(output_dimension)
    batch = resolve_batch_size(min(item.shape[0] for item in list(source_x) + [target_x]))

    if arm in {"cebra_joint_time", "cebra_joint_time_query_unlabelled"}:
        reason = MULTISESSION_TIME_REASON if arm == "cebra_joint_time" else QUERY_UNLABELLED_REASON
        if arm == "cebra_joint_time_query_unlabelled" and target_query_neural is None:
            raise CebraComparatorError("query-unlabelled variant requires unlabelled target query activity")
        return {
            "arm": arm,
            "status": MULTISESSION_TIME_VERDICT,
            "reason": reason,
            "cost": zero_adaptation_cost().as_dict(),
            "source_fit_quality": None,
            "target_score": None,
        }

    if arm == "cebra_no_adapt":
        verdict = no_adapt_verdict(source_counts, target_count)
        if verdict["verdict"] != "CEBRA_DEFINABLE":
            return {
                "arm": arm,
                "status": verdict["verdict"],
                "reason": verdict["reason"],
                "cost": zero_adaptation_cost().as_dict(),
                "source_fit_quality": None,
                "target_score": None,
            }
        estimator = fit_source_cebra(
            source_x, source_y, max_iterations=max_source_iterations, output_dimension=output_dimension
        )
        match_id = next(index for index, count in enumerate(source_counts) if count == target_count)
        source_embeddings = source_embeddings_concat(estimator, source_x)
        target_embeddings = transform_embedding(estimator, target_x, session_id=match_id)
        bundle = _readout_bundle(
            source_embeddings=source_embeddings,
            source_labels=np.concatenate(source_y, axis=0),
            target_embeddings=target_embeddings,
            target_labels=target_y,
            dataset=dataset,
        )
        return {
            "arm": arm,
            "status": "CEBRA_DEFINABLE",
            "reason": verdict["reason"],
            "cost": zero_adaptation_cost().as_dict(),
            "source_fit_quality": bundle["source_fit_quality"],
            "target_score": {
                "linear_ridge_r2": bundle["target"]["linear_ridge_r2"],
                "knn_r2": bundle["target"]["knn_r2"],
            },
            "matched_source_session_id": int(match_id),
        }

    labeled = True  # remaining runnable arms consume calib-prefix labels as the auxiliary

    if arm == "cebra_joint_behavior":
        joint_x = list(source_x) + [target_x]
        joint_y = list(source_y) + [target_y]
        start = time.monotonic()
        estimator = fit_joint_cebra(
            joint_x, joint_y, max_iterations=max_source_iterations, output_dimension=output_dimension
        )
        elapsed = time.monotonic() - start
        target_session_id = len(source_x)
        source_embeddings = source_embeddings_concat(estimator, source_x)
        target_embeddings = transform_embedding(estimator, target_x, session_id=target_session_id)
        cost = cost_from_estimator(
            estimator,
            session_n_neurons=[int(item.shape[1]) for item in joint_x],
            trainable_sessions=[True] * len(joint_x),
            batch_size=batch,
            n_iterations=max_source_iterations,
            wall_clock_s=elapsed,
            n_output=n_output,
        )
    elif arm == "cebra_frozen_source_adapt":
        source_estimator = fit_source_cebra(
            source_x,
            source_y if labeled else None,
            max_iterations=max_source_iterations,
            output_dimension=output_dimension,
        )
        freeze = list(range(len(source_x)))
        joint_x = list(source_x) + [target_x]
        joint_y = (list(source_y) + [target_y]) if labeled else None
        start = time.monotonic()
        estimator = fit_joint_cebra(
            joint_x,
            joint_y,
            max_iterations=max_adapt_iterations,
            output_dimension=output_dimension,
            freeze_sessions=freeze,
            init_from=source_estimator,
        )
        elapsed = time.monotonic() - start
        target_session_id = len(source_x)
        source_embeddings = source_embeddings_concat(estimator, source_x)
        target_embeddings = transform_embedding(estimator, target_x, session_id=target_session_id)
        trainable = [False] * len(source_x) + [True]
        cost = cost_from_estimator(
            estimator,
            session_n_neurons=[int(item.shape[1]) for item in joint_x],
            trainable_sessions=trainable,
            batch_size=batch,
            n_iterations=max_adapt_iterations,
            wall_clock_s=elapsed,
            n_output=n_output,
        )
    elif arm == "cebra_adapt_unaligned":
        source_estimator = fit_source_cebra(
            source_x,
            source_y if labeled else None,
            max_iterations=max_source_iterations,
            output_dimension=output_dimension,
        )
        adapted, cost = _fit_unaligned_adapt(
            source_estimator,
            source_x[0],
            source_y[0] if labeled else None,
            target_x,
            target_y if labeled else None,
            max_adapt_iterations=max_adapt_iterations,
            output_dimension=output_dimension,
            device=DEFAULT_DEVICE,
        )
        source_embeddings = source_embeddings_concat(source_estimator, source_x)
        target_embeddings = transform_embedding(adapted, target_x)
        estimator = adapted
    else:
        raise CebraComparatorError(f"unhandled arm: {arm}")

    bundle = _readout_bundle(
        source_embeddings=source_embeddings,
        source_labels=np.concatenate(source_y, axis=0),
        target_embeddings=target_embeddings,
        target_labels=target_y,
        dataset=dataset,
    )
    return {
        "arm": arm,
        "status": "CEBRA_DEFINABLE_NEGATIVE_CONTROL" if arm == NEGATIVE_CONTROL_ARM else "CEBRA_DEFINABLE",
        "reason": None,
        "cost": cost.as_dict(),
        "source_fit_quality": bundle["source_fit_quality"],
        "target_score": {
            "linear_ridge_r2": bundle["target"]["linear_ridge_r2"],
            "knn_r2": bundle["target"]["knn_r2"],
        },
        "n_trainable_parameters_on_estimator": int(getattr(estimator, "n_trainable_parameters_", -1)),
    }


def make_shared_latent_sessions(
    *,
    n_channels: Sequence[int] = POSITIVE_CONTROL_N_CHANNELS,
    n_samples: int = POSITIVE_CONTROL_N_SAMPLES,
    latent_dim: int = POSITIVE_CONTROL_LATENT_DIM,
    noise: float = POSITIVE_CONTROL_NOISE,
    seed_material: str = SEED_MATERIAL,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Coordinator F8 construction: different linear mixes of one shared latent."""
    rng = rng_from_material(seed_material + "-positive-control")
    t = np.linspace(0.0, 4.0 * np.pi, n_samples, endpoint=False)
    latent = np.stack([np.cos(t), np.sin(t)], axis=1)
    if latent_dim != 2:
        extra = rng.normal(size=(n_samples, latent_dim - 2))
        latent = np.concatenate([latent, extra], axis=1)
    sessions: list[np.ndarray] = []
    for width in n_channels:
        mix = rng.normal(size=(int(width), latent.shape[1]))
        neural = latent @ mix.T + float(noise) * rng.normal(size=(n_samples, int(width)))
        sessions.append(np.asarray(neural, dtype=np.float64))
    return sessions, np.asarray(latent, dtype=np.float64)


def run_positive_control_arm(
    arm: str,
    *,
    max_iterations: int = POSITIVE_CONTROL_ITERATIONS,
    max_adapt_iterations: int = POSITIVE_CONTROL_ADAPT_ITERATIONS,
) -> dict[str, Any]:
    n_fit = POSITIVE_CONTROL_N_SAMPLES
    seed_computational_rng(f"{SEED_MATERIAL}-positive-control-{arm}")
    query_neural = None
    if arm == "cebra_joint_time_query_unlabelled":
        xs_full, latent_full = make_shared_latent_sessions(n_samples=n_fit * 2)
        xs = [item[:n_fit] for item in xs_full]
        ys = [latent_full[:n_fit] for _ in xs]
        query_neural = xs_full[-1][n_fit:]
    else:
        xs, latent = make_shared_latent_sessions(n_samples=n_fit)
        ys = [latent for _ in xs]
    result = run_arm_on_synthetic_fold(
        arm=arm,
        neural_sessions=xs,
        label_sessions=ys,
        target_index=len(xs) - 1,
        dataset="subject_m",
        max_source_iterations=max_iterations,
        max_adapt_iterations=max_adapt_iterations,
        output_dimension=POSITIVE_CONTROL_OUTPUT_DIMENSION,
        target_query_neural=query_neural,
        target_query_labels=None,
    )
    target_r2 = None if result["target_score"] is None else float(result["target_score"]["linear_ridge_r2"])
    source_r2 = None if result["source_fit_quality"] is None else float(result["source_fit_quality"]["linear_ridge_r2"])
    return {
        "arm": arm,
        "status": result["status"],
        "source_r2": source_r2,
        "target_r2": target_r2,
        "cost": result["cost"],
        "bias": ARM_BIAS[arm],
        "threshold": POSITIVE_CONTROL_MIN_TARGET_R2,
        "unaligned_max": POSITIVE_CONTROL_UNALIGNED_MAX_TARGET_R2,
    }


def assert_positive_control(arm: str, report: Mapping[str, Any]) -> dict[str, Any]:
    target_r2 = report.get("target_r2")
    if arm == "cebra_no_adapt":
        require(
            report.get("status") == "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH",
            f"no_adapt must be undefined on mismatched N, got {report.get('status')}",
        )
        return {"arm": arm, "passed": True, "role": "undefined_on_mismatched_n", "target_r2": target_r2}
    if arm in {"cebra_joint_time", "cebra_joint_time_query_unlabelled"}:
        require(
            report.get("status") == MULTISESSION_TIME_VERDICT,
            f"{arm} must be {MULTISESSION_TIME_VERDICT}, got {report.get('status')}",
        )
        return {"arm": arm, "passed": True, "role": "undefined_multisession_requires_auxiliary", "target_r2": target_r2}
    if arm == NEGATIVE_CONTROL_ARM:
        require(
            target_r2 is not None and float(target_r2) < POSITIVE_CONTROL_UNALIGNED_MAX_TARGET_R2,
            f"negative control {arm} unexpectedly recovered the latent: target R2={target_r2}",
        )
        return {"arm": arm, "passed": True, "role": "negative_control", "target_r2": target_r2}
    if report.get("status") != "CEBRA_DEFINABLE" or target_r2 is None:
        raise PositiveControlError(f"{arm} is void: status={report.get('status')} target_r2={target_r2}")
    if float(target_r2) < POSITIVE_CONTROL_MIN_TARGET_R2:
        raise PositiveControlError(
            f"{arm} cannot recover a shared synthetic latent: target R2={target_r2} "
            f"< {POSITIVE_CONTROL_MIN_TARGET_R2}; numbers from this arm are void"
        )
    return {"arm": arm, "passed": True, "role": "required_gate", "target_r2": target_r2}


def run_positive_control_gate(
    *,
    arms: Sequence[str] | None = None,
    max_iterations: int = POSITIVE_CONTROL_ITERATIONS,
    max_adapt_iterations: int = POSITIVE_CONTROL_ADAPT_ITERATIONS,
) -> dict[str, Any]:
    selected = list(arms) if arms is not None else list(ARMS)
    reports = {
        arm: run_positive_control_arm(
            arm, max_iterations=max_iterations, max_adapt_iterations=max_adapt_iterations
        )
        for arm in selected
    }
    checks = {arm: assert_positive_control(arm, reports[arm]) for arm in selected}
    return {
        "status": "PASSED",
        "construction": {
            "n_channels": list(POSITIVE_CONTROL_N_CHANNELS),
            "n_samples": POSITIVE_CONTROL_N_SAMPLES,
            "latent": "shared 2-D (cos,sin) with per-session linear mixes plus noise",
            "readout": "source-fitted linear ridge, applied to target embeddings",
        },
        "min_target_r2": POSITIVE_CONTROL_MIN_TARGET_R2,
        "unaligned_max_target_r2": POSITIVE_CONTROL_UNALIGNED_MAX_TARGET_R2,
        "primary_arm": PRIMARY_ARM,
        "negative_control_arm": NEGATIVE_CONTROL_ARM,
        "arms": reports,
        "checks": checks,
        "arm_bias": dict(ARM_BIAS),
    }


def _readout_bundle(
    *,
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_embeddings: np.ndarray,
    target_labels: np.ndarray,
    dataset: str,
) -> dict[str, Any]:
    linear = fit_linear_readout(source_embeddings, source_labels)
    knn = fit_knn_readout(source_embeddings, source_labels)
    linear_source = predict_linear_readout(source_embeddings, linear)
    knn_source = predict_knn_readout(source_embeddings, knn)
    linear_target = predict_linear_readout(target_embeddings, linear)
    knn_target = predict_knn_readout(target_embeddings, knn)
    return {
        "primary_decoder": PRIMARY_DECODER,
        "secondary_decoder": SECONDARY_DECODER,
        "source_fit_quality": {
            "linear_ridge_r2": score_predictions(linear_source, source_labels, dataset=dataset),
            "knn_r2": score_predictions(knn_source, source_labels, dataset=dataset),
        },
        "target": {
            "linear_ridge_r2": score_predictions(linear_target, target_labels, dataset=dataset),
            "knn_r2": score_predictions(knn_target, target_labels, dataset=dataset),
            "predictions_linear": linear_target,
            "predictions_knn": knn_target,
        },
    }


def dry_run_report(*, dataset: str, repo_root: Path, view: str | None = None, data_dir: Path | None = None) -> dict[str, Any]:
    require(dataset in BOUND_DATASETS, f"unknown dataset: {dataset}")
    report: dict[str, Any] = {
        "dataset": dataset,
        "view": view if dataset == "subject_m" else dataset,
        "device": DEFAULT_DEVICE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol_date": PROTOCOL_DATE,
        "arms": list(ARMS),
        "primary_decoder": PRIMARY_DECODER,
        "secondary_decoder": SECONDARY_DECODER,
        "implementation_binding": dict(IMPLEMENTATION_BINDING),
        "integrity_gate": integrity_gate(dataset, view=view, reproduced=None),
        "scoring_arm_executed": False,
    }
    if dataset == "subject_m":
        require(view in VIEWS, "subject-M requires --view")
        report["calibration_budget_trials"] = SUBM_CALIBRATION_TRIALS
        report["also_report_if_cheap"] = list(SUBM_BUDGETS)
        report["cohort_sessions"] = list(SUBM_COHORT_SESSIONS)
        report["path_status"] = "not_opened"
        if data_dir is not None and data_dir.exists():
            missing = [
                name
                for name in SUBM_COHORT_SESSIONS
                if not (data_dir / f"{name}_behavior+ecephys.nwb").is_file()
            ]
            report["path_status"] = "ready" if not missing else "incomplete"
            report["missing_sessions"] = missing
    elif dataset == "rt":
        report["calibration_budget_trials"] = RT_CALIBRATION_TRIALS
        report["expected_folds"] = RT_EXPECTED_FOLDS
        report["native_direction"] = "degenerate_one_unique_value"
        report["auxiliary"] = "continuous_velocity_only"
        report["discovery"] = (
            "find_rt_sessions + rt_classical_comparators.session_name_from_nwb_path; "
            "NOT data/000129/sub-Indy"
        )
        report["path_status"] = "not_opened"
        if data_dir is not None:
            try:
                sessions = discover_rt_sessions(data_dir)
                report["path_status"] = "ready"
                report["n_sessions"] = len(sessions)
                report["session_names"] = list(sessions)
            except Exception as exc:
                report["path_status"] = "discovery_failed"
                report["discovery_error"] = f"{type(exc).__name__}: {exc}"
    elif dataset == "falcon_h1":
        report["calibration_budget_trials"] = H1_SUPPORT_TRIALS
        report["discovery"] = "sua_exploration.mc_maze.h1_sparse_event_endpoint.index_heldin_calib"
        report["n_channels"] = H1_CHANNELS
        report["velocity_dim"] = H1_VELOCITY_DIM
        report["expected_query_sha256"] = H1_EXPECTED_QUERY_SHA256
        report["held_out_in_scope"] = False
        if data_dir is not None:
            try:
                sessions = discover_h1_sessions(data_dir)
                report["path_status"] = "ready"
                report["n_sessions"] = len(sessions)
                report["session_names"] = list(sessions)
            except Exception as exc:
                report["path_status"] = "discovery_failed"
                report["discovery_error"] = f"{type(exc).__name__}: {exc}"
        else:
            report["path_status"] = "data_dir_not_supplied"
    elif dataset == "falcon_m2":
        from sua_exploration.mc_maze.native_m2_m24_ridge_w50 import EXPECTED_HELDOUT_LAYOUT, EXPECTED_HELDOUT_SESSIONS

        report["calibration_budget_trials"] = M2_CALIBRATION_TRIALS
        report["n_channels"] = M2_CHANNELS
        report["held_out_calib_in_scope"] = True
        report["discovery"] = "allowlist EXPECTED_HELDOUT_SESSIONS, not a raw glob"
        report["sessions"] = list(EXPECTED_HELDOUT_SESSIONS)
        report["m24_layout"] = {name: dict(EXPECTED_HELDOUT_LAYOUT[name]) for name in EXPECTED_HELDOUT_SESSIONS}
        if data_dir is not None:
            try:
                sessions = discover_m2_sessions(data_dir)
                report["path_status"] = "ready"
                report["n_sessions"] = len(sessions)
            except Exception as exc:
                report["path_status"] = "discovery_failed"
                report["discovery_error"] = f"{type(exc).__name__}: {exc}"
        else:
            report["path_status"] = "data_dir_not_supplied"
    report["part_a_structural_prior"] = audit_from_structural_prior(dataset, view=view)
    report["repo_root"] = str(repo_root)
    return report


def dry_run_all(*, repo_root: Path, data_dir_subject_m: Path | None, data_dir_rt: Path | None, data_dir_h1: Path | None, data_dir_m2: Path | None) -> dict[str, Any]:
    return {
        "schema": "cebra_comparator_dry_run_v1",
        "protocol_date": PROTOCOL_DATE,
        "device": DEFAULT_DEVICE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scoring_arm_executed": False,
        "adapters": {
            "subject_m_sua": dry_run_report(
                dataset="subject_m", repo_root=repo_root, view="sua", data_dir=data_dir_subject_m
            ),
            "subject_m_pseudo_mua": dry_run_report(
                dataset="subject_m", repo_root=repo_root, view="pseudo_mua", data_dir=data_dir_subject_m
            ),
            "rt": dry_run_report(dataset="rt", repo_root=repo_root, data_dir=data_dir_rt),
            "falcon_h1": dry_run_report(dataset="falcon_h1", repo_root=repo_root, data_dir=data_dir_h1),
            "falcon_m2": dry_run_report(dataset="falcon_m2", repo_root=repo_root, data_dir=data_dir_m2),
        },
        "primary_arm": PRIMARY_ARM,
        "negative_control_arm": NEGATIVE_CONTROL_ARM,
        "arm_bias": dict(ARM_BIAS),
        "query_activity_in_primary": QUERY_ACTIVITY_IN_PRIMARY,
        "target_query_labels_in_fit": TARGET_QUERY_LABELS_IN_FIT,
        "positive_control_gate": "required before any arm is interpreted; run --positive-control",
        "implementation_binding": dict(IMPLEMENTATION_BINDING),
        "h1_m2_decision": (
            "Bound in the skeleton because CEBRA is not in FALCON Table 1. "
            "Primary scoring datasets remain subject-M and RT. H1 discovery is "
            "index_heldin_calib only. M2 held-out-calib is in scope as FALCON's "
            "released few-shot split."
        ),
    }
