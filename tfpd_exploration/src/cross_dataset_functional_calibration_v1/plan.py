"""Frozen literals for cross-dataset functional calibration v1.

Interfaces only. Do not invent estimators, bases, or parent substitutions here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


class PlanError(RuntimeError):
    """Fail closed for static contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


PHASE = "cross_dataset_functional_calibration_v1"
CONTRACT_VERSION = "cdf_v1.0"
SCHEMA_INVENTORY = "cross_dataset_functional_calibration_inventory_v1"
SCHEMA_E1 = "cross_dataset_functional_calibration_e1_v1"
SCHEMA_E2 = "cross_dataset_functional_calibration_e2_v1"
SCHEMA_E3 = "cross_dataset_functional_calibration_e3_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CROSS_DATASET_FUNCTIONAL_CALIBRATION_V1_20260905.md"
WORKORDER_SHA256 = "1acd9f2372fa1c819648581c54dfb01e79821bd4299a02e46b33427ab0589eb7"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cross_dataset_functional_calibration_v1"
STAGE0_TIMESTAMP = "20260905_113700"
STAGE0_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/{STAGE0_TIMESTAMP}"
STAGE0_TERMINAL_SHA256 = "ea4c3227ad09d09610a8d78a38dbe400bf03ada0893f6058f35594073b36f875"
OWN_TOKEN = "run_cross_dataset_functional_calibration_v1.py"

# Bound documents actually used (hashes verified on the files present at Stage0).
BOUND_DOCUMENTS = (
    {
        "relative": WORKORDER_RELATIVE,
        "sha256": WORKORDER_SHA256,
        "role": "workorder_authority",
    },
    {
        "relative": "sua_exploration/docs/T4G_GENERALIZED_ANALYTIC_FUNCTIONAL_CARRIER_ROUTE_20260806.md",
        "sha256": "dbb6454597238b50aa975f337a8a6ca52767f5ff2986aa17c4cea2724aad4f1d",
        "role": "prior_art_generalized_encoding",
    },
    {
        "relative": "SPINT-main/docs/H1_CARRIERID_QUALITY_DIAGNOSTIC_PROGRAM.md",
        "sha256": "d5fe3bbe02117d638169ab08367b861f29e2a0b13f56c4a9a69926a6a723ab1c",
        "role": "h1_shrinkage_quality_boundary",
    },
    {
        "relative": "sua_exploration/docs/CURRENT_RESULTS.md",
        "sha256": "d933f06a25baffd23c3dfb3b0e09993b862305645b109b65ed6191e119e79997",
        "role": "matched_h1_content_width_evidence",
    },
    {
        "relative": "tfpd_exploration/h1_series_20260830/docs/RESULT_H1_FILM_CONTENT_DIAGNOSTIC_V5_20260904.md",
        "sha256": "380180c059e4ec613270ed38986df01b059e5f1d2fc1423eafffe8e384d9e072",
        "role": "h1_film_v5_profile_content_null",
    },
    {
        "relative": "tfpd_exploration/docs/RESULT_M1_EMG_RSYN3_STAGE0_20260902.md",
        "sha256": "86746837447fd1970f8199b80a4ca8f3e8d60d7be82050b122a2559453d33358",
        "role": "m1_rsyn3_stage0_reliability",
    },
    {
        "relative": "sua_exploration/docs/HANDOFF_M1_CALIBRATION_AWARE_BEHAVIOR_BOTTLENECK_20260824.md",
        "sha256": "7f455f8c4553fae7bcdeb9999f4d3c86273873e1e8bd8af254cceb0b4fb7e8a3",
        "role": "m1_calibration_aware_bottleneck_motivation_not_prediction",
    },
    {
        "relative": "tfpd_exploration/docs/RESULT_FABLE_TKD_M1_V1_20260905.md",
        "sha256": "387ad4a66fed8520de7b1e930a2b8f4c83a10b377ee12aef0cf7484ef2e982c9",
        "role": "m1_closed_form_consumer_negative",
    },
)

# Existing operator / receipt authorities. Do not fork these estimators.
H1_OPERATOR_RELATIVE = "SPINT-main/src/data/h1_m4_eb_pilot.py"
H1_OPERATOR_SHA256 = "c73c80fcec05d323052d9a4154a4cf7c46e90989a917c89116ce085e76b50a8a"
H1_RAW_RECEIPT_RELATIVE = (
    "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/"
    "H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
)
H1_RAW_RECEIPT_SHA256 = "660f78f86ed74b3950ff53946edc2db50979802e8f02a37b211c5e044c0ed4bb"
H1_EB_RECEIPT_RELATIVE = (
    "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/"
    "H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
)
H1_EB_RECEIPT_SHA256 = "13004595d5d5e28c4fb1316bf7119bd3cdb2197bbf5001abb53eec5d2881c964"
H1_FROZEN_PLAN_MANIFEST_RELATIVE = (
    "SPINT-main/pilot_artifacts/h1_carrierid_hu/source_authority_v1/fold0_frozen_eb_plan.manifest.json"
)
H1_FROZEN_PLAN_MANIFEST_SHA256 = "cbf4610a1244ea528aa2b42b3a7ce85220dffa486c40010188523a98f4f0e26c"
H1_DATA_DIR_RELATIVE = "SPINT-main/data/000954"
H1_CHANNELS = 176
H1_VELOCITY_DIM = 7
H1_CARRIER_DIM = 4
H1_BLOCK_BINS = 5
H1_BLOCK_SECONDS = 0.1
H1_RIDGE_LAMBDA = 100.0
H1_PCA_Q = 16
H1_SUPPORT_TRIALS_M3 = 3
H1_SUPPORT_TRIALS_M4 = 4
H1_ESTIMATOR_DIRECTION = "backward_decoder_weight"
H1_LABEL_TYPE = "7dof_velocity_block_mean"

M1_SYN3_OPERATOR_RELATIVE = "tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py"
M1_SYN3_OPERATOR_SHA256 = "f3ecdf614aaeafae316e35660f4119aca6b17c1ee136cb6a59497b3792b145e9"
M1_CARRIER_BANK_RELATIVE = "tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/carrier_bank.py"
M1_CARRIER_BANK_SHA256 = "b9723946077de7aa0bda9581b0a1146579b830676cdcf8cd2c0401ee47aa26cb"
M1_DATA_DIR_RELATIVE = "SPINT-main/data/000941"
M1_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
M1_FOLD0_TARGET = "ses-20120924"
M1_FOLD0_SOURCES = ("ses-20120926", "ses-20120927", "ses-20120928")
M1_SUPPORT_TRIALS = 10
M1_E2_EVAL_TRIALS = 10
M1_E2_NEURAL_STOP = 20
M1_QUERY_START = 10
M1_QUERY_STOP_EXCLUSIVE = 210
M1_WINDOW = 100
M1_TRIAL_LENGTH = 1024
M1_CHANNELS = 64
M1_EMG_DIM = 16
M1_RANK = 3
M1_CARRIER_DIM = 4
M1_BIN_SECONDS = 0.02
M1_RIDGE_LAMBDA = 1.0
M1_SCALE_FLOOR = 1.0e-8
M1_ESTIMATOR_DIRECTION = "forward_encoding_ridge"
M1_LABEL_TYPE = "signed_emg_native_16"
M1_RECTIFIER = "relu_nonnegative_projection"

TOKEN_PROBE_SUMMARY_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/token_probe_v1/summary.json"
TOKEN_PROBE_SUMMARY_SHA256 = "fb578154a63306e0c642b18548df778cfc64173a645cf4411a3a7be388d9625d"
BUDGET_PROBE_SUMMARY_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/budget_probe_v1/summary.json"
BUDGET_PROBE_SUMMARY_SHA256 = "d83327e535b7b20a1817a61e09c861ce58f1ed86697cf456c2181691402f1d41"
FOLD_LOCAL_RELIABILITY_RELATIVE = (
    "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/stage0/reliability_table.json"
)

# E1 frozen assay literals. Label-blind focal spacing; dedicated seed.
E1_N_FOCAL = 8
E1_N_MASKS = 16
E1_RETAIN_FRAC = 0.75
E1_SEED = 20260905
E1_NUMERICAL_EPS = 1.0e-12

# E2 frozen lag set. 100 ms / 20 ms bin = 5 bins. Not a sweep.
E2_LAG_MS = 100
E2_LAG_BINS = 5
E2_FIT_TRIALS = (0, 10)
E2_EVAL_TRIALS = (10, 20)
E2_DYNAMIC_RANK = 9

# P-stream training freeze from the workorder.
P_EPOCHS = 12
P_BATCH = 32
P_SEED = 42
P_LR = 1.0e-4
P_CLIP = 1.0
P_WEIGHT_DECAY = 1.0e-2
P_WARMUP_EPOCHS = 1
P_CARRIER_DIM = 4
P_REVISION_RELATIVE = (
    "tfpd_exploration/docs/REVISION_CROSS_DATASET_P_OPERATOR_V1_20260905.md"
)
P_REVISION_SHA256 = "81ef30c9b08f3a5d36ce9bb7400973aefa344816c5ba27b312a62847e1f90638"
P_REVISION_TIMESTAMP = "20260905_122000"
P_REVISION_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/{P_REVISION_TIMESTAMP}"
P_F_ETA_NAME = "row_normalized_nnmf_nnls_v1"
P_F_ETA_SHAPE = (3, 16)
P_PARENT_CLAIM = "CLEAN_OUTER_SESSION_FILE_EXCLUSION"
P_PREFLIGHT_TIMESTAMP = "20260905_123100"
P_PREFLIGHT_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/{P_PREFLIGHT_TIMESTAMP}"
P_PREFLIGHT_AUTHORITY_RELATIVE = (
    "tfpd_exploration/docs/REVIEW_CROSS_DATASET_P_OPERATOR_PREFLIGHT_FREEZE_V1_20260905.md"
)
P_PREFLIGHT_AUTHORITY_SHA256 = "cf96beae38daad658dec4a08dd031e310e60f85b6f9ce02ee7c07e03bd31d695"
P_NORMALIZER_NAME = "SOURCE_INITIAL_DICTIONARY_FROZEN_NORMALIZER_V1"
P_NNLS_SUPPORT_EPS = 1.0e-12
P_CARRIER_ATOL = 1.0e-8
P_CARRIER_RTOL = 1.0e-8
P_CONSUMER_ATOL = 1.0e-6
P_CONSUMER_RTOL = 1.0e-5
P_DISPOSABLE_PROFILE_STEPS = 100
P_DISPOSABLE_PROFILE_BATCH = 32
P_DISPOSABLE_PROFILE_BUDGET_MINUTES = 30
P_DISPOSABLE_PROFILE_ENV = "CDF_DISPOSABLE_PROFILE"
P_DISPOSABLE_PROFILE_SEED = 202609051
P_DISPOSABLE_PROFILE_TIMERS = (
    "support_read",
    "scipy_nnls_cpu_gpu",
    "active_set_solve",
    "ridge",
    "consumer_fwd_bwd",
    "checkpoint",
)

# Existing checkpoints whose bytes are inventoried. Selection is not implied.
Z_FIX_EPOCH011_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/z_fix/epoch_011.pt"
Z_FIX_EPOCH011_SHA256 = "55d7143b55e5b3e099ecace60f0f6eb30f569301bc471ab36685096e37601ae1"
S_FIX_EPOCH011_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix/epoch_011.pt"
S_FIX_EPOCH011_SHA256 = "7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a"
S_ACYC_EPOCH011_RELATIVE = "tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_acyc/epoch_011.pt"
S_ACYC_EPOCH011_SHA256 = "9708482c92619193517218af68a3bcf5c3c8fc444b09797c3a35e4fee2ddb71b"
FOLD0_SOURCE_TEACHER_RELATIVE = (
    "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/"
    "2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/"
    "checkpoints/best_ckpt/epoch_018.ckpt"
)
FOLD0_SOURCE_TEACHER_SHA256 = "f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be"
FOLD0_SOURCE_TEACHER_MANIFEST_RELATIVE = (
    "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold0/runs/"
    "2026-08-06-16-15-55-070150_rid-m1_afc4_source_decoder_fold0_dev20_resume_e1r1_fNone_s42/"
    "source_only_decoder_manifest.json"
)
ALLSOURCE_TEACHER_RELATIVE = (
    "SPINT-main/logs/train/runs/2026-07-21-19-11-01/checkpoints/best_ckpt/epoch_019.ckpt"
)
ALLSOURCE_TEACHER_SHA256 = "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"
H1_HC_CHECKPOINT_RELATIVE = (
    "SPINT-main/pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/checkpoints/fixed_epoch50/epoch_049.ckpt"
)
H1_HC_CHECKPOINT_SHA256 = "f23e83c9ee8ca6c11d3c6b86410e856d906ccc8c37486aa13ae2e3a2af008fff"

FORBIDDEN_PATH_TOKENS = (
    "held-out",
    "heldout",
    "minival",
    "evalai",
    "formal",
    "private",
    "test_ecephys",
)

OWNED_PATHS = (
    "tfpd_exploration/src/cross_dataset_functional_calibration_v1/",
    "tfpd_exploration/scripts/run_cross_dataset_functional_calibration_v1.py",
    "tfpd_exploration/tests/test_cross_dataset_functional_calibration_v1.py",
    "tfpd_exploration/tests/test_cross_dataset_functional_calibration_e1.py",
    "tfpd_exploration/tests/test_cross_dataset_functional_calibration_e2.py",
    "tfpd_exploration/tests/test_cross_dataset_functional_calibration_p_operator.py",
    "tfpd_exploration/tests/test_cross_dataset_functional_calibration_preflight.py",
    RESULT_ROOT_RELATIVE,
)

SEALED_FOREIGN_ROOTS = (
    "tfpd_exploration/src/m2_dual_track_v1/",
    "tfpd_exploration/results/m2_dual_track_v1/",
)


def focal_channel_indices(n_channels: int) -> tuple[int, ...]:
    """Eight roster-even indices. Label- and performance-blind."""
    _require(int(n_channels) >= E1_N_FOCAL, "roster shorter than eight channels")
    return tuple(int(index * int(n_channels) // E1_N_FOCAL) for index in range(E1_N_FOCAL))


def verify_bound_documents(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for item in BOUND_DOCUMENTS:
        path = Path(root) / str(item["relative"])
        _require(path.is_file(), f"missing bound document {item['relative']}")
        digest = sha256_bytes(path.read_bytes()) if path.stat().st_size < 80_000_000 else sha256_file(path)
        _require(digest == item["sha256"], f"sha drift: {item['relative']}")
        observed[str(item["relative"])] = digest
    workorder = Path(root) / WORKORDER_RELATIVE
    _require(sha256_bytes(workorder.read_bytes()) == WORKORDER_SHA256, "workorder sha drift")
    return observed


def reject_forbidden_path(path: str | Path) -> None:
    text = str(Path(path)).lower()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in text:
            raise PlanError(f"forbidden path token {token!r} in {path}")


def inherited_ridge_objective() -> dict[str, object]:
    """Exact M1 bank ridge objective. Not a naked lambda with a changed n."""
    return {
        "authority": M1_SYN3_OPERATOR_RELATIVE,
        "function": "tfpd_exploration.src.m1_emg_syn3_fcm_v1.syn3.fit_unit_ridge",
        "loss": "normalized_mse_plus_slope_penalty",
        "design": "column_stack(ones, z)",
        "n": "n_paired_rows_after_valid_intersection",
        "gram": "(X^T X) / n",
        "rhs": "(X^T y) / n",
        "penalty": "diag([0, lambda, ..., lambda]) on intercept-unpenalized slopes",
        "lambda": M1_RIDGE_LAMBDA,
        "feature_scaling_before_z": (
            "EMG RMS per channel, floor 1e-8, then source-frozen NNMF/NNLS; "
            "ridge consumes those z scores with no extra z-standardization"
        ),
        "solver": "numpy.linalg.solve on (gram + penalty)",
        "dtype": "float64",
        "normalized_by_n": True,
        "sample_count_changes_lambda": False,
        "note": (
            "Equivalent to min ||n^{-1/2}(X beta - y)||^2 + lambda ||beta_slopes||^2. "
            "Dynamic z has 9 slope columns; lambda stays 1.0 and n is the intersection count."
        ),
    }


def h1_ridge_objective() -> dict[str, object]:
    """Exact H1 frozen backward ridge. Distinct from the M1 /n bank convention."""
    return {
        "authority": H1_OPERATOR_RELATIVE,
        "function": "fit_deployment_carrier",
        "design": "column_stack(ones, z_pca)",
        "system": "X^T X + Lambda, Lambda[0,0]=0, Lambda[i,i]=lambda for i>0",
        "rhs": "X^T Y_velocity",
        "lambda": H1_RIDGE_LAMBDA,
        "normalized_by_n": False,
        "z": "(rates - source_mean) / source_scale @ pcs[:q].T",
        "q": H1_PCA_Q,
        "raw_rows": "(pcs[:q].T @ beta_slopes) / scale",
        "raw_carrier": "raw_rows @ U",
        "eb": "mu + (tau2 / (tau2 + projected_variance)) * (raw_carrier - mu)",
        "direction": H1_ESTIMATOR_DIRECTION,
        "dtype": "float64",
        "solver": "numpy.linalg.solve",
    }


def dry_cli_payload() -> dict[str, object]:
    return {
        "phase": PHASE,
        "contract_version": CONTRACT_VERSION,
        "workorder_sha256": WORKORDER_SHA256,
        "stage0_root": STAGE0_ROOT_RELATIVE,
        "estimators": {
            "h1": h1_ridge_objective(),
            "m1": inherited_ridge_objective(),
        },
        "e1": {
            "n_focal": E1_N_FOCAL,
            "n_masks": E1_N_MASKS,
            "retain_frac": E1_RETAIN_FRAC,
            "seed": E1_SEED,
            "decoder_r2": False,
        },
        "e2": {
            "lag_ms": E2_LAG_MS,
            "lag_bins": E2_LAG_BINS,
            "fit_trials": list(E2_FIT_TRIALS),
            "eval_trials": list(E2_EVAL_TRIALS),
            "adds_lags_to_p_pilot": False,
        },
        "p_stream": {
            "epochs": P_EPOCHS,
            "batch": P_BATCH,
            "seed": P_SEED,
            "parent_bytes": S_FIX_EPOCH011_SHA256,
            "parent_claim": P_PARENT_CLAIM,
            "f_eta_name": P_F_ETA_NAME,
            "f_eta_shape": list(P_F_ETA_SHAPE),
            "differentiable_solve": "torch.linalg.solve matching bank /n ridge",
            "named_revision": P_REVISION_RELATIVE,
            "normalizer": P_NORMALIZER_NAME,
            "preflight_root": P_PREFLIGHT_ROOT_RELATIVE,
            "disposable_profile_steps": P_DISPOSABLE_PROFILE_STEPS,
            "gpu_eligible": False,
        },
        "prospective_roots": {"stage0": STAGE0_ROOT_RELATIVE},
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_gpu_capability": False,
        "public_execution_authorized": False,
        "gpu_work_started": False,
    }


def require_mapping(value: Mapping[str, object], label: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


@dataclass(frozen=True)
class StageRootSpec:
    root_relative: str
