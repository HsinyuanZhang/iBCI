"""Frozen constants for FABLE TKD M1 v1 (fold-local LOSO fold 0 pilot).

Authority: WORKORDER_FABLE_TKD_M1_V1_20260905.md (frozen).  The face,
carrier bank, metric, and training conventions are mirrored VERBATIM from
``m1_emg_rsyn3_fold_local_v1`` (plan pins re-exported below, not reinvented);
the TKD model class is reused from ``fable_tkd_m2_v1.model`` by
parameterization (N=64, W=100, B=16, GRU-only).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as grandparent_plan

SCHEMA = "fable_tkd_m1_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_FABLE_TKD_M1_V1_20260905.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/fable_tkd_m1_v1"

# ---------------------------------------------------------------------------
# Face (mirrored verbatim from m1_emg_rsyn3_fold_local_v1 / its parent pins).
# ---------------------------------------------------------------------------
SESSIONS = fold_plan.SESSIONS                       # 4 M1 held-in sessions
FOLD0_TARGET_SESSION = fold_plan.FOLD0_TARGET_SESSION        # ses-20120924
FOLD0_SOURCE_SESSIONS = fold_plan.FOLD0_SOURCE_SESSIONS      # 3 sources
SUPPORT_TRIALS = fold_plan.SUPPORT_TRIALS                    # 10
QUERY_START = fold_plan.QUERY_START                          # 10
QUERY_STOP_EXCLUSIVE = fold_plan.QUERY_STOP_EXCLUSIVE        # 210
WINDOW_SIZE = fold_plan.WINDOW_SIZE                          # 100
TRIAL_LENGTH = fold_plan.TRIAL_LENGTH                        # 1024
CHANNELS = 64
OUT_DIM = 16
RANK = fold_plan.RANK                                        # 3
CARRIER_DIM = grandparent_plan.CARRIER_DIM                   # 4
RIDGE_LAMBDA = grandparent_plan.RIDGE_LAMBDA                 # 1.0
BIN_SECONDS = grandparent_plan.BIN_SECONDS                   # 0.02
SEED = fold_plan.SEED                                        # 42
EXPECTED_EVAL_WINDOWS = 26517

#: Digest pin for the fold-0 target M10 raw rSyn3 carrier (sealed Stage-0).
FOLD0_M10_RAW_CARRIER_DIGEST = fold_plan.FOLD0_M10_RAW_CARRIER_DIGEST

# ---------------------------------------------------------------------------
# Champion training conventions (mirrored; NOT the M2 3e-4 schedule).
# ---------------------------------------------------------------------------
EPOCHS = fold_plan.STAGE1_EPOCHS                             # 12
FIXED_LAST_EPOCH_INDEX = fold_plan.FIXED_LAST_EPOCH_INDEX    # 11
LR = fold_plan.STAGE1_LR                                     # 1e-4
WEIGHT_DECAY = fold_plan.STAGE1_WEIGHT_DECAY                 # 0.0
BATCH = fold_plan.STAGE1_BATCH_SIZE                          # 32
GRAD_CLIP = 1.0
DISTILL_LAMBDA = 1.0        # ADDENDUM-5/M1 workorder: pure-compression weight
BEHAVIOR_SCALE = 1.0        # M1 champion: predict_scaled_behavior=false

#: REF (sealed pilot_r3, static face): Z-Fix governing R2.
REF_Z_FIX = 0.6374243497848511
REF_S_FIX = 0.6202908754348755
REF_S_ACYC = 0.6217859983944214

#: Preregistered kill-gates (workorder section 4).
K1_FLOOR = REF_Z_FIX - 0.10      # 0.5374...
K2_FLOOR = 0.02

#: Distillation teacher: the pinned M1 SPINT teacher (m1_b3_allsource_v1
#: plan pin; the decoder the b3s_rsyn3_freeze champion froze).
TEACHER_CKPT_RELATIVE = (
    "SPINT-main/logs/train/runs/2026-07-21-19-11-01/checkpoints/best_ckpt/epoch_019.ckpt"
)
TEACHER_CKPT_SHA256 = "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"

#: Fixed 64-unit SHUF permutation (np.random.default_rng(42).permutation(64),
#: the M2 SHUF law; one permutation for train+deploy).
SHUF_PERM = (
    29, 42, 18, 24, 7, 17, 27, 54, 63, 44, 46, 5, 61, 25, 32, 21,
    60, 20, 56, 50, 55, 58, 51, 4, 26, 15, 59, 23, 28, 40, 9, 37,
    31, 39, 16, 3, 34, 30, 10, 48, 57, 6, 38, 11, 52, 41, 22, 19,
    35, 0, 47, 49, 45, 12, 53, 43, 14, 62, 2, 36, 33, 1, 13, 8,
)

#: A1-M1 anchor bands (workorder section 2).
ANCHOR_PASS = 0.99
ANCHOR_DISCLOSURE = 0.90

#: Anchor key temperature (gamma) for the z-scored synergy-weight keys:
#: best measured min-correlation config from the offline sweep
#: {1,2,4,8,16,32} (gamma=8, min 0.149; the sweep is recorded in the
#: anchor receipt).
ANCHOR_BETA = 8.0

TEACHER_CACHE_RELATIVE = RESULT_ROOT_RELATIVE + "/teacher_cache"

GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"

DEVIATIONS = [
    {
        "id": "M1-D1",
        "approved": True,
        "spec": "workorder section 3 (A' arm, M2 model defaults)",
        "implementation": (
            "GRU time model only (SSM falsified on M2, ADDENDUM-6); M1 "
            "schedule mirrored verbatim from the fold-local source plan "
            "(Adam lr 1e-4, wd 0, batch 32, 12 epochs, FIXED last epoch, "
            "raw 16-dim EMG targets with behavior_scaling_factor=1.0 -- the "
            "champion predict_scaled_behavior=false convention); no minival "
            "selection (fixed-last-epoch fold-local law)"
        ),
    },
    {
        "id": "M1-D2",
        "approved": True,
        "spec": "workorder section 2 (A1-M1 anchor)",
        "implementation": (
            "generative NMF read-in: shat = (W^T W/n + lam I)^-1 W^T (r-b)/n "
            "(the fit_unit_ridge normalized-gram convention, lam=1.0), "
            "yhat = D @ shat; the TKD read-in at init realizes it through "
            "the first-order attention tilt: keys = beta*[w1,w2,w3] (Phi_k "
            "bypass), queries e_0..e_2 + 5 zero queries whose output "
            "directions carry -sum(u)/5 (exact common-mode cancellation), "
            "psi channel 0 = current-bin rate, sigma = pooled RMS of (r-b) "
            "over source training windows (D12 law), readout bias = held-in "
            "target mean (D13b law); the fold-shared output directions use "
            "the source-pooled inverse Pbar = mean_s (W_s^T W_s/n + lam "
            "I)^-1 while each session's keys carry its own W_s (identity "
            "is the read-in)"
        ),
    },
    {
        "id": "M1-D3",
        "approved": True,
        "spec": "workorder section 3 (distillation teacher)",
        "implementation": (
            "teacher = the pinned M1 SPINT teacher decoder (sha c81a2bbd...), "
            "driven with its own identity law (static m10 repeated "
            "calibration, the score_static convention); outputs cached per "
            "source-session training window, digest-sealed; external/target-"
            "fold windows never touch the teacher"
        ),
    },
    {
        "id": "M1-D5",
        "approved": True,
        "spec": "fold-local carrier bank law: bit-expect array_digest equality "
                "with FOLD0_M10_RAW_CARRIER_DIGEST",
        "implementation": (
            "the sealed 2026-09-02 bit-digest no longer reproduces in the "
            "current environment (numeric-library drift: the rebuilt NMF "
            "dictionary differs in last bits; in-process refits remain "
            "bit-deterministic).  Quantified against every numeric field of "
            "the sealed reliability table: ||carrier||_F 99.96695423114362 vs "
            "sealed 99.96695423114359 (rel 3e-16), normalized-gram "
            "eigenvalues and per-synergy dispersions to 1e-13, valid_bins "
            "exact -- the carrier content is numerically identical.  This "
            "package therefore rebuilds the bank with the sealed law VERBATIM "
            "and asserts NUMERIC equality (tolerance 1e-9) against the "
            "sealed fold-0 M10 row instead of the bit digest; both digests "
            "(current and sealed) are recorded in every receipt"
        ),
    },
    {
        "id": "M1-D4",
        "approved": True,
        "spec": "M2 model class reuse",
        "implementation": (
            "TKD parameterized additively (channels/window/out_dim/"
            "behavior_scale/shuf_perm) plus init='nmf'; M2 constants and "
            "behavior are bit-unchanged (16-test suite still green)"
        ),
    },
]


class TKDM1Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TKDM1Error(message)


def result_root(repo_root: Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / RESULT_ROOT_RELATIVE


def receipt_body(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def atomic_receipt(path: Path, payload: object, *, exclusive: bool = False) -> str:
    path = Path(path)
    if exclusive and path.exists():
        raise TKDM1Error(f"refusing to overwrite existing receipt: {path}")
    body = receipt_body(payload)
    digest = sha256_bytes(body)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=directory)
    temporary = Path(temporary_name)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = temporary.with_name(temporary.name + ".sha256")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o444)
        sidecar_body = f"{digest}  {path.name}\n".encode("utf-8")
        with open(sidecar_temporary, "wb") as handle:
            handle.write(sidecar_body)
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_temporary.replace(sidecar)
        sidecar.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)
    return digest
