"""Frozen constants and laws for FABLE TKD M2 v1 (Wave 1: model/data/Stage 0).

Authority: ``tfpd_exploration/docs/SPEC_FABLE_TKD_M2_V1_IMPL_20260904.md``
(implementation spec), parent workorder
``tfpd_exploration/docs/WORKORDER_FABLE_TKD_M2_V1_20260904.md``, design
``FABLE0904_decoder_design1.md`` sections 2-3.  CPU only, float32,
deterministic, fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA = "fable_tkd_m2_v1"

#: This file lives at tfpd_exploration/src/fable_tkd_m2_v1/plan.py; the third
#: parent above it is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/fable_tkd_m2_v1"

SPEC_RELATIVE = "tfpd_exploration/docs/SPEC_FABLE_TKD_M2_V1_IMPL_20260904.md"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_FABLE_TKD_M2_V1_20260904.md"

# ---------------------------------------------------------------------------
# Reference anchors (sealed numbers, cited not recomputed).
# ---------------------------------------------------------------------------
#: Sealed M30 champion external equal-session mean (m2_hold_film_probe_v1).
SEALED_M30_EXTERNAL = 0.29521985196829853
#: Context references that do not enter any gate.
M33_EXTERNAL = 0.2991329172968798
MEANS_FILM_EXTERNAL = 0.32102

#: Champion (act30) checkpoint sha256, pinned by the frozen exporter.
CHAMPION_CKPT_SHA256 = (
    "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
)
CHAMPION_NORMALIZATION_SHA256 = (
    "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
)
#: REF-SHUF control checkpoint (champion ts4 whole-row-permutation arm).
TS4_CKPT_SHA256 = (
    "e385e2f408d6b3fe65645be934a8f69b3ab4e68b986e56e64512611cac040399"
)
TS4_CKPT_RELATIVE = (
    "streaming_calibration_exp/outputs/streaming_calibration/"
    "e8_ts4_m2_submission_control_m33q33_v1_s42_20260801_162010/checkpoints/best.ckpt"
)
CHAMPION_RUN_RELATIVE = (
    "streaming_calibration_exp/outputs/streaming_calibration/"
    "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806"
)

# ---------------------------------------------------------------------------
# Data / surface constants.
# ---------------------------------------------------------------------------
CHANNELS = 96
WINDOW = 50
OUT_DIM = 2
ACTIVITY_HORIZON = 30
RIDGE_LAMBDA = 0.1
BEHAVIOR_SCALE = 5.0

# ---------------------------------------------------------------------------
# Model constants (spec section 1).
# ---------------------------------------------------------------------------
D_V = 64
D_K = 64
N_QUERIES = 8
D_H = 128
SSM_LAYERS = 2
CONV_KERNEL = 10
CONV_CHANNELS = 16

# ---------------------------------------------------------------------------
# Training constants (Wave 2 consumes these; frozen now per spec).
# ---------------------------------------------------------------------------
EPOCHS = 30
LR = 3e-4
WD = 0.0
BATCH = 32
SEEDS = (42, 43, 44)
SHUFFLE_SEED = 42
PV_BETA = 4.0

#: The 8 equally spaced direction queries (psi_l = 2*pi*l/8).
QUERY_DIRECTIONS = [2.0 * 3.141592653589793 * l / 8 for l in range(8)]

# ---------------------------------------------------------------------------
# Session rosters (read once from the champion run's split_manifest.json and
# the frozen exporter's dataset objects; hardcoded per spec section 1).
# Source: streaming_calibration_exp/outputs/streaming_calibration/
#   m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/split_manifest.json
# (held-in), and the exporter's val_heldout_dataset roster (external 6,
# matching WORKORDER section 2: 2020-10-30 x2, 11-18, 11-19, 11-24 x2).
# ---------------------------------------------------------------------------
HELDIN_SESSIONS = (
    "ses-2020-10-19-Run1",
    "ses-2020-10-19-Run2",
    "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2",
    "ses-2020-10-27-Run1",
    "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
)
EXTERNAL_SESSIONS = (
    "ses-2020-10-30-Run1",
    "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1",
    "ses-2020-11-24-Run1",
    "ses-2020-11-24-Run2",
)
ALL_SESSIONS = tuple(HELDIN_SESSIONS) + tuple(EXTERNAL_SESSIONS)

# ---------------------------------------------------------------------------
# SHUF permutation: numpy default_rng(42).permutation(96), computed once and
# frozen verbatim (spec section 3.4).  Non-identity asserted in tests and at
# model construction.
# ---------------------------------------------------------------------------
SHUF_PERM = (
    57, 21, 75, 18, 33, 40, 93, 27, 51, 39, 2, 25, 24, 50, 94, 95,
    76, 59, 4, 85, 90, 37, 28, 69, 7, 61, 63, 26, 72, 74, 38, 0,
    82, 68, 10, 29, 42, 52, 17, 87, 73, 5, 91, 1, 46, 80, 88, 3,
    79, 89, 20, 83, 43, 71, 32, 56, 23, 92, 70, 15, 48, 55, 60, 45,
    9, 62, 31, 16, 44, 78, 67, 34, 30, 58, 84, 54, 49, 6, 86, 11,
    81, 41, 22, 19, 35, 64, 47, 12, 53, 14, 13, 66, 36, 65, 77, 8,
)

# ---------------------------------------------------------------------------
# Receipt law (workorder section 8; mirrors the screen's _atomic_json,
# extended with a .sha256 sidecar per spec section 0).
# ---------------------------------------------------------------------------
RECEIPT_LAW = {
    "attempt_first": (
        "results/fable_tkd_m2_v1/attempt.json is created (O_EXCL semantics: "
        "refuse to overwrite an existing attempt) before ANY data or model "
        "access"
    ),
    "atomic_write": "mkstemp + write + fsync + rename, then chmod 0444",
    "sidecar": (
        "every receipt file X gets X.sha256 containing "
        "'<hex>  <name>\\n' (sha256 of the receipt bytes)"
    ),
    "fail_closed": "any failed assert raises after writing failure.json",
    "sealed_readonly": [
        CHAMPION_RUN_RELATIVE,
        "tfpd_exploration/results",
        "SPINT-main",
    ],
}

#: Implementation deviations from the spec, recorded in every terminal
#: receipt (items D1/D2 are pre-approved in the spec itself).
DEVIATIONS = [
    {
        "id": "D1",
        "approved": True,
        "spec": "section 3.1 value sigma = softplus(W_sig @ t + b_sig) + 1e-3",
        "implementation": (
            "sigma = clamp(W_sig @ t + b_sig, min=1e-3) linear (softplus "
            "cannot represent the exact raw-m affine inverse); spec section "
            "3.1 correction note pre-approves this"
        ),
    },
    {
        "id": "D2",
        "approved": True,
        "spec": "section 3.3 Phi_k closed-form construction through the ReLU MLP",
        "implementation": (
            "Phi_k hidden-layer bypass: hidden dims 0..3 carry (t + 50) "
            "through a positive ReLU region (exact affine), hidden dims 4..63 "
            "carry a +0.01 bias only (zero W2 columns at init, alive for "
            "training); Phi_k(t) is the exact affine k = beta*[a, c, 0...]. "
            "q_l . k_i / beta = a*cos(psi) + c*sin(psi) = m_i*cos(angle), "
            "NOT cos(angle) itself: the /m division is not affine in t. "
            "A3's cos assertion is exact on synthetic T4 with m == 1 (where "
            "a*cos+c*sin == cos(angle)); on real T4 the PV anchor uses "
            "m-weighted soft binning. Spec section 3.3 pre-approves the "
            "hidden-layer bypass"
        ),
    },
    {
        "id": "D6",
        "approved": True,
        "spec": "spec section 3.3/section 4-A1 original reference (Wave 1) "
                "y_PV = sum rho (r-b)/m * [cos phi, sin phi]",
        "implementation": (
            "A1 reference replaced per WORKORDER ADDENDUM-1 (2026-09-04) by "
            "the classic depth-weighted PV y_PV = sum_i rho_i (rate10_i - "
            "b_i) [a_i, c_i] (no division, no m normalization); the original "
            "double-1/m reference was dominated by near-zero-m low-rho units "
            "(Wave 1 min corr 0.35).  Note: Wave 1's cited 0.981/0.962 was "
            "measured against the unit-direction variant sum rho (r-b) "
            "[a/m, c/m], not against this depth-weighted form; merge init "
            "defaults to uniform bin masses (exact sum_l dir(psi_l) = 0 "
            "common-mode cancellation)"
        ),
    },
    {
        "id": "D7",
        "approved": True,
        "spec": "spec section 1 PV_BETA = 4.0",
        "implementation": (
            "PV_BETA stays frozen at 4.0 in plan.py; authorized re-tuning of "
            "pv_beta and bin-mass mode (workorder section 9 / ADDENDUM-1 "
            "task) is done via runner CLI arguments recorded per-run in each "
            "receipt and terminal (initial run = spec defaults)"
        ),
    },
    {
        "id": "D8",
        "approved": True,
        "spec": "spec section 3.1 sigma clamp min 1e-3",
        "implementation": (
            "planner-directed 2026-09-05: sigma = clamp(W_sig @ t + b_sig, "
            "min=SIGMA_FLOOR) with SIGMA_FLOOR = 0.0070093626 = median of the "
            "held-in pooled raw m column; kills the 1/m^2 value amplification "
            "for near-zero-m units (PV-init still recovers raw m exactly for "
            "m >= floor); arms/faces/loss/lr/epochs/batch/gates unchanged"
        ),
    },
    {
        "id": "D9",
        "approved": True,
        "spec": "spec section 3.2 pv SSM init a ~ exp(-softplus)=0 "
                "(a_log = 30) with GLU gate bias +4",
        "implementation": (
            "planner-directed 2026-09-05: at PV init decay a = 0.9 "
            "(a_log = inverse_softplus(-ln 0.9)) and GLU gate bias +2.0; "
            "gamma = 0 still makes the layer an exact bitwise pass-through "
            "at init (out = z + 0*o), but d a / d a_log ~= -a*sigmoid(a_log) "
            "is no longer ~1e-13, so the recurrence timescale is trainable "
            "(at a ~ 1e-13 it froze: a_log stayed exactly at init through 30 "
            "epochs, capping within at ~0.12); B=I, C=I, gamma=0 unchanged"
        ),
    },
    {
        "id": "D10",
        "approved": True,
        "spec": "spec section 3.1 P zero-init (implicit)",
        "implementation": (
            "implementation fix 2026-09-05: P ~ N(0, 0.05) instead of zeros; "
            "the (epsilon=0, P=0) pair is a mutual saddle (dL/deps ~ P.u, "
            "dL/dP ~ eps) that left epsilon EXACTLY 0 after 30 epochs in the "
            "first pilot; inert at init since epsilon = 0 (A1 anchor "
            "unchanged bit-exactly)"
        ),
    },
    {
        "id": "D12",
        "approved": True,
        "spec": "spec section 3.1 value law v = rho * (u - mu)/sigma with "
                "sigma recovering per-unit raw m",
        "implementation": (
            "planner-directed 2026-09-05 (ADDENDUM-3): v = rho * (u - mu(t)) "
            "/ S_POOLED with a single global constant S_POOLED = 0.4288818656 "
            "= pooled RMS of (rate10 - b) over held-in post-30 windows; "
            "mu(t) still recovers raw b per unit; W_sig PV-inits to 0 and "
            "b_sig to S_POOLED (clamp min 1e-3 kept for API stability, "
            "inactive); per-unit 1/m removed because depth weighting is "
            "carried by the key scores (m*cos angle, I1 kappa m) and the "
            "per-unit m / median-floor scaling left half the units at "
            "O(20-100) value scale, collapsing training to the zero solution "
            "(pilots r4/r6); arms/faces/gates/schedule unchanged"
        ),
    },
    {
        "id": "D16",
        "approved": True,
        "spec": "spec section 2 item 4/5 (per-epoch selection on the same "
                "post-30 windows used for training)",
        "implementation": (
            "planner-directed 2026-09-05 (ADDENDUM-6): the Wave 2 spec's "
            "epoch selection used the training windows themselves (training-"
            "set selection, structurally picking the most overfit epoch); "
            "replaced by a chronological per-session 80/20 split -- the last "
            "20% of each session's post-30 windows are a disjoint minival "
            "face excluded from training and from teacher-target usage; the "
            "deployment epoch is the minival equal-session-mean argmax.  "
            "Within/external evaluation faces are unchanged; the "
            "external-vs-epoch curve is recorded as a disclosed diagnostic "
            "for the continuation rule ONLY, never for epoch choice"
        ),
    },
    {
        "id": "D14",
        "approved": True,
        "spec": "spec section 2 from-scratch loss (MSE on targets only)",
        "implementation": (
            "planner-directed 2026-09-05 (ADDENDUM-5): distillation main arm "
            "-- L = MSE(readout_out, y*5) + 0.1 * MSE(readout_out, "
            "teacher_out) with the frozen act30 champion as teacher "
            "(last-timestep x5-space outputs on the same held-in post-30 "
            "windows; teacher targets cached once per session with sha "
            "sidecars); SHUF/POOL semantics preserved (teacher is "
            "independent of the student's t permutation); deployment "
            "contract unchanged; from-scratch pilots r4-r8 retained as "
            "negative-result receipts"
        ),
    },
    {
        "id": "D15",
        "approved": True,
        "spec": "spec section 3.2 a = exp(-softplus(a_log)) recurrence",
        "implementation": (
            "planner-directed 2026-09-05 (ADDENDUM-5): SSM state bounding -- "
            "a = 0.98 * sigmoid(a_log_raw) (bounded away from 1, gradients "
            "alive; PV init targets a = 0.9) and a hard state clamp "
            "h in [-10, 10] applied identically in the parallel scan and "
            "the step recurrence (A2 parity re-verified <= 1e-6); "
            "tanh(gamma) retained; GRU arm bounded by construction"
        ),
    },
    {
        "id": "D13",
        "approved": True,
        "spec": "spec section 2 raw-target loss; section 3.3 unscaled merge "
                "init; SSM gamma as a free scalar",
        "implementation": (
            "planner-directed 2026-09-05 (ADDENDUM-4): (a) training loss on "
            "x5.0-scaled targets (readout_out vs y_raw*5.0), the champion "
            "convention -- the /5.0 division stays at eval/deployment only; "
            "(b) init output-scale calibration: merge rows scaled by "
            "PV_INIT_OUTPUT_SCALE = target_rms/anchor_rms (0.2189363672, "
            "held-in closed form) and readout bias = per-dim held-in target "
            "mean in x5 space -- affine, so the A1 correlation is unchanged "
            "up to float rounding; (c) SSM residual gain reparametrized as "
            "gamma = tanh(gamma_raw) (gamma_raw = 0 at PV init keeps the "
            "exact pass-through; loop gain bounded in (-1, 1))"
        ),
    },
    {
        "id": "D11",
        "approved": True,
        "spec": "spec section 3.3 psi conv channels 1..15 zero init",
        "implementation": (
            "implementation fix 2026-09-05: psi_linear columns 1..15 seeded "
            "N(0, 0.01) (conv channels output exactly 0 at init, so u and "
            "the A1 anchor are bit-unchanged) because with zero columns the "
            "conv channels received exactly zero gradient forever (they were "
            "still exactly zero after 30 epochs in the first pilot)"
        ),
    },
]


class TKDError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Wave 2 additions (spec SPEC_FABLE_TKD_M2_V1_WAVE2_20260904 section 1;
# additive only -- every Wave 1 constant above is unchanged).
# ---------------------------------------------------------------------------

#: GPU1 ownership (workorder section 8; Wave 2 spec section 0).
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
GPU_PREFLIGHT = {
    "util_max_percent": 5.0,
    "mem_max_mib": 500.0,
    "foreign_pids": "none tolerated",
}

#: Arms (internal "A2" is the workorder's A-prime main arm).
ARMS = ("A2", "A", "SHUF", "POOL", "B", "A2_GRU")
ARM_TABLE = {
    "A2": dict(epsilon="learnable", key_mode="identity", time_model="ssm",
               init="pv", frozen_readin=False),
    "A": dict(epsilon="zero", key_mode="identity", time_model="ssm",
              init="pv", frozen_readin=False),
    "SHUF": dict(epsilon="learnable", key_mode="shuffled", time_model="ssm",
                 init="pv", frozen_readin=False),
    "POOL": dict(epsilon="learnable", key_mode="pool", time_model="ssm",
                 init="pv", frozen_readin=False),
    "B": dict(epsilon="learnable", key_mode="identity", time_model="ssm",
              init="pv", frozen_readin=True),
    "A2_GRU": dict(epsilon="learnable", key_mode="identity", time_model="gru",
                   init="pv", frozen_readin=False),
}

#: The frozen PV init configuration selected by the ADDENDUM-3 sweep under
#: the D12 value law ({28, 84, 168} x {uniform, authority}; best min A1
#: correlation 0.9975).  Source of truth: the sealed receipt
#: results/fable_tkd_m2_v1/stage0_r4/a1_pv_anchor.json (pv_beta 28.0,
#: uniform masses) -- copied verbatim, NOT re-derived.  (Supersedes the
#: stage0_r2 config, which was optimal under the pre-D12 per-unit sigma law.)
PV_INIT_CONFIG = {
    "pv_beta": 28.0,
    "bin_masses_mode": "uniform",
    "bin_masses": (12.0,) * 8,
    "sealed_source": "tfpd_exploration/results/fable_tkd_m2_v1/stage0_r4/a1_pv_anchor.json",
}

#: REF per-session external R2 (act30 champion, probe p0 arm, full precision).
#: Source: tfpd_exploration/results/m2_hold_film_probe_v1/score.json
#:   arms/p0_zero_film_t4/summaries/external_official_query/per_session_r2
#: (equal-session mean = SEALED_M30_EXTERNAL above).
REF_EXTERNAL_PER_SESSION = {
    "ses-2020-10-30-Run1": 0.4575574651728199,
    "ses-2020-10-30-Run2": 0.413918166109874,
    "ses-2020-11-18-Run1": 0.2907288949699923,
    "ses-2020-11-19-Run1": 0.159329382195729,
    "ses-2020-11-24-Run1": 0.25785181150359004,
    "ses-2020-11-24-Run2": 0.1919333918577859,
}

#: D13(b) (planner-directed 2026-09-05, ADDENDUM-4): init output-scale
#: calibration, held-in closed form over all 102,379 post-30 windows.
#: target_rms(x5, both dims pooled) = 0.0510728574; anchor PV reference RMS
#: = 0.2332771760; scale = 0.2189363672.  readout bias = per-dim held-in
#: target mean in x5 space (essentially zero but kept exact).
PV_INIT_OUTPUT_SCALE = 0.2189363672
PV_READOUT_BIAS_X5 = (-2.0909375204465373e-05, -3.964696518504744e-05)
PV_INIT_SCALE_SOURCE = (
    "held-in post-30 windows, m30 T4 fits digest-bound to "
    "stage0/t4_authority.json; anchor RMS from the explicit depth-weighted PV "
    "reference (ADDENDUM-1 law)"
)

#: D14 (planner-directed 2026-09-05, ADDENDUM-5): distillation weight for the
#: main-arm training loss L = MSE(readout_out, y*5) + DISTILL_LAMBDA *
#: MSE(readout_out, teacher_out); teacher = frozen act30 champion
#: (sha CHAMPION_CKPT_SHA256) last-timestep output in x5 space on the SAME
#: held-in post-30 training windows (teacher identity via its own deployment
#: law: first-30 calibration block + B3S encoder).  Training-time compression
#: only; the deployment contract (TKD + closed-form T4/rho, zero-gradient,
#: streaming) is unchanged.
DISTILL_LAMBDA = 0.1
TEACHER_CACHE_RELATIVE = "tfpd_exploration/results/fable_tkd_m2_v1/teacher_cache"

#: D16 (planner-directed 2026-09-05, ADDENDUM-6): chronological per-session
#: 80/20 split of the post-30 windows -- first 80% train (task loss +
#: distillation), last 20% minival (excluded from training AND teacher-target
#: usage); the per-epoch selection metric is the minival equal-session mean
#: and the deployment epoch is its argmax (tie -> earlier).  Mirrors the
#: champion's disjoint held-in minival selection (which picked epoch 2/12).
MINIVAL_FRACTION = 0.2

GRAD_CLIP = 1.0
EVAL_BATCH = 1024
P0_R2_TOLERANCE = 1.0e-6

#: D8 (planner-directed 2026-09-05, SUPERSEDED BY D12): sigma floor = MEDIAN
#: of the held-in 7 session pooled raw m.  Kept for the audit trail; the
#: active value law is D12's S_POOLED below.
SIGMA_FLOOR = 0.0070093626
SIGMA_FLOOR_SOURCE = (
    "median of held-in pooled raw T4 m column (672 units, m30 law, digests "
    "bound to tfpd_exploration/results/fable_tkd_m2_v1/stage0/t4_authority.json)"
)

#: D12 (planner-directed 2026-09-05, ADDENDUM-3): the value path divides by a
#: single GLOBAL closed-form constant -- the pooled RMS of (rate10 - b) over
#: the held-in 7 sessions' post-30 training windows (all 102,379 windows x 50
#: bins x 96 units, n = 491,419,200 deviations, causal 10-bin mean law
#: mirroring the psi conv; per-session T4 digests verified equal to
#: stage0/t4_authority.json before computation).  Per-unit 1/m is removed:
#: depth weighting comes from the key scores (D6 / ADDENDUM-3).
S_POOLED = 0.4288818656
S_POOLED_SOURCE = (
    "sqrt(sum (rate10_i - b_i)^2 / n) over held-in post-30 windows (all bins, "
    "all units); m30 T4 fits digest-bound to "
    "tfpd_exploration/results/fable_tkd_m2_v1/stage0/t4_authority.json"
)

#: Pilot sanity gate (workorder section 6 Stage 1a; Wave 2 spec section 4).
SANITY_EXTERNAL_RANGE = (0.10, 0.45)
SANITY_MAX_NONMONOTONIC_EPOCHS = 2

#: Run artifacts root (checkpoints + per-run receipts).
RUNS_RELATIVE = "tfpd_exploration/results/fable_tkd_m2_v1/runs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TKDError(message)


def result_root(repo_root: Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / RESULT_ROOT_RELATIVE


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def receipt_body(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_receipt(path: Path, payload: object, *, exclusive: bool = False) -> str:
    """Atomic 0444 JSON receipt with a .sha256 sidecar; returns the hex.

    Mirrors ``m2_t4_activity_budget_screen_v1/physical.py::_atomic_json``
    (mkstemp + fsync + replace + chmod 0444) extended per spec section 0 with
    a same-name ``.sha256`` sidecar whose content is ``<hex>  <name>\\n``.
    ``exclusive=True`` refuses (fail-closed) when the target already exists.
    """
    path = Path(path)
    if exclusive and path.exists():
        raise TKDError(f"refusing to overwrite existing receipt: {path}")
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


def verify_sidecar(path: Path) -> None:
    """Fail-closed check that a receipt and its sidecar agree."""
    path = Path(path)
    body = path.read_bytes()
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing sidecar: {sidecar}")
    expected = f"{sha256_bytes(body)}  {path.name}\n"
    require(sidecar.read_text(encoding="utf-8") == expected,
            f"sidecar mismatch: {sidecar}")
