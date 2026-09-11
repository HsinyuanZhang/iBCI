"""Frozen contract constants for the dandi688 local-benchmark series (bench_v1).

Authority order:
  1. btransform_unified_v2/docs/DESIGN_688_LOCAL_BENCHMARK_PROTOCOL_V1_20260909.md
     (experiment design: split / arms / preregistered gates)
  2. btransform_unified_v2/docs/EXECUTION_688_RIFT_V1_20260907.md
     (frozen data/model contract: prepared_cache, 50-bin window, B3S E0,
      carrier [a_R, c_R, m_R, delta_b], 12 epochs with e8-e11 averaging,
      Nmax padding)

This package is independent of the rift_v1 runner code and only re-uses its
immutable prepared cache read-only.  No formal-test data is ever referenced
beyond a name-level rejection list.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

# --- identity -----------------------------------------------------------
SCHEMA = "dandi688_bench_v1"
# Carrier axis (t4/f0/ts4) is orthogonal to the identity-fusion axis.
# User ruling 2026-09-09 (verbatim): "M1/H1 上 add 是必须的（维度问题）；
# M2/688 上 concat 和 add（joint）都要试".
#   - M1/H1: add (proj_add) is mandatory -- the raw E0 width (700/100)
#     cannot be summed onto the 16 local conv channels without a projection;
#     those settled builds freeze proj_add and are NOT re-opened here.
#   - M2/688: both fusion modes are tested in parallel.  t4/f0/ts4 keep the
#     settled proj_add frontend (P: 50->16 added onto local, token_in 20);
#     t4_concat is the true carrier through the matched concat frontend
#     (v2 concat_model: [local16 | E0_50 | carrier4], token_in 70).  The
#     concat arm reports val-face readings but never enters the gates.
#
# vstate-688 series (user directive 2026-09-09, verbatim): "立刻开始准备
# vstate-688，注意消融——z 系列 = 完全不在新日期校准（bank 复用旧日期/source），
# f 系列 = 不使用任何标签校准（label-free）".
#   - vstate           main arm: signed-velocity-state carrier (dense
#                     cursor_vel calibration labels; see VSTATE_* below).
#   - vstate_concat    M2-mainline fusion reading of the vstate carrier
#                     (alignment matrix row "fusion": M2 trains R50 D4
#                     concat, so the maximal-correspondence bench needs a
#                     vstate-carrier x concat-frontend arm).  Carrier
#                     transform identical to vstate; only the model's
#                     identity fusion differs (token_in 70).  Readings
#                     only, never enters any gate (same law as t4_concat).
#   - z_vstate_srcbank z-series ablation ("完全不在新日期校准"): model and
#                     training identical to vstate, but each exam session's
#                     bank (E0 + carrier) is swapped for a train session's
#                     bank via the frozen nearest-date map (Z_SRCBANK_MAPS)
#                     at load time -- zero calibration on the new date.
#                     NB: the legacy arm name "z0" already means "E0 zeroed"
#                     (query-only); the new series deliberately uses the
#                     z_<...> prefix to avoid that collision.
#   - f_labelfree      f-series ablation ("不使用任何标签校准"): E0 and
#                     carrier use no behavioral labels at all -- E0 =
#                     post_pool(cat(pre_pool activity mean, zero side))
#                     (the ACTIVITY-ONLY identity, the clean ablation the
#                     earlier f0 diagnosis was missing); carrier = zero.
#                     Measures the floor of pure activity calibration, and
#                     vstate - f_labelfree is the total label-information
#                     increment (the quantity f0 could not isolate, because
#                     f0 keeps the label-melted frozen E0).
#   - norm_only        NORM_ONLY zero-calibration baseline (guide
#                     NORM_ONLY_ZERO_BASELINE_GUIDE_20260910.md sections
#                     2/3/6): E0 = the day's label-free per-unit pooled-rate
#                     DEVIATION from the source (train-session) rate
#                     distribution, broadcast over e0_dim; carrier = zero;
#                     input = raw neural data, untouched.  The identity lives
#                     in the norm_only variant cache bytes (src/.../
#                     norm_only.py); the arm itself is the same zero-carrier
#                     + identity-E0 law as f_labelfree.
ARMS = ("t4", "f0", "ts4", "t4_concat", "z0", "floor", "equiv_zero",
        "vstate", "vstate_concat", "z_vstate_srcbank", "f_labelfree",
        "norm_only")
FUSION_MODES = ("proj_add", "concat")
LOCAL_CONV_CHANNELS = 16  # frozen v1 CONV_CHANNELS width of the local k5 block

# --- component ablation (user clarification 2026-09-09, verbatim: ----------
# "我不是要对比离散标签和连续标签，而是 carrier + activity 对于跨 session
# 能力的消融实验，各自有多少效果"; final ruling same day: NESTED LADDER,
# verbatim "算 carrier 已经拿到 activity 信息，不用白不用") ------------------
# The MAIN comparison axis of this package is the nested-ladder contribution
# decomposition on top of t4 (the discrete directional-label baseline),
# judged on the exp1_narrow exam face (the cross-session face); the vstate
# series below is AUXILIARY exploration.
#
# Ladder (information NESTING: computing the carrier requires reading the
# calibration neural data, which already yields the activity identity --
# "有标签无身份" is not a realistic deployment cell, so NO CARRIER_ONLY rung
# in the preregistered decomposition):
#   FULL           t4            E0 = activity+carrier side melt, direct T on
#   ACTIVITY_ONLY  f_labelfree   E0 = zero-side remelt (encoder active), T = 0
#   NORM_ONLY      norm_only     E0 = per-unit pooled-rate deviation from the
#                                source rate distribution, broadcast over
#                                e0_dim (NO encoder, NO labels), T = 0
#                                (guide NORM_ONLY_ZERO_BASELINE_GUIDE_
#                                20260910.md -- the domain-standard
#                                per-session normalization baseline)
#   NONE           floor         E0 = 0, T = 0
#   equiv_zero     (control)     param-matched fixed random projection of the
#                                pre_pool activity mean (seed 42, never
#                                trained) + T = 0 -- separates "missing
#                                information" from "missing pathway/capacity".
#                                AUXILIARY control, not a ladder rung.
# The 2026-09-10 NORM_ONLY rung is inserted between NONE and ACTIVITY_ONLY;
# COMPONENT_LADDER_ORDER below is the authoritative ascending order and is
# re-asserted fail-closed at import (see _validate_component_ablation).
# Demoted to auxiliary by the same ruling: z0 (carrier-only cell; kept as an
# arm for exploration, never enters the preregistered decomposition) and
# z_vstate_srcbank.  z0 was REDEFINED from both-zero (query-only Z_NONE) to
# carrier-only by the 2026-09-09 clarification; the both-zero cell is now
# the floor arm, and the archived train_exp1_narrow_z0 results measure
# today's floor.  The random-projection melt of equiv_zero lives in its
# variant cache (scripts/build_vstate_cache.py); see src/.../equiv_zero.py.
COMPONENT_ABLATION: dict[str, Any] = {
    "schema": "dandi688_bench_v1_component_ablation",
    "clarification": "carrier + activity contributions to cross-session "
                     "ability (user 2026-09-09), based on t4 discrete labels; "
                     "nested-ladder final ruling same day: 算 carrier 已经拿到 "
                     "activity 信息，不用白不用",
    "base_arm": "t4",
    # ascending four-level E0 ladder (NORM_ONLY rung added 2026-09-10 by the
    # NORM_ONLY zero-baseline guide); the insertion order is the ladder.
    "ladder": {
        "NONE": "floor",
        "NORM_ONLY": "norm_only",
        "ACTIVITY_ONLY": "f_labelfree",
        "FULL": "t4",
    },
    # rung -> (E0 pathway, direct carrier token, what the rung measures)
    "rungs": {
        "floor": ("zero", "zero", "no-calibration-information floor"),
        "norm_only": ("per-unit pooled-rate deviation from the source rate "
                      "distribution (label-free), broadcast over e0_dim; NO "
                      "encoder pathway, NO labels, no trained identity",
                      "zero",
                      "domain-standard per-session normalization baseline "
                      "(zero-calibration cost: one mean/std subtraction per "
                      "unit from the day's own label-free rates)"),
        "f_labelfree": ("activity only (zero-side remelt; encoder weights "
                        "kept, all labels removed)", "zero",
                        "label-free activity calibration alone"),
        "t4": ("activity + carrier side (frozen label-melted E0)", "carrier",
               "full system"),
    },
    "control": {
        "equiv_zero": ("param-matched fixed random projection of the pre_pool "
                       "activity mean (no identity, no labels)", "zero",
                       "pathway/capacity control; AUXILIARY, not a rung"),
    },
    "auxiliary_arms": {
        "z0": ("zero", "carrier", "carrier-only cell; exploratory reading "
                         "only -- dropped from the preregistered ladder by "
                         "the nesting ruling (carrier presupposes activity)"),
        "z_vstate_srcbank": ("reused old-date bank", "identity",
                             "vstate-series; exploratory only"),
    },
    "f_labelfree_vs_floor": (
        "f_labelfree KEEPS the E0 pathway: the frozen B3S encoder weights and "
        "the activity content survive, only the behavioral-label side input is "
        "zeroed (E0 = post_pool(cat(pre_pool activity mean, zero side))).  "
        "floor REMOVES E0 entirely (zero tensor) plus the carrier token: the "
        "model gets only raw spikes + weights.  Their difference "
        "f_labelfree - floor isolates the PURE-ACTIVITY component of E0 "
        "(label-free)."
    ),
    "formulas": {
        "activity_independent": "f_labelfree - floor",
        "carrier_label_increment": "t4 - f_labelfree",
        "calibration_total": "t4 - floor",
    },
    "auxiliary_formulas": {
        "equiv_zero_pathway_value": "equiv_zero - floor",
        "equiv_zero_information_value": "t4 - equiv_zero",
        "z0_carrier_only_reading": "t4 - z0",
        "direct_carrier_marginal": "t4 - f0",
    },
    # four-level refinement of the preregistered 3-item decomposition: the
    # NORM_ONLY rung splits "activity_independent" into the zero-calibration
    # rate-deviation rung and the encoder-activity increment on top of it.
    # The 3-item formulas above stay frozen (they are the preregistered
    # decomposition of the 2026-09-09 ruling); these live beside them.
    "norm_only_formulas": {
        "rate_deviation_identity": "norm_only - floor",
        "encoder_activity_increment": "f_labelfree - norm_only",
        "label_free_activity_total": "f_labelfree - floor",
    },
    "norm_only_vs_f_labelfree": (
        "NORM_ONLY carries NO encoder pathway: E0 is the broadcast per-unit "
        "rate deviation itself, so the rung costs one mean/std subtraction "
        "per unit and no trained parameter.  f_labelfree (ACTIVITY_ONLY) "
        "passes the day's activity through the FROZEN B3S encoder weights "
        "with a zero side.  f_labelfree - norm_only therefore isolates "
        "exactly what the trained encoder pathway adds on top of the "
        "domain-standard normalization (representation richness, not the "
        "support set: both consume the SAME label-free calibration support)."
    ),
    "judged_on": "exp1_narrow exam face: the 4 cross-date sessions "
                 "(sub-C_ses-CO-20150713/0714/0715/0716) -- the cross-session "
                 "face; equal-session mean R^2",
    "secondary": "ts4 content control on the same face (channel<->carrier "
                 "correspondence destroyed, marginals identical)",
    "note_z0_redefinition": (
        "z0 was redefined from both-zero (query-only Z_NONE) to carrier-only "
        "by the 2026-09-09 clarification; the both-zero cell is the floor "
        "arm.  Archived results trained under the old z0 definition "
        "(results/train_exp1_narrow_z0) measure today's floor.  The nested-"
        "ladder ruling then demoted z0 (and z_vstate_srcbank) to auxiliary "
        "exploration: computing the carrier presupposes reading the "
        "calibration neural data, so a carrier-only cell is not a realistic "
        "deployment rung."
    ),
}

# --- NORM_ONLY zero-calibration baseline (guide -----------------------------
# NORM_ONLY_ZERO_BASELINE_GUIDE_20260910.md sections 2/3/4/5/6; the rung is
# implemented in src/dandi688_bench_v1/norm_only.py and baked into the
# "norm_only" variant cache by scripts/build_vstate_cache.py) ----------------
# Ascending ladder order (arm names); the rung names of COMPONENT_ABLATION
# ["ladder"] bind onto this order fail-closed at import.
COMPONENT_LADDER_ORDER = ("floor", "norm_only", "f_labelfree", "t4")
COMPONENT_LADDER_RUNGS = ("NONE", "NORM_ONLY", "ACTIVITY_ONLY", "FULL")
# Identity construction (guide section 3): z[u] = (rate_sess[u] - mu_src[u])
# / sigma_src[u], broadcast over e0_dim; carrier all-zero; input raw.
NORM_ONLY_VARIANT = "norm_only"
NORM_ONLY_CACHE_DIRNAME = "cache_norm_only"
NORM_ONLY_SIGMA_FLOOR = 1e-6  # guide section 3: sigma floor (kept verbatim)
NORM_ONLY_RATE_BIN_SECONDS = 0.02  # 688 BIN_MS = 20 (frozen contract)
NORM_ONLY_RATE_UNITS = (
    "counts per 20 ms bin (688 rate primitive family; the Hz value is this "
    "x 50 -- the constant factor cancels inside z, so it is not material)"
)
# the label-free calibration support the ACTIVITY_ONLY (f_labelfree) arm's E0
# consumes, hence the same support NORM_ONLY must pool over (guide section 3
# "支持集 = 该数据集 ACTIVITY_ONLY 臂 E0 所用的同一个支持集"): the
# ACTIVITY_SUPPORT_N = 30 first rewarded trials stored in the session's
# calib_trials tensor (multisession_datamodule._build_calib_trials).
NORM_ONLY_SUPPORT_TRIALS = 30  # = mc_maze.dandi688_sparse_event_t4_v1.plan
#                                 .ACTIVITY_SUPPORT_N = cp_film_v1.plan
#                                 .ACTIVITY_SUPPORT
NORM_ONLY_SUPPORT_NOTE = (
    "calib_trials [trial, TRIAL_LENGTH=100, unit] of the session record: 688 "
    "ACTIVITY_SUPPORT_N = 30 first rewarded trials, whole-trial windows, "
    "cubic-resampled to TRIAL_LENGTH bins (the exact tensor the ACTIVITY_ONLY "
    "arm's E0 encoder receives)"
)
# per-slot rule when the source list has no session carrying that unit slot
NORM_ONLY_UNDEFINED_SLOT_Z = 0.0
# a slot needs at least this many SOURCE sessions to be defined.  Two-sample
# source spreads are the 688 slot-misalignment artifact (sessions carry
# different neuron populations, so a late slot's "source distribution" can be
# built from 2 sessions whose rates nearly coincide): with 2 the exp1_narrow
# exam identities reached |z| = 130/79/88 (single-slot spikes, physically
# meaningless 80-130 sigma), i.e. far outside the O(1) scale the guide
# requires; with 3 the observed maximum is |z| = 8.3 while 55/91 source slots
# stay defined (3-11 of the exam sessions' 58-66 rows fall back to
# NORM_ONLY_UNDEFINED_SLOT_Z).  This is a DEFINITION threshold (which slots
# have a source reference at all), not a transformation of the identity:
# z = (rate - mu)/sigma is the guide's section-3 formula verbatim, and the
# 1e-6 sigma floor is kept unchanged.
NORM_ONLY_MIN_SOURCE_SESSIONS = 3
# fail-closed guard: a genuine per-unit rate deviation on a source
# distribution with a real spread stays far below this bound (the M1 HO
# rate lift of ~2.1x over a ~0.3 CV is z ~ 3.7; the 688 exp1_narrow exam
# identities peak at |z| = 8.3).  Exceeding it means the per-slot sigma is
# degenerate (floor hits), not that the day's rates drifted.
NORM_ONLY_MAX_ABS_Z = 20.0
# sanity checks of the ladder reading (guide section 5)
NORM_ONLY_SANITY_CHECKS: dict[str, str] = {
    "monotone": "NONE <= NORM_ONLY <= ACTIVITY_ONLY <= FULL (per task, the "
                "same face)",
    "construction_error": "NORM_ONLY < NONE -> the construction is wrong "
                          "(normalization made it worse): STOP and debug, do "
                          "not report",
    "stronger_than_activity": "NORM_ONLY > ACTIVITY_ONLY -> the mean-rate "
                              "identity beats the encoder: report honestly "
                              "and revisit the ACTIVITY_ONLY design",
    "dc_penalty_disclosure": "NORM_ONLY > 0.1 while the DC penalty (raw "
                             "chR2 - DC-centered chR2) is still > 0.1 -> the "
                             "offset is not fixed: attach the DC penalty "
                             "whenever the rung is quoted in the main table",
}
# guide section 4/5 main criteria of the other tasks (recorded for the M1/
# M2/H1 implementations of the same rung; the 688 face uses the ladder law
# above against floor / f_labelfree / t4 of EXP1_NARROW_BASELINE_R2).
NORM_ONLY_TASK_CRITERIA: dict[str, dict[str, float | str]] = {
    "M1": {"metric": "HO3 channel-weighted R2", "min": 0.3,
           "dc_penalty_max": 0.1, "e0_dim": 100},
    "M2": {"metric": "EXT6 equal-weight R2", "min": 0.15, "e0_dim": 50},
    "H1": {"metric": "HO-M3 grouped-7", "min": 0.25, "e0_dim": 700},
}


def norm_only_ladder_report(
    r2_floor: float,
    r2_norm_only: float,
    r2_f_labelfree: float,
    r2_t4: float,
    dc_penalty: float | None = None,
) -> dict[str, Any]:
    """Four-level ladder reading of the NORM_ONLY rung on ONE face (guide
    sections 5/6): recorded readings + the guide's sanity verdicts.  Pure
    function.

    Fails closed on the guide's "construction error" condition (NORM_ONLY
    below the NONE floor) -- that reading means the identity is wrong, so the
    caller must not consume it.  Every other verdict is a recorded reading,
    never a gate."""
    for name, value in (("floor", r2_floor), ("norm_only", r2_norm_only),
                        ("f_labelfree", r2_f_labelfree), ("t4", r2_t4)):
        require(isinstance(value, (int, float)) and np.isfinite(float(value)),
                f"ladder reading {name} must be a finite number, got {value!r}")
    r2 = {"floor": float(r2_floor), "norm_only": float(r2_norm_only),
          "f_labelfree": float(r2_f_labelfree), "t4": float(r2_t4)}
    if r2["norm_only"] < r2["floor"]:
        raise RuntimeError(
            "NORM_ONLY construction error (guide section 5): norm_only "
            f"{r2['norm_only']:.6f} < NONE floor {r2['floor']:.6f} -- "
            "normalization made the reading worse; STOP and debug the "
            "identity, do not report this rung"
        )
    monotone = (r2["floor"] <= r2["norm_only"] <= r2["f_labelfree"]
                <= r2["t4"])
    report: dict[str, Any] = {
        "schema": SCHEMA + "_norm_only_ladder",
        "ladder_order": list(COMPONENT_LADDER_ORDER),
        "rungs": dict(COMPONENT_ABLATION["ladder"]),
        "r2": r2,
        "deltas": {
            "rate_deviation_identity": r2["norm_only"] - r2["floor"],
            "encoder_activity_increment": r2["f_labelfree"] - r2["norm_only"],
            "label_free_activity_total": r2["f_labelfree"] - r2["floor"],
            "carrier_label_increment": r2["t4"] - r2["f_labelfree"],
            "calibration_total": r2["t4"] - r2["floor"],
        },
        "sanity": {
            "monotone": monotone,
            "construction_error": False,
            "stronger_than_activity": r2["norm_only"] > r2["f_labelfree"],
            "dc_penalty": None if dc_penalty is None else float(dc_penalty),
            "dc_penalty_flag": (
                None if dc_penalty is None
                else bool(r2["norm_only"] > 0.1 and float(dc_penalty) > 0.1)
            ),
        },
        "laws": dict(NORM_ONLY_SANITY_CHECKS),
        "formal_test_policy": FORMAL_TEST_POLICY,
    }
    return report

# --- vstate-688 recipe constants (M2 vstate4 recipe, PLAN_CARRIER_ITERATION_ --
# M2_688_20260909.md section 3.2, adapted to the 688 rate primitives; per-part
# M2-alignment audit: btransform_unified_v2/docs/
# CARRIER_M2_688_ALIGNMENT_MATRIX_20260909.md) -------------------------------
# 100 ms non-overlapping blocks inside the frozen R700/H300 windows; rates =
# spike-time searchsorted half-open counts / block duration (the 688 rate
# primitive family; bin-sum parity audited exact on grid-aligned edges);
# state W = [softplus(v_x/rms_x), softplus(-v_x/rms_x), softplus(v_y/rms_y),
# softplus(-v_y/rms_y)] with v = cursor_vel block mean; rms calibrated on
# TRAIN sessions only (exp1_narrow = the 9 train sessions, exp2_full = the
# 27); Poisson standardization with delta = block duration + n0 = 10-block
# shrinkage; readout a = R(+x) - R(-x), c = R(+y) - R(-y), m = hypot(a, c),
# b = mean_k R  (MAIN variant, M2 vstate4-identical fourth column -- user
# ruling 2026-09-09 final: 最大对应优先; the original delta_b readout
# b = mean_k R - mean_k R_hold is preserved as the switchable b_mode
# "hold_diff" ablation, cache variant vstate_b_hold, with a/c/m unchanged);
# column normalization train-only mean/std.  Support budget of the MAIN
# variant is the SAME M10 trial set as the t4 arm (candidate namespace,
# chronological first 10 rewarded trials); the vstate_full cache variant
# mirrors M2's "all support" semantics on all 30 activity-support trials.
VSTATE_BLOCK_SECONDS = 0.1          # 100 ms blocks
VSTATE_N0_BLOCKS = 10.0             # shrinkage: 10 pseudo-blocks per state
VSTATE_R700_SECONDS = 0.7           # frozen R700 motion window -> 7 blocks
VSTATE_H300_SECONDS = 0.3           # frozen H300 hold window   -> 3 blocks
VSTATE_STATE_COLUMNS = ("+x", "-x", "+y", "-y")
VSTATE_VELOCITY_SUBBINS = 5         # 5 x 20 ms bin centers inside each block
VSTATE_VELOCITY_BIN_SECONDS = 0.02  # 688 BIN_SIZE_MS = 20 (frozen contract)
VSTATE_SUPPORT_NAMESPACE = "candidate"   # same M10 support as the t4 arm
VSTATE_SUPPORT_POSITIONS = tuple(range(10))
# vstate_full cache variant (alignment matrix "support" row): all 30
# activity-support trials of the reliability-audit namespace, mirroring M2's
# use of ALL calib trials (M33); the legality gate is the 688 necessity.
VSTATE_FULL_SUPPORT_NAMESPACE = "reliability_audit"
VSTATE_FULL_SUPPORT_POSITIONS = tuple(range(30))  # = se_plan.ACTIVITY_SUPPORT_N
# fourth-column readout modes (alignment matrix "readout" row):
VSTATE_B_MODES = ("mean_k", "hold_diff")
VSTATE_B_MODE_MAIN = "mean_k"  # M2-identical b = mean_k R_u (ruling 2026-09-09)
VSTATE_HOLD = "signed_state_h300_same_affine_n0_10"  # hold treatment of the
# b_mode="hold_diff" ablation: R_hold uses the H300 blocks' own signed-state
# weights and the R700-fitted Poisson affine.
# cache variants of the vstate family (built by scripts/build_vstate_cache.py;
# consumed by the vstate / vstate_concat arms via --prepared-cache):
VSTATE_VARIANTS = ("vstate", "vstate_full", "vstate_b_hold")

# --- z-series (z_vstate_srcbank) frozen bank-source map ---------------------
# User directive 2026-09-09: the exam session's bank must NOT use its own
# date's calibration data; E0 and carrier are both reused from the train
# side.  Deterministic rule: nearest train session by |date difference|
# (ties broken toward the earlier date); every exam date must be STRICTLY
# later than its mapped train date (leakage law, asserted at import and at
# load time).  Under this rule every exp1_narrow exam session maps to the
# last train session 2015-07-10, and every exp2_full val session maps to
# the last train session 2015-07-16 -- the honest degeneracy of "deploy on
# a new date carrying only the most recent old bank".
Z_SRCBANK_RULE = (
    "nearest train session by absolute date difference; ties -> earlier "
    "date; exam date must be strictly later than the mapped train date"
)
Z_SRCBANK_MAPS: dict[str, dict[str, str]] = {
    "exp1_narrow": {
        "sub-C_ses-CO-20150713": "sub-C_ses-CO-20150710",
        "sub-C_ses-CO-20150714": "sub-C_ses-CO-20150710",
        "sub-C_ses-CO-20150715": "sub-C_ses-CO-20150710",
        "sub-C_ses-CO-20150716": "sub-C_ses-CO-20150710",
    },
    "exp2_full": {
        "sub-C_ses-CO-20151103": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151104": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151106": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151109": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151110": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151112": "sub-C_ses-CO-20150716",
    },
    # poverty face (2026-09-10): nearest-date rule maps every exam session to
    # the poverty window's last train session 2015-07-01.  Defined for law
    # totality (every protocol carries a z-table); no poverty z-arm is
    # scheduled by the 2026-09-10 directive.
    "exp1_poverty": {
        "sub-C_ses-CO-20150713": "sub-C_ses-CO-20150701",
        "sub-C_ses-CO-20150714": "sub-C_ses-CO-20150701",
        "sub-C_ses-CO-20150715": "sub-C_ses-CO-20150701",
        "sub-C_ses-CO-20150716": "sub-C_ses-CO-20150701",
    },
    # 2015-only remote exam (2026-09-09): nearest-date rule maps every val
    # session to the 2015 train face's last session 2015-07-16.  Defined for
    # law totality; no exp2015_full z-arm is scheduled.
    "exp2015_full": {
        "sub-C_ses-CO-20151103": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151104": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151106": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151109": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151110": "sub-C_ses-CO-20150716",
        "sub-C_ses-CO-20151112": "sub-C_ses-CO-20150716",
    },
    # 2016 remote exam (2026-09-10 ruling, first-91-by-table-order): every
    # 2016 exam session maps to the train face's last session 2015-07-16
    # (nearest by ~14 months).  Defined for law totality; no exp2016 z-arm
    # is scheduled.
    "exp2016": {
        name: "sub-C_ses-CO-20150716" for name in (
            "sub-C_ses-CO-20160909", "sub-C_ses-CO-20160912",
            "sub-C_ses-CO-20160914", "sub-C_ses-CO-20160915",
            "sub-C_ses-CO-20160919", "sub-C_ses-CO-20160921",
            "sub-C_ses-CO-20160923", "sub-C_ses-CO-20160929",
            "sub-C_ses-CO-20161005", "sub-C_ses-CO-20161006",
            "sub-C_ses-CO-20161007", "sub-C_ses-CO-20161011",
            "sub-C_ses-CO-20161013", "sub-C_ses-CO-20161021",
        )
    },
}

# --- prepared-cache variant binding for the vstate-688 arms -----------------
# The new arms are only legal on their matching variant cache family (the
# frozen contract-v2 cache carries no "variant" field and backs the five
# original arms unchanged).  Values are the ALLOWED variant sets: the
# vstate / vstate_concat arms accept every vstate-family variant (vstate =
# M10 mean_k main; vstate_full = M30 mean_k "all support"; vstate_b_hold =
# M10 delta_b ablation -- alignment-matrix variants), while z_vstate_srcbank
# is defined against the MAIN variant only.  Every vstate-family cache is
# additionally protocol-bound (its rms/column normalizer was fit on that
# protocol's train sessions); f_labelfree is label-free and protocol-free;
# norm_only is label-free but protocol-bound (its source rate distribution is
# fit on the protocol's train sessions).  PROTOCOL_BOUND_CACHE_VARIANTS above
# is the authoritative set consumed by assert_cache_variant_for_arm.
ARM_REQUIRED_CACHE_VARIANT: dict[str, frozenset[str]] = {
    "vstate": frozenset(VSTATE_VARIANTS),
    "vstate_concat": frozenset(VSTATE_VARIANTS),
    "z_vstate_srcbank": frozenset({"vstate"}),
    "f_labelfree": frozenset({"f_labelfree"}),
    # equiv_zero (component-ablation capacity control): the fixed-random-
    # projection E0 melt lives in the equiv_zero variant cache bytes; the
    # arm-level transform is the same zero-carrier law as f_labelfree
    # (belt-and-suspenders -- see arms.py).  Protocol-free: the random
    # projection consumes no labels and no protocol-bound normalizer.
    "equiv_zero": frozenset({"equiv_zero"}),
    # norm_only (NORM_ONLY zero-calibration baseline, guide 2026-09-10): the
    # broadcast rate-deviation identity lives in the norm_only variant cache
    # bytes; the arm-level transform is the same zero-carrier + identity-E0
    # law as f_labelfree.  Protocol-bound: mu_src/sigma_src are fit on the
    # ACTIVE protocol's train sessions (the source list is recorded in the
    # cache contract), so the cache cannot be reused across protocols.
    "norm_only": frozenset({"norm_only"}),
}
# cache variants whose recorded estimator is bound to the protocol it was
# BUILT for (assert_cache_variant_for_arm refuses a protocol mismatch):
# the vstate family (train-only rms + column normalizer) and norm_only
# (train-session source rate distribution).  Label-free, protocol-free
# variants (f_labelfree, equiv_zero) stay out of this set.
PROTOCOL_BOUND_CACHE_VARIANTS = tuple(VSTATE_VARIANTS) + (NORM_ONLY_VARIANT,)
# arms of the component-ablation matrix that need NO variant cache: t4/f0/z0/
# floor all run on the frozen contract-v2 cache (their cells are produced by
# the load-time transforms alone); f_labelfree/equiv_zero carry their E0 melts
# in their own variant caches.
COMPONENT_ARMS_ON_FROZEN_CACHE = ("t4", "f0", "z0", "floor")

# --- preregistered acceptance gates (design doc section 3) ---------------
GATE_T4_MINUS_F0 = 0.03    # val-face equal-session mean R2, T4 - F0 >= +0.03
GATE_T4_MINUS_TS4 = 0.03   # val-face equal-session mean R2, T4 - TS4 >= +0.03
GATE_VSTATE_MINUS_T4 = 0.03  # exam-face equal-session mean R2, vstate - t4 >= +0.03
REF_SPINT_T4_DEV6 = 0.5750  # sealed FABLE TKD pre-check reference (dev-6 face)
NONINFERIORITY_TOLERANCE = 0.01  # RIFT+T4 >= REF - 0.01 counts as transferable

# --- evaluation-face policy ----------------------------------------------
VAL_FACE_ROLE = "ext6-equivalent: 6 val sessions"
FORMAL_TEST_POLICY = (
    "sealed until all preregistered val-face verdicts land; one-shot unseal"
)

# --- GPU discipline (2026-09-09 revision unblocked training) -----------
GPU_POLICY = "ALLOWED_REVISION_20260909"
GPU_POLICY_NOTE = (
    "GPU training was unblocked by the user-authorized 2026-09-09 "
    "revision (carrier-iteration wave).  --stage train / --stage score "
    "accept --device (default cuda:0); --stage preflight remains CPU-only."
)
CPU_THREADS_DEFAULT = 2

# --- model-geometry ablation overrides (user directive 2026-09-10) ---------
# Orthogonal to the arm axis: SAME arms (t4 / f_labelfree), SAME caches, SAME
# training recipe; only the decoder geometry changes (runner --model-override,
# recorded in the bench contract so train/score stays fail-closed bound).
# Readings only -- these cells never enter the preregistered gates.
#   shortwin  W 50 -> 10 bins (200 ms of raw history): each frozen 50-bin
#             window contributes only its LAST 10 bins as model input; the
#             query targets stay byte-identical to the W=50 runs, so the
#             condition isolates exactly the raw-history budget (D1).
#   shallow   temporal depth 4 -> 1: a single local-attention layer spanning
#             the SAME 50-bin receptive field (bench-local subclass
#             dandi688_bench_v1.model_variants.RiftShallowDecoder; the
#             rift_v1 main package is never modified) (D2).
MODEL_OVERRIDES = ("none", "shortwin", "shallow")
SHORTWIN_WINDOW_BINS = 10
SHALLOW_TEMPORAL_LAYERS = 1

# --- ablation-condition comparison baseline (user directive 2026-09-10) ----
# Sealed exp1_narrow exam-face readings of the nested ladder (2026-09-09 runs;
# results/train_exp1_narrow_{t4,f_labelfree,floor}/score_receipt_*.json).  The
# 2026-09-10 condition axis (poverty / dir16 / shortwin / shallow) compares
# against these readings side by side; it never re-opens or re-trains them.
EXP1_NARROW_BASELINE_R2 = {
    "t4": 0.8729465513441879,
    "f_labelfree": 0.8256967064497345,
    "floor": 0.3246591735072727,
}
EXP1_NARROW_BASELINE_SOURCE = (
    "sealed score receipts of results/train_exp1_narrow_t4, "
    "train_exp1_narrow_f_labelfree, train_exp1_narrow_floor (2026-09-09), "
    "exp1_narrow exam face, equal-session mean R^2"
)


def ablation_condition_comparison(
    conditions: dict[str, dict[str, float | None]],
    baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Side-by-side carrier-increment comparison of the 2026-09-10 ablation
    conditions against the frozen exp1_narrow ladder readings.

    ``conditions`` maps a condition name -> {"t4": r2, "f_labelfree": r2 or
    None, "floor": r2 or None} (exam-face equal-session mean R^2 of that
    condition's runs).  Per condition the report records the nested-ladder
    increments of plan.COMPONENT_ABLATION["formulas"] and every reading's
    delta against the baseline condition.  When a condition carries no own
    f_labelfree rung (the carrier-axis conditions: t4_dir16, and the u1
    estimator-family comparator t4_u1_m10), the baseline f_labelfree is
    reused for the carrier increment under the disclosure that the
    ACTIVITY_ONLY cell consumes no carrier bytes and is condition-invariant
    on the carrier axis (this reuse is legal ONLY for carrier-axis
    conditions: poverty/shortwin/shallow conditions MUST pass their own
    f_labelfree rung -- reuse is refused there).  Pure function; readings
    only, never a gate."""
    baseline = dict(EXP1_NARROW_BASELINE_R2) if baseline is None else baseline
    require(set(baseline) >= {"t4", "f_labelfree", "floor"},
            "baseline ladder readings must cover t4/f_labelfree/floor")
    base_increment = baseline["t4"] - baseline["f_labelfree"]
    formulas = COMPONENT_ABLATION["formulas"]
    out_conditions: dict[str, Any] = {}
    for name, r2 in conditions.items():
        require("t4" in r2 and r2["t4"] is not None,
                f"condition {name} must carry a t4 reading")
        own_f = r2.get("f_labelfree")
        # carrier-axis conditions (the t4_dir16 family and the u1 comparator)
        # may reuse the baseline ACTIVITY_ONLY rung; geometry/data conditions
        # may not (their f_labelfree rung is condition-dependent and must be
        # measured).
        carrier_axis = "dir16" in name or "u1" in name
        if own_f is None and not carrier_axis:
            raise RuntimeError(
                f"condition {name} lacks its own f_labelfree rung; only "
                f"carrier-axis conditions may reuse the baseline rung"
            )
        f_used = own_f if own_f is not None else baseline["f_labelfree"]
        floor = r2.get("floor")
        entry: dict[str, Any] = {
            "r2": {"t4": r2["t4"], "f_labelfree": own_f, "floor": floor},
            "f_labelfree_reused_from_baseline": own_f is None,
            "carrier_label_increment": {
                "delta": r2["t4"] - f_used,
                "formula": formulas["carrier_label_increment"],
                "reading": "increment of carrier/label calibration on top of "
                           "activity under this condition (FULL over "
                           "ACTIVITY_ONLY)",
            },
        }
        if floor is not None:
            entry["activity_independent"] = {
                "delta": own_f - floor,
                "formula": formulas["activity_independent"],
            }
            entry["calibration_total"] = {
                "delta": r2["t4"] - floor,
                "formula": formulas["calibration_total"],
            }
        entry["vs_baseline"] = {
            "t4": r2["t4"] - baseline["t4"],
            "f_labelfree": (own_f - baseline["f_labelfree"]
                            if own_f is not None else None),
            "floor": (floor - baseline["floor"] if floor is not None else None),
            "carrier_label_increment": (r2["t4"] - f_used) - base_increment,
            "reading": "positive carrier_label_increment delta = the carrier's "
                       "marginal value is AMPLIFIED under this condition",
        }
        out_conditions[name] = entry
    return {
        "schema": SCHEMA + "_condition_comparison",
        "baseline": {
            "condition": "exp1_narrow",
            "r2": dict(baseline),
            "carrier_label_increment": base_increment,
            "source": EXP1_NARROW_BASELINE_SOURCE,
        },
        "conditions": out_conditions,
        "readings_only": True,
        "formal_test_policy": FORMAL_TEST_POLICY,
    }

# --- frozen data split ----------------------------------------------------
MANIFEST_RELATIVE = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
SPLIT_COUNTS = {"train": 27, "val": 6, "test": 6}

VAL_SESSIONS = (
    "sub-C_ses-CO-20151103", "sub-C_ses-CO-20151104", "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109", "sub-C_ses-CO-20151110", "sub-C_ses-CO-20151112",
)
# Rejection list ONLY: these names are never loaded by any stage of this
# package.  Encoding the names here lets loaders fail closed.
FORMAL_TEST_SESSIONS = (
    "sub-C_ses-CO-20151113", "sub-C_ses-CO-20151116", "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119", "sub-C_ses-CO-20151120", "sub-C_ses-CO-20151201",
)

# --- two-stage experiment protocols (ADDENDUM-TWO-STAGE, 2026-09-09) ------
# User ruling 2026-09-09: "先窄时间跨度后完整" (narrow span first, then full).
#   EXP-1 narrow: train = 2015-06-29..07-10 (9 sessions, 12 days inclusive,
#     mirroring M2's 10-day/7-session span); exam = 0713/0714/0715/0716 (the
#     "days-after" exam paper, 3-6 days after the last train session,
#     mirroring M2's ext4 structure).  ~1/3 of the full protocol's data.
#   EXP-2 full:   the original 27/6 protocol unchanged (train spans ~2 years,
#     val = the 6 val sessions ~4 months after the last train session).
# The two-stage difference (EXP-2 val minus EXP-1 exam, same architecture and
# recipe, only the time span differs) is the clean measurement of long-term
# drift cost.
EXP1_NARROW_TRAIN_SESSIONS = (
    "sub-C_ses-CO-20150629", "sub-C_ses-CO-20150630", "sub-C_ses-CO-20150701",
    "sub-C_ses-CO-20150703", "sub-C_ses-CO-20150706", "sub-C_ses-CO-20150707",
    "sub-C_ses-CO-20150708", "sub-C_ses-CO-20150709", "sub-C_ses-CO-20150710",
)
EXP1_NARROW_EXAM_SESSIONS = (
    "sub-C_ses-CO-20150713", "sub-C_ses-CO-20150714",
    "sub-C_ses-CO-20150715", "sub-C_ses-CO-20150716",
)
# EXTREME-DATA-POVERTY protocol (user directive 2026-09-10, ablation plan B):
# train = ONLY the earliest three sessions (2015-06-29/0630/0701), exam face
# UNCHANGED (0713-0716).  Train windows drop ~190k -> ~65k: decoder
# generalization falls, calibration reliance rises -- the condition under
# which the carrier's marginal value is expected to AMPLIFY.  Pure train-face
# subset of exp1_narrow: no new cache, the frozen cache rows are merely
# protocol-filtered (run_688_bench.load_rows); the nested ladder
# (t4 / f_labelfree / floor) is re-measured under this protocol.
EXP1_POVERTY_TRAIN_SESSIONS = tuple(EXP1_NARROW_TRAIN_SESSIONS[:3])
# 2015-ONLY REMOTE-EXAM protocol (user directive 2026-09-09): train = the 18
# sessions of the frozen manifest's 27-session train split whose dates sit in
# 2015 (the 2013 sessions are dropped), exam = the SAME 6 val sessions as
# exp2_full (2015-11, ~4 months after the last train session 2015-07-16).
# The ONLY difference from exp2_full is the removal of the 9 sessions dated
# 2013: the protocol isolates the effect of the two-year-old 2013 data on the
# remote exam.  Pure train-face subset of exp2_full -- no new cache; the
# frozen cache rows are merely protocol-filtered (run_688_bench.load_rows).
# Cross-protocol disclosure law of exp2_full applies unchanged: the
# exp1_narrow exam sessions (and train sessions) are members of this train
# face -- independent measurements, never compare faces across protocols.
EXP2015_FULL_TRAIN_SESSIONS = (
    "sub-C_ses-CO-20150309", "sub-C_ses-CO-20150311", "sub-C_ses-CO-20150312",
    "sub-C_ses-CO-20150313", "sub-C_ses-CO-20150319",
    "sub-C_ses-CO-20150629", "sub-C_ses-CO-20150630",
    "sub-C_ses-CO-20150701", "sub-C_ses-CO-20150703", "sub-C_ses-CO-20150706",
    "sub-C_ses-CO-20150707", "sub-C_ses-CO-20150708", "sub-C_ses-CO-20150709",
    "sub-C_ses-CO-20150710", "sub-C_ses-CO-20150713", "sub-C_ses-CO-20150714",
    "sub-C_ses-CO-20150715", "sub-C_ses-CO-20150716",
)
# 2016 REMOTE-EXAM protocol (user directive 2026-09-10): train = the same 18
# sessions as exp2015_full; exam = the 14 CO sessions of 2016 living in the
# DANDI directory (2016-09-09..2016-10-21), which sit OUTSIDE the frozen
# manifest entirely -- they were never eligible for it because every one of
# them carries 188-353 NWB units and the frozen roster law
# (multisession_datamodule.discover_nwb_files, max_units_exclusive=100)
# excludes any session with >= 100 units.  Admitting them therefore requires
# the disclosed exp2016 extension cache (scripts/build_exp2016_cache.py):
# rows built from NWB, T4 carrier + E0 recomputed with a column normalizer
# fit on the protocol's OWN 18 train sessions only, and a unit-selection law
# that fits the frozen Nmax-91 geometry (first 91 units of the NWB table by
# table order -- deterministic, zero discretion, diagnostics recorded).
EXP2016_EXAM_SESSIONS = (
    "sub-C_ses-CO-20160909", "sub-C_ses-CO-20160912", "sub-C_ses-CO-20160914",
    "sub-C_ses-CO-20160915", "sub-C_ses-CO-20160919", "sub-C_ses-CO-20160921",
    "sub-C_ses-CO-20160923", "sub-C_ses-CO-20160929", "sub-C_ses-CO-20161005",
    "sub-C_ses-CO-20161006", "sub-C_ses-CO-20161007", "sub-C_ses-CO-20161011",
    "sub-C_ses-CO-20161013", "sub-C_ses-CO-20161021",
)
# "ALL_TRAIN" / "ALL_VAL" are resolved against the frozen manifest at call
# time (see protocol_sessions); exp2_full is byte-identical to the pre-addendum
# behavior of this package.
PROTOCOLS: dict[str, dict[str, Any]] = {
    "exp1_narrow": {
        "train_sessions": tuple(EXP1_NARROW_TRAIN_SESSIONS),
        "exam_sessions": tuple(EXP1_NARROW_EXAM_SESSIONS),
        "span_days": 12,  # 2015-06-29..2015-07-10 inclusive
        "m2_anchor": "7 train/10d; ext4 days-after",
        "face_role": "ext4-equivalent: 4 exam sessions 3-6 days after the "
                     "12-day train window",
    },
    "exp2_full": {
        "train_sessions": "ALL_TRAIN",
        "exam_sessions": "ALL_VAL",
        "span_days": 652,  # 2013-10-03..2015-07-16 inclusive
        "m2_anchor": "27 train/2y; ext6 ~110d after last train",
        "face_role": VAL_FACE_ROLE,
    },
    # extreme data poverty (user directive 2026-09-10, plan B): train face
    # reduced to the earliest 3 sessions of exp1_narrow, exam face identical.
    "exp1_poverty": {
        "train_sessions": tuple(EXP1_POVERTY_TRAIN_SESSIONS),
        "exam_sessions": tuple(EXP1_NARROW_EXAM_SESSIONS),
        "span_days": 3,  # 2015-06-29..2015-07-01 inclusive
        "m2_anchor": "3 train/3d (extreme poverty); same ext4 days-after exam",
        "face_role": "poverty ablation: the exp1_narrow exam face (4 sessions "
                     "3-6 days after the FULL 12-day window; 12-15 days after "
                     "the poverty window's last train session)",
    },
    # 2015-only remote exam (user directive 2026-09-09): exp2_full minus the
    # 9 sessions dated 2013; the nested ladder (t4 / f_labelfree / floor) is
    # re-measured under this train face on the exp2_full val exam face.
    "exp2015_full": {
        "train_sessions": tuple(EXP2015_FULL_TRAIN_SESSIONS),
        "exam_sessions": "ALL_VAL",
        "span_days": 130,  # 2015-03-09..2015-07-16 inclusive
        "m2_anchor": "18 train/130d (2015 only); ext6 ~110d after last train",
        "face_role": "2015-only remote exam: the exp2_full val face (6 "
                     "sessions, 2015-11) scored against a train face that "
                     "drops the 2013 sessions -- the only difference from "
                     "exp2_full",
    },
    # 2016 remote exam (user directive 2026-09-10, ruling: first-91-by-table-
    # order): train = the exp2015_full 18-session face; exam = the 14 CO
    # sessions of 2016, which live OUTSIDE the frozen manifest (all violate
    # the frozen <100-unit roster law).  Requires the exp2016 extension cache
    # (scripts/build_exp2016_cache.py); the frozen 33-session cache does NOT
    # hold these rows and load_rows fails closed on it for this protocol.
    "exp2016": {
        "train_sessions": tuple(EXP2015_FULL_TRAIN_SESSIONS),
        "exam_sessions": tuple(EXP2016_EXAM_SESSIONS),
        "span_days": 130,  # train span 2015-03-09..2015-07-16 inclusive
        "m2_anchor": "18 train/130d (2015 only); ext14 ~14 months after "
                     "last train",
        "face_role": "2016 remote exam: 14 sessions (2016-09-09..2016-10-21) "
                     "outside the frozen manifest (roster law conflict "
                     "disclosed; admitted via the exp2016 extension cache's "
                     "first-91-by-table-order unit selection)",
    },
}
DEFAULT_PROTOCOL = "exp1_narrow"  # user ruling: narrow span first, then full
# SESSION-SUBSET DISCIPLINE, disclosed explicitly because it is easy to get
# wrong: the EXP-1 exam sessions live inside the EXP-2 train split (they are
# members of the manifest's 27-session train face).  Every receipt of this
# package carries this disclosure block verbatim.
CROSS_PROTOCOL_DISCLOSURE = (
    "SESSION-SUBSET DISCIPLINE: the EXP-1 narrow exam sessions "
    "(sub-C_ses-CO-20150713/0714/0715/0716) are members of the EXP-2 full "
    "27-session train split.  The two protocols are independent measurements: "
    "never compare evaluation faces across protocols and never reuse EXP-1 "
    "exam verdicts as selection evidence inside EXP-2 (and vice versa).  The "
    "same law covers exp2015_full (2015-only remote exam): its 18-session "
    "train face contains the EXP-1 narrow exam AND train sessions, so its "
    "val-face readings are again an independent measurement -- faces are "
    "never compared across protocols."
)


def session_date(session_name: str):
    """Parse the trailing YYYYMMDD of a session name into a datetime.date."""
    import re
    from datetime import datetime

    match = re.search(r"(\d{8})$", session_name)
    if match is None:
        raise RuntimeError(f"cannot parse a YYYYMMDD date out of {session_name!r}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def _validate_protocol_definitions() -> None:
    """IO-free fail-closed structural checks, run at import time."""
    from datetime import timedelta

    require(tuple(PROTOCOLS) == ("exp1_narrow", "exp2_full", "exp1_poverty",
                                 "exp2015_full", "exp2016"),
            "protocol names must be exactly (exp1_narrow, exp2_full, "
            "exp1_poverty, exp2015_full, exp2016)")
    for name, spec in PROTOCOLS.items():
        require({"train_sessions", "exam_sessions", "span_days", "m2_anchor",
                 "face_role"} <= set(spec), f"protocol {name} misses keys")
    train_dates = [session_date(n) for n in EXP1_NARROW_TRAIN_SESSIONS]
    exam_dates = [session_date(n) for n in EXP1_NARROW_EXAM_SESSIONS]
    require(len(train_dates) == 9 and len(exam_dates) == 4,
            "exp1_narrow must be 9 train + 4 exam sessions")
    require(not set(EXP1_NARROW_TRAIN_SESSIONS) & set(EXP1_NARROW_EXAM_SESSIONS),
            "exp1_narrow train/exam session sets must be disjoint")
    # exp1_poverty (user directive 2026-09-10): exactly the FIRST THREE
    # exp1_narrow train sessions (earliest dates), same exam face, every exam
    # strictly after the poverty window, no formal-test overlap.
    poverty_dates = [session_date(n) for n in EXP1_POVERTY_TRAIN_SESSIONS]
    require(EXP1_POVERTY_TRAIN_SESSIONS == tuple(EXP1_NARROW_TRAIN_SESSIONS[:3]),
            "exp1_poverty train face must be the first three exp1_narrow "
            "train sessions (earliest dates)")
    require(len(poverty_dates) == 3 and poverty_dates == sorted(poverty_dates),
            "exp1_poverty train sessions must be 3 date-ordered sessions")
    require(not set(EXP1_POVERTY_TRAIN_SESSIONS) & set(EXP1_NARROW_EXAM_SESSIONS),
            "exp1_poverty train/exam session sets must be disjoint")
    require((poverty_dates[-1] - poverty_dates[0]).days + 1
            == PROTOCOLS["exp1_poverty"]["span_days"] == 3,
            "exp1_poverty train span must be 3 days inclusive")
    for d in exam_dates:
        require(d > poverty_dates[-1],
                f"exp1_poverty exam session {d} must sit strictly after the "
                f"poverty window's last train session")
    require(set(EXP1_POVERTY_TRAIN_SESSIONS) & set(FORMAL_TEST_SESSIONS) == set(),
            "no exp1_poverty train session may be a formal-test session")
    # exp2015_full (user directive 2026-09-09): 18 date-ordered 2015 sessions,
    # 130-day inclusive span, every session strictly 2015-dated, exam = the
    # val face (strictly later, remote), no formal-test overlap anywhere.
    exp2015_dates = [session_date(n) for n in EXP2015_FULL_TRAIN_SESSIONS]
    require(len(exp2015_dates) == 18,
            "exp2015_full must be 18 train sessions (5 March + 2 June + "
            "11 July)")
    require(all(d.year == 2015 for d in exp2015_dates),
            "every exp2015_full train session must be dated 2015 (the 2013 "
            "sessions are the ones removed from exp2_full)")
    months = [d.month for d in exp2015_dates]
    require(months.count(3) == 5 and months.count(6) == 2 and months.count(7) == 11,
            "exp2015_full month histogram must be 5 (Mar) + 2 (Jun) + 11 (Jul)")
    require(exp2015_dates == sorted(exp2015_dates),
            "exp2015_full session tuple must be date-ordered")
    require((exp2015_dates[-1] - exp2015_dates[0]).days + 1
            == PROTOCOLS["exp2015_full"]["span_days"] == 130,
            "exp2015_full train span must be 130 days inclusive")
    require(PROTOCOLS["exp2015_full"]["exam_sessions"] == "ALL_VAL",
            "exp2015_full exam face must resolve to the frozen val split")
    require(not set(EXP2015_FULL_TRAIN_SESSIONS) & set(VAL_SESSIONS),
            "exp2015_full train/exam session sets must be disjoint")
    require(set(EXP2015_FULL_TRAIN_SESSIONS) & set(FORMAL_TEST_SESSIONS) == set(),
            "no exp2015_full session may be a formal-test session")
    for val_name in VAL_SESSIONS:
        require(session_date(val_name) > exp2015_dates[-1],
                f"exp2015_full exam session {val_name} must sit strictly "
                f"after the last train session {exp2015_dates[-1]}")
    # exp2016 (user directive 2026-09-10, first-91 ruling): same 18-session
    # 2015 train face, 14 date-ordered 2016 exam sessions entirely outside
    # the frozen manifest, every exam strictly after the last train session.
    exp2016_dates = [session_date(n) for n in EXP2016_EXAM_SESSIONS]
    require(len(exp2016_dates) == 14,
            "exp2016 must be 14 exam sessions (2016-09-09..2016-10-21)")
    require(all(d.year == 2016 for d in exp2016_dates),
            "every exp2016 exam session must be dated 2016")
    require(exp2016_dates == sorted(exp2016_dates),
            "exp2016 exam session tuple must be date-ordered")
    require(PROTOCOLS["exp2016"]["train_sessions"] == tuple(EXP2015_FULL_TRAIN_SESSIONS),
            "exp2016 train face must equal the exp2015_full train face")
    require(not set(EXP2016_EXAM_SESSIONS) & set(EXP2015_FULL_TRAIN_SESSIONS),
            "exp2016 train/exam session sets must be disjoint")
    require(not set(EXP2016_EXAM_SESSIONS) & set(VAL_SESSIONS)
            and not set(EXP2016_EXAM_SESSIONS) & set(FORMAL_TEST_SESSIONS)
            and not set(EXP2016_EXAM_SESSIONS) & set(EXP1_NARROW_EXAM_SESSIONS),
            "exp2016 exam sessions must be disjoint from every frozen face")
    for d in exp2016_dates:
        require(d > exp2015_dates[-1],
                f"exp2016 exam session {d} must sit strictly after the last "
                f"train session {exp2015_dates[-1]}")
    require(train_dates == sorted(train_dates) and exam_dates == sorted(exam_dates),
            "exp1_narrow session tuples must be date-ordered")
    require((train_dates[-1] - train_dates[0]).days + 1
            == PROTOCOLS["exp1_narrow"]["span_days"] == 12,
            "exp1_narrow train span must be 12 days inclusive")
    last_train = train_dates[-1]
    for d in exam_dates:
        gap = (d - last_train).days
        require(3 <= gap <= 6,
                f"exp1_narrow exam session {d} must sit 3-6 days after the "
                f"last train session (gap={gap})")
    require(set(EXP1_NARROW_EXAM_SESSIONS) & set(FORMAL_TEST_SESSIONS) == set()
            and set(EXP1_NARROW_TRAIN_SESSIONS) & set(FORMAL_TEST_SESSIONS) == set(),
            "no exp1_narrow session may be a formal-test session")
    require(DEFAULT_PROTOCOL in PROTOCOLS and DEFAULT_PROTOCOL == "exp1_narrow",
            "DEFAULT_PROTOCOL must be exp1_narrow (narrow span first)")
    require("20150713" in CROSS_PROTOCOL_DISCLOSURE
            and "train" in CROSS_PROTOCOL_DISCLOSURE,
            "CROSS_PROTOCOL_DISCLOSURE must name the exp1-exam-in-exp2-train overlap")


def protocol_sessions(protocol: str) -> dict[str, tuple[str, ...]]:
    """Resolve a protocol name into {"train": (...), "exam": (...)}.

    "ALL_TRAIN" resolves to the frozen manifest's 27-session train split,
    "ALL_VAL" to plan.VAL_SESSIONS; exp1_narrow sessions are returned as the
    frozen tuples above.  Raises ValueError on any unknown protocol name."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {tuple(PROTOCOLS)}")
    import json

    spec = PROTOCOLS[protocol]
    train, exam = spec["train_sessions"], spec["exam_sessions"]
    if train == "ALL_TRAIN" or exam == "ALL_VAL":
        splits = json.loads(manifest_path().read_text())["session_splits"]
        if train == "ALL_TRAIN":
            train = tuple(splits["train"])
        if exam == "ALL_VAL":
            exam = tuple(splits["val"])
    return {"train": tuple(train), "exam": tuple(exam)}


def verify_protocol_definitions() -> dict[str, dict[str, tuple[str, ...]]]:
    """Manifest-bound cross-checks of both protocols; returns the resolved
    session faces.  Called by every runner stage (fail closed before any
    cache row is loaded)."""
    import json

    resolved = {name: protocol_sessions(name) for name in PROTOCOLS}
    splits = json.loads(manifest_path().read_text())["session_splits"]
    train_split = tuple(splits["train"])
    require(len(train_split) == SPLIT_COUNTS["train"],
            f"manifest train split must hold {SPLIT_COUNTS['train']} sessions")
    require(set(resolved["exp1_narrow"]["train"]) <= set(train_split),
            "exp1_narrow train sessions must all sit inside the manifest train split")
    require(set(resolved["exp1_narrow"]["exam"]) <= set(train_split),
            "exp1_narrow exam sessions must all sit inside the manifest train "
            "split (the disclosed CROSS_PROTOCOL_DISCLOSURE overlap)")
    require(resolved["exp2_full"]["train"] == train_split,
            "exp2_full train face must be exactly the manifest train split")
    require(resolved["exp2_full"]["exam"] == tuple(splits["val"]) == VAL_SESSIONS,
            "exp2_full exam face must be exactly the manifest val split")
    # z_vstate_srcbank tables (user directive 2026-09-09) must re-verify
    # against the manifest-resolved faces of BOTH protocols before any
    # cache row is loaded.
    _check_z_srcbank_table("exp1_narrow", z_srcbank_map("exp1_narrow"),
                           resolved["exp1_narrow"]["train"],
                           resolved["exp1_narrow"]["exam"])
    _check_z_srcbank_table("exp2_full", z_srcbank_map("exp2_full"),
                           resolved["exp2_full"]["train"],
                           resolved["exp2_full"]["exam"])
    require(set(resolved["exp1_poverty"]["train"]) <= set(train_split),
            "exp1_poverty train sessions must all sit inside the manifest "
            "train split (poverty is a train-face subset of exp1_narrow)")
    require(resolved["exp1_poverty"]["exam"] == resolved["exp1_narrow"]["exam"],
            "exp1_poverty exam face must equal the exp1_narrow exam face "
            "(only the train face shrinks)")
    _check_z_srcbank_table("exp1_poverty", z_srcbank_map("exp1_poverty"),
                           resolved["exp1_poverty"]["train"],
                           resolved["exp1_poverty"]["exam"])
    # exp2015_full (user directive 2026-09-09): train face = exactly the
    # manifest train split minus the 2013-dated sessions (the ONLY difference
    # from exp2_full); exam face = the exp2_full val face; the disclosed
    # exp1_narrow overlap carries over (narrow train AND exam sit inside this
    # train face -- CROSS_PROTOCOL_DISCLOSURE).
    require(resolved["exp2015_full"]["train"] == tuple(EXP2015_FULL_TRAIN_SESSIONS),
            "exp2015_full train face must be exactly the frozen 18-session "
            "2015 tuple")
    require(set(resolved["exp2015_full"]["train"]) <= set(train_split),
            "exp2015_full train sessions must all sit inside the manifest "
            "train split")
    removed = set(train_split) - set(resolved["exp2015_full"]["train"])
    require(removed == {n for n in train_split if session_date(n).year == 2013}
            and len(removed) == 9,
            "exp2015_full must differ from exp2_full by exactly the 9 "
            "2013-dated train sessions")
    require(resolved["exp2015_full"]["exam"] == resolved["exp2_full"]["exam"],
            "exp2015_full exam face must equal the exp2_full val face")
    require(set(EXP1_NARROW_TRAIN_SESSIONS) | set(EXP1_NARROW_EXAM_SESSIONS)
            <= set(resolved["exp2015_full"]["train"]),
            "exp2015_full train face must contain the exp1_narrow train AND "
            "exam sessions (the disclosed cross-protocol overlap)")
    _check_z_srcbank_table("exp2015_full", z_srcbank_map("exp2015_full"),
                           resolved["exp2015_full"]["train"],
                           resolved["exp2015_full"]["exam"])
    # exp2016 (user directive 2026-09-10): train face identical to
    # exp2015_full; the 14 exam sessions sit outside the ENTIRE frozen
    # manifest (train/val/test) -- they were never roster-eligible (all
    # carry >= 188 NWB units vs the frozen <100-unit roster law) and are
    # admitted only via the disclosed exp2016 extension cache.
    require(resolved["exp2016"]["train"] == resolved["exp2015_full"]["train"],
            "exp2016 train face must equal the exp2015_full train face")
    manifest_all = set(train_split) | set(splits["val"]) | set(splits["test"])
    require(set(resolved["exp2016"]["exam"]) & manifest_all == set(),
            "exp2016 exam sessions must sit outside the whole frozen manifest")
    _check_z_srcbank_table("exp2016", z_srcbank_map("exp2016"),
                           resolved["exp2016"]["train"],
                           resolved["exp2016"]["exam"])
    return resolved


def protocol_receipt_block(protocol: str) -> dict[str, Any]:
    """Disclosure block embedded into every receipt/contract of a stage run:
    protocol name, explicit session lists, span and the cross-protocol rule."""
    sessions = protocol_sessions(protocol)
    spec = PROTOCOLS[protocol]
    return {
        "name": protocol,
        "train_sessions": list(sessions["train"]),
        "exam_sessions": list(sessions["exam"]),
        "n_train": len(sessions["train"]),
        "n_exam": len(sessions["exam"]),
        "span_days": spec["span_days"],
        "m2_anchor": spec["m2_anchor"],
        "face_role": spec["face_role"],
        "cross_protocol_disclosure": CROSS_PROTOCOL_DISCLOSURE,
    }


# --- z-series bank-source map (user directive 2026-09-09) -------------------
def nearest_train_session(exam_name: str, train_sessions: tuple[str, ...] | list[str]) -> str:
    """Deterministic rule behind Z_SRCBANK_MAPS: the train session with the
    smallest absolute date distance to ``exam_name``; ties broken toward the
    earlier date.  Pure function over session names."""
    exam_date = session_date(exam_name)
    best_name: str | None = None
    best_gap = -1
    best_date = None
    for name in train_sessions:
        date = session_date(name)
        gap = abs((date - exam_date).days)
        if (best_name is None or gap < best_gap
                or (gap == best_gap and date < best_date)):
            best_name, best_gap, best_date = name, gap, date
    if best_name is None:
        raise ValueError(f"no train sessions to map {exam_name!r} against")
    return best_name


def z_srcbank_map(protocol: str) -> dict[str, str]:
    """Frozen exam -> train bank-source table of the z_vstate_srcbank arm."""
    if protocol not in Z_SRCBANK_MAPS:
        raise ValueError(
            f"unknown protocol {protocol!r}; expected one of {tuple(PROTOCOLS)}"
        )
    return dict(Z_SRCBANK_MAPS[protocol])


def _check_z_srcbank_table(
    protocol: str, table: dict[str, str], train_sessions: tuple[str, ...],
    exam_sessions: tuple[str, ...],
) -> None:
    """Fail-closed law of a z-srcbank table: keys = the protocol's exam
    face, sources inside the protocol's train face, every exam date strictly
    after its mapped train date, and the table equals the nearest-date rule
    recomputed from scratch."""
    require(set(table) == set(exam_sessions),
            f"{protocol}: z-srcbank table keys must be exactly the exam face")
    for exam, src in table.items():
        require(src in train_sessions,
                f"{protocol}: z-srcbank source {src} for {exam} is not a "
                f"train session of the protocol")
        require(session_date(exam) > session_date(src),
                f"{protocol}: z-srcbank leakage -- exam {exam} must sit "
                f"strictly after its bank source {src}")
        recomputed = nearest_train_session(exam, train_sessions)
        require(recomputed == src,
                f"{protocol}: z-srcbank table for {exam} is {src} but the "
                f"nearest-date rule gives {recomputed}")


def _validate_z_srcbank_definitions() -> None:
    """IO-free import-time checks (exp1 tables); exp2 is manifest-bound and
    checked inside verify_protocol_definitions."""
    _check_z_srcbank_table(
        "exp1_narrow", Z_SRCBANK_MAPS["exp1_narrow"],
        tuple(EXP1_NARROW_TRAIN_SESSIONS), tuple(EXP1_NARROW_EXAM_SESSIONS),
    )
    _check_z_srcbank_table(
        "exp1_poverty", Z_SRCBANK_MAPS["exp1_poverty"],
        tuple(EXP1_POVERTY_TRAIN_SESSIONS), tuple(EXP1_NARROW_EXAM_SESSIONS),
    )
    exp2 = Z_SRCBANK_MAPS["exp2_full"]
    require(set(exp2) == set(VAL_SESSIONS),
            "exp2_full z-srcbank table keys must be exactly the val face")
    for exam, src in exp2.items():
        require(session_date(exam) > session_date(src),
                f"exp2_full: z-srcbank leakage -- {exam} must sit strictly "
                f"after its bank source {src}")
    exp2015 = Z_SRCBANK_MAPS["exp2015_full"]
    require(set(exp2015) == set(VAL_SESSIONS),
            "exp2015_full z-srcbank table keys must be exactly the val face")
    for exam, src in exp2015.items():
        require(session_date(exam) > session_date(src),
                f"exp2015_full: z-srcbank leakage -- {exam} must sit strictly "
                f"after its bank source {src}")
    exp2016 = Z_SRCBANK_MAPS["exp2016"]
    require(set(exp2016) == set(EXP2016_EXAM_SESSIONS),
            "exp2016 z-srcbank table keys must be exactly the 2016 exam face")
    for exam, src in exp2016.items():
        require(session_date(exam) > session_date(src),
                f"exp2016: z-srcbank leakage -- {exam} must sit strictly "
                f"after its bank source {src}")


# --- frozen prepared-cache binding (read-only reuse of rift_v1's bundle) --
# On-disk immutable cache (NOT the rift_v1 runner's unused default name
# dandi688_prepared_cache_v1): schema dandi688_rift_v2_prepared, 27+6
# sessions, formal_test_used=false, every array hash bound in
# prepared_contract.json.
PREPARED_CACHE_RELATIVE = (
    "btransform_unified_v2/results/rift_v1/dandi688_prepared_cache_contract_v2"
)
PREPARED_CACHE_SCHEMA = "dandi688_rift_v2_prepared"
PREPARED_CACHE_MAX_GIB = 2.0

# --- frozen model/training contract constants (mirror EXECUTION_688) ------
WINDOW_BINS = 50
CARRIER_DIM = 4
E0_DIM = 50
# concat-fusion token width (t4_concat arm): [local16 | E0_50 | carrier4]
CONCAT_TOKEN_IN = LOCAL_CONV_CHANNELS + E0_DIM + CARRIER_DIM  # 70
OUT_DIM = 2
SEED = 42
EPOCHS = 12
AVG_EPOCHS_ZERO_BASED = (8, 9, 10, 11)
BATCH = 32
MODEL_TASK = "dandi688_co"


def workspace_root() -> Path:
    """Absolute SPINT workspace root derived from this file's location."""
    root = Path(__file__).resolve().parents[4]
    if not (root / "sua_exploration").is_dir() or not (root / "btransform_unified_v2").is_dir():
        raise RuntimeError(f"workspace root sanity check failed: {root}")
    return root


def manifest_path() -> Path:
    return workspace_root() / MANIFEST_RELATIVE


def prepared_cache_path() -> Path:
    return workspace_root() / PREPARED_CACHE_RELATIVE


def require(condition: Any, message: str) -> None:
    """Fail-closed assertion used across this package (v1 plan.require style)."""
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def obj_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json_dumps(payload).encode("utf-8")
    ).hexdigest()


def json_dumps(payload: Any) -> str:
    import json
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def array_digest(array: np.ndarray) -> str:
    """SHA-256 over C-contiguous bytes; algorithm identical to
    btransform_unified_v1.bank.array_sha256 so digests are comparable with
    rift_v1 receipts (dtype str + shape + raw bytes)."""
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(data.dtype.str).encode("utf-8"))
    digest.update(str(data.shape).encode("utf-8"))
    digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def gate_report(r2_t4: float, r2_f0: float, r2_ts4: float) -> dict[str, Any]:
    """Preregistered gate arithmetic (val face).  Pure function; verdicts are
    recorded, never auto-consumed."""
    return {
        "schema": SCHEMA + "_gates",
        "r2": {"t4": r2_t4, "f0": r2_f0, "ts4": r2_ts4},
        "gate1_t4_minus_f0": {"delta": r2_t4 - r2_f0, "threshold": GATE_T4_MINUS_F0,
                              "pass": bool(r2_t4 - r2_f0 >= GATE_T4_MINUS_F0)},
        "gate2_t4_minus_ts4": {"delta": r2_t4 - r2_ts4, "threshold": GATE_T4_MINUS_TS4,
                               "pass": bool(r2_t4 - r2_ts4 >= GATE_T4_MINUS_TS4)},
        "gate3_noninferiority_vs_ref": {
            "ref_spint_t4_dev6": REF_SPINT_T4_DEV6,
            "tolerance": NONINFERIORITY_TOLERANCE,
            "floor": REF_SPINT_T4_DEV6 - NONINFERIORITY_TOLERANCE,
            "pass": bool(r2_t4 >= REF_SPINT_T4_DEV6 - NONINFERIORITY_TOLERANCE),
        },
        "formal_test_policy": FORMAL_TEST_POLICY,
    }


def vstate_gate_report(
    r2_vstate: float, r2_t4: float, r2_z_vstate: float, r2_f_labelfree: float
) -> dict[str, Any]:
    """Gate + ablation arithmetic for the vstate-688 series (user directive
    2026-09-09), exam-face equal-session mean R^2.

    gate:  vstate - t4 >= +0.03 (the vstate carrier must beat the frozen
           directional T4 under the matched M10 support).
    ablation readings (no pass/fail, both are measurements):
    z_vstate_srcbank - vstate = cost of zero new-date calibration (bank
           reused from the nearest-date train session);
    f_labelfree - vstate     = negative of the total label-information
           increment (the quantity the legacy f0 arm could not isolate,
           because f0 keeps the label-melted frozen E0)."""
    return {
        "schema": SCHEMA + "_vstate_gates",
        "r2": {"vstate": r2_vstate, "t4": r2_t4,
               "z_vstate_srcbank": r2_z_vstate, "f_labelfree": r2_f_labelfree},
        "gate_vstate_minus_t4": {
            "delta": r2_vstate - r2_t4,
            "threshold": GATE_VSTATE_MINUS_T4,
            "pass": bool(r2_vstate - r2_t4 >= GATE_VSTATE_MINUS_T4),
        },
        "ablation_calibration_cost": {
            "z_vstate_srcbank_minus_vstate": r2_z_vstate - r2_vstate,
            "reading": "cost of no new-date calibration (bank reuse via the "
                       "frozen nearest-date map)",
        },
        "ablation_label_information": {
            "f_labelfree_minus_vstate": r2_f_labelfree - r2_vstate,
            "vstate_minus_f_labelfree": r2_vstate - r2_f_labelfree,
            "reading": "total label-information increment of the vstate "
                       "carrier + label-melted E0 vs pure activity identity",
        },
        "formal_test_policy": FORMAL_TEST_POLICY,
    }


def component_ablation_report(
    r2_t4: float,
    r2_f_labelfree: float,
    r2_floor: float,
    r2_equiv_zero: float | None = None,
    r2_z0: float | None = None,
    r2_f0: float | None = None,
    r2_ts4: float | None = None,
    r2_norm_only: float | None = None,
    dc_penalty: float | None = None,
) -> dict[str, Any]:
    """Preregistered NESTED-LADDER decomposition (user clarification
    2026-09-09, nested-ladder final ruling same day), on the exp1_narrow
    exam face (the 4 cross-date sessions), equal-session mean R^2.  Pure
    function; every entry is a recorded reading, no pass/fail (the
    preregistered gates stay in gate_report/vstate_gate_report).

    ladder (information NESTING -- computing the carrier presupposes reading
    the calibration neural data, which already yields activity, so there is
    NO carrier-only rung: "算 carrier 已经拿到 activity 信息，不用白不用"):
      FULL          = t4            (activity + carrier side in E0, T on)
      ACTIVITY_ONLY = f_labelfree   (zero-side E0 remelt, T = 0)
      NORM_ONLY     = norm_only     (broadcast rate deviation, no encoder, T=0)
      NONE          = floor         (E0 = 0, T = 0)

    preregistered decomposition (COMPONENT_ABLATION["formulas"], exactly 3):
      activity_independent     = f_labelfree - floor
        the independent contribution of label-free activity calibration
        (ACTIVITY_ONLY over NONE);
      carrier_label_increment  = t4 - f_labelfree
        the increment of carrier/label calibration on top of activity
        (FULL over ACTIVITY_ONLY);
      calibration_total        = t4 - floor
        the total value of calibration information (FULL over NONE).
      (arithmetic law: activity_independent + carrier_label_increment
       == calibration_total.)

    auxiliary readings (optional, recorded when passed; never gates):
      equiv_zero pathway value  = equiv_zero - floor
        structural value of a param-matched random E0 pathway with NO
        information (if ~ floor: the pathway itself adds nothing; if ~ t4:
        the pathway matters even without encoded identity);
      equiv_zero information value = t4 - equiv_zero;
      z0 carrier-only reading  = t4 - z0 (exploratory cell demoted by the
        nesting ruling); f0 direct-carrier marginal = t4 - f0;
      ts4 content control (secondary face reading);
      norm_only four-level ladder (guide NORM_ONLY_ZERO_BASELINE_GUIDE_
        20260910: NONE < NORM_ONLY < ACTIVITY_ONLY < FULL) -- recorded when
        r2_norm_only is passed, with the guide's section-5 sanity verdicts
        (the construction-error verdict fails closed)."""
    formulas = COMPONENT_ABLATION["formulas"]
    auxiliary_formulas = COMPONENT_ABLATION["auxiliary_formulas"]
    r2: dict[str, float | None] = {
        "t4": r2_t4, "f_labelfree": r2_f_labelfree, "floor": r2_floor,
        "norm_only": r2_norm_only,
        "equiv_zero": r2_equiv_zero, "z0": r2_z0, "f0": r2_f0, "ts4": r2_ts4,
    }
    activity_independent = r2_f_labelfree - r2_floor
    carrier_label_increment = r2_t4 - r2_f_labelfree
    calibration_total = r2_t4 - r2_floor
    report: dict[str, Any] = {
        "schema": SCHEMA + "_component_ablation",
        "clarification": COMPONENT_ABLATION["clarification"],
        "base_arm": COMPONENT_ABLATION["base_arm"],
        "ladder": dict(COMPONENT_ABLATION["ladder"]),
        "ladder_order": list(COMPONENT_LADDER_ORDER),
        "norm_only_formulas": dict(COMPONENT_ABLATION["norm_only_formulas"]),
        "face": COMPONENT_ABLATION["judged_on"],
        "secondary_face_policy": COMPONENT_ABLATION["secondary"],
        "r2": r2,
        "decomposition": {
            "activity_independent": {
                "delta": activity_independent,
                "formula": formulas["activity_independent"],
                "reading": "independent contribution of label-free activity "
                           "calibration (ACTIVITY_ONLY over NONE): "
                           "f_labelfree keeps encoder weights + activity "
                           "content, floor removes E0 entirely",
            },
            "carrier_label_increment": {
                "delta": carrier_label_increment,
                "formula": formulas["carrier_label_increment"],
                "reading": "increment of carrier/label calibration on top of "
                           "activity (FULL over ACTIVITY_ONLY): the direct "
                           "carrier token + the label side inside E0",
            },
            "calibration_total": {
                "delta": calibration_total,
                "formula": formulas["calibration_total"],
                "reading": "total value of calibration information "
                           "(FULL over NONE)",
            },
        },
        "decomposition_law": "activity_independent + carrier_label_increment "
                             "== calibration_total (nested ladder)",
        "formal_test_policy": FORMAL_TEST_POLICY,
    }
    controls: dict[str, Any] = {}
    if r2_equiv_zero is not None:
        controls["equiv_zero"] = {
            "pathway_value": {
                "delta": r2_equiv_zero - r2_floor,
                "formula": auxiliary_formulas["equiv_zero_pathway_value"],
                "reading": "structural value of the param-matched random "
                           "E0 pathway carrying no identity/labels",
            },
            "information_value": {
                "delta": r2_t4 - r2_equiv_zero,
                "formula": auxiliary_formulas["equiv_zero_information_value"],
                "reading": "what the random projection cannot recover: "
                           "the trained-identity gap under a matched "
                           "pathway/capacity",
            },
            "param_parity_law": "trainable params of the equiv_zero "
                                "model == t4 (same model class); the "
                                "random projection's params == the "
                                "replaced post_pool pathway",
        }
    if r2_z0 is not None:
        controls["z0_carrier_only"] = {
            "delta": r2_t4 - r2_z0,
            "formula": auxiliary_formulas["z0_carrier_only_reading"],
            "reading": "AUXILIARY exploratory cell (dropped from the ladder "
                       "by the nesting ruling: computing the carrier "
                       "presupposes the calibration neural data)",
        }
    if r2_f0 is not None:
        controls["f0_direct_carrier_marginal"] = {
            "delta": r2_t4 - r2_f0,
            "formula": auxiliary_formulas["direct_carrier_marginal"],
            "reading": "AUXILIARY: marginal of the direct carrier token "
                       "under the full (label-melted) E0",
        }
    if r2_ts4 is not None:
        controls["ts4_content"] = {
            "delta": r2_t4 - r2_ts4,
            "reading": "content control (secondary): correspondence "
                       "destroyed, per-column marginals identical",
        }
    if r2_norm_only is not None:
        # four-level ladder reading (guide 2026-09-10 sections 2/5/6); the
        # construction-error verdict fails closed inside the helper.
        report["zero_calibration_baseline"] = norm_only_ladder_report(
            r2_floor, r2_norm_only, r2_f_labelfree, r2_t4, dc_penalty=dc_penalty,
        )
    if controls:
        report["controls"] = controls
    return report


def _validate_component_ablation() -> None:
    """IO-free fail-closed structural checks of the nested-ladder ablation."""
    ladder = COMPONENT_ABLATION["ladder"]
    rungs = COMPONENT_ABLATION["rungs"]
    require(ladder == {
        "NONE": "floor", "NORM_ONLY": "norm_only",
        "ACTIVITY_ONLY": "f_labelfree", "FULL": "t4",
    }, "the component-ablation ladder must be exactly NONE=floor, "
       "NORM_ONLY=norm_only, ACTIVITY_ONLY=f_labelfree, FULL=t4")
    # the ladder dict's insertion order IS the ascending rung order: it must
    # equal the frozen order tuple (guide NORM_ONLY_ZERO_BASELINE_GUIDE_
    # 20260910 section 2) -- a silently reordered ladder fails import.
    require(tuple(ladder) == COMPONENT_LADDER_RUNGS,
            f"the ladder rung order must be exactly {COMPONENT_LADDER_RUNGS}")
    require(tuple(ladder.values()) == COMPONENT_LADDER_ORDER,
            f"the ladder arms must ascend exactly as {COMPONENT_LADDER_ORDER}")
    require(tuple(rungs) == COMPONENT_LADDER_ORDER,
            f"component-ablation rungs must be exactly {COMPONENT_LADDER_ORDER}"
            " (ascending)")
    require(set(rungs) == set(ladder.values()),
            "every ladder rung must describe exactly its arm")
    require(set(COMPONENT_ABLATION["auxiliary_arms"]) == {"z0", "z_vstate_srcbank"},
            "z0 and z_vstate_srcbank must be demoted to auxiliary arms")
    require(COMPONENT_ABLATION["base_arm"] == "t4",
            "the component ablation is based on t4 (discrete labels)")
    every_component_arm = (
        set(rungs)
        | set(COMPONENT_ABLATION["control"])
        | set(COMPONENT_ABLATION["auxiliary_arms"])
        | {"f0"}
    )
    for arm in every_component_arm:
        require(arm in ARMS, f"component-ablation arm {arm} missing from ARMS")
    require(tuple(COMPONENT_ABLATION["formulas"]) == (
        "activity_independent", "carrier_label_increment", "calibration_total"),
        "the preregistered decomposition must be exactly the 3-item ladder")
    require("算 carrier" in COMPONENT_ABLATION["clarification"],
            "the nested-ladder ruling quote must stay attached")
    require("20150713" in COMPONENT_ABLATION["judged_on"],
            "the component ablation is judged on the exp1_narrow exam face")
    require("train_exp1_narrow_z0" in COMPONENT_ABLATION["note_z0_redefinition"]
            and "floor" in COMPONENT_ABLATION["note_z0_redefinition"],
            "the z0 redefinition note must stay attached")
    # NORM_ONLY rung wiring (guide 2026-09-10 sections 2/3/5): the arm exists,
    # is bound to its own variant cache, and carries the guide's sanity laws.
    require(COMPONENT_LADDER_ORDER[1] == NORM_ONLY_VARIANT
            and ladder["NORM_ONLY"] == NORM_ONLY_VARIANT,
            "NORM_ONLY must sit directly above NONE in the ladder")
    require(ARM_REQUIRED_CACHE_VARIANT.get(NORM_ONLY_VARIANT)
            == frozenset({NORM_ONLY_VARIANT}),
            "the norm_only arm must be bound to the norm_only variant cache")
    require(NORM_ONLY_TASK_CRITERIA["M1"]["min"] == 0.3
            and NORM_ONLY_TASK_CRITERIA["M1"]["dc_penalty_max"] == 0.1
            and NORM_ONLY_TASK_CRITERIA["M2"]["min"] == 0.15
            and NORM_ONLY_TASK_CRITERIA["H1"]["min"] == 0.25,
            "the guide section 5 pre-registered criteria must stay attached")
    require("STOP" in NORM_ONLY_SANITY_CHECKS["construction_error"],
            "the construction-error sanity law must fail loudly")
    require(tuple(COMPONENT_ABLATION["norm_only_formulas"]) == (
        "rate_deviation_identity", "encoder_activity_increment",
        "label_free_activity_total"),
        "the NORM_ONLY rung formulas must be exactly the 3-item refinement")
    for arm in COMPONENT_ARMS_ON_FROZEN_CACHE:
        require(arm not in ARM_REQUIRED_CACHE_VARIANT,
                f"component arm {arm} must run on the frozen cache")
    require(set(ARM_REQUIRED_CACHE_VARIANT) <= set(ARMS),
            "variant-bound arms must exist in ARMS")
    bound_variants = {v for allowed in ARM_REQUIRED_CACHE_VARIANT.values()
                      for v in allowed}
    require(set(PROTOCOL_BOUND_CACHE_VARIANTS) <= bound_variants,
            "protocol-bound cache variants must be variant-bound families")


# IO-free fail-closed structural checks of the two-stage protocol constants,
# executed at import time (after require() and the date helper are bound).
_validate_protocol_definitions()
_validate_z_srcbank_definitions()
_validate_component_ablation()


__all__ = [
    "SCHEMA", "ARMS", "FUSION_MODES", "LOCAL_CONV_CHANNELS", "CONCAT_TOKEN_IN",
    "GATE_T4_MINUS_F0", "GATE_T4_MINUS_TS4", "GATE_VSTATE_MINUS_T4",
    "REF_SPINT_T4_DEV6", "NONINFERIORITY_TOLERANCE",
    "VAL_FACE_ROLE", "FORMAL_TEST_POLICY",
    "GPU_POLICY", "GPU_POLICY_NOTE", "CPU_THREADS_DEFAULT",
    "MANIFEST_RELATIVE", "MANIFEST_SHA256", "SPLIT_COUNTS",
    "VAL_SESSIONS", "FORMAL_TEST_SESSIONS",
    "EXP1_NARROW_TRAIN_SESSIONS", "EXP1_NARROW_EXAM_SESSIONS",
    "EXP1_POVERTY_TRAIN_SESSIONS", "EXP2015_FULL_TRAIN_SESSIONS",
    "EXP2016_EXAM_SESSIONS",
    "PROTOCOLS", "DEFAULT_PROTOCOL", "CROSS_PROTOCOL_DISCLOSURE",
    "MODEL_OVERRIDES", "SHORTWIN_WINDOW_BINS", "SHALLOW_TEMPORAL_LAYERS",
    "EXP1_NARROW_BASELINE_R2", "EXP1_NARROW_BASELINE_SOURCE",
    "ablation_condition_comparison",
    "VSTATE_BLOCK_SECONDS", "VSTATE_N0_BLOCKS", "VSTATE_R700_SECONDS",
    "VSTATE_H300_SECONDS", "VSTATE_STATE_COLUMNS", "VSTATE_VELOCITY_SUBBINS",
    "VSTATE_VELOCITY_BIN_SECONDS", "VSTATE_SUPPORT_NAMESPACE",
    "VSTATE_SUPPORT_POSITIONS", "VSTATE_FULL_SUPPORT_NAMESPACE",
    "VSTATE_FULL_SUPPORT_POSITIONS", "VSTATE_B_MODES", "VSTATE_B_MODE_MAIN",
    "VSTATE_HOLD", "VSTATE_VARIANTS",
    "Z_SRCBANK_RULE", "Z_SRCBANK_MAPS", "ARM_REQUIRED_CACHE_VARIANT",
    "PROTOCOL_BOUND_CACHE_VARIANTS",
    "COMPONENT_ABLATION", "COMPONENT_ARMS_ON_FROZEN_CACHE",
    "COMPONENT_LADDER_ORDER", "COMPONENT_LADDER_RUNGS",
    "NORM_ONLY_VARIANT", "NORM_ONLY_CACHE_DIRNAME", "NORM_ONLY_SIGMA_FLOOR",
    "NORM_ONLY_RATE_BIN_SECONDS", "NORM_ONLY_RATE_UNITS",
    "NORM_ONLY_SUPPORT_TRIALS", "NORM_ONLY_SUPPORT_NOTE",
    "NORM_ONLY_UNDEFINED_SLOT_Z", "NORM_ONLY_SANITY_CHECKS",
    "NORM_ONLY_MIN_SOURCE_SESSIONS", "NORM_ONLY_MAX_ABS_Z",
    "NORM_ONLY_TASK_CRITERIA", "norm_only_ladder_report",
    "component_ablation_report",
    "session_date", "protocol_sessions", "verify_protocol_definitions",
    "protocol_receipt_block", "nearest_train_session", "z_srcbank_map",
    "PREPARED_CACHE_RELATIVE", "PREPARED_CACHE_SCHEMA", "PREPARED_CACHE_MAX_GIB",
    "WINDOW_BINS", "CARRIER_DIM", "E0_DIM", "OUT_DIM",
    "SEED", "EPOCHS", "AVG_EPOCHS_ZERO_BASED", "BATCH", "MODEL_TASK",
    "workspace_root", "manifest_path", "prepared_cache_path",
    "require", "file_sha256", "obj_sha256", "json_dumps", "array_digest",
    "gate_report", "vstate_gate_report",
]
