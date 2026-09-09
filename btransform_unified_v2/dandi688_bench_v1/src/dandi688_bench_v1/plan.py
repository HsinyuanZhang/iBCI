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
ARMS = ("t4", "f0", "ts4", "t4_concat", "z0",
        "vstate", "z_vstate_srcbank", "f_labelfree")
FUSION_MODES = ("proj_add", "concat")
LOCAL_CONV_CHANNELS = 16  # frozen v1 CONV_CHANNELS width of the local k5 block

# --- vstate-688 recipe constants (M2 vstate4 recipe, PLAN_CARRIER_ITERATION_ --
# M2_688_20260909.md section 3.2, adapted to the 688 rate primitives) --------
# 100 ms non-overlapping blocks inside the frozen R700/H300 windows; rates =
# spike-time searchsorted half-open counts / block duration (the 688 rate
# primitive family); state W = [softplus(v_x/rms_x), softplus(-v_x/rms_x),
# softplus(v_y/rms_y), softplus(-v_y/rms_y)] with v = cursor_vel block mean;
# rms calibrated on TRAIN sessions only (exp1_narrow = the 9 train sessions,
# exp2_full = the 27); Poisson standardization with delta = block duration +
# n0 = 10-block shrinkage; readout a = R(+x) - R(-x), c = R(+y) - R(-y),
# m = hypot(a, c), b = mean_k R - R_hold (hold = the conditional response of
# the H300 blocks, preserving the delta_b semantics); column normalization
# train-only mean/std.  Support budget is the SAME M10 trial set as the t4
# arm (candidate namespace, chronological first 10 rewarded trials) -- the
# support set does not change, only the labels swap direction -> velocity.
VSTATE_BLOCK_SECONDS = 0.1          # 100 ms blocks
VSTATE_N0_BLOCKS = 10.0             # shrinkage: 10 pseudo-blocks per state
VSTATE_R700_SECONDS = 0.7           # frozen R700 motion window -> 7 blocks
VSTATE_H300_SECONDS = 0.3           # frozen H300 hold window   -> 3 blocks
VSTATE_STATE_COLUMNS = ("+x", "-x", "+y", "-y")
VSTATE_VELOCITY_SUBBINS = 5         # 5 x 20 ms bin centers inside each block
VSTATE_VELOCITY_BIN_SECONDS = 0.02  # 688 BIN_SIZE_MS = 20 (frozen contract)
VSTATE_SUPPORT_NAMESPACE = "candidate"   # same M10 support as the t4 arm
VSTATE_SUPPORT_POSITIONS = tuple(range(10))
VSTATE_HOLD = "signed_state_h300_same_affine_n0_10"  # R_hold uses the H300
# blocks' own signed-state weights and the R700-fitted Poisson affine.

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
}

# --- prepared-cache variant binding for the vstate-688 arms -----------------
# The three new arms are only legal on their matching variant cache (the
# frozen contract-v2 cache carries no "variant" field and backs the five
# original arms unchanged).  The vstate cache is additionally protocol-bound
# (its rms/column normalizer was fit on that protocol's train sessions).
ARM_REQUIRED_CACHE_VARIANT: dict[str, str] = {
    "vstate": "vstate",
    "z_vstate_srcbank": "vstate",
    "f_labelfree": "f_labelfree",
}

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
    "exam verdicts as selection evidence inside EXP-2 (and vice versa)."
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

    require(tuple(PROTOCOLS) == ("exp1_narrow", "exp2_full"),
            "protocol names must be exactly (exp1_narrow, exp2_full)")
    for name, spec in PROTOCOLS.items():
        require({"train_sessions", "exam_sessions", "span_days", "m2_anchor",
                 "face_role"} <= set(spec), f"protocol {name} misses keys")
    train_dates = [session_date(n) for n in EXP1_NARROW_TRAIN_SESSIONS]
    exam_dates = [session_date(n) for n in EXP1_NARROW_EXAM_SESSIONS]
    require(len(train_dates) == 9 and len(exam_dates) == 4,
            "exp1_narrow must be 9 train + 4 exam sessions")
    require(not set(EXP1_NARROW_TRAIN_SESSIONS) & set(EXP1_NARROW_EXAM_SESSIONS),
            "exp1_narrow train/exam session sets must be disjoint")
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
    """IO-free import-time checks (exp1 table); exp2 is manifest-bound and
    checked inside verify_protocol_definitions."""
    _check_z_srcbank_table(
        "exp1_narrow", Z_SRCBANK_MAPS["exp1_narrow"],
        tuple(EXP1_NARROW_TRAIN_SESSIONS), tuple(EXP1_NARROW_EXAM_SESSIONS),
    )
    exp2 = Z_SRCBANK_MAPS["exp2_full"]
    require(set(exp2) == set(VAL_SESSIONS),
            "exp2_full z-srcbank table keys must be exactly the val face")
    for exam, src in exp2.items():
        require(session_date(exam) > session_date(src),
                f"exp2_full: z-srcbank leakage -- {exam} must sit strictly "
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


# IO-free fail-closed structural checks of the two-stage protocol constants,
# executed at import time (after require() and the date helper are bound).
_validate_protocol_definitions()
_validate_z_srcbank_definitions()


__all__ = [
    "SCHEMA", "ARMS", "FUSION_MODES", "LOCAL_CONV_CHANNELS", "CONCAT_TOKEN_IN",
    "GATE_T4_MINUS_F0", "GATE_T4_MINUS_TS4", "GATE_VSTATE_MINUS_T4",
    "REF_SPINT_T4_DEV6", "NONINFERIORITY_TOLERANCE",
    "VAL_FACE_ROLE", "FORMAL_TEST_POLICY",
    "GPU_POLICY", "GPU_POLICY_NOTE", "CPU_THREADS_DEFAULT",
    "MANIFEST_RELATIVE", "MANIFEST_SHA256", "SPLIT_COUNTS",
    "VAL_SESSIONS", "FORMAL_TEST_SESSIONS",
    "EXP1_NARROW_TRAIN_SESSIONS", "EXP1_NARROW_EXAM_SESSIONS",
    "PROTOCOLS", "DEFAULT_PROTOCOL", "CROSS_PROTOCOL_DISCLOSURE",
    "VSTATE_BLOCK_SECONDS", "VSTATE_N0_BLOCKS", "VSTATE_R700_SECONDS",
    "VSTATE_H300_SECONDS", "VSTATE_STATE_COLUMNS", "VSTATE_VELOCITY_SUBBINS",
    "VSTATE_VELOCITY_BIN_SECONDS", "VSTATE_SUPPORT_NAMESPACE",
    "VSTATE_SUPPORT_POSITIONS", "VSTATE_HOLD",
    "Z_SRCBANK_RULE", "Z_SRCBANK_MAPS", "ARM_REQUIRED_CACHE_VARIANT",
    "session_date", "protocol_sessions", "verify_protocol_definitions",
    "protocol_receipt_block", "nearest_train_session", "z_srcbank_map",
    "PREPARED_CACHE_RELATIVE", "PREPARED_CACHE_SCHEMA", "PREPARED_CACHE_MAX_GIB",
    "WINDOW_BINS", "CARRIER_DIM", "E0_DIM", "OUT_DIM",
    "SEED", "EPOCHS", "AVG_EPOCHS_ZERO_BASED", "BATCH", "MODEL_TASK",
    "workspace_root", "manifest_path", "prepared_cache_path",
    "require", "file_sha256", "obj_sha256", "json_dumps", "array_digest",
    "gate_report", "vstate_gate_report",
]
