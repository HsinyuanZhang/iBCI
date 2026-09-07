#!/usr/bin/env python3
"""Matched scorer for the sub-population invariance cells T / C / G / W.

Implements HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3
(pre-registered readings for cells T and G, the conjunctive C gate, and the
Step 0C unit-loss robustness curve as C's flatter-curve baseline) and §5
(receipt requirements) for the training cells of the decomposition round.
Cell W (§3 "Cell W - output head and within-window structure", built to
HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md §5 Priority 2) is the EXPLORATORY
cell: it has NO pre-registered gate, is scored NATIVE ONLY (no unit-loss
curve, no ensemble — the Step 0C wrapper replicates the PARENT decode path
and would silently drop the W residual), and reports BOTH granularities
because it is the cell intended to change the within-window profile.
Read-only: zero GPU training, zero target updates.

Per LANDED cell (each independently gated on its own terminal receipt +
SWA SHA; a cell whose terminal receipt does not exist yet is omitted with a
``not_landed`` note, never scored from a partial artifact):

1. Native governing score on BOTH surfaces (within 6 sub-C dev, external 15
   sub-M), dual granularity (GOVERNING last-bin/equal-session labelled; full
   window + window-weighted as DIAGNOSTICS), ``n_windows`` per session in
   every per-session entry, the 2014/2015 date blocks and
   ``sub-M_ses-CO-20141203`` reported separately.
2. Unit-loss robustness curve on the external surface using the Step 0C (i)
   protocol VERBATIM — fractions {0, 0.1, 0.25, 0.5, 0.75} x 3 mask seeds x
   ``zero_nogain`` PRIMARY + ``zero_gain`` + ``padding`` x 1 seed, with the
   0C paired checkpoint-independent seed rule — so cell masks and D masks are
   bit-identical.  The D curve is scored LIVE in the same pass (never
   hand-copied) and cross-checked against the sealed Step 0C receipt when
   present; per fraction the scorer emits cell-minus-D paired per-session
   deltas with sign counts (C's pre-registered claim is a FLATTER curve than
   D's).
3. Paired contrasts vs Arm A and vs D — same-engine live scoring of the armA
   and D SWAs in this pass, ``matched_scorer.paired_session_stats`` (mean,
   n_positive, bootstrap CI DESCRIPTIVE ONLY) — plus C-minus-T when T lands.
4. Pre-registered readings as machine-checkable fields with the rule strings
   recorded in the receipt: the T trichotomy, the G three-band reading and
   the conjunctive C gate (no bootstrap escape clause).
5. Every reference value is LOADED from SHA-verified sealed receipts (A2
   pooled + per-seed bars from a2_matched_rescore_v1_r1; the Arm A / D
   governing bars from sparsification_score_v1) — no hardcoded numbers.
6. The A2 development screen (both absolute bars) is reported DESCRIPTIVE
   ONLY as a seed-42 screen; the receipt states that seeds 43/44 remain
   mandatory for any A2 superiority claim.

The curve/ensemble machinery, the checkpoint-integrity binding for arm A / D
and the seed rule are IMPORTED from ``scripts/run_subpop_step0c.py`` (never
duplicated).  Authorization, loader, integrity and receipt conventions are
copied from ``scripts/run_sparsify_score.py`` / ``run_pop_robust_score.py``.

Cells land at different times, and the receipt root is immutable (fresh
directory per invocation): score each cell as it lands into
``results/subpop_score_v1``, then ``_r1`` / ``_r2`` for the later cells, or
wait and score all landed cells in one pass.

Extended for HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md (the AM/IM
decomposition round).  Cells AM (activity mask) and IM (identity mask) live
under their own root ``results/aimask_v1/``, draw from the SAME one-p-per-
forward stream as R/S2/T/G, apply a whole-unit ``F.dropout`` mask (gain
``1/(1-p)`` inherited from the call, no ``min_keep`` clamp) to ONE component
of ``src = activity + identity`` during TRAINING only — so their eval graph
is the literal parent sum and the Step 0C curve wrapper is VALID for them
(unlike W).  They are read ONLY through the pre-registered §3 decomposition
matrix, emitted as the receipt's top-level ``decomposition_reading`` block
with the rule string verbatim; no single-cell reading is ever taken.
Handoff §5.3 is also implemented: every receipt carries a top-level
``supersedes`` list (+ ``supersedes_note`` / ``supersedes_source``) naming
the prior receipt(s) this pass extends or re-scores, from ``--supersedes``
or auto-derived from the output-root family.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
# Three-layer `src` namespace (tfpd_exploration / streaming_calibration_exp /
# sua_exploration) resolved exactly once — the run_subpop_step0c.py pattern.
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))
import src as _src_pkg  # noqa: E402  (tfpd_exploration/src wins)

_STREAMING_SRC = str(REPO / "streaming_calibration_exp" / "src")
if _STREAMING_SRC not in _src_pkg.__path__:
    _src_pkg.__path__.append(_STREAMING_SRC)

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
BOUND_PATTERNS = (
    "scripts/run_subpop_score.py",
    "tests/test_subpop_score.py",
    "tests/test_subpop_score_w.py",
    "tests/test_subpop_score_aimask.py",
    "scripts/run_subpop_step0c.py",
    "src/tfpd_lane/subpop_cells.py",
    "src/tfpd_lane/consistency_cell.py",
    "src/tfpd_lane/temporal_residual_cell.py",
    "src/tfpd_lane/activity_identity_cells.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd_lane/arm_common.py",
    "scripts/run_a2_matched_rescore.py",
)
SEPARATE_SESSION = "sub-M_ses-CO-20141203"
CANONICAL_INITIAL_STATE = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
THETA_AUTHORITY_FILE = ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt"
STEP0C_RECEIPT = ROOT / "results/subpop_step0c_v1/step0c_receipt.json"

# ---- sealed reference receipts (SHA-verified before any value is read) ------
A2_R1_RECEIPT = ROOT / "results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json"
A2_R1_RECEIPT_SHA = "0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e"
SPARSIFY_SCORE_RECEIPT = (
    ROOT / "results/sparsification_score_v1/sparsification_score_receipt.json"
)
SPARSIFY_SCORE_RECEIPT_SHA = (
    "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"
)

# ---- the three cells (handoff §3) -------------------------------------------
CELL_SPECS = {
    "T": {
        "directory": "cellT_true_removal",
        "graph_builder": "src/tfpd_lane/subpop_cells.build_subpop_model(cell='T')",
        "builder": "subpop",
        "p_draws_per_step": 1,
        "law_fields": (
            "mask_structure", "min_keep", "rescaling_policy",
            "key_padding_mask_semantics", "p_distribution_and_clamp_policy",
            "generator_namespaces", "eval_mask_policy",
        ),
        "expected": {"min_keep": 4},
    },
    "C": {
        "directory": "cellC_paired_consistency",
        "graph_builder": "src/tfpd_lane/consistency_cell.build_consistency_model()",
        "builder": "consistency",
        "p_draws_per_step": 2,
        "law_fields": (
            "lambda", "lambda_frozen", "lambda_sweep", "consistency_site",
            "skip_floor", "skip_rule", "compute_option", "preregistration",
            "dropout_structure", "eval_mask",
        ),
        "expected": {"lambda": 0.1, "lambda_frozen": True, "num_heads": 2},
    },
    "G": {
        "directory": "cellG_global_gain",
        "graph_builder": "src/tfpd_lane/subpop_cells.build_subpop_model(cell='G')",
        "builder": "subpop",
        "p_draws_per_step": 1,
        "law_fields": (
            "mask_structure", "min_keep", "rescaling_policy",
            "p_distribution_and_clamp_policy", "generator_namespaces",
            "eval_mask_policy",
        ),
        "expected": {"min_keep": "n/a (no masking)"},
    },
    # the exploratory within-window-profile cell: an OUTPUT-HEAD change, not a
    # perturbation — no masking, no p stream, NO pre-registered gate
    "W": {
        "directory": "cellW_temporal_residual",
        "graph_builder": (
            "src/tfpd_lane/temporal_residual_cell.build_temporal_residual_model"
            "(seed=42)"
        ),
        "builder": "temporal_residual",
        "p_draws_per_step": 0,
        "graph_note": (
            "the exact Arm A graph (2 heads) PLUS the route-owned zero-init "
            "temporal latent residual head; the residual is ACTIVE at eval (it "
            "is the one W change; W has no train-mode perturbation at all)"
        ),
        "law_fields": (
            "head_structure", "initialization", "zero_init_guarantee",
            "non_causal_caveat", "gradient_flow", "rng_note", "loss",
            "dropout_structure", "w_head_parameters",
            "behavior_scaling_convention",
        ),
        "expected": {
            "behavior_scaling_convention": (
                "unscaled standardized behavior (exact Arm A replica)"
            ),
        },
    },
}
CELL_ROOT = "results/subpop_v1"
FULL_BUDGET_EPOCHS = 48

# ---- cells AM / IM (HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md §3) -------
# Both cells apply ONE shared whole-unit F.dropout mask draw per training
# forward to a SINGLE component of `src = activity + identity`; the gain
# 1/(1-p) is inherited from the F.dropout call itself (never reimplemented)
# and there is no min_keep clamp.  Their output root is their own
# (results/aimask_v1), and their eval path is the LITERAL parent sum, so the
# loaded graph is the exact Arm A graph — the same state keys, the same
# strict load, and a Step 0C curve wrapper that is VALID (see
# AIMASK_CURVE_VALIDITY_NOTE below).
AIMASK_CELL_ROOT = "results/aimask_v1"
AIMASK_CELLS = ("AM", "IM")
AIMASK_MASK_STRUCTURE = {
    "AM": "whole_unit_F_dropout_on_activity",
    "IM": "whole_unit_F_dropout_on_identity",
}
CELL_SPECS["AM"] = {
    "directory": "cellAM_activity_mask",
    "cell_root": AIMASK_CELL_ROOT,
    "graph_builder": (
        "src/tfpd_lane/activity_identity_cells.build_activity_identity_model"
        "(seed=42, cell='AM')"
    ),
    "builder": "activity_identity",
    "p_draws_per_step": 1,
    "graph_note": (
        "the exact Arm A graph (2 heads); the AM mask multiplies the ACTIVITY "
        "component only and is TRAIN-MODE-ONLY — the eval path is the literal "
        "parent sum `activity + identity`, which is why the Step 0C curve "
        "wrapper is VALID for this cell (unlike cell W)"
    ),
    "law_fields": (
        "mask_structure", "min_keep", "gain", "gain_rule", "application_site",
        "dropped_unit_becomes", "mask_code_path_matches_D",
        "p_distribution_and_clamp_policy", "p_law",
        "key_padding_mask_semantics", "generator_namespaces",
        "eval_mask_policy",
    ),
    "expected": {
        "mask_structure": AIMASK_MASK_STRUCTURE["AM"],
        "mask_code_path_matches_D": True,
    },
}
CELL_SPECS["IM"] = {
    "directory": "cellIM_identity_mask",
    "cell_root": AIMASK_CELL_ROOT,
    "graph_builder": (
        "src/tfpd_lane/activity_identity_cells.build_activity_identity_model"
        "(seed=42, cell='IM')"
    ),
    "builder": "activity_identity",
    "p_draws_per_step": 1,
    "graph_note": (
        "the exact Arm A graph (2 heads); the IM mask multiplies the IDENTITY "
        "component only and is TRAIN-MODE-ONLY — the eval path is the literal "
        "parent sum `activity + identity`, which is why the Step 0C curve "
        "wrapper is VALID for this cell (unlike cell W)"
    ),
    "law_fields": (
        "mask_structure", "min_keep", "gain", "gain_rule", "application_site",
        "dropped_unit_becomes", "mask_code_path_matches_D",
        "p_distribution_and_clamp_policy", "p_law",
        "key_padding_mask_semantics", "generator_namespaces",
        "eval_mask_policy",
    ),
    "expected": {
        "mask_structure": AIMASK_MASK_STRUCTURE["IM"],
        "mask_code_path_matches_D": True,
    },
}
# The agreed cell-law strings the AM/IM runner records in its terminal receipt:
# the gain comes from the F.dropout call itself (handoff §2), and min_keep is
# "none" for these cells (unlike cell T's clamp of 4).
AIMASK_EXPECTED_GAIN = "1/(1-p) inherited from F.dropout"
AIMASK_EXPECTED_MIN_KEEP = "none"

# ---- cell W: the exploratory reading (NO pre-registered gate) ---------------
W_READING_LABEL = "EXPLORATORY__NO_PREREGISTERED_GATE"
W_READING_NOTE = (
    "cell W (temporal latent residual output head) is the EXPLORATORY "
    "within-window-profile cell of handoff 2026-08-18 #3; NO gate was "
    "pre-registered for it, so this dict is a READING, never a verdict.  Both "
    "granularities are reported because W is the cell intended to change the "
    "within-window profile; last-bin equal-session remains GOVERNING and the "
    "full-window diagnostic can never rescue it.  Adoption requires its own "
    "follow-up: seeds 43/44 plus one Z4 sibling to establish whether any gain "
    "is generic or a better consumption of T4 "
    "(HANDOFF_SPINT_DECODER_DIRECTIONS_20260817 #5 Priority 2); no adoption "
    "decision may be taken from this receipt"
)

# ---- cells scored NATIVE ONLY (structurally excluded from the 0C curve) -----
# The Step 0C curve wrapper (run_subpop_step0c.SubpopEvalModel) replicates the
# PARENT decode path; for cell W it would silently DROP the temporal residual
# (delta), i.e. score the WRONG graph while reporting it as W.  A W curve must
# raise, never run — see refuse_cell_curve / curve_cells_for.
CURVE_EXCLUDED_CELLS = {
    "W": (
        "the Step 0C wrapper (run_subpop_step0c.SubpopEvalModel) replicates the "
        "PARENT decode path and therefore silently DROPS the W temporal "
        "residual (delta); a W curve would score the wrong graph, so cell W is "
        "scored NATIVE ONLY (no unit-loss curve, no subset ensemble)"
    ),
}

# ---- why AM/IM are the OPPOSITE case: the wrapper is VALID for them --------
# The AM/IM mask is a TRAINING-time intervention at the component site; at
# eval the perturbation is disabled and the forward is the literal parent sum
# `activity + identity` fed to the parent decoder (handoff 2026-08-19 §3
# "Evaluation path must be bitwise equal to the parent path").  The loaded
# graph therefore IS the parent graph — same state keys, strict load, and a
# SubpopEvalModel that reproduces it exactly — so AM/IM take the full 0C
# unit-loss curve: their test-time unit-loss robustness is well-posed.
AIMASK_CURVE_VALIDITY_NOTE = (
    "cells AM/IM are CURVED (unlike W): their perturbation is train-mode-only "
    "at the activity/identity component site, so the eval graph is the exact "
    "parent graph the Step 0C wrapper replicates; the curve measures their "
    "test-time unit-loss robustness with the same paired seed rule as D"
)

# ---- pre-registered decision rules (rule strings go in the receipt) --------
T_TRICHOTOMY_RULE = (
    "T-minus-D EXTERNAL paired governing mean delta d: |d| < 0.01 -> "
    "T_APPROX_EQUAL; d >= +0.01 -> T_GREATER; d <= -0.01 -> T_LESS"
)
T_CONSEQUENCES = {
    "T_GREATER": (
        "the placeholder was a handicap; genuine set-invariance is both a "
        "better method and a clean mechanism sentence — T becomes the "
        "headline system"
    ),
    "T_APPROX_EQUAL": (
        "placeholders are harmless; the effect is subset variation per se — "
        "describe it that way"
    ),
    "T_LESS": (
        "the constant placeholder / attention-sink mass is doing the work; "
        "surprising, and the 'population invariance' framing is wrong"
    ),
}
G_BAND_RULE = (
    "G-minus-ArmA EXTERNAL paired governing mean delta d: d >= +0.08 -> "
    "GAIN_EXPLAINS_MOST; +0.03 <= d < +0.08 -> GAIN_SECONDARY_TERM; "
    "d < +0.03 -> GAIN_NOT_THE_MECHANISM"
)
G_CONSEQUENCES = {
    "GAIN_EXPLAINS_MOST": (
        "over half of D's +0.1576; the population story is largely wrong — "
        "reframe around scale augmentation and re-derive everything"
    ),
    "GAIN_SECONDARY_TERM": (
        "gain is a real secondary term; D's claim must be stated as net of it"
    ),
    "GAIN_NOT_THE_MECHANISM": (
        "factor (b) is not the mechanism; D's population claim is clean"
    ),
}
C_GATE_RULE = (
    "conjunctive with NO bootstrap escape clause: (C-minus-D external paired "
    "governing mean >= +0.03) AND (>= 10 of 15 external sessions positive) "
    "AND (C-minus-D within paired governing mean >= -0.03)"
)
C_GATE_EXTERNAL_MEAN_MIN = 0.03
C_GATE_EXTERNAL_POSITIVE_MIN = 10
C_GATE_EXTERNAL_N_TOTAL = 15
C_GATE_WITHIN_MEAN_MIN = -0.03

# ---- the AM/IM decomposition matrix (handoff 2026-08-19 §3, §6) ------------
# Band: "equal to D" is |delta external governing| < 0.03, the T-minus-D band
# already accepted.  Every label below has its OWN explicit branch in
# decomposition_reading (the run_tfap_stage3.py:369-374 defect — a
# mechanism-pass-without-engineering-pass collapsed into BOTH_GATES_FAIL — is
# NOT reproduced here: the one unreachable band combination raises instead of
# being silently labelled).
AIMASK_APPROX_EQUAL_BAND = 0.03
AIMASK_LABELS = {
    "identity": "IDENTITY_ABLATION_IS_THE_MECHANISM",
    "activity": "ACTIVITY_SUBSAMPLING_IS_THE_MECHANISM",
    "joint": "JOINT_ABLATION_REQUIRED",
    "suspicious": "SUSPICIOUS__INVESTIGATE_MASK_AND_EVAL_PATH__DO_NOT_REPORT",
    "unanticipated": "UNANTICIPATED__SUPERIOR_TO_D__REPORT_AS_MEASURED_NO_CLAIM",
}
AIMASK_MATRIX_RULE = (
    "AM/IM decomposition matrix (HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819"
    ".md #3): per cell the EXTERNAL paired governing mean delta vs D, "
    "d = cell_swa - D_swa; band |d| < 0.03 -> APPROX_EQUAL_TO_D; "
    "d <= -0.03 -> MUCH_LESS_THAN_D; d >= +0.03 -> SUPERIOR_TO_D (out of "
    "matrix).  Rows: (IM APPROX_EQUAL_TO_D and AM MUCH_LESS_THAN_D) -> "
    "IDENTITY_ABLATION_IS_THE_MECHANISM; (AM APPROX_EQUAL_TO_D and IM "
    "MUCH_LESS_THAN_D) -> ACTIVITY_SUBSAMPLING_IS_THE_MECHANISM; (both "
    "MUCH_LESS_THAN_D) -> JOINT_ABLATION_REQUIRED; (both APPROX_EQUAL_TO_D) "
    "-> SUSPICIOUS__INVESTIGATE_MASK_AND_EVAL_PATH__DO_NOT_REPORT; any "
    "SUPERIOR_TO_D -> UNANTICIPATED__SUPERIOR_TO_D__REPORT_AS_MEASURED_"
    "NO_CLAIM.  The matrix emits ONLY when BOTH AM and IM have landed"
)
AIMASK_CONSEQUENCES = {
    AIMASK_LABELS["identity"]: (
        "the mechanism is identity/fingerprint ablation — headline rewrite; "
        "the claim becomes far more specific and connects to the carrier "
        "thesis that is the lane's actual subject"
    ),
    AIMASK_LABELS["activity"]: (
        "the mechanism is activity sub-sampling — the population story stands "
        "as written; this round is a confirmatory control"
    ),
    AIMASK_LABELS["joint"]: (
        "the mechanism requires joint ablation — the token must be fully "
        "replaced; population story stands in a stronger form; report as the "
        "decomposition's floor"
    ),
    AIMASK_LABELS["suspicious"]: (
        "suspicious — conflicts with R, where a matched amount of elementwise "
        "noise came in below baseline. Do not report as a finding; investigate "
        "the mask draw and the evaluation-path equality first"
    ),
    AIMASK_LABELS["unanticipated"]: (
        "out of the pre-registered matrix — report the number as measured and "
        "make NO claim from it; a cell superior to D requires its own handoff "
        "(and seeds 43/44 for any A2 comparison) before any use"
    ),
}
# which rows may be REPORTED as a finding (the SUSPICIOUS row is the sanity
# check, and the UNANTICIPATED row is report-as-measured with no claim)
AIMASK_REPORTABLE_AS_FINDING = {
    AIMASK_LABELS["identity"]: True,
    AIMASK_LABELS["activity"]: True,
    AIMASK_LABELS["joint"]: True,
    AIMASK_LABELS["suspicious"]: False,
    AIMASK_LABELS["unanticipated"]: False,
}
AIMASK_MECHANISM_SENTENCE = {
    AIMASK_LABELS["identity"]: (
        "REVOKED: 'training on random sub-populations' must change to "
        "stochastic suppression of the session-fingerprint identity signal, "
        "and every document asserting the population framing needs revision "
        "(predecessor handoff §0.1 included)"
    ),
    AIMASK_LABELS["activity"]: (
        "UNCHANGED: 'training on random sub-populations' stands as written"
    ),
    AIMASK_LABELS["joint"]: (
        "UNCHANGED but stronger: the population sentence stands with the "
        "token fully replaced (the decomposition's floor)"
    ),
    AIMASK_LABELS["suspicious"]: (
        "NO CHANGE PERMITTED: no sentence may be taken from the SUSPICIOUS "
        "row; investigate the mask draw and the evaluation-path equality first"
    ),
    AIMASK_LABELS["unanticipated"]: (
        "NO CHANGE PERMITTED: an out-of-matrix result licenses no mechanism "
        "sentence"
    ),
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _temporal_residual_module():
    """The route-owned cell W module (loaded once, reused from sys.modules)."""
    module = sys.modules.get("tfpd_lane_temporal_residual_cell")
    if module is None or not hasattr(module, "build_temporal_residual_model"):
        module = _load_module(
            "tfpd_lane_temporal_residual_cell",
            ROOT / "src/tfpd_lane/temporal_residual_cell.py",
        )
    return module


def w_head_state_keys() -> tuple[str, ...]:
    """The 8 W-head state keys, taken from the route module (never duplicated)."""
    return tuple(_temporal_residual_module().W_HEAD_STATE_KEYS)


def w_head_parameter_count() -> int:
    return int(_temporal_residual_module().W_HEAD_PARAMETER_COUNT)


ACTIVITY_IDENTITY_CELLS_PATH = ROOT / "src/tfpd_lane/activity_identity_cells.py"


def _activity_identity_module():
    """The route-owned AM/IM module — LAZY, resolved only when a cell lands.

    The module is being built in parallel (agreed interface:
    ``build_activity_identity_model(seed=42, cell="AM"|"IM")``), so nothing
    at import time of THIS scorer may touch it: it is loaded on first use —
    which can only happen for a cell whose terminal receipt exists — and a
    missing file is a hard, named failure rather than an import traceback.
    """
    module = sys.modules.get("tfpd_lane_activity_identity_cells")
    if module is not None and hasattr(module, "build_activity_identity_model"):
        return module
    if not ACTIVITY_IDENTITY_CELLS_PATH.is_file():
        raise SystemExit(
            "src/tfpd_lane/activity_identity_cells.py has not landed yet; the "
            "AM/IM graph builder is unavailable (a landed terminal receipt "
            "without its builder module is never silently scored)"
        )
    return _load_module(
        "tfpd_lane_activity_identity_cells", ACTIVITY_IDENTITY_CELLS_PATH
    )


# ---------------------------------------------------------------------------
# the W curve guard (the wrong-graph path must raise, never run silently)
# ---------------------------------------------------------------------------
def refuse_cell_curve(cell: str | None) -> None:
    """A curve on a native-only cell must RAISE, never silently score."""
    if cell in CURVE_EXCLUDED_CELLS:
        raise SystemExit(f"cell {cell} curve refused: {CURVE_EXCLUDED_CELLS[cell]}")


def curve_cells_for(landed_cells):
    """Structural native-only exclusion: the cells allowed into the 0C curve.

    Returns ``(curve_cells, excluded_notes)``; ``excluded_notes`` maps every
    landed native-only cell to the reason and is recorded in the receipt.
    """
    excluded = {
        cell: CURVE_EXCLUDED_CELLS[cell]
        for cell in landed_cells
        if cell in CURVE_EXCLUDED_CELLS
    }
    return [cell for cell in landed_cells if cell not in CURVE_EXCLUDED_CELLS], excluded


# ---------------------------------------------------------------------------
# pre-registered readings (pure functions — unit-tested on synthetic stats)
# ---------------------------------------------------------------------------
def t_trichotomy(delta: float | None) -> str | None:
    """Handoff §3 Cell T: T > D / T ~ D / T < D from |d| vs 0.01."""
    if delta is None:
        return None
    if delta >= 0.01:
        return "T_GREATER"
    if delta <= -0.01:
        return "T_LESS"
    return "T_APPROX_EQUAL"


def g_band(delta: float | None) -> str | None:
    """Handoff §3 Cell G: >= +0.08 / [+0.03, +0.08) / < +0.03."""
    if delta is None:
        return None
    if delta >= 0.08:
        return "GAIN_EXPLAINS_MOST"
    if delta >= 0.03:
        return "GAIN_SECONDARY_TERM"
    return "GAIN_NOT_THE_MECHANISM"


def aimask_cell_band(delta: float | None) -> str | None:
    """Handoff 2026-08-19 §3 band for ONE cell's external delta vs D.

    Three explicit branches, boundaries inclusive (the T/G convention):
    ``|d| < 0.03`` is APPROX_EQUAL_TO_D, ``d <= -0.03`` is
    MUCH_LESS_THAN_D, ``d >= +0.03`` is SUPERIOR_TO_D (out of matrix).
    """
    if delta is None:
        return None
    if abs(delta) < AIMASK_APPROX_EQUAL_BAND:
        return "APPROX_EQUAL_TO_D"
    if delta <= -AIMASK_APPROX_EQUAL_BAND:
        return "MUCH_LESS_THAN_D"
    return "SUPERIOR_TO_D"


def decomposition_reading(am_delta: float | None, im_delta: float | None) -> dict:
    """The pre-registered AM/IM reading matrix (handoff 2026-08-19 §3).

    Pure function on the two external governing paired-mean deltas
    (``cell_swa - D_swa``); ``None`` means the cell has not landed / was not
    scored.  The matrix emits ONLY when BOTH deltas are present — a single
    landed cell emits just its delta with ``matrix_incomplete``.  Every row
    label has an explicit branch below; the one combination that cannot occur
    after the band function raises rather than being silently labelled (the
    run_tfap_stage3.py:369-374 defect is not reproduced).
    """
    cells = {}
    for cell, delta in (("AM", am_delta), ("IM", im_delta)):
        if delta is None:
            continue
        cells[cell] = {
            "delta_external_governing_mean": float(delta),
            "approx_equal_to_d": bool(abs(delta) < AIMASK_APPROX_EQUAL_BAND),
            "band": aimask_cell_band(delta),
            "input": (
                f"{cell}_swa - D_swa, external, governing last bin, paired mean"
            ),
        }
    if not cells:
        matrix_status, row = "no_am_im_cells_scored", None
    elif len(cells) == 1:
        matrix_status, row = "matrix_incomplete", None
    else:
        matrix_status = "matrix_complete"
        am_band, im_band = cells["AM"]["band"], cells["IM"]["band"]
        if am_band == "SUPERIOR_TO_D" or im_band == "SUPERIOR_TO_D":
            row = AIMASK_LABELS["unanticipated"]
        elif am_band == "APPROX_EQUAL_TO_D" and im_band == "APPROX_EQUAL_TO_D":
            row = AIMASK_LABELS["suspicious"]
        elif am_band == "MUCH_LESS_THAN_D" and im_band == "MUCH_LESS_THAN_D":
            row = AIMASK_LABELS["joint"]
        elif am_band == "APPROX_EQUAL_TO_D" and im_band == "MUCH_LESS_THAN_D":
            row = AIMASK_LABELS["activity"]
        elif am_band == "MUCH_LESS_THAN_D" and im_band == "APPROX_EQUAL_TO_D":
            row = AIMASK_LABELS["identity"]
        else:
            raise SystemExit(
                f"unreachable AM/IM band combination: AM={am_band}, IM={im_band}"
            )
    reading = {
        "handoff": (
            "HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md #3 (pre-registered "
            "readings) / #6 (framing)"
        ),
        "rule": AIMASK_MATRIX_RULE,
        "band_definition": (
            "equal to D means |delta external governing| < 0.03 (the T-minus-D "
            "band already accepted, -0.0099)"
        ),
        "cells": cells,
        "matrix_status": matrix_status,
        "matrix_row": row,
        "consequence": AIMASK_CONSEQUENCES.get(row),
        "reportable_as_finding": (
            AIMASK_REPORTABLE_AS_FINDING[row] if row is not None else None
        ),
        "mechanism_sentence_status": (
            AIMASK_MECHANISM_SENTENCE[row] if row is not None else (
                "no mechanism-sentence change: the matrix has not emitted a row"
            )
        ),
        "framing_note": (
            "the mechanism sentence licensed today is 'training on random "
            "sub-populations' (T settled it); THIS ROUND CAN REVOKE IT — if IM "
            "~ D the sentence must change to identity suppression (handoff #6). "
            "The carrier remains necessary regardless of outcome; whichever of "
            "AM/IM fails is reported as a negative result that narrows the "
            "mechanism"
        ),
    }
    if matrix_status == "matrix_incomplete":
        reading["matrix_incomplete_note"] = (
            f"only {sorted(cells)} of AM/IM has landed; the pre-registered "
            "matrix needs BOTH cells, so no reading is emitted — the single "
            "delta above is recorded but must not be read against the matrix"
        )
    return reading


def c_gate_conditions(external: dict, within: dict) -> tuple[dict, bool]:
    """Handoff §3 Cell C gate: conjunctive, no bootstrap escape clause.

    ``external`` / ``within`` are paired-session stats dicts (mean, n_positive,
    n_total).  A bootstrap interval may NEVER rescue a failed mean: it is not
    read here at all.
    """
    conditions = {
        "external_mean_ge_plus_0.03": float(external["mean"]) >= C_GATE_EXTERNAL_MEAN_MIN,
        "external_positive_ge_10_of_15": (
            int(external["n_positive"]) >= C_GATE_EXTERNAL_POSITIVE_MIN
        ),
        "within_mean_ge_minus_0.03": float(within["mean"]) >= C_GATE_WITHIN_MEAN_MIN,
        "external_surface_is_15_sessions": (
            int(external["n_total"]) == C_GATE_EXTERNAL_N_TOTAL
        ),
    }
    return conditions, bool(all(conditions.values()))


def a2_development_screen(external_mean: float, within_mean: float,
                          a2_bars: dict) -> dict:
    """§7.3-style screen: BOTH absolute governing means must meet the bars.

    Descriptive only — a seed-42 screen, never a superiority claim.
    """
    return {
        "external_mean": float(external_mean),
        "external_meets_pooled_bar": bool(
            external_mean >= a2_bars["external"]
        ),
        "within_mean": float(within_mean),
        "within_meets_pooled_bar": bool(within_mean >= a2_bars["within"]),
        "development_screen_pass": bool(
            external_mean >= a2_bars["external"] and within_mean >= a2_bars["within"]
        ),
        "note": (
            "seed-42 development screen against the pooled A2 bars; "
            "DESCRIPTIVE ONLY, not a superiority claim"
        ),
    }


# ---------------------------------------------------------------------------
# references (LOADED from SHA-verified sealed receipts, never hardcoded)
# ---------------------------------------------------------------------------
def load_governing_references(sha256_file) -> dict:
    """A2 pooled + per-seed bars and the Arm A / D governing bars.

    Both receipts are body-SHA-verified BEFORE any value is read; every
    number is then pulled by pointer from the verified payload.
    """
    # --- A2 pooled + per seed (a2_matched_rescore_v1_r1) --------------------
    if not A2_R1_RECEIPT.is_file():
        raise SystemExit("A2 _r1 receipt missing")
    a2_sha = sha256_file(A2_R1_RECEIPT)
    if a2_sha != A2_R1_RECEIPT_SHA:
        raise SystemExit(f"A2 _r1 receipt SHA mismatch: {a2_sha}")
    a2_payload = json.loads(A2_R1_RECEIPT.read_text())
    pooled_ext = a2_payload["pooled_per_session"]["A2_t4_pooled_external"]
    pooled_win = a2_payload["pooled_per_session"]["A2_t4_pooled_within"]
    per_seed = {}
    for seed in (42, 43, 44):
        node = a2_payload["results"][f"A2_t4_s{seed}"]
        per_seed[str(seed)] = {
            "external": float(node["external"]["mean_r2"]),
            "within": float(node["within"]["mean_r2"]),
        }
    seed_ext = [v["external"] for v in per_seed.values()]
    arm_a = a2_payload["results"]["armA_direct_t4_48_swa_lastbin"]

    # --- Arm A / D governing bars (sparsification_score_v1) -----------------
    if not SPARSIFY_SCORE_RECEIPT.is_file():
        raise SystemExit("sparsification_score_v1 receipt missing")
    score_sha = sha256_file(SPARSIFY_SCORE_RECEIPT)
    if score_sha != SPARSIFY_SCORE_RECEIPT_SHA:
        raise SystemExit(f"sparsification_score_v1 receipt SHA mismatch: {score_sha}")
    score_payload = json.loads(SPARSIFY_SCORE_RECEIPT.read_text())
    bars = {}
    per_session_tables = {}
    # the shared one-p-per-forward stream of R/S2 (== T == G == AM == IM):
    # pulled by pointer from the SAME verified payload, never hardcoded
    shared_p = [
        (score_payload.get("cell_provenance", {}).get(cell) or {}).get(
            "p_sequence_sha256"
        )
        for cell in ("R", "S2")
    ]
    if not shared_p[0] or shared_p[0] != shared_p[1]:
        raise SystemExit(
            "R/S2 p_sequence_sha256 missing or disagreeing in the sealed "
            "sparsification receipt"
        )
    for key in ("armA_swa", "D_swa"):
        node = score_payload["results"][key]["governing_last_bin"]
        bars[key] = {
            "external": float(node["external"]["mean_r2"]),
            "within": float(node["within"]["mean_r2"]),
        }
        per_session_tables[key] = {
            surface: {row["session"]: float(row["r2"])
                      for row in node[surface]["per_session"]}
            for surface in ("within", "external")
        }
    # cross-receipt reconciliation: arm A last-bin in the A2 rescore must be
    # the arm A governing mean in the sparsification score
    arm_a_reconciled = (
        float(arm_a["external"]["mean_r2"]) == bars["armA_swa"]["external"]
        and float(arm_a["within"]["mean_r2"]) == bars["armA_swa"]["within"]
    )
    if not arm_a_reconciled:
        raise SystemExit("arm A governing mean disagrees across sealed receipts")

    return {
        "a2": {
            "pooled": {
                "external": float(sum(pooled_ext.values()) / len(pooled_ext)),
                "within": float(sum(pooled_win.values()) / len(pooled_win)),
            },
            "per_seed": per_seed,
            "per_seed_external_spread": float(max(seed_ext) - min(seed_ext)),
            "per_seed_external_max": float(max(seed_ext)),
            "receipt": str(A2_R1_RECEIPT),
            "receipt_sha256": a2_sha,
        },
        "governing_bars": {
            "armA": bars["armA_swa"],
            "D": bars["D_swa"],
            "per_session_governing_last_bin": per_session_tables,
            "receipt": str(SPARSIFY_SCORE_RECEIPT),
            "receipt_sha256": score_sha,
        },
        "shared_one_draw_p_sequence": {
            "value": shared_p[0],
            "bound_from": ["R", "S2"],
            "receipt": str(SPARSIFY_SCORE_RECEIPT),
            "receipt_sha256": score_sha,
            "note": (
                "the single one-p-per-forward stream shared by R/S2/T/G and "
                "(handoff 2026-08-19 §3) by AM/IM: one identical mask draw, "
                "the same F.dropout code path"
            ),
        },
        "armA_reconciled_across_receipts": True,
    }


def load_step0c_baseline(sha256_file, path=None, expected_sha: str | None = None) -> dict:
    """The sealed Step 0C curve (C's flatter-curve baseline), sidecar-verified.

    The Step 0C receipt is sealed by its own sidecar; an explicit expected body
    SHA may be supplied with ``--step0c-receipt-sha`` for strict binding.  A
    missing receipt is NOT fatal: the D curve is scored live in the same pass
    regardless, and the receipt records the baseline as unavailable.
    """
    receipt = Path(path) if path is not None else STEP0C_RECEIPT
    if not receipt.is_file():
        return {
            "available": False,
            "path": str(receipt),
            "note": (
                "Step 0C receipt not present; C's flatter-curve baseline is the "
                "D curve scored LIVE in this pass (same engine, same seed rule)"
            ),
        }
    body_sha = sha256_file(receipt)
    problems = []
    sidecar = Path(str(receipt) + ".sha256")
    if sidecar.is_file() and sidecar.read_text().split()[0] != body_sha:
        problems.append("sidecar SHA mismatch")
    if expected_sha is not None and body_sha != expected_sha:
        problems.append(f"body SHA != expected ({body_sha})")
    if problems:
        raise SystemExit(f"Step 0C baseline receipt rejected: {problems}")
    payload = json.loads(receipt.read_text())
    curves: dict[str, dict] = {}
    for model_key, blocks in payload.get("results", {}).items():
        if not isinstance(blocks, dict):
            continue
        for label, block in blocks.items():
            if not isinstance(block, dict) or "governing_mean_r2" not in block:
                continue
            curves.setdefault(model_key, {})[label] = {
                "governing_mean_r2": float(block["governing_mean_r2"]),
                "n_seeds": block.get("n_seeds"),
                "per_session_mean_over_seeds": {
                    session: float(value) for session, value
                    in (block.get("per_session_mean_over_seeds") or {}).items()
                },
            }
    return {
        "available": True,
        "path": str(receipt),
        "receipt_sha256": body_sha,
        "sidecar_verified": sidecar.is_file(),
        "status": payload.get("status"),
        "fractions": (payload.get("plan") or {}).get("fractions"),
        "n_mask_seeds": (payload.get("plan") or {}).get("n_mask_seeds"),
        "curves": curves,
    }


# ---------------------------------------------------------------------------
# the supersedes field (handoff 2026-08-19 §5.3): every receipt this scorer
# writes names the prior receipt(s) it extends or re-scores
# ---------------------------------------------------------------------------
RECEIPT_BASENAME = "subpop_score_receipt.json"


def family_predecessor(out_dir: Path) -> Path | None:
    """``results/subpop_score_v1_r2`` -> ``results/subpop_score_v1_r1``.

    ``_r1``'s predecessor is the unsuffixed BASE root of the family; ``None``
    for the base root itself and for any name that is not an ``_rN`` member
    of the default output-root family (unrelated explicit roots have no
    predecessor slot).
    """
    stem, separator, suffix = out_dir.name.rpartition("_r")
    if not separator or not suffix.isdigit():
        return None
    index = int(suffix)
    if index < 1:
        return None
    if index == 1:
        return out_dir.parent / stem
    return out_dir.parent / f"{stem}_r{index - 1}"


def predecessor_receipt_chain(out_dir: Path) -> list[str]:
    """The existing predecessor receipts of ``out_dir``, newest first.

    Walks the ``_rN`` family backwards while receipts exist and extends the
    chain TRANSITIVELY with each predecessor's own ``supersedes`` list (the
    earlier receipts lack the field entirely — the inherited defect this
    machinery exists to end — which simply stops the extension).  The walk
    STOPS at the first family member without a receipt: the sealed family is
    expected to be contiguous, and a gap is recorded as no chain rather than
    silently bridged.
    """
    chain: list[str] = []

    def _add(path: str) -> None:
        if path not in chain:  # a predecessor may already appear in a prior list
            chain.append(path)

    current = family_predecessor(out_dir)
    while current is not None:
        receipt_file = current / RECEIPT_BASENAME
        if not receipt_file.is_file():
            break
        _add(str(receipt_file))
        try:
            prior = json.loads(receipt_file.read_text()).get("supersedes") or []
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"unreadable predecessor receipt {receipt_file}: {error}"
            )
        for path in prior:
            _add(path)
        current = family_predecessor(current)
    return chain


def resolve_supersedes(out_dir: Path, explicit=None) -> dict:
    """The receipt's top-level supersedes block (handoff 2026-08-19 §5.3).

    Three explicit sources, no collapsed cases: operator-listed paths via
    ``--supersedes`` (each must exist), else the output-root family's
    existing predecessor receipts (auto-derived, transitive), else an empty
    list for a first run.
    """
    if explicit:
        paths = []
        for item in explicit:
            path = Path(item)
            if not path.is_file():
                raise SystemExit(f"--supersedes receipt not found: {path}")
            paths.append(str(path))
        return {
            "supersedes": paths,
            "supersedes_source": "explicit_cli",
            "supersedes_note": (
                "extends/re-scores the operator-listed prior receipt(s); they "
                "are read only here and stay sealed at their own roots"
            ),
        }
    chain = predecessor_receipt_chain(out_dir)
    if chain:
        return {
            "supersedes": chain,
            "supersedes_source": "auto_derived_from_output_root_family",
            "supersedes_note": (
                "auto-derived: this pass extends/re-scores the existing "
                "predecessor receipt(s) of the output-root family (newest "
                "first, then their own supersedes chain), so cumulative vs "
                "corrective is readable off the receipt (handoff 2026-08-19 "
                "§5.3)"
            ),
        }
    return {
        "supersedes": [],
        "supersedes_source": "none_first_run",
        "supersedes_note": (
            "first run: no predecessor receipt exists in this output root's "
            "family and no --supersedes was given"
        ),
    }


# ---------------------------------------------------------------------------
# cell terminal validation (adapted from run_sparsify_score.validate_cell_terminal)
# ---------------------------------------------------------------------------
def aimask_law_problems(cell: str, integrity: dict) -> list[str]:
    """Handoff 2026-08-19 §3 cell-law checks for the AM/IM mask.

    Five law fields are checked (mask_structure and mask_code_path_matches_D
    exactly, via the cell spec's ``expected``; gain, min_keep and the
    application site here): the gain must be the ``1/(1-p)`` INHERITED from
    the shared ``F.dropout`` call (never a reimplemented rescale) — recorded
    under ``gain`` or the landed runner's ``gain_rule`` name — min_keep must
    be "none" (or an explicit integer clamp if the runner records one), and
    the application site must mask EXACTLY this cell's component and not the
    other one.
    """
    problems: list[str] = []
    masked, untouched = (
        ("activity", "identity") if cell == "AM" else ("identity", "activity")
    )
    gain = str(integrity.get("gain_rule", integrity.get("gain", "")))
    if "1/(1-p)" not in gain or "dropout" not in gain.lower():
        problems.append(
            f"cell {cell} gain {gain!r} does not inherit 1/(1-p) from F.dropout"
        )
    min_keep = integrity.get("min_keep")
    if not (isinstance(min_keep, int) or "none" in str(min_keep).lower()):
        problems.append(
            f"cell {cell} min_keep {min_keep!r} is neither 'none' nor an "
            "integer clamp"
        )
    site = "".join(str(integrity.get("application_site", "")).split())
    if f"{masked}*mask" not in site:
        problems.append(
            f"cell {cell} application_site {integrity.get('application_site')!r} "
            f"does not mask the {masked} component"
        )
    if f"{untouched}*mask" in site:
        problems.append(
            f"cell {cell} application_site {integrity.get('application_site')!r} "
            f"masks the {untouched} component too (one-factor contrast violated)"
        )
    return problems


def cell_law_problems(cell: str, integrity: dict) -> list[str]:
    """Cell-identity checks on the perturbation law the terminal receipt binds."""
    problems: list[str] = []
    if cell in AIMASK_CELLS:
        problems.extend(aimask_law_problems(cell, integrity))
    elif cell == "T":
        if integrity.get("key_padding_mask_semantics") in (None, "n/a (no masking)"):
            problems.append("cell T records no key_padding_mask semantics")
        policy = str(integrity.get("rescaling_policy", ""))
        if "none" not in policy.lower():
            problems.append("cell T must not rescale survivors")
        clamp = (integrity.get("p_distribution_and_clamp_policy") or {}).get("clamp")
        if str(clamp).lower() != "none":
            problems.append(f"cell T p clamp {clamp!r} != 'none'")
    elif cell == "G":
        if integrity.get("mask_structure") != "none_global_gain_all_units":
            problems.append(
                f"cell G mask_structure {integrity.get('mask_structure')!r}"
            )
        clamp = str((integrity.get("p_distribution_and_clamp_policy") or {}).get("clamp", ""))
        if "0.95" not in clamp:
            problems.append(f"cell G p clamp {clamp!r} does not bind 0.95")
    elif cell == "W":
        head = integrity.get("head_structure") or {}
        if head.get("k_latent_frozen") != 8:
            problems.append(
                f"cell W k_latent_frozen {head.get('k_latent_frozen')!r} != 8"
            )
        if "none" not in str(head.get("k_sweep", "")).lower():
            problems.append(f"cell W K sweep {head.get('k_sweep')!r} is not frozen")
        if "zero" not in str(head.get("value_head", "")).lower():
            problems.append(
                f"cell W value_head {head.get('value_head')!r} is not recorded "
                "zero-initialized"
            )
        manifest = integrity.get("w_head_parameters") or {}
        if not manifest.get("counts_match"):
            problems.append("cell W added-parameter manifest counts_match != True")
        if sorted(manifest.get("names") or []) != sorted(w_head_state_keys()):
            problems.append(
                "cell W added-parameter manifest names are not exactly the "
                "W-head keys"
            )
        if manifest.get("total_parameters") != w_head_parameter_count():
            problems.append(
                f"cell W added-parameter count {manifest.get('total_parameters')!r} "
                "!= the frozen W-head count"
            )
        for field in ("zero_init_guarantee", "gradient_flow", "rng_note",
                      "non_causal_caveat", "loss"):
            if not integrity.get(field):
                problems.append(f"cell W records no {field}")
        if "no masking" not in str(integrity.get("loss", "")):
            problems.append("cell W loss must record 'no masking' (head-only change)")
    else:
        if integrity.get("consistency_site") != "behaviour predictions (never a latent)":
            problems.append(
                f"cell C consistency_site {integrity.get('consistency_site')!r}"
            )
        if integrity.get("skip_floor") != 1:
            problems.append(f"cell C skip_floor {integrity.get('skip_floor')!r} != 1")
        if (integrity.get("p_stream") or {}).get("draws_per_step") != 2:
            problems.append("cell C must record two p draws per step")
        if integrity.get("lambda_sweep") not in (None, "none"):
            problems.append(f"cell C lambda sweep {integrity.get('lambda_sweep')!r}")
    return problems


def w_terminal_problems(payload: dict, initial: dict) -> list[str]:
    """Cell W initial-state discipline: strict=False + the 8-key missing set.

    W ADDS parameters, so the T/C/G strict=True template does not apply: the
    canonical artifact is loaded with strict=False and the runner must have
    recorded (and proven) that the missing keys are EXACTLY the W-head keys,
    the unexpected keys empty, the shared-key subset byte-exact, the decode
    bitwise parent-equal at initialisation, and the residual awake by the end.
    """
    problems: list[str] = []
    if initial.get("strict_load") is not False:
        problems.append("cell W initial_state.strict_load must be False (W adds parameters)")
    if initial.get("missing_keys_exactly_w_head") is not True:
        problems.append("cell W initial_state.missing_keys_exactly_w_head is not True")
    if initial.get("unexpected_keys_empty") is not True:
        problems.append("cell W initial_state.unexpected_keys_empty is not True")
    if sorted(initial.get("missing_keys") or []) != sorted(w_head_state_keys()):
        problems.append(
            "cell W initial_state missing keys are not exactly the 8 W-head keys"
        )
    if initial.get("shared_subset_matches_artifact_state_sha256") is not True:
        problems.append(
            "cell W shared-subset state SHA binding is not recorded True"
        )
    if initial.get("zero_init_delta_exactly_zero") is not True:
        problems.append("cell W zero-init delta guarantee is not recorded True")
    if initial.get("eval_path_bitwise_parent_equal") is not True:
        problems.append("cell W eval-path bitwise parent equality is not recorded True")
    proofs = (payload.get("launch_proofs") or {}).get("zero_init_bitwise") or {}
    for field in ("torch_equal", "tensor_sha256_equal", "delta_exactly_zero"):
        if proofs.get(field) is not True:
            problems.append(
                f"cell W zero-init bitwise launch proof .{field} is not True"
            )
    diagnostics = payload.get("diagnostics_per_epoch") or []
    if diagnostics and not diagnostics[-1].get("value_head_woke"):
        problems.append(
            "cell W value_head never woke (delta stayed exactly zero through "
            "training)"
        )
    return problems


def validate_cell_terminal(cell, arm_common, canonical_payload, theta_payload,
                           root: Path | None = None):
    """Full validation of one landed cell's terminal receipt + sealed SWA.

    Returns ``None`` when the terminal receipt does not exist yet (the cell
    has not landed); raises ``SystemExit`` on ANY defect in a receipt that
    DOES exist — a landed-but-failed cell is never silently skipped.
    """
    import stat as _stat

    base = Path(root) if root is not None else ROOT
    spec = CELL_SPECS[cell]
    # AM/IM live under their own output root (results/aimask_v1); the T/C/G/W
    # cells stay under results/subpop_v1
    directory = base / spec.get("cell_root", CELL_ROOT) / spec["directory"]
    receipt_path = directory / "terminal_receipt.json"
    swa_path = directory / "swa_final4.pt"
    if not receipt_path.is_file():
        return None
    problems = []
    sidecar = Path(str(receipt_path) + ".sha256")
    if not sidecar.is_file():
        problems.append("terminal sidecar missing")
    if receipt_path.is_symlink() or (sidecar.is_file() and sidecar.is_symlink()):
        problems.append("symlinked receipt/sidecar")
    body_sha = arm_common.sha256_file(receipt_path)
    if sidecar.is_file() and body_sha != sidecar.read_text().split()[0]:
        problems.append("sidecar SHA mismatch")
    receipt_mode = _stat.S_IMODE(receipt_path.stat().st_mode)
    if receipt_mode != 0o444:
        problems.append(f"receipt mode {oct(receipt_mode)} != 0444")
    payload = json.loads(receipt_path.read_text())
    if payload.get("status") != "CELL_TERMINAL":
        problems.append(f"status {payload.get('status')!r}")
    if payload.get("smoke") is not False:
        problems.append("smoke receipt (non-authoritative)")
    if payload.get("max_train_steps") is not None:
        problems.append("truncated-step run")
    if payload.get("epochs_run") != FULL_BUDGET_EPOCHS:
        problems.append(f"epochs_run {payload.get('epochs_run')} != {FULL_BUDGET_EPOCHS}")
    if payload.get("invariant_failures"):
        problems.append("invariant failures recorded")
    if not payload.get("source_closure", {}).get("launch_final_closure_equal"):
        problems.append("launch/final closure inequality")
    initial = payload.get("initial_state", {})
    if initial.get("state_dict_sha256") != canonical_payload["state_sha256"]:
        problems.append("terminal-bound initial-state state SHA != canonical")
    if initial.get("artifact_sha256") != arm_common.sha256_file(CANONICAL_INITIAL_STATE):
        problems.append("terminal-bound initial-state artifact SHA drift")
    normalizer = payload.get("data_contract", {}).get(
        "behavior_normalizer_semantic_sha256", ""
    )
    if not str(normalizer).startswith("f062506c"):
        problems.append("terminal-bound normalizer semantic SHA drift")
    integrity = payload.get("integrity", {})
    if integrity.get("num_heads") != 2:
        problems.append(f"num_heads {integrity.get('num_heads')!r} != 2")
    # cell-identity checks on the perturbation law
    for field, expected in spec["expected"].items():
        if integrity.get(field) != expected:
            problems.append(
                f"integrity.{field} = {integrity.get(field)!r} != {expected!r}"
            )
    problems.extend(cell_law_problems(cell, integrity))
    # cell W: the strict=False canonical-load discipline (the 8-key W-head
    # missing set, shared-subset byte equality, zero-init proofs, wake-up)
    if cell == "W":
        problems.extend(w_terminal_problems(payload, initial))
    # theta-authority binding: recorded by the T/G runner; cell C has no
    # session authority (per-(batch, unit) masks) and records none
    if cell in ("T", "G"):
        theta_bound = integrity.get("theta_authority", {})
        if theta_bound.get("sha256") != arm_common.sha256_file(THETA_AUTHORITY_FILE):
            problems.append("terminal-bound theta-authority SHA drift")
        if theta_bound.get("authority_sha256") != theta_payload["authority_sha256"]:
            problems.append("terminal-bound theta authority_sha256 drift")
    # the sealed SWA artifact itself
    swa_sha = payload.get("swa", {}).get("sha256")
    if not swa_sha:
        problems.append("terminal receipt carries no swa sha256")
    if not swa_path.is_file():
        problems.append(f"SWA missing: {swa_path}")
    else:
        if swa_path.is_symlink():
            problems.append("symlinked SWA")
        live_sha = arm_common.sha256_file(swa_path)
        if swa_sha and live_sha != swa_sha:
            problems.append("SWA SHA mismatch vs terminal receipt")
        swa_sidecar = Path(str(swa_path) + ".sha256")
        if swa_sidecar.is_file() and swa_sidecar.read_text().split()[0] != live_sha:
            problems.append("SWA sidecar SHA mismatch")
    if problems:
        raise SystemExit(f"cell {cell} terminal validation failed: {problems}")
    record = {
        "cell": cell,
        "cell_name": spec["directory"],
        "terminal_receipt": str(receipt_path),
        "terminal_receipt_sha256": body_sha,
        "mode_0444": True,
        "non_symlink": True,
        "launch_final_closure_equal": True,
        "initial_state_reconciled": True,
        "normalizer_reconciled_f062506c": True,
        "theta_authority_reconciled": (
            True if cell in ("T", "G") else
            ("n/a (cell C masks are per-(batch, unit); no session authority)"
             if cell == "C" else
             f"n/a (cell {cell} binds its mask law through "
             "mask_code_path_matches_D and the shared one-draw p stream; no "
             "separate session authority is bound)")
        ),
        "swa_path": str(swa_path),
        "swa_sha256": swa_sha,
        "p_sequence_sha256": payload.get("p_sequence_sha256"),
        "p_draws_per_step": spec["p_draws_per_step"],
        "p_stream_total_draws": payload.get("p_stream_total_draws"),
        "epochs_run": payload.get("epochs_run"),
        "num_heads": integrity.get("num_heads"),
        "dropout_structure": integrity.get("dropout_structure"),
        "behavior_scaling_convention": integrity.get("behavior_scaling_convention"),
        "perturbation_law": {
            field: integrity[field] for field in spec["law_fields"] if field in integrity
        },
        "realized_perturbation_statistics": realized_perturbation_statistics(
            cell, payload.get("diagnostics_per_epoch", [])
        ),
    }
    if cell == "W":
        # W has NO p stream (an output-head change, no masking) and no session
        # theta authority; record n/a rather than None so the receipt states it
        record.update({
            "p_sequence_sha256": "n/a (no p stream: cell W is an output-head change; no masking)",
            "p_stream_total_draws": "n/a (no p stream: cell W is an output-head change; no masking)",
            "theta_authority_reconciled": (
                "n/a (cell W changes the output head; no session authority)"
            ),
        })
    return record


def realized_perturbation_statistics(cell: str, diagnostics: list[dict]) -> dict:
    """Handoff §5.6: the realized perturbation statistics, auditable not assumed."""
    if not diagnostics:
        return {"available": False, "note": "no per-epoch diagnostics in the receipt"}
    series = []
    for diag in diagnostics:
        if cell in ("T", "G"):
            summary = diag.get("perturbation_summary") or {}
            if cell == "T":
                series.append({
                    "epoch": diag.get("epoch"),
                    **{key: summary.get(key) for key in (
                        "n_forwards", "kept_fraction_mean", "surviving_units_min",
                        "surviving_units_q05", "surviving_units_median",
                        "surviving_units_mean", "surviving_units_q95",
                        "surviving_units_max", "min_keep_trigger_rate",
                        "min_keep_trigger_rows", "units_restored_by_min_keep",
                        "rows_below_min_keep_after_guard",
                        "all_masked_rows_after_guard")},
                })
            else:
                series.append({
                    "epoch": diag.get("epoch"),
                    **{key: summary.get(key) for key in (
                        "n_forwards", "gain_min", "gain_q25", "gain_median",
                        "gain_q75", "gain_max", "gain_mean", "p_raw_min",
                        "p_raw_median", "p_raw_max", "clamp_trigger_count",
                        "clamp_trigger_rate")},
                })
        elif cell == "W":
            # the delta wake-up diagnostics: the residual leaving exact zero
            probe = diag.get("temporal_head_probe") or {}
            series.append({
                "epoch": diag.get("epoch"),
                "delta_abs_mean": probe.get("delta_abs_mean"),
                "delta_abs_max": probe.get("delta_abs_max"),
                "delta_exactly_zero": probe.get("delta_exactly_zero"),
                "delta_to_base_scale_ratio": probe.get("delta_to_base_scale_ratio"),
                "base_abs_mean": probe.get("base_abs_mean"),
                "value_head_woke": diag.get("value_head_woke"),
                "value_head_weight_max_abs": probe.get("value_head_weight_max_abs"),
                "value_head_bias_max_abs": probe.get("value_head_bias_max_abs"),
            })
        elif cell in AIMASK_CELLS:
            # handoff 2026-08-19 §5.1: the realized perturbation statistics —
            # the p distribution, realized surviving-unit counts, the clamp
            # hit rate, and the same-code-path-as-D mask assertion
            summary = (
                diag.get("perturbation_summary") or diag.get("mask_summary") or {}
            )
            row = {"epoch": diag.get("epoch")}
            for key in (
                "n_forwards", "p_raw_min", "p_raw_median", "p_raw_max",
                "kept_fraction_mean", "surviving_units_min",
                "surviving_units_q05", "surviving_units_median",
                "surviving_units_mean", "surviving_units_q95",
                "surviving_units_max", "min_keep_trigger_rate",
                "min_keep_trigger_rows", "clamp_trigger_count",
                "clamp_trigger_rate",
            ):
                if key in summary:
                    row[key] = summary[key]
            if "mask_code_path_matches_D" in summary:
                row["mask_code_path_matches_D"] = summary[
                    "mask_code_path_matches_D"
                ]
            series.append(row)
        else:
            mask = diag.get("mask_summary") or {}
            series.append({
                "epoch": diag.get("epoch"),
                "jaccard_overlap": diag.get("jaccard_overlap"),
                "branch1_kept_fraction_mean": mask.get("branch1", {}).get("kept_fraction_mean"),
                "branch2_kept_fraction_mean": mask.get("branch2", {}).get("kept_fraction_mean"),
                "branch1_all_zero_samples": mask.get("branch1", {}).get("all_zero_branch_samples"),
                "branch2_all_zero_samples": mask.get("branch2", {}).get("all_zero_branch_samples"),
                "behavior_loss_sum_mean_per_step": diag.get("behavior_loss_sum_mean_per_step"),
                "consistency_loss_mean_per_step": diag.get("consistency_loss_mean_per_step"),
                "branch1_skipped_samples": diag.get("branch1_skipped_samples"),
                "branch2_skipped_samples": diag.get("branch2_skipped_samples"),
                "both_branches_skipped_samples": diag.get("both_branches_skipped_samples"),
            })
    return {
        "available": True,
        "n_epochs": len(series),
        "final_epoch": series[-1],
        "per_epoch_series": series,
    }


# ---------------------------------------------------------------------------
# curve aggregation / contrasts (the 0C registration arithmetic, reused)
# ---------------------------------------------------------------------------
def aggregate_seed_rows(label: str, kind: str, fraction, seed_rows: list[dict],
                        extra: dict | None = None) -> dict:
    """The exact aggregation of ``run_subpop_step0c.register``."""
    mean_over_seeds = float(np.mean([row["mean_r2"] for row in seed_rows]))
    per_session: dict[str, list[float]] = {}
    for row in seed_rows:
        for entry in row["per_session"]:
            per_session.setdefault(entry["session"], []).append(entry["r2"])
    return {
        "intervention": kind,
        "fraction_removed": fraction,
        "n_seeds": len(seed_rows),
        "governing_mean_r2": mean_over_seeds,
        "per_seed_mean_r2": [float(row["mean_r2"]) for row in seed_rows],
        "per_session_mean_over_seeds": {
            session: float(np.mean(values)) for session, values in per_session.items()
        },
        "n_windows_per_session": {
            entry["session"]: entry.get("n_windows")
            for entry in seed_rows[0]["per_session"]
        },
        "label": label,
        **(extra or {}),
    }


def curve_contrast(cell_block: dict, d_block: dict, matched_scorer) -> dict:
    """Per-session paired cell-minus-D deltas at ONE curve condition."""
    cell_table = cell_block["per_session_mean_over_seeds"]
    d_table = d_block["per_session_mean_over_seeds"]
    sessions = sorted(set(cell_table) & set(d_table))
    if len(sessions) != len(cell_table) or len(sessions) != len(d_table):
        raise SystemExit("curve session-roster mismatch between cell and D")
    deltas = [cell_table[s] - d_table[s] for s in sessions]
    stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
    stats["contrast"] = "cell - D (external, curve condition, seed-averaged per session)"
    return stats


def curve_drop_profile(blocks: dict, kind: str, fractions) -> dict:
    """Descriptive drop-from-fraction-0 profile for ONE model and curve kind."""
    base_block = blocks.get(f"{kind}_f0.0")
    if base_block is None:
        return {"available": False}
    base = float(base_block["governing_mean_r2"])
    drops, nonzero = {}, []
    for fraction in fractions:
        block = blocks.get(f"{kind}_f{fraction}")
        if block is None:
            continue
        drop = float(block["governing_mean_r2"]) - base
        drops[str(float(fraction))] = drop
        if float(fraction) > 0.0:
            nonzero.append(drop)
    return {
        "available": True,
        "base_fraction_0_mean_r2": base,
        "drop_from_fraction_0": drops,
        "mean_drop_over_nonzero_fractions": (
            float(np.mean(nonzero)) if nonzero else None
        ),
        "note": "DESCRIPTIVE ONLY; the pre-registered claim is per-fraction paired deltas vs D",
    }


def flatter_than_d(cell_profile: dict, d_profile: dict, fractions) -> dict:
    """Descriptive flattening comparison: a smaller drop is a flatter curve."""
    cell_drops = cell_profile.get("drop_from_fraction_0", {})
    d_drops = d_profile.get("drop_from_fraction_0", {})
    per_fraction, flatter = {}, 0
    for fraction in fractions:
        key = str(float(fraction))
        if float(fraction) <= 0.0 or key not in cell_drops or key not in d_drops:
            continue
        per_fraction[key] = {
            "cell_drop": cell_drops[key], "d_drop": d_drops[key],
            "cell_flatter": bool(cell_drops[key] > d_drops[key]),
        }
        flatter += int(cell_drops[key] > d_drops[key])
    return {
        "per_fraction": per_fraction,
        "n_nonzero_fractions": len(per_fraction),
        "n_fractions_flatter_than_d": flatter,
        "note": (
            "DESCRIPTIVE summary of the pre-registered flatter-curve claim; the "
            "governing evidence is the per-fraction paired deltas vs D"
        ),
    }


# ---------------------------------------------------------------------------
# diagnostics helpers (run_sparsify_score conventions)
# ---------------------------------------------------------------------------
def window_weighted_mean(per_session_rows):
    """Full-window DIAGNOSTIC aggregation weighted by per-session n_windows."""
    total_windows = sum(row.get("n_windows_scored", 0) for row in per_session_rows)
    if total_windows == 0:
        return None
    return float(
        sum(row["r2"] * row.get("n_windows_scored", 0) for row in per_session_rows)
        / total_windows
    )


def all_positions(dataset, starts_by_session):
    positions = []
    for position, (session, _start) in enumerate(dataset.window_indices):
        if session in starts_by_session:
            positions.append(position)
    return positions


def annotate_n_windows(block: dict) -> dict:
    """§5.1: every per_session entry carries ``n_windows`` (copy of the
    full-window scorer's ``n_windows_scored``), without mutating sealed modules."""
    for row in block["per_session"]:
        row.setdefault("n_windows", row.get("n_windows_scored"))
    return block


def build_cell_model(subpop, consistency_cell, cell: str, temporal_residual=None):
    """The exact Arm A graph via the ROUTE wrappers (handoff §3 cell builders).

    Cell W builds through ``temporal_residual_cell.build_temporal_residual_model``
    — the Arm A graph PLUS the zero-init temporal latent residual head (the
    head is ACTIVE at eval, which is exactly why the Step 0C curve wrapper is
    wrong for W; see refuse_cell_curve).  Cells AM/IM build through the LAZY
    ``activity_identity_cells.build_activity_identity_model`` (the module is
    resolved only here, i.e. only once the cell has landed).  The module is
    passed by the caller or resolved from sys.modules.
    """
    builder = CELL_SPECS[cell]["builder"]
    if builder == "subpop":
        return subpop.build_subpop_model(seed=42, cell=cell)
    if builder == "temporal_residual":
        module = temporal_residual or _temporal_residual_module()
        return module.build_temporal_residual_model(seed=42)
    if builder == "activity_identity":
        return _activity_identity_module().build_activity_identity_model(
            seed=42, cell=cell
        )
    return consistency_cell.build_consistency_model(seed=42)


def verify_p_stream_binding(provenance: dict,
                            shared_p_sequence: str | None = None) -> None:
    """One p per forward, ONE shared stream: T, G, AM and IM.

    T and G draw ONE p per step from the exact shared stream; C draws two; W
    has none at all, so its presence never enters these assertions.  Cells
    AM/IM join the SAME one-draw stream (handoff 2026-08-19 §3: one identical
    mask draw per forward, shared with D/R/S2/T/G), so:

    1. every landed one-draw cell must agree on the p-stream SHA (the T/G
       pair keeps its exact historical error message); and
    2. when ``shared_p_sequence`` is supplied (the R/S2 value LOADED from the
       SHA-verified sealed sparsification receipt), every landed AM/IM cell
       must equal it — the new cells are hard-bound to the sealed stream.
    """
    if "T" in provenance and "G" in provenance:
        if provenance["T"]["p_sequence_sha256"] != provenance["G"]["p_sequence_sha256"]:
            raise SystemExit("T/G p-stream SHA inequality")
    one_draw = {
        cell: record["p_sequence_sha256"]
        for cell, record in provenance.items()
        if record.get("p_draws_per_step") == 1
    }
    if len(one_draw) >= 2 and len(set(one_draw.values())) > 1:
        raise SystemExit(
            f"one-draw-per-step cells disagree on the p-stream SHA: {one_draw}"
        )
    if shared_p_sequence is not None:
        for cell in AIMASK_CELLS:
            if cell in one_draw and one_draw[cell] != shared_p_sequence:
                raise SystemExit(
                    f"cell {cell} p_sequence_sha256 != the sealed shared "
                    "R/S2/T/G one-draw stream"
                )


def w_reading(contrasts: dict) -> dict:
    """The cell W reading: EXPLORATORY, verdict-free, NO pre-registered gate.

    Assembles the paired contrasts vs Arm A and vs D (same-engine live scoring,
    ``matched_scorer.paired_session_stats``) on BOTH surfaces and BOTH
    granularities — the handoff reports both for W specifically, since it is
    the cell intended to change the within-window profile.
    """
    def grab(name):
        if name not in contrasts:
            raise SystemExit(f"cell W reading missing contrast {name!r}")
        return contrasts[name]

    return {
        "label": W_READING_LABEL,
        "verdict": None,
        "pass": None,
        "rule": "none — cell W is the exploratory cell; no gate was pre-registered",
        "input": (
            "W_swa - armA_swa and W_swa - D_swa, both surfaces, BOTH "
            "granularities (governing last-bin + diagnostic full window)"
        ),
        "w_minus_armA_external": grab("W_swa_minus_armA_swa_external_governing_last_bin"),
        "w_minus_armA_within": grab("W_swa_minus_armA_swa_within_governing_last_bin"),
        "w_minus_d_external": grab("W_swa_minus_D_swa_external_governing_last_bin"),
        "w_minus_d_within": grab("W_swa_minus_D_swa_within_governing_last_bin"),
        "diagnostic_full_window": {
            "w_minus_armA_external": grab(
                "W_swa_minus_armA_swa_external_diagnostic_full_window"),
            "w_minus_armA_within": grab(
                "W_swa_minus_armA_swa_within_diagnostic_full_window"),
            "w_minus_d_external": grab(
                "W_swa_minus_D_swa_external_diagnostic_full_window"),
            "w_minus_d_within": grab(
                "W_swa_minus_D_swa_within_diagnostic_full_window"),
        },
        "granularity_note": (
            "BOTH granularities are reported for W specifically (handoff #3 "
            "Cell W: it is the cell intended to change the within-window "
            "profile); last-bin equal-session remains GOVERNING and the "
            "full-window diagnostic can never rescue it"
        ),
        "adoption_note": (
            "EXPLORATORY: adoption requires its own follow-up — seeds 43/44 "
            "plus one Z4 sibling to establish whether any gain is generic or a "
            "better consumption of T4 (HANDOFF_SPINT_DECODER_DIRECTIONS_20260817 "
            "#5 Priority 2); no adoption decision may be taken from this receipt"
        ),
        "note": W_READING_NOTE,
    }


def perturbation_disabled(model) -> bool:
    for attribute in ("perturbation_enabled", "sparsification_enabled"):
        if hasattr(model, attribute) and getattr(model, attribute):
            return False
    return True


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--authorize-target", default="")
    parser.add_argument("--cells", nargs="+",
                        default=["T", "C", "G", "W", "AM", "IM"],
                        choices=sorted(CELL_SPECS))
    parser.add_argument("--supersedes", nargs="+", default=None,
                        help="prior subpop_score receipt path(s) this pass "
                             "extends/re-scores (handoff 2026-08-19 §5.3); "
                             "auto-derived from the output-root family when "
                             "omitted")
    parser.add_argument("--fractions", type=float, nargs="+", default=None)
    parser.add_argument("--n-mask-seeds", type=int, default=None)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--limit-windows-per-session", type=int, default=None)
    parser.add_argument("--skip-curve", action="store_true")
    parser.add_argument("--step0c-receipt", type=Path, default=None)
    parser.add_argument("--step0c-receipt-sha", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    authorized = args.authorize_target == AUTH_VALUE
    if not args.dry_run and not authorized:
        print("external scoring requires --authorize-target " + AUTH_VALUE, file=sys.stderr)
        return 3

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    step0c = _load_module(
        "tfpd_subpop_step0c_machinery", ROOT / "scripts/run_subpop_step0c.py"
    )
    fractions = tuple(args.fractions) if args.fractions is not None else step0c.SPEC_FRACTIONS
    n_mask_seeds = (
        args.n_mask_seeds if args.n_mask_seeds is not None else step0c.SPEC_N_MASK_SEEDS
    )

    # ---- references + landed cells, verified BEFORE any data is opened ------
    references = load_governing_references(arm_common.sha256_file)
    step0c_baseline = load_step0c_baseline(
        arm_common.sha256_file, path=args.step0c_receipt,
        expected_sha=args.step0c_receipt_sha,
    )
    canonical_payload = torch.load(
        CANONICAL_INITIAL_STATE, map_location="cpu", weights_only=False
    )
    theta_payload = torch.load(
        THETA_AUTHORITY_FILE, map_location="cpu", weights_only=False
    )
    provenance: dict[str, dict] = {}
    not_landed: dict[str, dict] = {}
    for cell in args.cells:
        record = validate_cell_terminal(
            cell, arm_common, canonical_payload, theta_payload
        )
        if record is None:
            not_landed[cell] = {
                "note": (
                    f"terminal receipt not present yet "
                    f"({CELL_SPECS[cell].get('cell_root', CELL_ROOT)}/"
                    f"{CELL_SPECS[cell]['directory']}); cell omitted"
                )
            }
        else:
            provenance[cell] = record
    landed = [cell for cell in args.cells if cell in provenance]
    # T and G draw ONE p per step from the exact shared stream; C draws two;
    # W has no p stream at all, and AM/IM join the SAME one-draw stream as
    # R/S2/T/G (handoff 2026-08-19 §3) — bound to the sealed R/S2 value
    verify_p_stream_binding(
        provenance,
        shared_p_sequence=references["shared_one_draw_p_sequence"]["value"],
    )

    # ---- the output root + the supersedes block (handoff §5.3) --------------
    out_dir = (
        Path(args.output_root) if args.output_root is not None
        else ROOT / ("results/subpop_score_v1_smoke" if args.smoke
                     else "results/subpop_score_v1")
    )
    supersedes_block = resolve_supersedes(out_dir, args.supersedes)

    plan = {
        "schema": "tfpd_subpop_score_v1",
        "handoff": "HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md #3/#5",
        "aimask_decomposition": {
            "handoff": (
                "HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md #3 (readings) / "
                "#5 (receipt) / #6 (framing)"
            ),
            "cells": list(AIMASK_CELLS),
            "cell_root": AIMASK_CELL_ROOT,
            "shared_p_sequence_bound_from": (
                references["shared_one_draw_p_sequence"]["bound_from"]
            ),
            "curve_validity": AIMASK_CURVE_VALIDITY_NOTE,
            "rule": AIMASK_MATRIX_RULE,
            "note": (
                "AM/IM are read ONLY through the pre-registered decomposition "
                "matrix; a single landed cell emits its delta with "
                "matrix_incomplete and no reading"
            ),
        },
        "cells_requested": list(args.cells),
        "cells_landed": landed,
        "cells_not_landed": sorted(not_landed),
        "surfaces": ["within 6 sub-C dev", "external 15 sub-M"],
        "granularity": {
            "governing_last_bin": (
                "GOVERNING: last timestep of window, variance-weighted R2, "
                "equal session weight"
            ),
            "diagnostic_full_window": (
                "DIAGNOSTIC ONLY: cannot rescue a failed governing mean"
            ),
        },
        "curve": {
            "protocol": "Step 0C (i) verbatim, imported from scripts/run_subpop_step0c.py",
            "fractions": [float(f) for f in fractions],
            "n_mask_seeds": n_mask_seeds,
            "kinds_primary_first": list(step0c.CURVE_KINDS),
            "padding": {"fractions": [float(f) for f in fractions if f > 0.0],
                        "n_seeds": 1},
            "seed_rule": step0c.SEED_RULE_DOC,
            "surface": "external 15 sub-M",
            "pairing": "D scored live in the same pass; checkpoint-independent masks give bit-identical draws",
            "native_only_cells": dict(CURVE_EXCLUDED_CELLS),
            "aimask_curve_validity": AIMASK_CURVE_VALIDITY_NOTE,
            "step0c_baseline_receipt": step0c_baseline.get("path"),
            "step0c_baseline_available": step0c_baseline["available"],
        },
        "references": {
            "a2_pooled_bars": references["a2"]["pooled"],
            "armA_governing": references["governing_bars"]["armA"],
            "D_governing": references["governing_bars"]["D"],
            "a2_r1_receipt_sha256": references["a2"]["receipt_sha256"],
            "sparsification_score_receipt_sha256": (
                references["governing_bars"]["receipt_sha256"]
            ),
        },
        "pre_registered_rules": {
            "T_trichotomy": T_TRICHOTOMY_RULE,
            "G_band": G_BAND_RULE,
            "C_gate": C_GATE_RULE,
            "AM_IM_decomposition_matrix": AIMASK_MATRIX_RULE,
        },
        "supersedes": supersedes_block["supersedes"],
        "supersedes_note": supersedes_block["supersedes_note"],
        "supersedes_source": supersedes_block["supersedes_source"],
        "smoke": bool(args.smoke),
        "limit_sessions": args.limit_sessions,
        "limit_windows_per_session": args.limit_windows_per_session,
        "authorized": authorized,
    }
    if args.dry_run:
        print(json.dumps({
            **plan,
            "status": "DRY_RUN__NO_NWB_OPENED",
            "landed_cell_provenance": {
                cell: {
                    "terminal_receipt": provenance[cell]["terminal_receipt"],
                    "terminal_receipt_sha256": provenance[cell]["terminal_receipt_sha256"],
                    "swa_sha256": provenance[cell]["swa_sha256"],
                    "epochs_run": provenance[cell]["epochs_run"],
                } for cell in landed
            },
            "not_landed": not_landed,
        }, indent=1))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3
    if out_dir.exists():
        print(f"fresh output root required: {out_dir}", file=sys.stderr)
        return 2
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True)
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    rescorer = _load_module(
        "tfpd_a2_rescorer_subpop", ROOT / "scripts/run_a2_matched_rescore.py"
    )
    pilot = _load_module(
        "tfpd_z4_pilot_subpop", ROOT / "scripts/run_z4_boundary_pilot.py"
    )
    subpop = _load_module(
        "tfpd_lane_subpop_cells", ROOT / "src/tfpd_lane/subpop_cells.py"
    )
    consistency_cell = _load_module(
        "tfpd_lane_consistency_cell", ROOT / "src/tfpd_lane/consistency_cell.py"
    )
    temporal_residual = _load_module(
        "tfpd_lane_temporal_residual_cell",
        ROOT / "src/tfpd_lane/temporal_residual_cell.py",
    )

    # ---- arm A / D checkpoint integrity (the sealed 0C binding, reused) -----
    checkpoint_integrity = step0c.verify_checkpoints(arm_common.sha256_file)
    torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])

    # ---- surfaces (exact A2-matched convention) -----------------------------
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    surfaces = rescorer.build_surfaces(args, a2, "t4")
    (within_ds, within_starts), (ext_ds, ext_starts) = surfaces
    if args.limit_sessions is not None:
        within_starts = {s: v for s, v in sorted(within_starts.items())[: args.limit_sessions]}
        ext_starts = {s: ext_starts[s] for s in sorted(ext_starts)[: args.limit_sessions]}
    if args.limit_windows_per_session is not None:
        within_starts = {s: v[: args.limit_windows_per_session]
                         for s, v in within_starts.items()}
        ext_starts = {s: v[: args.limit_windows_per_session]
                      for s, v in ext_starts.items()}
    print(json.dumps({
        "surface": "within", "n_sessions": len(within_starts),
        "sessions": sorted(within_starts)}), flush=True)
    print(json.dumps({
        "surface": "external", "n_sessions": len(ext_starts),
        "sessions": sorted(ext_starts)}), flush=True)

    # ---- native dual-granularity scoring (cells + live arm A / D) -----------
    model_specs = {
        "armA_swa": {"path": step0c.CHECKPOINTS["armA"]["path"], "cell": None},
        "D_swa": {"path": step0c.CHECKPOINTS["D"]["path"], "cell": None},
    }
    for cell in landed:
        model_specs[f"{cell}_swa"] = {"path": provenance[cell]["swa_path"], "cell": cell}

    results: dict[str, dict] = {}
    integrity_blocks: dict[str, dict] = {}
    for name, spec in model_specs.items():
        path = Path(spec["path"])
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {(k[len("model."):] if k.startswith("model.") else k): v
                 for k, v in state.items()}
        model = (
            build_cell_model(subpop, consistency_cell, spec["cell"], temporal_residual)
            if spec["cell"] else step0c.build_step0c_model(seed=42)
        )
        model.load_state_dict(state, strict=True)
        del state
        state_before = arm_common.state_sha256(model)
        model.to(device).eval()

        def forward(neural, calib, side, _m=model):
            return _m(neural, calib_trials=calib, side_features=side)

        governing = {
            surface: rescorer.score_last_bin(
                forward, ds, starts, device, matched_scorer.session_r2, output_scale=1.0
            )
            for surface, (ds, starts) in (
                ("within", (within_ds, within_starts)),
                ("external", (ext_ds, ext_starts)),
            )
        }
        diagnostic = {
            surface: annotate_n_windows(pilot.score_track(
                model, ds, all_positions(ds, starts), device, 128, 50,
                cap_per_session=None, forward_mode="identity_cached",
            ))
            for surface, (ds, starts) in (
                ("within", (within_ds, within_starts)),
                ("external", (ext_ds, ext_starts)),
            )
        }
        state_after = arm_common.state_sha256(model)
        if state_before != state_after:
            raise SystemExit(f"scoring mutated state: {name}")
        if not perturbation_disabled(model):
            raise SystemExit(f"perturbation leaked into the scoring path: {name}")
        results[name] = {
            "governing_last_bin": governing,
            "diagnostic_full_window": diagnostic,
        }
        block = {
            "path": str(path),
            "sha256": (
                provenance[spec["cell"]]["swa_sha256"] if spec["cell"]
                else checkpoint_integrity["armA" if name == "armA_swa" else "D"]["sha256"]
            ),
            "loaded_graph": (
                CELL_SPECS[spec["cell"]]["graph_builder"] if spec["cell"]
                else "src/tfpd/spintshape_module.build_spintshape_model(seed=42)"
            ),
            "graph_note": (
                # W overrides its note (its residual is ACTIVE at eval); the
                # T/C/G perturbation cells keep the original wording
                (CELL_SPECS[spec["cell"]].get("graph_note")
                 or "the exact Arm A graph (2 heads); the route perturbation is "
                    "train-mode-only and inert at eval")
                if spec["cell"]
                else "the run_sparsify_score.py build for armA_swa and D_swa"
            ),
            "strict_load": True,
            "state_unchanged_during_scoring": True,
            "grads_all_none_after": all(
                p.grad is None for p in model.parameters()
                if not isinstance(p, torch.nn.parameter.UninitializedParameter)
            ),
        }
        if spec["cell"]:
            provenance_cell = provenance[spec["cell"]]
            block.update({
                "cell": spec["cell"],
                "terminal_receipt": provenance_cell["terminal_receipt"],
                "terminal_receipt_sha256": provenance_cell["terminal_receipt_sha256"],
                "mode_0444": provenance_cell["mode_0444"],
                "non_symlink": provenance_cell["non_symlink"],
                "launch_final_closure_equal": provenance_cell["launch_final_closure_equal"],
                "initial_state_reconciled": provenance_cell["initial_state_reconciled"],
                "normalizer_reconciled_f062506c": provenance_cell["normalizer_reconciled_f062506c"],
                "theta_authority_reconciled": provenance_cell["theta_authority_reconciled"],
                "num_heads": provenance_cell["num_heads"],
                "dropout_structure": provenance_cell["dropout_structure"],
                "p_sequence_sha256": provenance_cell["p_sequence_sha256"],
                "p_stream_total_draws": provenance_cell["p_stream_total_draws"],
                "epochs_run": provenance_cell["epochs_run"],
                "perturbation_law": provenance_cell["perturbation_law"],
                "realized_perturbation_statistics": (
                    provenance_cell["realized_perturbation_statistics"]
                ),
            })
        else:
            block.update({
                "bound_from": checkpoint_integrity[
                    "armA" if name == "armA_swa" else "D"
                ]["bound_from"],
                "terminal_receipt": checkpoint_integrity[
                    "armA" if name == "armA_swa" else "D"
                ]["terminal_receipt"],
            })
        integrity_blocks[name] = block
        print(json.dumps({
            "model": name,
            "within_gov": round(governing["within"]["mean_r2"], 6),
            "external_gov": round(governing["external"]["mean_r2"], 6),
            "within_full": round(diagnostic["within"]["mean_r2"], 6),
            "external_full": round(diagnostic["external"]["mean_r2"], 6),
        }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- sealed-reference cross-check (as measured, never asserted away) ----
    sealed_crosscheck = {}
    for name, sealed in (
        ("armA_swa", references["governing_bars"]["armA"]),
        ("D_swa", references["governing_bars"]["D"]),
    ):
        live = results[name]["governing_last_bin"]["external"]["mean_r2"]
        sealed_crosscheck[name] = {
            "live_reproduced": live,
            "sealed_reference": sealed["external"],
            "live_minus_sealed": live - sealed["external"],
            "within_live_minus_sealed": (
                results[name]["governing_last_bin"]["within"]["mean_r2"] - sealed["within"]
            ),
            "note": (
                "same scorer path and graph; any nonzero value is reported as measured"
            ),
        }

    # ---- paired contrasts (§5: same-engine live pairing, no hand-copied) ----
    def per_session(model, surface, granularity="governing_last_bin"):
        return {
            row["session"]: row["r2"]
            for row in results[model][granularity][surface]["per_session"]
        }

    def paired(model_a, model_b, surface, granularity="governing_last_bin"):
        table_a = per_session(model_a, surface, granularity)
        table_b = per_session(model_b, surface, granularity)
        sessions = sorted(set(table_a) & set(table_b))
        if len(sessions) != len(table_a) or len(sessions) != len(table_b):
            raise SystemExit(f"session-roster mismatch: {model_a} vs {model_b}")
        deltas = [table_a[s] - table_b[s] for s in sessions]
        stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
        stats["contrast"] = f"{model_a} - {model_b} ({surface}, {granularity})"
        return stats

    contrast_pairs = [
        (f"{cell}_swa", "armA_swa") for cell in landed
    ] + [
        (f"{cell}_swa", "D_swa") for cell in landed
    ] + [
        # D's gain over Arm A is the reference the G band and every recovery
        # ratio are calibrated against — scored live, never hand-copied
        ("D_swa", "armA_swa"),
    ]
    if "C" in landed and "T" in landed:
        contrast_pairs.append(("C_swa", "T_swa"))
    contrasts = {}
    for granularity in ("governing_last_bin", "diagnostic_full_window"):
        for model_a, model_b in contrast_pairs:
            for surface in ("within", "external"):
                contrasts[f"{model_a}_minus_{model_b}_{surface}_{granularity}"] = paired(
                    model_a, model_b, surface, granularity
                )

    # ---- date blocks + the separately reported session (external, governing)
    n_windows_reference = {
        row["session"]: row.get("n_windows")
        for row in results["armA_swa"]["governing_last_bin"]["external"]["per_session"]
    }
    blocks: dict[str, dict] = {}
    block_stats: dict[str, dict] = {}
    armA_table = per_session("armA_swa", "external")
    d_table = per_session("D_swa", "external")
    for model in model_specs:
        table = per_session(model, "external")
        for session, value in table.items():
            bucket = "separate_20141203" if session == SEPARATE_SESSION \
                else step0c.date_block(session)
            blocks.setdefault(model, {}).setdefault(bucket, {})[session] = value
        for bucket, entries in blocks[model].items():
            sessions = sorted(entries)
            values = [entries[s] for s in sessions]
            def _block_delta(reference):
                return [entries[s] - reference[s] for s in sessions]
            block_stats.setdefault(model, {})[bucket] = {
                "n_sessions": len(sessions),
                "block_mean": float(np.mean(values)),
                "paired_delta_vs_armA_mean": float(np.mean(_block_delta(armA_table))),
                "paired_delta_vs_armA_n_positive": int(
                    sum(d > 0 for d in _block_delta(armA_table))
                ),
                "paired_delta_vs_D_mean": float(np.mean(_block_delta(d_table))),
                "paired_delta_vs_D_n_positive": int(
                    sum(d > 0 for d in _block_delta(d_table))
                ),
                "window_share_of_external": float(
                    sum(n_windows_reference.get(s, 0) for s in sessions)
                    / max(sum(n_windows_reference.values()), 1)
                ),
                "sessions": sessions,
            }

    # ---- Step 0C (i) unit-loss robustness curve on the external surface ----
    # Native-only cells are STRUCTURALLY excluded here (cell W: the 0C wrapper
    # replicates the PARENT decode and would silently drop the W residual);
    # refuse_cell_curve re-asserts the exclusion at the wrapper call site.
    curve_cells, native_only_cells = curve_cells_for(landed)
    curve_results: dict[str, dict] = {}
    curve_contrasts: dict[str, dict] = {}
    step0c_crosscheck: dict[str, dict] = {}
    if not args.skip_curve:
        curve_models = ["D_swa"] + [f"{cell}_swa" for cell in curve_cells]
        for name in curve_models:
            spec = model_specs[name]
            refuse_cell_curve(spec["cell"])  # the wrong-graph guard, per model
            state = torch.load(
                Path(spec["path"]), map_location="cpu", weights_only=False
            )["state_dict"]
            state = {(k[len("model."):] if k.startswith("model.") else k): v
                     for k, v in state.items()}
            model = (
                build_cell_model(subpop, consistency_cell, spec["cell"], temporal_residual)
                if spec["cell"] else step0c.build_step0c_model(seed=42)
            )
            model.load_state_dict(state, strict=True)
            del state
            state_before = arm_common.state_sha256(model)
            model.to(device).eval()
            wrapper = step0c.SubpopEvalModel(model, min_keep=step0c.MIN_KEEP)
            wrapper.eval()
            model_results: dict[str, dict] = {}

            wrapper.reset_mask_stats()
            native = step0c.score_condition(
                rescorer, matched_scorer, step0c.native_forward(wrapper),
                ext_ds, ext_starts, device, f"{name}/native",
            )
            model_results["native"] = aggregate_seed_rows(
                "native", "none", 0.0,
                [native | {"seed_idx": None, "mask_stats": wrapper.consume_mask_stats()}],
            )
            for kind in step0c.CURVE_KINDS:  # zero_nogain is PRIMARY
                for fraction in fractions:
                    label = f"{kind}_f{fraction}"  # the 0C condition key: paired masks
                    seed_rows = []
                    for seed_idx in range(n_mask_seeds):
                        wrapper.reset_mask_stats()
                        forward = step0c.curve_forward(
                            wrapper, kind, float(fraction), seed_idx, label
                        )
                        row = step0c.score_condition(
                            rescorer, matched_scorer, forward, ext_ds, ext_starts,
                            device, f"{name}/{label}/s{seed_idx}",
                        )
                        row["seed_idx"] = seed_idx
                        row["mask_stats"] = wrapper.consume_mask_stats()
                        seed_rows.append(row)
                    model_results[label] = aggregate_seed_rows(
                        label, kind, float(fraction), seed_rows
                    )
                    print(json.dumps({
                        "model": name, "condition": label,
                        "mean_r2": round(model_results[label]["governing_mean_r2"], 6),
                    }), flush=True)
            for fraction in [f for f in fractions if f > 0.0]:
                label = f"padding_f{fraction}"
                wrapper.reset_mask_stats()
                forward = step0c.curve_forward(wrapper, "padding", float(fraction), 0, label)
                row = step0c.score_condition(
                    rescorer, matched_scorer, forward, ext_ds, ext_starts, device,
                    f"{name}/{label}",
                )
                row["seed_idx"] = 0
                row["mask_stats"] = wrapper.consume_mask_stats()
                model_results[label] = aggregate_seed_rows(
                    label, "padding", float(fraction), [row]
                )
                print(json.dumps({
                    "model": name, "condition": label,
                    "mean_r2": round(model_results[label]["governing_mean_r2"], 6),
                }), flush=True)

            state_after = arm_common.state_sha256(model)
            if state_before != state_after:
                raise SystemExit(f"curve scoring mutated state: {name}")
            if not perturbation_disabled(model):
                raise SystemExit(f"perturbation leaked into the curve path: {name}")
            curve_results[name] = model_results
            del model, wrapper
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # fraction-0 internal identity + sealed Step 0C baseline cross-check
        for name in curve_results:
            native_mean = curve_results[name]["native"]["governing_mean_r2"]
            f0 = {}
            for kind in step0c.CURVE_KINDS:
                label = f"{kind}_f0.0"
                if label in curve_results[name]:
                    f0[kind] = {
                        "mean_r2": curve_results[name][label]["governing_mean_r2"],
                        "minus_native": (
                            curve_results[name][label]["governing_mean_r2"] - native_mean
                        ),
                    }
            step0c_crosscheck[name] = {
                "native_reproduced": native_mean,
                "native_minus_native_governing_score": (
                    native_mean
                    - results[name]["governing_last_bin"]["external"]["mean_r2"]
                ),
                "fraction_zero_points": f0,
                "note": ("the wrapper with no intervention must reproduce the native "
                         "governing external mean exactly; any nonzero value is "
                         "reported as measured"),
            }
            if step0c_baseline["available"]:
                sealed_curves = step0c_baseline["curves"].get(
                    name.removesuffix("_swa"), {}
                )
                deltas = {}
                for label, block in curve_results[name].items():
                    sealed = sealed_curves.get(label)
                    if sealed is not None:
                        deltas[label] = block["governing_mean_r2"] - sealed["governing_mean_r2"]
                step0c_crosscheck[name]["minus_sealed_step0c"] = deltas
                step0c_crosscheck[name]["step0c_receipt_sha256"] = (
                    step0c_baseline["receipt_sha256"]
                )

        # per-fraction cell-minus-D paired deltas with sign counts (C's claim)
        d_blocks = curve_results["D_swa"]
        for cell in curve_cells:
            name = f"{cell}_swa"
            cell_blocks = curve_results[name]
            per_kind = {}
            for kind in list(step0c.CURVE_KINDS) + ["padding"]:
                per_fraction = {}
                for fraction in fractions:
                    if kind == "padding" and float(fraction) <= 0.0:
                        continue
                    label = f"{kind}_f{fraction}"
                    if label not in cell_blocks or label not in d_blocks:
                        continue
                    per_fraction[str(float(fraction))] = {
                        "cell_mean_r2": cell_blocks[label]["governing_mean_r2"],
                        "d_mean_r2": d_blocks[label]["governing_mean_r2"],
                        "paired_delta": curve_contrast(
                            cell_blocks[label], d_blocks[label], matched_scorer
                        ),
                    }
                if per_fraction:
                    per_kind[kind] = {
                        "fractions": per_fraction,
                        "drop_profile_cell": curve_drop_profile(cell_blocks, kind, fractions),
                        "drop_profile_d": curve_drop_profile(d_blocks, kind, fractions),
                        "flatter_than_d_descriptive": flatter_than_d(
                            curve_drop_profile(cell_blocks, kind, fractions),
                            curve_drop_profile(d_blocks, kind, fractions),
                            fractions,
                        ),
                    }
            curve_contrasts[cell] = {
                "primary_kind": step0c.CURVE_KINDS[0],
                "kinds": per_kind,
                "claim": (
                    "C's pre-registered claim is a FLATTER curve than D's; the "
                    "per-fraction paired deltas with sign counts are the evidence"
                ) if cell == "C" else (
                    "per-fraction paired cell-minus-D deltas on the external surface"
                ),
            }

    # ---- pre-registered readings -------------------------------------------
    readings: dict[str, dict] = {}
    if "T" in landed:
        ext = contrasts["T_swa_minus_D_swa_external_governing_last_bin"]
        win = contrasts["T_swa_minus_D_swa_within_governing_last_bin"]
        verdict = t_trichotomy(ext["mean"])
        readings["T"] = {
            "rule": T_TRICHOTOMY_RULE,
            "verdict": verdict,
            "consequence": T_CONSEQUENCES.get(verdict),
            "input": "T_swa - D_swa, external, governing last bin, paired mean",
            "t_minus_d_external": ext,
            "t_minus_d_within": win,
            "t_minus_armA_external": contrasts[
                "T_swa_minus_armA_swa_external_governing_last_bin"
            ],
            "t_minus_armA_within": contrasts[
                "T_swa_minus_armA_swa_within_governing_last_bin"
            ],
            "framing_note": (
                "until T reports, the mechanism sentence is 'random unit-token "
                "ablation to a shared constant, with gain-compensated survivors'"
            ),
        }
    if "G" in landed:
        ext = contrasts["G_swa_minus_armA_swa_external_governing_last_bin"]
        verdict = g_band(ext["mean"])
        readings["G"] = {
            "rule": G_BAND_RULE,
            "verdict": verdict,
            "consequence": G_CONSEQUENCES.get(verdict),
            "input": "G_swa - armA_swa, external, governing last bin, paired mean",
            "g_minus_armA_external": ext,
            "g_minus_armA_within": contrasts[
                "G_swa_minus_armA_swa_within_governing_last_bin"
            ],
            "g_minus_d_external": contrasts["G_swa_minus_D_swa_external_governing_last_bin"],
            "d_minus_armA_external_reference": contrasts[
                "D_swa_minus_armA_swa_external_governing_last_bin"
            ],
            "band_calibration_note": (
                "the +0.08 band is 'over half of D's gain over Arm A', so D's live "
                "paired gain is reported beside the reading"
            ),
        }
    if "C" in landed:
        ext = contrasts["C_swa_minus_D_swa_external_governing_last_bin"]
        win = contrasts["C_swa_minus_D_swa_within_governing_last_bin"]
        conditions, gate_pass = c_gate_conditions(ext, win)
        readings["C"] = {
            "rule": C_GATE_RULE,
            "conditions": conditions,
            "pass": gate_pass,
            "bootstrap_escape_clause": "none",
            "input": "C_swa - D_swa, governing last bin, paired means",
            "c_minus_d_external": ext,
            "c_minus_d_within": win,
            "c_minus_armA_external": contrasts[
                "C_swa_minus_armA_swa_external_governing_last_bin"
            ],
            "c_minus_armA_within": contrasts[
                "C_swa_minus_armA_swa_within_governing_last_bin"
            ],
            "compute_confound_preregistration": (
                "if C - D external mean is small, a compute-matched D control at "
                "96 epochs is required before any claim"
            ),
        }
        if "T" in landed:
            readings["C"]["c_minus_t_external"] = contrasts[
                "C_swa_minus_T_swa_external_governing_last_bin"
            ]
            readings["C"]["c_minus_t_within"] = contrasts[
                "C_swa_minus_T_swa_within_governing_last_bin"
            ]
            readings["C"]["mechanism_sentence_status"] = (
                "T reported; the C mechanism sentence may be written"
            )
        else:
            readings["C"]["mechanism_sentence_status"] = (
                "T NOT landed; per handoff #3 item 5 the C mechanism sentence "
                "must not be written until T reports"
            )
    if "W" in landed:
        # exploratory cell: NO pre-registered gate — a labelled reading with
        # both-granularity contrasts, never a verdict
        readings["W"] = w_reading(contrasts)

    # ---- the AM/IM decomposition reading (handoff 2026-08-19 §3) -----------
    # The matrix input is the EXTERNAL GOVERNING paired mean vs D for each
    # landed cell; the full contrast blocks are attached for audit.  A single
    # landed cell yields matrix_incomplete (its delta is recorded but never
    # read against the matrix).
    am_ext = contrasts.get("AM_swa_minus_D_swa_external_governing_last_bin")
    im_ext = contrasts.get("IM_swa_minus_D_swa_external_governing_last_bin")
    decomposition = decomposition_reading(
        float(am_ext["mean"]) if am_ext is not None else None,
        float(im_ext["mean"]) if im_ext is not None else None,
    )
    for cell, block in (("AM", am_ext), ("IM", im_ext)):
        if block is None:
            continue
        decomposition["cells"][cell].update({
            "contrast_external_governing": block,
            "contrast_within_governing": contrasts.get(
                f"{cell}_swa_minus_D_swa_within_governing_last_bin"
            ),
            "contrast_external_full_window_diagnostic": contrasts.get(
                f"{cell}_swa_minus_D_swa_external_diagnostic_full_window"
            ),
            "contrast_vs_armA_external_governing": contrasts.get(
                f"{cell}_swa_minus_armA_swa_external_governing_last_bin"
            ),
        })

    # ---- A2 development screen (descriptive only) ---------------------------
    a2_bars = references["a2"]["pooled"]
    a2_screen = {}
    for cell in landed:
        node = results[f"{cell}_swa"]["governing_last_bin"]
        screen = a2_development_screen(
            node["external"]["mean_r2"], node["within"]["mean_r2"], a2_bars
        )
        screen["external_exceeds_best_a2_seed"] = bool(
            node["external"]["mean_r2"] >= references["a2"]["per_seed_external_max"]
        )
        screen["within_exceeds_worst_a2_seed"] = bool(
            node["within"]["mean_r2"]
            >= min(v["within"] for v in references["a2"]["per_seed"].values())
        )
        a2_screen[cell] = screen

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    receipt = {
        "schema": "tfpd_subpop_score_v1",
        "status": "SUBPOP_SCORE_SMOKE" if args.smoke else "SUBPOP_SCORED",
        "handoff": "HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md #3/#5/#6",
        "handoff_aimask": (
            "HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md #3 (pre-registered "
            "readings) / #5 (receipt requirements) / #6 (framing)"
        ),
        # handoff 2026-08-19 §5.3: the prior receipt(s) this pass extends or
        # re-scores — explicit via --supersedes, auto-derived from the
        # output-root family, or [] for a first run
        "supersedes": supersedes_block["supersedes"],
        "supersedes_note": supersedes_block["supersedes_note"],
        "supersedes_source": supersedes_block["supersedes_source"],
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan": plan,
        "governing_references": references,
        "step0c_baseline": step0c_baseline,
        "sealed_crosscheck": sealed_crosscheck,
        "curve_crosscheck": step0c_crosscheck,
        "cell_provenance": {
            cell: provenance[cell] for cell in landed
        },
        "not_landed": not_landed,
        "scored_artifacts": integrity_blocks,
        "results": results,
        "n_windows_per_session": {
            model: {
                surface: {
                    row["session"]: row.get("n_windows")
                    for row in results[model]["governing_last_bin"][surface]["per_session"]
                }
                for surface in ("within", "external")
            }
            for model in model_specs
        },
        "window_weighted_full_window_diagnostic": {
            "note": (
                "DIAGNOSTIC ONLY, never governing; equal-session last-bin remains "
                "the governing pair"
            ),
            "means": {
                model: {
                    surface: window_weighted_mean(
                        results[model]["diagnostic_full_window"][surface]["per_session"]
                    )
                    for surface in ("within", "external")
                }
                for model in model_specs
            },
        },
        "external_date_block_statistics": {
            "note": (
                "governing last-bin external; separate_20141203 excluded from the "
                "year blocks and reported on its own"
            ),
            "blocks": block_stats,
        },
        "external_date_blocks_and_separate_session": blocks,
        "contrasts": contrasts,
        "unit_loss_curve": {
            "protocol": "Step 0C (i) verbatim (scripts/run_subpop_step0c.py)",
            "fractions": [float(f) for f in fractions],
            "n_mask_seeds": n_mask_seeds,
            "seed_rule": step0c.SEED_RULE_DOC,
            "results": curve_results,
            "per_cell_minus_d_paired": curve_contrasts,
            "cells_native_only": {
                cell: {
                    "note": note,
                    "curve_run": False,
                    "reason": "structural exclusion; refuse_cell_curve raises if attempted",
                }
                for cell, note in native_only_cells.items()
            },
        },
        "pre_registered_readings": readings,
        "decomposition_reading": decomposition,
        "exploratory_readings_note": (
            "cells listed under pre_registered_readings with label "
            "EXPLORATORY__NO_PREREGISTERED_GATE (cell W) carry NO gate and NO "
            "verdict; their adoption requires its own follow-up (seeds 43/44 "
            "plus a Z4 sibling)"
        ) if "W" in readings else (
            "no exploratory cells landed in this pass"
        ),
        "a2_development_screen": {
            "bars": a2_bars,
            "per_seed": references["a2"]["per_seed"],
            "cells": a2_screen,
            "note": (
                "both absolute governing means must meet the pooled bars; "
                "DESCRIPTIVE seed-42 screen only — seeds 43/44 remain MANDATORY "
                "before any A2 superiority claim"
            ),
            "a2_superiority_claim_admissible": False,
        },
        "disclosed_confound": (
            "the A2/teacher lineage uses behavior_scaling_factor 5.0 with "
            "predict_scaled_behavior; the Arm A / D / T / C / G lineage has none "
            "(sparsification_step0_v1 behavior_scaling_parity_audit)"
        ),
        "granularity_labels": {
            "governing_last_bin": (
                "GOVERNING: last timestep, variance-weighted R2, equal session weight"
            ),
            "diagnostic_full_window": (
                "DIAGNOSTIC ONLY: cannot rescue a failed governing mean"
            ),
        },
        "disclosures": {
            "gpu_training_steps": 0,
            "target_updates_gradients_optimizer_steps": 0,
            "checkpoint_selection_performed": False,
            "checkpoints_modified": False,
            "formal_or_organizer_held_data_opened": False,
            "normalizer_refit_on_target": False,
            "sealed_files_modified": False,
            "lambda_sweep_performed": False,
            "smoke_run": bool(args.smoke),
            "sessions_truncated_to": args.limit_sessions,
            "windows_per_session_truncated_to": args.limit_windows_per_session,
            "curve_skipped": bool(args.skip_curve),
        },
        "source_closure": {
            "launch": closure,
            "final": closure_final,
            "launch_final_closure_equal": (
                closure["closure_sha256"] == closure_final["closure_sha256"]
            ),
        },
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "subpop_score_receipt.json", receipt)

    summary = {
        "cells_landed": landed,
        "not_landed": sorted(not_landed),
        "supersedes": supersedes_block["supersedes"],
        "supersedes_source": supersedes_block["supersedes_source"],
        "governing": {
            name: {
                "within": round(results[name]["governing_last_bin"]["within"]["mean_r2"], 6),
                "external": round(results[name]["governing_last_bin"]["external"]["mean_r2"], 6),
            } for name in model_specs
        },
        "decomposition_reading": {
            "matrix_status": decomposition["matrix_status"],
            "matrix_row": decomposition["matrix_row"],
            "reportable_as_finding": decomposition["reportable_as_finding"],
            "deltas_vs_d_external_governing": {
                cell: round(node["delta_external_governing_mean"], 6)
                for cell, node in decomposition["cells"].items()
            },
        } if decomposition["cells"] else {
            "matrix_status": decomposition["matrix_status"],
        },
        "readings": {
            cell: (
                {"label": readings[cell]["label"], "verdict": None, "pass": None}
                if cell == "W" else
                ({"verdict": readings[cell].get("verdict"), "pass": readings[cell].get("pass")}
                 if cell != "C" else
                 {"pass": readings[cell]["pass"], "conditions": readings[cell]["conditions"]})
            )
            for cell in readings
        },
        "a2_screen": {cell: a2_screen[cell]["development_screen_pass"] for cell in a2_screen},
        "sealed_crosscheck": {
            name: round(v["live_minus_sealed"], 12) for name, v in sealed_crosscheck.items()
        },
        "receipt": str(out_dir / "subpop_score_receipt.json"),
    }
    if not args.skip_curve and curve_contrasts:
        summary["curve_minus_d_primary"] = {
            cell: {
                fraction: {
                    "mean": round(node["paired_delta"]["mean"], 6),
                    "n_positive": node["paired_delta"]["n_positive"],
                }
                for fraction, node in
                curve_contrasts[cell]["kinds"][step0c.CURVE_KINDS[0]]["fractions"].items()
            }
            for cell in curve_contrasts
        }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
