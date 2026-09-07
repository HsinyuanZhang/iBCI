"""Frozen constants and laws for the M2 prerequisite audit (Part B2 phase 1).

Everything here is pre-registered before any M2 data is opened.  The audit
never writes to an existing result root and never trains anything.
"""

from pathlib import Path


SCHEMA = "cdm_p1_m2_v1"
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cdm_p1_m2_v1"

# ---------------------------------------------------------------------------
# The frozen M2 runtime bindings (identical to the sealed m2_t4 screen).
# ---------------------------------------------------------------------------

CHECKPOINT_SHA256 = (
    "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
)
NORMALIZATION_SHA256 = (
    "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
)
RIDGE_NORMALIZED_LAMBDA = 0.1
ACTIVITY_HORIZON = 30
CHANNELS = 96
TRIAL_LENGTH = 100
EXPECTED_WITHIN_SESSIONS = 7
EXPECTED_EXTERNAL_SESSIONS = 6

# ---------------------------------------------------------------------------
# Prerequisite (a): completed-trial boundary availability under the contract.
# ---------------------------------------------------------------------------

#: The official FALCON evaluation client actually served to a submitted M2
#: decoder.  The audit re-reads this exact file and records its per-step loop,
#: because that loop is the contract: for a ``continual`` task (h1/m1/m2) the
#: evaluator NEVER calls ``BCIDecoder.on_done`` and never calls ``observe``;
#: the decoder receives only ``reset(dataset_tags)`` and one
#: ``predict(neural_observations)`` per timestep.
EVALUATOR_MODULE = "falcon_challenge.evaluator"
EVALUATOR_SOURCE_MARKERS = {
    "continual_tasks": "self.continual = True",
    "on_done_gated_on_not_continual": "if not self.continual:",
    "predict_receives_neural_only": "step_prediction = decoder.predict(neural_observations)",
}
#: The two decisive facts asserted beyond the markers: the continual branch is
#: exactly the h1/m1/m2 split, and the gated call is the ONLY on_done call on
#: the regression-task path.
EVALUATOR_ASSERTIONS = {
    "continual_branch_covers_m2": "split in ['h1', 'm1', 'm2']",
    "gated_on_done_call": "decoder.on_done(trial_delta_obs)",
}
#: The deployed M2 submission decoder of this program (the cached-identity T4
#: image) makes ``on_done`` a no-op, so even a contract change that delivered
#: ``dones`` would not reach any state machine without a new sealed image.
DEPLOYED_DECODER_RELATIVE = "sua_exploration/evalai_t4_m2/t4_spint_decoder.py"
DEPLOYED_DECODER_ON_DONE_MARKER = "def on_done(self, dones: np.ndarray):"

#: The pre-registered family of CAUSAL, NEURAL-ONLY boundary statistics.  Each
#: statistic is a pure function of the neural stream up to and including bin
#: ``t`` (no behavior, no mask, no trial metadata, no future bins), so any
#: threshold on it is FSU/TTA-legal at inference.  The audit statistic is the
#: per-session rank AUC of each statistic against the true ``trial_change``
#: boundary indicator: AUC ~= 0.5 means NO threshold law exists at all.
BOUNDARY_STATISTICS = ("population_count", "abs_delta_population_count", "abs_delta_population_vector")
BOUNDARY_AUC_PASS_MIN = 0.65
BOUNDARY_VERDICTS = {
    "AVAILABLE_UNDER_OFFICIAL_CONTRACT": (
        "the official evaluator delivers completed-trial boundaries to the decoder"
    ),
    "RECONSTRUCTIBLE_CAUSAL_LAW": (
        "boundaries hidden by the contract, but a pre-registered causal "
        "neural-only law separates them above the frozen AUC floor on the "
        "M2 OWN source surface"
    ),
    "NOT_EVALUABLE_OFFICIAL_CONTRACT": (
        "boundaries hidden by the contract and no pre-registered causal "
        "neural-only law separates them on the source surface; P1 cannot run "
        "under the official M2 evaluation contract"
    ),
}

# ---------------------------------------------------------------------------
# Prerequisite (b): the T4-equivalent carrier and its support anchor.
# ---------------------------------------------------------------------------

#: The M2 per-unit carrier is ``fit_ridge_t4`` (the sealed m2_t4 law): design
#: ``[cos(theta), sin(theta), 1]``, penalty ``diag(n*lambda, n*lambda, 0)``,
#: ``lambda = 0.1``, ``solve(design.T @ design + penalty, design.T @ rates)``.
#: The Stage-O support-anchor law is the SAME algebra
#: (``A0 = Xs^T Xs + diag(n_s*lambda, n_s*lambda, 0)``, ``b0 = Xs^T r_s``), so
#: the anchor must reproduce the sealed carrier bit-exactly through the
#: float32 quantization path.
ANCHOR_PARITY_LAW = {
    "support_anchor": "A0 = Xs^T Xs + diag(n_s*lambda, n_s*lambda, 0); b0 = Xs^T r_s",
    "m2_carrier": "tfpd_exploration.src.calibration_budget_comparators_v1.fit_ridge_t4",
    "normalized_lambda": RIDGE_NORMALIZED_LAMBDA,
    "design_column_order": ("cos", "sin", "intercept"),
    "required_proof": "solve(A0, b0) -> [a, c, sqrt(a*a+c*c), b] float32 == the sealed fit bit-for-bit",
}

# ---------------------------------------------------------------------------
# Prerequisite (c): the direction parametrization.
# ---------------------------------------------------------------------------

#: The M2 task's direction parametrization: continuous center-out target
#: angles about the maze centre, snapped to the eight canonical directions of
#: the shared law (identical literals in ``falcon``/``sua_exploration`` and
#: ``cdm_core``).  Calibration-pool angles are available under the official
#: contract (the labelled calibration NWBs); QUERY-trial angles are not.
DIRECTION_LAW = {
    "calibration_support": "finite trial_target_angles of the labelled calibration pool",
    "canonical_snap": "nearest of the 8 canonical directions (-3pi/4 + k*pi/4)",
    "query_side_measurement": (
        "the deployable cross-group complementary-forward pseudo-direction law "
        "(no query labels); MOOT unless prerequisite (a) passes"
    ),
}


def result_root(repo_root: Path) -> Path:
    return Path(repo_root) / RESULT_ROOT_RELATIVE
