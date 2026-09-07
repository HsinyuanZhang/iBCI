"""Static AC3-0 contract: the frozen action-continuity representation screen.

Authority
---------
``tfpd_exploration/docs/ADDENDUM_ACTION_CONTINUITY_CONTRASTIVE_CARRIER_20260829.md``
including the six BINDING amendments of section 23 (operator decision,
2026-08-29).  Everything that could bias the AC3-0 screen is frozen HERE, in
module bytes that the attempt receipt pins by SHA-256 *before* any evaluation
row runs:

* the amended AC3-0 matrix (R0.5 R-GE is THE formal primary baseline);
* the input contract of section 8 (completed-trial trajectories of the frozen
  decoder under the four complementary-group views + the permitted causal
  per-view summary of section 8.2);
* the contrastive sampling contract of section 9 (C-Time / C-Action /
  cross-group auxiliary / hard negatives / shuffle control);
* the split discipline of section 10.2 (source-session grouped folds, whole
  trials in one split, source-only decoder outputs);
* the binding gates of section 10.4 AS AMENDED BY section 23, including the
  pre-registered E7-mirror disposition string;
* the section 16 shortcut controls and the section 19 stop conditions.

AC3-0 trains ONLY the tiny per-view encoders/direction readouts declared in
this module.  It never trains, updates, or re-forwards the frozen Cell-D
decoder except to materialize the per-trial four-group trajectories that
section 8.1 declares as its input, and that materialization is inference-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan

CELL = "AC3_ACTION_CONTINUITY_FROZEN_REPRESENTATION_SCREEN_V0"
SCHEMA = "ac3_action_continuity_v0"
ADDENDUM_RELATIVE = "tfpd_exploration/docs/ADDENDUM_ACTION_CONTINUITY_CONTRASTIVE_CARRIER_20260829.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/ac3_action_continuity_v0"

# ---------------------------------------------------------------------------
# Immutable predecessor evidence (addendum section 1.2; read-only anchors).
# ---------------------------------------------------------------------------

P2PRIME_RESULT_RELATIVE = "tfpd_exploration/results/learned_gate_p2prime_v1/result.json"
P2PRIME_RESULT_SHA256 = "0b1264318c37b31fa34d655c1bf89989c6c97090c17821a092f2c4daa1971303"
P2PRIME_TERMINAL_RELATIVE = "tfpd_exploration/results/learned_gate_p2prime_v1/terminal.json"
P2PRIME_TERMINAL_SHA256 = "0ab2c2fef8c0c2e880ae47e57567921e1d5dd95aae721169cc64369d3f115b36"
P2PRIME_VERDICT = "SMOOTHING_NO_CARRIER_VALUE_KEEP_OUTPUT_FILTER_ONLY"
SEALED_CELL_D_SWA_SHA256 = v1plan.SEALED_CELL_D_SWA_SHA256

# The P2' coherent-oracle replay that owns the utility estimand (imported, never
# rebuilt).  Its measured wall cost is recorded because it is what makes the
# AC3-0 utility rows fall outside this run's process gate.
P2PRIME_STAGE_COP_WALL_SECONDS = 11906.4

EXACT_DATA_ROOT_ENV = {
    "SUBC_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C",
    "SUBM_DATA_ROOT": "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M",
}

# ---------------------------------------------------------------------------
# Screen scope: source sessions only (addendum section 10.2).
# ---------------------------------------------------------------------------

SOURCE_SURFACE = "within"
SOURCE_SESSION_COUNT = 6
BUDGET = 4
GROUP_COUNT = 4
GROUPS = tuple(range(GROUP_COUNT))

#: The four source-cohort validation sessions are the source-side roster of the
#: frozen evaluation authority; the external-15 roster is never opened by this
#: screen (section 10.2 "no external-15 label or score selects any AC3
#: component"; section 16 control 7).
CONSTRUCTIONS_MATERIALIZED = ("raw", "smoothed_causal")

# ---------------------------------------------------------------------------
# Pre-registration 1: the amended AC3-0 matrix (addendum section 23.1).
# ---------------------------------------------------------------------------

ROWS = ("R0", "R0.5", "R1", "R2", "R3", "R4", "R5", "RS", "O2")
ROW_SPECS = {
    "R0": {
        "representation": "raw atan2 pseudo direction (single view)",
        "construction": "raw",
        "learned_parameters": 0,
        "role": "accepted raw baseline",
        "aggregation": (
            "primary metric averages the four complementary-group view estimates, "
            "matching the immutable P2' sub-study direction-error table; the "
            "group-0 single-view variant is reported alongside"
        ),
    },
    "R0.5": {
        "representation": "R-GE group ensemble: circular mean of the four complementary-group view directions",
        "construction": "raw",
        "learned_parameters": 0,
        "credibility": "rho_GE = resultant length of the four view directions",
        "role": "FORMAL PRIMARY BASELINE (section 23 amendment 1)",
    },
    "R1": {
        "representation": "P2-prime smoothed pseudo direction (trial-local causal EMA alpha=0.25)",
        "construction": "smoothed_causal",
        "learned_parameters": 0,
        "role": "smoothing negative control; already dead per the terminal P2' verdict",
    },
    "R2": {
        "representation": "ordinary supervised circular MLP on per-view trajectory summaries",
        "construction": "raw",
        "learned_parameters": "tiny encoder + circular direction head",
        "labels": "source completed-trial/bin direction labels (E7-mirror control)",
    },
    "R3": {
        "representation": "C-Time contrastive encoder (temporal-neighbor positives)",
        "construction": "raw",
        "learned_parameters": "tiny encoder + linear circular probe",
        "role": "pure continuity test / time-proximity control",
    },
    "R4": {
        "representation": "C-Action contrastive encoder (cross-session action-near positives)",
        "construction": "raw",
        "learned_parameters": "tiny encoder + linear circular probe",
        "role": "primary contrastive method",
    },
    "R5": {
        "representation": "C-Hybrid contrastive encoder (temporal + action positives)",
        "construction": "raw",
        "learned_parameters": "tiny encoder + linear circular probe",
    },
    "RS": {
        "representation": "shuffled C-Action (session-within permuted action labels)",
        "construction": "raw",
        "learned_parameters": "tiny encoder + linear circular probe",
        "role": "negative control; must degrade (addendum sections 9.5, 16.9)",
    },
    "O2": {
        "representation": "true completed-trial direction (frozen true-velocity pipeline)",
        "construction": "true",
        "learned_parameters": 0,
        "role": "leakage oracle / upper bound",
        "leakage_label": "LEAKAGE_DIAGNOSTIC_ONLY_NEVER_DEPLOYABLE",
    },
}

LEARNED_ROWS = ("R2", "R3", "R4", "R5", "RS")
CONTRASTIVE_ROWS = ("R3", "R4", "R5")
PRIMARY_CONTRASTIVE_ROWS = ("R4", "R5")

# ---------------------------------------------------------------------------
# Pre-registration 2: input contract (addendum section 8).
# ---------------------------------------------------------------------------

INPUT_CONTRACT = {
    "primary": "trajectory[s, trial, group] = [v_1..v_T], v_t in R^2 (physical units) of the frozen decoder",
    "source": "the accepted frozen Cell-D evaluator's four complementary-group held-unit forwards",
    "parent_line": (
        "the never-commit (activity-only parent) line of the P2' sub-study: the B3S "
        "activity FIFO advances by the frozen independent-activity law, the carrier "
        "never commits, so the four-group trajectories are source-only and carry no "
        "pseudo-direction feedback"
    ),
    "permitted_summary_features": [
        "integrated displacement (2)",
        "early/middle/late displacement (6)",
        "mean speed, peak speed (2)",
        "endpoint instantaneous direction (2)",
        "speed-weighted direction (2)",
        "path straightness ratio (1)",
        "valid-bin count in prefix (1)",
        "vector coherence |mean v| / mean|v| (1)",
        "speed dispersion (1)",
    ],
    "summary_dim_per_view": 18,
    "summary_causality": (
        "every feature reads only rows 0..t of one trial's own view and resets at the "
        "trial boundary (addendum section 8.2)"
    ),
    "excluded_in_v1": [
        "complementary-group disagreement / cross-group summary features are PERMITTED "
        "by section 8.2 but excluded in v1 so that cross-group auxiliary positives and "
        "the cross-group latent dispersion metric stay non-trivial (section 21 Q1 open)",
        "raw fixed-width neural vectors (not the v1 input)",
    ],
    "forbidden": [
        "target behavior labels",
        "future trials",
        "session ID as a learnable feature",
        "recording date or absolute timestamp as a learnable feature",
        "external R2 or target error",
        "target-fitted normalizers",
        "post-trial data",
        "a per-target-session trained encoder",
    ],
    "normalizer": (
        "per-feature z-scoring with mean/std computed on the TRAINING source sessions "
        "of each fold only, then frozen for that fold"
    ),
    "open_question_note": (
        "section 21 Q1 is open: the fixed-summary encoder is the v1 primary; the raw "
        "trajectory encoder is the declared alternative and is NOT run here"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 3: contrastive sampling contract (addendum section 9).
# ---------------------------------------------------------------------------

SAMPLING = {
    "state_unit": "(session, trial, group, bin t) -- a causal prefix of one completed trial under one view",
    "c_time": {
        "law": "(s, g, t) <-> (s, g, t + delta) inside one uninterrupted trial",
        "delta_bins": [5, 10],
        "delta_probabilities": [0.5, 0.5],
        "never_crosses": ["trial boundary", "session boundary", "gap/reset"],
        "requires": "both endpoints outside the rest condition",
    },
    "c_action": {
        "law": "(s1, g1, t1) <-> (s2, g2, t2) with s1 != s2 and circular distance(u_1, u_2) <= epsilon_action",
        "direction_coordinates": "u_theta = [cos(theta_true), sin(theta_true)] (circular)",
        "epsilon_action_rad": 0.39269908169872414,  # pi/8 == the frozen canonical-distance threshold
        "speed_ratio_window": [0.5, 2.0],
        "phase_tolerance": 0.25,
        "phase_definition": "t / T of the trial's own validity interval",
    },
    "cross_group_auxiliary": {
        "law": "same (session, trial, bin), different complementary group",
        "role": "AUXILIARY positives only; never the only positive type (section 16.3)",
        "swap_probability": 0.25,
        "max_fraction_of_positives": 0.5,
    },
    "hard_negatives": {
        "law": "speed/phase-matched opposite-direction states",
        "min_direction_distance_rad": 2.356194490192345,  # 3*pi/4
        "speed_ratio_window": [0.5, 2.0],
        "phase_tolerance": 0.25,
        "per_anchor": 2,
        "random_negatives": "in-batch negatives are the additional random pool",
    },
    "rest_condition": {
        "law": "true instantaneous speed below the REST_SPEED_QUANTILE quantile of the "
               "TRAINING sessions' true speeds (source-fold-only statistic)",
        "quantile": 0.15,
        "undefined_direction_rows": "excluded from every direction pair/loss and counted in the receipt",
    },
    "shuffle_control": {
        "law": "session-within permutation of the true direction labels used for C-Action "
               "eligibility (fixed seed), preserving marginals, destroying correspondence",
        "must": "destroy the action-conditioned advantage (sections 9.5, 16.9)",
    },
    "seeds": {"sampler": 20260829, "torch": 20260829, "numpy": 20260829},
}

# ---------------------------------------------------------------------------
# Pre-registration 4: split discipline (addendum section 10.2).
# ---------------------------------------------------------------------------

SPLIT_DISCIPLINE = {
    "outer_unit": "source session (leave-one-source-session-out over the 6 source sessions)",
    "inner_unit": "source session, never random windows",
    "trial_integrity": "all rows/bins of one trial remain in one split (splitting is per trial id)",
    "decoder_outputs": "source-only: the frozen source-trained Cell-D decoder, target-update-free",
    "hyperparameter_selection": (
        "embedding dimension and temperature are selected by one grouped leave-one-source-"
        "session-out CV over the source sessions only (never a held-out outer session's "
        "labels select a component; disclosed as a single grouped selection, not nested)"
    ),
    "external_usage": "none: the external-15 roster is never opened by this screen",
}

# ---------------------------------------------------------------------------
# Pre-registration 5: the tiny learned models (this IS the AC3-0 scope).
# ---------------------------------------------------------------------------

MODELS = {
    "encoder": "MLP 18 -> 32 (tanh) -> embedding dim d, L2-normalized",
    "embedding_grid": [8, 16],
    "temperature_grid": [0.5],
    "r2_head": (
        "linear direction head d -> 2 trained jointly with the encoder (circular loss) on "
        "the COMPLETED-TRIAL direction labels at the t=T states of the training sessions"
    ),
    "contrastive_probe": (
        "linear circular probe d -> 2 fitted by closed-form ridge (lambda=1e-3) on the "
        "training sessions' COMPLETED-TRIAL direction labels at the t=T states with a "
        "frozen encoder; the encoder itself is trained self-supervised on the section-9 "
        "state pairs and never sees a direction label"
    ),
    "estimand_note": (
        "every row's estimand is the frozen completed-trial integrated direction; the "
        "readout is fitted on that label at t=T because the bin-level endpoint direction "
        "is a different estimand (on this cohort it sits a mean 2.34 rad from the "
        "integrated completed-trial direction)"
    ),
    "probe_ridge_lambda": 0.001,
    "optimizer": "Adam lr=1e-3, batch 512, 100 epochs, torch CPU, single thread, fixed seed",
    "states_per_trial_view": 4,
    "trial_level_direction": "circular mean of the four view estimates at t=T",
    "trial_level_credibility_learned": "resultant length of the four view direction estimates",
    "parameter_budget": "at most ~2000 parameters per model",
    "no_gpu_training": True,
}

# ---------------------------------------------------------------------------
# Pre-registration 6: metrics (addendum section 10.3).
# ---------------------------------------------------------------------------

METRICS = {
    "circular_error": "mean circular distance of the row's completed-trial estimate vs the frozen true direction",
    "snap_mismatch": "fraction of estimates whose nearest canonical direction differs from the true one",
    "retrieval": (
        "for each trial, the nearest neighbour among OTHER source sessions' trials in "
        "the row's own representation; reported as the true-direction agreement rate "
        "and the mean circular error to the retrieved neighbour's true direction"
    ),
    "cross_group_dispersion": "1 - resultant length of the four view embeddings (learned rows)",
    "correct_minus_shuffle": "R4 metric minus the same metric under RS",
    "credibility_calibration": (
        "calibration curve of credibility (rho_GE for R0.5; resultant length for learned "
        "rows) vs true direction error, per session and pooled (section 23 amendment 4)"
    ),
    "calibration_bins": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001],
    "utility": (
        "coherent matched carrier utility under exact replay, owned by the imported P2' "
        "machinery (see UTILITY_ROWS)"
    ),
    "recovered_fraction_of_O2_minus_O1": "reported from the immutable P2' receipt only",
}

# ---------------------------------------------------------------------------
# Pre-registration 7: binding gates (section 10.4 AS AMENDED by section 23).
# ---------------------------------------------------------------------------

GATES = {
    "REPRESENTATION_GATE": {
        "applies_to": ["R3", "R4", "R5"],
        "requires_any": True,
        "circular_error_improvement_over_R0_rad": 0.10,
        "snap_mismatch_improvement_over_R0_pp": 10.0,
        "shuffle_control_must_degrade": True,
        "not_one_session_driven": (
            "the improvement over R0 must survive dropping any single source session "
            "(every leave-one-session-out pooled improvement >= the same threshold)"
        ),
        "target_update_counts_zero": True,
    },
    "CONTRASTIVE_CLAIM_GATE": {
        "amendment": "section 23 amendment 2: strengthens to beat max(R2, R-GE), not merely R0",
        "requires_any_of": ["R4", "R5"],
        "circular_error_margin_rad": 0.05,
        "coherent_utility_margin": 0.005,
        "baseline": "max(R2, R0.5) on the source grouped-OOF circular error",
        "disjunct_status": (
            "the coherent-utility disjunct is NOT evaluable inside this run's process gate "
            "(see UTILITY_ROWS); only the circular-error disjunct can fire here"
        ),
    },
    "E7_MIRROR": {
        "pre_registered_outcome": (
            "R2 (supervised head) improves direction and carrier utility; R4/R5 fail to "
            "beat R2; keep the supervised direction route, terminate the contrastive "
            "method claim -- a route success, not a method success"
        ),
        "disposition_string": "SUPERVISED_ROUTE_KEEP_CONTRASTIVE_CLAIM_TERMINATED",
        "firing_rule": (
            "fires when R2 improves the source grouped-OOF circular error over R0.5 AND "
            "the CONTRASTIVE_CLAIM_GATE does not pass; the carrier-utility clause of the "
            "pre-registered outcome is recorded as NOT_EVALUATED_PROCESS_GATE when the "
            "utility rows are not run"
        ),
        "report_exactly": "report as exactly the disposition string, a route success not a method success",
    },
    "DOWNSTREAM_ADVANCE": {
        "expression": "corrected-pseudo coherent utility - raw-pseudo coherent utility >= +0.01",
        "authority": "the imported P2' coherent-replay machinery",
        "status": "see UTILITY_ROWS",
    },
}

#: Pre-registered exact boolean semantics of the shuffle control.  RS degrades
#: when it is strictly worse than R4 on BOTH primary representation metrics.
SHUFFLE_DEGRADATION_RULE = {
    "expression": "(RS.circular_error > R4.circular_error) and (RS.snap_mismatch > R4.snap_mismatch)",
    "alternative_reading": "shuffled labels must not retain the claimed advantage",
}

UTILITY_ROWS = {
    "owner": "src.learned_gate_p2prime_v1.physical (_rollout_oracle / _horizon_counterfactual)",
    "law": "the P2' coherent counterfactual u_j on the fixed horizon H=5, exact replay",
    "status_this_run": "NOT_RUN_PROCESS_GATE",
    "disclosure": (
        "the coherent corrected-pseudo utility row needs per-proposal horizon "
        "counterfactuals on the frozen decoder; the identical machinery measured "
        f"{P2PRIME_STAGE_COP_WALL_SECONDS:.1f} s of wall time in P2' stage_cop, far "
        "outside this run's brief-inference (<10 min GPU) and <=30 min task gates.  The "
        "row is wired to the imported machinery and NOT executed; the downstream-advance "
        "gate and stop conditions 6/10 are therefore recorded PENDING, never as passed"
    ),
    "context_from_immutable_receipt": (
        "P2' m4 external coherent oracle opportunity O2-O1 = +0.0922 and O1-O0 = "
        "-0.000236 are quoted read-only from the immutable P2' result as the size of the "
        "remaining pseudo-direction opportunity"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registration 8: section 16 shortcut controls (audit rows + tests).
# ---------------------------------------------------------------------------

SHORTCUT_CONTROLS = {
    "session_id_predicts_target": "linear probe from the encoder input features to session ID (audit row)",
    "trial_split_leakage": "no train/validation split shares a trial id (structural test)",
    "cross_group_only_positives": "cross-group positives are auxiliary and capped (sampler counts)",
    "time_proximity_explains_result": "R3 (C-Time) row plus the correct-minus-shuffle metric",
    "speed_or_phase_explains_direction": (
        "hard negatives are speed/phase matched; retrieval agreement is additionally "
        "stratified by speed tercile and by phase"
    ),
    "rest_rows_silent": "rest/undefined-direction rows are excluded and counted",
    "target_labels_select_encoder": "the external roster is never opened (materialization proof)",
    "target_unlabeled_adaptation": "target update counts must remain zero",
    "shuffle_retains_advantage": "RS row",
    "representation_without_utility": "the downstream gate is PENDING, never silently passed",
}

STOP_CONDITIONS = {
    1: "no contrastive arm improves R0 circular error by 0.10 rad",
    2: "snap mismatch does not improve by 10 percentage points",
    3: "R4/R5 does not beat max(R2, R-GE) by the declared contrastive gate",
    4: "shuffled control does not degrade",
    5: "representation gain is driven by one source session",
    6: "corrected direction does not improve coherent carrier utility by +0.01",
    7: "the small learned gate has null/negative held-session realized utility",
    8: "a larger model is required before a small model shows any downstream value",
    9: "governing velocity performance degrades materially in the GPU pilot",
    10: "AC3 requires target parameter updates or target-selected hyperparameters",
    11: "trial/reset chronology cannot be proved",
    12: "external improvement cannot be separated from final-output filtering",
}

OWNED_PATHS = (
    ADDENDUM_RELATIVE,
    "tfpd_exploration/src/ac3_action_continuity_v1/__init__.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/plan.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/circular.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/summaries.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/sampler.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/encoder.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/matrix.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/gates.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/materialize.py",
    "tfpd_exploration/src/ac3_action_continuity_v1/screen.py",
    "tfpd_exploration/scripts/run_ac3_action_continuity_v0.py",
    "tfpd_exploration/tests/test_ac3_action_continuity_v0.py",
)

PROCESS_GATE = {
    "per_task_seconds": 1800,
    "gpu_inference_seconds": 600,
    "gpu_inference_precondition": "both GPUs idle (nvidia-smi util 0% and no compute processes) checked immediately before and after",
    "training_allowed": "only the tiny AC3-0 encoders/direction heads declared in MODELS (CPU)",
    "stop_rule": "any single task projected beyond a gate STOPs and is reported",
}

# ---------------------------------------------------------------------------
# Operator review requirements injected between materialize and screen
# (2026-08-29).  These EXTEND the section 23 amendments; they do not replace
# them.  The amendment receipt re-pins the owned bytes BEFORE the screen runs.
# ---------------------------------------------------------------------------

OPERATOR_REVIEW = {
    "received_at_stage": "after materialize, before select/screen evaluation rows",
    "input_lock": {
        "requirement": (
            "mandatory pre-screen gate: trajectories.npz must bind to the materialize "
            "receipt on exactly the same source sessions, trial order, four-group "
            "partition and validity masks, and every row must consume the byte-identical "
            "frozen inputs"
        ),
        "receipt_field": "input_lock",
        "single_digest_field": "input_lock.input_sha256",
        "every_row_binds": True,
        "fail_behavior": "raise before any arm runs",
    },
    "primary_baseline_row": "R0.5",
    "primary_comparison_set": ["R0", "R0.5", "R2", "R4", "R5", "RS", "O2"],
    "primary_baseline_role": "R-GE is THE formal primary baseline, not a diagnostic",
    "result_order": [
        "coherent_matched_carrier_utility",
        "circular_direction_error",
    ],
    "advance_rule": (
        "downstream carrier utility is reported FIRST; a direction-error improvement "
        "without a utility improvement does NOT advance"
    ),
    "output_filter": {
        "used_in_ac3_0": False,
        "rule": "all main comparisons on raw output only",
        "note": (
            "the P2' fixed output filter (causal EMA alpha=0.25) is not applied anywhere in "
            "AC3-0; row R1 is the PSEUDO-DIRECTION smoothing negative control, a different "
            "object from the output filter, and it never touches any other row's comparison"
        ),
    },
    "ac3_1_status": "HOLD",
    "ac3_2_status": "NO_GO",
    "screen_disposition": (
        "the screen result goes to the operator for the advance decision; no further stage "
        "is started by the screen itself"
    ),
    "operator_verdicts": {
        "R_GE_EQUIVALENT": {
            "string": "STOP_LEARNING_KEEP_ZERO_PARAM_ENSEMBLE",
            "condition": (
                "no learned row (R2/R3/R4/R5) improves over R0.5 by more than the "
                "predeclared equivalence band on the source grouped-OOF circular error"
            ),
            "equivalence_band_rad": 0.02,
            "band_disclosure": (
                "the operator review fixed the string but not the band; 0.02 rad is "
                "predeclared HERE before the screen runs (the program's established "
                "material-difference threshold, e.g. P2' KC1)"
            ),
        },
        "E7_MIRROR": {
            "string": "SUPERVISED_ROUTE_KEEP_CONTRASTIVE_CLAIM_TERMINATED",
            "condition": "R2 improves over R0.5 AND neither R4 nor R5 beats R2",
        },
        "ADVANCE": {
            "string": "ADVANCE_AC3_1",
            "condition": (
                "R4/R5 beats max(R2, R0.5) by the declared margin AND the coherent matched "
                "carrier utility improves in step"
            ),
            "utility_clause": (
                "requires the coherent utility row, which is PENDING under this run's "
                "process gate; this verdict therefore CANNOT fire from this screen"
            ),
        },
        "NO_GPU": {
            "string": "STOP_AC3_NO_GPU",
            "condition": "all methods' coherent matched carrier utility is ~ 0",
            "utility_clause": (
                "requires the coherent utility rows, PENDING under this run's process gate"
            ),
        },
    },
}



def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def owned_sha256s(root: Path) -> dict[str, str]:
    base = Path(root).absolute()
    return {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in OWNED_PATHS
    }


def pre_registration_payload() -> dict[str, object]:
    return {
        "row_specs": {key: dict(value) for key, value in ROW_SPECS.items()},
        "input_contract": dict(INPUT_CONTRACT),
        "sampling": json.loads(json.dumps(SAMPLING, sort_keys=True)),
        "split_discipline": dict(SPLIT_DISCIPLINE),
        "models": dict(MODELS),
        "metrics": dict(METRICS),
        "gates": json.loads(json.dumps(GATES, sort_keys=True)),
        "shuffle_degradation_rule": dict(SHUFFLE_DEGRADATION_RULE),
        "utility_rows": dict(UTILITY_ROWS),
        "shortcut_controls": dict(SHORTCUT_CONTROLS),
        "stop_conditions": {str(key): value for key, value in STOP_CONDITIONS.items()},
        "process_gate": dict(PROCESS_GATE),
        "operator_review": json.loads(json.dumps(OPERATOR_REVIEW, sort_keys=True)),
    }


def validate_pre_registration(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if tuple(result.get("row_specs", ())) != ROWS:
        raise ValueError("AC3-0 row topology pre-registration drift")
    if result.get("gates", {}).get("CONTRASTIVE_CLAIM_GATE", {}).get("baseline") != "max(R2, R0.5) on the source grouped-OOF circular error":
        raise ValueError("AC3-0 amended contrastive-claim gate pre-registration drift")
    if result.get("gates", {}).get("E7_MIRROR", {}).get("disposition_string") != GATES["E7_MIRROR"]["disposition_string"]:
        raise ValueError("AC3-0 E7-mirror disposition pre-registration drift")
    if result.get("utility_rows", {}).get("status_this_run") != "NOT_RUN_PROCESS_GATE":
        raise ValueError("AC3-0 utility-row process-gate pre-registration drift")
    if set(result.get("stop_conditions", ())) != {str(key) for key in STOP_CONDITIONS}:
        raise ValueError("AC3-0 stop-condition pre-registration drift")
    return result
