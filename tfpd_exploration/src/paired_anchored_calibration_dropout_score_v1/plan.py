"""Static PACD matched-score contract.  This module is intentionally torch-free."""
from __future__ import annotations

CELL = "PACD_MATCHED_SCORE_V1"
SCHEMA = "pacd_matched_score_v1"
WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PACD_MATCHED_SCORE_V1_20260831.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/paired_anchored_calibration_dropout_score_v1"
AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/paired_anchored_calibration_dropout_score_v1_authority"
SYSTEM_ORDER = ("P0", "P1", "P2", "T0", "C1", "SD")
SURFACE_ORDER = ("within", "external")
BUDGET_ORDER = (30, 10, 4)
REGIME_ORDER = ("honest_total", "activity_isolation")
ROSTER_COUNTS = {"within": 6, "external": 15}
EXPECTED_ROW_COUNT = len(SYSTEM_ORDER) * sum(ROSTER_COUNTS.values()) * len(BUDGET_ORDER) * len(REGIME_ORDER)
BOOTSTRAP_SEED = 42
P0_M30_FLOOR = -0.01
PRIMARY_DELTA = 0.03
PRIMARY_POSITIVE = 10
INCREMENTAL_DELTA = 0.01
INCREMENTAL_POSITIVE = 9
WORST_DELTA_FLOOR = -0.05

# These historical bytes are known and are descriptor-rehashed before a live
# attempt.  PACD producer body literals deliberately remain unavailable.
HISTORICAL_SYSTEMS = {
    "T0": {
        "terminal": "tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/terminal.json",
        "terminal_sha256": "3071d907df2a91cc409d85997ac3f401a9e1af05ab75d902f332370cb8e461c7",
        "swa": "tfpd_exploration/results/cal_aug_v1/t0_operator_disabled/swa_final4.pt",
        "swa_sha256": "b4781d71ae408ba306edc9268597b6ce83600138c0acaa08dbdf5a1a409e7e86",
    },
    "C1": {
        "terminal": "tfpd_exploration/results/cal_aug_v1/c1_prefix_cycle/terminal.json",
        "terminal_sha256": "320e2b9991c75ed1bcb87fab73643ca8a12bf398133d5358a33e32c8fd72d738",
        "swa": "tfpd_exploration/results/cal_aug_v1/c1_prefix_cycle/swa_final4.pt",
        "swa_sha256": "5cc24676777cb1eacb0ce5fd5174c55dc2efd5ca9ede69d24a85f935f8e89f8d",
    },
    "SD": {
        "terminal": "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json",
        "terminal_sha256": "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442",
        "swa": "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt",
        "swa_sha256": "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd",
    },
}

# P0/P1/P2 terminal/SWA/manifest/attempt/launch/source/checkpoint literals do
# not yet exist.  A caller cannot turn a guessed value into a live binding.
LIVE_PACD_PRODUCER_LITERALS = None

BOUND_PATTERNS = (
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/binding.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/score.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v1/lifecycle.py",
    "tfpd_exploration/src/cal_aug_v1/deployment.py",
    "tfpd_exploration/src/cal_aug_v1/__init__.py",
    "tfpd_exploration/src/cal_aug_v1/plan.py",
    "tfpd_exploration/src/cal_aug_v1/mechanism.py",
    "tfpd_exploration/src/cal_aug_v1/receipts.py",
    "tfpd_exploration/src/calibration_gap_v1/p4_stream_stats.py",
    "tfpd_exploration/src/calibration_gap_v1/__init__.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/low_cost_calibration_v1.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "tfpd_exploration/src/tfpd_lane/__init__.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "tfpd_exploration/scripts/run_pacd_matched_score_v1.py",
    WORK_ORDER_RELATIVE,
)
