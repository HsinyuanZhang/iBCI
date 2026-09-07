"""AJPF-C frozen constants: continual-law matched joint training."""
from __future__ import annotations

SCHEMA = "m2_ajpf_c_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_ajpf_c_v4"
V3_ROOT_RELATIVE = "tfpd_exploration/results/m2_ajpf_c_v3"
V2_ROOT_RELATIVE = "tfpd_exploration/results/m2_ajpf_c_v2"
WORKORDER_V2_ADDENDUM_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_AJPF_C_CONTINUAL_V2_ADDENDUM_20260903.md"
V1_FAILED_ROOT_RELATIVE = "tfpd_exploration/results/m2_ajpf_c_v1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_AJPF_C_CONTINUAL_V1_20260903.md"
PROBE_RECEIPT_RELATIVE = "tfpd_exploration/results/m2_continual_chunk_probe_v1/replay.json"

SELECTED_CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
SELECTED_STUDENT_STATE_SHA256 = "2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20"

# Reused AJPF laws (imported, not copied):
#   adapter.install_after_strict_load / require_positive_zero
#   runner._authorize_joint_trainable / _fresh_adam / _disable_clone_internal_unit_dropout
#   joint_training.one_joint_step (resident-injected)
#   dropout.governing_probability_and_mask / FirstCallMaskHook
#   runtime.task_only_last_bin_mse / BEHAVIOR_SCALING_FACTOR (5.0)
AJPF_V1_PACKAGE = "tfpd_exploration.src.m2_anchored_joint_postfusion_v1"
AJPF_PLAN_LR = {"encoder_alpha_lr": 1e-4, "decoder_lr": 1e-5}  # mirrors their plan literals

SEED = 42
EPOCHS = 12
BATCH_SIZE = 32  # AJPF plan.BATCH_SIZE
WINDOW_BINS = 50
TARGET_COORDINATE_BUDGET = 91_717  # AJPF canonical source coordinates per epoch
TARGET_BUDGET_TOLERANCE = 0.25     # [68k, 114k] accepted; stride disclosed in receipt
GROUP_BUDGET_PER_EPOCH = 3_600     # V2 addendum law: all states; per state windows
WINDOWS_PER_STATE_MAX = 320        # capped by min(state windows, dynamic allocation
                                   # = 80% of availability / n_states, ceiling 320,
                                   # floor 32, multiple of 32), split into <=32 groups

CHUNK_BINS = 100
POOL_CAPACITY = 30
LAWS = ("chunk100", "chunk100e")
MODULES = ("C-NAT", "C-R1", "Ce-NAT", "Ce-R1")
LAW_OF_MODULE = {"C-NAT": "chunk100", "C-R1": "chunk100", "Ce-NAT": "chunk100e", "Ce-R1": "chunk100e"}
ARMS_PER_LAW = {"chunk100": ("C-NAT", "C-R1"), "chunk100e": ("Ce-NAT", "Ce-R1")}

WALL_CAP_SECONDS_PER_LAW = 9_000
SMOKE_STEPS = 12
UPDATE_SENTINEL_ORDINALS = (0, 11)

# Score-side anchors (sealed screen)
SEALED_SCREEN_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_CELL = "ridge_activity30_m4"
R2_ABS_TOLERANCE = 1.0e-7
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000

# Pre-registered gates (external, official windows)
GATE_CANDIDATE_MEAN = 0.005     # >= +0.005 and >=4/6 -> official TTA candidate justified
GATE_PAPER_MEAN = 0.010         # >= +0.010 and >=4/6 and worst >= -0.015 -> paper-level
GATE_PAPER_WORST = -0.015
