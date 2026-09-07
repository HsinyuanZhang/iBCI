"""Frozen identities and budget for the PACD matched full-training family."""
from __future__ import annotations
from src.paired_anchored_calibration_dropout_v1 import plan as v1
CELL="PACD_MATCHED_FULL_TRAINING_V1"; SCHEMA="pacd_matched_full_training_v1"
WORK_ORDER_RELATIVE="tfpd_exploration/docs/WORKORDER_PACD_MATCHED_FULL_TRAINING_V1_20260831.md"
ROOT_BASE="tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1"
ARMS={"p0":{"short_m":30,"root":ROOT_BASE+"/p0_fullfull_seed42"},"p1":{"short_m":4,"root":ROOT_BASE+"/p1_m4_seed42"},"p2":{"short_m":10,"root":ROOT_BASE+"/p2_m10_seed42"}}
SEED=42; EPOCHS=48; STEPS_PER_EPOCH=33925; TOTAL_STEPS=EPOCHS*STEPS_PER_EPOCH; BATCH_SIZE=32; FINAL_EPOCHS=(44,45,46,47); SENTINELS=(0,1,16962,33924)
EXPECTED_INITIAL_STATE_SHA="65bacb85447df40ea5e03cffed03b1763d1964bfba50cbbe3d21d1614c07f2a3"
EXPECTED_BEHAVIOR_NORMALIZER_SHA="f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
EXPECTED_SIDE_NORMALIZER_SHA="293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
V2_THROUGHPUT_STEPS_PER_SECOND={"p0":5.876,"p1":10.077,"p2":9.148}
MECHANICAL_PROJECTED_HOURS={"p0":76.98,"p1":44.89,"p2":49.45}
V2_RELATIVE="tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42"
V2_ATTEMPT_SHA="ef2ebde24864c8105e47b6ef1c925149b128fe9531577dbf58fba227ea11eb3a"; V2_TERMINAL_SHA="a04be949665a5c57091ba2192793401f43950735142fdb8fac9e2ec6829f9dfe"; V2_CLOSURE_SHA="1b7ebf98a01e197718588eb9702949304f68b527ba08660ae88dd62857d11339"
BOUND_PATTERNS=tuple(v1.BOUND_PATTERNS)+("tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/__init__.py","tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/plan.py","tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/predecessor.py","tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/runner.py","tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/smoke.py","tfpd_exploration/scripts/run_pacd_full_training_v1.py",WORK_ORDER_RELATIVE)
REVIEW_EVIDENCE_PATHS=("tfpd_exploration/tests/test_pacd_full_training_v1.py","tfpd_exploration/src/paired_anchored_calibration_dropout_v1/core.py")
