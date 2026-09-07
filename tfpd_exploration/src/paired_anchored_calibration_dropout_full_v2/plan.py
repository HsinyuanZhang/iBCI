"""V2 identity; all science/runtime mechanics are inherited from full V1."""
from src.paired_anchored_calibration_dropout_full_v1 import plan as v1

CELL="PACD_MATCHED_FULL_TRAINING_V2"; SCHEMA="pacd_matched_full_training_v2"
WORK_ORDER_RELATIVE="tfpd_exploration/docs/WORKORDER_PACD_MATCHED_FULL_TRAINING_V2_20260831.md"
ROOT_BASE="tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2"
ARMS={"p0":{"short_m":30,"root":ROOT_BASE+"/p0_fullfull_seed42"},"p1":{"short_m":4,"root":ROOT_BASE+"/p1_m4_seed42"},"p2":{"short_m":10,"root":ROOT_BASE+"/p2_m10_seed42"}}
V1_FAILURE_RELATIVE="tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1/p0_fullfull_seed42"
V1_FAILURE_SHAS={"attempt.json":"803e32bedcf7f9b2129b79ff7769733ece0e9fc164bdc07e0a88eca2f3d77073","launch.json":"de1e8a75b56f5ebc82e82e886d69d46ff82f2c5887a317c9b108389ff8c2392c","source_authority.json":"d0524318304bad0f38742a85938e7801ec98d247ba4bd57fb55917921e06e124","failure.json":"65b125b982506c82a9daa9eb32d88fed1afca7355155abeab2c224f31b4c3d08"}
BOUND_PATTERNS=tuple(v1.BOUND_PATTERNS)+("tfpd_exploration/src/paired_anchored_calibration_dropout_full_v2/__init__.py","tfpd_exploration/src/paired_anchored_calibration_dropout_full_v2/plan.py","tfpd_exploration/src/paired_anchored_calibration_dropout_full_v2/predecessor.py","tfpd_exploration/src/paired_anchored_calibration_dropout_full_v2/smoke.py","tfpd_exploration/scripts/run_pacd_full_training_v2.py",WORK_ORDER_RELATIVE)

class _RuntimePlan:
 """Inherited numerical contract with V2-only identity/closure overrides."""
 CELL=CELL; SCHEMA=SCHEMA; ARMS=ARMS; BOUND_PATTERNS=BOUND_PATTERNS
 REVIEW_EVIDENCE_PATHS=v1.REVIEW_EVIDENCE_PATHS
 def __getattr__(self,name): return getattr(v1,name)
RUNTIME_PLAN=_RuntimePlan()
