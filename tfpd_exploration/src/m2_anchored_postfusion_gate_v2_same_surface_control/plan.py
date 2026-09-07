"""Pure V2 target-only contract; no Torch, data, or result-root access."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SCHEMA = "m2_anchored_postfusion_gate_v2_same_surface_control"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_postfusion_gate_v2_same_surface_control"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md"
WORKORDER_SHA256 = "b14c453903403e09c1e2c6c995df37bb6f6b367233a4667cafe6aeb7bb9a1303"
V1_ROOT_RELATIVE = "tfpd_exploration/results/m2_anchored_postfusion_gate_v1"
V1_BODIES = {
    "attempt.json": "e546cb34f3efc32b8d6056e5eb6dc74877a9a548543d51a1040d149f1e5407a7",
    "launch.json": "9747223a8a5afed21e9ff59df3d0c44bea85c9a244c6095bf5afa5ace7c9dfef",
    "source_authority.json": "ae1f02c97d507c28fb35165ab616100156e279fa855cb3976d61ab9ff55c0ccf",
    "alpha_selection.json": "c60a9e29928a8f559881745a727d76716b2b458c1aa3d892772f6b8b65eef908",
    "input_authority.json": "41185bceb1307cebc1afb2d7672c55bc71d601ba028a4c0d694310d8af366e24",
    "failure.json": "6e979a4b8abc9761cf6dd9b521bfc7c4a08422ba9d775530bb0cecfc72a23f1d",
}
V1_CLOSURE_SHA256 = "879ef63ce91da2c084285e72a34a7ba0d3a443246720f6bb0afb9e67e6d70934"
SELECTED_EPOCH = 11
REFIT_ALPHA = -0.20759029686450958
SELECTED_CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
SELECTED_STUDENT_STATE_SHA256 = "2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20"
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PYTHONPATH": "/home/xinyuan/Work_host/SPINT"}
SURFACES = ("external_post30_local", "within_post30")
ROSTER_SIZES = {"external_post30_local": 6, "within_post30": 7}
SYSTEMS = ("NATIVE-POOLED", "APFG-ZERO", "APFG-LEARNED")
LAWS = ("FIXED30", "UNCAPPED")
CPU_DECODE_BATCH_SIZE = 1024
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 42, 10_000
CLOSURE_RELATIVES = (WORKORDER_RELATIVE,
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v2_same_surface_control/__init__.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v2_same_surface_control/plan.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v2_same_surface_control/binding.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v2_same_surface_control/laws.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v2_same_surface_control/lifecycle.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v2_same_surface_control/physical.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v2_same_surface_control/driver.py",
 "tfpd_exploration/scripts/run_m2_anchored_postfusion_gate_v2_same_surface_control.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/__init__.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/plan.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/pools.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/laws.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/source_replay.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/scoring.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/selection.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/binding.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/runtime.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/training.py",
 "tfpd_exploration/src/m2_anchored_postfusion_gate_v1/lifecycle.py",
 "tfpd_exploration/src/pit_m2_v1/__init__.py",
 "tfpd_exploration/src/pit_m2_v1/plan.py",
 "tfpd_exploration/src/pit_m2_v1/hook.py",
 "tfpd_exploration/src/pit_m2_v1/schedule.py",
 "tfpd_exploration/src/pit_m2_v1/trainer.py",
 "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/__init__.py",
 "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py",
 "tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py",
 "tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py",
 "tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py",
 "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py",
 "tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py",
 "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py",
 "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/__init__.py",
 "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
 "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
 "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/__init__.py",
 "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/plan.py",
 "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
 "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/physical.py",
 "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
 "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
 "tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py",
 "tfpd_exploration/src/support_anchored_t4_stage_o_v1/__init__.py",
 "tfpd_exploration/src/support_anchored_t4_stage_o_v1/anchor.py",
 "tfpd_exploration/src/support_anchored_t4_stage_o_v1/block_refit.py",
 "tfpd_exploration/src/support_anchored_t4_stage_o_v1/gates.py",
 "tfpd_exploration/src/support_anchored_t4_stage_o_v1/plan.py",
 "tfpd_exploration/src/support_anchored_t4_stage_o_v1/replay.py",
 "tfpd_exploration/src/support_anchored_t4_stage_o_v1/trust_region.py",
 "tfpd_exploration/src/support_anchored_t4_stage_p_v1/__init__.py",
 "tfpd_exploration/src/support_anchored_t4_stage_p_v1/direction_estimator.py",
 "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gate.py",
 "tfpd_exploration/src/support_anchored_t4_stage_p_v1/gates.py",
 "tfpd_exploration/src/support_anchored_t4_stage_p_v1/plan.py",
 "tfpd_exploration/src/support_anchored_t4_stage_p_v1/replay.py",
 "tfpd_exploration/src/cross_session_worst_group_v1/__init__.py",
 "tfpd_exploration/src/cross_session_worst_group_v1/plan.py",
 "tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py",
 # The production coordinator reaches this complete local import closure
 # through PIT's source-only prepare and the G00m activity replay.  Keep
 # these leaves explicit: neither site-packages nor a glob is authority.
 "tfpd_exploration/src/__init__.py",
 "tfpd_exploration/src/calibration_budget_comparators_v1.py",
 "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/__init__.py",
 "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
 "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
 "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
 "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/__init__.py",
 "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/physical.py",
 "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/plan.py",
 "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_v2/transition.py",
 "streaming_calibration_exp/src/__init__.py",
 "streaming_calibration_exp/src/data/__init__.py",
 "streaming_calibration_exp/src/data/afc4_xls_v2.py",
 "streaming_calibration_exp/src/data/afc4_xls_v2_adapter.py",
 "streaming_calibration_exp/src/data/falcon_d4_features.py",
 "streaming_calibration_exp/src/data/falcon_k4_features.py",
 "streaming_calibration_exp/src/data/falcon_n4_features.py",
 "streaming_calibration_exp/src/data/falcon_t4_features.py",
 "streaming_calibration_exp/src/data/validation_protocol.py",
 "streaming_calibration_exp/src/models/__init__.py",
 "streaming_calibration_exp/src/models/falcon_module.py",
 "streaming_calibration_exp/src/models/components/__init__.py",
 "streaming_calibration_exp/src/models/components/carrier_noise_augmentation.py",
 "streaming_calibration_exp/src/models/components/correspondence_breaking.py",
 "streaming_calibration_exp/src/models/components/neuron_dropout.py",
 "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
 "streaming_calibration_exp/src/models/components/spint.py",
 "streaming_calibration_exp/src/models/components/streaming_encoders.py",
 "streaming_calibration_exp/src/models/components/streaming_spint.py",
 "streaming_calibration_exp/src/models/streaming_calibration_module.py",
 "streaming_calibration_exp/src/data/falcon_datamodule.py",
 "streaming_calibration_exp/third_party/__init__.py",
 "streaming_calibration_exp/third_party/catalyst/__init__.py",
 "streaming_calibration_exp/third_party/catalyst/distributed_sampler.py",
 "streaming_calibration_exp/third_party/falcon_challenge/__init__.py",
 "streaming_calibration_exp/third_party/falcon_challenge/filtering.py")
def canonical_json(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
def sha256_bytes(value): return hashlib.sha256(value).hexdigest()
def closure(root: Path):
    r=Path(root); out={}
    for rel in CLOSURE_RELATIVES:
        p=r/rel
        if not p.is_file(): raise ValueError(f"V2 closure leaf missing: {rel}")
        out[rel]=sha256_bytes(p.read_bytes())
    return out
def validate_static(root: Path):
    if sha256_bytes((Path(root)/WORKORDER_RELATIVE).read_bytes()) != WORKORDER_SHA256: raise ValueError("V2 workorder SHA drift")
    return {"workorder_sha256": WORKORDER_SHA256, "closure_sha256": sha256_bytes(canonical_json(closure(root)))}
