"""Torch-free AOF V1 contract."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
SCHEMA='m2_anchored_output_fusion_v1'; ROOT_RELATIVE='tfpd_exploration/results/m2_anchored_output_fusion_v1/source_screen'
DESIGN_RELATIVE='tfpd_exploration/docs/DESIGN_ANCHORED_OUTPUT_FUSION_V1_20260902.md'; DESIGN_SHA256='48f8f2a1ea7aa99267a230bb08d982e02ee706b955f6ebdac1d717158b998d8e'
WORKORDER_RELATIVE='tfpd_exploration/docs/WORKORDER_M2_ANCHORED_OUTPUT_FUSION_V1_20260902.md'; WORKORDER_SHA256='1cacd0f6d52ed3475997b5fcff184f502e20fb224e6270028f66908d1de3e1fe'
GPU_UUID='GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9'; ENV={'CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONPATH':'/home/xinyuan/Work_host/SPINT'}
CHECKPOINT_SHA256='25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e'; STUDENT_STATE_SHA256='2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20'
OFFICIAL_ACT30_RECEIPT_RELATIVE='tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/t4_m2_seed42_dopt4_act30_identity.receipt.json'; OFFICIAL_ACT30_RECEIPT_SHA256='6f90230f9f8f330edeec970ea0108defde24cd43ca3195fea264083cac6fa583'
OFFICIAL_ACT30_PAYLOAD_RELATIVE='tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/artifacts/t4_m2_seed42_dopt4_act30_identity.pkl'; OFFICIAL_ACT30_PAYLOAD_SHA256='e4ff17e857c0bab9bbd900bc737ca7c48476a44ed03725cefc5377d92b959261'
OFFICIAL_ACT30_SCORE_RELATIVE='tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json'; OFFICIAL_ACT30_SCORE_SHA256='6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce'
FIT_COUNT=5; VALIDATION_COUNT=2; STATIC_POOL=30; WINDOW_BINS=50; DECODE_BATCH=1024
CLOSURE=(
 DESIGN_RELATIVE,WORKORDER_RELATIVE,
 OFFICIAL_ACT30_RECEIPT_RELATIVE,'tfpd_exploration/submissions/evalai_m2_act30_dopt4_v1/export_act30_dopt4_payload.py',
 'tfpd_exploration/src/m2_anchored_output_fusion_v1/__init__.py','tfpd_exploration/src/m2_anchored_output_fusion_v1/plan.py','tfpd_exploration/src/m2_anchored_output_fusion_v1/core.py','tfpd_exploration/src/m2_anchored_output_fusion_v1/runner.py','tfpd_exploration/src/m2_anchored_output_fusion_v1/driver.py','tfpd_exploration/scripts/run_m2_anchored_output_fusion_v1.py',
 # Direct production imports and package initializers in their actual order.
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/__init__.py','tfpd_exploration/src/m2_anchored_postfusion_gate_v1/plan.py','tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py','tfpd_exploration/src/m2_anchored_postfusion_gate_v1/pools.py','tfpd_exploration/src/m2_anchored_postfusion_gate_v1/runtime.py','tfpd_exploration/src/m2_anchored_postfusion_gate_v1/selection.py','tfpd_exploration/src/m2_anchored_postfusion_gate_v1/source_replay.py','tfpd_exploration/src/m2_anchored_postfusion_gate_v1/training.py',
 'tfpd_exploration/src/pit_m2_v1/__init__.py','tfpd_exploration/src/pit_m2_v1/plan.py','tfpd_exploration/src/pit_m2_v1/schedule.py','tfpd_exploration/src/pit_m2_v1/hook.py','tfpd_exploration/src/pit_m2_v1/trainer.py',
 'tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/__init__.py','tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/plan.py','tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/binding.py','tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py',
 'tfpd_exploration/src/m2_postfusion_probe_v1/__init__.py','tfpd_exploration/src/m2_postfusion_probe_v1/memory.py',
 'tfpd_exploration/src/cdm_p1_m2_local_v1/__init__.py','tfpd_exploration/src/cdm_p1_m2_local_v1/plan.py','tfpd_exploration/src/cdm_p1_m2_local_v1/gates.py','tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py',
 'tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/__init__.py','tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py','tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py',
 'tfpd_exploration/src/m2_t4_activity_budget_screen_v1/__init__.py','tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py','tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py','tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py','tfpd_exploration/src/calibration_budget_comparators_v1.py',
 'tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/__init__.py','tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py',
 'tfpd_exploration/src/cross_session_worst_group_v1/__init__.py','tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py',
 'streaming_calibration_exp/src/models/components/streaming_spint.py','streaming_calibration_exp/src/models/components/streaming_encoders.py','streaming_calibration_exp/src/models/streaming_calibration_module.py','streaming_calibration_exp/src/data/falcon_datamodule.py',
)
def jsonb(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def closure(root):
 root=Path(root); out={}
 if len(CLOSURE) != len(set(CLOSURE)):
  raise ValueError('AOF closure contains duplicate keys')
 for r in CLOSURE:
  p=root/r
  if not p.is_file() or p.is_symlink():raise ValueError(f'AOF closure leaf missing/symlink: {r}')
  out[r]=file_sha(p)
 if tuple(out) != CLOSURE or set(out) != set(CLOSURE):
  raise ValueError('AOF closure key-set drift')
 return out
